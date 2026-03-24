"""
AIS ingestion layer for the custody package.

This module provides:
  - AISObservation          : typed, unit-normalised record parsed from a CSV source
  - observation_to_vessel   : map one observation to a fresh Vessel snapshot
  - observation_to_history_entry : map one observation to a HistoryEntry
  - parse_ais_csv           : read AIS observations from a file path or raw CSV string
  - ingest_ais_track        : replay a single vessel's observations through the
                              custody pipeline, producing a timeline matching the
                              simulation output schema

Conventions:
  - speed_knots stores the raw AIS value; speed_kmh is the derived property
    used by the rest of the pipeline (Vessel, HistoryEntry).
  - Timestamps without timezone information are treated as UTC.
  - Malformed rows are skipped with warnings.warn rather than raising, because
    real AIS feeds routinely contain gaps and corrupt messages.
  - parse_ais_csv output is sorted by (vessel_id, timestamp).
  - ingest_ais_track sorts its input by timestamp internally and requires all
    observations to belong to the same vessel.
"""
import csv
import io
import math
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone

import custody.config as config
from custody.anomalies import anomaly_breakdown, anomaly_score
from custody.behavior.detectors import vessel_proximity
from custody.behavior.state_machine import infer_state
from custody.decision_trace import build_decision_trace
from custody.features.proximity_features import group_records_by_time, nearest_vessel_from_records
from custody.models import BehaviorState, HistoryEntry, TrackState, Vessel
from custody.planner import plan_collection
from custody.sensors import get_sensor_opportunities
from custody.tracks import update_uncertainty


_REQUIRED_COLUMNS = {"vessel_id", "timestamp", "lat", "lon", "speed_knots", "heading_deg"}


def _stale_confidence(base_confidence: float, gap_seconds: float) -> float:
    """Apply AIS staleness decay to a custody confidence value.

    Gaps up to AIS_STALE_GAP_SECONDS are normal AIS reporting intervals and
    return base_confidence unchanged.  Beyond that, confidence decays
    exponentially with the excess gap time, floored at AIS_MIN_STALE_CONFIDENCE.

    This reflects the operational reality that a vessel which has gone quiet
    for an extended period is harder to maintain in custody than one reporting
    regularly — independent of how uncertain its position has become.

    Args:
        base_confidence: Custody confidence derived from TrackState (0–1).
        gap_seconds:     Elapsed seconds since the previous AIS observation.

    Returns:
        Adjusted confidence in [AIS_MIN_STALE_CONFIDENCE, base_confidence].
    """
    if gap_seconds <= config.AIS_STALE_GAP_SECONDS:
        return base_confidence
    excess = gap_seconds - config.AIS_STALE_GAP_SECONDS
    decay = math.exp(-config.AIS_STALE_CONFIDENCE_DECAY_RATE * excess)
    return max(base_confidence * decay, config.AIS_MIN_STALE_CONFIDENCE)


@dataclass(frozen=True)
class AISObservation:
    """A single parsed AIS position report.

    Attributes:
        vessel_id:    String identifier (MMSI or any label).
        timestamp:    UTC-aware observation time.
        lat:          Latitude in decimal degrees.
        lon:          Longitude in decimal degrees.
        speed_knots:  Speed over ground in knots (raw AIS unit).
        heading_deg:  Course or true heading in degrees.
    """
    vessel_id: str
    timestamp: datetime
    lat: float
    lon: float
    speed_knots: float
    heading_deg: float

    @property
    def speed_kmh(self) -> float:
        """Speed in km/h, converted from the raw knots value."""
        return self.speed_knots * 1.852


def observation_to_vessel(obs: AISObservation) -> Vessel:
    """Create a fresh Vessel snapshot from an AIS observation.

    History is intentionally empty — the caller is responsible for building
    it up as subsequent observations arrive.  speed_kmh (not raw knots) is
    passed to Vessel so that all downstream pipeline code sees consistent units.
    """
    return Vessel(
        id=obs.vessel_id,
        lat=obs.lat,
        lon=obs.lon,
        speed_kmh=obs.speed_kmh,
        heading_deg=obs.heading_deg,
        last_seen=obs.timestamp,
    )


def observation_to_history_entry(obs: AISObservation) -> HistoryEntry:
    """Convert an AIS observation to a HistoryEntry.

    HistoryEntry stores speed in km/h (matching what Vessel uses), so
    speed_kmh is used here rather than the raw knots value.
    """
    return HistoryEntry(
        lat=obs.lat,
        lon=obs.lon,
        timestamp=obs.timestamp,
        speed_kmh=obs.speed_kmh,
        heading_deg=obs.heading_deg,
    )


def parse_ais_csv(source: str) -> list[AISObservation]:
    """Parse AIS observations from a CSV file path or raw CSV string.

    Args:
        source: Either an absolute or relative file path, or a raw CSV string.
                Detection is by presence of a newline character — file paths
                never contain newlines; CSV text always does.

    Returns:
        List of AISObservation objects sorted by (vessel_id, timestamp).

    Raises:
        ValueError:  If any required column is absent from the header.
        OSError:     If source looks like a file path but the file cannot be opened.
    """
    if "\n" in source:
        text = source
    else:
        with open(source, newline="") as fh:
            text = fh.read()

    reader = csv.DictReader(io.StringIO(text))

    if reader.fieldnames is None:
        raise ValueError("CSV appears to be empty — no header row found")

    missing = _REQUIRED_COLUMNS - set(reader.fieldnames)
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {sorted(missing)}"
        )

    observations: list[AISObservation] = []
    for row_num, row in enumerate(reader, start=2):  # row 1 is the header
        try:
            ts = datetime.fromisoformat(row["timestamp"])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            obs = AISObservation(
                vessel_id=row["vessel_id"],
                timestamp=ts,
                lat=float(row["lat"]),
                lon=float(row["lon"]),
                speed_knots=float(row["speed_knots"]),
                heading_deg=float(row["heading_deg"]),
            )
            observations.append(obs)
        except (ValueError, KeyError) as exc:
            warnings.warn(
                f"Skipping malformed AIS row {row_num}: {exc}",
                stacklevel=2,
            )

    observations.sort(key=lambda o: (o.vessel_id, o.timestamp))
    return observations


def ingest_ais_track(
    observations: list[AISObservation],
    observer_position_from_obs: bool = False,
) -> list[dict]:
    """Replay a single vessel's AIS observations through the custody pipeline.

    Produces a timeline list whose records match the simulation output schema
    (same column names and semantics as run_simulation).  The key differences
    from simulation:

      - Position and kinematics come from observations, not dead-reckoning.
      - behavior_mode is always "observed" (no behavior schedule applies).
      - Elapsed time between steps is computed from observation timestamps, so
        irregular reporting intervals are handled correctly.
      - The first record is emitted conservatively: action="NONE",
        behavior_state=UNKNOWN, state_confidence=0.2 (no history yet).

    Args:
        observations: AIS observations for exactly one vessel.  Mixed vessel_ids
                      raise ValueError.  Input is sorted by timestamp internally.
        observer_position_from_obs: When True, each observation's lat/lon is
                      passed as the observer position to get_sensor_opportunities
                      and plan_collection, allowing orbital sensors (SAT-A/SAT-B)
                      to be considered based on actual vessel geometry.  When False
                      (default), observer position is omitted and orbital sensors
                      are excluded — preserving the original AIS replay behavior.

    Returns:
        List of timeline dicts, one per observation, in timestamp order.

    Raises:
        ValueError: If observations contain more than one distinct vessel_id.
    """
    if not observations:
        return []

    vessel_ids = {obs.vessel_id for obs in observations}
    if len(vessel_ids) > 1:
        raise ValueError(
            f"ingest_ais_track requires observations for exactly one vessel; "
            f"got mixed vessel_ids: {sorted(vessel_ids)}"
        )

    observations = sorted(observations, key=lambda o: o.timestamp)

    vessel = observation_to_vessel(observations[0])
    track = TrackState()

    # ── First observation: no history, conservative fixed outputs ─────────────
    first = observations[0]
    first_breakdown = anomaly_breakdown(vessel)
    first_score = anomaly_score(vessel)

    _first_lat = first.lat if observer_position_from_obs else None
    _first_lon = first.lon if observer_position_from_obs else None
    first_opps = get_sensor_opportunities(first.timestamp, _first_lat, _first_lon)
    first_trace = build_decision_trace(
        timestamp=first.timestamp,
        vessel_id=vessel.id,
        score=first_score,
        confidence=track.confidence,
        compound_boost=0.0,
        track=track,
        accessible_opportunities=first_opps,
        claimed_sensor_ids=set(),
        remaining_opportunities=first_opps,
        decision_action="NONE",
        decision_sensor_id=None,
    )

    timeline: list[dict] = [
        {
            "target_id": vessel.id,
            "time": first.timestamp,
            "lat": vessel.lat,
            "lon": vessel.lon,
            "uncertainty_km": track.uncertainty_km,
            "custody_confidence": track.confidence,
            "anomaly_score": first_score,
            "speed_kmh": vessel.speed_kmh,
            "heading_deg": vessel.heading_deg,
            "behavior_mode": "observed",
            "action": "NONE",
            "action_reason": "",
            "sensor_id": None,
            "sensor_type": None,
            "collection_result": None,
            "sensitive_zone": first_breakdown["sensitive_zone"].score,
            "loitering": first_breakdown["loitering"].score,
            "route_deviation": first_breakdown["route_deviation"].score,
            "behavior_state": BehaviorState.UNKNOWN.value,
            "state_confidence": 0.2,
            "decision_trace": first_trace,
        }
    ]

    # ── Subsequent observations ────────────────────────────────────────────────
    for i in range(1, len(observations)):
        prev_obs = observations[i - 1]
        curr_obs = observations[i]

        # Snapshot previous vessel state into history before updating position.
        vessel.history.append(observation_to_history_entry(prev_obs))

        # Elapsed time drives uncertainty growth — never assume a fixed interval.
        gap_seconds = (curr_obs.timestamp - prev_obs.timestamp).total_seconds()
        hours_elapsed = gap_seconds / 3600
        track.uncertainty_km = update_uncertainty(track.uncertainty_km, hours_elapsed)

        # Update vessel fields from incoming observation.
        vessel.lat = curr_obs.lat
        vessel.lon = curr_obs.lon
        vessel.speed_kmh = curr_obs.speed_kmh
        vessel.heading_deg = curr_obs.heading_deg
        vessel.last_seen = curr_obs.timestamp

        breakdown = anomaly_breakdown(vessel)
        score = anomaly_score(vessel)
        inferred_state, state_confidence = infer_state(vessel, vessel.history)

        _obs_lat = curr_obs.lat if observer_position_from_obs else None
        _obs_lon = curr_obs.lon if observer_position_from_obs else None
        opps = get_sensor_opportunities(curr_obs.timestamp, _obs_lat, _obs_lon)
        # Apply staleness decay: long observation gaps reduce confidence beyond
        # what position-uncertainty growth alone captures.
        confidence = _stale_confidence(track.confidence, gap_seconds)
        decision = plan_collection(
            track, score, confidence, breakdown, curr_obs.timestamp, opps,
            compound_boost=0.0,
            observer_lat=_obs_lat,
            observer_lon=_obs_lon,
        )
        trace = build_decision_trace(
            timestamp=curr_obs.timestamp,
            vessel_id=vessel.id,
            score=score,
            confidence=confidence,
            compound_boost=0.0,
            track=track,
            accessible_opportunities=opps,
            claimed_sensor_ids=set(),
            remaining_opportunities=opps,
            decision_action=decision.action,
            decision_sensor_id=decision.sensor_id,
            lookahead_boost=decision.lookahead_boost,
            nearest_pass_tts=decision.nearest_pass_tts,
            hold_reason=decision.hold_reason,
        )

        timeline.append(
            {
                "target_id": vessel.id,
                "time": curr_obs.timestamp,
                "lat": vessel.lat,
                "lon": vessel.lon,
                "uncertainty_km": track.uncertainty_km,
                "custody_confidence": confidence,
                "anomaly_score": score,
                "speed_kmh": vessel.speed_kmh,
                "heading_deg": vessel.heading_deg,
                "behavior_mode": "observed",
                "action": decision.action,
                "action_reason": decision.action_reason,
                "sensor_id": decision.sensor_id,
                "sensor_type": decision.sensor_type,
                "collection_result": decision.collection_result,
                "sensitive_zone": breakdown["sensitive_zone"].score,
                "loitering": breakdown["loitering"].score,
                "route_deviation": breakdown["route_deviation"].score,
                "behavior_state": inferred_state.value,
                "state_confidence": state_confidence,
                "decision_trace": trace,
            }
        )

    return timeline


def _add_proximity_scores(results: dict[str, list[dict]]) -> None:
    """Enrich every record in *results* with inter-vessel proximity fields.

    Groups all records across vessels by timestamp, then for each record calls
    the vessel_proximity detector against the other vessels present at that same
    timestamp.  Three fields are injected in-place:

        vessel_proximity_score  — 0.0, 0.5, or 1.0
        nearest_vessel_id       — target_id of the closest other vessel, or None
        nearest_vessel_km       — Haversine distance in km, or None

    For single-vessel replay (or timesteps with no other vessels present) all
    three fields are set to their zero/None defaults so downstream consumers
    always find the keys present.

    Mutates records in place; does not change the return shape of replay_ais_file.
    """
    # Flatten all records from all vessels into one list for timestamp grouping.
    all_records: list[dict] = [r for tl in results.values() for r in tl]
    by_time = group_records_by_time(all_records)

    for timeline in results.values():
        for record in timeline:
            others_at_time = [
                r for r in by_time.get(record["time"], [])
                if r is not record
            ]
            result = vessel_proximity(
                float(record["lat"]),
                float(record["lon"]),
                record["target_id"],
                others_at_time,
            )
            record["vessel_proximity_score"] = result.score
            if result.score > 0.0:
                record["nearest_vessel_id"] = result.metadata.get("nearest_vessel_id")
                record["nearest_vessel_km"] = result.metadata.get("distance_km")
            else:
                record["nearest_vessel_id"] = None
                record["nearest_vessel_km"] = None


def replay_ais_file(
    source: str,
    observer_position_from_obs: bool = False,
) -> dict[str, list[dict]]:
    """Replay all vessels from an AIS CSV source through the custody pipeline.

    Groups observations by vessel_id and calls ingest_ais_track() for each
    vessel independently.  parse_ais_csv guarantees observations are already
    sorted by (vessel_id, timestamp) before grouping.

    Args:
        source: File path or raw CSV string — passed directly to parse_ais_csv.
        observer_position_from_obs: Forwarded to ingest_ais_track for every
                      vessel.  When True, each vessel's own lat/lon is used as
                      the observer position so orbital sensors are considered.
                      Defaults to False to preserve original behavior.

    Returns:
        Dict keyed by vessel_id; each value is the timeline list returned by
        ingest_ais_track() for that vessel.

    Raises:
        ValueError:  Propagated from parse_ais_csv if required columns are missing.
        OSError:     Propagated from parse_ais_csv if a file path cannot be opened.
    """
    observations = parse_ais_csv(source)
    groups: dict[str, list[AISObservation]] = {}
    for obs in observations:
        groups.setdefault(obs.vessel_id, []).append(obs)
    results = {
        vid: ingest_ais_track(group, observer_position_from_obs=observer_position_from_obs)
        for vid, group in groups.items()
    }
    _add_proximity_scores(results)
    return results
