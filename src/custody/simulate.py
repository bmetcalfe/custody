"""
Multi-vessel simulation pipeline for the custody package.

run_simulation() advances two synthetic vessels through a 10-hour scenario
using a three-phase loop at each hourly timestep:

  Phase 1 — Advance all vessel states
    Each vessel's position, uncertainty, anomaly score, and behavioral state
    are updated independently.  Compound signals are evaluated here to produce
    a ``compound_boost`` for priority scoring.

  Phase 2 — Per-vessel sensor-access filtering and vessel ranking
    Sensor opportunities are fetched for each vessel from its own current
    position, so a vessel's orbital visibility reflects where it actually is.
    The ``sensor_access_count`` field on each record is the size of that
    vessel's individual pre-arbitration pool.  Vessels are then sorted by
    ``compute_target_priority`` (highest anomaly + lowest confidence first).

  Phase 3 — Sensor assignment (arbitration) and record emission
    Vessels are processed in priority order.  A ``claimed_sensor_ids`` set
    tracks every sensor_id consumed so far this timestep.  Each vessel starts
    from its own position-filtered pool and has already-claimed sensors
    removed before being passed to ``plan_collection``.  A vessel whose
    initial pool was non-empty but is now fully claimed receives PREEMPTED;
    a vessel whose initial pool was empty receives NO_SENSOR.  A
    position-unique sensor not visible to any higher-priority vessel is
    never in ``claimed_sensor_ids`` and remains available to its vessel.

Action codes emitted by the simulation:
  NONE      — tasking not warranted
  HOLD      — task value below threshold (freshness decay)
  TASK      — sensor assigned (may succeed or fail)
  NO_SENSOR — vessel had no accessible sensors after orbit-access filtering
  PREEMPTED — vessel had sensors initially, but all were claimed by
              higher-priority vessels before its turn

Note: AIS replay (ingest_ais_track) is single-vessel only; PREEMPTED can
never occur there.  The ``sensor_access_count`` field is simulation-specific
and is not emitted by the AIS replay path.
"""
from datetime import datetime, UTC, timedelta
import random

from custody.models import TrackState, Vessel
from custody.tracks import update_position, update_uncertainty
from custody.anomalies import anomaly_score, anomaly_breakdown
from custody.planner import plan_collection, compute_target_priority
from custody.behavior import apply_behavior_mode, get_behavior_mode
from custody.behavior.state_machine import infer_state
from custody.compounds import evaluate_compounds
from custody.decision_trace import build_decision_trace
from custody.sensors import get_sensor_opportunities

def run_simulation():
    random.seed(42)
    start_time = datetime(2026, 3, 23, 10, 0, tzinfo=UTC)
    end_time = datetime(2026, 3, 23, 19, 0, tzinfo=UTC)

    vessel_1 = Vessel(
        id="V001",
        lat=0.0,
        lon=0.0,
        speed_kmh=30,
        heading_deg=45,
        last_seen=start_time,
    )

    vessel_2 = Vessel(
        id="V002",
        lat=0.2,
        lon=0.1,
        speed_kmh=24,
        heading_deg=35,
        last_seen=start_time,
    )

    schedules = {
        "V001": [((0, 3), "transit"), ((4, 5), "approach"), ((6, 8), "loiter"), ((9, 12), "egress")],
        "V002": [((0, 4), "transit"), ((5, 6), "approach"), ((7, 9), "egress")],
    }

    tracks = {v.id: TrackState() for v in [vessel_1, vessel_2]}
    vessels = {v.id: v for v in [vessel_1, vessel_2]}
    timelines = {vid: [] for vid in vessels}

    current_time = start_time
    while current_time <= end_time:
        hour_index = int((current_time - start_time).total_seconds() // 3600)

        # Phase 1: advance all vessel states
        vessel_states = {}
        for vid, vessel in vessels.items():
            track = tracks[vid]
            mode = get_behavior_mode(hour_index, schedules[vid])
            vessel = apply_behavior_mode(vessel, mode)
            vessel = update_position(vessel, hours=1)
            vessel.last_seen = current_time
            track.uncertainty_km = update_uncertainty(track.uncertainty_km, hours=1)
            confidence = track.confidence
            breakdown = anomaly_breakdown(vessel)
            score = anomaly_score(vessel)
            inferred_state, state_confidence = infer_state(vessel, vessel.history)

            mini_record = {
                "time": current_time,
                "anomaly_score": score,
                "custody_confidence": confidence,
                "sensitive_zone": breakdown["sensitive_zone"].score,
                "loitering": breakdown["loitering"].score,
                "vessel_proximity_score": 0.0,
            }
            compounds = evaluate_compounds(mini_record)
            compound_boost = max((s.confidence for s in compounds), default=0.0)

            vessel_states[vid] = {
                "vessel": vessel,
                "track": track,
                "mode": mode,
                "score": score,
                "confidence": confidence,
                "breakdown": breakdown,
                "inferred_state": inferred_state,
                "state_confidence": state_confidence,
                "compound_boost": compound_boost,
            }
            vessels[vid] = vessel

        # Phase 2: per-vessel sensor-access filtering and vessel ranking
        #
        # Each vessel's orbital access is checked from its own current position,
        # so sensor_access_count reflects what that vessel can actually reach.
        for vid, state in vessel_states.items():
            vessel = state["vessel"]
            state["initial_opportunities"] = get_sensor_opportunities(
                current_time, vessel.lat, vessel.lon
            )

        ranked_vessels = sorted(
            vessel_states.items(),
            key=lambda kv: compute_target_priority(
                kv[1]["score"], kv[1]["confidence"], kv[1]["compound_boost"]
            ),
            reverse=True,
        )

        # Phase 3: assign sensors in priority order and record
        #
        # claimed_sensor_ids grows as higher-priority vessels take sensors.
        # Each vessel starts from its own position-filtered pool; sensors with
        # an id already in claimed_sensor_ids are excluded before the call to
        # plan_collection.  A sensor unique to one vessel's position is never
        # in claimed_sensor_ids and is therefore never blocked by arbitration.
        claimed_sensor_ids: set[str] = set()

        for vid, state in ranked_vessels:
            track = state["track"]
            score = state["score"]
            confidence = state["confidence"]
            breakdown = state["breakdown"]
            vessel = state["vessel"]
            vessel_initial = state["initial_opportunities"]
            compound_boost = state["compound_boost"]

            # Snapshot claimed IDs before this vessel's decision so the trace
            # reflects sensors consumed by *other* vessels only.
            claimed_before = set(claimed_sensor_ids)

            vessel_remaining = [
                o for o in vessel_initial if o.sensor_id not in claimed_sensor_ids
            ]
            preempted = bool(vessel_initial) and not vessel_remaining
            decision = plan_collection(
                track, score, confidence, breakdown, current_time,
                vessel_remaining, preempted=preempted,
                compound_boost=compound_boost,
                observer_lat=vessel.lat,
                observer_lon=vessel.lon,
            )
            if decision.action == "TASK" and decision.sensor_id:
                claimed_sensor_ids.add(decision.sensor_id)

            trace = build_decision_trace(
                timestamp=current_time,
                vessel_id=vid,
                score=score,
                confidence=confidence,
                compound_boost=compound_boost,
                track=track,
                accessible_opportunities=vessel_initial,
                claimed_sensor_ids=claimed_before,
                remaining_opportunities=vessel_remaining,
                decision_action=decision.action,
                decision_sensor_id=decision.sensor_id,
                lookahead_boost=decision.lookahead_boost,
                nearest_pass_tts=decision.nearest_pass_tts,
                hold_reason=decision.hold_reason,
            )

            timelines[vid].append({
                "target_id": vid,
                "time": current_time,
                "lat": vessel.lat,
                "lon": vessel.lon,
                "uncertainty_km": track.uncertainty_km,
                "custody_confidence": confidence,
                "history_length": len(vessel.history),
                "anomaly_score": score,
                "speed_kmh": vessel.speed_kmh,
                "heading_deg": vessel.heading_deg,
                "behavior_mode": state["mode"],
                "action": decision.action,
                "action_reason": decision.action_reason,
                "sensor_id": decision.sensor_id,
                "sensor_type": decision.sensor_type,
                "collection_result": decision.collection_result,
                "sensitive_zone": breakdown["sensitive_zone"].score,
                "loitering": breakdown["loitering"].score,
                "route_deviation": breakdown["route_deviation"].score,
                "behavior_state": state["inferred_state"].value,
                "state_confidence": state["state_confidence"],
                # Sensors accessible to this vessel before arbitration —
                # based on this vessel's own lat/lon, not a shared position.
                "sensor_access_count": len(vessel_initial),
                # Full structured decision trace for this vessel/timestep.
                "decision_trace": trace,
            })

        current_time += timedelta(hours=1)

    return [r for vid in timelines for r in timelines[vid]]


if __name__ == "__main__":
    records = run_simulation()
    for row in records[:12]:
        print(
            row["target_id"],
            row["time"].isoformat(),
            row["behavior_mode"],
            row["lat"],
            row["lon"],
            row["anomaly_score"],
            row["action"],
        )
