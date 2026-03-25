"""
Pairwise vessel interaction detector for the custody package.

Detects rendezvous-type interactions: two vessels that dwell in close
proximity for a minimum consecutive period.  A single proximity frame
is insufficient to trigger — the detector requires sustained close contact
to distinguish a genuine rendezvous from a momentary passing encounter.

Operating model
---------------
At each simulation timestep the detector receives:
  - current_positions: where every vessel is NOW (post-update)
  - vessel_histories:  prior records for each vessel (excluding current step)

For every vessel pair it:
  1. Checks whether the current distance is within the proximity threshold.
  2. Walks backward through both vessels' histories counting consecutive
     steps where they were also within the threshold (dwell_prior).
  3. If dwell_prior + 1 (current step) >= min_dwell_steps, a RendezvousEvent
     is emitted for that pair.

Confidence saturates at _CONFIDENCE_SATURATION_HOURS of dwell (default 4h).

Public API
----------
RendezvousEvent
    Frozen dataclass describing a detected rendezvous.

detect_rendezvous_events(current_positions, vessel_histories, timestamp,
                          proximity_km, min_dwell_steps, dt_hours)
    -> list[RendezvousEvent]
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations

from custody.features.proximity_features import haversine_km
from custody.config import RENDEZVOUS_PROXIMITY_KM, RENDEZVOUS_MIN_DWELL_STEPS

# Dwell duration (hours) at which confidence saturates at 1.0.
_CONFIDENCE_SATURATION_HOURS = 4.0


@dataclass(frozen=True)
class RendezvousEvent:
    """A detected vessel-to-vessel rendezvous interaction.

    Attributes:
        vessel_a:          Lexicographically smaller vessel ID.
        vessel_b:          Lexicographically larger vessel ID.
        timestamp:         Time at which the event was detected.
        dwell_hours:       Consecutive hours both vessels were within
                           the proximity threshold (including current step).
        min_separation_km: Minimum pairwise distance observed during
                           the dwell window.
        confidence:        Normalised confidence in [0, 1]; saturates
                           at _CONFIDENCE_SATURATION_HOURS of dwell.
        evidence:          Plain-English explanation of the evidence.
    """
    vessel_a: str
    vessel_b: str
    timestamp: datetime
    dwell_hours: float
    min_separation_km: float
    confidence: float
    evidence: str


def detect_rendezvous_events(
    current_positions: dict[str, tuple[float, float]],
    vessel_histories: dict[str, list[dict]],
    timestamp: datetime,
    proximity_km: float = RENDEZVOUS_PROXIMITY_KM,
    min_dwell_steps: int = RENDEZVOUS_MIN_DWELL_STEPS,
    dt_hours: float = 1.0,
) -> list[RendezvousEvent]:
    """Detect pairwise rendezvous events at a single timestep.

    For every pair of vessels, checks whether they are currently within
    ``proximity_km`` AND have been so for at least ``min_dwell_steps``
    consecutive timesteps (including the current one).  The dwell count
    is determined by walking backward through ``vessel_histories``.

    Args:
        current_positions: Mapping vessel_id → (lat, lon) for the current
                           simulation timestep.  All vessels present in
                           this dict are candidates.
        vessel_histories:  Mapping vessel_id → ordered list of prior records
                           (each dict must carry "lat" and "lon" keys).
                           The last element is the most recent prior step.
                           Vessels absent from this dict have empty histories.
        timestamp:         Current simulation time (written to event).
        proximity_km:      Distance threshold in km.  Both vessels must be
                           within this range to count as a dwell step.
        min_dwell_steps:   Minimum consecutive dwell steps (including the
                           current step) required to emit an event.
        dt_hours:          Duration of one simulation timestep in hours.
                           Used to compute dwell_hours.

    Returns:
        List of RendezvousEvent objects, one per qualifying pair.
        ``vessel_a < vessel_b`` lexicographically in every event.
        Returns an empty list when no pairs meet the dwell threshold.
    """
    vessel_ids = sorted(current_positions.keys())
    events: list[RendezvousEvent] = []

    for a_id, b_id in combinations(vessel_ids, 2):
        a_lat, a_lon = current_positions[a_id]
        b_lat, b_lon = current_positions[b_id]
        current_dist = haversine_km(a_lat, a_lon, b_lat, b_lon)

        if current_dist > proximity_km:
            continue  # not in proximity now — skip pair

        # Walk backward through prior history counting consecutive close steps.
        hist_a = vessel_histories.get(a_id, [])
        hist_b = vessel_histories.get(b_id, [])
        max_lookback = min(len(hist_a), len(hist_b))

        dwell_prior = 0
        min_prior_sep = math.inf
        for i in range(max_lookback):
            rec_a = hist_a[-(i + 1)]
            rec_b = hist_b[-(i + 1)]
            try:
                d = haversine_km(
                    float(rec_a["lat"]), float(rec_a["lon"]),
                    float(rec_b["lat"]), float(rec_b["lon"]),
                )
            except (KeyError, TypeError, ValueError):
                break  # malformed record — stop lookback
            if d <= proximity_km:
                dwell_prior += 1
                min_prior_sep = min(min_prior_sep, d)
            else:
                break  # gap in consecutive dwell — stop

        total_dwell_steps = 1 + dwell_prior  # current step + consecutive prior steps
        if total_dwell_steps < min_dwell_steps:
            continue

        dwell_hours = round(total_dwell_steps * dt_hours, 4)
        min_sep = round(
            min(current_dist, min_prior_sep if min_prior_sep < math.inf else current_dist),
            4,
        )
        confidence = round(min(1.0, dwell_hours / _CONFIDENCE_SATURATION_HOURS), 4)
        evidence = (
            f"vessels {a_id} and {b_id} within {proximity_km:.1f} km "
            f"for {dwell_hours:.1f}h (min separation {min_sep:.2f} km)"
        )

        events.append(RendezvousEvent(
            vessel_a=a_id,
            vessel_b=b_id,
            timestamp=timestamp,
            dwell_hours=dwell_hours,
            min_separation_km=min_sep,
            confidence=confidence,
            evidence=evidence,
        ))

    return events
