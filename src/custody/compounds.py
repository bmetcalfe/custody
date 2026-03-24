"""
Compound behavior detection for the custody package.

This module derives higher-level behavioral signals by combining existing
per-record signal values.  It is a read-only layer: it never modifies
records, never calls planner or pipeline logic, and never adds to the
anomaly composite score.

Each compound rule is a private _check_* function that:
  - accepts a plain timeline record dict
  - returns a CompoundSignal if the rule fires, or None otherwise
  - uses record.get() with safe defaults so missing fields never raise

Confidence derivation
---------------------
Each component contributes a value normalised to [0, 1] before combining.
Confidence = min(normalised component values).  This is conservative: a
compound only fires with high confidence when every contributing signal is
strong.

Normalisation per component
---------------------------
  loitering             : already in [0, 1] (detector caps at 1.0)
  loitering (proximity) : divide by 0.5 — moderate loitering saturates at 1.0
  sensitive_zone        : raw score ≤ 1.5; divide by 1.5 to normalise
  vessel_proximity_score: already in [0, 1] (0.0 / 0.5 / 1.0); use directly
  anomaly_score         : divide by CRITICAL_ANOMALY_THRESHOLD (cap at 1.0)
  custody_conf          : inverted — (1.0 - confidence); already in [0, 1]
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from custody.config import (
    CRITICAL_ANOMALY_THRESHOLD,
    HIGH_ANOMALY_THRESHOLD,
    LOW_CUSTODY_THRESHOLD,
    ZONE_COMPOUND_MIN_SCORE,
)

# Maximum raw score emitted by the sensitive_zone detector (inside zone).
_ZONE_SCORE_MAX = 1.5


@dataclass(frozen=True)
class CompoundSignal:
    """A higher-level behavioral observation derived from multiple existing signals.

    Attributes:
        code:       Stable identifier (e.g. "LOITERING_NEAR_ZONE").
        confidence: Normalised confidence in [0, 1] — min of normalised
                    component values.  Does NOT contribute to anomaly_score.
        evidence:   Human-readable explanation listing the contributing signals.
        components: Raw (un-normalised) values of the signals that fired.
        timestamp:  Observation time taken from the source record's "time" field.
    """
    code: str
    confidence: float
    evidence: str
    components: dict
    timestamp: datetime


# ---------------------------------------------------------------------------
# Private rule helpers
# ---------------------------------------------------------------------------

def _check_loitering_near_zone(record: dict) -> Optional[CompoundSignal]:
    """Fire when the vessel is loitering AND within or near a sensitive zone.

    Both signals must be above zero; the zone score must also meet the
    minimum compound threshold.
    """
    loitering = float(record.get("loitering", 0.0))
    zone = float(record.get("sensitive_zone", 0.0))

    if loitering <= 0.0 or zone < ZONE_COMPOUND_MIN_SCORE:
        return None

    # Normalise: loitering already [0,1]; zone divide by max possible score.
    norm_loitering = min(loitering, 1.0)
    norm_zone = min(zone / _ZONE_SCORE_MAX, 1.0)
    confidence = min(norm_loitering, norm_zone)

    return CompoundSignal(
        code="LOITERING_NEAR_ZONE",
        confidence=round(confidence, 4),
        evidence=(
            f"loitering ({loitering:.2f}) + sensitive_zone ({zone:.2f})"
        ),
        components={"loitering": loitering, "sensitive_zone": zone},
        timestamp=record["time"],
    )


def _check_proximity_near_zone(record: dict) -> Optional[CompoundSignal]:
    """Fire when another vessel is in proximity AND the host is near a sensitive zone.

    Represents a potential coordinated approach: two vessels converging on a
    sensitive area simultaneously.
    """
    proximity = float(record.get("vessel_proximity_score", 0.0))
    zone = float(record.get("sensitive_zone", 0.0))

    if proximity <= 0.0 or zone < ZONE_COMPOUND_MIN_SCORE:
        return None

    # Normalise: proximity already [0,1]; zone divide by max possible score.
    norm_proximity = min(proximity, 1.0)
    norm_zone = min(zone / _ZONE_SCORE_MAX, 1.0)
    confidence = min(norm_proximity, norm_zone)

    return CompoundSignal(
        code="PROXIMITY_NEAR_ZONE",
        confidence=round(confidence, 4),
        evidence=(
            f"vessel_proximity_score ({proximity:.2f}) + sensitive_zone ({zone:.2f})"
        ),
        components={"vessel_proximity_score": proximity, "sensitive_zone": zone},
        timestamp=record["time"],
    )


def _check_loitering_with_proximity(record: dict) -> Optional[CompoundSignal]:
    """Fire when the vessel is loitering AND another vessel is in close proximity.

    Represents a potential rendezvous or covert transfer: the host vessel is
    slow/stationary while a second vessel is nearby.
    """
    loitering = float(record.get("loitering", 0.0))
    proximity = float(record.get("vessel_proximity_score", 0.0))

    if loitering <= 0.0 or proximity <= 0.0:
        return None

    # Normalise: loitering divided by 0.5 so moderate loitering (≥ 0.5) saturates;
    # proximity already [0,1].
    norm_loitering = min(loitering / 0.5, 1.0)
    norm_proximity = min(proximity, 1.0)
    confidence = min(norm_loitering, norm_proximity)

    return CompoundSignal(
        code="LOITERING_WITH_PROXIMITY",
        confidence=round(confidence, 4),
        evidence=(
            f"loitering ({loitering:.2f}) + vessel_proximity_score ({proximity:.2f})"
        ),
        components={"loitering": loitering, "vessel_proximity_score": proximity},
        timestamp=record["time"],
    )


def _check_high_anomaly_low_custody(record: dict) -> Optional[CompoundSignal]:
    """Fire when anomaly score is high AND custody confidence is low.

    This represents the worst-case operational situation: the vessel is
    behaving anomalously but we have poor track quality.
    """
    anomaly = float(record.get("anomaly_score", 0.0))
    confidence_val = float(record.get("custody_confidence", 1.0))

    if anomaly <= HIGH_ANOMALY_THRESHOLD or confidence_val >= LOW_CUSTODY_THRESHOLD:
        return None

    # Normalise: anomaly divide by critical threshold (cap at 1); confidence
    # inverted (low confidence → high normalised contribution).
    norm_anomaly = min(anomaly / CRITICAL_ANOMALY_THRESHOLD, 1.0)
    norm_low_conf = min(1.0 - confidence_val, 1.0)
    confidence = min(norm_anomaly, norm_low_conf)

    return CompoundSignal(
        code="HIGH_ANOMALY_LOW_CUSTODY",
        confidence=round(confidence, 4),
        evidence=(
            f"anomaly_score ({anomaly:.2f}) + "
            f"custody_confidence ({confidence_val:.2f})"
        ),
        components={
            "anomaly_score": anomaly,
            "custody_confidence": confidence_val,
        },
        timestamp=record["time"],
    )


def _check_repeated_zone_entry(
    record: dict,
    window: list[dict],
) -> Optional[CompoundSignal]:
    """Fire when the vessel has entered a sensitive zone ≥ 2 times in the window.

    A zone entry is a consecutive pair in the window where the first record
    has sensitive_zone < ZONE_COMPOUND_MIN_SCORE and the second has
    sensitive_zone ≥ ZONE_COMPOUND_MIN_SCORE.  Requires at least 2 records
    in the window to form a consecutive pair; fewer records always return None.

    Confidence scales with the count: min(1.0, entry_count / 3.0).
    """
    if len(window) < 2:
        return None

    entry_count = 0
    for i in range(1, len(window)):
        prev_zone = float(window[i - 1].get("sensitive_zone", 0.0))
        curr_zone = float(window[i].get("sensitive_zone", 0.0))
        if prev_zone < ZONE_COMPOUND_MIN_SCORE and curr_zone >= ZONE_COMPOUND_MIN_SCORE:
            entry_count += 1

    if entry_count < 2:
        return None

    confidence = min(1.0, entry_count / 3.0)

    return CompoundSignal(
        code="REPEATED_ZONE_ENTRY",
        confidence=round(confidence, 4),
        evidence=f"{entry_count} zone entries detected in observation window",
        components={"entry_count": entry_count},
        timestamp=record["time"],
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_compounds(
    record: dict,
    window: Optional[list[dict]] = None,
) -> list[CompoundSignal]:
    """Evaluate all compound rules against one timeline record.

    Args:
        record: One timeline record dict.  Missing fields default to zero
                — this function never raises on absent keys.
        window: Ordered list of *prior* records from the same vessel timeline,
                i.e. all records that precede ``record`` in chronological order.
                Window-based rules (e.g. REPEATED_ZONE_ENTRY) use this history
                to detect multi-step patterns.  None is treated as an empty list.

    Returns:
        List of CompoundSignal objects whose conditions are satisfied.
        Empty list if no rules fire.
    """
    _window = window or []
    candidates = [
        _check_loitering_near_zone(record),
        _check_proximity_near_zone(record),
        _check_loitering_with_proximity(record),
        _check_high_anomaly_low_custody(record),
        _check_repeated_zone_entry(record, _window),
    ]
    return [s for s in candidates if s is not None]


def compounds_for_timeline(timeline: list[dict]) -> list[CompoundSignal]:
    """Evaluate compound rules across an entire single-vessel timeline.

    Passes a growing prefix window to each ``evaluate_compounds`` call so
    window-based rules (e.g. REPEATED_ZONE_ENTRY) have access to all
    records that precede the current position.

    Args:
        timeline: Ordered list of record dicts.  Must belong to the same
                  vessel.

    Returns:
        Flat list of CompoundSignal objects in ascending timeline order.
        Empty if no rules fire or the timeline is empty.
    """
    result: list[CompoundSignal] = []
    for i, record in enumerate(timeline):
        window = timeline[:i] if i > 0 else []
        result.extend(evaluate_compounds(record, window=window))
    return result
