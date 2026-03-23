"""
Alert evaluation layer for the custody package.

This module provides:
  - Alert              : immutable record describing a single triggered alert
  - evaluate_alerts    : evaluate all alert rules against one timeline record

Conventions:
  - Alerts are derived purely from timeline record dicts; no pipeline state
    is mutated.
  - Each rule is a private _check_* function returning Alert | None.
  - Rules that need a prior record receive prev_record; they must handle
    prev_record=None explicitly (first-record case).
  - Thresholds are imported from custody.config; none are hardcoded here.
  - Each alert carries a fresh context dict — no shared mutable state.

Alert levels:  "INFO" < "WARNING" < "CRITICAL"
Alert codes:   stable machine-readable strings used for filtering/display.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from custody.config import (
    CRITICAL_ANOMALY_THRESHOLD,
    HIGH_ANOMALY_THRESHOLD,
    LOW_CUSTODY_THRESHOLD,
)


@dataclass(frozen=True)
class Alert:
    """A single triggered alert for one vessel at one point in time.

    Attributes:
        vessel_id:  Target identifier from the timeline record.
        timestamp:  Observation time the alert was raised against.
        level:      Severity — "INFO", "WARNING", or "CRITICAL".
        code:       Stable machine-readable rule identifier.
        message:    Human-readable explanation.
        context:    Dict of raw values that caused the rule to fire.
    """
    vessel_id: str
    timestamp: datetime
    level: str
    code: str
    message: str
    context: dict


# ---------------------------------------------------------------------------
# Private rule helpers
# ---------------------------------------------------------------------------

def _check_critical_anomaly(record: dict) -> Optional[Alert]:
    score = float(record["anomaly_score"])
    if score > CRITICAL_ANOMALY_THRESHOLD:
        return Alert(
            vessel_id=record["target_id"],
            timestamp=record["time"],
            level="CRITICAL",
            code="CRITICAL_ANOMALY",
            message=f"Anomaly score {score:.2f} exceeds critical threshold {CRITICAL_ANOMALY_THRESHOLD}.",
            context={"anomaly_score": score, "threshold": CRITICAL_ANOMALY_THRESHOLD},
        )
    return None


def _check_high_anomaly(record: dict) -> Optional[Alert]:
    score = float(record["anomaly_score"])
    # Only fire WARNING if strictly below the CRITICAL threshold
    if HIGH_ANOMALY_THRESHOLD < score < CRITICAL_ANOMALY_THRESHOLD:
        return Alert(
            vessel_id=record["target_id"],
            timestamp=record["time"],
            level="WARNING",
            code="HIGH_ANOMALY",
            message=f"Anomaly score {score:.2f} exceeds warning threshold {HIGH_ANOMALY_THRESHOLD}.",
            context={"anomaly_score": score, "threshold": HIGH_ANOMALY_THRESHOLD},
        )
    return None


def _check_sensitive_zone_entry(
    record: dict, prev_record: Optional[dict]
) -> Optional[Alert]:
    if prev_record is None:
        return None
    was_outside = float(prev_record["sensitive_zone"]) == 0.0
    now_inside = float(record["sensitive_zone"]) > 0.0
    if was_outside and now_inside:
        return Alert(
            vessel_id=record["target_id"],
            timestamp=record["time"],
            level="WARNING",
            code="SENSITIVE_ZONE_ENTRY",
            message="Vessel has entered or approached the sensitive zone.",
            context={
                "sensitive_zone": float(record["sensitive_zone"]),
                "lat": float(record["lat"]),
                "lon": float(record["lon"]),
            },
        )
    return None


def _check_collection_failed(record: dict) -> Optional[Alert]:
    if record["action"] == "TASK" and record["collection_result"] == "FAILED":
        return Alert(
            vessel_id=record["target_id"],
            timestamp=record["time"],
            level="WARNING",
            code="COLLECTION_FAILED",
            message=(
                f"Collection attempt by sensor {record['sensor_id']} "
                f"({record['sensor_type']}) failed."
            ),
            context={
                "sensor_id": record["sensor_id"],
                "sensor_type": record["sensor_type"],
            },
        )
    return None


def _check_no_sensor_available(record: dict) -> Optional[Alert]:
    if record["action"] == "NO_SENSOR":
        return Alert(
            vessel_id=record["target_id"],
            timestamp=record["time"],
            level="WARNING",
            code="NO_SENSOR_AVAILABLE",
            message="Tasking required but no sensors are available at this time.",
            context={
                "anomaly_score": float(record["anomaly_score"]),
                "custody_confidence": float(record["custody_confidence"]),
            },
        )
    return None


def _check_low_custody(record: dict) -> Optional[Alert]:
    confidence = float(record["custody_confidence"])
    if confidence < LOW_CUSTODY_THRESHOLD:
        return Alert(
            vessel_id=record["target_id"],
            timestamp=record["time"],
            level="WARNING",
            code="LOW_CUSTODY",
            message=f"Custody confidence {confidence:.2f} is below threshold {LOW_CUSTODY_THRESHOLD}.",
            context={
                "custody_confidence": confidence,
                "uncertainty_km": float(record["uncertainty_km"]),
                "threshold": LOW_CUSTODY_THRESHOLD,
            },
        )
    return None


def _check_vessel_proximity(
    record: dict, prev_record: Optional[dict]
) -> Optional[Alert]:
    if prev_record is None:
        return None
    prev_score = float(prev_record.get("vessel_proximity_score", 0.0))
    curr_score = float(record.get("vessel_proximity_score", 0.0))
    if prev_score == 0.0 and curr_score > 0.0:
        nearest_id = record.get("nearest_vessel_id")
        distance_km = record.get("nearest_vessel_km")
        if nearest_id and distance_km is not None:
            message = (
                f"Vessel closed to within {float(distance_km):.1f} km "
                f"of vessel {nearest_id}."
            )
        else:
            message = "Vessel entered proximity range of another vessel."
        return Alert(
            vessel_id=record["target_id"],
            timestamp=record["time"],
            level="WARNING",
            code="VESSEL_PROXIMITY",
            message=message,
            context={
                "vessel_id": record["target_id"],
                "nearest_vessel_id": nearest_id,
                "distance_km": float(distance_km) if distance_km is not None else None,
                "vessel_proximity_score": curr_score,
            },
        )
    return None


def _check_loitering_confirmed(record: dict) -> Optional[Alert]:
    if (
        record["behavior_state"] == "loiter"
        and float(record["state_confidence"]) >= 0.85
    ):
        return Alert(
            vessel_id=record["target_id"],
            timestamp=record["time"],
            level="INFO",
            code="LOITERING_CONFIRMED",
            message="Vessel loitering behavior confirmed with high confidence.",
            context={
                "behavior_state": record["behavior_state"],
                "state_confidence": float(record["state_confidence"]),
            },
        )
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def alerts_for_timeline(timeline: list[dict]) -> list[Alert]:
    """Evaluate alerts across an entire single-vessel timeline.

    Args:
        timeline: Ordered list of record dicts from simulate_target or
                  ingest_ais_track.  Must all belong to the same vessel.

    Returns:
        Flat list of Alert objects in ascending timeline order.  Empty if
        no rules fire or the timeline is empty.
    """
    result = []
    for i, record in enumerate(timeline):
        prev = timeline[i - 1] if i > 0 else None
        result.extend(evaluate_alerts(record, prev))
    return result


def evaluate_alerts(
    record: dict,
    prev_record: Optional[dict] = None,
) -> list[Alert]:
    """Evaluate all alert rules against one timeline record.

    Args:
        record:      One record dict from simulate_target or ingest_ais_track.
        prev_record: The immediately preceding record for the same vessel, or
                     None if this is the first record.  Only rules that detect
                     transitions use this argument.

    Returns:
        List of Alert objects whose conditions are satisfied.  Empty if none fire.
    """
    candidates = [
        _check_critical_anomaly(record),
        _check_high_anomaly(record),
        _check_sensitive_zone_entry(record, prev_record),
        _check_collection_failed(record),
        _check_no_sensor_available(record),
        _check_low_custody(record),
        _check_loitering_confirmed(record),
        _check_vessel_proximity(record, prev_record),
    ]
    return [a for a in candidates if a is not None]
