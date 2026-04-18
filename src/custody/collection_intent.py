"""Collection-intent tracking and sensor-aware collection reasoning.

Derives what the system has already observed, how recently, and with
which sensor, then produces a collection intent that guides tasking.

All logic is deterministic.  No hidden state.

Public API
----------
Collection-intent constants (module-scope strings)
    SEARCH | CONFIRM | CHARACTERIZE | MONITOR

CollectionIntent (frozen dataclass)
    last_observation_time, last_sensor, hours_since, collection_intent,
    needs_cross_sensor, preferred_confirmation_sensor, rationale.

derive_observation_state(record, history_window) -> CollectionIntent
apply_observation_to_policy(policy_fields, obs_state) -> dict
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

# Hours without a successful observation before we shift to search/reacquisition.
# Raised from 6 to 12 so that sparse collection scenarios (typical in simulation)
# don't put most vessels into search mode unnecessarily.
STALE_OBSERVATION_HOURS = 12.0

# Hours for "recent" — within this window, cross-sensor confirmation is useful.
RECENT_OBSERVATION_HOURS = 4.0


# ---------------------------------------------------------------------------
# Collection intent
# ---------------------------------------------------------------------------

SEARCH       = "search"         # no recent observation; reacquisition needed
CONFIRM      = "confirm"        # recent signal; seek confirming observation
CHARACTERIZE = "characterize"   # high-confidence sustained anomaly; detailed imaging
MONITOR      = "monitor"        # routine tracking


# ---------------------------------------------------------------------------
# Sensor type classification
# ---------------------------------------------------------------------------

_OPTICAL_TYPES = {"high_resolution", "fast_revisit"}
_SAR_TYPES = {"all_weather"}


def _is_optical(sensor_type: str | None) -> bool:
    return sensor_type in _OPTICAL_TYPES


def _is_sar(sensor_type: str | None) -> bool:
    return sensor_type in _SAR_TYPES


def _cross_sensor(last_sensor_type: str | None) -> str | None:
    """Return the preferred cross-sensor type for confirmation.

    If last was optical → prefer SAR.  If last was SAR → prefer optical.
    """
    if _is_optical(last_sensor_type):
        return "all_weather"
    if _is_sar(last_sensor_type):
        return "high_resolution"
    return None


# ---------------------------------------------------------------------------
# Observation state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CollectionIntent:
    """Observation context for one entity at one timestep.

    Attributes:
        last_observation_time:          Timestamp of the most recent
                                        successful TASK with SUCCESS result.
        last_sensor_type:               Sensor type of that observation.
        hours_since_observation:        Hours elapsed since that observation.
                                        None if never observed.
        collection_intent:              search/confirm/characterize/monitor.
        needs_cross_sensor:             True if only one sensor type has
                                        observed the anomaly.
        preferred_confirmation_sensor:  Preferred sensor for next collection
                                        based on cross-sensor logic.
        rationale:                      Human-readable explanation.
    """
    last_observation_time: Optional[datetime]
    last_sensor_type: Optional[str]
    hours_since_observation: Optional[float]
    collection_intent: str
    needs_cross_sensor: bool
    preferred_confirmation_sensor: Optional[str]
    rationale: str


def derive_observation_state(
    record: dict,
    history_window: list[dict],
) -> CollectionIntent:
    """Derive observation state from current record and history.

    Scans history backwards for the most recent TASK/SUCCESS event to
    determine recency and sensor type.  Combines with anomaly state
    and confidence to assign collection intent.

    Args:
        record:          Current timestep record.
        history_window:  Prior records for this entity, oldest first.

    Returns:
        An :class:`CollectionIntent`.
    """
    # Find last successful observation
    last_time: Optional[datetime] = None
    last_sensor: Optional[str] = None
    sensor_types_used: set[str] = set()

    for r in reversed(history_window):
        if r.get("action") == "TASK" and r.get("collection_result") == "SUCCESS":
            s_type = r.get("sensor_type")
            if last_time is None:
                last_time = r.get("time") or r.get("timestamp")
                last_sensor = s_type
            if s_type:
                sensor_types_used.add(s_type)

    # Also check current record
    if record.get("action") == "TASK" and record.get("collection_result") == "SUCCESS":
        s_type = record.get("sensor_type")
        last_time = record.get("time") or record.get("timestamp")
        last_sensor = s_type
        if s_type:
            sensor_types_used.add(s_type)

    # Hours since last observation
    current_time = record.get("time") or record.get("timestamp")
    hours_since: Optional[float] = None
    if last_time is not None and current_time is not None:
        delta = (current_time - last_time).total_seconds() / 3600.0
        hours_since = round(max(0.0, delta), 2)

    # Anomaly context
    state = record.get("anomaly_state", "normal")
    agreement = record.get("anomaly_agreement", "normal")
    confidence = float(record.get("overall_confidence", 1.0))
    is_anomalous = state not in ("normal", "recovering")

    # Cross-sensor need
    needs_cross = (
        is_anomalous
        and len(sensor_types_used) == 1
        and hours_since is not None
        and hours_since <= RECENT_OBSERVATION_HOURS
    )
    cross_sensor = _cross_sensor(last_sensor) if needs_cross else None

    # Collection intent
    rationale_parts: list[str] = []

    if hours_since is None or hours_since >= STALE_OBSERVATION_HOURS:
        intent = SEARCH
        if hours_since is None:
            rationale_parts.append("no prior observation")
        else:
            rationale_parts.append(f"stale observation ({hours_since:.1f}h ago)")
        rationale_parts.append("search/reacquisition required")

    elif is_anomalous and state in ("sustained", "critical") and confidence >= 0.7:
        intent = CHARACTERIZE
        rationale_parts.append(f"sustained anomaly with high confidence ({confidence:.2f})")
        rationale_parts.append("detailed characterization warranted")

    elif is_anomalous and (agreement == "emerging" or confidence < 0.6):
        intent = CONFIRM
        if confidence < 0.6:
            rationale_parts.append(f"anomaly under low confidence ({confidence:.2f})")
        if agreement == "emerging":
            rationale_parts.append("ML-only anomaly not yet confirmed by heuristic")
        rationale_parts.append("confirmation observation needed")

    elif is_anomalous:
        intent = CONFIRM
        rationale_parts.append("anomaly present")
        rationale_parts.append("confirming observation recommended")

    else:
        intent = MONITOR
        rationale_parts.append("routine monitoring")

    # Cross-sensor rationale
    if needs_cross and cross_sensor:
        sensor_label = "EO" if _is_optical(cross_sensor) else "SAR"
        last_label = "SAR" if _is_sar(last_sensor) else "EO"
        rationale_parts.append(
            f"last observed by {last_label}; {sensor_label} preferred for cross-sensor confirmation"
        )

    if last_sensor and hours_since is not None and hours_since <= RECENT_OBSERVATION_HOURS:
        rationale_parts.insert(0, f"last {last_sensor} obs {hours_since:.1f}h ago")

    rationale = "; ".join(rationale_parts)

    return CollectionIntent(
        last_observation_time=last_time,
        last_sensor_type=last_sensor,
        hours_since_observation=hours_since,
        collection_intent=intent,
        needs_cross_sensor=needs_cross,
        preferred_confirmation_sensor=cross_sensor,
        rationale=rationale,
    )


# ---------------------------------------------------------------------------
# Policy adjustment
# ---------------------------------------------------------------------------

def apply_observation_to_policy(
    policy_fields: dict,
    obs: CollectionIntent,
) -> dict:
    """Adjust tasking policy fields based on observation reasoning.

    Modifications (non-mutating — returns a new dict):
      - Cross-sensor preference: if ``needs_cross_sensor`` and current
        effective_sensor_preference does not match, suggest the cross sensor.
      - Stale observation: reduce ``desired_revisit_hours`` by 30%.
      - Intent layering: set ``collection_intent`` and ``observation_rationale``.

    Does not override solar constraints (effective_sensor_preference
    stays SAR if night forced it).

    Args:
        policy_fields: Dict of current policy outputs from the record.
        obs:           Observation state from :func:`derive_observation_state`.

    Returns:
        Updated copy of policy_fields with observation-adjusted values.
    """
    out = dict(policy_fields)
    out["collection_intent"] = obs.collection_intent
    out["observation_rationale"] = obs.rationale
    out["needs_cross_sensor"] = obs.needs_cross_sensor
    out["last_sensor_type"] = obs.last_sensor_type
    out["hours_since_observation"] = obs.hours_since_observation

    # Cross-sensor preference adjustment
    eff = out.get("effective_sensor_preference", "any")
    if obs.needs_cross_sensor and obs.preferred_confirmation_sensor:
        cross = obs.preferred_confirmation_sensor
        # Don't override night-forced SAR with optical
        solar = out.get("solar_condition", "day")
        if solar == "night" and _is_optical(cross):
            pass  # solar constraint takes precedence
        elif eff != cross:
            out["effective_sensor_preference"] = cross
            out["sensor_rationale"] = (
                out.get("sensor_rationale", "") +
                f"; cross-sensor: {cross} preferred (last was {obs.last_sensor_type})"
            )

    # Stale observation: shorten revisit
    if obs.collection_intent == SEARCH:
        revisit = float(out.get("desired_revisit_hours", 12.0))
        out["desired_revisit_hours"] = round(revisit * 0.7, 1)

    return out
