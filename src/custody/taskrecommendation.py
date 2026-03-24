"""
Collection orchestration layer for custody.

Turns a Decision into a ranked queue of concrete sensing/task options.
This is Layer 6 of the SENTIENT-inspired architecture: the step that
answers "which asset, in which window, and why?" after the mission
reasoning layer has determined what kind of action is warranted.

TaskRecommendation answers:
  - What exact sensing action should happen next?   → sensor
  - In what time window?                            → window_start / window_end
  - Why this option?                                → reason
  - What is its expected value?                     → expected_value
  - What are the backups if it fails?               → fallbacks
  - Where does it sit in the queue?                 → rank

Layer boundary
--------------
  ✓ Operationalise the chosen action class
  ✓ Score concrete collection/task choices
  ✓ Produce a ranked queue

  ✗ Redo FusionAssessment
  ✗ Redo mission significance logic
  ✗ Replace Decision selection logic

Sensor vocabulary
-----------------
  OPTICAL    → SAT-A (high-resolution, ISS-like LEO, 51.6°)
  SAR        → SAT-B (all-weather, sun-synchronous, 97.8°)
  AIS_REFRESH → immediate AIS reacquisition window
  MONITOR    → passive monitoring (always available)

Public API
----------
TaskRecommendation
    Frozen dataclass: task_id, entity_id, sensor, window_start, window_end,
    expected_value, reason, fallbacks, rank.

build_task_recommendations(decision, fusion_assessment, record, track,
                           planner_context) -> list[TaskRecommendation]
    Primary builder.  Returns a ranked queue, sorted by expected_value
    descending.  Sensors with no available window within the horizon are
    omitted.  The queue always contains at least one entry (MONITOR is
    always available).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import custody.config as config
from custody.decision import Decision, PASSIVE_MONITOR, ELEVATE, TASK_OPTICAL, TASK_SAR, ESCALATE
from custody.decision_trace import DecisionTrace
from custody.fusion import FusionAssessment
from custody.models import TrackState
from custody.sensors import PassWindow, next_pass_window


# ---------------------------------------------------------------------------
# Sensor catalog mappings
# ---------------------------------------------------------------------------

# Abstract sensor label → satellite id for orbital window lookup
_SAT_FOR_SENSOR: dict[str, str] = {
    "OPTICAL": "SAT-A",
    "SAR":     "SAT-B",
}

# Candidate sensor ordering per decision action
_ACTION_CANDIDATES: dict[str, list[str]] = {
    TASK_SAR:        ["SAR", "OPTICAL", "MONITOR"],
    TASK_OPTICAL:    ["OPTICAL", "SAR", "MONITOR"],
    ELEVATE:         ["OPTICAL", "SAR", "AIS_REFRESH", "MONITOR"],
    PASSIVE_MONITOR: ["MONITOR", "AIS_REFRESH"],
    ESCALATE:        ["SAR", "OPTICAL", "MONITOR"],
}

# Fallback sensors per primary sensor (primary excluded)
_SENSOR_FALLBACKS: dict[str, list[str]] = {
    "SAR":         ["OPTICAL", "MONITOR"],
    "OPTICAL":     ["SAR", "AIS_REFRESH"],
    "AIS_REFRESH": ["OPTICAL", "MONITOR"],
    "MONITOR":     ["AIS_REFRESH"],
}

# Synthetic window durations for non-orbital sensors
_AIS_REFRESH_WINDOW_MINUTES   = 30
_MONITOR_WINDOW_HOURS         = 2


# ---------------------------------------------------------------------------
# Core data contract
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TaskRecommendation:
    """A concrete sensing/collection option for one entity.

    Attributes:
        task_id:       Human-readable unique identifier for this task
                       (e.g. ``task_v001_sar_20260324t1412z``).
        entity_id:     Entity being monitored.
        sensor:        Abstract sensor label: OPTICAL, SAR, AIS_REFRESH,
                       or MONITOR.
        window_start:  Start of the recommended collection window (UTC).
        window_end:    End of the recommended collection window (UTC).
        expected_value: Estimated information gain in [0, 1].  Higher =
                       more worth doing.
        reason:        Operator-grade one-sentence justification.
        fallbacks:     Ordered alternative sensors if this task fails.
        rank:          Position in the sorted queue (1 = highest value).
    """
    task_id: str
    entity_id: str
    sensor: str
    window_start: datetime
    window_end: datetime
    expected_value: float
    reason: str
    fallbacks: list[str]
    rank: int


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _clamp01(v: float) -> float:
    return min(max(v, 0.0), 1.0)


def _make_task_id(entity_id: str, sensor: str, window_start: datetime) -> str:
    """Generate a human-readable task identifier."""
    ts = window_start.strftime("%Y%m%dt%H%Mz")
    slug = entity_id.lower().replace("-", "").replace(" ", "_")
    return f"task_{slug}_{sensor.lower()}_{ts}"


def _candidate_sensors_for_action(
    decision: Decision,
    fusion_assessment: FusionAssessment,
) -> list[str]:
    """Return the ordered candidate sensor list for this decision action.

    For PASSIVE_MONITOR, AIS_REFRESH is added when uncertainty is elevated.
    For ESCALATE, the candidate order puts the FA-recommended source first.

    Args:
        decision:          The current Decision.
        fusion_assessment: The current FusionAssessment.

    Returns:
        Ordered list of sensor label strings.
    """
    action = decision.action

    if action == ESCALATE:
        rec = fusion_assessment.recommended_confirming_source
        primary   = "SAR" if rec == "SAR" else "OPTICAL"
        secondary = "OPTICAL" if primary == "SAR" else "SAR"
        return [primary, secondary, "MONITOR"]

    if action == PASSIVE_MONITOR:
        candidates = ["MONITOR"]
        if fusion_assessment.uncertainty > 0.40:
            candidates.append("AIS_REFRESH")
        return candidates

    return list(_ACTION_CANDIDATES.get(action, ["MONITOR"]))


def _select_window(
    sensor: str,
    planner_context,
    record: dict,
    track: TrackState,
) -> Optional[tuple[datetime, datetime, float]]:
    """Determine the best available window for a sensor.

    For orbital sensors (OPTICAL, SAR), queries the orbital pass catalog
    using the record's lat/lon.  If no pass exists within the horizon the
    sensor is considered unavailable and None is returned.

    For AIS_REFRESH and MONITOR, synthetic windows starting at the record
    time are always returned.

    Args:
        sensor:          Abstract sensor label.
        planner_context: Optional DecisionTrace (reserved for future use).
        record:          Timeline record dict (provides lat, lon, time).
        track:           TrackState (reserved for future use).

    Returns:
        ``(window_start, window_end, tts_seconds)`` or None if the sensor
        has no accessible window.
    """
    now: datetime = record.get("time")
    lat = record.get("lat")
    lon = record.get("lon")

    if sensor in _SAT_FOR_SENSOR:
        sat_id = _SAT_FOR_SENSOR[sensor]
        if lat is not None and lon is not None:
            pw: Optional[PassWindow] = next_pass_window(sat_id, lat, lon, now)
            if pw is None:
                return None   # No pass in horizon — sensor unavailable
            return pw.start_time, pw.end_time, pw.time_to_start_seconds
        # No position — fall back to planner context TTS if available
        if isinstance(planner_context, DecisionTrace):
            tts = planner_context.task_value.nearest_pass_time_to_start_seconds
            if tts is not None and tts >= 0:
                start = now + timedelta(seconds=tts)
                end   = start + timedelta(minutes=10)
                return start, end, float(tts)
        # No orbital info at all — sensor unavailable
        return None

    if sensor == "AIS_REFRESH":
        end = now + timedelta(minutes=_AIS_REFRESH_WINDOW_MINUTES)
        return now, end, 0.0

    if sensor == "MONITOR":
        end = now + timedelta(hours=_MONITOR_WINDOW_HOURS)
        return now, end, 0.0

    # Unknown sensor — unavailable
    return None


def _compute_sensor_fit(
    sensor: str,
    decision: Decision,
    fusion_assessment: FusionAssessment,
    record: dict,
    track: TrackState,
) -> float:
    """Score how well this sensor matches the decision and evidence picture.

    Returns a value in [0, 1].  Higher = stronger match.

    Args:
        sensor:            Abstract sensor label.
        decision:          The current Decision.
        fusion_assessment: The current FusionAssessment.
        record:            Timeline record dict.
        track:             TrackState.

    Returns:
        sensor_fit in [0, 1].
    """
    action = decision.action
    rec    = fusion_assessment.recommended_confirming_source

    # Perfect fit: sensor matches both the decision action and FA recommendation
    if sensor == "SAR" and (action == TASK_SAR or rec == "SAR"):
        return 1.0
    if sensor == "OPTICAL" and (action == TASK_OPTICAL or rec == "OPTICAL"):
        return 1.0
    if sensor == "AIS_REFRESH" and rec == "AIS":
        return 0.90

    # Strong fit: sensor is the natural primary for the action
    if sensor in ("SAR", "OPTICAL") and action == ESCALATE:
        return 0.80
    if sensor == "AIS_REFRESH" and fusion_assessment.uncertainty > 0.50:
        return 0.65
    if sensor == "MONITOR" and action == PASSIVE_MONITOR:
        return 0.80

    # Moderate fit: secondary sensor for ELEVATE
    if sensor in ("SAR", "OPTICAL") and action == ELEVATE:
        return 0.55
    if sensor == "AIS_REFRESH" and action == ELEVATE:
        return 0.45

    # MONITOR as a backstop for active tasking actions
    if sensor == "MONITOR":
        return 0.20

    return 0.40


def _compute_timing_score(tts_seconds: float) -> float:
    """Score window timing — sooner is better.

    Args:
        tts_seconds: Time-to-start in seconds.  0 means already in view.

    Returns:
        timing_score in [0, 1].
    """
    if tts_seconds <= 0:
        return 1.0
    if tts_seconds <= config.HOLD_LOOKAHEAD_THRESHOLD_SECONDS:   # default 1800s
        return 0.85
    if tts_seconds <= 3_600:
        return 0.65
    if tts_seconds <= 4 * 3_600:
        return 0.40
    return 0.20


def _compute_expected_value(
    sensor: str,
    decision: Decision,
    fusion_assessment: FusionAssessment,
    window_start: datetime,
    window_end: datetime,
    record: dict,
    track: TrackState,
) -> float:
    """Estimate information gain for this sensor/window combination.

    Formula (bounded [0, 1]):
        0.40 × decision.priority
      + 0.25 × decision.confidence
      + 0.20 × sensor_fit
      + 0.15 × timing_score

    Args:
        sensor:        Abstract sensor label.
        decision:      The current Decision.
        fusion_assessment: FusionAssessment for context.
        window_start:  Start of the candidate window.
        window_end:    End of the candidate window.
        record:        Timeline record dict.
        track:         TrackState.

    Returns:
        expected_value in [0, 1].
    """
    sensor_fit   = _compute_sensor_fit(sensor, decision, fusion_assessment, record, track)
    now: datetime = record.get("time")
    tts = max((window_start - now).total_seconds(), 0.0) if now else 0.0
    timing_score = _compute_timing_score(tts)

    return round(_clamp01(
        0.40 * decision.priority
        + 0.25 * decision.confidence
        + 0.20 * sensor_fit
        + 0.15 * timing_score
    ), 3)


def _build_reason(
    sensor: str,
    decision: Decision,
    fusion_assessment: FusionAssessment,
    window_start: datetime,
    record: dict,
    tts_seconds: float,
) -> str:
    """Generate an operator-grade one-sentence justification.

    Args:
        sensor:            Abstract sensor label.
        decision:          The current Decision.
        fusion_assessment: The current FusionAssessment.
        window_start:      Start of the recommended window.
        record:            Timeline record dict (for current time).
        tts_seconds:       Time-to-start in seconds.

    Returns:
        A single plain-English sentence.
    """
    tts_min = max(int(tts_seconds / 60), 0)

    if sensor == "SAR":
        if tts_seconds <= 0:
            return (
                "SAR is the highest-value confirming source and the satellite "
                "is currently overhead — task now."
            )
        if tts_min <= 5:
            return (
                f"SAR is the highest-value confirming source and an access "
                f"window opens in approximately {tts_min} minutes."
            )
        return (
            f"SAR offers all-weather confirmation capability with an access "
            f"window in approximately {tts_min} minutes."
        )

    if sensor == "OPTICAL":
        if tts_seconds <= 0:
            return (
                "Optical confirmation is available now and is the best match "
                "for the current evidence gap."
            )
        if tts_min <= 5:
            return (
                f"Optical confirmation is available in approximately "
                f"{tts_min} minutes and best matches the current evidence gap."
            )
        return (
            f"Optical tasking provides direct visual confirmation with an "
            f"access window in approximately {tts_min} minutes."
        )

    if sensor == "AIS_REFRESH":
        return (
            "AIS refresh is the fastest way to reduce custody uncertainty "
            "and restore track confidence."
        )

    if sensor == "MONITOR":
        if decision.action == PASSIVE_MONITOR:
            return (
                "Passive monitoring is sufficient because fused significance "
                "remains low and no active tasking is warranted."
            )
        return (
            "Continued monitoring maintains situational awareness while "
            "awaiting a higher-value collection opportunity."
        )

    return f"Collection via {sensor} is the best available option given current conditions."


def _fallbacks_for_sensor(sensor: str, decision: Decision) -> list[str]:
    """Return ordered fallback sensor alternatives, excluding the primary.

    Args:
        sensor:   Primary sensor label.
        decision: The current Decision (reserved for future context).

    Returns:
        Ordered list of alternative sensor label strings.
    """
    return list(_SENSOR_FALLBACKS.get(sensor, ["MONITOR"]))


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

def build_task_recommendations(
    decision: Decision,
    fusion_assessment: FusionAssessment,
    record: dict,
    track: TrackState,
    planner_context=None,
) -> list[TaskRecommendation]:
    """Build a ranked queue of TaskRecommendation objects.

    Generates candidate tasks from the decision action family, scores each
    by expected information value, sorts descending, and assigns ranks.
    Sensors with no available window within the orbital horizon are omitted.
    MONITOR is always available so the queue is never empty.

    Args:
        decision:          The current Decision for this entity/timestep.
        fusion_assessment: The current FusionAssessment (same timestep).
        record:            Timeline record dict (same record used for FA/Decision).
        track:             TrackState for this entity.
        planner_context:   Optional DecisionTrace from this timestep's planner.
                           Used as an orbital-window fallback when lat/lon are
                           absent from the record.  Passing None is safe.

    Returns:
        List of TaskRecommendation sorted by expected_value descending,
        with rank=1 being the highest-value option.  Never empty.
    """
    candidates = _candidate_sensors_for_action(decision, fusion_assessment)

    unranked: list[TaskRecommendation] = []
    for sensor in candidates:
        window = _select_window(sensor, planner_context, record, track)
        if window is None:
            continue   # No accessible window — omit this sensor

        window_start, window_end, tts_seconds = window

        expected_value = _compute_expected_value(
            sensor, decision, fusion_assessment,
            window_start, window_end, record, track,
        )
        reason    = _build_reason(
            sensor, decision, fusion_assessment, window_start, record, tts_seconds,
        )
        fallbacks = _fallbacks_for_sensor(sensor, decision)
        task_id   = _make_task_id(fusion_assessment.entity_id, sensor, window_start)

        unranked.append(TaskRecommendation(
            task_id=task_id,
            entity_id=fusion_assessment.entity_id,
            sensor=sensor,
            window_start=window_start,
            window_end=window_end,
            expected_value=expected_value,
            reason=reason,
            fallbacks=fallbacks,
            rank=0,  # assigned after sort
        ))

    # Sort descending by expected_value, then assign final ranks
    unranked.sort(key=lambda r: r.expected_value, reverse=True)

    return [
        TaskRecommendation(
            task_id=r.task_id,
            entity_id=r.entity_id,
            sensor=r.sensor,
            window_start=r.window_start,
            window_end=r.window_end,
            expected_value=r.expected_value,
            reason=r.reason,
            fallbacks=r.fallbacks,
            rank=i + 1,
        )
        for i, r in enumerate(unranked)
    ]
