from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from custody.config import ANOMALY_THRESHOLD
from custody.collection import apply_collection_effect, attempt_collection
from custody.models import TrackState
from custody.sensors import get_sensor_opportunities


@dataclass(frozen=True)
class CollectionDecision:
    """Result of one collection planning step.

    Attributes:
        action:            "NONE", "HOLD", "TASK", or "NO_SENSOR".
        action_reason:     Human-readable explanation for the action taken.
        sensor_id:         Sensor identifier when action is "TASK", else None.
        sensor_type:       Sensor type string when action is "TASK", else None.
        collection_result: "SUCCESS" or "FAILED" when action is "TASK",
                           else None.
        success:           True only when action is "TASK" and the collection
                           attempt succeeded.
        new_uncertainty:   Post-collection uncertainty_km when success is True,
                           else None.
    """
    action: str
    action_reason: str
    sensor_id: Optional[str]
    sensor_type: Optional[str]
    collection_result: Optional[str]
    success: bool
    new_uncertainty: Optional[float]


def plan_collection(
    track: TrackState,
    score: float,
    confidence: float,
    breakdown: dict,
    current_time: datetime,
) -> CollectionDecision:
    """Decide and execute one collection planning step.

    Evaluates whether to task a sensor, hold on a recent collection, or take
    no action.  When a collection attempt succeeds, ``track`` is updated in
    place via ``track.record_collection``.

    Args:
        track:        Per-track custody state (mutated on successful collection).
        score:        Current composite anomaly score.
        confidence:   Current custody confidence (before this step's collection).
        breakdown:    Anomaly breakdown dict from ``anomaly_breakdown()``.
        current_time: Timestamp of the current step (used for sensor lookup and
                      collection timing).

    Returns:
        A ``CollectionDecision`` describing what action was taken.
    """
    should_consider_tasking = score > ANOMALY_THRESHOLD or confidence < 0.7

    if not should_consider_tasking:
        return CollectionDecision(
            action="NONE",
            action_reason="",
            sensor_id=None,
            sensor_type=None,
            collection_result=None,
            success=False,
            new_uncertainty=None,
        )

    hours_since = track.hours_since_collection(current_time)
    recent_collection = hours_since is not None and hours_since < 2
    anomaly_worsened = score > (track.last_collection_anomaly_score + 0.25)

    should_hold = (
        recent_collection
        and confidence > 0.75
        and not anomaly_worsened
    )

    if should_hold:
        return CollectionDecision(
            action="HOLD",
            action_reason="recent successful collection still provides acceptable custody",
            sensor_id=None,
            sensor_type=None,
            collection_result=None,
            success=False,
            new_uncertainty=None,
        )

    opportunities = get_sensor_opportunities(current_time)

    if not opportunities:
        return CollectionDecision(
            action="NO_SENSOR",
            action_reason="no sensors available at this time",
            sensor_id=None,
            sensor_type=None,
            collection_result=None,
            success=False,
            new_uncertainty=None,
        )

    ranked = choose_best_opportunity(opportunities, score, confidence, breakdown)
    best, _best_score, reasons = ranked[0]
    action_reason = ", ".join(reasons) if reasons else ""

    success = attempt_collection(best, breakdown)

    if success:
        new_uncertainty = apply_collection_effect(track.uncertainty_km, best)
        track.record_collection(current_time, score, new_uncertainty)
        return CollectionDecision(
            action="TASK",
            action_reason=action_reason,
            sensor_id=best.sensor_id,
            sensor_type=best.sensor_type,
            collection_result="SUCCESS",
            success=True,
            new_uncertainty=new_uncertainty,
        )

    return CollectionDecision(
        action="TASK",
        action_reason=action_reason,
        sensor_id=best.sensor_id,
        sensor_type=best.sensor_type,
        collection_result="FAILED",
        success=False,
        new_uncertainty=None,
    )


def score_opportunity(opportunity, anomaly_score, custody_confidence, anomaly_breakdown):
    score = (
        anomaly_score * 2.0
        + (1 - custody_confidence) * 3.0
        + opportunity.success_prob * 1.5
        - opportunity.cost * 0.5
    )

    reasons = []

    if opportunity.sensor_type == "fast_revisit":
        score += 0.5
        reasons.append("good for rapid reacquisition")

    if anomaly_breakdown["sensitive_zone"].score > 0:
        score += 1.5
        reasons.append("target operating in sensitive zone")

    if anomaly_breakdown["sensitive_zone"].score > 0 and opportunity.sensor_type == "high_resolution":
        score += 1.0
        reasons.append("high resolution favored in sensitive zone")

    if anomaly_breakdown["loitering"].score > 0 and opportunity.sensor_type == "high_resolution":
        score += 0.5
        reasons.append("better for characterizing suspicious loitering")

    if custody_confidence < 0.6 and opportunity.sensor_type == "fast_revisit":
        score += 1.0
        reasons.append("custody risk favors fast revisit")

    if anomaly_score > 1.2 and opportunity.sensor_type == "all_weather":
        score += 0.5
        reasons.append("reliable fallback for elevated anomaly state")

    return score, reasons


def choose_best_opportunity(opportunities, anomaly_score, custody_confidence, anomaly_breakdown):
    scored = []

    for opp in opportunities:
        score, reasons = score_opportunity(
            opp, anomaly_score, custody_confidence, anomaly_breakdown
        )
        scored.append((opp, score, reasons))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored