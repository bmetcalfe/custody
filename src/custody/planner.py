import math
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Union

from custody.config import (
    ANOMALY_THRESHOLD,
    CRITICAL_ANOMALY_THRESHOLD,
    CUSTODY_TASK_CONFIDENCE_THRESHOLD,
    FAILURE_URGENCY_BOOST,
    FAILURE_URGENCY_MAX_COUNT,
    FRESHNESS_SUPPRESSION,
    HOLD_LOOKAHEAD_BOOST,
    HOLD_LOOKAHEAD_THRESHOLD_SECONDS,
    REVISIT_DECAY_HOURS,
    TASK_VALUE_THRESHOLD,
    WORSENING_BOOST,
)
from custody.collection import apply_collection_effect, attempt_collection
from custody.models import TrackState
from custody.sensors import SensorOpportunity


@dataclass(frozen=True)
class CollectionDecision:
    """Result of one collection planning step.

    Attributes:
        action:            "NONE", "HOLD", "TASK", "NO_SENSOR", or "PREEMPTED".
        action_reason:     Human-readable explanation for the action taken.
        sensor_id:         Sensor identifier when action is "TASK", else None.
        sensor_type:       Sensor type string when action is "TASK", else None.
        collection_result: "SUCCESS" or "FAILED" when action is "TASK",
                           else None.
        success:           True only when action is "TASK" and the collection
                           attempt succeeded.
        new_uncertainty:   Post-collection uncertainty_km when success is True,
                           else None.
        lookahead_boost:   The HOLD_LOOKAHEAD_BOOST value applied by the
                           orbital-pass lookahead logic at this step, or 0.0
                           if the lookahead did not fire.  Passed directly to
                           build_decision_trace so the trace records the exact
                           planner-computed value rather than re-deriving it.
        nearest_pass_tts:  Seconds to the nearest upcoming orbital pass that
                           triggered the lookahead, or None.  Companion to
                           lookahead_boost; also passed to the trace builder.
        hold_reason:       "freshness" when HOLD is due to task-value / freshness
                           suppression; "lookahead" when HOLD is due to an
                           imminent orbital pass with no sensor currently
                           available; None for all non-HOLD actions.
    """
    action: str
    action_reason: str
    sensor_id: Optional[str]
    sensor_type: Optional[str]
    collection_result: Optional[str]
    success: bool
    new_uncertainty: Optional[float]
    lookahead_boost: float = 0.0
    nearest_pass_tts: Optional[float] = None
    hold_reason: Optional[str] = None


def compute_target_priority(
    score: float,
    confidence: float,
    compound_boost: float = 0.0,
    return_breakdown: bool = False,
):
    """Return a priority value for cross-vessel tasking arbitration.

    Higher return value = task sooner.  Range is approximately [0, 1] but
    can exceed 1.0 when score is above CRITICAL_ANOMALY_THRESHOLD.

    Weighting rationale:
      - Anomaly score (55 %) is the primary driver: a vessel behaving
        anomalously should be tasked ahead of a nominal one.
      - Low custody confidence (35 %) increases priority: poor track
        quality is operationally urgent in its own right.
      - Compound boost (10 %) is a small escalation signal from multi-
        signal behavioral patterns.  It breaks ties and provides a soft
        escalation path without overriding the two primary drivers.
        Pass max(s.confidence for s in active_compounds) or 0.0; the
        planner does not import custody.compounds directly.

    Args:
        score:            Current composite anomaly score.
        confidence:       Current custody confidence in [0, 1].
        compound_boost:   Maximum active compound signal confidence in [0, 1].
                          Defaults to 0.0 (no compound signals active).
        return_breakdown: If True, return (total, PriorityBreakdown) instead
                          of just the total.  Defaults to False (unchanged
                          behavior).

    Returns:
        Priority float when return_breakdown is False.
        (float, PriorityBreakdown) tuple when return_breakdown is True.
    """
    from custody.decision_trace import PriorityBreakdown  # local import avoids cycle

    norm_anomaly = score / CRITICAL_ANOMALY_THRESHOLD  # primary driver
    norm_no_conf = 1.0 - confidence                    # low confidence → high priority
    total = norm_anomaly * 0.55 + norm_no_conf * 0.35 + compound_boost * 0.10

    if not return_breakdown:
        return total

    return total, PriorityBreakdown(
        anomaly_norm=norm_anomaly,
        uncertainty=norm_no_conf,
        compound_boost=compound_boost,
        total=total,
    )


def compute_task_value(
    score: float,
    confidence: float,
    hours_since_last_collection: float | None,
    last_collection_anomaly_score: float | None,
    compound_boost: float = 0.0,
    lookahead_boost: float = 0.0,
    nearest_pass_tts: Optional[float] = None,
    consecutive_failures: int = 0,
    return_breakdown: bool = False,
):
    """Return a task-value score indicating how urgently this vessel should be re-tasked.

    Higher return value = stronger case for tasking.  A value below
    TASK_VALUE_THRESHOLD is treated as HOLD in plan_collection.

    Model:
      base   = (score / CRITICAL) * 0.55 + (1 - confidence) * 0.35 + compound_boost * 0.10
             — anomaly is the primary driver; low custody confidence adds urgency.

      When a prior collection exists:
        freshness  = exp(-hours_since / REVISIT_DECAY_HOURS)
                   — 1.0 immediately after collection, decays toward 0 over time.
        worsening  = max(0, score - last_collection_anomaly_score)
                   — how much the anomaly has grown since the last collection.
        task_value = base - freshness * FRESHNESS_SUPPRESSION + worsening * WORSENING_BOOST
                   + lookahead_boost

      When no prior collection exists:
        task_value = base + lookahead_boost  (freshness suppression is not applied)

    Args:
        score:                        Current composite anomaly score.
        confidence:                   Current custody confidence in [0, 1].
        hours_since_last_collection:  Hours elapsed since the last successful
                                      collection, or None if never collected.
        last_collection_anomaly_score: Anomaly score recorded at the last
                                      collection, or None if never collected.
        compound_boost:               Maximum active compound signal confidence
                                      in [0, 1].  Defaults to 0.0.
        lookahead_boost:              Additional boost applied when an orbital
                                      pass is imminent and no sensor is currently
                                      available.  Supplied by plan_collection;
                                      callers should leave at the default 0.0.
        nearest_pass_tts:             Seconds to the nearest upcoming orbital
                                      pass that triggered the lookahead boost,
                                      or None.  Stored in the breakdown for
                                      diagnostic purposes only.
        return_breakdown:             If True, return (total, TaskValueBreakdown)
                                      instead of just the total.  Defaults to
                                      False (unchanged behavior).

    Returns:
        Task-value float when return_breakdown is False.
        (float, TaskValueBreakdown) tuple when return_breakdown is True.
    """
    from custody.decision_trace import TaskValueBreakdown  # local import avoids cycle

    base = (
        (score / CRITICAL_ANOMALY_THRESHOLD) * 0.55
        + (1.0 - confidence) * 0.35
        + compound_boost * 0.10
    )

    # Failure urgency: consecutive failed collections bias toward retasking.
    # Capped at FAILURE_URGENCY_MAX_COUNT failures × FAILURE_URGENCY_BOOST per.
    failure_boost = min(consecutive_failures, FAILURE_URGENCY_MAX_COUNT) * FAILURE_URGENCY_BOOST

    if hours_since_last_collection is None:
        freshness_decay = 0.0
        worsening_boost_amount = 0.0
        total = base + lookahead_boost + failure_boost
    else:
        freshness = math.exp(-hours_since_last_collection / REVISIT_DECAY_HOURS)
        worsening = max(0.0, score - (last_collection_anomaly_score or 0.0))
        freshness_decay = freshness * FRESHNESS_SUPPRESSION
        worsening_boost_amount = worsening * WORSENING_BOOST
        total = base - freshness_decay + worsening_boost_amount + lookahead_boost + failure_boost

    if not return_breakdown:
        return total

    return total, TaskValueBreakdown(
        base=base,
        freshness_decay=freshness_decay,
        worsening_boost=worsening_boost_amount,
        total=total,
        hold_threshold=TASK_VALUE_THRESHOLD,
        hold_eligible=(total < TASK_VALUE_THRESHOLD),
        lookahead_boost=lookahead_boost,
        nearest_pass_time_to_start_seconds=nearest_pass_tts,
        failure_boost=failure_boost,
    )


def plan_collection(
    track: TrackState,
    score: float,
    confidence: float,
    breakdown: dict,
    current_time: datetime,
    opportunities: list[SensorOpportunity],
    preempted: bool = False,
    compound_boost: float = 0.0,
    observer_lat: Optional[float] = None,
    observer_lon: Optional[float] = None,
) -> CollectionDecision:
    """Decide and execute one collection planning step for a single vessel.

    The caller is responsible for supplying ``opportunities`` — this function
    does *not* fetch sensors internally.  In a multi-vessel simulation the
    caller should:
      1. Fetch each vessel's sensor pool independently (orbit-access filtered
         from that vessel's position); record its size as ``sensor_access_count``.
      2. Sort vessels by ``compute_target_priority`` (highest first).
      3. Maintain a ``claimed_sensor_ids`` set across vessels for the timestep.
      4. For each vessel in priority order: exclude claimed ids from its pool,
         pass the remaining opportunities here, set ``preempted=True`` when
         the vessel's initial pool was non-empty but is now fully claimed.
      5. Add the chosen sensor_id to ``claimed_sensor_ids`` after a TASK.

    Action semantics:
      NONE      — tasking is not warranted (low anomaly + adequate confidence).
      HOLD      — task value is below TASK_VALUE_THRESHOLD; freshness
                  decay has not yet restored full re-task urgency.
                  Value recovers over time and increases with anomaly
                  worsening (see compute_task_value).
      TASK      — a sensor was selected and an attempt was made; check
                  ``decision.success`` and ``decision.collection_result``.
      NO_SENSOR — tasking is warranted but the passed-in ``opportunities``
                  list is empty and no sensors were globally available
                  (i.e., ``preempted=False``).
      PREEMPTED — tasking is warranted and sensors were available globally,
                  but all were already claimed by higher-priority vessels
                  before this vessel's turn (``preempted=True`` with an
                  empty ``opportunities`` list).

    When a collection attempt succeeds, ``track`` is updated in place via
    ``track.record_collection``.

    Args:
        track:        Per-track custody state (mutated on successful collection).
        score:        Current composite anomaly score.
        confidence:   Current custody confidence (before this step's collection).
        breakdown:    Anomaly breakdown dict from ``anomaly_breakdown()``.
        current_time: Timestamp of the current step (used for collection timing
                      and the revisit-suppression window check).
        opportunities: Sensor opportunities available to this vessel for this
                      step.  Normally the caller passes the *remaining* pool
                      after higher-priority vessels have claimed their sensors.
                      Pass ``[]`` to express no available sensors.
        preempted:      Set True when ``opportunities`` is empty because sensors
                        were consumed by higher-priority vessels at this timestep,
                        not because no sensors were globally scheduled.  Changes
                        the returned action from ``NO_SENSOR`` to ``PREEMPTED``.
                        Has no effect when ``opportunities`` is non-empty or when
                        the vessel would return NONE / HOLD anyway.
        compound_boost: Maximum active compound signal confidence in [0, 1].
                        Passed through to ``compute_task_value`` so that active
                        compound patterns (e.g. LOITERING_NEAR_ZONE) can lift
                        the task value above the HOLD threshold.  Defaults to
                        0.0 (no compound signals active).
        observer_lat:   Observer geodetic latitude in degrees (+N), or None.
                        Required for orbital-pass lookahead bias.  When None,
                        the lookahead feature is disabled and behaviour is
                        identical to pre-lookahead versions.
        observer_lon:   Observer geodetic longitude in degrees (+E), or None.
                        Required for orbital-pass lookahead bias.

    Returns:
        A ``CollectionDecision`` describing what action was taken.
    """
    # ── Orbital-pass lookahead ─────────────────────────────────────────────
    # Computed up front so it can influence both the task-value HOLD gate
    # and the fallback NO_SENSOR → HOLD conversion.
    # Conditions: sensor pool is empty (not due to preemption), and observer
    # position is known.  Applies only to orbital satellites (SAT-A, SAT-B);
    # schedule-based sensors (A1/B1/C1) are not forecast here.
    _lookahead_boost = 0.0
    _nearest_pass_tts: Optional[float] = None
    if (
        not opportunities
        and not preempted
        and observer_lat is not None
        and observer_lon is not None
    ):
        from custody.sensors import nearest_orbital_pass
        pw = nearest_orbital_pass(observer_lat, observer_lon, current_time)
        if pw is not None and pw.time_to_start_seconds <= HOLD_LOOKAHEAD_THRESHOLD_SECONDS:
            _lookahead_boost = HOLD_LOOKAHEAD_BOOST
            _nearest_pass_tts = pw.time_to_start_seconds

    should_consider_tasking = score > ANOMALY_THRESHOLD or confidence < CUSTODY_TASK_CONFIDENCE_THRESHOLD

    if not should_consider_tasking:
        return CollectionDecision(
            action="NONE",
            action_reason="",
            sensor_id=None,
            sensor_type=None,
            collection_result=None,
            success=False,
            new_uncertainty=None,
            lookahead_boost=_lookahead_boost,
            nearest_pass_tts=_nearest_pass_tts,
        )

    hours_since = track.hours_since_collection(current_time)
    if hours_since is not None:
        task_value = compute_task_value(
            score, confidence, hours_since, track.last_collection_anomaly_score,
            compound_boost, lookahead_boost=_lookahead_boost,
            consecutive_failures=track.consecutive_failures,
        )
        if task_value < TASK_VALUE_THRESHOLD:
            return CollectionDecision(
                action="HOLD",
                action_reason=(
                    f"task value {task_value:.3f} below threshold — "
                    "recent collection adequate"
                ),
                sensor_id=None,
                sensor_type=None,
                collection_result=None,
                success=False,
                new_uncertainty=None,
                lookahead_boost=_lookahead_boost,
                nearest_pass_tts=_nearest_pass_tts,
                hold_reason="freshness",
            )

    if not opportunities:
        if _lookahead_boost > 0.0:
            # Orbital pass is imminent; HOLD is more informative than NO_SENSOR.
            return CollectionDecision(
                action="HOLD",
                action_reason=(
                    f"orbital pass in {round(_nearest_pass_tts / 60.0, 1)} min "
                    "— holding for orbital window"
                ),
                sensor_id=None,
                sensor_type=None,
                collection_result=None,
                success=False,
                new_uncertainty=None,
                lookahead_boost=_lookahead_boost,
                nearest_pass_tts=_nearest_pass_tts,
                hold_reason="lookahead",
            )
        if preempted:
            return CollectionDecision(
                action="PREEMPTED",
                action_reason="sensors claimed by higher-priority vessels",
                sensor_id=None,
                sensor_type=None,
                collection_result=None,
                success=False,
                new_uncertainty=None,
            )
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

    track.record_failure(current_time)
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
    # Anomaly urgency: normalized to CRITICAL_ANOMALY_THRESHOLD so the scale is
    # consistent with compute_target_priority and compute_task_value.
    # Weight 2.0 on norm_anomaly ∈ [0, 1] keeps anomaly as the primary driver
    # without allowing raw score magnitude to dwarf capability terms.
    norm_anomaly = anomaly_score / CRITICAL_ANOMALY_THRESHOLD
    score = (
        norm_anomaly * 2.0           # anomaly urgency (primary)
        + (1 - custody_confidence) * 3.0  # custody fit
        + opportunity.success_prob * 1.5  # sensor capability
        - opportunity.cost * 0.5          # cost penalty
    )

    # Modest resolution preference: "high" is a secondary tiebreaker (0.3).
    # Kept well below sensor-type and breakdown bonuses (0.5–1.5) so it never
    # overrides a clearly better fit.
    if opportunity.resolution == "high":
        score += 0.3

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