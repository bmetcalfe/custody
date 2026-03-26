"""
Structured decision-trace layer for the custody planner.

Every planner decision can be decomposed into three sequential questions:

  1. Priority  — how urgently should this vessel be tasked relative to others?
  2. Task value — is the computed urgency high enough to act now (HOLD gate)?
  3. Arbitration — which sensors were accessible, claimed, and ultimately chosen?

DecisionTrace bundles the answers to all three into a single frozen record so
that any planner decision can be answered with structured data:

  "Why did this vessel get (or not get) a sensor at this timestep?"

Usage
-----
Traces are produced by both run_simulation() and ingest_ais_track() and stored
as ``decision_trace`` in each timeline record.  Use traces_to_rows() to flatten
them for display:

    records = run_simulation()
    traces = [r["decision_trace"] for r in records]
    rows = traces_to_rows(traces)

Build traces with build_decision_trace() — both execution paths share this
helper so trace shape is identical regardless of source.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class DecisionInputs:
    """Raw inputs fed into priority and task-value scoring.

    Attributes:
        anomaly_score:      Composite anomaly score at this timestep.
        anomaly_norm:       anomaly_score / CRITICAL_ANOMALY_THRESHOLD.
        custody_confidence: Custody confidence in [0, 1].
        compound_boost:     Max active compound signal confidence in [0, 1].
        freshness:          exp(-hours_since / REVISIT_DECAY_HOURS) if a prior
                            collection exists, else 0.0.  Non-zero means recent
                            collection suppresses re-tasking urgency.
        sensor_access_count: Number of sensors in this vessel's pre-arbitration
                             pool (orbit-access filtered from its position).
        consecutive_failures: Number of consecutive failed collection attempts.
                              0 when no failures or after a successful collection.
    """
    anomaly_score: float
    anomaly_norm: float
    custody_confidence: float
    compound_boost: float
    freshness: float
    sensor_access_count: int
    consecutive_failures: int = 0


@dataclass(frozen=True)
class PriorityBreakdown:
    """Component breakdown of compute_target_priority.

    total = anomaly_norm * 0.55 + uncertainty * 0.35 + compound_boost * 0.10

    Attributes:
        anomaly_norm:   Normalized anomaly contribution.
        uncertainty:    1 - custody_confidence (low confidence → high priority).
        compound_boost: Compound signal contribution.
        total:          Final priority score used for vessel ranking.
    """
    anomaly_norm: float
    uncertainty: float
    compound_boost: float
    total: float


@dataclass(frozen=True)
class TaskValueBreakdown:
    """Component breakdown of compute_task_value.

    total = base - freshness_decay + worsening_boost + lookahead_boost + failure_boost

    Attributes:
        base:            Priority-based urgency before freshness adjustment.
        freshness_decay: Amount subtracted due to recent collection
                         (freshness * FRESHNESS_SUPPRESSION).  0.0 if no
                         prior collection.
        worsening_boost: Amount added due to anomaly growth since last
                         collection (worsening * WORSENING_BOOST).  0.0 if
                         no prior collection or anomaly has not grown.
        lookahead_boost: Amount added because an orbital pass is imminent
                         and no sensor is currently available
                         (HOLD_LOOKAHEAD_BOOST when triggered, else 0.0).
                         Non-zero here indicates the lookahead bias fired.
        failure_boost:   Amount added due to consecutive failed collection
                         attempts (min(failures, 3) * FAILURE_URGENCY_BOOST).
                         0.0 when no failures have occurred.
        nearest_pass_time_to_start_seconds:
                         Seconds until the nearest upcoming orbital pass
                         that triggered the lookahead bias, or None if the
                         bias did not fire.
        total:           Final task-value score (includes all boosts).
        hold_threshold:  TASK_VALUE_THRESHOLD used for the HOLD gate.
        hold_eligible:   True when total < hold_threshold (HOLD would fire
                         if should_consider_tasking is also True).
    """
    base: float
    freshness_decay: float
    worsening_boost: float
    total: float
    hold_threshold: float
    hold_eligible: bool
    lookahead_boost: float = 0.0
    nearest_pass_time_to_start_seconds: Optional[float] = None
    failure_boost: float = 0.0


@dataclass(frozen=True)
class ArbitrationBreakdown:
    """Sensor-pool state at the moment of assignment for this vessel.

    Attributes:
        accessible_sensor_ids:       Sensor IDs in this vessel's orbit-access
                                     filtered pool before arbitration.
        claimed_by_higher_priority:  Subset of accessible sensors already
                                     consumed by higher-priority vessels this
                                     timestep.
        final_sensor_pool:           Sensors remaining after claimed IDs are
                                     removed; these are what plan_collection
                                     actually sees.
        chosen_sensor_id:            The sensor selected (TASK action), or
                                     None if no sensor was assigned.
        result:                      The action string returned by
                                     plan_collection: "NONE", "HOLD", "TASK",
                                     "NO_SENSOR", or "PREEMPTED".
        hold_reason:                 "freshness" when result is HOLD due to
                                     task-value / freshness suppression;
                                     "lookahead" when HOLD is due to an
                                     imminent orbital pass; None for all
                                     non-HOLD actions.  Exact value from
                                     CollectionDecision — not recomputed here.
    """
    accessible_sensor_ids: tuple[str, ...]
    claimed_by_higher_priority: tuple[str, ...]
    final_sensor_pool: tuple[str, ...]
    chosen_sensor_id: Optional[str]
    result: str
    hold_reason: Optional[str] = None


@dataclass(frozen=True)
class DecisionTrace:
    """Complete structured record of one planner decision.

    Bundles all inputs and intermediate breakdowns so any decision can be
    explained without re-running the planner.

    Attributes:
        timestamp:   Simulation timestep at which the decision was made.
        vessel_id:   Identifier of the vessel being planned for.
        inputs:      Raw scoring inputs.
        priority:    Priority-score breakdown (vessel ranking).
        task_value:  Task-value breakdown (HOLD gate).
        arbitration: Sensor-pool state and assignment outcome.
    """
    timestamp: datetime
    vessel_id: str
    inputs: DecisionInputs
    priority: PriorityBreakdown
    task_value: TaskValueBreakdown
    arbitration: ArbitrationBreakdown


# ---------------------------------------------------------------------------
# Display helper
# ---------------------------------------------------------------------------

def traces_to_rows(traces: list[DecisionTrace]) -> list[dict]:
    """Flatten DecisionTrace objects into display-ready dicts.

    Each row corresponds to one trace and contains all fields needed for
    the Decision Traces UI table.

    Args:
        traces: List of DecisionTrace objects (e.g. from run_simulation()).

    Returns:
        List of flat dicts with the following keys:

          Time, Vessel, Action, Hold Reason, Chosen Sensor,
          Priority, Priority Anomaly, Priority Uncertainty, Priority Compound,
          Task Value, Task Base, Freshness Decay, Worsening Boost,
          Lookahead Boost, Failure Boost, Nearest Pass TTS, Hold Eligible,
          Consecutive Failures,
          Accessible Sensors, Claimed Higher, Final Pool, Sensor Access Count
    """
    rows = []
    for t in traces:
        rows.append({
            "Time": t.timestamp,
            "Vessel": t.vessel_id,
            "Action": t.arbitration.result,
            "Hold Reason": t.arbitration.hold_reason,
            "Chosen Sensor": t.arbitration.chosen_sensor_id,
            "Priority": round(t.priority.total, 4),
            "Priority Anomaly": round(t.priority.anomaly_norm, 4),
            "Priority Uncertainty": round(t.priority.uncertainty, 4),
            "Priority Compound": round(t.priority.compound_boost, 4),
            "Task Value": round(t.task_value.total, 4),
            "Task Base": round(t.task_value.base, 4),
            "Freshness Decay": round(t.task_value.freshness_decay, 4),
            "Worsening Boost": round(t.task_value.worsening_boost, 4),
            "Lookahead Boost": round(t.task_value.lookahead_boost, 4),
            "Failure Boost": round(t.task_value.failure_boost, 4),
            "Nearest Pass TTS": t.task_value.nearest_pass_time_to_start_seconds,
            "Hold Eligible": t.task_value.hold_eligible,
            "Consecutive Failures": t.inputs.consecutive_failures,
            "Accessible Sensors": ", ".join(t.arbitration.accessible_sensor_ids),
            "Claimed Higher": ", ".join(t.arbitration.claimed_by_higher_priority),
            "Final Pool": ", ".join(t.arbitration.final_sensor_pool),
            "Sensor Access Count": t.inputs.sensor_access_count,
        })
    return rows


# ---------------------------------------------------------------------------
# Shared trace construction helper
# ---------------------------------------------------------------------------

def build_decision_trace(
    timestamp,
    vessel_id: str,
    score: float,
    confidence: float,
    compound_boost: float,
    track,
    accessible_opportunities: list,
    claimed_sensor_ids: set,
    remaining_opportunities: list,
    decision_action: str,
    decision_sensor_id,
    lookahead_boost: float = 0.0,
    nearest_pass_tts: Optional[float] = None,
    hold_reason: Optional[str] = None,
) -> "DecisionTrace":
    """Construct a DecisionTrace from planner inputs and arbitration state.

    Shared by both the simulation path (run_simulation) and the AIS replay
    path (ingest_ais_track) so that the trace shape is identical regardless
    of which execution path produced the record.

    Uses local imports to avoid circular dependencies (planner imports from
    this module; this helper calls back into planner for breakdowns).

    The lookahead values must be passed in directly from the CollectionDecision
    returned by plan_collection — this function does NOT recompute them.
    plan_collection is the single source of truth for lookahead_boost and
    nearest_pass_tts; the trace records exactly what the planner used.

    Args:
        timestamp:               Simulation or observation timestamp.
        vessel_id:               Target identifier.
        score:                   Composite anomaly score at this step.
        confidence:              Custody confidence in [0, 1].
        compound_boost:          Max active compound signal confidence in [0, 1].
        track:                   TrackState (provides hours_since_collection
                                 and last_collection_anomaly_score).
        accessible_opportunities: Sensor pool BEFORE arbitration (all sensors
                                  reachable from this vessel's position).
        claimed_sensor_ids:      Sensor IDs already consumed by higher-priority
                                 vessels this timestep.  Pass an empty set for
                                 single-vessel contexts (AIS replay).
        remaining_opportunities: Sensor pool AFTER removing claimed IDs — the
                                 list actually passed to plan_collection.
        decision_action:         Action string returned by plan_collection (or
                                 the hardcoded action for the first AIS record).
        decision_sensor_id:      Sensor id when action is "TASK", else None.
        lookahead_boost:         Exact value from CollectionDecision.lookahead_boost.
                                 0.0 when lookahead did not fire.
        nearest_pass_tts:        Exact value from CollectionDecision.nearest_pass_tts.
                                 None when lookahead did not fire.
        hold_reason:             Exact value from CollectionDecision.hold_reason.
                                 "freshness", "lookahead", or None.

    Returns:
        A fully populated DecisionTrace.
    """
    import math  # standard library — safe to import locally
    # Local imports break the planner → decision_trace → planner cycle.
    from custody.config import CRITICAL_ANOMALY_THRESHOLD, REVISIT_DECAY_HOURS
    from custody.planner import compute_target_priority, compute_task_value

    _, priority_bd = compute_target_priority(
        score, confidence, compound_boost, return_breakdown=True
    )

    hours_since = track.hours_since_collection(timestamp)
    _cons_failures = getattr(track, "consecutive_failures", 0)
    _, task_value_bd = compute_task_value(
        score, confidence, hours_since, track.last_collection_anomaly_score,
        compound_boost,
        lookahead_boost=lookahead_boost,
        nearest_pass_tts=nearest_pass_tts,
        consecutive_failures=_cons_failures,
        return_breakdown=True,
    )

    freshness = 0.0
    if hours_since is not None:
        freshness = math.exp(-hours_since / REVISIT_DECAY_HOURS)

    accessible_ids = tuple(o.sensor_id for o in accessible_opportunities)
    claimed_ids = tuple(
        o.sensor_id for o in accessible_opportunities
        if o.sensor_id in claimed_sensor_ids
    )

    return DecisionTrace(
        timestamp=timestamp,
        vessel_id=vessel_id,
        inputs=DecisionInputs(
            anomaly_score=score,
            anomaly_norm=score / CRITICAL_ANOMALY_THRESHOLD,
            custody_confidence=confidence,
            compound_boost=compound_boost,
            freshness=freshness,
            sensor_access_count=len(accessible_opportunities),
            consecutive_failures=_cons_failures,
        ),
        priority=priority_bd,
        task_value=task_value_bd,
        arbitration=ArbitrationBreakdown(
            accessible_sensor_ids=accessible_ids,
            claimed_by_higher_priority=claimed_ids,
            final_sensor_pool=tuple(o.sensor_id for o in remaining_opportunities),
            chosen_sensor_id=decision_sensor_id,
            result=decision_action,
            hold_reason=hold_reason,
        ),
    )
