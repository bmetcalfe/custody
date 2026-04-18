"""
Mission reasoning layer for custody.

Turns a FusionAssessment plus current mission/context state into a
human-readable operational judgment.  This is Layer 5 of the
seven-layer reasoning architecture: the step that answers "so what, and
what should happen next?" before the orchestration layer asks "which
asset and when?"

The Decision object answers:
  - What should we do now?      → action
  - How urgent is it?           → priority
  - How confident are we?       → confidence
  - Why did we choose it?       → why[]
  - What are the fallbacks?     → next_best_actions[]

Decision layer responsibility
------------------------------
  ✓ Choose the action type (PASSIVE_MONITOR → ESCALATE)
  ✓ Compute mission-facing priority
  ✓ Compute confidence in the chosen action
  ✓ Generate human-readable why bullets from real computed state
  ✓ Generate plausible ordered fallback alternatives

Decision layer must NOT
-----------------------
  ✗ Choose the exact sensor asset instance
  ✗ Choose exact window start/end times
  ✗ Choose ranked queue position

Those belong to Phase 5: TaskRecommendation.

Public API
----------
Decision
    Frozen dataclass: entity_id, timestamp, action, priority,
    confidence, why, next_best_actions.

build_decision(fusion_assessment, record, track, compounds,
               planner_context) -> Decision
    Primary builder.

decision_for_timeline(timeline, track, fusion_assessments,
                      planner_context_by_time) -> list[Decision]
    Apply build_decision across a vessel timeline paired with
    pre-built FusionAssessments.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import custody.config as config
from custody.compounds import CompoundSignal
from custody.decision_trace import DecisionTrace
from custody.belief_assessment import FusionAssessment
from custody.models import TrackState


# ---------------------------------------------------------------------------
# Action vocabulary
# ---------------------------------------------------------------------------

PASSIVE_MONITOR = "PASSIVE_MONITOR"
ELEVATE         = "ELEVATE"
TASK_OPTICAL    = "TASK_OPTICAL"
TASK_SAR        = "TASK_SAR"
ESCALATE        = "ESCALATE"

_ALL_ACTIONS = {PASSIVE_MONITOR, ELEVATE, TASK_OPTICAL, TASK_SAR, ESCALATE}

# Ordered fallback alternatives per primary action (primary excluded)
_NEXT_BEST: dict[str, list[str]] = {
    TASK_SAR:        [TASK_OPTICAL, ELEVATE, ESCALATE],
    TASK_OPTICAL:    [TASK_SAR, ELEVATE, PASSIVE_MONITOR],
    ELEVATE:         [TASK_OPTICAL, TASK_SAR, PASSIVE_MONITOR],
    ESCALATE:        [TASK_SAR, TASK_OPTICAL, ELEVATE],
    PASSIVE_MONITOR: [ELEVATE],
}


# ---------------------------------------------------------------------------
# Core data contract
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Decision:
    """Mission-level operational recommendation for one entity at one timestep.

    Attributes:
        entity_id:        Entity being assessed.
        timestamp:        Observation time this decision corresponds to.
        action:           Headline operational recommendation.  One of:
                          PASSIVE_MONITOR, ELEVATE, TASK_OPTICAL, TASK_SAR,
                          ESCALATE.
        priority:         Mission-facing urgency in [0, 1].  Not a copy of
                          fused_score — includes context bumps.
        confidence:       Confidence in the chosen action in [0, 1].  Drops
                          as uncertainty rises; rises when action clearly
                          matches the evidence.
        why:              Short plain-English rationale bullets assembled from
                          real computed state (3–5 items).
        next_best_actions: Ordered alternatives if the primary action cannot be
                          taken.  Primary action is excluded.
    """
    entity_id: str
    timestamp: datetime
    action: str
    priority: float
    confidence: float
    why: list[str]
    next_best_actions: list[str]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _clamp01(v: float) -> float:
    return min(max(v, 0.0), 1.0)


def _select_action(
    fusion_assessment: FusionAssessment,
    record: dict,
    track: TrackState,
    compounds: list[CompoundSignal],
    planner_context=None,
) -> str:
    """Choose the headline action from the fusion picture.

    Rules applied in priority order — first match wins.

    Args:
        fusion_assessment: Pre-built FusionAssessment for this timestep.
        record:            Timeline record dict.
        track:             TrackState for this entity.
        compounds:         Active CompoundSignal objects.
        planner_context:   Optional DecisionTrace.

    Returns:
        One of PASSIVE_MONITOR, ELEVATE, TASK_OPTICAL, TASK_SAR, ESCALATE.
    """
    fs = fusion_assessment.fused_score
    unc = fusion_assessment.uncertainty
    rec = fusion_assessment.recommended_confirming_source

    # Clear, low-concern situation
    if fs < 0.35 and unc < 0.50:
        return PASSIVE_MONITOR

    # High significance with unresolved uncertainty — human review warranted
    if fs >= 0.80 and unc >= 0.65:
        return ESCALATE

    # Confirming source drives the tasking action
    if rec == "SAR":
        return TASK_SAR
    if rec == "OPTICAL":
        return TASK_OPTICAL

    # Elevated concern without a clear confirming-source recommendation
    return ELEVATE


def _compute_priority(
    fusion_assessment: FusionAssessment,
    record: dict,
    track: TrackState,
    compounds: list[CompoundSignal],
    planner_context=None,
) -> float:
    """Mission-facing urgency score, bounded [0, 1].

    Starts from fused_score and applies context bumps for zone presence,
    active compound signals, and imminent orbital windows.

    Args:
        fusion_assessment: Pre-built FusionAssessment.
        record:            Timeline record dict.
        track:             TrackState.
        compounds:         Active CompoundSignal objects.
        planner_context:   Optional DecisionTrace.

    Returns:
        priority in [0, 1].
    """
    base = fusion_assessment.fused_score

    # Zone proximity — entity in or near a sensitive area adds urgency
    zone = float(record.get("sensitive_zone", 0.0))
    zone_bump = 0.08 if zone > 0 else 0.0

    # Compound signals — each active compound adds incremental urgency
    compound_bump = min(len(compounds) * 0.03, 0.09)

    # Worsening anomaly trend — current anomaly at or above the last-collected
    # level indicates the situation has not improved since last tasking
    anomaly = float(record.get("anomaly_score", 0.0))
    worsening_bump = 0.05 if (
        anomaly > config.HIGH_ANOMALY_THRESHOLD
        and anomaly >= track.last_collection_anomaly_score
    ) else 0.0

    # Imminent orbital window — collection opportunity is time-constrained
    orbital_bump = 0.0
    if isinstance(planner_context, DecisionTrace):
        tts = planner_context.task_value.nearest_pass_time_to_start_seconds
        if tts is not None and 0 < tts <= config.HOLD_LOOKAHEAD_THRESHOLD_SECONDS:
            orbital_bump = 0.05

    # Approaching zone — prediction layer signal
    _zone_prob = float(record.get("zone_probability", 0.0))
    _tte       = record.get("time_to_zone_hours")
    try:
        _tte_valid = _tte is not None and not math.isnan(float(_tte))
    except (TypeError, ValueError):
        _tte_valid = False
    prediction_bump = 0.06 if _zone_prob > 0.5 and _tte_valid else 0.0

    return round(_clamp01(
        base + zone_bump + compound_bump + worsening_bump + orbital_bump + prediction_bump
    ), 3)


def _compute_decision_confidence(
    fusion_assessment: FusionAssessment,
    action: str,
    record: dict,
    track: TrackState,
    compounds: list[CompoundSignal],
) -> float:
    """Confidence in the chosen action, bounded [0, 1].

    Drops as uncertainty rises; rises when the action naturally matches
    the evidence picture.

    Args:
        fusion_assessment: Pre-built FusionAssessment.
        action:            The selected action string.
        record:            Timeline record dict.
        track:             TrackState.
        compounds:         Active CompoundSignal objects.

    Returns:
        confidence in [0, 1].
    """
    rec = fusion_assessment.recommended_confirming_source
    fs  = fusion_assessment.fused_score

    # source_action_fit: how cleanly does the action match the evidence?
    if action == TASK_SAR and rec == "SAR":
        fit = 1.0
    elif action == TASK_OPTICAL and rec == "OPTICAL":
        fit = 1.0
    elif action == PASSIVE_MONITOR and fs < 0.35:
        fit = 0.95
    elif action == ESCALATE and fs >= 0.80:
        fit = 0.90
    elif action == ELEVATE:
        fit = 0.65
    else:
        # Action is reasonable but evidence alignment is weaker
        fit = 0.45

    return round(_clamp01(
        0.60 * (1.0 - fusion_assessment.uncertainty)
        + 0.25 * fs
        + 0.15 * fit
    ), 3)


def _build_why(
    fusion_assessment: FusionAssessment,
    action: str,
    record: dict,
    track: TrackState,
    compounds: list[CompoundSignal],
    planner_context=None,
) -> list[str]:
    """Assemble operator-grade rationale bullets from real computed state.

    Produces 3–5 bullets in priority order, capped at 5.  Each bullet
    is derived from an actual numeric or categorical value — no canned filler.

    Args:
        fusion_assessment: Pre-built FusionAssessment.
        action:            Selected action string.
        record:            Timeline record dict.
        track:             TrackState.
        compounds:         Active CompoundSignal objects.
        planner_context:   Optional DecisionTrace.

    Returns:
        List of 1–5 short plain-English strings.
    """
    bullets: list[str] = []
    fs   = fusion_assessment.fused_score
    unc  = fusion_assessment.uncertainty
    agr  = fusion_assessment.source_agreement
    conf = float(record.get("custody_confidence", 1.0))
    anom = float(record.get("anomaly_score", 0.0))
    zone = float(record.get("sensitive_zone", 0.0))

    # ── Lead with fused significance ──────────────────────────────────────
    if fs >= 0.70:
        bullets.append(
            "Fused significance is high, driven by elevated anomaly severity "
            "and degraded custody confidence."
        )
    elif fs >= 0.45:
        bullets.append(
            "Fused significance is moderate — the situation warrants increased "
            "monitoring attention."
        )
    else:
        bullets.append(
            "Fused significance is low — the entity does not currently require "
            "active intervention."
        )

    # ── Uncertainty ───────────────────────────────────────────────────────
    if unc >= 0.65:
        bullets.append(
            "Uncertainty is elevated; available evidence is insufficient "
            "to form a confident operational picture."
        )
    elif unc >= 0.40:
        bullets.append(
            "Some uncertainty persists — confirming evidence would strengthen "
            "the current assessment."
        )

    # ── Source agreement ──────────────────────────────────────────────────
    if agr < 0.50:
        bullets.append(
            "Evidence sources are inconsistent — behavioural signals are "
            "not fully corroborated by the AIS track."
        )

    # ── Zone context ──────────────────────────────────────────────────────
    if zone > 0.5:
        bullets.append("The entity is operating in or near a sensitive zone.")

    # ── Prediction — approaching zone ─────────────────────────────────────
    zone_prob = float(record.get("zone_probability", 0.0))
    tte        = record.get("time_to_zone_hours")
    if zone_prob > 0.5 and tte is not None:
        try:
            _tte = float(tte)
            if not math.isnan(_tte):
                bullets.append(
                    f"Trajectory analysis projects zone entry in {_tte:.1f}h "
                    f"(zone probability {zone_prob:.0%}); pre-tasking recommended."
                )
        except (TypeError, ValueError):
            pass

    # ── Compound signals ──────────────────────────────────────────────────
    if compounds:
        codes = ", ".join(
            cs.code.replace("_", " ").title() for cs in compounds
        )
        bullets.append(f"Active compound signal(s): {codes}.")

    # ── Action-specific rationale ─────────────────────────────────────────
    if action == TASK_SAR:
        bullets.append(
            "SAR tasking is recommended as the most reliable confirming "
            "source under current track and weather conditions."
        )
    elif action == TASK_OPTICAL:
        bullets.append(
            "Optical tasking is recommended to obtain direct visual "
            "confirmation of current behaviour."
        )
    elif action == ESCALATE:
        bullets.append(
            "Human review is warranted — significance is high and "
            "uncertainty remains unresolved."
        )
    elif action == ELEVATE:
        bullets.append(
            "Collection priority should be elevated pending further "
            "confirming evidence."
        )

    # Cap at 5 operator-grade bullets
    return bullets[:5]


def _build_next_best_actions(
    action: str,
    fusion_assessment: FusionAssessment,
    record: dict,
    track: TrackState,
    planner_context=None,
) -> list[str]:
    """Return ordered fallback alternatives, excluding the primary action.

    Args:
        action:            Primary selected action.
        fusion_assessment: Pre-built FusionAssessment.
        record:            Timeline record dict (reserved for future context).
        track:             TrackState (reserved for future context).
        planner_context:   Optional DecisionTrace (reserved for future context).

    Returns:
        Ordered list of alternative action strings.
    """
    return list(_NEXT_BEST.get(action, [ELEVATE]))


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

def build_decision(
    fusion_assessment: FusionAssessment,
    record: dict,
    track: TrackState,
    compounds: list[CompoundSignal],
    planner_context=None,
) -> Decision:
    """Build a Decision from a FusionAssessment and current context.

    This is an output artifact: frozen and deterministic for the same inputs.
    It answers the mission-layer question "what should happen now?" using the
    fused belief picture produced by build_fusion_assessment().

    Args:
        fusion_assessment: Pre-built FusionAssessment for this entity/timestep.
        record:            Timeline record dict (same record used to build FA).
        track:             TrackState for this entity at this timestep.
        compounds:         Active CompoundSignal objects (same list used for FA).
        planner_context:   Optional DecisionTrace from this timestep's planner
                           run.  Passing None is safe.

    Returns:
        A frozen Decision.
    """
    action = _select_action(
        fusion_assessment, record, track, compounds, planner_context
    )
    priority = _compute_priority(
        fusion_assessment, record, track, compounds, planner_context
    )
    confidence = _compute_decision_confidence(
        fusion_assessment, action, record, track, compounds
    )
    why = _build_why(
        fusion_assessment, action, record, track, compounds, planner_context
    )
    next_best = _build_next_best_actions(
        action, fusion_assessment, record, track, planner_context
    )

    return Decision(
        entity_id=fusion_assessment.entity_id,
        timestamp=fusion_assessment.timestamp,
        action=action,
        priority=priority,
        confidence=confidence,
        why=why,
        next_best_actions=next_best,
    )


# ---------------------------------------------------------------------------
# Timeline convenience wrapper
# ---------------------------------------------------------------------------

def decision_for_timeline(
    timeline: list[dict],
    track: TrackState,
    fusion_assessments: list[FusionAssessment],
    planner_context_by_time: Optional[dict] = None,
) -> list[Decision]:
    """Apply build_decision to every record in a single-vessel timeline.

    Args:
        timeline:              Ordered list of record dicts for one vessel.
        track:                 The vessel's TrackState.
        fusion_assessments:    Pre-built FusionAssessments in the same order
                               as timeline.  Must have the same length.
        planner_context_by_time:
                               Optional dict mapping datetime → DecisionTrace.
                               Pass None if planner traces are not available.

    Returns:
        List of Decision in the same order as timeline.

    Raises:
        ValueError: If timeline and fusion_assessments have different lengths.
    """
    if len(timeline) != len(fusion_assessments):
        raise ValueError(
            f"timeline ({len(timeline)}) and fusion_assessments "
            f"({len(fusion_assessments)}) must have the same length"
        )
    from custody.compounds import evaluate_compounds

    ctx_map = planner_context_by_time or {}
    result: list[Decision] = []

    for i, (record, fa) in enumerate(zip(timeline, fusion_assessments)):
        window = timeline[:i]
        compounds = evaluate_compounds(record, window=window)
        ctx = ctx_map.get(record.get("time"))
        result.append(build_decision(fa, record, track, compounds, ctx))

    return result
