"""
Agent-style decision roles for tasking.

A read-only interpretive layer that examines existing record, fusion, and
decision data through three operational lenses:

  Analyst   — behavioral significance and intelligence interpretation
  Collector — evidence gaps and sensor selection
  Operator  — mission urgency and escalation risk

Each role produces a RoleRecommendation.  The Watch Officer (synthesize)
resolves disagreements into a Synthesis with a simple majority / operator-
tiebreak rule.

This module does NOT participate in the simulation loop.  Roles are computed
at UI render time from pre-existing data.  The canonical Decision pipeline
is unchanged and remains authoritative.

Public API
----------
RoleRecommendation  — one role's output
Synthesis           — combined deliberation result
build_deliberation(record, fa, decision, compounds) -> Synthesis
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Optional

from custody.compounds import CompoundSignal
from custody.decision import (
    ELEVATE,
    ESCALATE,
    PASSIVE_MONITOR,
    TASK_OPTICAL,
    TASK_SAR,
    Decision,
)
from custody.belief_assessment import FusionAssessment


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RoleRecommendation:
    """One role's interpretive recommendation."""
    role: str                      # "analyst" | "collector" | "operator"
    recommended_action: str        # same vocabulary as Decision.action
    confidence: float              # [0, 1]
    rationale: str                 # one sentence
    key_factors: dict[str, float]  # 3-4 named numeric drivers
    expected_value: float          # [0, 1]


@dataclass(frozen=True)
class Synthesis:
    """Combined deliberation from all three roles."""
    analyst: RoleRecommendation
    collector: RoleRecommendation
    operator: RoleRecommendation
    agreement: str                          # "unanimous" | "majority" | "split"
    final_action: str                       # informational — does NOT override Decision.action
    resolution_reason: str                  # structured English sentence
    dissent: Optional[dict[str, str]]       # e.g. {"analyst": "TASK_OPTICAL"} or None
    why_not: dict[str, str]                 # rejected action → one-sentence reason


# ---------------------------------------------------------------------------
# Role functions
# ---------------------------------------------------------------------------

def _clamp01(v: float) -> float:
    return min(max(v, 0.0), 1.0)


def _safe_float(val, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        v = float(val)
        return default if math.isnan(v) else v
    except (TypeError, ValueError):
        return default


def analyst_recommend(
    record: dict,
    fa: FusionAssessment,
    decision: Decision,
    compounds: list[CompoundSignal],
) -> RoleRecommendation:
    """Behavioral significance lens.

    Interprets the intelligence picture — anomaly severity, behavioral
    patterns, temporal trend — without regard for sensor availability.
    """
    fs = fa.fused_score
    agr = fa.source_agreement
    anomaly_state = record.get("anomaly_state", "normal")
    behavior_state = record.get("behavior_state", "unknown")
    anomaly_score = _safe_float(record.get("anomaly_score"))

    # Action selection — behavioral significance (NOT sensor-driven)
    #
    # Analyst focuses on *what the behavior means*, not which sensor to use.
    # Sensor selection is Collector's job.  Analyst picks a tasking type only
    # when the behavioral pattern implies a specific modality.
    if fs < 0.35:
        action = PASSIVE_MONITOR
    elif fs >= 0.85 and anomaly_state in ("critical", "sustained"):
        action = ESCALATE
    elif anomaly_state in ("sustained", "confirmed") or (
        compounds and max(c.confidence for c in compounds) >= 0.7
    ):
        # Persistent or compound-confirmed — pick sensor by activity type
        if behavior_state in ("loitering", "approach", "rendezvous"):
            action = TASK_OPTICAL   # visual verification of activity
        else:
            action = TASK_SAR       # all-weather persistent watch
    elif fs >= 0.50 or anomaly_state == "emerging":
        action = ELEVATE
    else:
        action = PASSIVE_MONITOR

    # Rationale
    parts = []
    if anomaly_state != "normal":
        parts.append(f"{anomaly_state} anomaly ({anomaly_score:.2f})")
    else:
        parts.append(f"anomaly {anomaly_score:.2f}")
    if compounds:
        top = max(compounds, key=lambda c: c.confidence)
        parts.append(f"{top.code} compound")
    if behavior_state not in ("unknown", "transit"):
        parts.append(f"entity in {behavior_state} state")

    return RoleRecommendation(
        role="analyst",
        recommended_action=action,
        confidence=round(_clamp01(agr), 3),
        rationale="; ".join(parts) + ".",
        key_factors={
            "fused_score": round(fs, 3),
            "anomaly_score": round(anomaly_score, 3),
            "source_agreement": round(agr, 3),
            "compound_count": len(compounds),
        },
        expected_value=round(_clamp01(fs), 3),
    )


def collector_recommend(
    record: dict,
    fa: FusionAssessment,
    decision: Decision,
    compounds: list[CompoundSignal],
) -> RoleRecommendation:
    """Sensor/evidence lens.

    Interprets the evidence picture — what's stale, what's missing, what
    sensor would best reduce uncertainty.
    """
    unc = fa.uncertainty
    rec_src = fa.recommended_confirming_source
    missing = fa.missing_evidence
    hsc = _safe_float(record.get("hours_since_collection"), default=-1.0)
    health = record.get("custody_health", "HEALTHY")
    conf = _safe_float(record.get("custody_confidence"), default=1.0)

    # Action selection — evidence-gap driven
    if not missing and health == "HEALTHY":
        action = PASSIVE_MONITOR
    elif rec_src == "SAR":
        action = TASK_SAR
    elif rec_src == "OPTICAL":
        action = TASK_OPTICAL
    elif rec_src == "AIS":
        action = ELEVATE
    elif missing:
        action = ELEVATE
    else:
        action = PASSIVE_MONITOR

    # Rationale
    parts = []
    if rec_src:
        parts.append(f"{rec_src} confirmation needed")
    if missing:
        parts.append(f"gaps: {', '.join(missing[:2])}")
    if health != "HEALTHY":
        parts.append(f"track {health}")
    if hsc >= 0:
        parts.append(f"{hsc:.0f}h since last collection")

    ev = round(_clamp01(1.0 - unc), 3)

    return RoleRecommendation(
        role="collector",
        recommended_action=action,
        confidence=round(_clamp01(1.0 - unc), 3),
        rationale="; ".join(parts) + "." if parts else "No evidence gaps identified.",
        key_factors={
            "uncertainty": round(unc, 3),
            "hours_since_collection": round(hsc, 1) if hsc >= 0 else -1.0,
            "custody_confidence": round(conf, 3),
            "missing_evidence_count": len(missing),
        },
        expected_value=ev,
    )


def operator_recommend(
    record: dict,
    fa: FusionAssessment,
    decision: Decision,
    compounds: list[CompoundSignal],
) -> RoleRecommendation:
    """Mission urgency lens.

    Reads the operational context — zone approach, neglect, attention tier,
    trend — to assess whether the situation demands immediate action.
    """
    priority = decision.priority
    zone_prob = _safe_float(record.get("zone_probability"))
    tte = record.get("time_to_zone_hours")
    attention = record.get("attention_state", "BACKGROUND")
    neglect = bool(record.get("neglect_flag", False))
    neglect_hrs = _safe_float(record.get("neglect_hours"))
    dark = bool(record.get("dark_vessel_flag", False))
    esc_boost = _safe_float(record.get("escalation_boost"))

    # Check valid time-to-zone
    try:
        tte_val = float(tte) if tte is not None else None
        if tte_val is not None and math.isnan(tte_val):
            tte_val = None
    except (TypeError, ValueError):
        tte_val = None

    zone_imminent = zone_prob > 0.7 and tte_val is not None and tte_val < 4.0

    # Action selection — urgency-driven
    #
    # Operator reserves ESCALATE for truly extreme cases.  A dark vessel or
    # moderate-high priority alone warrants tasking, not escalation.
    if priority < 0.35 and not neglect and not dark:
        action = PASSIVE_MONITOR
    elif priority >= 0.90 or zone_imminent:
        action = ESCALATE
    elif dark:
        action = TASK_SAR           # SAR is the operational response to dark
    elif priority >= 0.50 or neglect or attention == "ACTIVE_CUSTODY":
        # Use fusion source hint when available
        if fa.recommended_confirming_source == "OPTICAL":
            action = TASK_OPTICAL
        else:
            action = TASK_SAR
    else:
        action = ELEVATE

    # Rationale
    parts = [f"priority {priority:.3f}"]
    if neglect:
        parts.append(f"neglect {neglect_hrs:.0f}h")
    if zone_prob > 0.5 and tte_val is not None:
        parts.append(f"zone approach (prob {zone_prob:.2f}, ETA {tte_val:.1f}h)")
    if dark:
        parts.append("AIS dark")
    if attention != "BACKGROUND":
        parts.append(f"attention {attention}")

    return RoleRecommendation(
        role="operator",
        recommended_action=action,
        confidence=round(_clamp01(priority), 3),
        rationale="; ".join(parts) + ".",
        key_factors={
            "priority": round(priority, 3),
            "zone_probability": round(zone_prob, 3),
            "neglect_hours": round(neglect_hrs, 1),
            "escalation_boost": round(esc_boost, 3),
        },
        expected_value=round(_clamp01(priority), 3),
    )


# ---------------------------------------------------------------------------
# Synthesis (Watch Officer)
# ---------------------------------------------------------------------------

def _first_clause(rationale: str) -> str:
    """Extract the leading clause from a rationale string."""
    return rationale.split(";")[0].rstrip(".")


def synthesize(
    analyst: RoleRecommendation,
    collector: RoleRecommendation,
    operator: RoleRecommendation,
) -> Synthesis:
    """Resolve three role recommendations into a single deliberation result.

    Rules:
      unanimous — all three agree → adopt
      majority  — 2 of 3 agree → adopt majority, note dissent
      split     — 3-way → Operator breaks tie
    """
    actions = [analyst.recommended_action, collector.recommended_action, operator.recommended_action]
    roles = [analyst, collector, operator]
    counts = Counter(actions)
    n_unique = len(counts)

    if n_unique == 1:
        return Synthesis(
            analyst=analyst,
            collector=collector,
            operator=operator,
            agreement="unanimous",
            final_action=actions[0],
            resolution_reason=f"All three roles recommend {actions[0]}.",
            dissent=None,
            why_not={},
        )

    if n_unique == 2:
        majority_action = counts.most_common(1)[0][0]
        outlier = next(r for r in roles if r.recommended_action != majority_action)
        agreeing = [r for r in roles if r.recommended_action == majority_action]
        agreeing_names = [r.role for r in agreeing]
        why_not = {
            outlier.recommended_action: (
                f"{outlier.role.capitalize()} preferred {outlier.recommended_action} "
                f"({_first_clause(outlier.rationale)}), but "
                f"{agreeing_names[0]} and {agreeing_names[1]} agreed on {majority_action}."
            ),
        }
        return Synthesis(
            analyst=analyst,
            collector=collector,
            operator=operator,
            agreement="majority",
            final_action=majority_action,
            resolution_reason=(
                f"{agreeing_names[0].capitalize()} and {agreeing_names[1]} agree on {majority_action}; "
                f"{outlier.role} dissent ({outlier.recommended_action})."
            ),
            dissent={outlier.role: outlier.recommended_action},
            why_not=why_not,
        )

    # 3-way split — Operator breaks tie
    dissent_map = {}
    why_not = {}
    for r in roles:
        if r.role != "operator":
            dissent_map[r.role] = r.recommended_action
            why_not[r.recommended_action] = (
                f"{r.role.capitalize()} preferred {r.recommended_action} "
                f"({_first_clause(r.rationale)}), but Operator broke tie "
                f"with {operator.recommended_action} (priority {operator.expected_value:.2f})."
            )
    return Synthesis(
        analyst=analyst,
        collector=collector,
        operator=operator,
        agreement="split",
        final_action=operator.recommended_action,
        resolution_reason=(
            f"Three-way split; Operator breaks tie with {operator.recommended_action}. "
            f"Analyst preferred {analyst.recommended_action}, "
            f"Collector preferred {collector.recommended_action}."
        ),
        dissent=dissent_map,
        why_not=why_not,
    )


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def build_deliberation(
    record: dict,
    fa: FusionAssessment,
    decision: Decision,
    compounds: list[CompoundSignal],
) -> Synthesis:
    """Build a full role deliberation from existing record data.

    This is the single entry point for the UI callback.  It calls all three
    role functions and the synthesizer.  Pure, deterministic, no side effects.
    """
    a = analyst_recommend(record, fa, decision, compounds)
    c = collector_recommend(record, fa, decision, compounds)
    o = operator_recommend(record, fa, decision, compounds)
    return synthesize(a, c, o)
