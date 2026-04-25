"""Plan execution simulation feedback loop (ADR-0021 Slice 23).

Simulates execution of a scheduler-lite plan by generating synthetic
returned evidence for each scheduled collect, feeding that evidence
back through the belief-update engine, and recomputing custody health
and collection recommendations.

This closes the prototype decision loop:
plan selection -> schedule feasibility -> simulated collect result ->
belief update -> custody health change -> next recommendation.

This is simulation only.  No imagery is downloaded or processed,
no execution authorizations are issued, no live tasking or sensor
command is issued, and no platform-access decisions are claimed.

Design constraints
------------------

- Deterministic: no wall-clock, no randomness.
- Pure stdlib: no new runtime dependencies.
- No imports from ``custody.detection.*``, ``custody.ingest.gfw_presence``,
  matcher runtime, or external SDKs.
"""
from __future__ import annotations

import json as _json
from dataclasses import dataclass
from enum import Enum

from custody.hypotheses.collection_value import (
    CollectionRecommendation,
    rank_collection_candidates,
)
from custody.hypotheses.custody_health import (
    CustodyHealthStatus,
    HypothesisCustodyHealth,
    assess_custody_health,
)
from custody.hypotheses.scheduler import (
    ScheduledCollect,
    SchedulePlan,
)
from custody.hypotheses.types import (
    HypothesisEvidence,
    HypothesisState,
)
from custody.hypotheses.update import update_state


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class ExecutionOutcomePolicy(Enum):
    FAVORABLE = "favorable"
    INCONCLUSIVE = "inconclusive"
    ADVERSE = "adverse"
    MIXED = "mixed"


@dataclass(frozen=True)
class SimulatedCollectResult:
    result_id: str
    scenario_id: str
    candidate_id: str
    window_id: str
    outcome_policy: str
    returned_evidence: tuple[HypothesisEvidence, ...]
    result_quality: float
    reason: str
    caveats: tuple[str, ...]


@dataclass(frozen=True)
class ExecutionSimulationReport:
    scenario_id: str
    outcome_policy: str
    pre_top_hypothesis: str | None
    post_top_hypothesis: str | None
    pre_health_status: CustodyHealthStatus
    post_health_status: CustodyHealthStatus
    pre_health_score: float
    post_health_score: float
    health_score_delta: float
    pre_ambiguity_pairs: tuple[tuple[str, str], ...]
    post_ambiguity_pairs: tuple[tuple[str, str], ...]
    ambiguity_resolved: bool
    scheduled_collects: tuple[ScheduledCollect, ...]
    simulated_results: tuple[SimulatedCollectResult, ...]
    post_recommendation: CollectionRecommendation
    summary: str
    caveats: tuple[str, ...]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


_BASE_CAVEATS: tuple[str, ...] = (
    "execution simulation uses synthetic returned evidence",
    "no live tasking or sensor command is issued",
    "no imagery was downloaded or processed",
    "outcome policy is deterministic and not a calibrated probability",
)


_VALID_POLICIES = frozenset({p.value for p in ExecutionOutcomePolicy})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _round4(x: float) -> float:
    return round(float(x), 4)


def _validate_policy(policy: str) -> str:
    if policy not in _VALID_POLICIES:
        raise ValueError(
            f"outcome_policy must be one of {sorted(_VALID_POLICIES)}; "
            f"got {policy!r}"
        )
    return policy


def _result_id(scenario_id: str, scheduled: ScheduledCollect) -> str:
    return (
        f"result_{scenario_id}_{scheduled.candidate_id}_{scheduled.window_id}"
    )


def _evidence_id(scenario_id: str, scheduled: ScheduledCollect) -> str:
    return (
        f"exec_sim_{scenario_id}_{scheduled.candidate_id}_{scheduled.window_id}"
    )


def _source_ref(scheduled: ScheduledCollect) -> str:
    return f"schedule:{scheduled.window_id}:{scheduled.candidate_id}"


def _confidence_for_collect(scheduled: ScheduledCollect, fallback: float) -> float:
    score = scheduled.schedule_score
    if score <= 0.0:
        return fallback
    return _clamp01(score)


# ---------------------------------------------------------------------------
# Outcome construction
# ---------------------------------------------------------------------------


def _empty_result(
    scheduled: ScheduledCollect,
    *,
    scenario_id: str,
    outcome_policy: str,
    reason: str,
) -> SimulatedCollectResult:
    return SimulatedCollectResult(
        result_id=_result_id(scenario_id, scheduled),
        scenario_id=scenario_id,
        candidate_id=scheduled.candidate_id,
        window_id=scheduled.window_id,
        outcome_policy=outcome_policy,
        returned_evidence=(),
        result_quality=0.0,
        reason=reason,
        caveats=_BASE_CAVEATS,
    )


def _build_evidence(
    *,
    scenario_id: str,
    scheduled: ScheduledCollect,
    supports: tuple[str, ...],
    contradicts: tuple[str, ...],
    confidence: float,
    reason: str,
) -> HypothesisEvidence:
    return HypothesisEvidence(
        evidence_id=_evidence_id(scenario_id, scheduled),
        source_ref=_source_ref(scheduled),
        source_kind="execution_simulation",
        scenario_id=scenario_id,
        timestamp=scheduled.end_time,
        supports=supports,
        contradicts=contradicts,
        confidence=_clamp01(confidence),
        weight=1.0,
        reason=reason,
    )


def _favorable_or_adverse_result(
    scheduled: ScheduledCollect,
    *,
    scenario_id: str,
    outcome_policy: str,
    primary_pair: tuple[str, str],
) -> SimulatedCollectResult:
    a, b = primary_pair
    if outcome_policy == "favorable":
        supports = (a,)
        contradicts = (b,)
        reason = (
            f"scheduled {scheduled.candidate_id} returned synthetic evidence "
            f"supporting {a} over {b}"
        )
    else:  # adverse
        supports = (b,)
        contradicts = (a,)
        reason = (
            f"scheduled {scheduled.candidate_id} returned synthetic evidence "
            f"supporting {b} over {a}"
        )
    confidence = _confidence_for_collect(scheduled, fallback=0.75)
    ev = _build_evidence(
        scenario_id=scenario_id,
        scheduled=scheduled,
        supports=supports,
        contradicts=contradicts,
        confidence=confidence,
        reason=reason,
    )
    return SimulatedCollectResult(
        result_id=_result_id(scenario_id, scheduled),
        scenario_id=scenario_id,
        candidate_id=scheduled.candidate_id,
        window_id=scheduled.window_id,
        outcome_policy=outcome_policy,
        returned_evidence=(ev,),
        result_quality=_clamp01(scheduled.schedule_score),
        reason=reason,
        caveats=_BASE_CAVEATS,
    )


def _inconclusive_result(
    scheduled: ScheduledCollect,
    *,
    scenario_id: str,
    confidence: float = 0.05,
    label: str = "inconclusive",
) -> SimulatedCollectResult:
    reason = (
        f"scheduled {scheduled.candidate_id} returned {label} synthetic "
        f"evidence; primary ambiguity is unchanged"
    )
    ev = _build_evidence(
        scenario_id=scenario_id,
        scheduled=scheduled,
        supports=(),
        contradicts=(),
        confidence=confidence,
        reason=reason,
    )
    return SimulatedCollectResult(
        result_id=_result_id(scenario_id, scheduled),
        scenario_id=scenario_id,
        candidate_id=scheduled.candidate_id,
        window_id=scheduled.window_id,
        outcome_policy=label,
        returned_evidence=(ev,),
        result_quality=_clamp01(scheduled.schedule_score),
        reason=reason,
        caveats=_BASE_CAVEATS,
    )


# ---------------------------------------------------------------------------
# Public: simulate single collect
# ---------------------------------------------------------------------------


def simulate_collect_result(
    state: HypothesisState,
    health: HypothesisCustodyHealth,
    scheduled_collect: ScheduledCollect,
    *,
    scenario_id: str,
    outcome_policy: str = "favorable",
) -> SimulatedCollectResult:
    """Synthesize one returned-evidence result for a scheduled collect."""
    _validate_policy(outcome_policy)

    if not health.ambiguity_pairs:
        reason = (
            "execution simulation is most useful when custody has surfaced "
            "an ambiguity; no primary pair available"
        )
        return _empty_result(
            scheduled_collect,
            scenario_id=scenario_id,
            outcome_policy=outcome_policy,
            reason=reason,
        )

    primary = health.ambiguity_pairs[0]

    if outcome_policy in ("favorable", "adverse"):
        return _favorable_or_adverse_result(
            scheduled_collect,
            scenario_id=scenario_id,
            outcome_policy=outcome_policy,
            primary_pair=primary,
        )
    if outcome_policy == "inconclusive":
        return _inconclusive_result(
            scheduled_collect,
            scenario_id=scenario_id,
            confidence=0.05,
            label="inconclusive",
        )
    # mixed: single-collect fallback - slightly higher than inconclusive.
    return _inconclusive_result(
        scheduled_collect,
        scenario_id=scenario_id,
        confidence=0.15,
        label="mixed",
    )


# ---------------------------------------------------------------------------
# Public: simulate full plan execution
# ---------------------------------------------------------------------------


def _ambiguity_resolved(
    pre_pairs: tuple[tuple[str, str], ...],
    post_pairs: tuple[tuple[str, str], ...],
) -> bool:
    if not pre_pairs:
        return False
    primary = pre_pairs[0]
    if not post_pairs:
        return True
    return primary not in post_pairs


def _summary(
    scenario_id: str,
    pre_health: HypothesisCustodyHealth,
    post_health: HypothesisCustodyHealth,
    n_results: int,
    ambiguity_resolved: bool,
    outcome_policy: str,
) -> str:
    delta = post_health.score - pre_health.score
    parts = [
        f"execution simulation for {scenario_id} ({outcome_policy})",
        f"{n_results} synthetic result(s) applied",
        f"health {pre_health.status.value} -> {post_health.status.value}",
        f"score {pre_health.score:.2f} -> {post_health.score:.2f} "
        f"(delta {delta:+.2f})",
    ]
    if ambiguity_resolved:
        parts.append("primary ambiguity resolved")
    else:
        parts.append("primary ambiguity unchanged")
    return "; ".join(parts)


def simulate_plan_execution(
    state: HypothesisState,
    schedule_plan: SchedulePlan,
    *,
    scenario_id: str,
    outcome_policy: str = "favorable",
) -> ExecutionSimulationReport:
    """Apply a scheduler-lite plan via synthetic returned evidence."""
    policy = _validate_policy(outcome_policy)

    pre_health = assess_custody_health(state)
    pre_top = state.top_hypothesis
    pre_pairs = pre_health.ambiguity_pairs

    ordered = tuple(sorted(
        schedule_plan.scheduled_collects,
        key=lambda s: (s.end_time, s.candidate_id),
    ))

    if policy == "mixed" and len(ordered) >= 2:
        per_collect_policies: tuple[str, ...] = tuple(
            "favorable" if i % 2 == 0 else "inconclusive"
            for i in range(len(ordered))
        )
    else:
        per_collect_policies = tuple(policy for _ in ordered)

    simulated_results: list[SimulatedCollectResult] = []
    for collect, eff in zip(ordered, per_collect_policies):
        sim = simulate_collect_result(
            state, pre_health, collect,
            scenario_id=scenario_id, outcome_policy=eff,
        )
        simulated_results.append(sim)

    all_evidence: list[HypothesisEvidence] = []
    for sim in simulated_results:
        all_evidence.extend(sim.returned_evidence)
    all_evidence.sort(
        key=lambda e: (
            e.timestamp.isoformat() if e.timestamp is not None else "",
            e.evidence_id,
        ),
    )

    if all_evidence:
        post_state = update_state(
            scenario_id, all_evidence, prior_state=state,
        )
    else:
        post_state = state

    post_health = assess_custody_health(post_state)
    post_recommendation = rank_collection_candidates(
        post_state, post_health, scenario_id=scenario_id,
    )

    health_score_delta = _round4(post_health.score - pre_health.score)
    ambiguity_resolved = _ambiguity_resolved(
        pre_pairs, post_health.ambiguity_pairs,
    )

    summary = _summary(
        scenario_id, pre_health, post_health,
        len(simulated_results), ambiguity_resolved, policy,
    )

    return ExecutionSimulationReport(
        scenario_id=scenario_id,
        outcome_policy=policy,
        pre_top_hypothesis=pre_top,
        post_top_hypothesis=post_state.top_hypothesis,
        pre_health_status=pre_health.status,
        post_health_status=post_health.status,
        pre_health_score=_round4(pre_health.score),
        post_health_score=_round4(post_health.score),
        health_score_delta=health_score_delta,
        pre_ambiguity_pairs=pre_pairs,
        post_ambiguity_pairs=post_health.ambiguity_pairs,
        ambiguity_resolved=ambiguity_resolved,
        scheduled_collects=ordered,
        simulated_results=tuple(simulated_results),
        post_recommendation=post_recommendation,
        summary=summary,
        caveats=_BASE_CAVEATS,
    )


# ---------------------------------------------------------------------------
# Serializers
# ---------------------------------------------------------------------------


def _iso(dt) -> str | None:
    return dt.isoformat() if dt is not None else None


def _evidence_to_dict(e: HypothesisEvidence) -> dict:
    return {
        "evidence_id": e.evidence_id,
        "source_ref": e.source_ref,
        "source_kind": e.source_kind,
        "scenario_id": e.scenario_id,
        "timestamp": _iso(e.timestamp),
        "supports": list(e.supports),
        "contradicts": list(e.contradicts),
        "confidence": e.confidence,
        "weight": e.weight,
        "reason": e.reason,
    }


def _result_to_dict(r: SimulatedCollectResult) -> dict:
    return {
        "result_id": r.result_id,
        "scenario_id": r.scenario_id,
        "candidate_id": r.candidate_id,
        "window_id": r.window_id,
        "outcome_policy": r.outcome_policy,
        "returned_evidence": [
            _evidence_to_dict(e) for e in r.returned_evidence
        ],
        "result_quality": r.result_quality,
        "reason": r.reason,
        "caveats": list(r.caveats),
    }


def _scheduled_to_dict(s: ScheduledCollect) -> dict:
    return {
        "candidate_id": s.candidate_id,
        "label": s.label,
        "window_id": s.window_id,
        "start_time": _iso(s.start_time),
        "end_time": _iso(s.end_time),
        "capacity_cost": s.capacity_cost,
        "schedule_score": s.schedule_score,
        "reason": s.reason,
        "caveats": list(s.caveats),
    }


def _recommendation_to_dict(rec: CollectionRecommendation) -> dict:
    return {
        "scenario_id": rec.scenario_id,
        "health_status": rec.health_status.value,
        "primary_ambiguity": (
            list(rec.primary_ambiguity)
            if rec.primary_ambiguity is not None else None
        ),
        "ranked_values": [
            {
                "candidate_id": cv.candidate.candidate_id,
                "label": cv.candidate.label,
                "score": cv.score,
                "disambiguates": (
                    list(cv.disambiguates)
                    if cv.disambiguates is not None else None
                ),
                "reason": cv.reason,
                "caveats": list(cv.caveats),
            }
            for cv in rec.ranked_values
        ],
        "summary": rec.summary,
    }


def execution_report_to_dict(report: ExecutionSimulationReport) -> dict:
    return {
        "scenario_id": report.scenario_id,
        "outcome_policy": report.outcome_policy,
        "pre_top_hypothesis": report.pre_top_hypothesis,
        "post_top_hypothesis": report.post_top_hypothesis,
        "pre_health_status": report.pre_health_status.value,
        "post_health_status": report.post_health_status.value,
        "pre_health_score": report.pre_health_score,
        "post_health_score": report.post_health_score,
        "health_score_delta": report.health_score_delta,
        "pre_ambiguity_pairs": [list(p) for p in report.pre_ambiguity_pairs],
        "post_ambiguity_pairs": [list(p) for p in report.post_ambiguity_pairs],
        "ambiguity_resolved": report.ambiguity_resolved,
        "scheduled_collects": [
            _scheduled_to_dict(s) for s in report.scheduled_collects
        ],
        "simulated_results": [
            _result_to_dict(r) for r in report.simulated_results
        ],
        "post_recommendation": _recommendation_to_dict(report.post_recommendation),
        "summary": report.summary,
        "caveats": list(report.caveats),
    }


def execution_report_to_json(report: ExecutionSimulationReport) -> str:
    return _json.dumps(execution_report_to_dict(report), indent=2)


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def _hr(title: str) -> str:
    underline = "-" * max(len(title), 3)
    return f"\n{title}\n{underline}\n"


def _format_pair_or_none(pairs: tuple[tuple[str, str], ...]) -> str:
    if not pairs:
        return "(none)"
    a, b = pairs[0]
    return f"{a} vs {b}"


def format_execution_text(report: ExecutionSimulationReport) -> str:
    parts: list[str] = []
    title = f"PLAN EXECUTION SIMULATION - {report.scenario_id.upper()}"
    parts.append(title + "\n")
    parts.append("=" * len(title) + "\n")
    parts.append(f"\nOutcome policy: {report.outcome_policy}\n")

    parts.append(_hr("Pre-execution"))
    parts.append(
        f"  health: {report.pre_health_status.value.upper()} "
        f"{report.pre_health_score:.2f}\n"
    )
    parts.append(
        f"  top hypothesis: {report.pre_top_hypothesis or '(none)'}\n"
    )
    parts.append(
        f"  primary ambiguity: {_format_pair_or_none(report.pre_ambiguity_pairs)}\n"
    )

    parts.append(_hr("Scheduled collects"))
    if not report.scheduled_collects:
        parts.append("  (none)\n")
    else:
        for s in report.scheduled_collects:
            parts.append(
                f"  {s.candidate_id:24s}  window={s.window_id}\n"
            )

    parts.append(_hr("Simulated returned evidence"))
    if not report.simulated_results:
        parts.append("  (none)\n")
    else:
        for r in report.simulated_results:
            if r.returned_evidence:
                ev = r.returned_evidence[0]
                if ev.supports or ev.contradicts:
                    sup = ", ".join(ev.supports) if ev.supports else "(none)"
                    con = ", ".join(ev.contradicts) if ev.contradicts else "(none)"
                    parts.append(
                        f"  {r.candidate_id:24s} -> supports {sup}, "
                        f"contradicts {con} (conf {ev.confidence:.2f}) "
                        f"[{r.outcome_policy}]\n"
                    )
                else:
                    parts.append(
                        f"  {r.candidate_id:24s} -> {r.outcome_policy} "
                        f"(conf {ev.confidence:.2f})\n"
                    )
            else:
                parts.append(
                    f"  {r.candidate_id:24s} -> no synthetic evidence "
                    f"(no primary ambiguity)\n"
                )

    parts.append(_hr("Post-execution"))
    parts.append(
        f"  health: {report.post_health_status.value.upper()} "
        f"{report.post_health_score:.2f}\n"
    )
    parts.append(
        f"  top hypothesis: {report.post_top_hypothesis or '(none)'}\n"
    )
    parts.append(
        f"  ambiguity resolved: "
        f"{'yes' if report.ambiguity_resolved else 'no'}\n"
    )
    parts.append(
        f"  health score delta: {report.health_score_delta:+.2f}\n"
    )

    parts.append(_hr("Before/after health"))
    parts.append(
        f"  {report.pre_health_status.value.upper()} "
        f"{report.pre_health_score:.2f}  ->  "
        f"{report.post_health_status.value.upper()} "
        f"{report.post_health_score:.2f}\n"
    )

    parts.append(_hr("Updated recommendation"))
    rec = report.post_recommendation
    if rec.ranked_values:
        top = rec.ranked_values[0]
        parts.append(
            f"  top: {top.candidate.candidate_id:24s}  "
            f"score={top.score:.2f}\n"
        )
        for cv in rec.ranked_values[1:3]:
            parts.append(
                f"       {cv.candidate.candidate_id:24s}  "
                f"score={cv.score:.2f}\n"
            )
    else:
        parts.append("  (none)\n")

    parts.append(_hr("Caveats"))
    for c in report.caveats:
        parts.append(f"  - {c}\n")

    return "".join(parts)


def format_execution_markdown(report: ExecutionSimulationReport) -> str:
    lines: list[str] = []
    name = report.scenario_id.capitalize()
    lines.append(f"# Plan Execution Simulation - {name}")
    lines.append("")
    lines.append(f"- **Outcome policy**: {report.outcome_policy}")
    lines.append("")

    lines.append("## Pre-execution")
    lines.append("")
    lines.append(
        f"- health: **{report.pre_health_status.value.upper()}** "
        f"({report.pre_health_score:.2f})"
    )
    lines.append(
        f"- top hypothesis: `{report.pre_top_hypothesis or '(none)'}`"
    )
    lines.append(
        f"- primary ambiguity: {_format_pair_or_none(report.pre_ambiguity_pairs)}"
    )
    lines.append("")

    lines.append("## Scheduled collects")
    lines.append("")
    if not report.scheduled_collects:
        lines.append("(none)")
    else:
        lines.append("| candidate | window |")
        lines.append("|---|---|")
        for s in report.scheduled_collects:
            lines.append(f"| {s.candidate_id} | {s.window_id} |")
    lines.append("")

    lines.append("## Simulated returned evidence")
    lines.append("")
    if not report.simulated_results:
        lines.append("(none)")
    else:
        lines.append("| candidate | supports | contradicts | confidence | policy |")
        lines.append("|---|---|---|---|---|")
        for r in report.simulated_results:
            if r.returned_evidence:
                ev = r.returned_evidence[0]
                sup = ", ".join(ev.supports) if ev.supports else "-"
                con = ", ".join(ev.contradicts) if ev.contradicts else "-"
                lines.append(
                    f"| {r.candidate_id} | {sup} | {con} | "
                    f"{ev.confidence:.2f} | {r.outcome_policy} |"
                )
            else:
                lines.append(
                    f"| {r.candidate_id} | - | - | - | {r.outcome_policy} |"
                )
    lines.append("")

    lines.append("## Post-execution")
    lines.append("")
    lines.append(
        f"- health: **{report.post_health_status.value.upper()}** "
        f"({report.post_health_score:.2f})"
    )
    lines.append(
        f"- top hypothesis: `{report.post_top_hypothesis or '(none)'}`"
    )
    lines.append(
        f"- ambiguity resolved: "
        f"{'yes' if report.ambiguity_resolved else 'no'}"
    )
    lines.append(
        f"- health score delta: {report.health_score_delta:+.2f}"
    )
    lines.append("")

    lines.append("## Before/after health")
    lines.append("")
    lines.append(
        f"- {report.pre_health_status.value.upper()} "
        f"{report.pre_health_score:.2f}  ->  "
        f"{report.post_health_status.value.upper()} "
        f"{report.post_health_score:.2f}"
    )
    lines.append("")

    lines.append("## Updated recommendation")
    lines.append("")
    rec = report.post_recommendation
    if rec.ranked_values:
        lines.append("| candidate | score |")
        lines.append("|---|---|")
        for cv in rec.ranked_values[:3]:
            lines.append(
                f"| {cv.candidate.candidate_id} | {cv.score:.2f} |"
            )
    else:
        lines.append("(none)")
    lines.append("")

    lines.append("## Caveats")
    lines.append("")
    for c in report.caveats:
        lines.append(f"- {c}")
    lines.append("")
    return "\n".join(lines)
