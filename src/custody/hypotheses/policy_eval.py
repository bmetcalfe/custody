"""Heuristic collection-policy evaluator (ADR-0021 Slice 12).

Compares six named, deterministic heuristic collection policies against
a single ranked candidate set under a shared :class:`PlanConstraint`,
using the upstream Slice 5 / 9 / 10 / 11 outputs.

This is a **deterministic policy-evaluation substrate**, not a trained
RL agent and not live tasking.  The harness is RL-ready in the sense
that it produces a deterministic comparison signal across named
strategies; it does not learn, schedule, or task anything.

Scope guardrails
----------------

- Pure standard library; no new runtime dependencies.
- No imports from ``custody.detection.*``, ``custody.ingest.gfw_presence``,
  ``custody.fusion.tracker``, ``custody.taskrecommendation``, or
  Sentinel SDKs.
- No wall-clock anywhere.
- Language: "heuristic policy", "policy evaluation", "planning utility
  proxy", "RL-ready evaluation substrate".  Never "trained RL",
  "reinforcement learning agent", "autonomous scheduler", "production
  scheduler", "tasking order", "live tasking", "revenue dollars".

Six required policies
---------------------

  1. ``value_optimized``               — uses the Slice 11 optimizer's
     recommended plan directly.
  2. ``ambiguity_first``                — ranks by Slice 10 expected
     ambiguity resolution.
  3. ``low_cost_first``                 — ranks by relative cost ascending.
  4. ``sar_repeat_first``               — fixed SAR-leaning priority order.
  5. ``optical_disambiguation_first``   — fixed optical-leaning order.
  6. ``ais_context_first``              — fixed AIS-context order.

All policies obey ``PlanConstraint`` (budget, max_collects, required,
excluded).  A policy that cannot produce a feasible selection emits an
empty plan plus an "infeasible" caveat.
"""
from __future__ import annotations

from dataclasses import dataclass

from custody.hypotheses.collection_value import (
    CollectionRecommendation,
    CollectionValue,
)
from custody.hypotheses.counterfactual import (
    CounterfactualCollectAssessment,
    CounterfactualReport,
)
from custody.hypotheses.mission_value import (
    MissionValueAssessment,
    MissionValueReport,
)
from custody.hypotheses.optimizer import (
    PlanConstraint,
    PlanOptimizationReport,
)


# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyDefinition:
    """Static description of a named heuristic policy."""
    policy_id: str
    label: str
    description: str


@dataclass(frozen=True)
class PolicyEvaluation:
    """One policy's ranked result."""
    policy_id: str
    label: str
    selected_candidate_ids: tuple[str, ...]
    total_cost: float
    planning_utility: float
    mission_value_total: float
    expected_ambiguity_resolution: float
    expected_health_score_delta: float
    rank: int
    reason: str
    caveats: tuple[str, ...]


@dataclass(frozen=True)
class PolicyEvaluationReport:
    """Comparison of multiple heuristic policies under one constraint."""
    scenario_id: str
    constraint: PlanConstraint
    evaluations: tuple[PolicyEvaluation, ...]
    winning_policy_id: str
    summary: str


# ---------------------------------------------------------------------------
# Policy definitions + fixed priority orders
# ---------------------------------------------------------------------------


_POLICY_DEFINITIONS: dict[str, PolicyDefinition] = {
    "value_optimized": PolicyDefinition(
        policy_id="value_optimized",
        label="Value optimized",
        description=(
            "Mirrors the Slice 11 optimizer's recommended plan; the "
            "optimizer already searched for the highest planning-utility "
            "subset under the same constraint."
        ),
    ),
    "ambiguity_first": PolicyDefinition(
        policy_id="ambiguity_first",
        label="Ambiguity-first",
        description=(
            "Ranks candidates by counterfactual expected ambiguity "
            "resolution (descending), tie-broken by candidate_id."
        ),
    ),
    "low_cost_first": PolicyDefinition(
        policy_id="low_cost_first",
        label="Low-cost-first",
        description=(
            "Ranks candidates by relative cost ascending, then by per-"
            "candidate planning utility descending, then candidate_id."
        ),
    ),
    "sar_repeat_first": PolicyDefinition(
        policy_id="sar_repeat_first",
        label="SAR-repeat-first",
        description=(
            "Fixed priority order favouring same-geometry SAR repeats."
        ),
    ),
    "optical_disambiguation_first": PolicyDefinition(
        policy_id="optical_disambiguation_first",
        label="Optical-disambiguation-first",
        description=(
            "Fixed priority order favouring optical context for "
            "structure-vs-activity ambiguity."
        ),
    ),
    "ais_context_first": PolicyDefinition(
        policy_id="ais_context_first",
        label="AIS-context-first",
        description=(
            "Fixed priority order favouring AIS coverage queries for "
            "AIS-dark vs ordinary-traffic ambiguity."
        ),
    ),
}

_DEFAULT_POLICY_IDS: tuple[str, ...] = (
    "value_optimized",
    "ambiguity_first",
    "low_cost_first",
    "sar_repeat_first",
    "optical_disambiguation_first",
    "ais_context_first",
)

_FIXED_PRIORITIES: dict[str, tuple[str, ...]] = {
    "sar_repeat_first": (
        "repeat_sar",
        "cross_geometry_sar",
        "higher_resolution_sar",
        "optical_context",
        "ais_coverage_query",
        "wait_or_monitor",
    ),
    "optical_disambiguation_first": (
        "optical_context",
        "cross_geometry_sar",
        "higher_resolution_sar",
        "repeat_sar",
        "ais_coverage_query",
        "wait_or_monitor",
    ),
    "ais_context_first": (
        "ais_coverage_query",
        "repeat_sar",
        "optical_context",
        "cross_geometry_sar",
        "higher_resolution_sar",
        "wait_or_monitor",
    ),
}


_BASE_CAVEATS: tuple[str, ...] = (
    "policies are deterministic heuristics; the substrate is RL-ready but "
    "no learned policies are evaluated here",
    "planning utility is a deterministic proxy, not a financial model",
    "recommendations name candidate collect types only; specific platform "
    "selection and schedules are downstream concerns",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _round4(x: float) -> float:
    return round(float(x), 4)


@dataclass(frozen=True)
class _CandidateInfo:
    """Per-candidate data needed by every policy."""
    candidate_id: str
    label: str
    cost: float
    mission_value: float
    expected_ambiguity_resolution: float
    expected_health_score_delta: float
    planning_utility: float


def _planning_utility(mv: float, amb_res: float, health_delta: float) -> float:
    """Same shape as the Slice 11 optimizer's per-candidate utility.

    Duplicated here intentionally so the policy-eval module stays
    decoupled from the optimizer's private helpers; both modules pin
    the formula in their own tests.
    """
    raw = (
        0.50 * mv
        + 0.30 * amb_res
        + 0.20 * _clamp01(health_delta)
    )
    return _round4(_clamp01(raw))


def _build_candidate_index(
    recommendation: CollectionRecommendation,
    mission_value_report: MissionValueReport,
    counterfactual_report: CounterfactualReport,
) -> dict[str, _CandidateInfo]:
    cv_by_id = {v.candidate.candidate_id: v for v in recommendation.ranked_values}
    mv_by_id = {a.candidate_id: a for a in mission_value_report.ranked_assessments}
    cf_by_id = {a.candidate_id: a for a in counterfactual_report.ranked_assessments}
    out: dict[str, _CandidateInfo] = {}
    for cid, cv in cv_by_id.items():
        mv = mv_by_id.get(cid)
        cf = cf_by_id.get(cid)
        if mv is None or cf is None:
            continue
        cost = cv.candidate.relative_cost
        if cost is None or cost < 0.0:
            cost = 0.5
        out[cid] = _CandidateInfo(
            candidate_id=cid,
            label=cv.candidate.label,
            cost=float(cost),
            mission_value=float(mv.total_value),
            expected_ambiguity_resolution=float(cf.expected_ambiguity_resolution),
            expected_health_score_delta=float(cf.expected_health_score_delta),
            planning_utility=_planning_utility(
                mv.total_value,
                cf.expected_ambiguity_resolution,
                cf.expected_health_score_delta,
            ),
        )
    return out


def _validate_required_excluded(
    candidates: dict[str, _CandidateInfo],
    constraint: PlanConstraint,
) -> str | None:
    """Return None when feasible, otherwise an infeasibility reason."""
    excluded = set(constraint.excluded_candidate_ids)
    required = sorted(set(constraint.required_candidate_ids))
    for rid in required:
        if rid in excluded:
            return (
                f"infeasible constraint: required candidate {rid!r} is also "
                f"in excluded list"
            )
        if rid not in candidates:
            return (
                f"infeasible constraint: required candidate {rid!r} is not "
                f"available in the recommendation"
            )
    if len(required) > constraint.max_collects:
        return (
            f"infeasible constraint: {len(required)} required candidates "
            f"exceeds max_collects={constraint.max_collects}"
        )
    required_cost = sum(candidates[rid].cost for rid in required)
    if required_cost > constraint.budget + 1e-9:
        return (
            f"infeasible constraint: required candidates total cost "
            f"{required_cost:.2f} exceeds budget {constraint.budget:.2f}"
        )
    return None


def _greedy_pick(
    candidates: dict[str, _CandidateInfo],
    constraint: PlanConstraint,
    ordered_ids: tuple[str, ...],
) -> tuple[_CandidateInfo, ...]:
    """Pick candidates in ``ordered_ids`` order while respecting constraints.

    Required candidates are added first (in candidate_id order, regardless
    of their position in ``ordered_ids``) so a policy whose priority list
    omits a required id still honours the constraint.  Excluded ids are
    skipped.
    """
    excluded = set(constraint.excluded_candidate_ids)
    required_ids = sorted(set(constraint.required_candidate_ids))
    selected: list[_CandidateInfo] = []
    selected_ids: set[str] = set()
    total_cost = 0.0

    for rid in required_ids:
        info = candidates[rid]
        selected.append(info)
        selected_ids.add(rid)
        total_cost += info.cost

    for cid in ordered_ids:
        if len(selected) >= constraint.max_collects:
            break
        if cid in selected_ids or cid in excluded:
            continue
        info = candidates.get(cid)
        if info is None:
            continue
        if total_cost + info.cost > constraint.budget + 1e-9:
            continue
        selected.append(info)
        selected_ids.add(cid)
        total_cost += info.cost

    return tuple(selected)


def _aggregate(selected: tuple[_CandidateInfo, ...]) -> tuple[float, float, float, float, float]:
    """Return (total_cost, total_utility, mv_total, amb_res, positive_delta_sum)."""
    total_cost = sum(c.cost for c in selected)
    total_utility = sum(c.planning_utility for c in selected)
    mv_total = sum(c.mission_value for c in selected)
    amb_res = sum(c.expected_ambiguity_resolution for c in selected)
    positive_delta = sum(
        max(0.0, c.expected_health_score_delta) for c in selected
    )
    return (
        _round4(total_cost),
        _round4(total_utility),
        _round4(mv_total),
        _round4(amb_res),
        _round4(positive_delta),
    )


# ---------------------------------------------------------------------------
# Policy strategies
# ---------------------------------------------------------------------------


def _select_value_optimized(
    candidates: dict[str, _CandidateInfo],
    optimization_report: PlanOptimizationReport,
) -> tuple[_CandidateInfo, ...]:
    """Mirror the optimizer's recommended plan."""
    selected: list[_CandidateInfo] = []
    for plan_item in optimization_report.recommended_plan.selected_items:
        info = candidates.get(plan_item.candidate_id)
        if info is not None:
            selected.append(info)
    return tuple(selected)


def _select_ambiguity_first(
    candidates: dict[str, _CandidateInfo],
    constraint: PlanConstraint,
) -> tuple[_CandidateInfo, ...]:
    ordered = tuple(
        c.candidate_id for c in sorted(
            candidates.values(),
            key=lambda c: (-c.expected_ambiguity_resolution, c.candidate_id),
        )
    )
    return _greedy_pick(candidates, constraint, ordered)


def _select_low_cost_first(
    candidates: dict[str, _CandidateInfo],
    constraint: PlanConstraint,
) -> tuple[_CandidateInfo, ...]:
    ordered = tuple(
        c.candidate_id for c in sorted(
            candidates.values(),
            key=lambda c: (c.cost, -c.planning_utility, c.candidate_id),
        )
    )
    return _greedy_pick(candidates, constraint, ordered)


def _select_fixed_priority(
    candidates: dict[str, _CandidateInfo],
    constraint: PlanConstraint,
    policy_id: str,
) -> tuple[_CandidateInfo, ...]:
    return _greedy_pick(candidates, constraint, _FIXED_PRIORITIES[policy_id])


def _evaluate_policy(
    policy_id: str,
    candidates: dict[str, _CandidateInfo],
    optimization_report: PlanOptimizationReport,
    constraint: PlanConstraint,
    infeasibility: str | None,
) -> PolicyEvaluation:
    definition = _POLICY_DEFINITIONS[policy_id]

    if infeasibility is not None:
        return PolicyEvaluation(
            policy_id=policy_id,
            label=definition.label,
            selected_candidate_ids=(),
            total_cost=0.0,
            planning_utility=0.0,
            mission_value_total=0.0,
            expected_ambiguity_resolution=0.0,
            expected_health_score_delta=0.0,
            rank=0,
            reason=(
                f"policy produced no feasible plan under constraints: "
                f"{infeasibility}"
            ),
            caveats=_BASE_CAVEATS + (infeasibility,),
        )

    if policy_id == "value_optimized":
        selected = _select_value_optimized(candidates, optimization_report)
    elif policy_id == "ambiguity_first":
        selected = _select_ambiguity_first(candidates, constraint)
    elif policy_id == "low_cost_first":
        selected = _select_low_cost_first(candidates, constraint)
    elif policy_id in _FIXED_PRIORITIES:
        selected = _select_fixed_priority(candidates, constraint, policy_id)
    else:  # defensive; validated upstream
        raise ValueError(f"unknown policy_id: {policy_id!r}")

    if not selected:
        return PolicyEvaluation(
            policy_id=policy_id,
            label=definition.label,
            selected_candidate_ids=(),
            total_cost=0.0,
            planning_utility=0.0,
            mission_value_total=0.0,
            expected_ambiguity_resolution=0.0,
            expected_health_score_delta=0.0,
            rank=0,
            reason="policy produced no feasible plan under constraints",
            caveats=_BASE_CAVEATS,
        )

    selected_sorted = tuple(sorted(selected, key=lambda c: c.candidate_id))
    total_cost, total_utility, mv_total, amb_res, pos_delta = _aggregate(
        selected_sorted,
    )
    ids = tuple(c.candidate_id for c in selected_sorted)
    return PolicyEvaluation(
        policy_id=policy_id,
        label=definition.label,
        selected_candidate_ids=ids,
        total_cost=total_cost,
        planning_utility=total_utility,
        mission_value_total=mv_total,
        expected_ambiguity_resolution=amb_res,
        expected_health_score_delta=pos_delta,
        rank=0,  # filled in after sort
        reason=(
            f"{definition.label} selected {len(ids)} candidate(s): "
            f"{', '.join(ids)}"
        ),
        caveats=_BASE_CAVEATS,
    )


def _rank_evaluations(
    evals: list[PolicyEvaluation],
) -> list[PolicyEvaluation]:
    """Sort by utility desc, amb_res desc, cost asc, policy_id asc; assign rank."""
    evals.sort(key=lambda e: (
        -e.planning_utility,
        -e.expected_ambiguity_resolution,
        e.total_cost,
        e.policy_id,
    ))
    ranked: list[PolicyEvaluation] = []
    for i, e in enumerate(evals, start=1):
        ranked.append(PolicyEvaluation(
            policy_id=e.policy_id,
            label=e.label,
            selected_candidate_ids=e.selected_candidate_ids,
            total_cost=e.total_cost,
            planning_utility=e.planning_utility,
            mission_value_total=e.mission_value_total,
            expected_ambiguity_resolution=e.expected_ambiguity_resolution,
            expected_health_score_delta=e.expected_health_score_delta,
            rank=i,
            reason=e.reason,
            caveats=e.caveats,
        ))
    return ranked


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _default_constraint() -> PlanConstraint:
    return PlanConstraint(budget=1.0, max_collects=2)


def evaluate_collection_policies(
    recommendation: CollectionRecommendation,
    mission_value_report: MissionValueReport,
    counterfactual_report: CounterfactualReport,
    optimization_report: PlanOptimizationReport,
    *,
    scenario_id: str,
    constraint: PlanConstraint | None = None,
    policies: tuple[str, ...] | None = None,
) -> PolicyEvaluationReport:
    """Evaluate the requested heuristic policies under one shared constraint.

    ``policies=None`` evaluates all six required policies.  Pass a tuple
    of policy_ids to evaluate a subset; unknown policy_ids raise
    :class:`ValueError`.
    """
    resolved_constraint = (
        constraint if constraint is not None else _default_constraint()
    )

    requested = (
        _DEFAULT_POLICY_IDS if policies is None else tuple(policies)
    )
    unknown = [p for p in requested if p not in _POLICY_DEFINITIONS]
    if unknown:
        valid = ", ".join(sorted(_POLICY_DEFINITIONS))
        raise ValueError(
            f"unknown policy_id(s) {unknown}; valid policies: {valid}"
        )
    # Preserve caller order while deduplicating.
    seen: set[str] = set()
    ordered_requested: list[str] = []
    for p in requested:
        if p not in seen:
            ordered_requested.append(p)
            seen.add(p)

    candidates = _build_candidate_index(
        recommendation, mission_value_report, counterfactual_report,
    )
    infeasibility = _validate_required_excluded(candidates, resolved_constraint)

    evals = [
        _evaluate_policy(
            pid, candidates, optimization_report, resolved_constraint,
            infeasibility,
        )
        for pid in ordered_requested
    ]
    ranked = _rank_evaluations(evals)
    winner = ranked[0]
    if winner.selected_candidate_ids:
        summary = (
            f"winning policy {winner.policy_id!r} at planning utility "
            f"{winner.planning_utility:.2f} ({len(winner.selected_candidate_ids)} "
            f"candidate(s)); ranking compares deterministic heuristics across "
            f"the requested policies"
        )
    else:
        summary = (
            "no policy produced a feasible plan under the supplied "
            "constraint; all policies returned empty selections"
        )

    return PolicyEvaluationReport(
        scenario_id=scenario_id,
        constraint=resolved_constraint,
        evaluations=tuple(ranked),
        winning_policy_id=winner.policy_id,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Text formatter
# ---------------------------------------------------------------------------


def format_policy_eval_text(report: PolicyEvaluationReport) -> str:
    """Human-readable policy-evaluation summary."""
    parts: list[str] = []
    title = f"COLLECTION POLICY EVALUATION - {report.scenario_id.upper()}"
    parts.append(title + "\n")
    parts.append("=" * len(title) + "\n")

    parts.append("\nConstraint\n")
    parts.append("----------\n")
    parts.append(f"  budget: {report.constraint.budget:.2f}\n")
    parts.append(f"  max collects: {report.constraint.max_collects}\n")
    if report.constraint.required_candidate_ids:
        parts.append(
            f"  required: {', '.join(report.constraint.required_candidate_ids)}\n"
        )
    if report.constraint.excluded_candidate_ids:
        parts.append(
            f"  excluded: {', '.join(report.constraint.excluded_candidate_ids)}\n"
        )

    parts.append("\nPolicies evaluated\n")
    parts.append("------------------\n")
    parts.append(
        "  " + ", ".join(e.policy_id for e in report.evaluations) + "\n"
    )

    parts.append("\nRanked policy results\n")
    parts.append("---------------------\n")
    for e in report.evaluations:
        candidates_str = (
            ", ".join(e.selected_candidate_ids)
            if e.selected_candidate_ids else "(none)"
        )
        parts.append(
            f"  {e.policy_id:30s}  utility={e.planning_utility:.2f}  "
            f"cost={e.total_cost:.2f}  candidates: {candidates_str}\n"
        )

    parts.append("\nWinning policy\n")
    parts.append("--------------\n")
    winner = next(
        e for e in report.evaluations if e.policy_id == report.winning_policy_id
    )
    parts.append(
        f"  {winner.policy_id} (planning utility {winner.planning_utility:.2f})\n"
    )

    parts.append("\nCaveats\n")
    parts.append("-------\n")
    for c in _BASE_CAVEATS:
        parts.append(f"  - {c}\n")

    parts.append(f"\n{report.summary}\n")
    return "".join(parts)
