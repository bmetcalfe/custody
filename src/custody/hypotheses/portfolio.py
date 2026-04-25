"""Portfolio-level collection allocation (ADR-0021 Slice 16).

Extends the single-scenario planning pipeline (Slices 5-14) to
cross-scenario portfolio allocation.  Given multiple planner queue
items, optimized plans, and mission-value / counterfactual outputs,
selects a portfolio of candidate collect TYPES under shared resource
constraints.

This is a **prototype portfolio allocation engine**.  It does not
issue execution authorizations, command sensors, manage constellations,
or control any production planning process.  Every portfolio plan
carries an explicit non-claim caveat.

Design constraints
------------------

- Deterministic: no wall-clock, no randomness.
- Pure standard library: no new runtime dependencies.
- No imports from ``custody.detection.*``, ``custody.ingest.gfw_presence``,
  matcher runtime, Sentinel SDKs, or real-data loaders.
- Language: "portfolio allocation", "candidate collect type",
  "portfolio score".  No financial-value, execution-authorization, or
  schedule-control language.

Scope distinction
-----------------

Planner queue (Slice 14) ranks scenario work items for human attention.
Portfolio allocation selects a cross-scenario set of candidate collect
types under shared resource constraints.  This module consumes queue
items + plan / value reports; it does not duplicate queue ranking
logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from custody.hypotheses.counterfactual import (
    CounterfactualCollectAssessment,
    CounterfactualReport,
)
from custody.hypotheses.mission_value import (
    MissionValueAssessment,
    MissionValueReport,
)
from custody.hypotheses.optimizer import PlanItem, PlanOptimizationReport
from custody.hypotheses.planner_queue import (
    PlannerQueueItem,
    PlannerQueueStatus,
)


# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PortfolioConstraint:
    budget: float
    max_collects: int
    max_collects_per_scenario: int = 2
    required_scenario_ids: tuple[str, ...] = ()
    excluded_scenario_ids: tuple[str, ...] = ()
    required_candidate_ids: tuple[str, ...] = ()
    excluded_candidate_ids: tuple[str, ...] = ()
    include_closed_items: bool = False


@dataclass(frozen=True)
class PortfolioCandidate:
    scenario_id: str
    candidate_id: str
    label: str
    cost: float
    planning_utility: float
    mission_value_proxy: float
    expected_ambiguity_resolution: float
    expected_health_score_delta: float
    queue_priority_score: float
    queue_status: str
    reason: str
    caveats: tuple[str, ...]


@dataclass(frozen=True)
class PortfolioPlanItem:
    scenario_id: str
    candidate_id: str
    label: str
    cost: float
    portfolio_score: float
    planning_utility: float
    queue_priority_score: float
    reason: str


@dataclass(frozen=True)
class PortfolioAllocationPlan:
    strategy: str
    selected_items: tuple[PortfolioPlanItem, ...]
    total_cost: float
    total_portfolio_score: float
    total_planning_utility: float
    total_expected_ambiguity_resolution: float
    scenario_coverage_count: int
    constraint: PortfolioConstraint
    summary: str
    caveats: tuple[str, ...]


@dataclass(frozen=True)
class PortfolioAllocationReport:
    scenario_ids: tuple[str, ...]
    constraint: PortfolioConstraint
    candidate_pool: tuple[PortfolioCandidate, ...]
    exhaustive_plan: PortfolioAllocationPlan
    greedy_plan: PortfolioAllocationPlan
    recommended_plan: PortfolioAllocationPlan
    comparison_summary: str


# ---------------------------------------------------------------------------
# Constants + helpers
# ---------------------------------------------------------------------------


# Caveat strings phrased to honour the Slice 16 language guardrails;
# see the dispatch design constraint #4 for the banned-bigram list.
_BASE_CAVEATS: tuple[str, ...] = (
    "portfolio allocation is decision support only",
    "candidate collect types are prototype recommendations, not "
    "execution authorizations",
    "downstream execution and platform scheduling are out of scope for "
    "this module",
    "planning utility and mission value are prototype proxies, not "
    "financial estimates",
)

_RECOMMENDED_EXHAUSTIVE_THRESHOLD = 14

_CLOSED_STATUSES: frozenset[PlannerQueueStatus] = frozenset({
    PlannerQueueStatus.APPROVED,
    PlannerQueueStatus.REJECTED,
    PlannerQueueStatus.RESOLVED,
})


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _round4(x: float) -> float:
    return round(float(x), 4)


def _portfolio_score(c: PortfolioCandidate) -> float:
    raw = (
        0.30 * _clamp01(c.queue_priority_score)
        + 0.25 * _clamp01(c.planning_utility)
        + 0.20 * _clamp01(c.mission_value_proxy)
        + 0.15 * _clamp01(c.expected_ambiguity_resolution)
        + 0.10 * _clamp01(c.expected_health_score_delta)
    )
    return _round4(_clamp01(raw))


# ---------------------------------------------------------------------------
# Candidate construction
# ---------------------------------------------------------------------------


def _index_by_scenario(items: tuple) -> dict[str, object]:
    out: dict[str, object] = {}
    for it in items:
        out[it.scenario_id] = it
    return out


def _mission_value_lookup(
    mv: MissionValueReport, candidate_id: str,
) -> float:
    for a in mv.ranked_assessments:
        if a.candidate_id == candidate_id:
            return a.total_value
    return 0.0


def _counterfactual_lookup(
    cf: CounterfactualReport, candidate_id: str,
) -> tuple[float, float]:
    """Return (expected_ambiguity_resolution, expected_health_score_delta)."""
    for a in cf.ranked_assessments:
        if a.candidate_id == candidate_id:
            return (
                a.expected_ambiguity_resolution,
                a.expected_health_score_delta,
            )
    return (0.0, 0.0)


def build_portfolio_candidates(
    *,
    queue_items: tuple[PlannerQueueItem, ...] | list[PlannerQueueItem],
    optimization_reports: tuple[PlanOptimizationReport, ...] | list[PlanOptimizationReport],
    mission_value_reports: tuple[MissionValueReport, ...] | list[MissionValueReport],
    counterfactual_reports: tuple[CounterfactualReport, ...] | list[CounterfactualReport],
) -> tuple[PortfolioCandidate, ...]:
    """Join per-scenario reports into a flat candidate pool.

    For each queue item, look up the corresponding reports by
    ``scenario_id`` and emit one :class:`PortfolioCandidate` per
    ``optimization_report.all_candidates`` entry.  Output is sorted by
    ``(scenario_id, candidate_id)`` for determinism.
    """
    opt_by_scenario = _index_by_scenario(tuple(optimization_reports))
    mv_by_scenario = _index_by_scenario(tuple(mission_value_reports))
    cf_by_scenario = _index_by_scenario(tuple(counterfactual_reports))

    out: list[PortfolioCandidate] = []
    for q in queue_items:
        opt = opt_by_scenario.get(q.scenario_id)
        mv = mv_by_scenario.get(q.scenario_id)
        cf = cf_by_scenario.get(q.scenario_id)
        if opt is None or mv is None or cf is None:
            continue
        for plan_item in opt.all_candidates:
            mv_total = _mission_value_lookup(mv, plan_item.candidate_id)
            amb_res, h_delta_alt = _counterfactual_lookup(
                cf, plan_item.candidate_id,
            )
            out.append(PortfolioCandidate(
                scenario_id=q.scenario_id,
                candidate_id=plan_item.candidate_id,
                label=plan_item.label,
                cost=plan_item.cost,
                planning_utility=plan_item.value,
                mission_value_proxy=_round4(mv_total),
                expected_ambiguity_resolution=_round4(amb_res),
                expected_health_score_delta=_round4(
                    plan_item.expected_health_score_delta
                ),
                queue_priority_score=q.priority_score,
                queue_status=q.status.value,
                reason=plan_item.reason,
                caveats=("portfolio candidate sourced from upstream plan items",),
            ))

    out.sort(key=lambda c: (c.scenario_id, c.candidate_id))
    return tuple(out)


# ---------------------------------------------------------------------------
# Eligibility filter
# ---------------------------------------------------------------------------


def _eligible_candidates(
    candidates: tuple[PortfolioCandidate, ...],
    constraint: PortfolioConstraint,
) -> tuple[tuple[PortfolioCandidate, ...], tuple[str, ...]]:
    """Apply default closed-status filter + scenario / candidate excludes.

    Returns ``(filtered, extra_caveats)``.
    """
    excluded_scenarios = set(constraint.excluded_scenario_ids)
    excluded_candidates = set(constraint.excluded_candidate_ids)
    extras: list[str] = []
    closed_values = {s.value for s in _CLOSED_STATUSES}

    keep: list[PortfolioCandidate] = []
    saw_closed = False
    for c in candidates:
        if c.scenario_id in excluded_scenarios:
            continue
        if c.candidate_id in excluded_candidates:
            continue
        if c.queue_status in closed_values:
            saw_closed = True
            if not constraint.include_closed_items:
                continue
        keep.append(c)
    if constraint.include_closed_items and saw_closed:
        extras.append(
            "closed queue items included; results may reflect already-"
            "actioned scenarios"
        )
    return tuple(keep), tuple(extras)


# ---------------------------------------------------------------------------
# Plan-construction helpers
# ---------------------------------------------------------------------------


def _items_for(subset: tuple[PortfolioCandidate, ...]) -> tuple[PortfolioPlanItem, ...]:
    out: list[PortfolioPlanItem] = []
    for c in subset:
        out.append(PortfolioPlanItem(
            scenario_id=c.scenario_id,
            candidate_id=c.candidate_id,
            label=c.label,
            cost=_round4(c.cost),
            portfolio_score=_portfolio_score(c),
            planning_utility=_round4(c.planning_utility),
            queue_priority_score=_round4(c.queue_priority_score),
            reason=c.reason,
        ))
    out.sort(key=lambda p: (p.scenario_id, p.candidate_id))
    return tuple(out)


def _empty_plan(
    strategy: str,
    constraint: PortfolioConstraint,
    extra_caveats: tuple[str, ...],
    summary: str,
) -> PortfolioAllocationPlan:
    return PortfolioAllocationPlan(
        strategy=strategy,
        selected_items=(),
        total_cost=0.0,
        total_portfolio_score=0.0,
        total_planning_utility=0.0,
        total_expected_ambiguity_resolution=0.0,
        scenario_coverage_count=0,
        constraint=constraint,
        summary=summary,
        caveats=_BASE_CAVEATS + extra_caveats,
    )


def _plan_from_subset(
    strategy: str,
    subset: tuple[PortfolioCandidate, ...],
    constraint: PortfolioConstraint,
    extra_caveats: tuple[str, ...],
) -> PortfolioAllocationPlan:
    items = _items_for(subset)
    total_cost = _round4(sum(c.cost for c in subset))
    total_score = _round4(sum(_portfolio_score(c) for c in subset))
    total_utility = _round4(sum(c.planning_utility for c in subset))
    total_amb = _round4(sum(c.expected_ambiguity_resolution for c in subset))
    coverage = len({c.scenario_id for c in subset})
    if not subset:
        summary = (
            f"{strategy} strategy returned no candidates under "
            f"budget={constraint.budget:.2f}, "
            f"max_collects={constraint.max_collects}"
        )
    else:
        joined = ", ".join(
            f"{c.scenario_id}/{c.candidate_id}" for c in items
        )
        summary = (
            f"{strategy} strategy selected {len(subset)} candidate(s): "
            f"{joined}; total portfolio score {total_score:.2f}, "
            f"scenario coverage {coverage}"
        )
    return PortfolioAllocationPlan(
        strategy=strategy,
        selected_items=items,
        total_cost=total_cost,
        total_portfolio_score=total_score,
        total_planning_utility=total_utility,
        total_expected_ambiguity_resolution=total_amb,
        scenario_coverage_count=coverage,
        constraint=constraint,
        summary=summary,
        caveats=_BASE_CAVEATS + extra_caveats,
    )


# ---------------------------------------------------------------------------
# Constraint feasibility checks
# ---------------------------------------------------------------------------


def _required_constraint_infeasible(
    candidates: tuple[PortfolioCandidate, ...],
    constraint: PortfolioConstraint,
) -> str | None:
    """Return None when feasibility is plausible, else a short reason."""
    pool_scenarios = {c.scenario_id for c in candidates}
    pool_ids = {c.candidate_id for c in candidates}
    for sid in constraint.required_scenario_ids:
        if sid not in pool_scenarios:
            return (
                f"infeasible constraint: required scenario {sid!r} is not "
                f"available in the candidate pool"
            )
    for cid in constraint.required_candidate_ids:
        if cid not in pool_ids:
            return (
                f"infeasible constraint: required candidate {cid!r} is not "
                f"available in the candidate pool"
            )
    if len(constraint.required_candidate_ids) > constraint.max_collects:
        return (
            f"infeasible constraint: {len(constraint.required_candidate_ids)} "
            f"required candidates exceeds max_collects={constraint.max_collects}"
        )
    return None


def _subset_satisfies_constraint(
    subset: tuple[PortfolioCandidate, ...],
    constraint: PortfolioConstraint,
) -> bool:
    if sum(c.cost for c in subset) > constraint.budget + 1e-9:
        return False

    per_scenario: dict[str, int] = {}
    for c in subset:
        per_scenario[c.scenario_id] = per_scenario.get(c.scenario_id, 0) + 1
        if per_scenario[c.scenario_id] > constraint.max_collects_per_scenario:
            return False

    selected_ids = {c.candidate_id for c in subset}
    for required in constraint.required_candidate_ids:
        if required not in selected_ids:
            return False

    selected_scenarios = {c.scenario_id for c in subset}
    for required in constraint.required_scenario_ids:
        if required not in selected_scenarios:
            return False

    return True


# ---------------------------------------------------------------------------
# Exhaustive optimizer
# ---------------------------------------------------------------------------


def _exhaustive_sort_key(subset: tuple[PortfolioCandidate, ...]) -> tuple:
    """Best subset comes first under min(); tie-break per dispatch."""
    total_score = sum(_portfolio_score(c) for c in subset)
    coverage = len({c.scenario_id for c in subset})
    total_amb = sum(c.expected_ambiguity_resolution for c in subset)
    total_cost = sum(c.cost for c in subset)
    ids = tuple(sorted((c.scenario_id, c.candidate_id) for c in subset))
    return (-total_score, -coverage, -total_amb, total_cost, ids)


def optimize_portfolio_exhaustive(
    candidates: tuple[PortfolioCandidate, ...],
    *,
    constraint: PortfolioConstraint,
) -> PortfolioAllocationPlan:
    """Exhaustive portfolio search over subsets up to ``max_collects`` size."""
    infeasibility = _required_constraint_infeasible(candidates, constraint)
    if infeasibility is not None:
        return _empty_plan(
            "exhaustive", constraint,
            extra_caveats=(infeasibility,),
            summary=(
                f"exhaustive strategy returned no feasible portfolio: "
                f"{infeasibility}"
            ),
        )

    best: tuple[PortfolioCandidate, ...] = ()
    best_key = _exhaustive_sort_key(())
    found_feasible = False
    cap = min(constraint.max_collects, len(candidates))

    for k in range(0, cap + 1):
        for combo in combinations(candidates, k):
            if not _subset_satisfies_constraint(combo, constraint):
                continue
            found_feasible = True
            key = _exhaustive_sort_key(combo)
            if key < best_key:
                best = combo
                best_key = key

    if not found_feasible:
        return _empty_plan(
            "exhaustive", constraint,
            extra_caveats=("no feasible portfolio under given constraints",),
            summary="exhaustive strategy returned no feasible portfolio",
        )

    return _plan_from_subset("exhaustive", best, constraint, ())


# ---------------------------------------------------------------------------
# Greedy optimizer
# ---------------------------------------------------------------------------


def _greedy_sort_key(c: PortfolioCandidate) -> tuple:
    score = _portfolio_score(c)
    cost = max(c.cost, 0.01)
    return (
        -(score / cost),
        -score,
        -c.queue_priority_score,
        c.scenario_id,
        c.candidate_id,
    )


def optimize_portfolio_greedy(
    candidates: tuple[PortfolioCandidate, ...],
    *,
    constraint: PortfolioConstraint,
) -> PortfolioAllocationPlan:
    """Greedy portfolio selection ordered by score / cost descending."""
    infeasibility = _required_constraint_infeasible(candidates, constraint)
    if infeasibility is not None:
        return _empty_plan(
            "greedy", constraint,
            extra_caveats=(infeasibility,),
            summary=(
                f"greedy strategy returned no feasible portfolio: "
                f"{infeasibility}"
            ),
        )

    excluded_candidates = set(constraint.excluded_candidate_ids)
    excluded_scenarios = set(constraint.excluded_scenario_ids)
    required_candidate_ids = set(constraint.required_candidate_ids)
    required_scenario_ids = set(constraint.required_scenario_ids)

    by_id = {c.candidate_id: c for c in candidates}
    selected: list[PortfolioCandidate] = []
    selected_ids: set[str] = set()
    per_scenario: dict[str, int] = {}
    total_cost = 0.0

    def _can_take(c: PortfolioCandidate) -> bool:
        if c.candidate_id in selected_ids:
            return False
        if c.scenario_id in excluded_scenarios:
            return False
        if c.candidate_id in excluded_candidates:
            return False
        if total_cost + c.cost > constraint.budget + 1e-9:
            return False
        if (per_scenario.get(c.scenario_id, 0) + 1
                > constraint.max_collects_per_scenario):
            return False
        if len(selected) >= constraint.max_collects:
            return False
        return True

    # Required candidates first (sorted by candidate_id for determinism).
    for rid in sorted(required_candidate_ids):
        c = by_id.get(rid)
        if c is None or not _can_take(c):
            return _empty_plan(
                "greedy", constraint,
                extra_caveats=(
                    f"infeasible constraint: required candidate {rid!r} "
                    f"could not be included under the supplied limits",
                ),
                summary=(
                    f"greedy strategy returned no feasible portfolio: "
                    f"required candidate {rid!r} could not fit"
                ),
            )
        selected.append(c)
        selected_ids.add(c.candidate_id)
        per_scenario[c.scenario_id] = per_scenario.get(c.scenario_id, 0) + 1
        total_cost += c.cost

    # Then greedily fill remaining capacity.
    remaining = sorted(
        (c for c in candidates if c.candidate_id not in selected_ids),
        key=_greedy_sort_key,
    )
    for c in remaining:
        if _can_take(c):
            selected.append(c)
            selected_ids.add(c.candidate_id)
            per_scenario[c.scenario_id] = (
                per_scenario.get(c.scenario_id, 0) + 1
            )
            total_cost += c.cost

    # Required scenarios must end up represented.
    selected_scenarios = {c.scenario_id for c in selected}
    missing_scenarios = required_scenario_ids - selected_scenarios
    if missing_scenarios:
        missing = ", ".join(sorted(missing_scenarios))
        return _empty_plan(
            "greedy", constraint,
            extra_caveats=(
                f"infeasible constraint: required scenario(s) {missing} "
                f"could not be represented under the supplied limits",
            ),
            summary=(
                f"greedy strategy returned no feasible portfolio: "
                f"missing required scenarios {missing}"
            ),
        )

    return _plan_from_subset("greedy", tuple(selected), constraint, ())


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _default_constraint() -> PortfolioConstraint:
    return PortfolioConstraint(budget=1.5, max_collects=3, max_collects_per_scenario=2)


def allocate_portfolio(
    candidates: tuple[PortfolioCandidate, ...],
    *,
    constraint: PortfolioConstraint | None = None,
) -> PortfolioAllocationReport:
    """Compose both baselines + a recommended portfolio over the candidate pool."""
    resolved = (
        constraint if constraint is not None else _default_constraint()
    )

    eligible, extra_caveats = _eligible_candidates(candidates, resolved)

    exhaustive = optimize_portfolio_exhaustive(eligible, constraint=resolved)
    greedy = optimize_portfolio_greedy(eligible, constraint=resolved)

    if extra_caveats:
        # Layer in any closed-items-included caveat to both plans.
        exhaustive = PortfolioAllocationPlan(
            strategy=exhaustive.strategy,
            selected_items=exhaustive.selected_items,
            total_cost=exhaustive.total_cost,
            total_portfolio_score=exhaustive.total_portfolio_score,
            total_planning_utility=exhaustive.total_planning_utility,
            total_expected_ambiguity_resolution=(
                exhaustive.total_expected_ambiguity_resolution
            ),
            scenario_coverage_count=exhaustive.scenario_coverage_count,
            constraint=exhaustive.constraint,
            summary=exhaustive.summary,
            caveats=exhaustive.caveats + extra_caveats,
        )
        greedy = PortfolioAllocationPlan(
            strategy=greedy.strategy,
            selected_items=greedy.selected_items,
            total_cost=greedy.total_cost,
            total_portfolio_score=greedy.total_portfolio_score,
            total_planning_utility=greedy.total_planning_utility,
            total_expected_ambiguity_resolution=(
                greedy.total_expected_ambiguity_resolution
            ),
            scenario_coverage_count=greedy.scenario_coverage_count,
            constraint=greedy.constraint,
            summary=greedy.summary,
            caveats=greedy.caveats + extra_caveats,
        )

    if len(eligible) <= _RECOMMENDED_EXHAUSTIVE_THRESHOLD:
        recommended = exhaustive
        rationale = (
            f"recommended portfolio uses the exhaustive strategy (eligible "
            f"candidate count {len(eligible)} <= threshold "
            f"{_RECOMMENDED_EXHAUSTIVE_THRESHOLD})"
        )
    else:
        recommended = greedy
        rationale = (
            f"recommended portfolio uses the greedy strategy (eligible "
            f"candidate count {len(eligible)} > threshold "
            f"{_RECOMMENDED_EXHAUSTIVE_THRESHOLD})"
        )

    same = tuple(
        (i.scenario_id, i.candidate_id) for i in exhaustive.selected_items
    ) == tuple(
        (i.scenario_id, i.candidate_id) for i in greedy.selected_items
    )
    if same:
        comparison = (
            f"exhaustive and greedy converged on the same portfolio "
            f"({len(exhaustive.selected_items)} candidate(s)); {rationale}"
        )
    else:
        comparison = (
            f"exhaustive (score {exhaustive.total_portfolio_score:.2f}) vs "
            f"greedy (score {greedy.total_portfolio_score:.2f}); {rationale}"
        )

    scenario_ids = tuple(sorted({c.scenario_id for c in candidates}))

    return PortfolioAllocationReport(
        scenario_ids=scenario_ids,
        constraint=resolved,
        candidate_pool=candidates,
        exhaustive_plan=exhaustive,
        greedy_plan=greedy,
        recommended_plan=recommended,
        comparison_summary=comparison,
    )


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def _candidate_to_dict(c: PortfolioCandidate) -> dict:
    return {
        "scenario_id": c.scenario_id,
        "candidate_id": c.candidate_id,
        "label": c.label,
        "cost": c.cost,
        "planning_utility": c.planning_utility,
        "mission_value_proxy": c.mission_value_proxy,
        "expected_ambiguity_resolution": c.expected_ambiguity_resolution,
        "expected_health_score_delta": c.expected_health_score_delta,
        "queue_priority_score": c.queue_priority_score,
        "queue_status": c.queue_status,
        "portfolio_score": _portfolio_score(c),
        "reason": c.reason,
        "caveats": list(c.caveats),
    }


def _plan_item_to_dict(p: PortfolioPlanItem) -> dict:
    return {
        "scenario_id": p.scenario_id,
        "candidate_id": p.candidate_id,
        "label": p.label,
        "cost": p.cost,
        "portfolio_score": p.portfolio_score,
        "planning_utility": p.planning_utility,
        "queue_priority_score": p.queue_priority_score,
        "reason": p.reason,
    }


def _constraint_to_dict(c: PortfolioConstraint) -> dict:
    return {
        "budget": c.budget,
        "max_collects": c.max_collects,
        "max_collects_per_scenario": c.max_collects_per_scenario,
        "required_scenario_ids": list(c.required_scenario_ids),
        "excluded_scenario_ids": list(c.excluded_scenario_ids),
        "required_candidate_ids": list(c.required_candidate_ids),
        "excluded_candidate_ids": list(c.excluded_candidate_ids),
        "include_closed_items": c.include_closed_items,
    }


def _plan_to_dict(plan: PortfolioAllocationPlan) -> dict:
    return {
        "strategy": plan.strategy,
        "selected_items": [_plan_item_to_dict(i) for i in plan.selected_items],
        "total_cost": plan.total_cost,
        "total_portfolio_score": plan.total_portfolio_score,
        "total_planning_utility": plan.total_planning_utility,
        "total_expected_ambiguity_resolution": (
            plan.total_expected_ambiguity_resolution
        ),
        "scenario_coverage_count": plan.scenario_coverage_count,
        "constraint": _constraint_to_dict(plan.constraint),
        "summary": plan.summary,
        "caveats": list(plan.caveats),
    }


def portfolio_report_to_json_object(report: PortfolioAllocationReport) -> dict:
    """Plain-dict representation suitable for ``json.dumps``."""
    return {
        "scenario_ids": list(report.scenario_ids),
        "constraint": _constraint_to_dict(report.constraint),
        "candidate_pool": [_candidate_to_dict(c) for c in report.candidate_pool],
        "recommended_plan": _plan_to_dict(report.recommended_plan),
        "exhaustive_plan": _plan_to_dict(report.exhaustive_plan),
        "greedy_plan": _plan_to_dict(report.greedy_plan),
        "comparison_summary": report.comparison_summary,
        "caveats": list(_BASE_CAVEATS),
    }


def _render_constraint(out: list[str], c: PortfolioConstraint) -> None:
    out.append("\nGlobal constraints:\n")
    out.append(f"  budget: {c.budget:.2f}\n")
    out.append(f"  max collects: {c.max_collects}\n")
    out.append(
        f"  max collects per scenario: {c.max_collects_per_scenario}\n"
    )
    if c.required_scenario_ids:
        out.append(
            f"  required scenarios: {', '.join(c.required_scenario_ids)}\n"
        )
    if c.excluded_scenario_ids:
        out.append(
            f"  excluded scenarios: {', '.join(c.excluded_scenario_ids)}\n"
        )
    if c.required_candidate_ids:
        out.append(
            f"  required candidates: {', '.join(c.required_candidate_ids)}\n"
        )
    if c.excluded_candidate_ids:
        out.append(
            f"  excluded candidates: {', '.join(c.excluded_candidate_ids)}\n"
        )
    if c.include_closed_items:
        out.append("  include closed items: yes\n")


def format_portfolio_text(report: PortfolioAllocationReport) -> str:
    parts: list[str] = []
    parts.append("PORTFOLIO ALLOCATION PLAN\n")
    parts.append("=========================\n")

    _render_constraint(parts, report.constraint)

    parts.append("\nCandidate pool:\n")
    if not report.candidate_pool:
        parts.append("  (no candidates available)\n")
    for c in report.candidate_pool:
        parts.append(
            f"  {c.scenario_id} / {c.candidate_id:24s}  cost={c.cost:.2f}  "
            f"score={_portfolio_score(c):.2f}\n"
        )

    plan = report.recommended_plan
    parts.append(
        f"\nRecommended portfolio ({plan.strategy}):\n"
    )
    if not plan.selected_items:
        parts.append("  (no candidates selected)\n")
    for i, item in enumerate(plan.selected_items, start=1):
        parts.append(
            f"  {i}. {item.scenario_id} / {item.candidate_id:24s}  "
            f"cost={item.cost:.2f}  score={item.portfolio_score:.2f}\n"
        )

    parts.append("\nTotals:\n")
    parts.append(
        f"  cost: {plan.total_cost:.2f} / {report.constraint.budget:.2f}\n"
    )
    parts.append(
        f"  portfolio score: {plan.total_portfolio_score:.2f}\n"
    )
    parts.append(
        f"  Scenario coverage: {plan.scenario_coverage_count}\n"
    )

    parts.append("\nGreedy vs exhaustive comparison:\n")
    parts.append(f"  {report.comparison_summary}\n")

    parts.append("\nCaveats:\n")
    for c in plan.caveats:
        parts.append(f"  - {c}\n")

    return "".join(parts)


def format_portfolio_markdown(report: PortfolioAllocationReport) -> str:
    parts: list[str] = []
    parts.append("# Portfolio Allocation Plan\n\n")

    parts.append("## Global constraints\n\n")
    c = report.constraint
    parts.append(f"- budget: {c.budget:.2f}\n")
    parts.append(f"- max collects: {c.max_collects}\n")
    parts.append(
        f"- max collects per scenario: {c.max_collects_per_scenario}\n"
    )
    if c.include_closed_items:
        parts.append("- include closed items: yes\n")
    parts.append("\n")

    parts.append("## Candidate pool\n\n")
    parts.append("| scenario | candidate | cost | portfolio score |\n")
    parts.append("|---|---|---|---|\n")
    for cand in report.candidate_pool:
        parts.append(
            f"| {cand.scenario_id} | {cand.candidate_id} | "
            f"{cand.cost:.2f} | {_portfolio_score(cand):.2f} |\n"
        )
    parts.append("\n")

    plan = report.recommended_plan
    parts.append(f"## Recommended portfolio ({plan.strategy})\n\n")
    if not plan.selected_items:
        parts.append("(no candidates selected)\n\n")
    else:
        parts.append("| # | scenario | candidate | cost | score |\n")
        parts.append("|---|---|---|---|---|\n")
        for i, item in enumerate(plan.selected_items, start=1):
            parts.append(
                f"| {i} | {item.scenario_id} | {item.candidate_id} | "
                f"{item.cost:.2f} | {item.portfolio_score:.2f} |\n"
            )
        parts.append("\n")

    parts.append("## Totals\n\n")
    parts.append(
        f"- cost: {plan.total_cost:.2f} / {report.constraint.budget:.2f}\n"
    )
    parts.append(
        f"- portfolio score: {plan.total_portfolio_score:.2f}\n"
    )
    parts.append(
        f"- Scenario coverage: {plan.scenario_coverage_count}\n\n"
    )

    parts.append("## Comparison\n\n")
    parts.append(f"{report.comparison_summary}\n\n")

    parts.append("## Caveats\n\n")
    for c in plan.caveats:
        parts.append(f"- {c}\n")

    return "".join(parts)
