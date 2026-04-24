"""Constrained collection-plan optimizer (ADR-0021 Slice 11).

Selects the best subset of candidate collect TYPES under simple
constraints (budget, max collects, required / excluded candidates) by
combining the upstream Slice 5 / 9 / 10 outputs into a single planning-
utility score.

Scope guardrails
----------------

- This is a **deterministic prototype optimizer over candidate collect
  TYPES**, not live tasking, not sensor scheduling, not autonomous
  constellation management, and not a calibrated financial model.
- Pure standard library; no scipy / sklearn / OR-Tools / pulp /
  networkx / external optimization dependency.
- No imports from ``custody.detection.*``, ``custody.ingest.gfw_presence``,
  ``custody.fusion.tracker``, ``custody.taskrecommendation``, or
  Sentinel SDKs.
- No wall-clock anywhere; output is fully deterministic across calls.

Planning utility (per candidate)
--------------------------------

  candidate_value =
      0.50 * mission_value.total_value
    + 0.30 * counterfactual.expected_ambiguity_resolution
    + 0.20 * clamp01(counterfactual.expected_health_score_delta)

clamped to [0.0, 1.0].  This is a planning-utility proxy, not a
financial model and not revenue.

Strategies
----------

- ``exhaustive`` — all subsets up to ``max_collects`` size, filtered
  by budget / required / excluded; pick the highest-utility subset.
  Deterministic tie-break: higher total expected ambiguity resolution,
  then lower total cost, then lexicographic tuple of sorted
  ``candidate_id``.
- ``greedy`` — required candidates first (in candidate_id order),
  then add by ``value / cost`` descending (cost-zero items first),
  ties broken by value descending then candidate_id ascending,
  stopping when budget or ``max_collects`` is exhausted.

The recommended plan defaults to the exhaustive plan when there are
``<= 10`` candidates and to the greedy plan otherwise.  The 10-item
threshold is intentionally conservative; even at 10 candidates with
``max_collects = 10`` the exhaustive search is 1024 subsets, well
within stdlib reach.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

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


# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlanConstraint:
    """Simple budget / max-collects / required / excluded constraint set."""
    budget: float
    max_collects: int
    required_candidate_ids: tuple[str, ...] = ()
    excluded_candidate_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlanItem:
    """One candidate in the plan with cost / utility breakdown."""
    candidate_id: str
    label: str
    cost: float
    value: float
    expected_ambiguity_resolution: float
    expected_health_score_delta: float
    reason: str


@dataclass(frozen=True)
class OptimizedPlan:
    """Result of one optimization strategy (``exhaustive`` or ``greedy``)."""
    strategy: str
    selected_items: tuple[PlanItem, ...]
    total_cost: float
    total_value: float
    total_expected_ambiguity_resolution: float
    total_expected_health_score_delta: float
    constraint: PlanConstraint
    summary: str
    caveats: tuple[str, ...]


@dataclass(frozen=True)
class PlanOptimizationReport:
    """Both baselines plus the recommended plan and a brief comparison.

    ``all_candidates`` exposes every PlanItem the optimizer saw (sorted
    by candidate_id) so the text formatter can list the full utility
    table without re-running the upstream pipeline.  Display-only.
    """
    scenario_id: str
    exhaustive_plan: OptimizedPlan
    greedy_plan: OptimizedPlan
    recommended_plan: OptimizedPlan
    comparison_summary: str
    all_candidates: tuple[PlanItem, ...] = ()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_RECOMMENDED_EXHAUSTIVE_THRESHOLD = 10

_BASE_CAVEATS: tuple[str, ...] = (
    "planning utility is a deterministic proxy, not a financial model",
    "recommendations name candidate collect types only; specific platform "
    "selection and schedules are downstream concerns",
    "outcome weights are heuristic, not calibrated probabilities",
)


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _round4(x: float) -> float:
    return round(float(x), 4)


def _round3(x: float) -> float:
    return round(float(x), 3)


def _candidate_value(
    mv: MissionValueAssessment,
    cf: CounterfactualCollectAssessment,
) -> float:
    raw = (
        0.50 * mv.total_value
        + 0.30 * cf.expected_ambiguity_resolution
        + 0.20 * _clamp01(cf.expected_health_score_delta)
    )
    return _round4(_clamp01(raw))


def _index_by_id(
    recommendation: CollectionRecommendation,
    mission_value_report: MissionValueReport,
    counterfactual_report: CounterfactualReport,
) -> tuple[
    dict[str, CollectionValue],
    dict[str, MissionValueAssessment],
    dict[str, CounterfactualCollectAssessment],
]:
    cv_by_id = {v.candidate.candidate_id: v for v in recommendation.ranked_values}
    mv_by_id = {a.candidate_id: a for a in mission_value_report.ranked_assessments}
    cf_by_id = {a.candidate_id: a for a in counterfactual_report.ranked_assessments}
    return cv_by_id, mv_by_id, cf_by_id


def _candidate_items(
    recommendation: CollectionRecommendation,
    mission_value_report: MissionValueReport,
    counterfactual_report: CounterfactualReport,
) -> tuple[tuple[PlanItem, ...], tuple[str, ...]]:
    """Build per-candidate PlanItem tuples and any data-quality caveats."""
    cv_by_id, mv_by_id, cf_by_id = _index_by_id(
        recommendation, mission_value_report, counterfactual_report,
    )
    items: list[PlanItem] = []
    caveats: list[str] = []

    for candidate_id, cv in cv_by_id.items():
        mv = mv_by_id.get(candidate_id)
        cf = cf_by_id.get(candidate_id)
        if mv is None or cf is None:
            caveats.append(
                f"candidate {candidate_id!r} missing mission-value or "
                f"counterfactual data; skipped"
            )
            continue

        cost = cv.candidate.relative_cost
        if cost is None or cost < 0.0:
            caveats.append(
                f"candidate {candidate_id!r} has invalid relative_cost; "
                f"defaulting to 0.5"
            )
            cost = 0.5

        items.append(PlanItem(
            candidate_id=candidate_id,
            label=cv.candidate.label,
            cost=_round3(cost),
            value=_candidate_value(mv, cf),
            expected_ambiguity_resolution=cf.expected_ambiguity_resolution,
            expected_health_score_delta=cf.expected_health_score_delta,
            reason=(
                f"planning utility = 0.50 x mission_value({mv.total_value:.2f}) "
                f"+ 0.30 x amb_res({cf.expected_ambiguity_resolution:.2f}) "
                f"+ 0.20 x clamp(health_delta({cf.expected_health_score_delta:+.2f}))"
            ),
        ))
    items.sort(key=lambda p: p.candidate_id)
    return tuple(items), tuple(caveats)


def _validate_required(
    items: tuple[PlanItem, ...],
    constraint: PlanConstraint,
) -> tuple[tuple[PlanItem, ...], tuple[str, ...] | None]:
    """Resolve required-candidate references; surface infeasibilities.

    Returns ``(required_items, infeasible_caveats)`` — when caveats is
    not None, the constraint is infeasible and callers should return an
    empty plan with those caveats.
    """
    by_id = {p.candidate_id: p for p in items}
    required: list[PlanItem] = []
    excluded = set(constraint.excluded_candidate_ids)
    for rid in sorted(set(constraint.required_candidate_ids)):
        if rid in excluded:
            return (), (
                f"infeasible constraint: required candidate {rid!r} is also "
                f"in excluded list",
            )
        item = by_id.get(rid)
        if item is None:
            return (), (
                f"infeasible constraint: required candidate {rid!r} is not "
                f"available in the recommendation",
            )
        required.append(item)

    if len(required) > constraint.max_collects:
        return (), (
            f"infeasible constraint: {len(required)} required candidates "
            f"exceeds max_collects={constraint.max_collects}",
        )
    required_cost = sum(p.cost for p in required)
    if required_cost > constraint.budget + 1e-9:
        return (), (
            f"infeasible constraint: required candidates total cost "
            f"{required_cost:.2f} exceeds budget {constraint.budget:.2f}",
        )
    return tuple(required), None


def _empty_plan(
    strategy: str,
    constraint: PlanConstraint,
    caveats: tuple[str, ...],
    summary: str,
) -> OptimizedPlan:
    return OptimizedPlan(
        strategy=strategy,
        selected_items=(),
        total_cost=0.0,
        total_value=0.0,
        total_expected_ambiguity_resolution=0.0,
        total_expected_health_score_delta=0.0,
        constraint=constraint,
        summary=summary,
        caveats=caveats,
    )


def _plan_from_items(
    strategy: str,
    selected: tuple[PlanItem, ...],
    constraint: PlanConstraint,
    caveats: tuple[str, ...],
) -> OptimizedPlan:
    total_cost = _round4(sum(p.cost for p in selected))
    total_value = _round4(sum(p.value for p in selected))
    total_amb = _round4(sum(p.expected_ambiguity_resolution for p in selected))
    total_delta = _round4(sum(p.expected_health_score_delta for p in selected))
    if not selected:
        summary = (
            f"{strategy} strategy returned no candidates under "
            f"budget={constraint.budget:.2f}, max_collects={constraint.max_collects}"
        )
    else:
        ids = ", ".join(p.candidate_id for p in selected)
        summary = (
            f"{strategy} strategy selected {len(selected)} candidate(s): {ids}; "
            f"total planning utility {total_value:.2f}, "
            f"expected ambiguity resolution {total_amb:.2f}, "
            f"expected health delta {total_delta:+.2f}"
        )
    return OptimizedPlan(
        strategy=strategy,
        selected_items=selected,
        total_cost=total_cost,
        total_value=total_value,
        total_expected_ambiguity_resolution=total_amb,
        total_expected_health_score_delta=total_delta,
        constraint=constraint,
        summary=summary,
        caveats=caveats,
    )


# ---------------------------------------------------------------------------
# Exhaustive search
# ---------------------------------------------------------------------------


def _exhaustive_plan(
    items: tuple[PlanItem, ...],
    constraint: PlanConstraint,
    base_caveats: tuple[str, ...],
) -> OptimizedPlan:
    required, infeasibility = _validate_required(items, constraint)
    if infeasibility is not None:
        return _empty_plan(
            "exhaustive", constraint,
            caveats=base_caveats + infeasibility,
            summary=(
                "exhaustive strategy returned no candidates: "
                + infeasibility[0]
            ),
        )

    excluded = set(constraint.excluded_candidate_ids)
    required_ids = {p.candidate_id for p in required}
    optional = tuple(
        p for p in items
        if p.candidate_id not in required_ids
        and p.candidate_id not in excluded
    )

    # We always include the required items; choose 0..(max_collects - len(required))
    # additional optional items.
    extra_capacity = constraint.max_collects - len(required)
    required_cost = sum(p.cost for p in required)
    remaining_budget = constraint.budget - required_cost

    best_subset: tuple[PlanItem, ...] = tuple(required)
    best_key = _subset_sort_key(best_subset)

    for k in range(0, extra_capacity + 1):
        for combo in combinations(optional, k):
            cost = sum(p.cost for p in combo)
            if cost > remaining_budget + 1e-9:
                continue
            subset = tuple(required) + combo
            key = _subset_sort_key(subset)
            if key < best_key:
                best_subset = subset
                best_key = key

    # Stable display order: candidate_id ascending.
    selected = tuple(sorted(best_subset, key=lambda p: p.candidate_id))
    return _plan_from_items("exhaustive", selected, constraint, base_caveats)


def _subset_sort_key(subset: tuple[PlanItem, ...]) -> tuple:
    """Return a tuple that orders feasible subsets best-first under ``min``.

    Primary: -total_value (higher value is better).
    Tie-break 1: -total expected_ambiguity_resolution.
    Tie-break 2: +total_cost (lower cost wins).
    Tie-break 3: lexicographic tuple of sorted candidate_ids.
    """
    total_value = sum(p.value for p in subset)
    total_amb = sum(p.expected_ambiguity_resolution for p in subset)
    total_cost = sum(p.cost for p in subset)
    ids = tuple(sorted(p.candidate_id for p in subset))
    return (-total_value, -total_amb, total_cost, ids)


# ---------------------------------------------------------------------------
# Greedy heuristic
# ---------------------------------------------------------------------------


_INF = float("inf")


def _value_per_cost(item: PlanItem) -> float:
    return _INF if item.cost <= 0.0 else item.value / item.cost


def _greedy_plan(
    items: tuple[PlanItem, ...],
    constraint: PlanConstraint,
    base_caveats: tuple[str, ...],
) -> OptimizedPlan:
    required, infeasibility = _validate_required(items, constraint)
    if infeasibility is not None:
        return _empty_plan(
            "greedy", constraint,
            caveats=base_caveats + infeasibility,
            summary=(
                "greedy strategy returned no candidates: "
                + infeasibility[0]
            ),
        )

    excluded = set(constraint.excluded_candidate_ids)
    required_ids = {p.candidate_id for p in required}
    candidates = [
        p for p in items
        if p.candidate_id not in required_ids
        and p.candidate_id not in excluded
    ]
    # Greedy ranking: value-per-cost desc, then value desc, then id asc.
    candidates.sort(
        key=lambda p: (-_value_per_cost(p), -p.value, p.candidate_id),
    )

    selected = list(required)
    total_cost = sum(p.cost for p in selected)
    for p in candidates:
        if len(selected) >= constraint.max_collects:
            break
        if total_cost + p.cost > constraint.budget + 1e-9:
            continue
        selected.append(p)
        total_cost += p.cost

    selected.sort(key=lambda p: p.candidate_id)
    return _plan_from_items("greedy", tuple(selected), constraint, base_caveats)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _default_constraint() -> PlanConstraint:
    return PlanConstraint(budget=1.0, max_collects=2)


def optimize_collection_plan(
    recommendation: CollectionRecommendation,
    mission_value_report: MissionValueReport,
    counterfactual_report: CounterfactualReport,
    *,
    scenario_id: str,
    constraint: PlanConstraint | None = None,
) -> PlanOptimizationReport:
    """Compose a constrained collection plan.

    Returns both the exhaustive and greedy baselines plus a recommended
    plan (exhaustive when ``<= 10`` candidates, greedy otherwise) and a
    short comparison summary.  Pure deterministic; no wall-clock.
    """
    resolved_constraint = (
        constraint if constraint is not None else _default_constraint()
    )
    items, data_caveats = _candidate_items(
        recommendation, mission_value_report, counterfactual_report,
    )
    base = _BASE_CAVEATS + data_caveats

    exhaustive = _exhaustive_plan(items, resolved_constraint, base)
    greedy = _greedy_plan(items, resolved_constraint, base)

    if len(items) <= _RECOMMENDED_EXHAUSTIVE_THRESHOLD:
        recommended = exhaustive
        rationale = (
            f"recommended plan uses the exhaustive strategy (candidate "
            f"count {len(items)} <= threshold {_RECOMMENDED_EXHAUSTIVE_THRESHOLD})"
        )
    else:
        recommended = greedy
        rationale = (
            f"recommended plan uses the greedy strategy (candidate count "
            f"{len(items)} > threshold {_RECOMMENDED_EXHAUSTIVE_THRESHOLD})"
        )

    same_ids = (
        tuple(p.candidate_id for p in exhaustive.selected_items)
        == tuple(p.candidate_id for p in greedy.selected_items)
    )
    if same_ids:
        comparison = (
            f"exhaustive and greedy converged on the same plan "
            f"({len(exhaustive.selected_items)} candidate(s)); {rationale}"
        )
    else:
        comparison = (
            f"exhaustive (utility {exhaustive.total_value:.2f}) vs greedy "
            f"(utility {greedy.total_value:.2f}); {rationale}"
        )

    return PlanOptimizationReport(
        scenario_id=scenario_id,
        exhaustive_plan=exhaustive,
        greedy_plan=greedy,
        recommended_plan=recommended,
        comparison_summary=comparison,
        all_candidates=items,
    )


# ---------------------------------------------------------------------------
# Text formatter
# ---------------------------------------------------------------------------


def _render_constraint(out: list[str], c: PlanConstraint) -> None:
    out.append("\nConstraints\n")
    out.append("-----------\n")
    out.append(f"  budget: {c.budget:.2f}\n")
    out.append(f"  max collects: {c.max_collects}\n")
    if c.required_candidate_ids:
        out.append(
            f"  required: {', '.join(c.required_candidate_ids)}\n"
        )
    if c.excluded_candidate_ids:
        out.append(
            f"  excluded: {', '.join(c.excluded_candidate_ids)}\n"
        )


def _render_plan(out: list[str], plan: OptimizedPlan, label: str) -> None:
    out.append(f"\n{label} (strategy: {plan.strategy})\n")
    out.append("-" * (len(label) + len(plan.strategy) + 12) + "\n")
    if not plan.selected_items:
        out.append("(no candidates selected)\n")
    else:
        for p in plan.selected_items:
            out.append(
                f"  {p.candidate_id:24s}  cost={p.cost:.2f}  "
                f"value={p.value:.2f}\n"
            )
    out.append("Total:\n")
    out.append(
        f"  cost: {plan.total_cost:.2f} / {plan.constraint.budget:.2f}\n"
    )
    out.append(f"  planning utility: {plan.total_value:.2f}\n")
    out.append(
        f"  expected ambiguity resolution: "
        f"{plan.total_expected_ambiguity_resolution:.2f}\n"
    )
    out.append(
        f"  expected health delta: "
        f"{plan.total_expected_health_score_delta:+.2f}\n"
    )


def format_optimizer_text(report: PlanOptimizationReport) -> str:
    """Human-readable plan-optimizer summary."""
    parts: list[str] = []
    title = f"OPTIMIZED COLLECTION PLAN - {report.scenario_id.upper()}"
    parts.append(title + "\n")
    parts.append("=" * len(title) + "\n")

    _render_constraint(parts, report.recommended_plan.constraint)

    parts.append("\nCandidate utility inputs\n")
    parts.append("------------------------\n")
    parts.append(
        f"  {'candidate_id':24s}  {'cost':>5s}  {'value':>6s}  "
        f"{'amb_res':>7s}  {'h_delta':>7s}\n"
    )
    for p in report.all_candidates:
        parts.append(
            f"  {p.candidate_id:24s}  {p.cost:5.2f}  {p.value:6.2f}  "
            f"{p.expected_ambiguity_resolution:7.2f}  "
            f"{p.expected_health_score_delta:+7.2f}\n"
        )

    _render_plan(parts, report.recommended_plan, "Recommended plan")

    parts.append("\nGreedy vs exhaustive comparison\n")
    parts.append("-------------------------------\n")
    parts.append(report.comparison_summary + "\n")

    parts.append("\nCaveats\n")
    parts.append("-------\n")
    for c in report.recommended_plan.caveats:
        parts.append(f"  - {c}\n")

    return "".join(parts)
