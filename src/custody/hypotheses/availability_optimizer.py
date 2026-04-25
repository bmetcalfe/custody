"""Availability-adjusted collection-plan optimization (ADR-0021 Slice 21).

Integrates scene-availability feasibility (Slice 20) into the
constrained collection-plan optimizer (Slice 11) so optimized plans
account for whether candidate collect TYPES are feasible in the
current metadata window.

This composes existing modules - it does not replace ``optimizer.py``
or ``scene_availability.py``.  It adjusts planning utility by
feasibility score and optionally excludes infeasible candidates.

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
from itertools import combinations

from custody.hypotheses.optimizer import (
    PlanItem,
    PlanOptimizationReport,
)
from custody.hypotheses.scene_availability import (
    AvailabilityAdjustedRecommendation,
    AvailabilityStatus,
    CollectFeasibilityAssessment,
)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AvailabilityOptimizerConstraint:
    budget: float
    max_collects: int
    min_feasibility_score: float = 0.0
    exclude_unavailable: bool = True
    required_candidate_ids: tuple[str, ...] = ()
    excluded_candidate_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class AvailabilityAdjustedPlanItem:
    candidate_id: str
    label: str
    cost: float
    base_planning_utility: float
    feasibility_score: float
    availability_status: str
    adjusted_utility: float
    reason: str
    caveats: tuple[str, ...]


@dataclass(frozen=True)
class AvailabilityOptimizedPlan:
    strategy: str
    selected_items: tuple[AvailabilityAdjustedPlanItem, ...]
    total_cost: float
    total_base_planning_utility: float
    total_adjusted_utility: float
    constraint: AvailabilityOptimizerConstraint
    summary: str
    caveats: tuple[str, ...]


@dataclass(frozen=True)
class AvailabilityOptimizationReport:
    scenario_id: str
    base_plan_summary: str
    availability_summary: str
    adjusted_candidate_pool: tuple[AvailabilityAdjustedPlanItem, ...]
    exhaustive_plan: AvailabilityOptimizedPlan
    greedy_plan: AvailabilityOptimizedPlan
    recommended_plan: AvailabilityOptimizedPlan
    comparison_summary: str


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


_RECOMMENDED_EXHAUSTIVE_THRESHOLD = 10

_BASE_CAVEATS: tuple[str, ...] = (
    "availability-adjusted optimization is decision support only",
    "adjusted utility combines planning utility with metadata-derived "
    "feasibility; no executable plan is implied",
    "recommendations name candidate collect types only; platform "
    "selection and scheduling are downstream concerns",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _round4(x: float) -> float:
    return round(float(x), 4)


def _default_constraint() -> AvailabilityOptimizerConstraint:
    return AvailabilityOptimizerConstraint(budget=1.0, max_collects=2)


# ---------------------------------------------------------------------------
# Candidate construction
# ---------------------------------------------------------------------------


def build_availability_adjusted_candidates(
    optimization_report: PlanOptimizationReport,
    availability_report: AvailabilityAdjustedRecommendation,
) -> tuple[AvailabilityAdjustedPlanItem, ...]:
    """Join optimizer PlanItems with availability feasibility assessments.

    Each output item carries the optimizer's planning utility plus the
    feasibility multiplier derived from availability metadata.

    Candidates not present in the availability report fall back to
    ``feasibility_score=0.5`` and ``availability_status="unknown"``.
    """
    by_id: dict[str, CollectFeasibilityAssessment] = {
        a.candidate_id: a for a in availability_report.assessments
    }
    out: list[AvailabilityAdjustedPlanItem] = []
    for item in optimization_report.all_candidates:
        a = by_id.get(item.candidate_id)
        if a is None:
            feasibility = 0.5
            status = AvailabilityStatus.UNKNOWN.value
            extra = (
                "no availability assessment available; assumed neutral",
            )
        else:
            feasibility = a.feasibility_score
            status = a.status.value
            extra = ()
        adjusted = _round4(_clamp01(item.value * feasibility))
        out.append(AvailabilityAdjustedPlanItem(
            candidate_id=item.candidate_id,
            label=item.label,
            cost=item.cost,
            base_planning_utility=item.value,
            feasibility_score=feasibility,
            availability_status=status,
            adjusted_utility=adjusted,
            reason=item.reason,
            caveats=extra,
        ))
    out.sort(key=lambda c: c.candidate_id)
    return tuple(out)


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------


def _eligible_candidates(
    candidates: tuple[AvailabilityAdjustedPlanItem, ...],
    constraint: AvailabilityOptimizerConstraint,
) -> tuple[tuple[AvailabilityAdjustedPlanItem, ...], tuple[str, ...]]:
    """Filter candidates by constraint, returning (eligible, extra_caveats)."""
    excluded_ids = set(constraint.excluded_candidate_ids)
    extra: list[str] = []
    eligible: list[AvailabilityAdjustedPlanItem] = []
    for c in candidates:
        if c.candidate_id in excluded_ids:
            extra.append(
                f"candidate {c.candidate_id!r} excluded by constraint"
            )
            continue
        if (
            constraint.exclude_unavailable
            and c.availability_status == AvailabilityStatus.UNAVAILABLE.value
        ):
            extra.append(
                f"candidate {c.candidate_id!r} excluded as unavailable in "
                f"availability metadata"
            )
            continue
        if c.feasibility_score < constraint.min_feasibility_score:
            extra.append(
                f"candidate {c.candidate_id!r} excluded: feasibility "
                f"{c.feasibility_score:.2f} < min {constraint.min_feasibility_score:.2f}"
            )
            continue
        eligible.append(c)
    return tuple(eligible), tuple(extra)


# ---------------------------------------------------------------------------
# Plan helpers
# ---------------------------------------------------------------------------


def _empty_plan(
    strategy: str,
    constraint: AvailabilityOptimizerConstraint,
    caveats: tuple[str, ...],
    summary: str,
) -> AvailabilityOptimizedPlan:
    return AvailabilityOptimizedPlan(
        strategy=strategy,
        selected_items=(),
        total_cost=0.0,
        total_base_planning_utility=0.0,
        total_adjusted_utility=0.0,
        constraint=constraint,
        summary=summary,
        caveats=caveats,
    )


def _plan_from_items(
    strategy: str,
    selected: tuple[AvailabilityAdjustedPlanItem, ...],
    constraint: AvailabilityOptimizerConstraint,
    caveats: tuple[str, ...],
) -> AvailabilityOptimizedPlan:
    total_cost = _round4(sum(p.cost for p in selected))
    total_base = _round4(sum(p.base_planning_utility for p in selected))
    total_adj = _round4(sum(p.adjusted_utility for p in selected))
    if not selected:
        summary = (
            f"{strategy} strategy returned no candidates under "
            f"budget={constraint.budget:.2f}, "
            f"max_collects={constraint.max_collects}"
        )
    else:
        ids = ", ".join(p.candidate_id for p in selected)
        summary = (
            f"{strategy} strategy selected {len(selected)} candidate(s): "
            f"{ids}; total adjusted utility {total_adj:.2f} "
            f"(base {total_base:.2f})"
        )
    return AvailabilityOptimizedPlan(
        strategy=strategy,
        selected_items=selected,
        total_cost=total_cost,
        total_base_planning_utility=total_base,
        total_adjusted_utility=total_adj,
        constraint=constraint,
        summary=summary,
        caveats=caveats,
    )


def _resolve_required(
    eligible: tuple[AvailabilityAdjustedPlanItem, ...],
    constraint: AvailabilityOptimizerConstraint,
) -> tuple[
    tuple[AvailabilityAdjustedPlanItem, ...],
    tuple[AvailabilityAdjustedPlanItem, ...],
    tuple[str, ...],
]:
    """Return (required_items, free_pool, missing_required_caveats)."""
    by_id = {c.candidate_id: c for c in eligible}
    required: list[AvailabilityAdjustedPlanItem] = []
    missing: list[str] = []
    for cid in constraint.required_candidate_ids:
        if cid in by_id:
            required.append(by_id[cid])
        else:
            missing.append(
                f"required candidate {cid!r} is not eligible "
                f"(excluded, infeasible, or absent)"
            )
    free = tuple(c for c in eligible if c.candidate_id not in {
        r.candidate_id for r in required
    })
    return tuple(sorted(required, key=lambda c: c.candidate_id)), free, tuple(missing)


# ---------------------------------------------------------------------------
# Exhaustive search
# ---------------------------------------------------------------------------


def _subset_sort_key(
    subset: tuple[AvailabilityAdjustedPlanItem, ...],
) -> tuple:
    total_adj = sum(p.adjusted_utility for p in subset)
    total_base = sum(p.base_planning_utility for p in subset)
    total_cost = sum(p.cost for p in subset)
    ids = tuple(sorted(p.candidate_id for p in subset))
    return (-total_adj, -total_base, total_cost, ids)


def _exhaustive_plan(
    candidates: tuple[AvailabilityAdjustedPlanItem, ...],
    constraint: AvailabilityOptimizerConstraint,
    base_caveats: tuple[str, ...],
) -> AvailabilityOptimizedPlan:
    required, free, missing = _resolve_required(candidates, constraint)
    caveats = base_caveats + missing

    if missing:
        return _empty_plan(
            "exhaustive", constraint, caveats,
            "exhaustive strategy could not satisfy required candidates",
        )

    required_cost = sum(r.cost for r in required)
    required_count = len(required)
    if (
        required_count > constraint.max_collects
        or required_cost > constraint.budget + 1e-9
    ):
        return _empty_plan(
            "exhaustive", constraint, caveats,
            "exhaustive strategy: required candidates exceed budget or "
            "max_collects",
        )

    remaining_budget = constraint.budget - required_cost
    remaining_slots = constraint.max_collects - required_count

    best: tuple[AvailabilityAdjustedPlanItem, ...] = tuple(required)
    best_key = _subset_sort_key(best)

    max_addl = min(remaining_slots, len(free))
    for size in range(0, max_addl + 1):
        for combo in combinations(free, size):
            extra_cost = sum(p.cost for p in combo)
            if extra_cost > remaining_budget + 1e-9:
                continue
            subset = tuple(required) + combo
            key = _subset_sort_key(subset)
            if key < best_key:
                best_key = key
                best = subset

    selected = tuple(sorted(best, key=lambda c: c.candidate_id))
    return _plan_from_items("exhaustive", selected, constraint, caveats)


# ---------------------------------------------------------------------------
# Greedy search
# ---------------------------------------------------------------------------


def _greedy_score_key(c: AvailabilityAdjustedPlanItem) -> tuple:
    cost = c.cost if c.cost > 0 else 1e-9
    ratio = c.adjusted_utility / cost
    return (-ratio, -c.adjusted_utility, c.candidate_id)


def _greedy_plan(
    candidates: tuple[AvailabilityAdjustedPlanItem, ...],
    constraint: AvailabilityOptimizerConstraint,
    base_caveats: tuple[str, ...],
) -> AvailabilityOptimizedPlan:
    required, free, missing = _resolve_required(candidates, constraint)
    caveats = base_caveats + missing

    if missing:
        return _empty_plan(
            "greedy", constraint, caveats,
            "greedy strategy could not satisfy required candidates",
        )

    required_cost = sum(r.cost for r in required)
    if (
        len(required) > constraint.max_collects
        or required_cost > constraint.budget + 1e-9
    ):
        return _empty_plan(
            "greedy", constraint, caveats,
            "greedy strategy: required candidates exceed budget or "
            "max_collects",
        )

    selected = list(required)
    spent = required_cost
    slots = constraint.max_collects - len(required)

    ranked = sorted(free, key=_greedy_score_key)
    for c in ranked:
        if slots <= 0:
            break
        if spent + c.cost > constraint.budget + 1e-9:
            continue
        selected.append(c)
        spent += c.cost
        slots -= 1

    selected_sorted = tuple(sorted(selected, key=lambda c: c.candidate_id))
    return _plan_from_items("greedy", selected_sorted, constraint, caveats)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def optimize_availability_adjusted_plan(
    optimization_report: PlanOptimizationReport,
    availability_report: AvailabilityAdjustedRecommendation,
    *,
    scenario_id: str,
    constraint: AvailabilityOptimizerConstraint | None = None,
) -> AvailabilityOptimizationReport:
    """Compose an availability-adjusted optimized collection plan."""
    resolved = constraint if constraint is not None else _default_constraint()

    candidates = build_availability_adjusted_candidates(
        optimization_report, availability_report,
    )
    eligible, extra_caveats = _eligible_candidates(candidates, resolved)
    base = _BASE_CAVEATS + extra_caveats

    exhaustive = _exhaustive_plan(eligible, resolved, base)
    greedy = _greedy_plan(eligible, resolved, base)

    if len(eligible) <= _RECOMMENDED_EXHAUSTIVE_THRESHOLD:
        recommended = exhaustive
        rationale = (
            f"recommended plan uses the exhaustive strategy (candidate "
            f"count {len(eligible)} <= threshold "
            f"{_RECOMMENDED_EXHAUSTIVE_THRESHOLD})"
        )
    else:
        recommended = greedy
        rationale = (
            f"recommended plan uses the greedy strategy (candidate count "
            f"{len(eligible)} > threshold "
            f"{_RECOMMENDED_EXHAUSTIVE_THRESHOLD})"
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
            f"exhaustive (adjusted utility "
            f"{exhaustive.total_adjusted_utility:.2f}) vs greedy "
            f"(adjusted utility {greedy.total_adjusted_utility:.2f}); "
            f"{rationale}"
        )

    return AvailabilityOptimizationReport(
        scenario_id=scenario_id,
        base_plan_summary=optimization_report.comparison_summary,
        availability_summary=availability_report.summary,
        adjusted_candidate_pool=candidates,
        exhaustive_plan=exhaustive,
        greedy_plan=greedy,
        recommended_plan=recommended,
        comparison_summary=comparison,
    )


# ---------------------------------------------------------------------------
# Serializers
# ---------------------------------------------------------------------------


def _item_to_dict(item: AvailabilityAdjustedPlanItem) -> dict:
    return {
        "candidate_id": item.candidate_id,
        "label": item.label,
        "cost": item.cost,
        "base_planning_utility": item.base_planning_utility,
        "feasibility_score": item.feasibility_score,
        "availability_status": item.availability_status,
        "adjusted_utility": item.adjusted_utility,
        "reason": item.reason,
        "caveats": list(item.caveats),
    }


def _constraint_to_dict(c: AvailabilityOptimizerConstraint) -> dict:
    return {
        "budget": c.budget,
        "max_collects": c.max_collects,
        "min_feasibility_score": c.min_feasibility_score,
        "exclude_unavailable": c.exclude_unavailable,
        "required_candidate_ids": list(c.required_candidate_ids),
        "excluded_candidate_ids": list(c.excluded_candidate_ids),
    }


def _plan_to_dict(plan: AvailabilityOptimizedPlan) -> dict:
    return {
        "strategy": plan.strategy,
        "selected_items": [_item_to_dict(i) for i in plan.selected_items],
        "total_cost": plan.total_cost,
        "total_base_planning_utility": plan.total_base_planning_utility,
        "total_adjusted_utility": plan.total_adjusted_utility,
        "constraint": _constraint_to_dict(plan.constraint),
        "summary": plan.summary,
        "caveats": list(plan.caveats),
    }


def report_to_json_object(report: AvailabilityOptimizationReport) -> dict:
    return {
        "scenario_id": report.scenario_id,
        "base_plan_summary": report.base_plan_summary,
        "availability_summary": report.availability_summary,
        "adjusted_candidate_pool": [
            _item_to_dict(c) for c in report.adjusted_candidate_pool
        ],
        "recommended_plan": _plan_to_dict(report.recommended_plan),
        "exhaustive_plan": _plan_to_dict(report.exhaustive_plan),
        "greedy_plan": _plan_to_dict(report.greedy_plan),
        "comparison_summary": report.comparison_summary,
    }


def report_to_json(report: AvailabilityOptimizationReport) -> str:
    return _json.dumps(report_to_json_object(report), indent=2)


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def format_availability_optimizer_text(
    report: AvailabilityOptimizationReport,
) -> str:
    parts: list[str] = []
    title = (
        f"AVAILABILITY-ADJUSTED OPTIMIZED PLAN - {report.scenario_id.upper()}"
    )
    parts.append(title + "\n")
    parts.append("=" * len(title) + "\n")

    parts.append("\nBase optimized plan\n")
    parts.append("-------------------\n")
    parts.append(f"  {report.base_plan_summary}\n")

    parts.append("\nAvailability feasibility\n")
    parts.append("------------------------\n")
    parts.append(f"  {report.availability_summary}\n")

    parts.append("\nAdjusted candidate pool\n")
    parts.append("-----------------------\n")
    parts.append(
        f"  {'candidate_id':24s}  {'cost':>5s}  {'base':>6s}  "
        f"{'feas':>5s}  {'adj':>6s}  {'status':>12s}\n"
    )
    for c in report.adjusted_candidate_pool:
        parts.append(
            f"  {c.candidate_id:24s}  {c.cost:5.2f}  "
            f"{c.base_planning_utility:6.2f}  "
            f"{c.feasibility_score:5.2f}  "
            f"{c.adjusted_utility:6.2f}  "
            f"{c.availability_status:>12s}\n"
        )

    plan = report.recommended_plan
    header = (
        f"Recommended availability-adjusted plan ({plan.strategy})"
    )
    parts.append("\n" + header + "\n")
    parts.append("-" * len(header) + "\n")
    if not plan.selected_items:
        parts.append("  (no candidates selected)\n")
    else:
        for p in plan.selected_items:
            parts.append(
                f"  {p.candidate_id:24s}  cost={p.cost:.2f}  "
                f"adj_utility={p.adjusted_utility:.2f}  "
                f"({p.availability_status})\n"
            )
    parts.append("Total:\n")
    parts.append(
        f"  cost: {plan.total_cost:.2f} / {plan.constraint.budget:.2f}\n"
    )
    parts.append(
        f"  adjusted utility: {plan.total_adjusted_utility:.2f}\n"
    )
    parts.append(
        f"  base planning utility: {plan.total_base_planning_utility:.2f}\n"
    )

    parts.append("\nGreedy vs exhaustive comparison\n")
    parts.append("-------------------------------\n")
    parts.append(f"  {report.comparison_summary}\n")

    parts.append("\nCaveats\n")
    parts.append("-------\n")
    for c in plan.caveats:
        parts.append(f"  - {c}\n")

    return "".join(parts)


def format_availability_optimizer_markdown(
    report: AvailabilityOptimizationReport,
) -> str:
    lines: list[str] = []
    name = report.scenario_id.capitalize()
    lines.append(f"# Availability-Adjusted Optimized Plan - {name}")
    lines.append("")
    lines.append("## Base optimized plan")
    lines.append("")
    lines.append(report.base_plan_summary)
    lines.append("")
    lines.append("## Availability feasibility")
    lines.append("")
    lines.append(report.availability_summary)
    lines.append("")
    lines.append("## Adjusted candidate pool")
    lines.append("")
    lines.append(
        "| candidate | cost | base utility | feasibility | "
        "adjusted utility | status |"
    )
    lines.append("|---|---|---|---|---|---|")
    for c in report.adjusted_candidate_pool:
        lines.append(
            f"| {c.candidate_id} | {c.cost:.2f} | "
            f"{c.base_planning_utility:.2f} | {c.feasibility_score:.2f} | "
            f"{c.adjusted_utility:.2f} | {c.availability_status} |"
        )
    lines.append("")

    plan = report.recommended_plan
    lines.append(f"## Recommended plan ({plan.strategy})")
    lines.append("")
    if not plan.selected_items:
        lines.append("(no candidates selected)")
    else:
        lines.append("| # | candidate | cost | adjusted utility | status |")
        lines.append("|---|---|---|---|---|")
        for i, p in enumerate(plan.selected_items, start=1):
            lines.append(
                f"| {i} | {p.candidate_id} | {p.cost:.2f} | "
                f"{p.adjusted_utility:.2f} | {p.availability_status} |"
            )
    lines.append("")
    lines.append("## Totals")
    lines.append("")
    lines.append(
        f"- cost: {plan.total_cost:.2f} / {plan.constraint.budget:.2f}"
    )
    lines.append(
        f"- adjusted utility: {plan.total_adjusted_utility:.2f}"
    )
    lines.append(
        f"- base planning utility: {plan.total_base_planning_utility:.2f}"
    )
    lines.append("")
    lines.append("## Comparison")
    lines.append("")
    lines.append(report.comparison_summary)
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    for c in plan.caveats:
        lines.append(f"- {c}")
    lines.append("")
    return "\n".join(lines)
