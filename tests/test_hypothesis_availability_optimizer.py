"""Tests for :mod:`custody.hypotheses.availability_optimizer` (ADR-0021 Slice 21)."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from custody.hypotheses.availability_optimizer import (
    AvailabilityAdjustedPlanItem,
    AvailabilityOptimizationReport,
    AvailabilityOptimizedPlan,
    AvailabilityOptimizerConstraint,
    build_availability_adjusted_candidates,
    format_availability_optimizer_markdown,
    format_availability_optimizer_text,
    optimize_availability_adjusted_plan,
    report_to_json,
    report_to_json_object,
)
from custody.hypotheses.optimizer import (
    OptimizedPlan,
    PlanConstraint,
    PlanItem,
    PlanOptimizationReport,
)
from custody.hypotheses.scene_availability import (
    AvailabilityAdjustedRecommendation,
    AvailabilityAdjustedValue,
    AvailabilityStatus,
    CollectFeasibilityAssessment,
)


# ---------------------------------------------------------------------------
# Synthetic upstream reports
# ---------------------------------------------------------------------------


def _make_plan_item(cid: str, value: float, cost: float = 0.5) -> PlanItem:
    return PlanItem(
        candidate_id=cid,
        label=cid.replace("_", " "),
        cost=cost,
        value=value,
        expected_ambiguity_resolution=0.5,
        expected_health_score_delta=0.1,
        reason=f"reason for {cid}",
    )


def _empty_optimized_plan(strategy: str) -> OptimizedPlan:
    return OptimizedPlan(
        strategy=strategy,
        selected_items=(),
        total_cost=0.0,
        total_value=0.0,
        total_expected_ambiguity_resolution=0.0,
        total_expected_health_score_delta=0.0,
        constraint=PlanConstraint(budget=1.0, max_collects=2),
        summary="x",
        caveats=(),
    )


def _make_optimization_report(
    items: tuple[PlanItem, ...],
    *,
    scenario_id: str = "tennent",
) -> PlanOptimizationReport:
    return PlanOptimizationReport(
        scenario_id=scenario_id,
        exhaustive_plan=_empty_optimized_plan("exhaustive"),
        greedy_plan=_empty_optimized_plan("greedy"),
        recommended_plan=_empty_optimized_plan("recommended"),
        comparison_summary=f"base optimized plan summary for {scenario_id}",
        all_candidates=items,
    )


def _make_assessment(
    cid: str, status: AvailabilityStatus, score: float,
) -> CollectFeasibilityAssessment:
    return CollectFeasibilityAssessment(
        candidate_id=cid,
        status=status,
        feasibility_score=score,
        supporting_record_ids=(),
        reason=f"feasibility for {cid}",
        caveats=(),
    )


def _make_availability_report(
    assessments: tuple[CollectFeasibilityAssessment, ...],
    *,
    scenario_id: str = "tennent",
) -> AvailabilityAdjustedRecommendation:
    return AvailabilityAdjustedRecommendation(
        scenario_id=scenario_id,
        assessments=assessments,
        adjusted_values=(),
        summary=f"availability summary for {scenario_id}",
        caveats=("metadata-only bridge; no imagery was downloaded",),
    )


def _standard_pair() -> tuple[
    PlanOptimizationReport, AvailabilityAdjustedRecommendation,
]:
    items = (
        _make_plan_item("optical_context", 0.85, cost=0.5),
        _make_plan_item("repeat_sar", 0.60, cost=0.4),
        _make_plan_item("cross_geometry_sar", 0.70, cost=0.6),
        _make_plan_item("higher_resolution_sar", 0.50, cost=0.7),
        _make_plan_item("ais_coverage_query", 0.30, cost=0.2),
        _make_plan_item("wait_or_monitor", 0.15, cost=0.0),
    )
    assessments = (
        _make_assessment("optical_context", AvailabilityStatus.FEASIBLE, 0.85),
        _make_assessment("repeat_sar", AvailabilityStatus.FEASIBLE, 0.70),
        _make_assessment("cross_geometry_sar", AvailabilityStatus.PARTIAL, 0.35),
        _make_assessment("higher_resolution_sar", AvailabilityStatus.UNAVAILABLE, 0.0),
        _make_assessment("ais_coverage_query", AvailabilityStatus.FEASIBLE, 0.70),
        _make_assessment("wait_or_monitor", AvailabilityStatus.FEASIBLE, 0.25),
    )
    return _make_optimization_report(items), _make_availability_report(assessments)


# ---------------------------------------------------------------------------
# Candidate construction
# ---------------------------------------------------------------------------


def test_build_availability_adjusted_candidates_joins_correctly() -> None:
    opt, avail = _standard_pair()
    candidates = build_availability_adjusted_candidates(opt, avail)
    by_id = {c.candidate_id: c for c in candidates}
    assert by_id["optical_context"].base_planning_utility == 0.85
    assert by_id["optical_context"].feasibility_score == 0.85
    assert by_id["repeat_sar"].feasibility_score == 0.70


def test_adjusted_utility_equals_base_times_feasibility() -> None:
    opt, avail = _standard_pair()
    candidates = build_availability_adjusted_candidates(opt, avail)
    for c in candidates:
        assert abs(
            c.adjusted_utility - c.base_planning_utility * c.feasibility_score
        ) < 1e-4


def test_unknown_candidate_gets_default_feasibility() -> None:
    items = (_make_plan_item("mystery_collect", 0.5),)
    opt = _make_optimization_report(items)
    avail = _make_availability_report(())
    candidates = build_availability_adjusted_candidates(opt, avail)
    assert len(candidates) == 1
    assert candidates[0].feasibility_score == 0.5
    assert candidates[0].availability_status == "unknown"


def test_candidates_sorted_by_candidate_id() -> None:
    opt, avail = _standard_pair()
    candidates = build_availability_adjusted_candidates(opt, avail)
    ids = [c.candidate_id for c in candidates]
    assert ids == sorted(ids)


# ---------------------------------------------------------------------------
# Eligibility filtering
# ---------------------------------------------------------------------------


def test_unavailable_excluded_by_default() -> None:
    opt, avail = _standard_pair()
    constraint = AvailabilityOptimizerConstraint(budget=2.0, max_collects=4)
    report = optimize_availability_adjusted_plan(
        opt, avail, scenario_id="tennent", constraint=constraint,
    )
    pool_ids = {c.candidate_id for c in report.adjusted_candidate_pool}
    selected_ids = {p.candidate_id for p in report.recommended_plan.selected_items}
    assert "higher_resolution_sar" in pool_ids
    assert "higher_resolution_sar" not in selected_ids


def test_include_unavailable_keeps_all() -> None:
    opt, avail = _standard_pair()
    constraint = AvailabilityOptimizerConstraint(
        budget=10.0, max_collects=10, exclude_unavailable=False,
    )
    report = optimize_availability_adjusted_plan(
        opt, avail, scenario_id="tennent", constraint=constraint,
    )
    pool_ids = [c.candidate_id for c in report.adjusted_candidate_pool]
    assert "higher_resolution_sar" in pool_ids
    selected_ids = {p.candidate_id for p in report.recommended_plan.selected_items}
    # When unavailable, adjusted_utility = 0 so it shouldn't be picked anyway,
    # but it should be eligible (no exclusion caveat for it).
    caveats = " ".join(report.recommended_plan.caveats).lower()
    assert "higher_resolution_sar" not in caveats


def test_min_feasibility_score_filters() -> None:
    opt, avail = _standard_pair()
    constraint = AvailabilityOptimizerConstraint(
        budget=10.0, max_collects=10, min_feasibility_score=0.5,
    )
    report = optimize_availability_adjusted_plan(
        opt, avail, scenario_id="tennent", constraint=constraint,
    )
    selected_ids = {p.candidate_id for p in report.recommended_plan.selected_items}
    # cross_geometry_sar (0.35), wait_or_monitor (0.25), unavailable should
    # be filtered out
    assert "cross_geometry_sar" not in selected_ids
    assert "wait_or_monitor" not in selected_ids


def test_excluded_candidate_ids_respected() -> None:
    opt, avail = _standard_pair()
    constraint = AvailabilityOptimizerConstraint(
        budget=10.0, max_collects=10,
        excluded_candidate_ids=("optical_context",),
    )
    report = optimize_availability_adjusted_plan(
        opt, avail, scenario_id="tennent", constraint=constraint,
    )
    selected_ids = {p.candidate_id for p in report.recommended_plan.selected_items}
    assert "optical_context" not in selected_ids


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------


def test_exhaustive_selects_highest_adjusted_utility() -> None:
    opt, avail = _standard_pair()
    constraint = AvailabilityOptimizerConstraint(budget=1.0, max_collects=2)
    report = optimize_availability_adjusted_plan(
        opt, avail, scenario_id="tennent", constraint=constraint,
    )
    plan = report.exhaustive_plan
    # Top adjusted utility candidates are optical_context (0.85*0.85=0.7225)
    # and repeat_sar (0.60*0.70=0.42).  Total cost = 0.5+0.4 = 0.9 within
    # budget 1.0.
    selected_ids = {p.candidate_id for p in plan.selected_items}
    assert "optical_context" in selected_ids
    assert "repeat_sar" in selected_ids


def test_greedy_deterministic() -> None:
    opt, avail = _standard_pair()
    constraint = AvailabilityOptimizerConstraint(budget=1.0, max_collects=2)
    a = optimize_availability_adjusted_plan(
        opt, avail, scenario_id="tennent", constraint=constraint,
    )
    b = optimize_availability_adjusted_plan(
        opt, avail, scenario_id="tennent", constraint=constraint,
    )
    assert a.greedy_plan.selected_items == b.greedy_plan.selected_items


def test_budget_constraint_respected() -> None:
    opt, avail = _standard_pair()
    constraint = AvailabilityOptimizerConstraint(budget=0.5, max_collects=4)
    report = optimize_availability_adjusted_plan(
        opt, avail, scenario_id="tennent", constraint=constraint,
    )
    assert report.recommended_plan.total_cost <= constraint.budget + 1e-6


def test_max_collects_constraint_respected() -> None:
    opt, avail = _standard_pair()
    constraint = AvailabilityOptimizerConstraint(budget=10.0, max_collects=1)
    report = optimize_availability_adjusted_plan(
        opt, avail, scenario_id="tennent", constraint=constraint,
    )
    assert len(report.recommended_plan.selected_items) <= 1


def test_required_candidate_ids_included() -> None:
    opt, avail = _standard_pair()
    constraint = AvailabilityOptimizerConstraint(
        budget=10.0, max_collects=4,
        required_candidate_ids=("ais_coverage_query",),
    )
    report = optimize_availability_adjusted_plan(
        opt, avail, scenario_id="tennent", constraint=constraint,
    )
    selected_ids = {p.candidate_id for p in report.recommended_plan.selected_items}
    assert "ais_coverage_query" in selected_ids


def test_exhaustive_and_greedy_converge_small_pool() -> None:
    items = (
        _make_plan_item("a", 0.8, cost=0.5),
        _make_plan_item("b", 0.6, cost=0.4),
    )
    assessments = (
        _make_assessment("a", AvailabilityStatus.FEASIBLE, 0.9),
        _make_assessment("b", AvailabilityStatus.FEASIBLE, 0.7),
    )
    opt = _make_optimization_report(items)
    avail = _make_availability_report(assessments)
    constraint = AvailabilityOptimizerConstraint(budget=1.0, max_collects=2)
    report = optimize_availability_adjusted_plan(
        opt, avail, scenario_id="tennent", constraint=constraint,
    )
    ex_ids = tuple(p.candidate_id for p in report.exhaustive_plan.selected_items)
    gr_ids = tuple(p.candidate_id for p in report.greedy_plan.selected_items)
    assert ex_ids == gr_ids


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def test_recommended_plan_is_exhaustive_for_small_pool() -> None:
    opt, avail = _standard_pair()
    constraint = AvailabilityOptimizerConstraint(budget=1.0, max_collects=2)
    report = optimize_availability_adjusted_plan(
        opt, avail, scenario_id="tennent", constraint=constraint,
    )
    assert report.recommended_plan.strategy == "exhaustive"


def test_report_summaries_non_empty() -> None:
    opt, avail = _standard_pair()
    report = optimize_availability_adjusted_plan(
        opt, avail, scenario_id="tennent",
    )
    assert report.base_plan_summary
    assert report.availability_summary
    assert report.comparison_summary


def test_report_adjusted_candidate_pool_complete() -> None:
    opt, avail = _standard_pair()
    report = optimize_availability_adjusted_plan(
        opt, avail, scenario_id="tennent",
    )
    pool_ids = {c.candidate_id for c in report.adjusted_candidate_pool}
    src_ids = {c.candidate_id for c in opt.all_candidates}
    assert pool_ids == src_ids


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_report_to_json_object_serializable() -> None:
    opt, avail = _standard_pair()
    report = optimize_availability_adjusted_plan(
        opt, avail, scenario_id="tennent",
    )
    obj = report_to_json_object(report)
    encoded = json.dumps(obj)
    assert json.loads(encoded) == obj


def test_report_to_json_parseable() -> None:
    opt, avail = _standard_pair()
    report = optimize_availability_adjusted_plan(
        opt, avail, scenario_id="tennent",
    )
    parsed = json.loads(report_to_json(report))
    assert parsed["scenario_id"] == "tennent"
    assert "recommended_plan" in parsed


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def test_text_format_contains_required_sections() -> None:
    opt, avail = _standard_pair()
    report = optimize_availability_adjusted_plan(
        opt, avail, scenario_id="tennent",
    )
    text = format_availability_optimizer_text(report)
    assert "AVAILABILITY-ADJUSTED OPTIMIZED PLAN" in text
    assert "Base optimized plan" in text
    assert "Availability feasibility" in text
    assert "Adjusted candidate pool" in text
    assert "Recommended availability-adjusted plan" in text
    assert "Caveats" in text


def test_markdown_format_contains_heading() -> None:
    opt, avail = _standard_pair()
    report = optimize_availability_adjusted_plan(
        opt, avail, scenario_id="whitsun",
    )
    md = format_availability_optimizer_markdown(report)
    assert md.startswith("# Availability-Adjusted Optimized Plan")


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------


def test_no_forbidden_imports() -> None:
    import custody.hypotheses.availability_optimizer as mod
    tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
    forbidden = (
        "custody.detection",
        "custody.ingest.gfw_presence",
        "sentinelhub",
        "custody.fusion.tracker",
        "custody.taskrecommendation",
    )
    offending: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if any(alias.name.startswith(p) for p in forbidden):
                    offending.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod_name = node.module or ""
            if any(mod_name.startswith(p) for p in forbidden):
                offending.append(mod_name)
    assert not offending, f"forbidden imports: {offending}"


def test_source_file_no_forbidden_language() -> None:
    import custody.hypotheses.availability_optimizer as mod
    src = Path(mod.__file__).read_text(encoding="utf-8").lower()
    forbidden = (
        "live tasking",
        "tasking order",
        "sensor command",
        "sentinel integration",
        "platform access",
        "production scheduler",
        "autonomous execution",
        "collection order",
        "revenue dollars",
    )
    for needle in forbidden:
        assert needle not in src, f"forbidden token {needle!r} in source"
