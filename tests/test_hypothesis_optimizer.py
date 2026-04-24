"""Tests for :mod:`custody.hypotheses.optimizer` (ADR-0021 Slice 11).

Pin the constrained collection-plan optimizer: planning-utility formula,
exhaustive optimum + tie-break, greedy ordering with required/excluded
honored, infeasibility caveats, recommended-plan threshold, scope
guardrails (no forbidden imports, no wall-clock, no revenue / tasking
language), and end-to-end ranking against the Tennent and Whitsun
synthetic ambiguous states.
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from pathlib import Path

import pytest

from custody.hypotheses.collection_value import (
    CollectionRecommendation,
    rank_collection_candidates,
)
from custody.hypotheses.counterfactual import (
    CounterfactualReport,
    simulate_counterfactual_collects,
)
from custody.hypotheses.custody_health import (
    CustodyHealthStatus,
    HypothesisCustodyHealth,
    assess_custody_health,
)
from custody.hypotheses.mission_value import (
    MissionValueReport,
    attribute_mission_value,
)
from custody.hypotheses.optimizer import (
    OptimizedPlan,
    PlanConstraint,
    PlanItem,
    PlanOptimizationReport,
    format_optimizer_text,
    optimize_collection_plan,
)
from custody.hypotheses.registry import (
    SCENARIO_TENNENT,
    SCENARIO_WHITSUN,
    TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
    TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
    WHITSUN_AIS_DARK_OR_POORLY_OBSERVED_VESSELS,
    WHITSUN_VESSEL_CLUSTER_ACTIVITY,
)
from custody.hypotheses.types import HypothesisEvidence
from custody.hypotheses.update import update_state


T_RECENT = datetime(2023, 8, 13, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Helpers — build the upstream pipeline once
# ---------------------------------------------------------------------------


def _ev(
    evidence_id: str,
    scenario_id: str,
    supports: tuple[str, ...] = (),
    contradicts: tuple[str, ...] = (),
    *,
    confidence: float = 0.6,
) -> HypothesisEvidence:
    return HypothesisEvidence(
        evidence_id=evidence_id,
        source_ref=f"observation:{evidence_id}",
        source_kind="observation",
        scenario_id=scenario_id,
        timestamp=T_RECENT,
        supports=supports,
        contradicts=contradicts,
        confidence=confidence,
        weight=1.0,
        reason="synthetic",
    )


def _ambiguous_pipeline(
    scenario_id: str,
    hid_a: str,
    hid_b: str,
) -> tuple[
    HypothesisCustodyHealth,
    CollectionRecommendation,
    MissionValueReport,
    CounterfactualReport,
]:
    state = update_state(
        scenario_id,
        evidence=(
            _ev("ev-a", scenario_id, supports=(hid_a,), confidence=1.0),
            _ev("ev-b", scenario_id, supports=(hid_b,), confidence=1.0),
        ),
    )
    health = assess_custody_health(state, as_of=T_RECENT)
    assert health.status is CustodyHealthStatus.AMBIGUOUS
    rec = rank_collection_candidates(state, health)
    mv = attribute_mission_value(rec, health)
    cf = simulate_counterfactual_collects(state, health, rec)
    return health, rec, mv, cf


@pytest.fixture(scope="module")
def tennent_pipeline():
    return _ambiguous_pipeline(
        SCENARIO_TENNENT,
        TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
        TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
    )


@pytest.fixture(scope="module")
def whitsun_pipeline():
    return _ambiguous_pipeline(
        SCENARIO_WHITSUN,
        WHITSUN_AIS_DARK_OR_POORLY_OBSERVED_VESSELS,
        WHITSUN_VESSEL_CLUSTER_ACTIVITY,
    )


def _ids(plan: OptimizedPlan) -> tuple[str, ...]:
    return tuple(p.candidate_id for p in plan.selected_items)


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


def test_plan_constraint_is_frozen() -> None:
    c = PlanConstraint(budget=1.0, max_collects=2)
    with pytest.raises(FrozenInstanceError):
        c.budget = 0.0  # type: ignore[misc]


def test_plan_item_is_frozen() -> None:
    p = PlanItem(
        candidate_id="x", label="x", cost=0.1, value=0.5,
        expected_ambiguity_resolution=0.0, expected_health_score_delta=0.0,
        reason="x",
    )
    with pytest.raises(FrozenInstanceError):
        p.cost = 0.0  # type: ignore[misc]


def test_optimized_plan_is_frozen() -> None:
    plan = OptimizedPlan(
        strategy="greedy", selected_items=(),
        total_cost=0.0, total_value=0.0,
        total_expected_ambiguity_resolution=0.0,
        total_expected_health_score_delta=0.0,
        constraint=PlanConstraint(budget=1.0, max_collects=1),
        summary="x", caveats=(),
    )
    with pytest.raises(FrozenInstanceError):
        plan.summary = "y"  # type: ignore[misc]


def test_plan_optimization_report_is_frozen() -> None:
    plan = OptimizedPlan(
        strategy="greedy", selected_items=(),
        total_cost=0.0, total_value=0.0,
        total_expected_ambiguity_resolution=0.0,
        total_expected_health_score_delta=0.0,
        constraint=PlanConstraint(budget=1.0, max_collects=1),
        summary="x", caveats=(),
    )
    rep = PlanOptimizationReport(
        scenario_id="x", exhaustive_plan=plan, greedy_plan=plan,
        recommended_plan=plan, comparison_summary="x",
    )
    with pytest.raises(FrozenInstanceError):
        rep.comparison_summary = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Planning-utility formula
# ---------------------------------------------------------------------------


def test_candidate_value_formula_combines_three_components(tennent_pipeline) -> None:
    health, rec, mv, cf = tennent_pipeline
    report = optimize_collection_plan(
        rec, mv, cf, scenario_id=SCENARIO_TENNENT,
    )
    by_id = {p.candidate_id: p for p in report.all_candidates}
    mv_by_id = {a.candidate_id: a for a in mv.ranked_assessments}
    cf_by_id = {a.candidate_id: a for a in cf.ranked_assessments}

    for cid, item in by_id.items():
        mv_a = mv_by_id[cid]
        cf_a = cf_by_id[cid]
        clamped_delta = max(
            0.0, min(1.0, cf_a.expected_health_score_delta),
        )
        expected_raw = (
            0.50 * mv_a.total_value
            + 0.30 * cf_a.expected_ambiguity_resolution
            + 0.20 * clamped_delta
        )
        expected_clamped = max(0.0, min(1.0, expected_raw))
        assert item.value == pytest.approx(round(expected_clamped, 4))


def test_candidate_values_clamped_zero_to_one(tennent_pipeline) -> None:
    health, rec, mv, cf = tennent_pipeline
    report = optimize_collection_plan(
        rec, mv, cf, scenario_id=SCENARIO_TENNENT,
    )
    for p in report.all_candidates:
        assert 0.0 <= p.value <= 1.0


# ---------------------------------------------------------------------------
# Default optimization on Tennent + Whitsun
# ---------------------------------------------------------------------------


def test_tennent_recommended_plan_includes_optical_context(tennent_pipeline) -> None:
    health, rec, mv, cf = tennent_pipeline
    report = optimize_collection_plan(
        rec, mv, cf, scenario_id=SCENARIO_TENNENT,
    )
    assert "optical_context" in _ids(report.recommended_plan)


def test_whitsun_recommended_plan_includes_ais_coverage_query(whitsun_pipeline) -> None:
    health, rec, mv, cf = whitsun_pipeline
    report = optimize_collection_plan(
        rec, mv, cf, scenario_id=SCENARIO_WHITSUN,
    )
    assert "ais_coverage_query" in _ids(report.recommended_plan)


def test_recommended_plan_uses_exhaustive_for_six_candidates(tennent_pipeline) -> None:
    health, rec, mv, cf = tennent_pipeline
    report = optimize_collection_plan(
        rec, mv, cf, scenario_id=SCENARIO_TENNENT,
    )
    assert report.recommended_plan.strategy == "exhaustive"


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------


def test_budget_constraint_excludes_over_budget_subsets(tennent_pipeline) -> None:
    health, rec, mv, cf = tennent_pipeline
    tight = PlanConstraint(budget=0.3, max_collects=2)
    report = optimize_collection_plan(
        rec, mv, cf, scenario_id=SCENARIO_TENNENT, constraint=tight,
    )
    assert report.recommended_plan.total_cost <= tight.budget + 1e-9


def test_max_collects_one_returns_at_most_one_item(tennent_pipeline) -> None:
    health, rec, mv, cf = tennent_pipeline
    one = PlanConstraint(budget=1.0, max_collects=1)
    report = optimize_collection_plan(
        rec, mv, cf, scenario_id=SCENARIO_TENNENT, constraint=one,
    )
    assert len(report.recommended_plan.selected_items) <= 1


def test_max_collects_zero_returns_empty_plan(tennent_pipeline) -> None:
    health, rec, mv, cf = tennent_pipeline
    none = PlanConstraint(budget=1.0, max_collects=0)
    report = optimize_collection_plan(
        rec, mv, cf, scenario_id=SCENARIO_TENNENT, constraint=none,
    )
    assert report.recommended_plan.selected_items == ()


def test_required_candidate_present_in_plan(tennent_pipeline) -> None:
    health, rec, mv, cf = tennent_pipeline
    require_repeat = PlanConstraint(
        budget=1.0, max_collects=2,
        required_candidate_ids=("repeat_sar",),
    )
    report = optimize_collection_plan(
        rec, mv, cf, scenario_id=SCENARIO_TENNENT,
        constraint=require_repeat,
    )
    assert "repeat_sar" in _ids(report.recommended_plan)


def test_excluded_candidate_absent_from_plan(tennent_pipeline) -> None:
    health, rec, mv, cf = tennent_pipeline
    exclude_optical = PlanConstraint(
        budget=1.0, max_collects=2,
        excluded_candidate_ids=("optical_context",),
    )
    report = optimize_collection_plan(
        rec, mv, cf, scenario_id=SCENARIO_TENNENT,
        constraint=exclude_optical,
    )
    assert "optical_context" not in _ids(report.recommended_plan)


def test_required_exceeding_budget_returns_empty_plan(tennent_pipeline) -> None:
    """Required candidates whose total cost exceeds the budget -> infeasible."""
    health, rec, mv, cf = tennent_pipeline
    impossible = PlanConstraint(
        budget=0.05, max_collects=2,
        required_candidate_ids=("higher_resolution_sar",),  # cost ~0.80
    )
    report = optimize_collection_plan(
        rec, mv, cf, scenario_id=SCENARIO_TENNENT, constraint=impossible,
    )
    assert report.recommended_plan.selected_items == ()
    joined = " ".join(report.recommended_plan.caveats).lower()
    assert "infeasible" in joined


def test_required_exceeding_max_collects_returns_empty_plan(tennent_pipeline) -> None:
    health, rec, mv, cf = tennent_pipeline
    impossible = PlanConstraint(
        budget=1.0, max_collects=1,
        required_candidate_ids=("optical_context", "repeat_sar"),
    )
    report = optimize_collection_plan(
        rec, mv, cf, scenario_id=SCENARIO_TENNENT, constraint=impossible,
    )
    assert report.recommended_plan.selected_items == ()
    joined = " ".join(report.recommended_plan.caveats).lower()
    assert "infeasible" in joined


def test_required_not_in_recommendation_returns_empty_plan(tennent_pipeline) -> None:
    health, rec, mv, cf = tennent_pipeline
    bogus = PlanConstraint(
        budget=1.0, max_collects=2,
        required_candidate_ids=("nonexistent_collect",),
    )
    report = optimize_collection_plan(
        rec, mv, cf, scenario_id=SCENARIO_TENNENT, constraint=bogus,
    )
    assert report.recommended_plan.selected_items == ()


def test_required_in_excluded_returns_empty_plan(tennent_pipeline) -> None:
    health, rec, mv, cf = tennent_pipeline
    contradictory = PlanConstraint(
        budget=1.0, max_collects=2,
        required_candidate_ids=("repeat_sar",),
        excluded_candidate_ids=("repeat_sar",),
    )
    report = optimize_collection_plan(
        rec, mv, cf, scenario_id=SCENARIO_TENNENT, constraint=contradictory,
    )
    assert report.recommended_plan.selected_items == ()


# ---------------------------------------------------------------------------
# Greedy + zero-cost handling
# ---------------------------------------------------------------------------


def test_greedy_plan_does_not_divide_by_zero_for_wait_or_monitor(tennent_pipeline) -> None:
    """wait_or_monitor has cost 0.0; greedy must rank it without ZeroDivisionError."""
    health, rec, mv, cf = tennent_pipeline
    report = optimize_collection_plan(
        rec, mv, cf, scenario_id=SCENARIO_TENNENT,
    )
    # If the greedy_plan exists at all, no exception was raised during build.
    assert isinstance(report.greedy_plan, OptimizedPlan)


def test_greedy_plan_is_deterministic(tennent_pipeline) -> None:
    health, rec, mv, cf = tennent_pipeline
    r1 = optimize_collection_plan(rec, mv, cf, scenario_id=SCENARIO_TENNENT)
    r2 = optimize_collection_plan(rec, mv, cf, scenario_id=SCENARIO_TENNENT)
    assert _ids(r1.greedy_plan) == _ids(r2.greedy_plan)
    assert r1.greedy_plan.total_value == r2.greedy_plan.total_value


# ---------------------------------------------------------------------------
# Exhaustive determinism + tie-break
# ---------------------------------------------------------------------------


def test_exhaustive_plan_is_deterministic(tennent_pipeline) -> None:
    health, rec, mv, cf = tennent_pipeline
    r1 = optimize_collection_plan(rec, mv, cf, scenario_id=SCENARIO_TENNENT)
    r2 = optimize_collection_plan(rec, mv, cf, scenario_id=SCENARIO_TENNENT)
    assert _ids(r1.exhaustive_plan) == _ids(r2.exhaustive_plan)


def test_exhaustive_picks_highest_value_subset(tennent_pipeline) -> None:
    """Sanity: the recommended plan's total_value should be no worse than
    any single-item plan of equal max-collects."""
    health, rec, mv, cf = tennent_pipeline
    one = PlanConstraint(budget=1.0, max_collects=1)
    report = optimize_collection_plan(
        rec, mv, cf, scenario_id=SCENARIO_TENNENT, constraint=one,
    )
    # When max_collects=1, the recommended plan should pick the single
    # highest-value affordable candidate.
    items = sorted(report.all_candidates, key=lambda p: -p.value)
    affordable = [p for p in items if p.cost <= 1.0]
    assert affordable
    assert _ids(report.recommended_plan) == (affordable[0].candidate_id,)


# ---------------------------------------------------------------------------
# Scope guardrails — language
# ---------------------------------------------------------------------------


_FORBIDDEN_PHRASES = (
    "revenue dollars",
    "actual revenue",
    "profit",
    "production scheduler",
    "tasking order",
    "live tasking",
)


def test_no_forbidden_language_in_summaries_or_caveats(tennent_pipeline) -> None:
    health, rec, mv, cf = tennent_pipeline
    report = optimize_collection_plan(
        rec, mv, cf, scenario_id=SCENARIO_TENNENT,
    )
    blobs: list[str] = [
        report.comparison_summary,
        report.recommended_plan.summary,
        report.exhaustive_plan.summary,
        report.greedy_plan.summary,
    ]
    for plan in (
        report.recommended_plan, report.exhaustive_plan, report.greedy_plan,
    ):
        blobs.extend(plan.caveats)
        for item in plan.selected_items:
            blobs.extend([item.reason, item.label])

    text = " ".join(b.lower() for b in blobs)
    for needle in _FORBIDDEN_PHRASES:
        assert needle not in text, (
            f"forbidden phrase {needle!r} appears in optimizer output"
        )


def test_no_forbidden_language_in_text_formatter(tennent_pipeline) -> None:
    health, rec, mv, cf = tennent_pipeline
    report = optimize_collection_plan(
        rec, mv, cf, scenario_id=SCENARIO_TENNENT,
    )
    text = format_optimizer_text(report).lower()
    for needle in _FORBIDDEN_PHRASES:
        assert needle not in text


# ---------------------------------------------------------------------------
# Scope guardrails — imports + wall-clock
# ---------------------------------------------------------------------------


def test_optimizer_module_has_no_forbidden_imports() -> None:
    import ast
    import custody.hypotheses.optimizer as mod
    tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
    forbidden_prefixes = (
        "custody.detection",
        "custody.ingest.gfw_presence",
        "sentinelhub",
        "custody.fusion.tracker",
        "custody.taskrecommendation",
        # External optimization deps are also banned per dispatch.
        "scipy",
        "sklearn",
        "ortools",
        "pulp",
        "networkx",
    )
    offending: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if any(alias.name.startswith(p) for p in forbidden_prefixes):
                    offending.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod_name = node.module or ""
            if any(mod_name.startswith(p) for p in forbidden_prefixes):
                offending.append(mod_name)
    assert not offending, (
        f"optimizer.py imports forbidden modules: {offending}"
    )


def test_optimizer_source_has_no_wall_clock() -> None:
    import custody.hypotheses.optimizer as mod
    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert "datetime.now" not in src
    assert "datetime.utcnow" not in src


# ---------------------------------------------------------------------------
# Caveats include the proxy disclaimer
# ---------------------------------------------------------------------------


def test_caveats_include_planning_utility_proxy_disclaimer(tennent_pipeline) -> None:
    health, rec, mv, cf = tennent_pipeline
    report = optimize_collection_plan(
        rec, mv, cf, scenario_id=SCENARIO_TENNENT,
    )
    blob = " ".join(report.recommended_plan.caveats).lower()
    assert "planning utility" in blob
    assert "proxy" in blob


def test_text_format_renders_constraints_block(tennent_pipeline) -> None:
    health, rec, mv, cf = tennent_pipeline
    report = optimize_collection_plan(
        rec, mv, cf, scenario_id=SCENARIO_TENNENT,
    )
    text = format_optimizer_text(report)
    assert "OPTIMIZED COLLECTION PLAN" in text
    assert "Constraints" in text
    assert "Recommended plan" in text
    assert "Greedy vs exhaustive comparison" in text


# ---------------------------------------------------------------------------
# all_candidates surface
# ---------------------------------------------------------------------------


def test_all_candidates_includes_every_candidate(tennent_pipeline) -> None:
    health, rec, mv, cf = tennent_pipeline
    report = optimize_collection_plan(
        rec, mv, cf, scenario_id=SCENARIO_TENNENT,
    )
    rec_ids = {v.candidate.candidate_id for v in rec.ranked_values}
    all_ids = {p.candidate_id for p in report.all_candidates}
    assert all_ids == rec_ids
