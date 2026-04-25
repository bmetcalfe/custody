"""Tests for :mod:`custody.hypotheses.portfolio` (ADR-0021 Slice 16).

Pin candidate construction (per-scenario join), eligibility filtering
(closed-status defaults + scenario / candidate excludes), exhaustive +
greedy optimizers under shared constraints, deterministic tie-breakers,
formatters, and language / import scope guardrails.
"""
from __future__ import annotations

import json
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
    PlanConstraint,
    PlanOptimizationReport,
    optimize_collection_plan,
)
from custody.hypotheses.planner_queue import (
    PlannerQueueItem,
    PlannerQueueStatus,
    create_queue_item,
)
from custody.hypotheses.planner_review import (
    ReviewAction,
    create_planner_review_record,
    create_review_subject_from_optimized_plan,
)
from custody.hypotheses.policy_eval import evaluate_collection_policies
from custody.hypotheses.portfolio import (
    PortfolioAllocationPlan,
    PortfolioAllocationReport,
    PortfolioCandidate,
    PortfolioConstraint,
    PortfolioPlanItem,
    allocate_portfolio,
    build_portfolio_candidates,
    format_portfolio_markdown,
    format_portfolio_text,
    portfolio_report_to_json_object,
)
from custody.hypotheses.registry import (
    SCENARIO_TENNENT,
    SCENARIO_WHITSUN,
    TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
    TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
    WHITSUN_AIS_DARK_OR_POORLY_OBSERVED_VESSELS,
    WHITSUN_VESSEL_CLUSTER_ACTIVITY,
)
from custody.hypotheses.types import HypothesisEvidence, HypothesisState
from custody.hypotheses.update import update_state


T_RECENT = datetime(2023, 8, 13, 12, 0, 0, tzinfo=timezone.utc)
T_REVIEW = datetime(2026, 4, 24, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Pipeline fixtures
# ---------------------------------------------------------------------------


def _ev(
    evidence_id: str, scenario_id: str,
    supports: tuple[str, ...] = (),
    contradicts: tuple[str, ...] = (),
    *, confidence: float = 0.6,
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


def _ambiguous_pipeline(scenario_id: str, hid_a: str, hid_b: str):
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
    opt = optimize_collection_plan(rec, mv, cf, scenario_id=scenario_id)
    policy = evaluate_collection_policies(
        rec, mv, cf, opt, scenario_id=scenario_id,
    )
    qi = create_queue_item(
        scenario_id=scenario_id, health=health, recommendation=rec,
        mission_value_report=mv, counterfactual_report=cf,
        optimization_report=opt, policy_report=policy,
    )
    return qi, opt, mv, cf


@pytest.fixture(scope="module")
def tennent_bundle():
    return _ambiguous_pipeline(
        SCENARIO_TENNENT,
        TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
        TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
    )


@pytest.fixture(scope="module")
def whitsun_bundle():
    return _ambiguous_pipeline(
        SCENARIO_WHITSUN,
        WHITSUN_AIS_DARK_OR_POORLY_OBSERVED_VESSELS,
        WHITSUN_VESSEL_CLUSTER_ACTIVITY,
    )


def _approve_review(opt):
    subject = create_review_subject_from_optimized_plan(opt)
    return create_planner_review_record(
        subject, action=ReviewAction.APPROVE, operator_reason="ok",
        reviewed_at=T_REVIEW,
    )


def _candidates_for(tennent_bundle, whitsun_bundle):
    t_qi, t_opt, t_mv, t_cf = tennent_bundle
    w_qi, w_opt, w_mv, w_cf = whitsun_bundle
    return build_portfolio_candidates(
        queue_items=(t_qi, w_qi),
        optimization_reports=(t_opt, w_opt),
        mission_value_reports=(t_mv, w_mv),
        counterfactual_reports=(t_cf, w_cf),
    )


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


def test_portfolio_constraint_is_frozen() -> None:
    c = PortfolioConstraint(budget=1.5, max_collects=3)
    with pytest.raises(FrozenInstanceError):
        c.budget = 0.5  # type: ignore[misc]


def test_portfolio_candidate_is_frozen() -> None:
    cand = PortfolioCandidate(
        scenario_id="x", candidate_id="y", label="z",
        cost=0.0, planning_utility=0.0, mission_value_proxy=0.0,
        expected_ambiguity_resolution=0.0, expected_health_score_delta=0.0,
        queue_priority_score=0.0, queue_status="pending_review",
        reason="r", caveats=(),
    )
    with pytest.raises(FrozenInstanceError):
        cand.cost = 1.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Candidate construction
# ---------------------------------------------------------------------------


def test_build_portfolio_candidates_joins_by_scenario(
    tennent_bundle, whitsun_bundle,
) -> None:
    pool = _candidates_for(tennent_bundle, whitsun_bundle)
    scenarios = {c.scenario_id for c in pool}
    assert scenarios == {SCENARIO_TENNENT, SCENARIO_WHITSUN}
    # Each candidate's scenario_id matches its origin queue item.
    for c in pool:
        assert c.scenario_id in scenarios


def test_build_portfolio_candidates_sorted_deterministically(
    tennent_bundle, whitsun_bundle,
) -> None:
    pool = _candidates_for(tennent_bundle, whitsun_bundle)
    sorted_pool = sorted(
        pool, key=lambda c: (c.scenario_id, c.candidate_id),
    )
    assert list(pool) == sorted_pool


# ---------------------------------------------------------------------------
# Eligibility (closed-status defaults)
# ---------------------------------------------------------------------------


def test_closed_statuses_excluded_by_default(
    tennent_bundle, whitsun_bundle,
) -> None:
    """If we re-create the Tennent queue item with an APPROVE review, the
    candidate pool should still include both scenarios but allocate_portfolio
    should drop the closed (APPROVED) items by default."""
    t_qi, t_opt, t_mv, t_cf = tennent_bundle
    w_qi, w_opt, w_mv, w_cf = whitsun_bundle

    # Recreate Tennent queue item with an APPROVE review (status=APPROVED).
    state = update_state(
        SCENARIO_TENNENT,
        evidence=(
            _ev("ev-a", SCENARIO_TENNENT,
                supports=(TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,),
                confidence=1.0),
            _ev("ev-b", SCENARIO_TENNENT,
                supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,),
                confidence=1.0),
        ),
    )
    health = assess_custody_health(state, as_of=T_RECENT)
    rec = rank_collection_candidates(state, health)
    review = _approve_review(t_opt)
    t_qi_approved = create_queue_item(
        scenario_id=SCENARIO_TENNENT, health=health, recommendation=rec,
        mission_value_report=t_mv, counterfactual_report=t_cf,
        optimization_report=t_opt, review_record=review,
    )
    assert t_qi_approved.status is PlannerQueueStatus.APPROVED

    candidates = build_portfolio_candidates(
        queue_items=(t_qi_approved, w_qi),
        optimization_reports=(t_opt, w_opt),
        mission_value_reports=(t_mv, w_mv),
        counterfactual_reports=(t_cf, w_cf),
    )
    report = allocate_portfolio(candidates)
    selected_scenarios = {
        i.scenario_id for i in report.recommended_plan.selected_items
    }
    assert SCENARIO_TENNENT not in selected_scenarios


def test_include_closed_items_includes_all_statuses_with_caveat(
    tennent_bundle, whitsun_bundle,
) -> None:
    t_qi, t_opt, t_mv, t_cf = tennent_bundle
    w_qi, w_opt, w_mv, w_cf = whitsun_bundle

    state = update_state(
        SCENARIO_TENNENT,
        evidence=(
            _ev("ev-a", SCENARIO_TENNENT,
                supports=(TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,),
                confidence=1.0),
            _ev("ev-b", SCENARIO_TENNENT,
                supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,),
                confidence=1.0),
        ),
    )
    health = assess_custody_health(state, as_of=T_RECENT)
    rec = rank_collection_candidates(state, health)
    review = _approve_review(t_opt)
    t_qi_approved = create_queue_item(
        scenario_id=SCENARIO_TENNENT, health=health, recommendation=rec,
        mission_value_report=t_mv, counterfactual_report=t_cf,
        optimization_report=t_opt, review_record=review,
    )

    candidates = build_portfolio_candidates(
        queue_items=(t_qi_approved, w_qi),
        optimization_reports=(t_opt, w_opt),
        mission_value_reports=(t_mv, w_mv),
        counterfactual_reports=(t_cf, w_cf),
    )
    constraint = PortfolioConstraint(
        budget=1.5, max_collects=3, include_closed_items=True,
    )
    report = allocate_portfolio(candidates, constraint=constraint)
    joined = " ".join(report.recommended_plan.caveats).lower()
    assert "closed queue items included" in joined


# ---------------------------------------------------------------------------
# Portfolio score clamping
# ---------------------------------------------------------------------------


def test_portfolio_score_in_unit_interval(tennent_bundle, whitsun_bundle) -> None:
    pool = _candidates_for(tennent_bundle, whitsun_bundle)
    report = allocate_portfolio(pool)
    for item in report.recommended_plan.selected_items:
        assert 0.0 <= item.portfolio_score <= 1.0


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------


def test_exhaustive_selects_highest_scoring_subset(
    tennent_bundle, whitsun_bundle,
) -> None:
    pool = _candidates_for(tennent_bundle, whitsun_bundle)
    report = allocate_portfolio(pool)
    plan = report.recommended_plan
    # The recommended plan must be at least as good as any single-item plan.
    one_constraint = PortfolioConstraint(
        budget=1.5, max_collects=1, max_collects_per_scenario=1,
    )
    one_report = allocate_portfolio(pool, constraint=one_constraint)
    assert (
        plan.total_portfolio_score
        >= one_report.recommended_plan.total_portfolio_score
    )


def test_budget_constraint_excludes_over_budget(tennent_bundle, whitsun_bundle) -> None:
    pool = _candidates_for(tennent_bundle, whitsun_bundle)
    constraint = PortfolioConstraint(
        budget=0.30, max_collects=3, max_collects_per_scenario=2,
    )
    report = allocate_portfolio(pool, constraint=constraint)
    assert report.recommended_plan.total_cost <= constraint.budget + 1e-9


def test_max_collects_constraint_caps_selection(
    tennent_bundle, whitsun_bundle,
) -> None:
    pool = _candidates_for(tennent_bundle, whitsun_bundle)
    constraint = PortfolioConstraint(
        budget=10.0, max_collects=2, max_collects_per_scenario=2,
    )
    report = allocate_portfolio(pool, constraint=constraint)
    assert len(report.recommended_plan.selected_items) <= 2


def test_max_collects_per_scenario_constraint(tennent_bundle, whitsun_bundle) -> None:
    pool = _candidates_for(tennent_bundle, whitsun_bundle)
    constraint = PortfolioConstraint(
        budget=10.0, max_collects=6, max_collects_per_scenario=1,
    )
    report = allocate_portfolio(pool, constraint=constraint)
    counts: dict[str, int] = {}
    for item in report.recommended_plan.selected_items:
        counts[item.scenario_id] = counts.get(item.scenario_id, 0) + 1
    for n in counts.values():
        assert n <= 1


def test_required_scenario_ids_honored(tennent_bundle, whitsun_bundle) -> None:
    pool = _candidates_for(tennent_bundle, whitsun_bundle)
    constraint = PortfolioConstraint(
        budget=1.5, max_collects=3, max_collects_per_scenario=2,
        required_scenario_ids=(SCENARIO_WHITSUN,),
    )
    report = allocate_portfolio(pool, constraint=constraint)
    selected = {
        i.scenario_id for i in report.recommended_plan.selected_items
    }
    assert SCENARIO_WHITSUN in selected


def test_excluded_scenario_ids_honored(tennent_bundle, whitsun_bundle) -> None:
    pool = _candidates_for(tennent_bundle, whitsun_bundle)
    constraint = PortfolioConstraint(
        budget=1.5, max_collects=3, max_collects_per_scenario=2,
        excluded_scenario_ids=(SCENARIO_TENNENT,),
    )
    report = allocate_portfolio(pool, constraint=constraint)
    selected = {
        i.scenario_id for i in report.recommended_plan.selected_items
    }
    assert SCENARIO_TENNENT not in selected


def test_required_candidate_ids_honored(tennent_bundle, whitsun_bundle) -> None:
    pool = _candidates_for(tennent_bundle, whitsun_bundle)
    constraint = PortfolioConstraint(
        budget=1.5, max_collects=3, max_collects_per_scenario=2,
        required_candidate_ids=("ais_coverage_query",),
    )
    report = allocate_portfolio(pool, constraint=constraint)
    selected = {
        i.candidate_id for i in report.recommended_plan.selected_items
    }
    assert "ais_coverage_query" in selected


def test_excluded_candidate_ids_honored(tennent_bundle, whitsun_bundle) -> None:
    pool = _candidates_for(tennent_bundle, whitsun_bundle)
    constraint = PortfolioConstraint(
        budget=1.5, max_collects=3, max_collects_per_scenario=2,
        excluded_candidate_ids=("optical_context",),
    )
    report = allocate_portfolio(pool, constraint=constraint)
    selected = {
        i.candidate_id for i in report.recommended_plan.selected_items
    }
    assert "optical_context" not in selected


def test_infeasible_required_produces_empty_plan_with_caveat(
    tennent_bundle, whitsun_bundle,
) -> None:
    pool = _candidates_for(tennent_bundle, whitsun_bundle)
    impossible = PortfolioConstraint(
        budget=1.5, max_collects=3, max_collects_per_scenario=2,
        required_candidate_ids=("definitely_not_a_candidate",),
    )
    report = allocate_portfolio(pool, constraint=impossible)
    assert report.recommended_plan.selected_items == ()
    joined = " ".join(report.recommended_plan.caveats).lower()
    assert "infeasible" in joined or "no feasible" in joined


# ---------------------------------------------------------------------------
# Determinism + tie-breakers
# ---------------------------------------------------------------------------


def test_greedy_is_deterministic(tennent_bundle, whitsun_bundle) -> None:
    pool = _candidates_for(tennent_bundle, whitsun_bundle)
    a = allocate_portfolio(pool).greedy_plan
    b = allocate_portfolio(pool).greedy_plan
    assert (
        tuple((i.scenario_id, i.candidate_id) for i in a.selected_items)
        == tuple((i.scenario_id, i.candidate_id) for i in b.selected_items)
    )


def test_exhaustive_is_deterministic(tennent_bundle, whitsun_bundle) -> None:
    pool = _candidates_for(tennent_bundle, whitsun_bundle)
    a = allocate_portfolio(pool).exhaustive_plan
    b = allocate_portfolio(pool).exhaustive_plan
    assert (
        tuple((i.scenario_id, i.candidate_id) for i in a.selected_items)
        == tuple((i.scenario_id, i.candidate_id) for i in b.selected_items)
    )


def test_recommended_uses_exhaustive_for_small_pool(
    tennent_bundle, whitsun_bundle,
) -> None:
    pool = _candidates_for(tennent_bundle, whitsun_bundle)
    # Two scenarios * 6 candidates = 12 eligible (under threshold 14).
    report = allocate_portfolio(pool)
    assert report.recommended_plan.strategy == "exhaustive"


def test_scenario_coverage_count_matches_distinct_scenarios(
    tennent_bundle, whitsun_bundle,
) -> None:
    pool = _candidates_for(tennent_bundle, whitsun_bundle)
    plan = allocate_portfolio(pool).recommended_plan
    expected = len({i.scenario_id for i in plan.selected_items})
    assert plan.scenario_coverage_count == expected


# ---------------------------------------------------------------------------
# Language guardrails
# ---------------------------------------------------------------------------


_FORBIDDEN = (
    "revenue dollars",
    "profit",
    "real revenue",
    "tasking order",
    "live tasking",
    "sensor command",
    "production scheduler",
    "autonomous constellation management",
    "real mps control",
    "collection order",
    "sentinel integration",
)


def test_no_forbidden_language_in_summaries_caveats(
    tennent_bundle, whitsun_bundle,
) -> None:
    pool = _candidates_for(tennent_bundle, whitsun_bundle)
    report = allocate_portfolio(pool)
    blobs: list[str] = [report.comparison_summary]
    for plan in (
        report.recommended_plan,
        report.exhaustive_plan,
        report.greedy_plan,
    ):
        blobs.append(plan.summary)
        blobs.extend(plan.caveats)
        for item in plan.selected_items:
            blobs.extend([item.label, item.reason])
    for cand in report.candidate_pool:
        blobs.extend([cand.label, cand.reason])
        blobs.extend(cand.caveats)
    text = " ".join(blobs).lower()
    for needle in _FORBIDDEN:
        assert needle not in text, (
            f"forbidden token {needle!r} appears in portfolio output"
        )


def test_source_file_no_forbidden_language() -> None:
    import custody.hypotheses.portfolio as mod
    src = Path(mod.__file__).read_text(encoding="utf-8").lower()
    for needle in _FORBIDDEN:
        assert needle not in src, (
            f"forbidden token {needle!r} appears in portfolio.py source"
        )


# ---------------------------------------------------------------------------
# Import-boundary scan
# ---------------------------------------------------------------------------


def test_no_forbidden_imports() -> None:
    import ast
    import custody.hypotheses.portfolio as mod
    tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
    forbidden_prefixes = (
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
                if any(alias.name.startswith(p) for p in forbidden_prefixes):
                    offending.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod_name = node.module or ""
            if any(mod_name.startswith(p) for p in forbidden_prefixes):
                offending.append(mod_name)
    assert not offending, (
        f"portfolio.py imports forbidden modules: {offending}"
    )


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def test_text_format_contains_required_sections(
    tennent_bundle, whitsun_bundle,
) -> None:
    pool = _candidates_for(tennent_bundle, whitsun_bundle)
    report = allocate_portfolio(pool)
    text = format_portfolio_text(report)
    assert "PORTFOLIO ALLOCATION PLAN" in text
    assert "Global constraints" in text
    assert "Recommended portfolio" in text
    assert "Caveats" in text


def test_json_round_trips_deterministically(
    tennent_bundle, whitsun_bundle,
) -> None:
    pool = _candidates_for(tennent_bundle, whitsun_bundle)
    report = allocate_portfolio(pool)
    a = json.dumps(portfolio_report_to_json_object(report), indent=2)
    b = json.dumps(portfolio_report_to_json_object(report), indent=2)
    assert a == b
    parsed = json.loads(a)
    assert "scenario_ids" in parsed
    assert "constraint" in parsed
    assert "candidate_pool" in parsed
    assert "recommended_plan" in parsed


def test_markdown_starts_with_h1(tennent_bundle, whitsun_bundle) -> None:
    pool = _candidates_for(tennent_bundle, whitsun_bundle)
    md = format_portfolio_markdown(allocate_portfolio(pool))
    assert md.startswith("# Portfolio Allocation Plan\n")
