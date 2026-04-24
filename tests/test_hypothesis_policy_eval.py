"""Tests for :mod:`custody.hypotheses.policy_eval` (ADR-0021 Slice 12).

Pin the six named heuristic policies, constraint enforcement,
deterministic ranking, frozen dataclasses, scope guardrails (no
forbidden imports, no wall-clock, no trained-RL / live-tasking
language), and end-to-end behaviour against the Tennent / Whitsun
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
    PlanConstraint,
    PlanOptimizationReport,
    optimize_collection_plan,
)
from custody.hypotheses.policy_eval import (
    PolicyDefinition,
    PolicyEvaluation,
    PolicyEvaluationReport,
    evaluate_collection_policies,
    format_policy_eval_text,
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
# Pipeline fixtures
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
    *,
    constraint: PlanConstraint | None = None,
) -> tuple[
    HypothesisCustodyHealth,
    CollectionRecommendation,
    MissionValueReport,
    CounterfactualReport,
    PlanOptimizationReport,
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
    opt = optimize_collection_plan(
        rec, mv, cf, scenario_id=scenario_id, constraint=constraint,
    )
    return health, rec, mv, cf, opt


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


def _eval_by_id(report: PolicyEvaluationReport) -> dict[str, PolicyEvaluation]:
    return {e.policy_id: e for e in report.evaluations}


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


def test_policy_definition_is_frozen() -> None:
    d = PolicyDefinition(policy_id="x", label="x", description="x")
    with pytest.raises(FrozenInstanceError):
        d.label = "y"  # type: ignore[misc]


def test_policy_evaluation_is_frozen() -> None:
    e = PolicyEvaluation(
        policy_id="x", label="x", selected_candidate_ids=(),
        total_cost=0.0, planning_utility=0.0, mission_value_total=0.0,
        expected_ambiguity_resolution=0.0, expected_health_score_delta=0.0,
        rank=1, reason="x", caveats=(),
    )
    with pytest.raises(FrozenInstanceError):
        e.rank = 2  # type: ignore[misc]


def test_policy_evaluation_report_is_frozen() -> None:
    e = PolicyEvaluation(
        policy_id="x", label="x", selected_candidate_ids=(),
        total_cost=0.0, planning_utility=0.0, mission_value_total=0.0,
        expected_ambiguity_resolution=0.0, expected_health_score_delta=0.0,
        rank=1, reason="x", caveats=(),
    )
    rep = PolicyEvaluationReport(
        scenario_id="x",
        constraint=PlanConstraint(budget=1.0, max_collects=1),
        evaluations=(e,), winning_policy_id="x", summary="x",
    )
    with pytest.raises(FrozenInstanceError):
        rep.summary = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Default policies
# ---------------------------------------------------------------------------


_DEFAULT_POLICY_IDS = (
    "value_optimized",
    "ambiguity_first",
    "low_cost_first",
    "sar_repeat_first",
    "optical_disambiguation_first",
    "ais_context_first",
)


def test_default_evaluates_all_six_policies(tennent_pipeline) -> None:
    health, rec, mv, cf, opt = tennent_pipeline
    report = evaluate_collection_policies(
        rec, mv, cf, opt, scenario_id=SCENARIO_TENNENT,
    )
    assert {e.policy_id for e in report.evaluations} == set(_DEFAULT_POLICY_IDS)


def test_winning_policy_id_matches_rank_one(tennent_pipeline) -> None:
    health, rec, mv, cf, opt = tennent_pipeline
    report = evaluate_collection_policies(
        rec, mv, cf, opt, scenario_id=SCENARIO_TENNENT,
    )
    rank_one = next(e for e in report.evaluations if e.rank == 1)
    assert report.winning_policy_id == rank_one.policy_id


def test_evaluations_ranked_deterministically(tennent_pipeline) -> None:
    health, rec, mv, cf, opt = tennent_pipeline
    report = evaluate_collection_policies(
        rec, mv, cf, opt, scenario_id=SCENARIO_TENNENT,
    )
    items = report.evaluations
    for x, y in zip(items, items[1:]):
        # Primary: planning_utility desc.
        assert x.planning_utility >= y.planning_utility
        if x.planning_utility == y.planning_utility:
            assert x.expected_ambiguity_resolution >= y.expected_ambiguity_resolution
            if x.expected_ambiguity_resolution == y.expected_ambiguity_resolution:
                if x.total_cost == y.total_cost:
                    assert x.policy_id < y.policy_id
                else:
                    assert x.total_cost <= y.total_cost
    # Ranks are 1..N in order.
    assert [e.rank for e in items] == list(range(1, len(items) + 1))


# ---------------------------------------------------------------------------
# Per-policy semantics
# ---------------------------------------------------------------------------


def test_value_optimized_mirrors_optimizer_recommended_plan(tennent_pipeline) -> None:
    health, rec, mv, cf, opt = tennent_pipeline
    report = evaluate_collection_policies(
        rec, mv, cf, opt, scenario_id=SCENARIO_TENNENT,
    )
    expected_ids = tuple(
        sorted(p.candidate_id for p in opt.recommended_plan.selected_items)
    )
    actual = _eval_by_id(report)["value_optimized"]
    assert actual.selected_candidate_ids == expected_ids


def test_ambiguity_first_picks_highest_resolution_candidates(tennent_pipeline) -> None:
    health, rec, mv, cf, opt = tennent_pipeline
    report = evaluate_collection_policies(
        rec, mv, cf, opt, scenario_id=SCENARIO_TENNENT,
    )
    actual = _eval_by_id(report)["ambiguity_first"]
    # Top counterfactual amb_res for Tennent is optical_context (0.85),
    # then cross_geometry_sar (0.75); both should fit under default
    # budget 1.0 (0.40 + 0.70 = 1.10 > 1.0, so only one of them fits with
    # something cheaper).  At minimum optical_context must be present.
    assert "optical_context" in actual.selected_candidate_ids


def test_low_cost_first_starts_with_cheapest(tennent_pipeline) -> None:
    health, rec, mv, cf, opt = tennent_pipeline
    report = evaluate_collection_policies(
        rec, mv, cf, opt, scenario_id=SCENARIO_TENNENT,
    )
    actual = _eval_by_id(report)["low_cost_first"]
    # Cheapest candidates are wait_or_monitor (0.0) and
    # ais_coverage_query (0.10); under budget 1.0 with max_collects 2
    # both should be selected.
    assert "wait_or_monitor" in actual.selected_candidate_ids
    assert "ais_coverage_query" in actual.selected_candidate_ids


def test_sar_repeat_first_prefers_repeat_sar(tennent_pipeline) -> None:
    health, rec, mv, cf, opt = tennent_pipeline
    report = evaluate_collection_policies(
        rec, mv, cf, opt, scenario_id=SCENARIO_TENNENT,
    )
    actual = _eval_by_id(report)["sar_repeat_first"]
    assert "repeat_sar" in actual.selected_candidate_ids


def test_optical_disambiguation_first_prefers_optical_context(tennent_pipeline) -> None:
    health, rec, mv, cf, opt = tennent_pipeline
    report = evaluate_collection_policies(
        rec, mv, cf, opt, scenario_id=SCENARIO_TENNENT,
    )
    actual = _eval_by_id(report)["optical_disambiguation_first"]
    assert "optical_context" in actual.selected_candidate_ids


def test_ais_context_first_prefers_ais_coverage_query(whitsun_pipeline) -> None:
    health, rec, mv, cf, opt = whitsun_pipeline
    report = evaluate_collection_policies(
        rec, mv, cf, opt, scenario_id=SCENARIO_WHITSUN,
    )
    actual = _eval_by_id(report)["ais_context_first"]
    assert "ais_coverage_query" in actual.selected_candidate_ids


# ---------------------------------------------------------------------------
# Constraint enforcement
# ---------------------------------------------------------------------------


def test_every_policy_obeys_budget(tennent_pipeline) -> None:
    health, rec, mv, cf, opt = tennent_pipeline
    tight = PlanConstraint(budget=0.3, max_collects=2)
    # Re-run optimizer under the tight constraint so value_optimized
    # mirrors a plan that respects it.
    opt2 = optimize_collection_plan(
        rec, mv, cf, scenario_id=SCENARIO_TENNENT, constraint=tight,
    )
    report = evaluate_collection_policies(
        rec, mv, cf, opt2, scenario_id=SCENARIO_TENNENT, constraint=tight,
    )
    for e in report.evaluations:
        assert e.total_cost <= tight.budget + 1e-9, (
            f"{e.policy_id} cost {e.total_cost} exceeds budget {tight.budget}"
        )


def test_every_policy_obeys_max_collects(tennent_pipeline) -> None:
    health, rec, mv, cf, opt = tennent_pipeline
    one = PlanConstraint(budget=1.0, max_collects=1)
    opt2 = optimize_collection_plan(
        rec, mv, cf, scenario_id=SCENARIO_TENNENT, constraint=one,
    )
    report = evaluate_collection_policies(
        rec, mv, cf, opt2, scenario_id=SCENARIO_TENNENT, constraint=one,
    )
    for e in report.evaluations:
        assert len(e.selected_candidate_ids) <= 1


def test_required_candidate_present_in_every_policy(tennent_pipeline) -> None:
    health, rec, mv, cf, opt = tennent_pipeline
    require = PlanConstraint(
        budget=1.0, max_collects=2,
        required_candidate_ids=("repeat_sar",),
    )
    opt2 = optimize_collection_plan(
        rec, mv, cf, scenario_id=SCENARIO_TENNENT, constraint=require,
    )
    report = evaluate_collection_policies(
        rec, mv, cf, opt2, scenario_id=SCENARIO_TENNENT, constraint=require,
    )
    for e in report.evaluations:
        assert "repeat_sar" in e.selected_candidate_ids, (
            f"{e.policy_id} did not honour required candidate"
        )


def test_excluded_candidate_absent_from_every_policy(tennent_pipeline) -> None:
    health, rec, mv, cf, opt = tennent_pipeline
    exclude = PlanConstraint(
        budget=1.0, max_collects=2,
        excluded_candidate_ids=("optical_context",),
    )
    opt2 = optimize_collection_plan(
        rec, mv, cf, scenario_id=SCENARIO_TENNENT, constraint=exclude,
    )
    report = evaluate_collection_policies(
        rec, mv, cf, opt2, scenario_id=SCENARIO_TENNENT, constraint=exclude,
    )
    for e in report.evaluations:
        assert "optical_context" not in e.selected_candidate_ids, (
            f"{e.policy_id} did not honour excluded candidate"
        )


def test_infeasible_constraint_returns_empty_selections_with_caveat(
    tennent_pipeline,
) -> None:
    health, rec, mv, cf, opt = tennent_pipeline
    impossible = PlanConstraint(
        budget=0.05, max_collects=2,
        required_candidate_ids=("higher_resolution_sar",),  # cost ~0.80
    )
    # Optimizer also returns empty under this constraint; we pass a fresh
    # call so its plan reflects the constraint.
    opt2 = optimize_collection_plan(
        rec, mv, cf, scenario_id=SCENARIO_TENNENT, constraint=impossible,
    )
    report = evaluate_collection_policies(
        rec, mv, cf, opt2, scenario_id=SCENARIO_TENNENT, constraint=impossible,
    )
    for e in report.evaluations:
        assert e.selected_candidate_ids == ()
        joined = " ".join(e.caveats).lower()
        assert "infeasible" in joined or "no feasible plan" in joined


# ---------------------------------------------------------------------------
# Selectable subset of policies
# ---------------------------------------------------------------------------


def test_policies_parameter_limits_evaluated_set(tennent_pipeline) -> None:
    health, rec, mv, cf, opt = tennent_pipeline
    report = evaluate_collection_policies(
        rec, mv, cf, opt, scenario_id=SCENARIO_TENNENT,
        policies=("value_optimized", "low_cost_first"),
    )
    assert {e.policy_id for e in report.evaluations} == {
        "value_optimized", "low_cost_first",
    }


def test_unknown_policy_id_raises_value_error(tennent_pipeline) -> None:
    health, rec, mv, cf, opt = tennent_pipeline
    with pytest.raises(ValueError) as excinfo:
        evaluate_collection_policies(
            rec, mv, cf, opt, scenario_id=SCENARIO_TENNENT,
            policies=("not_a_real_policy",),
        )
    assert "not_a_real_policy" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Aggregate metric properties
# ---------------------------------------------------------------------------


def test_metrics_are_non_negative(tennent_pipeline) -> None:
    health, rec, mv, cf, opt = tennent_pipeline
    report = evaluate_collection_policies(
        rec, mv, cf, opt, scenario_id=SCENARIO_TENNENT,
    )
    for e in report.evaluations:
        assert e.planning_utility >= 0.0
        assert e.mission_value_total >= 0.0
        assert e.expected_ambiguity_resolution >= 0.0
        assert e.expected_health_score_delta >= 0.0
        assert e.total_cost >= 0.0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_output_is_deterministic_across_two_calls(tennent_pipeline) -> None:
    health, rec, mv, cf, opt = tennent_pipeline
    r1 = evaluate_collection_policies(
        rec, mv, cf, opt, scenario_id=SCENARIO_TENNENT,
    )
    r2 = evaluate_collection_policies(
        rec, mv, cf, opt, scenario_id=SCENARIO_TENNENT,
    )
    s1 = [
        (e.policy_id, e.selected_candidate_ids, e.planning_utility, e.rank)
        for e in r1.evaluations
    ]
    s2 = [
        (e.policy_id, e.selected_candidate_ids, e.planning_utility, e.rank)
        for e in r2.evaluations
    ]
    assert s1 == s2


def test_text_format_is_deterministic_across_two_calls(whitsun_pipeline) -> None:
    health, rec, mv, cf, opt = whitsun_pipeline
    r = evaluate_collection_policies(
        rec, mv, cf, opt, scenario_id=SCENARIO_WHITSUN,
    )
    assert format_policy_eval_text(r) == format_policy_eval_text(r)


# ---------------------------------------------------------------------------
# Tennent-specific / Whitsun-specific narrative checks
# ---------------------------------------------------------------------------


def test_tennent_value_or_optical_ranks_highly(tennent_pipeline) -> None:
    """Under the default constraint, value_optimized or
    optical_disambiguation_first should rank in the top three for Tennent.
    Other policies (e.g. ambiguity_first) may legitimately tie on the
    same selection set; tie-break is alphabetic by policy_id."""
    health, rec, mv, cf, opt = tennent_pipeline
    report = evaluate_collection_policies(
        rec, mv, cf, opt, scenario_id=SCENARIO_TENNENT,
    )
    by_id = _eval_by_id(report)
    assert (
        by_id["value_optimized"].rank <= 3
        or by_id["optical_disambiguation_first"].rank <= 3
    )


def test_whitsun_ais_context_first_performs_well(whitsun_pipeline) -> None:
    """ais_context_first should land in the top half for Whitsun's
    AIS-related ambiguity (rank 1 through 3)."""
    health, rec, mv, cf, opt = whitsun_pipeline
    report = evaluate_collection_policies(
        rec, mv, cf, opt, scenario_id=SCENARIO_WHITSUN,
    )
    actual = _eval_by_id(report)["ais_context_first"]
    assert actual.rank <= 3, (
        f"ais_context_first rank {actual.rank} should be in top 3 for Whitsun"
    )


# ---------------------------------------------------------------------------
# Scope guardrails — language
# ---------------------------------------------------------------------------


_FORBIDDEN_PHRASES = (
    "trained rl",
    "rl agent",
    "reinforcement learning agent",
    "autonomous scheduler",
    "production scheduler",
    "tasking order",
    "live tasking",
    "revenue dollars",
)


def test_no_forbidden_language_in_summaries_or_caveats(tennent_pipeline) -> None:
    health, rec, mv, cf, opt = tennent_pipeline
    report = evaluate_collection_policies(
        rec, mv, cf, opt, scenario_id=SCENARIO_TENNENT,
    )
    blobs = [report.summary]
    for e in report.evaluations:
        blobs.append(e.reason)
        blobs.append(e.label)
        blobs.extend(e.caveats)
    text = " ".join(b.lower() for b in blobs)
    for needle in _FORBIDDEN_PHRASES:
        assert needle not in text, (
            f"forbidden phrase {needle!r} appears in policy-eval output"
        )


def test_no_forbidden_language_in_text_formatter(tennent_pipeline) -> None:
    health, rec, mv, cf, opt = tennent_pipeline
    report = evaluate_collection_policies(
        rec, mv, cf, opt, scenario_id=SCENARIO_TENNENT,
    )
    text = format_policy_eval_text(report).lower()
    for needle in _FORBIDDEN_PHRASES:
        assert needle not in text


# ---------------------------------------------------------------------------
# Scope guardrails — imports + wall-clock
# ---------------------------------------------------------------------------


def test_policy_eval_module_has_no_forbidden_imports() -> None:
    import ast
    import custody.hypotheses.policy_eval as mod
    tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
    forbidden_prefixes = (
        "custody.detection",
        "custody.ingest.gfw_presence",
        "sentinelhub",
        "custody.fusion.tracker",
        "custody.taskrecommendation",
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
        f"policy_eval.py imports forbidden modules: {offending}"
    )


def test_policy_eval_source_has_no_wall_clock() -> None:
    import custody.hypotheses.policy_eval as mod
    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert "datetime.now" not in src
    assert "datetime.utcnow" not in src
