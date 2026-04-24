"""Tests for :mod:`custody.hypotheses.counterfactual` (ADR-0021 Slice 10).

Pin the deterministic three-outcome decomposition under AMBIGUOUS custody,
the cautious single-outcome path under non-ambiguous custody, weight
normalization, deterministic sorting, mission-value passthrough, the
no-calibrated-probabilities language invariant, and the import-boundary
invariant.
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
    CounterfactualCollectAssessment,
    CounterfactualOutcome,
    CounterfactualReport,
    format_counterfactual_text,
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


# ---------------------------------------------------------------------------
# Helpers (mirror the pattern from test_hypothesis_collection_value.py)
# ---------------------------------------------------------------------------


def _ev(
    evidence_id: str,
    scenario_id: str,
    supports: tuple[str, ...] = (),
    contradicts: tuple[str, ...] = (),
    *,
    confidence: float = 0.6,
    weight: float = 1.0,
    timestamp: datetime | None = T_RECENT,
) -> HypothesisEvidence:
    return HypothesisEvidence(
        evidence_id=evidence_id,
        source_ref=f"observation:{evidence_id}",
        source_kind="observation",
        scenario_id=scenario_id,
        timestamp=timestamp,
        supports=supports,
        contradicts=contradicts,
        confidence=confidence,
        weight=weight,
        reason="synthetic",
    )


def _canonical(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def _ambiguous(
    scenario_id: str,
    hid_a: str,
    hid_b: str,
) -> tuple[HypothesisState, HypothesisCustodyHealth, CollectionRecommendation]:
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
    return state, health, rec


def _outcome_by_label(
    assessment: CounterfactualCollectAssessment,
    label_prefix: str,
) -> CounterfactualOutcome | None:
    for o in assessment.outcomes:
        if o.label.startswith(label_prefix):
            return o
    return None


def _assessment_by_id(
    report: CounterfactualReport,
) -> dict[str, CounterfactualCollectAssessment]:
    return {a.candidate_id: a for a in report.ranked_assessments}


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


def test_outcome_is_frozen() -> None:
    o = CounterfactualOutcome(
        outcome_id="x", label="x", likelihood_weight=0.5,
        resulting_top_hypothesis=None,
        resulting_health_status=CustodyHealthStatus.HEALTHY,
        resulting_health_score=0.9, ambiguity_pairs=(),
        health_score_delta=0.1, ambiguity_resolved=True,
        evidence_trace=(), reason="x",
    )
    with pytest.raises(FrozenInstanceError):
        o.likelihood_weight = 0.0  # type: ignore[misc]


def test_assessment_is_frozen() -> None:
    o = CounterfactualOutcome(
        outcome_id="x", label="x", likelihood_weight=1.0,
        resulting_top_hypothesis=None,
        resulting_health_status=CustodyHealthStatus.HEALTHY,
        resulting_health_score=0.9, ambiguity_pairs=(),
        health_score_delta=0.0, ambiguity_resolved=True,
        evidence_trace=(), reason="x",
    )
    a = CounterfactualCollectAssessment(
        candidate_id="x", candidate_label="x",
        current_health_status=CustodyHealthStatus.HEALTHY,
        primary_ambiguity=None, current_mission_value=None,
        expected_health_score_delta=0.0,
        expected_ambiguity_resolution=0.0,
        outcomes=(o,), summary="x",
    )
    with pytest.raises(FrozenInstanceError):
        a.summary = "y"  # type: ignore[misc]


def test_report_is_frozen() -> None:
    r = CounterfactualReport(
        scenario_id="x", primary_ambiguity=None,
        ranked_assessments=(), summary="x",
    )
    with pytest.raises(FrozenInstanceError):
        r.summary = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Three outcomes per candidate under AMBIGUOUS custody
# ---------------------------------------------------------------------------


def test_ambiguous_state_produces_three_outcomes_per_candidate() -> None:
    state, health, rec = _ambiguous(
        SCENARIO_TENNENT,
        TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
        TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
    )
    report = simulate_counterfactual_collects(state, health, rec)
    for a in report.ranked_assessments:
        assert len(a.outcomes) == 3, (
            f"{a.candidate_id} should have 3 outcomes, got {len(a.outcomes)}"
        )


def test_resolve_toward_first_uses_first_hypothesis() -> None:
    """resolve_toward_first should drive top toward the first id in canonical pair."""
    state, health, rec = _ambiguous(
        SCENARIO_TENNENT,
        TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
        TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
    )
    report = simulate_counterfactual_collects(state, health, rec)
    pair = report.primary_ambiguity
    assert pair is not None
    a_id, b_id = pair
    assert a_id < b_id  # canonical
    assess = _assessment_by_id(report)["optical_context"]
    out_first = _outcome_by_label(assess, "resolve_toward_first")
    assert out_first is not None
    # The outcome's reason names the first hypothesis as the supported side.
    assert a_id in out_first.reason
    assert b_id in out_first.reason


def test_resolve_toward_second_uses_second_hypothesis() -> None:
    state, health, rec = _ambiguous(
        SCENARIO_TENNENT,
        TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
        TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
    )
    report = simulate_counterfactual_collects(state, health, rec)
    pair = report.primary_ambiguity
    assert pair is not None
    a_id, b_id = pair
    assess = _assessment_by_id(report)["optical_context"]
    out_second = _outcome_by_label(assess, "resolve_toward_second")
    assert out_second is not None
    # Resolves toward b: name b in supports, a in contradicts.
    assert f"supports {b_id}" in out_second.reason
    assert f"contradicts {a_id}" in out_second.reason


def test_inconclusive_outcome_does_not_resolve_ambiguity() -> None:
    state, health, rec = _ambiguous(
        SCENARIO_TENNENT,
        TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
        TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
    )
    report = simulate_counterfactual_collects(state, health, rec)
    for a in report.ranked_assessments:
        out_inconcl = _outcome_by_label(a, "inconclusive")
        assert out_inconcl is not None
        assert out_inconcl.ambiguity_resolved is False


def test_resolve_outcomes_can_resolve_ambiguity_for_tennent() -> None:
    """At least one resolve outcome should drive ambiguity off for top candidates."""
    state, health, rec = _ambiguous(
        SCENARIO_TENNENT,
        TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
        TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
    )
    report = simulate_counterfactual_collects(state, health, rec)
    top = report.ranked_assessments[0]
    resolved_outcomes = [o for o in top.outcomes if o.ambiguity_resolved]
    assert resolved_outcomes, (
        f"top candidate {top.candidate_id} should have at least one "
        "ambiguity-resolving outcome"
    )


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------


def test_expected_ambiguity_resolution_tracks_resolution_potential() -> None:
    """Higher candidate scores should yield higher expected_ambiguity_resolution."""
    state, health, rec = _ambiguous(
        SCENARIO_TENNENT,
        TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
        TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
    )
    report = simulate_counterfactual_collects(state, health, rec)
    # optical_context (score 0.85) > cross_geometry_sar (0.75) > repeat_sar (0.60)
    by_id = _assessment_by_id(report)
    assert (
        by_id["optical_context"].expected_ambiguity_resolution
        >= by_id["cross_geometry_sar"].expected_ambiguity_resolution
        >= by_id["repeat_sar"].expected_ambiguity_resolution
    )


def test_expected_health_score_delta_positive_for_top_candidate() -> None:
    state, health, rec = _ambiguous(
        SCENARIO_TENNENT,
        TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
        TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
    )
    report = simulate_counterfactual_collects(state, health, rec)
    top = report.ranked_assessments[0]
    assert top.expected_health_score_delta > 0.0


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------


def test_assessments_sorted_deterministically() -> None:
    state, health, rec = _ambiguous(
        SCENARIO_TENNENT,
        TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
        TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
    )
    report = simulate_counterfactual_collects(state, health, rec)
    items = report.ranked_assessments
    for x, y in zip(items, items[1:]):
        # Primary: expected_ambiguity_resolution descending.
        assert x.expected_ambiguity_resolution >= y.expected_ambiguity_resolution
        if x.expected_ambiguity_resolution == y.expected_ambiguity_resolution:
            # Secondary: expected_health_score_delta descending.
            assert x.expected_health_score_delta >= y.expected_health_score_delta
            if x.expected_health_score_delta == y.expected_health_score_delta:
                # Tertiary: candidate_id ascending.
                assert x.candidate_id < y.candidate_id


def test_max_results_truncates_after_sorting() -> None:
    state, health, rec = _ambiguous(
        SCENARIO_TENNENT,
        TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
        TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
    )
    full = simulate_counterfactual_collects(state, health, rec)
    top3 = simulate_counterfactual_collects(state, health, rec, max_results=3)
    assert len(top3.ranked_assessments) == 3
    full_top3 = [a.candidate_id for a in full.ranked_assessments[:3]]
    actual_top3 = [a.candidate_id for a in top3.ranked_assessments]
    assert actual_top3 == full_top3


# ---------------------------------------------------------------------------
# mission_value_report integration
# ---------------------------------------------------------------------------


def test_mission_value_report_attaches_current_mission_value() -> None:
    state, health, rec = _ambiguous(
        SCENARIO_TENNENT,
        TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
        TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
    )
    mv_report = attribute_mission_value(rec, health)
    report = simulate_counterfactual_collects(
        state, health, rec, mission_value_report=mv_report,
    )
    for a in report.ranked_assessments:
        assert a.current_mission_value is not None
        # Should match the mission-value report value for this candidate.
        mv_for = next(
            x.total_value for x in mv_report.ranked_assessments
            if x.candidate_id == a.candidate_id
        )
        assert a.current_mission_value == pytest.approx(mv_for)


def test_no_mission_value_report_leaves_mission_value_none() -> None:
    state, health, rec = _ambiguous(
        SCENARIO_TENNENT,
        TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
        TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
    )
    report = simulate_counterfactual_collects(state, health, rec)
    for a in report.ranked_assessments:
        assert a.current_mission_value is None


# ---------------------------------------------------------------------------
# Non-ambiguous paths
# ---------------------------------------------------------------------------


def test_healthy_state_produces_cautious_single_outcome() -> None:
    """HEALTHY custody -> one cautious outcome, no resolve-toward branches."""
    state = update_state(
        SCENARIO_TENNENT,
        evidence=(_ev("ev1", SCENARIO_TENNENT,
                      supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,),
                      confidence=0.85),),
    )
    health = assess_custody_health(state, as_of=T_RECENT)
    assert health.status is CustodyHealthStatus.HEALTHY
    rec = rank_collection_candidates(state, health)
    report = simulate_counterfactual_collects(state, health, rec)
    assert report.primary_ambiguity is None
    for a in report.ranked_assessments:
        assert len(a.outcomes) == 1
        assert a.outcomes[0].likelihood_weight == 1.0
        assert "non-ambiguous" in a.outcomes[0].label.lower()
    assert "AMBIGUOUS" in report.summary or "ambiguous" in report.summary.lower()


# ---------------------------------------------------------------------------
# Weight normalization
# ---------------------------------------------------------------------------


def test_likelihood_weights_sum_to_one_per_candidate() -> None:
    state, health, rec = _ambiguous(
        SCENARIO_TENNENT,
        TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
        TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
    )
    report = simulate_counterfactual_collects(state, health, rec)
    for a in report.ranked_assessments:
        total = sum(o.likelihood_weight for o in a.outcomes)
        assert total == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# No calibrated-probability language in any output
# ---------------------------------------------------------------------------


def test_no_calibrated_probability_language_in_text_or_summary() -> None:
    state, health, rec = _ambiguous(
        SCENARIO_TENNENT,
        TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
        TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
    )
    report = simulate_counterfactual_collects(state, health, rec)
    text = format_counterfactual_text(report)
    forbidden = (
        "calibrated probability",
        "sensor performance probability",
        "actual outcome likelihood",
        "live tasking",
        "production scheduler",
    )
    lowered = text.lower()
    summary_lowered = report.summary.lower()
    for needle in forbidden:
        assert needle not in lowered, (
            f"forbidden phrase {needle!r} appears in counterfactual text output"
        )
        assert needle not in summary_lowered, (
            f"forbidden phrase {needle!r} appears in counterfactual summary"
        )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_output_is_deterministic_across_two_calls() -> None:
    state, health, rec = _ambiguous(
        SCENARIO_TENNENT,
        TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
        TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
    )
    r1 = simulate_counterfactual_collects(state, health, rec)
    r2 = simulate_counterfactual_collects(state, health, rec)
    s1 = [
        (a.candidate_id, a.expected_ambiguity_resolution,
         a.expected_health_score_delta)
        for a in r1.ranked_assessments
    ]
    s2 = [
        (a.candidate_id, a.expected_ambiguity_resolution,
         a.expected_health_score_delta)
        for a in r2.ranked_assessments
    ]
    assert s1 == s2


def test_text_format_is_deterministic_across_two_calls() -> None:
    state, health, rec = _ambiguous(
        SCENARIO_WHITSUN,
        WHITSUN_AIS_DARK_OR_POORLY_OBSERVED_VESSELS,
        WHITSUN_VESSEL_CLUSTER_ACTIVITY,
    )
    r = simulate_counterfactual_collects(state, health, rec)
    assert format_counterfactual_text(r) == format_counterfactual_text(r)


# ---------------------------------------------------------------------------
# scenario_id passthrough
# ---------------------------------------------------------------------------


def test_scenario_id_defaults_to_state() -> None:
    state, health, rec = _ambiguous(
        SCENARIO_TENNENT,
        TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
        TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
    )
    report = simulate_counterfactual_collects(state, health, rec)
    assert report.scenario_id == SCENARIO_TENNENT


def test_scenario_id_can_be_overridden() -> None:
    state, health, rec = _ambiguous(
        SCENARIO_TENNENT,
        TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
        TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
    )
    report = simulate_counterfactual_collects(
        state, health, rec, scenario_id="custom",
    )
    assert report.scenario_id == "custom"


# ---------------------------------------------------------------------------
# Import-boundary invariants
# ---------------------------------------------------------------------------


def test_counterfactual_module_has_no_forbidden_imports() -> None:
    import ast
    import custody.hypotheses.counterfactual as mod
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
        f"counterfactual.py imports forbidden modules: {offending}"
    )


def test_counterfactual_source_has_no_wall_clock() -> None:
    import custody.hypotheses.counterfactual as mod
    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert "datetime.now" not in src
    assert "datetime.utcnow" not in src
