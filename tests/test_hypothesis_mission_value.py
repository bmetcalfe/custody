"""Tests for :mod:`custody.hypotheses.mission_value` (ADR-0021 Slice 9).

Pins the mission-value attribution proxy: component decomposition, score
clamping, deterministic ordering, scenario-specific behaviour, the
mission-value-proxy caveat, absence of revenue claims, and the
import-boundary invariant (no Sentinel / VLM / matcher-runtime /
real-data-loader imports).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from custody.hypotheses.collection_value import (
    CollectionRecommendation,
    rank_collection_candidates,
)
from custody.hypotheses.custody_health import (
    CustodyHealthStatus,
    HypothesisCustodyHealth,
    assess_custody_health,
)
from custody.hypotheses.mission_value import (
    MissionValueAssessment,
    MissionValueComponent,
    MissionValueReport,
    attribute_mission_value,
    format_mission_value_text,
)
from custody.hypotheses.registry import (
    SCENARIO_TENNENT,
    SCENARIO_WHITSUN,
    TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
    TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
    TENNENT_STATIONARY_SAR_SCATTER_OR_REEF_CLUTTER,
    WHITSUN_AIS_DARK_OR_POORLY_OBSERVED_VESSELS,
    WHITSUN_DETECTOR_CLUTTER_FALSE_POSITIVES,
    WHITSUN_VESSEL_CLUSTER_ACTIVITY,
)
from custody.hypotheses.types import HypothesisEvidence, HypothesisState
from custody.hypotheses.update import update_state


T_RECENT = datetime(2023, 8, 13, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
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


def _ambiguous_state_health_rec(
    scenario_id: str,
    hid_a: str,
    hid_b: str,
) -> tuple[HypothesisState, HypothesisCustodyHealth, CollectionRecommendation]:
    """Build AMBIGUOUS state + health + recommendation for two hypotheses."""
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


def _score_by_id(report: MissionValueReport) -> dict[str, float]:
    return {a.candidate_id: a.total_value for a in report.ranked_assessments}


def _component_by_name(
    assessment: MissionValueAssessment,
) -> dict[str, float]:
    return {c.name: c.contribution for c in assessment.components}


def _assessment_by_id(
    report: MissionValueReport,
) -> dict[str, MissionValueAssessment]:
    return {a.candidate_id: a for a in report.ranked_assessments}


# ---------------------------------------------------------------------------
# Component decomposition — Tennent
# ---------------------------------------------------------------------------


def test_tennent_optical_context_has_high_ambiguity_reduction() -> None:
    """optical_context for Tennent construction-vs-fixed should have
    the highest ambiguity_reduction among candidates."""
    _, health, rec = _ambiguous_state_health_rec(
        SCENARIO_TENNENT,
        TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
        TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
    )
    report = attribute_mission_value(rec, health)
    assessments = _assessment_by_id(report)
    optical = assessments["optical_context"]
    optical_ar = _component_by_name(optical)["ambiguity_reduction"]

    # Verify optical_context has the highest ambiguity_reduction.
    for a in report.ranked_assessments:
        if a.candidate_id != "optical_context":
            other_ar = _component_by_name(a)["ambiguity_reduction"]
            assert optical_ar >= other_ar, (
                f"optical_context ambiguity_reduction {optical_ar} should be "
                f">= {a.candidate_id} {other_ar}"
            )


def test_tennent_optical_context_ranks_first_by_total_value() -> None:
    _, health, rec = _ambiguous_state_health_rec(
        SCENARIO_TENNENT,
        TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
        TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
    )
    report = attribute_mission_value(rec, health)
    assert report.ranked_assessments[0].candidate_id == "optical_context"


# ---------------------------------------------------------------------------
# Component decomposition — Whitsun
# ---------------------------------------------------------------------------


def test_whitsun_ais_query_has_high_ambiguity_reduction_for_ais_pair() -> None:
    """ais_coverage_query for Whitsun AIS-dark-vs-cluster should have
    the highest ambiguity_reduction."""
    _, health, rec = _ambiguous_state_health_rec(
        SCENARIO_WHITSUN,
        WHITSUN_AIS_DARK_OR_POORLY_OBSERVED_VESSELS,
        WHITSUN_VESSEL_CLUSTER_ACTIVITY,
    )
    report = attribute_mission_value(rec, health)
    assessments = _assessment_by_id(report)
    ais = assessments["ais_coverage_query"]
    ais_ar = _component_by_name(ais)["ambiguity_reduction"]

    for a in report.ranked_assessments:
        if a.candidate_id != "ais_coverage_query":
            other_ar = _component_by_name(a)["ambiguity_reduction"]
            assert ais_ar >= other_ar


# ---------------------------------------------------------------------------
# Cost and latency penalties reduce total value
# ---------------------------------------------------------------------------


def test_cost_penalty_reduces_total_value() -> None:
    """Candidates with higher relative_cost should have lower total_value,
    all else being equal."""
    _, health, rec = _ambiguous_state_health_rec(
        SCENARIO_TENNENT,
        TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
        TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
    )
    report = attribute_mission_value(rec, health)
    for a in report.ranked_assessments:
        cost_comp = _component_by_name(a)["cost_penalty"]
        assert cost_comp <= 0.0, (
            f"{a.candidate_id} cost_penalty should be non-positive, "
            f"got {cost_comp}"
        )


def test_latency_penalty_reduces_total_value() -> None:
    _, health, rec = _ambiguous_state_health_rec(
        SCENARIO_TENNENT,
        TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
        TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
    )
    report = attribute_mission_value(rec, health)
    for a in report.ranked_assessments:
        latency_comp = _component_by_name(a)["latency_penalty"]
        assert latency_comp <= 0.0


# ---------------------------------------------------------------------------
# Mission relevance changes total value
# ---------------------------------------------------------------------------


def test_higher_mission_relevance_increases_total_value() -> None:
    _, health, rec = _ambiguous_state_health_rec(
        SCENARIO_TENNENT,
        TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
        TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
    )
    report_low = attribute_mission_value(rec, health, mission_relevance=0.1)
    report_high = attribute_mission_value(rec, health, mission_relevance=0.9)

    scores_low = _score_by_id(report_low)
    scores_high = _score_by_id(report_high)
    for cid in scores_low:
        assert scores_high[cid] >= scores_low[cid], (
            f"{cid} total_value should increase with mission_relevance"
        )


def test_mission_relevance_component_scales_with_weight() -> None:
    _, health, rec = _ambiguous_state_health_rec(
        SCENARIO_TENNENT,
        TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
        TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
    )
    report_lo = attribute_mission_value(rec, health, mission_relevance=0.2)
    report_hi = attribute_mission_value(rec, health, mission_relevance=0.8)

    for a_lo, a_hi in zip(
        report_lo.ranked_assessments, report_hi.ranked_assessments
    ):
        mr_lo = _component_by_name(a_lo)["mission_relevance"]
        mr_hi = _component_by_name(a_hi)["mission_relevance"]
        assert mr_hi > mr_lo


# ---------------------------------------------------------------------------
# Total value clamping
# ---------------------------------------------------------------------------


def test_total_values_clamped_zero_to_one() -> None:
    _, health, rec = _ambiguous_state_health_rec(
        SCENARIO_TENNENT,
        TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
        TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
    )
    for mr in (0.0, 0.5, 1.0):
        report = attribute_mission_value(rec, health, mission_relevance=mr)
        for a in report.ranked_assessments:
            assert 0.0 <= a.total_value <= 1.0, (
                f"{a.candidate_id} total_value {a.total_value} out of [0,1]"
            )


def test_extreme_mission_relevance_still_clamped() -> None:
    """mission_relevance > 1.0 or < 0.0 should clamp internally."""
    _, health, rec = _ambiguous_state_health_rec(
        SCENARIO_TENNENT,
        TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
        TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
    )
    for mr in (-1.0, 5.0):
        report = attribute_mission_value(rec, health, mission_relevance=mr)
        for a in report.ranked_assessments:
            assert 0.0 <= a.total_value <= 1.0


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------


def test_report_sorted_by_total_value_descending() -> None:
    _, health, rec = _ambiguous_state_health_rec(
        SCENARIO_TENNENT,
        TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
        TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
    )
    report = attribute_mission_value(rec, health)
    values = [a.total_value for a in report.ranked_assessments]
    assert values == sorted(values, reverse=True)


def test_tiebreak_by_candidate_id_ascending() -> None:
    """When total_value ties, candidate_id ascending is the tiebreaker."""
    _, health, rec = _ambiguous_state_health_rec(
        SCENARIO_TENNENT,
        TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
        TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
    )
    report = attribute_mission_value(rec, health)
    for a1, a2 in zip(
        report.ranked_assessments, report.ranked_assessments[1:]
    ):
        if a1.total_value == a2.total_value:
            assert a1.candidate_id < a2.candidate_id


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_output_is_deterministic_across_two_calls() -> None:
    _, health, rec = _ambiguous_state_health_rec(
        SCENARIO_TENNENT,
        TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
        TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
    )
    r1 = attribute_mission_value(rec, health)
    r2 = attribute_mission_value(rec, health)
    s1 = [(a.candidate_id, a.total_value) for a in r1.ranked_assessments]
    s2 = [(a.candidate_id, a.total_value) for a in r2.ranked_assessments]
    assert s1 == s2


# ---------------------------------------------------------------------------
# Caveats — mission-value proxy language
# ---------------------------------------------------------------------------


def test_all_assessments_carry_proxy_caveat() -> None:
    _, health, rec = _ambiguous_state_health_rec(
        SCENARIO_TENNENT,
        TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
        TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
    )
    report = attribute_mission_value(rec, health)
    for a in report.ranked_assessments:
        proxy_caveats = [
            c for c in a.caveats if "proxy" in c.lower()
        ]
        assert proxy_caveats, (
            f"{a.candidate_id} must carry a 'mission-value proxy' caveat"
        )


def test_no_revenue_claims_in_output() -> None:
    """The output must never claim 'actual revenue'."""
    _, health, rec = _ambiguous_state_health_rec(
        SCENARIO_TENNENT,
        TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
        TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
    )
    report = attribute_mission_value(rec, health)
    text = format_mission_value_text(report)
    assert "actual revenue" not in text.lower()
    assert "revenue" not in text.lower()
    # Also check the summary and caveats.
    assert "actual revenue" not in report.summary.lower()
    for a in report.ranked_assessments:
        for c in a.caveats:
            assert "actual revenue" not in c.lower()


# ---------------------------------------------------------------------------
# False-positive risk penalty — clutter pairs
# ---------------------------------------------------------------------------


def test_clutter_pair_incurs_false_positive_penalty() -> None:
    """When the ambiguity pair includes a clutter hypothesis, the
    false_positive_risk_penalty should be non-zero for low-scoring
    candidates."""
    _, health, rec = _ambiguous_state_health_rec(
        SCENARIO_TENNENT,
        TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
        TENNENT_STATIONARY_SAR_SCATTER_OR_REEF_CLUTTER,
    )
    report = attribute_mission_value(rec, health)
    for a in report.ranked_assessments:
        fp = _component_by_name(a)["false_positive_risk_penalty"]
        assert fp <= 0.0


def test_whitsun_clutter_pair_incurs_penalty() -> None:
    _, health, rec = _ambiguous_state_health_rec(
        SCENARIO_WHITSUN,
        WHITSUN_DETECTOR_CLUTTER_FALSE_POSITIVES,
        WHITSUN_VESSEL_CLUSTER_ACTIVITY,
    )
    report = attribute_mission_value(rec, health)
    for a in report.ranked_assessments:
        fp = _component_by_name(a)["false_positive_risk_penalty"]
        assert fp <= 0.0


def test_non_clutter_pair_has_zero_false_positive_penalty() -> None:
    _, health, rec = _ambiguous_state_health_rec(
        SCENARIO_WHITSUN,
        WHITSUN_AIS_DARK_OR_POORLY_OBSERVED_VESSELS,
        WHITSUN_VESSEL_CLUSTER_ACTIVITY,
    )
    report = attribute_mission_value(rec, health)
    for a in report.ranked_assessments:
        fp = _component_by_name(a)["false_positive_risk_penalty"]
        assert fp == 0.0


# ---------------------------------------------------------------------------
# Component completeness
# ---------------------------------------------------------------------------


def test_all_seven_components_present() -> None:
    _, health, rec = _ambiguous_state_health_rec(
        SCENARIO_TENNENT,
        TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
        TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
    )
    report = attribute_mission_value(rec, health)
    expected_names = {
        "ambiguity_reduction",
        "custody_health_improvement",
        "mission_relevance",
        "timeliness",
        "cost_penalty",
        "latency_penalty",
        "false_positive_risk_penalty",
    }
    for a in report.ranked_assessments:
        actual_names = {c.name for c in a.components}
        assert actual_names == expected_names, (
            f"{a.candidate_id} missing components: "
            f"{expected_names - actual_names}"
        )


def test_every_component_has_reason() -> None:
    _, health, rec = _ambiguous_state_health_rec(
        SCENARIO_TENNENT,
        TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
        TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
    )
    report = attribute_mission_value(rec, health)
    for a in report.ranked_assessments:
        for c in a.components:
            assert c.reason, f"{a.candidate_id}.{c.name} has empty reason"


# ---------------------------------------------------------------------------
# Text formatter
# ---------------------------------------------------------------------------


def test_format_mission_value_text_includes_top_candidates() -> None:
    _, health, rec = _ambiguous_state_health_rec(
        SCENARIO_TENNENT,
        TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
        TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
    )
    report = attribute_mission_value(rec, health)
    text = format_mission_value_text(report)
    assert "Mission value proxy:" in text
    assert "Contributions:" in text
    assert "ambiguity_reduction" in text


# ---------------------------------------------------------------------------
# Non-ambiguous health statuses
# ---------------------------------------------------------------------------


def test_healthy_state_produces_low_total_values() -> None:
    state = update_state(
        SCENARIO_TENNENT,
        evidence=(_ev("ev1", SCENARIO_TENNENT,
                      supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,),
                      confidence=0.85),),
    )
    health = assess_custody_health(state, as_of=T_RECENT)
    assert health.status is CustodyHealthStatus.HEALTHY
    rec = rank_collection_candidates(state, health)
    report = attribute_mission_value(rec, health)
    # Healthy state → low ambiguity reduction across all candidates.
    for a in report.ranked_assessments:
        assert a.total_value <= 0.50, (
            f"{a.candidate_id} should have modest value under HEALTHY, "
            f"got {a.total_value}"
        )


def test_lost_state_produces_minimal_total_values() -> None:
    state = update_state(SCENARIO_TENNENT, evidence=())
    health = assess_custody_health(state)
    assert health.status is CustodyHealthStatus.LOST
    rec = rank_collection_candidates(state, health)
    report = attribute_mission_value(rec, health)
    for a in report.ranked_assessments:
        assert a.total_value <= 0.35


# ---------------------------------------------------------------------------
# scenario_id passthrough
# ---------------------------------------------------------------------------


def test_scenario_id_defaults_to_recommendation() -> None:
    _, health, rec = _ambiguous_state_health_rec(
        SCENARIO_TENNENT,
        TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
        TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
    )
    report = attribute_mission_value(rec, health)
    assert report.scenario_id == SCENARIO_TENNENT


def test_scenario_id_can_be_overridden() -> None:
    _, health, rec = _ambiguous_state_health_rec(
        SCENARIO_TENNENT,
        TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
        TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
    )
    report = attribute_mission_value(rec, health, scenario_id="custom")
    assert report.scenario_id == "custom"


# ---------------------------------------------------------------------------
# Import-boundary invariants
# ---------------------------------------------------------------------------


def test_mission_value_module_has_no_forbidden_imports() -> None:
    """Sentinel / VLM / matcher-runtime / GFW / real-data-loader imports
    are forbidden."""
    import ast
    import custody.hypotheses.mission_value as mod
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
        f"mission_value.py imports forbidden modules: {offending}; "
        "Slice 9 is a value-attribution proxy, not a platform integration"
    )


def test_no_revenue_language_in_module_source() -> None:
    """The module source must use 'mission-value proxy' language,
    not 'revenue' or 'actual revenue'."""
    import custody.hypotheses.mission_value as mod
    source = Path(mod.__file__).read_text(encoding="utf-8")
    assert "actual revenue" not in source.lower(), (
        "mission_value.py must not claim 'actual revenue'"
    )
