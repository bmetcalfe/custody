"""Tests for :mod:`custody.hypotheses.collection_value` (ADR-0021 Slice 5).

Pins the strategy-table rankings for each covered ambiguity pair, the
fallback behaviour for uncovered pairs, the non-ambiguous status paths
(HEALTHY / LOST / STALE / DEGRADED), deterministic ordering, score
clamping, and the import-boundary invariant (no Sentinel / VLM /
matcher-runtime / real-data-loader imports).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from custody.hypotheses.collection_value import (
    CollectionCandidate,
    CollectionRecommendation,
    CollectionValue,
    rank_collection_candidates,
)
from custody.hypotheses.custody_health import (
    CustodyHealthStatus,
    HypothesisCustodyHealth,
    assess_custody_health,
)
from custody.hypotheses.registry import (
    SCENARIO_TENNENT,
    SCENARIO_WHITSUN,
    TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
    TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
    TENNENT_STATIONARY_SAR_SCATTER_OR_REEF_CLUTTER,
    TENNENT_TRANSIENT_VESSEL_ACTIVITY,
    WHITSUN_AIS_DARK_OR_POORLY_OBSERVED_VESSELS,
    WHITSUN_DETECTOR_CLUTTER_FALSE_POSITIVES,
    WHITSUN_TRANSIENT_ANCHORAGE_OR_FISHING_PRESENCE,
    WHITSUN_VESSEL_CLUSTER_ACTIVITY,
)
from custody.hypotheses.types import HypothesisEvidence, HypothesisState
from custody.hypotheses.update import update_state


T0 = datetime(2023, 7, 2, 12, 0, 0, tzinfo=timezone.utc)
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


def _ambiguous_state_and_health(
    scenario_id: str,
    hid_a: str,
    hid_b: str,
) -> tuple[HypothesisState, HypothesisCustodyHealth]:
    """Build a state where hid_a and hid_b are both saturated → AMBIGUOUS."""
    state = update_state(
        scenario_id,
        evidence=(
            _ev("ev-a", scenario_id, supports=(hid_a,), confidence=1.0),
            _ev("ev-b", scenario_id, supports=(hid_b,), confidence=1.0),
        ),
    )
    health = assess_custody_health(state, as_of=T_RECENT)
    assert health.status is CustodyHealthStatus.AMBIGUOUS
    # The canonical pair must be present in the ambiguity set.
    assert _canonical(hid_a, hid_b) in health.ambiguity_pairs
    return state, health


def _sorted_scores(rec: CollectionRecommendation) -> list[tuple[str, float]]:
    return [(v.candidate.candidate_id, v.score) for v in rec.ranked_values]


def _score_by_id(rec: CollectionRecommendation) -> dict[str, float]:
    return {v.candidate.candidate_id: v.score for v in rec.ranked_values}


# ---------------------------------------------------------------------------
# Covered strategy pairs — Tennent
# ---------------------------------------------------------------------------


def test_tennent_construction_vs_fixed_structure_recommends_optical_context_first() -> None:
    state, health = _ambiguous_state_and_health(
        SCENARIO_TENNENT,
        TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
        TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
    )
    rec = rank_collection_candidates(state, health)
    top = rec.ranked_values[0]
    assert top.candidate.candidate_id == "optical_context"
    assert rec.primary_ambiguity == _canonical(
        TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
        TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
    )


def test_construction_vs_fixed_cross_geo_ranks_above_ais_query() -> None:
    state, health = _ambiguous_state_and_health(
        SCENARIO_TENNENT,
        TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
        TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
    )
    rec = rank_collection_candidates(state, health)
    scores = _score_by_id(rec)
    assert scores["cross_geometry_sar"] > scores["ais_coverage_query"]


def test_fixed_structure_vs_stationary_scatter_ranks_cross_geo_high() -> None:
    state, health = _ambiguous_state_and_health(
        SCENARIO_TENNENT,
        TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
        TENNENT_STATIONARY_SAR_SCATTER_OR_REEF_CLUTTER,
    )
    rec = rank_collection_candidates(state, health)
    top = rec.ranked_values[0]
    assert top.candidate.candidate_id == "cross_geometry_sar"


def test_fixed_vs_transient_vessel_ranks_repeat_sar_high() -> None:
    state, health = _ambiguous_state_and_health(
        SCENARIO_TENNENT,
        TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
        TENNENT_TRANSIENT_VESSEL_ACTIVITY,
    )
    rec = rank_collection_candidates(state, health)
    scores = _score_by_id(rec)
    # repeat_sar should beat cross_geometry_sar for this pair
    assert scores["repeat_sar"] > scores["cross_geometry_sar"]


# ---------------------------------------------------------------------------
# Covered strategy pairs — Whitsun
# ---------------------------------------------------------------------------


def test_vessel_cluster_vs_detector_clutter_ranks_repeat_sar_or_optical_high() -> None:
    state, health = _ambiguous_state_and_health(
        SCENARIO_WHITSUN,
        WHITSUN_VESSEL_CLUSTER_ACTIVITY,
        WHITSUN_DETECTOR_CLUTTER_FALSE_POSITIVES,
    )
    rec = rank_collection_candidates(state, health)
    top_ids = {v.candidate.candidate_id for v in rec.ranked_values[:2]}
    assert {"repeat_sar", "optical_context"}.issubset(top_ids)


def test_ais_dark_vs_vessel_cluster_ranks_ais_coverage_query_first() -> None:
    state, health = _ambiguous_state_and_health(
        SCENARIO_WHITSUN,
        WHITSUN_AIS_DARK_OR_POORLY_OBSERVED_VESSELS,
        WHITSUN_VESSEL_CLUSTER_ACTIVITY,
    )
    rec = rank_collection_candidates(state, health)
    assert rec.ranked_values[0].candidate.candidate_id == "ais_coverage_query"


def test_transient_anchorage_vs_vessel_cluster_ranks_repeat_sar_first() -> None:
    state, health = _ambiguous_state_and_health(
        SCENARIO_WHITSUN,
        WHITSUN_TRANSIENT_ANCHORAGE_OR_FISHING_PRESENCE,
        WHITSUN_VESSEL_CLUSTER_ACTIVITY,
    )
    rec = rank_collection_candidates(state, health)
    assert rec.ranked_values[0].candidate.candidate_id == "repeat_sar"


# ---------------------------------------------------------------------------
# Fallback for unknown ambiguity pairs
# ---------------------------------------------------------------------------


def test_unknown_ambiguity_pair_returns_fallback_low_confidence() -> None:
    """Construct a hand-built AMBIGUOUS health with a pair not in the table."""
    state = update_state(
        SCENARIO_TENNENT,
        evidence=(_ev("ev1", SCENARIO_TENNENT,
                      supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,),
                      confidence=0.5),),
    )
    unknown_pair = ("alpha_hypothesis", "beta_hypothesis")
    fake_health = HypothesisCustodyHealth(
        status=CustodyHealthStatus.AMBIGUOUS,
        score=0.5,
        top_hypothesis="alpha_hypothesis",
        top_score=0.6,
        second_hypothesis="beta_hypothesis",
        second_score=0.55,
        top_two_margin=0.05,
        ambiguity_pairs=(unknown_pair,),
        latest_evidence_at=T_RECENT,
        drivers=("synthetic",),
        reason="synthetic",
    )
    rec = rank_collection_candidates(state, fake_health)
    # Every score is the small uniform fallback value.
    for v in rec.ranked_values:
        assert v.score == pytest.approx(0.10)
    # At least one caveat names the "outside covered strategy table" condition.
    for v in rec.ranked_values:
        assert any("outside covered strategy table" in c.lower() for c in v.caveats)
    assert "outside the covered strategy table" in rec.summary.lower()


# ---------------------------------------------------------------------------
# Non-ambiguous status paths
# ---------------------------------------------------------------------------


def test_healthy_state_recommends_wait_or_monitor_in_summary() -> None:
    state = update_state(
        SCENARIO_TENNENT,
        evidence=(_ev("ev1", SCENARIO_TENNENT,
                      supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,),
                      confidence=0.85),),
    )
    health = assess_custody_health(state, as_of=T_RECENT)
    assert health.status is CustodyHealthStatus.HEALTHY
    rec = rank_collection_candidates(state, health)
    top = rec.ranked_values[0].candidate.candidate_id
    assert top == "wait_or_monitor"
    assert "separated" in rec.summary.lower() or "wait" in rec.summary.lower()
    assert rec.primary_ambiguity is None


def test_lost_state_summary_says_insufficient_evidence() -> None:
    state = update_state(SCENARIO_TENNENT, evidence=())
    health = assess_custody_health(state)
    assert health.status is CustodyHealthStatus.LOST
    rec = rank_collection_candidates(state, health)
    assert "insufficient evidence" in rec.summary.lower()
    # Every score should be low (<= 0.30) — nothing to target.
    for v in rec.ranked_values:
        assert v.score <= 0.30


def test_stale_state_summary_recommends_reacquisition() -> None:
    state = update_state(
        SCENARIO_TENNENT,
        evidence=(_ev("ev1", SCENARIO_TENNENT,
                      supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,),
                      confidence=0.85, timestamp=T0),),
    )
    as_of = T0 + timedelta(days=60)
    health = assess_custody_health(state, as_of=as_of)
    assert health.status is CustodyHealthStatus.STALE
    rec = rank_collection_candidates(state, health)
    # Repeat-SAR should be a leading candidate for reacquisition.
    assert rec.ranked_values[0].candidate.candidate_id == "repeat_sar"
    assert (
        "reacquisition" in rec.summary.lower()
        or "confirmation" in rec.summary.lower()
        or "stale" in rec.summary.lower()
    )


def test_degraded_state_summary_is_cautious() -> None:
    state = update_state(
        SCENARIO_TENNENT,
        evidence=(_ev("ev1", SCENARIO_TENNENT,
                      supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,),
                      confidence=0.3),),
    )
    health = assess_custody_health(state, as_of=T_RECENT)
    assert health.status is CustodyHealthStatus.DEGRADED
    rec = rank_collection_candidates(state, health)
    # Generic confirmation candidates, cautious reasons.
    top_id = rec.ranked_values[0].candidate.candidate_id
    assert top_id in {"repeat_sar", "optical_context"}
    assert (
        "degraded" in rec.summary.lower()
        or "confirmation" in rec.summary.lower()
    )


# ---------------------------------------------------------------------------
# Score clamping, ordering, max_results
# ---------------------------------------------------------------------------


def test_all_scores_are_clamped_between_zero_and_one() -> None:
    state, health = _ambiguous_state_and_health(
        SCENARIO_TENNENT,
        TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
        TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
    )
    rec = rank_collection_candidates(state, health)
    for v in rec.ranked_values:
        assert 0.0 <= v.score <= 1.0


def test_ranked_values_sorted_descending_with_candidate_id_tiebreak() -> None:
    state, health = _ambiguous_state_and_health(
        SCENARIO_TENNENT,
        TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
        TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
    )
    rec = rank_collection_candidates(state, health)
    pairs = _sorted_scores(rec)
    for (_, s1), (_, s2) in zip(pairs, pairs[1:]):
        assert s1 >= s2
    # For any score tie, candidate_id must be ascending.
    for (id1, s1), (id2, s2) in zip(pairs, pairs[1:]):
        if s1 == s2:
            assert id1 < id2


def test_output_is_deterministic_across_two_calls() -> None:
    state, health = _ambiguous_state_and_health(
        SCENARIO_TENNENT,
        TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
        TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
    )
    rec1 = rank_collection_candidates(state, health)
    rec2 = rank_collection_candidates(state, health)
    assert _sorted_scores(rec1) == _sorted_scores(rec2)


def test_max_results_truncates_after_sorting() -> None:
    state, health = _ambiguous_state_and_health(
        SCENARIO_TENNENT,
        TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
        TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
    )
    rec_full = rank_collection_candidates(state, health)
    rec_top3 = rank_collection_candidates(state, health, max_results=3)
    assert len(rec_top3.ranked_values) == 3
    assert _sorted_scores(rec_top3) == _sorted_scores(rec_full)[:3]


def test_max_results_none_returns_full_list() -> None:
    state, health = _ambiguous_state_and_health(
        SCENARIO_TENNENT,
        TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
        TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
    )
    rec = rank_collection_candidates(state, health, max_results=None)
    # 6 candidate types per the module: repeat_sar, cross_geometry_sar,
    # higher_resolution_sar, optical_context, ais_coverage_query, wait_or_monitor.
    assert len(rec.ranked_values) == 6


# ---------------------------------------------------------------------------
# CollectionValue shape
# ---------------------------------------------------------------------------


def test_collection_value_fields_populated() -> None:
    state, health = _ambiguous_state_and_health(
        SCENARIO_TENNENT,
        TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
        TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
    )
    rec = rank_collection_candidates(state, health)
    v = rec.ranked_values[0]
    assert isinstance(v, CollectionValue)
    assert isinstance(v.candidate, CollectionCandidate)
    assert v.reason
    assert v.disambiguates == rec.primary_ambiguity
    assert isinstance(v.caveats, tuple)


# ---------------------------------------------------------------------------
# Import-boundary invariants
# ---------------------------------------------------------------------------


def test_collection_value_module_has_no_forbidden_imports() -> None:
    """Sentinel / VLM / matcher-runtime / GFW / real-data-loader imports are forbidden."""
    import ast
    import custody.hypotheses.collection_value as mod
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
        f"collection_value.py imports forbidden modules: {offending}; "
        "Slice 5 ranks collect TYPES, not platform integrations"
    )


def test_no_candidate_label_claims_sentinel_integration() -> None:
    """Candidate labels must stay sensor-generic (no Sentinel-1/Sentinel-2 etc.)."""
    state, health = _ambiguous_state_and_health(
        SCENARIO_TENNENT,
        TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
        TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
    )
    rec = rank_collection_candidates(state, health)
    for v in rec.ranked_values:
        blob = f"{v.candidate.candidate_id} {v.candidate.label} {v.candidate.description}"
        lower = blob.lower()
        for platform in ("sentinel-1", "sentinel-2", "iceye", "umbra", "planetscope", "maxar"):
            assert platform not in lower, (
                f"candidate {v.candidate.candidate_id!r} names platform {platform!r} "
                "in its public surface; keep candidate types sensor-generic"
            )
