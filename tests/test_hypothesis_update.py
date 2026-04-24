"""Tests for :mod:`custody.hypotheses.update` (ADR-0021).

Covers deterministic weighted additive belief update, tie-break ordering,
uncertainty and ambiguity-boost behaviour, prior-state seeding, and the
per-evidence explanation trace.  Scoring is intentionally simple for the
Slice 1 contract — Bayesian rigour is not yet required.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from custody.hypotheses.registry import (
    SCENARIO_TENNENT,
    SCENARIO_WHITSUN,
    TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
    TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
    TENNENT_NO_MEANINGFUL_ACTIVITY,
    TENNENT_STATIONARY_SAR_SCATTER_OR_REEF_CLUTTER,
    TENNENT_TRANSIENT_VESSEL_ACTIVITY,
    WHITSUN_VESSEL_CLUSTER_ACTIVITY,
    get_hypotheses,
)
from custody.hypotheses.types import HypothesisEvidence, HypothesisState
from custody.hypotheses.update import update_state


T0 = datetime(2023, 7, 2, 12, 0, 0, tzinfo=timezone.utc)
T1 = datetime(2023, 7, 3, 12, 0, 0, tzinfo=timezone.utc)
T2 = datetime(2023, 8, 7, 12, 0, 0, tzinfo=timezone.utc)


def _ev(
    evidence_id: str,
    supports: tuple[str, ...] = (),
    contradicts: tuple[str, ...] = (),
    *,
    confidence: float = 0.6,
    weight: float = 1.0,
    timestamp=T0,
    scenario_id: str = SCENARIO_TENNENT,
    reason: str = "synthetic",
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
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Seeding from priors
# ---------------------------------------------------------------------------


def test_empty_evidence_returns_prior_seeded_state() -> None:
    state = update_state(SCENARIO_TENNENT, evidence=())
    assert isinstance(state, HypothesisState)
    assert state.scenario_id == SCENARIO_TENNENT
    assert state.timestamp is None
    hs = get_hypotheses(SCENARIO_TENNENT)
    for h in hs:
        assert state.scores[h.hypothesis_id] == pytest.approx(h.prior)


def test_update_picks_top_from_priors_with_deterministic_tie_break() -> None:
    """All priors equal → top is the alphabetically-first hypothesis_id."""
    state = update_state(SCENARIO_TENNENT, evidence=())
    ids = sorted(h.hypothesis_id for h in get_hypotheses(SCENARIO_TENNENT))
    assert state.top_hypothesis == ids[0]


# ---------------------------------------------------------------------------
# Basic support/contradict scoring
# ---------------------------------------------------------------------------


def test_support_increases_score() -> None:
    base = update_state(SCENARIO_TENNENT, evidence=())
    base_score = base.scores[TENNENT_FIXED_RECLAMATION_OR_STRUCTURE]

    state = update_state(
        SCENARIO_TENNENT,
        evidence=(_ev("ev1", supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,),
                      confidence=0.8, weight=1.0),),
    )
    assert state.scores[TENNENT_FIXED_RECLAMATION_OR_STRUCTURE] > base_score
    assert state.scores[TENNENT_FIXED_RECLAMATION_OR_STRUCTURE] == pytest.approx(
        min(1.0, base_score + 0.8)
    )


def test_contradict_decreases_score_and_floors_at_zero() -> None:
    state = update_state(
        SCENARIO_TENNENT,
        evidence=(
            _ev("ev1", contradicts=(TENNENT_NO_MEANINGFUL_ACTIVITY,),
                confidence=1.0, weight=1.0),
        ),
    )
    assert state.scores[TENNENT_NO_MEANINGFUL_ACTIVITY] == pytest.approx(0.0)


def test_score_clamped_to_one() -> None:
    state = update_state(
        SCENARIO_TENNENT,
        evidence=(
            _ev("ev1", supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,),
                confidence=1.0, weight=5.0),
        ),
    )
    assert state.scores[TENNENT_FIXED_RECLAMATION_OR_STRUCTURE] <= 1.0


def test_top_hypothesis_reflects_dominant_support() -> None:
    state = update_state(
        SCENARIO_TENNENT,
        evidence=(
            _ev("ev1", supports=(TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,),
                confidence=0.9, weight=1.0),
            _ev("ev2", supports=(TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,),
                confidence=0.5, weight=1.0),
        ),
    )
    assert state.top_hypothesis == TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY


# ---------------------------------------------------------------------------
# Uncertainty and ambiguity boost
# ---------------------------------------------------------------------------


def test_uncertainty_is_one_minus_top_score_when_margin_is_wide() -> None:
    state = update_state(
        SCENARIO_TENNENT,
        evidence=(
            _ev("ev1", supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,),
                confidence=0.6, weight=1.0),
        ),
    )
    top_score = state.scores[state.top_hypothesis]
    second = sorted(state.scores.values(), reverse=True)[1]
    margin = top_score - second
    if margin >= 0.1:
        assert state.uncertainty == pytest.approx(1.0 - top_score)


def test_ambiguity_boost_applies_when_top_two_are_close() -> None:
    """Margin < 0.1 → uncertainty += 0.1 (clamped to 1)."""
    state = update_state(
        SCENARIO_TENNENT,
        evidence=(
            _ev("ev1", supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,),
                confidence=0.05, weight=1.0),
            _ev("ev2", supports=(TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,),
                confidence=0.03, weight=1.0),
        ),
    )
    top_score = state.scores[state.top_hypothesis]
    second = sorted(state.scores.values(), reverse=True)[1]
    assert (top_score - second) < 0.1
    assert state.uncertainty == pytest.approx(min(1.0, (1.0 - top_score) + 0.1))


# ---------------------------------------------------------------------------
# Tie-break determinism
# ---------------------------------------------------------------------------


def test_tie_break_is_deterministic_on_hypothesis_id_sort_order() -> None:
    """Two hypotheses end with identical scores → earliest sorted id wins."""
    state = update_state(
        SCENARIO_TENNENT,
        evidence=(
            _ev("ev1", supports=(TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,),
                confidence=0.1, weight=1.0),
            _ev("ev2", supports=(TENNENT_TRANSIENT_VESSEL_ACTIVITY,),
                confidence=0.1, weight=1.0),
        ),
    )
    a = state.scores[TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY]
    b = state.scores[TENNENT_TRANSIENT_VESSEL_ACTIVITY]
    assert a == pytest.approx(b)
    assert state.top_hypothesis == sorted(
        [TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
         TENNENT_TRANSIENT_VESSEL_ACTIVITY]
    )[0]


def test_repeated_update_is_deterministic() -> None:
    evs = (
        _ev("ev1", supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,), confidence=0.4),
        _ev("ev2", supports=(TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,), confidence=0.3),
        _ev("ev3", contradicts=(TENNENT_NO_MEANINGFUL_ACTIVITY,), confidence=0.5),
    )
    s1 = update_state(SCENARIO_TENNENT, evidence=evs)
    s2 = update_state(SCENARIO_TENNENT, evidence=evs)
    assert s1.scores == s2.scores
    assert s1.top_hypothesis == s2.top_hypothesis
    assert s1.uncertainty == s2.uncertainty
    assert s1.explanation == s2.explanation


# ---------------------------------------------------------------------------
# Prior-state seeding
# ---------------------------------------------------------------------------


def test_prior_state_seeds_running_belief() -> None:
    first = update_state(
        SCENARIO_TENNENT,
        evidence=(
            _ev("ev1", supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,),
                confidence=0.4, weight=1.0, timestamp=T0),
        ),
    )
    second = update_state(
        SCENARIO_TENNENT,
        evidence=(
            _ev("ev2", supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,),
                confidence=0.3, weight=1.0, timestamp=T1),
        ),
        prior_state=first,
    )
    # Score should accumulate from the running belief
    assert (
        second.scores[TENNENT_FIXED_RECLAMATION_OR_STRUCTURE]
        > first.scores[TENNENT_FIXED_RECLAMATION_OR_STRUCTURE]
    )


# ---------------------------------------------------------------------------
# Timestamp propagation
# ---------------------------------------------------------------------------


def test_state_timestamp_is_max_of_evidence_timestamps() -> None:
    state = update_state(
        SCENARIO_TENNENT,
        evidence=(
            _ev("ev1", supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,), timestamp=T0),
            _ev("ev2", supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,), timestamp=T2),
            _ev("ev3", supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,), timestamp=T1),
        ),
    )
    assert state.timestamp == T2


def test_state_timestamp_is_none_when_all_evidence_timestamps_are_none() -> None:
    state = update_state(
        SCENARIO_TENNENT,
        evidence=(
            _ev("ev1", supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,), timestamp=None),
        ),
    )
    assert state.timestamp is None


# ---------------------------------------------------------------------------
# Explanation trace
# ---------------------------------------------------------------------------


def test_explanation_has_one_line_per_evidence() -> None:
    evs = (
        _ev("ev1", supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,), confidence=0.4),
        _ev("ev2", contradicts=(TENNENT_NO_MEANINGFUL_ACTIVITY,), confidence=0.2),
    )
    state = update_state(SCENARIO_TENNENT, evidence=evs)
    assert len(state.explanation) == 2
    assert "ev1" in state.explanation[0]
    assert "ev2" in state.explanation[1]


def test_explanation_describes_sign_and_target() -> None:
    state = update_state(
        SCENARIO_TENNENT,
        evidence=(
            _ev("ev1", supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,),
                confidence=0.5, weight=1.0, reason="bright scatterer"),
            _ev("ev2", contradicts=(TENNENT_NO_MEANINGFUL_ACTIVITY,),
                confidence=0.3, weight=1.0, reason="scene shows activity"),
        ),
    )
    # First line is a support (+) for fixed_reclamation_or_structure
    assert "+0.50" in state.explanation[0] or "+0.5" in state.explanation[0]
    assert TENNENT_FIXED_RECLAMATION_OR_STRUCTURE in state.explanation[0]
    assert "bright scatterer" in state.explanation[0]
    # Second line is a contradict (-) for no_meaningful_activity
    assert "-0.30" in state.explanation[1] or "-0.3" in state.explanation[1]
    assert TENNENT_NO_MEANINGFUL_ACTIVITY in state.explanation[1]


# ---------------------------------------------------------------------------
# Supporting/contradicting evidence grouping
# ---------------------------------------------------------------------------


def test_supporting_and_contradicting_evidence_is_grouped() -> None:
    evs = (
        _ev("ev1", supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,)),
        _ev("ev2", supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,)),
        _ev("ev3", contradicts=(TENNENT_NO_MEANINGFUL_ACTIVITY,)),
    )
    state = update_state(SCENARIO_TENNENT, evidence=evs)

    sup = state.supporting_evidence[TENNENT_FIXED_RECLAMATION_OR_STRUCTURE]
    assert {e.evidence_id for e in sup} == {"ev1", "ev2"}

    con = state.contradicting_evidence[TENNENT_NO_MEANINGFUL_ACTIVITY]
    assert {e.evidence_id for e in con} == {"ev3"}


# ---------------------------------------------------------------------------
# Cross-scenario isolation
# ---------------------------------------------------------------------------


def test_whitsun_scenario_produces_whitsun_scores_only() -> None:
    state = update_state(
        SCENARIO_WHITSUN,
        evidence=(
            _ev("ev1", supports=(WHITSUN_VESSEL_CLUSTER_ACTIVITY,),
                scenario_id=SCENARIO_WHITSUN, confidence=0.5),
        ),
    )
    whitsun_ids = {h.hypothesis_id for h in get_hypotheses(SCENARIO_WHITSUN)}
    assert set(state.scores) == whitsun_ids


def test_unknown_scenario_raises_value_error() -> None:
    with pytest.raises(ValueError):
        update_state("atlantis", evidence=())
