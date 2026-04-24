"""Tests for :mod:`custody.hypotheses.custody_health` (ADR-0021 Slice 4).

Pin the cascade (LOST > STALE > AMBIGUOUS > HEALTHY > DEGRADED), the
canonicalization of ``ambiguity_pairs``, the driver-string rationale,
and the default-``as_of`` behaviour (uses ``state.timestamp``, not
wall-clock).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from custody.hypotheses.custody_health import (
    CustodyHealthStatus,
    HypothesisCustodyHealth,
    assess_custody_health,
)
from custody.hypotheses.registry import (
    SCENARIO_TENNENT,
    TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
    TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
    TENNENT_NO_MEANINGFUL_ACTIVITY,
    TENNENT_STATIONARY_SAR_SCATTER_OR_REEF_CLUTTER,
    TENNENT_TRANSIENT_VESSEL_ACTIVITY,
)
from custody.hypotheses.types import HypothesisEvidence, HypothesisState
from custody.hypotheses.update import update_state


T0 = datetime(2023, 7, 2, 12, 0, 0, tzinfo=timezone.utc)
T_RECENT = datetime(2023, 8, 13, 12, 0, 0, tzinfo=timezone.utc)


def _ev(
    evidence_id: str,
    supports: tuple[str, ...] = (),
    contradicts: tuple[str, ...] = (),
    *,
    confidence: float = 0.6,
    weight: float = 1.0,
    timestamp: datetime | None = T0,
) -> HypothesisEvidence:
    return HypothesisEvidence(
        evidence_id=evidence_id,
        source_ref=f"observation:{evidence_id}",
        source_kind="observation",
        scenario_id=SCENARIO_TENNENT,
        timestamp=timestamp,
        supports=supports,
        contradicts=contradicts,
        confidence=confidence,
        weight=weight,
        reason="synthetic",
    )


# ---------------------------------------------------------------------------
# LOST
# ---------------------------------------------------------------------------


def test_no_evidence_returns_lost() -> None:
    state = update_state(SCENARIO_TENNENT, evidence=())
    health = assess_custody_health(state)
    assert health.status is CustodyHealthStatus.LOST
    assert health.score == 0.0


def test_priors_only_state_returns_lost() -> None:
    """All scores at priors -> LOST even if state was somehow timestamped."""
    state = update_state(SCENARIO_TENNENT, evidence=())
    health = assess_custody_health(state)
    assert health.status is CustodyHealthStatus.LOST


def test_lost_drivers_mention_priors() -> None:
    state = update_state(SCENARIO_TENNENT, evidence=())
    health = assess_custody_health(state)
    assert health.drivers
    assert any("prior" in d.lower() or "no useful evidence" in d.lower()
               for d in health.drivers)


# ---------------------------------------------------------------------------
# HEALTHY
# ---------------------------------------------------------------------------


def test_clear_top_with_margin_is_healthy() -> None:
    state = update_state(
        SCENARIO_TENNENT,
        evidence=(
            _ev("ev1", supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,),
                confidence=0.8, weight=1.0, timestamp=T_RECENT),
        ),
    )
    health = assess_custody_health(state, as_of=T_RECENT)
    assert health.status is CustodyHealthStatus.HEALTHY
    assert health.score >= 0.70
    assert health.top_hypothesis == TENNENT_FIXED_RECLAMATION_OR_STRUCTURE
    assert health.top_two_margin is not None
    assert health.top_two_margin >= 0.25


def test_healthy_has_non_empty_drivers_with_threshold_rationale() -> None:
    state = update_state(
        SCENARIO_TENNENT,
        evidence=(
            _ev("ev1", supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,),
                confidence=0.85, timestamp=T_RECENT),
        ),
    )
    health = assess_custody_health(state, as_of=T_RECENT)
    assert health.status is CustodyHealthStatus.HEALTHY
    assert health.drivers
    joined = " ".join(health.drivers).lower()
    assert "0.65" in joined or "threshold" in joined
    assert "0.25" in joined or "margin" in joined


# ---------------------------------------------------------------------------
# AMBIGUOUS
# ---------------------------------------------------------------------------


def test_narrow_margin_is_ambiguous() -> None:
    state = update_state(
        SCENARIO_TENNENT,
        evidence=(
            _ev("ev1", supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,),
                confidence=0.40, timestamp=T_RECENT),
            _ev("ev2", supports=(TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,),
                confidence=0.35, timestamp=T_RECENT),
        ),
    )
    health = assess_custody_health(state, as_of=T_RECENT)
    assert health.status is CustodyHealthStatus.AMBIGUOUS
    # Top-two margin must be < ambiguity_margin (default 0.15)
    assert health.top_two_margin is not None and health.top_two_margin < 0.15


def test_multiple_saturated_hypotheses_is_ambiguous() -> None:
    state = update_state(
        SCENARIO_TENNENT,
        evidence=(
            _ev("ev1", supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,),
                confidence=1.0, timestamp=T_RECENT),
            _ev("ev2", supports=(TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,),
                confidence=1.0, timestamp=T_RECENT),
        ),
    )
    health = assess_custody_health(state, as_of=T_RECENT)
    assert health.status is CustodyHealthStatus.AMBIGUOUS
    saturated_driver = " ".join(health.drivers).lower()
    assert "saturated" in saturated_driver
    assert "2 hypotheses" in saturated_driver or "0.95" in saturated_driver


def test_ambiguous_score_in_expected_range() -> None:
    state = update_state(
        SCENARIO_TENNENT,
        evidence=(
            _ev("ev1", supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,),
                confidence=0.40, timestamp=T_RECENT),
            _ev("ev2", supports=(TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,),
                confidence=0.35, timestamp=T_RECENT),
        ),
    )
    health = assess_custody_health(state, as_of=T_RECENT)
    assert health.status is CustodyHealthStatus.AMBIGUOUS
    assert 0.35 <= health.score <= 0.60


# ---------------------------------------------------------------------------
# STALE
# ---------------------------------------------------------------------------


def test_stale_evidence_is_stale() -> None:
    state = update_state(
        SCENARIO_TENNENT,
        evidence=(
            _ev("ev1", supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,),
                confidence=0.8, timestamp=T0),
        ),
    )
    # 45 days after T0 — well past default stale_after_days=30.
    as_of = T0 + timedelta(days=45)
    health = assess_custody_health(state, as_of=as_of)
    assert health.status is CustodyHealthStatus.STALE
    assert health.score <= 0.25
    joined = " ".join(health.drivers).lower()
    assert "days" in joined
    assert "30" in joined


def test_stale_check_skipped_when_both_as_of_and_timestamp_none() -> None:
    """No as_of and no state.timestamp -> STALE cannot be evaluated."""
    ev_no_ts = _ev(
        "ev1", supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,),
        confidence=0.85, timestamp=None,
    )
    state = update_state(SCENARIO_TENNENT, evidence=(ev_no_ts,))
    assert state.timestamp is None
    health = assess_custody_health(state)  # as_of=None falls back to state.timestamp (also None)
    assert health.status is not CustodyHealthStatus.STALE


# ---------------------------------------------------------------------------
# DEGRADED
# ---------------------------------------------------------------------------


def test_weak_evidence_is_degraded() -> None:
    """Evidence present but below healthy thresholds and not ambiguous."""
    # Low-confidence single evidence item; top_score around 0.3 + 0.2 prior = 0.4.
    state = update_state(
        SCENARIO_TENNENT,
        evidence=(
            _ev("ev1", supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,),
                confidence=0.3, weight=1.0, timestamp=T_RECENT),
        ),
    )
    health = assess_custody_health(state, as_of=T_RECENT)
    assert health.status is CustodyHealthStatus.DEGRADED
    assert 0.40 <= health.score <= 0.65
    joined = " ".join(health.drivers).lower()
    assert "below" in joined


# ---------------------------------------------------------------------------
# Cascade precedence
# ---------------------------------------------------------------------------


def test_lost_wins_over_stale() -> None:
    """No evidence (priors-only) + ancient as_of still returns LOST."""
    state = update_state(SCENARIO_TENNENT, evidence=())
    health = assess_custody_health(
        state, as_of=T0 + timedelta(days=365), stale_after_days=30,
    )
    assert health.status is CustodyHealthStatus.LOST


def test_stale_wins_over_ambiguous() -> None:
    """Narrow margin + old evidence -> STALE beats AMBIGUOUS."""
    state = update_state(
        SCENARIO_TENNENT,
        evidence=(
            _ev("ev1", supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,),
                confidence=0.4, timestamp=T0),
            _ev("ev2", supports=(TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,),
                confidence=0.35, timestamp=T0),
        ),
    )
    as_of = T0 + timedelta(days=45)
    health = assess_custody_health(state, as_of=as_of)
    assert health.status is CustodyHealthStatus.STALE


# ---------------------------------------------------------------------------
# Ambiguity pairs
# ---------------------------------------------------------------------------


def test_margin_ambiguity_includes_top_two_pair_canonicalized() -> None:
    state = update_state(
        SCENARIO_TENNENT,
        evidence=(
            _ev("ev1", supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,),
                confidence=0.40, timestamp=T_RECENT),
            _ev("ev2", supports=(TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,),
                confidence=0.35, timestamp=T_RECENT),
        ),
    )
    health = assess_custody_health(state, as_of=T_RECENT)
    assert health.status is CustodyHealthStatus.AMBIGUOUS
    assert len(health.ambiguity_pairs) >= 1
    a, b = sorted((
        TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
        TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
    ))
    assert (a, b) in health.ambiguity_pairs


def test_saturation_ambiguity_includes_all_combinations_canonicalized_and_sorted() -> None:
    state = update_state(
        SCENARIO_TENNENT,
        evidence=(
            _ev("ev1", supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,),
                confidence=1.0, timestamp=T_RECENT),
            _ev("ev2", supports=(TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,),
                confidence=1.0, timestamp=T_RECENT),
            _ev("ev3", supports=(TENNENT_TRANSIENT_VESSEL_ACTIVITY,),
                confidence=1.0, timestamp=T_RECENT),
        ),
    )
    health = assess_custody_health(state, as_of=T_RECENT)
    assert health.status is CustodyHealthStatus.AMBIGUOUS

    # Three saturated hypotheses -> C(3,2) = 3 pairs.
    assert len(health.ambiguity_pairs) == 3
    # All pairs must be canonical (a < b lexicographically).
    for a, b in health.ambiguity_pairs:
        assert a < b
    # The tuple must be sorted ascending.
    assert list(health.ambiguity_pairs) == sorted(health.ambiguity_pairs)


def test_ambiguity_pairs_empty_when_not_ambiguous() -> None:
    state = update_state(
        SCENARIO_TENNENT,
        evidence=(
            _ev("ev1", supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,),
                confidence=0.85, timestamp=T_RECENT),
        ),
    )
    health = assess_custody_health(state, as_of=T_RECENT)
    assert health.status is CustodyHealthStatus.HEALTHY
    assert health.ambiguity_pairs == ()


# ---------------------------------------------------------------------------
# Drivers non-empty for every status + threshold rationale
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", list(CustodyHealthStatus))
def test_drivers_non_empty_for_every_status(status: CustodyHealthStatus) -> None:
    """Construct a state that lands in each status, confirm drivers are populated."""
    if status is CustodyHealthStatus.LOST:
        state = update_state(SCENARIO_TENNENT, evidence=())
        health = assess_custody_health(state)
    elif status is CustodyHealthStatus.STALE:
        state = update_state(
            SCENARIO_TENNENT,
            evidence=(_ev("ev1", supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,),
                          confidence=0.8, timestamp=T0),),
        )
        health = assess_custody_health(state, as_of=T0 + timedelta(days=60))
    elif status is CustodyHealthStatus.AMBIGUOUS:
        state = update_state(
            SCENARIO_TENNENT,
            evidence=(
                _ev("ev1", supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,),
                    confidence=0.40, timestamp=T_RECENT),
                _ev("ev2", supports=(TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,),
                    confidence=0.35, timestamp=T_RECENT),
            ),
        )
        health = assess_custody_health(state, as_of=T_RECENT)
    elif status is CustodyHealthStatus.HEALTHY:
        state = update_state(
            SCENARIO_TENNENT,
            evidence=(_ev("ev1", supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,),
                          confidence=0.85, timestamp=T_RECENT),),
        )
        health = assess_custody_health(state, as_of=T_RECENT)
    elif status is CustodyHealthStatus.DEGRADED:
        state = update_state(
            SCENARIO_TENNENT,
            evidence=(_ev("ev1", supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,),
                          confidence=0.3, timestamp=T_RECENT),),
        )
        health = assess_custody_health(state, as_of=T_RECENT)
    else:  # pragma: no cover - enum exhaustiveness
        pytest.skip(f"no fixture for {status}")
    assert health.status is status
    assert health.drivers, f"drivers must be non-empty for status {status}"


# ---------------------------------------------------------------------------
# Score bounds
# ---------------------------------------------------------------------------


def test_score_always_between_zero_and_one() -> None:
    # Two scenarios across the status range, assert clamping.
    scenarios = [
        update_state(SCENARIO_TENNENT, evidence=()),
        update_state(SCENARIO_TENNENT, evidence=(
            _ev("ev1", supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,),
                confidence=1.0, timestamp=T_RECENT),
        )),
        update_state(SCENARIO_TENNENT, evidence=(
            _ev("ev1", supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,),
                confidence=0.40, timestamp=T_RECENT),
            _ev("ev2", supports=(TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,),
                confidence=0.35, timestamp=T_RECENT),
        )),
    ]
    for st in scenarios:
        h = assess_custody_health(st, as_of=T_RECENT)
        assert 0.0 <= h.score <= 1.0


# ---------------------------------------------------------------------------
# Default-as_of uses state.timestamp, not wall-clock
# ---------------------------------------------------------------------------


def test_default_as_of_uses_state_timestamp_not_wall_clock() -> None:
    """A state whose evidence is from 2023 must NOT be STALE in 2026 when as_of=None."""
    state = update_state(
        SCENARIO_TENNENT,
        evidence=(
            _ev("ev1", supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,),
                confidence=0.85, timestamp=T0),  # T0 is 2023-07-02
        ),
    )
    # No as_of -> default to state.timestamp == T0; comparing T0 to T0 is not stale.
    health = assess_custody_health(state)
    assert health.status is not CustodyHealthStatus.STALE


# ---------------------------------------------------------------------------
# Return type + latest_evidence_at plumbing
# ---------------------------------------------------------------------------


def test_returns_hypothesis_custody_health_with_latest_evidence_at() -> None:
    state = update_state(
        SCENARIO_TENNENT,
        evidence=(_ev("ev1", supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,),
                      confidence=0.8, timestamp=T_RECENT),),
    )
    health = assess_custody_health(state, as_of=T_RECENT)
    assert isinstance(health, HypothesisCustodyHealth)
    # latest_evidence_at is read from state.timestamp, per the dispatch.
    assert health.latest_evidence_at == T_RECENT
    assert health.top_hypothesis == TENNENT_FIXED_RECLAMATION_OR_STRUCTURE
    assert health.top_score > 0.0
