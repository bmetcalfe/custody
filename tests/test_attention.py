"""
Tests for the selective custody attention model (orchestration/attention.py).

Covers:
- derive_attention_state for representative inputs
- apply_tracking_directive_floor (directive-as-floor semantics)
- compute_neglect_weight per tier
- explain_attention_state content
- Portfolio-level behavioral properties: background neglect does not pollute rankings,
  manually tracked vessel remains relevant, anomalous vessel outranks routine traffic
"""
from __future__ import annotations

import pytest
from datetime import datetime, timezone, timedelta

from custody.orchestration.attention import (
    derive_attention_state,
    apply_tracking_directive_floor,
    compute_neglect_weight,
    explain_attention_state,
    BACKGROUND,
    WATCHLIST,
    ACTIVE_CUSTODY,
    DIRECTIVE_NONE,
    DIRECTIVE_MAINTAIN_CUSTODY,
)

UTC = timezone.utc
_T0 = datetime(2026, 3, 23, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# derive_attention_state
# ---------------------------------------------------------------------------

class TestDeriveAttentionState:

    def test_routine_vessel_is_background(self):
        state = derive_attention_state(anomaly_score=0.1, custody_confidence=0.9, zone_score=0.0)
        assert state == BACKGROUND

    def test_mildly_anomalous_is_watchlist(self):
        state = derive_attention_state(anomaly_score=0.6, custody_confidence=0.9, zone_score=0.0)
        assert state == WATCHLIST

    def test_threshold_anomaly_watchlist(self):
        state = derive_attention_state(anomaly_score=0.5, custody_confidence=0.9, zone_score=0.0)
        assert state == WATCHLIST

    def test_high_anomaly_is_active_custody(self):
        state = derive_attention_state(anomaly_score=1.5, custody_confidence=0.9, zone_score=0.0)
        assert state == ACTIVE_CUSTODY

    def test_very_high_anomaly_is_active_custody(self):
        state = derive_attention_state(anomaly_score=2.5, custody_confidence=0.8, zone_score=0.0)
        assert state == ACTIVE_CUSTODY

    def test_zone_proximity_watchlist(self):
        state = derive_attention_state(anomaly_score=0.1, custody_confidence=0.9, zone_score=0.25)
        assert state == WATCHLIST

    def test_zone_presence_active_custody(self):
        state = derive_attention_state(anomaly_score=0.1, custody_confidence=0.9, zone_score=0.6)
        assert state == ACTIVE_CUSTODY

    def test_very_low_confidence_active_custody(self):
        state = derive_attention_state(anomaly_score=0.1, custody_confidence=0.30)
        assert state == ACTIVE_CUSTODY

    def test_degraded_confidence_watchlist(self):
        state = derive_attention_state(anomaly_score=0.1, custody_confidence=0.60)
        assert state == WATCHLIST

    def test_nominal_confidence_background(self):
        state = derive_attention_state(anomaly_score=0.1, custody_confidence=0.75)
        assert state == BACKGROUND

    def test_worst_signal_wins(self):
        """Low zone score + high anomaly → ACTIVE_CUSTODY (not WATCHLIST)."""
        state = derive_attention_state(anomaly_score=2.0, custody_confidence=0.9, zone_score=0.1)
        assert state == ACTIVE_CUSTODY

    def test_valid_return_values(self):
        valid = {BACKGROUND, WATCHLIST, ACTIVE_CUSTODY}
        for a, c, z in [(0.1, 0.9, 0.0), (0.6, 0.8, 0.1), (2.0, 0.5, 0.6)]:
            assert derive_attention_state(a, c, z) in valid


# ---------------------------------------------------------------------------
# apply_tracking_directive_floor
# ---------------------------------------------------------------------------

class TestApplyTrackingDirectiveFloor:

    def test_none_directive_no_change_background(self):
        assert apply_tracking_directive_floor(BACKGROUND, DIRECTIVE_NONE) == BACKGROUND

    def test_none_directive_no_change_watchlist(self):
        assert apply_tracking_directive_floor(WATCHLIST, DIRECTIVE_NONE) == WATCHLIST

    def test_none_directive_no_change_active(self):
        assert apply_tracking_directive_floor(ACTIVE_CUSTODY, DIRECTIVE_NONE) == ACTIVE_CUSTODY

    def test_maintain_custody_raises_background_to_active(self):
        result = apply_tracking_directive_floor(BACKGROUND, DIRECTIVE_MAINTAIN_CUSTODY)
        assert result == ACTIVE_CUSTODY

    def test_maintain_custody_raises_watchlist_to_active(self):
        result = apply_tracking_directive_floor(WATCHLIST, DIRECTIVE_MAINTAIN_CUSTODY)
        assert result == ACTIVE_CUSTODY

    def test_maintain_custody_keeps_active_custody(self):
        result = apply_tracking_directive_floor(ACTIVE_CUSTODY, DIRECTIVE_MAINTAIN_CUSTODY)
        assert result == ACTIVE_CUSTODY

    def test_result_never_lower_than_input(self):
        for tier in [BACKGROUND, WATCHLIST, ACTIVE_CUSTODY]:
            for directive in [DIRECTIVE_NONE, DIRECTIVE_MAINTAIN_CUSTODY]:
                result = apply_tracking_directive_floor(tier, directive)
                from custody.orchestration.attention import _TIER_ORDER
                assert _TIER_ORDER.index(result) >= _TIER_ORDER.index(tier)


# ---------------------------------------------------------------------------
# compute_neglect_weight
# ---------------------------------------------------------------------------

class TestComputeNeglectWeight:

    def test_background_zero_weight(self):
        assert compute_neglect_weight(BACKGROUND) == 0.0

    def test_watchlist_is_between_zero_and_one(self):
        w = compute_neglect_weight(WATCHLIST)
        assert 0.0 < w < 1.0

    def test_active_custody_full_weight(self):
        assert compute_neglect_weight(ACTIVE_CUSTODY) == 1.0

    def test_monotone_ordering(self):
        assert compute_neglect_weight(BACKGROUND) < compute_neglect_weight(WATCHLIST)
        assert compute_neglect_weight(WATCHLIST) < compute_neglect_weight(ACTIVE_CUSTODY)

    def test_unknown_tier_defaults_to_full_weight(self):
        """Unknown tier should default to 1.0 (conservative/safe)."""
        assert compute_neglect_weight("UNKNOWN_TIER") == 1.0


# ---------------------------------------------------------------------------
# explain_attention_state
# ---------------------------------------------------------------------------

class TestExplainAttentionState:

    def test_returns_nonempty_string(self):
        s = explain_attention_state(BACKGROUND, DIRECTIVE_NONE, 0.1, 0.0, 0.9)
        assert isinstance(s, str) and len(s) > 0

    def test_contains_tier_name(self):
        for tier in [BACKGROUND, WATCHLIST, ACTIVE_CUSTODY]:
            s = explain_attention_state(tier, DIRECTIVE_NONE, 0.1, 0.0, 0.9)
            assert tier in s

    def test_routine_behavior_mentioned_for_background(self):
        s = explain_attention_state(BACKGROUND, DIRECTIVE_NONE, 0.1, 0.0, 0.9)
        assert "routine" in s.lower()

    def test_directive_mentioned_when_maintain_custody(self):
        s = explain_attention_state(ACTIVE_CUSTODY, DIRECTIVE_MAINTAIN_CUSTODY, 0.1, 0.0, 0.9)
        assert "directive" in s.lower() or "MAINTAIN" in s

    def test_elevated_anomaly_mentioned(self):
        s = explain_attention_state(ACTIVE_CUSTODY, DIRECTIVE_NONE, 2.0, 0.0, 0.9)
        assert "anomaly" in s.lower()

    def test_zone_entry_mentioned(self):
        s = explain_attention_state(ACTIVE_CUSTODY, DIRECTIVE_NONE, 0.1, 0.6, 0.9)
        assert "zone" in s.lower()

    def test_low_confidence_mentioned(self):
        s = explain_attention_state(ACTIVE_CUSTODY, DIRECTIVE_NONE, 0.1, 0.0, 0.20)
        assert "confidence" in s.lower()


# ---------------------------------------------------------------------------
# Portfolio-level behavioral properties
# ---------------------------------------------------------------------------

def _record(
    eid: str,
    anomaly: float = 0.1,
    conf: float = 0.9,
    unc_km: float = 5.0,
    action: str = "NONE",
    hsc: float | None = 0.0,
    directive: str = "NONE",
    zone: float = 0.0,
) -> dict:
    return {
        "target_id":            eid,
        "anomaly_score":        anomaly,
        "custody_confidence":   conf,
        "uncertainty_km":       unc_km,
        "action":               action,
        "hours_since_collection": hsc,
        "tracking_directive":   directive,
        "sensitive_zone":       zone,
    }


class TestPortfolioAttentionBehavior:
    """
    Key behavioral properties of the attention-gated portfolio scorer.
    These tests validate that the system operates correctly at the portfolio level,
    not just at the attention module level.
    """

    def _rank(self, records, t=_T0, start=_T0):
        from custody.orchestration.portfolio import rank_portfolio
        return rank_portfolio(timestamp=t, timestep_records=records,
                              scenario_start=start)

    def test_background_vessel_is_background_tier(self):
        pa = self._rank([_record("BG", anomaly=0.1, conf=0.9)])
        assert pa.by_entity()["BG"].attention_state == BACKGROUND

    def test_watchlist_vessel_is_watchlist_tier(self):
        pa = self._rank([_record("W", anomaly=0.6, conf=0.9)])
        assert pa.by_entity()["W"].attention_state == WATCHLIST

    def test_high_anomaly_is_active_custody_tier(self):
        pa = self._rank([_record("A", anomaly=2.0, conf=0.9)])
        assert pa.by_entity()["A"].attention_state == ACTIVE_CUSTODY

    def test_maintain_custody_directive_forces_active_custody(self):
        pa = self._rank([_record("P", anomaly=0.1, conf=0.9, directive="MAINTAIN_CUSTODY")])
        item = pa.by_entity()["P"]
        assert item.attention_state == ACTIVE_CUSTODY
        assert item.tracking_directive == "MAINTAIN_CUSTODY"

    def test_attention_basis_nonempty(self):
        pa = self._rank([_record("X")])
        assert len(pa.by_entity()["X"].attention_basis) > 0

    def test_background_neglected_ranks_below_active_custody_nominal(self):
        """
        A BACKGROUND vessel with 48h of neglect should rank BELOW a MAINTAIN_CUSTODY
        vessel with nominal behavior (low anomaly, low neglect).
        """
        t = _T0 + timedelta(hours=48)
        bg_neglected = _record("BG",    anomaly=0.1, conf=0.9, hsc=None)   # 48h unobserved
        ac_nominal   = _record("PORT",  anomaly=0.1, conf=0.9, hsc=0.0,
                                directive="MAINTAIN_CUSTODY")

        pa = self._rank([bg_neglected, ac_nominal], t=t)
        by_e = pa.by_entity()

        assert by_e["BG"].attention_state   == BACKGROUND
        assert by_e["PORT"].attention_state == ACTIVE_CUSTODY
        # Active-custody nominal vessel should outrank (lower rank number) background-neglected
        assert by_e["PORT"].portfolio_rank < by_e["BG"].portfolio_rank

    def test_anomalous_vessel_outranks_manual_custody_nominal(self):
        """A genuinely anomalous vessel should still rank above a nominal manually-tracked one."""
        pa = self._rank([
            _record("BRAVO", anomaly=2.5, conf=0.8),
            _record("PORT",  anomaly=0.1, conf=0.9, directive="MAINTAIN_CUSTODY"),
        ])
        by_e = pa.by_entity()
        assert by_e["BRAVO"].portfolio_rank < by_e["PORT"].portfolio_rank

    def test_background_neglect_does_not_push_score_above_active_nominal(self):
        """
        Background vessel neglected 100h should score lower than
        an active-custody vessel with zero neglect and zero anomaly.
        """
        t = _T0 + timedelta(hours=100)
        bg_long_neglect = _record("BG",  anomaly=0.1, conf=0.9, hsc=None)
        ac_fresh        = _record("AC",  anomaly=0.1, conf=0.9, hsc=0.0,
                                  directive="MAINTAIN_CUSTODY")

        pa = self._rank([bg_long_neglect, ac_fresh], t=t)
        by_e = pa.by_entity()
        assert by_e["AC"].portfolio_score > by_e["BG"].portfolio_score

    def test_portfolio_score_in_unit_interval_for_all_tiers(self):
        records = [
            _record("BG",    anomaly=0.0, conf=1.0, hsc=200.0),             # extreme background
            _record("W",     anomaly=0.6, conf=0.8, hsc=5.0),               # watchlist
            _record("AC",    anomaly=2.5, conf=0.3, hsc=50.0,               # active, bad state
                    directive="MAINTAIN_CUSTODY"),
        ]
        t = _T0 + timedelta(hours=200)
        for item in self._rank(records, t=t).items:
            assert 0.0 <= item.portfolio_score <= 1.0, (
                f"{item.entity_id} score {item.portfolio_score} out of [0,1]"
            )

    def test_all_three_tiers_reachable_in_one_timestep(self):
        records = [
            _record("BG",  anomaly=0.1, conf=0.9),
            _record("W",   anomaly=0.6, conf=0.9),
            _record("AC",  anomaly=2.0, conf=0.9),
        ]
        pa = self._rank(records)
        states = {item.attention_state for item in pa.items}
        assert BACKGROUND    in states
        assert WATCHLIST     in states
        assert ACTIVE_CUSTODY in states
