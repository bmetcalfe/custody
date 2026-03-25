"""
Tests for dark-vessel / AIS-dropout functionality.

Covers:
  - dark_vessel module: mark_dark, build_dark_reason, is_dark_relevant
  - attention.py: apply_dark_vessel_floor
  - Simulation integration: DARK_VESSEL_SMOKE scenario
"""
from __future__ import annotations

import pytest
from datetime import datetime, timezone

from custody.models import TrackState, Vessel
from custody.dark_vessel import mark_dark, build_dark_reason, is_dark_relevant
from custody.orchestration.attention import (
    apply_dark_vessel_floor,
    BACKGROUND, WATCHLIST, ACTIVE_CUSTODY,
    DIRECTIVE_NONE, DIRECTIVE_MAINTAIN_CUSTODY,
    explain_attention_state,
)
from custody.simulation import run_multi_target_simulation, DARK_VESSEL_SMOKE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_T0 = datetime(2026, 3, 23, 0, 0, tzinfo=timezone.utc)


def _vessel(lat: float = 1.0, lon: float = 0.5) -> Vessel:
    return Vessel(
        id="TEST",
        lat=lat,
        lon=lon,
        speed_kmh=5.0,
        heading_deg=90.0,
        last_seen=_T0,
    )


def _track() -> TrackState:
    return TrackState()


# ---------------------------------------------------------------------------
# Unit: mark_dark
# ---------------------------------------------------------------------------

class TestMarkDark:
    def test_sets_is_dark(self):
        track = _track()
        v = _vessel()
        mark_dark(track, v, _T0, anomaly_score=0.4)
        assert track.is_dark is True

    def test_freezes_position(self):
        track = _track()
        v = _vessel(lat=2.0, lon=3.0)
        mark_dark(track, v, _T0, anomaly_score=0.7)
        assert track.last_known_lat == pytest.approx(2.0)
        assert track.last_known_lon == pytest.approx(3.0)

    def test_freezes_timestamp(self):
        track = _track()
        v = _vessel()
        mark_dark(track, v, _T0, anomaly_score=0.3)
        assert track.dark_since == _T0
        assert track.last_known_time == _T0

    def test_freezes_anomaly_score(self):
        track = _track()
        v = _vessel()
        mark_dark(track, v, _T0, anomaly_score=1.2)
        assert track.last_known_anomaly == pytest.approx(1.2)

    def test_track_initially_not_dark(self):
        track = _track()
        assert track.is_dark is False
        assert track.dark_since is None
        assert track.last_known_lat is None
        assert track.last_known_lon is None


# ---------------------------------------------------------------------------
# Unit: build_dark_reason
# ---------------------------------------------------------------------------

class TestBuildDarkReason:
    def test_immediate_dropout(self):
        reason = build_dark_reason(_T0, _T0)
        assert "AIS" in reason
        assert "last known" in reason.lower()

    def test_elapsed_time_shown(self):
        from datetime import timedelta
        t1 = _T0 + timedelta(hours=5)
        reason = build_dark_reason(_T0, t1)
        assert "5.0" in reason

    def test_returns_string(self):
        from datetime import timedelta
        assert isinstance(build_dark_reason(_T0, _T0 + timedelta(hours=2)), str)


# ---------------------------------------------------------------------------
# Unit: is_dark_relevant
# ---------------------------------------------------------------------------

class TestIsDarkRelevant:
    def test_maintain_custody_always_relevant(self):
        for tier in (BACKGROUND, WATCHLIST, ACTIVE_CUSTODY):
            assert is_dark_relevant(tier, DIRECTIVE_MAINTAIN_CUSTODY) is True

    def test_watchlist_relevant_without_directive(self):
        assert is_dark_relevant(WATCHLIST, DIRECTIVE_NONE) is True

    def test_active_custody_relevant_without_directive(self):
        assert is_dark_relevant(ACTIVE_CUSTODY, DIRECTIVE_NONE) is True

    def test_background_not_relevant_without_directive(self):
        assert is_dark_relevant(BACKGROUND, DIRECTIVE_NONE) is False


# ---------------------------------------------------------------------------
# Unit: apply_dark_vessel_floor
# ---------------------------------------------------------------------------

class TestApplyDarkVesselFloor:
    def test_no_effect_when_not_dark(self):
        assert apply_dark_vessel_floor(BACKGROUND, False, DIRECTIVE_NONE) == BACKGROUND
        assert apply_dark_vessel_floor(WATCHLIST, False, DIRECTIVE_MAINTAIN_CUSTODY) == WATCHLIST

    def test_background_dark_no_directive_unchanged(self):
        result = apply_dark_vessel_floor(BACKGROUND, True, DIRECTIVE_NONE)
        assert result == BACKGROUND

    def test_watchlist_dark_no_directive_raises_to_active(self):
        result = apply_dark_vessel_floor(WATCHLIST, True, DIRECTIVE_NONE)
        assert result == ACTIVE_CUSTODY

    def test_active_custody_dark_stays_active(self):
        result = apply_dark_vessel_floor(ACTIVE_CUSTODY, True, DIRECTIVE_NONE)
        assert result == ACTIVE_CUSTODY

    def test_background_dark_with_directive_raises_to_active(self):
        result = apply_dark_vessel_floor(BACKGROUND, True, DIRECTIVE_MAINTAIN_CUSTODY)
        assert result == ACTIVE_CUSTODY


# ---------------------------------------------------------------------------
# Unit: explain_attention_state with dark_flag
# ---------------------------------------------------------------------------

class TestExplainAttentionStateDark:
    def test_dark_flag_appears_in_explanation(self):
        basis = explain_attention_state(
            ACTIVE_CUSTODY, DIRECTIVE_MAINTAIN_CUSTODY,
            anomaly_score=0.2, zone_score=0.0, custody_confidence=0.9,
            dark_flag=True,
        )
        assert "AIS dark" in basis or "transponder" in basis.lower()

    def test_no_dark_mention_when_not_dark(self):
        basis = explain_attention_state(
            BACKGROUND, DIRECTIVE_NONE,
            anomaly_score=0.1, zone_score=0.0, custody_confidence=0.95,
            dark_flag=False,
        )
        assert "dark" not in basis.lower()


# ---------------------------------------------------------------------------
# Integration: DARK_VESSEL_SMOKE scenario
# ---------------------------------------------------------------------------

class TestDarkVesselSmokeIntegration:
    @pytest.fixture(scope="class")
    def records(self):
        return run_multi_target_simulation(DARK_VESSEL_SMOKE)

    @pytest.fixture(scope="class")
    def dark1_records(self, records):
        return [r for r in records if r["target_id"] == "DARK-1"]

    # ── Presence ──────────────────────────────────────────────────────────

    def test_dark1_present(self, dark1_records):
        assert len(dark1_records) > 0

    def test_expected_timestep_count(self, dark1_records):
        # 8-hour scenario with dt=1: hours 0..8 inclusive = 9 timesteps
        assert len(dark1_records) == 9

    def test_dark_vessel_flag_field_always_present(self, dark1_records):
        assert all("dark_vessel_flag" in r for r in dark1_records)

    def test_dark_vessel_reason_field_always_present(self, dark1_records):
        assert all("dark_vessel_reason" in r for r in dark1_records)

    def test_dark_since_field_present(self, dark1_records):
        assert all("dark_since" in r for r in dark1_records)

    def test_last_known_lat_lon_fields_present(self, dark1_records):
        assert all("last_known_lat" in r for r in dark1_records)
        assert all("last_known_lon" in r for r in dark1_records)

    # ── Pre-dropout behaviour (hours 0–2) ─────────────────────────────────

    def test_not_dark_before_dropout(self, dark1_records):
        pre = [r for r in dark1_records if r["time"].hour < 3]
        assert all(r["dark_vessel_flag"] is False for r in pre)

    def test_lat_lon_non_null_before_dropout(self, dark1_records):
        pre = [r for r in dark1_records if r["time"].hour < 3]
        assert all(r["lat"] is not None for r in pre)
        assert all(r["lon"] is not None for r in pre)

    def test_dark_since_none_before_dropout(self, dark1_records):
        pre = [r for r in dark1_records if r["time"].hour < 3]
        assert all(r["dark_since"] is None for r in pre)

    # ── Dropout onset (hour 3) ─────────────────────────────────────────────

    def test_dark_flag_set_at_dropout_hour(self, dark1_records):
        dropout_record = next(r for r in dark1_records if r["time"].hour == 3)
        assert dropout_record["dark_vessel_flag"] is True

    def test_dark_since_set_at_dropout_hour(self, dark1_records):
        dropout_record = next(r for r in dark1_records if r["time"].hour == 3)
        assert dropout_record["dark_since"] is not None
        assert dropout_record["dark_since"].hour == 3

    def test_last_known_position_set_at_dropout(self, dark1_records):
        dropout_record = next(r for r in dark1_records if r["time"].hour == 3)
        assert dropout_record["last_known_lat"] is not None
        assert dropout_record["last_known_lon"] is not None

    # ── Post-dropout behaviour (hours 4–8) ────────────────────────────────

    def test_dark_flag_persists_after_dropout(self, dark1_records):
        post = [r for r in dark1_records if r["time"].hour > 3]
        assert all(r["dark_vessel_flag"] is True for r in post)

    def test_position_frozen_after_dropout(self, dark1_records):
        """All post-dropout records must report the same (frozen) lat/lon."""
        post = [r for r in dark1_records if r["time"].hour >= 3]
        lats = {r["lat"] for r in post}
        lons = {r["lon"] for r in post}
        assert len(lats) == 1, "lat should be frozen after dropout"
        assert len(lons) == 1, "lon should be frozen after dropout"

    def test_anomaly_score_frozen_after_dropout(self, dark1_records):
        """Post-dropout anomaly scores must all equal the frozen value."""
        post = [r for r in dark1_records if r["time"].hour >= 3]
        scores = {round(r["anomaly_score"], 6) for r in post}
        assert len(scores) == 1, "anomaly_score should be frozen after dropout"

    def test_dark_vessel_reason_is_string_post_dropout(self, dark1_records):
        post = [r for r in dark1_records if r["time"].hour > 3]
        assert all(isinstance(r["dark_vessel_reason"], str) for r in post)

    def test_reason_mentions_elapsed_time_post_dropout(self, dark1_records):
        # At hour 5 (2h after dropout), reason should mention elapsed time
        h5 = next((r for r in dark1_records if r["time"].hour == 5), None)
        if h5 is not None:
            assert "2.0" in h5["dark_vessel_reason"] or "h" in h5["dark_vessel_reason"]

    # ── Uncertainty growth ─────────────────────────────────────────────────

    def test_uncertainty_grows_after_dropout(self, dark1_records):
        """Positional uncertainty must grow even when AIS is dark."""
        unc_at_3 = next(r for r in dark1_records if r["time"].hour == 3)["uncertainty_km"]
        unc_at_8 = next(r for r in dark1_records if r["time"].hour == 8)["uncertainty_km"]
        assert unc_at_8 > unc_at_3

    # ── Portfolio / attention ──────────────────────────────────────────────

    def test_portfolio_fields_present(self, dark1_records):
        for r in dark1_records:
            assert "portfolio_rank" in r
            assert "attention_state" in r

    def test_attention_state_active_custody_post_dropout(self, dark1_records):
        """DARK-1 has MAINTAIN_CUSTODY; dark floor must push to ACTIVE_CUSTODY."""
        post = [r for r in dark1_records if r["time"].hour > 3]
        assert all(r["attention_state"] == ACTIVE_CUSTODY for r in post)

    # ── Background vessels unaffected ─────────────────────────────────────

    def test_background_vessels_not_dark(self, records):
        bg_records = [r for r in records if r["target_id"] != "DARK-1"]
        assert all(r["dark_vessel_flag"] is False for r in bg_records)

    def test_background_vessels_last_known_none(self, records):
        bg_records = [r for r in records if r["target_id"] != "DARK-1"]
        assert all(r["last_known_lat"] is None for r in bg_records)
        assert all(r["last_known_lon"] is None for r in bg_records)
