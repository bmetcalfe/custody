"""
Tests for twilight-aware tasking behavior.

Covers:
  1. twilight + routine tier keeps optical
  2. twilight + urgent tier switches to SAR
  3. twilight + critical tier switches to SAR
  4. twilight + elevated tier keeps optical (degraded note)
  5. night behavior unchanged (still switches to SAR)
  6. day behavior unchanged (keeps optical)
  7. twilight rationale mentions marginal light for urgent
  8. twilight rationale mentions degraded for lower tiers
  9. sensor_confidence reduced in twilight with optical preference
 10. confidence action bias can shift to confirm under twilight
"""
from __future__ import annotations

import pytest

from custody.tasking_policy import (
    compute_policy,
    SENSOR_HIGH_RES, SENSOR_SAR, SENSOR_FAST,
    TIER_ROUTINE, TIER_ELEVATED, TIER_PRIORITY, TIER_URGENT, TIER_CRITICAL,
)
from custody.confidence import sensor_confidence, compute_overall_confidence, confidence_action_bias


def _rec(tier_state="normal", agreement="normal", custody=0.7,
         solar="day", eo_suit=1.0, sun_elev=45.0, duration=0, priority=0.3):
    return {
        "priority_score": priority,
        "anomaly_state": tier_state,
        "anomaly_agreement": agreement,
        "ml_anomaly_duration_hours": duration,
        "custody_confidence": custody,
        "solar_condition": solar,
        "eo_suitability": eo_suit,
        "sar_suitability": 1.0,
        "sun_elevation_deg": sun_elev,
    }


# ---------------------------------------------------------------------------
# Effective sensor preference under twilight
# ---------------------------------------------------------------------------

class TestTwilightSensorPreference:

    def test_twilight_routine_keeps_optical(self):
        """Routine tier under twilight: optical viable, not switched."""
        r = _rec(solar="twilight", eo_suit=0.6, sun_elev=2.0)
        p = compute_policy(r)
        # Routine → sensor_preference = "any", not optical-class
        # So effective stays "any" (no switch needed)
        assert p.effective_sensor_preference == p.sensor_preference

    def test_twilight_urgent_switches_to_sar(self):
        """Urgent tier with optical preference under twilight → SAR."""
        r = _rec(tier_state="sustained", agreement="confirmed",
                 duration=5, custody=0.7, priority=0.9,
                 solar="twilight", eo_suit=0.6, sun_elev=2.0)
        p = compute_policy(r)
        assert p.sensor_preference == SENSOR_HIGH_RES  # anomaly-driven wants EO
        assert p.effective_sensor_preference == SENSOR_SAR  # twilight switches it

    def test_twilight_critical_switches_to_sar(self):
        """Critical tier under twilight → SAR."""
        r = _rec(tier_state="critical", agreement="confirmed",
                 duration=5, custody=0.3, priority=0.9,
                 solar="twilight", eo_suit=0.6, sun_elev=2.0)
        p = compute_policy(r)
        # Critical + weak custody already selects SAR in sensor_preference
        assert p.effective_sensor_preference == SENSOR_SAR

    def test_twilight_elevated_keeps_optical(self):
        """Elevated tier under twilight: optical retained (not urgent enough to switch)."""
        r = _rec(tier_state="emerging", agreement="emerging",
                 custody=0.8, priority=0.6,
                 solar="twilight", eo_suit=0.6, sun_elev=2.0)
        p = compute_policy(r)
        assert p.sensor_preference == SENSOR_FAST  # emerging → fast_revisit
        assert p.effective_sensor_preference == SENSOR_FAST  # twilight keeps it


# ---------------------------------------------------------------------------
# Day and night unchanged
# ---------------------------------------------------------------------------

class TestDayNightUnchanged:

    def test_night_still_switches_to_sar(self):
        r = _rec(tier_state="confirmed", agreement="confirmed",
                 priority=0.8,
                 solar="night", eo_suit=0.0, sun_elev=-20.0)
        p = compute_policy(r)
        assert p.effective_sensor_preference == SENSOR_SAR

    def test_day_keeps_optical(self):
        r = _rec(tier_state="sustained", agreement="confirmed",
                 duration=5, custody=0.7, priority=0.9,
                 solar="day", eo_suit=1.0, sun_elev=45.0)
        p = compute_policy(r)
        assert p.sensor_preference == SENSOR_HIGH_RES
        assert p.effective_sensor_preference == SENSOR_HIGH_RES


# ---------------------------------------------------------------------------
# Rationale
# ---------------------------------------------------------------------------

class TestTwilightRationale:

    def test_urgent_twilight_mentions_marginal_light(self):
        r = _rec(tier_state="sustained", agreement="confirmed",
                 duration=5, priority=0.9,
                 solar="twilight", eo_suit=0.6, sun_elev=2.0)
        p = compute_policy(r)
        assert "marginal light" in p.sensor_rationale.lower()

    def test_elevated_twilight_mentions_degraded(self):
        r = _rec(tier_state="emerging", agreement="emerging",
                 priority=0.6,
                 solar="twilight", eo_suit=0.6, sun_elev=2.0)
        p = compute_policy(r)
        assert "degraded" in p.sensor_rationale.lower()


# ---------------------------------------------------------------------------
# Confidence integration
# ---------------------------------------------------------------------------

class TestTwilightConfidence:

    def test_sensor_confidence_reduced_in_twilight(self):
        """Optical preference under twilight should have reduced sensor_confidence."""
        r = _rec(solar="twilight", eo_suit=0.6)
        r["effective_sensor_preference"] = "fast_revisit"  # optical-class
        sc = sensor_confidence(r)
        assert sc < 1.0
        assert sc == pytest.approx(0.6, abs=0.05)

    def test_twilight_can_trigger_confirm_bias(self):
        """Under twilight with other weak factors, overall confidence may
        drop below threshold, shifting aggressive action to confirm."""
        r = _rec(solar="twilight", eo_suit=0.6, custody=0.4)
        r["effective_sensor_preference"] = "fast_revisit"
        # Sparse history and no baseline compound the low sensor confidence
        conf = compute_overall_confidence(r, [])  # empty history
        # With h_conf=0.2, b_conf=0.3, s_conf≈0.6, c_conf=0.4:
        # overall ≈ 0.35*0.2 + 0.30*0.3 + 0.20*0.6 + 0.15*0.4 = 0.07+0.09+0.12+0.06 = 0.34
        assert conf["overall_confidence"] < 0.5
        biased = confidence_action_bias("increase_attention", conf["overall_confidence"])
        assert biased == "confirm"
