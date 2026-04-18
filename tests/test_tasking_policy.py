"""
Tests for custody.tasking_policy — adaptive tasking from anomaly reasoning.

Covers:
  1.  critical anomaly gets CRITICAL tier
  2.  sustained anomaly gets URGENT tier
  3.  confirmed anomaly gets PRIORITY tier
  4.  emerging anomaly gets ELEVATED tier
  5.  normal state gets ROUTINE tier
  6.  weak custody raises tier by one step
  7.  recovering triggers COOLDOWN action
  8.  critical triggers INTENSIFY action
  9.  confirmed triggers CONFIRM action
 10.  normal triggers MAINTAIN action
 11.  critical tier has shortest revisit
 12.  routine tier has longest revisit
 13.  long duration reduces revisit further
 14.  sensor preference: urgent + weak custody → all_weather (SAR)
 15.  sensor preference: emerging → fast_revisit
 16.  rationale is non-empty and mentions state
 17.  policy is a frozen dataclass
 18.  simulation records carry policy fields
"""
from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import FrozenInstanceError

import pytest

from custody.tasking_policy import (
    TaskingPolicy,
    compute_policy,
    TIER_ROUTINE, TIER_ELEVATED, TIER_PRIORITY, TIER_URGENT, TIER_CRITICAL,
    ACTION_MAINTAIN, ACTION_INCREASE_ATTENTION, ACTION_CONFIRM,
    ACTION_INTENSIFY, ACTION_COOLDOWN,
    SENSOR_ANY, SENSOR_HIGH_RES, SENSOR_SAR, SENSOR_FAST,
)

UTC = timezone.utc
_T0 = datetime(2026, 3, 23, 10, 0, tzinfo=UTC)


def _rec(priority=0.3, state="normal", agreement="normal",
         duration=0, custody=0.8):
    return {
        "priority_score": priority,
        "anomaly_state": state,
        "anomaly_agreement": agreement,
        "ml_anomaly_duration_hours": duration,
        "custody_confidence": custody,
    }


# ---------------------------------------------------------------------------
# Tier classification
# ---------------------------------------------------------------------------

class TestTierClassification:

    def test_critical(self):
        p = compute_policy(_rec(state="critical"))
        assert p.tasking_tier == TIER_CRITICAL

    def test_sustained(self):
        p = compute_policy(_rec(state="sustained"))
        assert p.tasking_tier == TIER_URGENT

    def test_confirmed_short_is_elevated(self):
        """Confirmed without duration stays elevated."""
        p = compute_policy(_rec(state="confirmed"))
        assert p.tasking_tier == TIER_ELEVATED

    def test_confirmed_persistent_is_priority(self):
        """Confirmed with duration >= 2 reaches priority."""
        p = compute_policy(_rec(state="confirmed", duration=3))
        assert p.tasking_tier == TIER_PRIORITY

    def test_emerging_short_is_routine(self):
        """Brief emerging (no duration) stays routine."""
        p = compute_policy(_rec(state="emerging"))
        assert p.tasking_tier == TIER_ROUTINE

    def test_emerging_persistent_is_elevated(self):
        """Emerging with duration >= 2 reaches elevated."""
        p = compute_policy(_rec(state="emerging", duration=3))
        assert p.tasking_tier == TIER_ELEVATED

    def test_normal(self):
        p = compute_policy(_rec())
        assert p.tasking_tier == TIER_ROUTINE

    def test_weak_custody_without_anomaly_stays_routine(self):
        """Weak custody alone (no anomaly) no longer elevates."""
        p = compute_policy(_rec(state="normal", custody=0.2))
        assert p.tasking_tier == TIER_ROUTINE

    def test_weak_custody_with_anomaly_stays_at_level(self):
        """Weak custody + anomaly evidence: tier raise now requires
        confirmed/sustained state at priority level."""
        r = _rec(state="normal", custody=0.2)
        r["anomaly_score"] = 0.8
        p = compute_policy(r)
        # Normal state + weak anomaly → elevated (from fused threshold)
        # but custody alone no longer promotes beyond that
        assert p.tasking_tier in (TIER_ROUTINE, TIER_ELEVATED)

    def test_high_fused_score_with_anomaly_raises_tier(self):
        """High fused score + anomaly evidence → elevated or priority."""
        from custody.belief_assessment import FusionAssessment
        from datetime import datetime, timezone
        r = _rec(priority=0.9, state="normal")
        r["anomaly_score"] = 1.0  # above ANOMALY_THRESHOLD
        r["fusion_assessment"] = FusionAssessment(
            entity_id="X", timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            fused_score=0.9, uncertainty=0.3, source_agreement=0.8,
            missing_evidence=[], recommended_confirming_source=None,
        )
        p = compute_policy(r)
        assert p.tasking_tier in (TIER_PRIORITY, TIER_ELEVATED)


# ---------------------------------------------------------------------------
# Monitoring actions
# ---------------------------------------------------------------------------

class TestMonitoringActions:

    def test_recovering_cooldown(self):
        p = compute_policy(_rec(state="recovering"))
        assert p.monitoring_action == ACTION_COOLDOWN

    def test_critical_intensify(self):
        p = compute_policy(_rec(state="critical"))
        assert p.monitoring_action == ACTION_INTENSIFY

    def test_confirmed_confirm(self):
        p = compute_policy(_rec(state="confirmed", agreement="confirmed"))
        assert p.monitoring_action == ACTION_CONFIRM

    def test_normal_maintain(self):
        p = compute_policy(_rec())
        assert p.monitoring_action == ACTION_MAINTAIN

    def test_elevated_increase(self):
        """Persistent emerging → elevated → increase_attention."""
        p = compute_policy(_rec(state="emerging", duration=3))
        assert p.monitoring_action == ACTION_INCREASE_ATTENTION


# ---------------------------------------------------------------------------
# Revisit intervals
# ---------------------------------------------------------------------------

class TestRevisit:

    def test_critical_shortest(self):
        p = compute_policy(_rec(state="critical"))
        assert p.desired_revisit_hours <= 1.0

    def test_routine_longest(self):
        p = compute_policy(_rec())
        assert p.desired_revisit_hours >= 12.0

    def test_long_duration_reduces_revisit(self):
        short = compute_policy(_rec(state="confirmed", duration=1))
        long = compute_policy(_rec(state="confirmed", duration=8))
        assert long.desired_revisit_hours <= short.desired_revisit_hours


# ---------------------------------------------------------------------------
# Sensor preference
# ---------------------------------------------------------------------------

class TestSensorPreference:

    def test_urgent_weak_custody_sar(self):
        p = compute_policy(_rec(state="sustained", custody=0.2))
        assert p.sensor_preference == SENSOR_SAR

    def test_emerging_fast_revisit(self):
        p = compute_policy(_rec(state="emerging", agreement="emerging"))
        assert p.sensor_preference == SENSOR_FAST

    def test_normal_any(self):
        p = compute_policy(_rec())
        assert p.sensor_preference == SENSOR_ANY


# ---------------------------------------------------------------------------
# Rationale and dataclass
# ---------------------------------------------------------------------------

class TestRationaleAndStructure:

    def test_rationale_nonempty(self):
        p = compute_policy(_rec(state="sustained", duration=5))
        assert len(p.rationale) > 10
        assert "sustained" in p.rationale

    def test_frozen(self):
        p = compute_policy(_rec())
        with pytest.raises((FrozenInstanceError, AttributeError)):
            p.tasking_tier = "bogus"


# ---------------------------------------------------------------------------
# Simulation integration
# ---------------------------------------------------------------------------

class TestSimulationIntegration:

    def test_records_carry_policy_fields(self):
        from custody.simulate import run_simulation
        records = run_simulation()
        required = {"tasking_tier", "desired_revisit_hours",
                     "monitoring_action", "sensor_preference", "tasking_rationale"}
        for r in records[:5]:
            missing = required - set(r.keys())
            assert not missing, f"Missing: {missing}"

    def test_normal_vessels_get_routine(self):
        from custody.simulate import run_simulation
        records = run_simulation()
        # At h0, most vessels should be routine
        first_ts = [r for r in records if r["time"] == records[0]["time"]]
        routine = [r for r in first_ts if r["tasking_tier"] == "routine"]
        assert len(routine) >= len(first_ts) // 2
