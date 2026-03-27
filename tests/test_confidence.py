"""
Tests for custody.confidence — uncertainty and confidence modeling.

Covers:
  1.  history_confidence: empty history → low
  2.  history_confidence: 6+ records → 1.0
  3.  history_confidence: 3 records → intermediate
  4.  baseline_confidence: no relative → 0.3
  5.  baseline_confidence: relative present → 1.0
  6.  baseline_confidence: NaN relative → 0.3
  7.  sensor_confidence: EO at night → low
  8.  sensor_confidence: SAR always high
  9.  sensor_confidence: EO in daylight → high
 10.  overall_confidence: all factors high → high category
 11.  overall_confidence: sparse history + no baseline → low/moderate
 12.  overall_confidence: rationale mentions weak factors
 13.  apply_confidence_to_priority: high confidence → no damping
 14.  apply_confidence_to_priority: low confidence → damped
 15.  apply_confidence_to_priority: bounded [0, 1]
 16.  confidence_action_bias: low conf shifts intensify → confirm
 17.  confidence_action_bias: high conf keeps intensify
 18.  confidence_action_bias: maintain unchanged regardless
 19.  simulation records carry confidence fields
 20.  first timestep has lower history_confidence than later
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from custody.confidence import (
    history_confidence,
    baseline_confidence,
    sensor_confidence,
    compute_overall_confidence,
    apply_confidence_to_priority,
    confidence_action_bias,
)

UTC = timezone.utc
_T0 = datetime(2026, 3, 23, 10, 0, tzinfo=UTC)


def _rec(**overrides):
    base = {
        "target_id": "TEST",
        "time": _T0,
        "custody_confidence": 0.8,
        "eo_suitability": 1.0,
        "sar_suitability": 1.0,
        "effective_sensor_preference": "any",
        "solar_condition": "day",
    }
    base.update(overrides)
    return base


def _history(n):
    return [_rec(time=_T0 - timedelta(hours=i)) for i in range(n, 0, -1)]


# ---------------------------------------------------------------------------
# history_confidence
# ---------------------------------------------------------------------------

class TestHistoryConfidence:

    def test_empty_history_low(self):
        assert history_confidence(_rec(), []) == pytest.approx(0.2)

    def test_full_history_high(self):
        assert history_confidence(_rec(), _history(6)) == 1.0

    def test_partial_history_intermediate(self):
        c = history_confidence(_rec(), _history(3))
        assert 0.5 < c < 0.9


# ---------------------------------------------------------------------------
# baseline_confidence
# ---------------------------------------------------------------------------

class TestBaselineConfidence:

    def test_no_relative_low(self):
        assert baseline_confidence(_rec()) == 0.3

    def test_relative_present_high(self):
        assert baseline_confidence(_rec(ml_anomaly_relative=0.5)) == 1.0

    def test_nan_relative_low(self):
        assert baseline_confidence(_rec(ml_anomaly_relative=float("nan"))) == 0.3


# ---------------------------------------------------------------------------
# sensor_confidence
# ---------------------------------------------------------------------------

class TestSensorConfidence:

    def test_eo_night_low(self):
        r = _rec(eo_suitability=0.0, effective_sensor_preference="high_resolution")
        assert sensor_confidence(r) == 0.3

    def test_sar_always_high(self):
        r = _rec(eo_suitability=0.0, effective_sensor_preference="all_weather")
        assert sensor_confidence(r) == 1.0

    def test_eo_day_high(self):
        r = _rec(eo_suitability=1.0, effective_sensor_preference="high_resolution")
        assert sensor_confidence(r) == 1.0


# ---------------------------------------------------------------------------
# overall_confidence
# ---------------------------------------------------------------------------

class TestOverallConfidence:

    def test_all_high(self):
        r = _rec(ml_anomaly_relative=0.5, custody_confidence=0.9)
        c = compute_overall_confidence(r, _history(8))
        assert c["overall_confidence"] >= 0.75
        assert c["confidence_category"] == "high"

    def test_sparse_history_no_baseline(self):
        r = _rec(custody_confidence=0.4)
        c = compute_overall_confidence(r, [])
        assert c["overall_confidence"] < 0.75
        assert c["confidence_category"] in ("moderate", "low")

    def test_rationale_mentions_weak_factors(self):
        r = _rec(custody_confidence=0.3)
        c = compute_overall_confidence(r, [])
        assert "sparse history" in c["confidence_rationale"]
        assert "weak custody" in c["confidence_rationale"]

    def test_rationale_adequate_when_strong(self):
        r = _rec(ml_anomaly_relative=0.5, custody_confidence=0.9)
        c = compute_overall_confidence(r, _history(8))
        assert "adequate" in c["confidence_rationale"]


# ---------------------------------------------------------------------------
# Priority damping
# ---------------------------------------------------------------------------

class TestPriorityDamping:

    def test_high_confidence_no_damping(self):
        adj = apply_confidence_to_priority(0.8, 0.9)
        assert adj == 0.8

    def test_low_confidence_damped(self):
        adj = apply_confidence_to_priority(0.8, 0.3)
        assert adj < 0.8

    def test_bounded(self):
        assert apply_confidence_to_priority(1.0, 0.0) >= 0.0
        assert apply_confidence_to_priority(1.0, 1.0) <= 1.0


# ---------------------------------------------------------------------------
# Action bias
# ---------------------------------------------------------------------------

class TestActionBias:

    def test_low_confidence_shifts_intensify(self):
        assert confidence_action_bias("intensify", 0.3) == "confirm"

    def test_low_confidence_shifts_increase_attention(self):
        assert confidence_action_bias("increase_attention", 0.3) == "confirm"

    def test_high_confidence_keeps_intensify(self):
        assert confidence_action_bias("intensify", 0.8) == "intensify"

    def test_maintain_unchanged(self):
        assert confidence_action_bias("maintain", 0.3) == "maintain"

    def test_cooldown_unchanged(self):
        assert confidence_action_bias("cooldown", 0.3) == "cooldown"


# ---------------------------------------------------------------------------
# Simulation integration
# ---------------------------------------------------------------------------

class TestSimulationIntegration:

    @pytest.fixture(scope="class")
    def records(self):
        from custody.simulate import run_simulation
        return run_simulation()

    def test_records_have_confidence_fields(self, records):
        required = {
            "history_confidence", "baseline_confidence",
            "sensor_confidence", "overall_confidence",
            "confidence_category", "confidence_rationale",
            "confidence_adjusted_priority",
        }
        for r in records[:5]:
            missing = required - set(r.keys())
            assert not missing, f"Missing: {missing}"

    def test_first_step_lower_history_confidence(self, records):
        """First timestep has no prior history → low history_confidence."""
        vid = records[0]["target_id"]
        first = next(r for r in records if r["target_id"] == vid)
        later = [r for r in records if r["target_id"] == vid][-1]
        assert first["history_confidence"] < later["history_confidence"]

    def test_overall_in_unit_interval(self, records):
        for r in records:
            assert 0.0 <= r["overall_confidence"] <= 1.0

    def test_confidence_adjusted_priority_present(self, records):
        for r in records:
            assert "confidence_adjusted_priority" in r
            assert 0.0 <= r["confidence_adjusted_priority"] <= 1.0
