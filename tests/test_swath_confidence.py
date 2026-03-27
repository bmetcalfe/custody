"""
Tests for confidence-aware swath/grouped collection valuation.

Covers:
  1. _target_value prefers confidence_adjusted_priority
  2. _target_value falls back through priority chain
  3. high-confidence targets produce stronger uplift
  4. low-confidence targets produce weaker uplift (via damped priority)
  5. mean_covered_confidence computed correctly
  6. rationale mentions low confidence when mean is low
  7. rationale mentions high confidence when mean is high
  8. isolated target: unchanged behavior
  9. uplift cap still applies
 10. simulation records carry mean_covered_confidence
"""
from __future__ import annotations

import pytest

from custody.swath import assess_swath, _target_value


def _target(tid, lat, lon, conf_adj=None, portfolio=None, priority=None, anomaly=0.0, overall_conf=0.8):
    r = {"target_id": tid, "lat": lat, "lon": lon, "anomaly_score": anomaly, "overall_confidence": overall_conf}
    if conf_adj is not None:
        r["confidence_adjusted_priority"] = conf_adj
    if portfolio is not None:
        r["portfolio_score"] = portfolio
    if priority is not None:
        r["priority_score"] = priority
    return r


# ---------------------------------------------------------------------------
# _target_value priority chain
# ---------------------------------------------------------------------------

class TestTargetValueChain:

    def test_prefers_confidence_adjusted(self):
        r = _target("A", 30, -88, conf_adj=0.65, portfolio=0.8, priority=0.7)
        assert _target_value(r) == 0.65

    def test_falls_back_to_portfolio(self):
        r = _target("A", 30, -88, portfolio=0.7, priority=0.5)
        assert _target_value(r) == 0.7

    def test_falls_back_to_priority(self):
        r = _target("A", 30, -88, priority=0.6)
        assert _target_value(r) == 0.6

    def test_falls_back_to_anomaly(self):
        r = _target("A", 30, -88, anomaly=1.5)
        assert _target_value(r) == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# Confidence-aware uplift
# ---------------------------------------------------------------------------

class TestConfidenceAwareUplift:

    def test_high_confidence_stronger_uplift(self):
        """High-confidence covered targets produce more uplift."""
        records_high = [
            _target("A", 30.0, -88.0, conf_adj=0.5),
            _target("B", 30.01, -88.01, conf_adj=0.7),  # high conf_adj
        ]
        records_low = [
            _target("A", 30.0, -88.0, conf_adj=0.5),
            _target("B", 30.01, -88.01, conf_adj=0.2),  # low conf_adj (damped)
        ]
        sa_high = assess_swath(records_high[0], "all_weather", records_high)
        sa_low = assess_swath(records_low[0], "all_weather", records_low)
        assert sa_high.value_uplift > sa_low.value_uplift

    def test_mean_covered_confidence_computed(self):
        records = [
            _target("A", 30.0, -88.0, conf_adj=0.5, overall_conf=0.9),
            _target("B", 30.01, -88.01, conf_adj=0.3, overall_conf=0.4),
        ]
        sa = assess_swath(records[0], "all_weather", records)
        # Mean of 0.9 and 0.4 = 0.65
        assert sa.mean_covered_confidence == pytest.approx(0.65, abs=0.01)

    def test_low_confidence_rationale(self):
        records = [
            _target("A", 30.0, -88.0, conf_adj=0.5, overall_conf=0.3),
            _target("B", 30.01, -88.01, conf_adj=0.3, overall_conf=0.2),
        ]
        sa = assess_swath(records[0], "all_weather", records)
        assert "low confidence" in sa.rationale

    def test_high_confidence_rationale(self):
        records = [
            _target("A", 30.0, -88.0, conf_adj=0.5, overall_conf=0.9),
            _target("B", 30.01, -88.01, conf_adj=0.7, overall_conf=0.85),
        ]
        sa = assess_swath(records[0], "all_weather", records)
        assert "high-confidence" in sa.rationale


# ---------------------------------------------------------------------------
# Preserved behaviors
# ---------------------------------------------------------------------------

class TestPreservedBehaviors:

    def test_isolated_target_unchanged(self):
        records = [
            _target("A", 30.0, -88.0, conf_adj=0.6),
            _target("B", 40.0, -70.0, conf_adj=0.9),  # far away
        ]
        sa = assess_swath(records[0], "high_resolution", records)
        assert sa.covered_target_count == 1
        assert sa.value_uplift == 0.0
        assert sa.mean_covered_confidence > 0

    def test_uplift_still_capped(self):
        records = [_target("A", 30.0, -88.0, conf_adj=0.5)]
        for i in range(10):
            records.append(_target(f"X{i}", 30.0 + i * 0.001, -88.0, conf_adj=0.9))
        sa = assess_swath(records[0], "all_weather", records)
        from custody.config import SWATH_UPLIFT_CAP
        assert sa.value_uplift <= SWATH_UPLIFT_CAP


# ---------------------------------------------------------------------------
# Simulation integration
# ---------------------------------------------------------------------------

class TestSimulationIntegration:

    def test_records_carry_mean_confidence(self):
        from custody.simulate import run_simulation
        records = run_simulation()
        task_recs = [r for r in records if r["action"] == "TASK"]
        for r in task_recs[:5]:
            assert "mean_covered_confidence" in r
