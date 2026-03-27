"""
Tests for custody.ml.normalize — per-vessel relative anomaly scoring.

Covers:
  1. build_baselines computes correct stats
  2. build_baselines respects min_records
  3. compute_relative_score: median → 0.5
  4. compute_relative_score: p95 → ~0.95
  5. compute_relative_score: above p99 → near 1.0
  6. compute_relative_score: below median → < 0.5
  7. compute_relative_score: handles zero std
  8. add_relative_scores adds column to DataFrame
  9. add_relative_scores NaN for unknown vessel
 10. slow vessel gets high relative score for moderate absolute score
 11. fast vessel gets low relative score for same absolute score
 12. save/load round-trip preserves baselines
 13. reasoning uses relative threshold when enabled
 14. reasoning falls back to absolute when relative absent
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import pytest

import custody.config as config
from custody.ml.normalize import (
    VesselBaseline,
    build_baselines,
    compute_relative_score,
    add_relative_scores,
    save_baselines,
    load_baselines,
)
from custody.reasoning import classify_agreement, AnomalyAgreement

UTC = timezone.utc
_T0 = datetime(2024, 9, 20, 0, 0, tzinfo=UTC)


def _scored_df():
    """Two vessels with very different score distributions."""
    rows = []
    rng = np.random.default_rng(42)
    # Slow vessel: scores in [0.05, 0.35]
    for h in range(24):
        rows.append({
            "mmsi": 111, "timestamp": _T0 + timedelta(hours=h),
            "ml_anomaly_score": 0.15 + rng.normal(0, 0.05),
        })
    # Fast vessel: scores in [0.6, 0.95]
    for h in range(24):
        rows.append({
            "mmsi": 222, "timestamp": _T0 + timedelta(hours=h),
            "ml_anomaly_score": 0.75 + rng.normal(0, 0.08),
        })
    df = pd.DataFrame(rows)
    df["ml_anomaly_score"] = df["ml_anomaly_score"].clip(0, 1)
    return df


@pytest.fixture
def scored():
    return _scored_df()


@pytest.fixture
def baselines(scored):
    return build_baselines(scored)


# ---------------------------------------------------------------------------
# build_baselines
# ---------------------------------------------------------------------------

class TestBuildBaselines:

    def test_correct_stats(self, scored, baselines):
        bl = baselines[111]
        scores = scored[scored["mmsi"] == 111]["ml_anomaly_score"]
        assert bl.mean == pytest.approx(scores.mean(), abs=0.001)
        assert bl.count == 24

    def test_min_records(self, scored):
        bl = build_baselines(scored, min_records=100)
        assert len(bl) == 0  # both vessels have only 24 records


# ---------------------------------------------------------------------------
# compute_relative_score
# ---------------------------------------------------------------------------

class TestRelativeScore:

    def test_median_maps_to_half(self, baselines):
        bl = baselines[111]
        rel = compute_relative_score(bl.p50, bl)
        assert abs(rel - 0.5) < 0.01

    def test_p95_maps_near_095(self, baselines):
        bl = baselines[111]
        rel = compute_relative_score(bl.p95, bl)
        assert abs(rel - 0.95) < 0.01

    def test_above_p99_near_one(self, baselines):
        bl = baselines[111]
        rel = compute_relative_score(bl.p99 + 0.1, bl)
        assert rel > 0.99

    def test_below_median_less_than_half(self, baselines):
        bl = baselines[111]
        rel = compute_relative_score(bl.p50 - 0.05, bl)
        assert rel < 0.5

    def test_zero_std_handled(self):
        bl = VesselBaseline(mmsi=999, mean=0.3, std=0.0,
                            p50=0.3, p90=0.3, p95=0.3, p99=0.3, count=10)
        assert compute_relative_score(0.3, bl) == 0.5
        assert compute_relative_score(0.5, bl) == 1.0


# ---------------------------------------------------------------------------
# add_relative_scores
# ---------------------------------------------------------------------------

class TestAddRelativeScores:

    def test_adds_column(self, scored, baselines):
        result = add_relative_scores(scored, baselines)
        assert "ml_anomaly_relative" in result.columns
        assert len(result) == len(scored)

    def test_nan_for_unknown_vessel(self, scored, baselines):
        extra = pd.DataFrame([{
            "mmsi": 999, "timestamp": _T0, "ml_anomaly_score": 0.5,
        }])
        combined = pd.concat([scored, extra], ignore_index=True)
        result = add_relative_scores(combined, baselines)
        unknown_row = result[result["mmsi"] == 999].iloc[0]
        assert np.isnan(unknown_row["ml_anomaly_relative"])


# ---------------------------------------------------------------------------
# Cross-vessel comparison
# ---------------------------------------------------------------------------

class TestCrossVesselComparison:

    def test_slow_vessel_high_relative_for_moderate_absolute(self, baselines):
        """A score of 0.35 is high for slow vessel 111 (normally ~0.15),
        but low for fast vessel 222 (normally ~0.75)."""
        rel_slow = compute_relative_score(0.35, baselines[111])
        rel_fast = compute_relative_score(0.35, baselines[222])
        assert rel_slow > rel_fast
        assert rel_slow > 0.9  # extreme for vessel 111
        assert rel_fast < 0.3  # below normal for vessel 222


# ---------------------------------------------------------------------------
# Save/load round-trip
# ---------------------------------------------------------------------------

class TestSaveLoad:

    def test_round_trip(self, baselines, tmp_path):
        path = tmp_path / "baselines.joblib"
        save_baselines(baselines, path)
        loaded = load_baselines(path)
        assert set(loaded.keys()) == set(baselines.keys())
        for mmsi in baselines:
            assert loaded[mmsi].mean == pytest.approx(baselines[mmsi].mean)
            assert loaded[mmsi].count == baselines[mmsi].count


# ---------------------------------------------------------------------------
# Reasoning integration
# ---------------------------------------------------------------------------

class TestReasoningIntegration:

    def test_relative_threshold_when_enabled(self):
        """With USE_RELATIVE_ML_THRESHOLD=True, a high relative score
        should trigger EMERGING even when absolute score is low."""
        old_use = config.USE_RELATIVE_ML_THRESHOLD
        old_thresh = config.RELATIVE_ML_HIGH_THRESHOLD
        try:
            config.USE_RELATIVE_ML_THRESHOLD = True
            config.RELATIVE_ML_HIGH_THRESHOLD = 0.95
            record = {
                "ml_anomaly_score": 0.35,      # low absolute
                "ml_anomaly_relative": 0.97,   # high relative
                "anomaly_score": 0.2,           # low heuristic
            }
            assert classify_agreement(record) == AnomalyAgreement.EMERGING
        finally:
            config.USE_RELATIVE_ML_THRESHOLD = old_use
            config.RELATIVE_ML_HIGH_THRESHOLD = old_thresh

    def test_falls_back_to_absolute_when_relative_absent(self):
        """When USE_RELATIVE_ML_THRESHOLD=True but ml_anomaly_relative
        is missing, fall back to absolute threshold."""
        old_use = config.USE_RELATIVE_ML_THRESHOLD
        try:
            config.USE_RELATIVE_ML_THRESHOLD = True
            record = {
                "ml_anomaly_score": 0.9,
                "anomaly_score": 0.2,
            }
            # No ml_anomaly_relative → falls back to absolute 0.9 >= 0.8
            assert classify_agreement(record) == AnomalyAgreement.EMERGING
        finally:
            config.USE_RELATIVE_ML_THRESHOLD = old_use

    def test_absolute_mode_ignores_relative(self):
        """With USE_RELATIVE_ML_THRESHOLD=False (default), relative
        score is ignored."""
        assert config.USE_RELATIVE_ML_THRESHOLD is False
        record = {
            "ml_anomaly_score": 0.3,
            "ml_anomaly_relative": 0.99,
            "anomaly_score": 0.2,
        }
        # Absolute 0.3 < 0.8 → NORMAL despite high relative
        assert classify_agreement(record) == AnomalyAgreement.NORMAL
