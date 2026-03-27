"""
Tests for custody.ml.features — ML feature extraction.

Uses small synthetic fixtures.

Covers:
  1. heading_delta circular wraparound (350 → 10 = +20)
  2. heading_delta reverse wraparound (10 → 350 = -20)
  3. heading_delta no wrap (90 → 180 = +90)
  4. extract_features returns all expected keys
  5. extract_features with empty history → zero deltas
  6. extract_features speed_delta correct
  7. extract_features heading_delta correct
  8. extract_features rolling stats with full window
  9. extract_features no future leakage
 10. build_feature_frame returns correct shape
 11. build_feature_frame has expected columns
 12. build_feature_frame preserves vessel grouping
 13. build_feature_frame first row has zero deltas
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import pytest

from custody.ml.features import (
    FEATURE_COLUMNS,
    build_feature_frame,
    extract_features,
    heading_delta,
)

UTC = timezone.utc
_T0 = datetime(2024, 9, 20, 0, 0, tzinfo=UTC)


def _record(hour=0, speed=10.0, heading=90.0, lat=30.0, lon=-88.0, mmsi=1):
    return {
        "mmsi": mmsi,
        "timestamp": _T0 + timedelta(hours=hour),
        "lat": lat,
        "lon": lon,
        "speed_kmh": speed,
        "heading_deg": heading,
    }


# ---------------------------------------------------------------------------
# heading_delta
# ---------------------------------------------------------------------------

class TestHeadingDelta:

    def test_clockwise_wrap(self):
        # 350 → 10 = +20
        assert abs(heading_delta(10.0, 350.0) - 20.0) < 0.01

    def test_counter_clockwise_wrap(self):
        # 10 → 350 = -20
        assert abs(heading_delta(350.0, 10.0) - (-20.0)) < 0.01

    def test_no_wrap(self):
        assert abs(heading_delta(180.0, 90.0) - 90.0) < 0.01

    def test_zero_delta(self):
        assert heading_delta(45.0, 45.0) == 0.0

    def test_half_circle(self):
        assert abs(heading_delta(270.0, 90.0)) == 180.0


# ---------------------------------------------------------------------------
# extract_features
# ---------------------------------------------------------------------------

class TestExtractFeatures:

    def test_returns_all_keys(self):
        feats = extract_features(_record(), [])
        assert set(FEATURE_COLUMNS).issubset(feats.keys())

    def test_empty_history_zero_deltas(self):
        feats = extract_features(_record(), [])
        assert feats["speed_delta"] == 0.0
        assert feats["heading_delta"] == 0.0
        assert feats["lat_delta"] == 0.0
        assert feats["lon_delta"] == 0.0

    def test_speed_delta(self):
        prev = _record(hour=0, speed=8.0)
        curr = _record(hour=1, speed=12.0)
        feats = extract_features(curr, [prev])
        assert abs(feats["speed_delta"] - 4.0) < 0.01

    def test_heading_delta_circular(self):
        prev = _record(hour=0, heading=350.0)
        curr = _record(hour=1, heading=10.0)
        feats = extract_features(curr, [prev])
        assert abs(feats["heading_delta"] - 20.0) < 0.01

    def test_rolling_stats(self):
        history = [_record(hour=i, speed=float(10 + i)) for i in range(6)]
        curr = _record(hour=6, speed=20.0)
        feats = extract_features(curr, history)
        speeds = [10, 11, 12, 13, 14, 15]
        assert abs(feats["speed_mean_6h"] - np.mean(speeds)) < 0.01
        assert abs(feats["speed_std_6h"] - np.std(speeds, ddof=0)) < 0.01

    def test_no_future_leakage(self):
        """Rolling window must not include the current record."""
        history = [_record(hour=i, speed=5.0) for i in range(3)]
        curr = _record(hour=3, speed=100.0)  # spike
        feats = extract_features(curr, history)
        # speed_mean_6h should be 5.0 (from history), NOT contaminated by 100.0
        assert abs(feats["speed_mean_6h"] - 5.0) < 0.01


# ---------------------------------------------------------------------------
# build_feature_frame
# ---------------------------------------------------------------------------

class TestBuildFeatureFrame:

    @pytest.fixture
    def small_df(self):
        rows = []
        for mmsi in [1, 2]:
            for h in range(10):
                rows.append(_record(hour=h, speed=10.0 + h, heading=90.0 + h * 5, mmsi=mmsi))
        df = pd.DataFrame(rows)
        df.sort_values(["mmsi", "timestamp"], inplace=True)
        return df

    def test_correct_shape(self, small_df):
        ff = build_feature_frame(small_df)
        assert len(ff) == len(small_df)

    def test_expected_columns(self, small_df):
        ff = build_feature_frame(small_df)
        for col in FEATURE_COLUMNS:
            assert col in ff.columns
        assert "mmsi" in ff.columns
        assert "timestamp" in ff.columns

    def test_preserves_vessel_grouping(self, small_df):
        ff = build_feature_frame(small_df)
        assert set(ff["mmsi"].unique()) == {1, 2}
        assert ff.groupby("mmsi").size().tolist() == [10, 10]

    def test_first_row_zero_deltas(self, small_df):
        ff = build_feature_frame(small_df)
        for mmsi in [1, 2]:
            first = ff[ff["mmsi"] == mmsi].iloc[0]
            assert first["speed_delta"] == 0.0
            assert first["heading_delta"] == 0.0
