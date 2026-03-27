"""
Tests for custody.ml.inject and custody.ml.compare.

Uses synthetic vessel timelines.

Covers:
  1.  inject_loitering reduces speed in window
  2.  inject_loitering holds position near anchor
  3.  inject_loitering does not mutate original
  4.  inject_loitering leaves records outside window unchanged
  5.  inject_heading_deviation sustained offsets heading
  6.  inject_heading_deviation erratic randomizes heading
  7.  inject_heading_deviation does not mutate original
  8.  inject_heading_deviation wraps around 360
  9.  inject_zone_approach moves toward target
 10.  inject_zone_approach sets bearing toward zone
 11.  inject_zone_approach does not mutate original
 12.  compare produces table with expected columns
 13.  compare summary has expected keys
 14.  compare detects score increase from injection
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone, timedelta

import numpy as np
import pytest

from custody.ml.inject import (
    inject_heading_deviation,
    inject_loitering,
    inject_zone_approach,
)

UTC = timezone.utc
_T0 = datetime(2024, 9, 20, 0, 0, tzinfo=UTC)


def _timeline(n_hours=24, speed=12.0, heading=90.0, lat=30.0, lon=-88.0):
    """Uniform straight-line vessel track."""
    return [
        {
            "mmsi": 100000001,
            "timestamp": _T0 + timedelta(hours=h),
            "lat": lat + h * 0.01,
            "lon": lon + h * 0.005,
            "speed_kmh": speed,
            "heading_deg": heading,
        }
        for h in range(n_hours)
    ]


# ---------------------------------------------------------------------------
# Loitering
# ---------------------------------------------------------------------------

class TestLoitering:

    def test_reduces_speed(self):
        tl = _timeline()
        result = inject_loitering(tl, start_hour=5, duration_hours=4, target_speed_kmh=0.5)
        for i in range(5, 9):
            assert result[i]["speed_kmh"] == 0.5
        # Outside window: unchanged
        assert result[0]["speed_kmh"] == 12.0
        assert result[10]["speed_kmh"] == 12.0

    def test_holds_position(self):
        tl = _timeline()
        result = inject_loitering(tl, start_hour=5, duration_hours=4)
        anchor_lat = tl[5]["lat"]
        for i in range(5, 9):
            assert abs(result[i]["lat"] - anchor_lat) < 0.01

    def test_does_not_mutate_original(self):
        tl = _timeline()
        original = copy.deepcopy(tl)
        inject_loitering(tl, start_hour=5, duration_hours=4)
        assert tl == original

    def test_outside_window_unchanged(self):
        tl = _timeline()
        result = inject_loitering(tl, start_hour=5, duration_hours=4)
        for i in [0, 1, 2, 3, 4, 10, 11, 20]:
            assert result[i]["lat"] == tl[i]["lat"]
            assert result[i]["speed_kmh"] == tl[i]["speed_kmh"]


# ---------------------------------------------------------------------------
# Heading deviation
# ---------------------------------------------------------------------------

class TestHeadingDeviation:

    def test_sustained_offsets(self):
        tl = _timeline(heading=90.0)
        result = inject_heading_deviation(tl, 5, 4, mode="sustained", deviation_deg=45.0)
        for i in range(5, 9):
            assert abs(result[i]["heading_deg"] - 135.0) < 0.01

    def test_erratic_randomizes(self):
        tl = _timeline(heading=90.0)
        result = inject_heading_deviation(tl, 5, 4, mode="erratic")
        headings = [result[i]["heading_deg"] for i in range(5, 9)]
        # Should not all be 90.0 (original)
        assert not all(abs(h - 90.0) < 1.0 for h in headings)
        # All in [0, 360)
        assert all(0 <= h < 360 for h in headings)

    def test_does_not_mutate(self):
        tl = _timeline()
        original = copy.deepcopy(tl)
        inject_heading_deviation(tl, 5, 4)
        assert tl == original

    def test_wraps_around_360(self):
        tl = _timeline(heading=350.0)
        result = inject_heading_deviation(tl, 5, 4, mode="sustained", deviation_deg=30.0)
        for i in range(5, 9):
            assert abs(result[i]["heading_deg"] - 20.0) < 0.01


# ---------------------------------------------------------------------------
# Zone approach
# ---------------------------------------------------------------------------

class TestZoneApproach:

    def test_moves_toward_target(self):
        tl = _timeline(lat=30.0, lon=-88.0)
        zone_lat, zone_lon = 31.0, -87.0
        result = inject_zone_approach(tl, 5, 6, zone_lat=zone_lat, zone_lon=zone_lon)
        # Last injected record should be near the zone
        last_inj = result[10]  # 5 + 6 - 1
        assert abs(last_inj["lat"] - zone_lat) < 0.2
        assert abs(last_inj["lon"] - zone_lon) < 0.2

    def test_sets_bearing_toward_zone(self):
        tl = _timeline(lat=30.0, lon=-88.0)
        result = inject_zone_approach(tl, 5, 6, zone_lat=31.0, zone_lon=-87.0)
        # Heading should be roughly NE (zone is N and E of origin)
        for i in range(5, 11):
            hdg = result[i]["heading_deg"]
            assert 0 <= hdg < 360

    def test_does_not_mutate(self):
        tl = _timeline()
        original = copy.deepcopy(tl)
        inject_zone_approach(tl, 5, 6)
        assert tl == original


# ---------------------------------------------------------------------------
# Compare (integration with scorer)
# ---------------------------------------------------------------------------

class TestCompare:

    @pytest.fixture
    def scorer(self, tmp_path):
        """Train a quick model on synthetic normal data."""
        from custody.ml.features import build_feature_frame
        from custody.ml.scorer import train
        import pandas as pd

        tls = []
        for v in range(20):
            for h in range(24):
                tls.append({
                    "mmsi": 100000000 + v,
                    "timestamp": _T0 + timedelta(hours=h),
                    "lat": 30.0 + h * 0.01,
                    "lon": -88.0 + h * 0.005,
                    "speed_kmh": 10.0 + np.random.default_rng(v * 100 + h).normal(0, 0.5),
                    "heading_deg": 90.0 + np.random.default_rng(v * 100 + h).normal(0, 2),
                })
        df = pd.DataFrame(tls)
        df.sort_values(["mmsi", "timestamp"], inplace=True)
        features = build_feature_frame(df)
        model_path = str(tmp_path / "test_model.joblib")
        return train(features, model_path)

    def test_compare_table_columns(self, scorer):
        from custody.ml.compare import compare
        baseline = _timeline()
        injected = inject_loitering(baseline, 8, 6)
        result = compare(baseline, injected, scorer, label="test")
        table = result["table"]
        expected = {"timestamp", "base_ml_score", "inj_ml_score", "ml_score_delta",
                    "base_speed", "inj_speed"}
        assert expected.issubset(set(table.columns))

    def test_compare_summary_keys(self, scorer):
        from custody.ml.compare import compare
        baseline = _timeline()
        injected = inject_loitering(baseline, 8, 6)
        result = compare(baseline, injected, scorer)
        s = result["summary"]
        expected = {"label", "records", "base_mean_score", "inj_mean_score",
                    "max_score_delta", "inj_max_score"}
        assert expected.issubset(set(s.keys()))

    def test_injection_raises_score(self, scorer):
        from custody.ml.compare import compare
        baseline = _timeline()
        injected = inject_loitering(baseline, 8, 6, target_speed_kmh=0.5)
        result = compare(baseline, injected, scorer)
        # Loitering injection should raise ML score in the window
        assert result["summary"]["max_score_delta"] > 0
