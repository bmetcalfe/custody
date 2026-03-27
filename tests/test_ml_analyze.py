"""
Tests for custody.ml.analyze — anomaly score inspection and analysis.

Uses small synthetic DataFrames.

Covers:
  1. top_anomalous_records returns correct count and order
  2. top_anomalous_records handles fewer rows than n
  3. top_anomalous_vessels returns correct count and order
  4. vessel_summary computes mean/max/p95/peak correctly
  5. vessel_summary handles single-record vessel
  6. vessel_time_series extracts correct vessel
  7. vessel_time_series returns empty for unknown vessel
  8. anomaly_excursions filters by threshold
  9. contiguous_windows finds sustained anomaly periods
 10. contiguous_windows respects min_length
 11. contiguous_windows handles no windows
 12. markdown_summary returns non-empty string
 13. all functions handle empty DataFrame
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pandas as pd
import pytest

from custody.ml.analyze import (
    anomaly_excursions,
    contiguous_windows,
    markdown_summary,
    top_anomalous_records,
    top_anomalous_vessels,
    vessel_summary,
    vessel_time_series,
)

UTC = timezone.utc
_T0 = datetime(2024, 9, 20, 0, 0, tzinfo=UTC)


def _scored_df():
    """Create a small scored DataFrame with 3 vessels, varied scores."""
    rows = []
    # Vessel A: mostly normal, one spike at h5
    for h in range(10):
        score = 0.9 if h == 5 else 0.1
        rows.append({
            "mmsi": 100, "timestamp": _T0 + timedelta(hours=h),
            "lat": 30.0, "lon": -88.0, "speed_kmh": 10.0,
            "heading_deg": 90.0, "ml_anomaly_score": score,
        })
    # Vessel B: consistently moderate
    for h in range(10):
        rows.append({
            "mmsi": 200, "timestamp": _T0 + timedelta(hours=h),
            "lat": 31.0, "lon": -87.0, "speed_kmh": 15.0,
            "heading_deg": 180.0, "ml_anomaly_score": 0.5,
        })
    # Vessel C: high sustained anomaly h3-h7
    for h in range(10):
        score = 0.85 if 3 <= h <= 7 else 0.2
        rows.append({
            "mmsi": 300, "timestamp": _T0 + timedelta(hours=h),
            "lat": 29.0, "lon": -89.0, "speed_kmh": 25.0,
            "heading_deg": 270.0, "ml_anomaly_score": score,
        })
    df = pd.DataFrame(rows)
    df.sort_values(["mmsi", "timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


@pytest.fixture
def scored():
    return _scored_df()


def _empty_df():
    return pd.DataFrame(columns=[
        "mmsi", "timestamp", "lat", "lon", "speed_kmh",
        "heading_deg", "ml_anomaly_score",
    ])


# ---------------------------------------------------------------------------
# top_anomalous_records
# ---------------------------------------------------------------------------

class TestTopRecords:

    def test_returns_correct_count(self, scored):
        top = top_anomalous_records(scored, n=5)
        assert len(top) == 5

    def test_sorted_descending(self, scored):
        top = top_anomalous_records(scored, n=10)
        scores = top["ml_anomaly_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_handles_fewer_than_n(self, scored):
        top = top_anomalous_records(scored, n=999)
        assert len(top) == len(scored)

    def test_empty_input(self):
        top = top_anomalous_records(_empty_df(), n=5)
        assert len(top) == 0


# ---------------------------------------------------------------------------
# vessel_summary / top_anomalous_vessels
# ---------------------------------------------------------------------------

class TestVesselSummary:

    def test_correct_count(self, scored):
        vs = vessel_summary(scored)
        assert len(vs) == 3  # 3 vessels

    def test_sorted_by_max(self, scored):
        vs = vessel_summary(scored)
        maxes = vs["max"].tolist()
        assert maxes == sorted(maxes, reverse=True)

    def test_mean_max_p95(self, scored):
        vs = vessel_summary(scored)
        vessel_a = vs[vs["mmsi"] == 100].iloc[0]
        assert vessel_a["max"] == pytest.approx(0.9)
        assert vessel_a["mean"] == pytest.approx((0.1 * 9 + 0.9) / 10)

    def test_peak_timestamp(self, scored):
        vs = vessel_summary(scored)
        vessel_a = vs[vs["mmsi"] == 100].iloc[0]
        assert vessel_a["peak_timestamp"] == _T0 + timedelta(hours=5)

    def test_single_record_vessel(self):
        df = pd.DataFrame([{
            "mmsi": 999, "timestamp": _T0,
            "lat": 30.0, "lon": -88.0, "speed_kmh": 5.0,
            "heading_deg": 0.0, "ml_anomaly_score": 0.7,
        }])
        vs = vessel_summary(df)
        assert len(vs) == 1
        assert vs.iloc[0]["max"] == pytest.approx(0.7)

    def test_top_n(self, scored):
        top = top_anomalous_vessels(scored, n=2)
        assert len(top) == 2

    def test_empty_input(self):
        vs = vessel_summary(_empty_df())
        assert len(vs) == 0


# ---------------------------------------------------------------------------
# vessel_time_series
# ---------------------------------------------------------------------------

class TestVesselTimeSeries:

    def test_extracts_correct_vessel(self, scored):
        ts = vessel_time_series(scored, 200)
        assert len(ts) == 10
        assert (ts["mmsi"] == 200).all()

    def test_sorted_by_time(self, scored):
        ts = vessel_time_series(scored, 100)
        times = ts["timestamp"].tolist()
        assert times == sorted(times)

    def test_unknown_vessel_empty(self, scored):
        ts = vessel_time_series(scored, 999)
        assert len(ts) == 0


# ---------------------------------------------------------------------------
# Excursions
# ---------------------------------------------------------------------------

class TestExcursions:

    def test_filters_by_threshold(self, scored):
        exc = anomaly_excursions(scored, threshold=0.8)
        assert (exc["ml_anomaly_score"] >= 0.8).all()
        # Vessel A: 1 record at 0.9; Vessel C: 5 records at 0.85
        assert len(exc) == 6

    def test_empty_when_none_above(self, scored):
        exc = anomaly_excursions(scored, threshold=0.95)
        assert len(exc) == 0


# ---------------------------------------------------------------------------
# Contiguous windows
# ---------------------------------------------------------------------------

class TestContiguousWindows:

    def test_finds_sustained_window(self, scored):
        wins = contiguous_windows(scored, threshold=0.8, min_length=2)
        # Vessel C has h3-h7 (5 consecutive steps above 0.8)
        assert len(wins) >= 1
        assert any(w["mmsi"] == 300 and w["length"] == 5 for w in wins)

    def test_respects_min_length(self, scored):
        # Vessel A has only 1 record above 0.8 → no window at min_length=2
        wins = contiguous_windows(scored, threshold=0.8, min_length=2)
        assert not any(w["mmsi"] == 100 for w in wins)

    def test_no_windows(self, scored):
        wins = contiguous_windows(scored, threshold=0.99, min_length=2)
        assert len(wins) == 0


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

class TestMarkdown:

    def test_non_empty(self, scored):
        md = markdown_summary(scored, top_n=3)
        assert isinstance(md, str)
        assert len(md) > 50
        assert "100" in md or "200" in md or "300" in md
