"""
Tests for custody.ingest.noaa_preprocess — NOAA AIS preprocessing pipeline.

Uses small synthetic fixtures; does NOT require the real 800MB NOAA file.

Covers:
  1. load_raw selects correct columns
  2. load_raw renames columns to internal names
  3. load_raw applies bounding-box filter
  4. clean fills speed nulls with 0.0
  5. clean forward-fills heading per vessel
  6. clean adds speed_kmh column with correct conversion
  7. clean drops rows with null lat/lon
  8. select_vessels respects min_obs threshold
  9. select_vessels returns at most n_vessels
 10. select_vessels is deterministic with same seed
 11. resample_hourly produces one row per vessel per hour
 12. resample_hourly computes mean lat/lon
 13. resample_hourly handles circular heading mean
 14. to_timelines produces correct dict structure
 15. to_timelines records have all required keys
 16. to_timelines timestamps are datetime objects
 17. preprocess_noaa_day end-to-end returns timelines
 18. _circular_mean handles 0/360 wraparound
"""
from __future__ import annotations

import io
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from custody.ingest.noaa_preprocess import (
    _circular_mean,
    clean,
    load_raw,
    preprocess_noaa_day,
    resample_hourly,
    select_vessels,
    to_timelines,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_csv(rows: list[dict], path: Path) -> Path:
    """Write rows to a CSV file at path and return the path."""
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def small_csv(tmp_path) -> Path:
    """A small NOAA-format CSV with 3 vessels, ~5 rows each."""
    rows = []
    base = datetime(2024, 9, 24, 0, 0, 0)
    for mmsi in [100000001, 100000002, 100000003]:
        for h in range(20):  # 20 observations over ~5 hours
            ts = base.replace(hour=h // 4, minute=(h % 4) * 15)
            rows.append({
                "mmsi": mmsi,
                "base_date_time": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "latitude": 30.0 + h * 0.01 + (mmsi % 10) * 0.1,
                "longitude": -88.0 + h * 0.005,
                "sog": 5.0 + h * 0.1,
                "cog": 90.0 + h * 2.0,
                "heading": 91.0,
                "vessel_name": f"VESSEL_{mmsi}",
                "imo": "",
                "call_sign": "",
                "vessel_type": 70,
                "status": 0,
                "length": 100,
                "width": 20,
                "draft": 5.0,
                "cargo": 70,
                "transceiver": "A",
            })
    return _make_csv(rows, tmp_path / "test_ais.csv")


@pytest.fixture
def raw_df(small_csv) -> pd.DataFrame:
    return load_raw(small_csv)


@pytest.fixture
def cleaned_df(raw_df) -> pd.DataFrame:
    return clean(raw_df)


# ---------------------------------------------------------------------------
# load_raw
# ---------------------------------------------------------------------------

class TestLoadRaw:

    def test_selects_core_columns(self, raw_df):
        expected = {"mmsi", "timestamp", "lat", "lon", "speed_knots", "heading_deg"}
        assert set(raw_df.columns) == expected

    def test_renames_columns(self, raw_df):
        assert "base_date_time" not in raw_df.columns
        assert "timestamp" in raw_df.columns
        assert "latitude" not in raw_df.columns
        assert "lat" in raw_df.columns

    def test_sorted_by_mmsi_timestamp(self, raw_df):
        for mmsi, group in raw_df.groupby("mmsi"):
            ts = group["timestamp"].tolist()
            assert ts == sorted(ts)

    def test_bbox_filter(self, small_csv):
        # Only lat >= 30.1 should survive (vessel 100000002 starts at 30.1+)
        df = load_raw(small_csv, bbox=(30.15, 31.0, -89.0, -87.0))
        assert len(df) > 0
        assert df["lat"].min() >= 30.15

    def test_timestamps_are_utc(self, raw_df):
        assert raw_df["timestamp"].dt.tz is not None


# ---------------------------------------------------------------------------
# clean
# ---------------------------------------------------------------------------

class TestClean:

    def test_fills_speed_nulls(self, raw_df):
        df = raw_df.copy()
        df.loc[0, "speed_knots"] = None
        result = clean(df)
        assert result["speed_knots"].isna().sum() == 0

    def test_ffills_heading_per_vessel(self, raw_df):
        df = raw_df.copy()
        # Set heading to NaN for some rows in one vessel
        mask = (df["mmsi"] == df["mmsi"].iloc[0]) & (df.index > 2)
        df.loc[mask, "heading_deg"] = None
        result = clean(df)
        assert result["heading_deg"].isna().sum() == 0

    def test_adds_speed_kmh(self, cleaned_df):
        assert "speed_kmh" in cleaned_df.columns
        # Check conversion: first row's speed_knots * 1.852
        row = cleaned_df.iloc[0]
        assert abs(row["speed_kmh"] - row["speed_knots"] * 1.852) < 0.01

    def test_drops_null_latlon(self, raw_df):
        df = raw_df.copy()
        df.loc[0, "lat"] = None
        result = clean(df)
        assert result["lat"].isna().sum() == 0
        assert len(result) < len(df)


# ---------------------------------------------------------------------------
# select_vessels
# ---------------------------------------------------------------------------

class TestSelectVessels:

    def test_respects_min_obs(self, cleaned_df):
        result = select_vessels(cleaned_df, n_vessels=10, min_obs=100)
        # Each vessel has 20 obs; min_obs=100 → no vessels eligible
        assert len(result) == 0

    def test_limits_to_n_vessels(self, cleaned_df):
        result = select_vessels(cleaned_df, n_vessels=2, min_obs=1)
        assert result["mmsi"].nunique() <= 2

    def test_deterministic_with_seed(self, cleaned_df):
        r1 = select_vessels(cleaned_df, n_vessels=2, min_obs=1, seed=99)
        r2 = select_vessels(cleaned_df, n_vessels=2, min_obs=1, seed=99)
        assert set(r1["mmsi"].unique()) == set(r2["mmsi"].unique())

    def test_different_seed_may_differ(self, cleaned_df):
        r1 = select_vessels(cleaned_df, n_vessels=1, min_obs=1, seed=1)
        r2 = select_vessels(cleaned_df, n_vessels=1, min_obs=1, seed=2)
        # With 3 vessels, different seeds may pick different ones
        # (not guaranteed, but very likely)
        # Just verify both return valid results
        assert r1["mmsi"].nunique() == 1
        assert r2["mmsi"].nunique() == 1


# ---------------------------------------------------------------------------
# resample_hourly
# ---------------------------------------------------------------------------

class TestResampleHourly:

    def test_one_row_per_vessel_per_hour(self, cleaned_df):
        subset = select_vessels(cleaned_df, n_vessels=1, min_obs=1)
        hourly = resample_hourly(subset)
        mmsi = hourly["mmsi"].iloc[0]
        vessel = hourly[hourly["mmsi"] == mmsi]
        # Each hour should have at most 1 row
        hours = vessel["timestamp"].dt.floor("h")
        assert hours.nunique() == len(vessel)

    def test_computes_mean_position(self, cleaned_df):
        subset = select_vessels(cleaned_df, n_vessels=1, min_obs=1)
        hourly = resample_hourly(subset)
        # Hourly lat should be between min and max of raw observations
        raw_vessel = subset[subset["mmsi"] == hourly["mmsi"].iloc[0]]
        assert hourly["lat"].min() >= raw_vessel["lat"].min() - 0.001
        assert hourly["lat"].max() <= raw_vessel["lat"].max() + 0.001

    def test_has_speed_kmh(self, cleaned_df):
        subset = select_vessels(cleaned_df, n_vessels=1, min_obs=1)
        hourly = resample_hourly(subset)
        assert "speed_kmh" in hourly.columns
        assert hourly["speed_kmh"].isna().sum() == 0


# ---------------------------------------------------------------------------
# to_timelines
# ---------------------------------------------------------------------------

class TestToTimelines:

    def test_returns_dict_keyed_by_mmsi_string(self, cleaned_df):
        subset = select_vessels(cleaned_df, n_vessels=2, min_obs=1)
        hourly = resample_hourly(subset)
        timelines = to_timelines(hourly)
        assert isinstance(timelines, dict)
        assert all(isinstance(k, str) for k in timelines)
        assert len(timelines) <= 2

    def test_records_have_required_keys(self, cleaned_df):
        subset = select_vessels(cleaned_df, n_vessels=1, min_obs=1)
        hourly = resample_hourly(subset)
        timelines = to_timelines(hourly)
        required = {"timestamp", "lat", "lon", "speed_kmh", "heading_deg", "source"}
        for mmsi, records in timelines.items():
            for r in records:
                assert required.issubset(r.keys()), f"Missing keys: {required - r.keys()}"

    def test_timestamps_are_datetime(self, cleaned_df):
        subset = select_vessels(cleaned_df, n_vessels=1, min_obs=1)
        hourly = resample_hourly(subset)
        timelines = to_timelines(hourly)
        for records in timelines.values():
            for r in records:
                assert isinstance(r["timestamp"], datetime)

    def test_source_is_ais(self, cleaned_df):
        subset = select_vessels(cleaned_df, n_vessels=1, min_obs=1)
        hourly = resample_hourly(subset)
        timelines = to_timelines(hourly)
        for records in timelines.values():
            assert all(r["source"] == "ais" for r in records)


# ---------------------------------------------------------------------------
# _circular_mean
# ---------------------------------------------------------------------------

class TestCircularMean:

    def test_simple_average(self):
        assert abs(_circular_mean(np.array([90.0, 90.0])) - 90.0) < 0.01

    def test_wraparound(self):
        # Mean of 350° and 10° should be ~0° (not 180°)
        result = _circular_mean(np.array([350.0, 10.0]))
        assert result < 20.0 or result > 340.0

    def test_empty_returns_zero(self):
        assert _circular_mean(np.array([])) == 0.0

    def test_single_value(self):
        assert abs(_circular_mean(np.array([45.0])) - 45.0) < 0.01


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------

class TestEndToEnd:

    def test_preprocess_noaa_day(self, small_csv):
        timelines = preprocess_noaa_day(
            small_csv, n_vessels=2, min_obs=5, seed=42,
        )
        assert isinstance(timelines, dict)
        assert len(timelines) <= 2
        for records in timelines.values():
            assert len(records) >= 1
            assert "speed_kmh" in records[0]

    def test_with_parquet_output(self, small_csv, tmp_path):
        out = tmp_path / "test_output.parquet"
        timelines = preprocess_noaa_day(
            small_csv, n_vessels=2, min_obs=5, seed=42,
            output_parquet=str(out),
        )
        assert out.exists()
        reloaded = pd.read_parquet(out)
        assert len(reloaded) > 0
        assert "mmsi" in reloaded.columns
