"""Tests for :mod:`custody.detection.quality`.

Unit tests exercise the stat-computation and flag-threshold logic on
synthetic arrays.  Integration tests run against two real scenes to confirm
the calibrated thresholds separate the Tennent 2023-07-02 healthy scene
(green) from the Tennent 2023-08-09 low-SNR null (red).

Integration tests skip automatically if the raw scene files aren't on disk
(CI / fresh clones without Umbra mirror).  Skip logic keeps the fast-path
unit tests exercisable without the ~140 GB raw-data dependency.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from custody.detection.quality import (
    IntensityStats,
    SceneQualityReport,
    _compute_stats,
    _flag_from_stats,
    assess_scene_quality,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCENE_07_02 = (
    REPO_ROOT / "data/raw/umbra/sar-data/tasks/ship_detection_testdata"
    / "f0730a1d-2bf7-4193-b2fe-ec8bfb2e1aef/2023-07-02-14-00-55_UMBRA-05"
    / "2023-07-02-14-00-55_UMBRA-05_GEC.tif"
)
SCENE_08_09 = (
    REPO_ROOT / "data/raw/umbra/sar-data/tasks/ship_detection_testdata"
    / "b422de0d-b6fc-4340-aa85-6c87f3805175/2023-08-09-02-23-25_UMBRA-06"
    / "2023-08-09-02-23-25_UMBRA-06_GEC.tif"
)


# ---------------------------------------------------------------------------
# Unit tests — stat computation
# ---------------------------------------------------------------------------


def test_compute_stats_on_non_zero_pixels():
    """Stats are computed on non-zero pixels only; zeros are excluded."""
    arr = np.zeros((100, 100), dtype=np.uint8)
    arr[0:50, :] = 50           # half uniform at 50
    arr[50:100, :] = 150        # half uniform at 150
    stats = _compute_stats(arr, reasons=[])
    # Median of [50 × 5000, 150 × 5000] is 100 (boundary between halves) — but
    # numpy percentile interpolates, so p50 is exactly 100.
    assert stats.p50 == pytest.approx(100.0)
    assert stats.p99 == pytest.approx(150.0)
    assert stats.dynamic_range == pytest.approx(50.0)
    assert stats.mean == pytest.approx(100.0)


def test_compute_stats_falls_back_when_too_few_non_zero_pixels():
    """With fewer than the minimum non-zero pixels, stats fall back to full array."""
    arr = np.zeros((100, 100), dtype=np.uint8)
    arr[0, 0:10] = 200  # only 10 non-zero pixels
    reasons: list[str] = []
    stats = _compute_stats(arr, reasons)
    # Full-array stats: 9990 zeros + 10 of value 200 → p50 == 0
    assert stats.p50 == pytest.approx(0.0)
    assert any("too few non-zero pixels" in r for r in reasons)


def test_compute_stats_dynamic_range_matches_p99_minus_p50():
    """dynamic_range field must equal p99 - p50."""
    rng = np.random.default_rng(0)
    arr = rng.integers(1, 200, size=(200, 200), dtype=np.uint8)
    stats = _compute_stats(arr, reasons=[])
    assert stats.dynamic_range == pytest.approx(stats.p99 - stats.p50, abs=1e-9)


# ---------------------------------------------------------------------------
# Unit tests — flag logic
# ---------------------------------------------------------------------------


def _mk_stats(mean=50.0, std=5.0, p50=50.0, p90=55.0, p99=58.0, dr=8.0) -> IntensityStats:
    return IntensityStats(mean=mean, std=std, p50=p50, p90=p90, p99=p99, dynamic_range=dr)


def test_flag_aoi_red_below_15():
    """AOI dynamic_range below 15 → red (calibrated for 08-09's DR=9)."""
    aoi = _mk_stats(dr=9.0)
    full = _mk_stats(dr=9.0)
    flag, reasons = _flag_from_stats(full, aoi)
    assert flag == "red"
    assert any("AOI" in r and "dynamic_range" in r and "9.0" in r for r in reasons)


def test_flag_aoi_yellow_15_to_25():
    """AOI dynamic_range in [15, 25) → yellow."""
    aoi = _mk_stats(dr=20.0)
    flag, _ = _flag_from_stats(_mk_stats(), aoi)
    assert flag == "yellow"


def test_flag_aoi_green_at_or_above_25():
    """AOI dynamic_range >= 25 → green (Tennent 08-07 at DR=35 must pass)."""
    aoi = _mk_stats(dr=35.0)
    flag, reasons = _flag_from_stats(_mk_stats(), aoi)
    assert flag == "green"
    assert reasons == []


def test_flag_full_scene_when_no_aoi_red_below_10():
    """Full-scene DR below 10 → red (matches 08-09's 9 when AOI not given)."""
    full = _mk_stats(dr=9.0)
    flag, reasons = _flag_from_stats(full, None)
    assert flag == "red"
    assert any("full-scene" in r for r in reasons)


def test_flag_full_scene_whitsun_healthy_dr_12_is_green():
    """Whitsun healthy full-scene DR=12 must pass as green (boundary case).

    This is the critical calibration point: if the threshold were >= 12, the
    two real Whitsun scenes would false-flag as yellow.  Threshold uses strict
    less-than so DR=12 is green.
    """
    full = _mk_stats(dr=12.0)
    flag, _ = _flag_from_stats(full, None)
    assert flag == "green"


def test_flag_full_scene_yellow_at_11():
    """Full-scene DR of 11 (between 10 and 12) → yellow."""
    full = _mk_stats(dr=11.0)
    flag, _ = _flag_from_stats(full, None)
    assert flag == "yellow"


def test_flag_prefers_aoi_over_full_scene_when_both_present():
    """When AOI is given, the flag is based on AOI stats, not full-scene."""
    full = _mk_stats(dr=50.0)   # healthy full scene
    aoi = _mk_stats(dr=5.0)     # low-SNR AOI — should dominate
    flag, reasons = _flag_from_stats(full, aoi)
    assert flag == "red"
    assert any("AOI" in r for r in reasons)


# ---------------------------------------------------------------------------
# Integration tests — real scenes
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not SCENE_07_02.exists(),
    reason=f"Scene not on disk: {SCENE_07_02}",
)
def test_real_scene_tennent_20230702_flags_green():
    """Tennent 2023-07-02 — healthy baseline scene; expect green under Tennent AOI."""
    # 2 km AOI centered on Tennent Reef — derive pixel bounds via rasterio.
    from custody.detection.sar_common import crop_to_aoi, read_geotiff
    img, transform, crs_wkt = read_geotiff(SCENE_07_02)
    _, _, bounds = crop_to_aoi(
        img, transform, 8.855687, 114.665145, crs_wkt, box_half_km=1.0,
    )
    aoi_bounds = (
        bounds["row_start"], bounds["row_end"],
        bounds["col_start"], bounds["col_end"],
    )
    report = assess_scene_quality(SCENE_07_02, aoi_bounds=aoi_bounds)
    assert report.quality_flag == "green", (
        f"07-02 expected green; got {report.quality_flag} with reasons {report.flag_reasons}"
    )
    # Sanity: AOI dynamic_range should be well above the 15 m threshold
    assert report.aoi_stats is not None
    assert report.aoi_stats.dynamic_range >= 25.0


@pytest.mark.skipif(
    not SCENE_08_09.exists(),
    reason=f"Scene not on disk: {SCENE_08_09}",
)
def test_real_scene_tennent_20230809_flags_red():
    """Tennent 2023-08-09 — low-SNR null acquisition; expect red under Tennent AOI."""
    from custody.detection.sar_common import crop_to_aoi, read_geotiff
    img, transform, crs_wkt = read_geotiff(SCENE_08_09)
    _, _, bounds = crop_to_aoi(
        img, transform, 8.855687, 114.665145, crs_wkt, box_half_km=1.0,
    )
    aoi_bounds = (
        bounds["row_start"], bounds["row_end"],
        bounds["col_start"], bounds["col_end"],
    )
    report = assess_scene_quality(SCENE_08_09, aoi_bounds=aoi_bounds)
    assert report.quality_flag == "red", (
        f"08-09 expected red; got {report.quality_flag} with reasons {report.flag_reasons}"
    )
    assert report.aoi_stats is not None
    assert report.aoi_stats.dynamic_range < 15.0
    assert any("red threshold" in r for r in report.flag_reasons)


# ---------------------------------------------------------------------------
# Synthetic end-to-end — no raw-scene dependency
# ---------------------------------------------------------------------------


def test_assess_scene_quality_end_to_end_on_synthetic_tif(tmp_path: Path):
    """Write a synthetic GeoTIFF with a bright feature, run the screen end-to-end."""
    H = W = 2000
    arr = np.full((H, W), 50, dtype=np.uint8)         # uniform water speckle
    arr[800:1200, 800:1200] = 200                      # bright structure in the center
    path = tmp_path / "synthetic.tif"
    with rasterio.open(
        path, "w", driver="GTiff",
        height=H, width=W, count=1, dtype=np.uint8,
        crs="EPSG:32650",
        transform=from_origin(500_000, 1_000_000, 1.0, 1.0),
    ) as dst:
        dst.write(arr, 1)
    # AOI covering the bright structure only
    aoi_bounds = (800, 1200, 800, 1200)
    report = assess_scene_quality(path, aoi_bounds=aoi_bounds)
    # AOI is pure 200 → p99=p50=200 → dynamic_range=0 → red.
    # That's honest: an AOI of uniform pixels has no dynamic range.  The
    # screen flags it, which is the desired behavior — flat imagery never
    # produces detectable features.
    assert report.quality_flag == "red"
    assert report.aoi_stats is not None
    assert report.aoi_stats.dynamic_range == pytest.approx(0.0)


def test_assess_scene_quality_without_aoi_uses_full_scene_thresholds(tmp_path: Path):
    """No aoi_bounds → full-scene thresholds apply (looser, calibrated for Whitsun)."""
    H = W = 2000
    rng = np.random.default_rng(42)
    # Synthetic "water only" scene with speckle — full-scene DR low, should flag
    # (mimics a Whitsun-like coverage with no vessels).
    arr = rng.integers(45, 55, size=(H, W), dtype=np.uint8)
    path = tmp_path / "synthetic_no_aoi.tif"
    with rasterio.open(
        path, "w", driver="GTiff",
        height=H, width=W, count=1, dtype=np.uint8,
        crs="EPSG:32650",
        transform=from_origin(500_000, 1_000_000, 1.0, 1.0),
    ) as dst:
        dst.write(arr, 1)
    report = assess_scene_quality(path, aoi_bounds=None)
    assert report.aoi_stats is None
    # DR is ~9 here (50→55 roughly at the quantiles) — should trip red under
    # full-scene thresholds (< 10).
    assert report.quality_flag == "red"


def test_scene_quality_report_is_frozen_dataclass():
    """Report is immutable — callers can't mutate the flag after computation."""
    stats = _mk_stats()
    report = SceneQualityReport(
        scene_path="x", full_scene_stats=stats, aoi_stats=None,
        quality_flag="green", flag_reasons=[],
    )
    with pytest.raises(AttributeError):
        report.quality_flag = "red"  # type: ignore[misc]
