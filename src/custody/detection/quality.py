"""Pre-flight SAR scene quality screening.

Quick intensity-stats sanity check run before committing $$ to VLM inference
on a scene.  Catches uniformly low-contrast / low-SNR acquisitions that would
produce zero detections anyway.  Sub-second per scene via rasterio overviews
or decimated reads; numpy + rasterio only.

The module is pure: it returns a :class:`SceneQualityReport`.  Callers decide
what to do with the flag (green / yellow / red).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import rasterio


@dataclass(frozen=True)
class IntensityStats:
    """Summary intensity statistics over the non-zero pixels of a SAR image."""

    mean: float
    std: float
    p50: float
    p90: float
    p99: float
    dynamic_range: float  # p99 - p50


@dataclass(frozen=True)
class SceneQualityReport:
    """Pre-flight quality report for one SAR scene."""

    scene_path: str
    full_scene_stats: IntensityStats
    aoi_stats: IntensityStats | None
    quality_flag: Literal["green", "yellow", "red"]
    flag_reasons: list[str]


# Minimum non-zero pixel count before we trust the zero-masked stats.  Below
# this we fall back to full-array stats and emit a reason.
_MIN_NONZERO_PIXELS = 1024


def _compute_stats(arr: np.ndarray, reasons: list[str]) -> IntensityStats:
    """Compute :class:`IntensityStats` over non-zero pixels of ``arr``.

    Falls back to the full array (and records a reason) if too few non-zero
    pixels remain after masking.
    """
    flat = arr.ravel()
    nonzero = flat[flat > 0]
    if nonzero.size < _MIN_NONZERO_PIXELS:
        reasons.append(
            f"too few non-zero pixels ({nonzero.size}) — using full array"
        )
        nonzero = flat
    p50, p90, p99 = np.percentile(nonzero, [50.0, 90.0, 99.0])
    return IntensityStats(
        mean=float(nonzero.mean()),
        std=float(nonzero.std()),
        p50=float(p50),
        p90=float(p90),
        p99=float(p99),
        dynamic_range=float(p99 - p50),
    )


def assess_scene_quality(
    scene_path: Path | str,
    aoi_bounds: tuple[int, int, int, int] | None = None,
    decimation: int = 8,
) -> SceneQualityReport:
    """Compute full-scene (and optionally AOI) intensity stats and flag quality.

    ``aoi_bounds`` is ``(row_start, row_end, col_start, col_end)`` in
    *full-resolution* pixel coordinates; the AOI region is extracted from the
    *decimated* array, so no second full-resolution read happens.
    """
    scene_path = Path(scene_path)
    reasons: list[str] = []

    with rasterio.open(scene_path) as src:
        h, w = src.height, src.width
        out_h = max(1, h // decimation)
        out_w = max(1, w // decimation)
        img = src.read(1, out_shape=(out_h, out_w))

    full_stats = _compute_stats(img, reasons)

    aoi_stats: IntensityStats | None = None
    if aoi_bounds is not None:
        r0, r1, c0, c1 = aoi_bounds
        r0d = r0 // decimation
        r1d = max(r0d + 1, r1 // decimation)
        c0d = c0 // decimation
        c1d = max(c0d + 1, c1 // decimation)
        aoi_arr = img[r0d:r1d, c0d:c1d]
        if aoi_arr.size == 0:
            reasons.append(
                f"aoi_bounds {aoi_bounds} collapsed to empty region after decimation"
            )
        else:
            aoi_stats = _compute_stats(aoi_arr, reasons)

    flag, threshold_reasons = _flag_from_stats(full_stats, aoi_stats)
    reasons.extend(threshold_reasons)

    return SceneQualityReport(
        scene_path=str(scene_path),
        full_scene_stats=full_stats,
        aoi_stats=aoi_stats,
        quality_flag=flag,
        flag_reasons=reasons,
    )


# Thresholds calibrated from the 7-scene dataset (2026-04-21).  The calibration
# script `day0/scratch/quality_calibration_run.py` prints the per-scene stats;
# the thresholds below separate the 08-09 low-SNR null from the six healthy
# scenes with comfortable margins.
#
# dynamic_range (p99 - p50) is the separator, not p99 alone — at uint8 GEC the
# full-scene p99 of the 08-09 null (58) is nearly identical to healthy Whitsun
# full-scene p99 (60-61), so any p99-only rule would either false-flag Whitsun
# or miss 08-09.  dynamic_range cleanly discriminates: 08-09 DR=9 vs Whitsun
# DR=12 vs healthy Tennent full DR=41+ / AOI DR=35+.
#
# AOI stats are systematically brighter than full-scene stats for Tennent
# (the reef concentrates the bright-return population into the AOI), so AOI
# thresholds can be tighter than full-scene thresholds.  When no AOI is given
# (e.g. Whitsun full-scene runs), we fall back to full-scene thresholds.
_AOI_DR_RED = 15.0
_AOI_DR_YELLOW = 25.0
_FULL_DR_RED = 10.0
_FULL_DR_YELLOW = 12.0


def _flag_from_stats(
    full_stats: IntensityStats,
    aoi_stats: IntensityStats | None,
) -> tuple[Literal["green", "yellow", "red"], list[str]]:
    """Map intensity stats to quality flag via calibrated dynamic-range thresholds."""
    reasons: list[str] = []
    if aoi_stats is not None:
        dr = aoi_stats.dynamic_range
        source = "AOI"
        red_t, yellow_t = _AOI_DR_RED, _AOI_DR_YELLOW
    else:
        dr = full_stats.dynamic_range
        source = "full-scene"
        red_t, yellow_t = _FULL_DR_RED, _FULL_DR_YELLOW
    if dr < red_t:
        reasons.append(
            f"{source} dynamic_range {dr:.1f} below red threshold {red_t:.1f}"
        )
        return "red", reasons
    if dr < yellow_t:
        reasons.append(
            f"{source} dynamic_range {dr:.1f} below yellow threshold {yellow_t:.1f}"
        )
        return "yellow", reasons
    return "green", reasons
