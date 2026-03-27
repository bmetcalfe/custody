"""Per-vessel relative anomaly scoring.

Computes vessel-specific baselines from historical data and produces a
relative anomaly score that measures deviation from each vessel's normal
behavior rather than a fixed global threshold.

Design:
  - ``VesselBaseline`` stores per-vessel mean, std, and percentiles
    computed from training/historical data.
  - ``build_baselines()`` creates baselines from a scored DataFrame.
  - ``compute_relative_score()`` transforms an absolute ML score into
    a vessel-relative score in [0, 1].
  - ``add_relative_scores()`` enriches a DataFrame with the new field.

The relative score uses a percentile-rank approach: for each vessel,
the score is mapped to where it falls in that vessel's historical
distribution.  A score at the vessel's 95th percentile → relative 0.95.
This naturally adapts to each vessel's range.

Public API
----------
VesselBaseline (frozen dataclass)
build_baselines(df) -> dict[int, VesselBaseline]
compute_relative_score(score, baseline) -> float
add_relative_scores(df, baselines) -> pd.DataFrame
save_baselines(baselines, path)
load_baselines(path) -> dict[int, VesselBaseline]
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VesselBaseline:
    """Per-vessel score distribution summary.

    Attributes:
        mmsi:   Vessel identifier.
        mean:   Mean ml_anomaly_score in baseline period.
        std:    Std dev of ml_anomaly_score.
        p50:    Median.
        p90:    90th percentile.
        p95:    95th percentile.
        p99:    99th percentile.
        count:  Number of baseline records.
    """
    mmsi: int
    mean: float
    std: float
    p50: float
    p90: float
    p95: float
    p99: float
    count: int


def build_baselines(
    df: pd.DataFrame,
    score_col: str = "ml_anomaly_score",
    min_records: int = 10,
) -> dict[int, VesselBaseline]:
    """Compute per-vessel baselines from a scored DataFrame.

    Args:
        df:          DataFrame with mmsi and score_col columns.
        score_col:   Column name for the anomaly score.
        min_records: Minimum records required to build a baseline.
                     Vessels with fewer records are excluded.

    Returns:
        Dict mapping mmsi (int) → VesselBaseline.
    """
    baselines: dict[int, VesselBaseline] = {}
    for mmsi, group in df.groupby("mmsi"):
        scores = group[score_col].dropna()
        if len(scores) < min_records:
            continue
        baselines[int(mmsi)] = VesselBaseline(
            mmsi=int(mmsi),
            mean=float(scores.mean()),
            std=float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
            p50=float(scores.quantile(0.50)),
            p90=float(scores.quantile(0.90)),
            p95=float(scores.quantile(0.95)),
            p99=float(scores.quantile(0.99)),
            count=len(scores),
        )
    return baselines


def compute_relative_score(
    score: float,
    baseline: VesselBaseline,
) -> float:
    """Transform an absolute ML score into a vessel-relative score [0, 1].

    Uses a percentile-interpolation approach:
      - score <= p50 → maps to [0.0, 0.5]
      - score in (p50, p90] → maps to [0.5, 0.9]
      - score in (p90, p95] → maps to [0.9, 0.95]
      - score in (p95, p99] → maps to [0.95, 0.99]
      - score > p99 → maps to [0.99, 1.0]

    This ensures that a score at a vessel's 95th percentile always maps
    to ~0.95, regardless of the vessel's absolute score range.

    Args:
        score:    Absolute ml_anomaly_score.
        baseline: Vessel's historical baseline.

    Returns:
        Relative score in [0, 1].
    """
    # Interpolation breakpoints: (threshold, output)
    points = [
        (0.0,          0.0),
        (baseline.p50, 0.50),
        (baseline.p90, 0.90),
        (baseline.p95, 0.95),
        (baseline.p99, 0.99),
    ]

    # Handle edge case: all scores identical (std=0)
    if baseline.std == 0.0:
        return 0.5 if score <= baseline.mean else 1.0

    # Find the segment and interpolate
    for i in range(len(points) - 1):
        lo_score, lo_out = points[i]
        hi_score, hi_out = points[i + 1]
        if score <= hi_score:
            span = hi_score - lo_score
            if span <= 0:
                return hi_out
            t = (score - lo_score) / span
            return lo_out + t * (hi_out - lo_out)

    # Above p99: extrapolate toward 1.0
    if baseline.p99 > baseline.p95:
        excess = (score - baseline.p99) / (baseline.p99 - baseline.p95)
        return min(0.99 + 0.01 * excess, 1.0)
    return 1.0


def add_relative_scores(
    df: pd.DataFrame,
    baselines: dict[int, VesselBaseline],
    score_col: str = "ml_anomaly_score",
    output_col: str = "ml_anomaly_relative",
) -> pd.DataFrame:
    """Add per-vessel relative anomaly scores to a DataFrame.

    Vessels without a baseline get NaN in the output column.

    Args:
        df:         Scored DataFrame with mmsi and score_col.
        baselines:  Dict from :func:`build_baselines`.
        score_col:  Input score column.
        output_col: Output column name.

    Returns:
        Copy of df with the new column added.
    """
    out = df.copy()
    relative = np.full(len(out), np.nan)

    for i, row in out.iterrows():
        mmsi = int(row["mmsi"])
        bl = baselines.get(mmsi)
        if bl is not None:
            relative[i] = compute_relative_score(float(row[score_col]), bl)

    out[output_col] = relative
    return out


def save_baselines(baselines: dict[int, VesselBaseline], path: str | Path) -> None:
    """Save baselines to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(baselines, path)


def load_baselines(path: str | Path) -> dict[int, VesselBaseline]:
    """Load baselines from disk."""
    return joblib.load(path)
