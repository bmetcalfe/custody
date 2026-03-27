"""Inspection and analysis helpers for ML anomaly scores.

Operates on scored parquet DataFrames produced by the train/score
pipeline.  All functions are pure — no side effects, no plotting.

Public API
----------
load_scored_parquet(path) -> pd.DataFrame
top_anomalous_records(df, n=50) -> pd.DataFrame
top_anomalous_vessels(df, n=20) -> pd.DataFrame
vessel_time_series(df, vessel_id) -> pd.DataFrame
vessel_summary(df) -> pd.DataFrame
anomaly_excursions(df, threshold=0.8) -> pd.DataFrame
contiguous_windows(df, threshold=0.8, min_length=2) -> list[dict]
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_scored_parquet(path: str | Path) -> pd.DataFrame:
    """Load a scored parquet and ensure standard column types.

    Returns DataFrame sorted by (mmsi, timestamp).
    """
    df = pd.read_parquet(path)
    df.sort_values(["mmsi", "timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ---------------------------------------------------------------------------
# Record-level
# ---------------------------------------------------------------------------

_RECORD_COLS = [
    "mmsi", "timestamp", "lat", "lon", "speed_kmh", "heading_deg",
    "ml_anomaly_score",
]


def top_anomalous_records(df: pd.DataFrame, n: int = 50) -> pd.DataFrame:
    """Return the *n* highest-scoring records.

    Includes feature deltas if present in the DataFrame.
    """
    if df.empty:
        return df.copy()
    cols = [c for c in _RECORD_COLS if c in df.columns]
    for extra in ["speed_delta", "heading_delta", "lat_delta", "lon_delta"]:
        if extra in df.columns:
            cols.append(extra)
    out = df.nlargest(n, "ml_anomaly_score")[cols].copy()
    out.reset_index(drop=True, inplace=True)
    return out


# ---------------------------------------------------------------------------
# Vessel-level
# ---------------------------------------------------------------------------

def vessel_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-vessel anomaly score summary.

    Returns DataFrame with columns: mmsi, count, mean, max, p95,
    peak_timestamp, peak_score.  Sorted by max descending.
    """
    def _agg(g):
        scores = g["ml_anomaly_score"]
        peak_idx = scores.idxmax()
        return pd.Series({
            "count":          len(g),
            "mean":           scores.mean(),
            "max":            scores.max(),
            "p95":            scores.quantile(0.95) if len(g) >= 2 else scores.max(),
            "peak_timestamp": g.loc[peak_idx, "timestamp"],
            "peak_score":     scores.max(),
            "peak_speed":     g.loc[peak_idx, "speed_kmh"],
            "peak_lat":       g.loc[peak_idx, "lat"],
            "peak_lon":       g.loc[peak_idx, "lon"],
        })

    if df.empty:
        return pd.DataFrame(columns=[
            "mmsi", "count", "mean", "max", "p95",
            "peak_timestamp", "peak_score", "peak_speed", "peak_lat", "peak_lon",
        ])
    summary = df.groupby("mmsi").apply(_agg, include_groups=False)
    summary.reset_index(inplace=True)
    summary.sort_values("max", ascending=False, inplace=True)
    summary.reset_index(drop=True, inplace=True)
    return summary


def top_anomalous_vessels(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Return vessel summary for the top *n* most anomalous vessels."""
    return vessel_summary(df).head(n)


# ---------------------------------------------------------------------------
# Time series
# ---------------------------------------------------------------------------

def vessel_time_series(df: pd.DataFrame, vessel_id: int | str) -> pd.DataFrame:
    """Extract all records for one vessel, sorted by time.

    Args:
        df:         Scored DataFrame.
        vessel_id:  MMSI as int or string.

    Returns:
        DataFrame with all columns for that vessel, time-sorted.
        Empty DataFrame if vessel not found.
    """
    vid = int(vessel_id)
    out = df[df["mmsi"] == vid].sort_values("timestamp").copy()
    out.reset_index(drop=True, inplace=True)
    return out


# ---------------------------------------------------------------------------
# Excursion detection
# ---------------------------------------------------------------------------

def anomaly_excursions(
    df: pd.DataFrame,
    threshold: float = 0.8,
) -> pd.DataFrame:
    """Return all records where ml_anomaly_score >= threshold.

    Adds an ``excursion`` boolean column to the result.
    """
    exc = df[df["ml_anomaly_score"] >= threshold].copy()
    exc.reset_index(drop=True, inplace=True)
    return exc


def contiguous_windows(
    df: pd.DataFrame,
    threshold: float = 0.8,
    min_length: int = 2,
) -> list[dict]:
    """Identify contiguous above-threshold anomaly windows per vessel.

    Args:
        df:          Scored DataFrame sorted by (mmsi, timestamp).
        threshold:   Score threshold for inclusion.
        min_length:  Minimum window length in timesteps.

    Returns:
        List of dicts with keys: mmsi, start, end, length, max_score,
        mean_score.  Sorted by max_score descending.
    """
    windows: list[dict] = []

    for mmsi, group in df.groupby("mmsi"):
        group = group.sort_values("timestamp")
        above = group["ml_anomaly_score"] >= threshold
        # Find contiguous runs
        run_start = None
        run_scores: list[float] = []
        run_times: list = []

        for idx, (_, row) in enumerate(group.iterrows()):
            if above.iloc[idx]:
                if run_start is None:
                    run_start = row["timestamp"]
                run_scores.append(row["ml_anomaly_score"])
                run_times.append(row["timestamp"])
            else:
                if run_start is not None and len(run_scores) >= min_length:
                    windows.append({
                        "mmsi":       int(mmsi),
                        "start":      run_start,
                        "end":        run_times[-1],
                        "length":     len(run_scores),
                        "max_score":  max(run_scores),
                        "mean_score": sum(run_scores) / len(run_scores),
                    })
                run_start = None
                run_scores = []
                run_times = []

        # Close final window
        if run_start is not None and len(run_scores) >= min_length:
            windows.append({
                "mmsi":       int(mmsi),
                "start":      run_start,
                "end":        run_times[-1],
                "length":     len(run_scores),
                "max_score":  max(run_scores),
                "mean_score": sum(run_scores) / len(run_scores),
            })

    windows.sort(key=lambda w: w["max_score"], reverse=True)
    return windows


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------

def markdown_summary(df: pd.DataFrame, top_n: int = 5) -> str:
    """Produce a compact markdown summary of the top anomalous vessels.

    Args:
        df:     Scored DataFrame.
        top_n:  Number of vessels to include.

    Returns:
        Markdown string.
    """
    vs = top_anomalous_vessels(df, n=top_n)
    lines = [f"## Top {top_n} Anomalous Vessels\n"]
    lines.append("| MMSI | Records | Mean | Max | p95 | Peak Time | Peak Speed |")
    lines.append("|------|---------|------|-----|-----|-----------|------------|")
    for _, row in vs.iterrows():
        ts = str(row["peak_timestamp"]).split("+")[0]
        lines.append(
            f"| {int(row['mmsi'])} | {int(row['count'])} | "
            f"{row['mean']:.3f} | {row['max']:.3f} | {row['p95']:.3f} | "
            f"{ts} | {row['peak_speed']:.1f} km/h |"
        )

    wins = contiguous_windows(df, threshold=0.8, min_length=2)
    if wins:
        lines.append(f"\n## Sustained Anomaly Windows (score >= 0.8, length >= 2h)\n")
        lines.append("| MMSI | Start | End | Length | Max | Mean |")
        lines.append("|------|-------|-----|--------|-----|------|")
        for w in wins[:10]:
            s = str(w["start"]).split("+")[0]
            e = str(w["end"]).split("+")[0]
            lines.append(
                f"| {w['mmsi']} | {s} | {e} | {w['length']}h | "
                f"{w['max_score']:.3f} | {w['mean_score']:.3f} |"
            )

    return "\n".join(lines)
