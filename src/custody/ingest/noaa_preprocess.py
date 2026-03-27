"""Preprocess raw NOAA AIS daily CSV into Custody-friendly vessel timelines.

Reads a single-day NOAA AIS CSV (~8M rows, 17 columns), reduces to core
fields, cleans, resamples to hourly cadence per vessel, and outputs a
compact dict of vessel timelines ready for ML feature extraction or
direct ingestion via custody.ais.

Usage as a library:

    from custody.ingest.noaa_preprocess import preprocess_noaa_day
    timelines = preprocess_noaa_day("ais-2024-09-24.csv", n_vessels=30)

Usage as a CLI:

    uv run python -m custody.ingest.noaa_preprocess \\
        --input ais-2024-09-24.csv \\
        --output timelines.parquet \\
        --n-vessels 30 \\
        --min-obs 12

Design rules:
  - Memory-conscious: reads only needed columns, processes in one pass.
  - Deterministic: same input + seed → same output.
  - Configurable: vessel count, min observations, bounding box, seed.
  - Output compatible with existing custody.ais.AISObservation schema.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Column mapping: NOAA raw → internal names
# ---------------------------------------------------------------------------

_NOAA_COLUMNS = {
    "mmsi":           "mmsi",
    "base_date_time": "timestamp",
    "latitude":       "lat",
    "longitude":      "lon",
    "sog":            "speed_knots",  # speed over ground in knots
    "cog":            "heading_deg",  # course over ground in degrees
}

_USE_COLS = list(_NOAA_COLUMNS.keys())

_KNOTS_TO_KMH = 1.852


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_raw(
    path: str | Path,
    bbox: Optional[tuple[float, float, float, float]] = None,
) -> pd.DataFrame:
    """Read a NOAA AIS daily CSV, keeping only core columns.

    Args:
        path:  File path to the raw CSV.
        bbox:  Optional (min_lat, max_lat, min_lon, max_lon) bounding box.
               Rows outside the box are dropped immediately after read.

    Returns:
        DataFrame with columns: mmsi, timestamp, lat, lon, speed_knots, heading_deg.
        Sorted by mmsi, timestamp.
    """
    df = pd.read_csv(
        path,
        usecols=_USE_COLS,
        dtype={"mmsi": "int64"},
        parse_dates=["base_date_time"],
    )
    df.rename(columns=_NOAA_COLUMNS, inplace=True)

    # Timezone-aware UTC
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")

    # Bounding-box filter (before any further processing)
    if bbox is not None:
        min_lat, max_lat, min_lon, max_lon = bbox
        df = df[
            (df["lat"] >= min_lat) & (df["lat"] <= max_lat)
            & (df["lon"] >= min_lon) & (df["lon"] <= max_lon)
        ]

    df.sort_values(["mmsi", "timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize a raw AIS DataFrame.

    - Fills speed_knots nulls with 0.0 (stationary assumption).
    - Forward-fills heading_deg per vessel (course doesn't change if
      speed is zero); remaining nulls filled with 0.0.
    - Drops rows still missing lat or lon.
    - Adds speed_kmh column.

    Args:
        df: Output of :func:`load_raw`.

    Returns:
        Cleaned DataFrame with added ``speed_kmh`` column.
    """
    df = df.copy()
    df["speed_knots"] = df["speed_knots"].fillna(0.0)
    df["heading_deg"] = df.groupby("mmsi")["heading_deg"].ffill().fillna(0.0)
    df.dropna(subset=["lat", "lon"], inplace=True)
    df["speed_kmh"] = df["speed_knots"] * _KNOTS_TO_KMH
    return df


def select_vessels(
    df: pd.DataFrame,
    n_vessels: int = 30,
    min_obs: int = 12,
    seed: int = 42,
) -> pd.DataFrame:
    """Select a random subset of vessels with sufficient observations.

    Args:
        df:         Cleaned AIS DataFrame.
        n_vessels:  Target number of vessels to keep.
        min_obs:    Minimum raw observations required per vessel.
        seed:       Random seed for reproducibility.

    Returns:
        DataFrame filtered to the selected vessels.
    """
    counts = df.groupby("mmsi").size()
    eligible = counts[counts >= min_obs].index
    rng = np.random.default_rng(seed)
    chosen = rng.choice(eligible, size=min(n_vessels, len(eligible)), replace=False)
    return df[df["mmsi"].isin(chosen)].copy()


def resample_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample each vessel's track to hourly cadence.

    Aggregation per hour:
      - lat, lon: mean position (centroid of reports in that hour)
      - speed_knots, speed_kmh: mean speed
      - heading_deg: circular mean heading

    Hours with no observations are dropped (no interpolation).

    Args:
        df: Cleaned, vessel-filtered DataFrame.

    Returns:
        DataFrame with one row per (vessel, hour), sorted by mmsi, timestamp.
    """
    df = df.copy()
    df.set_index("timestamp", inplace=True)

    parts = []
    for mmsi, group in df.groupby("mmsi"):
        hourly = group.resample("1h").agg({
            "mmsi":         "first",
            "lat":          "mean",
            "lon":          "mean",
            "speed_knots":  "mean",
            "speed_kmh":    "mean",
            "heading_deg":  lambda h: _circular_mean(h.dropna().values),
        })
        hourly.dropna(subset=["lat"], inplace=True)
        hourly["mmsi"] = mmsi
        parts.append(hourly)

    if not parts:
        return pd.DataFrame(columns=df.columns)

    result = pd.concat(parts)
    result.reset_index(inplace=True)
    result.sort_values(["mmsi", "timestamp"], inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


def to_timelines(df: pd.DataFrame) -> dict[str, list[dict]]:
    """Convert a resampled DataFrame to Custody-friendly timeline dicts.

    Args:
        df: Hourly-resampled DataFrame from :func:`resample_hourly`.

    Returns:
        Dict keyed by MMSI string.  Each value is a list of record dicts
        with keys: timestamp, lat, lon, speed_kmh, heading_deg, source.
        Timestamps are datetime objects (UTC-aware).
    """
    timelines: dict[str, list[dict]] = {}
    for mmsi, group in df.groupby("mmsi"):
        key = str(int(mmsi))
        records = []
        for _, row in group.iterrows():
            records.append({
                "timestamp": row["timestamp"],
                "lat":        round(float(row["lat"]), 6),
                "lon":        round(float(row["lon"]), 6),
                "speed_kmh":  round(float(row["speed_kmh"]), 2),
                "heading_deg": round(float(row["heading_deg"]), 1),
                "source":     "ais",
            })
        timelines[key] = records
    return timelines


def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """Save a resampled DataFrame to Parquet for efficient reload.

    Args:
        df:   Hourly-resampled DataFrame.
        path: Output file path (.parquet).
    """
    df.to_parquet(path, index=False)


def load_parquet(path: str | Path) -> pd.DataFrame:
    """Load a previously saved Parquet file.

    Args:
        path: Path to the .parquet file.

    Returns:
        DataFrame matching :func:`resample_hourly` output schema.
    """
    return pd.read_parquet(path)


def preprocess_noaa_day(
    input_path: str | Path,
    n_vessels: int = 30,
    min_obs: int = 12,
    seed: int = 42,
    bbox: Optional[tuple[float, float, float, float]] = None,
    output_parquet: Optional[str | Path] = None,
) -> dict[str, list[dict]]:
    """End-to-end preprocessing pipeline for one NOAA AIS daily CSV.

    Args:
        input_path:      Path to the raw NOAA CSV.
        n_vessels:        Number of vessels to sample.
        min_obs:          Minimum raw observations per vessel.
        seed:             Random seed.
        bbox:             Optional (min_lat, max_lat, min_lon, max_lon).
        output_parquet:   If set, save the resampled DataFrame here.

    Returns:
        Dict of vessel timelines (see :func:`to_timelines`).
    """
    raw = load_raw(input_path, bbox=bbox)
    cleaned = clean(raw)
    subset = select_vessels(cleaned, n_vessels=n_vessels, min_obs=min_obs, seed=seed)
    hourly = resample_hourly(subset)
    if output_parquet:
        save_parquet(hourly, output_parquet)
    return to_timelines(hourly)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _circular_mean(angles_deg: np.ndarray) -> float:
    """Compute the circular (angular) mean of an array of degree values.

    Returns the mean angle in [0, 360).  Returns 0.0 for empty input.
    """
    if len(angles_deg) == 0:
        return 0.0
    rads = np.radians(angles_deg)
    mean_sin = np.mean(np.sin(rads))
    mean_cos = np.mean(np.cos(rads))
    mean_rad = np.arctan2(mean_sin, mean_cos)
    return float(np.degrees(mean_rad) % 360)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def _cli():
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess a NOAA AIS daily CSV into Custody-friendly timelines.",
    )
    parser.add_argument("--input", required=True, help="Path to raw NOAA AIS CSV")
    parser.add_argument("--output", default=None, help="Output Parquet path (optional)")
    parser.add_argument("--json", default=None, help="Output JSON timeline path (optional)")
    parser.add_argument("--n-vessels", type=int, default=30, help="Number of vessels to sample")
    parser.add_argument("--min-obs", type=int, default=12, help="Minimum observations per vessel")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--bbox", type=float, nargs=4, default=None,
                        metavar=("MIN_LAT", "MAX_LAT", "MIN_LON", "MAX_LON"),
                        help="Bounding box filter")
    args = parser.parse_args()

    bbox = tuple(args.bbox) if args.bbox else None

    print(f"Loading {args.input}...")
    timelines = preprocess_noaa_day(
        input_path=args.input,
        n_vessels=args.n_vessels,
        min_obs=args.min_obs,
        seed=args.seed,
        bbox=bbox,
        output_parquet=args.output,
    )

    n_records = sum(len(tl) for tl in timelines.values())
    print(f"Produced {len(timelines)} vessel timelines ({n_records} total records)")

    if args.output:
        print(f"Saved Parquet to {args.output}")

    if args.json:
        # Serialize datetimes to ISO strings for JSON
        serializable = {}
        for k, records in timelines.items():
            serializable[k] = [
                {**r, "timestamp": r["timestamp"].isoformat()}
                for r in records
            ]
        with open(args.json, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"Saved JSON to {args.json}")

    # Print a sample
    sample_key = next(iter(timelines))
    sample = timelines[sample_key]
    print(f"\nSample vessel {sample_key} ({len(sample)} records):")
    for r in sample[:3]:
        print(f"  {r['timestamp']}  lat={r['lat']}  lon={r['lon']}  "
              f"speed={r['speed_kmh']} km/h  hdg={r['heading_deg']}°")
    if len(sample) > 3:
        print(f"  ... ({len(sample) - 3} more)")


if __name__ == "__main__":
    _cli()
