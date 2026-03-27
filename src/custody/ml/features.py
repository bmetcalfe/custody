"""Per-timestep feature extraction for ML anomaly scoring.

Operates on hourly AIS parquet DataFrames produced by
``custody.ingest.noaa_preprocess``.  Features are computed from
past/current information only — no future leakage.

Public API
----------
extract_features(record, history_window) -> dict[str, float]
    Compute features for one timestep from the current record and a
    window of prior records.

build_feature_frame(df, history_hours=6) -> pd.DataFrame
    Compute features for all rows in a per-vessel sorted DataFrame.
    Returns a DataFrame aligned 1:1 with the input rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "speed_kmh",
    "heading_deg",
    "lat",
    "lon",
    "speed_delta",
    "heading_delta",
    "lat_delta",
    "lon_delta",
    "speed_mean_6h",
    "speed_std_6h",
    "heading_std_6h",
]

_DEFAULT_HISTORY = 6  # hours


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def heading_delta(current: float, previous: float) -> float:
    """Signed circular heading difference in [-180, 180].

    Positive = clockwise turn.  Handles the 0/360 wraparound correctly:
    e.g. 350 → 10 = +20, not −340.
    """
    d = (current - previous) % 360.0
    if d > 180.0:
        d -= 360.0
    return d


def _circular_std(angles_deg: np.ndarray) -> float:
    """Circular standard deviation in degrees.  Returns 0 for < 2 values."""
    if len(angles_deg) < 2:
        return 0.0
    rads = np.radians(angles_deg)
    R = np.sqrt(np.mean(np.sin(rads)) ** 2 + np.mean(np.cos(rads)) ** 2)
    # R ∈ [0, 1]; R=1 means all identical, R=0 means uniform
    R = min(R, 1.0)
    return float(np.degrees(np.sqrt(-2.0 * np.log(R))))


# ---------------------------------------------------------------------------
# Record-level extraction
# ---------------------------------------------------------------------------

def extract_features(
    record: dict,
    history_window: list[dict],
) -> dict[str, float]:
    """Compute features for one timestep.

    Args:
        record:         Current timestep dict with speed_kmh, heading_deg,
                        lat, lon.
        history_window: Prior records for this vessel, ordered oldest-first.
                        May be empty (first timestep).

    Returns:
        Dict of feature name → float value.  All features are always
        present; short history → 0.0 for deltas and rolling stats.
    """
    speed = float(record.get("speed_kmh", 0.0))
    hdg = float(record.get("heading_deg", 0.0))
    lat = float(record.get("lat", 0.0))
    lon = float(record.get("lon", 0.0))

    # Deltas (require at least 1 prior record)
    if history_window:
        prev = history_window[-1]
        speed_d = speed - float(prev.get("speed_kmh", 0.0))
        hdg_d = heading_delta(hdg, float(prev.get("heading_deg", 0.0)))
        lat_d = lat - float(prev.get("lat", 0.0))
        lon_d = lon - float(prev.get("lon", 0.0))
    else:
        speed_d = 0.0
        hdg_d = 0.0
        lat_d = 0.0
        lon_d = 0.0

    # Rolling window (up to last 6 records, NOT including current)
    window = history_window[-_DEFAULT_HISTORY:]
    if window:
        speeds = np.array([float(r.get("speed_kmh", 0.0)) for r in window])
        headings = np.array([float(r.get("heading_deg", 0.0)) for r in window])
        speed_mean = float(np.mean(speeds))
        speed_std = float(np.std(speeds, ddof=0)) if len(speeds) > 1 else 0.0
        hdg_std = _circular_std(headings)
    else:
        speed_mean = speed
        speed_std = 0.0
        hdg_std = 0.0

    return {
        "speed_kmh":      speed,
        "heading_deg":    hdg,
        "lat":            lat,
        "lon":            lon,
        "speed_delta":    speed_d,
        "heading_delta":  hdg_d,
        "lat_delta":      lat_d,
        "lon_delta":      lon_d,
        "speed_mean_6h":  speed_mean,
        "speed_std_6h":   speed_std,
        "heading_std_6h": hdg_std,
    }


# ---------------------------------------------------------------------------
# DataFrame-level extraction
# ---------------------------------------------------------------------------

def build_feature_frame(
    df: pd.DataFrame,
    history_hours: int = _DEFAULT_HISTORY,
) -> pd.DataFrame:
    """Compute features for every row in a vessel-sorted DataFrame.

    Args:
        df:             Hourly AIS DataFrame with columns: mmsi, timestamp,
                        lat, lon, speed_kmh, heading_deg.  Must be sorted
                        by (mmsi, timestamp).
        history_hours:  Number of prior records to use for rolling features.

    Returns:
        DataFrame with one row per input row and columns from
        :data:`FEATURE_COLUMNS` plus ``mmsi`` and ``timestamp`` for
        alignment.  Rows are in the same order as the input.
    """
    result_rows: list[dict] = []

    for mmsi, group in df.groupby("mmsi", sort=False):
        records = group.to_dict("records")
        for i, record in enumerate(records):
            window = records[max(0, i - history_hours):i]
            feats = extract_features(record, window)
            feats["mmsi"] = mmsi
            feats["timestamp"] = record["timestamp"]
            result_rows.append(feats)

    out = pd.DataFrame(result_rows)
    # Ensure column order
    meta = ["mmsi", "timestamp"]
    return out[meta + FEATURE_COLUMNS]
