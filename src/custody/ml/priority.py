"""Vessel priority scoring for ML-enriched timelines.

Converts anomaly detection signals into actionable prioritization.
Operates on records already enriched by the reasoning layer
(anomaly_agreement, anomaly_state, ml_anomaly_duration_hours,
escalation_boost, ml_anomaly_combined).

Priority score is distinct from fused_score — it incorporates temporal
context (persistence, agreement, escalation) that fused_score does not.

Public API
----------
compute_priority(record) -> float
add_priority_scores(df) -> pd.DataFrame
rank_vessels(df, n=20) -> pd.DataFrame
vessel_priority_summary(df) -> pd.DataFrame
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# State and agreement weights
# ---------------------------------------------------------------------------

_STATE_BOOST = {
    "critical":   0.25,
    "sustained":  0.15,
    "confirmed":  0.10,
    "emerging":   0.05,
    "recovering": 0.02,
    "normal":     0.0,
}

_AGREEMENT_BOOST = {
    "confirmed":      0.10,
    "emerging":       0.05,
    "rule_triggered": 0.03,
    "normal":         0.0,
}


# ---------------------------------------------------------------------------
# Record-level priority
# ---------------------------------------------------------------------------

def compute_priority(record: dict) -> float:
    """Compute a priority score for one enriched record.

    Formula:
        base = max(fused_score, ml_anomaly_combined)
        + state_boost       (0.0–0.25 based on anomaly_state)
        + agreement_boost   (0.0–0.10 based on anomaly_agreement)
        + duration_boost    (0.01 per hour, capped at 0.10)
        + custody_pressure  (0.15 * (1 - custody_confidence))
        + escalation_boost  (passed through from reasoning layer)

    Returns a value in [0, 1], clamped.

    Sustained, confirmed, high-ML anomalies with weak custody rank highest.
    Brief isolated spikes produce minimal priority above their base score.
    """
    # Base: the stronger of fused and ML combined
    fused = float(record.get("fused_score", 0.0))
    fa = record.get("fusion_assessment")
    if fa is not None and hasattr(fa, "fused_score"):
        fused = max(fused, fa.fused_score)
    ml_combined = float(record.get("ml_anomaly_combined", 0.0))
    base = max(fused, ml_combined)

    # State boost
    state = record.get("anomaly_state", "normal")
    s_boost = _STATE_BOOST.get(state, 0.0)

    # Agreement boost
    agreement = record.get("anomaly_agreement", "normal")
    a_boost = _AGREEMENT_BOOST.get(agreement, 0.0)

    # Duration boost: 0.01 per hour above threshold, capped at 0.10
    duration = int(record.get("ml_anomaly_duration_hours", 0))
    d_boost = min(duration * 0.01, 0.10)

    # Custody pressure: weak custody increases priority
    custody = float(record.get("custody_confidence", 1.0))
    c_pressure = 0.15 * (1.0 - min(max(custody, 0.0), 1.0))

    # Escalation from reasoning layer
    esc = float(record.get("escalation_boost", 0.0))

    score = base + s_boost + a_boost + d_boost + c_pressure + esc
    return round(min(max(score, 0.0), 1.0), 4)


# ---------------------------------------------------------------------------
# DataFrame-level operations
# ---------------------------------------------------------------------------

def add_priority_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``priority_score`` column to a DataFrame of enriched records.

    Records must have been enriched by the reasoning layer (i.e. contain
    anomaly_state, anomaly_agreement, ml_anomaly_duration_hours, etc.).
    Missing fields are handled with safe defaults.

    Returns a copy.
    """
    out = df.copy()
    out["priority_score"] = [
        compute_priority(row.to_dict()) for _, row in out.iterrows()
    ]
    return out


def rank_vessels(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Rank vessels by their peak priority_score.

    Args:
        df: DataFrame with ``priority_score`` and ``mmsi`` columns.
        n:  Number of top vessels to return.

    Returns:
        DataFrame with one row per vessel, sorted by priority descending:
        mmsi, max_priority, mean_priority, peak_timestamp, peak_state,
        peak_agreement, total_hours_elevated, records.
    """
    if df.empty or "priority_score" not in df.columns:
        return pd.DataFrame(columns=[
            "mmsi", "max_priority", "mean_priority", "peak_timestamp",
            "peak_state", "peak_agreement", "total_hours_elevated", "records",
        ])

    rows = []
    for mmsi, group in df.groupby("mmsi"):
        peak_idx = group["priority_score"].idxmax()
        peak_row = group.loc[peak_idx]
        elevated = group["priority_score"] >= 0.5
        rows.append({
            "mmsi":                 int(mmsi),
            "max_priority":         round(float(group["priority_score"].max()), 4),
            "mean_priority":        round(float(group["priority_score"].mean()), 4),
            "peak_timestamp":       peak_row.get("timestamp"),
            "peak_state":           peak_row.get("anomaly_state", "normal"),
            "peak_agreement":       peak_row.get("anomaly_agreement", "normal"),
            "total_hours_elevated": int(elevated.sum()),
            "records":              len(group),
        })
    result = pd.DataFrame(rows)
    result.sort_values("max_priority", ascending=False, inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result.head(n)


def vessel_priority_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-vessel summary with priority + anomaly state distribution.

    Returns DataFrame with: mmsi, records, max_priority, mean_priority,
    hours_elevated, state_counts (dict), dominant_state.
    """
    rows = []
    for mmsi, group in df.groupby("mmsi"):
        state_counts = group.get("anomaly_state", pd.Series(dtype=str)).value_counts().to_dict()
        dominant = max(state_counts, key=state_counts.get) if state_counts else "normal"
        elevated = group["priority_score"] >= 0.5 if "priority_score" in group.columns else pd.Series([False])
        rows.append({
            "mmsi":             int(mmsi),
            "records":          len(group),
            "max_priority":     round(float(group["priority_score"].max()), 4) if "priority_score" in group.columns else 0.0,
            "mean_priority":    round(float(group["priority_score"].mean()), 4) if "priority_score" in group.columns else 0.0,
            "hours_elevated":   int(elevated.sum()),
            "state_counts":     state_counts,
            "dominant_state":   dominant,
        })
    result = pd.DataFrame(rows)
    result.sort_values("max_priority", ascending=False, inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result
