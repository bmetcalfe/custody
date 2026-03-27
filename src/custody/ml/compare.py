"""Baseline vs injected track comparison with reasoning enrichment.

Scores both timelines through the ML feature extraction + scorer pipeline,
enriches with temporal anomaly reasoning (agreement, persistence,
escalation, state), and produces a side-by-side comparison table.

Public API
----------
score_timeline(timeline, scorer) -> pd.DataFrame
compare(baseline, injected, scorer, label="injected") -> dict
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from custody.ml.features import build_feature_frame, FEATURE_COLUMNS
from custody.reasoning import enrich_record


def _enrich_timeline_reasoning(records: list[dict]) -> list[dict]:
    """Run reasoning enrichment across a scored timeline."""
    enriched = []
    prev_state = None
    for i, r in enumerate(records):
        history = records[:i]
        e = enrich_record(r, history, previous_state=prev_state)
        prev_state = e["anomaly_state"]
        enriched.append(e)
    return enriched


def score_timeline(
    timeline: list[dict],
    scorer,
) -> pd.DataFrame:
    """Build features, score, and enrich a single vessel timeline.

    Args:
        timeline: List of record dicts with timestamp, lat, lon,
                  speed_kmh, heading_deg, and mmsi.
        scorer:   An ``MLAnomalyScorer`` instance.

    Returns:
        DataFrame with original fields + feature columns +
        ``ml_anomaly_score`` + reasoning fields.
    """
    df = pd.DataFrame(timeline)
    if "mmsi" not in df.columns:
        df["mmsi"] = 0
    df.sort_values(["mmsi", "timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    features = build_feature_frame(df)
    scores = scorer.score(features)

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = features[col].values
    df["ml_anomaly_score"] = scores

    # Reasoning enrichment
    records = df.to_dict("records")
    enriched = _enrich_timeline_reasoning(records)
    for field in ("anomaly_agreement", "anomaly_state",
                  "ml_anomaly_duration_hours", "fused_anomaly_duration_hours",
                  "is_sustained_anomaly", "escalation_boost"):
        df[field] = [r[field] for r in enriched]

    return df


_REASONING_FIELDS = [
    "anomaly_agreement", "anomaly_state",
    "ml_anomaly_duration_hours", "escalation_boost",
]


def compare(
    baseline: list[dict],
    injected: list[dict],
    scorer,
    label: str = "injected",
) -> dict:
    """Score both timelines and produce a comparison with reasoning.

    Returns:
        Dict with keys:
            "table":   DataFrame with side-by-side scores and reasoning.
            "summary": Dict of summary metrics including state transitions.
    """
    base_df = score_timeline(baseline, scorer)
    inj_df = score_timeline(injected, scorer)

    table = pd.DataFrame({
        "timestamp":            base_df["timestamp"],
        "base_speed":           base_df["speed_kmh"],
        "inj_speed":            inj_df["speed_kmh"],
        "base_heading":         base_df["heading_deg"],
        "inj_heading":          inj_df["heading_deg"],
        "base_lat":             base_df["lat"],
        "inj_lat":              inj_df["lat"],
        "base_lon":             base_df["lon"],
        "inj_lon":              inj_df["lon"],
        "base_ml_score":        base_df["ml_anomaly_score"],
        "inj_ml_score":         inj_df["ml_anomaly_score"],
        "ml_score_delta":       inj_df["ml_anomaly_score"].values - base_df["ml_anomaly_score"].values,
        "base_agreement":       base_df["anomaly_agreement"],
        "inj_agreement":        inj_df["anomaly_agreement"],
        "base_state":           base_df["anomaly_state"],
        "inj_state":            inj_df["anomaly_state"],
        "inj_duration_hours":   inj_df["ml_anomaly_duration_hours"],
        "inj_escalation":       inj_df["escalation_boost"],
    })
    for feat in ["speed_delta", "heading_delta"]:
        if feat in base_df.columns and feat in inj_df.columns:
            table[f"base_{feat}"] = base_df[feat]
            table[f"inj_{feat}"] = inj_df[feat]

    # Summary metrics
    delta = table["ml_score_delta"]
    inj_scores = table["inj_ml_score"]
    threshold = 0.8

    above = inj_scores >= threshold
    first_above = None
    if above.any():
        first_above = str(table.loc[above.idxmax(), "timestamp"])

    # State transition summary
    states = inj_df["anomaly_state"].tolist()
    state_set = set(states)
    first_of = {}
    for target in ["emerging", "confirmed", "sustained", "critical", "recovering"]:
        for i, s in enumerate(states):
            if s == target and target not in first_of:
                first_of[target] = str(inj_df.iloc[i]["timestamp"])

    summary = {
        "label":                label,
        "records":              len(table),
        "base_mean_score":      round(float(table["base_ml_score"].mean()), 4),
        "inj_mean_score":       round(float(inj_scores.mean()), 4),
        "max_score_delta":      round(float(delta.max()), 4),
        "mean_score_delta":     round(float(delta.mean()), 4),
        "inj_max_score":        round(float(inj_scores.max()), 4),
        "inj_records_above_08": int(above.sum()),
        "first_above_08":       first_above,
        "total_hours_above_08": int(above.sum()),
        "peak_escalation":      round(float(inj_df["escalation_boost"].max()), 4),
        "max_duration_hours":   int(inj_df["ml_anomaly_duration_hours"].max()),
        "states_observed":      sorted(state_set),
        "first_emerging":       first_of.get("emerging"),
        "first_confirmed":      first_of.get("confirmed"),
        "first_sustained":      first_of.get("sustained"),
        "first_critical":       first_of.get("critical"),
        "first_recovering":     first_of.get("recovering"),
    }

    return {"table": table, "summary": summary}
