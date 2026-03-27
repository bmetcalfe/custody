"""Multi-day train/score workflow for AIS anomaly detection.

Loads preprocessed daily parquet files, builds feature frames, trains
an Isolation Forest, and scores held-out days.

Public API
----------
load_daily_parquets(paths) -> pd.DataFrame
    Concatenate multiple daily parquets into one sorted DataFrame.

train_on_days(train_paths, model_path, **kwargs) -> MLAnomalyScorer
    Build features from train days and fit model.

score_days(test_paths, scorer) -> pd.DataFrame
    Build features from test days and append ml_anomaly_score.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from custody.ml.features import build_feature_frame
from custody.ml.scorer import MLAnomalyScorer, train


def load_daily_parquets(paths: list[str | Path]) -> pd.DataFrame:
    """Concatenate daily parquet files into one sorted DataFrame.

    Args:
        paths: List of parquet file paths (from noaa_preprocess).

    Returns:
        Combined DataFrame sorted by (mmsi, timestamp).
    """
    frames = [pd.read_parquet(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df.sort_values(["mmsi", "timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def train_on_days(
    train_paths: list[str | Path],
    model_path: str | Path = "models/isolation_forest.joblib",
    **kwargs,
) -> MLAnomalyScorer:
    """Load train-day parquets, extract features, fit Isolation Forest.

    Args:
        train_paths: Parquet files for training days.
        model_path:  Where to save the model artifact.
        **kwargs:    Forwarded to :func:`scorer.train` (n_estimators, etc.).

    Returns:
        A ready-to-use :class:`MLAnomalyScorer`.
    """
    df = load_daily_parquets(train_paths)
    features = build_feature_frame(df)
    return train(features, model_path, **kwargs)


def score_days(
    test_paths: list[str | Path],
    scorer: MLAnomalyScorer,
) -> pd.DataFrame:
    """Load test-day parquets, extract features, score with trained model.

    Args:
        test_paths: Parquet files for test days.
        scorer:     A trained :class:`MLAnomalyScorer`.

    Returns:
        DataFrame with all original columns plus ``ml_anomaly_score``
        and all feature columns.  Sorted by (mmsi, timestamp).
    """
    df = load_daily_parquets(test_paths)
    features = build_feature_frame(df)
    scores = scorer.score(features)
    # Merge: align features + scores back onto the raw data
    features["ml_anomaly_score"] = scores
    # Join the score and feature columns onto the original data
    out = df.copy()
    out = out.merge(
        features[["mmsi", "timestamp", "ml_anomaly_score"]],
        on=["mmsi", "timestamp"],
        how="left",
    )
    return out
