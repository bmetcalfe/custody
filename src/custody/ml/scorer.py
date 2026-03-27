"""Isolation Forest anomaly scorer — train, save, load, score.

Design:
  - ``train()`` fits an IsolationForest on a feature matrix and saves
    the model artifact to disk.
  - ``MLAnomalyScorer`` loads a saved model and scores new feature
    vectors, producing ``ml_anomaly_score`` in [0, 1] where higher =
    more anomalous.
  - Score normalization: sklearn's ``decision_function`` returns negative
    values for anomalies.  We negate and rescale using the training
    distribution's percentiles so the output is a stable [0, 1] score.

Public API
----------
train(feature_df, model_path, **kwargs) -> MLAnomalyScorer
    Fit and save.  Returns a ready-to-use scorer.

MLAnomalyScorer(model_path)
    Load a saved model and call ``.score(feature_df) -> np.ndarray``.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from custody.ml.features import FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    feature_df: pd.DataFrame,
    model_path: str | Path,
    n_estimators: int = 200,
    contamination: float = 0.05,
    random_state: int = 42,
) -> "MLAnomalyScorer":
    """Fit an Isolation Forest on *feature_df* and save to *model_path*.

    Args:
        feature_df:    DataFrame with at least :data:`FEATURE_COLUMNS`.
        model_path:    Where to write the joblib artifact.
        n_estimators:  Number of trees.
        contamination: Expected fraction of anomalies (for threshold, not
                       used in scoring — we use continuous scores).
        random_state:  Seed for reproducibility.

    Returns:
        A ready-to-use :class:`MLAnomalyScorer`.
    """
    X = feature_df[FEATURE_COLUMNS].values.astype(np.float64)

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X)

    # Compute normalization parameters from training data so test scores
    # are on the same scale.
    raw_scores = model.decision_function(X)
    # decision_function: lower (more negative) = more anomalous
    # We negate so higher = more anomalous, then rescale to [0, 1]
    negated = -raw_scores
    p01 = float(np.percentile(negated, 1))
    p99 = float(np.percentile(negated, 99))

    artifact = {
        "model": model,
        "norm_p01": p01,
        "norm_p99": p99,
    }
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)

    return MLAnomalyScorer(model_path)


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class MLAnomalyScorer:
    """Load a trained Isolation Forest and score new feature vectors.

    Attributes:
        model:    The fitted IsolationForest.
        norm_p01: 1st percentile of negated training scores (low end).
        norm_p99: 99th percentile of negated training scores (high end).
    """

    def __init__(self, model_path: str | Path):
        artifact = joblib.load(model_path)
        self.model: IsolationForest = artifact["model"]
        self.norm_p01: float = artifact["norm_p01"]
        self.norm_p99: float = artifact["norm_p99"]

    def score(self, feature_df: pd.DataFrame) -> np.ndarray:
        """Return ``ml_anomaly_score`` array in [0, 1], higher = more anomalous.

        Args:
            feature_df: DataFrame with at least :data:`FEATURE_COLUMNS`.

        Returns:
            1-D numpy array of length ``len(feature_df)``.
        """
        X = feature_df[FEATURE_COLUMNS].values.astype(np.float64)
        raw = self.model.decision_function(X)
        negated = -raw
        span = self.norm_p99 - self.norm_p01
        if span <= 0:
            span = 1.0
        scaled = (negated - self.norm_p01) / span
        return np.clip(scaled, 0.0, 1.0)
