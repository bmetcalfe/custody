"""
Tests for custody.ml.scorer — Isolation Forest training and scoring.

Uses small synthetic data.

Covers:
  1. train produces a model file
  2. train returns a usable scorer
  3. scorer.score returns array of correct length
  4. scores are in [0, 1]
  5. known outlier scores higher than inlier
  6. scorer round-trip: save + load + score
  7. train_on_days + score_days integration
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from custody.ml.features import FEATURE_COLUMNS, build_feature_frame
from custody.ml.scorer import MLAnomalyScorer, train

UTC = timezone.utc
_T0 = datetime(2024, 9, 20, 0, 0, tzinfo=UTC)


def _make_normal_df(n_vessels=20, n_hours=24, seed=42):
    """Create a DataFrame of normal vessel behavior."""
    rng = np.random.default_rng(seed)
    rows = []
    for v in range(n_vessels):
        base_speed = rng.uniform(5, 15)
        base_heading = rng.uniform(0, 360)
        base_lat = rng.uniform(28, 32)
        base_lon = rng.uniform(-90, -85)
        for h in range(n_hours):
            rows.append({
                "mmsi": 100000000 + v,
                "timestamp": _T0 + timedelta(hours=h),
                "lat": base_lat + h * 0.01 + rng.normal(0, 0.001),
                "lon": base_lon + h * 0.005 + rng.normal(0, 0.001),
                "speed_kmh": base_speed + rng.normal(0, 0.5),
                "heading_deg": (base_heading + rng.normal(0, 2)) % 360,
            })
    df = pd.DataFrame(rows)
    df.sort_values(["mmsi", "timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _make_anomalous_df(n_hours=24):
    """Create a single vessel with erratic behavior."""
    rng = np.random.default_rng(99)
    rows = []
    for h in range(n_hours):
        rows.append({
            "mmsi": 999999999,
            "timestamp": _T0 + timedelta(hours=h),
            "lat": 30.0 + rng.uniform(-0.5, 0.5),
            "lon": -88.0 + rng.uniform(-0.5, 0.5),
            "speed_kmh": rng.uniform(0, 30),  # wild speed variation
            "heading_deg": rng.uniform(0, 360),  # random heading each step
        })
    return pd.DataFrame(rows)


@pytest.fixture
def normal_features():
    return build_feature_frame(_make_normal_df())


@pytest.fixture
def model_path(tmp_path):
    return tmp_path / "test_model.joblib"


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class TestTrain:

    def test_produces_model_file(self, normal_features, model_path):
        train(normal_features, model_path)
        assert model_path.exists()

    def test_returns_scorer(self, normal_features, model_path):
        scorer = train(normal_features, model_path)
        assert isinstance(scorer, MLAnomalyScorer)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

class TestScoring:

    @pytest.fixture
    def scorer(self, normal_features, model_path):
        return train(normal_features, model_path)

    def test_score_length(self, scorer, normal_features):
        scores = scorer.score(normal_features)
        assert len(scores) == len(normal_features)

    def test_scores_in_unit_interval(self, scorer, normal_features):
        scores = scorer.score(normal_features)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_outlier_scores_higher(self, scorer):
        """A vessel with erratic behavior should score higher than normal."""
        normal_df = _make_normal_df(n_vessels=5, n_hours=24, seed=77)
        anom_df = _make_anomalous_df(n_hours=24)
        combined = pd.concat([normal_df, anom_df], ignore_index=True)
        combined.sort_values(["mmsi", "timestamp"], inplace=True)
        features = build_feature_frame(combined)
        scores = scorer.score(features)

        # Split scores by vessel type
        anom_mask = features["mmsi"] == 999999999
        normal_mean = scores[~anom_mask].mean()
        anom_mean = scores[anom_mask].mean()
        assert anom_mean > normal_mean, (
            f"Anomalous vessel mean ({anom_mean:.4f}) should be > "
            f"normal mean ({normal_mean:.4f})"
        )

    def test_round_trip(self, normal_features, model_path):
        """Save, reload, score — should produce identical results."""
        scorer1 = train(normal_features, model_path)
        scores1 = scorer1.score(normal_features)
        scorer2 = MLAnomalyScorer(model_path)
        scores2 = scorer2.score(normal_features)
        np.testing.assert_array_equal(scores1, scores2)


# ---------------------------------------------------------------------------
# Integration with train.py workflow
# ---------------------------------------------------------------------------

class TestTrainScoreWorkflow:

    def test_day_based_workflow(self, tmp_path):
        """Simulate multi-day train/test with parquet files."""
        from custody.ml.train import load_daily_parquets, train_on_days, score_days

        # Create 3 "daily" parquets
        for day in range(3):
            df = _make_normal_df(n_vessels=5, n_hours=24, seed=day)
            df["timestamp"] = df["timestamp"] + timedelta(days=day)
            df.to_parquet(tmp_path / f"day{day}.parquet", index=False)

        train_paths = [str(tmp_path / "day0.parquet"), str(tmp_path / "day1.parquet")]
        test_paths = [str(tmp_path / "day2.parquet")]
        model_path = str(tmp_path / "model.joblib")

        scorer = train_on_days(train_paths, model_path)
        scored = score_days(test_paths, scorer)

        assert "ml_anomaly_score" in scored.columns
        assert len(scored) > 0
        assert scored["ml_anomaly_score"].between(0, 1).all()
