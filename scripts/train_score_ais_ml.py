#!/usr/bin/env python
"""Train an Isolation Forest on AIS daily parquets and score held-out days.

Usage:
    uv run python scripts/train_score_ais_ml.py \
        --train data/ais-2024-09-15_hourly_300.parquet ... \
        --test  data/ais-2024-09-22_hourly_300.parquet ... \
        --model-out models/isolation_forest.joblib \
        --scored-out data/ais_test_scored.parquet

    # Or use the built-in defaults for the 300-vessel + 30-vessel sets:
    uv run python scripts/train_score_ais_ml.py --defaults
"""
import os
import sys
import glob
import argparse

_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_repo, "src"))

import numpy as np
import pandas as pd
from custody.ml.train import load_daily_parquets, train_on_days, score_days
from custody.ml.features import build_feature_frame


def _default_paths():
    """Return (train_300, test_300, holdout_30) path lists."""
    data = os.path.join(_repo, "data")
    all_300 = sorted(glob.glob(os.path.join(data, "ais-2024-09-*_hourly_300.parquet")))
    train_300 = all_300[:7]   # Sep 15–21
    test_300 = all_300[7:]    # Sep 22–24
    holdout_30 = sorted(glob.glob(os.path.join(data, "ais-2024-09-*_hourly.parquet")))
    # Exclude the old ais_2024_09_24_hourly.parquet (underscore variant)
    holdout_30 = [p for p in holdout_30 if "_300" not in p and "ais-2024-09-" in os.path.basename(p)]
    return train_300, test_300, holdout_30


def _print_score_summary(label: str, df: pd.DataFrame):
    scores = df["ml_anomaly_score"].dropna()
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Rows:   {len(df):,}")
    print(f"  Scored: {len(scores):,}")
    print(f"  Score distribution:")
    print(f"    mean:   {scores.mean():.4f}")
    print(f"    std:    {scores.std():.4f}")
    print(f"    min:    {scores.min():.4f}")
    print(f"    p25:    {scores.quantile(0.25):.4f}")
    print(f"    median: {scores.median():.4f}")
    print(f"    p75:    {scores.quantile(0.75):.4f}")
    print(f"    p95:    {scores.quantile(0.95):.4f}")
    print(f"    max:    {scores.max():.4f}")
    # Top anomalous records
    top = df.nlargest(5, "ml_anomaly_score")
    print(f"\n  Top 5 anomalous records:")
    for _, row in top.iterrows():
        print(f"    mmsi={int(row['mmsi'])}  {row['timestamp']}  "
              f"speed={row['speed_kmh']:.1f} km/h  score={row['ml_anomaly_score']:.4f}")
    # Per-vessel summary
    vessel_stats = df.groupby("mmsi")["ml_anomaly_score"].agg(["mean", "max"]).sort_values("max", ascending=False)
    print(f"\n  Top 5 anomalous vessels (by max score):")
    for mmsi, row in vessel_stats.head(5).iterrows():
        print(f"    {int(mmsi):>12}  mean={row['mean']:.4f}  max={row['max']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train and score AIS anomaly model")
    parser.add_argument("--train", nargs="+", help="Training day parquet paths")
    parser.add_argument("--test", nargs="+", help="Test day parquet paths")
    parser.add_argument("--holdout", nargs="*", default=None, help="Holdout (30-vessel) parquet paths")
    parser.add_argument("--model-out", default="models/isolation_forest.joblib")
    parser.add_argument("--scored-out", default="data/ais_test_scored.parquet")
    parser.add_argument("--holdout-out", default="data/ais_holdout_scored.parquet")
    parser.add_argument("--defaults", action="store_true", help="Use default 300/30 vessel paths")
    args = parser.parse_args()

    if args.defaults:
        train_paths, test_paths, holdout_paths = _default_paths()
    else:
        train_paths = args.train or []
        test_paths = args.test or []
        holdout_paths = args.holdout or []

    if not train_paths:
        parser.error("No training files. Use --train or --defaults.")

    # --- Train ---
    print(f"Training on {len(train_paths)} day files...")
    for p in train_paths:
        print(f"  {os.path.basename(p)}")
    train_df = load_daily_parquets(train_paths)
    print(f"  Train rows: {len(train_df):,}  vessels: {train_df['mmsi'].nunique()}")

    train_features = build_feature_frame(train_df)
    from custody.ml.scorer import train as train_model
    scorer = train_model(train_features, args.model_out)
    print(f"  Model saved to {args.model_out}")

    # --- Score train set (for reference) ---
    train_scores = scorer.score(train_features)
    print(f"\n  Train score stats: mean={train_scores.mean():.4f}  "
          f"std={train_scores.std():.4f}  p95={np.percentile(train_scores, 95):.4f}")

    # --- Score test days (temporal generalization) ---
    if test_paths:
        print(f"\nScoring {len(test_paths)} test day files...")
        scored_test = score_days(test_paths, scorer)
        scored_test.to_parquet(args.scored_out, index=False)
        print(f"  Saved to {args.scored_out}")
        _print_score_summary("TEMPORAL GENERALIZATION (300-vessel, days 8-10)", scored_test)

    # --- Score holdout (population generalization) ---
    if holdout_paths:
        print(f"\nScoring {len(holdout_paths)} holdout (30-vessel) files...")
        scored_holdout = score_days(holdout_paths, scorer)
        scored_holdout.to_parquet(args.holdout_out, index=False)
        print(f"  Saved to {args.holdout_out}")
        _print_score_summary("POPULATION GENERALIZATION (30-vessel holdout)", scored_holdout)

    print("\nDone.")


if __name__ == "__main__":
    main()
