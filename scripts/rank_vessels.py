#!/usr/bin/env python
"""Rank vessels by priority from scored + enriched AIS data.

Usage:
    uv run python scripts/rank_vessels.py
    uv run python scripts/rank_vessels.py --input data/ais_test_scored.parquet --top 30
    uv run python scripts/rank_vessels.py --csv-out data/priority_ranking.csv
"""
import os
import sys
import argparse

_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_repo, "src"))

import pandas as pd
from custody.ml.normalize import build_baselines, add_relative_scores
from custody.ml.compare import _enrich_timeline_reasoning
from custody.ml.priority import add_priority_scores, rank_vessels, vessel_priority_summary
from custody.reasoning import ml_combined_score


def main():
    parser = argparse.ArgumentParser(description="Rank vessels by priority score")
    parser.add_argument("--input", default=os.path.join(_repo, "data", "ais_test_scored.parquet"))
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--csv-out", default=None)
    args = parser.parse_args()

    print(f"Loading {os.path.basename(args.input)}...")
    df = pd.read_parquet(args.input)
    print(f"  {len(df):,} records, {df['mmsi'].nunique()} vessels")

    # Step 1: Add relative scores
    print("Computing per-vessel baselines and relative scores...")
    baselines = build_baselines(df)
    df = add_relative_scores(df, baselines)

    # Step 2: Add combined score
    df["ml_anomaly_combined"] = df.apply(
        lambda r: ml_combined_score(r.to_dict()), axis=1
    )

    # Step 3: Enrich with reasoning (agreement, state, persistence, escalation)
    print("Running reasoning enrichment...")
    enriched_records = []
    for mmsi, group in df.groupby("mmsi"):
        group = group.sort_values("timestamp")
        records = group.to_dict("records")
        enriched = _enrich_timeline_reasoning(records)
        enriched_records.extend(enriched)
    df = pd.DataFrame(enriched_records)

    # Step 4: Compute priority scores
    print("Computing priority scores...")
    df = add_priority_scores(df)

    # Step 5: Rank and display
    top = rank_vessels(df, n=args.top)
    summary = vessel_priority_summary(df)

    print(f"\n{'=' * 80}")
    print(f"  TOP {args.top} VESSELS BY PRIORITY")
    print(f"{'=' * 80}")
    print(f"  {'MMSI':>12}  {'MaxPri':>7}  {'MeanPri':>8}  {'Hours>0.5':>9}  {'State':>12}  {'Agreement':>15}  {'Peak Time'}")
    print(f"  {'----':>12}  {'------':>7}  {'-------':>8}  {'---------':>9}  {'-----':>12}  {'---------':>15}  {'---------'}")
    for _, row in top.iterrows():
        ts = str(row["peak_timestamp"]).split("+")[0] if row["peak_timestamp"] else "—"
        print(f"  {int(row['mmsi']):>12}  {row['max_priority']:>7.4f}  {row['mean_priority']:>8.4f}  "
              f"{row['total_hours_elevated']:>9}  {row['peak_state']:>12}  "
              f"{row['peak_agreement']:>15}  {ts}")

    # Fleet-level stats
    print(f"\n{'=' * 80}")
    print(f"  FLEET SUMMARY")
    print(f"{'=' * 80}")
    p = df["priority_score"]
    print(f"  Records: {len(df):,}")
    print(f"  Priority: mean={p.mean():.4f}  p90={p.quantile(0.9):.4f}  "
          f"p95={p.quantile(0.95):.4f}  max={p.max():.4f}")
    elevated = (p >= 0.5).sum()
    high = (p >= 0.7).sum()
    print(f"  Records >= 0.5: {elevated:,} ({100*elevated/len(df):.1f}%)")
    print(f"  Records >= 0.7: {high:,} ({100*high/len(df):.1f}%)")

    # State distribution
    if "anomaly_state" in df.columns:
        states = df["anomaly_state"].value_counts()
        print(f"\n  Anomaly state distribution:")
        for state, count in states.items():
            print(f"    {state:>12}: {count:>6} ({100*count/len(df):.1f}%)")

    if args.csv_out:
        top.to_csv(args.csv_out, index=False)
        print(f"\n  Saved to {args.csv_out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
