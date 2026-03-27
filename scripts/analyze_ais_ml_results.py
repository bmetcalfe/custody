#!/usr/bin/env python
"""Inspect and summarize ML anomaly scores from scored parquet files.

Usage:
    # Default: analyze the test scored output
    uv run python scripts/analyze_ais_ml_results.py

    # Analyze a specific file
    uv run python scripts/analyze_ais_ml_results.py --input data/ais_test_scored.parquet

    # Focus on one vessel
    uv run python scripts/analyze_ais_ml_results.py --vessel 303260000

    # Write CSV outputs
    uv run python scripts/analyze_ais_ml_results.py --csv-dir data/analysis/

    # Plot a vessel
    uv run python scripts/analyze_ais_ml_results.py --vessel 303260000 --plot
"""
import os
import sys
import argparse

_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_repo, "src"))

import pandas as pd
from custody.ml.analyze import (
    load_scored_parquet,
    top_anomalous_records,
    top_anomalous_vessels,
    vessel_time_series,
    vessel_summary,
    anomaly_excursions,
    contiguous_windows,
    markdown_summary,
)


def _plot_vessel(ts: pd.DataFrame, vessel_id: int, out_path: str | None = None):
    """Simple matplotlib plot: score, speed, heading_delta over time."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed — skipping plot")
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"MMSI {vessel_id} — ML Anomaly Analysis", fontsize=13)

    # Score
    axes[0].plot(ts["timestamp"], ts["ml_anomaly_score"], color="#d73a49", linewidth=1.2)
    axes[0].axhline(0.8, color="#888", linestyle="--", linewidth=0.7, label="threshold=0.8")
    axes[0].fill_between(ts["timestamp"], ts["ml_anomaly_score"], alpha=0.15, color="#d73a49")
    axes[0].set_ylabel("ML Anomaly Score")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend(fontsize=8)

    # Speed
    axes[1].plot(ts["timestamp"], ts["speed_kmh"], color="#1f78b4", linewidth=1.0)
    axes[1].set_ylabel("Speed (km/h)")

    # Heading delta (if available)
    if "heading_delta" in ts.columns:
        axes[2].bar(ts["timestamp"], ts["heading_delta"], width=0.035, color="#e07b00", alpha=0.7)
        axes[2].set_ylabel("Heading Delta (°)")
    else:
        axes[2].plot(ts["timestamp"], ts["heading_deg"], color="#2ea043", linewidth=1.0)
        axes[2].set_ylabel("Heading (°)")

    axes[2].set_xlabel("Time (UTC)")
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=120)
        print(f"  Plot saved to {out_path}")
    else:
        path = os.path.join(_repo, "data", f"vessel_{vessel_id}_analysis.png")
        plt.savefig(path, dpi=120)
        print(f"  Plot saved to {path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze ML anomaly scores")
    parser.add_argument("--input", default=os.path.join(_repo, "data", "ais_test_scored.parquet"),
                        help="Scored parquet path")
    parser.add_argument("--holdout", default=None,
                        help="Optional holdout scored parquet for comparison")
    parser.add_argument("--vessel", type=int, default=None,
                        help="Focus on a specific MMSI")
    parser.add_argument("--csv-dir", default=None,
                        help="Directory to write CSV outputs")
    parser.add_argument("--plot", action="store_true",
                        help="Generate matplotlib plot for focused vessel")
    parser.add_argument("--top-n", type=int, default=20,
                        help="Number of top records/vessels to show")
    args = parser.parse_args()

    df = load_scored_parquet(args.input)
    print(f"Loaded {len(df):,} records from {os.path.basename(args.input)}")
    print(f"Vessels: {df['mmsi'].nunique()}  |  "
          f"Score range: [{df['ml_anomaly_score'].min():.4f}, {df['ml_anomaly_score'].max():.4f}]")

    # ── Vessel summary ───────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  TOP {args.top_n} ANOMALOUS VESSELS")
    print(f"{'=' * 70}")
    vs = top_anomalous_vessels(df, n=args.top_n)
    for _, row in vs.iterrows():
        ts_str = str(row["peak_timestamp"]).split("+")[0]
        print(f"  {int(row['mmsi']):>12}  records={int(row['count']):>3}  "
              f"mean={row['mean']:.3f}  max={row['max']:.3f}  p95={row['p95']:.3f}  "
              f"peak={ts_str}  speed={row['peak_speed']:.1f} km/h")

    # ── Top records ──────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  TOP {args.top_n} ANOMALOUS RECORDS")
    print(f"{'=' * 70}")
    top = top_anomalous_records(df, n=args.top_n)
    for _, row in top.iterrows():
        ts_str = str(row["timestamp"]).split("+")[0]
        extras = ""
        if "speed_delta" in row.index:
            extras += f"  Δspd={row['speed_delta']:+.1f}"
        if "heading_delta" in row.index:
            extras += f"  Δhdg={row['heading_delta']:+.1f}°"
        print(f"  {int(row['mmsi']):>12}  {ts_str}  "
              f"lat={row['lat']:.4f}  lon={row['lon']:.4f}  "
              f"speed={row['speed_kmh']:.1f} km/h  score={row['ml_anomaly_score']:.4f}"
              f"{extras}")

    # ── Excursions ───────────────────────────────────────────────────────
    excursions = anomaly_excursions(df, threshold=0.8)
    print(f"\n  Excursions (score >= 0.8): {len(excursions)} records across "
          f"{excursions['mmsi'].nunique() if len(excursions) > 0 else 0} vessels")

    # ── Contiguous windows ───────────────────────────────────────────────
    wins = contiguous_windows(df, threshold=0.8, min_length=2)
    if wins:
        print(f"\n  Sustained anomaly windows (>= 2h above 0.8): {len(wins)}")
        for w in wins[:5]:
            s = str(w["start"]).split("+")[0]
            e = str(w["end"]).split("+")[0]
            print(f"    mmsi={w['mmsi']}  {s} ->{e}  "
                  f"length={w['length']}h  max={w['max_score']:.3f}  mean={w['mean_score']:.3f}")

    # ── Markdown ─────────────────────────────────────────────────────────
    md = markdown_summary(df, top_n=5)
    print(f"\n{'=' * 70}")
    print(md)

    # ── Single vessel focus ──────────────────────────────────────────────
    if args.vessel:
        print(f"\n{'=' * 70}")
        print(f"  VESSEL FOCUS: {args.vessel}")
        print(f"{'=' * 70}")
        ts = vessel_time_series(df, args.vessel)
        if ts.empty:
            print(f"  Vessel {args.vessel} not found in data.")
        else:
            scores = ts["ml_anomaly_score"]
            print(f"  Records: {len(ts)}")
            print(f"  Score: mean={scores.mean():.4f}  max={scores.max():.4f}  "
                  f"p95={scores.quantile(0.95):.4f}")
            print(f"  Speed range: [{ts['speed_kmh'].min():.1f}, {ts['speed_kmh'].max():.1f}] km/h")
            print(f"\n  Timeline (showing all records):")
            for _, row in ts.iterrows():
                ts_str = str(row["timestamp"]).split("+")[0]
                marker = " <<" if row["ml_anomaly_score"] >= 0.8 else ""
                print(f"    {ts_str}  speed={row['speed_kmh']:>6.1f}  "
                      f"hdg={row['heading_deg']:>6.1f}  "
                      f"score={row['ml_anomaly_score']:.4f}{marker}")
            if args.plot:
                _plot_vessel(ts, args.vessel)

    # ── CSV output ───────────────────────────────────────────────────────
    if args.csv_dir:
        os.makedirs(args.csv_dir, exist_ok=True)
        vs.to_csv(os.path.join(args.csv_dir, "vessel_summary.csv"), index=False)
        top.to_csv(os.path.join(args.csv_dir, "top_records.csv"), index=False)
        if wins:
            pd.DataFrame(wins).to_csv(os.path.join(args.csv_dir, "anomaly_windows.csv"), index=False)
        print(f"\n  CSV outputs written to {args.csv_dir}/")

    # ── Holdout comparison ───────────────────────────────────────────────
    if args.holdout:
        hdf = load_scored_parquet(args.holdout)
        print(f"\n{'=' * 70}")
        print(f"  HOLDOUT COMPARISON ({os.path.basename(args.holdout)})")
        print(f"{'=' * 70}")
        print(f"  Holdout records: {len(hdf):,}  vessels: {hdf['mmsi'].nunique()}")
        hs = hdf["ml_anomaly_score"]
        ms = df["ml_anomaly_score"]
        print(f"  Test    score: mean={ms.mean():.4f}  p95={ms.quantile(0.95):.4f}")
        print(f"  Holdout score: mean={hs.mean():.4f}  p95={hs.quantile(0.95):.4f}")
        h_exc = anomaly_excursions(hdf, threshold=0.8)
        print(f"  Holdout excursions (>= 0.8): {len(h_exc)} records, "
              f"{h_exc['mmsi'].nunique() if len(h_exc) > 0 else 0} vessels")

    print("\nDone.")


if __name__ == "__main__":
    main()
