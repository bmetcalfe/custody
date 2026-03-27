#!/usr/bin/env python
"""Inject anomalous behavior into a real AIS track and compare scores.

Usage:
    # Loitering injection on a specific vessel
    uv run python scripts/inject_and_compare_ais.py \
        --scored data/ais_test_scored.parquet \
        --vessel 367080090 \
        --inject loitering --start 8 --duration 6

    # Heading deviation
    uv run python scripts/inject_and_compare_ais.py \
        --scored data/ais_test_scored.parquet \
        --vessel 367080090 \
        --inject heading --start 8 --duration 6 --mode erratic

    # Zone approach
    uv run python scripts/inject_and_compare_ais.py \
        --scored data/ais_test_scored.parquet \
        --vessel 367080090 \
        --inject zone --start 8 --duration 8 \
        --zone-lat 30.0 --zone-lon -88.0

    # Run all three injections on one vessel and plot
    uv run python scripts/inject_and_compare_ais.py \
        --scored data/ais_test_scored.parquet \
        --vessel 367080090 --all --plot
"""
import os
import sys
import argparse

_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_repo, "src"))

import pandas as pd
from custody.ml.analyze import vessel_time_series
from custody.ml.compare import compare
from custody.ml.inject import inject_heading_deviation, inject_loitering, inject_zone_approach
from custody.ml.scorer import MLAnomalyScorer


def _print_comparison(result: dict):
    s = result["summary"]
    print(f"\n  --- {s['label']} ---")
    print(f"  Records: {s['records']}")
    print(f"  Base mean ML score:  {s['base_mean_score']:.4f}")
    print(f"  Inj  mean ML score:  {s['inj_mean_score']:.4f}")
    print(f"  Max score delta:     {s['max_score_delta']:+.4f}")
    print(f"  Mean score delta:    {s['mean_score_delta']:+.4f}")
    print(f"  Inj max score:       {s['inj_max_score']:.4f}")
    print(f"  Hours above 0.8:     {s['total_hours_above_08']}")
    if s["first_above_08"]:
        print(f"  First above 0.8:     {s['first_above_08']}")

    # Show the window where injection is most visible
    table = result["table"]
    delta = table["ml_score_delta"]
    peak_idx = delta.idxmax()
    window = table.iloc[max(0, peak_idx - 2):peak_idx + 3]
    print(f"\n  Peak delta window:")
    for _, row in window.iterrows():
        ts = str(row["timestamp"]).split("+")[0]
        marker = " <<" if row["inj_ml_score"] >= 0.8 else ""
        print(f"    {ts}  base={row['base_ml_score']:.3f}  "
              f"inj={row['inj_ml_score']:.3f}  "
              f"delta={row['ml_score_delta']:+.3f}  "
              f"speed={row['base_speed']:.1f}->{row['inj_speed']:.1f}{marker}")


def _plot_comparison(results: list[dict], vessel_id: int, out_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed -- skipping plot")
        return

    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=True)
    if n == 1:
        axes = [axes]
    fig.suptitle(f"MMSI {vessel_id} -- Anomaly Injection Comparison", fontsize=13)

    for ax, result in zip(axes, results):
        table = result["table"]
        label = result["summary"]["label"]
        ax.plot(table["timestamp"], table["base_ml_score"],
                color="#888", linewidth=1, label="Baseline ML Score")
        ax.plot(table["timestamp"], table["inj_ml_score"],
                color="#d73a49", linewidth=1.5, label=f"{label} ML Score")
        ax.fill_between(table["timestamp"], table["inj_ml_score"],
                        table["base_ml_score"], alpha=0.15, color="#d73a49")
        ax.axhline(0.8, color="#555", linestyle="--", linewidth=0.7)
        ax.set_ylabel("ML Anomaly Score")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8, loc="upper left")
        ax.set_title(label, fontsize=10)

    axes[-1].set_xlabel("Time (UTC)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"\n  Plot saved to {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Inject anomalies and compare ML scores")
    parser.add_argument("--scored", default=os.path.join(_repo, "data", "ais_test_scored.parquet"))
    parser.add_argument("--model", default=os.path.join(_repo, "models", "isolation_forest.joblib"))
    parser.add_argument("--vessel", type=int, required=True)
    parser.add_argument("--inject", choices=["loitering", "heading", "zone"], default=None)
    parser.add_argument("--all", action="store_true", help="Run all three injection types")
    parser.add_argument("--start", type=int, default=8, help="Injection start hour index")
    parser.add_argument("--duration", type=int, default=6)
    parser.add_argument("--mode", default="erratic", help="Heading mode: sustained or erratic")
    parser.add_argument("--deviation", type=float, default=90.0, help="Heading deviation degrees")
    parser.add_argument("--zone-lat", type=float, default=30.0)
    parser.add_argument("--zone-lon", type=float, default=-88.0)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--csv-out", default=None, help="Write comparison CSV")
    args = parser.parse_args()

    df = pd.read_parquet(args.scored)
    ts = vessel_time_series(df, args.vessel)
    if ts.empty:
        print(f"Vessel {args.vessel} not found.")
        return

    baseline = ts.to_dict("records")
    print(f"Vessel {args.vessel}: {len(baseline)} records")
    print(f"  Speed range: [{ts['speed_kmh'].min():.1f}, {ts['speed_kmh'].max():.1f}] km/h")
    print(f"  Lat range: [{ts['lat'].min():.4f}, {ts['lat'].max():.4f}]")

    scorer = MLAnomalyScorer(args.model)

    inject_types = []
    if args.all:
        inject_types = ["loitering", "heading", "zone"]
    elif args.inject:
        inject_types = [args.inject]
    else:
        parser.error("Specify --inject TYPE or --all")

    results = []
    for inj_type in inject_types:
        if inj_type == "loitering":
            injected = inject_loitering(baseline, args.start, args.duration)
            label = f"Loitering (h{args.start}-h{args.start + args.duration})"
        elif inj_type == "heading":
            injected = inject_heading_deviation(
                baseline, args.start, args.duration,
                mode=args.mode, deviation_deg=args.deviation,
            )
            label = f"Heading {args.mode} (h{args.start}-h{args.start + args.duration})"
        elif inj_type == "zone":
            injected = inject_zone_approach(
                baseline, args.start, args.duration,
                zone_lat=args.zone_lat, zone_lon=args.zone_lon,
            )
            label = f"Zone approach (h{args.start}-h{args.start + args.duration})"
        else:
            continue

        result = compare(baseline, injected, scorer, label=label)
        results.append(result)
        _print_comparison(result)

    if args.csv_out and results:
        all_tables = pd.concat(
            [r["table"].assign(variant=r["summary"]["label"]) for r in results],
            ignore_index=True,
        )
        all_tables.to_csv(args.csv_out, index=False)
        print(f"\n  CSV written to {args.csv_out}")

    if args.plot and results:
        plot_path = os.path.join(_repo, "data", f"inject_{args.vessel}.png")
        _plot_comparison(results, args.vessel, plot_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
