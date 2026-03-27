#!/usr/bin/env python
"""Validate temporal anomaly reasoning with four controlled injection scenarios.

Scenario A: Short anomaly (2h)        → expect EMERGING, NOT sustained
Scenario B: Sustained anomaly (6h)    → expect SUSTAINED, escalation ramp
Scenario C: Sustained + low custody   → expect CRITICAL, max escalation
Scenario D: Recovery after sustained   → expect RECOVERING → NORMAL

Usage:
    uv run python scripts/validate_reasoning.py
    uv run python scripts/validate_reasoning.py --vessel 367454140
    uv run python scripts/validate_reasoning.py --plot
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


def _find_candidate(df: pd.DataFrame) -> int:
    """Find a vessel with decent coverage and moderate baseline score."""
    for mmsi, g in df.groupby("mmsi"):
        if len(g) >= 20 and g["speed_kmh"].mean() > 2 and g["ml_anomaly_score"].mean() < 0.4:
            return int(mmsi)
    return int(df["mmsi"].iloc[0])


def _print_transition_trace(result: dict, label: str):
    """Print a compact transition trace table."""
    table = result["table"]
    s = result["summary"]
    print(f"\n{'=' * 80}")
    print(f"  SCENARIO: {label}")
    print(f"{'=' * 80}")
    print(f"  States observed: {', '.join(s['states_observed'])}")
    print(f"  Peak escalation: {s['peak_escalation']:.4f}")
    print(f"  Max duration: {s['max_duration_hours']}h")
    for key in ("first_emerging", "first_confirmed", "first_sustained",
                "first_critical", "first_recovering"):
        val = s.get(key)
        if val:
            print(f"  {key}: {val.split('+')[0]}")

    print(f"\n  {'Hour':>4}  {'ML':>6}  {'State':>12}  {'Agree':>15}  {'Dur':>3}  {'Esc':>6}  {'Speed':>7}  {'Hdg':>6}")
    print(f"  {'----':>4}  {'------':>6}  {'------------':>12}  {'---------------':>15}  {'---':>3}  {'------':>6}  {'-------':>7}  {'------':>6}")
    for i, (_, row) in enumerate(table.iterrows()):
        ts = str(row["timestamp"]).split(" ")[1].split("+")[0][:5]
        state = row["inj_state"]
        marker = ""
        if state in ("sustained", "critical"):
            marker = " <<"
        elif state == "recovering":
            marker = " vv"
        elif state in ("emerging", "confirmed"):
            marker = " !"
        print(f"  {i:>4}  {row['inj_ml_score']:>6.3f}  {state:>12}  {row['inj_agreement']:>15}  "
              f"{int(row['inj_duration_hours']):>3}  {row['inj_escalation']:>6.4f}  "
              f"{row['inj_speed']:>7.1f}  {row['inj_heading']:>6.1f}{marker}")


def _plot_scenarios(results: list[dict], vessel_id: int, out_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed -- skipping plot")
        return

    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(16, 3.5 * n), sharex=False)
    if n == 1:
        axes = [axes]
    fig.suptitle(f"MMSI {vessel_id} -- Reasoning Validation", fontsize=13)

    state_colors = {
        "normal": "#555", "emerging": "#e8c547", "confirmed": "#e07b00",
        "sustained": "#d73a49", "critical": "#ff2222", "recovering": "#2ea043",
    }

    for ax, result in zip(axes, results):
        table = result["table"]
        label = result["summary"]["label"]
        hours = range(len(table))

        ax.plot(hours, table["base_ml_score"], color="#666", linewidth=0.8,
                label="Baseline ML", linestyle="--")
        ax.plot(hours, table["inj_ml_score"], color="#d73a49", linewidth=1.5,
                label="Injected ML")

        # Color markers by state
        for i, (_, row) in enumerate(table.iterrows()):
            c = state_colors.get(row["inj_state"], "#555")
            ax.scatter(i, row["inj_ml_score"], color=c, s=20, zorder=5)

        # Escalation as bar overlay
        ax2 = ax.twinx()
        ax2.bar(hours, table["inj_escalation"], alpha=0.2, color="#e8c547", label="Escalation")
        ax2.set_ylabel("Escalation", fontsize=8, color="#e8c547")
        ax2.set_ylim(0, 0.15)

        ax.axhline(0.8, color="#444", linestyle=":", linewidth=0.7)
        ax.set_ylabel("ML Score")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=7, loc="upper left")

    axes[-1].set_xlabel("Hour index")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    print(f"\n  Plot saved to {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Validate reasoning with injection scenarios")
    parser.add_argument("--scored", default=os.path.join(_repo, "data", "ais_test_scored.parquet"))
    parser.add_argument("--model", default=os.path.join(_repo, "models", "isolation_forest.joblib"))
    parser.add_argument("--vessel", type=int, default=None)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    df = pd.read_parquet(args.scored)
    vessel_id = args.vessel or _find_candidate(df)
    ts = vessel_time_series(df, vessel_id)
    if ts.empty:
        print(f"Vessel {vessel_id} not found.")
        return
    baseline = ts.to_dict("records")
    scorer = MLAnomalyScorer(args.model)

    print(f"Vessel {vessel_id}: {len(baseline)} records")
    print(f"  Speed range: [{ts['speed_kmh'].min():.1f}, {ts['speed_kmh'].max():.1f}] km/h")

    results = []

    # ── Scenario A: Short anomaly (2h erratic heading) ───────────────
    inj_a = inject_heading_deviation(baseline, start_hour=10, duration_hours=2, mode="erratic")
    res_a = compare(baseline, inj_a, scorer, label="A: Short anomaly (2h erratic heading)")
    _print_transition_trace(res_a, "A: Short anomaly (2h)")
    results.append(res_a)

    # ── Scenario B: Sustained anomaly (6h zone approach) ─────────────
    inj_b = inject_zone_approach(baseline, start_hour=8, duration_hours=6,
                                 zone_lat=30.0, zone_lon=-88.0, approach_speed_kmh=15.0)
    res_b = compare(baseline, inj_b, scorer, label="B: Sustained anomaly (6h zone approach)")
    _print_transition_trace(res_b, "B: Sustained anomaly (6h)")
    results.append(res_b)

    # ── Scenario C: Sustained + simulated low custody ────────────────
    # Inject zone approach AND mark custody_confidence low in the window
    inj_c_base = inject_zone_approach(baseline, start_hour=6, duration_hours=8,
                                       zone_lat=30.0, zone_lon=-88.0, approach_speed_kmh=18.0)
    import copy
    inj_c = [copy.deepcopy(r) for r in inj_c_base]
    for i in range(6, min(14, len(inj_c))):
        inj_c[i]["custody_confidence"] = 0.2  # simulate weak custody
        inj_c[i]["anomaly_score"] = 1.5  # simulate high heuristic
    res_c = compare(baseline, inj_c, scorer, label="C: Sustained + low custody (8h, custody=0.2)")
    _print_transition_trace(res_c, "C: Sustained + low custody")
    results.append(res_c)

    # ── Scenario D: Recovery (6h anomaly then normal) ────────────────
    inj_d = inject_zone_approach(baseline, start_hour=6, duration_hours=6,
                                 zone_lat=30.0, zone_lon=-88.0, approach_speed_kmh=15.0)
    # The track naturally reverts after the injection window
    res_d = compare(baseline, inj_d, scorer, label="D: Recovery (6h anomaly then revert)")
    _print_transition_trace(res_d, "D: Recovery")
    results.append(res_d)

    # ── Validation summary ───────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print(f"  VALIDATION SUMMARY")
    print(f"{'=' * 80}")
    for r in results:
        s = r["summary"]
        states = ", ".join(s["states_observed"])
        print(f"  {s['label'][:50]:<50}  states=[{states}]  "
              f"peak_esc={s['peak_escalation']:.4f}  max_dur={s['max_duration_hours']}h")

    if args.plot:
        out_path = os.path.join(_repo, "data", f"reasoning_validation_{vessel_id}.png")
        _plot_scenarios(results, vessel_id, out_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
