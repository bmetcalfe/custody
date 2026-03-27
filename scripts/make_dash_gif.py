#!/usr/bin/env python
"""Generate a presentation GIF showing the Custody Dash portfolio view.

Renders each timestep as a frame: KPI strip + ranked portfolio table + map
sketch, assembled into a ~15-20 second GIF.

Usage:
    uv run python scripts/make_dash_gif.py
    uv run python scripts/make_dash_gif.py --output docs/progression.gif --fps 3
"""
import os
import sys
import argparse

_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_repo, "src"))
sys.path.insert(0, os.path.join(_repo, "src", "app"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
from PIL import Image
import io

from custody.simulation import run_multi_target_simulation, PORTFOLIO_SCENARIO
from portfolio_overview import build_overview_df, compute_kpi_counts, derive_display_status
import pandas as pd


# ---------------------------------------------------------------------------
# Colors (dark theme, matching Dash UI)
# ---------------------------------------------------------------------------

BG_COLOR = "#121212"
TEXT_COLOR = "#dddddd"
MUTED = "#888888"
BORDER = "#2d2d2d"

STATUS_COLORS = {
    "NEEDS ACTION": "#e63737",
    "PREEMPTED":    "#d78219",
    "NEGLECTED":    "#d7c31e",
    "STALE":        "#af692d",
    "APPROACHING":  "#b450dc",
    "WATCH":        "#55a5eb",
    "HEALTHY":      "#9ba5af",
}

STATUS_MAP_RGBA = {
    "NEEDS ACTION": (0.90, 0.22, 0.22, 0.9),
    "PREEMPTED":    (0.84, 0.51, 0.10, 0.9),
    "NEGLECTED":    (0.84, 0.76, 0.12, 0.9),
    "STALE":        (0.69, 0.41, 0.18, 0.8),
    "APPROACHING":  (0.71, 0.31, 0.86, 0.9),
    "WATCH":        (0.33, 0.65, 0.92, 0.8),
    "HEALTHY":      (0.61, 0.65, 0.69, 0.5),
}

TIER_COLORS = {
    "routine":  "#555555",
    "elevated": "#d78219",
    "priority": "#e63737",
    "urgent":   "#ff2222",
    "critical": "#ff0000",
}

ZONE_COLOR = (1.0, 0.84, 0.0, 0.15)
ZONE_EDGE = (1.0, 0.84, 0.0, 0.7)


def _render_frame(
    timestep_df: pd.DataFrame,
    all_records: list[dict],
    step_idx: int,
    n_steps: int,
    timestamp_str: str,
    figsize=(14, 7.5),
):
    """Render one timestep as a matplotlib figure."""
    fig = plt.figure(figsize=figsize, facecolor=BG_COLOR)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 5], width_ratios=[3, 2],
                  hspace=0.12, wspace=0.08)

    # ── KPI strip (top left) ────────────────────────────────────────
    ax_kpi = fig.add_subplot(gs[0, :])
    ax_kpi.set_facecolor(BG_COLOR)
    ax_kpi.axis("off")

    kpis = compute_kpi_counts(timestep_df)
    kpi_items = [
        ("Tracked", kpis["total"], TEXT_COLOR),
        ("Needs Action", kpis["needs_action"], "#e63737" if kpis["needs_action"] > 0 else MUTED),
        ("Elevated", sum(1 for _, r in timestep_df.iterrows() if r.get("tasking_tier") == "elevated"), "#d78219"),
        ("Priority", sum(1 for _, r in timestep_df.iterrows() if r.get("tasking_tier") in ("priority", "urgent")), "#e63737"),
        ("Neglected", kpis["neglected"], "#d7c31e" if kpis["neglected"] > 0 else MUTED),
        ("Stale/Lost", kpis["stale_or_lost"], "#af692d" if kpis["stale_or_lost"] > 0 else MUTED),
    ]

    for i, (label, value, color) in enumerate(kpi_items):
        x = 0.02 + i * 0.16
        ax_kpi.text(x, 0.7, str(value), fontsize=22, fontweight="bold",
                    color=color, transform=ax_kpi.transAxes, fontfamily="monospace")
        ax_kpi.text(x, 0.15, label, fontsize=8, color=MUTED,
                    transform=ax_kpi.transAxes)

    # Timestamp and step indicator
    ax_kpi.text(0.98, 0.5, f"Step {step_idx}/{n_steps - 1}  {timestamp_str}",
                fontsize=9, color=MUTED, transform=ax_kpi.transAxes,
                ha="right", va="center", fontfamily="monospace")

    # Title
    ax_kpi.text(0.0, 1.15, "CUSTODY", fontsize=13, fontweight="bold",
                color=TEXT_COLOR, transform=ax_kpi.transAxes)
    ax_kpi.text(0.09, 1.15, "  Portfolio Overview", fontsize=10,
                color=MUTED, transform=ax_kpi.transAxes)

    # ── Portfolio table (bottom left) ───────────────────────────────
    ax_tbl = fig.add_subplot(gs[1, 0])
    ax_tbl.set_facecolor(BG_COLOR)
    ax_tbl.axis("off")

    overview = build_overview_df(timestep_df)
    if overview.empty:
        ax_tbl.text(0.5, 0.5, "No data", color=MUTED, ha="center", fontsize=12)
    else:
        # Show top 15 rows
        display = overview.head(15)
        cols = ["Rank", "Entity", "Status", "Anomaly", "Health", "Action"]
        col_widths = [0.06, 0.14, 0.18, 0.10, 0.14, 0.14]
        col_x = [sum(col_widths[:i]) for i in range(len(cols))]

        # Header
        for j, col in enumerate(cols):
            ax_tbl.text(col_x[j], 0.97, col, fontsize=7.5, fontweight="bold",
                        color="#cccccc", transform=ax_tbl.transAxes)

        # Rows
        for i, (_, row) in enumerate(display.iterrows()):
            y = 0.92 - i * 0.058
            if y < 0.02:
                break

            status = str(row.get("Status", "HEALTHY"))
            tier = "routine"
            for _, orig_r in timestep_df.iterrows():
                if str(orig_r.get("target_id", "")) == str(row.get("Entity", "")):
                    tier = str(orig_r.get("tasking_tier", "routine"))
                    break

            # Row background tint
            if tier in ("priority", "urgent"):
                ax_tbl.axhspan(y - 0.02, y + 0.04, xmin=0, xmax=0.78,
                               color="#4d1a1a", alpha=0.5)
            elif tier == "elevated":
                ax_tbl.axhspan(y - 0.02, y + 0.04, xmin=0, xmax=0.78,
                               color="#3d3520", alpha=0.4)

            row_color = TEXT_COLOR if tier != "routine" else "#999999"
            vals = [
                str(row.get("Rank", "")),
                str(row.get("Entity", "")),
                status,
                str(row.get("Anomaly", "")),
                str(row.get("Health", "")),
                str(row.get("Action", "")),
            ]
            for j, val in enumerate(vals):
                c = STATUS_COLORS.get(status, row_color) if j == 2 else row_color
                ax_tbl.text(col_x[j], y, val, fontsize=7, color=c,
                            transform=ax_tbl.transAxes, fontfamily="monospace")

    # ── Map (bottom right) ──────────────────────────────────────────
    ax_map = fig.add_subplot(gs[1, 1])
    ax_map.set_facecolor("#0a0a12")

    # Zone polygon
    from custody.config import ZONES
    for z in ZONES:
        rect = mpatches.FancyBboxPatch(
            (z.min_lon, z.min_lat), z.max_lon - z.min_lon, z.max_lat - z.min_lat,
            boxstyle="round,pad=0.02", facecolor=ZONE_COLOR, edgecolor=ZONE_EDGE,
            linewidth=1.5,
        )
        ax_map.add_patch(rect)

    # Vessel dots
    for _, r in timestep_df.iterrows():
        lat = float(r.get("lat", 0))
        lon = float(r.get("lon", 0))
        ds = derive_display_status(r.to_dict())
        rgba = STATUS_MAP_RGBA.get(ds, (0.5, 0.5, 0.5, 0.4))
        size = 50 if ds in ("NEEDS ACTION", "PREEMPTED") else (35 if ds != "HEALTHY" else 15)
        ax_map.scatter(lon, lat, c=[rgba], s=size, edgecolors="none", zorder=3)

        # Label top-5 + scripted
        rank = int(r.get("portfolio_rank", 99))
        tid = str(r.get("target_id", ""))
        if rank <= 5 or not tid.startswith("BG-"):
            ax_map.text(lon + 0.04, lat + 0.03, tid, fontsize=5,
                        color="#cccccc", fontfamily="monospace", zorder=4)

    # Map styling
    ax_map.set_xlim(-2.5, 4.5)
    ax_map.set_ylim(-2.5, 4.5)
    ax_map.tick_params(colors=MUTED, labelsize=6)
    ax_map.set_xlabel("Longitude", fontsize=7, color=MUTED)
    ax_map.set_ylabel("Latitude", fontsize=7, color=MUTED)
    for spine in ax_map.spines.values():
        spine.set_color(BORDER)

    # Convert to PIL Image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor=BG_COLOR, edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    parser = argparse.ArgumentParser(description="Generate Custody portfolio GIF")
    parser.add_argument("--output", default=os.path.join(_repo, "docs", "progression.gif"))
    parser.add_argument("--fps", type=int, default=3, help="Frames per second")
    args = parser.parse_args()

    print("Running PORTFOLIO_SCENARIO...")
    records = run_multi_target_simulation(PORTFOLIO_SCENARIO)
    times = sorted({r["time"] for r in records})
    n_steps = len(times)
    print(f"  {n_steps} timesteps, {len({r['target_id'] for r in records})} entities")

    frames = []
    for step, t in enumerate(times):
        ts_recs = [r for r in records if r["time"] == t]
        ts_df = pd.DataFrame(ts_recs)
        ts_str = t.strftime("%b %d, %H:%M UTC")

        print(f"  Rendering step {step}/{n_steps - 1} ({ts_str})...")
        img = _render_frame(ts_df, records, step, n_steps, ts_str)
        frames.append(img)

    # Assemble GIF
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    duration_ms = int(1000 / args.fps)

    # Hold first and last frames longer
    durations = [duration_ms] * len(frames)
    durations[0] = 1500   # 1.5s on first frame
    durations[-1] = 2000  # 2s on last frame

    frames[0].save(
        args.output,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )

    total_sec = sum(durations) / 1000
    file_kb = os.path.getsize(args.output) / 1024
    print(f"\nSaved {args.output}")
    print(f"  {len(frames)} frames, {total_sec:.1f}s total, {file_kb:.0f} KB")


if __name__ == "__main__":
    main()
