#!/usr/bin/env python
"""Generate the final production GIF for Custody README.

Two-panel layout: portfolio table (left 55%) + map (right 45%).
Three text overlays. ~10 seconds. Dark theme. No clutter.

Usage:
    uv run python scripts/make_final_gif.py
"""
import os
import sys
import io

_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_repo, "src"))
sys.path.insert(0, os.path.join(_repo, "src", "app"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from PIL import Image, ImageDraw, ImageFont
import pandas as pd

from custody.simulation import run_multi_target_simulation, PORTFOLIO_SCENARIO
from portfolio_overview import build_overview_df, derive_display_status
from custody.config import ZONES


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

BG = "#0f1117"
MUTED = "#555555"
BORDER = "#2a2a2a"

# Tier text: routine nearly invisible, elevated warm, priority bright
TIER_TEXT = {
    "routine":  "#363636",      # very dark — disappears into background
    "elevated": "#f0c050",      # bright warm amber — unmistakable jump
    "priority": "#ff7070",      # bright salmon-red — impossible to miss
    "urgent":   "#ff5555",
    "critical": "#ff3333",
}

# Tier row backgrounds: elevated has a clear warm tint, priority glows
TIER_BG = {
    "priority": ("#5a1818", 0.85),  # strong red glow
    "urgent":   ("#6a1010", 0.90),
    "elevated": ("#4a3010", 0.65),  # warm amber band — visible jump
}

# Fix 2: HEALTHY → ROUTINE in display
STATUS_DISPLAY = {
    "HEALTHY":      "ROUTINE",
    "NEEDS ACTION": "PRIORITY",
    "APPROACHING":  "APPROACHING",
    "WATCH":        "ELEVATED",
}

STATUS_COLORS = {
    "ROUTINE":      "#363636",
    "PRIORITY":     "#ff5555",
    "APPROACHING":  "#b450dc",
    "ELEVATED":     "#e8b84a",
}

# Map dots — anomaly-only emphasis
MAP_RGBA = {
    "NEEDS ACTION": (1.00, 0.35, 0.35, 1.0),   # red — priority
    "APPROACHING":  (0.80, 0.40, 0.95, 0.95),  # purple
    "WATCH":        (0.90, 0.72, 0.20, 0.90),   # amber — elevated
    "HEALTHY":      (0.45, 0.50, 0.55, 0.50),   # visible gray
}

SCRIPTED = {"BRAVO-1", "ECHO-1", "ECHO-2", "PORT-1"}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _render_frame(ts_df, step, n_steps, ts_str):
    """Render one frame: table left + map right."""
    fig = plt.figure(figsize=(12, 6), facecolor=BG, dpi=100)
    gs = GridSpec(1, 2, figure=fig, width_ratios=[55, 45], wspace=0.04)

    # ── Table (left) ────────────────────────────────────────────────
    ax_t = fig.add_subplot(gs[0, 0])
    ax_t.set_facecolor(BG)
    ax_t.axis("off")

    overview = build_overview_df(ts_df)
    display = overview.head(15) if not overview.empty else overview

    # Header
    cols = ["Rank", "Entity", "Status", "Action"]
    x_pos = [0.01, 0.08, 0.24, 0.50]
    for j, col in enumerate(cols):
        ax_t.text(x_pos[j], 0.97, col, fontsize=7.5, fontweight="bold",
                  color="#888888", transform=ax_t.transAxes, fontfamily="monospace")

    # Separator
    ax_t.plot([0.01, 0.68], [0.95, 0.95], color=BORDER, linewidth=0.5,
              transform=ax_t.transAxes, clip_on=False)

    for i, (_, row) in enumerate(display.iterrows()):
        y = 0.91 - i * 0.057
        if y < 0.01:
            break

        status_raw = str(row.get("Status", "HEALTHY"))
        status = STATUS_DISPLAY.get(status_raw, status_raw)
        entity = str(row.get("Entity", ""))

        # Visual tier derived from DISPLAY STATUS only (not engine tasking_tier).
        # This ensures row highlighting matches the anomaly-only status model.
        if status in ("PRIORITY", "NEEDS ACTION"):
            vis_tier = "priority"
        elif status in ("ELEVATED", "WATCH"):
            vis_tier = "elevated"
        elif status == "APPROACHING":
            vis_tier = "elevated"
        else:
            vis_tier = "routine"

        # Row background — disabled to eliminate any possible bleed.
        # Priority emphasis is text-only (red text + bold).
        # No axhspan, no patches, no fills.

        # Text colors driven by visual tier
        tc = TIER_TEXT.get(vis_tier, "#363636")
        sc = STATUS_COLORS.get(status, tc)

        # Priority rows get bold + larger font
        is_priority = vis_tier == "priority"
        entity_size = 7.5 if is_priority else 6.8
        status_size = 7.2 if is_priority else 6.8

        vals = [str(row.get("Rank", "")), entity, status, str(row.get("Action", ""))]
        sizes = [6.8, entity_size, status_size, 6.8]
        for j, (val, sz) in enumerate(zip(vals, sizes)):
            c = sc if j == 2 else tc
            weight = "bold" if is_priority and j in (1, 2) else "normal"
            ax_t.text(x_pos[j], y, val, fontsize=sz, color=c, fontweight=weight,
                      transform=ax_t.transAxes, fontfamily="monospace")

    # Timestamp
    ax_t.text(0.78, 0.97, ts_str, fontsize=7, color=MUTED,
              transform=ax_t.transAxes, ha="right", fontfamily="monospace")

    # ── Map (right) ─────────────────────────────────────────────────
    ax_m = fig.add_subplot(gs[0, 1])
    ax_m.set_facecolor("#080810")

    # Zone
    for z in ZONES:
        rect = mpatches.FancyBboxPatch(
            (z.min_lon, z.min_lat), z.max_lon - z.min_lon, z.max_lat - z.min_lat,
            boxstyle="round,pad=0.01",
            facecolor=(1, 0.84, 0, 0.10), edgecolor=(1, 0.84, 0, 0.40),
            linewidth=1.0,
        )
        ax_m.add_patch(rect)

    # Vessels — Fix 3: size varies much more by tier
    for _, r in ts_df.iterrows():
        lat = float(r.get("lat", 0))
        lon = float(r.get("lon", 0))
        tid = str(r.get("target_id", ""))
        ds = derive_display_status(r.to_dict())
        rgba = MAP_RGBA.get(ds, (0.3, 0.3, 0.3, 0.2))

        # Map dot size driven by display status (anomaly-only)
        if ds == "NEEDS ACTION":
            size = 110
            edge = "white"
            ew = 1.5
        elif ds in ("WATCH", "APPROACHING"):
            size = 55
            edge = "#e8b84a"
            ew = 0.8
        elif tid in SCRIPTED:
            size = 25
            edge = "none"
            ew = 0
        else:
            size = 18
            edge = "none"
            ew = 0

        ax_m.scatter(lon, lat, c=[rgba], s=size, edgecolors=edge,
                     linewidths=ew, zorder=3)

        # Labels: scripted + anomaly-status vessels only
        show_label = tid in SCRIPTED or ds in ("NEEDS ACTION", "WATCH", "APPROACHING")
        if show_label:
            label_c = "#dddddd" if ds == "NEEDS ACTION" else "#999999"
            ax_m.text(lon + 0.06, lat + 0.06, tid, fontsize=5.5,
                      color=label_c, fontfamily="monospace", zorder=4)

    ax_m.set_xlim(-2.3, 4.3)
    ax_m.set_ylim(-2.3, 4.3)
    ax_m.tick_params(colors="#333333", labelsize=5)
    for spine in ax_m.spines.values():
        spine.set_color(BORDER)
    ax_m.set_xlabel("")
    ax_m.set_ylabel("")

    # Convert to PIL — fixed bbox to prevent frame-to-frame shift
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100,
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# Fix 4: Lighter, thinner annotation bar
def _add_overlay(img, text):
    """Add a bottom-center caption with thin translucent strip."""
    w, h = img.size

    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except (OSError, IOError):
        font = ImageFont.load_default()

    # Measure text
    tmp_draw = ImageDraw.Draw(img)
    bbox = tmp_draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    # Thin, subtle strip — caption feel
    strip_h = th + 8
    strip_y = h - strip_h - 4

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    odraw.rectangle([(0, strip_y), (w, strip_y + strip_h)],
                    fill=(0, 0, 0, 95))     # 37% opacity — very light
    odraw.text(((w - tw) // 2, strip_y + 3), text, font=font,
               fill=(200, 200, 200, 210))   # soft gray-white
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def main():
    out_path = os.path.join(_repo, "docs", "progression.gif")

    print("Running PORTFOLIO_SCENARIO...")
    records = run_multi_target_simulation(PORTFOLIO_SCENARIO)
    times = sorted({r["time"] for r in records})
    n = len(times)
    print(f"  {n} timesteps, {len({r['target_id'] for r in records})} entities")

    # ── Frame timing ─────────────────────────────────────────────────
    # Fix 5: Clear arc with deliberate pause at peak
    frame_durations = []
    annotations = {}

    for step in range(n):
        if step == 0:
            dur = 3800      # hold opening
        elif step <= 6:
            dur = 1200      # baseline
        elif step <= 16:
            dur = 1100      # emerging
        elif step == 20:
            dur = 2700      # PEAK HOLD
        elif step <= 24:
            dur = 1300      # around peak
        elif step <= 34:
            dur = 850       # resolution
        elif step == n - 1:
            dur = 4500      # hold final calm
        else:
            dur = 1200

        frame_durations.append(dur)

    # Annotations — one at a time, clean transitions
    for step in range(0, 8):
        annotations[step] = "Most vessels remain routine"
    for step in range(11, 19):
        annotations[step] = "A small subset shows anomalous behavior"
    for step in range(19, 27):
        annotations[step] = "Only persistent anomalies reach priority"
    # No annotation for final resolution frames

    # ── Render ───────────────────────────────────────────────────────
    frames = []
    for step, t in enumerate(times):
        ts_recs = [r for r in records if r["time"] == t]
        ts_df = pd.DataFrame(ts_recs)
        ts_str = t.strftime("%b %d, %H:%M")

        print(f"  Frame {step}/{n-1} ({ts_str})")
        img = _render_frame(ts_df, step, n, ts_str)

        if step in annotations:
            img = _add_overlay(img, annotations[step])

        frames.append(img)

    # ── Assemble GIF ─────────────────────────────────────────────────
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_durations,
        loop=0,
        optimize=True,
    )

    total_s = sum(frame_durations) / 1000
    size_kb = os.path.getsize(out_path) / 1024
    print(f"\nSaved {out_path}")
    print(f"  {len(frames)} frames, {total_s:.1f}s, {size_kb:.0f} KB")


if __name__ == "__main__":
    main()
