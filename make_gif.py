"""
Generate an animated GIF showing the V001 reasoning-chain progression.

Layout: map (left) | reasoning stack (right)

Usage:
    uv run python make_gif.py

Output:
    docs/progression.gif  (~30 s, 10 frames, one per simulation hour)
"""
from __future__ import annotations

import sys
sys.path.insert(0, "src")

from datetime import datetime
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from PIL import Image

from custody.config import ZONES
from custody.simulate import run_simulation
from custody.compounds import evaluate_compounds
from custody.decision import build_decision
from custody.belief_assessment import build_fusion_assessment
from custody.models import TrackState
from custody.taskrecommendation import build_task_recommendations


# ── palette ─────────────────────────────────────────────────────────────────

BG     = "#0e1117"
CARD   = "#1a1d23"
BORDER = "#2d2d2d"
TEXT   = "#e0e0e0"
MUTED  = "#888888"
ACCENT = "#1f78b4"

MAP_SEA    = "#0a1520"
MAP_GRID   = "#1a2535"
TRACK_NORM = "#00c8ff"
TRACK_ANOM = "#ff8c00"
ZONE_FILL  = (1.0, 0.84, 0.0, 0.12)
ZONE_EDGE  = (1.0, 0.84, 0.0, 0.75)

ACTION_COLOR = {
    "PASSIVE_MONITOR": "#2ea043",
    "ELEVATE":         "#b08000",
    "TASK_OPTICAL":    "#e07b00",
    "TASK_SAR":        "#c94a00",
    "ESCALATE":        "#d73a49",
}
ACTION_LABEL = {
    "PASSIVE_MONITOR": "PASSIVE MONITOR",
    "ELEVATE":         "ELEVATE",
    "TASK_OPTICAL":    "TASK OPTICAL",
    "TASK_SAR":        "TASK SAR",
    "ESCALATE":        "ESCALATE",
}
SENSOR_COLOR = {
    "SAR":         "#1f78b4",
    "OPTICAL":     "#e07b00",
    "MONITOR":     "#555555",
    "AIS_REFRESH": "#2ea043",
}


# ── helpers ──────────────────────────────────────────────────────────────────

def _derive_track(record: dict, prefix: list[dict]) -> TrackState:
    unc  = float(record.get("uncertainty_km", 5.0))
    last_t, last_a = None, 0.0
    for r in reversed(prefix):
        result = r.get("collection_result")
        if r.get("action") == "TASK" and result not in (None, "", "—", "NONE"):
            t = r.get("time")
            if isinstance(t, datetime):
                last_t = t
                last_a = float(r.get("anomaly_score", 0.0))
            break
    return TrackState(
        uncertainty_km=unc,
        last_collection_time=last_t,
        last_collection_anomaly_score=last_a,
    )


def _hbar(ax, x, y, w, h, value: float, color: str) -> None:
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0", lw=0, facecolor=BORDER, zorder=2))
    ax.add_patch(FancyBboxPatch((x, y), max(value * w, 0.004), h,
        boxstyle="round,pad=0", lw=0, facecolor=color, zorder=3))


def _badge(ax, cx, cy, text, color, fs=9, w=0.13, h=0.032):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.007", lw=0, facecolor=color, zorder=4))
    ax.text(cx, cy, text, ha="center", va="center",
            color="white", fontsize=fs, fontweight="bold", zorder=5)


# ── map renderer ─────────────────────────────────────────────────────────────

def _render_map(ax, v001_all: list[dict], v002_all: list[dict],
                idx: int, record: dict) -> None:
    ax.set_facecolor(MAP_SEA)
    ax.set_xlim(0.0, 2.15)
    ax.set_ylim(-0.05, 2.05)
    ax.tick_params(colors=MUTED, labelsize=5.5)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.set_xlabel("Longitude", color=MUTED, fontsize=6)
    ax.set_ylabel("Latitude",  color=MUTED, fontsize=6)

    # grid
    ax.grid(color=MAP_GRID, linewidth=0.4, zorder=0)

    # sensitive zone
    for z in ZONES:
        w = z.max_lon - z.min_lon
        h = z.max_lat - z.min_lat
        ax.add_patch(mpatches.Rectangle(
            (z.min_lon, z.min_lat), w, h,
            linewidth=1.2, edgecolor=ZONE_EDGE, facecolor=ZONE_FILL, zorder=2))
        ax.text(z.min_lon + w / 2, z.max_lat + 0.04, z.name,
                ha="center", color="#ccaa00", fontsize=5.5, zorder=3)

    # V001 track up to current step
    v001_prev = v001_all[:idx + 1]
    for i in range(1, len(v001_prev)):
        p0 = v001_prev[i - 1]
        p1 = v001_prev[i]
        anom = float(p1.get("anomaly_score", 0))
        color = TRACK_ANOM if anom > 0.5 else TRACK_NORM
        ax.plot([p0["lon"], p1["lon"]], [p0["lat"], p1["lat"]],
                color=color, lw=1.8, zorder=4)

    # task / hold markers along V001 track
    for r in v001_prev:
        if r.get("action") == "TASK":
            ax.scatter(r["lon"], r["lat"], s=30, color="#ff3c3c",
                       zorder=6, edgecolors="white", linewidths=0.4)
        elif r.get("action") == "HOLD":
            ax.scatter(r["lon"], r["lat"], s=18, color="#50dc78",
                       zorder=6, edgecolors="white", linewidths=0.4)

    # V002 position at this timestep
    now_time = record.get("time")
    v2_now = [r for r in v002_all if r.get("time") == now_time]
    if v2_now:
        ax.scatter(v2_now[0]["lon"], v2_now[0]["lat"], s=22,
                   color="#aaaaaa", zorder=5, alpha=0.7)

    # current V001 position + uncertainty ring
    lat = float(record["lat"])
    lon = float(record["lon"])
    unc_deg = float(record.get("uncertainty_km", 5.0)) / 111.0  # rough km->deg
    ax.add_patch(Circle((lon, lat), unc_deg,
        color="#ff5050", alpha=0.18, zorder=5))
    ax.add_patch(Circle((lon, lat), unc_deg,
        fill=False, edgecolor="#ff5050", lw=0.8, alpha=0.7, zorder=6))
    ax.scatter(lon, lat, s=55, color="white", zorder=7,
               edgecolors="#cccccc", linewidths=0.6)

    # mini legend
    legend_items = [
        (TRACK_NORM, "Normal track"),
        (TRACK_ANOM, "Anomalous"),
        ("#ff3c3c",  "TASK"),
        ("#50dc78",  "HOLD"),
        ("#aaaaaa",  "Other target"),
    ]
    for i, (c, lbl) in enumerate(legend_items):
        ax.plot([], [], "o" if i >= 2 else "-",
                color=c, markersize=4, linewidth=1.5, label=lbl)
    ax.legend(fontsize=5, loc="lower right", framealpha=0.5,
              facecolor="#111", labelcolor="white",
              borderpad=0.5, handlelength=1.2)

    ax.set_title("Track Map  ·  V001", color=MUTED, fontsize=7, pad=4)


# ── reasoning panel renderer ──────────────────────────────────────────────────

def _render_reasoning(ax, record: dict, prefix: list[dict],
                       time_str: str) -> None:
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    track     = _derive_track(record, prefix)
    compounds = evaluate_compounds(record, window=prefix)
    fa        = build_fusion_assessment(record, compounds, track)
    decision  = build_decision(fa, record, track, compounds)
    tasks     = build_task_recommendations(decision, fa, record, track)
    now: datetime = record.get("time")

    behavior = record.get("behavior_state", "—")

    # header
    ax.text(0.02, 0.975, "V001", color=TEXT, fontsize=9.5, fontweight="bold", va="top")
    ax.text(0.98, 0.975, time_str, color=MUTED, fontsize=9, va="top", ha="right")
    ax.axhline(0.956, color=BORDER, lw=0.8)

    # current state strip
    ax.text(0.02, 0.940, "CURRENT STATE", color=MUTED, fontsize=6, fontweight="bold", va="top")
    strip = [
        ("Behavior",    behavior),
        ("Uncertainty", f"{record.get('uncertainty_km','—')} km"),
        ("Anomaly",     f"{record.get('anomaly_score', 0):.2f}"),
    ]
    for i, (lbl, val) in enumerate(strip):
        xi = 0.02 + i * 0.32
        ax.text(xi, 0.921, lbl, color=MUTED, fontsize=5.8, va="top")
        ax.text(xi, 0.906, val, color=TEXT, fontsize=7.5, fontweight="bold", va="top")
    ax.axhline(0.889, color=BORDER, lw=0.5)

    # ── fusion assessment ───────────────────────────────────────────────────
    y = 0.875
    ax.text(0.02, y, "FUSION ASSESSMENT", color=MUTED, fontsize=5.8, fontweight="bold", va="top")
    ax.axhline(y - 0.012, color=BORDER, lw=0.4)

    bary = y - 0.040
    barh = 0.016
    barw = 0.27
    bars = [
        ("FUSED SCORE",      fa.fused_score,      ACCENT,    0.02),
        ("UNCERTAINTY",      fa.uncertainty,      "#d97706",  0.36),
        ("SOURCE AGREEMENT", fa.source_agreement, "#2ea043",  0.70),
    ]
    for lbl, val, color, bx in bars:
        ax.text(bx, bary + 0.003, lbl, color=MUTED, fontsize=5.2, va="bottom")
        _hbar(ax, bx, bary - barh, barw, barh, val, color)
        ax.text(bx + barw + 0.007, bary - barh / 2, f"{val:.3f}",
                color=TEXT, fontsize=6.5, va="center")

    y_src = bary - barh - 0.030
    src = fa.recommended_confirming_source or "—"
    src_color = SENSOR_COLOR.get(src, "#555")
    ax.text(0.02, y_src + 0.012, "CONFIRMING SOURCE", color=MUTED, fontsize=5.2)
    _badge(ax, 0.02 + 0.048, y_src - 0.001, src, src_color, fs=7, w=0.088, h=0.022)

    ax.text(0.36, y_src + 0.012, "MISSING EVIDENCE", color=MUTED, fontsize=5.2)
    miss = ",  ".join(fa.missing_evidence) if fa.missing_evidence else "None"
    ax.text(0.36, y_src, miss, color=TEXT, fontsize=6.2, va="top")

    ax.axhline(y_src - 0.026, color=BORDER, lw=0.4)

    # ── decision ─────────────────────────────────────────────────────────────
    y_dec = y_src - 0.038
    ax.text(0.02, y_dec, "DECISION", color=MUTED, fontsize=5.8, fontweight="bold", va="top")
    ax.axhline(y_dec - 0.010, color=BORDER, lw=0.4)

    action_color = ACTION_COLOR.get(decision.action, "#555")
    action_label = ACTION_LABEL.get(decision.action, decision.action)

    badge_y = y_dec - 0.042
    _badge(ax, 0.13, badge_y, action_label, action_color, fs=9.5, w=0.21, h=0.034)

    for xi, lbl, val in [(0.44, "Priority",   decision.priority),
                          (0.72, "Confidence", decision.confidence)]:
        ax.text(xi, badge_y + 0.018, lbl, color=MUTED, fontsize=6, ha="center")
        ax.text(xi, badge_y - 0.002, f"{val:.3f}", color=TEXT, fontsize=9,
                fontweight="bold", ha="center", va="center")

    y_why = badge_y - 0.032
    for j, bullet in enumerate(decision.why[:3]):
        ax.text(0.02, y_why - j * 0.020,
                f"{j+1}.  {bullet[:90]}",
                color=TEXT, fontsize=5.8, va="top")

    ax.axhline(y_why - len(decision.why[:3]) * 0.020 - 0.005, color=BORDER, lw=0.4)

    # ── task queue ────────────────────────────────────────────────────────────
    y_tq = y_why - len(decision.why[:3]) * 0.020 - 0.018
    ax.text(0.02, y_tq, "TASK QUEUE", color=MUTED, fontsize=5.8, fontweight="bold", va="top")
    ax.axhline(y_tq - 0.010, color=BORDER, lw=0.4)

    rank_colors = ["#d73a49", "#e07b00", "#555"]
    y_card = y_tq - 0.026
    card_h = 0.054
    for i, task in enumerate(tasks[:3]):
        if y_card - card_h < 0.01:
            break
        rc = rank_colors[min(i, 2)]
        ax.add_patch(FancyBboxPatch(
            (0.02, y_card - card_h), 0.96, card_h - 0.004,
            boxstyle="round,pad=0.005", lw=0.5,
            edgecolor=BORDER, facecolor=CARD, zorder=2))

        circ = plt.Circle((0.055, y_card - card_h / 2), 0.012, color=rc, zorder=4)
        ax.add_patch(circ)
        ax.text(0.055, y_card - card_h / 2, str(task.rank),
                color="white", fontsize=6, fontweight="bold",
                ha="center", va="center", zorder=5)

        sc = SENSOR_COLOR.get(task.sensor, "#aaa")
        ax.text(0.078, y_card - card_h / 2 + 0.011, task.sensor,
                color=sc, fontsize=7.5, fontweight="bold", zorder=4)
        win_str = task.window_start.strftime("%H:%Mz") if task.window_start else "—"
        if now and task.window_start:
            tts = max(0, int((task.window_start - now).total_seconds() / 60))
            tts_str = f"TTS {tts} min" if tts else "TTS now"
        else:
            tts_str = "—"
        ax.text(0.078, y_card - card_h / 2 - 0.005, f"{win_str}  ·  {tts_str}",
                color=MUTED, fontsize=5.5, zorder=4)
        ax.text(0.97, y_card - card_h / 2 + 0.006, f"EV {task.expected_value:.3f}",
                color=TEXT, fontsize=7, fontweight="bold", ha="right", zorder=4)
        fall_str = " · ".join(task.fallbacks) if task.fallbacks else "—"
        ax.text(0.078, y_card - card_h + 0.005, f"Fallbacks: {fall_str}",
                color=MUTED, fontsize=5.2, zorder=4)
        y_card -= card_h + 0.005


# ── frame assembler ──────────────────────────────────────────────────────────

def render_frame(record: dict, prefix: list[dict],
                 v001_all: list[dict], v002_all: list[dict],
                 idx: int) -> Image.Image:
    now: datetime = record.get("time")
    time_str = now.strftime("%H:%Mz") if now else "—"

    fig = plt.figure(figsize=(13.5, 5.6), facecolor=BG)

    # two columns: map 43%, reasoning 57%
    ax_map  = fig.add_axes([0.02, 0.07, 0.40, 0.88])
    ax_info = fig.add_axes([0.45, 0.02, 0.54, 0.96])
    ax_info.set_facecolor(BG)

    _render_map(ax_map, v001_all, v002_all, idx, record)
    _render_reasoning(ax_info, record, prefix, time_str)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=130, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Running simulation...")
    timeline = run_simulation()

    v001 = [r for r in timeline if r.get("target_id") == "V001"]
    v002 = [r for r in timeline if r.get("target_id") == "V002"]
    print(f"  {len(v001)} V001 timesteps, {len(v002)} V002 timesteps")

    frames: list[Image.Image] = []
    for idx, record in enumerate(v001):
        prefix = v001[:idx]
        t = record.get("time")
        ts = t.strftime("%H:%Mz") if t else f"step {idx}"
        print(f"  rendering {ts}...", end="  ", flush=True)
        img = render_frame(record, prefix, v001, v002, idx)
        frames.append(img)
        print("done")

    if not frames:
        print("No frames generated.")
        return

    n = len(frames)
    duration_ms = max(int(30_000 / n), 800)
    out_path = "docs/progression.gif"
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    print(f"\nSaved: {out_path}  ({n} frames x {duration_ms} ms each)")


if __name__ == "__main__":
    main()
