"""
Generate an animated GIF showing the V001 reasoning-chain progression.

Usage:
    uv run python make_gif.py

Output:
    docs/progression.gif  (~30 s, 10 frames, one per simulation hour)
"""
from __future__ import annotations

import sys
sys.path.insert(0, "src")

from datetime import datetime, timezone
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from PIL import Image

from custody.simulate import run_simulation
from custody.compounds import evaluate_compounds
from custody.decision import build_decision
from custody.fusion import build_fusion_assessment
from custody.models import TrackState
from custody.taskrecommendation import build_task_recommendations


# ── colour palette ─────────────────────────────────────────────────────────

BG       = "#0e1117"
CARD     = "#1a1d23"
BORDER   = "#2d2d2d"
TEXT     = "#e0e0e0"
MUTED    = "#888888"
ACCENT   = "#1f78b4"

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
    "SAR":     "#1f78b4",
    "OPTICAL": "#e07b00",
    "MONITOR": "#555555",
    "AIS_REFRESH": "#2ea043",
}


# ── helpers ─────────────────────────────────────────────────────────────────

def _derive_track(record: dict, prefix: list[dict]) -> TrackState:
    unc = float(record.get("uncertainty_km", 5.0))
    last_t = None
    last_a = 0.0
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


def _bar(ax, x, y, w, h, value: float, color: str, bg: str = BORDER) -> None:
    """Draw a horizontal progress bar."""
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0", linewidth=0,
        facecolor=bg, zorder=2,
    ))
    fill_w = max(value * w, 0.004)
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), fill_w, h,
        boxstyle="round,pad=0", linewidth=0,
        facecolor=color, zorder=3,
    ))


def _badge(ax, cx, cy, text: str, color: str,
           fontsize: float = 9, w: float = 0.14, h: float = 0.035) -> None:
    ax.add_patch(mpatches.FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.008", linewidth=0,
        facecolor=color, zorder=4,
    ))
    ax.text(cx, cy, text, ha="center", va="center",
            color="white", fontsize=fontsize, fontweight="bold", zorder=5)


# ── frame renderer ─────────────────────────────────────────────────────────

def render_frame(record: dict, prefix: list[dict]) -> Image.Image:
    track     = _derive_track(record, prefix)
    compounds = evaluate_compounds(record, window=prefix)
    fa        = build_fusion_assessment(record, compounds, track)
    decision  = build_decision(fa, record, track, compounds)
    tasks     = build_task_recommendations(decision, fa, record, track)

    # ── layout ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5.4))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    now: datetime = record.get("time")
    time_str = now.strftime("%H:%Mz") if now else "—"
    behavior = record.get("behavior_state", record.get("behavior", "—"))

    # ── header ──────────────────────────────────────────────────────────────
    ax.text(0.03, 0.965, "CUSTODY  ·  V001", color=TEXT,
            fontsize=9, fontweight="bold", va="top")
    ax.text(0.97, 0.965, time_str, color=MUTED,
            fontsize=9, va="top", ha="right")
    ax.axhline(0.948, color=BORDER, linewidth=0.8)

    # ── current state strip ─────────────────────────────────────────────────
    ax.text(0.03, 0.930, "CURRENT STATE", color=MUTED,
            fontsize=6.5, fontweight="bold", va="top", alpha=0.8)

    labels_vals = [
        ("Behavior",      behavior),
        ("Uncertainty",   f"{record.get('uncertainty_km', '—')} km"),
        ("Anomaly",       f"{record.get('anomaly_score', 0):.2f}"),
    ]
    for i, (lbl, val) in enumerate(labels_vals):
        xi = 0.03 + i * 0.22
        ax.text(xi, 0.908, lbl, color=MUTED, fontsize=6.5, va="top")
        ax.text(xi, 0.892, val, color=TEXT, fontsize=8, fontweight="bold", va="top")

    ax.axhline(0.874, color=BORDER, linewidth=0.6)

    # ── fusion assessment ───────────────────────────────────────────────────
    y_fa = 0.860
    ax.text(0.03, y_fa, "FUSION ASSESSMENT", color=MUTED,
            fontsize=6.5, fontweight="bold", va="top", alpha=0.8)
    ax.axhline(y_fa - 0.010, color=BORDER, linewidth=0.5)

    bar_y = y_fa - 0.040
    bar_h = 0.018
    bar_w = 0.26

    items = [
        ("FUSED SCORE",      fa.fused_score,      ACCENT,    0.03),
        ("UNCERTAINTY",      fa.uncertainty,      "#d97706",  0.36),
        ("SOURCE AGREEMENT", fa.source_agreement, "#2ea043",  0.69),
    ]
    for lbl, val, color, bx in items:
        ax.text(bx, bar_y + 0.004, lbl, color=MUTED, fontsize=5.8, va="bottom")
        _bar(ax, bx, bar_y - bar_h, bar_w, bar_h, val, color)
        ax.text(bx + bar_w + 0.008, bar_y - bar_h / 2, f"{val:.3f}",
                color=TEXT, fontsize=7, va="center")

    # confirming source badge
    src = fa.recommended_confirming_source or "—"
    src_color = SENSOR_COLOR.get(src, "#555")
    y_src = bar_y - bar_h - 0.034
    ax.text(0.03, y_src + 0.014, "CONFIRMING SOURCE", color=MUTED, fontsize=5.8)
    _badge(ax, 0.03 + 0.055, y_src - 0.002, src, src_color,
           fontsize=8, w=0.10, h=0.026)

    # missing evidence
    ax.text(0.36, y_src + 0.014, "MISSING EVIDENCE", color=MUTED, fontsize=5.8)
    miss_txt = ",  ".join(fa.missing_evidence) if fa.missing_evidence else "None"
    ax.text(0.36, y_src, miss_txt, color=TEXT, fontsize=7, va="top")

    ax.axhline(y_src - 0.030, color=BORDER, linewidth=0.5)

    # ── decision ─────────────────────────────────────────────────────────────
    y_dec = y_src - 0.044
    ax.text(0.03, y_dec, "DECISION", color=MUTED,
            fontsize=6.5, fontweight="bold", va="top", alpha=0.8)
    ax.axhline(y_dec - 0.010, color=BORDER, linewidth=0.5)

    action_color = ACTION_COLOR.get(decision.action, "#555")
    action_label = ACTION_LABEL.get(decision.action, decision.action)

    # action badge — large centrepiece
    badge_y = y_dec - 0.046
    _badge(ax, 0.14, badge_y, action_label, action_color,
           fontsize=10.5, w=0.22, h=0.038)

    # priority + confidence
    for xi, lbl, val in [(0.44, "Priority", decision.priority),
                          (0.68, "Confidence", decision.confidence)]:
        ax.text(xi, badge_y + 0.022, lbl, color=MUTED, fontsize=6.5,
                ha="center", va="center")
        ax.text(xi, badge_y - 0.004, f"{val:.3f}", color=TEXT, fontsize=10,
                fontweight="bold", ha="center", va="center")

    # why bullets (up to 3)
    y_why = badge_y - 0.036
    for j, bullet in enumerate(decision.why[:3]):
        ax.text(0.03, y_why - j * 0.022,
                f"{j+1}.  {bullet[:85]}",
                color=TEXT, fontsize=6.4, va="top", wrap=False)

    ax.axhline(y_why - len(decision.why[:3]) * 0.022 - 0.006, color=BORDER, linewidth=0.5)

    # ── task queue ───────────────────────────────────────────────────────────
    y_tq = y_why - len(decision.why[:3]) * 0.022 - 0.020
    ax.text(0.03, y_tq, "TASK QUEUE", color=MUTED,
            fontsize=6.5, fontweight="bold", va="top", alpha=0.8)
    ax.axhline(y_tq - 0.010, color=BORDER, linewidth=0.5)

    rank_colors = ["#d73a49", "#e07b00", "#555"]
    y_card = y_tq - 0.028
    card_h = 0.058
    for i, task in enumerate(tasks[:3]):
        if y_card - card_h < 0.01:
            break
        rc = rank_colors[min(i, 2)]
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.03, y_card - card_h), 0.94, card_h - 0.004,
            boxstyle="round,pad=0.006", linewidth=0.6,
            edgecolor=BORDER, facecolor=CARD, zorder=2,
        ))
        # rank circle
        circ = plt.Circle((0.065, y_card - card_h / 2), 0.013,
                           color=rc, zorder=4)
        ax.add_patch(circ)
        ax.text(0.065, y_card - card_h / 2, str(task.rank),
                color="white", fontsize=6.5, fontweight="bold",
                ha="center", va="center", zorder=5)

        sensor_color = SENSOR_COLOR.get(task.sensor, "#aaa")
        ax.text(0.090, y_card - card_h / 2 + 0.013, task.sensor,
                color=sensor_color, fontsize=8, fontweight="bold", zorder=4)
        win_str = task.window_start.strftime("%H:%Mz") if task.window_start else "—"
        if now and task.window_start:
            tts = max(0, int((task.window_start - now).total_seconds() / 60))
            tts_str = f"TTS {tts} min" if tts else "TTS now"
        else:
            tts_str = "—"
        ax.text(0.090, y_card - card_h / 2 - 0.004, f"{win_str}  ·  {tts_str}",
                color=MUTED, fontsize=6.2, zorder=4)
        ax.text(0.96, y_card - card_h / 2 + 0.008, f"EV {task.expected_value:.3f}",
                color=TEXT, fontsize=7.5, fontweight="bold",
                ha="right", zorder=4)
        fall_str = " · ".join(task.fallbacks) if task.fallbacks else "—"
        ax.text(0.090, y_card - card_h + 0.006, f"Fallbacks: {fall_str}",
                color=MUTED, fontsize=5.8, zorder=4)

        y_card -= card_h + 0.006

    plt.tight_layout(pad=0.2)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=140, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    print("Running simulation…")
    timeline = run_simulation()

    # Filter to V001 records only
    v001 = [r for r in timeline if r.get("target_id") == "V001"]
    print(f"  {len(v001)} V001 timesteps found")

    frames: list[Image.Image] = []
    for idx, record in enumerate(v001):
        prefix = v001[:idx]
        t = record.get("time")
        ts = t.strftime("%H:%Mz") if t else f"step {idx}"
        print(f"  rendering {ts}…", end="  ", flush=True)
        img = render_frame(record, prefix)
        frames.append(img)
        print("done")

    if not frames:
        print("No frames generated — aborting.")
        return

    # Total duration ~30 s across all frames
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
