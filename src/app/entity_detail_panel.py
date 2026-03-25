"""
Entity detail panel for the Custody Streamlit app.

Computes and renders the three-layer reasoning stack for a selected entity
at a specific playback step:

    FusionAssessment  →  Decision  →  TaskRecommendation queue

Public API
----------
render_entity_detail_panel(record, prefix_window)
    Main entry point.  Accepts the current record dict and the ordered
    prefix of prior records for the same entity.  Renders all three
    reasoning sections (Fusion Assessment, Decision, Task Queue) in place.
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Optional

import streamlit as st

from custody.compounds import evaluate_compounds
from custody.decision import Decision, build_decision
from custody.fusion import FusionAssessment, build_fusion_assessment
from custody.models import TrackState
from custody.taskrecommendation import TaskRecommendation, build_task_recommendations


# ---------------------------------------------------------------------------
# Action display config
# ---------------------------------------------------------------------------

_ACTION_COLOR: dict[str, str] = {
    "PASSIVE_MONITOR": "#2ea043",
    "ELEVATE":         "#b08000",
    "TASK_OPTICAL":    "#e07b00",
    "TASK_SAR":        "#c94a00",
    "ESCALATE":        "#d73a49",
}

_ACTION_LABEL: dict[str, str] = {
    "PASSIVE_MONITOR": "PASSIVE MONITOR",
    "ELEVATE":         "ELEVATE",
    "TASK_OPTICAL":    "TASK OPTICAL",
    "TASK_SAR":        "TASK SAR",
    "ESCALATE":        "ESCALATE",
}

_SOURCE_COLOR: dict[str, str] = {
    "OPTICAL": "#e07b00",
    "SAR":     "#1f78b4",
    "AIS":     "#2ea043",
}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _entity_id(record: dict) -> str:
    return (
        record.get("entity_id")
        or record.get("vessel_id")
        or record.get("target_id")
        or "unknown"
    )


def _derive_track_state(record: dict, prefix_window: list[dict]) -> TrackState:
    """Build a TrackState from the current record and its history prefix.

    Uses the record's uncertainty_km directly.  Scans the prefix backwards
    for the most recent TASK event with a non-empty collection_result to
    populate last_collection_time and last_collection_anomaly_score.
    """
    uncertainty_km = float(record.get("uncertainty_km", 5.0))
    last_collection_time: Optional[datetime] = None
    last_collection_anomaly_score: float = 0.0

    for r in reversed(prefix_window):
        result = r.get("collection_result")
        if r.get("action") == "TASK" and result not in (None, "", "—", "NONE"):
            t = r.get("time")
            if isinstance(t, datetime):
                last_collection_time = t
                last_collection_anomaly_score = float(r.get("anomaly_score", 0.0))
            break

    return TrackState(
        uncertainty_km=uncertainty_km,
        last_collection_time=last_collection_time,
        last_collection_anomaly_score=last_collection_anomaly_score,
    )


def _panel_summary(fa: FusionAssessment, decision: Decision) -> str:
    """One-sentence summary for the top of the reasoning panel."""
    fs  = fa.fused_score
    unc = fa.uncertainty
    fs_word  = "elevated" if fs  >= 0.60 else ("moderate" if fs  >= 0.35 else "low")
    unc_word = "unresolved" if unc >= 0.50 else ("moderate" if unc >= 0.30 else "low")
    action   = decision.action.replace("_", " ")
    return (
        f"Fused significance is **{fs_word}** ({fs:.2f}), "
        f"uncertainty is **{unc_word}** ({unc:.2f}), "
        f"and the system recommends **{action}** as the highest-value action."
    )


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def _render_prediction(record: dict) -> None:
    """Render the Prediction section if prediction data is present."""
    zone_prob = record.get("zone_probability")
    if zone_prob is None:
        return
    try:
        zone_prob = float(zone_prob)
        if math.isnan(zone_prob):
            return
    except (TypeError, ValueError):
        return

    st.markdown(
        "<div class='section-label'>Prediction</div>",
        unsafe_allow_html=True,
    )

    tte     = record.get("time_to_zone_hours")
    fa      = record.get("future_anomaly")
    pc      = record.get("prediction_confidence")
    horizon = record.get("prediction_horizon_hours")
    reason  = record.get("prediction_reason", "—")

    p1, p2, p3, p4, p5 = st.columns(5)
    with p1:
        st.metric("Zone Prob", f"{zone_prob:.0%}")
    with p2:
        try:
            _tte = float(tte) if tte is not None else None
            tte_str = f"{_tte:.1f}h" if _tte is not None and not math.isnan(_tte) else "—"
        except (TypeError, ValueError):
            tte_str = "—"
        st.metric("Time to Zone", tte_str)
    with p3:
        try:
            fa_str = f"{float(fa):.3f}" if fa is not None else "—"
        except (TypeError, ValueError):
            fa_str = "—"
        st.metric("Future Anomaly", fa_str)
    with p4:
        try:
            pc_str = f"{float(pc):.0%}" if pc is not None else "—"
        except (TypeError, ValueError):
            pc_str = "—"
        st.metric("Confidence", pc_str)
    with p5:
        try:
            hz_str = f"{float(horizon):.1f}h" if horizon is not None else "—"
        except (TypeError, ValueError):
            hz_str = "—"
        st.metric("Horizon", hz_str)

    if reason and reason != "—":
        st.markdown(
            f"<p style='font-size:0.80rem;color:#aaa;margin:4px 0 0 0;'>{reason}</p>",
            unsafe_allow_html=True,
        )


def _render_fusion_assessment(fa: FusionAssessment) -> None:
    st.markdown(
        "<div class='section-label'>Fusion Assessment</div>",
        unsafe_allow_html=True,
    )

    # Three score bars
    f1, f2, f3 = st.columns(3)
    with f1:
        st.caption("FUSED SCORE")
        st.progress(fa.fused_score, text=f"{fa.fused_score:.3f}")
    with f2:
        st.caption("UNCERTAINTY")
        st.progress(fa.uncertainty, text=f"{fa.uncertainty:.3f}")
    with f3:
        st.caption("SOURCE AGREEMENT")
        st.progress(fa.source_agreement, text=f"{fa.source_agreement:.3f}")

    # Confirming source badge | missing evidence list
    b1, b2 = st.columns([1, 3])
    with b1:
        src   = fa.recommended_confirming_source or "—"
        color = _SOURCE_COLOR.get(src, "#555")
        st.markdown(
            f"<div style='margin-top:10px;'>"
            f"<span style='color:#999;font-size:0.72rem;letter-spacing:0.06em;'>"
            f"CONFIRMING SOURCE</span><br>"
            f"<span style='display:inline-block;margin-top:4px;background:{color};"
            f"color:#fff;padding:3px 12px;border-radius:4px;"
            f"font-size:0.88rem;font-weight:700;'>{src}</span></div>",
            unsafe_allow_html=True,
        )
    with b2:
        st.markdown(
            "<span style='color:#999;font-size:0.72rem;letter-spacing:0.06em;'>"
            "MISSING EVIDENCE</span>",
            unsafe_allow_html=True,
        )
        if fa.missing_evidence:
            bullets = "".join(f"<li>{e}</li>" for e in fa.missing_evidence)
            st.markdown(
                f"<ul style='margin:4px 0 0 0;padding-left:1.2em;"
                f"font-size:0.82rem;line-height:1.6;'>{bullets}</ul>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<span style='color:#555;font-size:0.82rem;'>None identified.</span>",
                unsafe_allow_html=True,
            )


def _render_decision(decision: Decision) -> None:
    st.markdown(
        "<div class='section-label'>Decision</div>",
        unsafe_allow_html=True,
    )

    color = _ACTION_COLOR.get(decision.action, "#555")
    label = _ACTION_LABEL.get(decision.action, decision.action)

    # Badge + priority + confidence
    d1, d2, d3 = st.columns([3, 1, 1])
    with d1:
        st.markdown(
            f"<div style='padding-top:4px;'>"
            f"<span style='background:{color};color:#fff;"
            f"padding:5px 16px;border-radius:5px;"
            f"font-size:1.05rem;font-weight:700;letter-spacing:0.06em;'>"
            f"{label}</span></div>",
            unsafe_allow_html=True,
        )
    with d2:
        st.metric("Priority", f"{decision.priority:.3f}")
    with d3:
        st.metric("Confidence", f"{decision.confidence:.3f}")

    # Why bullets
    if decision.why:
        items = "".join(
            f"<li style='margin-bottom:5px;'>{b}</li>" for b in decision.why
        )
        st.markdown(
            f"<ol style='margin:10px 0 6px 0;padding-left:1.4em;"
            f"font-size:0.83rem;line-height:1.55;'>{items}</ol>",
            unsafe_allow_html=True,
        )

    # Next best actions as arrow-separated pills
    if decision.next_best_actions:
        pills = " &nbsp;→&nbsp; ".join(
            f"<span style='background:#2d2d2d;color:#ccc;"
            f"padding:2px 9px;border-radius:4px;"
            f"font-size:0.74rem;font-family:monospace;'>{a}</span>"
            for a in decision.next_best_actions
        )
        st.markdown(
            f"<div style='margin-top:8px;'>"
            f"<span style='color:#999;font-size:0.72rem;letter-spacing:0.06em;'>"
            f"NEXT BEST ACTIONS</span><br>"
            f"<div style='margin-top:5px;'>{pills}</div></div>",
            unsafe_allow_html=True,
        )


def _render_task_queue(tasks: list[TaskRecommendation], now: Optional[datetime]) -> None:
    st.markdown(
        "<div class='section-label'>Task Queue</div>",
        unsafe_allow_html=True,
    )

    if not tasks:
        st.markdown(
            "<span style='color:#666;'>No task recommendations available for this step.</span>",
            unsafe_allow_html=True,
        )
        return

    for task in tasks:
        window_str = task.window_start.strftime("%H:%Mz") if task.window_start else "—"
        if now and task.window_start:
            tts_sec = max(0.0, (task.window_start - now).total_seconds())
            tts_str = f"{int(tts_sec / 60)} min" if tts_sec > 0 else "now"
        else:
            tts_str = "—"

        fallback_str = (
            " · ".join(task.fallbacks) if task.fallbacks else "none"
        )

        rank_color = (
            "#d73a49" if task.rank == 1
            else "#e07b00" if task.rank == 2
            else "#555"
        )

        st.markdown(
            f"<div style='border:1px solid #2d2d2d;border-radius:6px;"
            f"padding:10px 14px;margin-bottom:8px;'>"

            # Header row: rank badge, sensor, window, TTS, expected value
            f"<div style='display:flex;align-items:center;gap:12px;flex-wrap:wrap;'>"
            f"<span style='background:{rank_color};color:#fff;border-radius:50%;"
            f"width:22px;height:22px;display:inline-flex;align-items:center;"
            f"justify-content:center;font-size:0.75rem;font-weight:700;'>"
            f"{task.rank}</span>"
            f"<span style='font-weight:700;font-size:0.92rem;font-family:monospace;'>"
            f"{task.sensor}</span>"
            f"<span style='color:#999;font-size:0.78rem;'>{window_str}</span>"
            f"<span style='color:#999;font-size:0.78rem;'>TTS {tts_str}</span>"
            f"<span style='margin-left:auto;font-size:0.85rem;font-weight:600;'>"
            f"EV {task.expected_value:.3f}</span>"
            f"</div>"

            # Reason
            f"<div style='margin-top:6px;font-size:0.81rem;line-height:1.45;'>"
            f"{task.reason}</div>"

            # Fallbacks
            f"<div style='margin-top:4px;color:#666;font-size:0.74rem;'>"
            f"Fallbacks: {fallback_str}</div>"

            f"</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def render_entity_detail_panel(record: dict, prefix_window: list[dict]) -> None:
    """Compute and render the full reasoning stack for one entity at one step.

    Runs: FusionAssessment → Decision → TaskRecommendation, then renders
    three sections: Fusion Assessment, Decision, and Task Queue.

    Args:
        record:        The current timeline record dict (selected step).
        prefix_window: All prior records for this entity, in order, excluding
                       the current record.  Used for compound evaluation and
                       TrackState derivation.
    """
    # ── Pipeline computation ─────────────────────────────────────────────────
    try:
        track     = _derive_track_state(record, prefix_window)
        compounds = evaluate_compounds(record, window=prefix_window)
        fa        = build_fusion_assessment(record, compounds, track)
        decision  = build_decision(fa, record, track, compounds)
        tasks     = build_task_recommendations(decision, fa, record, track)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Reasoning pipeline unavailable for this step: {exc}")
        return

    # ── One-sentence summary ─────────────────────────────────────────────────
    st.markdown(
        f"<p style='font-size:0.85rem;color:#ccc;margin:10px 0 2px 0;'>"
        f"{_panel_summary(fa, decision)}</p>",
        unsafe_allow_html=True,
    )

    # ── Three reasoning sections ─────────────────────────────────────────────
    now: Optional[datetime] = record.get("time")

    _render_prediction(record)
    _render_fusion_assessment(fa)
    _render_decision(decision)
    _render_task_queue(tasks, now)
