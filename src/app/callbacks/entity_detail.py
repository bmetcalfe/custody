"""Entity detail callbacks: reasoning chain + supporting tables.

Split into two callbacks per the design constraints:
  1. Reasoning callback: prediction, fusion, decision, summary, task queue
  2. Tables callback: alerts, compounds, compound history, orbital passes, traces
"""
from __future__ import annotations

import math
from datetime import datetime

from dash import Dash, Input, Output, html, no_update

import state as app_state
from adapter import entity_timeline_up_to
from entity_detail_data import derive_track_state, panel_summary

from custody.alerts import alerts_for_timeline
from custody.compounds import evaluate_compounds
from custody.decision import build_decision
from custody.decision_trace import traces_to_rows
from custody.fusion import build_fusion_assessment
from custody.roles import build_deliberation
from custody.taskrecommendation import build_task_recommendations

from compound_panels import (
    aggregate_compound_history,
    build_active_compounds_df,
    build_compound_history_df,
)
from orbital_passes_panel import build_orbital_passes_rows

from layout.entity_detail import (
    DETAIL_SUMMARY,
    PRED_ZONE_PROB, PRED_TIME_TO_ZONE, PRED_FUTURE_ANOMALY,
    PRED_CONFIDENCE, PRED_HORIZON,
    ML_ANOMALY_SCORE, ANOMALY_AGREEMENT_BADGE, ANOMALY_STATE_BADGE, ANOMALY_DURATION,
    FUSION_SCORE, FUSION_UNCERTAINTY, FUSION_AGREEMENT,
    FUSION_SOURCE, FUSION_MISSING,
    DETAIL_HEADING,
    DECISION_ACTION, DECISION_PRIORITY, DECISION_CONFIDENCE,
    DECISION_WHY, DECISION_FALLBACKS, TASK_QUEUE,
    ALERTS_TABLE, COMPOUNDS_TABLE, COMPOUND_HISTORY_TABLE,
    ORBITAL_TABLE, TRACES_TABLE, DETAIL_ACCORDION,
    ROLE_ANALYST_CARD, ROLE_COLLECTOR_CARD, ROLE_OPERATOR_CARD,
    ROLE_RESOLUTION, ROLE_WHY_NOT,
    COND_LOCAL_TIME, COND_SUN_STATE, COND_OPTICAL_VIABLE,
    COND_SAR_VIABLE, COND_CLOUD_COVER,
)

# Action → badge color
_ACTION_COLOR = {
    "PASSIVE_MONITOR": "#2ea043",
    "ELEVATE":         "#b08000",
    "TASK_OPTICAL":    "#e07b00",
    "TASK_SAR":        "#c94a00",
    "ESCALATE":        "#d73a49",
}

# Confirming source → badge color
_SOURCE_COLOR = {
    "OPTICAL": "#e07b00",
    "SAR":     "#1f78b4",
    "AIS":     "#2ea043",
}


def _fmt(val, fmt_str=".3f", fallback="—"):
    """Safe format a possibly-None/NaN value."""
    if val is None:
        return fallback
    try:
        v = float(val)
        if math.isnan(v):
            return fallback
        return f"{v:{fmt_str}}"
    except (TypeError, ValueError):
        return fallback


def _collection_conditions(utc_time, longitude):
    """Derive collection conditions from UTC time and longitude.

    Returns a dict with local_time, sun_state, optical, sar, cloud_cover.
    """
    fallback = {
        "local_time": "—", "sun_state": "—",
        "optical": "—", "sar": "Yes", "cloud_cover": "0% (stub)",
    }
    if utc_time is None or longitude is None:
        return fallback
    try:
        lon = float(longitude)
        offset_hours = lon / 15.0
        if hasattr(utc_time, "hour"):
            total_minutes = utc_time.hour * 60 + utc_time.minute + int(offset_hours * 60)
        else:
            return fallback
        # Wrap into 0–1439
        total_minutes = total_minutes % 1440
        local_h = total_minutes // 60
        local_m = total_minutes % 60
        local_str = f"{local_h:02d}:{local_m:02d}"
        is_day = 6 <= local_h < 18
        return {
            "local_time": local_str,
            "sun_state": "Day" if is_day else "Night",
            "optical": "Yes" if is_day else "No",
            "sar": "Yes",
            "cloud_cover": "0% (stub)",
        }
    except (TypeError, ValueError):
        return fallback


def _build_situation_summary(record: dict, fa, decision, prefix: list[dict]) -> str:
    """One concise human-readable sentence summarizing the entity's situation.

    Derived entirely from existing record fields and stored objects.
    """
    eid = record.get("target_id", "Entity")
    health = record.get("custody_health", "HEALTHY")
    anomaly = float(record.get("anomaly_score", 0.0))
    hsc = record.get("hours_since_collection")
    failures = int(record.get("consecutive_failures", 0))
    dark = record.get("dark_vessel_flag", False)
    zone = float(record.get("sensitive_zone", 0.0))

    # Build concern fragments
    concerns = []
    if dark:
        concerns.append("AIS dark")
    if zone > 0.5:
        concerns.append("inside sensitive zone")
    elif zone > 0:
        concerns.append("near sensitive zone")
    if anomaly >= 1.5:
        concerns.append("critically elevated anomaly")
    elif anomaly >= 0.8:
        concerns.append("elevated anomaly")
    if health in ("STALE", "LOST"):
        concerns.append(f"custody {health.lower()}")
    elif health == "DEGRADING":
        concerns.append("custody degrading")
    if failures >= 2:
        concerns.append(f"{failures} consecutive collection failures")

    # Time without collection
    if hsc is not None and hsc > 4:
        concerns.append(f"no successful collection in {hsc:.0f}h")

    if not concerns:
        action_str = decision.action.replace("_", " ").lower()
        return f"{eid} is in routine status. System recommends {action_str}."

    concern_str = ", ".join(concerns[:-1]) + (" and " + concerns[-1] if len(concerns) > 1 else concerns[0])
    action_str = decision.action.replace("_", " ").lower()
    return f"{eid} is deteriorating due to {concern_str}. System recommends {action_str}."


def register(app: Dash) -> None:
    """Register entity detail callbacks."""

    # ── Callback A: reasoning chain (prediction + fusion + decision + tasks)

    _reasoning_outputs = [
        Output(DETAIL_HEADING, "children"),
        Output(DETAIL_SUMMARY, "children"),
        Output(PRED_ZONE_PROB, "children"),
        Output(PRED_TIME_TO_ZONE, "children"),
        Output(PRED_FUTURE_ANOMALY, "children"),
        Output(PRED_CONFIDENCE, "children"),
        Output(PRED_HORIZON, "children"),
        Output(ANOMALY_AGREEMENT_BADGE, "children"),
        Output(ANOMALY_AGREEMENT_BADGE, "style"),
        Output(ANOMALY_STATE_BADGE, "children"),
        Output(ANOMALY_STATE_BADGE, "style"),
        Output(ML_ANOMALY_SCORE, "children"),
        Output(ANOMALY_DURATION, "children"),
        Output(FUSION_SCORE, "children"),
        Output(FUSION_UNCERTAINTY, "children"),
        Output(FUSION_AGREEMENT, "children"),
        Output(FUSION_SOURCE, "children"),
        Output(FUSION_MISSING, "children"),
        Output(DECISION_ACTION, "children"),
        Output(DECISION_ACTION, "style"),
        Output(DECISION_PRIORITY, "children"),
        Output(DECISION_CONFIDENCE, "children"),
        Output(DECISION_WHY, "children"),
        Output(DECISION_FALLBACKS, "children"),
        Output(TASK_QUEUE, "children"),
        Output(ROLE_ANALYST_CARD, "children"),
        Output(ROLE_COLLECTOR_CARD, "children"),
        Output(ROLE_OPERATOR_CARD, "children"),
        Output(ROLE_RESOLUTION, "children"),
        Output(ROLE_WHY_NOT, "children"),
        Output(COND_LOCAL_TIME, "children"),
        Output(COND_SUN_STATE, "children"),
        Output(COND_OPTICAL_VIABLE, "children"),
        Output(COND_SAR_VIABLE, "children"),
        Output(COND_CLOUD_COVER, "children"),
    ]
    N_REASONING = len(_reasoning_outputs)

    @app.callback(
        *_reasoning_outputs,
        Input(app_state.SCENARIO_KEY, "data"),
        Input(app_state.TIMESTEP_INDEX, "data"),
        Input(app_state.SELECTED_ENTITY, "data"),
    )
    def update_reasoning(scenario_key, timestep_idx, entity_id):
        _blank_badge = {"fontSize": "0.8rem", "fontWeight": "600",
                        "padding": "2px 8px", "borderRadius": "3px"}
        _blank_action = {"fontSize": "1.0rem", "fontWeight": "700", "padding": "4px 14px",
                         "borderRadius": "5px", "backgroundColor": "#555", "color": "#fff"}
        blank = (
            "Entity Detail",                    # heading
            "Select an entity to see its reasoning chain.",  # summary
            "—", "—", "—", "—", "—",           # 5 prediction metrics
            "—", _blank_badge,                  # agreement badge + style
            "—", _blank_badge,                  # state badge + style
            "—",                                # ML anomaly score
            "—",                                # duration
            "—", "—", "—",                      # fused, uncertainty, agreement
            "—", "—",                           # source, missing
            "—", _blank_action,                 # action text + style
            "—", "—",                           # priority, confidence
            "—",                                # why
            "",                                 # fallbacks
            "No task recommendations.",         # task queue
            "—",                                # role analyst card
            "—",                                # role collector card
            "—",                                # role operator card
            "—",                                # role resolution
            "",                                 # role why-not
            "—", "—", "—", "—", "—",           # collection conditions
        )

        if not scenario_key or timestep_idx is None or entity_id is None:
            return blank

        records = app_state.get_records(scenario_key)
        timeline = entity_timeline_up_to(records, entity_id, timestep_idx)
        if not timeline:
            return blank

        current = timeline[-1]
        prefix = timeline[:-1]

        # Fusion + Decision: prefer stored objects
        fa = current.get("fusion_assessment")
        decision = current.get("mission_decision")

        if fa is None or decision is None:
            track = derive_track_state(current, prefix)
            compounds = evaluate_compounds(current, window=prefix)
            fa = build_fusion_assessment(current, compounds, track)
            decision = build_decision(fa, current, track, compounds)

        # Task recommendations
        track = derive_track_state(current, prefix)
        compounds = evaluate_compounds(current, window=prefix)
        tasks = build_task_recommendations(decision, fa, current, track)

        # Role deliberation
        delib = build_deliberation(current, fa, decision, compounds)

        # Summary
        summary = _build_situation_summary(current, fa, decision, prefix)

        # Prediction values
        pred_zp = _fmt(current.get("zone_probability"), ".0%")
        if pred_zp != "—":
            pred_zp = f"{float(current['zone_probability']):.0%}"
        pred_tte = _fmt(current.get("time_to_zone_hours"), ".1f")
        if pred_tte != "—":
            pred_tte += "h"
        pred_fa = _fmt(current.get("future_anomaly"))
        pred_conf = _fmt(current.get("prediction_confidence"), ".0%")
        if pred_conf != "—":
            pred_conf = f"{float(current['prediction_confidence']):.0%}"
        pred_hz = _fmt(current.get("prediction_horizon_hours"), ".1f")
        if pred_hz != "—":
            pred_hz += "h"

        # Fusion values
        fs = f"{fa.fused_score:.3f}"
        fu = f"{fa.uncertainty:.3f}"
        fag = f"{fa.source_agreement:.3f}"
        _src_name = fa.recommended_confirming_source or "—"
        _src_color = _SOURCE_COLOR.get(_src_name, "#555")
        f_src = html.Span(_src_name, style={
            "backgroundColor": _src_color, "color": "#fff",
            "padding": "2px 10px", "borderRadius": "4px",
            "fontWeight": "700",
        }) if _src_name != "—" else "—"
        f_miss = ", ".join(fa.missing_evidence) if fa.missing_evidence else "None identified"

        # Decision values
        action_label = decision.action.replace("_", " ")
        action_color = _ACTION_COLOR.get(decision.action, "#555")
        action_style = {
            "fontSize": "1.0rem", "fontWeight": "700",
            "padding": "4px 14px", "borderRadius": "5px",
            "backgroundColor": action_color, "color": "#fff",
        }
        priority = f"{decision.priority:.3f}"
        confidence = f"{decision.confidence:.3f}"

        # Why bullets
        if decision.why:
            why_children = html.Ol([
                html.Li(b, style={"marginBottom": "4px"}) for b in decision.why
            ], style={"margin": "6px 0", "paddingLeft": "1.4em", "fontSize": "0.82rem"})
        else:
            why_children = "—"

        # Fallbacks
        if decision.next_best_actions:
            fallback_str = "Next best: " + " → ".join(decision.next_best_actions)
        else:
            fallback_str = ""

        # Task queue cards
        if tasks:
            task_children = []
            now = current.get("time")
            for t in tasks:
                window_str = t.window_start.strftime("%H:%Mz") if t.window_start else "—"
                if now and t.window_start:
                    tts_sec = max(0.0, (t.window_start - now).total_seconds())
                    tts_str = f"{int(tts_sec / 60)} min" if tts_sec > 0 else "now"
                else:
                    tts_str = "—"
                fallbacks = " · ".join(t.fallbacks) if t.fallbacks else "none"
                is_top = t.rank == 1
                border_color = "#4a9" if is_top else "#2d2d2d"
                bg = "#0d1f18" if is_top else "transparent"
                rank_label = [
                    html.Span(f"#{t.rank} ", style={"fontWeight": "700", "marginRight": "8px"}),
                ]
                if is_top:
                    rank_label.append(html.Span(
                        "RECOMMENDED",
                        style={
                            "fontSize": "0.6rem", "fontWeight": "700",
                            "color": "#4a9", "letterSpacing": "0.08em",
                            "marginRight": "8px",
                        },
                    ))
                task_children.append(html.Div(
                    [
                        *rank_label,
                        html.Span(t.sensor, style={"fontWeight": "600", "fontFamily": "monospace"}),
                        html.Span(f"  {window_str}  TTS {tts_str}", style={"color": "#999", "fontSize": "0.78rem"}),
                        html.Span(f"  EV {t.expected_value:.3f}", style={"fontWeight": "600", "marginLeft": "auto"}),
                        html.Div(t.reason, style={"fontSize": "0.8rem", "marginTop": "4px"}),
                        html.Div(f"Fallbacks: {fallbacks}", style={"fontSize": "0.72rem", "color": "#666", "marginTop": "2px"}),
                    ],
                    style={
                        "border": f"1px solid {border_color}", "borderRadius": "6px",
                        "padding": "8px 12px", "marginBottom": "6px",
                        "backgroundColor": bg,
                    },
                ))
            task_queue_el = html.Div(task_children)
        else:
            task_queue_el = "No task recommendations."

        # Anomaly reasoning fields
        _agr = current.get("anomaly_agreement", "normal")
        _agr_colors = {
            "confirmed": {"backgroundColor": "#4d1a1a", "color": "#f88"},
            "emerging": {"backgroundColor": "#3d3520", "color": "#e8c547"},
            "rule_triggered": {"backgroundColor": "#2d2d3d", "color": "#8af"},
            "normal": {"backgroundColor": "#1e1e1e", "color": "#888"},
        }
        _agr_style = {**{"fontSize": "0.8rem", "fontWeight": "600",
                         "padding": "2px 8px", "borderRadius": "3px"},
                      **_agr_colors.get(_agr, _agr_colors["normal"])}
        _state = current.get("anomaly_state", "normal")
        _state_colors = {
            "critical": {"backgroundColor": "#5a1010", "color": "#f66"},
            "sustained": {"backgroundColor": "#4d1a1a", "color": "#f88"},
            "confirmed": {"backgroundColor": "#3d2020", "color": "#fa8"},
            "emerging": {"backgroundColor": "#3d3520", "color": "#e8c547"},
            "recovering": {"backgroundColor": "#1f3d2d", "color": "#6c6"},
            "normal": {"backgroundColor": "#1e1e1e", "color": "#888"},
        }
        _state_style = {**{"fontSize": "0.8rem", "fontWeight": "600",
                           "padding": "2px 8px", "borderRadius": "3px"},
                        **_state_colors.get(_state, _state_colors["normal"])}
        _ml_dur = current.get("ml_anomaly_duration_hours", 0)
        _dur_str = f"{_ml_dur}h" if _ml_dur > 0 else "—"

        # Role card renderer with disagreement highlighting
        def _role_card(rec, agrees_with_final):
            act_color = _ACTION_COLOR.get(rec.recommended_action, "#555")
            border_color = "#2ea043" if agrees_with_final else "#b08000"
            return html.Div([
                html.Span(
                    rec.recommended_action.replace("_", " "),
                    style={
                        "fontSize": "0.78rem", "fontWeight": "700",
                        "padding": "2px 8px", "borderRadius": "3px",
                        "backgroundColor": act_color, "color": "#fff",
                    },
                ),
                html.Div(
                    f"Conf {rec.confidence:.2f}  ·  EV {rec.expected_value:.2f}",
                    style={"fontSize": "0.7rem", "color": "#999", "marginTop": "4px"},
                ),
                html.Div(
                    rec.rationale,
                    style={"fontSize": "0.72rem", "color": "#bbb",
                           "fontStyle": "italic", "marginTop": "4px"},
                ),
            ], style={"borderLeft": f"3px solid {border_color}", "paddingLeft": "8px"})

        _final = delib.final_action
        role_analyst = _role_card(delib.analyst, delib.analyst.recommended_action == _final)
        role_collector = _role_card(delib.collector, delib.collector.recommended_action == _final)
        role_operator = _role_card(delib.operator, delib.operator.recommended_action == _final)

        _agr_icon = {"unanimous": "#2ea043", "majority": "#b08000", "split": "#d73a49"}
        role_resolution = html.Span([
            html.Span(
                f"Watch Officer: {delib.agreement.upper()} ",
                style={"fontWeight": "700",
                       "color": _agr_icon.get(delib.agreement, "#aaa")},
            ),
            html.Span(
                f"— {delib.resolution_reason}",
                style={"color": "#aaa"},
            ),
        ])

        # "Why not?" rendering
        if delib.why_not:
            why_not_children = [
                html.Div(
                    f"Why not {action.replace('_', ' ')}? {reason}",
                    style={"fontSize": "0.72rem", "color": "#999", "marginTop": "2px"},
                )
                for action, reason in delib.why_not.items()
            ]
            role_why_not = html.Div(why_not_children, style={"marginTop": "4px"})
        else:
            role_why_not = ""

        # Collection conditions
        conds = _collection_conditions(current.get("time"), current.get("lon"))

        return (
            entity_id,  # heading
            summary,
            pred_zp, pred_tte, pred_fa, pred_conf, pred_hz,
            _agr.upper(), _agr_style,           # agreement badge
            _state.upper(), _state_style,        # state badge
            _fmt(current.get("ml_anomaly_score", 0.0)),  # ML score
            _dur_str,                            # duration
            fs, fu, fag, f_src, f_miss,
            action_label, action_style,
            priority, confidence,
            why_children, fallback_str,
            task_queue_el,
            role_analyst,
            role_collector,
            role_operator,
            role_resolution,
            role_why_not,
            conds["local_time"],
            conds["sun_state"],
            conds["optical"],
            conds["sar"],
            conds["cloud_cover"],
        )

    # ── Callback B: supporting tables (alerts, compounds, orbital, traces)

    @app.callback(
        Output(ALERTS_TABLE, "data"),
        Output(COMPOUNDS_TABLE, "data"),
        Output(COMPOUND_HISTORY_TABLE, "data"),
        Output(ORBITAL_TABLE, "data"),
        Output(TRACES_TABLE, "data"),
        Output(DETAIL_ACCORDION, "active_item"),
        Input(app_state.SCENARIO_KEY, "data"),
        Input(app_state.TIMESTEP_INDEX, "data"),
        Input(app_state.SELECTED_ENTITY, "data"),
    )
    def update_detail_tables(scenario_key, timestep_idx, entity_id):
        empty = ([], [], [], [], [], [])

        if not scenario_key or timestep_idx is None or entity_id is None:
            return empty

        records = app_state.get_records(scenario_key)
        timeline = entity_timeline_up_to(records, entity_id, timestep_idx)
        if not timeline:
            return empty

        current = timeline[-1]
        prefix = timeline[:-1]

        # Alerts — deduplicate repeated codes into counts
        alerts = alerts_for_timeline(timeline)
        _alert_counts: dict[tuple, int] = {}
        _alert_latest_ts: dict[tuple, object] = {}
        for a in alerts:
            key = (a.level, a.code)
            _alert_counts[key] = _alert_counts.get(key, 0) + 1
            if key not in _alert_latest_ts or a.timestamp > _alert_latest_ts[key]:
                _alert_latest_ts[key] = a.timestamp
        alerts_data = [
            {
                "level": level,
                "code": f"{code} ({count})" if count > 1 else code,
                "message": next(a.message for a in alerts if a.level == level and a.code == code),
                "timestamp": _alert_latest_ts[(level, code)].strftime("%Y-%m-%d %H:%M:%Sz")
                if hasattr(_alert_latest_ts[(level, code)], "strftime")
                else str(_alert_latest_ts[(level, code)]),
            }
            for (level, code), count in _alert_counts.items()
        ]

        # Compounds: active at current step
        compounds = evaluate_compounds(current, window=prefix)
        active_df = build_active_compounds_df(compounds)
        active_data = active_df.to_dict("records") if not active_df.empty else []

        # Compound history: aggregated over full prefix + current
        all_compounds = []
        for i, r in enumerate(timeline):
            w = timeline[:i]
            all_compounds.extend(evaluate_compounds(r, window=w))
        agg = aggregate_compound_history(all_compounds)
        hist_df = build_compound_history_df(agg)
        hist_data = hist_df.to_dict("records") if not hist_df.empty else []

        # Orbital passes
        orbital_rows = build_orbital_passes_rows(
            float(current["lat"]), float(current["lon"]), current["time"],
        )
        for row in orbital_rows:
            for k, v in row.items():
                if isinstance(v, datetime):
                    row[k] = v.strftime("%H:%M:%Sz")

        # Decision traces
        traces = [r["decision_trace"] for r in timeline if r.get("decision_trace")]
        trace_rows = traces_to_rows(traces)
        for row in trace_rows:
            if "Time" in row and isinstance(row["Time"], datetime):
                row["Time"] = str(row["Time"]).split("+")[0]

        # Auto-expand accordion sections that have data
        active_items = []
        if alerts_data:
            active_items.append("acc-alerts")
        if active_data:
            active_items.append("acc-compounds")
        if hist_data:
            active_items.append("acc-compound-history")
        # Orbital passes always have 6 rows; expand only if a pass is imminent
        imminent = any(
            r.get("Status") == "IN VIEW" or (
                isinstance(r.get("Time to Start (min)"), (int, float))
                and r["Time to Start (min)"] <= 30
            )
            for r in orbital_rows
        )
        if imminent:
            active_items.append("acc-orbital")
        if trace_rows:
            active_items.append("acc-traces")

        return (
            alerts_data,
            active_data,
            hist_data,
            orbital_rows,
            trace_rows,
            active_items,
        )
