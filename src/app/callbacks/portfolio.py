"""Portfolio callbacks: KPI strip, table, and event feed updates."""
from __future__ import annotations

import pandas as pd
from dash import Dash, Input, Output, html, no_update

import state as app_state
from adapter import records_at_timestep, timestep_as_dataframe
from layout.overview import (
    KPI_TOTAL,
    KPI_NEEDS_ACTION,
    KPI_NEGLECTED,
    KPI_STALE_LOST,
    KPI_PREEMPTED,
    KPI_APPROACHING,
    PORTFOLIO_TABLE,
    EVENT_FEED,
)
from overview_events import build_event_feed
from portfolio_overview import build_overview_df, compute_kpi_counts

# Status → row highlight — tiered urgency gradient (dark-theme)
# Critical: strong red, High: medium red, Moderate: amber, Neutral: subtle
_STATUS_COLORS = {
    "NEEDS ACTION": "#4d1a1a",   # critical — strong red
    "STALE":        "#3d2020",   # critical — dark red (custody lost/stale)
    "NEGLECTED":    "#3d2e1f",   # high — amber-brown
    "PREEMPTED":    "#3d3520",   # high — amber
    "APPROACHING":  "#2d2340",   # moderate — muted purple
    "WATCH":        "#1f2d3d",   # moderate — dark blue
    "HEALTHY":      "transparent",
}

# Event type → left-border color
_EVENT_COLOR = {
    "zone_entry":     "#b08000",
    "zone_approach":  "#7b3fa0",
    "health_change":  "#d14343",
    "neglect":        "#d7c31e",
    "rank_change":    "#1f6fa8",
    "preemption":     "#d78219",
}


_ACTION_COLORS = {
    "TASK":      {"backgroundColor": "#1a3d1a", "color": "#6fcf6f"},
    "PREEMPTED": {"backgroundColor": "#3d3520", "color": "#d7a832"},
    "NO_SENSOR": {"backgroundColor": "#3d1f1f", "color": "#d77"},
}


def _build_row_styles(data: list[dict]) -> list[dict]:
    """Return DataTable style_data_conditional rules."""
    styles = []
    # Status row background
    for status, color in _STATUS_COLORS.items():
        if color == "transparent":
            continue
        styles.append({
            "if": {"filter_query": f'{{Status}} = "{status}"'},
            "backgroundColor": color,
        })
    # Action column highlighting
    for action, action_style in _ACTION_COLORS.items():
        styles.append({
            "if": {"filter_query": f'{{Action}} = "{action}"', "column_id": "Action"},
            **action_style, "fontWeight": "600",
        })
    return styles


def _build_event_cards(events: list[dict]) -> list:
    """Convert event dicts into styled Dash html.Div cards."""
    if not events:
        return [html.P("No events at this timestep.", style={"color": "#666", "fontSize": "0.8rem"})]
    cards = []
    for evt in events:
        etype = evt.get("event_type", "")
        border_color = _EVENT_COLOR.get(etype, "#555")
        cards.append(html.Div(
            [
                html.Span(
                    evt.get("entity_id", ""),
                    style={"fontWeight": "700", "fontFamily": "monospace", "marginRight": "8px"},
                ),
                html.Span(
                    evt.get("description", ""),
                    style={"fontSize": "0.8rem"},
                ),
            ],
            style={
                "borderLeft": f"3px solid {border_color}",
                "padding": "4px 10px",
                "marginBottom": "4px",
                "fontSize": "0.82rem",
            },
        ))
    return cards


import re

# Label mappings for reason segments
_REASON_LABELS = {
    "projected zone entry": "Zone",
    "elevated anomaly": "Anomaly",
    "neglected": "Neglect",
    "custody lost": "Custody",
    "custody stale": "Custody",
    "custody degrading": "Custody",
    "routine monitoring": "Status",
}


def _format_reason(raw: str) -> str:
    """Reformat a reason string into compact labeled segments.

    Input:  "Ranked 3/30: projected zone entry in 0.0h; elevated anomaly (2.50); neglected 7.0h without collection"
    Output: "Zone: 0.0h | Anomaly: 2.50 | Neglect: 7.0h"
    """
    # Strip the "Ranked N/M: " prefix
    text = re.sub(r"^Ranked \d+/\d+:\s*", "", raw)
    if not text:
        return raw

    parts = [s.strip() for s in text.split(";")]
    segments = []
    for part in parts:
        if "zone entry" in part:
            m = re.search(r"([\d.]+)h", part)
            segments.append(f"Zone: {m.group(1)}h" if m else "Zone: yes")
        elif "anomaly" in part:
            m = re.search(r"\(([\d.]+)\)", part)
            segments.append(f"Anomaly: {m.group(1)}" if m else "Anomaly: high")
        elif "neglected" in part:
            m = re.search(r"([\d.]+)h", part)
            segments.append(f"Neglect: {m.group(1)}h" if m else "Neglect: yes")
        elif "custody" in part:
            segments.append(f"Custody: {part.strip()}")
        elif "routine" in part:
            segments.append("Routine")
        else:
            segments.append(part)
    return " | ".join(segments) if segments else raw


def register(app: Dash) -> None:
    """Register portfolio + event feed callbacks."""

    @app.callback(
        Output(KPI_TOTAL, "children"),
        Output(KPI_NEEDS_ACTION, "children"),
        Output(KPI_NEGLECTED, "children"),
        Output(KPI_STALE_LOST, "children"),
        Output(KPI_PREEMPTED, "children"),
        Output(KPI_APPROACHING, "children"),
        Output(PORTFOLIO_TABLE, "data"),
        Output(PORTFOLIO_TABLE, "style_data_conditional"),
        Output(EVENT_FEED, "children"),
        Input(app_state.SCENARIO_KEY, "data"),
        Input(app_state.TIMESTEP_INDEX, "data"),
    )
    def update_portfolio(scenario_key, timestep_idx):
        empty = ("—", "—", "—", "—", "—", "—", [], [], [])
        if not scenario_key or timestep_idx is None:
            return empty

        records = app_state.get_records(scenario_key)
        ts_df = timestep_as_dataframe(records, timestep_idx)
        if ts_df.empty:
            return empty

        overview_df = build_overview_df(ts_df)
        kpis = compute_kpi_counts(ts_df)
        table_data = overview_df.to_dict("records")
        # Format reason column for readability
        for row in table_data:
            if "Reason" in row and row["Reason"]:
                row["Reason"] = _format_reason(str(row["Reason"]))
        styles = _build_row_styles(table_data)

        # Event feed: compare current vs previous timestep
        if timestep_idx > 0:
            prev_df = timestep_as_dataframe(records, timestep_idx - 1)
        else:
            prev_df = pd.DataFrame()
        events = build_event_feed(ts_df, prev_df) if not prev_df.empty else []
        event_cards = _build_event_cards(events)

        return (
            str(kpis["total"]),
            str(kpis["needs_action"]),
            str(kpis["neglected"]),
            str(kpis["stale_or_lost"]),
            str(kpis["preempted"]),
            str(kpis["approaching"]),
            table_data,
            styles,
            event_cards,
        )
