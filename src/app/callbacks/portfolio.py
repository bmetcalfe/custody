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

# Status → row highlight color (muted, dark-theme friendly)
_STATUS_COLORS = {
    "NEEDS ACTION": "#3d1f1f",
    "PREEMPTED":    "#3d3520",
    "NEGLECTED":    "#3d2e1f",
    "STALE":        "#2d2d3d",
    "APPROACHING":  "#1f2d3d",
    "WATCH":        "#1f3d2d",
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


def _build_row_styles(data: list[dict]) -> list[dict]:
    """Return DataTable style_data_conditional rules for status colors."""
    styles = []
    for status, color in _STATUS_COLORS.items():
        if color == "transparent":
            continue
        styles.append({
            "if": {
                "filter_query": f'{{Status}} = "{status}"',
            },
            "backgroundColor": color,
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
