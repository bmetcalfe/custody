"""Portfolio callbacks: KPI strip and table updates on timestep change."""
from __future__ import annotations

from dash import Dash, Input, Output, no_update

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
)
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


def register(app: Dash) -> None:
    """Register portfolio update callback on *app*."""

    @app.callback(
        Output(KPI_TOTAL, "children"),
        Output(KPI_NEEDS_ACTION, "children"),
        Output(KPI_NEGLECTED, "children"),
        Output(KPI_STALE_LOST, "children"),
        Output(KPI_PREEMPTED, "children"),
        Output(KPI_APPROACHING, "children"),
        Output(PORTFOLIO_TABLE, "data"),
        Output(PORTFOLIO_TABLE, "style_data_conditional"),
        Input(app_state.SCENARIO_KEY, "data"),
        Input(app_state.TIMESTEP_INDEX, "data"),
    )
    def update_portfolio(scenario_key, timestep_idx):
        if not scenario_key or timestep_idx is None:
            return "—", "—", "—", "—", "—", "—", [], []

        records = app_state.get_records(scenario_key)
        ts_df = timestep_as_dataframe(records, timestep_idx)
        if ts_df.empty:
            return "—", "—", "—", "—", "—", "—", [], []

        overview_df = build_overview_df(ts_df)
        kpis = compute_kpi_counts(ts_df)

        table_data = overview_df.to_dict("records")
        styles = _build_row_styles(table_data)

        return (
            str(kpis["total"]),
            str(kpis["needs_action"]),
            str(kpis["neglected"]),
            str(kpis["stale_or_lost"]),
            str(kpis["preempted"]),
            str(kpis["approaching"]),
            table_data,
            styles,
        )
