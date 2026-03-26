"""Overview layout: KPI strip and portfolio table."""
from __future__ import annotations

from dash import dash_table, html
import dash_bootstrap_components as dbc

# Component IDs — importable by callbacks
KPI_TOTAL = "kpi-total"
KPI_NEEDS_ACTION = "kpi-needs-action"
KPI_NEGLECTED = "kpi-neglected"
KPI_STALE_LOST = "kpi-stale-lost"
KPI_PREEMPTED = "kpi-preempted"
KPI_APPROACHING = "kpi-approaching"

PORTFOLIO_TABLE = "portfolio-table"


def _kpi_card(card_id: str, label: str) -> dbc.Col:
    """One KPI metric card with a placeholder value."""
    return dbc.Col(
        dbc.Card(
            dbc.CardBody(
                [
                    html.P(label, style={"fontSize": "0.7rem", "color": "#999", "margin": 0}),
                    html.H4("—", id=card_id, style={"margin": 0}),
                ],
                style={"padding": "8px 12px"},
            ),
        ),
        width=2,
    )


def build_kpi_strip() -> dbc.Row:
    """Return a row of 6 KPI cards with placeholder values.

    Callback fills in the actual numbers via the card IDs.
    """
    return dbc.Row(
        [
            _kpi_card(KPI_TOTAL, "Total"),
            _kpi_card(KPI_NEEDS_ACTION, "Needs Action"),
            _kpi_card(KPI_NEGLECTED, "Neglected"),
            _kpi_card(KPI_STALE_LOST, "Stale / Lost"),
            _kpi_card(KPI_PREEMPTED, "Preempted"),
            _kpi_card(KPI_APPROACHING, "Approaching"),
        ],
        className="g-2 mb-3",
    )


def build_portfolio_table() -> dash_table.DataTable:
    """Return an empty DataTable configured for the portfolio overview.

    Columns match the output of ``portfolio_overview.build_overview_df()``.
    Data is filled by the portfolio callback.
    """
    columns = [
        {"name": "Rank", "id": "Rank"},
        {"name": "Entity", "id": "Entity"},
        {"name": "Status", "id": "Status"},
        {"name": "Attention", "id": "Attention"},
        {"name": "Anomaly", "id": "Anomaly"},
        {"name": "Health", "id": "Health"},
        {"name": "Neglect h", "id": "Neglect h"},
        {"name": "Action", "id": "Action"},
        {"name": "Deferred For", "id": "Deferred For"},
        {"name": "Reason", "id": "Reason"},
    ]
    return dash_table.DataTable(
        id=PORTFOLIO_TABLE,
        columns=columns,
        data=[],
        row_selectable="single",
        selected_rows=[],
        sort_action="native",
        page_size=30,
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": "#1e1e1e",
            "color": "#ccc",
            "fontWeight": "600",
            "fontSize": "0.75rem",
        },
        style_cell={
            "backgroundColor": "#121212",
            "color": "#ddd",
            "fontSize": "0.8rem",
            "padding": "6px 10px",
            "border": "1px solid #2d2d2d",
        },
        style_data_conditional=[],
    )


# Event feed container ID
EVENT_FEED = "event-feed"


def build_event_feed_container() -> html.Div:
    """Empty container populated by the event feed callback."""
    return html.Div(id=EVENT_FEED, style={"marginBottom": "12px"})


def build_overview_layout() -> html.Div:
    """Assemble the full overview panel: KPI strip + table + map + events."""
    from layout.map_panel import build_map_panel

    return html.Div(
        [
            build_kpi_strip(),
            build_portfolio_table(),
            build_map_panel(),
            build_event_feed_container(),
        ],
        style={"padding": "12px"},
    )
