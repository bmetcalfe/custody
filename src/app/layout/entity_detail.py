"""Entity detail layout: reasoning chain + supporting tables.

All components are created with placeholder/empty content and explicit IDs.
Callbacks populate them when an entity is selected.
"""
from __future__ import annotations

from dash import dash_table, html
import dash_bootstrap_components as dbc

# ---------------------------------------------------------------------------
# Component IDs — reasoning section
# ---------------------------------------------------------------------------
DETAIL_HEADING = "detail-heading"
DETAIL_SUMMARY = "detail-summary"

PRED_ZONE_PROB = "pred-zone-prob"
PRED_TIME_TO_ZONE = "pred-time-to-zone"
PRED_FUTURE_ANOMALY = "pred-future-anomaly"
PRED_CONFIDENCE = "pred-confidence"
PRED_HORIZON = "pred-horizon"

FUSION_SCORE = "fusion-score"
FUSION_UNCERTAINTY = "fusion-uncertainty"
FUSION_AGREEMENT = "fusion-agreement"
FUSION_SOURCE = "fusion-source"
FUSION_MISSING = "fusion-missing"

DECISION_ACTION = "decision-action"
DECISION_PRIORITY = "decision-priority"
DECISION_CONFIDENCE = "decision-confidence"
DECISION_WHY = "decision-why"
DECISION_FALLBACKS = "decision-fallbacks"

TASK_QUEUE = "task-queue"

# ---------------------------------------------------------------------------
# Component IDs — tables section
# ---------------------------------------------------------------------------
ALERTS_TABLE = "detail-alerts-table"
COMPOUNDS_TABLE = "detail-compounds-table"
COMPOUND_HISTORY_TABLE = "detail-compound-history-table"
ORBITAL_TABLE = "detail-orbital-table"
TRACES_TABLE = "detail-traces-table"
DETAIL_ACCORDION = "detail-accordion"

# ---------------------------------------------------------------------------
# Wrapper ID for show/hide when no entity selected
# ---------------------------------------------------------------------------
DETAIL_CONTAINER = "detail-container"


def _metric(label: str, value_id: str) -> dbc.Col:
    return dbc.Col(
        [
            html.P(label, style={"fontSize": "0.65rem", "color": "#999", "margin": 0}),
            html.Span("—", id=value_id, style={"fontSize": "0.95rem", "fontWeight": "600"}),
        ],
        width="auto",
        style={"minWidth": "80px"},
    )


def _section_label(text: str) -> html.Div:
    return html.Div(
        text,
        style={
            "fontSize": "0.62rem", "fontWeight": "600", "letterSpacing": "0.09em",
            "textTransform": "uppercase", "color": "#888",
            "borderBottom": "1px solid #2d2d2d", "paddingBottom": "3px",
            "marginTop": "14px", "marginBottom": "6px",
        },
    )


def _simple_table(table_id: str, columns: list[dict]) -> dash_table.DataTable:
    return dash_table.DataTable(
        id=table_id,
        columns=columns,
        data=[],
        sort_action="native",
        page_size=20,
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": "#1e1e1e", "color": "#ccc",
            "fontWeight": "600", "fontSize": "0.72rem",
        },
        style_cell={
            "backgroundColor": "#121212", "color": "#ddd",
            "fontSize": "0.78rem", "padding": "5px 8px",
            "border": "1px solid #2d2d2d",
        },
    )


def build_entity_detail_layout() -> html.Div:
    """Return the full entity detail panel with empty placeholders."""
    return html.Div(
        id=DETAIL_CONTAINER,
        children=[
            # Entity heading
            html.H5("Entity Detail", id=DETAIL_HEADING, style={
                "fontFamily": "monospace", "margin": "10px 0 4px 0",
            }),
            # Summary sentence
            html.P("Select an entity to see its reasoning chain.",
                    id=DETAIL_SUMMARY,
                    style={"fontSize": "0.85rem", "color": "#ccc", "margin": "0 0 2px 0"}),

            # ── Prediction ───────────────────────────────────────────────
            _section_label("Prediction"),
            dbc.Row(
                [
                    _metric("Zone Prob", PRED_ZONE_PROB),
                    _metric("Time to Zone", PRED_TIME_TO_ZONE),
                    _metric("Future Anomaly", PRED_FUTURE_ANOMALY),
                    _metric("Confidence", PRED_CONFIDENCE),
                    _metric("Horizon", PRED_HORIZON),
                ],
                className="g-3 mb-2",
            ),

            # ── Fusion Assessment ────────────────────────────────────────
            _section_label("Fusion Assessment"),
            dbc.Row(
                [
                    _metric("Fused Score", FUSION_SCORE),
                    _metric("Uncertainty", FUSION_UNCERTAINTY),
                    _metric("Source Agreement", FUSION_AGREEMENT),
                ],
                className="g-3 mb-1",
            ),
            dbc.Row(
                [
                    dbc.Col([
                        html.Span("Confirming Source: ", style={"fontSize": "0.72rem", "color": "#999"}),
                        html.Span("—", id=FUSION_SOURCE, style={"fontSize": "0.85rem", "fontWeight": "700"}),
                    ], width="auto"),
                    dbc.Col([
                        html.Span("Missing Evidence: ", style={"fontSize": "0.72rem", "color": "#999"}),
                        html.Span("—", id=FUSION_MISSING, style={"fontSize": "0.82rem"}),
                    ]),
                ],
                className="g-2 mb-2",
            ),

            # ── Decision ─────────────────────────────────────────────────
            _section_label("Decision"),
            dbc.Row(
                [
                    dbc.Col([
                        html.Span("—", id=DECISION_ACTION, style={
                            "fontSize": "1.05rem", "fontWeight": "700",
                            "padding": "5px 16px", "borderRadius": "5px",
                            "backgroundColor": "#555", "color": "#fff",
                            "letterSpacing": "0.06em",
                        }),
                    ], width="auto"),
                    dbc.Col([
                        html.P("Priority", style={"fontSize": "0.62rem", "color": "#999", "margin": 0}),
                        html.Span("—", id=DECISION_PRIORITY, style={
                            "fontSize": "1.2rem", "fontWeight": "700", "color": "#fff",
                        }),
                    ], width="auto", style={"minWidth": "90px"}),
                    dbc.Col([
                        html.P("Confidence", style={"fontSize": "0.62rem", "color": "#999", "margin": 0}),
                        html.Span("—", id=DECISION_CONFIDENCE, style={
                            "fontSize": "0.85rem", "fontWeight": "400", "color": "#aaa",
                        }),
                    ], width="auto", style={"minWidth": "80px"}),
                ],
                className="g-3 mb-1 align-items-center",
            ),
            html.Div("—", id=DECISION_WHY, style={
                "fontSize": "0.82rem", "lineHeight": "1.55", "margin": "6px 0",
            }),
            html.Div(id=DECISION_FALLBACKS, style={
                "fontSize": "0.74rem", "color": "#999", "marginBottom": "8px",
            }),

            # ── Task Queue ───────────────────────────────────────────────
            _section_label("Task Queue"),
            html.Div("No task recommendations.", id=TASK_QUEUE, style={
                "fontSize": "0.82rem", "color": "#666",
            }),

            # ── Supporting tables (collapsible accordion) ─────────────────
            # Accordion items auto-expand when they have data (controlled
            # by the tables callback setting DETAIL_ACCORDION.active_item).
            _section_label("Supporting Data"),
            dbc.Accordion(
                id=DETAIL_ACCORDION,
                children=[
                    dbc.AccordionItem(
                        dash_table.DataTable(
                            id=ALERTS_TABLE,
                            columns=[
                                {"name": "Level", "id": "level"},
                                {"name": "Code", "id": "code"},
                                {"name": "Message", "id": "message"},
                            ],
                            data=[],
                            sort_action="native",
                            page_size=20,
                            style_table={"overflowX": "auto"},
                            style_header={
                                "backgroundColor": "#1e1e1e", "color": "#ccc",
                                "fontWeight": "600", "fontSize": "0.72rem",
                            },
                            style_cell={
                                "backgroundColor": "#121212", "color": "#ddd",
                                "fontSize": "0.78rem", "padding": "5px 8px",
                                "border": "1px solid #2d2d2d",
                            },
                            style_data_conditional=[
                                {"if": {"filter_query": '{level} = "CRITICAL"', "column_id": "level"},
                                 "color": "#e55", "fontWeight": "700"},
                                {"if": {"filter_query": '{level} = "WARNING"', "column_id": "level"},
                                 "color": "#d7a832", "fontWeight": "600"},
                                {"if": {"filter_query": '{level} = "INFO"', "column_id": "level"},
                                 "color": "#888"},
                            ],
                        ),
                        title="Alerts",
                        item_id="acc-alerts",
                    ),
                    dbc.AccordionItem(
                        _simple_table(COMPOUNDS_TABLE, [
                            {"name": "Code", "id": "Code"},
                            {"name": "Conf", "id": "Conf"},
                            {"name": "Evidence", "id": "Evidence"},
                        ]),
                        title="Active Compounds",
                        item_id="acc-compounds",
                    ),
                    dbc.AccordionItem(
                        dash_table.DataTable(
                            id=COMPOUND_HISTORY_TABLE,
                            columns=[
                                {"name": "Code", "id": "Code"},
                                {"name": "Count", "id": "Count"},
                                {"name": "Peak Conf", "id": "Peak Conf"},
                                {"name": "First Seen", "id": "First Seen"},
                                {"name": "Last Seen", "id": "Last Seen"},
                            ],
                            data=[],
                            sort_action="native",
                            page_size=20,
                            style_table={"overflowX": "auto"},
                            style_header={
                                "backgroundColor": "#1e1e1e", "color": "#ccc",
                                "fontWeight": "600", "fontSize": "0.72rem",
                            },
                            style_cell={
                                "backgroundColor": "#121212", "color": "#ddd",
                                "fontSize": "0.78rem", "padding": "5px 8px",
                                "border": "1px solid #2d2d2d",
                            },
                            style_data_conditional=[
                                # Emphasize high-count compounds
                                {"if": {"column_id": "Count"},
                                 "fontWeight": "700", "color": "#e8c547"},
                                {"if": {"column_id": "Peak Conf"},
                                 "fontWeight": "600"},
                            ],
                        ),
                        title="Compound History",
                        item_id="acc-compound-history",
                    ),
                    dbc.AccordionItem(
                        _simple_table(ORBITAL_TABLE, [
                            {"name": "Satellite", "id": "Satellite"},
                            {"name": "Start", "id": "Start"},
                            {"name": "Duration (min)", "id": "Duration (min)"},
                            {"name": "Time to Start (min)", "id": "Time to Start (min)"},
                            {"name": "Status", "id": "Status"},
                        ]),
                        title="Orbital Passes",
                        item_id="acc-orbital",
                    ),
                    dbc.AccordionItem(
                        _simple_table(TRACES_TABLE, [
                            {"name": "Time", "id": "Time"},
                            {"name": "Action", "id": "Action"},
                            {"name": "Priority", "id": "Priority"},
                            {"name": "Task Value", "id": "Task Value"},
                            {"name": "Hold Eligible", "id": "Hold Eligible"},
                            {"name": "Failure Boost", "id": "Failure Boost"},
                            {"name": "Chosen Sensor", "id": "Chosen Sensor"},
                        ]),
                        title="Decision Traces",
                        item_id="acc-traces",
                    ),
                ],
                active_item=[],
                always_open=True,
                flush=True,
            ),
        ],
        style={"padding": "0 12px"},
    )
