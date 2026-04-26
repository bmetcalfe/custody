"""Whitsun decision-trace replay panel.

A self-contained Dash layout that renders the canonical Whitsun
custody / reacquisition trace at
``data/demo/whitsun_decision_trace.fixture.json``.  Read-only,
fixture-backed, and independent of the rest of the dashboard.

This module builds the layout tree only.  Callbacks live in
``callbacks/whitsun_replay.py``.

Visual discipline:

- Honest labels for source class (Umbra / Sentinel-1 / Sentinel-2 /
  simulated) and ``data_mode`` (fixture / simulated).
- Confidence weights surfaced explicitly per observation.
- Policy rationale is labelled "Heuristic advisory / RL-ready slot",
  never "trained RL decision".
- If an event does not yet have a given artifact, the panel renders a
  small "not available at this step" state instead of breaking.
"""
from __future__ import annotations

from dash import dcc, html
import dash_bootstrap_components as dbc

from custody.demo import load_whitsun_decision_trace


# ---------------------------------------------------------------------------
# Component IDs (importable by callbacks)
# ---------------------------------------------------------------------------


WHITSUN_REPLAY_ROOT = "whitsun-replay-root"
WHITSUN_SELECTED_EVENT_STORE = "whitsun-replay-selected-event"
WHITSUN_TIMELINE_RADIO = "whitsun-replay-timeline-radio"
WHITSUN_TIMELINE_STEP_COUNTER = "whitsun-replay-step-counter"

WHITSUN_HEADER = "whitsun-replay-header"
WHITSUN_EVENT_SUMMARY = "whitsun-replay-event-summary"
WHITSUN_OBSERVATIONS_PANEL = "whitsun-replay-observations"
WHITSUN_OPTIONS_TABLE = "whitsun-replay-options-table"
WHITSUN_SCORE_BREAKDOWN = "whitsun-replay-score-breakdown"
WHITSUN_POLICY_RATIONALE = "whitsun-replay-policy-rationale"
WHITSUN_HUMAN_ACTION = "whitsun-replay-human-action"
WHITSUN_OUTCOME = "whitsun-replay-outcome"
WHITSUN_COUNTERFACTUALS = "whitsun-replay-counterfactuals"
WHITSUN_FOLLOWUP = "whitsun-replay-followup"
WHITSUN_CONTEXT_MAP = "whitsun-replay-context-map"

# Sidebar swap targets — driven by the main Tabs value.
WHITSUN_SIDEBAR_OVERVIEW = "custody-main-sidebar-overview"
WHITSUN_SIDEBAR_REPLAY = "custody-main-sidebar-whitsun"
CUSTODY_MAIN_TABS = "custody-main-tabs"


# ---------------------------------------------------------------------------
# Static palette
# ---------------------------------------------------------------------------


_PANEL_BG = "#1a1a1a"
_PANEL_BORDER = "#2d2d2d"
_MUTED = "#9ca3af"
_TEXT = "#e5e7eb"
_ACCENT = "#5eead4"
_WARN = "#fbbf24"


# ---------------------------------------------------------------------------
# Source / data-mode badges
# ---------------------------------------------------------------------------


_SOURCE_BADGE_COLOR = {
    "umbra-sar": "primary",
    "sentinel-1": "info",
    "sentinel-2": "secondary",
    "simulated": "warning",
}


def source_badge(source: str | None) -> dbc.Badge:
    color = _SOURCE_BADGE_COLOR.get(str(source), "light")
    label = (source or "(unknown)").upper()
    return dbc.Badge(label, color=color, className="me-1")


def data_mode_badge(mode: str | None) -> dbc.Badge:
    label = (mode or "(unknown)").upper()
    color = "warning" if mode == "simulated" else "secondary"
    return dbc.Badge(label, color=color, className="me-1")


def _muted(text: str) -> html.Span:
    return html.Span(text, style={"color": _MUTED, "fontSize": "0.75rem"})


def _panel(title: str, body, panel_id: str | None = None) -> dbc.Card:
    body_kwargs = {"style": {"padding": "10px 12px"}}
    if panel_id is not None:
        body_kwargs["id"] = panel_id
    return dbc.Card(
        [
            dbc.CardHeader(
                title,
                style={
                    "backgroundColor": _PANEL_BORDER,
                    "color": _TEXT,
                    "fontSize": "0.85rem",
                    "fontWeight": "600",
                    "padding": "6px 12px",
                },
            ),
            dbc.CardBody(body, **body_kwargs),
        ],
        style={
            "backgroundColor": _PANEL_BG,
            "border": f"1px solid {_PANEL_BORDER}",
            "marginBottom": "10px",
        },
    )


# ---------------------------------------------------------------------------
# Public layout factory
# ---------------------------------------------------------------------------


def build_whitsun_sidebar_block() -> html.Div:
    """Replacement sidebar shown when the Whitsun replay tab is active.

    Compact read-only block.  No controls; the existing scenario /
    timeline / entity controls do not affect the Whitsun replay and
    therefore are hidden by a tab-switch callback.
    """
    return html.Div(
        [
            html.H5(
                "Whitsun replay",
                style={"color": _TEXT, "marginBottom": "4px"},
            ),
            html.Div(
                "read-only fixture",
                style={"color": _MUTED, "fontSize": "0.78rem"},
            ),
            html.Hr(style={"borderColor": _PANEL_BORDER, "margin": "10px 0"}),
            html.Ul(
                [
                    html.Li("progressive mission replay"),
                    html.Li("no live tasking"),
                    html.Li("no live inference"),
                ],
                style={
                    "color": _MUTED,
                    "fontSize": "0.75rem",
                    "paddingLeft": "18px",
                    "lineHeight": "1.4",
                },
            ),
            html.Div(
                "Use the timeline on the right to step through the "
                "14-event scenario.",
                style={
                    "color": _MUTED,
                    "fontSize": "0.72rem",
                    "marginTop": "10px",
                    "fontStyle": "italic",
                },
            ),
        ],
        style={"padding": "16px"},
    )


def build_whitsun_replay_layout() -> html.Div:
    """Return the complete Whitsun replay panel tree."""
    trace = load_whitsun_decision_trace()
    meta = trace.metadata
    pairs = trace.event_label_pairs()
    initial_event_id = trace.first_event_id() or ""

    timeline = dcc.RadioItems(
        id=WHITSUN_TIMELINE_RADIO,
        options=[
            {
                "label": html.Span(
                    f"{idx + 1:02d}. {label}",
                    style={"fontSize": "0.78rem"},
                ),
                "value": event_id,
            }
            for idx, (event_id, label) in enumerate(pairs)
        ],
        value=initial_event_id,
        labelClassName="whitsun-timeline-row",
    )

    timeline_subhead = html.Div(
        "Click an event to step through. Panels reveal "
        "progressively as the scenario unfolds.",
        style={
            "color": _MUTED,
            "fontSize": "0.72rem",
            "fontStyle": "italic",
            "marginBottom": "8px",
        },
    )

    timeline_header = html.Div(
        [
            html.Span("Event timeline (14 steps)"),
            html.Span(
                "Step — / 14",
                id=WHITSUN_TIMELINE_STEP_COUNTER,
                style={
                    "color": _ACCENT,
                    "fontSize": "0.78rem",
                    "fontWeight": "600",
                    "marginLeft": "auto",
                },
            ),
        ],
        style={"display": "flex", "alignItems": "center"},
    )

    context_map_body = html.Div(
        [
            html.Div(
                "Map visualization is planned for the next slice. "
                "Below: AOI, revealed tracks, and latest custody "
                "snapshot.",
                style={
                    "color": _MUTED,
                    "fontSize": "0.72rem",
                    "fontStyle": "italic",
                    "marginBottom": "6px",
                },
            ),
            html.Div(
                id=WHITSUN_CONTEXT_MAP,
                style={"color": _MUTED, "fontSize": "0.75rem"},
            ),
        ],
    )

    header = html.Div(
        [
            html.H4(
                meta.get("title") or "Whitsun decision replay",
                style={"color": _TEXT, "marginBottom": "4px"},
            ),
            html.Div(
                [
                    data_mode_badge(meta.get("data_mode")),
                    source_badge(meta.get("primary_evidence_source")),
                    _muted(
                        " · scenario "
                        f"{meta.get('scenario_id') or '(unknown)'}"
                    ),
                ],
            ),
            html.Div(id=WHITSUN_HEADER, style={"marginTop": "6px"}),
        ],
        style={"marginBottom": "10px"},
    )

    return html.Div(
        [
            dcc.Store(id=WHITSUN_SELECTED_EVENT_STORE, data=initial_event_id),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            header,
                            _panel(
                                timeline_header,
                                html.Div([timeline_subhead, timeline]),
                            ),
                            _panel(
                                "Context (map placeholder)",
                                context_map_body,
                            ),
                        ],
                        width=4,
                        style={"paddingRight": "10px"},
                    ),
                    dbc.Col(
                        [
                            _panel(
                                "Decision inspector",
                                html.Div(id=WHITSUN_EVENT_SUMMARY),
                            ),
                            _panel(
                                "Observations and evidence",
                                html.Div(id=WHITSUN_OBSERVATIONS_PANEL),
                            ),
                            _panel(
                                "Candidate tasking options",
                                html.Div(id=WHITSUN_OPTIONS_TABLE),
                            ),
                            _panel(
                                "Score breakdown",
                                html.Div(id=WHITSUN_SCORE_BREAKDOWN),
                            ),
                            _panel(
                                "Policy rationale",
                                html.Div(id=WHITSUN_POLICY_RATIONALE),
                            ),
                            _panel(
                                "Human action",
                                html.Div(id=WHITSUN_HUMAN_ACTION),
                            ),
                            _panel(
                                "Outcome",
                                html.Div(id=WHITSUN_OUTCOME),
                            ),
                            _panel(
                                "Counterfactuals (rejected options)",
                                html.Div(id=WHITSUN_COUNTERFACTUALS),
                            ),
                            _panel(
                                "Follow-up recommendation",
                                html.Div(id=WHITSUN_FOLLOWUP),
                            ),
                        ],
                        width=8,
                    ),
                ],
                className="g-0",
            ),
        ],
        id=WHITSUN_REPLAY_ROOT,
        style={
            "padding": "12px",
            "backgroundColor": "#121212",
            "color": _TEXT,
        },
    )
