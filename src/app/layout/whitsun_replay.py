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
                    style={
                        "color": _TEXT,
                        "fontSize": "0.78rem",
                        "marginLeft": "4px",
                    },
                ),
                "value": event_id,
            }
            for idx, (event_id, label) in enumerate(pairs)
        ],
        value=initial_event_id,
        labelStyle={
            "display": "block",
            "padding": "2px 0",
            "color": _TEXT,
        },
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
                                "Event timeline (14 steps)",
                                timeline,
                            ),
                            _panel(
                                "Context map",
                                html.Div(
                                    id=WHITSUN_CONTEXT_MAP,
                                    style={
                                        "color": _MUTED,
                                        "fontSize": "0.75rem",
                                    },
                                ),
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
                                "Policy rationale (heuristic advisory; "
                                "RL-ready slot, not a trained RL decision)",
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
