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

import dash_deck
import pydeck as pdk
from dash import dcc, html
import dash_bootstrap_components as dbc

from custody.demo import load_whitsun_decision_trace, load_map_overlays
from layout.map_overlays_helpers import (
    LAYER_TOGGLE_OPTIONS,
    build_deck_json,
    default_visibility,
)


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

# Map / overlay panel IDs.
WHITSUN_MAP_DECK = "whitsun-replay-map-deck"
WHITSUN_MAP_LAYER_TOGGLES = "whitsun-replay-map-layers"
WHITSUN_MAP_OPACITY = "whitsun-replay-map-opacity"
WHITSUN_MAP_OVERLAY_BADGES = "whitsun-replay-map-overlay-badges"
WHITSUN_MAP_MISSING_IMAGERY = "whitsun-replay-map-missing-imagery"

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

    aoi_center_lat = float(meta.get("aoi_center_lat_deg") or 9.98)
    aoi_center_lon = float(meta.get("aoi_center_lon_deg") or 114.63)

    initial_overlays = load_map_overlays()
    initial_deck = build_deck_json(
        overlays=tuple(
            o for o in initial_overlays
            if o.scenario_id == "whitsun"
            and o.visible_from_event_ordinal <= 1
        ),
        layer_visibility=default_visibility(),
        opacity=0.6,
        center_lat=aoi_center_lat,
        center_lon=aoi_center_lon,
    )

    map_body = html.Div(
        [
            html.Div(
                "Footprints + AOI on a real map.  No georeferenced "
                "imagery is committed yet, so observation footprints "
                "currently render as outlines and the panel says so "
                "explicitly per source.",
                style={
                    "color": _MUTED,
                    "fontSize": "0.72rem",
                    "fontStyle": "italic",
                    "marginBottom": "8px",
                },
            ),
            dash_deck.DeckGL(
                id=WHITSUN_MAP_DECK,
                data=initial_deck,
                mapboxKey="",
                tooltip={"text": "{tooltip}"},
                style={
                    "width": "100%", "height": "320px",
                    "position": "relative",
                    "borderRadius": "4px", "overflow": "hidden",
                },
            ),
            html.Div(
                [
                    html.Span(
                        "layers:",
                        style={"color": _MUTED, "fontSize": "0.72rem"},
                    ),
                    dcc.Checklist(
                        id=WHITSUN_MAP_LAYER_TOGGLES,
                        options=[
                            {"label": label, "value": key}
                            for key, label in LAYER_TOGGLE_OPTIONS
                        ],
                        value=[k for k, _ in LAYER_TOGGLE_OPTIONS],
                        inline=True,
                        inputStyle={"marginRight": "4px"},
                        labelStyle={
                            "color": _TEXT,
                            "fontSize": "0.72rem",
                            "marginRight": "8px",
                        },
                    ),
                ],
                style={
                    "marginTop": "8px",
                    "display": "flex",
                    "flexWrap": "wrap",
                    "alignItems": "center",
                    "gap": "8px",
                },
            ),
            html.Div(
                [
                    html.Span(
                        "image opacity:",
                        style={"color": _MUTED, "fontSize": "0.72rem"},
                    ),
                    dcc.Slider(
                        id=WHITSUN_MAP_OPACITY,
                        min=0.0, max=1.0, step=0.05, value=0.6,
                        marks=None,
                        tooltip={"placement": "bottom",
                                 "always_visible": False},
                    ),
                ],
                style={"marginTop": "4px"},
            ),
            html.Div(
                id=WHITSUN_MAP_OVERLAY_BADGES,
                style={"marginTop": "8px"},
            ),
            html.Div(
                id=WHITSUN_MAP_MISSING_IMAGERY,
                style={
                    "marginTop": "6px",
                    "color": _MUTED,
                    "fontSize": "0.7rem",
                    "fontStyle": "italic",
                },
            ),
            html.Div(
                id=WHITSUN_CONTEXT_MAP,
                style={
                    "marginTop": "8px",
                    "color": _MUTED,
                    "fontSize": "0.72rem",
                },
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
                                "Map / evidence overlays",
                                map_body,
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
