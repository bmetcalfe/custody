"""Sidebar layout: scenario selector, timeline slider, entity dropdown."""
from __future__ import annotations

from dash import dcc, html
import dash_bootstrap_components as dbc

# Component IDs — importable by callbacks
SCENARIO_DROPDOWN = "scenario-dropdown"
TIMELINE_SLIDER = "timeline-slider"
TIMESTEP_DISPLAY = "timestep-display"
ENTITY_DROPDOWN = "entity-dropdown"
BTN_PREV_STEP = "btn-prev-step"
BTN_NEXT_STEP = "btn-next-step"


def build_sidebar(
    scenario_names: list[str],
    default_scenario: str,
) -> html.Div:
    """Return the sidebar component tree.

    All controls start in a sensible default state.  Callbacks populate
    dynamic values (slider max, entity options) after scenario load.
    """
    return html.Div(
        [
            html.H5("Custody", style={"marginBottom": "2px", "color": "#f3f4f6"}),
            html.P(
                "Anomaly-aware custody and collection planning",
                style={"color": "#9ca3af", "fontSize": "0.75rem", "marginBottom": "16px"},
            ),

            # ── Scenario selector ────────────────────────────────────────
            dbc.Label("Scenario", html_for=SCENARIO_DROPDOWN, size="sm",
                      style={"color": "#e5e7eb"}),
            dcc.Dropdown(
                id=SCENARIO_DROPDOWN,
                options=[{"label": n, "value": n} for n in scenario_names],
                value=default_scenario,
                clearable=False,
                style={"marginBottom": "16px"},
            ),

            # ── Timeline slider ──────────────────────────────────────────
            dbc.Label("Timeline step", html_for=TIMELINE_SLIDER, size="sm",
                      style={"color": "#e5e7eb"}),
            dcc.Slider(
                id=TIMELINE_SLIDER,
                min=0,
                max=0,       # updated by scenario_changed callback
                step=1,
                value=0,
                marks=None,  # updated by scenario_changed callback
                tooltip={"placement": "bottom", "always_visible": False},
            ),
            html.Div(
                [
                    dbc.Button(
                        "\u25C0", id=BTN_PREV_STEP, size="sm", color="secondary",
                        outline=True, disabled=True,
                        style={"minWidth": "36px", "padding": "2px 8px"},
                    ),
                    html.Span(
                        "Step 0",
                        id=TIMESTEP_DISPLAY,
                        style={
                            "fontSize": "0.75rem", "color": "#aaa",
                            "flex": "1", "textAlign": "center",
                        },
                    ),
                    dbc.Button(
                        "\u25B6", id=BTN_NEXT_STEP, size="sm", color="secondary",
                        outline=True,
                        style={"minWidth": "36px", "padding": "2px 8px"},
                    ),
                ],
                style={
                    "display": "flex", "alignItems": "center",
                    "gap": "6px", "marginBottom": "16px",
                },
            ),

            # ── Entity selector ──────────────────────────────────────────
            dbc.Label("Entity", html_for=ENTITY_DROPDOWN, size="sm",
                      style={"color": "#e5e7eb"}),
            dcc.Dropdown(
                id=ENTITY_DROPDOWN,
                options=[],   # populated by scenario_changed callback
                value=None,
                placeholder="Select entity...",
            ),
        ],
        style={"padding": "16px"},
    )
