"""Dash application entrypoint for the Custody dashboard.

Usage:
    uv run python src/app/dash_app.py
"""
from __future__ import annotations

import os
import sys

# Ensure both 'custody' (engine) and 'src/app' (UI modules) are importable
# when running this file directly.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in [os.path.join(_repo_root, "src"), os.path.join(_repo_root, "src", "app")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dash import Dash, dcc, html
import dash_bootstrap_components as dbc

import state
from adapter import scenario_names
from layout.sidebar import build_sidebar
from layout.overview import build_overview_layout
from layout.entity_detail import build_entity_detail_layout
from layout.tennent_monitoring import (
    TENNENT_SIDEBAR_BLOCK,
    TENNENT_TAB_VALUE,
    build_tennent_monitoring_layout,
    build_tennent_sidebar_block,
)
from layout.whitsun_replay import (
    WHITSUN_SIDEBAR_OVERVIEW,
    WHITSUN_SIDEBAR_REPLAY,
    build_whitsun_replay_layout,
    build_whitsun_sidebar_block,
)
from callbacks import (
    navigation, portfolio, map_layers, entity_detail, whitsun_replay,
)

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
)
app.title = "Custody"

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

_scenarios = scenario_names()
_default = _scenarios[0]  # multi_day_72h

app.layout = html.Div(
    [
        *state.build_stores(),
        dbc.Row(
            [
                dbc.Col(
                    html.Div([
                        html.Div(
                            build_sidebar(_scenarios, _default),
                            id=WHITSUN_SIDEBAR_OVERVIEW,
                        ),
                        html.Div(
                            build_whitsun_sidebar_block(),
                            id=WHITSUN_SIDEBAR_REPLAY,
                            style={"display": "none"},
                        ),
                        html.Div(
                            build_tennent_sidebar_block(),
                            id=TENNENT_SIDEBAR_BLOCK,
                            style={"display": "none"},
                        ),
                    ]),
                    width=2,
                    style={"borderRight": "1px solid #2d2d2d", "minHeight": "100vh"},
                ),
                dbc.Col(
                    dcc.Tabs(
                        id="custody-main-tabs",
                        value="tab-custody-overview",
                        children=[
                            dcc.Tab(
                                label="Custody overview",
                                value="tab-custody-overview",
                                children=html.Div([
                                    build_overview_layout(),
                                    html.Hr(style={"borderColor": "#2d2d2d", "margin": "8px 0"}),
                                    build_entity_detail_layout(),
                                ]),
                            ),
                            dcc.Tab(
                                label="Whitsun replay (fixture)",
                                value="tab-whitsun-replay",
                                children=build_whitsun_replay_layout(),
                            ),
                            dcc.Tab(
                                label="Tennent monitoring (fixture)",
                                value=TENNENT_TAB_VALUE,
                                children=build_tennent_monitoring_layout(),
                            ),
                        ],
                        colors={
                            "border": "#2d2d2d",
                            "primary": "#5eead4",
                            "background": "#1a1a1a",
                        },
                    ),
                    width=10,
                ),
            ],
            className="g-0",
        ),
    ],
    style={"backgroundColor": "#121212", "color": "#ddd", "minHeight": "100vh"},
)

# ---------------------------------------------------------------------------
# Register callbacks
# ---------------------------------------------------------------------------

navigation.register(app)
portfolio.register(app)
map_layers.register(app)
entity_detail.register(app)
whitsun_replay.register(app)

# ---------------------------------------------------------------------------
# Dev server
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=8050)
