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

from dash import Dash, html
import dash_bootstrap_components as dbc

import state
from adapter import scenario_names
from layout.sidebar import build_sidebar
from layout.overview import build_overview_layout
from callbacks import navigation, portfolio

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
                    build_sidebar(_scenarios, _default),
                    width=3,
                    style={"borderRight": "1px solid #2d2d2d", "minHeight": "100vh"},
                ),
                dbc.Col(
                    build_overview_layout(),
                    width=9,
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

# ---------------------------------------------------------------------------
# Dev server
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=8050)
