"""Overview map layout using dash-deck (pydeck wrapper)."""
from __future__ import annotations

import json

import dash_deck
import pydeck as pdk
from dash import html

# Component IDs
OVERVIEW_MAP = "overview-map"


def _empty_deck_json() -> str:
    """Return a minimal empty pydeck Deck as JSON for initial render."""
    deck = pdk.Deck(
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        initial_view_state=pdk.ViewState(latitude=1.0, longitude=0.5, zoom=6),
        layers=[],
    )
    return deck.to_json()


def build_map_panel() -> html.Div:
    """Return the overview map component with an empty initial state."""
    return html.Div(
        [
            dash_deck.DeckGL(
                id=OVERVIEW_MAP,
                data=_empty_deck_json(),
                tooltip={"text": "{label}"},
                style={"width": "100%", "height": "480px", "position": "relative"},
            ),
            html.Div(
                [
                    html.Span("● Needs Action ", style={"color": "#e63737", "fontSize": "0.7rem", "marginRight": "10px"}),
                    html.Span("● Preempted ", style={"color": "#d78219", "fontSize": "0.7rem", "marginRight": "10px"}),
                    html.Span("● Neglected ", style={"color": "#d7c31e", "fontSize": "0.7rem", "marginRight": "10px"}),
                    html.Span("● Stale ", style={"color": "#af692d", "fontSize": "0.7rem", "marginRight": "10px"}),
                    html.Span("● Watch ", style={"color": "#55a5eb", "fontSize": "0.7rem", "marginRight": "10px"}),
                    html.Span("● Healthy ", style={"color": "#9ba5af", "fontSize": "0.7rem"}),
                ],
                style={"padding": "4px 0", "marginTop": "4px"},
            ),
        ],
        className="mb-3",
    )
