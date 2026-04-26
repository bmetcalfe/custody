"""Callbacks for the Tennent monitoring tab map.

Tennent is static — no event timeline.  This single callback responds
to the user's layer-toggle and opacity changes.
"""
from __future__ import annotations

from dash import Dash, Input, Output, html
import dash_bootstrap_components as dbc

from custody.demo import load_map_overlays, overlays_for_scenario

from layout.map_overlays_helpers import (
    build_deck_json,
    overlays_with_missing_imagery,
    visibility_from_checked,
)
from layout.tennent_monitoring import (
    TENNENT_MAP_DECK,
    TENNENT_MAP_LAYER_TOGGLES,
    TENNENT_MAP_MISSING_IMAGERY,
    TENNENT_MAP_OPACITY,
    TENNENT_MAP_OVERLAY_BADGES,
)


_MUTED = "#9ca3af"


# Module-level cache: overlays loaded once at import.
_OVERLAYS = load_map_overlays()
_TENNENT_OVERLAYS = overlays_for_scenario(_OVERLAYS, "tennent")

# Tennent AOI center sourced from the AOI metadata block in the
# committed fixture.  See data/demo/tennent_aoi.fixture.geojson.
_TENNENT_CENTER_LAT = 8.8583
_TENNENT_CENTER_LON = 114.6561


def _build_overlay_badges(overlays) -> html.Div:
    if not overlays:
        return html.Div(
            "no overlays available",
            style={"color": _MUTED, "fontSize": "0.7rem"},
        )
    chips = []
    for o in overlays:
        if o.observation_id is None:
            chips.append(
                dbc.Badge(
                    o.display_name,
                    color="light", text_color="dark",
                    className="me-1",
                ),
            )
            continue
        cw = (
            f" w={o.confidence_weight:.2f}"
            if o.confidence_weight is not None else ""
        )
        chips.append(
            dbc.Badge(
                f"{o.display_name} · {o.image_kind}{cw}",
                color=(
                    "primary" if o.source == "umbra"
                    else "info" if o.source == "sentinel-1"
                    else "warning" if o.source == "sentinel-2"
                    else "secondary"
                ),
                className="me-1",
            ),
        )
    return html.Div(
        chips,
        style={"display": "flex", "flexWrap": "wrap", "gap": "4px"},
    )


def _build_missing_note(overlays) -> html.Div:
    missing = overlays_with_missing_imagery(overlays)
    if not missing:
        return html.Div()
    items = [
        html.Li(
            f"{o.display_name} — {o.missing_asset_reason or 'no asset'}",
            style={
                "color": _MUTED, "fontSize": "0.7rem",
                "fontStyle": "italic",
            },
        )
        for o in missing
    ]
    return html.Div(
        [
            html.Span(
                "image asset not available for:",
                style={"color": _MUTED, "fontSize": "0.7rem"},
            ),
            html.Ul(items, style={"paddingLeft": "16px", "margin": "2px 0"}),
        ],
    )


def register(app: Dash) -> None:
    @app.callback(
        Output(TENNENT_MAP_DECK, "data"),
        Output(TENNENT_MAP_OVERLAY_BADGES, "children"),
        Output(TENNENT_MAP_MISSING_IMAGERY, "children"),
        Input(TENNENT_MAP_LAYER_TOGGLES, "value"),
        Input(TENNENT_MAP_OPACITY, "value"),
    )
    def _refresh_tennent_map(checked_layers, opacity):
        visibility = visibility_from_checked(checked_layers)
        deck_json = build_deck_json(
            overlays=_TENNENT_OVERLAYS,
            layer_visibility=visibility,
            opacity=float(opacity if opacity is not None else 0.6),
            center_lat=_TENNENT_CENTER_LAT,
            center_lon=_TENNENT_CENTER_LON,
            zoom=12.0,
        )
        badges = _build_overlay_badges(_TENNENT_OVERLAYS)
        missing = _build_missing_note(_TENNENT_OVERLAYS)
        return deck_json, badges, missing
