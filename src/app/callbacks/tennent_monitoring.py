"""Callbacks for the Tennent monitoring tab map.

Tennent is static — no event timeline.  The single callback responds
to the user's base-layer toggles, per-overlay toggles, and opacity
changes.
"""
from __future__ import annotations

from dash import Dash, Input, Output, html
import dash_bootstrap_components as dbc

from custody.demo import load_map_overlays, overlays_for_scenario
from custody.demo.map_overlays import (
    format_overlay_label,
    is_observation_overlay,
)

from layout.map_overlays_helpers import (
    base_visibility_from_checked,
    build_deck_json,
    overlays_with_missing_imagery,
)
from layout.tennent_monitoring import (
    TENNENT_MAP_DECK,
    TENNENT_MAP_LAYER_TOGGLES,
    TENNENT_MAP_MISSING_IMAGERY,
    TENNENT_MAP_OPACITY,
    TENNENT_MAP_OVERLAY_BADGES,
    TENNENT_OVERLAY_TOGGLES,
)


_MUTED = "#9ca3af"


# Module-level cache: overlays loaded once at import.
_OVERLAYS = load_map_overlays()
_TENNENT_OVERLAYS = overlays_for_scenario(_OVERLAYS, "tennent")

# Tennent AOI center sourced from the AOI metadata block in the
# committed fixture.  See data/demo/tennent_aoi.fixture.geojson.
_TENNENT_CENTER_LAT = 8.8583
_TENNENT_CENTER_LON = 114.6561


def _format_overlay_badge_text(o) -> str:
    """Audience-facing badge text for one Tennent overlay."""
    return format_overlay_label(o)


def _build_overlay_badges(overlays) -> html.Div:
    """One badge per overlay in audience-facing prose."""
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
        chips.append(
            dbc.Badge(
                _format_overlay_badge_text(o),
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
        Input(TENNENT_OVERLAY_TOGGLES, "value"),
        Input(TENNENT_MAP_OPACITY, "value"),
    )
    def _refresh_tennent_map(checked_layers, checked_overlay_ids, opacity):
        selected = set(checked_overlay_ids or ())
        active = tuple(
            o for o in _TENNENT_OVERLAYS
            if (not is_observation_overlay(o)) or o.overlay_id in selected
        )
        active_obs = tuple(o for o in active if is_observation_overlay(o))
        base_visibility = base_visibility_from_checked(checked_layers)
        deck_json = build_deck_json(
            overlays=active,
            base_visibility=base_visibility,
            opacity=float(opacity if opacity is not None else 1.0),
            center_lat=_TENNENT_CENTER_LAT,
            center_lon=_TENNENT_CENTER_LON,
            zoom=12.0,
        )
        badges = _build_overlay_badges(active_obs)
        missing = _build_missing_note(active_obs)
        return deck_json, badges, missing
