"""Tennent fixed-site monitoring tab.

Read-only Dash panel that renders the Tennent Reef Sentinel observation
fixture as a fixed-site / infrastructure-change **context** view.  This
is intentionally NOT a custody / reacquisition decision replay; it
demonstrates that the same normalized ``ObservationArtifact`` schema
(see :mod:`custody.ingest.sentinel`) supports a second mission
archetype.

This module is a layout factory only.  No callbacks are registered:
the panel is fully static once loaded, so there are no inputs to
react to.
"""
from __future__ import annotations

import json as _json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import dash_deck
from dash import dcc, html
import dash_bootstrap_components as dbc

from custody.demo import load_evidence_manifest, load_map_overlays
from custody.demo.map_overlays import (
    format_overlay_label,
    is_observation_overlay,
)
from custody.ingest.sentinel import load_observation_cache
from layout.evidence_viewer import build_tennent_evidence_panel
from layout.map_overlays_helpers import (
    BASE_LAYER_TOGGLE_OPTIONS,
    build_deck_json,
    default_base_visibility,
)

# Reuse the small style helpers from the Whitsun replay layout so the
# two tabs render with identical visual discipline.
from layout.whitsun_replay import (
    _ACCENT,
    _MUTED,
    _PANEL_BG,
    _PANEL_BORDER,
    _TEXT,
    _panel,
    data_mode_badge,
    source_badge,
)


# ---------------------------------------------------------------------------
# Public IDs / values (importable by dash_app.py and the sidebar callback)
# ---------------------------------------------------------------------------


TENNENT_TAB_VALUE = "tab-tennent-monitoring"
TENNENT_SIDEBAR_BLOCK = "custody-main-sidebar-tennent"
TENNENT_MONITORING_ROOT = "tennent-monitoring-root"

# Map / overlay panel IDs.
TENNENT_MAP_DECK = "tennent-monitoring-map-deck"
TENNENT_MAP_LAYER_TOGGLES = "tennent-monitoring-map-layers"
TENNENT_OVERLAY_TOGGLES = "tennent-monitoring-overlay-toggles"
TENNENT_MAP_OPACITY = "tennent-monitoring-map-opacity"
TENNENT_MAP_OVERLAY_BADGES = "tennent-monitoring-map-overlay-badges"
TENNENT_MAP_MISSING_IMAGERY = "tennent-monitoring-map-missing-imagery"


# ---------------------------------------------------------------------------
# Fixture paths
# ---------------------------------------------------------------------------


_REPO_ROOT = Path(__file__).resolve().parents[3]
_TENNENT_AOI_PATH = (
    _REPO_ROOT / "data" / "demo" / "tennent_aoi.fixture.geojson"
)
_TENNENT_OBSERVATIONS_PATH = (
    _REPO_ROOT / "data" / "demo"
    / "tennent_sentinel_observations.fixture.json"
)


def _load_aoi_metadata() -> Mapping[str, Any]:
    blob = _json.loads(_TENNENT_AOI_PATH.read_text(encoding="utf-8"))
    return blob.get("metadata") or {}


# ---------------------------------------------------------------------------
# Small render utilities
# ---------------------------------------------------------------------------


def _kv(label: str, value) -> html.Div:
    return html.Div(
        [
            html.Span(
                f"{label}: ",
                style={"color": _MUTED, "fontSize": "0.78rem"},
            ),
            html.Span(
                str(value), style={"color": _TEXT, "fontSize": "0.85rem"},
            ),
        ],
        style={"marginBottom": "2px"},
    )


def _muted_caption(text: str) -> html.Div:
    return html.Div(
        text,
        style={
            "color": _MUTED,
            "fontSize": "0.72rem",
            "fontStyle": "italic",
            "marginBottom": "6px",
        },
    )


def _ul(items: list[str]) -> html.Ul:
    return html.Ul(
        [
            html.Li(
                t,
                style={
                    "color": _TEXT,
                    "fontSize": "0.82rem",
                    "marginBottom": "2px",
                },
            )
            for t in items
        ],
        style={"paddingLeft": "20px", "margin": "4px 0 0 0"},
    )


def _muted_ul(items: list[str]) -> html.Ul:
    return html.Ul(
        [
            html.Li(
                t,
                style={
                    "color": _MUTED,
                    "fontSize": "0.78rem",
                    "marginBottom": "2px",
                },
            )
            for t in items
        ],
        style={"paddingLeft": "20px", "margin": "4px 0 0 0"},
    )


# ---------------------------------------------------------------------------
# Sidebar block (rendered when the Tennent tab is active)
# ---------------------------------------------------------------------------


def build_tennent_sidebar_block() -> html.Div:
    """Read-only sidebar shown when the Tennent monitoring tab is active."""
    return html.Div(
        [
            html.H5(
                "Tennent monitoring",
                style={"color": _TEXT, "marginBottom": "4px"},
            ),
            html.Div(
                "read-only fixture",
                style={"color": _MUTED, "fontSize": "0.78rem"},
            ),
            html.Hr(style={"borderColor": _PANEL_BORDER, "margin": "10px 0"}),
            html.Ul(
                [
                    html.Li("fixed-site monitoring scenario"),
                    html.Li("Sentinel context only"),
                    html.Li("no live inference"),
                    html.Li("no live change detection"),
                ],
                style={
                    "color": _MUTED,
                    "fontSize": "0.75rem",
                    "paddingLeft": "18px",
                    "lineHeight": "1.4",
                },
            ),
            html.Div(
                "Demonstrates that the Sentinel observation schema is "
                "not Whitsun-specific.",
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


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _build_header(aoi_meta: Mapping[str, Any]) -> html.Div:
    return html.Div(
        [
            html.H4(
                "Tennent Reef — fixed-site monitoring",
                style={"color": _TEXT, "marginBottom": "4px"},
            ),
            html.Div(
                [
                    data_mode_badge(aoi_meta.get("data_mode", "fixture")),
                    dbc.Badge(
                        "FIXED-SITE MONITORING",
                        color="info", className="me-1",
                    ),
                    dbc.Badge(
                        "SENTINEL OBSERVATION CONTEXT",
                        color="secondary", className="me-1",
                    ),
                ],
            ),
            _muted_caption(
                "Read-only Sentinel observation context for the Tennent "
                "Reef AOI.  Not a live change-detection pipeline yet."
            ),
        ],
        style={"marginBottom": "10px"},
    )


def _build_aoi_panel(aoi_meta: Mapping[str, Any]) -> dbc.Card:
    body = html.Div(
        [
            _kv(
                "AOI center (lat, lon)",
                f"{aoi_meta.get('center_lat_deg')}, "
                f"{aoi_meta.get('center_lon_deg')}",
            ),
            _kv(
                "bbox (W / E / S / N)",
                f"{aoi_meta.get('bbox_lon_west_deg')} / "
                f"{aoi_meta.get('bbox_lon_east_deg')} / "
                f"{aoi_meta.get('bbox_lat_south_deg')} / "
                f"{aoi_meta.get('bbox_lat_north_deg')}",
            ),
            _kv("scenario_type", "fixed_site_monitoring"),
            _kv("data_mode", aoi_meta.get("data_mode", "fixture")),
            _kv(
                "scenario_role",
                aoi_meta.get(
                    "scenario_role",
                    "fixed-site monitoring / construction context",
                ),
            ),
            html.Div(
                "Caveat: placeholder AOI, replace before operational use.",
                style={
                    "color": _MUTED,
                    "fontSize": "0.72rem",
                    "fontStyle": "italic",
                    "marginTop": "6px",
                },
            ),
        ],
    )
    return _panel("AOI / context", body)


def _build_map_panel(aoi_meta: Mapping[str, Any]) -> dbc.Card:
    """Map / overlay panel for the Tennent tab.

    Static (no event timeline): every Tennent overlay is visible at
    ordinal 0, so revealing is not gated.  The same layer toggles +
    opacity slider as the Whitsun panel so the screen-share viewer
    sees a consistent UI between scenarios.
    """
    overlays = load_map_overlays()
    tennent_overlays = tuple(
        o for o in overlays if o.scenario_id == "tennent"
    )
    center_lat = float(aoi_meta.get("center_lat_deg") or 8.8583)
    center_lon = float(aoi_meta.get("center_lon_deg") or 114.6561)

    initial_default_obs_ids = tuple(
        o.overlay_id for o in tennent_overlays
        if is_observation_overlay(o) and o.default_visible
    )
    initial_default_set = set(initial_default_obs_ids)
    initial_active_overlays = tuple(
        o for o in tennent_overlays
        if (not is_observation_overlay(o))
        or o.overlay_id in initial_default_set
    )
    initial_deck = build_deck_json(
        overlays=initial_active_overlays,
        base_visibility=default_base_visibility(),
        opacity=1.0,
        center_lat=center_lat,
        center_lon=center_lon,
        zoom=12.0,
    )
    # Tennent has no event timeline; every overlay is always available.
    overlay_options = [
        {"label": format_overlay_label(o), "value": o.overlay_id}
        for o in tennent_overlays
        if is_observation_overlay(o)
    ]
    overlay_options.sort(
        key=lambda opt: (opt["label"]),
    )
    layer_default_keys = ["aoi"]
    body = html.Div(
        [
            _muted_caption(
                "Umbra SAR preview overlays are generated from "
                "committed GEC scenes.  Sentinel observations "
                "currently render as weak-signal footprints unless "
                "preview imagery is available."
            ),
            dash_deck.DeckGL(
                id=TENNENT_MAP_DECK,
                data=initial_deck,
                mapboxKey="",
                tooltip={"text": "{tooltip}"},
                style={
                    "width": "100%", "height": "420px",
                    "position": "relative",
                    "borderRadius": "4px", "overflow": "hidden",
                },
            ),
            html.Div(
                [
                    html.Span(
                        "base layers:",
                        style={"color": _MUTED, "fontSize": "0.72rem"},
                    ),
                    dcc.Checklist(
                        id=TENNENT_MAP_LAYER_TOGGLES,
                        options=[
                            {"label": label, "value": key}
                            for key, label in BASE_LAYER_TOGGLE_OPTIONS
                        ],
                        value=layer_default_keys,
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
                        "imagery overlays:",
                        style={"color": _MUTED, "fontSize": "0.72rem"},
                    ),
                    dcc.Checklist(
                        id=TENNENT_OVERLAY_TOGGLES,
                        options=overlay_options,
                        value=list(initial_default_obs_ids),
                        inputStyle={"marginRight": "4px"},
                        labelStyle={
                            "color": _TEXT,
                            "fontSize": "0.72rem",
                            "marginRight": "0",
                            "display": "block",
                        },
                        style={"marginLeft": "8px"},
                    ),
                ],
                style={
                    "marginTop": "6px",
                    "display": "flex",
                    "flexWrap": "wrap",
                    "alignItems": "flex-start",
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
                        id=TENNENT_MAP_OPACITY,
                        min=0.0, max=1.0, step=0.05, value=1.0,
                        marks=None,
                        tooltip={"placement": "bottom",
                                 "always_visible": False},
                    ),
                ],
                style={"marginTop": "4px"},
            ),
            html.Div(
                id=TENNENT_MAP_OVERLAY_BADGES,
                style={"marginTop": "8px"},
            ),
            _build_tennent_map_legend(),
            html.Div(
                id=TENNENT_MAP_MISSING_IMAGERY,
                style={
                    "marginTop": "6px",
                    "color": _MUTED,
                    "fontSize": "0.7rem",
                    "fontStyle": "italic",
                },
            ),
        ],
    )
    return _panel("Map / evidence overlays", body)


def _build_tennent_map_legend() -> html.Div:
    line_style = {
        "color": _MUTED,
        "fontSize": "0.7rem",
        "fontStyle": "italic",
        "lineHeight": "1.4",
    }
    return html.Div(
        [
            html.Div(
                "Umbra SAR — high-confidence confirmation imagery",
                style=line_style,
            ),
            html.Div(
                "Sentinel — weak-signal cueing / context",
                style=line_style,
            ),
            html.Div(
                "footprint-only — no image preview currently loaded",
                style=line_style,
            ),
        ],
        style={
            "marginTop": "8px",
            "padding": "6px 8px",
            "borderTop": f"1px solid {_PANEL_BORDER}",
        },
    )


def _build_observations_panel() -> dbc.Card:
    obs = load_observation_cache(_TENNENT_OBSERVATIONS_PATH)

    header = html.Tr(
        [
            html.Th("source"),
            html.Th("timestamp"),
            html.Th("cloud %"),
            html.Th("confidence_weight"),
            html.Th("cue usable"),
            html.Th("context usable"),
        ],
        style={"color": _MUTED, "fontSize": "0.75rem"},
    )

    body_rows: list[html.Tr] = []
    detail_rows: list[html.Div] = []
    for o in obs:
        cloud = (
            f"{o.cloud_coverage:.0f}" if o.cloud_coverage is not None
            else "n/a"
        )
        body_rows.append(
            html.Tr(
                [
                    html.Td(source_badge(o.source)),
                    html.Td(o.timestamp or "n/a"),
                    html.Td(cloud),
                    html.Td(f"{o.confidence_weight:.2f}"),
                    html.Td("yes" if o.usable_for_detection else "no"),
                    html.Td("yes" if o.usable_for_context else "no"),
                ],
                style={
                    "color": _TEXT, "fontSize": "0.78rem",
                    "borderBottom": "1px solid #2d2d2d",
                },
            ),
        )
        detail_rows.append(
            html.Div(
                [
                    html.Strong(
                        o.observation_id,
                        style={"color": _TEXT, "fontSize": "0.78rem"},
                    ),
                    data_mode_badge(o.data_mode),
                    _muted_ul(list(o.caveats or ())),
                ],
                style={
                    "padding": "6px 0",
                    "borderBottom": "1px solid #2d2d2d",
                },
            ),
        )

    table = html.Table(
        [html.Thead(header), html.Tbody(body_rows)],
        style={"width": "100%", "borderCollapse": "collapse"},
    )

    body = html.Div(
        [
            _muted_caption(
                "Three Sentinel observations from the committed fixture: "
                "one Sentinel-1 GRD, one low-cloud Sentinel-2 L2A, one "
                "cloudy Sentinel-2 L2A.  Confidence weights are demo "
                "heuristics."
            ),
            table,
            html.Div(
                "Per-record caveats:",
                style={
                    "color": _MUTED, "fontSize": "0.75rem",
                    "marginTop": "10px", "marginBottom": "4px",
                },
            ),
            html.Div(detail_rows),
        ],
    )
    return _panel("Sentinel observation timeline", body)


def _build_interpretation_panel() -> dbc.Card:
    body = html.Div(
        [
            html.Div(
                [
                    dbc.Badge(
                        "WEAK-SIGNAL CUE",
                        color="warning", className="me-2",
                    ),
                    html.Span(
                        "Sentinel-1 / Sentinel-2 are weak-signal cueing "
                        "layers in this view, not high-confidence proof. "
                        "Use them as change candidates / tasking cues. "
                        "Umbra (when tasked) remains the high-confidence "
                        "confirmation layer.",
                        style={
                            "color": _MUTED,
                            "fontSize": "0.78rem",
                            "fontStyle": "italic",
                        },
                    ),
                ],
                style={
                    "padding": "6px 8px",
                    "marginBottom": "8px",
                    "border": "1px solid #2d2d2d",
                    "borderRadius": "4px",
                    "backgroundColor": "rgba(251, 191, 36, 0.06)",
                },
            ),
            _ul(
                [
                    "Sentinel-1 and Sentinel-2 provide low-confidence "
                    "temporal context around the Tennent AOI between "
                    "tasked high-confidence collects (Umbra-style).  "
                    "They are useful for cueing higher-resolution "
                    "tasking; they do not stand on their own as proof "
                    "of change.",
                    "Cloudy Sentinel-2 (cloud_coverage > 25%) is "
                    "context-only: usable_for_detection is false, "
                    "usable_for_context is true.  Treat any cloudy "
                    "Sentinel-2 signal as a tasking cue at best.",
                    "This tab demonstrates that the same normalized "
                    "ObservationArtifact schema works for a different "
                    "mission archetype than the Whitsun maritime "
                    "custody / reacquisition replay.",
                ],
            ),
            html.Div(
                "See docs/sentinel_ingestion_demo_aois.md for the full "
                "AOI dispatch and confidence-weight table.",
                style={
                    "color": _MUTED,
                    "fontSize": "0.72rem",
                    "marginTop": "10px",
                    "fontStyle": "italic",
                },
            ),
        ],
    )
    return _panel("Site-monitoring interpretation", body)


def _build_not_yet_panel() -> dbc.Card:
    body = html.Div(
        [
            _muted_caption(
                "What this tab does NOT yet provide.  These are out of "
                "scope for this slice and explicitly not implied by any "
                "panel above."
            ),
            _muted_ul(
                [
                    "no live imagery rendering",
                    "no site-change classifier",
                    "no construction-change detection",
                    "no policy / tasking decision loop",
                    "no operator approval workflow",
                ],
            ),
        ],
    )
    return _panel("Not yet implemented", body)


# ---------------------------------------------------------------------------
# Top-level layout factory
# ---------------------------------------------------------------------------


def build_tennent_monitoring_layout() -> html.Div:
    """Return the complete Tennent monitoring tab tree."""
    aoi_meta = _load_aoi_metadata()
    evidence_scenes = load_evidence_manifest()
    return html.Div(
        [
            _build_header(aoi_meta),
            _build_aoi_panel(aoi_meta),
            _build_map_panel(aoi_meta),
            build_tennent_evidence_panel(evidence_scenes),
            _build_observations_panel(),
            _build_interpretation_panel(),
            _build_not_yet_panel(),
        ],
        id=TENNENT_MONITORING_ROOT,
        style={
            "padding": "12px",
            "backgroundColor": "#121212",
            "color": _TEXT,
        },
    )
