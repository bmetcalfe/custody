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

from dash import html
import dash_bootstrap_components as dbc

from custody.ingest.sentinel import load_observation_cache

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


def _build_observations_panel() -> dbc.Card:
    obs = load_observation_cache(_TENNENT_OBSERVATIONS_PATH)

    header = html.Tr(
        [
            html.Th("source"),
            html.Th("timestamp"),
            html.Th("cloud %"),
            html.Th("confidence_weight"),
            html.Th("for detection"),
            html.Th("for context"),
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
            _ul(
                [
                    "Sentinel-1 and Sentinel-2 provide public temporal "
                    "context around the Tennent AOI between any tasked "
                    "high-confidence collects (Umbra-style).",
                    "Cloudy Sentinel-2 (cloud_coverage > 25%) is "
                    "context-only: usable_for_detection is false, but "
                    "usable_for_context is true.",
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
    return html.Div(
        [
            _build_header(aoi_meta),
            _build_aoi_panel(aoi_meta),
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
