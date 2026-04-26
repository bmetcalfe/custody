"""Shared Evidence Viewer panel for Whitsun replay + Tennent monitoring.

The Evidence Viewer renders the actual SAR chip the VLM analysed for
each Umbra collect plus the per-detection bboxes the VLM emitted.  It
is purely a renderer over ``data/demo/evidence_manifest.fixture.json``;
no live inference, no SAR I/O at runtime, no decision-layer mutation.

Usage:

  - Whitsun replay tab: a single ``build_whitsun_evidence_panel()``
    mounts the panel directly under the map; ``callbacks.evidence``
    drives image visibility and detection list from the selected
    event ordinal.

  - Tennent monitoring tab: ``build_tennent_evidence_panel()`` adds a
    per-date selector + the same image / detection list controls.

Component IDs are exported so callbacks can address them.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from dash import dcc, html
import dash_bootstrap_components as dbc

from custody.demo import EvidenceScene


_PANEL_BG = "#1a1a1a"
_PANEL_BORDER = "#2d2d2d"
_MUTED = "#9ca3af"
_TEXT = "#e5e7eb"
_ACCENT = "#5eead4"
_WARN = "#fbbf24"


# ---------------------------------------------------------------------------
# Component IDs
# ---------------------------------------------------------------------------


# Whitsun replay
WHITSUN_EVIDENCE_PANEL = "whitsun-evidence-panel"
WHITSUN_EVIDENCE_HEADER = "whitsun-evidence-header"
WHITSUN_EVIDENCE_IMAGE = "whitsun-evidence-image"
WHITSUN_EVIDENCE_TOGGLES = "whitsun-evidence-toggles"
WHITSUN_EVIDENCE_DETECTIONS = "whitsun-evidence-detections"
WHITSUN_EVIDENCE_FOOTER = "whitsun-evidence-footer"

# Tennent monitoring
TENNENT_EVIDENCE_PANEL = "tennent-evidence-panel"
TENNENT_EVIDENCE_DATE_SELECT = "tennent-evidence-date-select"
TENNENT_EVIDENCE_HEADER = "tennent-evidence-header"
TENNENT_EVIDENCE_IMAGE = "tennent-evidence-image"
TENNENT_EVIDENCE_TOGGLES = "tennent-evidence-toggles"
TENNENT_EVIDENCE_DETECTIONS = "tennent-evidence-detections"
TENNENT_EVIDENCE_FOOTER = "tennent-evidence-footer"


# ---------------------------------------------------------------------------
# Toggle option set used by both tabs
# ---------------------------------------------------------------------------


EVIDENCE_TOGGLE_OPTIONS: tuple[tuple[str, str], ...] = (
    ("raw", "Raw SAR"),
    ("detections", "Detections"),
    ("labels", "Labels"),
    ("confidence", "Confidence"),
)
# When ``detections`` is on, the annotated PNG (which already bakes in
# per-detection rectangles + confidence labels) is shown.  ``labels`` /
# ``confidence`` are surfaced as visible toggles so the operator
# understands what is in the annotated layer; toggling them off in v1
# falls back to the raw image (we cannot strip just one element from a
# pre-rendered raster without re-rendering, which is out of scope for
# this slice).


def _placeholder(text: str) -> html.Div:
    return html.Div(
        text,
        style={
            "color": _MUTED,
            "fontSize": "0.78rem",
            "fontStyle": "italic",
            "padding": "12px",
            "border": f"1px dashed {_PANEL_BORDER}",
            "borderRadius": "4px",
            "textAlign": "center",
        },
    )


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
# Whitsun evidence panel
# ---------------------------------------------------------------------------


_ZOOM_HINT = (
    "Scroll to zoom · drag to pan · double-click or use the modebar "
    "home button to reset the view."
)


def build_whitsun_evidence_panel() -> dbc.Card:
    body = html.Div(
        [
            html.Div(
                "Real Umbra GEC chip the VLM analysed at this step. "
                "The annotated layer shows the bounding boxes the VLM "
                "emitted (red rectangles + confidence values). Sentinel "
                "scenes are not surfaced here — Sentinel remains a "
                "weak-signal cueing layer.",
                style={
                    "color": _MUTED, "fontSize": "0.72rem",
                    "fontStyle": "italic", "marginBottom": "8px",
                },
            ),
            html.Div(
                _ZOOM_HINT,
                style={
                    "color": _MUTED, "fontSize": "0.7rem",
                    "fontStyle": "italic", "marginBottom": "4px",
                },
            ),
            html.Div(id=WHITSUN_EVIDENCE_HEADER),
            html.Div(
                [
                    html.Span(
                        "show:",
                        style={
                            "color": _MUTED, "fontSize": "0.72rem",
                            "marginRight": "8px",
                        },
                    ),
                    dcc.Checklist(
                        id=WHITSUN_EVIDENCE_TOGGLES,
                        options=[
                            {"label": label, "value": key}
                            for key, label in EVIDENCE_TOGGLE_OPTIONS
                        ],
                        value=[
                            "raw", "detections", "labels", "confidence",
                        ],
                        inline=True,
                        inputStyle={"marginRight": "4px"},
                        labelStyle={
                            "color": _TEXT,
                            "fontSize": "0.72rem",
                            "marginRight": "12px",
                        },
                    ),
                ],
                style={
                    "marginTop": "6px", "marginBottom": "6px",
                    "display": "flex", "alignItems": "center",
                    "flexWrap": "wrap", "gap": "8px",
                },
            ),
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(id=WHITSUN_EVIDENCE_IMAGE),
                        width=8,
                        style={"paddingRight": "8px"},
                    ),
                    dbc.Col(
                        html.Div(
                            id=WHITSUN_EVIDENCE_DETECTIONS,
                            style={
                                "maxHeight": "560px",
                                "overflowY": "auto",
                            },
                        ),
                        width=4,
                    ),
                ],
                className="g-0",
            ),
            html.Div(
                id=WHITSUN_EVIDENCE_FOOTER,
                style={
                    "marginTop": "6px",
                    "color": _MUTED,
                    "fontSize": "0.7rem",
                    "fontStyle": "italic",
                },
            ),
        ],
        id=WHITSUN_EVIDENCE_PANEL,
    )
    return _panel("Evidence viewer (real Umbra chip + VLM detections)", body)


# ---------------------------------------------------------------------------
# Tennent evidence panel
# ---------------------------------------------------------------------------


def build_tennent_evidence_panel(scenes: Iterable[EvidenceScene]) -> dbc.Card:
    tennent_scenes = sorted(
        (s for s in scenes if s.scenario_id == "tennent"),
        key=lambda s: s.collection_time,
    )
    options = [
        {"label": s.collection_time, "value": s.scene_id}
        for s in tennent_scenes
    ]
    initial_value = options[0]["value"] if options else None

    body = html.Div(
        [
            html.Div(
                "Real Umbra GEC chip per collect plus the bounding "
                "boxes the VLM emitted (red rectangles + confidence "
                "values).  Sentinel scenes are intentionally absent — "
                "Sentinel remains a weak-signal cueing layer until "
                "preview imagery is committed.",
                style={
                    "color": _MUTED, "fontSize": "0.72rem",
                    "fontStyle": "italic", "marginBottom": "8px",
                },
            ),
            html.Div(
                _ZOOM_HINT,
                style={
                    "color": _MUTED, "fontSize": "0.7rem",
                    "fontStyle": "italic", "marginBottom": "4px",
                },
            ),
            html.Div(
                [
                    html.Span(
                        "Umbra collect:",
                        style={
                            "color": _MUTED, "fontSize": "0.72rem",
                            "marginRight": "8px",
                        },
                    ),
                    dcc.RadioItems(
                        id=TENNENT_EVIDENCE_DATE_SELECT,
                        options=options,
                        value=initial_value,
                        inline=True,
                        inputStyle={"marginRight": "4px"},
                        labelStyle={
                            "color": _TEXT,
                            "fontSize": "0.72rem",
                            "marginRight": "12px",
                        },
                    ),
                ],
                style={
                    "display": "flex", "alignItems": "center",
                    "flexWrap": "wrap", "gap": "8px",
                    "marginBottom": "8px",
                },
            ),
            html.Div(id=TENNENT_EVIDENCE_HEADER),
            html.Div(
                [
                    html.Span(
                        "show:",
                        style={
                            "color": _MUTED, "fontSize": "0.72rem",
                            "marginRight": "8px",
                        },
                    ),
                    dcc.Checklist(
                        id=TENNENT_EVIDENCE_TOGGLES,
                        options=[
                            {"label": label, "value": key}
                            for key, label in EVIDENCE_TOGGLE_OPTIONS
                        ],
                        value=[
                            "raw", "detections", "labels", "confidence",
                        ],
                        inline=True,
                        inputStyle={"marginRight": "4px"},
                        labelStyle={
                            "color": _TEXT,
                            "fontSize": "0.72rem",
                            "marginRight": "12px",
                        },
                    ),
                ],
                style={
                    "marginTop": "6px", "marginBottom": "6px",
                    "display": "flex", "alignItems": "center",
                    "flexWrap": "wrap", "gap": "8px",
                },
            ),
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(id=TENNENT_EVIDENCE_IMAGE),
                        width=8,
                        style={"paddingRight": "8px"},
                    ),
                    dbc.Col(
                        html.Div(
                            id=TENNENT_EVIDENCE_DETECTIONS,
                            style={
                                "maxHeight": "560px",
                                "overflowY": "auto",
                            },
                        ),
                        width=4,
                    ),
                ],
                className="g-0",
            ),
            html.Div(
                id=TENNENT_EVIDENCE_FOOTER,
                style={
                    "marginTop": "6px",
                    "color": _MUTED,
                    "fontSize": "0.7rem",
                    "fontStyle": "italic",
                },
            ),
        ],
        id=TENNENT_EVIDENCE_PANEL,
    )
    return _panel(
        "Evidence viewer (real Umbra chips + VLM detections)", body,
    )
