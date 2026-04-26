"""Callbacks for the Evidence Viewer panel (Whitsun + Tennent)."""
from __future__ import annotations

from collections.abc import Iterable

import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
import dash_bootstrap_components as dbc

from custody.demo import (
    EvidenceScene,
    evidence_scene_for_whitsun_event_ordinal,
    evidence_scenes_for_scenario,
    load_evidence_manifest,
)
from custody.demo.decision_trace import DecisionTrace
from custody.demo import load_whitsun_decision_trace

from layout.evidence_viewer import (
    TENNENT_EVIDENCE_DATE_SELECT,
    TENNENT_EVIDENCE_DETECTIONS,
    TENNENT_EVIDENCE_FOOTER,
    TENNENT_EVIDENCE_HEADER,
    TENNENT_EVIDENCE_IMAGE,
    TENNENT_EVIDENCE_TOGGLES,
    WHITSUN_EVIDENCE_DETECTIONS,
    WHITSUN_EVIDENCE_FOOTER,
    WHITSUN_EVIDENCE_HEADER,
    WHITSUN_EVIDENCE_IMAGE,
    WHITSUN_EVIDENCE_TOGGLES,
)
from layout.whitsun_replay import WHITSUN_SELECTED_EVENT_STORE


_TEXT = "#e5e7eb"
_MUTED = "#9ca3af"
_ACCENT = "#5eead4"


# Module-level cache: manifest + decision trace loaded once at import.
_SCENES: tuple[EvidenceScene, ...] = load_evidence_manifest()
_TRACE: DecisionTrace = load_whitsun_decision_trace()


# ---------------------------------------------------------------------------
# Whitsun event ordinals at which to show evidence
# ---------------------------------------------------------------------------


# Event 02: raw SAR evidence (no detections)
# Event 03+: SAR + detections (the VLM-analysis step in the trace)
# Event 12/13: simulated follow-up Umbra — no committed raster
WHITSUN_RAW_FROM_ORDINAL = 2
WHITSUN_DETECTIONS_FROM_ORDINAL = 3
WHITSUN_FOLLOWUP_FROM_ORDINAL = 12


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------


def _na(message: str) -> html.Div:
    return html.Div(
        message,
        style={
            "color": _MUTED, "fontSize": "0.78rem",
            "fontStyle": "italic", "padding": "12px",
            "border": "1px dashed #2d2d2d", "borderRadius": "4px",
            "textAlign": "center",
        },
    )


def _scene_header(scene: EvidenceScene) -> html.Div:
    rows = [
        html.Strong(
            f"{scene.scenario_id.title()} {scene.collection_time}",
            style={"color": _TEXT, "fontSize": "0.85rem"},
        ),
        html.Span(
            f"  ·  {scene.detection_count} VLM detection"
            f"{'s' if scene.detection_count != 1 else ''}",
            style={"color": _MUTED, "fontSize": "0.75rem"},
        ),
    ]
    if scene.detector_version:
        rows.append(
            html.Span(
                f"  ·  detector {scene.detector_version}",
                style={"color": _MUTED, "fontSize": "0.72rem"},
            ),
        )
    return html.Div(rows, style={"marginBottom": "4px"})


def _scene_footer(scene: EvidenceScene) -> html.Div:
    return html.Div(
        f"image: {scene.image_width_px}×{scene.image_height_px} px "
        f"(downsampled {scene.downsample_scale}× from real GEC) · "
        f"source: {scene.source_gec.split('/')[-1]} · "
        f"detections: {scene.source_parquet}",
        style={
            "color": _MUTED, "fontSize": "0.68rem",
            "fontStyle": "italic",
        },
    )


# Standard modebar config for the zoomable image graphs.  Scroll-wheel
# zoom is enabled; selection / lasso / toImage are removed because they
# do not apply to image inspection; the home button (resetScale2d) +
# double-click on the plot both reset the view.
_GRAPH_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "scrollZoom": True,
    "modeBarButtonsToRemove": [
        "select2d",
        "lasso2d",
        "toImage",
    ],
    "doubleClick": "reset",
}


def _build_zoomable_image_figure(
    *, url: str, width: int, height: int, alt: str,
) -> go.Figure:
    """A Plotly figure that renders ``url`` as a zoomable, pannable
    image in pixel coordinates.

    Aspect ratio is preserved by anchoring Y to X with a 1:1 scale.
    Y-axis range is reversed so row 0 of the image sits at the top of
    the plot (image-pixel convention)."""
    fig = go.Figure()
    fig.add_layout_image(
        dict(
            source=url,
            xref="x", yref="y",
            x=0, y=0, sizex=width, sizey=height,
            xanchor="left", yanchor="top",
            sizing="contain",
            layer="below",
        ),
    )
    fig.update_xaxes(
        visible=False,
        range=[0, width],
        constrain="domain",
    )
    fig.update_yaxes(
        visible=False,
        range=[height, 0],
        scaleanchor="x",
        scaleratio=1,
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        dragmode="pan",
        plot_bgcolor="#1a1a1a",
        paper_bgcolor="#1a1a1a",
        # ``meta`` rides along on the figure so accessibility tools and
        # tests can identify what the figure is showing.
        meta={"alt": alt, "asset_url": url},
    )
    return fig


def _image_for(scene: EvidenceScene, toggles: Iterable[str]):
    """Pick raw vs annotated PNG based on toggle state and return a
    zoomable ``dcc.Graph``.  When both toggles are off, render a
    plain placeholder (no graph)."""
    show_detections = "detections" in (toggles or ())
    show_raw = "raw" in (toggles or ())
    if not show_raw and not show_detections:
        return _na("All evidence layers are off; toggle Raw or Detections.")
    url = (
        scene.annotated_asset_url if show_detections
        else scene.raw_asset_url
    )
    fig = _build_zoomable_image_figure(
        url=url,
        width=int(scene.image_width_px),
        height=int(scene.image_height_px),
        alt=f"Umbra SAR evidence for {scene.collection_time}",
    )
    return dcc.Graph(
        id=f"evidence-graph-{scene.scene_id}",
        figure=fig,
        config=_GRAPH_CONFIG,
        style={
            "height": "600px",
            "border": "1px solid #2d2d2d",
            "borderRadius": "4px",
        },
    )


def _detection_row(det) -> html.Div:
    conf_label = (
        f"{det.confidence:.2f}" if det.confidence is not None else "n/a"
    )
    length_label = (
        f"{det.vessel_length_est_m:.0f} m"
        if det.vessel_length_est_m is not None else "—"
    )
    latlon = (
        f"{det.lat:.5f}, {det.lon:.5f}"
        if det.lat is not None and det.lon is not None else "—"
    )
    short_id = det.detection_id.split("-")[-1] if det.detection_id else "?"
    return html.Div(
        [
            html.Div(
                [
                    html.Strong(
                        f"#{short_id}",
                        style={"color": _ACCENT, "fontSize": "0.75rem"},
                    ),
                    html.Span(
                        f"  ·  conf {conf_label}",
                        style={"color": _MUTED, "fontSize": "0.72rem"},
                    ),
                ],
            ),
            html.Div(
                f"lat/lon: {latlon}",
                style={"color": _TEXT, "fontSize": "0.72rem"},
            ),
            html.Div(
                f"vessel length est: {length_label}",
                style={"color": _MUTED, "fontSize": "0.7rem"},
            ),
        ],
        style={
            "padding": "4px 6px",
            "borderBottom": "1px solid #2d2d2d",
        },
    )


def _detection_list(scene: EvidenceScene) -> html.Div:
    if not scene.detections:
        return _na("No detections in this collect.")
    return html.Div(
        [
            html.Div(
                f"{scene.detection_count} detections "
                f"(scroll for full list)",
                style={
                    "color": _MUTED, "fontSize": "0.7rem",
                    "marginBottom": "4px",
                },
            ),
        ] + [_detection_row(d) for d in scene.detections],
    )


# ---------------------------------------------------------------------------
# Whitsun-specific helpers
# ---------------------------------------------------------------------------


def _whitsun_followup_message() -> html.Div:
    return html.Div(
        [
            dbc.Badge(
                "FOLLOW-UP COLLECT",
                color="warning", className="me-2",
            ),
            html.Span(
                "The follow-up Umbra collect is simulated demo "
                "scaffolding for events 12 / 13.  No committed raster "
                "exists.  Evidence will become available once a real "
                "follow-up scene is ingested.",
                style={
                    "color": _MUTED, "fontSize": "0.78rem",
                    "fontStyle": "italic",
                },
            ),
        ],
        style={
            "padding": "8px 10px",
            "border": "1px solid #2d2d2d",
            "borderRadius": "4px",
            "backgroundColor": "rgba(251, 191, 36, 0.06)",
        },
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register(app: Dash) -> None:
    """Register Evidence Viewer callbacks on *app*."""

    @app.callback(
        Output(WHITSUN_EVIDENCE_HEADER, "children"),
        Output(WHITSUN_EVIDENCE_IMAGE, "children"),
        Output(WHITSUN_EVIDENCE_DETECTIONS, "children"),
        Output(WHITSUN_EVIDENCE_FOOTER, "children"),
        Input(WHITSUN_SELECTED_EVENT_STORE, "data"),
        Input(WHITSUN_EVIDENCE_TOGGLES, "value"),
    )
    def _refresh_whitsun_evidence(event_id, toggles):
        event = _TRACE.get_event(event_id) if event_id else {}
        ord_ = int((event or {}).get("ordinal") or 0)

        if ord_ < WHITSUN_RAW_FROM_ORDINAL:
            return (
                "",
                _na(
                    "Evidence chip becomes available once Umbra has "
                    "collected (event 02)."
                ),
                _na("No detections yet."),
                "",
            )

        scene = evidence_scene_for_whitsun_event_ordinal(_SCENES, ord_)
        if scene is None:
            return (
                "",
                _na("No Whitsun evidence scene mapped to this event."),
                _na("No detections."),
                "",
            )

        toggles = list(toggles or ())
        # Event 02 shows raw SAR only (no detections yet, before VLM
        # analysis is "revealed" at event 03).  Force the detections
        # toggle off so we can't accidentally show analysis output
        # before its narrative step.
        if ord_ < WHITSUN_DETECTIONS_FROM_ORDINAL:
            toggles = [t for t in toggles if t != "detections"]
            detection_block = _na(
                "VLM analysis is not yet revealed at this step "
                "(starts at event 03)."
            )
        else:
            detection_block = _detection_list(scene)

        # Event 12/13: surface the simulated-follow-up note alongside
        # the original collect's evidence (we still show the 2023-12-06
        # evidence as the most-recent committed real raster).
        if ord_ >= WHITSUN_FOLLOWUP_FROM_ORDINAL:
            footer = html.Div(
                [
                    _whitsun_followup_message(),
                    html.Div(style={"height": "6px"}),
                    _scene_footer(scene),
                ],
            )
        else:
            footer = _scene_footer(scene)

        return (
            _scene_header(scene),
            _image_for(scene, toggles),
            detection_block,
            footer,
        )

    @app.callback(
        Output(TENNENT_EVIDENCE_HEADER, "children"),
        Output(TENNENT_EVIDENCE_IMAGE, "children"),
        Output(TENNENT_EVIDENCE_DETECTIONS, "children"),
        Output(TENNENT_EVIDENCE_FOOTER, "children"),
        Input(TENNENT_EVIDENCE_DATE_SELECT, "value"),
        Input(TENNENT_EVIDENCE_TOGGLES, "value"),
    )
    def _refresh_tennent_evidence(scene_id, toggles):
        if not scene_id:
            return (
                "",
                _na("Select a Tennent collect to view its evidence."),
                _na("No collect selected."),
                "",
            )
        tennent_scenes = evidence_scenes_for_scenario(_SCENES, "tennent")
        scene = next(
            (s for s in tennent_scenes if s.scene_id == scene_id), None,
        )
        if scene is None:
            return (
                "",
                _na(f"No evidence found for {scene_id}."),
                _na("No detections."),
                "",
            )
        return (
            _scene_header(scene),
            _image_for(scene, toggles),
            _detection_list(scene),
            _scene_footer(scene),
        )
