"""Tests for the Evidence Viewer panel + manifest.

The Evidence Viewer renders the actual SAR chip the VLM analysed for
each Umbra collect plus the per-detection bboxes the VLM emitted.
These tests validate:

  * the committed evidence manifest loads cleanly and references real
    files on disk
  * the Whitsun replay panel hides analysis output before its
    narrative reveal step (event 03)
  * the Tennent panel can list every Umbra collect that has detection
    output and an honest "no detections" message when applicable
  * Sentinel scenes are NOT surfaced as evidence (Sentinel remains a
    weak-signal cueing layer)

No live HTTP, no Dash server is started.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "data" / "demo" / "evidence_manifest.fixture.json"


# ---------------------------------------------------------------------------
# Manifest loader
# ---------------------------------------------------------------------------


def _import_evidence():
    import sys
    src = REPO_ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    import importlib
    return importlib.import_module("custody.demo.evidence")


def test_evidence_manifest_loads() -> None:
    ev = _import_evidence()
    scenes = ev.load_evidence_manifest()
    assert isinstance(scenes, tuple)
    assert all(isinstance(s, ev.EvidenceScene) for s in scenes)
    assert len(scenes) >= 5  # 4 Tennent + 1 Whitsun-trace + 1 off-trace


def test_evidence_manifest_has_required_scenes() -> None:
    ev = _import_evidence()
    scenes = ev.load_evidence_manifest()
    by_id = {s.scene_id: s for s in scenes}
    # Must cover every Tennent date the VLM was run on.
    for sid in (
        "tennent-2023-07-02", "tennent-2023-07-23",
        "tennent-2023-08-07", "tennent-2023-08-13",
    ):
        assert sid in by_id, f"missing Tennent evidence scene {sid}"
    # Whitsun trace scene must exist and be tied to event ordinal 2.
    whitsun = by_id.get("whitsun-2023-12-06")
    assert whitsun is not None
    assert whitsun.whitsun_event_ordinal == 2


def test_evidence_manifest_paths_exist_on_disk() -> None:
    ev = _import_evidence()
    scenes = ev.load_evidence_manifest()
    assert scenes
    for s in scenes:
        raw = REPO_ROOT / s.raw_image_path
        annotated = REPO_ROOT / s.annotated_image_path
        assert raw.exists(), f"raw image missing for {s.scene_id}: {raw}"
        assert annotated.exists(), (
            f"annotated image missing for {s.scene_id}: {annotated}"
        )
        assert s.image_width_px > 0
        assert s.image_height_px > 0


def test_evidence_manifest_has_real_detections() -> None:
    ev = _import_evidence()
    scenes = ev.load_evidence_manifest()
    # At least the 2023-07-02 Tennent and 2023-12-06 Whitsun scenes
    # must have non-zero detections drawn from the committed parquets.
    by_id = {s.scene_id: s for s in scenes}
    assert by_id["tennent-2023-07-02"].detection_count > 0
    assert by_id["whitsun-2023-12-06"].detection_count > 0
    # Spot-check a detection record.
    det = by_id["whitsun-2023-12-06"].detections[0]
    assert det.confidence is not None
    assert det.bbox_x1 is not None and det.bbox_y1 is not None
    assert det.bbox_x2 > det.bbox_x1
    assert det.bbox_y2 > det.bbox_y1


def test_evidence_manifest_excludes_sentinel() -> None:
    """Sentinel must not appear as evidence — it is a weak-signal cueing
    layer, never confirmation/evidence."""
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    blob = json.dumps(payload).lower()
    # Anti-pattern phrases must not appear in the evidence manifest.
    for forbidden in ("sentinel proves", "definitive change"):
        assert forbidden not in blob
    # No scene should advertise itself as Sentinel.
    for scene in payload["scenes"]:
        sid = scene.get("scene_id", "")
        assert "sentinel" not in sid.lower()


def test_evidence_scene_for_whitsun_event_ordinal_gating() -> None:
    ev = _import_evidence()
    scenes = ev.load_evidence_manifest()
    # Before event 2: no Whitsun evidence.
    assert ev.evidence_scene_for_whitsun_event_ordinal(scenes, 1) is None
    # Event 2 onward: Whitsun 2023-12-06 evidence.
    s = ev.evidence_scene_for_whitsun_event_ordinal(scenes, 2)
    assert s is not None and s.scene_id == "whitsun-2023-12-06"
    # Later events: still the same most-recent Whitsun-trace scene.
    s = ev.evidence_scene_for_whitsun_event_ordinal(scenes, 14)
    assert s is not None and s.scene_id == "whitsun-2023-12-06"


# ---------------------------------------------------------------------------
# Whitsun replay panel: ordinal-driven reveal of evidence
# ---------------------------------------------------------------------------


def _import_evidence_callbacks():
    import sys
    src_app = REPO_ROOT / "src" / "app"
    src = REPO_ROOT / "src"
    for p in (str(src), str(src_app)):
        if p not in sys.path:
            sys.path.insert(0, p)
    for k in list(sys.modules):
        if k.startswith("layout.") or k.startswith("callbacks."):
            del sys.modules[k]
    import callbacks.evidence_viewer as mod
    return mod


def test_whitsun_evidence_renders_no_chip_before_event_two() -> None:
    cb = _import_evidence_callbacks()
    rendered = str(cb._na("anything")).lower()  # sanity: helper compiles
    assert "anything" in rendered
    # Direct invocation of the registered callback function isn't
    # straightforward with Dash's decorator; instead, exercise the
    # helpers we own and rely on integration coverage at the layout
    # level.
    scenes = cb._SCENES
    s = cb.evidence_scene_for_whitsun_event_ordinal(scenes, 1)
    assert s is None


def test_whitsun_evidence_renders_chip_at_event_two() -> None:
    cb = _import_evidence_callbacks()
    s = cb.evidence_scene_for_whitsun_event_ordinal(cb._SCENES, 2)
    assert s is not None
    assert s.scene_id == "whitsun-2023-12-06"
    assert s.detection_count > 0


def test_whitsun_evidence_image_helper_picks_annotated_when_detections_on() -> None:
    cb = _import_evidence_callbacks()
    s = cb.evidence_scene_for_whitsun_event_ordinal(cb._SCENES, 3)
    img = cb._image_for(s, ["raw", "detections"])
    # Confirm via the figure's meta payload (set by
    # _build_zoomable_image_figure) so this test isn't sensitive to
    # how Dash stringifies the component tree.
    fig = img.figure
    assert fig.layout.meta["asset_url"] == s.annotated_asset_url


def test_whitsun_evidence_image_helper_picks_raw_when_only_raw_on() -> None:
    cb = _import_evidence_callbacks()
    s = cb.evidence_scene_for_whitsun_event_ordinal(cb._SCENES, 3)
    img = cb._image_for(s, ["raw"])
    fig = img.figure
    assert fig.layout.meta["asset_url"] == s.raw_asset_url
    assert s.annotated_asset_url != s.raw_asset_url


# ---------------------------------------------------------------------------
# Zoom / pan / reset behaviour
# ---------------------------------------------------------------------------


def test_evidence_image_returns_zoomable_dcc_graph() -> None:
    """The image area should be a dcc.Graph so we get Plotly's
    built-in zoom/pan/reset controls instead of a static <img>."""
    from dash import dcc
    cb = _import_evidence_callbacks()
    s = cb.evidence_scene_for_whitsun_event_ordinal(cb._SCENES, 3)
    component = cb._image_for(s, ["raw"])
    assert isinstance(component, dcc.Graph)


def test_evidence_graph_preserves_aspect_ratio() -> None:
    """Square SAR pixels: y-axis must scale to the x-axis with ratio 1."""
    cb = _import_evidence_callbacks()
    s = cb.evidence_scene_for_whitsun_event_ordinal(cb._SCENES, 3)
    fig = cb._image_for(s, ["raw"]).figure
    assert fig.layout.yaxis.scaleanchor == "x"
    assert fig.layout.yaxis.scaleratio == 1


def test_evidence_graph_uses_image_pixel_coordinates() -> None:
    """X / Y axis ranges must match the image pixel dimensions and the
    Y range must be reversed so row 0 sits at the top."""
    cb = _import_evidence_callbacks()
    s = cb.evidence_scene_for_whitsun_event_ordinal(cb._SCENES, 3)
    fig = cb._image_for(s, ["raw"]).figure
    assert list(fig.layout.xaxis.range) == [0, s.image_width_px]
    # Reversed Y so image row 0 is at the top.
    assert list(fig.layout.yaxis.range) == [s.image_height_px, 0]


def test_evidence_graph_hides_axes() -> None:
    cb = _import_evidence_callbacks()
    s = cb.evidence_scene_for_whitsun_event_ordinal(cb._SCENES, 3)
    fig = cb._image_for(s, ["raw"]).figure
    assert fig.layout.xaxis.visible is False
    assert fig.layout.yaxis.visible is False


def test_evidence_graph_config_supports_scroll_zoom_and_reset() -> None:
    cb = _import_evidence_callbacks()
    s = cb.evidence_scene_for_whitsun_event_ordinal(cb._SCENES, 3)
    component = cb._image_for(s, ["raw"])
    cfg = component.config
    assert cfg.get("scrollZoom") is True
    assert cfg.get("displayModeBar") is True
    assert cfg.get("doubleClick") == "reset"
    # Selection / lasso / toImage are removed; the standard zoom / pan /
    # zoomIn / zoomOut / resetScale / autoScale stay in.
    removed = set(cfg.get("modeBarButtonsToRemove") or [])
    assert {"select2d", "lasso2d"} <= removed


def test_evidence_graph_default_dragmode_is_pan() -> None:
    cb = _import_evidence_callbacks()
    s = cb.evidence_scene_for_whitsun_event_ordinal(cb._SCENES, 3)
    fig = cb._image_for(s, ["raw"]).figure
    assert fig.layout.dragmode == "pan"


def test_zoom_hint_present_in_both_panels() -> None:
    """Visible operator hint explains how to zoom / pan / reset."""
    whit = _import_layout("layout.whitsun_replay")
    rendered_w = str(whit.build_whitsun_replay_layout()).lower()
    assert "scroll to zoom" in rendered_w
    assert "double-click" in rendered_w

    tenn = _import_layout("layout.tennent_monitoring")
    rendered_t = str(tenn.build_tennent_monitoring_layout()).lower()
    assert "scroll to zoom" in rendered_t
    assert "double-click" in rendered_t


def test_whitsun_followup_message_present_for_event_twelve_plus() -> None:
    cb = _import_evidence_callbacks()
    msg = str(cb._whitsun_followup_message())
    lower = msg.lower()
    assert "follow-up" in lower
    assert "simulated" in lower


# ---------------------------------------------------------------------------
# Whitsun + Tennent layouts mount the evidence panel
# ---------------------------------------------------------------------------


def _walk_ids(component) -> list[str]:
    seen: list[str] = []

    def _walk(c):
        if c is None:
            return
        cid = getattr(c, "id", None)
        if cid is not None:
            seen.append(str(cid))
        children = getattr(c, "children", None)
        if children is None:
            return
        if isinstance(children, list):
            for ch in children:
                _walk(ch)
        else:
            _walk(children)

    _walk(component)
    return seen


def _import_layout(module_name: str):
    import sys
    src_app = REPO_ROOT / "src" / "app"
    src = REPO_ROOT / "src"
    for p in (str(src), str(src_app)):
        if p not in sys.path:
            sys.path.insert(0, p)
    for k in list(sys.modules):
        if k.startswith("layout.") or k.startswith("callbacks."):
            del sys.modules[k]
    import importlib
    return importlib.import_module(module_name)


def test_whitsun_layout_mounts_evidence_panel() -> None:
    mod = _import_layout("layout.whitsun_replay")
    layout = mod.build_whitsun_replay_layout()
    ids = _walk_ids(layout)
    assert "whitsun-evidence-panel" in ids
    assert "whitsun-evidence-image" in ids
    assert "whitsun-evidence-detections" in ids


def test_tennent_layout_mounts_evidence_panel_with_date_selector() -> None:
    mod = _import_layout("layout.tennent_monitoring")
    layout = mod.build_tennent_monitoring_layout()
    ids = _walk_ids(layout)
    assert "tennent-evidence-panel" in ids
    assert "tennent-evidence-date-select" in ids
    assert "tennent-evidence-image" in ids
    rendered = str(layout)
    # Each Tennent collect with a parquet should be selectable by date.
    for date in ("2023-07-02", "2023-07-23", "2023-08-07", "2023-08-13"):
        assert date in rendered, (
            f"Tennent evidence date selector missing {date}"
        )


def test_tennent_evidence_panel_default_selects_first_collect() -> None:
    cb = _import_evidence_callbacks()
    tennent_scenes = cb.evidence_scenes_for_scenario(cb._SCENES, "tennent")
    assert tennent_scenes
    # Sorted by collection_time, first should be 2023-07-02.
    sorted_scenes = sorted(tennent_scenes, key=lambda s: s.collection_time)
    assert sorted_scenes[0].scene_id == "tennent-2023-07-02"


# ---------------------------------------------------------------------------
# Map remains available — Evidence Viewer is additive, not replacing
# ---------------------------------------------------------------------------


def test_whitsun_layout_still_has_map() -> None:
    mod = _import_layout("layout.whitsun_replay")
    layout = mod.build_whitsun_replay_layout()
    ids = _walk_ids(layout)
    assert "whitsun-replay-map-deck" in ids


def test_tennent_layout_still_has_map() -> None:
    mod = _import_layout("layout.tennent_monitoring")
    layout = mod.build_tennent_monitoring_layout()
    ids = _walk_ids(layout)
    assert "tennent-monitoring-map-deck" in ids


# ---------------------------------------------------------------------------
# Sentinel cueing context — separate from VLM detection evidence
# ---------------------------------------------------------------------------


def test_sentinel_cueing_div_mounted_on_both_panels() -> None:
    whit = _import_layout("layout.whitsun_replay")
    ids_w = _walk_ids(whit.build_whitsun_replay_layout())
    assert "whitsun-sentinel-cueing" in ids_w

    tenn = _import_layout("layout.tennent_monitoring")
    ids_t = _walk_ids(tenn.build_tennent_monitoring_layout())
    assert "tennent-sentinel-cueing" in ids_t


def test_sentinel_cueing_section_carries_required_disclaimer() -> None:
    """The dispatch-mandated cueing copy must be in the rendered output."""
    from custody.demo import load_map_overlays
    from custody.demo.map_overlays import overlays_for_scenario
    from layout.evidence_viewer import (
        _render_sentinel_cueing_section, SENTINEL_CUEING_COPY,
    )
    overlays = overlays_for_scenario(load_map_overlays(), "tennent")
    rendered = str(_render_sentinel_cueing_section(overlays))
    # Verbatim disclaimer.
    assert SENTINEL_CUEING_COPY in rendered
    # Honest framing markers.
    lower = rendered.lower()
    assert "weak-signal cue" in lower
    # Sentinel must NOT be advertised as confirmation evidence.
    assert "high-confidence confirmation" not in lower
    assert "definitive change" not in lower
    assert "sentinel proves" not in lower


def test_sentinel_label_says_image_when_preview_present() -> None:
    """``overlay_kind_label`` must distinguish Sentinel-with-preview
    from Sentinel-footprint-only."""
    from custody.demo.map_overlays import OverlayArtifact, overlay_kind_label

    with_preview = OverlayArtifact(
        overlay_id="x", scenario_id="whitsun", observation_id="o",
        source="sentinel-2", sensor_type="optical", display_name="S2",
        data_mode="fixture",
        image_path="src/app/assets/evidence/sentinel/whitsun/o/preview.png",
        asset_url="/assets/evidence/sentinel/whitsun/o/preview.png",
        image_kind="sentinel_preview",
        bounds=(114.45, 9.75, 114.85, 10.25),
        geometry=None,
        opacity_default=0.55, visible_from_event_ordinal=4, z_index=25,
        confidence_weight=0.30, usable_for_detection=True,
        usable_for_context=True,
        caveats=("weak-signal cueing layer; not high-confidence proof",),
        missing_asset_reason=None,
    )
    without_preview = OverlayArtifact(
        **{**with_preview.__dict__,
           "image_path": None, "asset_url": None,
           "image_kind": "footprint-only"},
    )
    assert overlay_kind_label(with_preview) == "weak-signal image"
    assert overlay_kind_label(without_preview) == "weak-signal footprint"
    # Neither must say "image overlay" — that term is reserved for
    # high-confidence Umbra previews.
    assert overlay_kind_label(with_preview) != "image overlay"


def test_sentinel_render_uses_thumbnail_when_preview_exists(tmp_path) -> None:
    """When an overlay has image_path/asset_url, the row renders a
    thumbnail; otherwise it renders a footprint-only placeholder."""
    from custody.demo.map_overlays import OverlayArtifact
    from layout.evidence_viewer import _render_sentinel_overlay_row

    with_preview = OverlayArtifact(
        overlay_id="x", scenario_id="tennent", observation_id="o",
        source="sentinel-2", sensor_type="optical",
        display_name="Sentinel-2 L2A 2023-07-18 (12% cloud)",
        data_mode="fixture",
        image_path="src/app/assets/evidence/sentinel/tennent/o/preview.png",
        asset_url="/assets/evidence/sentinel/tennent/o/preview.png",
        image_kind="sentinel_preview",
        bounds=(114.55, 8.78, 114.75, 8.93),
        geometry=None,
        opacity_default=0.55, visible_from_event_ordinal=0, z_index=25,
        confidence_weight=0.30, usable_for_detection=True,
        usable_for_context=True,
        caveats=("weak-signal cueing layer; not high-confidence proof",),
        missing_asset_reason=None,
    )
    rendered_with = str(_render_sentinel_overlay_row(with_preview))
    assert "/assets/evidence/sentinel/tennent/o/preview.png" in rendered_with

    without = OverlayArtifact(
        **{**with_preview.__dict__,
           "image_path": None, "asset_url": None,
           "image_kind": "footprint-only",
           "missing_asset_reason": "no committed Sentinel-2 RGB preview"},
    )
    rendered_without = str(_render_sentinel_overlay_row(without))
    assert "/assets/evidence/sentinel/" not in rendered_without
    # No raster: the row carries the missing_asset_reason text and
    # the kind label demotes to "weak-signal footprint".
    assert "no committed Sentinel-2 RGB preview" in rendered_without
    assert "weak-signal footprint" in rendered_without.lower()


def test_sentinel_cueing_section_empty_when_no_sentinel_overlays() -> None:
    """An empty / Umbra-only overlay list renders the honest 'none
    available' message, not a thumbnail grid."""
    from layout.evidence_viewer import _render_sentinel_cueing_section
    from custody.demo.map_overlays import OverlayArtifact

    umbra_only = (
        OverlayArtifact(
            overlay_id="u", scenario_id="whitsun", observation_id="o",
            source="umbra", sensor_type="sar", display_name="Umbra",
            data_mode="real", image_path="src/app/assets/x.png",
            asset_url="/assets/x.png", image_kind="png",
            bounds=(0.0, 0.0, 1.0, 1.0), geometry=None,
            opacity_default=1.0, visible_from_event_ordinal=0,
            z_index=30, confidence_weight=1.0,
            usable_for_detection=True, usable_for_context=True,
            caveats=(), missing_asset_reason=None,
        ),
    )
    rendered = str(_render_sentinel_cueing_section(umbra_only)).lower()
    assert "no sentinel cueing context available" in rendered


def test_sentinel_overlays_never_advertise_high_confidence_evidence() -> None:
    """Cross-cutting honesty guard: Sentinel overlays in the manifest
    must never carry confirmation-grade language."""
    from custody.demo import load_map_overlays
    overlays = load_map_overlays()
    sentinel = [
        o for o in overlays
        if o.source in ("sentinel-1", "sentinel-2")
    ]
    assert sentinel, "expected committed Sentinel overlays"
    for o in sentinel:
        # confirmation_layer flag must be False.
        assert o.confirmation_layer is False, o.overlay_id
        # weak_signal flag must be True.
        assert o.weak_signal is True, o.overlay_id
        # The existing caveats legitimately use the phrase
        # ``not high-confidence proof`` to disclaim Sentinel; the guard
        # we want is that no Sentinel overlay AFFIRMS high confidence.
        blob = (
            (o.display_name or "")
            + " "
            + " ".join(o.caveats or ())
        ).lower()
        assert "confirmed detection" not in blob, o.overlay_id
        assert "definitive change" not in blob, o.overlay_id
        assert "sentinel proves" not in blob, o.overlay_id
        # Affirmative high-confidence framing is forbidden; the
        # negative form ("not high-confidence ...") is allowed.
        for affirmative in (
            "is high-confidence",
            "high-confidence confirmation",
            "high-confidence detection",
        ):
            assert affirmative not in blob, (
                f"{o.overlay_id} carries forbidden affirmative "
                f"high-confidence framing: {affirmative}"
            )
