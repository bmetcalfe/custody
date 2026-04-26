"""Tests for the map overlay manifest, the deck-rendering helpers,
and the Dash map panels for the Whitsun and Tennent tabs.

No live HTTP, no Dash server is started.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from custody.demo import (
    MAP_OVERLAYS_PATH,
    OverlayArtifact,
    available_overlays_for,
    has_image_asset,
    load_map_overlays,
    overlays_for_scenario,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Manifest loader
# ---------------------------------------------------------------------------


def test_overlay_manifest_loads() -> None:
    overlays = load_map_overlays()
    assert isinstance(overlays, tuple)
    assert all(isinstance(o, OverlayArtifact) for o in overlays)
    assert len(overlays) >= 8


def test_overlay_manifest_default_path_resolves() -> None:
    assert MAP_OVERLAYS_PATH.exists()


def test_overlay_manifest_explicit_path_load(tmp_path: Path) -> None:
    blob = json.loads(MAP_OVERLAYS_PATH.read_text(encoding="utf-8"))
    out = tmp_path / "overlays.json"
    out.write_text(json.dumps(blob))
    overlays = load_map_overlays(out)
    assert overlays


def test_overlay_invalid_scenario_raises(tmp_path: Path) -> None:
    out = tmp_path / "bad.json"
    out.write_text(json.dumps({
        "overlays": [{
            "overlay_id": "x", "scenario_id": "mars", "source": "umbra",
            "image_kind": "footprint-only",
        }],
    }))
    with pytest.raises(ValueError):
        load_map_overlays(out)


def test_overlay_invalid_image_kind_raises(tmp_path: Path) -> None:
    out = tmp_path / "bad.json"
    out.write_text(json.dumps({
        "overlays": [{
            "overlay_id": "x", "scenario_id": "whitsun", "source": "umbra",
            "image_kind": "bogus",
        }],
    }))
    with pytest.raises(ValueError):
        load_map_overlays(out)


# ---------------------------------------------------------------------------
# Whitsun ordinal gating
# ---------------------------------------------------------------------------


def _ids(overlays):
    return tuple(o.overlay_id for o in overlays)


def test_whitsun_ordinal_01_only_aoi() -> None:
    overlays = load_map_overlays()
    avail = available_overlays_for(
        overlays, scenario_id="whitsun", current_ordinal=1,
    )
    ids = _ids(avail)
    assert "whitsun-aoi" in ids
    # No imagery / footprint overlays at ordinal 1.
    assert not any(
        o.observation_id is not None for o in avail
    ), f"event 01 should expose no observation overlays; got {ids}"


def test_whitsun_umbra_overlay_appears_at_ordinal_2() -> None:
    overlays = load_map_overlays()
    avail = available_overlays_for(
        overlays, scenario_id="whitsun", current_ordinal=2,
    )
    ids = _ids(avail)
    assert "whitsun-umbra-20231206" in ids
    assert "whitsun-sentinel-1-grd-20231210" not in ids
    assert "whitsun-sentinel-2-low-cloud-20231212" not in ids


def test_whitsun_sentinel_2_overlays_appear_at_ordinal_5() -> None:
    """Sentinel-2 cue arrives at trace ordinal 5 (after Candidate
    tracks initialise at ordinal 4 in the dispatch-ordered trace)."""
    overlays = load_map_overlays()
    avail = available_overlays_for(
        overlays, scenario_id="whitsun", current_ordinal=5,
    )
    ids = _ids(avail)
    assert "whitsun-sentinel-2-low-cloud-20231212" in ids
    assert "whitsun-sentinel-2-cloudy-20231215" in ids
    # Sentinel-1 still gated until ordinal 6.
    assert "whitsun-sentinel-1-grd-20231210" not in ids


def test_whitsun_sentinel_1_overlay_appears_at_ordinal_6() -> None:
    """Sentinel-1 cue arrives at trace ordinal 6, immediately after S2."""
    overlays = load_map_overlays()
    avail = available_overlays_for(
        overlays, scenario_id="whitsun", current_ordinal=6,
    )
    assert "whitsun-sentinel-1-grd-20231210" in _ids(avail)


def test_whitsun_followup_overlay_appears_at_ordinal_13() -> None:
    """Follow-up Umbra collect + outcome are merged at trace ordinal 13."""
    overlays = load_map_overlays()
    avail = available_overlays_for(
        overlays, scenario_id="whitsun", current_ordinal=13,
    )
    assert "whitsun-umbra-followup-20231213" in _ids(avail)
    # Pre-merge step (ordinal 12 = "Human approves") must NOT include it.
    avail_12 = available_overlays_for(
        overlays, scenario_id="whitsun", current_ordinal=12,
    )
    assert "whitsun-umbra-followup-20231213" not in _ids(avail_12)


def test_whitsun_ordinal_14_includes_all_whitsun_overlays() -> None:
    overlays = load_map_overlays()
    avail = available_overlays_for(
        overlays, scenario_id="whitsun", current_ordinal=14,
    )
    expected = {
        "whitsun-aoi",
        "whitsun-umbra-20231206",
        "whitsun-sentinel-1-grd-20231210",
        "whitsun-sentinel-2-low-cloud-20231212",
        "whitsun-sentinel-2-cloudy-20231215",
        "whitsun-umbra-followup-20231213",
    }
    assert expected.issubset(set(_ids(avail)))


# ---------------------------------------------------------------------------
# Tennent overlays — static (no event timeline)
# ---------------------------------------------------------------------------


def test_tennent_overlays_present_static() -> None:
    overlays = load_map_overlays()
    avail = overlays_for_scenario(overlays, "tennent")
    ids = set(_ids(avail))
    assert "tennent-aoi" in ids
    assert "tennent-sentinel-1-grd-20230715" in ids
    assert "tennent-sentinel-2-low-cloud-20230718" in ids
    assert "tennent-sentinel-2-cloudy-20230728" in ids


def test_tennent_overlays_use_tennent_aoi_bbox() -> None:
    overlays = load_map_overlays()
    avail = overlays_for_scenario(overlays, "tennent")
    for o in avail:
        bounds = o.bounds
        assert bounds is not None
        # Tennent AOI roughly 114.55-114.75 / 8.78-8.93.
        assert 114.4 < bounds[0] < 114.8
        assert 8.7 < bounds[1] < 9.0


# ---------------------------------------------------------------------------
# Missing imagery is labelled honestly
# ---------------------------------------------------------------------------


def test_no_overlay_silently_pretends_to_be_imagery() -> None:
    overlays = load_map_overlays()
    for o in overlays:
        # A footprint-only overlay must explicitly admit it has no
        # image asset.  Observation-bearing footprint-only overlays
        # must carry a non-empty missing_asset_reason.
        if o.image_kind == "footprint-only" and o.observation_id is not None:
            assert o.missing_asset_reason, (
                f"observation overlay {o.overlay_id} is footprint-only "
                f"but has no missing_asset_reason"
            )
            assert not has_image_asset(o)


# ---------------------------------------------------------------------------
# Real Umbra overlays from data/raw/umbra
# ---------------------------------------------------------------------------


def test_real_umbra_overlays_present() -> None:
    overlays = load_map_overlays()
    real = [o for o in overlays if o.data_mode == "real"]
    # Per the inventory: 1 Whitsun (2023-12-06) + 5 Tennent + 1 Whitsun
    # 2024-03-20 = 7 real overlays after dedup.
    assert len(real) >= 6
    assert all(has_image_asset(o) for o in real)
    assert all(o.image_kind == "png" for o in real)
    assert all(o.source == "umbra" for o in real)


def test_real_umbra_overlay_has_existing_asset() -> None:
    """Every real overlay's image_path should resolve to a committed file."""
    overlays = load_map_overlays()
    real = [o for o in overlays if o.data_mode == "real"]
    assert real, "expected real Umbra overlays in the manifest"
    for o in real:
        path = REPO_ROOT / o.image_path
        assert path.exists(), (
            f"overlay {o.overlay_id} image_path {o.image_path!r} not found"
        )
        # PNGs should be modest in size (< 2 MB) so the dashboard load
        # stays snappy.
        assert path.stat().st_size < 2_000_000, (
            f"overlay {o.overlay_id} preview is {path.stat().st_size} bytes"
        )


def test_real_overlay_asset_url_starts_with_assets_overlays() -> None:
    overlays = load_map_overlays()
    real = [o for o in overlays if o.data_mode == "real"]
    assert real
    for o in real:
        assert (o.asset_url or "").startswith("/assets/overlays/")


def test_real_overlay_bounds_inside_aoi() -> None:
    """Real Umbra footprints should sit inside their scenario AOI bbox."""
    overlays = load_map_overlays()
    real = [o for o in overlays if o.data_mode == "real"]
    aoi_by_scenario = {
        "whitsun": (114.45, 9.75, 114.85, 10.25),
        "tennent": (114.55, 8.78, 114.75, 8.93),
    }
    for o in real:
        aoi = aoi_by_scenario[o.scenario_id]
        assert o.bounds is not None
        # Generous tolerance — footprints sometimes extend a hair past
        # the AOI corner depending on squint / range geometry.
        tol = 0.05
        assert o.bounds[0] >= aoi[0] - tol, o.overlay_id
        assert o.bounds[1] >= aoi[1] - tol, o.overlay_id
        assert o.bounds[2] <= aoi[2] + tol, o.overlay_id
        assert o.bounds[3] <= aoi[3] + tol, o.overlay_id


def test_whitsun_real_umbra_visible_from_event_2() -> None:
    overlays = load_map_overlays()
    whitsun_real = [
        o for o in overlays
        if o.scenario_id == "whitsun" and o.data_mode == "real"
        and o.source == "umbra"
    ]
    assert whitsun_real
    # The trace's Umbra observation is revealed at event 02; that's
    # when the real raster should appear on the map.
    assert any(o.visible_from_event_ordinal == 2 for o in whitsun_real)


def test_tennent_real_overlays_static() -> None:
    """Tennent Umbra overlays use ordinal 0 (always visible)."""
    overlays = load_map_overlays()
    tennent_real = [
        o for o in overlays
        if o.scenario_id == "tennent" and o.data_mode == "real"
    ]
    assert len(tennent_real) >= 5
    for o in tennent_real:
        assert o.visible_from_event_ordinal == 0


def test_simulated_followup_stays_footprint_only() -> None:
    """Honest scoping: the simulated Whitsun follow-up has no real
    raster, so it must not be promoted to data_mode 'real'."""
    overlays = load_map_overlays()
    fu = [
        o for o in overlays
        if o.overlay_id == "whitsun-umbra-followup-20231213"
    ]
    assert len(fu) == 1
    assert fu[0].data_mode == "simulated"
    assert fu[0].image_kind == "footprint-only"
    assert not has_image_asset(fu[0])


def test_sentinel_overlays_in_mixed_state() -> None:
    """The canonical manifest reflects a real Sentinel Hub run with
    quality gates applied:

      * Sentinel-2 entries with valid unique previews are promoted
        (image_kind=sentinel_preview, asset_url set).
      * Sentinel-1 entries kept footprint-only because Sentinel Hub
        returned empty placeholders for those acquisitions.
      * Sentinel-2 duplicates kept footprint-only with a clear
        "Duplicate preview of <obs>" reason.

    Every Sentinel overlay — promoted or footprint-only — preserves
    weak_signal=True / confirmation_layer=False.  The intent is to
    prove multi-source pipeline handling + graceful fallback, not
    coverage perfection."""
    overlays = load_map_overlays()
    sentinel = [
        o for o in overlays if o.source in ("sentinel-1", "sentinel-2")
    ]
    assert sentinel

    promoted = [o for o in sentinel if o.image_kind == "sentinel_preview"]
    footprint_only = [
        o for o in sentinel if o.image_kind == "footprint-only"
    ]
    # Mixed state: at least one of each, proving both pipelines.
    assert promoted, "expected at least one promoted Sentinel preview"
    assert footprint_only, (
        "expected at least one footprint-only Sentinel overlay"
    )

    # Promoted overlays must carry an asset url and have no
    # missing_asset_reason left over from their pre-fetch state.
    for o in promoted:
        assert has_image_asset(o), o.overlay_id
        assert o.asset_url and o.asset_url.startswith(
            "/assets/evidence/sentinel/"
        )
        assert not o.missing_asset_reason
        # Promoted Sentinel-2 only — Sentinel-1 returned empty
        # placeholders this run and was rejected by the quality gate.
        assert o.source == "sentinel-2", o.overlay_id

    # Footprint-only overlays must declare WHY (empty placeholder or
    # duplicate); the dashboard surfaces this verbatim.
    for o in footprint_only:
        assert not has_image_asset(o), o.overlay_id
        reason = o.missing_asset_reason or ""
        assert reason, o.overlay_id

    # Honesty markers preserved across both groups.
    for o in sentinel:
        assert o.weak_signal is True, o.overlay_id
        assert o.confirmation_layer is False, o.overlay_id


# ---------------------------------------------------------------------------
# BitmapLayer rendering
# ---------------------------------------------------------------------------


def test_deck_json_includes_bitmap_layer_for_real_overlays() -> None:
    """Tennent renders BitmapLayers from two distinct asset trees:
    Umbra GEC previews under ``/assets/overlays/`` and quality-gated
    Sentinel-2 Process API previews under
    ``/assets/evidence/sentinel/...``.  Both are valid raster
    sources; the test pins the union, not Umbra alone."""
    helpers = _import_helpers()
    overlays = load_map_overlays()
    avail = overlays_for_scenario(overlays, "tennent")
    spec = json.loads(helpers.build_deck_json(
        overlays=avail,
        layer_visibility=helpers.default_visibility(),
        opacity=0.8,
        center_lat=8.86, center_lon=114.66,
    ))
    bitmap_layers = [
        l for l in spec.get("layers", [])
        if l.get("@@type") == "BitmapLayer"
    ]
    assert bitmap_layers, "expected at least one BitmapLayer for Tennent"
    # At least one Umbra GEC preview is always present; that's the
    # tasked confirmation imagery the demo can't function without.
    umbra_bitmaps = [
        l for l in bitmap_layers
        if l.get("image", "").startswith("/assets/overlays/")
    ]
    assert umbra_bitmaps, (
        "expected at least one Umbra BitmapLayer under /assets/overlays/"
    )
    for l in bitmap_layers:
        url = l.get("image", "")
        assert (
            url.startswith("/assets/overlays/")
            or url.startswith("/assets/evidence/sentinel/")
        ), f"BitmapLayer image {url!r} not from a known asset tree"
        assert isinstance(l.get("bounds"), list)
        assert len(l["bounds"]) == 4


def test_umbra_overlay_omitted_drops_its_bitmap_layer() -> None:
    """The dynamic overlay manager controls visibility by filtering
    overlays before they reach build_deck_json; the resulting deck
    spec should never contain an Umbra BitmapLayer when no Umbra
    overlays are passed in.

    Sentinel-2 previews under /assets/evidence/sentinel/ remain
    valid raster sources independently and may still render — that's
    the multi-source pipeline working as designed.
    """
    helpers = _import_helpers()
    overlays = load_map_overlays()
    avail_no_umbra = tuple(
        o for o in overlays_for_scenario(overlays, "tennent")
        if o.source != "umbra"
    )
    spec = json.loads(helpers.build_deck_json(
        overlays=avail_no_umbra,
        base_visibility=helpers.default_base_visibility(),
        opacity=0.8,
        center_lat=8.86, center_lon=114.66,
    ))
    bitmap_layers = [
        l for l in spec.get("layers", [])
        if l.get("@@type") == "BitmapLayer"
    ]
    umbra_bitmaps = [
        l for l in bitmap_layers
        if l.get("image", "").startswith("/assets/overlays/")
    ]
    assert not umbra_bitmaps, (
        "Umbra BitmapLayers should be absent when no Umbra overlays "
        "are passed"
    )


def test_whitsun_event_1_has_no_bitmap_layer() -> None:
    """At event 01 only the AOI is revealed; no Umbra raster yet."""
    helpers = _import_helpers()
    overlays = load_map_overlays()
    avail = available_overlays_for(
        overlays, scenario_id="whitsun", current_ordinal=1,
    )
    spec = json.loads(helpers.build_deck_json(
        overlays=avail,
        layer_visibility=helpers.default_visibility(),
        opacity=0.8,
        center_lat=9.98, center_lon=114.63,
    ))
    bitmap_layers = [
        l for l in spec.get("layers", [])
        if l.get("@@type") == "BitmapLayer"
    ]
    assert not bitmap_layers


def test_whitsun_event_2_has_real_umbra_bitmap_layer() -> None:
    helpers = _import_helpers()
    overlays = load_map_overlays()
    avail = available_overlays_for(
        overlays, scenario_id="whitsun", current_ordinal=2,
    )
    spec = json.loads(helpers.build_deck_json(
        overlays=avail,
        layer_visibility=helpers.default_visibility(),
        opacity=0.8,
        center_lat=9.98, center_lon=114.63,
    ))
    bitmap_layers = [
        l for l in spec.get("layers", [])
        if l.get("@@type") == "BitmapLayer"
    ]
    assert bitmap_layers, (
        "expected the real Umbra BitmapLayer to appear at event 02"
    )


def test_aoi_overlays_dont_need_missing_reason() -> None:
    overlays = load_map_overlays()
    aois = [o for o in overlays if o.observation_id is None]
    for o in aois:
        # AOIs are inherently footprint-only by design — no reason needed.
        assert o.image_kind == "footprint-only"


# ---------------------------------------------------------------------------
# Deck JSON helper produces parseable output
# ---------------------------------------------------------------------------


def _import_helpers():
    import sys
    src_app = REPO_ROOT / "src" / "app"
    if str(src_app) not in sys.path:
        sys.path.insert(0, str(src_app))
    for k in list(sys.modules):
        if k.startswith("layout."):
            del sys.modules[k]
    import layout.map_overlays_helpers as mod
    return mod


def test_build_deck_json_returns_parseable_json() -> None:
    helpers = _import_helpers()
    overlays = load_map_overlays()
    avail = overlays_for_scenario(overlays, "tennent")
    out = helpers.build_deck_json(
        overlays=avail,
        layer_visibility=helpers.default_visibility(),
        opacity=0.6,
        center_lat=8.86, center_lon=114.66,
    )
    parsed = json.loads(out)
    assert "layers" in parsed
    assert isinstance(parsed["layers"], list)


def test_caller_owns_per_source_filtering() -> None:
    """Per-source filtering moved out of build_deck_json; the caller
    now passes the already-selected overlay list.  Confirm Sentinel
    footprints are absent when their overlays are filtered out at the
    call site."""
    helpers = _import_helpers()
    overlays = load_map_overlays()
    avail = tuple(
        o for o in overlays_for_scenario(overlays, "tennent")
        if o.source not in ("sentinel-1", "sentinel-2")
    )
    blob = helpers.build_deck_json(
        overlays=avail,
        base_visibility=helpers.default_base_visibility(),
        opacity=0.6,
        center_lat=8.86, center_lon=114.66,
    )
    serialized = blob
    assert "tennent-sentinel-1" not in serialized
    assert "tennent-sentinel-2" not in serialized


def test_umbra_previews_are_neutral_grayscale_rgba() -> None:
    """Umbra preview PNGs must be perfectly grayscale (R=G=B) with a
    sane alpha channel: nodata pixels alpha=0 so the dark basemap
    shows through cleanly, valid pixels alpha=255 so the SAR isn't
    blended with the basemap into a teal cast."""
    pytest.importorskip("PIL")
    import numpy as np
    from PIL import Image

    overlay_dir = REPO_ROOT / "src" / "app" / "assets" / "overlays"
    pngs = sorted(overlay_dir.glob("*.png"))
    assert pngs, f"no Umbra preview PNGs found under {overlay_dir}"

    for png in pngs:
        img = np.array(Image.open(png))
        assert img.ndim == 3 and img.shape[2] == 4, (
            f"{png.name} must be RGBA, got shape {img.shape}"
        )
        rgb = img[..., :3].astype(int)
        alpha = img[..., 3]

        # Every alpha is either 0 (nodata, fully transparent) or 255
        # (valid SAR, fully opaque).  Anything in between would suggest
        # the BitmapLayer is partially blending with the basemap and
        # would re-introduce the cyan/teal cast we are guarding
        # against.
        unique_alphas = set(np.unique(alpha).tolist())
        assert unique_alphas <= {0, 255}, (
            f"{png.name} carries non-binary alpha {unique_alphas}; "
            f"expected only 0 / 255"
        )
        assert 0 in unique_alphas, (
            f"{png.name} has no transparent pixels; nodata mask is "
            f"not being applied"
        )

        valid = alpha == 255
        assert valid.any(), f"{png.name} has no valid (alpha=255) pixels"
        valid_rgb = rgb[valid]
        r = valid_rgb[:, 0]
        g = valid_rgb[:, 1]
        b = valid_rgb[:, 2]
        # Strict R=G=B for every valid pixel — the preview is
        # generated with a single grayscale value broadcast into all
        # three channels, so any deviation would mean a colour
        # transform was applied somewhere.
        assert np.array_equal(r, g), (
            f"{png.name} R != G in valid pixels — preview is not "
            f"neutral grayscale"
        )
        assert np.array_equal(g, b), (
            f"{png.name} G != B in valid pixels — preview is not "
            f"neutral grayscale"
        )
        # And the median |R-G|, |G-B| metrics the dispatch asks for.
        assert int(np.median(np.abs(r - g))) == 0
        assert int(np.median(np.abs(g - b))) == 0


def test_aoi_default_style_is_outline_only_with_zero_fill() -> None:
    """The AOI polygon must default to outline-only so it never washes
    out the imagery overlays.  ``filled=False`` is the primary guard;
    a fully-transparent ``getFillColor`` is the belt-and-suspenders
    backstop in case ``filled=False`` is regressed in pydeck."""
    helpers = _import_helpers()
    overlays = load_map_overlays()
    avail = overlays_for_scenario(overlays, "tennent")
    spec = json.loads(helpers.build_deck_json(
        overlays=avail,
        base_visibility=helpers.default_base_visibility(),
        opacity=0.6,
        center_lat=8.86, center_lon=114.66,
    ))
    polygon_layers = [
        l for l in spec.get("layers", [])
        if l.get("@@type") == "PolygonLayer"
    ]
    # The AOI layer is the first PolygonLayer (it sits beneath the
    # imagery in the z-order).
    aoi_layer = polygon_layers[0]
    assert aoi_layer.get("filled") is False, (
        "AOI must be outline-only by default"
    )
    fill = aoi_layer.get("getFillColor")
    assert isinstance(fill, list) and len(fill) == 4, fill
    # Either fully transparent (alpha 0) or, if a fallback fill is
    # ever introduced, no greater than the documented 0.05 cap.
    alpha_max = helpers._AOI_MAX_FALLBACK_FILL_ALPHA
    assert fill[3] == 0 or fill[3] <= alpha_max, (
        f"AOI default fill alpha must be 0 (outline-only) or "
        f"<= {alpha_max} (very-low fallback); got {fill[3]}"
    )
    # Outline must remain readable but not heavy.
    line_min = aoi_layer.get("lineWidthMinPixels")
    assert line_min is not None and line_min <= 2, line_min


def test_aoi_renders_below_imagery_overlays() -> None:
    """Z-order must put AOI beneath BitmapLayers so the imagery sits
    on top, with footprints / tracks / custody markers above the
    imagery."""
    helpers = _import_helpers()
    overlays = load_map_overlays()
    avail = overlays_for_scenario(overlays, "tennent")
    spec = json.loads(helpers.build_deck_json(
        overlays=avail,
        base_visibility=helpers.default_base_visibility(),
        opacity=0.6,
        center_lat=8.86, center_lon=114.66,
    ))
    layer_types = [l.get("@@type") for l in spec.get("layers", [])]
    # First Polygon (AOI) before any Bitmap.
    first_polygon = layer_types.index("PolygonLayer")
    first_bitmap = (
        layer_types.index("BitmapLayer")
        if "BitmapLayer" in layer_types else None
    )
    assert first_bitmap is not None, "expected at least one BitmapLayer"
    assert first_polygon < first_bitmap, (
        f"AOI polygon must render below imagery; "
        f"got AOI at {first_polygon} and first bitmap at {first_bitmap}"
    )
    # Footprint polygon (last PolygonLayer) sits above all bitmaps.
    last_bitmap = max(
        i for i, t in enumerate(layer_types) if t == "BitmapLayer"
    )
    last_polygon = max(
        i for i, t in enumerate(layer_types) if t == "PolygonLayer"
    )
    assert last_polygon > last_bitmap, (
        "footprint outlines must render on top of imagery"
    )


def test_aoi_toggle_off_drops_aoi_layer() -> None:
    """The AOI base toggle must still be honoured even though the
    layer renders outline-only by default."""
    helpers = _import_helpers()
    overlays = load_map_overlays()
    avail = overlays_for_scenario(overlays, "tennent")
    visibility = helpers.default_base_visibility()
    visibility["aoi"] = False
    spec = json.loads(helpers.build_deck_json(
        overlays=avail,
        base_visibility=visibility,
        opacity=0.6,
        center_lat=8.86, center_lon=114.66,
    ))
    layer_types = [l.get("@@type") for l in spec.get("layers", [])]
    # When AOI is off, only the footprint PolygonLayer (above imagery)
    # remains.  Confirm that PolygonLayer comes AFTER any BitmapLayer.
    if "BitmapLayer" in layer_types and "PolygonLayer" in layer_types:
        first_polygon = layer_types.index("PolygonLayer")
        first_bitmap = layer_types.index("BitmapLayer")
        assert first_polygon > first_bitmap, (
            "with AOI off, the only PolygonLayer is the footprint, "
            "which must sit above the imagery"
        )


def test_aoi_layer_present_for_whitsun_at_event_one() -> None:
    """Whitsun event 01 reveals only the AOI; even with no observation
    overlays, the AOI outline must render."""
    helpers = _import_helpers()
    overlays = load_map_overlays()
    avail = available_overlays_for(
        overlays, scenario_id="whitsun", current_ordinal=1,
    )
    spec = json.loads(helpers.build_deck_json(
        overlays=avail,
        base_visibility=helpers.default_base_visibility(),
        opacity=0.6,
        center_lat=9.98, center_lon=114.63,
    ))
    polygon_layers = [
        l for l in spec.get("layers", [])
        if l.get("@@type") == "PolygonLayer"
    ]
    assert polygon_layers, "AOI PolygonLayer must render at event 01"
    assert polygon_layers[0].get("filled") is False


def test_base_visibility_from_checked_excludes_unchecked_keys() -> None:
    helpers = _import_helpers()
    vis = helpers.base_visibility_from_checked(["aoi"])
    assert vis["aoi"] is True
    assert vis["tracks"] is False
    assert vis["custody"] is False
    # Old per-source keys are no longer part of the base toggle set.
    assert "umbra" not in vis
    assert "sentinel_1" not in vis
    assert "footprints" not in vis


# ---------------------------------------------------------------------------
# Dash layouts mount the map IDs
# ---------------------------------------------------------------------------


def _import_whitsun_layout():
    import sys
    src_app = REPO_ROOT / "src" / "app"
    if str(src_app) not in sys.path:
        sys.path.insert(0, str(src_app))
    for k in list(sys.modules):
        if k.startswith("layout.") or k.startswith("callbacks."):
            del sys.modules[k]
    import layout.whitsun_replay as mod
    return mod


def _import_tennent_layout():
    import sys
    src_app = REPO_ROOT / "src" / "app"
    if str(src_app) not in sys.path:
        sys.path.insert(0, str(src_app))
    for k in list(sys.modules):
        if k.startswith("layout.") or k.startswith("callbacks."):
            del sys.modules[k]
    import layout.tennent_monitoring as mod
    return mod


def test_whitsun_map_ids_present_in_layout() -> None:
    mod = _import_whitsun_layout()
    rendered = str(mod.build_whitsun_replay_layout())
    assert "whitsun-replay-map-deck" in rendered
    assert "whitsun-replay-map-layers" in rendered
    assert "whitsun-replay-map-opacity" in rendered
    assert "whitsun-replay-map-overlay-badges" in rendered
    assert "whitsun-replay-map-missing-imagery" in rendered


def test_tennent_map_ids_present_in_layout() -> None:
    mod = _import_tennent_layout()
    rendered = str(mod.build_tennent_monitoring_layout())
    assert "tennent-monitoring-map-deck" in rendered
    assert "tennent-monitoring-map-layers" in rendered
    assert "tennent-monitoring-map-opacity" in rendered
    assert "tennent-monitoring-map-overlay-badges" in rendered
    assert "tennent-monitoring-map-missing-imagery" in rendered


def test_full_dash_app_registers_both_map_callbacks() -> None:
    import sys
    src_app = REPO_ROOT / "src" / "app"
    if str(src_app) not in sys.path:
        sys.path.insert(0, str(src_app))
    for k in list(sys.modules):
        if k == "dash_app" or k.startswith("layout.") or k.startswith("callbacks."):
            del sys.modules[k]
    import dash_app
    ids = list(dash_app.app.callback_map.keys())
    assert any("whitsun-replay-map-deck" in cid for cid in ids)
    assert any("tennent-monitoring-map-deck" in cid for cid in ids)
