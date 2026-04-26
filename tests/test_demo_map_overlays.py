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


def test_whitsun_sentinel_2_overlays_appear_at_ordinal_4() -> None:
    overlays = load_map_overlays()
    avail = available_overlays_for(
        overlays, scenario_id="whitsun", current_ordinal=4,
    )
    ids = _ids(avail)
    assert "whitsun-sentinel-2-low-cloud-20231212" in ids
    assert "whitsun-sentinel-2-cloudy-20231215" in ids
    # Sentinel-1 still gated until ordinal 5.
    assert "whitsun-sentinel-1-grd-20231210" not in ids


def test_whitsun_sentinel_1_overlay_appears_at_ordinal_5() -> None:
    overlays = load_map_overlays()
    avail = available_overlays_for(
        overlays, scenario_id="whitsun", current_ordinal=5,
    )
    assert "whitsun-sentinel-1-grd-20231210" in _ids(avail)


def test_whitsun_followup_overlay_appears_at_ordinal_12() -> None:
    overlays = load_map_overlays()
    avail = available_overlays_for(
        overlays, scenario_id="whitsun", current_ordinal=12,
    )
    assert "whitsun-umbra-followup-20231213" in _ids(avail)


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


def test_sentinel_overlays_stay_footprint_only() -> None:
    """No Sentinel imagery is committed yet — every Sentinel entry
    must remain footprint-only with an explicit missing_asset_reason."""
    overlays = load_map_overlays()
    sentinel = [o for o in overlays if o.source in ("sentinel-1", "sentinel-2")]
    assert sentinel
    for o in sentinel:
        assert o.image_kind == "footprint-only"
        assert not has_image_asset(o)
        assert o.missing_asset_reason


# ---------------------------------------------------------------------------
# BitmapLayer rendering
# ---------------------------------------------------------------------------


def test_deck_json_includes_bitmap_layer_for_real_overlays() -> None:
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
    for l in bitmap_layers:
        assert l.get("image", "").startswith("/assets/overlays/")
        assert isinstance(l.get("bounds"), list)
        assert len(l["bounds"]) == 4


def test_umbra_toggle_hides_real_bitmap_layers() -> None:
    helpers = _import_helpers()
    overlays = load_map_overlays()
    avail = overlays_for_scenario(overlays, "tennent")
    visibility = helpers.default_visibility()
    visibility["umbra"] = False
    spec = json.loads(helpers.build_deck_json(
        overlays=avail,
        layer_visibility=visibility,
        opacity=0.8,
        center_lat=8.86, center_lon=114.66,
    ))
    bitmap_layers = [
        l for l in spec.get("layers", [])
        if l.get("@@type") == "BitmapLayer"
    ]
    assert not bitmap_layers, (
        "BitmapLayers should be hidden when the Umbra toggle is off"
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


def test_layer_visibility_filters_per_source() -> None:
    helpers = _import_helpers()
    overlays = load_map_overlays()
    avail = overlays_for_scenario(overlays, "tennent")
    visibility = helpers.default_visibility()
    visibility["sentinel_1"] = False
    visibility["sentinel_2"] = False
    blob = helpers.build_deck_json(
        overlays=avail,
        layer_visibility=visibility,
        opacity=0.6,
        center_lat=8.86, center_lon=114.66,
    )
    parsed = json.loads(blob)
    # PolygonLayer for footprints should still exist (AOI), but
    # Sentinel-1 / Sentinel-2 footprints filtered out.
    serialized = json.dumps(parsed)
    assert "tennent-sentinel-1" not in serialized
    assert "tennent-sentinel-2" not in serialized


def test_visibility_from_checked_excludes_unchecked_keys() -> None:
    helpers = _import_helpers()
    vis = helpers.visibility_from_checked(["aoi", "footprints"])
    assert vis["aoi"] is True
    assert vis["footprints"] is True
    assert vis["umbra"] is False
    assert vis["sentinel_1"] is False
    assert vis["tracks"] is False


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
