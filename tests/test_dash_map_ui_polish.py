"""Tests for the focused copy/UX polish pass on the map + overlay UI.

The pass replaces stale "no georeferenced imagery" copy, rebrands
Sentinel detection labels to weak-signal cueing terms, switches the
overlay badges to audience-facing prose, raises the Whitsun map to
primary-evidence height, defaults Tennent to a baseline-only Umbra
date set, and adds a legend on both tabs.

No live HTTP, no Dash server is started.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _import_layout(module_name: str):
    import sys
    src_app = REPO_ROOT / "src" / "app"
    if str(src_app) not in sys.path:
        sys.path.insert(0, str(src_app))
    for k in list(sys.modules):
        if k.startswith("layout.") or k.startswith("callbacks."):
            del sys.modules[k]
    import importlib
    return importlib.import_module(module_name)


def _import_callbacks(module_name: str):
    return _import_layout(module_name)


def _height_px(rendered: str, expected_id: str) -> int | None:
    """Find the inline ``height: NNNpx`` on the component carrying ``id=expected_id``."""
    pattern = re.compile(
        r"id=['\"]" + re.escape(expected_id) + r"['\"]"
        r"[^)]*?height['\"]?\s*[:=]\s*['\"]?(\d+)px",
        re.IGNORECASE,
    )
    m = pattern.search(rendered)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Item 1: stale "no georeferenced imagery" copy is gone; new copy present
# ---------------------------------------------------------------------------


_NEW_MAP_INTRO_FRAGMENT = (
    "umbra sar preview overlays are generated from "
    "committed gec scenes"
)


def test_whitsun_map_intro_uses_new_copy() -> None:
    mod = _import_layout("layout.whitsun_replay")
    rendered = str(mod.build_whitsun_replay_layout()).lower()
    assert _NEW_MAP_INTRO_FRAGMENT in rendered
    # Stale copy must be gone.
    assert "no georeferenced imagery is committed yet" not in rendered


def test_tennent_map_intro_uses_new_copy() -> None:
    mod = _import_layout("layout.tennent_monitoring")
    rendered = str(mod.build_tennent_monitoring_layout()).lower()
    assert _NEW_MAP_INTRO_FRAGMENT in rendered
    assert "no georeferenced imagery is committed yet" not in rendered


# ---------------------------------------------------------------------------
# Item 2: Sentinel "for detection" / "for context" relabeled
# ---------------------------------------------------------------------------


def test_tennent_observations_table_uses_cue_usable_columns() -> None:
    mod = _import_layout("layout.tennent_monitoring")
    rendered = str(mod.build_tennent_monitoring_layout()).lower()
    assert "cue usable" in rendered
    assert "context usable" in rendered
    # Old labels must be gone.
    assert ">for detection<" not in rendered
    assert ">for context<" not in rendered


def test_whitsun_observations_panel_uses_cue_usable_labels() -> None:
    rep = _import_callbacks("callbacks.whitsun_replay")
    # Step where Sentinel is revealed (ordinal 5).
    rendered = str(rep._render_observations(rep._TRACE, 5)).lower()
    assert "cue usable" in rendered
    assert "context usable" in rendered
    # The bare snake_case labels should not surface in the rendered tree.
    assert "usable_for_detection" not in rendered
    assert "usable_for_context" not in rendered


# ---------------------------------------------------------------------------
# Item 3: overlay badges use audience-facing prose
# ---------------------------------------------------------------------------


def test_whitsun_overlay_badges_use_image_overlay_label() -> None:
    rep = _import_callbacks("callbacks.whitsun_replay")
    from custody.demo import available_overlays_for
    from custody.demo.map_overlays import is_observation_overlay
    overlays = tuple(
        o for o in available_overlays_for(
            rep._OVERLAYS, scenario_id="whitsun", current_ordinal=14,
        )
        if is_observation_overlay(o)
    )
    rendered = str(rep._build_overlay_badges(overlays)).lower()
    # New format markers.
    assert "image overlay" in rendered
    assert "confidence" in rendered
    # No "png w=1.00"-style strings.
    assert " w=" not in rendered
    assert "png w=" not in rendered


def test_whitsun_overlay_badges_use_weak_signal_label() -> None:
    rep = _import_callbacks("callbacks.whitsun_replay")
    from custody.demo import available_overlays_for
    from custody.demo.map_overlays import is_observation_overlay
    overlays = tuple(
        o for o in available_overlays_for(
            rep._OVERLAYS, scenario_id="whitsun", current_ordinal=14,
        )
        if is_observation_overlay(o)
    )
    rendered = str(rep._build_overlay_badges(overlays)).lower()
    # Sentinel observations carry the weak-signal label (image/footprint
    # tag varies depending on whether a preview is committed).
    assert "weak-signal" in rendered


def test_tennent_overlay_badges_use_audience_facing_prose() -> None:
    cb = _import_callbacks("callbacks.tennent_monitoring")
    from custody.demo.map_overlays import is_observation_overlay
    obs_overlays = tuple(
        o for o in cb._TENNENT_OVERLAYS if is_observation_overlay(o)
    )
    rendered = str(cb._build_overlay_badges(obs_overlays)).lower()
    assert "image overlay" in rendered  # Umbra GEC png overlays
    assert "weak-signal" in rendered  # Sentinel
    assert "confidence" in rendered
    assert " w=" not in rendered


# ---------------------------------------------------------------------------
# Item 4: Whitsun map is at least 500px tall
# ---------------------------------------------------------------------------


def test_whitsun_map_renders_at_primary_evidence_height() -> None:
    mod = _import_layout("layout.whitsun_replay")
    rendered = str(mod.build_whitsun_replay_layout())
    h = _height_px(rendered, "whitsun-replay-map-deck")
    assert h is not None, "Whitsun deck height not found in layout"
    assert h >= 500, f"Whitsun map height {h}px < 500px (must be primary evidence)"


# ---------------------------------------------------------------------------
# Item 5: Tennent default Umbra-date toggles
# ---------------------------------------------------------------------------


def _walk_checklist(component, checklist_id: str):
    from dash import dcc
    found = None

    def _walk(c):
        nonlocal found
        if found is not None:
            return
        if isinstance(c, dcc.Checklist) and getattr(c, "id", None) == checklist_id:
            found = c
            return
        children = getattr(c, "children", None)
        if children is None:
            return
        if isinstance(children, list):
            for ch in children:
                _walk(ch)
        else:
            _walk(children)

    _walk(component)
    return found


def test_tennent_default_overlay_selection_is_baseline_umbra_only() -> None:
    mod = _import_layout("layout.tennent_monitoring")
    layout = mod.build_tennent_monitoring_layout()
    cl = _walk_checklist(layout, mod.TENNENT_OVERLAY_TOGGLES)
    assert cl is not None, "Tennent layout must include the per-overlay Checklist"
    value = list(cl.value or [])
    assert value == ["tennent-umbra-20230702"], (
        f"default per-overlay selection must be baseline Umbra only, got {value}"
    )
    # Every Tennent observation overlay should appear as an option, even
    # those off-by-default (operator can opt them in).
    option_values = {opt["value"] for opt in (cl.options or [])}
    expected = {
        "tennent-umbra-20230702", "tennent-umbra-20230723",
        "tennent-umbra-20230807", "tennent-umbra-20230809",
        "tennent-umbra-20230813",
        "tennent-sentinel-1-grd-20230715",
        "tennent-sentinel-2-low-cloud-20230718",
        "tennent-sentinel-2-cloudy-20230728",
    }
    assert expected <= option_values, (
        f"expected per-overlay options for every Tennent observation; "
        f"missing {expected - option_values}"
    )


def test_whitsun_overlay_options_are_event_aware() -> None:
    """The dynamic per-overlay manager must only list overlays whose
    visible_from_event_ordinal has been reached."""
    rep = _import_callbacks("callbacks.whitsun_replay")
    from custody.demo import available_overlays_for
    from custody.demo.map_overlays import is_observation_overlay

    def _option_ids_at_ordinal(ord_: int) -> set[str]:
        avail = [
            o for o in available_overlays_for(
                rep._OVERLAYS, scenario_id="whitsun", current_ordinal=ord_,
            )
            if is_observation_overlay(o)
        ]
        from layout.whitsun_replay import _overlay_toggle_options
        return {opt["value"] for opt in _overlay_toggle_options(avail)}

    # Event 01: only AOI is revealed; no observation overlays yet.
    assert _option_ids_at_ordinal(1) == set()
    # Event 02: Umbra primary collect.
    ord2 = _option_ids_at_ordinal(2)
    assert any("umbra" in oid and "20231206" in oid for oid in ord2), ord2
    assert not any("sentinel" in oid for oid in ord2)
    # Event 04: Sentinel-2 weak-signal layer arrives.
    ord4 = _option_ids_at_ordinal(4)
    assert any("sentinel-2" in oid for oid in ord4), ord4
    assert not any("sentinel-1" in oid for oid in ord4), (
        f"Sentinel-1 should not appear before event 5; got {ord4}"
    )
    # Event 05: Sentinel-1 weak-signal layer arrives.
    ord5 = _option_ids_at_ordinal(5)
    assert any("sentinel-1" in oid for oid in ord5), ord5
    # Event 11: follow-up Umbra (ordinal 12) must NOT appear early.
    ord11 = _option_ids_at_ordinal(11)
    assert not any(
        "umbra-followup-20231213" in oid for oid in ord11
    ), f"future follow-up Umbra leaked at ord 11; got {ord11}"
    # Event 12: follow-up Umbra appears.
    ord12 = _option_ids_at_ordinal(12)
    assert any("umbra-followup-20231213" in oid for oid in ord12), ord12


def test_default_visible_drives_initial_selection() -> None:
    """Default-visible overlays must be on at first render; non-default
    overlays must be off until the user toggles them on."""
    from custody.demo import load_map_overlays
    from custody.demo.map_overlays import is_observation_overlay
    overlays = load_map_overlays()
    obs = [o for o in overlays if is_observation_overlay(o)]
    # Tennent baseline must be on; cloudy / non-baseline / late dates off.
    by_id = {o.overlay_id: o for o in obs}
    assert by_id["tennent-umbra-20230702"].default_visible is True
    assert by_id["tennent-umbra-20230813"].default_visible is False
    assert by_id["tennent-sentinel-2-cloudy-20230728"].default_visible is False
    # Whitsun primary Umbra and weak-signal Sentinel layers are on by
    # default (they ARE the demo).  The cloudy Sentinel-2 is off.
    assert by_id["whitsun-umbra-20231206"].default_visible is True
    assert by_id["whitsun-sentinel-2-low-cloud-20231212"].default_visible is True
    assert by_id["whitsun-sentinel-1-grd-20231210"].default_visible is True
    assert by_id["whitsun-sentinel-2-cloudy-20231215"].default_visible is False


def test_overlay_label_format_matches_dispatch_examples() -> None:
    """Per-image labels must follow the dispatch examples."""
    from custody.demo.map_overlays import format_overlay_label, load_map_overlays
    by_id = {o.overlay_id: o for o in load_map_overlays()}
    umbra_label = format_overlay_label(by_id["whitsun-umbra-20231206"])
    # "Umbra SAR 2023-12-06 · 0.25 m · image overlay · confidence 1.00"
    assert "Umbra SAR" in umbra_label
    assert "2023-12-06" in umbra_label
    assert "image overlay" in umbra_label
    assert "confidence 1.00" in umbra_label
    s2_label = format_overlay_label(
        by_id["whitsun-sentinel-2-low-cloud-20231212"],
    )
    # "Sentinel-2 2023-12-12 · 8% cloud · weak-signal image/footprint · confidence 0.30"
    assert "Sentinel-2" in s2_label
    assert "2023-12-12" in s2_label
    assert "8% cloud" in s2_label
    assert "weak-signal" in s2_label
    assert "confidence 0.30" in s2_label


def test_footprint_only_overlays_carry_clear_kind_label() -> None:
    """Sentinel overlays without preview imagery must clearly identify
    as footprint-only (not as image overlays)."""
    from custody.demo.map_overlays import (
        format_overlay_label,
        has_image_asset,
        load_map_overlays,
    )
    s1_overlay = next(
        o for o in load_map_overlays()
        if o.overlay_id == "tennent-sentinel-1-grd-20230715"
    )
    assert not has_image_asset(s1_overlay)
    label = format_overlay_label(s1_overlay)
    assert "footprint" in label.lower()
    assert "image overlay" not in label  # full image-overlay language reserved


def test_tennent_base_layer_default_is_aoi_only() -> None:
    mod = _import_layout("layout.tennent_monitoring")
    layout = mod.build_tennent_monitoring_layout()
    cl = _walk_checklist(layout, mod.TENNENT_MAP_LAYER_TOGGLES)
    assert cl is not None
    value = list(cl.value or [])
    assert value == ["aoi"], (
        f"base toggles must default to AOI only on Tennent, got {value}"
    )
    # The legacy per-source / footprints switches must be gone.
    option_values = {opt["value"] for opt in (cl.options or [])}
    for legacy in ("umbra", "sentinel_1", "sentinel_2", "footprints"):
        assert legacy not in option_values


# ---------------------------------------------------------------------------
# Item 6: legend present on both tabs with the three required lines
# ---------------------------------------------------------------------------


_LEGEND_LINES = (
    "umbra sar — high-confidence confirmation imagery",
    "sentinel — weak-signal cueing / context",
    "footprint-only — no image preview currently loaded",
)


def test_whitsun_map_legend_present() -> None:
    mod = _import_layout("layout.whitsun_replay")
    rendered = str(mod.build_whitsun_replay_layout()).lower()
    for line in _LEGEND_LINES:
        assert line in rendered, f"Whitsun legend missing line: {line!r}"


def test_tennent_map_legend_present() -> None:
    mod = _import_layout("layout.tennent_monitoring")
    rendered = str(mod.build_tennent_monitoring_layout()).lower()
    for line in _LEGEND_LINES:
        assert line in rendered, f"Tennent legend missing line: {line!r}"
