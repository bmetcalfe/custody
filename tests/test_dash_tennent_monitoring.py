"""Tests for the Tennent monitoring tab.

Covers the layout factory, the sidebar block, the three-way sidebar
swap, and the third tab in the full Dash app.  No live HTTP, no Dash
server is started.
"""
from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _import_layout_module():
    import sys
    src_app = REPO_ROOT / "src" / "app"
    if str(src_app) not in sys.path:
        sys.path.insert(0, str(src_app))
    for k in list(sys.modules):
        if k.startswith("layout."):
            del sys.modules[k]
    import layout.tennent_monitoring as mod
    return mod


def _import_dash_app():
    import sys
    src_app = REPO_ROOT / "src" / "app"
    if str(src_app) not in sys.path:
        sys.path.insert(0, str(src_app))
    for k in list(sys.modules):
        if k == "dash_app" or k.startswith("layout.") or k.startswith("callbacks."):
            del sys.modules[k]
    import dash_app
    return dash_app


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


# ---------------------------------------------------------------------------
# Sidebar block
# ---------------------------------------------------------------------------


def test_tennent_sidebar_block_factory_exists() -> None:
    mod = _import_layout_module()
    block = mod.build_tennent_sidebar_block()
    rendered = str(block)
    assert "Tennent monitoring" in rendered
    assert "read-only fixture" in rendered
    assert "fixed-site monitoring scenario" in rendered
    assert "Sentinel context only" in rendered
    assert "no live inference" in rendered
    assert "no live change detection" in rendered


# ---------------------------------------------------------------------------
# Layout factory
# ---------------------------------------------------------------------------


def test_tennent_monitoring_layout_loads_offline() -> None:
    """Layout factory must read both committed Tennent fixtures cleanly."""
    mod = _import_layout_module()
    layout = mod.build_tennent_monitoring_layout()
    assert layout is not None
    rendered = str(layout)
    # Header strings.
    assert "Tennent Reef" in rendered
    assert "fixed-site monitoring" in rendered
    assert "not a live change-detection pipeline" in rendered.lower()


def test_layout_includes_aoi_panel_with_required_fields() -> None:
    mod = _import_layout_module()
    rendered = str(mod.build_tennent_monitoring_layout())
    assert "AOI / context" in rendered
    assert "scenario_type" in rendered
    assert "fixed_site_monitoring" in rendered
    assert "data_mode" in rendered
    # Tennent-published center coordinates from the AOI metadata.
    assert "8.8583" in rendered
    assert "114.6561" in rendered
    assert "placeholder" in rendered.lower()
    assert "replace before operational use" in rendered.lower()


def test_layout_includes_three_sentinel_observations() -> None:
    mod = _import_layout_module()
    rendered = str(mod.build_tennent_monitoring_layout())
    # All three fixture observation IDs surface in the table / detail rows.
    assert "fixture-s1-grd-tennent-20230715" in rendered
    assert "fixture-s2-l2a-tennent-20230718-low-cloud" in rendered
    assert "fixture-s2-l2a-tennent-20230728-cloudy" in rendered


def test_layout_distinguishes_low_cloud_and_cloudy_s2() -> None:
    mod = _import_layout_module()
    rendered = str(mod.build_tennent_monitoring_layout())
    # Cloud cover values must appear so the operator can distinguish.
    assert "12" in rendered  # low-cloud %
    assert "85" in rendered  # cloudy %


def test_layout_includes_interpretation_panel() -> None:
    mod = _import_layout_module()
    rendered = str(mod.build_tennent_monitoring_layout())
    assert "Site-monitoring interpretation" in rendered
    # Sentinel weak-signal framing (see test_sentinel_framing.py for the
    # full disclaimer text).
    assert "weak-signal cueing" in rendered or "weak-signal cue" in rendered
    assert "low-confidence temporal context" in rendered
    assert "context-only" in rendered or "context only" in rendered
    assert (
        "different mission archetype" in rendered
        or "schema works" in rendered
    )


def test_layout_includes_not_yet_implemented_panel() -> None:
    mod = _import_layout_module()
    rendered = str(mod.build_tennent_monitoring_layout())
    assert "Not yet implemented" in rendered
    assert "no live imagery rendering" in rendered
    assert "no site-change classifier" in rendered
    assert "no construction-change detection" in rendered
    assert "no policy / tasking decision loop" in rendered.lower() or \
        "no policy/tasking decision loop" in rendered.lower()
    assert "no operator approval workflow" in rendered


# ---------------------------------------------------------------------------
# Three-tab integration in the full Dash app
# ---------------------------------------------------------------------------


def test_dash_app_has_three_main_tabs() -> None:
    dash_app = _import_dash_app()
    rendered = str(dash_app.app.layout)
    assert "Custody overview" in rendered
    assert "Whitsun replay (fixture)" in rendered
    assert "Tennent monitoring (fixture)" in rendered


def test_dash_app_has_three_sidebar_swap_targets() -> None:
    dash_app = _import_dash_app()
    ids = _walk_ids(dash_app.app.layout)
    assert "custody-main-sidebar-overview" in ids
    assert "custody-main-sidebar-whitsun" in ids
    assert "custody-main-sidebar-tennent" in ids


def test_dash_app_has_tennent_root_id() -> None:
    dash_app = _import_dash_app()
    ids = _walk_ids(dash_app.app.layout)
    assert "tennent-monitoring-root" in ids


# ---------------------------------------------------------------------------
# Sidebar swap callback supports all three tabs
# ---------------------------------------------------------------------------


def test_sidebar_callback_outputs_cover_all_three_sidebars() -> None:
    dash_app = _import_dash_app()
    sidebar_keys = [
        cid for cid in dash_app.app.callback_map
        if "custody-main-sidebar-overview" in cid
        and "custody-main-sidebar-whitsun" in cid
        and "custody-main-sidebar-tennent" in cid
    ]
    assert sidebar_keys, (
        "expected a single callback whose outputs cover all three "
        "sidebar swap targets"
    )


def test_sidebar_callback_unchanged_count() -> None:
    """Tab-swap is one callback (extended outputs); collapse adds two;
    map overlays add two more (Whitsun + Tennent); the dynamic overlay
    manager on Whitsun adds one; the Evidence Viewer adds two more
    (Whitsun + Tennent); the Sentinel cueing context adds two more
    (Whitsun + Tennent)."""
    dash_app = _import_dash_app()
    assert len(dash_app.app.callback_map) == 23


# ---------------------------------------------------------------------------
# Existing Whitsun replay still works
# ---------------------------------------------------------------------------


def test_whitsun_replay_tab_still_present() -> None:
    dash_app = _import_dash_app()
    ids = _walk_ids(dash_app.app.layout)
    # Sentinels for the Whitsun replay layout.
    assert "whitsun-replay-root" in ids
    assert "whitsun-replay-timeline-radio" in ids
    assert "whitsun-replay-step-counter" in ids
