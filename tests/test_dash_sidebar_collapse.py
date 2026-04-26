"""Tests for the global sidebar collapse / expand toggle.

Covers the layout wiring (store, button, column IDs), the pure
collapse-state and apply-collapse logic, and the registered Dash
callbacks.  No live HTTP, no Dash server is started.
"""
from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


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


def _import_collapse_module():
    import sys
    src_app = REPO_ROOT / "src" / "app"
    if str(src_app) not in sys.path:
        sys.path.insert(0, str(src_app))
    for k in list(sys.modules):
        if k.startswith("callbacks."):
            del sys.modules[k]
    import callbacks.sidebar_collapse as mod
    return mod


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
# Layout wiring
# ---------------------------------------------------------------------------


def test_collapse_store_in_layout() -> None:
    dash_app = _import_dash_app()
    ids = _walk_ids(dash_app.app.layout)
    assert "custody-sidebar-collapsed-store" in ids


def test_toggle_button_in_layout() -> None:
    dash_app = _import_dash_app()
    ids = _walk_ids(dash_app.app.layout)
    assert "custody-sidebar-toggle" in ids


def test_sidebar_and_main_col_ids_in_layout() -> None:
    dash_app = _import_dash_app()
    ids = _walk_ids(dash_app.app.layout)
    assert "custody-main-sidebar-col" in ids
    assert "custody-main-content-col" in ids


# ---------------------------------------------------------------------------
# Default state: open
# ---------------------------------------------------------------------------


def test_default_open_state_widths_and_button_text() -> None:
    mod = _import_collapse_module()
    sidebar_style, main_width, button_text = mod.render_collapse(False)
    assert sidebar_style.get("display") != "none"
    assert main_width == 10
    assert button_text == "Hide panel"


def test_default_open_preserves_border_and_min_height() -> None:
    mod = _import_collapse_module()
    sidebar_style, _, _ = mod.render_collapse(False)
    assert "borderRight" in sidebar_style
    assert "minHeight" in sidebar_style


# ---------------------------------------------------------------------------
# Collapsed state
# ---------------------------------------------------------------------------


def test_collapsed_state_hides_sidebar() -> None:
    mod = _import_collapse_module()
    sidebar_style, main_width, button_text = mod.render_collapse(True)
    assert sidebar_style.get("display") == "none"
    assert main_width == 12
    assert button_text == "Show panel"


def test_collapsed_state_keeps_button_visible_via_main_col() -> None:
    """The toggle button is mounted inside the main column, not the
    sidebar, so collapsing the sidebar must not hide the button."""
    dash_app = _import_dash_app()

    def _find(c, target_id):
        if getattr(c, "id", None) == target_id:
            return c
        children = getattr(c, "children", None)
        if children is None:
            return None
        if isinstance(children, list):
            for ch in children:
                got = _find(ch, target_id)
                if got is not None:
                    return got
        else:
            return _find(children, target_id)
        return None

    main_col = _find(dash_app.app.layout, "custody-main-content-col")
    assert main_col is not None
    # The toggle button id appears somewhere inside the main col.
    assert _find(main_col, "custody-sidebar-toggle") is not None


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


def test_compute_collapse_state_initial_no_click_keeps_current() -> None:
    mod = _import_collapse_module()
    assert mod.compute_collapse_state(False, None) is False
    assert mod.compute_collapse_state(True, None) is True
    assert mod.compute_collapse_state(False, 0) is False


def test_compute_collapse_state_click_toggles() -> None:
    mod = _import_collapse_module()
    assert mod.compute_collapse_state(False, 1) is True
    assert mod.compute_collapse_state(True, 1) is False
    # Subsequent clicks (n_clicks=2, 3, ...) still produce a flip
    # because the store data carries the latest committed state.
    assert mod.compute_collapse_state(False, 2) is True


# ---------------------------------------------------------------------------
# Callback registration
# ---------------------------------------------------------------------------


def test_apply_callback_outputs_three_targets() -> None:
    """The apply-collapse callback must drive the sidebar col style,
    the main col width, and the toggle button text."""
    dash_app = _import_dash_app()
    matched = [
        cid for cid in dash_app.app.callback_map
        if "custody-main-sidebar-col.style" in cid
        and "custody-main-content-col.width" in cid
        and "custody-sidebar-toggle.children" in cid
    ]
    assert matched, "apply-collapse callback not registered with all three outputs"


def test_toggle_callback_writes_collapsed_store() -> None:
    dash_app = _import_dash_app()
    matched = [
        cid for cid in dash_app.app.callback_map
        if "custody-sidebar-collapsed-store.data" in cid
    ]
    assert matched, "toggle callback not registered"


# ---------------------------------------------------------------------------
# Existing tab-swap callback is preserved (regression guard)
# ---------------------------------------------------------------------------


def test_tab_swap_callback_still_drives_three_sidebars() -> None:
    dash_app = _import_dash_app()
    matched = [
        cid for cid in dash_app.app.callback_map
        if "custody-main-sidebar-overview" in cid
        and "custody-main-sidebar-whitsun" in cid
        and "custody-main-sidebar-tennent" in cid
    ]
    assert matched, "tab-swap callback no longer covers all three sidebar swap targets"
