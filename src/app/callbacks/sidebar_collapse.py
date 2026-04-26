"""Sidebar collapse / expand toggle.

A small button in the main column header lets a presenter collapse the
left sidebar to give the replay tabs more horizontal space during a
screen-share, then restore it.  Works across all three tabs and is
independent of the existing tab-specific sidebar block swap.

The pure functions :func:`render_collapse` and
:func:`compute_collapse_state` carry the logic so unit tests can
exercise the state transitions without mounting Dash.
"""
from __future__ import annotations

from dash import Dash, Input, Output, State


# ---------------------------------------------------------------------------
# Public component / store IDs
# ---------------------------------------------------------------------------


SIDEBAR_COLLAPSED_STORE = "custody-sidebar-collapsed-store"
SIDEBAR_TOGGLE_BUTTON = "custody-sidebar-toggle"
SIDEBAR_COL = "custody-main-sidebar-col"
MAIN_CONTENT_COL = "custody-main-content-col"


_SIDEBAR_BORDER = "1px solid #2d2d2d"


# ---------------------------------------------------------------------------
# Pure logic — unit-testable without Dash
# ---------------------------------------------------------------------------


def render_collapse(collapsed: bool) -> tuple[dict, int, str]:
    """Return ``(sidebar_col_style, main_col_width, toggle_button_text)``.

    Open state (``collapsed=False``):
      sidebar_col_style preserves the right-border + min-height; main
      column width is 10/12; toggle reads "Hide panel".

    Collapsed state (``collapsed=True``):
      sidebar_col_style adds ``display: none`` so the column drops out
      of layout; main column width is 12/12 (full row); toggle reads
      "Show panel" so the operator can reopen it.
    """
    base_style = {"borderRight": _SIDEBAR_BORDER, "minHeight": "100vh"}
    if collapsed:
        return ({**base_style, "display": "none"}, 12, "Show panel")
    return (base_style, 10, "Hide panel")


def compute_collapse_state(
    current: bool | None,
    n_clicks: int | None,
) -> bool:
    """Flip the store on a real click; otherwise preserve current state."""
    if not n_clicks:  # None or 0 — initial render or no click yet
        return bool(current)
    return not bool(current)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register(app: Dash) -> None:
    """Register the two collapse callbacks on *app*."""

    @app.callback(
        Output(SIDEBAR_COLLAPSED_STORE, "data"),
        Input(SIDEBAR_TOGGLE_BUTTON, "n_clicks"),
        State(SIDEBAR_COLLAPSED_STORE, "data"),
        prevent_initial_call=True,
    )
    def _toggle(n_clicks, current):
        return compute_collapse_state(current, n_clicks)

    @app.callback(
        Output(SIDEBAR_COL, "style"),
        Output(MAIN_CONTENT_COL, "width"),
        Output(SIDEBAR_TOGGLE_BUTTON, "children"),
        Input(SIDEBAR_COLLAPSED_STORE, "data"),
    )
    def _apply(collapsed):
        return render_collapse(bool(collapsed))
