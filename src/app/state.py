"""Canonical state definitions for the Dash app.

Store IDs are string constants referenced by layout modules and callbacks.
The server-side record cache avoids serializing large record lists with
frozen dataclass objects into the browser.

Usage:
    from state import SCENARIO_KEY, TIMESTEP_INDEX, get_records, build_stores
"""
from __future__ import annotations

from dash import dcc

from adapter import load_scenario

# ---------------------------------------------------------------------------
# Store IDs — referenced by layout and callback modules
# ---------------------------------------------------------------------------

SCENARIO_KEY = "scenario-key"
"""Scenario name string stored in the browser.  Used as a lookup key into
the server-side record cache via :func:`get_records`."""

TIMESTEP_INDEX = "timestep-index"
"""Current timestep index (int).  Single source of truth for what time
the UI displays.  Updated by the timeline slider callback."""

SELECTED_ENTITY = "selected-entity"
"""Currently selected entity ID (str or None).  Updated by table row
click or entity dropdown."""

VIEW_MODE = "view-mode"
"""Active view: ``"overview"`` or ``"entity_detail"``.  Reserved for
Phase 2; included now to establish the ID."""

# ---------------------------------------------------------------------------
# Server-side record cache
# ---------------------------------------------------------------------------

_record_cache: dict[str, list[dict]] = {}


def get_records(scenario_name: str) -> list[dict]:
    """Return cached records for *scenario_name*, loading if needed.

    Records are kept in a module-level dict keyed by scenario name.
    Appropriate for a single-user demo; no eviction policy.
    """
    if scenario_name not in _record_cache:
        _record_cache[scenario_name] = load_scenario(scenario_name)
    return _record_cache[scenario_name]


# ---------------------------------------------------------------------------
# Store component builder
# ---------------------------------------------------------------------------

def build_stores() -> list[dcc.Store]:
    """Return the ``dcc.Store`` components to include in ``app.layout``."""
    return [
        dcc.Store(id=SCENARIO_KEY, data=None),
        dcc.Store(id=TIMESTEP_INDEX, data=0),
        dcc.Store(id=SELECTED_ENTITY, data=None),
        dcc.Store(id=VIEW_MODE, data="overview"),
    ]
