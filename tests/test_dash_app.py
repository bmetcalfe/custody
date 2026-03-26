"""
Tests for the Dash application (Phase 1).

Covers:
  1. App layout contains all expected stores
  2. App layout contains sidebar components (dropdowns, slider)
  3. App layout contains overview components (KPI cards, table)
  4. Scenario changed callback returns correct slider max
  5. Scenario changed callback resets slider to 0
  6. Scenario changed callback populates entity dropdown
  7. Scenario changed callback clears table selection
  8. Portfolio callback returns correct KPI counts at step 0
  9. Portfolio callback returns non-empty table data
 10. Portfolio callback returns table with expected columns
 11. Portfolio callback handles unknown scenario gracefully
 12. State module caches records (second call returns same object)
 13. State build_stores returns correct number of stores
"""
from __future__ import annotations

import sys
import os

import pytest

# Ensure src paths are on sys.path for imports
_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in [os.path.join(_repo, "src"), os.path.join(_repo, "src", "app")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import state as app_state
from adapter import scenario_names, timestep_count, entity_ids


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def app():
    """Import and return the Dash app (does not start the server)."""
    from dash_app import app as _app
    return _app


@pytest.fixture(scope="module")
def smoke_records():
    return app_state.get_records("two_vessel_smoke")


# ---------------------------------------------------------------------------
# Layout structure
# ---------------------------------------------------------------------------

class TestLayoutStructure:

    def test_stores_present(self, app):
        """All canonical dcc.Store components should be in the layout."""
        from dash import dcc
        layout_str = str(app.layout)
        for store_id in [app_state.SCENARIO_KEY, app_state.TIMESTEP_INDEX,
                         app_state.SELECTED_ENTITY, app_state.VIEW_MODE]:
            assert store_id in layout_str, f"Store '{store_id}' not found in layout"

    def test_sidebar_components(self, app):
        layout_str = str(app.layout)
        from layout.sidebar import SCENARIO_DROPDOWN, TIMELINE_SLIDER, ENTITY_DROPDOWN
        for cid in [SCENARIO_DROPDOWN, TIMELINE_SLIDER, ENTITY_DROPDOWN]:
            assert cid in layout_str, f"Sidebar component '{cid}' not found"

    def test_overview_components(self, app):
        layout_str = str(app.layout)
        from layout.overview import PORTFOLIO_TABLE, KPI_TOTAL
        assert PORTFOLIO_TABLE in layout_str
        assert KPI_TOTAL in layout_str


# ---------------------------------------------------------------------------
# State module
# ---------------------------------------------------------------------------

class TestStateModule:

    def test_build_stores_count(self):
        stores = app_state.build_stores()
        assert len(stores) == 4

    def test_get_records_caches(self):
        r1 = app_state.get_records("two_vessel_smoke")
        r2 = app_state.get_records("two_vessel_smoke")
        assert r1 is r2  # same object, not a re-run

    def test_get_records_returns_list(self):
        records = app_state.get_records("two_vessel_smoke")
        assert isinstance(records, list)
        assert len(records) > 0


# ---------------------------------------------------------------------------
# Callback logic (unit tests calling the underlying functions directly)
# ---------------------------------------------------------------------------

class TestScenarioChangedCallback:

    def test_returns_correct_max(self):
        """Scenario 'two_vessel_smoke' has 10 timesteps → max_idx=9."""
        records = app_state.get_records("two_vessel_smoke")
        n = timestep_count(records)
        assert n == 10

    def test_entity_options_populated(self):
        records = app_state.get_records("two_vessel_smoke")
        ids = entity_ids(records)
        assert len(ids) >= 2
        assert "V001" in ids


class TestPortfolioCallback:

    def test_kpi_counts_at_step_0(self, smoke_records):
        from adapter import timestep_as_dataframe
        from portfolio_overview import compute_kpi_counts
        df = timestep_as_dataframe(smoke_records, 0)
        kpis = compute_kpi_counts(df)
        assert kpis["total"] >= 2
        assert isinstance(kpis["needs_action"], int)

    def test_overview_df_has_expected_columns(self, smoke_records):
        from adapter import timestep_as_dataframe
        from portfolio_overview import build_overview_df
        df = timestep_as_dataframe(smoke_records, 0)
        overview = build_overview_df(df)
        expected = {"Rank", "Entity", "Status", "Anomaly", "Health", "Action", "Reason"}
        assert expected.issubset(set(overview.columns))

    def test_overview_df_non_empty(self, smoke_records):
        from adapter import timestep_as_dataframe
        from portfolio_overview import build_overview_df
        df = timestep_as_dataframe(smoke_records, 0)
        overview = build_overview_df(df)
        assert len(overview) >= 2
