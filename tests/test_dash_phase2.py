"""
Tests for Dash Phase 2: map, entity detail, tables.

Covers:
  1.  Layout contains overview map component
  2.  Layout contains entity detail components (summary, fusion, decision, tables)
  3.  Map callback produces valid pydeck JSON
  4.  Map callback includes zone polygon layer
  5.  Map callback includes entity scatter layer
  6.  Map callback includes ground track layer
  7.  Map callback handles selected entity highlight
  8.  Entity reasoning callback returns valid summary
  9.  Entity reasoning callback returns fusion values
 10.  Entity reasoning callback returns decision action
 11.  Entity reasoning callback returns task queue
 12.  Entity reasoning callback handles no entity gracefully
 13.  Entity tables callback returns alerts data
 14.  Entity tables callback returns compound data
 15.  Entity tables callback returns orbital passes
 16.  Entity tables callback returns trace rows
 17.  Entity tables callback handles no entity gracefully
 18.  Event feed callback returns events for mid-scenario step
 19.  All 9 callbacks registered
"""
from __future__ import annotations

import json
import sys
import os

import pytest

_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in [os.path.join(_repo, "src"), os.path.join(_repo, "src", "app")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import state as app_state
from adapter import entity_ids, records_at_timestep


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def app():
    from dash_app import app as _app
    return _app


@pytest.fixture(scope="module")
def smoke_records():
    return app_state.get_records("two_vessel_smoke")


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

class TestPhase2Layout:

    def test_map_component_in_layout(self, app):
        from layout.map_panel import OVERVIEW_MAP
        assert OVERVIEW_MAP in str(app.layout)

    def test_detail_components_in_layout(self, app):
        from layout.entity_detail import (
            DETAIL_SUMMARY, FUSION_SCORE, DECISION_ACTION,
            ALERTS_TABLE, TRACES_TABLE,
        )
        layout_str = str(app.layout)
        for cid in [DETAIL_SUMMARY, FUSION_SCORE, DECISION_ACTION,
                     ALERTS_TABLE, TRACES_TABLE]:
            assert cid in layout_str, f"Component '{cid}' not in layout"

    def test_event_feed_in_layout(self, app):
        from layout.overview import EVENT_FEED
        assert EVENT_FEED in str(app.layout)


# ---------------------------------------------------------------------------
# Map callback (unit test the builder function directly)
# ---------------------------------------------------------------------------

class TestMapCallback:

    def test_produces_valid_json(self, smoke_records):
        from callbacks.map_layers import _build_deck_json
        ts_recs = records_at_timestep(smoke_records, 5)
        deck_json = _build_deck_json(ts_recs, None)
        parsed = json.loads(deck_json)
        assert "layers" in parsed
        assert "initialViewState" in parsed

    def test_includes_zone_layer(self, smoke_records):
        from callbacks.map_layers import _build_deck_json
        ts_recs = records_at_timestep(smoke_records, 5)
        parsed = json.loads(_build_deck_json(ts_recs, None))
        layer_types = [l.get("@@type") for l in parsed["layers"]]
        assert "PolygonLayer" in layer_types

    def test_includes_entity_scatter(self, smoke_records):
        from callbacks.map_layers import _build_deck_json
        ts_recs = records_at_timestep(smoke_records, 5)
        parsed = json.loads(_build_deck_json(ts_recs, None))
        scatter_layers = [l for l in parsed["layers"] if l.get("@@type") == "ScatterplotLayer"]
        assert len(scatter_layers) >= 1

    def test_includes_ground_tracks(self, smoke_records):
        from callbacks.map_layers import _build_deck_json
        ts_recs = records_at_timestep(smoke_records, 5)
        parsed = json.loads(_build_deck_json(ts_recs, None))
        path_layers = [l for l in parsed["layers"] if l.get("@@type") == "PathLayer"]
        assert len(path_layers) >= 1  # ground tracks

    def test_selected_entity_highlight(self, smoke_records):
        from callbacks.map_layers import _build_deck_json
        ts_recs = records_at_timestep(smoke_records, 5)
        ids = entity_ids(smoke_records)
        deck_with = json.loads(_build_deck_json(ts_recs, ids[0]))
        deck_without = json.loads(_build_deck_json(ts_recs, None))
        # Selected entity adds a highlight ring (extra ScatterplotLayer)
        assert len(deck_with["layers"]) > len(deck_without["layers"])


# ---------------------------------------------------------------------------
# Entity reasoning callback (unit test underlying logic)
# ---------------------------------------------------------------------------

class TestEntityReasoningCallback:

    @pytest.fixture
    def entity_data(self, smoke_records):
        from adapter import entity_timeline_up_to
        eid = entity_ids(smoke_records)[0]
        timeline = entity_timeline_up_to(smoke_records, eid, 5)
        return timeline[-1], timeline[:-1]

    def test_stored_fusion_present(self, entity_data):
        current, _ = entity_data
        fa = current.get("fusion_assessment")
        assert fa is not None
        assert 0.0 <= fa.fused_score <= 1.0

    def test_stored_decision_present(self, entity_data):
        current, _ = entity_data
        dec = current.get("mission_decision")
        assert dec is not None
        valid_actions = {"PASSIVE_MONITOR", "ELEVATE", "TASK_OPTICAL", "TASK_SAR", "ESCALATE"}
        assert dec.action in valid_actions

    def test_panel_summary_nonempty(self, entity_data):
        from entity_detail_data import panel_summary
        current, _ = entity_data
        fa = current["fusion_assessment"]
        dec = current["mission_decision"]
        s = panel_summary(fa, dec)
        assert len(s) > 10

    def test_task_recommendations_buildable(self, entity_data):
        from entity_detail_data import derive_track_state
        from custody.taskrecommendation import build_task_recommendations
        current, prefix = entity_data
        fa = current["fusion_assessment"]
        dec = current["mission_decision"]
        track = derive_track_state(current, prefix)
        tasks = build_task_recommendations(dec, fa, current, track)
        assert isinstance(tasks, list)
        assert len(tasks) >= 1  # MONITOR is always present


# ---------------------------------------------------------------------------
# Entity tables callback (unit test underlying data functions)
# ---------------------------------------------------------------------------

class TestEntityTablesCallback:

    def test_alerts_for_timeline(self, smoke_records):
        from custody.alerts import alerts_for_timeline
        from adapter import entity_timeline_up_to
        eid = entity_ids(smoke_records)[0]
        timeline = entity_timeline_up_to(smoke_records, eid, 8)
        alerts = alerts_for_timeline(timeline)
        assert isinstance(alerts, list)
        # Every alert carries a timestamp from the simulation record
        for a in alerts:
            assert hasattr(a, "timestamp") and a.timestamp is not None

    def test_compound_evaluation(self, smoke_records):
        from custody.compounds import evaluate_compounds
        from adapter import entity_timeline_up_to
        eid = entity_ids(smoke_records)[0]
        timeline = entity_timeline_up_to(smoke_records, eid, 5)
        current = timeline[-1]
        prefix = timeline[:-1]
        compounds = evaluate_compounds(current, window=prefix)
        assert isinstance(compounds, list)

    def test_orbital_passes(self, smoke_records):
        from orbital_passes_panel import build_orbital_passes_rows
        from adapter import entity_timeline_up_to
        eid = entity_ids(smoke_records)[0]
        timeline = entity_timeline_up_to(smoke_records, eid, 5)
        current = timeline[-1]
        rows = build_orbital_passes_rows(
            float(current["lat"]), float(current["lon"]), current["time"],
        )
        assert isinstance(rows, list)
        assert len(rows) == 6  # 6 orbital satellites

    def test_trace_rows(self, smoke_records):
        from custody.decision_trace import traces_to_rows
        from adapter import entity_timeline_up_to
        eid = entity_ids(smoke_records)[0]
        timeline = entity_timeline_up_to(smoke_records, eid, 5)
        traces = [r["decision_trace"] for r in timeline if r.get("decision_trace")]
        rows = traces_to_rows(traces)
        assert isinstance(rows, list)
        assert len(rows) == len(traces)

    def test_no_entity_returns_empty(self):
        """Tables callback should return empty lists when no entity is selected."""
        # Direct test: empty timeline → empty data
        from custody.alerts import alerts_for_timeline
        assert alerts_for_timeline([]) == []


# ---------------------------------------------------------------------------
# Collection conditions helper
# ---------------------------------------------------------------------------

class TestCollectionConditions:

    def test_day_case(self):
        from datetime import datetime
        sys.path.insert(0, os.path.join(_repo, "src", "app", "callbacks"))
        from entity_detail import _collection_conditions
        # UTC 12:00 at lon=0 → local 12:00 → Day
        utc = datetime(2026, 4, 1, 12, 0, 0)
        c = _collection_conditions(utc, 0.0)
        assert c["sun_state"] == "Day"
        assert c["optical"] == "Yes"
        assert c["local_time"] == "12:00"

    def test_night_case(self):
        from datetime import datetime
        sys.path.insert(0, os.path.join(_repo, "src", "app", "callbacks"))
        from entity_detail import _collection_conditions
        # UTC 02:00 at lon=0 → local 02:00 → Night
        utc = datetime(2026, 4, 1, 2, 0, 0)
        c = _collection_conditions(utc, 0.0)
        assert c["sun_state"] == "Night"
        assert c["optical"] == "No"
        assert c["local_time"] == "02:00"

    def test_longitude_offset(self):
        from datetime import datetime
        sys.path.insert(0, os.path.join(_repo, "src", "app", "callbacks"))
        from entity_detail import _collection_conditions
        # UTC 12:00 at lon=90 → offset +6h → local 18:00 → Night (>=18)
        utc = datetime(2026, 4, 1, 12, 0, 0)
        c = _collection_conditions(utc, 90.0)
        assert c["sun_state"] == "Night"
        assert c["local_time"] == "18:00"

    def test_missing_values(self):
        sys.path.insert(0, os.path.join(_repo, "src", "app", "callbacks"))
        from entity_detail import _collection_conditions
        c = _collection_conditions(None, None)
        assert c["local_time"] == "—"
        assert c["sar"] == "Yes"

    def test_negative_longitude(self):
        from datetime import datetime
        sys.path.insert(0, os.path.join(_repo, "src", "app", "callbacks"))
        from entity_detail import _collection_conditions
        # UTC 04:00 at lon=-75 → offset -5h → local 23:00 (prev day) → Night
        utc = datetime(2026, 4, 1, 4, 0, 0)
        c = _collection_conditions(utc, -75.0)
        assert c["sun_state"] == "Night"
        assert c["local_time"] == "23:00"


# ---------------------------------------------------------------------------
# Event feed
# ---------------------------------------------------------------------------

class TestEventFeed:

    def test_events_at_mid_scenario(self, smoke_records):
        from adapter import timestep_as_dataframe
        from overview_events import build_event_feed
        ts_df = timestep_as_dataframe(smoke_records, 5)
        prev_df = timestep_as_dataframe(smoke_records, 4)
        events = build_event_feed(ts_df, prev_df)
        assert isinstance(events, list)
        # Events may or may not fire; just verify the shape
        for evt in events:
            assert "entity_id" in evt
            assert "event_type" in evt
            assert "description" in evt


# ---------------------------------------------------------------------------
# Callback count
# ---------------------------------------------------------------------------

class TestCallbackCount:

    def test_callbacks_registered(self, app):
        # Phase 1: 5 nav + 1 portfolio = 6
        # Phase 2: +1 map + 2 entity detail = 3
        # Phase 3: +1 timestep display + 1 step buttons = 2
        # Whitsun replay: +1 selected-event sync + 1 fan-out
        #   refresh-all-panels + 1 sidebar-block-swap = 3
        # Sidebar collapse: +1 toggle store + 1 apply collapse = 2
        # Map overlays: +1 whitsun map + 1 tennent map = 2
        # Dynamic overlay manager: +1 whitsun overlay-toggle manager = 1
        # Evidence viewer: +1 whitsun + 1 tennent = 2
        # Total: 21
        assert len(app.callback_map) == 21
