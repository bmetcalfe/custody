"""
Tests for the engine-to-UI adapter layer (src/app/adapter.py).

Covers:
  1.  scenario_names returns a non-empty list of strings
  2.  load_scenario returns records for every known scenario name
  3.  load_scenario raises KeyError for unknown scenario
  4.  entity_ids returns sorted unique IDs
  5.  timestep_count matches expected for known scenarios
  6.  records_at_timestep returns correct entity count per step
  7.  records_at_timestep returns empty list for out-of-range step
  8.  entity_timeline returns all records for one entity
  9.  entity_timeline returns empty for unknown entity
 10.  entity_timeline_up_to returns prefix through step (inclusive)
 11.  entity_timeline_up_to returns empty for out-of-range step
 12.  timestep_as_dataframe returns DataFrame with correct shape
 13.  entity_timeline_as_dataframe returns DataFrame with correct shape
 14.  Records carry live objects (fusion_assessment, mission_decision)
 15.  Adapter round-trip: load → slice → records match original
"""
from __future__ import annotations

import sys
import os
from datetime import datetime

import pandas as pd
import pytest

# Ensure src/app is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "app"))

from adapter import (
    scenario_names,
    load_scenario,
    entity_ids,
    timestep_count,
    records_at_timestep,
    entity_timeline,
    entity_timeline_up_to,
    timestep_as_dataframe,
    entity_timeline_as_dataframe,
)
from custody.belief_assessment import FusionAssessment
from custody.decision import Decision


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def smoke_records():
    """Two-vessel smoke scenario — fast to load."""
    return load_scenario("two_vessel_smoke")


# ---------------------------------------------------------------------------
# Scenario catalog
# ---------------------------------------------------------------------------

class TestScenarioCatalog:

    def test_scenario_names_non_empty(self):
        names = scenario_names()
        assert len(names) >= 3
        assert all(isinstance(n, str) for n in names)

    def test_known_scenarios_present(self):
        names = scenario_names()
        assert "multi_day_72h" in names
        assert "portfolio_36h" in names
        assert "two_vessel_smoke" in names

    def test_load_unknown_raises(self):
        with pytest.raises(KeyError):
            load_scenario("nonexistent_scenario")


# ---------------------------------------------------------------------------
# Record-level accessors
# ---------------------------------------------------------------------------

class TestRecordAccessors:

    def test_entity_ids_sorted_unique(self, smoke_records):
        ids = entity_ids(smoke_records)
        assert ids == sorted(set(ids))
        assert len(ids) >= 2  # two-vessel smoke has V001, V002

    def test_timestep_count(self, smoke_records):
        count = timestep_count(smoke_records)
        # TWO_VESSEL_SMOKE: 9h + 1 inclusive = 10 timesteps
        assert count == 10

    def test_records_at_timestep_entity_count(self, smoke_records):
        recs = records_at_timestep(smoke_records, 0)
        n_entities = len(entity_ids(smoke_records))
        assert len(recs) == n_entities

    def test_records_at_timestep_all_same_time(self, smoke_records):
        recs = records_at_timestep(smoke_records, 3)
        times = {r["time"] for r in recs}
        assert len(times) == 1

    def test_records_at_timestep_out_of_range(self, smoke_records):
        assert records_at_timestep(smoke_records, -1) == []
        assert records_at_timestep(smoke_records, 9999) == []

    def test_entity_timeline_ordered(self, smoke_records):
        ids = entity_ids(smoke_records)
        tl = entity_timeline(smoke_records, ids[0])
        times = [r["time"] for r in tl]
        assert times == sorted(times)
        assert len(tl) == timestep_count(smoke_records)

    def test_entity_timeline_unknown_entity(self, smoke_records):
        assert entity_timeline(smoke_records, "NONEXISTENT") == []

    def test_entity_timeline_up_to_inclusive(self, smoke_records):
        ids = entity_ids(smoke_records)
        full = entity_timeline(smoke_records, ids[0])
        prefix = entity_timeline_up_to(smoke_records, ids[0], 3)
        # step 3 means timesteps 0,1,2,3 → 4 records
        assert len(prefix) == 4
        assert prefix == full[:4]

    def test_entity_timeline_up_to_out_of_range(self, smoke_records):
        ids = entity_ids(smoke_records)
        assert entity_timeline_up_to(smoke_records, ids[0], -1) == []
        assert entity_timeline_up_to(smoke_records, ids[0], 9999) == []


# ---------------------------------------------------------------------------
# DataFrame convenience
# ---------------------------------------------------------------------------

class TestDataFrameConvenience:

    def test_timestep_as_dataframe_shape(self, smoke_records):
        df = timestep_as_dataframe(smoke_records, 0)
        assert isinstance(df, pd.DataFrame)
        n_entities = len(entity_ids(smoke_records))
        assert len(df) == n_entities
        assert "target_id" in df.columns

    def test_entity_timeline_as_dataframe_shape(self, smoke_records):
        ids = entity_ids(smoke_records)
        df = entity_timeline_as_dataframe(smoke_records, ids[0])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == timestep_count(smoke_records)
        assert "time" in df.columns


# ---------------------------------------------------------------------------
# Live objects on records
# ---------------------------------------------------------------------------

class TestLiveObjects:

    def test_records_carry_fusion_assessment(self, smoke_records):
        for r in smoke_records[:10]:
            assert isinstance(r.get("fusion_assessment"), FusionAssessment)

    def test_records_carry_mission_decision(self, smoke_records):
        for r in smoke_records[:10]:
            assert isinstance(r.get("mission_decision"), Decision)


# ---------------------------------------------------------------------------
# Round-trip consistency
# ---------------------------------------------------------------------------

class TestRoundTrip:

    def test_slice_and_reassemble(self, smoke_records):
        """Slicing by timestep and reassembling should recover all records."""
        n_steps = timestep_count(smoke_records)
        reassembled = []
        for step in range(n_steps):
            reassembled.extend(records_at_timestep(smoke_records, step))
        assert len(reassembled) == len(smoke_records)
        # Every original record should be in the reassembled set (by identity)
        original_ids = {id(r) for r in smoke_records}
        reassembled_ids = {id(r) for r in reassembled}
        assert original_ids == reassembled_ids
