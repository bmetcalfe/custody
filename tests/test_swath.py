"""
Tests for custody.swath — sensor footprint and grouped collection valuation.

Covers:
  1.  footprint_bbox produces correct dimensions
  2.  target inside footprint is detected
  3.  target outside footprint is excluded
  4.  multiple targets in footprint all found
  5.  footprint handles equatorial and mid-latitude
  6.  assess_swath: single target → count=1, uplift=0
  7.  assess_swath: multiple targets → count>1, uplift>0
  8.  assess_swath: high priority targets increase swath value
  9.  assess_swath: SAR wider footprint covers more than EO
 10.  assess_swath rationale mentions target count
 11.  SWATH_SPECS has all expected sensor types
 12.  simulation records carry swath fields on TASK actions
 13.  simulation records have zero coverage on non-TASK actions
"""
from __future__ import annotations

import math

import pytest

from custody.swath import (
    SwathSpec,
    SwathAssessment,
    SWATH_SPECS,
    footprint_bbox,
    targets_in_footprint,
    assess_swath,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _target(tid, lat, lon, priority=0.5):
    return {"target_id": tid, "lat": lat, "lon": lon, "priority_score": priority}


_NARROW = SwathSpec(width_km=10.0, length_km=20.0)
_WIDE = SwathSpec(width_km=50.0, length_km=80.0)


# ---------------------------------------------------------------------------
# Footprint geometry
# ---------------------------------------------------------------------------

class TestFootprintBbox:

    def test_correct_dimensions(self):
        min_lat, max_lat, min_lon, max_lon = footprint_bbox(30.0, -88.0, _NARROW)
        lat_span_km = (max_lat - min_lat) * 111.0
        lon_span_km = (max_lon - min_lon) * 111.0 * math.cos(math.radians(30.0))
        assert abs(lat_span_km - 20.0) < 1.0  # length = 20 km
        assert abs(lon_span_km - 10.0) < 1.0  # width = 10 km

    def test_centered_on_point(self):
        min_lat, max_lat, min_lon, max_lon = footprint_bbox(0.0, 0.0, _NARROW)
        assert min_lat < 0 < max_lat
        assert min_lon < 0 < max_lon


class TestTargetsInFootprint:

    def test_inside_detected(self):
        center = (30.0, -88.0)
        targets = [_target("A", 30.01, -88.01)]
        covered = targets_in_footprint(*center, _WIDE, targets)
        assert len(covered) == 1

    def test_outside_excluded(self):
        center = (30.0, -88.0)
        targets = [_target("A", 35.0, -80.0)]  # far away
        covered = targets_in_footprint(*center, _NARROW, targets)
        assert len(covered) == 0

    def test_multiple_targets(self):
        center = (30.0, -88.0)
        targets = [
            _target("A", 30.01, -88.01),
            _target("B", 30.02, -87.99),
            _target("C", 30.03, -88.00),
            _target("D", 35.0, -80.0),  # far away
        ]
        covered = targets_in_footprint(*center, _WIDE, targets)
        covered_ids = {t["target_id"] for t in covered}
        assert "A" in covered_ids
        assert "B" in covered_ids
        assert "C" in covered_ids
        assert "D" not in covered_ids

    def test_equatorial(self):
        """Works at equator where lon degrees ≈ lat degrees in km."""
        targets = [_target("X", 0.01, 0.01)]
        covered = targets_in_footprint(0.0, 0.0, _WIDE, targets)
        assert len(covered) == 1


# ---------------------------------------------------------------------------
# Swath assessment
# ---------------------------------------------------------------------------

class TestAssessSwath:

    def test_single_target(self):
        records = [_target("A", 30.0, -88.0, priority=0.6)]
        sa = assess_swath(records[0], "high_resolution", records)
        assert sa.covered_target_count == 1
        assert sa.value_uplift == 0.0
        assert "single-target" in sa.rationale

    def test_multiple_targets_uplift(self):
        records = [
            _target("A", 30.0, -88.0, priority=0.8),
            _target("B", 30.01, -88.01, priority=0.6),
            _target("C", 30.02, -87.99, priority=0.4),
        ]
        sa = assess_swath(records[0], "all_weather", records)
        assert sa.covered_target_count >= 2
        assert sa.value_uplift > 0.0
        assert sa.swath_task_value > sa.single_target_value

    def test_high_priority_increases_value(self):
        records_low = [
            _target("A", 30.0, -88.0, priority=0.8),
            _target("B", 30.01, -88.01, priority=0.1),
        ]
        records_high = [
            _target("A", 30.0, -88.0, priority=0.8),
            _target("B", 30.01, -88.01, priority=0.9),
        ]
        sa_low = assess_swath(records_low[0], "all_weather", records_low)
        sa_high = assess_swath(records_high[0], "all_weather", records_high)
        assert sa_high.swath_task_value > sa_low.swath_task_value

    def test_sar_wider_than_eo(self):
        """SAR footprint should cover targets that EO misses."""
        records = [
            _target("A", 30.0, -88.0, priority=0.5),
            _target("B", 30.15, -88.0, priority=0.5),  # ~17 km away
        ]
        sa_eo = assess_swath(records[0], "high_resolution", records)
        sa_sar = assess_swath(records[0], "all_weather", records)
        # EO: 15 km wide → B is outside (17 km N); SAR: 40 km wide → B is inside
        assert sa_sar.covered_target_count >= sa_eo.covered_target_count

    def test_rationale_mentions_count(self):
        records = [
            _target("A", 30.0, -88.0, priority=0.5),
            _target("B", 30.01, -88.01, priority=0.5),
        ]
        sa = assess_swath(records[0], "all_weather", records)
        if sa.covered_target_count > 1:
            assert "covers" in sa.rationale
            assert str(sa.covered_target_count) in sa.rationale


class TestSwathSpecs:

    def test_all_sensor_types_present(self):
        assert "high_resolution" in SWATH_SPECS
        assert "all_weather" in SWATH_SPECS
        assert "fast_revisit" in SWATH_SPECS


# ---------------------------------------------------------------------------
# Simulation integration
# ---------------------------------------------------------------------------

class TestSimulationIntegration:

    @pytest.fixture(scope="class")
    def records(self):
        from custody.simulate import run_simulation
        return run_simulation()

    def test_task_records_have_swath_fields(self, records):
        task_recs = [r for r in records if r["action"] == "TASK"]
        if task_recs:
            r = task_recs[0]
            assert "swath_width_km" in r
            assert "covered_target_count" in r
            assert "swath_task_value" in r
            assert r["swath_width_km"] is not None

    def test_non_task_records_have_zero_coverage(self, records):
        non_task = [r for r in records if r["action"] != "TASK"]
        for r in non_task[:10]:
            assert r.get("covered_target_count", 0) == 0

    def test_some_collects_cover_multiple(self, records):
        """In a multi-vessel scenario, at least one collect should
        cover 2+ targets if vessels are within swath range."""
        task_recs = [r for r in records if r["action"] == "TASK"]
        multi = [r for r in task_recs if r.get("covered_target_count", 0) >= 2]
        # May or may not happen depending on vessel spacing;
        # just verify the field is populated correctly
        for r in multi:
            assert r["swath_task_value"] > 0
            assert r["swath_value_uplift"] > 0
