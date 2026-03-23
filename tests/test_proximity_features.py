"""
Tests for custody/features/proximity_features.py — Step 14a.

Covers:
  - haversine_km: known distances, identical points, poles, symmetry
  - nearest_vessel_from_records: empty, single, multiple, self-exclusion,
    missing target_id, tie-breaking by insertion order
  - group_records_by_time: empty, single group, multiple groups, order
"""
import math

import pytest

from custody.features.proximity_features import (
    group_records_by_time,
    haversine_km,
    nearest_vessel_from_records,
)


# ---------------------------------------------------------------------------
# haversine_km
# ---------------------------------------------------------------------------

class TestHaversineKm:
    def test_identical_points_returns_zero(self):
        assert haversine_km(1.3, 0.75, 1.3, 0.75) == 0.0

    def test_zero_zero_to_zero_one_degree_lon(self):
        # 1° of longitude at equator ≈ 111.19 km
        dist = haversine_km(0.0, 0.0, 0.0, 1.0)
        assert dist == pytest.approx(111.19, rel=0.005)

    def test_zero_zero_to_one_degree_lat(self):
        # 1° of latitude ≈ 111.19 km
        dist = haversine_km(0.0, 0.0, 1.0, 0.0)
        assert dist == pytest.approx(111.19, rel=0.005)

    def test_known_distance_singapore_to_batam(self):
        # Singapore (1.3521, 103.8198) to Batam (1.1301, 104.0529) ≈ 30–40 km
        dist = haversine_km(1.3521, 103.8198, 1.1301, 104.0529)
        assert 25.0 < dist < 45.0

    def test_symmetry(self):
        d1 = haversine_km(1.0, 2.0, 3.0, 4.0)
        d2 = haversine_km(3.0, 4.0, 1.0, 2.0)
        assert d1 == pytest.approx(d2)

    def test_returns_nonnegative(self):
        for args in [
            (0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 90.0, 180.0),
            (-45.0, -90.0, 45.0, 90.0),
        ]:
            assert haversine_km(*args) >= 0.0

    def test_antipodal_points_near_max(self):
        # Antipodal ≈ half Earth circumference ≈ 20015 km
        dist = haversine_km(0.0, 0.0, 0.0, 180.0)
        assert 20000.0 < dist < 20100.0

    def test_small_separation_not_zero(self):
        dist = haversine_km(0.0, 0.0, 0.0, 0.001)
        assert dist > 0.0

    def test_result_is_float(self):
        assert isinstance(haversine_km(1.0, 2.0, 3.0, 4.0), float)


# ---------------------------------------------------------------------------
# nearest_vessel_from_records
# ---------------------------------------------------------------------------

def _rec(target_id: str, lat: float, lon: float) -> dict:
    return {"target_id": target_id, "lat": lat, "lon": lon}


class TestNearestVesselFromRecords:
    def test_empty_others_returns_none_and_inf(self):
        vid, dist = nearest_vessel_from_records(1.0, 1.0, "V001", [])
        assert vid is None
        assert dist == math.inf

    def test_single_other_returns_it(self):
        others = [_rec("V002", 1.0, 1.1)]
        vid, dist = nearest_vessel_from_records(1.0, 1.0, "V001", others)
        assert vid == "V002"
        assert dist == pytest.approx(haversine_km(1.0, 1.0, 1.0, 1.1))

    def test_self_excluded_by_vessel_id(self):
        others = [_rec("V001", 1.0, 1.0)]   # same id as reference
        vid, dist = nearest_vessel_from_records(1.0, 1.0, "V001", others)
        assert vid is None
        assert dist == math.inf

    def test_self_excluded_even_when_mixed_with_others(self):
        others = [
            _rec("V001", 1.0, 1.0),   # self — should be skipped
            _rec("V002", 1.0, 1.5),
        ]
        vid, dist = nearest_vessel_from_records(1.0, 1.0, "V001", others)
        assert vid == "V002"

    def test_nearest_of_multiple_selected(self):
        others = [
            _rec("V002", 1.0, 1.5),   # farther
            _rec("V003", 1.0, 1.1),   # nearer
            _rec("V004", 1.0, 2.0),   # farthest
        ]
        vid, dist = nearest_vessel_from_records(1.0, 1.0, "V001", others)
        assert vid == "V003"
        assert dist < haversine_km(1.0, 1.0, 1.0, 1.5)

    def test_distance_value_is_correct(self):
        others = [_rec("V002", 2.0, 0.0)]
        vid, dist = nearest_vessel_from_records(0.0, 0.0, "V001", others)
        assert dist == pytest.approx(haversine_km(0.0, 0.0, 2.0, 0.0))

    def test_record_missing_target_id_skipped(self):
        others = [{"lat": 1.0, "lon": 1.1}]   # no target_id key
        vid, dist = nearest_vessel_from_records(1.0, 1.0, "V001", others)
        assert vid is None
        assert dist == math.inf

    def test_all_self_records_returns_none(self):
        others = [_rec("V001", 1.0, 1.0), _rec("V001", 1.5, 1.5)]
        vid, dist = nearest_vessel_from_records(1.0, 1.0, "V001", others)
        assert vid is None
        assert dist == math.inf

    def test_tie_broken_by_insertion_order(self):
        # Two vessels equidistant — first in list wins.
        others = [
            _rec("V002", 0.0, 1.0),
            _rec("V003", 1.0, 0.0),
        ]
        d1 = haversine_km(0.0, 0.0, 0.0, 1.0)
        d2 = haversine_km(0.0, 0.0, 1.0, 0.0)
        # Both should be equal (symmetry of lat/lon swap at equator is approximate)
        if abs(d1 - d2) < 0.01:
            vid, _ = nearest_vessel_from_records(0.0, 0.0, "V001", others)
            assert vid == "V002"   # first encountered

    def test_returns_tuple_of_correct_types(self):
        others = [_rec("V002", 1.0, 1.0)]
        vid, dist = nearest_vessel_from_records(0.0, 0.0, "V001", others)
        assert isinstance(vid, str)
        assert isinstance(dist, float)


# ---------------------------------------------------------------------------
# group_records_by_time
# ---------------------------------------------------------------------------

from datetime import datetime, UTC


_T1 = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
_T2 = datetime(2026, 1, 1, 11, 0, tzinfo=UTC)
_T3 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


class TestGroupRecordsByTime:
    def test_empty_list_returns_empty_dict(self):
        assert group_records_by_time([]) == {}

    def test_single_record_creates_one_group(self):
        records = [{"time": _T1, "target_id": "V001"}]
        result = group_records_by_time(records)
        assert list(result.keys()) == [_T1]
        assert result[_T1] == records

    def test_two_records_same_time_grouped_together(self):
        r1 = {"time": _T1, "target_id": "V001"}
        r2 = {"time": _T1, "target_id": "V002"}
        result = group_records_by_time([r1, r2])
        assert len(result) == 1
        assert set(r["target_id"] for r in result[_T1]) == {"V001", "V002"}

    def test_two_records_different_times_two_groups(self):
        r1 = {"time": _T1, "target_id": "V001"}
        r2 = {"time": _T2, "target_id": "V001"}
        result = group_records_by_time([r1, r2])
        assert set(result.keys()) == {_T1, _T2}
        assert result[_T1] == [r1]
        assert result[_T2] == [r2]

    def test_three_timestamps_three_groups(self):
        records = [
            {"time": _T1, "target_id": "V001"},
            {"time": _T2, "target_id": "V001"},
            {"time": _T3, "target_id": "V001"},
            {"time": _T2, "target_id": "V002"},
        ]
        result = group_records_by_time(records)
        assert set(result.keys()) == {_T1, _T2, _T3}
        assert len(result[_T2]) == 2

    def test_insertion_order_preserved_within_group(self):
        r1 = {"time": _T1, "target_id": "V001"}
        r2 = {"time": _T1, "target_id": "V002"}
        r3 = {"time": _T1, "target_id": "V003"}
        result = group_records_by_time([r1, r2, r3])
        assert result[_T1] == [r1, r2, r3]

    def test_custom_time_key(self):
        records = [{"ts": _T1, "target_id": "V001"}, {"ts": _T2, "target_id": "V001"}]
        result = group_records_by_time(records, time_key="ts")
        assert set(result.keys()) == {_T1, _T2}
