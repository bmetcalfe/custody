"""Tests for ground track sampling helpers (orbit.satellite_subpoint + ground_track module)."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from custody.orbit import load_tle, satellite_subpoint
from custody.sensors import orbital_satrecs
from ground_track import satellite_current_positions

# Reference TLE used across tests (EO-MIO-1 from sensors catalog)
_REF_TIME = datetime(2026, 3, 23, 12, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# satellite_subpoint (orbit.py)
# ---------------------------------------------------------------------------

class TestSatelliteSubpoint:
    """satellite_subpoint returns (lat, lon) within valid geodetic ranges."""

    def _satrec(self):
        return list(orbital_satrecs().values())[0]

    def test_returns_tuple(self):
        result = satellite_subpoint(self._satrec(), _REF_TIME)
        assert result is not None
        assert isinstance(result, tuple) and len(result) == 2

    def test_latitude_in_range(self):
        lat, _ = satellite_subpoint(self._satrec(), _REF_TIME)
        assert -90.0 <= lat <= 90.0

    def test_longitude_in_range(self):
        _, lon = satellite_subpoint(self._satrec(), _REF_TIME)
        assert -180.0 <= lon <= 180.0

    def test_consistent_with_same_time(self):
        r1 = satellite_subpoint(self._satrec(), _REF_TIME)
        r2 = satellite_subpoint(self._satrec(), _REF_TIME)
        assert r1 == r2

    def test_different_times_give_different_positions(self):
        from datetime import timedelta
        t2 = _REF_TIME + timedelta(minutes=10)
        r1 = satellite_subpoint(self._satrec(), _REF_TIME)
        r2 = satellite_subpoint(self._satrec(), t2)
        assert r1 != r2

    def test_all_catalog_satellites_return_valid_point(self):
        for sat_id, satrec in orbital_satrecs().items():
            result = satellite_subpoint(satrec, _REF_TIME)
            assert result is not None, f"{sat_id} returned None"
            lat, lon = result
            assert -90.0 <= lat <= 90.0, f"{sat_id} lat out of range: {lat}"
            assert -180.0 <= lon <= 180.0, f"{sat_id} lon out of range: {lon}"


# ---------------------------------------------------------------------------
# ground_track_segments
# ---------------------------------------------------------------------------

class TestGroundTrackSegments:
    def test_returns_list(self):
        from ground_track import ground_track_segments
        result = ground_track_segments("EO-MIO-1", _REF_TIME)
        assert isinstance(result, list)

    def test_unknown_satellite_returns_empty(self):
        from ground_track import ground_track_segments
        assert ground_track_segments("NO-SUCH-SAT", _REF_TIME) == []

    def test_segments_are_lists_of_lonlat(self):
        from ground_track import ground_track_segments
        segs = ground_track_segments("SAR-1", _REF_TIME)
        for seg in segs:
            assert isinstance(seg, list)
            assert len(seg) >= 2
            for pt in seg:
                assert len(pt) == 2
                lon, lat = pt
                assert -180.0 <= lon <= 180.0
                assert -90.0 <= lat <= 90.0

    def test_point_count_bounded(self):
        """Total points across all segments <= n_points (some may be dropped at splits)."""
        from ground_track import ground_track_segments
        n = 10
        segs = ground_track_segments("EO-SSO-1", _REF_TIME, n_points=n)
        total = sum(len(s) for s in segs)
        assert total <= n

    def test_minimum_two_points_per_segment(self):
        from ground_track import ground_track_segments
        for seg in ground_track_segments("EO-MIO-2", _REF_TIME):
            assert len(seg) >= 2


# ---------------------------------------------------------------------------
# _split_antimeridian
# ---------------------------------------------------------------------------

class TestSplitAntimeridian:
    def test_no_split_for_continuous_track(self):
        from ground_track import _split_antimeridian
        pts = [[0.0, 1.0], [5.0, 2.0], [10.0, 3.0]]
        segs = _split_antimeridian(pts)
        assert len(segs) == 1
        assert segs[0] == pts

    def test_splits_at_antimeridian(self):
        from ground_track import _split_antimeridian
        # 170 → -170 is a 340° jump → split; the leading single-point segment is discarded
        pts = [[170.0, 1.0], [-170.0, 2.0], [-160.0, 3.0]]
        segs = _split_antimeridian(pts)
        # [170] is a single-point segment → discarded; [-170, -160] is kept
        assert len(segs) == 1
        assert segs[0] == [[-170.0, 2.0], [-160.0, 3.0]]

    def test_splits_two_valid_segments(self):
        from ground_track import _split_antimeridian
        # Both sides have >= 2 points
        pts = [[160.0, 1.0], [170.0, 2.0], [-170.0, 3.0], [-160.0, 4.0]]
        segs = _split_antimeridian(pts)
        assert len(segs) == 2
        assert segs[0] == [[160.0, 1.0], [170.0, 2.0]]
        assert segs[1] == [[-170.0, 3.0], [-160.0, 4.0]]

    def test_empty_input_returns_empty(self):
        from ground_track import _split_antimeridian
        assert _split_antimeridian([]) == []

    def test_single_point_returns_empty(self):
        from ground_track import _split_antimeridian
        assert _split_antimeridian([[5.0, 1.0]]) == []

    def test_two_points_no_jump_returns_one_segment(self):
        from ground_track import _split_antimeridian
        pts = [[5.0, 1.0], [10.0, 2.0]]
        segs = _split_antimeridian(pts)
        assert len(segs) == 1
        assert segs[0] == pts


# ---------------------------------------------------------------------------
# all_ground_track_layers_data
# ---------------------------------------------------------------------------

class TestAllGroundTrackLayersData:
    def test_returns_list(self):
        from ground_track import all_ground_track_layers_data
        result = all_ground_track_layers_data(_REF_TIME)
        assert isinstance(result, list)

    def test_non_empty_for_valid_time(self):
        from ground_track import all_ground_track_layers_data
        result = all_ground_track_layers_data(_REF_TIME)
        assert len(result) > 0

    def test_rows_have_path_and_satellite_id(self):
        from ground_track import all_ground_track_layers_data
        for row in all_ground_track_layers_data(_REF_TIME):
            assert "path" in row
            assert "satellite_id" in row
            assert isinstance(row["path"], list)
            assert len(row["path"]) >= 2

    def test_satellite_ids_are_known(self):
        from ground_track import all_ground_track_layers_data
        known = set(orbital_satrecs().keys())
        for row in all_ground_track_layers_data(_REF_TIME):
            assert row["satellite_id"] in known

    def test_covers_all_catalog_satellites(self):
        """Every catalog satellite should contribute at least one segment."""
        from ground_track import all_ground_track_layers_data
        known = set(orbital_satrecs().keys())
        seen = {row["satellite_id"] for row in all_ground_track_layers_data(_REF_TIME)}
        assert known == seen


# ---------------------------------------------------------------------------
# orbital_satrecs (sensors.py public accessor)
# ---------------------------------------------------------------------------

class TestOrbitalSatrecs:
    def test_returns_dict(self):
        assert isinstance(orbital_satrecs(), dict)

    def test_contains_expected_satellites(self):
        keys = set(orbital_satrecs().keys())
        expected = {"EO-MIO-1", "EO-MIO-2", "EO-SSO-1", "EO-SSO-2", "SAR-1", "SAR-2"}
        assert keys == expected

    def test_returns_same_object_on_repeated_calls(self):
        assert orbital_satrecs() is orbital_satrecs()


# ---------------------------------------------------------------------------
# satellite_current_positions
# ---------------------------------------------------------------------------

class TestSatelliteCurrentPositions:
    def test_returns_list(self):
        result = satellite_current_positions(_REF_TIME)
        assert isinstance(result, list)

    def test_one_entry_per_catalog_satellite(self):
        result = satellite_current_positions(_REF_TIME)
        assert len(result) == len(orbital_satrecs())

    def test_rows_have_required_keys(self):
        for row in satellite_current_positions(_REF_TIME):
            assert "lon" in row and "lat" in row and "satellite_id" in row

    def test_positions_are_valid_geodetic(self):
        for row in satellite_current_positions(_REF_TIME):
            assert -180.0 <= row["lon"] <= 180.0
            assert  -90.0 <= row["lat"] <=  90.0

    def test_satellite_ids_match_catalog(self):
        known = set(orbital_satrecs().keys())
        for row in satellite_current_positions(_REF_TIME):
            assert row["satellite_id"] in known

    def test_different_times_give_different_positions(self):
        from datetime import timedelta
        r1 = {r["satellite_id"]: (r["lat"], r["lon"])
              for r in satellite_current_positions(_REF_TIME)}
        r2 = {r["satellite_id"]: (r["lat"], r["lon"])
              for r in satellite_current_positions(_REF_TIME + timedelta(minutes=15))}
        assert r1 != r2
