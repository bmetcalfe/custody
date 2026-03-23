"""
Tests for replay_ais_file (Step 11b).

Verifies that the multi-track replay layer correctly groups observations by
vessel_id, dispatches each group to ingest_ais_track, and returns a dict keyed
by vessel_id.  Single-vessel correctness is covered by test_ais_ingest.py;
these tests focus on grouping semantics and edge cases specific to multi-track
replay.
"""
from datetime import datetime, timezone, timedelta
import textwrap

import pytest

from custody.ais import AISObservation, ingest_ais_track, replay_ais_file

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

T0 = datetime(2026, 3, 23, 10, 0, tzinfo=timezone.utc)


def _obs(vessel_id, offset_hours, lat=0.0, lon=0.0, speed_knots=10.0, heading_deg=45.0):
    return AISObservation(
        vessel_id=vessel_id,
        timestamp=T0 + timedelta(hours=offset_hours),
        lat=lat,
        lon=lon,
        speed_knots=speed_knots,
        heading_deg=heading_deg,
    )


def _csv(*rows, header=True):
    """Build a minimal valid CSV string from dicts, optionally including a header."""
    cols = ["vessel_id", "timestamp", "lat", "lon", "speed_knots", "heading_deg"]
    lines = []
    if header:
        lines.append(",".join(cols))
    for r in rows:
        ts = r["timestamp"].isoformat() if isinstance(r["timestamp"], datetime) else r["timestamp"]
        lines.append(f"{r['vessel_id']},{ts},{r['lat']},{r['lon']},{r['speed_knots']},{r['heading_deg']}")
    return "\n".join(lines)


def _row(vessel_id, offset_hours, lat=0.0, lon=0.0, speed_knots=10.0, heading_deg=45.0):
    return {
        "vessel_id": vessel_id,
        "timestamp": T0 + timedelta(hours=offset_hours),
        "lat": lat,
        "lon": lon,
        "speed_knots": speed_knots,
        "heading_deg": heading_deg,
    }


# ---------------------------------------------------------------------------
# Empty source
# ---------------------------------------------------------------------------

class TestEmptySource:
    def test_empty_csv_returns_empty_dict(self):
        csv = "vessel_id,timestamp,lat,lon,speed_knots,heading_deg\n"
        result = replay_ais_file(csv)
        assert result == {}

    def test_return_type_is_dict(self):
        csv = "vessel_id,timestamp,lat,lon,speed_knots,heading_deg\n"
        result = replay_ais_file(csv)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Single vessel
# ---------------------------------------------------------------------------

class TestSingleVessel:
    def test_single_vessel_key_present(self):
        csv = _csv(_row("V001", 0), _row("V001", 1), _row("V001", 2))
        result = replay_ais_file(csv)
        assert set(result.keys()) == {"V001"}

    def test_single_vessel_matches_ingest_ais_track(self):
        obs = [_obs("V001", i) for i in range(3)]
        csv = _csv(*[_row("V001", i) for i in range(3)])
        multi_result = replay_ais_file(csv)
        direct_result = ingest_ais_track(obs)
        assert len(multi_result["V001"]) == len(direct_result)

    def test_single_vessel_timeline_length(self):
        n = 5
        csv = _csv(*[_row("V001", i) for i in range(n)])
        result = replay_ais_file(csv)
        assert len(result["V001"]) == n

    def test_single_vessel_first_record_target_id(self):
        csv = _csv(_row("V001", 0), _row("V001", 1))
        result = replay_ais_file(csv)
        assert result["V001"][0]["target_id"] == "V001"

    def test_single_obs_vessel_returns_one_record(self):
        """A vessel with exactly one observation gets the first-record treatment."""
        csv = _csv(_row("SOLO", 0))
        result = replay_ais_file(csv)
        assert len(result["SOLO"]) == 1
        assert result["SOLO"][0]["action"] == "NONE"


# ---------------------------------------------------------------------------
# Two vessels
# ---------------------------------------------------------------------------

class TestTwoVessels:
    def test_two_vessel_keys(self):
        csv = _csv(
            _row("V001", 0), _row("V001", 1),
            _row("V002", 0), _row("V002", 1),
        )
        result = replay_ais_file(csv)
        assert set(result.keys()) == {"V001", "V002"}

    def test_two_vessel_timeline_lengths(self):
        csv = _csv(
            _row("V001", 0), _row("V001", 1), _row("V001", 2),
            _row("V002", 0), _row("V002", 1),
        )
        result = replay_ais_file(csv)
        assert len(result["V001"]) == 3
        assert len(result["V002"]) == 2

    def test_two_vessel_target_ids_in_timelines(self):
        csv = _csv(
            _row("V001", 0), _row("V001", 1),
            _row("V002", 0), _row("V002", 1),
        )
        result = replay_ais_file(csv)
        assert all(r["target_id"] == "V001" for r in result["V001"])
        assert all(r["target_id"] == "V002" for r in result["V002"])

    def test_two_vessel_custody_pipeline_is_independent(self):
        """V002's position must not affect V001's core custody pipeline fields.

        Proximity scores legitimately differ when V002 moves, so this test
        compares only the fields produced by ingest_ais_track, excluding the
        three proximity columns added by the post-processing pass.
        """
        _PROXIMITY_KEYS = {"vessel_proximity_score", "nearest_vessel_id", "nearest_vessel_km"}

        csv_a = _csv(
            _row("V001", 0, lat=0.0), _row("V001", 1, lat=0.01),
            _row("V002", 0, lat=0.5), _row("V002", 1, lat=0.51),
        )
        csv_b = _csv(
            _row("V001", 0, lat=0.0), _row("V001", 1, lat=0.01),
            _row("V002", 0, lat=5.0), _row("V002", 1, lat=5.01),
        )
        result_a = replay_ais_file(csv_a)
        result_b = replay_ais_file(csv_b)

        def _strip_proximity(timeline):
            return [{k: v for k, v in r.items() if k not in _PROXIMITY_KEYS} for r in timeline]

        assert _strip_proximity(result_a["V001"]) == _strip_proximity(result_b["V001"])

    def test_two_vessel_last_records_correct_timestamps(self):
        csv = _csv(
            _row("V001", 0), _row("V001", 2),
            _row("V002", 0), _row("V002", 3),
        )
        result = replay_ais_file(csv)
        assert result["V001"][-1]["time"] == T0 + timedelta(hours=2)
        assert result["V002"][-1]["time"] == T0 + timedelta(hours=3)


# ---------------------------------------------------------------------------
# Three vessels, unordered rows
# ---------------------------------------------------------------------------

class TestThreeVesselsUnordered:
    def test_interleaved_rows_correct_grouping(self):
        """Rows interleaved across vessels are grouped correctly."""
        csv = _csv(
            _row("C", 0), _row("A", 0), _row("B", 0),
            _row("A", 1), _row("C", 1), _row("B", 1),
            _row("B", 2), _row("A", 2), _row("C", 2),
        )
        result = replay_ais_file(csv)
        assert set(result.keys()) == {"A", "B", "C"}
        assert len(result["A"]) == 3
        assert len(result["B"]) == 3
        assert len(result["C"]) == 3

    def test_interleaved_target_ids_correct(self):
        csv = _csv(
            _row("C", 0), _row("A", 0), _row("B", 0),
            _row("A", 1), _row("C", 1), _row("B", 1),
        )
        result = replay_ais_file(csv)
        for vid in ("A", "B", "C"):
            assert all(r["target_id"] == vid for r in result[vid])

    def test_timestamps_sorted_within_each_vessel(self):
        """Even with shuffled input rows, each vessel's timeline is in time order."""
        csv = _csv(
            _row("V1", 2), _row("V2", 1), _row("V1", 0),
            _row("V2", 2), _row("V1", 1), _row("V2", 0),
        )
        result = replay_ais_file(csv)
        for vid in ("V1", "V2"):
            times = [r["time"] for r in result[vid]]
            assert times == sorted(times)


# ---------------------------------------------------------------------------
# Schema spot-check
# ---------------------------------------------------------------------------

class TestOutputSchema:
    _REQUIRED_KEYS = {
        "target_id", "time", "lat", "lon",
        "uncertainty_km", "custody_confidence", "anomaly_score",
        "speed_kmh", "heading_deg", "behavior_mode",
        "action", "action_reason", "sensor_id", "sensor_type", "collection_result",
        "sensitive_zone", "loitering", "route_deviation",
        "behavior_state", "state_confidence",
        # proximity fields — always present after replay_ais_file post-processing
        "vessel_proximity_score", "nearest_vessel_id", "nearest_vessel_km",
    }

    def test_all_required_keys_present_multi_vessel(self):
        csv = _csv(
            _row("V001", 0), _row("V001", 1),
            _row("V002", 0), _row("V002", 1),
        )
        result = replay_ais_file(csv)
        for vid, timeline in result.items():
            for record in timeline:
                missing = self._REQUIRED_KEYS - record.keys()
                assert not missing, f"vessel {vid}: record missing keys {missing}"

    def test_behavior_mode_is_observed(self):
        csv = _csv(_row("V001", 0), _row("V001", 1), _row("V002", 0), _row("V002", 1))
        result = replay_ais_file(csv)
        for timeline in result.values():
            for record in timeline:
                assert record["behavior_mode"] == "observed"


# ---------------------------------------------------------------------------
# Error propagation
# ---------------------------------------------------------------------------

class TestErrorPropagation:
    def test_missing_required_column_raises_value_error(self):
        bad_csv = "vessel_id,timestamp,lat,lon,speed_knots\n"  # missing heading_deg
        with pytest.raises(ValueError, match="missing required columns"):
            replay_ais_file(bad_csv)

    def test_empty_string_raises_os_error(self):
        # No newline → treated as a file path → OSError (FileNotFoundError subclass)
        with pytest.raises(OSError):
            replay_ais_file("")


# ---------------------------------------------------------------------------
# Proximity enrichment — Step 14d
# ---------------------------------------------------------------------------

_PROXIMITY_KEYS = {"vessel_proximity_score", "nearest_vessel_id", "nearest_vessel_km"}

# Roughly 3 km apart at equatorial latitudes (0.03° ≈ 3.3 km).
_CLOSE_LAT_OFFSET = 0.03
# Roughly 200 km apart (2.0° ≈ 222 km).
_FAR_LAT_OFFSET = 2.0


class TestProximityEnrichmentSingleVessel:
    def test_proximity_keys_present_in_all_records(self):
        csv = _csv(_row("V001", 0), _row("V001", 1), _row("V001", 2))
        result = replay_ais_file(csv)
        for record in result["V001"]:
            assert _PROXIMITY_KEYS <= record.keys(), \
                f"Missing proximity keys in record: {record.keys() - _PROXIMITY_KEYS}"

    def test_single_vessel_proximity_score_is_zero(self):
        csv = _csv(_row("V001", 0), _row("V001", 1))
        result = replay_ais_file(csv)
        for record in result["V001"]:
            assert record["vessel_proximity_score"] == 0.0

    def test_single_vessel_nearest_id_is_none(self):
        csv = _csv(_row("V001", 0), _row("V001", 1))
        result = replay_ais_file(csv)
        for record in result["V001"]:
            assert record["nearest_vessel_id"] is None

    def test_single_vessel_nearest_km_is_none(self):
        csv = _csv(_row("V001", 0), _row("V001", 1))
        result = replay_ais_file(csv)
        for record in result["V001"]:
            assert record["nearest_vessel_km"] is None


class TestProximityEnrichmentTwoVessels:
    def test_all_records_have_proximity_keys(self):
        csv = _csv(
            _row("V001", 0), _row("V001", 1),
            _row("V002", 0), _row("V002", 1),
        )
        result = replay_ais_file(csv)
        for vid, timeline in result.items():
            for record in timeline:
                assert _PROXIMITY_KEYS <= record.keys(), \
                    f"vessel {vid}: missing proximity keys"

    def test_close_vessels_get_nonzero_proximity_score(self):
        # Both vessels at the same timestamp, ~3 km apart — within warning range (15 km).
        csv = _csv(
            _row("V001", 0, lat=0.0, lon=0.0),
            _row("V001", 1, lat=0.0, lon=0.0),
            _row("V002", 0, lat=_CLOSE_LAT_OFFSET, lon=0.0),
            _row("V002", 1, lat=_CLOSE_LAT_OFFSET, lon=0.0),
        )
        result = replay_ais_file(csv)
        # At least one record for each vessel should have a nonzero proximity score.
        v001_scores = [r["vessel_proximity_score"] for r in result["V001"]]
        v002_scores = [r["vessel_proximity_score"] for r in result["V002"]]
        assert any(s > 0.0 for s in v001_scores), "V001 expected nonzero proximity"
        assert any(s > 0.0 for s in v002_scores), "V002 expected nonzero proximity"

    def test_close_vessels_nearest_id_is_the_other(self):
        csv = _csv(
            _row("V001", 0, lat=0.0, lon=0.0),
            _row("V002", 0, lat=_CLOSE_LAT_OFFSET, lon=0.0),
        )
        result = replay_ais_file(csv)
        # The record with proximity score > 0 should name the correct neighbour.
        v001_rec = result["V001"][0]
        v002_rec = result["V002"][0]
        if v001_rec["vessel_proximity_score"] > 0.0:
            assert v001_rec["nearest_vessel_id"] == "V002"
        if v002_rec["vessel_proximity_score"] > 0.0:
            assert v002_rec["nearest_vessel_id"] == "V001"

    def test_far_vessels_stay_at_zero_proximity(self):
        csv = _csv(
            _row("V001", 0, lat=0.0,            lon=0.0),
            _row("V001", 1, lat=0.0,            lon=0.0),
            _row("V002", 0, lat=_FAR_LAT_OFFSET, lon=0.0),
            _row("V002", 1, lat=_FAR_LAT_OFFSET, lon=0.0),
        )
        result = replay_ais_file(csv)
        for vid in ("V001", "V002"):
            for record in result[vid]:
                assert record["vessel_proximity_score"] == 0.0, \
                    f"vessel {vid} unexpectedly has proximity score > 0"

    def test_proximity_only_matches_same_timestamp(self):
        # V001 at T0, V002 only at T0+1 — they never share a timestamp.
        csv = _csv(
            _row("V001", 0, lat=0.0, lon=0.0),
            _row("V002", 1, lat=_CLOSE_LAT_OFFSET, lon=0.0),
        )
        result = replay_ais_file(csv)
        # No shared timestamps → proximity scores should all be zero.
        for vid in ("V001", "V002"):
            for record in result[vid]:
                assert record["vessel_proximity_score"] == 0.0, \
                    f"vessel {vid} got nonzero proximity despite no shared timestamps"

    def test_unsorted_input_proximity_correct(self):
        """Interleaved/unsorted rows must still produce correct proximity scores."""
        csv = _csv(
            _row("V002", 0, lat=_CLOSE_LAT_OFFSET, lon=0.0),
            _row("V001", 0, lat=0.0, lon=0.0),   # same timestamp, different order
            _row("V001", 1, lat=0.0, lon=0.0),
            _row("V002", 1, lat=_CLOSE_LAT_OFFSET, lon=0.0),
        )
        result = replay_ais_file(csv)
        v001_scores = [r["vessel_proximity_score"] for r in result["V001"]]
        v002_scores = [r["vessel_proximity_score"] for r in result["V002"]]
        assert any(s > 0.0 for s in v001_scores)
        assert any(s > 0.0 for s in v002_scores)
