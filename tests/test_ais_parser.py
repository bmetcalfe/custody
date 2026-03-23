"""
Tests for AISObservation and parse_ais_csv (Step 9a).

These tests cover only the parser layer: data types, unit conversion,
sorting, malformed-row handling, and column validation.  No custody
pipeline logic is exercised here.
"""
import warnings
from datetime import datetime, timezone

import pytest

from custody.ais import AISObservation, observation_to_history_entry, observation_to_vessel, parse_ais_csv
from custody.models import HistoryEntry, Vessel


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

MINIMAL_CSV = """\
vessel_id,timestamp,lat,lon,speed_knots,heading_deg
V001,2026-03-23T10:00:00+00:00,0.0,0.0,10.0,45.0
V001,2026-03-23T11:00:00+00:00,0.1,0.1,10.0,45.0
V001,2026-03-23T12:00:00+00:00,0.2,0.2,10.0,45.0
"""

MULTI_VESSEL_CSV = """\
vessel_id,timestamp,lat,lon,speed_knots,heading_deg
V002,2026-03-23T12:00:00+00:00,0.5,0.5,8.0,90.0
V001,2026-03-23T11:00:00+00:00,0.1,0.1,10.0,45.0
V002,2026-03-23T10:00:00+00:00,0.4,0.4,8.0,90.0
V001,2026-03-23T10:00:00+00:00,0.0,0.0,10.0,45.0
"""


# ---------------------------------------------------------------------------
# parse_ais_csv — basic parsing
# ---------------------------------------------------------------------------

def test_parse_minimal_csv_returns_three_records():
    obs = parse_ais_csv(MINIMAL_CSV)
    assert len(obs) == 3


def test_all_records_are_ais_observations():
    obs = parse_ais_csv(MINIMAL_CSV)
    assert all(isinstance(o, AISObservation) for o in obs)


def test_vessel_id_preserved_as_string():
    obs = parse_ais_csv(MINIMAL_CSV)
    assert all(o.vessel_id == "V001" for o in obs)


def test_lat_lon_parsed_correctly():
    obs = parse_ais_csv(MINIMAL_CSV)
    assert obs[0].lat == pytest.approx(0.0)
    assert obs[0].lon == pytest.approx(0.0)
    assert obs[1].lat == pytest.approx(0.1)


def test_speed_knots_stored_as_raw_value():
    obs = parse_ais_csv(MINIMAL_CSV)
    assert obs[0].speed_knots == pytest.approx(10.0)


def test_heading_deg_parsed_correctly():
    obs = parse_ais_csv(MINIMAL_CSV)
    assert obs[0].heading_deg == pytest.approx(45.0)


# ---------------------------------------------------------------------------
# Timestamp handling
# ---------------------------------------------------------------------------

def test_timestamp_is_utc_aware():
    obs = parse_ais_csv(MINIMAL_CSV)
    for o in obs:
        assert o.timestamp.tzinfo is not None


def test_timestamp_is_correct_value():
    obs = parse_ais_csv(MINIMAL_CSV)
    expected = datetime(2026, 3, 23, 10, 0, tzinfo=timezone.utc)
    assert obs[0].timestamp == expected


def test_naive_timestamp_is_treated_as_utc():
    csv_text = (
        "vessel_id,timestamp,lat,lon,speed_knots,heading_deg\n"
        "V001,2026-03-23T10:00:00,0.0,0.0,10.0,45.0\n"
    )
    obs = parse_ais_csv(csv_text)
    assert obs[0].timestamp.tzinfo is not None
    assert obs[0].timestamp == datetime(2026, 3, 23, 10, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# speed_kmh property
# ---------------------------------------------------------------------------

def test_speed_kmh_converts_knots():
    obs = parse_ais_csv(MINIMAL_CSV)
    assert obs[0].speed_kmh == pytest.approx(10.0 * 1.852)


def test_speed_kmh_zero_knots():
    csv_text = (
        "vessel_id,timestamp,lat,lon,speed_knots,heading_deg\n"
        "V001,2026-03-23T10:00:00+00:00,0.0,0.0,0.0,45.0\n"
    )
    obs = parse_ais_csv(csv_text)
    assert obs[0].speed_kmh == pytest.approx(0.0)


def test_speed_kmh_is_derived_not_stored():
    # AISObservation is frozen; speed_kmh is a computed property, not a field.
    obs = AISObservation(
        vessel_id="X",
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        lat=0.0, lon=0.0,
        speed_knots=5.0,
        heading_deg=0.0,
    )
    assert obs.speed_kmh == pytest.approx(5.0 * 1.852)


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------

def test_output_sorted_by_vessel_id_then_timestamp():
    obs = parse_ais_csv(MULTI_VESSEL_CSV)
    vessel_ids = [o.vessel_id for o in obs]
    timestamps = [o.timestamp for o in obs]
    # All V001 before V002
    assert vessel_ids == ["V001", "V001", "V002", "V002"]
    # Within each vessel, ascending time
    assert timestamps[0] < timestamps[1]
    assert timestamps[2] < timestamps[3]


def test_single_vessel_sorted_by_timestamp():
    csv_text = (
        "vessel_id,timestamp,lat,lon,speed_knots,heading_deg\n"
        "V001,2026-03-23T12:00:00+00:00,0.2,0.2,10.0,45.0\n"
        "V001,2026-03-23T10:00:00+00:00,0.0,0.0,10.0,45.0\n"
        "V001,2026-03-23T11:00:00+00:00,0.1,0.1,10.0,45.0\n"
    )
    obs = parse_ais_csv(csv_text)
    times = [o.timestamp for o in obs]
    assert times == sorted(times)


# ---------------------------------------------------------------------------
# Malformed row handling
# ---------------------------------------------------------------------------

def test_malformed_row_is_skipped_not_raised():
    csv_text = (
        "vessel_id,timestamp,lat,lon,speed_knots,heading_deg\n"
        "V001,2026-03-23T10:00:00+00:00,0.0,0.0,10.0,45.0\n"
        "V001,2026-03-23T11:00:00+00:00,NOT_A_FLOAT,0.1,10.0,45.0\n"  # bad lat
        "V001,2026-03-23T12:00:00+00:00,0.2,0.2,10.0,45.0\n"
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        obs = parse_ais_csv(csv_text)
    assert len(obs) == 2
    assert any("malformed" in str(w.message).lower() for w in caught)


def test_malformed_timestamp_is_skipped_with_warning():
    csv_text = (
        "vessel_id,timestamp,lat,lon,speed_knots,heading_deg\n"
        "V001,NOT_A_TIMESTAMP,0.0,0.0,10.0,45.0\n"
        "V001,2026-03-23T11:00:00+00:00,0.1,0.1,10.0,45.0\n"
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        obs = parse_ais_csv(csv_text)
    assert len(obs) == 1
    assert len(caught) == 1


def test_warning_includes_row_number():
    csv_text = (
        "vessel_id,timestamp,lat,lon,speed_knots,heading_deg\n"
        "V001,2026-03-23T10:00:00+00:00,BAD,0.0,10.0,45.0\n"
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        parse_ais_csv(csv_text)
    assert any("2" in str(w.message) for w in caught)  # header is row 1, data row is row 2


# ---------------------------------------------------------------------------
# Missing required columns
# ---------------------------------------------------------------------------

def test_missing_required_column_raises_value_error():
    csv_text = (
        "vessel_id,timestamp,lat,lon,speed_knots\n"  # heading_deg missing
        "V001,2026-03-23T10:00:00+00:00,0.0,0.0,10.0\n"
    )
    with pytest.raises(ValueError, match="missing required columns"):
        parse_ais_csv(csv_text)


def test_error_message_names_missing_column():
    csv_text = (
        "vessel_id,timestamp,lat,lon,heading_deg\n"  # speed_knots missing
        "V001,2026-03-23T10:00:00+00:00,0.0,0.0,45.0\n"
    )
    with pytest.raises(ValueError, match="speed_knots"):
        parse_ais_csv(csv_text)


# ---------------------------------------------------------------------------
# observation_to_vessel
# ---------------------------------------------------------------------------

_OBS = AISObservation(
    vessel_id="V001",
    timestamp=datetime(2026, 3, 23, 10, 0, tzinfo=timezone.utc),
    lat=1.3,
    lon=0.7,
    speed_knots=10.0,
    heading_deg=90.0,
)


def test_observation_to_vessel_returns_vessel():
    assert isinstance(observation_to_vessel(_OBS), Vessel)


def test_observation_to_vessel_id():
    assert observation_to_vessel(_OBS).id == "V001"


def test_observation_to_vessel_lat():
    assert observation_to_vessel(_OBS).lat == pytest.approx(1.3)


def test_observation_to_vessel_lon():
    assert observation_to_vessel(_OBS).lon == pytest.approx(0.7)


def test_observation_to_vessel_uses_speed_kmh_not_knots():
    v = observation_to_vessel(_OBS)
    assert v.speed_kmh == pytest.approx(10.0 * 1.852)
    assert v.speed_kmh != pytest.approx(10.0)  # raw knots value must not appear


def test_observation_to_vessel_heading():
    assert observation_to_vessel(_OBS).heading_deg == pytest.approx(90.0)


def test_observation_to_vessel_last_seen():
    assert observation_to_vessel(_OBS).last_seen == datetime(2026, 3, 23, 10, 0, tzinfo=timezone.utc)


def test_observation_to_vessel_last_seen_is_timezone_aware():
    assert observation_to_vessel(_OBS).last_seen.tzinfo is not None


def test_observation_to_vessel_history_is_empty():
    assert observation_to_vessel(_OBS).history == []


# ---------------------------------------------------------------------------
# observation_to_history_entry
# ---------------------------------------------------------------------------

def test_observation_to_history_entry_returns_history_entry():
    assert isinstance(observation_to_history_entry(_OBS), HistoryEntry)


def test_observation_to_history_entry_lat():
    assert observation_to_history_entry(_OBS).lat == pytest.approx(1.3)


def test_observation_to_history_entry_lon():
    assert observation_to_history_entry(_OBS).lon == pytest.approx(0.7)


def test_observation_to_history_entry_timestamp():
    assert observation_to_history_entry(_OBS).timestamp == datetime(2026, 3, 23, 10, 0, tzinfo=timezone.utc)


def test_observation_to_history_entry_timestamp_is_timezone_aware():
    assert observation_to_history_entry(_OBS).timestamp.tzinfo is not None


def test_observation_to_history_entry_uses_speed_kmh_not_knots():
    entry = observation_to_history_entry(_OBS)
    assert entry.speed_kmh == pytest.approx(10.0 * 1.852)
    assert entry.speed_kmh != pytest.approx(10.0)


def test_observation_to_history_entry_heading():
    assert observation_to_history_entry(_OBS).heading_deg == pytest.approx(90.0)


# ---------------------------------------------------------------------------
# AISObservation is immutable
# ---------------------------------------------------------------------------

def test_ais_observation_is_immutable():
    obs = AISObservation(
        vessel_id="V001",
        timestamp=datetime(2026, 3, 23, 10, tzinfo=timezone.utc),
        lat=0.0, lon=0.0,
        speed_knots=10.0,
        heading_deg=45.0,
    )
    with pytest.raises((AttributeError, TypeError)):
        obs.lat = 99.0  # type: ignore[misc]
