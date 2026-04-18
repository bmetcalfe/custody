"""
Tests for ingest_ais_track (Step 9c).

These tests verify that the AIS ingestion path correctly replays observations
through the custody pipeline and produces a timeline matching the established
output schema.  No simulation machinery is tested here.
"""
import random
from datetime import datetime, timezone, timedelta

import pytest

from custody.ais import AISObservation, ingest_ais_track
from custody.config import ZONES
from custody.models import BehaviorState

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

T0 = datetime(2026, 3, 23, 10, 0, tzinfo=timezone.utc)  # even hour → sensors available

_REQUIRED_COLUMNS = [
    "target_id", "time", "lat", "lon",
    "uncertainty_km", "custody_confidence", "anomaly_score",
    "speed_kmh", "heading_deg", "behavior_mode",
    "action", "action_reason", "sensor_id", "sensor_type", "collection_result",
    "sensitive_zone", "loitering", "route_deviation",
    "behavior_state", "state_confidence",
]

_VALID_BEHAVIOR_STATES = {m.value for m in BehaviorState}
_ALLOWED_CONFIDENCE_LEVELS = {0.2, 0.6, 0.85}


def _obs(vessel_id, timestamp, lat, lon, speed_knots=10.0, heading_deg=45.0):
    return AISObservation(
        vessel_id=vessel_id,
        timestamp=timestamp,
        lat=lat,
        lon=lon,
        speed_knots=speed_knots,
        heading_deg=heading_deg,
    )


def _normal_track(n=5, vessel_id="V001", spacing_hours=1):
    """n observations outside the sensitive zone, 1 hour apart by default."""
    return [
        _obs(vessel_id, T0 + timedelta(hours=i * spacing_hours), lat=0.0, lon=0.0)
        for i in range(n)
    ]


def _zone_lat():
    """A latitude inside the first configured zone."""
    z = ZONES[0]
    return (z.min_lat + z.max_lat) / 2


def _zone_lon():
    """A longitude inside the first configured zone."""
    z = ZONES[0]
    return (z.min_lon + z.max_lon) / 2


# ---------------------------------------------------------------------------
# Basic contract
# ---------------------------------------------------------------------------

def test_ingest_returns_records():
    assert len(ingest_ais_track(_normal_track())) > 0


def test_ingest_record_count_matches_observation_count():
    obs = _normal_track(n=7)
    assert len(ingest_ais_track(obs)) == 7


def test_ingest_empty_list_returns_empty():
    assert ingest_ais_track([]) == []


def test_ingest_single_observation_returns_one_record():
    obs = [_obs("V001", T0, 0.0, 0.0)]
    assert len(ingest_ais_track(obs)) == 1


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

def test_ingest_output_has_required_columns():
    records = ingest_ais_track(_normal_track())
    for col in _REQUIRED_COLUMNS:
        assert col in records[0], f"Missing column: {col}"


def test_ingest_all_records_have_required_columns():
    records = ingest_ais_track(_normal_track(n=5))
    for i, rec in enumerate(records):
        missing = [c for c in _REQUIRED_COLUMNS if c not in rec]
        assert not missing, f"Record {i} missing columns: {missing}"


def test_ingest_target_id_in_every_record():
    records = ingest_ais_track(_normal_track())
    assert all(r["target_id"] == "V001" for r in records)


def test_ingest_behavior_mode_is_observed_in_every_record():
    records = ingest_ais_track(_normal_track())
    assert all(r["behavior_mode"] == "observed" for r in records)


# ---------------------------------------------------------------------------
# First observation handling
# ---------------------------------------------------------------------------

def test_ingest_first_record_action_is_none():
    records = ingest_ais_track(_normal_track())
    assert records[0]["action"] == "NONE"


def test_ingest_first_record_behavior_state_is_unknown():
    records = ingest_ais_track(_normal_track())
    assert records[0]["behavior_state"] == BehaviorState.UNKNOWN.value


def test_ingest_first_record_state_confidence_is_0_2():
    records = ingest_ais_track(_normal_track())
    assert records[0]["state_confidence"] == pytest.approx(0.2)


def test_ingest_first_record_anomaly_score_is_non_negative():
    records = ingest_ais_track(_normal_track())
    assert records[0]["anomaly_score"] >= 0.0


# ---------------------------------------------------------------------------
# Uncertainty growth (elapsed time from timestamps)
# ---------------------------------------------------------------------------

def test_ingest_uncertainty_grows_between_observations():
    obs = _normal_track(n=4)
    records = ingest_ais_track(obs)
    uncertainties = [r["uncertainty_km"] for r in records]
    # Without a successful collection, each step must be >= the previous.
    # (Collection can reduce uncertainty, so we only assert non-decreasing when
    #  action != "SUCCESS" — but for a normal low-anomaly track far from the
    #  zone, no tasking fires and uncertainty grows monotonically.)
    for i in range(1, len(uncertainties)):
        assert uncertainties[i] >= uncertainties[i - 1], (
            f"Uncertainty decreased at step {i} without collection: "
            f"{uncertainties[i - 1]} → {uncertainties[i]}"
        )


def test_ingest_uncertainty_growth_uses_elapsed_hours_not_fixed_one():
    """30-minute intervals should grow uncertainty at half the 1-hour rate."""
    obs_30min = _normal_track(n=3, spacing_hours=0.5)
    obs_1hr = _normal_track(n=3, spacing_hours=1.0)

    records_30min = ingest_ais_track(obs_30min)
    records_1hr = ingest_ais_track(obs_1hr)

    # After 2 steps from the same start uncertainty, the 30-min track should
    # have grown less than the 1-hour track.
    assert records_30min[-1]["uncertainty_km"] < records_1hr[-1]["uncertainty_km"]


# ---------------------------------------------------------------------------
# Anomaly scoring
# ---------------------------------------------------------------------------

def test_ingest_anomaly_score_is_non_negative_all_records():
    records = ingest_ais_track(_normal_track(n=5))
    assert all(r["anomaly_score"] >= 0.0 for r in records)


def test_ingest_high_anomaly_obs_inside_zone_considered_for_tasking():
    """An observation inside SENSITIVE_ZONE should trigger non-NONE action."""
    random.seed(99)  # deterministic collection outcome
    obs = [
        _obs("V001", T0, lat=0.0, lon=0.0),                              # baseline, outside zone
        _obs("V001", T0 + timedelta(hours=1), lat=0.0, lon=0.0),         # outside zone
        _obs("V001", T0 + timedelta(hours=2),                            # inside zone, even hour → sensors available
             lat=_zone_lat(), lon=_zone_lon()),
    ]
    records = ingest_ais_track(obs)
    zone_record = records[2]
    assert zone_record["action"] in {"TASK", "HOLD", "NO_SENSOR"}, (
        f"Expected tasking consideration for in-zone observation; got action={zone_record['action']!r}"
    )


# ---------------------------------------------------------------------------
# Behavior state and confidence
# ---------------------------------------------------------------------------

def test_ingest_behavior_state_is_valid_enum_value_all_records():
    records = ingest_ais_track(_normal_track(n=5))
    for i, rec in enumerate(records):
        assert rec["behavior_state"] in _VALID_BEHAVIOR_STATES, (
            f"Record {i} has invalid behavior_state: {rec['behavior_state']!r}"
        )


def test_ingest_state_confidence_is_allowed_level_all_records():
    records = ingest_ais_track(_normal_track(n=5))
    for i, rec in enumerate(records):
        assert rec["state_confidence"] in _ALLOWED_CONFIDENCE_LEVELS, (
            f"Record {i} has unexpected state_confidence: {rec['state_confidence']}"
        )


# ---------------------------------------------------------------------------
# Mixed vessel_id validation
# ---------------------------------------------------------------------------

def test_ingest_mixed_vessel_ids_raises_value_error():
    obs = [
        _obs("V001", T0, 0.0, 0.0),
        _obs("V002", T0 + timedelta(hours=1), 0.1, 0.1),
    ]
    with pytest.raises(ValueError, match="mixed vessel_ids"):
        ingest_ais_track(obs)


def test_ingest_mixed_vessel_ids_error_lists_ids():
    obs = [
        _obs("ALPHA", T0, 0.0, 0.0),
        _obs("BETA", T0 + timedelta(hours=1), 0.0, 0.0),
    ]
    with pytest.raises(ValueError, match="ALPHA"):
        ingest_ais_track(obs)


# ---------------------------------------------------------------------------
# Input ordering
# ---------------------------------------------------------------------------

def test_ingest_sorts_input_by_timestamp():
    """Out-of-order input should produce a correctly ordered timeline."""
    obs = [
        _obs("V001", T0 + timedelta(hours=2), 0.2, 0.0),
        _obs("V001", T0, 0.0, 0.0),
        _obs("V001", T0 + timedelta(hours=1), 0.1, 0.0),
    ]
    records = ingest_ais_track(obs)
    times = [r["time"] for r in records]
    assert times == sorted(times)
