"""
Tests for opt-in orbital sensor access during AIS replay.

observer_position_from_obs=False (default):
  - current behavior preserved exactly
  - orbital sensors (EO-MIO-1 / SAR-1 etc.) never appear
  - lookahead fields remain 0.0 / None

observer_position_from_obs=True:
  - each record's lat/lon is passed as observer position
  - orbital sensors appear when geometry allows
  - schedule-based sensors (A1/B1/C1) still appear as before
  - lookahead fields can be populated when a pass is nearby
  - planner actions remain valid; decision_trace is present

Coverage:
  1. Default mode preserves current outputs
  2. Default mode: no orbital sensors in any record
  3. Opt-in mode: orbital sensor appears when satellite in view
  4. Opt-in mode: no false positives when satellite not in view
  5. Decision trace present in both modes
  6. Planner action valid in both modes
  7. Lookahead boost is 0.0 / nearest_pass_tts is None in default mode
  8. Lookahead fields can be non-zero in opt-in mode when applicable
  9. replay_ais_file threads the flag to all vessels
 10. First record is always NONE (conservative bootstrap), both modes
"""
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from custody.ais import AISObservation, ingest_ais_track, replay_ais_file
from custody.decision_trace import DecisionTrace
from custody.sensors import PassWindow, SensorOpportunity

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# 14:00 on 2026-03-23 — near a known orbital window (SAR-1 in view at ~14:00 UTC at lat/lon ~0.5)
_T_NEAR = datetime(2026, 3, 23, 14, 0, tzinfo=UTC)
# 10:00 — outside any orbital window
_T_FAR = datetime(2026, 3, 23, 10, 0, tzinfo=UTC)

# Equatorial position used in most tests
_LAT, _LON = 0.5, 0.5

_VALID_ACTIONS = {"NONE", "TASK", "HOLD", "NO_SENSOR", "PREEMPTED"}


def _obs(vessel_id, timestamp, lat=_LAT, lon=_LON, speed_knots=5.0, heading_deg=0.0):
    return AISObservation(
        vessel_id=vessel_id,
        timestamp=timestamp,
        lat=lat,
        lon=lon,
        speed_knots=speed_knots,
        heading_deg=heading_deg,
    )


def _two_obs(t0=_T_FAR, lat=_LAT, lon=_LON, vid="V001"):
    """Minimal two-observation track — first record is bootstrap, second is planned."""
    return [
        _obs(vid, t0, lat=lat, lon=lon),
        _obs(vid, t0 + timedelta(hours=1), lat=lat, lon=lon),
    ]


# A fake EO-MIO-1 opportunity that get_sensor_opportunities would return when in view.
_FAKE_SAT_A_OPP = SensorOpportunity(
    sensor_id="EO-MIO-1",
    sensor_type="high_resolution",
    success_prob=0.70,
    resolution="high",
    cost=2.5,
    available_from=_T_NEAR,
    available_to=_T_NEAR,
    satellite_id="EO-MIO-1",
)

_FAKE_NEAR_PASS = PassWindow(
    satellite_id="EO-MIO-1",
    start_time=_T_NEAR + timedelta(minutes=15),
    end_time=_T_NEAR + timedelta(minutes=22),
    duration_seconds=420.0,
    time_to_start_seconds=900.0,
)


# ---------------------------------------------------------------------------
# 1. Default mode: output structure unchanged
# ---------------------------------------------------------------------------

class TestDefaultModePreservation:
    def test_returns_list(self):
        records = ingest_ais_track(_two_obs())
        assert isinstance(records, list)

    def test_record_count_matches_observations(self):
        obs = _two_obs()
        records = ingest_ais_track(obs)
        assert len(records) == len(obs)

    def test_required_keys_present(self):
        required = {
            "target_id", "time", "lat", "lon", "action", "sensor_id",
            "decision_trace", "action_reason",
        }
        for record in ingest_ais_track(_two_obs()):
            assert required <= set(record.keys())

    def test_explicit_false_identical_to_default(self):
        """Passing observer_position_from_obs=False is identical to the default."""
        obs = _two_obs()
        default_records = ingest_ais_track(obs)
        explicit_records = ingest_ais_track(obs, observer_position_from_obs=False)
        # Compare actions and sensor_ids across all records
        for d, e in zip(default_records, explicit_records):
            assert d["action"] == e["action"]
            assert d["sensor_id"] == e["sensor_id"]


# ---------------------------------------------------------------------------
# 2. Default mode: no orbital sensors
# ---------------------------------------------------------------------------

class TestDefaultModeNoOrbitalSensors:
    def _all_sensor_ids(self, records):
        return {r["sensor_id"] for r in records if r["sensor_id"] is not None}

    def test_sat_a_never_assigned_in_default_mode(self):
        obs = _two_obs(t0=_T_NEAR)
        records = ingest_ais_track(obs, observer_position_from_obs=False)
        assert "EO-MIO-1" not in self._all_sensor_ids(records)

    def test_sat_b_never_assigned_in_default_mode(self):
        obs = _two_obs(t0=_T_NEAR)
        records = ingest_ais_track(obs, observer_position_from_obs=False)
        assert "SAR-1" not in self._all_sensor_ids(records)

    def test_default_mode_uses_no_observer_position(self):
        """In default mode, get_sensor_opportunities is called without lat/lon."""
        obs = _two_obs()
        call_args = []

        def recording_get_opps(t, lat=None, lon=None):
            call_args.append((lat, lon))
            return []

        with patch("custody.ais.get_sensor_opportunities", side_effect=recording_get_opps):
            ingest_ais_track(obs, observer_position_from_obs=False)

        assert all(lat is None and lon is None for lat, lon in call_args)


# ---------------------------------------------------------------------------
# 3. Opt-in mode: orbital sensors appear when in view
# ---------------------------------------------------------------------------

class TestOptInOrbitalAccess:
    def test_sat_a_can_appear_in_opt_in_mode(self):
        """When EO-MIO-1 is in view and opt-in is enabled, it can be assigned."""
        obs = _two_obs(t0=_T_NEAR)

        def fake_opps(t, lat=None, lon=None):
            if lat is not None:
                return [_FAKE_SAT_A_OPP]
            return []

        with patch("custody.ais.get_sensor_opportunities", side_effect=fake_opps):
            records = ingest_ais_track(obs, observer_position_from_obs=True)

        sensor_ids = {r["sensor_id"] for r in records if r["sensor_id"] is not None}
        # EO-MIO-1 must appear in the opportunity pool; whether it's assigned depends
        # on planner thresholds — check the trace instead of the action directly.
        traces = [r["decision_trace"] for r in records if isinstance(r.get("decision_trace"), DecisionTrace)]
        accessible = {sid for t in traces for sid in t.arbitration.accessible_sensor_ids}
        assert "EO-MIO-1" in accessible

    def test_opt_in_passes_observer_position_to_get_sensor_opportunities(self):
        """With opt-in, lat/lon are forwarded (not None) for non-first records."""
        obs = _two_obs(lat=_LAT, lon=_LON)
        call_args = []

        def recording_get_opps(t, lat=None, lon=None):
            call_args.append((lat, lon))
            return []

        with patch("custody.ais.get_sensor_opportunities", side_effect=recording_get_opps):
            ingest_ais_track(obs, observer_position_from_obs=True)

        # Both calls (first + second record) should carry lat/lon
        assert all(lat is not None and lon is not None for lat, lon in call_args)

    def test_opt_in_passes_observer_position_to_plan_collection(self):
        """With opt-in, plan_collection receives observer_lat / observer_lon."""
        obs = _two_obs()
        plan_calls = []

        import custody.ais as ais_mod
        original_plan = ais_mod.plan_collection

        def recording_plan(*args, **kwargs):
            plan_calls.append(kwargs)
            return original_plan(*args, **kwargs)

        with patch("custody.ais.plan_collection", side_effect=recording_plan):
            ingest_ais_track(obs, observer_position_from_obs=True)

        assert plan_calls, "plan_collection was never called"
        for call in plan_calls:
            assert call.get("observer_lat") is not None
            assert call.get("observer_lon") is not None


# ---------------------------------------------------------------------------
# 4. Opt-in mode: no false positives when satellite not in view
# ---------------------------------------------------------------------------

class TestOptInNoFalsePositives:
    def test_no_orbital_sensor_when_not_in_view(self):
        """Opt-in with no orbital opportunities → no orbital sensor assigned."""
        obs = _two_obs(t0=_T_FAR)

        # Force get_sensor_opportunities to return nothing regardless of position
        with patch("custody.ais.get_sensor_opportunities", return_value=[]):
            records = ingest_ais_track(obs, observer_position_from_obs=True)

        sensor_ids = {r["sensor_id"] for r in records if r["sensor_id"] is not None}
        assert "EO-MIO-1" not in sensor_ids
        assert "SAR-1" not in sensor_ids


# ---------------------------------------------------------------------------
# 5. Decision trace present in both modes
# ---------------------------------------------------------------------------

class TestDecisionTracePresent:
    def test_trace_present_in_default_mode(self):
        for record in ingest_ais_track(_two_obs()):
            assert isinstance(record.get("decision_trace"), DecisionTrace)

    def test_trace_present_in_opt_in_mode(self):
        for record in ingest_ais_track(_two_obs(), observer_position_from_obs=True):
            assert isinstance(record.get("decision_trace"), DecisionTrace)


# ---------------------------------------------------------------------------
# 6. Planner action valid in both modes
# ---------------------------------------------------------------------------

class TestPlannerActionValid:
    def test_actions_valid_in_default_mode(self):
        for record in ingest_ais_track(_two_obs()):
            assert record["action"] in _VALID_ACTIONS

    def test_actions_valid_in_opt_in_mode(self):
        for record in ingest_ais_track(_two_obs(), observer_position_from_obs=True):
            assert record["action"] in _VALID_ACTIONS


# ---------------------------------------------------------------------------
# 7. Lookahead fields are 0.0 / None in default mode
# ---------------------------------------------------------------------------

class TestLookaheadDefaultMode:
    def test_lookahead_boost_zero_in_default_mode(self):
        """No observer position → nearest_orbital_pass is never called → boost stays 0."""
        obs = _two_obs(t0=_T_NEAR)
        records = ingest_ais_track(obs, observer_position_from_obs=False)
        for record in records:
            trace = record.get("decision_trace")
            if isinstance(trace, DecisionTrace):
                assert trace.task_value.lookahead_boost == 0.0

    def test_nearest_pass_tts_none_in_default_mode(self):
        obs = _two_obs(t0=_T_NEAR)
        records = ingest_ais_track(obs, observer_position_from_obs=False)
        for record in records:
            trace = record.get("decision_trace")
            if isinstance(trace, DecisionTrace):
                assert trace.task_value.nearest_pass_time_to_start_seconds is None


# ---------------------------------------------------------------------------
# 8. Lookahead fields can be populated in opt-in mode
# ---------------------------------------------------------------------------

class TestLookaheadOptInMode:
    def test_lookahead_boost_nonzero_when_pass_is_near(self):
        """Opt-in + near pass + no other sensors → lookahead_boost > 0."""
        obs = _two_obs(t0=_T_NEAR)

        # Empty opportunity pool so the lookahead branch can fire.
        with patch("custody.ais.get_sensor_opportunities", return_value=[]), \
             patch("custody.sensors.nearest_orbital_pass", return_value=_FAKE_NEAR_PASS):
            records = ingest_ais_track(obs, observer_position_from_obs=True)

        boosts = [
            record["decision_trace"].task_value.lookahead_boost
            for record in records
            if isinstance(record.get("decision_trace"), DecisionTrace)
        ]
        assert any(b > 0.0 for b in boosts), (
            "Expected at least one record with lookahead_boost > 0 when a near pass exists"
        )

    def test_nearest_pass_tts_populated_when_pass_is_near(self):
        obs = _two_obs(t0=_T_NEAR)

        with patch("custody.ais.get_sensor_opportunities", return_value=[]), \
             patch("custody.sensors.nearest_orbital_pass", return_value=_FAKE_NEAR_PASS):
            records = ingest_ais_track(obs, observer_position_from_obs=True)

        tts_values = [
            record["decision_trace"].task_value.nearest_pass_time_to_start_seconds
            for record in records
            if isinstance(record.get("decision_trace"), DecisionTrace)
            and record["decision_trace"].task_value.nearest_pass_time_to_start_seconds is not None
        ]
        assert tts_values, "Expected at least one record with nearest_pass_time_to_start_seconds set"


# ---------------------------------------------------------------------------
# 9. replay_ais_file threads the flag to all vessels
# ---------------------------------------------------------------------------

class TestReplayAisFileFlag:
    _CSV = "\n".join([
        "vessel_id,timestamp,lat,lon,speed_knots,heading_deg",
        f"V001,{_T_FAR.isoformat()},{_LAT},{_LON},5.0,0.0",
        f"V001,{(_T_FAR + timedelta(hours=1)).isoformat()},{_LAT},{_LON},5.0,0.0",
        f"V002,{_T_FAR.isoformat()},{_LAT},{_LON},5.0,0.0",
        f"V002,{(_T_FAR + timedelta(hours=1)).isoformat()},{_LAT},{_LON},5.0,0.0",
    ])

    def test_replay_default_returns_both_vessels(self):
        result = replay_ais_file(self._CSV)
        assert set(result.keys()) == {"V001", "V002"}

    def test_replay_opt_in_returns_both_vessels(self):
        result = replay_ais_file(self._CSV, observer_position_from_obs=True)
        assert set(result.keys()) == {"V001", "V002"}

    def test_replay_threads_flag_via_plan_collection_calls(self):
        """With opt-in, every plan_collection call receives observer lat/lon."""
        import custody.ais as ais_mod
        plan_calls = []
        original_plan = ais_mod.plan_collection

        def recording_plan(*args, **kwargs):
            plan_calls.append(kwargs)
            return original_plan(*args, **kwargs)

        with patch("custody.ais.plan_collection", side_effect=recording_plan):
            replay_ais_file(self._CSV, observer_position_from_obs=True)

        assert plan_calls, "plan_collection was never called"
        for call in plan_calls:
            assert call.get("observer_lat") is not None
            assert call.get("observer_lon") is not None

    def test_replay_default_no_observer_position_forwarded(self):
        """Default mode: plan_collection never receives observer position."""
        import custody.ais as ais_mod
        plan_calls = []
        original_plan = ais_mod.plan_collection

        def recording_plan(*args, **kwargs):
            plan_calls.append(kwargs)
            return original_plan(*args, **kwargs)

        with patch("custody.ais.plan_collection", side_effect=recording_plan):
            replay_ais_file(self._CSV, observer_position_from_obs=False)

        for call in plan_calls:
            assert call.get("observer_lat") is None
            assert call.get("observer_lon") is None


# ---------------------------------------------------------------------------
# 10. First record is always NONE in both modes
# ---------------------------------------------------------------------------

class TestFirstRecordBootstrap:
    def test_first_record_is_none_in_default_mode(self):
        obs = _two_obs()
        records = ingest_ais_track(obs, observer_position_from_obs=False)
        assert records[0]["action"] == "NONE"

    def test_first_record_is_none_in_opt_in_mode(self):
        obs = _two_obs()
        records = ingest_ais_track(obs, observer_position_from_obs=True)
        assert records[0]["action"] == "NONE"
