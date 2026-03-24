"""
Tests for the orbital-pass lookahead HOLD bias in plan_collection.

The lookahead fires ONLY when:
  - the sensor pool is empty (not due to preemption)
  - the observer position is known (observer_lat/lon provided)
  - the nearest upcoming orbital pass is within HOLD_LOOKAHEAD_THRESHOLD_SECONDS

Effect: NO_SENSOR is converted to HOLD so the planner signals "wait for the
incoming orbital window" rather than "no sensors exist".

Coverage:
  1. No change when a sensor is already available now
  2. No change when no future orbital pass exists within the boost threshold
  3. HOLD bias applies when nearest orbital pass is within threshold
  4. HOLD bias does NOT apply when nearest pass is outside threshold
  5. Among multiple orbital satellites the nearest upcoming pass is used
  6. Decision trace (TaskValueBreakdown) records lookahead_boost and
     nearest_pass_time_to_start_seconds correctly
  7. Existing planner behaviour is unchanged when observer position is omitted
  8. Preempted vessels are not converted to HOLD by the lookahead
  9. compute_task_value lookahead_boost is additive with other components
 10. traces_to_rows includes Lookahead Boost and Nearest Pass TTS columns
"""
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from custody.decision_trace import build_decision_trace, traces_to_rows
from custody.models import TrackState
from custody.planner import compute_task_value, plan_collection
from custody.sensors import PassWindow, SensorOpportunity, nearest_orbital_pass
import custody.planner as planner_module
import custody.config as config

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# (0.5, 0.5) is the observer from test_orbit_forecast:
#   SAT-A visible ~14:15–14:22 UTC on 2026-03-23  (tts from 13:00 ≈ 75 min)
#   SAT-A tts from 14:00 ≈ 15 min  → within HOLD_LOOKAHEAD_THRESHOLD_SECONDS=1800s
_OBS_LAT = 0.5
_OBS_LON = 0.5

# 13:00 — both passes are > 30 min away (SAT-A ~75 min, SAT-B ~115 min)
_T_FAR = datetime(2026, 3, 23, 13, 0, tzinfo=UTC)

# 14:00 — SAT-A pass is ~15 min away → within 30-min threshold
_T_NEAR = datetime(2026, 3, 23, 14, 0, tzinfo=UTC)

_NOMINAL_BD = {
    "sensitive_zone":  type("R", (), {"score": 0.0})(),
    "loitering":       type("R", (), {"score": 0.0})(),
    "route_deviation": type("R", (), {"score": 0.0})(),
}

_ONE_SENSOR = [
    SensorOpportunity(
        sensor_id="A1",
        sensor_type="fast_revisit",
        success_prob=1.0,
        resolution="medium",
        cost=1.0,
        available_from=_T_FAR,
        available_to=_T_FAR,
    )
]

# A fake near pass (15 min away) for mock-based tests
_FAKE_NEAR_PASS = PassWindow(
    satellite_id="SAT-A",
    start_time=_T_NEAR + timedelta(minutes=15),
    end_time=_T_NEAR + timedelta(minutes=22),
    duration_seconds=420.0,
    time_to_start_seconds=900.0,  # 15 min — within 1800s threshold
)

# A fake far pass (45 min away) for mock-based tests
_FAKE_FAR_PASS = PassWindow(
    satellite_id="SAT-A",
    start_time=_T_FAR + timedelta(minutes=45),
    end_time=_T_FAR + timedelta(minutes=52),
    duration_seconds=420.0,
    time_to_start_seconds=2700.0,  # 45 min — outside 1800s threshold
)


def _fresh_track() -> TrackState:
    return TrackState()


def _collected_track(hours_ago: float, anomaly_at_collection: float = 0.8) -> TrackState:
    track = TrackState()
    track.record_collection(
        _T_FAR - timedelta(hours=hours_ago), anomaly_at_collection, new_uncertainty=4.0
    )
    return track


# ---------------------------------------------------------------------------
# 1. No change when a sensor is currently available
# ---------------------------------------------------------------------------

class TestSensorAvailable:
    def test_lookahead_not_triggered_when_sensor_present(self):
        """When opportunities is non-empty, lookahead is never computed."""
        # Even with observer position at a time when a pass would be imminent,
        # the lookahead must not fire because a sensor IS available.
        with patch("custody.sensors.nearest_orbital_pass",
                   side_effect=AssertionError("should not be called")):
            decision = plan_collection(
                _fresh_track(), score=1.0, confidence=0.4,
                breakdown=_NOMINAL_BD, current_time=_T_NEAR,
                opportunities=_ONE_SENSOR,
                observer_lat=_OBS_LAT, observer_lon=_OBS_LON,
            )
        # action is whatever the planner decides (TASK) — not the point here
        assert decision.action != "NO_SENSOR"

    def test_action_unaffected_by_observer_position_when_sensor_present(self):
        """Providing observer_lat/lon has no effect when a sensor is available."""
        without = plan_collection(
            _fresh_track(), score=1.0, confidence=0.4,
            breakdown=_NOMINAL_BD, current_time=_T_NEAR,
            opportunities=_ONE_SENSOR,
        )
        with_pos = plan_collection(
            _fresh_track(), score=1.0, confidence=0.4,
            breakdown=_NOMINAL_BD, current_time=_T_NEAR,
            opportunities=_ONE_SENSOR,
            observer_lat=_OBS_LAT, observer_lon=_OBS_LON,
        )
        assert without.action == with_pos.action


# ---------------------------------------------------------------------------
# 2. No change when no future orbital pass exists within the boost threshold
# ---------------------------------------------------------------------------

class TestNoPassWithinThreshold:
    def test_no_sensor_when_nearest_pass_is_beyond_threshold(self):
        """When the nearest upcoming orbital pass is beyond the threshold, NO_SENSOR."""
        with patch("custody.sensors.nearest_orbital_pass",
                          return_value=_FAKE_FAR_PASS):
            decision = plan_collection(
                _fresh_track(), score=1.0, confidence=0.4,
                breakdown=_NOMINAL_BD, current_time=_T_FAR,
                opportunities=[],
                observer_lat=_OBS_LAT, observer_lon=_OBS_LON,
            )
        assert decision.action == "NO_SENSOR"

    def test_no_sensor_when_nearest_orbital_pass_is_none(self):
        """When nearest_orbital_pass returns None, NO_SENSOR is unchanged."""
        with patch("custody.sensors.nearest_orbital_pass", return_value=None):
            decision = plan_collection(
                _fresh_track(), score=1.0, confidence=0.4,
                breakdown=_NOMINAL_BD, current_time=_T_FAR,
                opportunities=[],
                observer_lat=_OBS_LAT, observer_lon=_OBS_LON,
            )
        assert decision.action == "NO_SENSOR"

    def test_real_orbital_timing_far_is_no_sensor(self):
        """Integration: at 13:00, both orbital passes are >30 min away → NO_SENSOR."""
        decision = plan_collection(
            _fresh_track(), score=1.0, confidence=0.4,
            breakdown=_NOMINAL_BD, current_time=_T_FAR,
            opportunities=[],
            observer_lat=_OBS_LAT, observer_lon=_OBS_LON,
        )
        assert decision.action == "NO_SENSOR"


# ---------------------------------------------------------------------------
# 3. HOLD bias applies when nearest orbital pass is within threshold
# ---------------------------------------------------------------------------

class TestHoldBiasApplies:
    def test_hold_when_pass_is_imminent_mock(self):
        """Mock: nearest pass within threshold → HOLD instead of NO_SENSOR."""
        with patch("custody.sensors.nearest_orbital_pass",
                          return_value=_FAKE_NEAR_PASS):
            decision = plan_collection(
                _fresh_track(), score=1.0, confidence=0.4,
                breakdown=_NOMINAL_BD, current_time=_T_NEAR,
                opportunities=[],
                observer_lat=_OBS_LAT, observer_lon=_OBS_LON,
            )
        assert decision.action == "HOLD"

    def test_hold_action_reason_mentions_orbital_window(self):
        """HOLD reason from lookahead should reference the orbital window."""
        with patch("custody.sensors.nearest_orbital_pass",
                          return_value=_FAKE_NEAR_PASS):
            decision = plan_collection(
                _fresh_track(), score=1.0, confidence=0.4,
                breakdown=_NOMINAL_BD, current_time=_T_NEAR,
                opportunities=[],
                observer_lat=_OBS_LAT, observer_lon=_OBS_LON,
            )
        assert "orbital" in decision.action_reason.lower()

    def test_real_orbital_timing_near_is_hold(self):
        """Integration: at 14:00, SAT-A pass is ~15 min away → HOLD."""
        decision = plan_collection(
            _fresh_track(), score=1.0, confidence=0.4,
            breakdown=_NOMINAL_BD, current_time=_T_NEAR,
            opportunities=[],
            observer_lat=_OBS_LAT, observer_lon=_OBS_LON,
        )
        assert decision.action == "HOLD"

    def test_hold_fires_even_without_prior_collection(self):
        """Lookahead HOLD fires on a fresh track (no prior collection) too."""
        with patch("custody.sensors.nearest_orbital_pass",
                          return_value=_FAKE_NEAR_PASS):
            decision = plan_collection(
                _fresh_track(), score=1.0, confidence=0.4,
                breakdown=_NOMINAL_BD, current_time=_T_NEAR,
                opportunities=[],
                observer_lat=_OBS_LAT, observer_lon=_OBS_LON,
            )
        assert decision.action == "HOLD"


# ---------------------------------------------------------------------------
# 4. No HOLD bias when nearest pass is outside threshold
# ---------------------------------------------------------------------------

class TestHoldBiasDoesNotApply:
    def test_no_sensor_at_threshold_boundary(self):
        """A pass just beyond the threshold (threshold + 1s) → NO_SENSOR."""
        just_outside = PassWindow(
            satellite_id="SAT-A",
            start_time=_T_FAR + timedelta(seconds=config.HOLD_LOOKAHEAD_THRESHOLD_SECONDS + 60),
            end_time=_T_FAR + timedelta(seconds=config.HOLD_LOOKAHEAD_THRESHOLD_SECONDS + 480),
            duration_seconds=420.0,
            time_to_start_seconds=config.HOLD_LOOKAHEAD_THRESHOLD_SECONDS + 60,
        )
        with patch("custody.sensors.nearest_orbital_pass",
                          return_value=just_outside):
            decision = plan_collection(
                _fresh_track(), score=1.0, confidence=0.4,
                breakdown=_NOMINAL_BD, current_time=_T_FAR,
                opportunities=[],
                observer_lat=_OBS_LAT, observer_lon=_OBS_LON,
            )
        assert decision.action == "NO_SENSOR"

    def test_hold_fires_at_threshold_boundary(self):
        """A pass exactly at the threshold → HOLD (condition is <=)."""
        at_threshold = PassWindow(
            satellite_id="SAT-A",
            start_time=_T_FAR + timedelta(seconds=config.HOLD_LOOKAHEAD_THRESHOLD_SECONDS),
            end_time=_T_FAR + timedelta(seconds=config.HOLD_LOOKAHEAD_THRESHOLD_SECONDS + 420),
            duration_seconds=420.0,
            time_to_start_seconds=config.HOLD_LOOKAHEAD_THRESHOLD_SECONDS,
        )
        with patch("custody.sensors.nearest_orbital_pass",
                          return_value=at_threshold):
            decision = plan_collection(
                _fresh_track(), score=1.0, confidence=0.4,
                breakdown=_NOMINAL_BD, current_time=_T_FAR,
                opportunities=[],
                observer_lat=_OBS_LAT, observer_lon=_OBS_LON,
            )
        assert decision.action == "HOLD"


# ---------------------------------------------------------------------------
# 5. Among multiple orbital satellites, the nearest pass is used
# ---------------------------------------------------------------------------

class TestNearestPassSelection:
    def test_nearest_orbital_pass_returns_soonest(self):
        """nearest_orbital_pass returns the satellite with the smallest tts."""
        near = PassWindow(
            satellite_id="SAT-A",
            start_time=_T_FAR + timedelta(minutes=10),
            end_time=_T_FAR + timedelta(minutes=17),
            duration_seconds=420.0,
            time_to_start_seconds=600.0,
        )
        far = PassWindow(
            satellite_id="SAT-B",
            start_time=_T_FAR + timedelta(minutes=25),
            end_time=_T_FAR + timedelta(minutes=32),
            duration_seconds=420.0,
            time_to_start_seconds=1500.0,
        )
        # Both within threshold — nearest should be returned.
        # side_effect covers all 6 catalog entries (SAT-A, SAT-A2, SAT-A3, SAT-B, SAT-B2, SAT-B3).
        with patch("custody.sensors.next_pass_window", side_effect=[near, None, None, far, None, None]):
            result = nearest_orbital_pass(_OBS_LAT, _OBS_LON, _T_FAR)
        assert result is not None
        assert result.satellite_id == "SAT-A"
        assert result.time_to_start_seconds == 600.0

    def test_nearest_orbital_pass_ignores_none(self):
        """If one satellite has no upcoming pass, the other's window is used."""
        far = PassWindow(
            satellite_id="SAT-B",
            start_time=_T_FAR + timedelta(minutes=20),
            end_time=_T_FAR + timedelta(minutes=27),
            duration_seconds=420.0,
            time_to_start_seconds=1200.0,
        )
        # side_effect covers all 6 catalog entries (SAT-A, SAT-A2, SAT-A3, SAT-B, SAT-B2, SAT-B3).
        with patch("custody.sensors.next_pass_window", side_effect=[None, None, None, far, None, None]):
            result = nearest_orbital_pass(_OBS_LAT, _OBS_LON, _T_FAR)
        assert result is not None
        assert result.satellite_id == "SAT-B"

    def test_nearest_orbital_pass_returns_none_when_all_none(self):
        """nearest_orbital_pass returns None when no satellite has an upcoming pass."""
        with patch("custody.sensors.next_pass_window", return_value=None):
            result = nearest_orbital_pass(_OBS_LAT, _OBS_LON, _T_FAR)
        assert result is None

    def test_lookahead_uses_nearest_satellite(self):
        """When two satellites have passes, the nearest one determines the boost."""
        near = _FAKE_NEAR_PASS  # tts=900s, within threshold
        far_pass = _FAKE_FAR_PASS   # tts=2700s, outside threshold
        # nearest_orbital_pass returns the near one (min tts)
        with patch("custody.sensors.nearest_orbital_pass", return_value=near):
            decision = plan_collection(
                _fresh_track(), score=1.0, confidence=0.4,
                breakdown=_NOMINAL_BD, current_time=_T_NEAR,
                opportunities=[],
                observer_lat=_OBS_LAT, observer_lon=_OBS_LON,
            )
        assert decision.action == "HOLD"


# ---------------------------------------------------------------------------
# 6. Decision trace reflects lookahead_boost correctly
# ---------------------------------------------------------------------------

class TestDecisionTrace:
    """build_decision_trace records exactly the values passed in — no recomputation."""

    def _make_trace(self, lookahead_fires: bool):
        """Build a trace by passing values directly (no mocking needed)."""
        track = _fresh_track()
        lb = config.HOLD_LOOKAHEAD_BOOST if lookahead_fires else 0.0
        tts = _FAKE_NEAR_PASS.time_to_start_seconds if lookahead_fires else None
        return build_decision_trace(
            timestamp=_T_NEAR,
            vessel_id="V-TEST",
            score=1.0,
            confidence=0.4,
            compound_boost=0.0,
            track=track,
            accessible_opportunities=[],
            claimed_sensor_ids=set(),
            remaining_opportunities=[],
            decision_action="HOLD",
            decision_sensor_id=None,
            lookahead_boost=lb,
            nearest_pass_tts=tts,
        )

    def test_lookahead_boost_nonzero_when_passed(self):
        trace = self._make_trace(lookahead_fires=True)
        assert trace.task_value.lookahead_boost == pytest.approx(config.HOLD_LOOKAHEAD_BOOST)

    def test_lookahead_boost_zero_when_not_passed(self):
        trace = self._make_trace(lookahead_fires=False)
        assert trace.task_value.lookahead_boost == pytest.approx(0.0)

    def test_nearest_pass_tts_populated_when_passed(self):
        trace = self._make_trace(lookahead_fires=True)
        assert trace.task_value.nearest_pass_time_to_start_seconds == pytest.approx(
            _FAKE_NEAR_PASS.time_to_start_seconds
        )

    def test_nearest_pass_tts_none_when_not_passed(self):
        trace = self._make_trace(lookahead_fires=False)
        assert trace.task_value.nearest_pass_time_to_start_seconds is None

    def test_lookahead_boost_included_in_total(self):
        """task_value.total includes the lookahead_boost contribution."""
        trace_with = self._make_trace(lookahead_fires=True)
        trace_without = self._make_trace(lookahead_fires=False)
        diff = trace_with.task_value.total - trace_without.task_value.total
        assert diff == pytest.approx(config.HOLD_LOOKAHEAD_BOOST, abs=1e-9)

    def test_arbitrary_values_passed_through_exactly(self):
        """build_decision_trace records whatever values are passed — no clamping or rederivation."""
        sentinel_boost = 0.7777
        sentinel_tts = 123.456
        track = _fresh_track()
        trace = build_decision_trace(
            timestamp=_T_NEAR,
            vessel_id="V-TEST",
            score=1.0,
            confidence=0.4,
            compound_boost=0.0,
            track=track,
            accessible_opportunities=[],
            claimed_sensor_ids=set(),
            remaining_opportunities=[],
            decision_action="HOLD",
            decision_sensor_id=None,
            lookahead_boost=sentinel_boost,
            nearest_pass_tts=sentinel_tts,
        )
        assert trace.task_value.lookahead_boost == pytest.approx(sentinel_boost)
        assert trace.task_value.nearest_pass_time_to_start_seconds == pytest.approx(sentinel_tts)

    def test_trace_values_match_planner_decision_exactly(self):
        """End-to-end: trace records the exact values from CollectionDecision."""
        with patch("custody.sensors.nearest_orbital_pass",
                   return_value=_FAKE_NEAR_PASS):
            decision = plan_collection(
                _fresh_track(), score=1.0, confidence=0.4,
                breakdown=_NOMINAL_BD, current_time=_T_NEAR,
                opportunities=[],
                observer_lat=_OBS_LAT, observer_lon=_OBS_LON,
            )
        track = _fresh_track()
        trace = build_decision_trace(
            timestamp=_T_NEAR,
            vessel_id="V-TEST",
            score=1.0,
            confidence=0.4,
            compound_boost=0.0,
            track=track,
            accessible_opportunities=[],
            claimed_sensor_ids=set(),
            remaining_opportunities=[],
            decision_action=decision.action,
            decision_sensor_id=decision.sensor_id,
            lookahead_boost=decision.lookahead_boost,
            nearest_pass_tts=decision.nearest_pass_tts,
        )
        assert trace.task_value.lookahead_boost == decision.lookahead_boost
        assert trace.task_value.nearest_pass_time_to_start_seconds == decision.nearest_pass_tts

    def test_collection_decision_carries_lookahead_fields(self):
        """CollectionDecision.lookahead_boost and .nearest_pass_tts are set by planner."""
        with patch("custody.sensors.nearest_orbital_pass",
                   return_value=_FAKE_NEAR_PASS):
            decision = plan_collection(
                _fresh_track(), score=1.0, confidence=0.4,
                breakdown=_NOMINAL_BD, current_time=_T_NEAR,
                opportunities=[],
                observer_lat=_OBS_LAT, observer_lon=_OBS_LON,
            )
        assert decision.lookahead_boost == pytest.approx(config.HOLD_LOOKAHEAD_BOOST)
        assert decision.nearest_pass_tts == pytest.approx(_FAKE_NEAR_PASS.time_to_start_seconds)

    def test_collection_decision_lookahead_zero_when_not_fired(self):
        """When lookahead does not fire, CollectionDecision carries 0.0/None."""
        decision = plan_collection(
            _fresh_track(), score=1.0, confidence=0.4,
            breakdown=_NOMINAL_BD, current_time=_T_FAR,
            opportunities=[],
            observer_lat=_OBS_LAT, observer_lon=_OBS_LON,
        )
        assert decision.action == "NO_SENSOR"
        assert decision.lookahead_boost == pytest.approx(0.0)
        assert decision.nearest_pass_tts is None

    def test_build_decision_trace_does_not_call_nearest_orbital_pass(self):
        """build_decision_trace no longer calls nearest_orbital_pass — values come from caller."""
        track = _fresh_track()
        with patch("custody.sensors.nearest_orbital_pass",
                   side_effect=AssertionError("build_decision_trace must not recompute lookahead")):
            # Should not raise — build_decision_trace takes values directly
            build_decision_trace(
                timestamp=_T_NEAR,
                vessel_id="V-TEST",
                score=1.0,
                confidence=0.4,
                compound_boost=0.0,
                track=track,
                accessible_opportunities=[],
                claimed_sensor_ids=set(),
                remaining_opportunities=[],
                decision_action="HOLD",
                decision_sensor_id=None,
                lookahead_boost=config.HOLD_LOOKAHEAD_BOOST,
                nearest_pass_tts=900.0,
            )


# ---------------------------------------------------------------------------
# 7. Existing planner behaviour unchanged when observer position is omitted
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    def test_no_sensor_without_observer_position(self):
        """Without observer_lat/lon, empty pool → NO_SENSOR (pre-lookahead behaviour)."""
        decision = plan_collection(
            _fresh_track(), score=1.0, confidence=0.4,
            breakdown=_NOMINAL_BD, current_time=_T_NEAR,
            opportunities=[],
        )
        assert decision.action == "NO_SENSOR"

    def test_hold_gate_unchanged_with_sensors_available(self):
        """HOLD gate still fires from task_value (freshness) when sensor is present."""
        track = _collected_track(hours_ago=0.5, anomaly_at_collection=0.8)
        decision = plan_collection(
            track, score=0.8, confidence=0.8,
            breakdown=_NOMINAL_BD, current_time=_T_NEAR,
            opportunities=_ONE_SENSOR,
            observer_lat=_OBS_LAT, observer_lon=_OBS_LON,
        )
        assert decision.action == "HOLD"

    def test_none_action_unchanged(self):
        """Low-anomaly / high-confidence vessel still returns NONE."""
        decision = plan_collection(
            _fresh_track(), score=0.1, confidence=0.9,
            breakdown=_NOMINAL_BD, current_time=_T_NEAR,
            opportunities=[],
            observer_lat=_OBS_LAT, observer_lon=_OBS_LON,
        )
        assert decision.action == "NONE"

    def test_compute_task_value_default_no_lookahead(self):
        """Calling compute_task_value without lookahead args preserves prior behaviour."""
        result_old_style = compute_task_value(0.8, 0.7, None, None)
        result_explicit = compute_task_value(0.8, 0.7, None, None,
                                             lookahead_boost=0.0,
                                             nearest_pass_tts=None)
        assert result_old_style == pytest.approx(result_explicit, abs=1e-9)


# ---------------------------------------------------------------------------
# 8. Preempted vessels are NOT converted to HOLD by lookahead
# ---------------------------------------------------------------------------

class TestPreemptedNotAffected:
    def test_preempted_not_converted_to_hold(self):
        """preempted=True bypasses lookahead; action stays PREEMPTED."""
        with patch("custody.sensors.nearest_orbital_pass",
                          return_value=_FAKE_NEAR_PASS):
            decision = plan_collection(
                _fresh_track(), score=1.0, confidence=0.4,
                breakdown=_NOMINAL_BD, current_time=_T_NEAR,
                opportunities=[], preempted=True,
                observer_lat=_OBS_LAT, observer_lon=_OBS_LON,
            )
        assert decision.action == "PREEMPTED"


# ---------------------------------------------------------------------------
# 9. compute_task_value lookahead_boost is additive
# ---------------------------------------------------------------------------

class TestComputeTaskValueLookahead:
    def test_lookahead_boost_is_additive_no_prior_collection(self):
        base = compute_task_value(0.8, 0.7, None, None, lookahead_boost=0.0)
        boosted = compute_task_value(0.8, 0.7, None, None, lookahead_boost=0.15)
        assert boosted - base == pytest.approx(0.15, abs=1e-9)

    def test_lookahead_boost_is_additive_with_prior_collection(self):
        base = compute_task_value(0.8, 0.7, 2.0, 0.8, lookahead_boost=0.0)
        boosted = compute_task_value(0.8, 0.7, 2.0, 0.8, lookahead_boost=0.15)
        assert boosted - base == pytest.approx(0.15, abs=1e-9)

    def test_breakdown_includes_lookahead_fields(self):
        _, bd = compute_task_value(0.8, 0.7, None, None,
                                   lookahead_boost=0.15, nearest_pass_tts=900.0,
                                   return_breakdown=True)
        assert bd.lookahead_boost == pytest.approx(0.15)
        assert bd.nearest_pass_time_to_start_seconds == pytest.approx(900.0)

    def test_breakdown_lookahead_zero_by_default(self):
        _, bd = compute_task_value(0.8, 0.7, None, None, return_breakdown=True)
        assert bd.lookahead_boost == pytest.approx(0.0)
        assert bd.nearest_pass_time_to_start_seconds is None


# ---------------------------------------------------------------------------
# 10. traces_to_rows includes new columns
# ---------------------------------------------------------------------------

class TestTracesToRows:
    def _make_trace_with_lookahead(self):
        track = _fresh_track()
        return build_decision_trace(
            timestamp=_T_NEAR,
            vessel_id="V-TEST",
            score=1.0,
            confidence=0.4,
            compound_boost=0.0,
            track=track,
            accessible_opportunities=[],
            claimed_sensor_ids=set(),
            remaining_opportunities=[],
            decision_action="HOLD",
            decision_sensor_id=None,
            lookahead_boost=config.HOLD_LOOKAHEAD_BOOST,
            nearest_pass_tts=_FAKE_NEAR_PASS.time_to_start_seconds,
        )

    def test_rows_contain_lookahead_boost_key(self):
        trace = self._make_trace_with_lookahead()
        rows = traces_to_rows([trace])
        assert "Lookahead Boost" in rows[0]

    def test_rows_contain_nearest_pass_tts_key(self):
        trace = self._make_trace_with_lookahead()
        rows = traces_to_rows([trace])
        assert "Nearest Pass TTS" in rows[0]

    def test_lookahead_boost_value_in_row(self):
        trace = self._make_trace_with_lookahead()
        rows = traces_to_rows([trace])
        assert rows[0]["Lookahead Boost"] == pytest.approx(
            round(config.HOLD_LOOKAHEAD_BOOST, 4)
        )

    def test_nearest_pass_tts_value_in_row(self):
        trace = self._make_trace_with_lookahead()
        rows = traces_to_rows([trace])
        assert rows[0]["Nearest Pass TTS"] == pytest.approx(
            _FAKE_NEAR_PASS.time_to_start_seconds
        )
