"""
Tests for CollectionDecision.hold_reason and its propagation through the trace.

hold_reason disambiguates the two HOLD sub-cases:
  "freshness"  — task value / freshness suppression kept the vessel on hold
  "lookahead"  — no sensor currently available but an orbital pass is imminent

For all non-HOLD actions (NONE, TASK, NO_SENSOR, PREEMPTED) hold_reason is None.

Coverage:
  1. freshness HOLD -> hold_reason="freshness"
  2. lookahead HOLD -> hold_reason="lookahead"
  3. NONE  -> hold_reason=None
  4. TASK  -> hold_reason=None
  5. NO_SENSOR -> hold_reason=None
  6. PREEMPTED -> hold_reason=None
  7. trace.arbitration.hold_reason matches decision.hold_reason exactly
  8. traces_to_rows row contains "Hold Reason" key with correct value
  9. Planner is the single source of truth: no recomputation in trace layer
"""
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from custody.decision_trace import build_decision_trace, traces_to_rows
from custody.models import TrackState
from custody.planner import plan_collection
from custody.sensors import PassWindow, SensorOpportunity
import custody.config as config


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

T0 = datetime(2026, 3, 23, 10, 0, tzinfo=UTC)

# 14:00 — SAT-A pass ~15 min away → within HOLD_LOOKAHEAD_THRESHOLD_SECONDS
_T_NEAR = datetime(2026, 3, 23, 14, 0, tzinfo=UTC)
_OBS_LAT, _OBS_LON = 0.5, 0.5

_NOMINAL_BD = {
    "sensitive_zone":  type("R", (), {"score": 0.0})(),
    "loitering":       type("R", (), {"score": 0.0})(),
    "route_deviation": type("R", (), {"score": 0.0})(),
}

_ONE_SENSOR = [
    SensorOpportunity(
        sensor_id="A1", sensor_type="fast_revisit",
        success_prob=1.0, resolution="medium", cost=1.0,
        available_from=T0, available_to=T0,
    )
]

_FAKE_NEAR_PASS = PassWindow(
    satellite_id="SAT-A",
    start_time=_T_NEAR + timedelta(minutes=15),
    end_time=_T_NEAR + timedelta(minutes=22),
    duration_seconds=420.0,
    time_to_start_seconds=900.0,
)


def _fresh_track() -> TrackState:
    return TrackState()


def _collected_track(hours_ago: float, anomaly: float = 0.8) -> TrackState:
    t = TrackState()
    t.record_collection(T0 - timedelta(hours=hours_ago), anomaly, new_uncertainty=4.0)
    return t


# ---------------------------------------------------------------------------
# 1. Freshness HOLD
# ---------------------------------------------------------------------------

class TestFreshnessHold:
    def _decision(self, **kwargs):
        track = _collected_track(hours_ago=0.5)
        return plan_collection(
            track, score=0.8, confidence=0.8,
            breakdown=_NOMINAL_BD, current_time=T0,
            opportunities=_ONE_SENSOR, **kwargs,
        )

    def test_hold_reason_is_freshness(self):
        d = self._decision()
        assert d.action == "HOLD"
        assert d.hold_reason == "freshness"

    def test_hold_reason_freshness_regardless_of_observer_position(self):
        """Freshness HOLD fires before the sensor check; observer position is irrelevant."""
        d = self._decision(observer_lat=_OBS_LAT, observer_lon=_OBS_LON)
        assert d.action == "HOLD"
        assert d.hold_reason == "freshness"


# ---------------------------------------------------------------------------
# 2. Lookahead HOLD
# ---------------------------------------------------------------------------

class TestLookaheadHold:
    def test_hold_reason_is_lookahead(self):
        with patch("custody.sensors.nearest_orbital_pass",
                   return_value=_FAKE_NEAR_PASS):
            d = plan_collection(
                _fresh_track(), score=1.0, confidence=0.4,
                breakdown=_NOMINAL_BD, current_time=_T_NEAR,
                opportunities=[],
                observer_lat=_OBS_LAT, observer_lon=_OBS_LON,
            )
        assert d.action == "HOLD"
        assert d.hold_reason == "lookahead"

    def test_real_orbital_timing_lookahead_hold_reason(self):
        """Integration: at 14:00, SAT-A pass ~15 min away → hold_reason='lookahead'."""
        d = plan_collection(
            _fresh_track(), score=1.0, confidence=0.4,
            breakdown=_NOMINAL_BD, current_time=_T_NEAR,
            opportunities=[],
            observer_lat=_OBS_LAT, observer_lon=_OBS_LON,
        )
        assert d.action == "HOLD"
        assert d.hold_reason == "lookahead"


# ---------------------------------------------------------------------------
# 3–6. Non-HOLD actions: hold_reason is None
# ---------------------------------------------------------------------------

class TestNonHoldActionsHaveNoReason:
    def test_none_action_has_no_hold_reason(self):
        d = plan_collection(
            _fresh_track(), score=0.1, confidence=0.9,
            breakdown=_NOMINAL_BD, current_time=T0,
            opportunities=_ONE_SENSOR,
        )
        assert d.action == "NONE"
        assert d.hold_reason is None

    def test_task_action_has_no_hold_reason(self):
        d = plan_collection(
            _fresh_track(), score=1.0, confidence=0.4,
            breakdown=_NOMINAL_BD, current_time=T0,
            opportunities=_ONE_SENSOR,
        )
        assert d.action == "TASK"
        assert d.hold_reason is None

    def test_no_sensor_has_no_hold_reason(self):
        d = plan_collection(
            _fresh_track(), score=1.0, confidence=0.4,
            breakdown=_NOMINAL_BD, current_time=T0,
            opportunities=[],
        )
        assert d.action == "NO_SENSOR"
        assert d.hold_reason is None

    def test_preempted_has_no_hold_reason(self):
        d = plan_collection(
            _fresh_track(), score=1.0, confidence=0.4,
            breakdown=_NOMINAL_BD, current_time=T0,
            opportunities=[], preempted=True,
        )
        assert d.action == "PREEMPTED"
        assert d.hold_reason is None

    def test_no_sensor_without_observer_has_no_hold_reason(self):
        """NO_SENSOR (no lookahead because no observer position) → hold_reason=None."""
        d = plan_collection(
            _fresh_track(), score=1.0, confidence=0.4,
            breakdown=_NOMINAL_BD, current_time=_T_NEAR,
            opportunities=[],
        )
        assert d.action == "NO_SENSOR"
        assert d.hold_reason is None


# ---------------------------------------------------------------------------
# 7. Trace records exact planner hold_reason
# ---------------------------------------------------------------------------

class TestTraceHoldReason:
    def _trace_from(self, decision, track=None):
        if track is None:
            track = _fresh_track()
        return build_decision_trace(
            timestamp=T0,
            vessel_id="V-TEST",
            score=0.8,
            confidence=0.8,
            compound_boost=0.0,
            track=track,
            accessible_opportunities=[],
            claimed_sensor_ids=set(),
            remaining_opportunities=[],
            decision_action=decision.action,
            decision_sensor_id=decision.sensor_id,
            lookahead_boost=decision.lookahead_boost,
            nearest_pass_tts=decision.nearest_pass_tts,
            hold_reason=decision.hold_reason,
        )

    def test_freshness_hold_reason_in_trace(self):
        track = _collected_track(hours_ago=0.5)
        d = plan_collection(
            track, score=0.8, confidence=0.8,
            breakdown=_NOMINAL_BD, current_time=T0,
            opportunities=_ONE_SENSOR,
        )
        trace = self._trace_from(d)
        assert trace.arbitration.hold_reason == "freshness"

    def test_lookahead_hold_reason_in_trace(self):
        with patch("custody.sensors.nearest_orbital_pass",
                   return_value=_FAKE_NEAR_PASS):
            d = plan_collection(
                _fresh_track(), score=1.0, confidence=0.4,
                breakdown=_NOMINAL_BD, current_time=_T_NEAR,
                opportunities=[],
                observer_lat=_OBS_LAT, observer_lon=_OBS_LON,
            )
        trace = self._trace_from(d)
        assert trace.arbitration.hold_reason == "lookahead"

    def test_none_action_hold_reason_in_trace(self):
        d = plan_collection(
            _fresh_track(), score=0.1, confidence=0.9,
            breakdown=_NOMINAL_BD, current_time=T0,
            opportunities=_ONE_SENSOR,
        )
        trace = self._trace_from(d)
        assert trace.arbitration.hold_reason is None

    def test_trace_hold_reason_matches_decision_exactly(self):
        """The trace records exactly what the decision carries — no recomputation."""
        track = _collected_track(hours_ago=0.5)
        d = plan_collection(
            track, score=0.8, confidence=0.8,
            breakdown=_NOMINAL_BD, current_time=T0,
            opportunities=_ONE_SENSOR,
        )
        trace = self._trace_from(d, track=_collected_track(hours_ago=0.5))
        assert trace.arbitration.hold_reason == d.hold_reason

    def test_arbitrary_hold_reason_passed_through(self):
        """build_decision_trace records whatever string is passed — no validation."""
        trace = build_decision_trace(
            timestamp=T0,
            vessel_id="V-TEST",
            score=0.8,
            confidence=0.8,
            compound_boost=0.0,
            track=_fresh_track(),
            accessible_opportunities=[],
            claimed_sensor_ids=set(),
            remaining_opportunities=[],
            decision_action="HOLD",
            decision_sensor_id=None,
            hold_reason="freshness",
        )
        assert trace.arbitration.hold_reason == "freshness"


# ---------------------------------------------------------------------------
# 8. traces_to_rows includes Hold Reason with correct value
# ---------------------------------------------------------------------------

class TestTracesToRowsHoldReason:
    def _make_hold_trace(self, reason: str):
        return build_decision_trace(
            timestamp=T0,
            vessel_id="V-TEST",
            score=0.8,
            confidence=0.8,
            compound_boost=0.0,
            track=_fresh_track(),
            accessible_opportunities=[],
            claimed_sensor_ids=set(),
            remaining_opportunities=[],
            decision_action="HOLD",
            decision_sensor_id=None,
            hold_reason=reason,
        )

    def test_hold_reason_key_present(self):
        rows = traces_to_rows([self._make_hold_trace("freshness")])
        assert "Hold Reason" in rows[0]

    def test_freshness_value_in_row(self):
        rows = traces_to_rows([self._make_hold_trace("freshness")])
        assert rows[0]["Hold Reason"] == "freshness"

    def test_lookahead_value_in_row(self):
        rows = traces_to_rows([self._make_hold_trace("lookahead")])
        assert rows[0]["Hold Reason"] == "lookahead"

    def test_none_hold_reason_in_non_hold_row(self):
        trace = build_decision_trace(
            timestamp=T0,
            vessel_id="V-TEST",
            score=0.8,
            confidence=0.8,
            compound_boost=0.0,
            track=_fresh_track(),
            accessible_opportunities=[],
            claimed_sensor_ids=set(),
            remaining_opportunities=[],
            decision_action="NO_SENSOR",
            decision_sensor_id=None,
            hold_reason=None,
        )
        rows = traces_to_rows([trace])
        assert rows[0]["Hold Reason"] is None
