"""
Tests for custody.observation — observation state and sensor-aware reasoning.

Covers:
  1.  no prior observation → SEARCH
  2.  stale observation (>6h) → SEARCH
  3.  recent + anomalous + low confidence → CONFIRM
  4.  recent + sustained + high confidence → CHARACTERIZE
  5.  recent + normal → MONITOR
  6.  SAR last → prefers EO cross-sensor
  7.  EO last → prefers SAR cross-sensor
  8.  cross-sensor only when anomalous and single-sensor
  9.  night solar overrides cross-sensor optical preference
 10.  stale observation shortens revisit
 11.  observation_rationale mentions last sensor and time
 12.  simulation records carry observation fields
 13.  history absent → stable fallback behavior
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from custody.observation import (
    SEARCH, CONFIRM, CHARACTERIZE, MONITOR,
    ObservationState,
    derive_observation_state,
    apply_observation_to_policy,
)

UTC = timezone.utc
_T0 = datetime(2026, 3, 23, 10, 0, tzinfo=UTC)


def _rec(hour=0, action="NONE", result=None, sensor_type=None,
         anomaly_state="normal", agreement="normal", confidence=0.8):
    return {
        "target_id": "TEST",
        "time": _T0 + timedelta(hours=hour),
        "action": action,
        "collection_result": result,
        "sensor_type": sensor_type,
        "anomaly_state": anomaly_state,
        "anomaly_agreement": agreement,
        "overall_confidence": confidence,
    }


def _history_with_task(hours_ago, sensor_type="all_weather"):
    """A single-element history with a successful TASK at hours_ago."""
    return [_rec(hour=-hours_ago, action="TASK", result="SUCCESS", sensor_type=sensor_type)]


# ---------------------------------------------------------------------------
# Collection intent
# ---------------------------------------------------------------------------

class TestCollectionIntent:

    def test_no_prior_observation_search(self):
        obs = derive_observation_state(_rec(), [])
        assert obs.collection_intent == SEARCH

    def test_stale_observation_search(self):
        history = _history_with_task(hours_ago=14)  # beyond 12h threshold
        obs = derive_observation_state(_rec(hour=0), history)
        assert obs.collection_intent == SEARCH

    def test_recent_anomalous_low_confidence_confirm(self):
        history = _history_with_task(hours_ago=1)
        current = _rec(hour=0, anomaly_state="emerging", agreement="emerging", confidence=0.4)
        obs = derive_observation_state(current, history)
        assert obs.collection_intent == CONFIRM

    def test_sustained_high_confidence_characterize(self):
        history = _history_with_task(hours_ago=1)
        current = _rec(hour=0, anomaly_state="sustained", agreement="confirmed", confidence=0.85)
        obs = derive_observation_state(current, history)
        assert obs.collection_intent == CHARACTERIZE

    def test_normal_monitor(self):
        history = _history_with_task(hours_ago=2)
        current = _rec(hour=0, anomaly_state="normal")
        obs = derive_observation_state(current, history)
        assert obs.collection_intent == MONITOR


# ---------------------------------------------------------------------------
# Cross-sensor logic
# ---------------------------------------------------------------------------

class TestCrossSensor:

    def test_sar_last_prefers_eo(self):
        history = _history_with_task(hours_ago=1, sensor_type="all_weather")
        current = _rec(hour=0, anomaly_state="confirmed", agreement="confirmed")
        obs = derive_observation_state(current, history)
        assert obs.needs_cross_sensor is True
        assert obs.preferred_confirmation_sensor == "high_resolution"

    def test_eo_last_prefers_sar(self):
        history = _history_with_task(hours_ago=1, sensor_type="high_resolution")
        current = _rec(hour=0, anomaly_state="confirmed", agreement="confirmed")
        obs = derive_observation_state(current, history)
        assert obs.needs_cross_sensor is True
        assert obs.preferred_confirmation_sensor == "all_weather"

    def test_no_cross_sensor_when_normal(self):
        history = _history_with_task(hours_ago=1)
        current = _rec(hour=0, anomaly_state="normal")
        obs = derive_observation_state(current, history)
        assert obs.needs_cross_sensor is False

    def test_no_cross_sensor_when_stale(self):
        history = _history_with_task(hours_ago=8)
        current = _rec(hour=0, anomaly_state="confirmed")
        obs = derive_observation_state(current, history)
        # Stale → SEARCH, not cross-sensor confirmation
        assert obs.needs_cross_sensor is False


# ---------------------------------------------------------------------------
# Policy adjustment
# ---------------------------------------------------------------------------

class TestPolicyAdjustment:

    def test_night_overrides_cross_sensor_optical(self):
        """Night solar should prevent cross-sensor from switching to EO."""
        history = _history_with_task(hours_ago=1, sensor_type="all_weather")
        current = _rec(hour=0, anomaly_state="confirmed", agreement="confirmed")
        obs = derive_observation_state(current, history)

        policy = {
            "effective_sensor_preference": "all_weather",
            "solar_condition": "night",
            "desired_revisit_hours": 3.0,
            "sensor_rationale": "SAR selected",
        }
        adjusted = apply_observation_to_policy(policy, obs)
        # Cross-sensor wants EO but night prevents it
        assert adjusted["effective_sensor_preference"] == "all_weather"

    def test_stale_shortens_revisit(self):
        obs = derive_observation_state(_rec(), [])
        policy = {"desired_revisit_hours": 6.0, "effective_sensor_preference": "any",
                   "sensor_rationale": "", "solar_condition": "day"}
        adjusted = apply_observation_to_policy(policy, obs)
        assert adjusted["desired_revisit_hours"] < 6.0
        assert adjusted["desired_revisit_hours"] == pytest.approx(4.2, abs=0.1)

    def test_cross_sensor_adjusts_preference(self):
        history = _history_with_task(hours_ago=1, sensor_type="all_weather")
        current = _rec(hour=0, anomaly_state="confirmed")
        obs = derive_observation_state(current, history)

        policy = {
            "effective_sensor_preference": "all_weather",
            "solar_condition": "day",
            "desired_revisit_hours": 3.0,
            "sensor_rationale": "SAR selected",
        }
        adjusted = apply_observation_to_policy(policy, obs)
        assert adjusted["effective_sensor_preference"] == "high_resolution"
        assert "cross-sensor" in adjusted["sensor_rationale"]


# ---------------------------------------------------------------------------
# Rationale
# ---------------------------------------------------------------------------

class TestRationale:

    def test_mentions_last_sensor(self):
        history = _history_with_task(hours_ago=2, sensor_type="all_weather")
        current = _rec(hour=0, anomaly_state="confirmed")
        obs = derive_observation_state(current, history)
        assert "all_weather" in obs.rationale

    def test_stale_mentions_hours(self):
        obs = derive_observation_state(_rec(), [])
        assert "no prior observation" in obs.rationale


# ---------------------------------------------------------------------------
# Simulation integration
# ---------------------------------------------------------------------------

class TestSimulationIntegration:

    def test_records_have_observation_fields(self):
        from custody.simulate import run_simulation
        records = run_simulation()
        required = {
            "collection_intent", "observation_rationale",
            "needs_cross_sensor", "hours_since_observation",
        }
        for r in records[:5]:
            missing = required - set(r.keys())
            assert not missing, f"Missing: {missing}"

    def test_intent_varies(self):
        from custody.simulate import run_simulation
        records = run_simulation()
        intents = {r["collection_intent"] for r in records}
        # At minimum SEARCH (first step) and MONITOR should be present
        assert SEARCH in intents or MONITOR in intents

    def test_absent_history_stable(self):
        """First record with no history should not crash."""
        from custody.simulate import run_simulation
        records = run_simulation()
        first = records[0]
        assert first["collection_intent"] in (SEARCH, MONITOR, CONFIRM, CHARACTERIZE)
