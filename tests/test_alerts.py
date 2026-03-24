"""
Tests for the alerting layer.

Each rule is tested in isolation:
  - fires when its condition is exactly met
  - does not fire when its condition is not met
  - threshold edge cases: strictly below, equal, strictly above
  - transition rules handle prev_record=None correctly
"""
from datetime import datetime, timedelta, timezone

import pytest

from custody.alerts import Alert, alerts_for_timeline, evaluate_alerts
from custody.config import (
    CRITICAL_ANOMALY_THRESHOLD,
    HIGH_ANOMALY_THRESHOLD,
    LOW_CUSTODY_THRESHOLD,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

T0 = datetime(2026, 3, 23, 10, 0, tzinfo=timezone.utc)


def _record(**overrides) -> dict:
    """Return a minimal valid timeline record dict with safe defaults."""
    base = {
        "target_id": "V001",
        "time": T0,
        "lat": 0.0,
        "lon": 0.0,
        "uncertainty_km": 5.0,
        "custody_confidence": 0.85,
        "anomaly_score": 0.0,
        "speed_kmh": 18.0,
        "heading_deg": 45.0,
        "behavior_mode": "transit",
        "action": "NONE",
        "action_reason": "",
        "sensor_id": None,
        "sensor_type": None,
        "collection_result": None,
        "sensitive_zone": 0.0,
        "loitering": 0.0,
        "route_deviation": 0.0,
        "behavior_state": "transit",
        "state_confidence": 0.6,
    }
    base.update(overrides)
    return base


def _codes(alerts: list[Alert]) -> set[str]:
    return {a.code for a in alerts}


# ---------------------------------------------------------------------------
# evaluate_alerts: return type and structure
# ---------------------------------------------------------------------------

class TestEvaluateAlertsContract:
    def test_returns_list(self):
        assert isinstance(evaluate_alerts(_record()), list)

    def test_empty_for_nominal_record(self):
        assert evaluate_alerts(_record()) == []

    def test_alert_fields_present(self):
        r = _record(anomaly_score=CRITICAL_ANOMALY_THRESHOLD + 0.1)
        alerts = evaluate_alerts(r)
        assert len(alerts) == 1
        a = alerts[0]
        assert a.vessel_id == "V001"
        assert a.timestamp == T0
        assert a.level in {"INFO", "WARNING", "CRITICAL"}
        assert isinstance(a.code, str) and a.code
        assert isinstance(a.message, str) and a.message
        assert isinstance(a.context, dict)

    def test_alert_is_frozen(self):
        r = _record(anomaly_score=CRITICAL_ANOMALY_THRESHOLD + 0.1)
        a = evaluate_alerts(r)[0]
        with pytest.raises((AttributeError, TypeError)):
            a.level = "INFO"  # type: ignore[misc]

    def test_context_dicts_are_independent(self):
        """Two alerts from the same record must not share the same context dict."""
        r = _record(
            anomaly_score=CRITICAL_ANOMALY_THRESHOLD + 0.1,
            custody_confidence=LOW_CUSTODY_THRESHOLD - 0.1,
        )
        alerts = evaluate_alerts(r)
        assert len(alerts) >= 2
        ctx_ids = [id(a.context) for a in alerts]
        assert len(set(ctx_ids)) == len(ctx_ids), "context dicts are shared — must be independent"


# ---------------------------------------------------------------------------
# CRITICAL_ANOMALY
# ---------------------------------------------------------------------------

class TestCriticalAnomaly:
    def test_fires_above_threshold(self):
        r = _record(anomaly_score=CRITICAL_ANOMALY_THRESHOLD + 0.01)
        assert "CRITICAL_ANOMALY" in _codes(evaluate_alerts(r))

    def test_does_not_fire_at_threshold(self):
        r = _record(anomaly_score=CRITICAL_ANOMALY_THRESHOLD)
        assert "CRITICAL_ANOMALY" not in _codes(evaluate_alerts(r))

    def test_does_not_fire_below_threshold(self):
        r = _record(anomaly_score=CRITICAL_ANOMALY_THRESHOLD - 0.01)
        assert "CRITICAL_ANOMALY" not in _codes(evaluate_alerts(r))

    def test_level_is_critical(self):
        r = _record(anomaly_score=CRITICAL_ANOMALY_THRESHOLD + 0.1)
        a = next(a for a in evaluate_alerts(r) if a.code == "CRITICAL_ANOMALY")
        assert a.level == "CRITICAL"

    def test_context_contains_score_and_threshold(self):
        score = CRITICAL_ANOMALY_THRESHOLD + 0.1
        r = _record(anomaly_score=score)
        a = next(a for a in evaluate_alerts(r) if a.code == "CRITICAL_ANOMALY")
        assert "anomaly_score" in a.context
        assert "threshold" in a.context
        assert a.context["threshold"] == CRITICAL_ANOMALY_THRESHOLD


# ---------------------------------------------------------------------------
# HIGH_ANOMALY
# ---------------------------------------------------------------------------

class TestHighAnomaly:
    def test_fires_above_high_threshold(self):
        score = HIGH_ANOMALY_THRESHOLD + 0.01
        r = _record(anomaly_score=score)
        assert "HIGH_ANOMALY" in _codes(evaluate_alerts(r))

    def test_does_not_fire_at_high_threshold(self):
        r = _record(anomaly_score=HIGH_ANOMALY_THRESHOLD)
        assert "HIGH_ANOMALY" not in _codes(evaluate_alerts(r))

    def test_does_not_fire_below_high_threshold(self):
        r = _record(anomaly_score=HIGH_ANOMALY_THRESHOLD - 0.01)
        assert "HIGH_ANOMALY" not in _codes(evaluate_alerts(r))

    def test_does_not_fire_above_critical_threshold(self):
        """CRITICAL_ANOMALY takes over; HIGH_ANOMALY must not also fire."""
        r = _record(anomaly_score=CRITICAL_ANOMALY_THRESHOLD + 0.1)
        assert "HIGH_ANOMALY" not in _codes(evaluate_alerts(r))

    def test_does_not_fire_at_critical_threshold(self):
        r = _record(anomaly_score=CRITICAL_ANOMALY_THRESHOLD)
        assert "HIGH_ANOMALY" not in _codes(evaluate_alerts(r))

    def test_level_is_warning(self):
        score = HIGH_ANOMALY_THRESHOLD + 0.01
        r = _record(anomaly_score=score)
        a = next(a for a in evaluate_alerts(r) if a.code == "HIGH_ANOMALY")
        assert a.level == "WARNING"

    def test_context_contains_score_and_threshold(self):
        score = HIGH_ANOMALY_THRESHOLD + 0.1
        r = _record(anomaly_score=score)
        a = next(a for a in evaluate_alerts(r) if a.code == "HIGH_ANOMALY")
        assert a.context["threshold"] == HIGH_ANOMALY_THRESHOLD


# ---------------------------------------------------------------------------
# SENSITIVE_ZONE_ENTRY
# ---------------------------------------------------------------------------

class TestSensitiveZoneEntry:
    def test_fires_on_entry(self):
        prev = _record(sensitive_zone=0.0)
        curr = _record(sensitive_zone=0.5)
        assert "SENSITIVE_ZONE_ENTRY" in _codes(evaluate_alerts(curr, prev))

    def test_does_not_fire_when_already_inside(self):
        prev = _record(sensitive_zone=0.3)
        curr = _record(sensitive_zone=0.5)
        assert "SENSITIVE_ZONE_ENTRY" not in _codes(evaluate_alerts(curr, prev))

    def test_does_not_fire_when_outside_both(self):
        prev = _record(sensitive_zone=0.0)
        curr = _record(sensitive_zone=0.0)
        assert "SENSITIVE_ZONE_ENTRY" not in _codes(evaluate_alerts(curr, prev))

    def test_does_not_fire_on_exit(self):
        prev = _record(sensitive_zone=0.5)
        curr = _record(sensitive_zone=0.0)
        assert "SENSITIVE_ZONE_ENTRY" not in _codes(evaluate_alerts(curr, prev))

    def test_prev_none_does_not_fire(self):
        curr = _record(sensitive_zone=0.5)
        assert "SENSITIVE_ZONE_ENTRY" not in _codes(evaluate_alerts(curr, prev_record=None))

    def test_level_is_warning(self):
        prev = _record(sensitive_zone=0.0)
        curr = _record(sensitive_zone=0.5, lat=1.2, lon=0.7)
        a = next(a for a in evaluate_alerts(curr, prev) if a.code == "SENSITIVE_ZONE_ENTRY")
        assert a.level == "WARNING"

    def test_context_contains_position(self):
        prev = _record(sensitive_zone=0.0)
        curr = _record(sensitive_zone=0.5, lat=1.2, lon=0.7)
        a = next(a for a in evaluate_alerts(curr, prev) if a.code == "SENSITIVE_ZONE_ENTRY")
        assert a.context["lat"] == pytest.approx(1.2)
        assert a.context["lon"] == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# COLLECTION_FAILED
# ---------------------------------------------------------------------------

class TestCollectionFailed:
    def test_fires_on_failed_tasking(self):
        r = _record(action="TASK", collection_result="FAILED",
                    sensor_id="SAT-1", sensor_type="high_resolution")
        assert "COLLECTION_FAILED" in _codes(evaluate_alerts(r))

    def test_does_not_fire_on_success(self):
        r = _record(action="TASK", collection_result="SUCCESS",
                    sensor_id="SAT-1", sensor_type="high_resolution")
        assert "COLLECTION_FAILED" not in _codes(evaluate_alerts(r))

    def test_does_not_fire_on_none(self):
        r = _record(action="NONE", collection_result=None)
        assert "COLLECTION_FAILED" not in _codes(evaluate_alerts(r))

    def test_does_not_fire_on_hold(self):
        r = _record(action="HOLD", collection_result=None)
        assert "COLLECTION_FAILED" not in _codes(evaluate_alerts(r))

    def test_level_is_warning(self):
        r = _record(action="TASK", collection_result="FAILED",
                    sensor_id="SAT-1", sensor_type="high_resolution")
        a = next(a for a in evaluate_alerts(r) if a.code == "COLLECTION_FAILED")
        assert a.level == "WARNING"

    def test_context_contains_sensor_info(self):
        r = _record(action="TASK", collection_result="FAILED",
                    sensor_id="SAT-1", sensor_type="high_resolution")
        a = next(a for a in evaluate_alerts(r) if a.code == "COLLECTION_FAILED")
        assert a.context["sensor_id"] == "SAT-1"
        assert a.context["sensor_type"] == "high_resolution"


# ---------------------------------------------------------------------------
# NO_SENSOR_AVAILABLE
# ---------------------------------------------------------------------------

class TestNoSensorAvailable:
    def test_fires_on_no_sensor(self):
        r = _record(action="NO_SENSOR", anomaly_score=0.6, custody_confidence=0.6)
        assert "NO_SENSOR_AVAILABLE" in _codes(evaluate_alerts(r))

    def test_does_not_fire_on_none(self):
        r = _record(action="NONE")
        assert "NO_SENSOR_AVAILABLE" not in _codes(evaluate_alerts(r))

    def test_does_not_fire_on_task(self):
        r = _record(action="TASK", collection_result="SUCCESS",
                    sensor_id="SAT-1", sensor_type="high_resolution")
        assert "NO_SENSOR_AVAILABLE" not in _codes(evaluate_alerts(r))

    def test_level_is_warning(self):
        r = _record(action="NO_SENSOR", anomaly_score=0.6, custody_confidence=0.6)
        a = next(a for a in evaluate_alerts(r) if a.code == "NO_SENSOR_AVAILABLE")
        assert a.level == "WARNING"

    def test_context_contains_score_and_confidence(self):
        r = _record(action="NO_SENSOR", anomaly_score=0.7, custody_confidence=0.55)
        a = next(a for a in evaluate_alerts(r) if a.code == "NO_SENSOR_AVAILABLE")
        assert "anomaly_score" in a.context
        assert "custody_confidence" in a.context


# ---------------------------------------------------------------------------
# LOW_CUSTODY
# ---------------------------------------------------------------------------

class TestLowCustody:
    def test_fires_below_threshold(self):
        r = _record(custody_confidence=LOW_CUSTODY_THRESHOLD - 0.01)
        assert "LOW_CUSTODY" in _codes(evaluate_alerts(r))

    def test_does_not_fire_at_threshold(self):
        r = _record(custody_confidence=LOW_CUSTODY_THRESHOLD)
        assert "LOW_CUSTODY" not in _codes(evaluate_alerts(r))

    def test_does_not_fire_above_threshold(self):
        r = _record(custody_confidence=LOW_CUSTODY_THRESHOLD + 0.01)
        assert "LOW_CUSTODY" not in _codes(evaluate_alerts(r))

    def test_level_is_warning(self):
        r = _record(custody_confidence=LOW_CUSTODY_THRESHOLD - 0.1)
        a = next(a for a in evaluate_alerts(r) if a.code == "LOW_CUSTODY")
        assert a.level == "WARNING"

    def test_context_contains_confidence_and_uncertainty(self):
        r = _record(custody_confidence=0.3, uncertainty_km=12.0)
        a = next(a for a in evaluate_alerts(r) if a.code == "LOW_CUSTODY")
        assert a.context["custody_confidence"] == pytest.approx(0.3)
        assert a.context["uncertainty_km"] == pytest.approx(12.0)
        assert a.context["threshold"] == LOW_CUSTODY_THRESHOLD


# ---------------------------------------------------------------------------
# LOITERING_CONFIRMED
# ---------------------------------------------------------------------------

class TestLoiteringConfirmed:
    def test_fires_on_loiter_with_high_confidence(self):
        r = _record(behavior_state="loiter", state_confidence=0.85)
        assert "LOITERING_CONFIRMED" in _codes(evaluate_alerts(r))

    def test_fires_above_confidence_threshold(self):
        r = _record(behavior_state="loiter", state_confidence=0.86)
        assert "LOITERING_CONFIRMED" in _codes(evaluate_alerts(r))

    def test_does_not_fire_below_confidence_threshold(self):
        r = _record(behavior_state="loiter", state_confidence=0.84)
        assert "LOITERING_CONFIRMED" not in _codes(evaluate_alerts(r))

    def test_does_not_fire_for_non_loiter_state(self):
        for state in ("transit", "approach", "egress", "idle", "unknown"):
            r = _record(behavior_state=state, state_confidence=0.9)
            assert "LOITERING_CONFIRMED" not in _codes(evaluate_alerts(r)), \
                f"Unexpected LOITERING_CONFIRMED for state={state!r}"

    def test_level_is_info(self):
        r = _record(behavior_state="loiter", state_confidence=0.85)
        a = next(a for a in evaluate_alerts(r) if a.code == "LOITERING_CONFIRMED")
        assert a.level == "INFO"

    def test_context_contains_state_and_confidence(self):
        r = _record(behavior_state="loiter", state_confidence=0.9)
        a = next(a for a in evaluate_alerts(r) if a.code == "LOITERING_CONFIRMED")
        assert a.context["behavior_state"] == "loiter"
        assert a.context["state_confidence"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Multiple alerts from one record
# ---------------------------------------------------------------------------

class TestMultipleAlerts:
    def test_critical_anomaly_and_low_custody_together(self):
        r = _record(
            anomaly_score=CRITICAL_ANOMALY_THRESHOLD + 0.1,
            custody_confidence=LOW_CUSTODY_THRESHOLD - 0.1,
        )
        codes = _codes(evaluate_alerts(r))
        assert "CRITICAL_ANOMALY" in codes
        assert "LOW_CUSTODY" in codes

    def test_collection_failed_and_high_anomaly_together(self):
        r = _record(
            anomaly_score=HIGH_ANOMALY_THRESHOLD + 0.1,
            action="TASK",
            collection_result="FAILED",
            sensor_id="SAT-1",
            sensor_type="high_resolution",
        )
        codes = _codes(evaluate_alerts(r))
        assert "HIGH_ANOMALY" in codes
        assert "COLLECTION_FAILED" in codes

    def test_critical_suppresses_high_anomaly(self):
        """CRITICAL_ANOMALY and HIGH_ANOMALY must never both fire."""
        r = _record(anomaly_score=CRITICAL_ANOMALY_THRESHOLD + 0.5)
        codes = _codes(evaluate_alerts(r))
        assert "CRITICAL_ANOMALY" in codes
        assert "HIGH_ANOMALY" not in codes


# ---------------------------------------------------------------------------
# alerts_for_timeline (Step 12b)
# ---------------------------------------------------------------------------

class TestAlertsForTimeline:
    def test_empty_timeline_returns_empty_list(self):
        assert alerts_for_timeline([]) == []

    def test_returns_list(self):
        assert isinstance(alerts_for_timeline([]), list)

    def test_single_record_no_alerts(self):
        result = alerts_for_timeline([_record()])
        assert result == []

    def test_single_record_with_alert(self):
        r = _record(anomaly_score=CRITICAL_ANOMALY_THRESHOLD + 0.1)
        result = alerts_for_timeline([r])
        assert len(result) == 1
        assert result[0].code == "CRITICAL_ANOMALY"

    def test_single_record_uses_prev_none(self):
        """SENSITIVE_ZONE_ENTRY must not fire on a lone record even with zone > 0."""
        r = _record(sensitive_zone=1.0)
        codes = _codes(alerts_for_timeline([r]))
        assert "SENSITIVE_ZONE_ENTRY" not in codes

    def test_multi_record_alert_order_matches_timeline(self):
        """Alerts appear in the same order as the records that triggered them."""
        t1 = T0
        t2 = T0 + timedelta(hours=1)
        t3 = T0 + timedelta(hours=2)
        timeline = [
            _record(time=t1, anomaly_score=0.0),
            _record(time=t2, anomaly_score=HIGH_ANOMALY_THRESHOLD + 0.1),
            _record(time=t3, anomaly_score=CRITICAL_ANOMALY_THRESHOLD + 0.1),
        ]
        result = alerts_for_timeline(timeline)
        assert len(result) == 2
        assert result[0].code == "HIGH_ANOMALY"
        assert result[0].timestamp == t2
        assert result[1].code == "CRITICAL_ANOMALY"
        assert result[1].timestamp == t3

    def test_transition_rule_fires_once_at_correct_step(self):
        """SENSITIVE_ZONE_ENTRY fires exactly once, at the step where entry occurs."""
        timeline = [
            _record(time=T0,                          sensitive_zone=0.0),
            _record(time=T0 + timedelta(hours=1),     sensitive_zone=0.0),
            _record(time=T0 + timedelta(hours=2),     sensitive_zone=0.5),  # entry here
            _record(time=T0 + timedelta(hours=3),     sensitive_zone=0.8),  # already inside
            _record(time=T0 + timedelta(hours=4),     sensitive_zone=0.3),  # still inside
        ]
        entry_alerts = [a for a in alerts_for_timeline(timeline)
                        if a.code == "SENSITIVE_ZONE_ENTRY"]
        assert len(entry_alerts) == 1
        assert entry_alerts[0].timestamp == T0 + timedelta(hours=2)

    def test_multiple_alerts_from_one_record_all_included(self):
        """All alerts fired by a single record appear in the flat output."""
        r = _record(
            anomaly_score=CRITICAL_ANOMALY_THRESHOLD + 0.1,
            custody_confidence=LOW_CUSTODY_THRESHOLD - 0.1,
        )
        result = alerts_for_timeline([r])
        codes = _codes(result)
        assert "CRITICAL_ANOMALY" in codes
        assert "LOW_CUSTODY" in codes

    def test_no_spurious_alerts_on_nominal_timeline(self):
        timeline = [_record(time=T0 + timedelta(hours=i)) for i in range(5)]
        assert alerts_for_timeline(timeline) == []

    def test_alert_vessel_id_matches_record(self):
        r = _record(target_id="TARGET_X", anomaly_score=CRITICAL_ANOMALY_THRESHOLD + 0.1)
        result = alerts_for_timeline([r])
        assert all(a.vessel_id == "TARGET_X" for a in result)


# ---------------------------------------------------------------------------
# VESSEL_PROXIMITY — Step 14c
# ---------------------------------------------------------------------------

def _prox_record(**overrides) -> dict:
    """Extend the base record with proximity fields at safe defaults (score=0)."""
    base = _record(
        vessel_proximity_score=0.0,
        nearest_vessel_id=None,
        nearest_vessel_km=None,
    )
    base.update(overrides)
    return base


class TestVesselProximityAlert:
    def test_fires_on_zero_to_nonzero_transition(self):
        prev = _prox_record(vessel_proximity_score=0.0)
        curr = _prox_record(
            vessel_proximity_score=0.5,
            nearest_vessel_id="V002",
            nearest_vessel_km=12.0,
        )
        assert "VESSEL_PROXIMITY" in _codes(evaluate_alerts(curr, prev))

    def test_does_not_fire_when_score_stays_above_zero(self):
        prev = _prox_record(
            vessel_proximity_score=0.5,
            nearest_vessel_id="V002",
            nearest_vessel_km=12.0,
        )
        curr = _prox_record(
            vessel_proximity_score=1.0,
            nearest_vessel_id="V002",
            nearest_vessel_km=3.0,
        )
        assert "VESSEL_PROXIMITY" not in _codes(evaluate_alerts(curr, prev))

    def test_does_not_fire_when_score_stays_zero(self):
        prev = _prox_record(vessel_proximity_score=0.0)
        curr = _prox_record(vessel_proximity_score=0.0)
        assert "VESSEL_PROXIMITY" not in _codes(evaluate_alerts(curr, prev))

    def test_does_not_fire_on_first_record_prev_none(self):
        curr = _prox_record(
            vessel_proximity_score=1.0,
            nearest_vessel_id="V002",
            nearest_vessel_km=3.0,
        )
        assert "VESSEL_PROXIMITY" not in _codes(evaluate_alerts(curr, prev_record=None))

    def test_missing_vessel_proximity_score_treated_as_zero(self):
        # Neither record has the key — should not crash or fire.
        prev = _record()   # no proximity keys at all
        curr = _record()
        assert "VESSEL_PROXIMITY" not in _codes(evaluate_alerts(curr, prev))

    def test_missing_in_prev_only_treated_as_zero(self):
        # prev lacks the key → treated as 0.0; curr has score > 0 → should fire.
        prev = _record()   # no vessel_proximity_score key
        curr = _prox_record(
            vessel_proximity_score=0.5,
            nearest_vessel_id="V002",
            nearest_vessel_km=12.0,
        )
        assert "VESSEL_PROXIMITY" in _codes(evaluate_alerts(curr, prev))

    def test_context_includes_nearest_vessel_id(self):
        prev = _prox_record(vessel_proximity_score=0.0)
        curr = _prox_record(
            vessel_proximity_score=0.5,
            nearest_vessel_id="V002",
            nearest_vessel_km=12.0,
        )
        alert = next(a for a in evaluate_alerts(curr, prev) if a.code == "VESSEL_PROXIMITY")
        assert alert.context["nearest_vessel_id"] == "V002"

    def test_context_includes_distance_km(self):
        prev = _prox_record(vessel_proximity_score=0.0)
        curr = _prox_record(
            vessel_proximity_score=0.5,
            nearest_vessel_id="V002",
            nearest_vessel_km=12.0,
        )
        alert = next(a for a in evaluate_alerts(curr, prev) if a.code == "VESSEL_PROXIMITY")
        assert alert.context["distance_km"] == pytest.approx(12.0)

    def test_context_includes_vessel_id(self):
        prev = _prox_record(vessel_proximity_score=0.0)
        curr = _prox_record(
            target_id="V001",
            vessel_proximity_score=0.5,
            nearest_vessel_id="V002",
            nearest_vessel_km=12.0,
        )
        alert = next(a for a in evaluate_alerts(curr, prev) if a.code == "VESSEL_PROXIMITY")
        assert alert.context["vessel_id"] == "V001"

    def test_message_contains_nearest_vessel_id(self):
        prev = _prox_record(vessel_proximity_score=0.0)
        curr = _prox_record(
            vessel_proximity_score=0.5,
            nearest_vessel_id="V002",
            nearest_vessel_km=12.0,
        )
        alert = next(a for a in evaluate_alerts(curr, prev) if a.code == "VESSEL_PROXIMITY")
        assert "V002" in alert.message

    def test_message_contains_distance(self):
        prev = _prox_record(vessel_proximity_score=0.0)
        curr = _prox_record(
            vessel_proximity_score=0.5,
            nearest_vessel_id="V002",
            nearest_vessel_km=12.0,
        )
        alert = next(a for a in evaluate_alerts(curr, prev) if a.code == "VESSEL_PROXIMITY")
        assert "12.0" in alert.message

    def test_level_is_warning(self):
        prev = _prox_record(vessel_proximity_score=0.0)
        curr = _prox_record(
            vessel_proximity_score=1.0,
            nearest_vessel_id="V002",
            nearest_vessel_km=3.0,
        )
        alert = next(a for a in evaluate_alerts(curr, prev) if a.code == "VESSEL_PROXIMITY")
        assert alert.level == "WARNING"

    def test_fires_only_once_across_timeline(self):
        """Proximity alert fires on entry, not on every step while vessels remain close."""
        timeline = [
            _prox_record(time=T0,                              vessel_proximity_score=0.0),
            _prox_record(time=T0 + timedelta(hours=1),        vessel_proximity_score=0.5,
                         nearest_vessel_id="V002", nearest_vessel_km=12.0),
            _prox_record(time=T0 + timedelta(hours=2),        vessel_proximity_score=1.0,
                         nearest_vessel_id="V002", nearest_vessel_km=3.0),
            _prox_record(time=T0 + timedelta(hours=3),        vessel_proximity_score=0.5,
                         nearest_vessel_id="V002", nearest_vessel_km=11.0),
        ]
        prox_alerts = [a for a in alerts_for_timeline(timeline) if a.code == "VESSEL_PROXIMITY"]
        assert len(prox_alerts) == 1
        assert prox_alerts[0].timestamp == T0 + timedelta(hours=1)
