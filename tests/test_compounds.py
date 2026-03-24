"""
Tests for custody/compounds.py.

Covers:
  - CompoundSignal dataclass construction, immutability, timestamp field
  - LOITERING_NEAR_ZONE: fires, does not fire, partial conditions
  - HIGH_ANOMALY_LOW_CUSTODY: fires, does not fire, boundary conditions
  - PROXIMITY_NEAR_ZONE: fires, does not fire, missing key safety
  - LOITERING_WITH_PROXIMITY: fires, does not fire, missing key safety
  - REPEATED_ZONE_ENTRY: fires, entry counting, window size guards
  - evaluate_compounds: return type, missing keys, multiple rules in one record
  - compounds_for_timeline: empty, single record, multi-record, ordering
  - confidence invariants: always in [0, 1]
  - simulation-style records (no proximity field) never crash or spuriously fire
"""
from datetime import datetime, UTC, timedelta

import pytest

from custody.compounds import CompoundSignal, compounds_for_timeline, evaluate_compounds
from custody.config import (
    CRITICAL_ANOMALY_THRESHOLD,
    HIGH_ANOMALY_THRESHOLD,
    LOW_CUSTODY_THRESHOLD,
    ZONE_COMPOUND_MIN_SCORE,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

T0 = datetime(2026, 3, 23, 10, 0, tzinfo=UTC)
_ZONE_SCORE_MAX = 1.5   # mirrors the module constant


def _rec(**overrides) -> dict:
    """Minimal safe timeline record; all signals at zero/nominal defaults."""
    base = {
        "target_id": "V001",
        "time": T0,
        "anomaly_score": 0.0,
        "custody_confidence": 0.85,
        "loitering": 0.0,
        "sensitive_zone": 0.0,
        "vessel_proximity_score": 0.0,
    }
    base.update(overrides)
    return base


def _codes(signals: list[CompoundSignal]) -> set[str]:
    return {s.code for s in signals}


# ---------------------------------------------------------------------------
# CompoundSignal dataclass
# ---------------------------------------------------------------------------

class TestCompoundSignal:
    def test_construction(self):
        s = CompoundSignal(
            code="TEST", confidence=0.7, evidence="test",
            components={"a": 1.0}, timestamp=T0,
        )
        assert s.code == "TEST"
        assert s.confidence == 0.7
        assert s.timestamp == T0

    def test_is_frozen(self):
        s = CompoundSignal(
            code="TEST", confidence=0.7, evidence="test",
            components={}, timestamp=T0,
        )
        with pytest.raises((AttributeError, TypeError)):
            s.code = "OTHER"  # type: ignore[misc]

    def test_timestamp_field_present(self):
        s = CompoundSignal(
            code="X", confidence=0.5, evidence="x",
            components={}, timestamp=T0,
        )
        assert isinstance(s.timestamp, datetime)


# ---------------------------------------------------------------------------
# LOITERING_NEAR_ZONE
# ---------------------------------------------------------------------------

class TestLoiteringNearZone:
    def test_fires_when_both_signals_active(self):
        r = _rec(loitering=0.5, sensitive_zone=ZONE_COMPOUND_MIN_SCORE)
        assert "LOITERING_NEAR_ZONE" in _codes(evaluate_compounds(r))

    def test_does_not_fire_when_loitering_zero(self):
        r = _rec(loitering=0.0, sensitive_zone=1.5)
        assert "LOITERING_NEAR_ZONE" not in _codes(evaluate_compounds(r))

    def test_does_not_fire_when_zone_below_minimum(self):
        r = _rec(loitering=0.5, sensitive_zone=ZONE_COMPOUND_MIN_SCORE - 0.01)
        assert "LOITERING_NEAR_ZONE" not in _codes(evaluate_compounds(r))

    def test_does_not_fire_when_both_zero(self):
        r = _rec(loitering=0.0, sensitive_zone=0.0)
        assert "LOITERING_NEAR_ZONE" not in _codes(evaluate_compounds(r))

    def test_fires_at_exact_zone_minimum_threshold(self):
        r = _rec(loitering=0.5, sensitive_zone=ZONE_COMPOUND_MIN_SCORE)
        assert "LOITERING_NEAR_ZONE" in _codes(evaluate_compounds(r))

    def test_confidence_in_range(self):
        r = _rec(loitering=0.5, sensitive_zone=1.5)
        sig = next(s for s in evaluate_compounds(r) if s.code == "LOITERING_NEAR_ZONE")
        assert 0.0 <= sig.confidence <= 1.0

    def test_confidence_is_min_of_normalised_components(self):
        loitering_val = 0.4
        zone_val = 1.5
        r = _rec(loitering=loitering_val, sensitive_zone=zone_val)
        sig = next(s for s in evaluate_compounds(r) if s.code == "LOITERING_NEAR_ZONE")
        expected = min(min(loitering_val, 1.0), min(zone_val / _ZONE_SCORE_MAX, 1.0))
        assert sig.confidence == pytest.approx(expected, abs=1e-4)

    def test_components_reflect_raw_values(self):
        r = _rec(loitering=0.6, sensitive_zone=0.75)
        sig = next(s for s in evaluate_compounds(r) if s.code == "LOITERING_NEAR_ZONE")
        assert sig.components["loitering"] == pytest.approx(0.6)
        assert sig.components["sensitive_zone"] == pytest.approx(0.75)

    def test_timestamp_matches_record_time(self):
        t = T0 + timedelta(hours=3)
        r = _rec(loitering=0.5, sensitive_zone=1.0, time=t)
        sig = next(s for s in evaluate_compounds(r) if s.code == "LOITERING_NEAR_ZONE")
        assert sig.timestamp == t

    def test_evidence_contains_both_values(self):
        r = _rec(loitering=0.5, sensitive_zone=1.0)
        sig = next(s for s in evaluate_compounds(r) if s.code == "LOITERING_NEAR_ZONE")
        assert "loitering" in sig.evidence
        assert "sensitive_zone" in sig.evidence

    def test_high_loitering_limited_by_zone_confidence(self):
        # zone barely over threshold → confidence should be low
        r = _rec(loitering=1.0, sensitive_zone=ZONE_COMPOUND_MIN_SCORE)
        sig = next(s for s in evaluate_compounds(r) if s.code == "LOITERING_NEAR_ZONE")
        # norm_zone = ZONE_COMPOUND_MIN_SCORE / 1.5 < norm_loitering = 1.0
        assert sig.confidence == pytest.approx(
            ZONE_COMPOUND_MIN_SCORE / _ZONE_SCORE_MAX, abs=1e-4
        )


# ---------------------------------------------------------------------------
# HIGH_ANOMALY_LOW_CUSTODY
# ---------------------------------------------------------------------------

class TestHighAnomalyLowCustody:
    def test_fires_when_both_conditions_met(self):
        r = _rec(
            anomaly_score=HIGH_ANOMALY_THRESHOLD + 0.1,
            custody_confidence=LOW_CUSTODY_THRESHOLD - 0.1,
        )
        assert "HIGH_ANOMALY_LOW_CUSTODY" in _codes(evaluate_compounds(r))

    def test_does_not_fire_when_anomaly_at_threshold(self):
        # Strictly greater than HIGH_ANOMALY_THRESHOLD required.
        r = _rec(
            anomaly_score=HIGH_ANOMALY_THRESHOLD,
            custody_confidence=LOW_CUSTODY_THRESHOLD - 0.1,
        )
        assert "HIGH_ANOMALY_LOW_CUSTODY" not in _codes(evaluate_compounds(r))

    def test_does_not_fire_when_confidence_at_threshold(self):
        # Strictly less than LOW_CUSTODY_THRESHOLD required.
        r = _rec(
            anomaly_score=HIGH_ANOMALY_THRESHOLD + 0.1,
            custody_confidence=LOW_CUSTODY_THRESHOLD,
        )
        assert "HIGH_ANOMALY_LOW_CUSTODY" not in _codes(evaluate_compounds(r))

    def test_does_not_fire_when_anomaly_low(self):
        r = _rec(anomaly_score=0.2, custody_confidence=0.1)
        assert "HIGH_ANOMALY_LOW_CUSTODY" not in _codes(evaluate_compounds(r))

    def test_does_not_fire_when_confidence_high(self):
        r = _rec(anomaly_score=HIGH_ANOMALY_THRESHOLD + 0.5, custody_confidence=0.9)
        assert "HIGH_ANOMALY_LOW_CUSTODY" not in _codes(evaluate_compounds(r))

    def test_confidence_in_range(self):
        r = _rec(
            anomaly_score=CRITICAL_ANOMALY_THRESHOLD,
            custody_confidence=0.1,
        )
        sig = next(s for s in evaluate_compounds(r) if s.code == "HIGH_ANOMALY_LOW_CUSTODY")
        assert 0.0 <= sig.confidence <= 1.0

    def test_confidence_is_min_of_normalised_components(self):
        anomaly_val = HIGH_ANOMALY_THRESHOLD + 0.2
        conf_val = LOW_CUSTODY_THRESHOLD - 0.2
        r = _rec(anomaly_score=anomaly_val, custody_confidence=conf_val)
        sig = next(s for s in evaluate_compounds(r) if s.code == "HIGH_ANOMALY_LOW_CUSTODY")
        norm_anomaly = min(anomaly_val / CRITICAL_ANOMALY_THRESHOLD, 1.0)
        norm_low_conf = min(1.0 - conf_val, 1.0)
        expected = min(norm_anomaly, norm_low_conf)
        assert sig.confidence == pytest.approx(expected, abs=1e-4)

    def test_components_reflect_raw_values(self):
        anomaly_val = HIGH_ANOMALY_THRESHOLD + 0.3
        conf_val = LOW_CUSTODY_THRESHOLD - 0.15
        r = _rec(anomaly_score=anomaly_val, custody_confidence=conf_val)
        sig = next(s for s in evaluate_compounds(r) if s.code == "HIGH_ANOMALY_LOW_CUSTODY")
        assert sig.components["anomaly_score"] == pytest.approx(anomaly_val)
        assert sig.components["custody_confidence"] == pytest.approx(conf_val)

    def test_timestamp_matches_record_time(self):
        t = T0 + timedelta(hours=5)
        r = _rec(
            anomaly_score=HIGH_ANOMALY_THRESHOLD + 0.1,
            custody_confidence=LOW_CUSTODY_THRESHOLD - 0.1,
            time=t,
        )
        sig = next(s for s in evaluate_compounds(r) if s.code == "HIGH_ANOMALY_LOW_CUSTODY")
        assert sig.timestamp == t

    def test_evidence_mentions_both_fields(self):
        r = _rec(
            anomaly_score=HIGH_ANOMALY_THRESHOLD + 0.1,
            custody_confidence=LOW_CUSTODY_THRESHOLD - 0.1,
        )
        sig = next(s for s in evaluate_compounds(r) if s.code == "HIGH_ANOMALY_LOW_CUSTODY")
        assert "anomaly_score" in sig.evidence
        assert "custody_confidence" in sig.evidence

    def test_extreme_anomaly_confidence_capped_at_one(self):
        # anomaly_score far above critical threshold → norm_anomaly capped at 1.0
        r = _rec(
            anomaly_score=CRITICAL_ANOMALY_THRESHOLD * 10,
            custody_confidence=0.0,
        )
        sig = next(s for s in evaluate_compounds(r) if s.code == "HIGH_ANOMALY_LOW_CUSTODY")
        assert sig.confidence <= 1.0


# ---------------------------------------------------------------------------
# evaluate_compounds
# ---------------------------------------------------------------------------

class TestEvaluateCompounds:
    def test_returns_list(self):
        assert isinstance(evaluate_compounds(_rec()), list)

    def test_empty_when_no_signals_active(self):
        assert evaluate_compounds(_rec()) == []

    def test_missing_loitering_key_does_not_crash(self):
        r = {k: v for k, v in _rec().items() if k != "loitering"}
        result = evaluate_compounds(r)
        assert isinstance(result, list)

    def test_missing_sensitive_zone_key_does_not_crash(self):
        r = {k: v for k, v in _rec().items() if k != "sensitive_zone"}
        result = evaluate_compounds(r)
        assert isinstance(result, list)

    def test_missing_anomaly_score_key_does_not_crash(self):
        r = {k: v for k, v in _rec().items() if k != "anomaly_score"}
        result = evaluate_compounds(r)
        assert isinstance(result, list)

    def test_both_rules_fire_simultaneously(self):
        r = _rec(
            loitering=0.5,
            sensitive_zone=1.0,
            anomaly_score=HIGH_ANOMALY_THRESHOLD + 0.1,
            custody_confidence=LOW_CUSTODY_THRESHOLD - 0.1,
        )
        codes = _codes(evaluate_compounds(r))
        assert "LOITERING_NEAR_ZONE" in codes
        assert "HIGH_ANOMALY_LOW_CUSTODY" in codes

    def test_window_argument_accepted_without_error(self):
        r = _rec()
        result = evaluate_compounds(r, window=[_rec(), _rec()])
        assert isinstance(result, list)

    def test_all_confidence_values_in_range(self):
        r = _rec(
            loitering=0.8, sensitive_zone=1.2,
            anomaly_score=HIGH_ANOMALY_THRESHOLD + 0.5,
            custody_confidence=LOW_CUSTODY_THRESHOLD - 0.2,
        )
        for sig in evaluate_compounds(r):
            assert 0.0 <= sig.confidence <= 1.0, \
                f"{sig.code} confidence {sig.confidence} out of range"


# ---------------------------------------------------------------------------
# compounds_for_timeline
# ---------------------------------------------------------------------------

class TestCompoundsForTimeline:
    def test_empty_timeline_returns_empty_list(self):
        assert compounds_for_timeline([]) == []

    def test_single_record_no_signals(self):
        assert compounds_for_timeline([_rec()]) == []

    def test_single_record_fires_signal(self):
        r = _rec(loitering=0.5, sensitive_zone=1.0)
        result = compounds_for_timeline([r])
        assert len(result) == 1
        assert result[0].code == "LOITERING_NEAR_ZONE"

    def test_multi_record_signals_in_timeline_order(self):
        t1 = T0
        t2 = T0 + timedelta(hours=1)
        t3 = T0 + timedelta(hours=2)
        timeline = [
            _rec(time=t1),                                           # no signal
            _rec(time=t2, loitering=0.5, sensitive_zone=1.0),       # fires
            _rec(time=t3, loitering=0.8, sensitive_zone=1.5),       # fires
        ]
        result = compounds_for_timeline(timeline)
        assert len(result) == 2
        assert result[0].timestamp == t2
        assert result[1].timestamp == t3

    def test_all_timestamps_correspond_to_source_records(self):
        timeline = [
            _rec(time=T0 + timedelta(hours=i), loitering=0.5, sensitive_zone=1.0)
            for i in range(4)
        ]
        result = compounds_for_timeline(timeline)
        result_times = {s.timestamp for s in result}
        record_times = {r["time"] for r in timeline}
        assert result_times <= record_times

    def test_returns_flat_list(self):
        timeline = [_rec(loitering=0.5, sensitive_zone=1.0) for _ in range(3)]
        result = compounds_for_timeline(timeline)
        assert isinstance(result, list)
        assert all(isinstance(s, CompoundSignal) for s in result)


# ---------------------------------------------------------------------------
# PROXIMITY_NEAR_ZONE
# ---------------------------------------------------------------------------

class TestProximityNearZone:
    def test_fires_when_both_conditions_met(self):
        r = _rec(vessel_proximity_score=0.5, sensitive_zone=ZONE_COMPOUND_MIN_SCORE)
        assert "PROXIMITY_NEAR_ZONE" in _codes(evaluate_compounds(r))

    def test_fires_with_critical_proximity(self):
        r = _rec(vessel_proximity_score=1.0, sensitive_zone=1.5)
        assert "PROXIMITY_NEAR_ZONE" in _codes(evaluate_compounds(r))

    def test_does_not_fire_when_proximity_zero(self):
        r = _rec(vessel_proximity_score=0.0, sensitive_zone=1.5)
        assert "PROXIMITY_NEAR_ZONE" not in _codes(evaluate_compounds(r))

    def test_does_not_fire_when_zone_below_minimum(self):
        r = _rec(vessel_proximity_score=0.5, sensitive_zone=ZONE_COMPOUND_MIN_SCORE - 0.01)
        assert "PROXIMITY_NEAR_ZONE" not in _codes(evaluate_compounds(r))

    def test_does_not_fire_when_both_zero(self):
        r = _rec(vessel_proximity_score=0.0, sensitive_zone=0.0)
        assert "PROXIMITY_NEAR_ZONE" not in _codes(evaluate_compounds(r))

    def test_missing_proximity_key_does_not_crash(self):
        r = {k: v for k, v in _rec(sensitive_zone=1.5).items()
             if k != "vessel_proximity_score"}
        result = evaluate_compounds(r)
        assert isinstance(result, list)
        assert "PROXIMITY_NEAR_ZONE" not in _codes(result)

    def test_missing_zone_key_does_not_crash(self):
        r = {k: v for k, v in _rec(vessel_proximity_score=0.5).items()
             if k != "sensitive_zone"}
        result = evaluate_compounds(r)
        assert isinstance(result, list)
        assert "PROXIMITY_NEAR_ZONE" not in _codes(result)

    def test_confidence_in_range(self):
        r = _rec(vessel_proximity_score=0.5, sensitive_zone=1.5)
        sig = next(s for s in evaluate_compounds(r) if s.code == "PROXIMITY_NEAR_ZONE")
        assert 0.0 <= sig.confidence <= 1.0

    def test_confidence_is_min_of_normalised_components(self):
        proximity_val = 0.5
        zone_val = 1.5
        r = _rec(vessel_proximity_score=proximity_val, sensitive_zone=zone_val)
        sig = next(s for s in evaluate_compounds(r) if s.code == "PROXIMITY_NEAR_ZONE")
        expected = min(min(proximity_val, 1.0), min(zone_val / _ZONE_SCORE_MAX, 1.0))
        assert sig.confidence == pytest.approx(expected, abs=1e-4)

    def test_zone_limits_confidence_when_low(self):
        # proximity at max, zone barely over threshold → zone limits confidence
        r = _rec(vessel_proximity_score=1.0, sensitive_zone=ZONE_COMPOUND_MIN_SCORE)
        sig = next(s for s in evaluate_compounds(r) if s.code == "PROXIMITY_NEAR_ZONE")
        assert sig.confidence == pytest.approx(
            ZONE_COMPOUND_MIN_SCORE / _ZONE_SCORE_MAX, abs=1e-4
        )

    def test_components_reflect_raw_values(self):
        r = _rec(vessel_proximity_score=0.5, sensitive_zone=0.75)
        sig = next(s for s in evaluate_compounds(r) if s.code == "PROXIMITY_NEAR_ZONE")
        assert sig.components["vessel_proximity_score"] == pytest.approx(0.5)
        assert sig.components["sensitive_zone"] == pytest.approx(0.75)

    def test_timestamp_matches_record_time(self):
        t = T0 + timedelta(hours=2)
        r = _rec(vessel_proximity_score=0.5, sensitive_zone=1.0, time=t)
        sig = next(s for s in evaluate_compounds(r) if s.code == "PROXIMITY_NEAR_ZONE")
        assert sig.timestamp == t

    def test_evidence_mentions_both_fields(self):
        r = _rec(vessel_proximity_score=0.5, sensitive_zone=1.0)
        sig = next(s for s in evaluate_compounds(r) if s.code == "PROXIMITY_NEAR_ZONE")
        assert "vessel_proximity_score" in sig.evidence
        assert "sensitive_zone" in sig.evidence


# ---------------------------------------------------------------------------
# LOITERING_WITH_PROXIMITY
# ---------------------------------------------------------------------------

class TestLoiteringWithProximity:
    def test_fires_when_both_conditions_met(self):
        r = _rec(loitering=0.3, vessel_proximity_score=0.5)
        assert "LOITERING_WITH_PROXIMITY" in _codes(evaluate_compounds(r))

    def test_fires_with_critical_proximity_and_high_loitering(self):
        r = _rec(loitering=0.8, vessel_proximity_score=1.0)
        assert "LOITERING_WITH_PROXIMITY" in _codes(evaluate_compounds(r))

    def test_does_not_fire_when_loitering_zero(self):
        r = _rec(loitering=0.0, vessel_proximity_score=1.0)
        assert "LOITERING_WITH_PROXIMITY" not in _codes(evaluate_compounds(r))

    def test_does_not_fire_when_proximity_zero(self):
        r = _rec(loitering=0.8, vessel_proximity_score=0.0)
        assert "LOITERING_WITH_PROXIMITY" not in _codes(evaluate_compounds(r))

    def test_does_not_fire_when_both_zero(self):
        r = _rec(loitering=0.0, vessel_proximity_score=0.0)
        assert "LOITERING_WITH_PROXIMITY" not in _codes(evaluate_compounds(r))

    def test_missing_proximity_key_does_not_crash(self):
        r = {k: v for k, v in _rec(loitering=0.8).items()
             if k != "vessel_proximity_score"}
        result = evaluate_compounds(r)
        assert isinstance(result, list)
        assert "LOITERING_WITH_PROXIMITY" not in _codes(result)

    def test_missing_loitering_key_does_not_crash(self):
        r = {k: v for k, v in _rec(vessel_proximity_score=0.5).items()
             if k != "loitering"}
        result = evaluate_compounds(r)
        assert isinstance(result, list)
        assert "LOITERING_WITH_PROXIMITY" not in _codes(result)

    def test_confidence_in_range(self):
        r = _rec(loitering=0.6, vessel_proximity_score=0.5)
        sig = next(s for s in evaluate_compounds(r) if s.code == "LOITERING_WITH_PROXIMITY")
        assert 0.0 <= sig.confidence <= 1.0

    def test_confidence_is_min_of_normalised_components(self):
        loitering_val = 0.4
        proximity_val = 0.5
        r = _rec(loitering=loitering_val, vessel_proximity_score=proximity_val)
        sig = next(s for s in evaluate_compounds(r) if s.code == "LOITERING_WITH_PROXIMITY")
        expected = min(min(loitering_val / 0.5, 1.0), min(proximity_val, 1.0))
        assert sig.confidence == pytest.approx(expected, abs=1e-4)

    def test_loitering_saturates_at_half_value(self):
        # loitering=0.5 → norm=1.0; confidence limited by proximity
        r = _rec(loitering=0.5, vessel_proximity_score=0.5)
        sig = next(s for s in evaluate_compounds(r) if s.code == "LOITERING_WITH_PROXIMITY")
        assert sig.confidence == pytest.approx(0.5, abs=1e-4)

    def test_loitering_above_norm_capped_at_one(self):
        # loitering > 0.5 → norm_loitering capped at 1.0
        r = _rec(loitering=1.0, vessel_proximity_score=1.0)
        sig = next(s for s in evaluate_compounds(r) if s.code == "LOITERING_WITH_PROXIMITY")
        assert sig.confidence == pytest.approx(1.0, abs=1e-4)

    def test_components_reflect_raw_values(self):
        r = _rec(loitering=0.4, vessel_proximity_score=1.0)
        sig = next(s for s in evaluate_compounds(r) if s.code == "LOITERING_WITH_PROXIMITY")
        assert sig.components["loitering"] == pytest.approx(0.4)
        assert sig.components["vessel_proximity_score"] == pytest.approx(1.0)

    def test_timestamp_matches_record_time(self):
        t = T0 + timedelta(hours=4)
        r = _rec(loitering=0.6, vessel_proximity_score=0.5, time=t)
        sig = next(s for s in evaluate_compounds(r) if s.code == "LOITERING_WITH_PROXIMITY")
        assert sig.timestamp == t

    def test_evidence_mentions_both_fields(self):
        r = _rec(loitering=0.6, vessel_proximity_score=0.5)
        sig = next(s for s in evaluate_compounds(r) if s.code == "LOITERING_WITH_PROXIMITY")
        assert "loitering" in sig.evidence
        assert "vessel_proximity_score" in sig.evidence


# ---------------------------------------------------------------------------
# Simulation-record safety (no proximity field)
# ---------------------------------------------------------------------------

class TestSimulationRecordSafety:
    """Records from simulate.py / ingest_ais_track never carry vessel_proximity_score.

    These tests confirm that the proximity-based compound rules are safe to
    evaluate against such records and never fire spuriously.
    """

    def _sim_rec(self, **overrides) -> dict:
        """Minimal simulation-style record with no proximity field."""
        base = {
            "target_id": "V001",
            "time": T0,
            "anomaly_score": 0.0,
            "custody_confidence": 0.85,
            "loitering": 0.0,
            "sensitive_zone": 0.0,
        }
        base.update(overrides)
        return base

    def test_no_proximity_field_does_not_crash(self):
        result = evaluate_compounds(self._sim_rec())
        assert isinstance(result, list)

    def test_no_proximity_field_proximity_near_zone_does_not_fire(self):
        result = evaluate_compounds(self._sim_rec(sensitive_zone=1.5))
        assert "PROXIMITY_NEAR_ZONE" not in _codes(result)

    def test_no_proximity_field_loitering_with_proximity_does_not_fire(self):
        result = evaluate_compounds(self._sim_rec(loitering=0.8))
        assert "LOITERING_WITH_PROXIMITY" not in _codes(result)

    def test_loitering_near_zone_still_fires_without_proximity_field(self):
        result = evaluate_compounds(
            self._sim_rec(loitering=0.5, sensitive_zone=1.0)
        )
        assert "LOITERING_NEAR_ZONE" in _codes(result)

    def test_all_signals_active_simulation_record(self):
        """A sim record with all non-proximity signals active fires exactly one rule."""
        r = self._sim_rec(loitering=0.5, sensitive_zone=1.0)
        codes = _codes(evaluate_compounds(r))
        assert "LOITERING_NEAR_ZONE" in codes
        assert "PROXIMITY_NEAR_ZONE" not in codes
        assert "LOITERING_WITH_PROXIMITY" not in codes


# ---------------------------------------------------------------------------
# REPEATED_ZONE_ENTRY
# ---------------------------------------------------------------------------

class TestRepeatedZoneEntry:
    """Tests for the window-based REPEATED_ZONE_ENTRY compound rule."""

    def _entry_window(self, n_entries: int) -> list[dict]:
        """Build a window with exactly n_entries zone-entry transitions.

        Pattern: outside, inside, outside, inside, ... (2n records).
        """
        records = []
        for i in range(n_entries * 2):
            zone = ZONE_COMPOUND_MIN_SCORE if i % 2 == 1 else 0.0
            records.append(_rec(sensitive_zone=zone))
        return records

    # -- fires / does not fire -----------------------------------------------

    def test_fires_with_two_entries(self):
        r = _rec()
        assert "REPEATED_ZONE_ENTRY" in _codes(
            evaluate_compounds(r, window=self._entry_window(2))
        )

    def test_fires_with_three_entries(self):
        r = _rec()
        assert "REPEATED_ZONE_ENTRY" in _codes(
            evaluate_compounds(r, window=self._entry_window(3))
        )

    def test_does_not_fire_with_one_entry(self):
        window = [_rec(sensitive_zone=0.0), _rec(sensitive_zone=ZONE_COMPOUND_MIN_SCORE)]
        r = _rec()
        assert "REPEATED_ZONE_ENTRY" not in _codes(evaluate_compounds(r, window=window))

    def test_does_not_fire_with_no_entries(self):
        window = [_rec(sensitive_zone=0.0)] * 4
        r = _rec()
        assert "REPEATED_ZONE_ENTRY" not in _codes(evaluate_compounds(r, window=window))

    def test_does_not_fire_when_always_inside(self):
        # No transitions — vessel never leaves, so no entries counted.
        window = [_rec(sensitive_zone=ZONE_COMPOUND_MIN_SCORE)] * 4
        r = _rec()
        assert "REPEATED_ZONE_ENTRY" not in _codes(evaluate_compounds(r, window=window))

    def test_exit_transitions_do_not_count(self):
        # Only inside→outside transitions; zero entry_count.
        window = [
            _rec(sensitive_zone=ZONE_COMPOUND_MIN_SCORE),
            _rec(sensitive_zone=0.0),
            _rec(sensitive_zone=ZONE_COMPOUND_MIN_SCORE),
            _rec(sensitive_zone=0.0),
        ]
        r = _rec()
        assert "REPEATED_ZONE_ENTRY" not in _codes(evaluate_compounds(r, window=window))

    # -- window size guards --------------------------------------------------

    def test_window_none_does_not_crash(self):
        result = evaluate_compounds(_rec(), window=None)
        assert isinstance(result, list)
        assert "REPEATED_ZONE_ENTRY" not in _codes(result)

    def test_empty_window_does_not_fire(self):
        assert "REPEATED_ZONE_ENTRY" not in _codes(
            evaluate_compounds(_rec(), window=[])
        )

    def test_single_record_window_does_not_fire(self):
        assert "REPEATED_ZONE_ENTRY" not in _codes(
            evaluate_compounds(_rec(), window=[_rec()])
        )

    # -- boundary threshold --------------------------------------------------

    def test_boundary_exactly_at_threshold_counts_as_entry(self):
        # Transition from just below → exactly ZONE_COMPOUND_MIN_SCORE.
        window = [
            _rec(sensitive_zone=ZONE_COMPOUND_MIN_SCORE - 0.01),
            _rec(sensitive_zone=ZONE_COMPOUND_MIN_SCORE),
            _rec(sensitive_zone=ZONE_COMPOUND_MIN_SCORE - 0.01),
            _rec(sensitive_zone=ZONE_COMPOUND_MIN_SCORE),
        ]
        r = _rec()
        assert "REPEATED_ZONE_ENTRY" in _codes(evaluate_compounds(r, window=window))

    def test_below_threshold_does_not_count_as_entry(self):
        # Both records below threshold — no transition.
        window = [
            _rec(sensitive_zone=0.0),
            _rec(sensitive_zone=ZONE_COMPOUND_MIN_SCORE - 0.01),
            _rec(sensitive_zone=0.0),
            _rec(sensitive_zone=ZONE_COMPOUND_MIN_SCORE - 0.01),
        ]
        r = _rec()
        assert "REPEATED_ZONE_ENTRY" not in _codes(evaluate_compounds(r, window=window))

    # -- confidence ----------------------------------------------------------

    def test_confidence_two_entries(self):
        r = _rec()
        sig = next(
            s for s in evaluate_compounds(r, window=self._entry_window(2))
            if s.code == "REPEATED_ZONE_ENTRY"
        )
        assert sig.confidence == pytest.approx(2.0 / 3.0, abs=1e-4)

    def test_confidence_three_entries_is_one(self):
        r = _rec()
        sig = next(
            s for s in evaluate_compounds(r, window=self._entry_window(3))
            if s.code == "REPEATED_ZONE_ENTRY"
        )
        assert sig.confidence == pytest.approx(1.0, abs=1e-4)

    def test_confidence_capped_at_one_for_many_entries(self):
        r = _rec()
        sig = next(
            s for s in evaluate_compounds(r, window=self._entry_window(5))
            if s.code == "REPEATED_ZONE_ENTRY"
        )
        assert sig.confidence <= 1.0

    def test_confidence_in_range(self):
        r = _rec()
        sig = next(
            s for s in evaluate_compounds(r, window=self._entry_window(2))
            if s.code == "REPEATED_ZONE_ENTRY"
        )
        assert 0.0 <= sig.confidence <= 1.0

    # -- payload -------------------------------------------------------------

    def test_components_has_entry_count(self):
        r = _rec()
        sig = next(
            s for s in evaluate_compounds(r, window=self._entry_window(2))
            if s.code == "REPEATED_ZONE_ENTRY"
        )
        assert sig.components["entry_count"] == 2

    def test_timestamp_is_from_current_record(self):
        t = T0 + timedelta(hours=7)
        r = _rec(time=t)
        sig = next(
            s for s in evaluate_compounds(r, window=self._entry_window(2))
            if s.code == "REPEATED_ZONE_ENTRY"
        )
        assert sig.timestamp == t

    def test_evidence_is_non_empty_string(self):
        r = _rec()
        sig = next(
            s for s in evaluate_compounds(r, window=self._entry_window(2))
            if s.code == "REPEATED_ZONE_ENTRY"
        )
        assert isinstance(sig.evidence, str) and len(sig.evidence) > 0

    # -- compounds_for_timeline integration ----------------------------------

    def test_fires_in_timeline_after_two_entries(self):
        """compounds_for_timeline fires REPEATED_ZONE_ENTRY once enough history exists."""
        timeline = [
            _rec(time=T0,                         sensitive_zone=0.0),
            _rec(time=T0 + timedelta(hours=1),    sensitive_zone=ZONE_COMPOUND_MIN_SCORE),
            _rec(time=T0 + timedelta(hours=2),    sensitive_zone=0.0),
            _rec(time=T0 + timedelta(hours=3),    sensitive_zone=ZONE_COMPOUND_MIN_SCORE),
            _rec(time=T0 + timedelta(hours=4),    sensitive_zone=0.0),
        ]
        result = compounds_for_timeline(timeline)
        assert "REPEATED_ZONE_ENTRY" in _codes(result)

    def test_does_not_fire_in_short_timeline(self):
        """Timeline too short to accumulate two entries does not fire."""
        timeline = [
            _rec(time=T0,                      sensitive_zone=0.0),
            _rec(time=T0 + timedelta(hours=1), sensitive_zone=ZONE_COMPOUND_MIN_SCORE),
            _rec(time=T0 + timedelta(hours=2), sensitive_zone=0.0),
        ]
        result = compounds_for_timeline(timeline)
        assert "REPEATED_ZONE_ENTRY" not in _codes(result)

    # -- simulation safety ---------------------------------------------------

    def test_simulation_records_no_crash(self):
        """Simulation-style records (no proximity field) never crash."""
        sim_rec = {
            "target_id": "V001", "time": T0,
            "anomaly_score": 0.0, "custody_confidence": 0.85,
            "loitering": 0.0, "sensitive_zone": 0.0,
        }
        window = [
            {**sim_rec, "sensitive_zone": 0.0},
            {**sim_rec, "sensitive_zone": ZONE_COMPOUND_MIN_SCORE},
            {**sim_rec, "sensitive_zone": 0.0},
            {**sim_rec, "sensitive_zone": ZONE_COMPOUND_MIN_SCORE},
        ]
        result = evaluate_compounds(sim_rec, window=window)
        assert isinstance(result, list)
        assert "REPEATED_ZONE_ENTRY" in _codes(result)
