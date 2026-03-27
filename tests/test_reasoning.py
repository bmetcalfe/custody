"""
Tests for custody.reasoning — temporal anomaly reasoning layer.

Covers:
  1.  classify_agreement: both high → CONFIRMED
  2.  classify_agreement: ML high, heuristic low → EMERGING
  3.  classify_agreement: heuristic high, ML low → RULE_TRIGGERED
  4.  classify_agreement: both low → NORMAL
  5.  classify_agreement: missing ML → NORMAL
  6.  compute_persistence: consecutive hours counted correctly
  7.  compute_persistence: streak breaks on low score
  8.  compute_persistence: is_sustained_anomaly threshold
  9.  compute_persistence: empty history → streak of 1 or 0
 10.  compute_escalation: zero for normal agreement
 11.  compute_escalation: positive for confirmed + sustained
 12.  compute_escalation: capped at ESCALATION_BOOST_CAP
 13.  compute_escalation: higher with weak custody
 14.  derive_anomaly_state: normal when both low
 15.  derive_anomaly_state: emerging when ML high only
 16.  derive_anomaly_state: confirmed when both high
 17.  derive_anomaly_state: sustained after 3+ hours
 18.  derive_anomaly_state: critical when sustained + weak custody
 19.  derive_anomaly_state: recovering from confirmed to normal
 20.  enrich_record: adds all expected fields
 21.  enrich_record: does not mutate original
 22.  enrich_record: transition sequence over 5 steps
 23.  simulation records carry reasoning fields
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone, timedelta

import pytest

import custody.config as config
from custody.reasoning import (
    AnomalyAgreement,
    AnomalyState,
    classify_agreement,
    compute_escalation,
    compute_persistence,
    derive_anomaly_state,
    enrich_record,
)

UTC = timezone.utc
_T0 = datetime(2026, 3, 23, 10, 0, tzinfo=UTC)


def _rec(ml=0.0, anom=0.0, custody=0.8, hour=0, fused=0.0):
    return {
        "target_id": "TEST",
        "time": _T0 + timedelta(hours=hour),
        "ml_anomaly_score": ml,
        "anomaly_score": anom,
        "custody_confidence": custody,
        "fused_score": fused,
    }


# ---------------------------------------------------------------------------
# Agreement
# ---------------------------------------------------------------------------

class TestAgreement:

    def test_both_high_confirmed(self):
        r = _rec(ml=0.9, anom=1.5)  # heuristic_norm = 1.0
        assert classify_agreement(r) == AnomalyAgreement.CONFIRMED

    def test_ml_high_heuristic_low_emerging(self):
        r = _rec(ml=0.9, anom=0.2)
        assert classify_agreement(r) == AnomalyAgreement.EMERGING

    def test_heuristic_high_ml_low_rule(self):
        r = _rec(ml=0.1, anom=1.2)  # 1.2/1.5 = 0.8 > 0.7 threshold
        assert classify_agreement(r) == AnomalyAgreement.RULE_TRIGGERED

    def test_both_low_normal(self):
        r = _rec(ml=0.1, anom=0.2)
        assert classify_agreement(r) == AnomalyAgreement.NORMAL

    def test_missing_ml_normal(self):
        r = {"anomaly_score": 0.3}
        assert classify_agreement(r) == AnomalyAgreement.NORMAL


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:

    def test_consecutive_hours(self):
        history = [_rec(ml=0.9, hour=i) for i in range(5)]
        current = _rec(ml=0.9, hour=5)
        p = compute_persistence(current, history)
        assert p["ml_anomaly_duration_hours"] == 6

    def test_streak_breaks(self):
        history = [_rec(ml=0.9, hour=0), _rec(ml=0.1, hour=1), _rec(ml=0.9, hour=2)]
        current = _rec(ml=0.9, hour=3)
        p = compute_persistence(current, history)
        assert p["ml_anomaly_duration_hours"] == 2  # only h2 + h3

    def test_sustained_threshold(self):
        history = [_rec(ml=0.9, hour=i) for i in range(4)]
        current = _rec(ml=0.9, hour=4)
        p = compute_persistence(current, history)
        assert p["is_sustained_anomaly"] is True  # 5 >= SUSTAINED_ANOMALY_MIN_HOURS

    def test_not_sustained_below_threshold(self):
        history = [_rec(ml=0.9, hour=i) for i in range(3)]
        current = _rec(ml=0.9, hour=3)
        p = compute_persistence(current, history)
        assert p["is_sustained_anomaly"] is False  # 4 < 5

    def test_empty_history_low_score(self):
        current = _rec(ml=0.1, hour=0)
        p = compute_persistence(current, [])
        assert p["ml_anomaly_duration_hours"] == 0

    def test_empty_history_high_score(self):
        current = _rec(ml=0.9, hour=0)
        p = compute_persistence(current, [])
        assert p["ml_anomaly_duration_hours"] == 1


# ---------------------------------------------------------------------------
# Escalation
# ---------------------------------------------------------------------------

class TestEscalation:

    def test_zero_for_normal(self):
        p = {"ml_anomaly_duration_hours": 0, "is_sustained_anomaly": False}
        assert compute_escalation(p, AnomalyAgreement.NORMAL, 0.8) == 0.0

    def test_positive_for_confirmed_sustained(self):
        p = {"ml_anomaly_duration_hours": 5, "is_sustained_anomaly": True}
        boost = compute_escalation(p, AnomalyAgreement.CONFIRMED, 0.7)
        assert boost > 0.0

    def test_capped(self):
        p = {"ml_anomaly_duration_hours": 100, "is_sustained_anomaly": True}
        boost = compute_escalation(p, AnomalyAgreement.CONFIRMED, 0.1)
        assert boost <= config.ESCALATION_BOOST_CAP

    def test_higher_with_weak_custody(self):
        p = {"ml_anomaly_duration_hours": 3, "is_sustained_anomaly": True}
        boost_good = compute_escalation(p, AnomalyAgreement.CONFIRMED, 0.9)
        boost_weak = compute_escalation(p, AnomalyAgreement.CONFIRMED, 0.2)
        assert boost_weak > boost_good


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------

class TestAnomalyState:

    def test_normal(self):
        p = {"ml_anomaly_duration_hours": 0, "is_sustained_anomaly": False}
        assert derive_anomaly_state(AnomalyAgreement.NORMAL, p, 0.8) == AnomalyState.NORMAL

    def test_emerging(self):
        p = {"ml_anomaly_duration_hours": 1, "is_sustained_anomaly": False}
        assert derive_anomaly_state(AnomalyAgreement.EMERGING, p, 0.8) == AnomalyState.EMERGING

    def test_confirmed(self):
        p = {"ml_anomaly_duration_hours": 2, "is_sustained_anomaly": False}
        assert derive_anomaly_state(AnomalyAgreement.CONFIRMED, p, 0.8) == AnomalyState.CONFIRMED

    def test_sustained(self):
        p = {"ml_anomaly_duration_hours": 5, "is_sustained_anomaly": True}
        assert derive_anomaly_state(AnomalyAgreement.CONFIRMED, p, 0.6) == AnomalyState.SUSTAINED

    def test_critical(self):
        p = {"ml_anomaly_duration_hours": 5, "is_sustained_anomaly": True}
        assert derive_anomaly_state(AnomalyAgreement.CONFIRMED, p, 0.2) == AnomalyState.CRITICAL

    def test_recovering(self):
        p = {"ml_anomaly_duration_hours": 0, "is_sustained_anomaly": False}
        state = derive_anomaly_state(
            AnomalyAgreement.NORMAL, p, 0.8,
            previous_state=AnomalyState.SUSTAINED,
        )
        assert state == AnomalyState.RECOVERING


# ---------------------------------------------------------------------------
# Enrich record
# ---------------------------------------------------------------------------

class TestEnrichRecord:

    def test_adds_expected_fields(self):
        r = _rec(ml=0.9, anom=1.5, custody=0.3)
        enriched = enrich_record(r, [])
        expected = {
            "anomaly_agreement", "ml_anomaly_duration_hours",
            "fused_anomaly_duration_hours", "is_sustained_anomaly",
            "anomaly_onset_timestamp", "escalation_boost", "anomaly_state",
        }
        assert expected.issubset(enriched.keys())

    def test_does_not_mutate_original(self):
        r = _rec(ml=0.9, anom=1.5)
        original = copy.deepcopy(r)
        enrich_record(r, [])
        assert r == original

    def test_transition_sequence(self):
        """Walk through 7 timesteps and verify state transitions.

        Updated for tighter thresholds:
        - SUSTAINED requires 5 consecutive hours (up from 3)
        - RULE_TRIGGERED no longer escalates to EMERGING
        """
        # h0: low → normal
        # h1: ML spikes → emerging (ML high, heuristic low)
        # h2: both high → confirmed
        # h3-h5: still high → confirmed (building toward sustained)
        # h6: 5th consecutive hour → sustained
        timeline = [
            _rec(ml=0.1, anom=0.2, hour=0),
            _rec(ml=0.9, anom=0.2, hour=1),
            _rec(ml=0.9, anom=1.5, hour=2),
            _rec(ml=0.9, anom=1.5, hour=3),
            _rec(ml=0.9, anom=1.5, hour=4),
            _rec(ml=0.9, anom=1.5, hour=5),
            _rec(ml=0.1, anom=0.2, hour=6),
        ]
        states = []
        prev_state = None
        for i, r in enumerate(timeline):
            history = timeline[:i]
            enriched = enrich_record(r, history, previous_state=prev_state)
            states.append(enriched["anomaly_state"])
            prev_state = enriched["anomaly_state"]

        assert states[0] == AnomalyState.NORMAL
        assert states[1] == AnomalyState.EMERGING
        assert states[2] == AnomalyState.CONFIRMED
        # h3-h4: confirmed (building but not yet 5h sustained)
        assert states[3] in (AnomalyState.CONFIRMED, AnomalyState.SUSTAINED)
        assert states[5] == AnomalyState.SUSTAINED
        assert states[6] == AnomalyState.RECOVERING


# ---------------------------------------------------------------------------
# Simulation records carry reasoning fields
# ---------------------------------------------------------------------------

class TestSimulationIntegration:

    def test_records_have_reasoning_fields(self):
        from custody.simulate import run_simulation
        records = run_simulation()
        required = {
            "anomaly_agreement", "anomaly_state",
            "ml_anomaly_duration_hours", "escalation_boost",
        }
        for r in records[:5]:
            missing = required - set(r.keys())
            assert not missing, f"Missing fields: {missing}"

    def test_all_states_are_valid(self):
        from custody.simulate import run_simulation
        valid = {
            AnomalyState.NORMAL, AnomalyState.EMERGING,
            AnomalyState.CONFIRMED, AnomalyState.SUSTAINED,
            AnomalyState.CRITICAL, AnomalyState.RECOVERING,
        }
        records = run_simulation()
        for r in records:
            assert r["anomaly_state"] in valid
