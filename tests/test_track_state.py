"""
Unit tests for TrackState.

TrackState owns per-track custody metadata: uncertainty, collection timing,
and the last anomaly score recorded at collection.  These tests verify the
data model only — no simulation policy, no sensor objects.
"""
from datetime import datetime, UTC, timedelta

import pytest

from custody.models import TrackState


T0 = datetime(2026, 3, 23, 10, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Construction and defaults
# ---------------------------------------------------------------------------

def test_default_uncertainty():
    assert TrackState().uncertainty_km == 5.0


def test_default_last_collection_time_is_none():
    assert TrackState().last_collection_time is None


def test_default_last_collection_anomaly_score_is_zero():
    assert TrackState().last_collection_anomaly_score == 0.0


def test_custom_initial_uncertainty():
    ts = TrackState(uncertainty_km=12.5)
    assert ts.uncertainty_km == 12.5


# ---------------------------------------------------------------------------
# hours_since_collection
# ---------------------------------------------------------------------------

def test_hours_since_collection_returns_none_before_any_collection():
    ts = TrackState()
    assert ts.hours_since_collection(T0) is None


def test_hours_since_collection_exact_one_hour():
    ts = TrackState(last_collection_time=T0)
    assert ts.hours_since_collection(T0 + timedelta(hours=1)) == pytest.approx(1.0)


def test_hours_since_collection_fractional():
    ts = TrackState(last_collection_time=T0)
    result = ts.hours_since_collection(T0 + timedelta(minutes=90))
    assert result == pytest.approx(1.5)


def test_hours_since_collection_zero_when_now_equals_collection_time():
    ts = TrackState(last_collection_time=T0)
    assert ts.hours_since_collection(T0) == pytest.approx(0.0)


def test_hours_since_collection_large_gap():
    ts = TrackState(last_collection_time=T0)
    result = ts.hours_since_collection(T0 + timedelta(hours=48))
    assert result == pytest.approx(48.0)


# ---------------------------------------------------------------------------
# record_collection
# ---------------------------------------------------------------------------

def test_record_collection_updates_uncertainty():
    ts = TrackState()
    ts.record_collection(T0, anomaly_score=0.8, new_uncertainty=3.5)
    assert ts.uncertainty_km == pytest.approx(3.5)


def test_record_collection_updates_last_collection_time():
    ts = TrackState()
    ts.record_collection(T0, anomaly_score=0.8, new_uncertainty=3.5)
    assert ts.last_collection_time == T0


def test_record_collection_updates_anomaly_score():
    ts = TrackState()
    ts.record_collection(T0, anomaly_score=1.2, new_uncertainty=4.0)
    assert ts.last_collection_anomaly_score == pytest.approx(1.2)


def test_record_collection_second_call_overwrites_first():
    ts = TrackState()
    t1 = T0
    t2 = T0 + timedelta(hours=3)
    ts.record_collection(t1, anomaly_score=0.5, new_uncertainty=6.0)
    ts.record_collection(t2, anomaly_score=1.0, new_uncertainty=3.0)
    assert ts.last_collection_time == t2
    assert ts.uncertainty_km == pytest.approx(3.0)
    assert ts.last_collection_anomaly_score == pytest.approx(1.0)


def test_hours_since_collection_reflects_record_collection():
    ts = TrackState()
    ts.record_collection(T0, anomaly_score=0.0, new_uncertainty=5.0)
    result = ts.hours_since_collection(T0 + timedelta(hours=2))
    assert result == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# confidence property
# ---------------------------------------------------------------------------

def test_confidence_matches_custody_confidence_formula():
    import math
    ts = TrackState(uncertainty_km=25.0)
    expected = math.exp(-25.0 / 50)
    assert ts.confidence == pytest.approx(expected)


def test_confidence_decreases_as_uncertainty_grows():
    ts_low = TrackState(uncertainty_km=5.0)
    ts_high = TrackState(uncertainty_km=30.0)
    assert ts_low.confidence > ts_high.confidence


def test_confidence_updates_after_record_collection():
    ts = TrackState(uncertainty_km=40.0)
    before = ts.confidence
    ts.record_collection(T0, anomaly_score=0.5, new_uncertainty=5.0)
    assert ts.confidence > before
