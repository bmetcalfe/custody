"""
Tests for AIS staleness confidence decay.

_stale_confidence(base, gap_seconds) applies an exponential penalty when
gap_seconds > AIS_STALE_GAP_SECONDS, floored at AIS_MIN_STALE_CONFIDENCE.
ingest_ais_track applies it to the per-record confidence that flows into
plan_collection and the decision trace.

Coverage:
  1. Short gaps (≤ threshold): no decay — base returned unchanged
  2. Gap exactly at threshold: no decay
  3. Long gaps: confidence is reduced
  4. Longer gaps produce lower confidence than shorter stale gaps (monotone)
  5. Floor: confidence never falls below AIS_MIN_STALE_CONFIDENCE
  6. AIS replay records reflect decayed confidence
  7. planner receives the decayed confidence, not raw track.confidence
  8. decision_trace confidence field reflects the decay
  9. Non-AIS paths (simulation) are unaffected
 10. decay uses config constants, not hardcoded values
"""
import math
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

import custody.config as config
from custody.ais import AISObservation, _stale_confidence, ingest_ais_track
from custody.decision_trace import DecisionTrace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

T0 = datetime(2026, 3, 23, 10, 0, tzinfo=UTC)
_BASE = 0.85  # representative high-confidence starting value


def _obs(vessel_id, timestamp, lat=0.0, lon=0.0):
    return AISObservation(
        vessel_id=vessel_id,
        timestamp=timestamp,
        lat=lat,
        lon=lon,
        speed_knots=5.0,
        heading_deg=0.0,
    )


def _gap_track(gap_seconds: float, n=2):
    """Minimal track: n observations separated by gap_seconds."""
    return [
        _obs("V001", T0 + timedelta(seconds=i * gap_seconds))
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# 1 & 2. Short gaps and exact threshold — no decay
# ---------------------------------------------------------------------------

class TestNoDecayForShortGaps:
    def test_zero_gap_returns_base(self):
        assert _stale_confidence(_BASE, 0.0) == _BASE

    def test_gap_below_threshold_returns_base(self):
        gap = config.AIS_STALE_GAP_SECONDS - 1
        assert _stale_confidence(_BASE, gap) == _BASE

    def test_gap_exactly_at_threshold_returns_base(self):
        assert _stale_confidence(_BASE, config.AIS_STALE_GAP_SECONDS) == _BASE

    def test_small_gap_returns_base_regardless_of_base_value(self):
        for base in (0.2, 0.5, 0.9):
            assert _stale_confidence(base, 60.0) == base


# ---------------------------------------------------------------------------
# 3. Long gaps reduce confidence
# ---------------------------------------------------------------------------

class TestLongGapsDecayConfidence:
    def test_long_gap_reduces_confidence(self):
        long_gap = config.AIS_STALE_GAP_SECONDS + 7200  # 2 h excess
        assert _stale_confidence(_BASE, long_gap) < _BASE

    def test_confidence_strictly_less_than_base_for_stale_gap(self):
        stale_gap = config.AIS_STALE_GAP_SECONDS * 3
        result = _stale_confidence(_BASE, stale_gap)
        assert result < _BASE

    def test_decay_is_exponential_in_excess_gap(self):
        """At excess = 1/rate the decay factor should be ~1/e."""
        rate = config.AIS_STALE_CONFIDENCE_DECAY_RATE
        characteristic_excess = 1.0 / rate  # seconds
        gap = config.AIS_STALE_GAP_SECONDS + characteristic_excess
        result = _stale_confidence(_BASE, gap)
        expected = _BASE * math.exp(-1.0)
        # Allow for floor clamping
        assert result == pytest.approx(max(expected, config.AIS_MIN_STALE_CONFIDENCE), abs=1e-6)


# ---------------------------------------------------------------------------
# 4. Monotonicity — longer stale gap → lower confidence
# ---------------------------------------------------------------------------

class TestMonotonicity:
    def test_longer_gap_gives_lower_confidence(self):
        gap_medium = config.AIS_STALE_GAP_SECONDS + 3600
        gap_long = config.AIS_STALE_GAP_SECONDS + 10800
        assert _stale_confidence(_BASE, gap_medium) > _stale_confidence(_BASE, gap_long)

    def test_three_gap_levels_are_ordered(self):
        short_stale = config.AIS_STALE_GAP_SECONDS + 1800
        medium_stale = config.AIS_STALE_GAP_SECONDS + 7200
        long_stale = config.AIS_STALE_GAP_SECONDS + 25200
        c_short = _stale_confidence(_BASE, short_stale)
        c_medium = _stale_confidence(_BASE, medium_stale)
        c_long = _stale_confidence(_BASE, long_stale)
        assert c_short >= c_medium >= c_long


# ---------------------------------------------------------------------------
# 5. Floor — confidence never falls below AIS_MIN_STALE_CONFIDENCE
# ---------------------------------------------------------------------------

class TestFloor:
    def test_very_long_gap_hits_floor(self):
        very_long = config.AIS_STALE_GAP_SECONDS + 1_000_000
        result = _stale_confidence(_BASE, very_long)
        assert result == pytest.approx(config.AIS_MIN_STALE_CONFIDENCE)

    def test_result_never_below_floor(self):
        for excess in (0, 3600, 36000, 360000, 3_600_000):
            gap = config.AIS_STALE_GAP_SECONDS + excess
            result = _stale_confidence(_BASE, gap)
            assert result >= config.AIS_MIN_STALE_CONFIDENCE

    def test_floor_matches_config_constant(self):
        """Floor is AIS_MIN_STALE_CONFIDENCE, not a hardcoded value."""
        result = _stale_confidence(_BASE, config.AIS_STALE_GAP_SECONDS + 1_000_000)
        assert result == pytest.approx(config.AIS_MIN_STALE_CONFIDENCE)


# ---------------------------------------------------------------------------
# 6. AIS replay records reflect decayed confidence
# ---------------------------------------------------------------------------

class TestReplayRecordsDecayedConfidence:
    def test_short_gap_record_confidence_is_not_decayed(self):
        """1-hour gap (= threshold): record confidence matches track.confidence."""
        obs = _gap_track(config.AIS_STALE_GAP_SECONDS)
        records = ingest_ais_track(obs)
        # Second record: gap == threshold → no decay
        second = records[1]
        # confidence in record should equal track.confidence (no staleness penalty)
        from custody.tracks import custody_confidence, update_uncertainty
        expected_uncertainty = update_uncertainty(5.0, config.AIS_STALE_GAP_SECONDS / 3600)
        expected_confidence = custody_confidence(expected_uncertainty)
        assert second["custody_confidence"] == pytest.approx(expected_confidence, abs=1e-6)

    def test_long_gap_record_confidence_is_lower_than_short_gap(self):
        """A 6-hour gap should produce lower confidence than a 1-hour gap."""
        short_obs = _gap_track(config.AIS_STALE_GAP_SECONDS)       # 1 h = threshold
        long_obs = _gap_track(config.AIS_STALE_GAP_SECONDS + 18000) # 6 h

        short_records = ingest_ais_track(short_obs)
        long_records = ingest_ais_track(long_obs)

        short_conf = short_records[1]["custody_confidence"]
        long_conf = long_records[1]["custody_confidence"]
        assert long_conf < short_conf

    def test_very_long_gap_confidence_hits_floor(self):
        obs = _gap_track(config.AIS_STALE_GAP_SECONDS + 1_000_000)
        records = ingest_ais_track(obs)
        assert records[1]["custody_confidence"] == pytest.approx(
            config.AIS_MIN_STALE_CONFIDENCE, abs=1e-6
        )


# ---------------------------------------------------------------------------
# 7. Planner receives the decayed confidence
# ---------------------------------------------------------------------------

class TestPlannerReceivesDecayedConfidence:
    def test_plan_collection_called_with_decayed_confidence(self):
        """With a stale gap, plan_collection's confidence arg < track.confidence."""
        stale_gap = config.AIS_STALE_GAP_SECONDS + 18000  # 5 h excess
        obs = _gap_track(stale_gap)

        plan_calls = []
        import custody.ais as ais_mod
        original = ais_mod.plan_collection

        def recording_plan(track, score, confidence, *args, **kwargs):
            plan_calls.append(confidence)
            return original(track, score, confidence, *args, **kwargs)

        with patch("custody.ais.plan_collection", side_effect=recording_plan):
            ingest_ais_track(obs)

        assert plan_calls, "plan_collection was never called"
        # The confidence passed must be the decayed value, not raw track.confidence
        from custody.tracks import custody_confidence, update_uncertainty
        raw_uncertainty = update_uncertainty(5.0, stale_gap / 3600)
        raw_confidence = custody_confidence(raw_uncertainty)
        decayed = _stale_confidence(raw_confidence, stale_gap)
        assert plan_calls[0] == pytest.approx(decayed, abs=1e-6)

    def test_planner_gets_lower_confidence_for_stale_gap(self):
        """Planner's confidence arg is lower for a stale track than a fresh one."""
        fresh_gap = config.AIS_STALE_GAP_SECONDS  # no decay
        stale_gap = config.AIS_STALE_GAP_SECONDS + 18000

        fresh_calls = []
        stale_calls = []
        import custody.ais as ais_mod
        original = ais_mod.plan_collection

        def recording_fresh(track, score, confidence, *args, **kwargs):
            fresh_calls.append(confidence)
            return original(track, score, confidence, *args, **kwargs)

        def recording_stale(track, score, confidence, *args, **kwargs):
            stale_calls.append(confidence)
            return original(track, score, confidence, *args, **kwargs)

        with patch("custody.ais.plan_collection", side_effect=recording_fresh):
            ingest_ais_track(_gap_track(fresh_gap))
        with patch("custody.ais.plan_collection", side_effect=recording_stale):
            ingest_ais_track(_gap_track(stale_gap))

        assert stale_calls[0] < fresh_calls[0]


# ---------------------------------------------------------------------------
# 8. Decision trace confidence reflects the decay
# ---------------------------------------------------------------------------

class TestTraceConfidenceReflectsDecay:
    def test_trace_confidence_lower_for_stale_track(self):
        fresh_records = ingest_ais_track(_gap_track(config.AIS_STALE_GAP_SECONDS))
        stale_records = ingest_ais_track(
            _gap_track(config.AIS_STALE_GAP_SECONDS + 18000)
        )

        fresh_trace = fresh_records[1]["decision_trace"]
        stale_trace = stale_records[1]["decision_trace"]

        assert isinstance(fresh_trace, DecisionTrace)
        assert isinstance(stale_trace, DecisionTrace)
        # Confidence is recorded in trace.inputs.custody_confidence
        assert (
            stale_trace.inputs.custody_confidence
            < fresh_trace.inputs.custody_confidence
        )

    def test_trace_confidence_matches_record_confidence(self):
        """Trace inputs.custody_confidence matches record['custody_confidence']."""
        stale_records = ingest_ais_track(
            _gap_track(config.AIS_STALE_GAP_SECONDS + 7200)
        )
        record = stale_records[1]
        trace = record["decision_trace"]
        assert isinstance(trace, DecisionTrace)
        assert trace.inputs.custody_confidence == pytest.approx(
            record["custody_confidence"], abs=1e-6
        )


# ---------------------------------------------------------------------------
# 9. Non-AIS paths are unaffected
# ---------------------------------------------------------------------------

class TestSimulationUnaffected:
    def test_simulation_output_unchanged(self):
        """run_simulation should still work and not call _stale_confidence."""
        from custody.simulate import run_simulation
        records = run_simulation()
        assert len(records) > 0
        valid_actions = {"NONE", "TASK", "HOLD", "NO_SENSOR", "PREEMPTED"}
        for r in records:
            assert r["action"] in valid_actions

    def test_stale_confidence_not_imported_by_simulate(self):
        """simulate.py has no dependency on the AIS staleness helper."""
        import ast
        import pathlib
        src = (
            pathlib.Path(__file__).parent.parent / "src" / "custody" / "simulate.py"
        ).read_text(encoding="utf-8")
        assert "_stale_confidence" not in src


# ---------------------------------------------------------------------------
# 10. Decay uses config constants, not hardcoded values
# ---------------------------------------------------------------------------

class TestUsesConfigConstants:
    def test_threshold_boundary_matches_config(self):
        """One second below threshold → no decay; one second above → decay."""
        below = config.AIS_STALE_GAP_SECONDS - 1
        above = config.AIS_STALE_GAP_SECONDS + 1
        assert _stale_confidence(_BASE, below) == _BASE
        assert _stale_confidence(_BASE, above) < _BASE

    def test_rate_determines_decay_amount(self):
        """Decay factor at excess e is exp(-rate * e), matching the config rate."""
        excess = 7200.0  # 2 hours above threshold
        gap = config.AIS_STALE_GAP_SECONDS + excess
        result = _stale_confidence(_BASE, gap)
        expected_decay = math.exp(-config.AIS_STALE_CONFIDENCE_DECAY_RATE * excess)
        expected = max(_BASE * expected_decay, config.AIS_MIN_STALE_CONFIDENCE)
        assert result == pytest.approx(expected, abs=1e-9)
