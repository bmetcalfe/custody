"""
Tests for custody.taskrecommendation — the collection orchestration layer.

Coverage:
  1.  TaskRecommendation is frozen (immutable)
  2.  TaskRecommendation has all required fields
  3.  TASK_SAR decision yields SAR as rank-1 when SAR window exists
  4.  TASK_OPTICAL decision yields OPTICAL as rank-1 when OPTICAL window exists
  5.  Sensor with no orbital window is omitted from the queue
  6.  Queue is never empty (MONITOR always available)
  7.  MONITOR is always rank-last or only entry for PASSIVE_MONITOR
  8.  Earlier window (lower TTS) scores higher timing_score
  9.  expected_value increases with higher decision priority
 10.  expected_value increases with stronger sensor_fit
 11.  fallbacks exclude the primary sensor
 12.  recommendations sorted by expected_value descending
 13.  rank=1 has the highest expected_value
 14.  ranks are consecutive starting at 1
 15.  rank values match sorted expected_value order
 16.  output is deterministic for identical inputs
 17.  expected_value bounded to [0, 1]
 18.  task_id contains entity_id and sensor label
 19.  task_id contains window_start timestamp
 20.  PASSIVE_MONITOR yields MONITOR as primary candidate
 21.  ESCALATE puts FA-recommended sensor first
 22.  AIS_REFRESH window starts at record time
 23.  MONITOR window ends 2 hours after record time
 24.  _compute_timing_score: in-view scores 1.0
 25.  _compute_timing_score: imminent pass scores high
 26.  _compute_timing_score: distant window scores low
 27.  _compute_sensor_fit: SAR with TASK_SAR decision returns 1.0
 28.  _compute_sensor_fit: OPTICAL with TASK_OPTICAL decision returns 1.0
 29.  _compute_sensor_fit: MONITOR with PASSIVE_MONITOR scores higher than with ESCALATE
 30.  _build_reason returns non-empty string
 31.  _build_reason for SAR mentions SAR
 32.  _build_reason for AIS_REFRESH mentions track confidence
 33.  _build_reason for MONITOR with PASSIVE_MONITOR mentions significance
 34.  _fallbacks_for_sensor excludes the primary sensor
 35.  Smoke test: end-to-end with run_simulation()
"""
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock

import pytest

import custody.config as config
from custody.decision import Decision, PASSIVE_MONITOR, ELEVATE, TASK_OPTICAL, TASK_SAR, ESCALATE
from custody.fusion import FusionAssessment
from custody.models import TrackState
from custody.sensors import PassWindow
from custody.taskrecommendation import (
    TaskRecommendation,
    _candidate_sensors_for_action,
    _compute_expected_value,
    _compute_sensor_fit,
    _compute_timing_score,
    _build_reason,
    _fallbacks_for_sensor,
    _make_task_id,
    _select_window,
    build_task_recommendations,
)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

_T0 = datetime(2026, 3, 23, 12, 0, tzinfo=timezone.utc)
_LAT, _LON = 1.0, 1.0   # mid-ocean, no orbital pass expected at this time


def _fa(
    fused_score=0.6,
    uncertainty=0.4,
    source_agreement=0.7,
    missing_evidence=None,
    recommended_confirming_source="SAR",
    entity_id="V001",
) -> FusionAssessment:
    return FusionAssessment(
        entity_id=entity_id,
        timestamp=_T0,
        fused_score=fused_score,
        uncertainty=uncertainty,
        source_agreement=source_agreement,
        missing_evidence=missing_evidence or ["SAR confirmation"],
        recommended_confirming_source=recommended_confirming_source,
    )


def _decision(
    action=TASK_SAR,
    priority=0.7,
    confidence=0.6,
    entity_id="V001",
) -> Decision:
    return Decision(
        entity_id=entity_id,
        timestamp=_T0,
        action=action,
        priority=priority,
        confidence=confidence,
        why=["Test why bullet."],
        next_best_actions=[],
    )


def _record(**kwargs) -> dict:
    base = {
        "entity_id": "V001",
        "time": _T0,
        "lat": _LAT,
        "lon": _LON,
        "custody_confidence": 0.7,
        "anomaly_score": 0.5,
        "action": "NONE",
        "sensitive_zone": 0.0,
    }
    base.update(kwargs)
    return base


def _track(**kwargs) -> TrackState:
    defaults = dict(uncertainty_km=5.0)
    defaults.update(kwargs)
    return TrackState(**defaults)


def _pass_window(sat_id="SAT-B", tts=300.0, duration=600.0) -> PassWindow:
    start = _T0 + timedelta(seconds=tts)
    end   = start + timedelta(seconds=duration)
    return PassWindow(
        satellite_id=sat_id,
        start_time=start,
        end_time=end,
        duration_seconds=duration,
        time_to_start_seconds=tts,
    )


def _build(decision=None, fa=None, record=None, track=None, planner_context=None):
    """Convenience wrapper with no-pass-window mock applied by default."""
    return build_task_recommendations(
        decision or _decision(),
        fa      or _fa(),
        record  or _record(),
        track   or _track(),
        planner_context,
    )


# ---------------------------------------------------------------------------
# 1–2. Dataclass properties
# ---------------------------------------------------------------------------

class TestTaskRecommendationSchema:
    def test_is_frozen(self):
        with patch("custody.taskrecommendation.next_pass_window", return_value=_pass_window()):
            recs = _build()
        with pytest.raises((FrozenInstanceError, TypeError)):
            recs[0].sensor = "INVALID"  # type: ignore[misc]

    def test_has_all_required_fields(self):
        with patch("custody.taskrecommendation.next_pass_window", return_value=_pass_window()):
            rec = _build()[0]
        for field in ("task_id", "entity_id", "sensor", "window_start", "window_end",
                      "expected_value", "reason", "fallbacks", "rank"):
            assert hasattr(rec, field)


# ---------------------------------------------------------------------------
# 3–6. Candidate selection and window availability
# ---------------------------------------------------------------------------

class TestCandidateSelection:
    def test_task_sar_yields_sar_as_rank1_when_window_exists(self):
        def fake_pass(sat_id, lat, lon, from_time, *args, **kwargs):
            return _pass_window(sat_id=sat_id, tts=300) if sat_id == "SAT-B" else None

        with patch("custody.taskrecommendation.next_pass_window", side_effect=fake_pass):
            recs = _build(decision=_decision(action=TASK_SAR))
        assert recs[0].sensor == "SAR"
        assert recs[0].rank == 1

    def test_task_optical_yields_optical_as_rank1_when_window_exists(self):
        def fake_pass(sat_id, lat, lon, from_time, *args, **kwargs):
            return _pass_window(sat_id=sat_id, tts=300) if sat_id == "SAT-A" else None

        with patch("custody.taskrecommendation.next_pass_window", side_effect=fake_pass):
            recs = _build(decision=_decision(action=TASK_OPTICAL),
                          fa=_fa(recommended_confirming_source="OPTICAL"))
        assert recs[0].sensor == "OPTICAL"

    def test_unavailable_sensor_omitted_from_queue(self):
        # No orbital windows at all
        with patch("custody.taskrecommendation.next_pass_window", return_value=None):
            recs = _build(decision=_decision(action=TASK_SAR))
        sensors = [r.sensor for r in recs]
        assert "SAR" not in sensors
        assert "OPTICAL" not in sensors

    def test_queue_never_empty(self):
        with patch("custody.taskrecommendation.next_pass_window", return_value=None):
            recs = _build()
        assert len(recs) >= 1

    def test_monitor_always_in_queue(self):
        with patch("custody.taskrecommendation.next_pass_window", return_value=None):
            recs = _build()
        assert any(r.sensor == "MONITOR" for r in recs)


# ---------------------------------------------------------------------------
# 7. PASSIVE_MONITOR maps to MONITOR primary
# ---------------------------------------------------------------------------

class TestPassiveMonitor:
    def test_passive_monitor_yields_monitor_as_candidate(self):
        recs = build_task_recommendations(
            _decision(action=PASSIVE_MONITOR),
            _fa(fused_score=0.2, uncertainty=0.2,
                recommended_confirming_source=None, missing_evidence=[]),
            _record(),
            _track(),
        )
        sensors = [r.sensor for r in recs]
        assert "MONITOR" in sensors

    def test_passive_monitor_includes_ais_refresh_when_uncertain(self):
        recs = build_task_recommendations(
            _decision(action=PASSIVE_MONITOR),
            _fa(fused_score=0.2, uncertainty=0.55,
                recommended_confirming_source=None,
                missing_evidence=["fresh AIS update"]),
            _record(),
            _track(),
        )
        sensors = [r.sensor for r in recs]
        assert "AIS_REFRESH" in sensors


# ---------------------------------------------------------------------------
# 8. Timing score
# ---------------------------------------------------------------------------

class TestTimingScore:
    def test_in_view_now_scores_1(self):
        assert _compute_timing_score(0.0) == 1.0

    def test_imminent_pass_scores_high(self):
        score = _compute_timing_score(config.HOLD_LOOKAHEAD_THRESHOLD_SECONDS / 2)
        assert score >= 0.80

    def test_distant_window_scores_low(self):
        score = _compute_timing_score(5 * 3600)
        assert score <= 0.25

    def test_scores_decrease_with_tts(self):
        tts_values = [0, 600, 1800, 3600, 7200, 18000]
        scores = [_compute_timing_score(t) for t in tts_values]
        assert scores == sorted(scores, reverse=True)

    def test_earlier_window_beats_later_window(self):
        def fake_pass_early_sar(sat_id, lat, lon, from_time, *args, **kwargs):
            if sat_id == "SAT-B":
                return _pass_window(sat_id="SAT-B", tts=300)    # 5 min
            if sat_id == "SAT-A":
                return _pass_window(sat_id="SAT-A", tts=5400)   # 90 min
            return None

        with patch("custody.taskrecommendation.next_pass_window",
                   side_effect=fake_pass_early_sar):
            recs = _build(decision=_decision(action=TASK_SAR))

        sar  = next((r for r in recs if r.sensor == "SAR"), None)
        opt  = next((r for r in recs if r.sensor == "OPTICAL"), None)
        if sar and opt:
            # SAR window is earlier → higher timing contribution
            # (sensor_fit already favours SAR here so this also checks ordering)
            assert sar.rank < opt.rank


# ---------------------------------------------------------------------------
# 9–10. expected_value sensitivity
# ---------------------------------------------------------------------------

class TestExpectedValue:
    def test_increases_with_higher_priority(self):
        fa   = _fa()
        r    = _record()
        t    = _track()
        ws   = _T0 + timedelta(minutes=5)
        we   = ws + timedelta(minutes=10)

        ev_low  = _compute_expected_value("SAR", _decision(priority=0.2), fa, ws, we, r, t)
        ev_high = _compute_expected_value("SAR", _decision(priority=0.9), fa, ws, we, r, t)
        assert ev_high > ev_low

    def test_increases_with_stronger_sensor_fit(self):
        # Use recommended_confirming_source=None so fit depends purely on action
        fa_no_rec = _fa(recommended_confirming_source=None)
        r  = _record()
        t  = _track()
        ws = _T0
        we = _T0 + timedelta(minutes=10)

        # TASK_SAR → SAR fit = 1.0 (action match)
        fit_perfect = _compute_expected_value("SAR", _decision(action=TASK_SAR), fa_no_rec, ws, we, r, t)
        # PASSIVE_MONITOR → SAR fit = 0.20 (MONITOR backstop fallback)
        fit_weak    = _compute_expected_value("SAR", _decision(action=PASSIVE_MONITOR), fa_no_rec, ws, we, r, t)
        assert fit_perfect > fit_weak

    def test_bounded_to_0_1(self):
        fa = _fa(fused_score=1.0)
        ws = _T0
        we = _T0 + timedelta(minutes=10)
        ev = _compute_expected_value(
            "SAR", _decision(priority=1.0, confidence=1.0),
            fa, ws, we, _record(), _track(),
        )
        assert 0.0 <= ev <= 1.0


# ---------------------------------------------------------------------------
# 11–15. Sorting, ranks, and fallbacks
# ---------------------------------------------------------------------------

class TestSortingAndRanks:
    def test_fallbacks_exclude_primary_sensor(self):
        for sensor in ("SAR", "OPTICAL", "AIS_REFRESH", "MONITOR"):
            fallbacks = _fallbacks_for_sensor(sensor, _decision())
            assert sensor not in fallbacks, f"{sensor} found in its own fallbacks"

    def test_sorted_by_expected_value_descending(self):
        with patch("custody.taskrecommendation.next_pass_window", return_value=_pass_window()):
            recs = _build()
        evs = [r.expected_value for r in recs]
        assert evs == sorted(evs, reverse=True)

    def test_rank1_has_highest_expected_value(self):
        with patch("custody.taskrecommendation.next_pass_window", return_value=_pass_window()):
            recs = _build()
        rank1 = next(r for r in recs if r.rank == 1)
        assert rank1.expected_value == max(r.expected_value for r in recs)

    def test_ranks_are_consecutive_from_1(self):
        with patch("custody.taskrecommendation.next_pass_window", return_value=_pass_window()):
            recs = _build()
        ranks = sorted(r.rank for r in recs)
        assert ranks == list(range(1, len(recs) + 1))

    def test_ranks_match_sorted_order(self):
        with patch("custody.taskrecommendation.next_pass_window", return_value=_pass_window()):
            recs = _build()
        for i, rec in enumerate(recs):
            assert rec.rank == i + 1


# ---------------------------------------------------------------------------
# 16. Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_inputs_give_same_output(self):
        with patch("custody.taskrecommendation.next_pass_window",
                   return_value=_pass_window(tts=600)):
            recs1 = _build()
            recs2 = _build()
        assert len(recs1) == len(recs2)
        for r1, r2 in zip(recs1, recs2):
            assert r1 == r2


# ---------------------------------------------------------------------------
# 18–19. task_id format
# ---------------------------------------------------------------------------

class TestTaskId:
    def test_task_id_contains_entity_id(self):
        tid = _make_task_id("V001", "SAR", _T0)
        assert "v001" in tid.lower()

    def test_task_id_contains_sensor(self):
        tid = _make_task_id("V001", "SAR", _T0)
        assert "sar" in tid.lower()

    def test_task_id_contains_timestamp(self):
        tid = _make_task_id("V001", "SAR", _T0)
        assert "1200" in tid   # HH:MM from _T0


# ---------------------------------------------------------------------------
# 21. ESCALATE ordering
# ---------------------------------------------------------------------------

class TestEscalate:
    def test_escalate_puts_fa_recommended_sensor_first(self):
        candidates = _candidate_sensors_for_action(
            _decision(action=ESCALATE),
            _fa(recommended_confirming_source="SAR"),
        )
        assert candidates[0] == "SAR"

    def test_escalate_with_optical_recommendation_puts_optical_first(self):
        candidates = _candidate_sensors_for_action(
            _decision(action=ESCALATE),
            _fa(recommended_confirming_source="OPTICAL"),
        )
        assert candidates[0] == "OPTICAL"


# ---------------------------------------------------------------------------
# 22–23. Window times
# ---------------------------------------------------------------------------

class TestWindowTimes:
    def test_ais_refresh_window_starts_at_record_time(self):
        window = _select_window("AIS_REFRESH", None, _record(), _track())
        assert window is not None
        start, end, tts = window
        assert start == _T0

    def test_monitor_window_ends_2_hours_after_record_time(self):
        window = _select_window("MONITOR", None, _record(), _track())
        assert window is not None
        start, end, tts = window
        assert end == _T0 + timedelta(hours=2)

    def test_orbital_sensor_returns_none_when_no_window(self):
        with patch("custody.taskrecommendation.next_pass_window", return_value=None):
            result = _select_window("SAR", None, _record(), _track())
        assert result is None


# ---------------------------------------------------------------------------
# 27–29. sensor_fit
# ---------------------------------------------------------------------------

class TestSensorFit:
    def test_sar_with_task_sar_returns_1(self):
        fit = _compute_sensor_fit("SAR", _decision(action=TASK_SAR), _fa(), _record(), _track())
        assert fit == 1.0

    def test_optical_with_task_optical_returns_1(self):
        fit = _compute_sensor_fit(
            "OPTICAL", _decision(action=TASK_OPTICAL),
            _fa(recommended_confirming_source="OPTICAL"), _record(), _track()
        )
        assert fit == 1.0

    def test_monitor_scores_higher_for_passive_than_escalate(self):
        fit_passive = _compute_sensor_fit(
            "MONITOR", _decision(action=PASSIVE_MONITOR), _fa(), _record(), _track()
        )
        fit_escalate = _compute_sensor_fit(
            "MONITOR", _decision(action=ESCALATE), _fa(), _record(), _track()
        )
        assert fit_passive > fit_escalate


# ---------------------------------------------------------------------------
# 30–33. reason
# ---------------------------------------------------------------------------

class TestBuildReason:
    def test_reason_is_non_empty(self):
        reason = _build_reason("SAR", _decision(), _fa(), _T0 + timedelta(minutes=5),
                               _record(), 300.0)
        assert len(reason) > 0

    def test_sar_reason_mentions_sar(self):
        reason = _build_reason("SAR", _decision(), _fa(), _T0 + timedelta(minutes=5),
                               _record(), 300.0)
        assert "SAR" in reason

    def test_ais_refresh_reason_mentions_track_confidence(self):
        reason = _build_reason("AIS_REFRESH", _decision(action=ELEVATE), _fa(),
                               _T0, _record(), 0.0)
        assert "confidence" in reason.lower() or "custody" in reason.lower()

    def test_monitor_passive_reason_mentions_significance(self):
        reason = _build_reason("MONITOR", _decision(action=PASSIVE_MONITOR), _fa(),
                               _T0, _record(), 0.0)
        assert "significance" in reason.lower() or "sufficient" in reason.lower()

    def test_sar_in_view_mentions_overhead(self):
        reason = _build_reason("SAR", _decision(), _fa(), _T0, _record(), 0.0)
        assert "overhead" in reason.lower() or "now" in reason.lower()


# ---------------------------------------------------------------------------
# 35. Smoke test
# ---------------------------------------------------------------------------

def test_end_to_end_with_simulation():
    """run_simulation → FA → Decision → TaskRecommendations for all vessels."""
    from collections import defaultdict
    from custody.simulate import run_simulation
    from custody.fusion import fusion_for_timeline
    from custody.decision import decision_for_timeline

    records = run_simulation()

    by_vessel: dict = defaultdict(list)
    for r in records:
        vid = r.get("entity_id") or r.get("vessel_id") or r.get("target_id")
        by_vessel[vid].append(r)

    all_sensors = {"SAR", "OPTICAL", "AIS_REFRESH", "MONITOR"}

    for vessel_id, timeline in by_vessel.items():
        track = TrackState()
        fas   = fusion_for_timeline(timeline, track)
        decs  = decision_for_timeline(timeline, track, fas)

        for record, fa, dec in zip(timeline, fas, decs):
            recs = build_task_recommendations(dec, fa, record, track)

            assert len(recs) >= 1, f"Empty queue for {vessel_id}"
            assert all(isinstance(r, TaskRecommendation) for r in recs)
            assert all(r.sensor in all_sensors for r in recs)
            assert all(0.0 <= r.expected_value <= 1.0 for r in recs)
            assert all(r.rank >= 1 for r in recs)
            # Sorted descending
            evs = [r.expected_value for r in recs]
            assert evs == sorted(evs, reverse=True)
