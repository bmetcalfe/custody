"""
Tests for custody/planner.py.

Covers:
  - compute_target_priority: ordering, edge cases, weight contributions
  - compute_task_value: freshness decay, worsening boost, monotonicity
  - plan_collection returns a valid CollectionDecision in all branches
  - HOLD fires when task value is below TASK_VALUE_THRESHOLD
  - Task value recovers over time and rises with anomaly worsening
  - CUSTODY_TASK_CONFIDENCE_THRESHOLD gates tasking eligibility
  - PREEMPTED fires when sensors were globally available but pre-empted
  - PREEMPTED does not override HOLD or NONE
"""
import math
from datetime import datetime, UTC, timedelta
from unittest.mock import patch

import pytest

from custody.models import TrackState
from custody.planner import compute_target_priority, compute_task_value, plan_collection, score_opportunity
from custody.sensors import SensorOpportunity, get_sensor_opportunities
import custody.config as config
import custody.planner as planner_module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# hour 10 → A1 available (even hour); used as the default current_time
T0 = datetime(2026, 3, 23, 10, 0, tzinfo=UTC)

_NOMINAL_BREAKDOWN = {
    "sensitive_zone":  type("R", (), {"score": 0.0})(),
    "loitering":       type("R", (), {"score": 0.0})(),
    "route_deviation": type("R", (), {"score": 0.0})(),
}

# Sensor list available at T0 (A1: fast_revisit)
_SENSORS = get_sensor_opportunities(T0)

# A single minimal sensor for tests that just need some opportunity present
_ONE_SENSOR = [
    SensorOpportunity(
        sensor_id="TEST1",
        sensor_type="fast_revisit",
        success_prob=1.0,
        resolution="medium",
        cost=1.0,
        available_from=T0,
        available_to=T0,
    )
]


def _fresh_track() -> TrackState:
    return TrackState()


def _collected_track(hours_ago: float, anomaly_at_collection: float = 0.0) -> TrackState:
    """Return a TrackState with a successful collection `hours_ago` hours before T0."""
    track = TrackState()
    collection_time = T0 - timedelta(hours=hours_ago)
    track.record_collection(collection_time, anomaly_at_collection, new_uncertainty=4.0)
    return track


# ---------------------------------------------------------------------------
# Basic contract
# ---------------------------------------------------------------------------

class TestPlanCollectionContract:
    def test_returns_none_for_nominal_state(self):
        """Low anomaly, high confidence, no recent collection → NONE."""
        decision = plan_collection(_fresh_track(), score=0.1, confidence=0.9,
                                   breakdown=_NOMINAL_BREAKDOWN, current_time=T0,
                                   opportunities=_SENSORS)
        assert decision.action == "NONE"

    def test_action_field_is_always_valid(self):
        for score, conf in [(0.1, 0.9), (1.0, 0.9), (0.6, 0.4)]:
            decision = plan_collection(_fresh_track(), score=score, confidence=conf,
                                       breakdown=_NOMINAL_BREAKDOWN, current_time=T0,
                                       opportunities=_SENSORS)
            assert decision.action in {"NONE", "HOLD", "TASK", "NO_SENSOR", "PREEMPTED"}

    def test_no_sensor_when_opportunities_empty(self):
        """When opportunities list is empty, action is NO_SENSOR."""
        decision = plan_collection(_fresh_track(), score=1.0, confidence=0.4,
                                   breakdown=_NOMINAL_BREAKDOWN, current_time=T0,
                                   opportunities=[])
        assert decision.action == "NO_SENSOR"

    def test_task_when_anomaly_high_and_sensors_available(self):
        decision = plan_collection(_fresh_track(), score=1.0, confidence=0.9,
                                   breakdown=_NOMINAL_BREAKDOWN, current_time=T0,
                                   opportunities=_SENSORS)
        assert decision.action == "TASK"

    def test_task_when_confidence_below_threshold(self):
        """confidence below CUSTODY_TASK_CONFIDENCE_THRESHOLD triggers tasking."""
        decision = plan_collection(
            _fresh_track(),
            score=0.1,
            confidence=config.CUSTODY_TASK_CONFIDENCE_THRESHOLD - 0.1,
            breakdown=_NOMINAL_BREAKDOWN,
            current_time=T0,
            opportunities=_SENSORS,
        )
        assert decision.action == "TASK"

    def test_none_when_confidence_at_threshold(self):
        """confidence exactly at CUSTODY_TASK_CONFIDENCE_THRESHOLD is not enough alone."""
        decision = plan_collection(
            _fresh_track(),
            score=0.1,
            confidence=config.CUSTODY_TASK_CONFIDENCE_THRESHOLD,
            breakdown=_NOMINAL_BREAKDOWN,
            current_time=T0,
            opportunities=_SENSORS,
        )
        assert decision.action == "NONE"

    def test_no_prior_collection_never_holds(self):
        """A fresh track with no prior collection is never HOLD (nothing to suppress)."""
        decision = plan_collection(
            _fresh_track(), score=1.0, confidence=0.9,
            breakdown=_NOMINAL_BREAKDOWN, current_time=T0,
            opportunities=_SENSORS,
        )
        assert decision.action != "HOLD"


# ---------------------------------------------------------------------------
# compute_task_value — pure function tests
# ---------------------------------------------------------------------------

class TestComputeTaskValue:
    def test_no_prior_collection_returns_base(self):
        """Without a prior collection, task value equals the base formula."""
        score, confidence = 1.0, 0.5
        expected = (
            (score / config.CRITICAL_ANOMALY_THRESHOLD) * 0.55
            + (1.0 - confidence) * 0.35
        )
        result = compute_task_value(score, confidence, None, None)
        assert result == pytest.approx(expected, abs=1e-9)

    def test_recent_collection_reduces_task_value(self):
        """A very recent collection lowers task value compared to no collection."""
        base = compute_task_value(0.8, 0.7, None, None)
        recent = compute_task_value(0.8, 0.7, 0.1, 0.8)   # 6 minutes ago
        assert recent < base

    def test_task_value_increases_with_time(self):
        """Task value rises monotonically as time since last collection grows."""
        values = [compute_task_value(0.8, 0.7, h, 0.8) for h in [0, 1, 3, 6, 12]]
        assert values == sorted(values), "task value should be monotonically increasing with time"

    def test_anomaly_worsening_increases_task_value(self):
        """Anomaly worsening relative to the last collection increases task value."""
        no_worsening = compute_task_value(0.8, 0.7, 1.5, 0.8)   # same score
        worsened     = compute_task_value(0.8, 0.7, 1.5, 0.5)   # improved from 0.5 → 0.8
        assert worsened > no_worsening

    def test_no_worsening_boost_when_score_improves(self):
        """Score lower than last collection (improvement) adds no worsening boost."""
        improved   = compute_task_value(0.5, 0.7, 1.5, 0.9)  # score fell from 0.9 to 0.5
        no_change  = compute_task_value(0.5, 0.7, 1.5, 0.5)  # same
        assert improved == pytest.approx(no_change, abs=1e-9)

    def test_compound_boost_is_additive(self):
        """compound_boost adds a small positive increment to task value."""
        without = compute_task_value(0.8, 0.7, None, None, compound_boost=0.0)
        with_b  = compute_task_value(0.8, 0.7, None, None, compound_boost=0.5)
        assert with_b > without

    def test_freshness_suppression_can_go_negative(self):
        """Immediately after collection (hours=0) task value may be negative."""
        v = compute_task_value(0.8, 0.8, 0.0, 0.8)
        assert v < 0.0

    def test_freshness_fully_decays(self):
        """After a very long time the freshness term is negligible."""
        very_old  = compute_task_value(0.8, 0.7, 1000.0, 0.8)
        no_prior  = compute_task_value(0.8, 0.7, None, None)
        assert abs(very_old - no_prior) < 0.001


# ---------------------------------------------------------------------------
# HOLD via task-value model
# ---------------------------------------------------------------------------

class TestTaskValueHold:
    def test_fresh_collection_yields_hold(self):
        """Collection just made → task value below threshold → HOLD."""
        track = _collected_track(hours_ago=0.5, anomaly_at_collection=0.8)
        decision = plan_collection(track, score=0.8, confidence=0.8,
                                   breakdown=_NOMINAL_BREAKDOWN, current_time=T0,
                                   opportunities=_SENSORS)
        assert decision.action == "HOLD"

    def test_old_collection_allows_retask(self):
        """Collection old enough that freshness has decayed → TASK."""
        track = _collected_track(hours_ago=8.0, anomaly_at_collection=0.8)
        decision = plan_collection(track, score=0.8, confidence=0.5,
                                   breakdown=_NOMINAL_BREAKDOWN, current_time=T0,
                                   opportunities=_SENSORS)
        assert decision.action == "TASK"

    def test_task_value_comparison_orders_correctly(self):
        """Recently-collected vessel has lower task value than same vessel with no collection."""
        score, confidence = 0.9, 0.7
        track_fresh    = _fresh_track()
        track_recent   = _collected_track(hours_ago=0.5, anomaly_at_collection=0.9)
        val_fresh  = compute_task_value(score, confidence, None, None)
        val_recent = compute_task_value(
            score, confidence,
            track_recent.hours_since_collection(T0),
            track_recent.last_collection_anomaly_score,
        )
        assert val_fresh > val_recent

    def test_significant_worsening_overrides_hold(self):
        """Large anomaly increase since last collection drives task value above threshold."""
        # collected recently when score was low; score has jumped significantly
        track = _collected_track(hours_ago=0.5, anomaly_at_collection=0.3)
        decision = plan_collection(track, score=1.4, confidence=0.7,
                                   breakdown=_NOMINAL_BREAKDOWN, current_time=T0,
                                   opportunities=_SENSORS)
        assert decision.action == "TASK"

    def test_small_worsening_preserves_hold(self):
        """Negligible anomaly change keeps task value below threshold → HOLD."""
        track = _collected_track(hours_ago=0.5, anomaly_at_collection=0.8)
        decision = plan_collection(track, score=0.81, confidence=0.8,
                                   breakdown=_NOMINAL_BREAKDOWN, current_time=T0,
                                   opportunities=_SENSORS)
        assert decision.action == "HOLD"

    def test_hold_action_reason_mentions_task_value(self):
        """HOLD action_reason should reference the task value."""
        track = _collected_track(hours_ago=0.5, anomaly_at_collection=0.8)
        decision = plan_collection(track, score=0.8, confidence=0.8,
                                   breakdown=_NOMINAL_BREAKDOWN, current_time=T0,
                                   opportunities=_SENSORS)
        assert "task value" in decision.action_reason.lower()

    def test_hold_then_task_as_time_passes(self):
        """Same vessel transitions from HOLD to TASK as time advances."""
        anomaly_at_collection = 0.8
        hold_seen = False
        task_seen = False
        for hours_ago in [0.5, 1, 2, 4, 6, 10]:
            track = _collected_track(hours_ago=hours_ago,
                                     anomaly_at_collection=anomaly_at_collection)
            decision = plan_collection(track, score=0.8, confidence=0.6,
                                       breakdown=_NOMINAL_BREAKDOWN, current_time=T0,
                                       opportunities=_SENSORS)
            if decision.action == "HOLD":
                hold_seen = True
            elif decision.action == "TASK":
                task_seen = True
        assert hold_seen, "expected HOLD for recent collections"
        assert task_seen, "expected TASK for old collections"

    def test_compound_boost_near_hold_threshold_enables_tasking(self):
        """When task value is just below the HOLD threshold, a compound_boost
        of 1.0 (max) adds 0.10 to task value and lifts it above the threshold.

        Parameters chosen so that without boost task_value ≈ 0.179 < 0.2 (HOLD)
        and with boost=1.0 task_value ≈ 0.279 > 0.2 (proceeds to TASK).
        hours_ago=3.0 puts freshness at exp(-1) ≈ 0.368.
        """
        track = _collected_track(hours_ago=3.0, anomaly_at_collection=0.8)
        decision_no_boost = plan_collection(
            track, score=0.8, confidence=0.8,
            breakdown=_NOMINAL_BREAKDOWN, current_time=T0,
            opportunities=_ONE_SENSOR,
        )
        assert decision_no_boost.action == "HOLD", (
            "expected HOLD without compound_boost (task value below threshold)"
        )

        track2 = _collected_track(hours_ago=3.0, anomaly_at_collection=0.8)
        decision_boosted = plan_collection(
            track2, score=0.8, confidence=0.8,
            breakdown=_NOMINAL_BREAKDOWN, current_time=T0,
            opportunities=_ONE_SENSOR, compound_boost=1.0,
        )
        assert decision_boosted.action == "TASK", (
            "expected TASK with compound_boost=1.0 lifting task value above threshold"
        )

    def test_default_compound_boost_preserves_prior_behavior(self):
        """Calling plan_collection without compound_boost and with compound_boost=0.0
        produce identical results — the default is a no-op."""
        track_a = _collected_track(hours_ago=3.0, anomaly_at_collection=0.8)
        track_b = _collected_track(hours_ago=3.0, anomaly_at_collection=0.8)
        decision_default = plan_collection(
            track_a, score=0.8, confidence=0.8,
            breakdown=_NOMINAL_BREAKDOWN, current_time=T0,
            opportunities=_ONE_SENSOR,
        )
        decision_explicit = plan_collection(
            track_b, score=0.8, confidence=0.8,
            breakdown=_NOMINAL_BREAKDOWN, current_time=T0,
            opportunities=_ONE_SENSOR, compound_boost=0.0,
        )
        assert decision_default.action == decision_explicit.action


# ---------------------------------------------------------------------------
# Threshold sensitivity — confirms config constants drive behavior
# ---------------------------------------------------------------------------

class TestThresholdSensitivity:
    def test_raising_custody_threshold_triggers_more_tasking(self):
        """Patching CUSTODY_TASK_CONFIDENCE_THRESHOLD higher makes previously-adequate
        confidence insufficient, triggering tasking."""
        confidence = config.CUSTODY_TASK_CONFIDENCE_THRESHOLD + 0.05  # currently above threshold
        track = _fresh_track()

        # Baseline: low anomaly, confidence above threshold → NONE
        baseline = plan_collection(track, score=0.1, confidence=confidence,
                                   breakdown=_NOMINAL_BREAKDOWN, current_time=T0,
                                   opportunities=_SENSORS)
        assert baseline.action == "NONE"

        # Raise threshold above current confidence → now considered for tasking
        higher = confidence + 0.1
        with patch.object(planner_module, "CUSTODY_TASK_CONFIDENCE_THRESHOLD", higher):
            decision = plan_collection(track, score=0.1, confidence=confidence,
                                       breakdown=_NOMINAL_BREAKDOWN, current_time=T0,
                                       opportunities=_SENSORS)
        assert decision.action == "TASK"

    def test_raising_task_value_threshold_extends_hold(self):
        """A higher TASK_VALUE_THRESHOLD causes a vessel that would be TASK to be HOLD."""
        # Use a collection old enough to normally allow re-tasking
        track = _collected_track(hours_ago=5.0, anomaly_at_collection=0.8)
        baseline = plan_collection(track, score=0.8, confidence=0.6,
                                   breakdown=_NOMINAL_BREAKDOWN, current_time=T0,
                                   opportunities=_SENSORS)
        assert baseline.action == "TASK"

        # Raise threshold to a value we know exceeds the current task value
        with patch.object(planner_module, "TASK_VALUE_THRESHOLD", 10.0):
            decision = plan_collection(track, score=0.8, confidence=0.6,
                                       breakdown=_NOMINAL_BREAKDOWN, current_time=T0,
                                       opportunities=_SENSORS)
        assert decision.action == "HOLD"

    def test_raising_decay_hours_slows_recovery(self):
        """A larger REVISIT_DECAY_HOURS keeps freshness elevated longer, suppressing re-tasking.

        In exp(-t/τ), a larger τ means slower decay: freshness stays high at t=4h
        when τ is large, resulting in more suppression and a lower task value.
        """
        track = _collected_track(hours_ago=4.0, anomaly_at_collection=0.8)
        baseline = plan_collection(track, score=0.8, confidence=0.6,
                                   breakdown=_NOMINAL_BREAKDOWN, current_time=T0,
                                   opportunities=_SENSORS)
        assert baseline.action == "TASK"

        # Very large decay constant → freshness barely decays at 4h → task_value stays low
        with patch.object(planner_module, "REVISIT_DECAY_HOURS", 1000.0):
            decision = plan_collection(track, score=0.8, confidence=0.6,
                                       breakdown=_NOMINAL_BREAKDOWN, current_time=T0,
                                       opportunities=_SENSORS)
        assert decision.action == "HOLD"


# ---------------------------------------------------------------------------
# Arbitration / PREEMPTED
# ---------------------------------------------------------------------------

class TestArbitration:
    def test_preempted_when_sensors_claimed(self):
        """Vessel that needs tasking but gets preempted=True and empty opportunities → PREEMPTED."""
        decision = plan_collection(
            _fresh_track(), score=1.0, confidence=0.4,
            breakdown=_NOMINAL_BREAKDOWN, current_time=T0,
            opportunities=[], preempted=True,
        )
        assert decision.action == "PREEMPTED"

    def test_no_sensor_when_not_preempted_and_empty(self):
        """Empty opportunities with preempted=False (default) → NO_SENSOR, not PREEMPTED."""
        decision = plan_collection(
            _fresh_track(), score=1.0, confidence=0.4,
            breakdown=_NOMINAL_BREAKDOWN, current_time=T0,
            opportunities=[],
        )
        assert decision.action == "NO_SENSOR"

    def test_preempted_does_not_override_hold(self):
        """Vessel with low task value → HOLD, even when preempted=True."""
        track = _collected_track(hours_ago=1.5, anomaly_at_collection=0.8)
        decision = plan_collection(
            track, score=0.8, confidence=0.8,
            breakdown=_NOMINAL_BREAKDOWN, current_time=T0,
            opportunities=[], preempted=True,
        )
        assert decision.action == "HOLD"

    def test_preempted_does_not_override_none(self):
        """Vessel not needing tasking → NONE, even when preempted=True."""
        decision = plan_collection(
            _fresh_track(), score=0.1, confidence=0.9,
            breakdown=_NOMINAL_BREAKDOWN, current_time=T0,
            opportunities=[], preempted=True,
        )
        assert decision.action == "NONE"


# ---------------------------------------------------------------------------
# compute_target_priority
# ---------------------------------------------------------------------------

class TestComputeTargetPriority:
    def test_returns_float(self):
        assert isinstance(compute_target_priority(0.5, 0.8), float)

    def test_minimum_inputs_give_lowest_priority(self):
        """score=0, confidence=1, no boost → all terms zero → priority=0."""
        assert compute_target_priority(0.0, 1.0, 0.0) == 0.0

    def test_critical_anomaly_full_confidence_gives_known_value(self):
        """score=CRITICAL_ANOMALY_THRESHOLD, confidence=1.0 → only anomaly term fires."""
        expected = 1.0 * 0.55 + 0.0 * 0.35 + 0.0 * 0.10
        result = compute_target_priority(config.CRITICAL_ANOMALY_THRESHOLD, 1.0)
        assert result == pytest.approx(expected, abs=1e-6)

    def test_zero_confidence_zero_anomaly_gives_known_value(self):
        """score=0, confidence=0 → only no-confidence term fires."""
        expected = 0.0 * 0.55 + 1.0 * 0.35 + 0.0 * 0.10
        result = compute_target_priority(0.0, 0.0)
        assert result == pytest.approx(expected, abs=1e-6)

    def test_critical_anomaly_outranks_high_anomaly(self):
        """A vessel at critical anomaly has higher priority than one at high anomaly."""
        critical = compute_target_priority(config.CRITICAL_ANOMALY_THRESHOLD, 0.8)
        high = compute_target_priority(config.HIGH_ANOMALY_THRESHOLD, 0.8)
        assert critical > high

    def test_lower_confidence_increases_priority(self):
        """Same anomaly score: lower confidence vessel ranks higher."""
        low_conf = compute_target_priority(0.6, 0.3)
        high_conf = compute_target_priority(0.6, 0.9)
        assert low_conf > high_conf

    def test_compound_boost_breaks_ties(self):
        """Two identical score/confidence vessels: compound boost determines order."""
        without_boost = compute_target_priority(0.6, 0.7, compound_boost=0.0)
        with_boost    = compute_target_priority(0.6, 0.7, compound_boost=0.5)
        assert with_boost > without_boost

    def test_compound_boost_capped_contribution(self):
        """Max compound_boost=1.0 adds exactly 0.10 to priority."""
        base  = compute_target_priority(0.6, 0.7, compound_boost=0.0)
        boosted = compute_target_priority(0.6, 0.7, compound_boost=1.0)
        assert boosted - base == pytest.approx(0.10, abs=1e-6)

    def test_ordering_reflects_operational_urgency(self):
        """Critical-anomaly low-custody vessel outranks a nominal vessel."""
        urgent  = compute_target_priority(config.CRITICAL_ANOMALY_THRESHOLD, 0.2)
        nominal = compute_target_priority(0.1, 0.9)
        assert urgent > nominal

    def test_can_exceed_one_for_supercritical_anomaly(self):
        """score above CRITICAL_ANOMALY_THRESHOLD produces priority > 0.55."""
        result = compute_target_priority(config.CRITICAL_ANOMALY_THRESHOLD * 1.5, 1.0)
        assert result > 0.55

    def test_priority_is_monotone_in_anomaly_score(self):
        """Increasing score strictly increases priority (all else equal)."""
        priorities = [compute_target_priority(s, 0.8) for s in [0.0, 0.5, 1.0, 1.5, 2.0]]
        assert priorities == sorted(priorities)

    def test_priority_is_monotone_in_low_confidence(self):
        """Decreasing confidence strictly increases priority (all else equal)."""
        priorities = [compute_target_priority(0.6, c) for c in [1.0, 0.8, 0.6, 0.4, 0.2]]
        assert priorities == sorted(priorities)


# ---------------------------------------------------------------------------
# score_opportunity — normalized anomaly and resolution preference
# ---------------------------------------------------------------------------

def _opp(
    sensor_type: str = "fast_revisit",
    success_prob: float = 0.8,
    resolution: str = "medium",
    cost: float = 1.0,
) -> "SensorOpportunity":
    """Minimal SensorOpportunity fixture for score_opportunity tests."""
    return SensorOpportunity(
        sensor_id="TEST",
        sensor_type=sensor_type,
        success_prob=success_prob,
        resolution=resolution,
        cost=cost,
        available_from=T0,
        available_to=T0,
    )


class TestScoreOpportunity:
    def test_higher_anomaly_score_increases_opportunity_score(self):
        """Normalized anomaly is monotone: higher score → higher opportunity score."""
        opp = _opp()
        scores = [
            score_opportunity(opp, s, 0.7, _NOMINAL_BREAKDOWN)[0]
            for s in [0.0, 0.5, 1.0, 1.5, 2.0]
        ]
        assert scores == sorted(scores), (
            "opportunity score should increase monotonically with anomaly score"
        )

    def test_high_resolution_scores_above_medium_when_all_else_equal(self):
        """'high' resolution adds exactly 0.3 compared to 'medium' when all else equal."""
        medium = _opp(resolution="medium")
        high   = _opp(resolution="high")
        s_medium, _ = score_opportunity(medium, 0.8, 0.7, _NOMINAL_BREAKDOWN)
        s_high,   _ = score_opportunity(high,   0.8, 0.7, _NOMINAL_BREAKDOWN)
        assert s_high - s_medium == pytest.approx(0.3, abs=1e-9)

    def test_resolution_does_not_override_success_prob_advantage(self):
        """A 'medium' sensor with high success_prob outscores a 'high' resolution
        sensor with low success_prob — resolution is a secondary tiebreaker only.

        success_prob difference of 0.4 contributes 0.6 (×1.5 weight), which
        exceeds the 0.3 resolution bonus, so the capable medium sensor wins.
        """
        capable_medium = _opp(resolution="medium", success_prob=1.0)
        weak_high      = _opp(resolution="high",   success_prob=0.6)
        s_capable, _ = score_opportunity(capable_medium, 0.8, 0.7, _NOMINAL_BREAKDOWN)
        s_weak,    _ = score_opportunity(weak_high,      0.8, 0.7, _NOMINAL_BREAKDOWN)
        assert s_capable > s_weak, (
            "higher success_prob advantage (×1.5) should outweigh resolution bonus (0.3)"
        )

    def test_resolution_does_not_override_breakdown_fit(self):
        """A 'medium' sensor that matches a sensitive-zone breakdown (bonus +1.5)
        outscores a 'high' resolution sensor with no breakdown match."""
        zone_breakdown = {
            "sensitive_zone":  type("R", (), {"score": 1.0})(),
            "loitering":       type("R", (), {"score": 0.0})(),
            "route_deviation": type("R", (), {"score": 0.0})(),
        }
        medium_zone_match = _opp(resolution="medium")
        high_no_match     = _opp(resolution="high")
        s_zone, _ = score_opportunity(medium_zone_match, 0.8, 0.7, zone_breakdown)
        s_high, _ = score_opportunity(high_no_match,     0.8, 0.7, _NOMINAL_BREAKDOWN)
        assert s_zone > s_high, (
            "breakdown-fit bonus (1.5) should dominate resolution preference (0.3)"
        )
