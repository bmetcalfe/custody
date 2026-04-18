"""
Tests for custody.decision — the mission reasoning layer.

Coverage:
  1.  Decision is frozen (immutable)
  2.  Decision has timestamp field
  3.  Decision has all required fields
  4.  Low fused_score + low uncertainty maps to PASSIVE_MONITOR
  5.  High fused_score + high uncertainty maps to ESCALATE
  6.  recommended_confirming_source SAR maps to TASK_SAR
  7.  recommended_confirming_source OPTICAL maps to TASK_OPTICAL
  8.  Mid-range without clear source maps to ELEVATE
  9.  priority increases with fused_score
 10.  priority bumped for sensitive zone presence
 11.  priority bumped for active compound signals
 12.  priority bounded to [0, 1]
 13.  confidence decreases as uncertainty rises
 14.  confidence high when action clearly matches evidence (fit=1.0)
 15.  confidence bounded to [0, 1]
 16.  why[] is non-empty for every action
 17.  why[] for PASSIVE_MONITOR mentions low significance
 18.  why[] for ESCALATE mentions high significance and uncertainty
 19.  why[] for TASK_SAR mentions SAR tasking
 20.  why[] for TASK_OPTICAL mentions optical tasking
 21.  why[] for ELEVATE mentions elevated priority
 22.  why[] length is between 1 and 5
 23.  why[] mentions compound signals when compounds are active
 24.  why[] mentions zone when entity is in sensitive zone
 25.  next_best_actions excludes the primary action
 26.  next_best_actions for PASSIVE_MONITOR is ["ELEVATE"]
 27.  next_best_actions for TASK_SAR includes TASK_OPTICAL
 28.  next_best_actions for ESCALATE includes TASK_SAR and ELEVATE
 29.  output is deterministic for same inputs
 30.  decision_for_timeline returns one Decision per record
 31.  decision_for_timeline raises on mismatched lengths
 32.  decision_for_timeline preserves order
 33.  Smoke test: end-to-end with run_simulation()
 34.  prediction: why[] includes zone approach bullet when zone_probability > 0.5
 35.  prediction: why[] omits zone approach bullet when zone_probability <= 0.5
 36.  prediction: why[] omits zone approach bullet when time_to_zone_hours is None
 37.  prediction: priority bumped when zone_probability > 0.5 and tte valid
 38.  prediction: priority NOT bumped when zone_probability <= 0.5
 39.  prediction: priority NOT bumped when time_to_zone_hours is None
 40.  prediction: priority bump bounded to [0, 1]
"""
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from collections import defaultdict

import pytest

import custody.config as config
from custody.compounds import CompoundSignal
from custody.decision import (
    Decision,
    PASSIVE_MONITOR, ELEVATE, TASK_OPTICAL, TASK_SAR, ESCALATE,
    _select_action, _compute_priority, _compute_decision_confidence,
    _build_why, _build_next_best_actions,
    build_decision, decision_for_timeline,
)
from custody.belief_assessment import FusionAssessment, build_fusion_assessment
from custody.models import TrackState


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

_T0 = datetime(2026, 3, 23, 12, 0, tzinfo=timezone.utc)


def _fa(
    fused_score=0.4,
    uncertainty=0.3,
    source_agreement=0.8,
    missing_evidence=None,
    recommended_confirming_source=None,
    entity_id="V001",
) -> FusionAssessment:
    return FusionAssessment(
        entity_id=entity_id,
        timestamp=_T0,
        fused_score=fused_score,
        uncertainty=uncertainty,
        source_agreement=source_agreement,
        missing_evidence=missing_evidence or [],
        recommended_confirming_source=recommended_confirming_source,
    )


def _record(**kwargs) -> dict:
    base = {
        "entity_id": "V001",
        "time": _T0,
        "custody_confidence": 0.8,
        "anomaly_score": 0.2,
        "action": "NONE",
        "sensitive_zone": 0.0,
        "loitering": 0.0,
        "route_deviation": 0.0,
        "vessel_proximity_score": 0.0,
    }
    base.update(kwargs)
    return base


def _track(**kwargs) -> TrackState:
    defaults = dict(uncertainty_km=5.0, last_collection_time=None,
                    last_collection_anomaly_score=0.0)
    defaults.update(kwargs)
    return TrackState(**defaults)


def _compound(code="LOITERING_NEAR_ZONE", confidence=0.7) -> CompoundSignal:
    return CompoundSignal(
        code=code, confidence=confidence,
        evidence="test", components={}, timestamp=_T0,
    )


def _build(fa=None, record=None, track=None, compounds=None, planner_context=None):
    return build_decision(
        fa or _fa(),
        record or _record(),
        track or _track(),
        compounds if compounds is not None else [],
        planner_context,
    )


# ---------------------------------------------------------------------------
# 1–3. Dataclass properties
# ---------------------------------------------------------------------------

class TestDecisionSchema:
    def test_decision_is_frozen(self):
        d = _build()
        with pytest.raises((FrozenInstanceError, TypeError)):
            d.action = "INVALID"  # type: ignore[misc]

    def test_has_timestamp(self):
        assert _build().timestamp == _T0

    def test_has_all_required_fields(self):
        d = _build()
        assert hasattr(d, "entity_id")
        assert hasattr(d, "timestamp")
        assert hasattr(d, "action")
        assert hasattr(d, "priority")
        assert hasattr(d, "confidence")
        assert hasattr(d, "why")
        assert hasattr(d, "next_best_actions")


# ---------------------------------------------------------------------------
# 4–8. Action selection
# ---------------------------------------------------------------------------

class TestActionSelection:
    def test_low_score_low_uncertainty_gives_passive_monitor(self):
        fa = _fa(fused_score=0.20, uncertainty=0.30)
        assert _select_action(fa, _record(), _track(), []) == PASSIVE_MONITOR

    def test_high_score_high_uncertainty_gives_escalate(self):
        fa = _fa(fused_score=0.85, uncertainty=0.70)
        assert _select_action(fa, _record(), _track(), []) == ESCALATE

    def test_sar_source_gives_task_sar(self):
        fa = _fa(fused_score=0.55, uncertainty=0.40,
                 recommended_confirming_source="SAR")
        assert _select_action(fa, _record(), _track(), []) == TASK_SAR

    def test_optical_source_gives_task_optical(self):
        fa = _fa(fused_score=0.55, uncertainty=0.40,
                 recommended_confirming_source="OPTICAL")
        assert _select_action(fa, _record(), _track(), []) == TASK_OPTICAL

    def test_mid_range_no_source_gives_elevate(self):
        fa = _fa(fused_score=0.50, uncertainty=0.40,
                 recommended_confirming_source=None)
        assert _select_action(fa, _record(), _track(), []) == ELEVATE

    def test_escalate_requires_both_high_score_and_high_uncertainty(self):
        """High score alone (low uncertainty) should NOT trigger ESCALATE."""
        fa = _fa(fused_score=0.85, uncertainty=0.40)
        action = _select_action(fa, _record(), _track(), [])
        assert action != ESCALATE

    def test_passive_requires_both_low_score_and_low_uncertainty(self):
        """Low score with high uncertainty should NOT trigger PASSIVE_MONITOR."""
        fa = _fa(fused_score=0.20, uncertainty=0.70)
        action = _select_action(fa, _record(), _track(), [])
        assert action != PASSIVE_MONITOR


# ---------------------------------------------------------------------------
# 9–12. Priority
# ---------------------------------------------------------------------------

class TestPriority:
    def test_priority_increases_with_fused_score(self):
        priorities = [
            _compute_priority(_fa(fused_score=fs), _record(), _track(), [])
            for fs in [0.1, 0.3, 0.5, 0.7, 0.9]
        ]
        assert priorities == sorted(priorities)

    def test_zone_presence_bumps_priority(self):
        no_zone = _compute_priority(_fa(), _record(sensitive_zone=0.0), _track(), [])
        in_zone = _compute_priority(_fa(), _record(sensitive_zone=1.0), _track(), [])
        assert in_zone > no_zone

    def test_compound_signals_bump_priority(self):
        no_compounds = _compute_priority(_fa(), _record(), _track(), [])
        with_compounds = _compute_priority(
            _fa(), _record(), _track(),
            [_compound(), _compound(code="PROXIMITY_NEAR_ZONE")],
        )
        assert with_compounds > no_compounds

    def test_priority_bounded_to_0_1(self):
        p = _compute_priority(
            _fa(fused_score=1.0),
            _record(sensitive_zone=1.5, anomaly_score=5.0),
            _track(),
            [_compound()] * 10,
        )
        assert 0.0 <= p <= 1.0

    def test_priority_low_when_fused_score_low(self):
        p = _compute_priority(_fa(fused_score=0.05), _record(), _track(), [])
        assert p < 0.4


# ---------------------------------------------------------------------------
# 13–15. Confidence
# ---------------------------------------------------------------------------

class TestConfidence:
    def test_confidence_decreases_as_uncertainty_rises(self):
        confs = [
            _compute_decision_confidence(_fa(uncertainty=u), ELEVATE, _record(), _track(), [])
            for u in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        ]
        assert confs == sorted(confs, reverse=True)

    def test_high_fit_gives_higher_confidence(self):
        """TASK_SAR with rec=SAR (fit=1.0) > ELEVATE (fit=0.65) at same uncertainty."""
        fa_sar = _fa(fused_score=0.6, uncertainty=0.4,
                     recommended_confirming_source="SAR")
        conf_sar = _compute_decision_confidence(fa_sar, TASK_SAR, _record(), _track(), [])
        conf_elv = _compute_decision_confidence(fa_sar, ELEVATE, _record(), _track(), [])
        assert conf_sar > conf_elv

    def test_confidence_bounded_to_0_1(self):
        c = _compute_decision_confidence(
            _fa(fused_score=1.0, uncertainty=0.0), PASSIVE_MONITOR, _record(), _track(), []
        )
        assert 0.0 <= c <= 1.0
        c2 = _compute_decision_confidence(
            _fa(fused_score=0.0, uncertainty=1.0), TASK_SAR, _record(), _track(), []
        )
        assert 0.0 <= c2 <= 1.0


# ---------------------------------------------------------------------------
# 16–24. why[]
# ---------------------------------------------------------------------------

class TestWhy:
    def test_why_non_empty_for_all_actions(self):
        for action in [PASSIVE_MONITOR, ELEVATE, TASK_OPTICAL, TASK_SAR, ESCALATE]:
            fa = _fa(fused_score=0.5, uncertainty=0.4)
            bullets = _build_why(fa, action, _record(), _track(), [])
            assert len(bullets) > 0, f"Empty why for {action}"

    def test_why_length_1_to_5(self):
        bullets = _build_why(_fa(), ELEVATE, _record(), _track(), [])
        assert 1 <= len(bullets) <= 5

    def test_passive_monitor_mentions_low_significance(self):
        bullets = _build_why(
            _fa(fused_score=0.15, uncertainty=0.20),
            PASSIVE_MONITOR, _record(), _track(), [],
        )
        text = " ".join(bullets).lower()
        assert "low" in text or "not" in text

    def test_escalate_mentions_human_review(self):
        bullets = _build_why(
            _fa(fused_score=0.85, uncertainty=0.70),
            ESCALATE, _record(), _track(), [],
        )
        assert any("human" in b.lower() or "review" in b.lower() for b in bullets)

    def test_task_sar_mentions_sar(self):
        bullets = _build_why(
            _fa(fused_score=0.6, uncertainty=0.5, recommended_confirming_source="SAR"),
            TASK_SAR, _record(), _track(), [],
        )
        assert any("SAR" in b for b in bullets)

    def test_task_optical_mentions_optical(self):
        bullets = _build_why(
            _fa(fused_score=0.6, uncertainty=0.4, recommended_confirming_source="OPTICAL"),
            TASK_OPTICAL, _record(), _track(), [],
        )
        assert any("ptical" in b for b in bullets)

    def test_elevate_mentions_elevated_priority(self):
        bullets = _build_why(_fa(), ELEVATE, _record(), _track(), [])
        assert any("elevat" in b.lower() or "priority" in b.lower() for b in bullets)

    def test_compounds_mentioned_when_active(self):
        cs = _compound(code="LOITERING_NEAR_ZONE")
        bullets = _build_why(_fa(), ELEVATE, _record(), _track(), [cs])
        assert any("compound" in b.lower() or "Loitering" in b for b in bullets)

    def test_zone_mentioned_when_entity_in_zone(self):
        bullets = _build_why(_fa(), ELEVATE, _record(sensitive_zone=1.0), _track(), [])
        assert any("zone" in b.lower() or "sensitive" in b.lower() for b in bullets)


# ---------------------------------------------------------------------------
# 25–28. next_best_actions
# ---------------------------------------------------------------------------

class TestNextBestActions:
    def test_primary_excluded_from_next_best(self):
        for action in [PASSIVE_MONITOR, ELEVATE, TASK_OPTICAL, TASK_SAR, ESCALATE]:
            nba = _build_next_best_actions(action, _fa(), _record(), _track())
            assert action not in nba, f"{action} appears in its own next_best"

    def test_passive_monitor_fallback_is_elevate(self):
        nba = _build_next_best_actions(PASSIVE_MONITOR, _fa(), _record(), _track())
        assert nba == [ELEVATE]

    def test_task_sar_includes_task_optical(self):
        nba = _build_next_best_actions(TASK_SAR, _fa(), _record(), _track())
        assert TASK_OPTICAL in nba

    def test_escalate_includes_task_sar_and_elevate(self):
        nba = _build_next_best_actions(ESCALATE, _fa(), _record(), _track())
        assert TASK_SAR in nba
        assert ELEVATE in nba

    def test_next_best_is_a_list(self):
        assert isinstance(_build_next_best_actions(ELEVATE, _fa(), _record(), _track()), list)


# ---------------------------------------------------------------------------
# 29. Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_inputs_give_same_output(self):
        fa = _fa(fused_score=0.65, uncertainty=0.45, recommended_confirming_source="SAR")
        r = _record(anomaly_score=0.7, sensitive_zone=1.0)
        t = _track()
        cs = [_compound()]
        d1 = build_decision(fa, r, t, cs)
        d2 = build_decision(fa, r, t, cs)
        assert d1 == d2


# ---------------------------------------------------------------------------
# 30–32. decision_for_timeline
# ---------------------------------------------------------------------------

class TestDecisionForTimeline:
    def _simple_timeline(self, n=3):
        from custody.models import TrackState
        records = [_record(time=datetime(2026, 1, 1, h, tzinfo=timezone.utc))
                   for h in range(n)]
        track = _track()
        fas = [_fa() for _ in range(n)]
        return records, track, fas

    def test_returns_one_per_record(self):
        records, track, fas = self._simple_timeline(4)
        decisions = decision_for_timeline(records, track, fas)
        assert len(decisions) == 4

    def test_raises_on_mismatched_lengths(self):
        records, track, fas = self._simple_timeline(3)
        with pytest.raises(ValueError):
            decision_for_timeline(records, track, fas[:2])

    def test_preserves_order(self):
        n = 3
        times = [datetime(2026, 1, 1, h, tzinfo=timezone.utc) for h in range(n)]
        records = [_record(time=t) for t in times]
        track = _track()
        # Build FAs from the same records so timestamps align
        fas = [build_fusion_assessment(r, [], track) for r in records]
        decisions = decision_for_timeline(records, track, fas)
        assert [d.timestamp for d in decisions] == times

    def test_empty_timeline_returns_empty(self):
        assert decision_for_timeline([], _track(), []) == []


# ---------------------------------------------------------------------------
# 34–40. Prediction layer wiring
# ---------------------------------------------------------------------------

class TestPredictionWhy:
    """34–36: prediction bullet in why[] based on zone_probability and tte."""

    def test_why_includes_prediction_bullet_when_approaching(self):
        r = _record(zone_probability=0.7, time_to_zone_hours=3.5)
        fa = _fa(fused_score=0.55, uncertainty=0.4, recommended_confirming_source="SAR")
        why = _build_why(fa, TASK_SAR, r, _track(), [])
        combined = " ".join(why).lower()
        assert "zone" in combined and ("3.5" in combined or "project" in combined)

    def test_why_omits_prediction_bullet_when_prob_low(self):
        r = _record(zone_probability=0.3, time_to_zone_hours=2.0)
        fa = _fa(fused_score=0.55, uncertainty=0.4)
        why = _build_why(fa, ELEVATE, r, _track(), [])
        combined = " ".join(why).lower()
        assert "project" not in combined or "2.0h" not in combined

    def test_why_omits_prediction_bullet_when_tte_none(self):
        r = _record(zone_probability=0.8, time_to_zone_hours=None)
        fa = _fa(fused_score=0.55, uncertainty=0.4, recommended_confirming_source="SAR")
        why = _build_why(fa, TASK_SAR, r, _track(), [])
        # No bullet mentioning h projection without a tte value
        combined = " ".join(why).lower()
        assert "0.8" not in combined or "trajectory" not in combined


class TestPredictionPriority:
    """37–40: prediction_bump in _compute_priority."""

    def test_priority_bumped_when_approaching(self):
        base_record = _record(zone_probability=0.0, time_to_zone_hours=None)
        approach_record = _record(zone_probability=0.7, time_to_zone_hours=4.0)
        fa = _fa(fused_score=0.5, uncertainty=0.3)
        p_base = _compute_priority(fa, base_record, _track(), [])
        p_approach = _compute_priority(fa, approach_record, _track(), [])
        assert p_approach > p_base

    def test_priority_not_bumped_when_prob_low(self):
        base_record = _record(zone_probability=0.0, time_to_zone_hours=None)
        low_prob_record = _record(zone_probability=0.3, time_to_zone_hours=2.0)
        fa = _fa(fused_score=0.5, uncertainty=0.3)
        p_base = _compute_priority(fa, base_record, _track(), [])
        p_low = _compute_priority(fa, low_prob_record, _track(), [])
        assert p_base == p_low

    def test_priority_not_bumped_when_tte_none(self):
        base_record = _record(zone_probability=0.0, time_to_zone_hours=None)
        no_tte_record = _record(zone_probability=0.8, time_to_zone_hours=None)
        fa = _fa(fused_score=0.5, uncertainty=0.3)
        p_base = _compute_priority(fa, base_record, _track(), [])
        p_no_tte = _compute_priority(fa, no_tte_record, _track(), [])
        assert p_base == p_no_tte

    def test_priority_bounded_at_one_even_with_prediction_bump(self):
        r = _record(
            sensitive_zone=1.0, anomaly_score=0.95,
            zone_probability=0.9, time_to_zone_hours=1.0,
        )
        fa = _fa(fused_score=1.0, uncertainty=0.9)
        p = _compute_priority(fa, r, _track(last_collection_anomaly_score=0.0), [_compound()])
        assert p <= 1.0


# ---------------------------------------------------------------------------
# 33. Smoke test
# ---------------------------------------------------------------------------

def test_end_to_end_with_simulation():
    """run_simulation → FusionAssessments → Decisions: all valid."""
    from custody.simulate import run_simulation
    from custody.belief_assessment import fusion_for_timeline

    records = run_simulation()

    by_vessel: dict = defaultdict(list)
    for r in records:
        vid = r.get("entity_id") or r.get("vessel_id") or r.get("target_id")
        by_vessel[vid].append(r)

    all_actions = {PASSIVE_MONITOR, ELEVATE, TASK_OPTICAL, TASK_SAR, ESCALATE}

    for vessel_id, timeline in by_vessel.items():
        track = TrackState()
        fas = fusion_for_timeline(timeline, track)
        decisions = decision_for_timeline(timeline, track, fas)

        assert len(decisions) == len(timeline)
        for d in decisions:
            assert isinstance(d, Decision)
            assert d.entity_id == vessel_id
            assert d.action in all_actions
            assert 0.0 <= d.priority <= 1.0
            assert 0.0 <= d.confidence <= 1.0
            assert len(d.why) >= 1
            assert d.action not in d.next_best_actions
