"""Tests for the agent-style decision roles layer (custody.roles)."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from custody.compounds import CompoundSignal
from custody.decision import (
    ELEVATE,
    ESCALATE,
    PASSIVE_MONITOR,
    TASK_OPTICAL,
    TASK_SAR,
    Decision,
)
from custody.fusion import FusionAssessment
from custody.roles import (
    RoleRecommendation,
    Synthesis,
    _first_clause,
    analyst_recommend,
    build_deliberation,
    collector_recommend,
    operator_recommend,
    synthesize,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TS = datetime(2024, 9, 15, 12, 0, tzinfo=timezone.utc)


def _fa(
    fused_score=0.5,
    uncertainty=0.4,
    source_agreement=0.6,
    missing_evidence=None,
    recommended_confirming_source=None,
):
    return FusionAssessment(
        entity_id="V1",
        timestamp=_TS,
        fused_score=fused_score,
        uncertainty=uncertainty,
        source_agreement=source_agreement,
        missing_evidence=missing_evidence or [],
        recommended_confirming_source=recommended_confirming_source,
    )


def _decision(action=ELEVATE, priority=0.5, confidence=0.6):
    return Decision(
        entity_id="V1",
        timestamp=_TS,
        action=action,
        priority=priority,
        confidence=confidence,
        why=["test"],
        next_best_actions=[],
    )


def _record(**overrides):
    base = {
        "target_id": "V1",
        "time": _TS,
        "anomaly_score": 0.8,
        "anomaly_state": "normal",
        "behavior_state": "transit",
        "custody_confidence": 0.7,
        "custody_health": "HEALTHY",
        "hours_since_collection": 4.0,
        "action": "NONE",
        "sensor_type": None,
        "zone_probability": 0.0,
        "time_to_zone_hours": None,
        "attention_state": "BACKGROUND",
        "neglect_flag": False,
        "neglect_hours": 0.0,
        "dark_vessel_flag": False,
        "escalation_boost": 0.0,
        "sensitive_zone": 0.0,
    }
    base.update(overrides)
    return base


def _compound(code="LOITERING_NEAR_ZONE", confidence=0.7):
    return CompoundSignal(
        code=code,
        confidence=confidence,
        evidence="test",
        components={},
        timestamp=_TS,
    )


# ---------------------------------------------------------------------------
# Analyst
# ---------------------------------------------------------------------------

class TestAnalyst:

    def test_low_concern_returns_passive_monitor(self):
        fa = _fa(fused_score=0.2, source_agreement=0.9)
        r = analyst_recommend(_record(), fa, _decision(), [])
        assert r.recommended_action == PASSIVE_MONITOR
        assert r.role == "analyst"

    def test_critical_sustained_returns_escalate(self):
        """Only truly extreme behavioral persistence triggers ESCALATE."""
        fa = _fa(fused_score=0.90, source_agreement=0.3)
        rec = _record(anomaly_state="sustained", anomaly_score=1.5)
        r = analyst_recommend(rec, fa, _decision(), [])
        assert r.recommended_action == ESCALATE

    def test_high_fused_normal_state_returns_elevate(self):
        """High fused score alone (no persistent state) → ELEVATE, not ESCALATE."""
        fa = _fa(fused_score=0.85, source_agreement=0.4)
        r = analyst_recommend(_record(), fa, _decision(), [])
        assert r.recommended_action == ELEVATE

    def test_confirmed_anomaly_loitering_returns_optical(self):
        """Behavioral activity (loitering) → analyst picks OPTICAL for visual confirmation."""
        fa = _fa(fused_score=0.6)
        rec = _record(anomaly_state="confirmed", behavior_state="loitering")
        r = analyst_recommend(rec, fa, _decision(), [])
        assert r.recommended_action == TASK_OPTICAL

    def test_confirmed_anomaly_transit_returns_sar(self):
        """Non-activity behavior (transit) → analyst picks SAR for persistent watch."""
        fa = _fa(fused_score=0.6)
        rec = _record(anomaly_state="confirmed", behavior_state="transit")
        r = analyst_recommend(rec, fa, _decision(), [])
        assert r.recommended_action == TASK_SAR

    def test_strong_compound_returns_tasking(self):
        """High-confidence compound triggers tasking even if anomaly_state is normal."""
        fa = _fa(fused_score=0.6)
        comp = _compound("LOITERING_NEAR_ZONE", 0.8)
        rec = _record(behavior_state="approach")
        r = analyst_recommend(rec, fa, _decision(), [comp])
        assert r.recommended_action == TASK_OPTICAL

    def test_moderate_fused_returns_elevate(self):
        fa = _fa(fused_score=0.55, source_agreement=0.6)
        r = analyst_recommend(_record(), fa, _decision(), [])
        assert r.recommended_action == ELEVATE

    def test_emerging_anomaly_returns_elevate(self):
        """Emerging state alone → ELEVATE (watch, don't task yet)."""
        fa = _fa(fused_score=0.45)
        rec = _record(anomaly_state="emerging")
        r = analyst_recommend(rec, fa, _decision(), [])
        assert r.recommended_action == ELEVATE

    def test_confidence_is_source_agreement(self):
        fa = _fa(source_agreement=0.72)
        r = analyst_recommend(_record(), fa, _decision(), [])
        assert r.confidence == 0.72

    def test_expected_value_is_fused_score(self):
        fa = _fa(fused_score=0.63)
        r = analyst_recommend(_record(), fa, _decision(), [])
        assert r.expected_value == 0.63

    def test_rationale_includes_compound(self):
        fa = _fa(fused_score=0.6)
        comp = _compound("HIGH_ANOMALY_LOW_CUSTODY", 0.8)
        r = analyst_recommend(_record(), fa, _decision(), [comp])
        assert "HIGH_ANOMALY_LOW_CUSTODY" in r.rationale


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------

class TestCollector:

    def test_no_gaps_healthy_returns_passive(self):
        fa = _fa(uncertainty=0.2, missing_evidence=[], recommended_confirming_source=None)
        rec = _record(custody_health="HEALTHY")
        r = collector_recommend(rec, fa, _decision(), [])
        assert r.recommended_action == PASSIVE_MONITOR

    def test_sar_confirmation_needed(self):
        fa = _fa(uncertainty=0.6, missing_evidence=["SAR confirmation"],
                 recommended_confirming_source="SAR")
        r = collector_recommend(_record(), fa, _decision(), [])
        assert r.recommended_action == TASK_SAR

    def test_optical_confirmation_needed(self):
        fa = _fa(uncertainty=0.5, missing_evidence=["optical confirmation"],
                 recommended_confirming_source="OPTICAL")
        r = collector_recommend(_record(), fa, _decision(), [])
        assert r.recommended_action == TASK_OPTICAL

    def test_ais_source_returns_elevate(self):
        fa = _fa(missing_evidence=["fresh AIS update"],
                 recommended_confirming_source="AIS")
        r = collector_recommend(_record(), fa, _decision(), [])
        assert r.recommended_action == ELEVATE

    def test_confidence_is_one_minus_uncertainty(self):
        fa = _fa(uncertainty=0.35)
        r = collector_recommend(_record(), fa, _decision(), [])
        assert r.confidence == 0.65

    def test_expected_value_is_one_minus_uncertainty(self):
        fa = _fa(uncertainty=0.4)
        r = collector_recommend(_record(), fa, _decision(), [])
        assert r.expected_value == 0.6

    def test_rationale_mentions_health(self):
        fa = _fa(missing_evidence=["SAR confirmation"],
                 recommended_confirming_source="SAR")
        rec = _record(custody_health="STALE")
        r = collector_recommend(rec, fa, _decision(), [])
        assert "STALE" in r.rationale


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------

class TestOperator:

    def test_low_priority_returns_passive(self):
        d = _decision(priority=0.2)
        r = operator_recommend(_record(), _fa(), d, [])
        assert r.recommended_action == PASSIVE_MONITOR

    def test_very_high_priority_returns_escalate(self):
        """Only priority >= 0.90 triggers ESCALATE."""
        d = _decision(priority=0.92)
        r = operator_recommend(_record(), _fa(), d, [])
        assert r.recommended_action == ESCALATE

    def test_high_priority_below_090_returns_task(self):
        """Priority 0.85 is high but NOT enough for ESCALATE — should task."""
        d = _decision(priority=0.85)
        r = operator_recommend(_record(), _fa(), d, [])
        assert r.recommended_action in (TASK_SAR, TASK_OPTICAL)

    def test_dark_vessel_returns_task_sar(self):
        """Dark vessel → SAR tasking (not ESCALATE)."""
        d = _decision(priority=0.5)
        rec = _record(dark_vessel_flag=True)
        r = operator_recommend(rec, _fa(), d, [])
        assert r.recommended_action == TASK_SAR

    def test_zone_imminent_returns_escalate(self):
        d = _decision(priority=0.5)
        rec = _record(zone_probability=0.8, time_to_zone_hours=2.0)
        r = operator_recommend(rec, _fa(), d, [])
        assert r.recommended_action == ESCALATE

    def test_zone_not_imminent_at_5h(self):
        """Zone ETA 5h with prob 0.8 is no longer imminent (threshold tightened to 4h)."""
        d = _decision(priority=0.5)
        rec = _record(zone_probability=0.8, time_to_zone_hours=5.0)
        r = operator_recommend(rec, _fa(), d, [])
        assert r.recommended_action != ESCALATE

    def test_moderate_priority_with_neglect_returns_task_sar(self):
        d = _decision(priority=0.55)
        rec = _record(neglect_flag=True, neglect_hours=6.0)
        r = operator_recommend(rec, _fa(), d, [])
        assert r.recommended_action == TASK_SAR

    def test_moderate_priority_optical_source_returns_task_optical(self):
        """Operator respects fusion source hint when tasking."""
        d = _decision(priority=0.55)
        fa = _fa(recommended_confirming_source="OPTICAL")
        r = operator_recommend(_record(), fa, d, [])
        assert r.recommended_action == TASK_OPTICAL

    def test_active_custody_returns_task_sar(self):
        d = _decision(priority=0.45)
        rec = _record(attention_state="ACTIVE_CUSTODY")
        r = operator_recommend(rec, _fa(), d, [])
        assert r.recommended_action == TASK_SAR

    def test_confidence_is_priority(self):
        d = _decision(priority=0.73)
        r = operator_recommend(_record(), _fa(), d, [])
        assert r.confidence == 0.73

    def test_expected_value_is_priority(self):
        d = _decision(priority=0.61)
        r = operator_recommend(_record(), _fa(), d, [])
        assert r.expected_value == 0.61


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

class TestSynthesis:

    def test_unanimous(self):
        a = RoleRecommendation("analyst", ESCALATE, 0.8, "r", {}, 0.8)
        c = RoleRecommendation("collector", ESCALATE, 0.7, "r", {}, 0.7)
        o = RoleRecommendation("operator", ESCALATE, 0.9, "r", {}, 0.9)
        s = synthesize(a, c, o)
        assert s.agreement == "unanimous"
        assert s.final_action == ESCALATE
        assert s.dissent is None
        assert "All three" in s.resolution_reason
        assert s.why_not == {}

    def test_majority(self):
        a = RoleRecommendation("analyst", TASK_OPTICAL, 0.8, "analyst rationale; extra", {}, 0.8)
        c = RoleRecommendation("collector", TASK_SAR, 0.7, "r", {}, 0.7)
        o = RoleRecommendation("operator", TASK_SAR, 0.9, "r", {}, 0.9)
        s = synthesize(a, c, o)
        assert s.agreement == "majority"
        assert s.final_action == TASK_SAR
        assert s.dissent == {"analyst": TASK_OPTICAL}
        assert "dissent" in s.resolution_reason
        assert TASK_OPTICAL in s.why_not
        assert len(s.why_not) == 1
        assert "Analyst" in s.why_not[TASK_OPTICAL]

    def test_split_operator_breaks_tie(self):
        a = RoleRecommendation("analyst", TASK_OPTICAL, 0.8, "analyst reason", {}, 0.8)
        c = RoleRecommendation("collector", TASK_SAR, 0.7, "collector reason", {}, 0.7)
        o = RoleRecommendation("operator", ESCALATE, 0.9, "operator reason", {}, 0.9)
        s = synthesize(a, c, o)
        assert s.agreement == "split"
        assert s.final_action == ESCALATE
        assert s.dissent == {"analyst": TASK_OPTICAL, "collector": TASK_SAR}
        assert "Operator breaks tie" in s.resolution_reason
        assert len(s.why_not) == 2
        assert TASK_OPTICAL in s.why_not
        assert TASK_SAR in s.why_not


# ---------------------------------------------------------------------------
# build_deliberation (integration)
# ---------------------------------------------------------------------------

class TestBuildDeliberation:

    def test_returns_well_formed_synthesis(self):
        rec = _record(anomaly_score=1.2, anomaly_state="confirmed")
        fa = _fa(fused_score=0.75, uncertainty=0.5,
                 missing_evidence=["SAR confirmation"],
                 recommended_confirming_source="SAR")
        d = _decision(action=TASK_SAR, priority=0.8)
        s = build_deliberation(rec, fa, d, [])
        assert isinstance(s, Synthesis)
        assert s.analyst.role == "analyst"
        assert s.collector.role == "collector"
        assert s.operator.role == "operator"
        assert s.agreement in ("unanimous", "majority", "split")
        assert s.final_action in (PASSIVE_MONITOR, ELEVATE, TASK_OPTICAL, TASK_SAR, ESCALATE)

    def test_with_compounds(self):
        rec = _record(anomaly_score=1.0)
        fa = _fa(fused_score=0.6, recommended_confirming_source="OPTICAL")
        d = _decision(priority=0.6)
        comps = [_compound()]
        s = build_deliberation(rec, fa, d, comps)
        assert s.analyst.key_factors["compound_count"] == 1

    def test_no_import_cycle(self):
        """Importing roles should not cause circular import errors."""
        import custody.roles  # noqa: F811
        assert hasattr(custody.roles, "build_deliberation")


# ---------------------------------------------------------------------------
# _first_clause helper
# ---------------------------------------------------------------------------

class TestFirstClause:

    def test_single_clause(self):
        assert _first_clause("anomaly 1.40.") == "anomaly 1.40"

    def test_multi_clause(self):
        assert _first_clause("SAR needed; track STALE; 8h since collection.") == "SAR needed"

    def test_empty(self):
        assert _first_clause("") == ""


# ---------------------------------------------------------------------------
# "Why not?" content
# ---------------------------------------------------------------------------

class TestWhyNot:

    def test_unanimous_empty(self):
        a = RoleRecommendation("analyst", ESCALATE, 0.8, "r", {}, 0.8)
        c = RoleRecommendation("collector", ESCALATE, 0.7, "r", {}, 0.7)
        o = RoleRecommendation("operator", ESCALATE, 0.9, "r", {}, 0.9)
        s = synthesize(a, c, o)
        assert s.why_not == {}

    def test_majority_mentions_dissenting_role(self):
        a = RoleRecommendation("analyst", TASK_OPTICAL, 0.8, "behavioral significance.", {}, 0.8)
        c = RoleRecommendation("collector", TASK_SAR, 0.7, "SAR needed.", {}, 0.7)
        o = RoleRecommendation("operator", TASK_SAR, 0.9, "urgent.", {}, 0.9)
        s = synthesize(a, c, o)
        reason = s.why_not[TASK_OPTICAL]
        assert "Analyst" in reason
        assert "TASK_OPTICAL" in reason
        assert "TASK_SAR" in reason

    def test_split_has_two_entries(self):
        a = RoleRecommendation("analyst", TASK_OPTICAL, 0.8, "sig high.", {}, 0.8)
        c = RoleRecommendation("collector", TASK_SAR, 0.7, "evidence gap.", {}, 0.7)
        o = RoleRecommendation("operator", ESCALATE, 0.9, "priority 0.91.", {}, 0.9)
        s = synthesize(a, c, o)
        assert len(s.why_not) == 2
        assert "Analyst" in s.why_not[TASK_OPTICAL]
        assert "Collector" in s.why_not[TASK_SAR]
        # Both mention operator's winning action
        assert "ESCALATE" in s.why_not[TASK_OPTICAL]
        assert "ESCALATE" in s.why_not[TASK_SAR]

    def test_why_not_values_are_nonempty_strings(self):
        a = RoleRecommendation("analyst", TASK_OPTICAL, 0.8, "r.", {}, 0.8)
        c = RoleRecommendation("collector", TASK_SAR, 0.7, "r.", {}, 0.7)
        o = RoleRecommendation("operator", ESCALATE, 0.9, "r.", {}, 0.9)
        s = synthesize(a, c, o)
        for action, reason in s.why_not.items():
            assert isinstance(reason, str)
            assert len(reason) > 10

    def test_majority_why_not_uses_first_clause(self):
        a = RoleRecommendation("analyst", TASK_OPTICAL, 0.8,
                               "sustained anomaly; LOITER compound; approach state.", {}, 0.8)
        c = RoleRecommendation("collector", TASK_SAR, 0.7, "r.", {}, 0.7)
        o = RoleRecommendation("operator", TASK_SAR, 0.9, "r.", {}, 0.9)
        s = synthesize(a, c, o)
        reason = s.why_not[TASK_OPTICAL]
        assert "sustained anomaly" in reason
        # Should NOT contain the second clause
        assert "LOITER compound" not in reason


# ---------------------------------------------------------------------------
# Scenario diversity — end-to-end deliberation outcomes
# ---------------------------------------------------------------------------

class TestScenarioDiversity:
    """Verify the three target scenario types actually occur."""

    def test_unanimous_calm(self):
        """Low-threat entity → all three roles agree on PASSIVE_MONITOR."""
        rec = _record(anomaly_score=0.1, anomaly_state="normal",
                      behavior_state="transit", custody_health="HEALTHY",
                      hours_since_collection=1.0)
        fa = _fa(fused_score=0.15, uncertainty=0.1, source_agreement=0.95,
                 missing_evidence=[])
        d = _decision(action=PASSIVE_MONITOR, priority=0.1)
        s = build_deliberation(rec, fa, d, [])
        assert s.agreement == "unanimous"
        assert s.final_action == PASSIVE_MONITOR

    def test_evidence_gap_majority(self):
        """Collector and Operator agree on tasking; Analyst sees only moderate
        behavioral interest → ELEVATE.  Result is majority, not unanimous."""
        rec = _record(anomaly_score=0.5, anomaly_state="normal",
                      behavior_state="transit", custody_health="STALE",
                      hours_since_collection=12.0)
        fa = _fa(fused_score=0.55, uncertainty=0.6,
                 missing_evidence=["SAR confirmation"],
                 recommended_confirming_source="SAR")
        d = _decision(action=TASK_SAR, priority=0.55)
        s = build_deliberation(rec, fa, d, [])
        # Analyst → ELEVATE (moderate fused, normal state, no compounds)
        # Collector → TASK_SAR (SAR confirmation needed)
        # Operator → TASK_SAR (priority 0.55)
        assert s.analyst.recommended_action == ELEVATE
        assert s.collector.recommended_action == TASK_SAR
        assert s.operator.recommended_action == TASK_SAR
        assert s.agreement == "majority"
        assert s.final_action == TASK_SAR

    def test_true_split(self):
        """Three-way split: analyst sees behavioral pattern (TASK_OPTICAL),
        collector wants SAR confirmation, operator wants to just elevate."""
        rec = _record(anomaly_score=0.6, anomaly_state="confirmed",
                      behavior_state="loitering", custody_health="DEGRADED",
                      hours_since_collection=8.0)
        fa = _fa(fused_score=0.60, uncertainty=0.5,
                 missing_evidence=["SAR confirmation"],
                 recommended_confirming_source="SAR")
        d = _decision(action=ELEVATE, priority=0.40)
        s = build_deliberation(rec, fa, d, [])
        # Analyst → TASK_OPTICAL (confirmed + loitering)
        # Collector → TASK_SAR (SAR confirmation needed)
        # Operator → ELEVATE (priority 0.40, no neglect/dark)
        assert s.analyst.recommended_action == TASK_OPTICAL
        assert s.collector.recommended_action == TASK_SAR
        assert s.operator.recommended_action == ELEVATE
        assert s.agreement == "split"

    def test_dark_vessel_divergence(self):
        """Dark vessel: operator tasks SAR (not ESCALATE), analyst elevates,
        collector tasks SAR — majority on TASK_SAR."""
        rec = _record(anomaly_score=0.5, anomaly_state="normal",
                      behavior_state="transit", dark_vessel_flag=True,
                      hours_since_collection=6.0)
        fa = _fa(fused_score=0.50, uncertainty=0.5,
                 missing_evidence=["SAR confirmation"],
                 recommended_confirming_source="SAR")
        d = _decision(action=TASK_SAR, priority=0.55)
        s = build_deliberation(rec, fa, d, [])
        assert s.operator.recommended_action == TASK_SAR
        assert s.operator.recommended_action != ESCALATE
