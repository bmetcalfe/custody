"""
Tests for ML anomaly score integration into the fusion layer.

Covers:
  1. fused_score identical when FUSION_W_ML = 0.0 (default)
  2. fused_score changes when FUSION_W_ML > 0 and ml_anomaly_score > 0
  3. missing ml_anomaly_score field handled as 0.0
  4. source_agreement unchanged when ML weight is 0
  5. source_agreement incorporates ML alignment when weight > 0
  6. decision trace includes ml_anomaly_score field
  7. decision trace ml_anomaly_score = 0 for simulated records
  8. config toggle: setting weight to 0 restores original behavior
  9. fused_score stays in [0, 1] with ML enabled
 10. record schema includes ml_anomaly_score in simulation output
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

import custody.config as config
from custody.compounds import CompoundSignal
from custody.fusion import _compute_fused_score, _compute_source_agreement, build_fusion_assessment
from custody.models import TrackState

UTC = timezone.utc
_T0 = datetime(2026, 3, 23, 10, 0, tzinfo=UTC)


def _record(**overrides):
    base = {
        "target_id": "TEST",
        "time": _T0,
        "anomaly_score": 0.8,
        "custody_confidence": 0.6,
        "sensitive_zone": 0.5,
        "loitering": 0.3,
        "action": "NONE",
        "vessel_proximity_score": 0.0,
    }
    base.update(overrides)
    return base


def _track():
    return TrackState()


# ---------------------------------------------------------------------------
# Preserve original behavior when ML weight = 0
# ---------------------------------------------------------------------------

class TestMLWeightZero:
    """When FUSION_W_ML = 0.0, behavior must be identical to pre-ML baseline."""

    def test_fused_score_ignores_ml_when_weight_zero(self):
        # With default config (W_ML = 0.0), ML score should have no effect
        assert config.FUSION_W_ML == 0.0
        score_no_ml = _compute_fused_score(0.8, 0.3, 0.6, ml_anomaly_score=0.0)
        score_with_ml = _compute_fused_score(0.8, 0.3, 0.6, ml_anomaly_score=0.9)
        assert score_no_ml == score_with_ml

    def test_fused_score_matches_original_formula(self):
        """With default weights, should match: 0.45*anom + 0.35*compound + 0.20*(1-custody)"""
        assert config.FUSION_W_ML == 0.0
        result = _compute_fused_score(0.8, 0.4, 0.7)
        expected = round(min(max(0.45 * 0.8 + 0.35 * 0.4 + 0.20 * 0.3, 0), 1), 3)
        assert result == pytest.approx(expected)

    def test_source_agreement_unchanged_when_ml_zero(self):
        record = _record()
        compounds = []
        track = _track()
        assert config.FUSION_W_ML == 0.0
        # Should use original formula: 0.60 * compound_alignment + 0.40 * custody
        agr1 = _compute_source_agreement(record, compounds, track)
        record_with_ml = _record(ml_anomaly_score=0.9)
        agr2 = _compute_source_agreement(record_with_ml, compounds, track)
        assert agr1 == agr2

    def test_full_assessment_unchanged_when_ml_zero(self):
        record = _record(ml_anomaly_score=0.95)
        fa = build_fusion_assessment(record, [], _track())
        record_no_ml = _record()
        fa_no_ml = build_fusion_assessment(record_no_ml, [], _track())
        # With W_ML=0.0, both should produce identical fused_score
        assert fa.fused_score == fa_no_ml.fused_score


# ---------------------------------------------------------------------------
# ML weight > 0 changes behavior
# ---------------------------------------------------------------------------

class TestMLWeightPositive:

    def test_fused_score_changes_with_ml(self):
        # Temporarily set ML weight > 0
        old_w_h = config.FUSION_W_HEURISTIC
        old_w_ml = config.FUSION_W_ML
        try:
            config.FUSION_W_HEURISTIC = 0.30
            config.FUSION_W_ML = 0.15
            score_low_ml = _compute_fused_score(0.5, 0.3, 0.7, ml_anomaly_score=0.1)
            score_high_ml = _compute_fused_score(0.5, 0.3, 0.7, ml_anomaly_score=0.9)
            assert score_high_ml > score_low_ml
        finally:
            config.FUSION_W_HEURISTIC = old_w_h
            config.FUSION_W_ML = old_w_ml

    def test_fused_score_in_unit_interval_with_ml(self):
        old_w_h = config.FUSION_W_HEURISTIC
        old_w_ml = config.FUSION_W_ML
        try:
            config.FUSION_W_HEURISTIC = 0.30
            config.FUSION_W_ML = 0.15
            for ml in [0.0, 0.5, 1.0]:
                for anom in [0.0, 0.5, 1.0]:
                    for comp in [0.0, 0.5, 1.0]:
                        for cust in [0.0, 0.5, 1.0]:
                            s = _compute_fused_score(anom, comp, cust, ml)
                            assert 0.0 <= s <= 1.0
        finally:
            config.FUSION_W_HEURISTIC = old_w_h
            config.FUSION_W_ML = old_w_ml

    def test_source_agreement_includes_ml_when_weight_positive(self):
        old_w_ml = config.FUSION_W_ML
        try:
            config.FUSION_W_ML = 0.15
            # ML agrees with heuristic → higher agreement
            record_agree = _record(anomaly_score=1.0, ml_anomaly_score=0.67)
            agr_agree = _compute_source_agreement(record_agree, [], _track())

            # ML disagrees with heuristic → lower agreement
            record_disagree = _record(anomaly_score=1.0, ml_anomaly_score=0.1)
            agr_disagree = _compute_source_agreement(record_disagree, [], _track())

            assert agr_agree > agr_disagree
        finally:
            config.FUSION_W_ML = old_w_ml


# ---------------------------------------------------------------------------
# Missing ML field
# ---------------------------------------------------------------------------

class TestMissingML:

    def test_missing_ml_field_treated_as_zero(self):
        record = {"target_id": "X", "time": _T0, "anomaly_score": 0.5,
                  "custody_confidence": 0.7}
        fa = build_fusion_assessment(record, [], _track())
        assert fa.fused_score > 0  # works without crash


# ---------------------------------------------------------------------------
# Decision trace
# ---------------------------------------------------------------------------

class TestDecisionTraceML:

    def test_trace_includes_ml_field(self):
        from custody.decision_trace import build_decision_trace
        track = TrackState(last_collection_time=_T0)
        trace = build_decision_trace(
            timestamp=_T0, vessel_id="TEST",
            score=0.5, confidence=0.7, compound_boost=0.0,
            track=track,
            accessible_opportunities=[], claimed_sensor_ids=set(),
            remaining_opportunities=[],
            decision_action="NONE", decision_sensor_id=None,
            ml_anomaly_score=0.42,
        )
        assert trace.inputs.ml_anomaly_score == pytest.approx(0.42)

    def test_trace_ml_defaults_to_zero(self):
        from custody.decision_trace import build_decision_trace
        track = TrackState(last_collection_time=_T0)
        trace = build_decision_trace(
            timestamp=_T0, vessel_id="TEST",
            score=0.5, confidence=0.7, compound_boost=0.0,
            track=track,
            accessible_opportunities=[], claimed_sensor_ids=set(),
            remaining_opportunities=[],
            decision_action="NONE", decision_sensor_id=None,
        )
        assert trace.inputs.ml_anomaly_score == 0.0

    def test_traces_to_rows_includes_ml(self):
        from custody.decision_trace import build_decision_trace, traces_to_rows
        track = TrackState(last_collection_time=_T0)
        trace = build_decision_trace(
            timestamp=_T0, vessel_id="TEST",
            score=0.5, confidence=0.7, compound_boost=0.0,
            track=track,
            accessible_opportunities=[], claimed_sensor_ids=set(),
            remaining_opportunities=[],
            decision_action="NONE", decision_sensor_id=None,
            ml_anomaly_score=0.55,
        )
        rows = traces_to_rows([trace])
        assert "ML Anomaly Score" in rows[0]
        assert rows[0]["ML Anomaly Score"] == pytest.approx(0.55, abs=0.001)


# ---------------------------------------------------------------------------
# Simulation record schema
# ---------------------------------------------------------------------------

class TestSimulationRecordSchema:

    def test_ml_anomaly_score_field_present(self):
        from custody.simulate import run_simulation
        records = run_simulation()
        for r in records[:5]:
            assert "ml_anomaly_score" in r
            assert r["ml_anomaly_score"] == 0.0  # simulated = no ML
