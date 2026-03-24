"""
Tests for custody.fusion — the fusion and belief layer.

Coverage:
  1.  FusionAssessment is frozen (immutable)
  2.  FusionAssessment has timestamp field
  3.  fused_score is in [0, 1]
  4.  uncertainty is in [0, 1]
  5.  source_agreement is in [0, 1]
  6.  fused_score rises when anomaly_score rises
  7.  fused_score rises when top_compound_confidence rises
  8.  fused_score rises when custody confidence falls
  9.  uncertainty rises as custody confidence falls
 10.  uncertainty rises when source_agreement falls
 11.  stale/weak custody adds "fresh AIS update" to missing_evidence
 12.  high anomaly near zone without TASK adds "optical confirmation"
 13.  high anomaly + low confidence adds "SAR confirmation"
 14.  zone-related compound codes add "RF cross-cue"
 15.  NO_SENSOR action adds "repeat observation in next access window"
 16.  recommended_confirming_source is "OPTICAL" when visual confirmation needed
 17.  recommended_confirming_source is "SAR" when uncertainty is high
 18.  recommended_confirming_source is "AIS" when track is stale, no anomaly
 19.  recommended_confirming_source is None when no specific action warranted
 20.  output is deterministic for same inputs
 21.  bounded fields stay in [0, 1] for extreme inputs
 22.  _compute_fused_score weighted formula
 23.  _compute_uncertainty weighted formula
 24.  _compute_source_agreement: low anomaly + no compounds → high agreement
 25.  _compute_source_agreement: high anomaly + no compounds → lower agreement
 26.  _compute_source_agreement: compounds corroborating anomaly → high agreement
 27.  build_fusion_assessment returns FusionAssessment
 28.  build_fusion_assessment uses record time as timestamp
 29.  build_fusion_assessment entity_id falls back to vessel_id
 30.  build_fusion_assessment tolerates empty compounds list
 31.  build_fusion_assessment accepts None planner_context
 32.  fusion_for_timeline returns one assessment per record
 33.  fusion_for_timeline with empty timeline returns empty list
 34.  fusion_for_timeline order is preserved
 35.  Smoke test: end-to-end with run_simulation()
"""
from datetime import datetime, timezone
from dataclasses import FrozenInstanceError

import pytest

import custody.config as config
from custody.compounds import CompoundSignal
from custody.fusion import (
    FusionAssessment,
    _compute_fused_score,
    _compute_source_agreement,
    _compute_uncertainty,
    _infer_missing_evidence,
    _recommend_confirming_source,
    build_fusion_assessment,
    fusion_for_timeline,
)
from custody.models import TrackState


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

_T0 = datetime(2026, 3, 23, 12, 0, tzinfo=timezone.utc)


def _record(**kwargs):
    """Minimal timeline record with safe defaults."""
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


def _build(**kwargs) -> FusionAssessment:
    """Call build_fusion_assessment with convenient overrides."""
    record = kwargs.pop("record", _record())
    compounds = kwargs.pop("compounds", [])
    track = kwargs.pop("track", _track())
    planner_context = kwargs.pop("planner_context", None)
    # Allow simple field overrides to be applied to the record
    for k, v in kwargs.items():
        record[k] = v
    return build_fusion_assessment(record, compounds, track, planner_context)


# ---------------------------------------------------------------------------
# 1–2. Dataclass properties
# ---------------------------------------------------------------------------

class TestDataclassProperties:
    def test_fusion_assessment_is_frozen(self):
        fa = _build()
        with pytest.raises((FrozenInstanceError, TypeError)):
            fa.fused_score = 99.0  # type: ignore[misc]

    def test_has_timestamp_field(self):
        fa = _build()
        assert fa.timestamp == _T0

    def test_has_all_required_fields(self):
        fa = _build()
        assert hasattr(fa, "entity_id")
        assert hasattr(fa, "timestamp")
        assert hasattr(fa, "fused_score")
        assert hasattr(fa, "uncertainty")
        assert hasattr(fa, "source_agreement")
        assert hasattr(fa, "missing_evidence")
        assert hasattr(fa, "recommended_confirming_source")


# ---------------------------------------------------------------------------
# 3–5. Field ranges
# ---------------------------------------------------------------------------

class TestFieldRanges:
    def test_fused_score_in_range(self):
        assert 0.0 <= _build().fused_score <= 1.0

    def test_uncertainty_in_range(self):
        assert 0.0 <= _build().uncertainty <= 1.0

    def test_source_agreement_in_range(self):
        assert 0.0 <= _build().source_agreement <= 1.0

    def test_extreme_high_inputs_stay_bounded(self):
        fa = _build(
            anomaly_score=99.0,
            custody_confidence=0.0,
            compounds=[_compound(confidence=1.0)] * 5,
        )
        assert 0.0 <= fa.fused_score <= 1.0
        assert 0.0 <= fa.uncertainty <= 1.0
        assert 0.0 <= fa.source_agreement <= 1.0

    def test_extreme_low_inputs_stay_bounded(self):
        fa = _build(anomaly_score=0.0, custody_confidence=1.0)
        assert 0.0 <= fa.fused_score <= 1.0
        assert 0.0 <= fa.uncertainty <= 1.0
        assert 0.0 <= fa.source_agreement <= 1.0


# ---------------------------------------------------------------------------
# 6–8. fused_score sensitivity (via _compute_fused_score)
# ---------------------------------------------------------------------------

class TestFusedScore:
    def test_rises_with_anomaly_score(self):
        scores = [
            _compute_fused_score(a, 0.0, 0.8)
            for a in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        ]
        assert scores == sorted(scores)

    def test_rises_with_top_compound_confidence(self):
        scores = [
            _compute_fused_score(0.3, c, 0.8)
            for c in [0.0, 0.25, 0.5, 0.75, 1.0]
        ]
        assert scores == sorted(scores)

    def test_rises_when_custody_falls(self):
        scores = [
            _compute_fused_score(0.3, 0.0, conf)
            for conf in [1.0, 0.75, 0.5, 0.25, 0.0]
        ]
        assert scores == sorted(scores)

    def test_formula_weights_match_spec(self):
        """0.45 * anomaly + 0.35 * compound + 0.20 * custody_gap."""
        result = _compute_fused_score(
            anomaly_score=0.6, top_compound_confidence=0.4, custody_confidence=0.7
        )
        expected = 0.45 * 0.6 + 0.35 * 0.4 + 0.20 * 0.3
        assert result == pytest.approx(expected, abs=1e-3)


# ---------------------------------------------------------------------------
# 9–10. uncertainty sensitivity (via _compute_uncertainty)
# ---------------------------------------------------------------------------

class TestUncertainty:
    def test_rises_as_custody_confidence_falls(self):
        uncertainties = [
            _compute_uncertainty(conf, 0.8, 0.0)
            for conf in [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
        ]
        assert uncertainties == sorted(uncertainties)

    def test_rises_when_source_agreement_falls(self):
        uncertainties = [
            _compute_uncertainty(0.7, sa, 0.0)
            for sa in [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
        ]
        assert uncertainties == sorted(uncertainties)

    def test_formula_uses_evidence_gap(self):
        """Increasing evidence_gap_score increases uncertainty."""
        u_low = _compute_uncertainty(0.7, 0.8, 0.0)
        u_high = _compute_uncertainty(0.7, 0.8, 1.0)
        assert u_high > u_low


# ---------------------------------------------------------------------------
# 11–15. missing_evidence
# ---------------------------------------------------------------------------

class TestMissingEvidence:
    def test_fresh_ais_when_confidence_below_50_pct(self):
        gaps = _infer_missing_evidence(
            _record(custody_confidence=0.4), _track(), []
        )
        assert "fresh AIS update" in gaps

    def test_no_fresh_ais_when_confidence_high(self):
        gaps = _infer_missing_evidence(
            _record(custody_confidence=0.7), _track(), []
        )
        assert "fresh AIS update" not in gaps

    def test_optical_when_high_anomaly_near_zone_not_tasked(self):
        gaps = _infer_missing_evidence(
            _record(
                anomaly_score=config.HIGH_ANOMALY_THRESHOLD + 0.1,
                sensitive_zone=1.0,
                action="NONE",
            ),
            _track(), [],
        )
        assert "optical confirmation" in gaps

    def test_no_optical_when_tasked(self):
        gaps = _infer_missing_evidence(
            _record(
                anomaly_score=config.HIGH_ANOMALY_THRESHOLD + 0.1,
                sensitive_zone=1.0,
                action="TASK",
            ),
            _track(), [],
        )
        assert "optical confirmation" not in gaps

    def test_sar_when_low_confidence_and_high_anomaly(self):
        gaps = _infer_missing_evidence(
            _record(
                custody_confidence=0.3,
                anomaly_score=config.HIGH_ANOMALY_THRESHOLD + 0.1,
            ),
            _track(), [],
        )
        assert "SAR confirmation" in gaps

    def test_rf_crosscue_for_zone_related_compound(self):
        cs = _compound(code="LOITERING_NEAR_ZONE")
        gaps = _infer_missing_evidence(_record(), _track(), [cs])
        assert "RF cross-cue" in gaps

    def test_no_rf_crosscue_for_unrelated_compound(self):
        cs = _compound(code="HIGH_ANOMALY_LOW_CUSTODY")
        gaps = _infer_missing_evidence(_record(), _track(), [cs])
        assert "RF cross-cue" not in gaps

    def test_repeat_observation_when_no_sensor(self):
        gaps = _infer_missing_evidence(
            _record(action="NO_SENSOR"), _track(), []
        )
        assert "repeat observation in next access window" in gaps


# ---------------------------------------------------------------------------
# 16–19. recommended_confirming_source
# ---------------------------------------------------------------------------

class TestRecommendedConfirmingSource:
    def test_optical_when_visual_confirmation_needed(self):
        missing = ["optical confirmation"]
        rec = _recommend_confirming_source(
            _record(custody_confidence=0.7), _track(), [], missing
        )
        assert rec == "OPTICAL"

    def test_sar_when_confidence_low_and_optical_needed(self):
        """Low confidence + optical confirmation needed → SAR preferred."""
        missing = ["optical confirmation"]
        rec = _recommend_confirming_source(
            _record(custody_confidence=0.3), _track(), [], missing
        )
        assert rec == "SAR"

    def test_sar_when_sar_confirmation_in_missing(self):
        missing = ["fresh AIS update", "SAR confirmation"]
        rec = _recommend_confirming_source(
            _record(), _track(), [], missing
        )
        assert rec == "SAR"

    def test_ais_when_track_stale_no_anomaly(self):
        missing = ["fresh AIS update"]
        rec = _recommend_confirming_source(
            _record(custody_confidence=0.4, anomaly_score=0.1),
            _track(), [], missing,
        )
        assert rec == "AIS"

    def test_none_when_no_evidence_gaps(self):
        rec = _recommend_confirming_source(
            _record(custody_confidence=0.9, anomaly_score=0.05),
            _track(), [], [],
        )
        assert rec is None


# ---------------------------------------------------------------------------
# 24–26. source_agreement
# ---------------------------------------------------------------------------

class TestSourceAgreement:
    def test_low_anomaly_no_compounds_gives_high_agreement(self):
        sa = _compute_source_agreement(
            _record(anomaly_score=0.1, custody_confidence=0.9),
            [], _track(),
        )
        assert sa > 0.7

    def test_high_anomaly_no_compounds_gives_lower_agreement(self):
        sa_low_anom = _compute_source_agreement(
            _record(anomaly_score=0.1, custody_confidence=0.8), [], _track()
        )
        sa_high_anom = _compute_source_agreement(
            _record(anomaly_score=0.9, custody_confidence=0.8), [], _track()
        )
        assert sa_high_anom < sa_low_anom

    def test_compounds_corroborating_anomaly_gives_high_agreement(self):
        cs = _compound(confidence=0.8)
        sa = _compute_source_agreement(
            _record(anomaly_score=config.CRITICAL_ANOMALY_THRESHOLD * 0.8),
            [cs], _track(),
        )
        assert sa > 0.6


# ---------------------------------------------------------------------------
# 20–21. Determinism and bounds
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_inputs_give_same_output(self):
        kwargs = dict(
            record=_record(anomaly_score=0.6, custody_confidence=0.4,
                           sensitive_zone=1.0),
            compounds=[_compound()],
            track=_track(uncertainty_km=10.0),
        )
        fa1 = build_fusion_assessment(**kwargs)
        fa2 = build_fusion_assessment(**kwargs)
        assert fa1 == fa2


# ---------------------------------------------------------------------------
# 27–31. build_fusion_assessment
# ---------------------------------------------------------------------------

class TestBuildFusionAssessment:
    def test_returns_fusion_assessment(self):
        assert isinstance(_build(), FusionAssessment)

    def test_uses_record_time_as_timestamp(self):
        t = datetime(2025, 6, 1, 9, 0, tzinfo=timezone.utc)
        fa = _build(record=_record(time=t))
        assert fa.timestamp == t

    def test_entity_id_from_vessel_id_fallback(self):
        record = _record()
        del record["entity_id"]
        record["vessel_id"] = "VESSEL_X"
        fa = build_fusion_assessment(record, [], _track())
        assert fa.entity_id == "VESSEL_X"

    def test_empty_compounds_is_valid(self):
        fa = build_fusion_assessment(_record(), [], _track())
        assert isinstance(fa, FusionAssessment)

    def test_none_planner_context_is_valid(self):
        fa = build_fusion_assessment(_record(), [], _track(), None)
        assert isinstance(fa, FusionAssessment)


# ---------------------------------------------------------------------------
# 32–34. fusion_for_timeline
# ---------------------------------------------------------------------------

class TestFusionForTimeline:
    def test_returns_one_per_record(self):
        timeline = [_record(entity_id="V001")] * 5
        result = fusion_for_timeline(timeline, _track())
        assert len(result) == 5

    def test_empty_timeline_returns_empty(self):
        assert fusion_for_timeline([], _track()) == []

    def test_order_preserved(self):
        t1 = datetime(2026, 1, 1, 10, tzinfo=timezone.utc)
        t2 = datetime(2026, 1, 1, 11, tzinfo=timezone.utc)
        t3 = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
        timeline = [_record(time=t) for t in (t1, t2, t3)]
        result = fusion_for_timeline(timeline, _track())
        assert [fa.timestamp for fa in result] == [t1, t2, t3]


# ---------------------------------------------------------------------------
# 35. Smoke test
# ---------------------------------------------------------------------------

def test_end_to_end_with_simulation():
    """run_simulation → split by vessel → fusion_for_timeline → all valid."""
    from custody.simulate import run_simulation
    from collections import defaultdict

    records = run_simulation()

    # Group by entity and build a fresh TrackState per vessel
    by_vessel: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        vid = r.get("entity_id") or r.get("vessel_id") or r.get("target_id")
        by_vessel[vid].append(r)

    for vessel_id, timeline in by_vessel.items():
        track = TrackState()
        assessments = fusion_for_timeline(timeline, track)
        assert len(assessments) == len(timeline)
        for fa in assessments:
            assert isinstance(fa, FusionAssessment)
            assert fa.entity_id == vessel_id
            assert 0.0 <= fa.fused_score <= 1.0
            assert 0.0 <= fa.uncertainty <= 1.0
            assert 0.0 <= fa.source_agreement <= 1.0
            assert isinstance(fa.missing_evidence, list)


def test_smoke_with_simulation_target_id_key():
    """Simulation records use 'target_id' — builder should resolve it."""
    from custody.simulate import run_simulation
    records = run_simulation()
    track = TrackState()
    fa = build_fusion_assessment(records[0], [], track)
    assert fa.entity_id != "unknown"
