"""
Tests for entity_detail_data.py — pure-data helpers extracted from
entity_detail_panel.py for UI-framework-independent reuse.

Covers:
  1. entity_id extracts target_id
  2. entity_id falls back to vessel_id
  3. entity_id falls back to entity_id key
  4. entity_id returns "unknown" when no ID key present
  5. derive_track_state sets uncertainty from record
  6. derive_track_state finds last collection in prefix
  7. derive_track_state handles empty prefix
  8. derive_track_state skips non-TASK records
  9. panel_summary returns non-empty string
 10. panel_summary mentions action
 11. panel_summary varies with fused_score level
"""
from __future__ import annotations

import sys
import os
from datetime import datetime, timezone

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "app"))

from entity_detail_data import entity_id, derive_track_state, panel_summary
from custody.fusion import FusionAssessment
from custody.decision import Decision

UTC = timezone.utc
_T0 = datetime(2026, 3, 23, 10, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# entity_id
# ---------------------------------------------------------------------------

class TestEntityId:

    def test_extracts_target_id(self):
        assert entity_id({"target_id": "V001"}) == "V001"

    def test_falls_back_to_vessel_id(self):
        assert entity_id({"vessel_id": "V002"}) == "V002"

    def test_falls_back_to_entity_id_key(self):
        assert entity_id({"entity_id": "E003"}) == "E003"

    def test_returns_unknown(self):
        assert entity_id({}) == "unknown"

    def test_prefers_entity_id_over_vessel_id(self):
        assert entity_id({"entity_id": "E1", "vessel_id": "V1", "target_id": "T1"}) == "E1"


# ---------------------------------------------------------------------------
# derive_track_state
# ---------------------------------------------------------------------------

class TestDeriveTrackState:

    def test_sets_uncertainty_from_record(self):
        track = derive_track_state({"uncertainty_km": 25.0}, [])
        assert track.uncertainty_km == 25.0

    def test_default_uncertainty(self):
        track = derive_track_state({}, [])
        assert track.uncertainty_km == 5.0

    def test_finds_last_collection(self):
        prefix = [
            {"action": "NONE", "time": _T0},
            {"action": "TASK", "collection_result": "SUCCESS", "time": _T0, "anomaly_score": 0.8},
        ]
        track = derive_track_state({"uncertainty_km": 10.0}, prefix)
        assert track.last_collection_time == _T0
        assert track.last_collection_anomaly_score == 0.8

    def test_empty_prefix(self):
        track = derive_track_state({"uncertainty_km": 10.0}, [])
        assert track.last_collection_time is None
        assert track.last_collection_anomaly_score == 0.0

    def test_skips_non_task_records(self):
        prefix = [
            {"action": "HOLD", "collection_result": None, "time": _T0, "anomaly_score": 0.5},
            {"action": "NONE", "collection_result": None, "time": _T0, "anomaly_score": 0.3},
        ]
        track = derive_track_state({"uncertainty_km": 10.0}, prefix)
        assert track.last_collection_time is None

    def test_skips_task_with_no_result(self):
        prefix = [
            {"action": "TASK", "collection_result": None, "time": _T0, "anomaly_score": 0.5},
        ]
        track = derive_track_state({"uncertainty_km": 10.0}, prefix)
        assert track.last_collection_time is None


# ---------------------------------------------------------------------------
# panel_summary
# ---------------------------------------------------------------------------

class TestPanelSummary:

    @staticmethod
    def _fa(fused_score=0.5, uncertainty=0.4):
        return FusionAssessment(
            entity_id="X", timestamp=_T0,
            fused_score=fused_score, uncertainty=uncertainty,
            source_agreement=0.8, missing_evidence=[], recommended_confirming_source=None,
        )

    @staticmethod
    def _dec(action="ELEVATE"):
        return Decision(
            entity_id="X", timestamp=_T0,
            action=action, priority=0.5, confidence=0.7,
            why=["test"], next_best_actions=["PASSIVE_MONITOR"],
        )

    def test_returns_nonempty_string(self):
        result = panel_summary(self._fa(), self._dec())
        assert isinstance(result, str)
        assert len(result) > 10

    def test_mentions_action(self):
        result = panel_summary(self._fa(), self._dec("TASK_SAR"))
        assert "TASK SAR" in result

    def test_elevated_fused_score(self):
        result = panel_summary(self._fa(fused_score=0.75), self._dec())
        assert "elevated" in result

    def test_low_fused_score(self):
        result = panel_summary(self._fa(fused_score=0.10), self._dec())
        assert "low" in result

    def test_moderate_fused_score(self):
        result = panel_summary(self._fa(fused_score=0.45), self._dec())
        assert "moderate" in result
