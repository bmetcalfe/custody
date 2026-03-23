"""
Unit tests for custody/models.py domain types.

Covers:
  - BehaviorState enum membership and values
  - DetectorResult construction, field access, immutability, and metadata extension
  - Existing HistoryEntry and Vessel types are not retested here (covered
    transitively by test_anomalies.py and test_dashboard.py).
"""
import pytest

from custody.models import BehaviorState, DetectorResult


# ---------------------------------------------------------------------------
# BehaviorState
# ---------------------------------------------------------------------------

class TestBehaviorState:
    EXPECTED_STATES = {"TRANSIT", "APPROACH", "LOITER", "EGRESS", "IDLE", "UNKNOWN"}

    def test_all_expected_members_exist(self):
        actual = {member.name for member in BehaviorState}
        assert actual == self.EXPECTED_STATES

    def test_values_are_lowercase_strings(self):
        for member in BehaviorState:
            assert isinstance(member.value, str)
            assert member.value == member.value.lower()

    def test_value_matches_name_lowercased(self):
        for member in BehaviorState:
            assert member.value == member.name.lower()

    def test_lookup_by_value(self):
        assert BehaviorState("transit") is BehaviorState.TRANSIT
        assert BehaviorState("loiter") is BehaviorState.LOITER
        assert BehaviorState("unknown") is BehaviorState.UNKNOWN

    def test_unknown_value_raises(self):
        with pytest.raises(ValueError):
            BehaviorState("flying")

    def test_members_are_distinct(self):
        members = list(BehaviorState)
        assert len(members) == len(self.EXPECTED_STATES)


# ---------------------------------------------------------------------------
# DetectorResult
# ---------------------------------------------------------------------------

class TestDetectorResult:
    def test_basic_construction(self):
        r = DetectorResult(name="loitering", score=0.5, evidence="speed below threshold")
        assert r.name == "loitering"
        assert r.score == 0.5
        assert r.evidence == "speed below threshold"

    def test_metadata_defaults_to_empty_dict(self):
        r = DetectorResult(name="x", score=0.0, evidence="none")
        assert r.metadata == {}

    def test_metadata_accepts_structured_data(self):
        meta = {"slow_steps": 3, "threshold_kmh": 5.0}
        r = DetectorResult(name="loitering", score=0.8, evidence="3 slow steps", metadata=meta)
        assert r.metadata["slow_steps"] == 3
        assert r.metadata["threshold_kmh"] == 5.0

    def test_is_immutable(self):
        r = DetectorResult(name="zone", score=1.5, evidence="inside zone")
        with pytest.raises((AttributeError, TypeError)):
            r.score = 0.0  # type: ignore[misc]

    def test_score_can_be_zero(self):
        r = DetectorResult(name="route_deviation", score=0.0, evidence="heading within tolerance")
        assert r.score == 0.0

    def test_score_can_be_fractional(self):
        r = DetectorResult(name="sensitive_zone", score=0.375, evidence="partial halo proximity")
        assert r.score == pytest.approx(0.375)

    def test_evidence_is_non_empty_string(self):
        r = DetectorResult(name="loitering", score=0.5, evidence="speed 2 km/h < threshold 5 km/h")
        assert isinstance(r.evidence, str)
        assert len(r.evidence) > 0

    def test_name_identifies_detector(self):
        for name in ("sensitive_zone", "loitering", "route_deviation"):
            r = DetectorResult(name=name, score=0.0, evidence="-")
            assert r.name == name

    def test_two_equal_results_compare_equal(self):
        r1 = DetectorResult(name="x", score=1.0, evidence="e")
        r2 = DetectorResult(name="x", score=1.0, evidence="e")
        assert r1 == r2

    def test_differing_scores_are_not_equal(self):
        r1 = DetectorResult(name="x", score=0.5, evidence="e")
        r2 = DetectorResult(name="x", score=1.0, evidence="e")
        assert r1 != r2
