"""Tests for :mod:`custody.hypotheses.types` (ADR-0021).

Covers the three frozen dataclasses — HypothesisEvidence, Hypothesis,
HypothesisState — that form the hypothesis-layer contract.  The layer
sits alongside (not inside) the existing seven-layer entity reasoning
stack; these types must not replace or reshape FusionAssessment.
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from custody.hypotheses.types import (
    Hypothesis,
    HypothesisEvidence,
    HypothesisState,
)


# ---------------------------------------------------------------------------
# HypothesisEvidence
# ---------------------------------------------------------------------------


def _mk_evidence(**overrides) -> HypothesisEvidence:
    kw = dict(
        evidence_id="ev-001",
        source_ref="observation:obs-42",
        source_kind="observation",
        scenario_id="tennent",
        timestamp=datetime(2023, 7, 2, 12, 0, 0, tzinfo=timezone.utc),
        supports=("fixed_reclamation_or_structure",),
        contradicts=("no_meaningful_activity",),
        confidence=0.7,
        weight=1.0,
        reason="bright persistent scatterer at reef coordinates",
    )
    kw.update(overrides)
    return HypothesisEvidence(**kw)


def test_evidence_is_frozen() -> None:
    ev = _mk_evidence()
    with pytest.raises(FrozenInstanceError):
        ev.confidence = 0.1  # type: ignore[misc]


def test_evidence_stores_typed_source_ref_string() -> None:
    """source_ref is a typed reference string — not an inline source object."""
    ev = _mk_evidence(source_ref="scene:tennent_20230702_umbra-05")
    assert isinstance(ev.source_ref, str)
    assert ev.source_ref.startswith("scene:")


def test_evidence_allows_null_source_ref() -> None:
    """Evidence without a single backing source is allowed (e.g. derived signals)."""
    ev = _mk_evidence(source_ref=None)
    assert ev.source_ref is None


def test_evidence_supports_and_contradicts_are_tuples() -> None:
    ev = _mk_evidence()
    assert isinstance(ev.supports, tuple)
    assert isinstance(ev.contradicts, tuple)


# ---------------------------------------------------------------------------
# Hypothesis
# ---------------------------------------------------------------------------


def test_hypothesis_is_frozen() -> None:
    h = Hypothesis(
        hypothesis_id="fixed_reclamation_or_structure",
        label="Fixed reclamation or structure",
        description="Persistent human-made feature on the reef.",
        prior=0.2,
        scenario_id="tennent",
    )
    with pytest.raises(FrozenInstanceError):
        h.prior = 0.5  # type: ignore[misc]


def test_hypothesis_fields_round_trip() -> None:
    h = Hypothesis(
        hypothesis_id="vessel_cluster_activity",
        label="Vessel cluster",
        description="Multiple vessels co-located over several scenes.",
        prior=0.2,
        scenario_id="whitsun",
    )
    assert h.hypothesis_id == "vessel_cluster_activity"
    assert h.scenario_id == "whitsun"
    assert h.prior == 0.2


# ---------------------------------------------------------------------------
# HypothesisState
# ---------------------------------------------------------------------------


def test_state_is_frozen() -> None:
    state = HypothesisState(
        scenario_id="tennent",
        timestamp=None,
        scores={"fixed_reclamation_or_structure": 0.3},
        top_hypothesis="fixed_reclamation_or_structure",
        uncertainty=0.7,
        supporting_evidence={},
        contradicting_evidence={},
        explanation=(),
    )
    with pytest.raises(FrozenInstanceError):
        state.uncertainty = 0.0  # type: ignore[misc]


def test_state_accepts_null_top_hypothesis_and_timestamp() -> None:
    """A brand-new state with no evidence may have no top hypothesis or timestamp."""
    state = HypothesisState(
        scenario_id="tennent",
        timestamp=None,
        scores={},
        top_hypothesis=None,
        uncertainty=1.0,
        supporting_evidence={},
        contradicting_evidence={},
        explanation=(),
    )
    assert state.top_hypothesis is None
    assert state.timestamp is None


def test_state_explanation_is_tuple() -> None:
    state = HypothesisState(
        scenario_id="whitsun",
        timestamp=None,
        scores={"vessel_cluster_activity": 0.4},
        top_hypothesis="vessel_cluster_activity",
        uncertainty=0.6,
        supporting_evidence={},
        contradicting_evidence={},
        explanation=("ev-1: +0.2 to vessel_cluster_activity (clustered detections)",),
    )
    assert isinstance(state.explanation, tuple)
    assert state.explanation[0].startswith("ev-1")
