"""Tests for Tennent scenario evidence generators (ADR-0021 Slice 3).

The generators are the first place domain heuristics enter the hypothesis
layer.  Tests pin the catalog mapping (signal → supports/contradicts),
quality→confidence, deterministic evidence_ids, and the guardrail that all
evidence validates against the Tennent registry via the Slice 2 adapters.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from custody.fusion.observations import PositionObservation
from custody.fusion.scenes import Scene
from custody.fusion.temporal import Match
from custody.hypotheses.registry import (
    SCENARIO_TENNENT,
    TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
    TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
    TENNENT_NO_MEANINGFUL_ACTIVITY,
    TENNENT_STATIONARY_SAR_SCATTER_OR_REEF_CLUTTER,
    TENNENT_TRANSIENT_VESSEL_ACTIVITY,
    get_hypothesis_ids,
)
from custody.hypotheses.scenarios import (
    _quality_to_confidence,
    tennent_evidence_from_match,
    tennent_evidence_from_scene,
)
from custody.hypotheses.types import HypothesisEvidence


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


T_EPOCH = datetime(2023, 7, 2, 14, 0, 55, tzinfo=timezone.utc).timestamp()


def _mk_scene(scene_id: str = "tennent_20230702_umbra-01") -> Scene:
    return Scene(
        scene_id=scene_id,
        sensor="umbra-01",
        acquisition_time=T_EPOCH,
        pixel_size_m=0.25,
        center_lat=8.856,
        center_lon=114.665,
        footprint_latlon=(
            (8.84, 114.65), (8.84, 114.68),
            (8.87, 114.68), (8.87, 114.65),
        ),
        raw_scene_path=None,
        quality_flag="yellow",
    )


def _mk_match() -> Match:
    return Match(obs_a_id="t-obs-a", obs_b_id="t-obs-b", distance_m=5.0)


# ---------------------------------------------------------------------------
# _quality_to_confidence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "flag,expected",
    [("green", 1.0), ("yellow", 0.6), ("red", 0.2)],
)
def test_quality_to_confidence_maps_known_flags(flag: str, expected: float) -> None:
    assert _quality_to_confidence(flag) == expected


def test_quality_to_confidence_unknown_flag_raises_listing_valid() -> None:
    with pytest.raises(ValueError) as excinfo:
        _quality_to_confidence("nonsense")
    msg = str(excinfo.value)
    for valid in ("green", "yellow", "red"):
        assert valid in msg


# ---------------------------------------------------------------------------
# tennent_evidence_from_scene
# ---------------------------------------------------------------------------


def test_scene_persistent_scatterer_supports_fixed_contradicts_no_meaningful() -> None:
    scene = _mk_scene()
    evs = tennent_evidence_from_scene(
        scene, persistent_scatterer_detected=True, change_signal_strength=0.0,
        quality_flag="yellow",
    )
    assert len(evs) == 1
    ev = evs[0]
    assert TENNENT_FIXED_RECLAMATION_OR_STRUCTURE in ev.supports
    assert TENNENT_NO_MEANINGFUL_ACTIVITY in ev.contradicts


def test_scene_change_signal_supports_construction_contradicts_stationary() -> None:
    scene = _mk_scene()
    evs = tennent_evidence_from_scene(
        scene, persistent_scatterer_detected=False, change_signal_strength=0.30,
        quality_flag="yellow",
    )
    assert len(evs) == 1
    ev = evs[0]
    assert TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY in ev.supports
    assert TENNENT_STATIONARY_SAR_SCATTER_OR_REEF_CLUTTER in ev.contradicts


def test_scene_both_signals_produce_two_separate_evidence_items() -> None:
    scene = _mk_scene()
    evs = tennent_evidence_from_scene(
        scene, persistent_scatterer_detected=True, change_signal_strength=0.35,
        quality_flag="green",
    )
    assert len(evs) == 2
    support_sets = [set(e.supports) for e in evs]
    assert {TENNENT_FIXED_RECLAMATION_OR_STRUCTURE} in support_sets
    assert {TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY} in support_sets


def test_scene_below_change_threshold_does_not_emit_construction() -> None:
    scene = _mk_scene()
    evs = tennent_evidence_from_scene(
        scene, persistent_scatterer_detected=False, change_signal_strength=0.24,
        quality_flag="yellow",
    )
    assert evs == ()


def test_scene_confidence_follows_quality_flag() -> None:
    scene = _mk_scene()
    for flag, expected in [("green", 1.0), ("yellow", 0.6), ("red", 0.2)]:
        evs = tennent_evidence_from_scene(
            scene, persistent_scatterer_detected=True, change_signal_strength=0.0,
            quality_flag=flag,
        )
        assert evs[0].confidence == expected


# ---------------------------------------------------------------------------
# tennent_evidence_from_match
# ---------------------------------------------------------------------------


def test_match_low_displacement_supports_fixed_contradicts_transient() -> None:
    ev = tennent_evidence_from_match(_mk_match(), displacement_m=5.0, quality_flag="green")
    assert TENNENT_FIXED_RECLAMATION_OR_STRUCTURE in ev.supports
    assert TENNENT_TRANSIENT_VESSEL_ACTIVITY in ev.contradicts


def test_match_moderate_displacement_supports_construction_contradicts_stationary() -> None:
    ev = tennent_evidence_from_match(_mk_match(), displacement_m=25.0, quality_flag="yellow")
    assert TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY in ev.supports
    assert TENNENT_STATIONARY_SAR_SCATTER_OR_REEF_CLUTTER in ev.contradicts


def test_match_high_displacement_supports_transient_contradicts_fixed() -> None:
    ev = tennent_evidence_from_match(_mk_match(), displacement_m=45.0, quality_flag="green")
    assert TENNENT_TRANSIENT_VESSEL_ACTIVITY in ev.supports
    assert TENNENT_FIXED_RECLAMATION_OR_STRUCTURE in ev.contradicts


def test_match_boundary_10m_goes_to_moderate() -> None:
    """Spec: displacement_m < 10.0 is low; [10.0, 40.0) is moderate."""
    ev = tennent_evidence_from_match(_mk_match(), displacement_m=10.0, quality_flag="green")
    assert TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY in ev.supports


def test_match_boundary_40m_goes_to_high() -> None:
    """Spec: [10.0, 40.0) moderate; d >= 40.0 high."""
    ev = tennent_evidence_from_match(_mk_match(), displacement_m=40.0, quality_flag="green")
    assert TENNENT_TRANSIENT_VESSEL_ACTIVITY in ev.supports


# ---------------------------------------------------------------------------
# Registry guardrail + determinism
# ---------------------------------------------------------------------------


def _all_ids_valid(evs, scenario_id: str) -> bool:
    valid = set(get_hypothesis_ids(scenario_id))
    for e in evs:
        if not all(h in valid for h in e.supports):
            return False
        if not all(h in valid for h in e.contradicts):
            return False
    return True


def test_all_scene_evidence_validates_against_tennent_registry() -> None:
    scene = _mk_scene()
    evs = tennent_evidence_from_scene(
        scene, persistent_scatterer_detected=True, change_signal_strength=0.35,
        quality_flag="yellow",
    )
    assert _all_ids_valid(evs, SCENARIO_TENNENT)
    for ev in evs:
        assert isinstance(ev, HypothesisEvidence)
        assert ev.scenario_id == SCENARIO_TENNENT


def test_all_match_evidence_validates_against_tennent_registry() -> None:
    for d in (5.0, 25.0, 45.0):
        ev = tennent_evidence_from_match(_mk_match(), displacement_m=d, quality_flag="yellow")
        assert _all_ids_valid((ev,), SCENARIO_TENNENT)


def test_scene_evidence_ids_are_deterministic() -> None:
    scene = _mk_scene()
    evs1 = tennent_evidence_from_scene(
        scene, persistent_scatterer_detected=True, change_signal_strength=0.30,
        quality_flag="yellow",
    )
    evs2 = tennent_evidence_from_scene(
        scene, persistent_scatterer_detected=True, change_signal_strength=0.30,
        quality_flag="yellow",
    )
    assert [e.evidence_id for e in evs1] == [e.evidence_id for e in evs2]


def test_match_evidence_id_is_deterministic() -> None:
    m = _mk_match()
    ev1 = tennent_evidence_from_match(m, displacement_m=5.0, quality_flag="yellow")
    ev2 = tennent_evidence_from_match(m, displacement_m=5.0, quality_flag="yellow")
    assert ev1.evidence_id == ev2.evidence_id
