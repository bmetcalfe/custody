"""Tests for Whitsun scenario evidence generators (ADR-0021 Slice 3).

Pins the catalog mapping for vessel-cluster / transient-anchorage /
AIS-dark / no-persistent-activity signals and guards the never-claim-
dark-vessels-under-solid-coverage invariant.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from custody.fusion.scenes import Scene
from custody.fusion.temporal import Match
from custody.hypotheses.registry import (
    SCENARIO_WHITSUN,
    WHITSUN_AIS_DARK_OR_POORLY_OBSERVED_VESSELS,
    WHITSUN_DETECTOR_CLUTTER_FALSE_POSITIVES,
    WHITSUN_NO_PERSISTENT_ACTIVITY,
    WHITSUN_TRANSIENT_ANCHORAGE_OR_FISHING_PRESENCE,
    WHITSUN_VESSEL_CLUSTER_ACTIVITY,
    get_hypothesis_ids,
)
from custody.hypotheses.scenarios import (
    _ais_coverage_signal,
    whitsun_evidence_from_match,
    whitsun_evidence_from_scene,
)
from custody.hypotheses.types import HypothesisEvidence


T_EPOCH = datetime(2023, 12, 6, 10, 0, 0, tzinfo=timezone.utc).timestamp()


def _mk_scene(scene_id: str = "whitsun_20231206_umbra-04") -> Scene:
    return Scene(
        scene_id=scene_id,
        sensor="umbra-04",
        acquisition_time=T_EPOCH,
        pixel_size_m=0.25,
        center_lat=9.98,
        center_lon=114.63,
        footprint_latlon=(
            (9.96, 114.60), (9.96, 114.66),
            (10.00, 114.66), (10.00, 114.60),
        ),
        raw_scene_path=None,
        quality_flag="yellow",
    )


def _mk_match() -> Match:
    return Match(obs_a_id="w-obs-a", obs_b_id="w-obs-b", distance_m=15.0)


# ---------------------------------------------------------------------------
# _ais_coverage_signal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flag", ["solid", "sparse", "none"])
def test_ais_coverage_signal_accepts_valid_flags(flag: str) -> None:
    # Return value shape is an implementation detail; the important behaviour is
    # "accepted without raising".  Downstream semantics are exercised via the
    # scene-generator tests.
    _ais_coverage_signal(flag)


def test_ais_coverage_signal_unknown_raises_listing_valid() -> None:
    with pytest.raises(ValueError) as excinfo:
        _ais_coverage_signal("nonsense")
    msg = str(excinfo.value)
    for valid in ("solid", "sparse", "none"):
        assert valid in msg


# ---------------------------------------------------------------------------
# whitsun_evidence_from_scene
# ---------------------------------------------------------------------------


def test_scene_cluster_with_three_or_more_vessels() -> None:
    scene = _mk_scene()
    evs = whitsun_evidence_from_scene(
        scene, vessel_count=5, ais_coverage_flag="solid", quality_flag="green",
    )
    support_sets = [set(e.supports) for e in evs]
    assert {WHITSUN_VESSEL_CLUSTER_ACTIVITY} in support_sets
    # And that same evidence contradicts no_persistent_activity
    for ev in evs:
        if WHITSUN_VESSEL_CLUSTER_ACTIVITY in ev.supports:
            assert WHITSUN_NO_PERSISTENT_ACTIVITY in ev.contradicts


@pytest.mark.parametrize("n", [1, 2])
def test_scene_transient_anchorage_with_one_or_two_vessels(n: int) -> None:
    scene = _mk_scene()
    evs = whitsun_evidence_from_scene(
        scene, vessel_count=n, ais_coverage_flag="solid", quality_flag="yellow",
    )
    support_sets = [set(e.supports) for e in evs]
    assert {WHITSUN_TRANSIENT_ANCHORAGE_OR_FISHING_PRESENCE} in support_sets
    # Never cluster at n=1 or 2
    assert {WHITSUN_VESSEL_CLUSTER_ACTIVITY} not in support_sets


def test_scene_ais_dark_emitted_when_sparse_and_vessels_present() -> None:
    scene = _mk_scene()
    evs = whitsun_evidence_from_scene(
        scene, vessel_count=5, ais_coverage_flag="sparse", quality_flag="yellow",
    )
    support_sets = [set(e.supports) for e in evs]
    assert {WHITSUN_AIS_DARK_OR_POORLY_OBSERVED_VESSELS} in support_sets
    assert {WHITSUN_VESSEL_CLUSTER_ACTIVITY} in support_sets
    assert len(evs) == 2  # cluster + ais-dark, separate evidence items


def test_scene_ais_dark_also_emitted_when_none_coverage_and_vessels_present() -> None:
    scene = _mk_scene()
    evs = whitsun_evidence_from_scene(
        scene, vessel_count=2, ais_coverage_flag="none", quality_flag="yellow",
    )
    support_sets = [set(e.supports) for e in evs]
    assert {WHITSUN_AIS_DARK_OR_POORLY_OBSERVED_VESSELS} in support_sets


def test_scene_never_claims_ais_dark_under_solid_coverage_with_zero_vessels() -> None:
    scene = _mk_scene()
    evs = whitsun_evidence_from_scene(
        scene, vessel_count=0, ais_coverage_flag="solid", quality_flag="green",
    )
    for ev in evs:
        assert WHITSUN_AIS_DARK_OR_POORLY_OBSERVED_VESSELS not in ev.supports


def test_scene_solid_coverage_with_zero_vessels_supports_no_persistent() -> None:
    scene = _mk_scene()
    evs = whitsun_evidence_from_scene(
        scene, vessel_count=0, ais_coverage_flag="solid", quality_flag="green",
    )
    support_sets = [set(e.supports) for e in evs]
    assert {WHITSUN_NO_PERSISTENT_ACTIVITY} in support_sets
    for ev in evs:
        if WHITSUN_NO_PERSISTENT_ACTIVITY in ev.supports:
            assert WHITSUN_VESSEL_CLUSTER_ACTIVITY in ev.contradicts


def test_scene_zero_vessels_under_sparse_or_none_does_not_emit_no_persistent() -> None:
    """The no-persistent claim requires SOLID coverage — otherwise AIS absence is uninformative."""
    scene = _mk_scene()
    for flag in ("sparse", "none"):
        evs = whitsun_evidence_from_scene(
            scene, vessel_count=0, ais_coverage_flag=flag, quality_flag="yellow",
        )
        for ev in evs:
            assert WHITSUN_NO_PERSISTENT_ACTIVITY not in ev.supports


def test_scene_confidence_follows_quality_flag() -> None:
    scene = _mk_scene()
    for flag, expected in [("green", 1.0), ("yellow", 0.6), ("red", 0.2)]:
        evs = whitsun_evidence_from_scene(
            scene, vessel_count=5, ais_coverage_flag="solid", quality_flag=flag,
        )
        for ev in evs:
            assert ev.confidence == expected


# ---------------------------------------------------------------------------
# whitsun_evidence_from_match
# ---------------------------------------------------------------------------


def test_match_cluster_supports_cluster_contradicts_clutter() -> None:
    ev = whitsun_evidence_from_match(_mk_match(), matched_cluster=True, quality_flag="green")
    assert WHITSUN_VESSEL_CLUSTER_ACTIVITY in ev.supports
    assert WHITSUN_DETECTOR_CLUTTER_FALSE_POSITIVES in ev.contradicts


def test_match_noncluster_supports_transient_anchorage_contradicts_cluster() -> None:
    ev = whitsun_evidence_from_match(_mk_match(), matched_cluster=False, quality_flag="yellow")
    assert WHITSUN_TRANSIENT_ANCHORAGE_OR_FISHING_PRESENCE in ev.supports
    assert WHITSUN_VESSEL_CLUSTER_ACTIVITY in ev.contradicts


# ---------------------------------------------------------------------------
# Registry guardrail
# ---------------------------------------------------------------------------


def test_all_generated_evidence_validates_against_whitsun_registry() -> None:
    valid = set(get_hypothesis_ids(SCENARIO_WHITSUN))
    scene = _mk_scene()

    combos = [
        (5, "solid"), (5, "sparse"), (5, "none"),
        (2, "solid"), (2, "sparse"), (2, "none"),
        (0, "solid"), (0, "sparse"), (0, "none"),
    ]
    for n, flag in combos:
        evs = whitsun_evidence_from_scene(
            scene, vessel_count=n, ais_coverage_flag=flag, quality_flag="yellow",
        )
        for ev in evs:
            assert isinstance(ev, HypothesisEvidence)
            assert ev.scenario_id == SCENARIO_WHITSUN
            assert all(h in valid for h in ev.supports)
            assert all(h in valid for h in ev.contradicts)

    for matched in (True, False):
        ev = whitsun_evidence_from_match(_mk_match(), matched_cluster=matched, quality_flag="yellow")
        assert isinstance(ev, HypothesisEvidence)
        assert all(h in valid for h in ev.supports)
        assert all(h in valid for h in ev.contradicts)
