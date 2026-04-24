"""Tests for :mod:`custody.hypotheses.registry` (ADR-0021).

The registry is scenario-specific by design: Tennent and Whitsun must not
share hypothesis labels, and unknown scenarios must fail loudly.
"""
from __future__ import annotations

import math

import pytest

from custody.hypotheses.registry import (
    SCENARIO_TENNENT,
    SCENARIO_WHITSUN,
    TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
    TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
    TENNENT_STATIONARY_SAR_SCATTER_OR_REEF_CLUTTER,
    TENNENT_TRANSIENT_VESSEL_ACTIVITY,
    TENNENT_NO_MEANINGFUL_ACTIVITY,
    WHITSUN_VESSEL_CLUSTER_ACTIVITY,
    WHITSUN_TRANSIENT_ANCHORAGE_OR_FISHING_PRESENCE,
    WHITSUN_AIS_DARK_OR_POORLY_OBSERVED_VESSELS,
    WHITSUN_DETECTOR_CLUTTER_FALSE_POSITIVES,
    WHITSUN_NO_PERSISTENT_ACTIVITY,
    get_hypotheses,
    get_hypothesis_ids,
)
from custody.hypotheses.types import Hypothesis


EXPECTED_TENNENT_IDS = {
    TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
    TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
    TENNENT_STATIONARY_SAR_SCATTER_OR_REEF_CLUTTER,
    TENNENT_TRANSIENT_VESSEL_ACTIVITY,
    TENNENT_NO_MEANINGFUL_ACTIVITY,
}

EXPECTED_WHITSUN_IDS = {
    WHITSUN_VESSEL_CLUSTER_ACTIVITY,
    WHITSUN_TRANSIENT_ANCHORAGE_OR_FISHING_PRESENCE,
    WHITSUN_AIS_DARK_OR_POORLY_OBSERVED_VESSELS,
    WHITSUN_DETECTOR_CLUTTER_FALSE_POSITIVES,
    WHITSUN_NO_PERSISTENT_ACTIVITY,
}


def test_scenario_id_constants_are_stable() -> None:
    assert SCENARIO_TENNENT == "tennent"
    assert SCENARIO_WHITSUN == "whitsun"


def test_tennent_set_matches_spec() -> None:
    ids = set(get_hypothesis_ids(SCENARIO_TENNENT))
    assert ids == EXPECTED_TENNENT_IDS


def test_whitsun_set_matches_spec() -> None:
    ids = set(get_hypothesis_ids(SCENARIO_WHITSUN))
    assert ids == EXPECTED_WHITSUN_IDS


def test_tennent_and_whitsun_sets_are_distinct() -> None:
    """Hypothesis IDs must not leak between scenarios (ADR-0012)."""
    overlap = EXPECTED_TENNENT_IDS & EXPECTED_WHITSUN_IDS
    assert overlap == set(), f"hypothesis ids overlap between scenarios: {overlap}"

    tennent = set(get_hypothesis_ids(SCENARIO_TENNENT))
    whitsun = set(get_hypothesis_ids(SCENARIO_WHITSUN))
    assert tennent.isdisjoint(whitsun)


def test_get_hypotheses_returns_hypothesis_objects() -> None:
    hs = get_hypotheses(SCENARIO_TENNENT)
    assert isinstance(hs, tuple)
    assert len(hs) == 5
    for h in hs:
        assert isinstance(h, Hypothesis)
        assert h.scenario_id == SCENARIO_TENNENT


def test_priors_are_uniform_and_sum_to_one() -> None:
    for scenario in (SCENARIO_TENNENT, SCENARIO_WHITSUN):
        hs = get_hypotheses(scenario)
        expected = 1.0 / len(hs)
        for h in hs:
            assert h.prior == pytest.approx(expected)
        assert math.isclose(sum(h.prior for h in hs), 1.0)


def test_unknown_scenario_raises_value_error_with_valid_ids() -> None:
    with pytest.raises(ValueError) as excinfo:
        get_hypotheses("atlantis")
    msg = str(excinfo.value)
    assert SCENARIO_TENNENT in msg
    assert SCENARIO_WHITSUN in msg


def test_unknown_scenario_ids_also_raises() -> None:
    with pytest.raises(ValueError):
        get_hypothesis_ids("atlantis")
