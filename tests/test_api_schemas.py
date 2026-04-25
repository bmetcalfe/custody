"""Tests for :mod:`custody.api.schemas` (ADR-0021 Slice 17).

Pin request defaults, validation helpers, request-id determinism, and
the JSON-serializability of :class:`ApiResponse`.
"""
from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

from custody.api.schemas import (
    ApiResponse,
    CollectRankingRequest,
    DecisionPacketRequest,
    OptimizePlanRequest,
    PlannerQueueRequest,
    PolicyEvaluationRequest,
    PortfolioAllocationRequest,
    _make_request_id,
    response_to_dict,
    validate_positive_float,
    validate_positive_int,
    validate_scenario_id,
    validate_scenario_ids,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_decision_packet_request_defaults() -> None:
    r = DecisionPacketRequest(scenario_id="tennent")
    assert r.include_mission_value is False
    assert r.output_format == "json"


def test_collect_ranking_request_defaults() -> None:
    r = CollectRankingRequest(scenario_id="tennent")
    assert r.max_results is None


def test_optimize_plan_request_defaults() -> None:
    r = OptimizePlanRequest(scenario_id="tennent")
    assert r.budget == 1.0
    assert r.max_collects == 2
    assert r.strategy == "recommended"


def test_policy_evaluation_request_defaults() -> None:
    r = PolicyEvaluationRequest(scenario_id="whitsun")
    assert r.budget == 1.0
    assert r.max_collects == 2
    assert r.policies is None


def test_planner_queue_request_defaults() -> None:
    r = PlannerQueueRequest()
    assert r.scenario_ids == ("tennent", "whitsun")
    assert r.include_reviews is False


def test_portfolio_allocation_request_defaults() -> None:
    r = PortfolioAllocationRequest()
    assert r.scenario_ids == ("tennent", "whitsun")
    assert r.budget == 1.5
    assert r.max_collects == 3
    assert r.max_collects_per_scenario == 2


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario_id", ["tennent", "whitsun"])
def test_validate_scenario_id_accepts_valid(scenario_id: str) -> None:
    validate_scenario_id(scenario_id)


def test_validate_scenario_id_rejects_invalid() -> None:
    with pytest.raises(ValueError):
        validate_scenario_id("invalid")


def test_validate_scenario_ids_rejects_invalid_member() -> None:
    with pytest.raises(ValueError):
        validate_scenario_ids(("tennent", "invalid"))


def test_validate_scenario_ids_rejects_empty() -> None:
    with pytest.raises(ValueError):
        validate_scenario_ids(())


def test_validate_positive_float_rejects_zero() -> None:
    with pytest.raises(ValueError):
        validate_positive_float("budget", 0.0)


def test_validate_positive_float_rejects_negative() -> None:
    with pytest.raises(ValueError):
        validate_positive_float("budget", -1.0)


def test_validate_positive_int_rejects_zero() -> None:
    with pytest.raises(ValueError):
        validate_positive_int("max_collects", 0)


def test_validate_positive_int_rejects_negative() -> None:
    with pytest.raises(ValueError):
        validate_positive_int("max_collects", -1)


# ---------------------------------------------------------------------------
# Request ID
# ---------------------------------------------------------------------------


def test_make_request_id_deterministic() -> None:
    a = _make_request_id("/x", {"k": 1, "v": "abc"})
    b = _make_request_id("/x", {"v": "abc", "k": 1})  # key order should not matter
    assert a == b


def test_make_request_id_differs_for_different_inputs() -> None:
    a = _make_request_id("/x", {"k": 1})
    b = _make_request_id("/x", {"k": 2})
    c = _make_request_id("/y", {"k": 1})
    assert a != b
    assert a != c


# ---------------------------------------------------------------------------
# ApiResponse / response_to_dict
# ---------------------------------------------------------------------------


def test_api_response_is_frozen() -> None:
    r = ApiResponse(
        request_id="x", generated_at="2026-01-01T00:00:00+00:00",
        endpoint="/x", status="ok", payload={}, caveats=(),
    )
    with pytest.raises(FrozenInstanceError):
        r.status = "error"  # type: ignore[misc]


def test_response_to_dict_is_json_serializable() -> None:
    r = ApiResponse(
        request_id="x", generated_at="2026-01-01T00:00:00+00:00",
        endpoint="/x", status="ok",
        payload={"value": 1, "items": [1, 2]},
        caveats=("a", "b"),
    )
    d = response_to_dict(r)
    encoded = json.dumps(d)
    parsed = json.loads(encoded)
    assert parsed["status"] == "ok"
    assert parsed["caveats"] == ["a", "b"]
