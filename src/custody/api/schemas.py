"""Request / response schemas for the local decision API (Slice 17).

All schemas are frozen dataclasses returning plain Python types.  No
Pydantic, no FastAPI models -- pure standard library.

This is a prototype service contract.  Schemas do not imply API
stability, SLA guarantees, or versioned backwards compatibility.
"""
from __future__ import annotations

import hashlib
import json as _json
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Request dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecisionPacketRequest:
    scenario_id: str
    include_mission_value: bool = False
    output_format: str = "json"


@dataclass(frozen=True)
class CollectRankingRequest:
    scenario_id: str
    max_results: int | None = None


@dataclass(frozen=True)
class OptimizePlanRequest:
    scenario_id: str
    budget: float = 1.0
    max_collects: int = 2
    strategy: str = "recommended"  # "recommended" | "exhaustive" | "greedy"


@dataclass(frozen=True)
class PolicyEvaluationRequest:
    scenario_id: str
    budget: float = 1.0
    max_collects: int = 2
    policies: tuple[str, ...] | None = None


@dataclass(frozen=True)
class PlannerQueueRequest:
    scenario_ids: tuple[str, ...] = ("tennent", "whitsun")
    include_reviews: bool = False


@dataclass(frozen=True)
class PortfolioAllocationRequest:
    scenario_ids: tuple[str, ...] = ("tennent", "whitsun")
    budget: float = 1.5
    max_collects: int = 3
    max_collects_per_scenario: int = 2


# ---------------------------------------------------------------------------
# Response wrapper
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ApiResponse:
    request_id: str
    generated_at: str        # ISO 8601 UTC string
    endpoint: str
    status: str              # "ok" or "error"
    payload: dict
    caveats: tuple[str, ...]


def response_to_dict(resp: ApiResponse) -> dict:
    """Render an :class:`ApiResponse` as a plain JSON-serializable dict."""
    return {
        "request_id": resp.request_id,
        "generated_at": resp.generated_at,
        "endpoint": resp.endpoint,
        "status": resp.status,
        "payload": resp.payload,
        "caveats": list(resp.caveats),
    }


# ---------------------------------------------------------------------------
# Request ID
# ---------------------------------------------------------------------------


def _make_request_id(endpoint: str, request_dict: dict) -> str:
    """Deterministic request ID from endpoint + request content."""
    blob = _json.dumps(
        {"endpoint": endpoint, **request_dict}, sort_keys=True, default=str,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


_VALID_SCENARIO_IDS = frozenset({"tennent", "whitsun"})


def validate_scenario_id(scenario_id: str) -> None:
    if scenario_id not in _VALID_SCENARIO_IDS:
        raise ValueError(
            f"invalid scenario_id: {scenario_id!r}; must be one of "
            f"{sorted(_VALID_SCENARIO_IDS)}"
        )


def validate_scenario_ids(scenario_ids: tuple[str, ...]) -> None:
    if not scenario_ids:
        raise ValueError("scenario_ids must not be empty")
    for s in scenario_ids:
        validate_scenario_id(s)


def validate_positive_float(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


def validate_positive_int(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
