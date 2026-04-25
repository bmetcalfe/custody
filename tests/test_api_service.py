"""Tests for :mod:`custody.api.service` (ADR-0021 Slice 17).

Each service function is exercised on the synthetic Tennent / Whitsun
narratives.  All requests pass an explicit ``generated_at`` to keep
output deterministic.  Scope guardrails (no forbidden imports, no
forbidden language, JSON serializability) are enforced.
"""
from __future__ import annotations

import ast
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from custody.api.schemas import (
    CollectRankingRequest,
    DecisionPacketRequest,
    OptimizePlanRequest,
    PlannerQueueRequest,
    PolicyEvaluationRequest,
    PortfolioAllocationRequest,
)
from custody.api.service import (
    build_collect_ranking_response,
    build_decision_packet_response,
    build_optimize_plan_response,
    build_planner_queue_response,
    build_policy_evaluation_response,
    build_portfolio_allocation_response,
    health_check,
)


T_GEN = datetime(2026, 1, 1, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


def test_health_check_returns_ok() -> None:
    r = health_check(generated_at=T_GEN)
    assert r["status"] == "ok"
    assert r["payload"]["service"] == "custody-decision-api"
    assert "tennent" in r["payload"]["scenarios_available"]
    assert "whitsun" in r["payload"]["scenarios_available"]


def test_health_check_caveats_include_prototype() -> None:
    r = health_check(generated_at=T_GEN)
    joined = " ".join(r["caveats"]).lower()
    assert "local prototype api only" in joined


# ---------------------------------------------------------------------------
# Decision packet
# ---------------------------------------------------------------------------


def test_decision_packet_tennent() -> None:
    r = build_decision_packet_response(
        DecisionPacketRequest(scenario_id="tennent"),
        generated_at=T_GEN,
    )
    assert r["status"] == "ok"
    assert r["endpoint"] == "/decision-packet"
    assert r["payload"]["scenario_id"] == "tennent"


def test_decision_packet_whitsun() -> None:
    r = build_decision_packet_response(
        DecisionPacketRequest(scenario_id="whitsun"),
        generated_at=T_GEN,
    )
    assert r["payload"]["scenario_id"] == "whitsun"


def test_decision_packet_with_mission_value() -> None:
    r = build_decision_packet_response(
        DecisionPacketRequest(scenario_id="tennent", include_mission_value=True),
        generated_at=T_GEN,
    )
    assert "mission_value" in r["payload"]


# ---------------------------------------------------------------------------
# Collect ranking
# ---------------------------------------------------------------------------


def test_collect_ranking_includes_candidate_collects() -> None:
    r = build_collect_ranking_response(
        CollectRankingRequest(scenario_id="tennent"),
        generated_at=T_GEN,
    )
    payload = r["payload"]
    assert "ranked_values" in payload
    assert len(payload["ranked_values"]) >= 1
    first = payload["ranked_values"][0]
    assert "candidate_id" in first
    assert "score" in first


# ---------------------------------------------------------------------------
# Optimize plan
# ---------------------------------------------------------------------------


def test_optimize_plan_includes_constraints_and_plan() -> None:
    r = build_optimize_plan_response(
        OptimizePlanRequest(scenario_id="tennent", budget=1.0, max_collects=2),
        generated_at=T_GEN,
    )
    p = r["payload"]
    assert "constraint" in p
    assert p["constraint"]["budget"] == 1.0
    assert "selected_items" in p
    assert "total_cost" in p


def test_optimize_plan_strategy_exhaustive() -> None:
    r = build_optimize_plan_response(
        OptimizePlanRequest(scenario_id="tennent", strategy="exhaustive"),
        generated_at=T_GEN,
    )
    assert r["payload"]["strategy"] == "exhaustive"


# ---------------------------------------------------------------------------
# Policy evaluation
# ---------------------------------------------------------------------------


def test_policy_evaluation_includes_winning_policy() -> None:
    r = build_policy_evaluation_response(
        PolicyEvaluationRequest(scenario_id="tennent"),
        generated_at=T_GEN,
    )
    p = r["payload"]
    assert "winning_policy_id" in p
    assert "evaluations" in p
    assert len(p["evaluations"]) >= 1


# ---------------------------------------------------------------------------
# Planner queue
# ---------------------------------------------------------------------------


def test_planner_queue_includes_ranked_items() -> None:
    r = build_planner_queue_response(
        PlannerQueueRequest(scenario_ids=("tennent", "whitsun")),
        generated_at=T_GEN,
    )
    p = r["payload"]
    assert "items" in p
    scenarios = {item["scenario_id"] for item in p["items"]}
    assert scenarios == {"tennent", "whitsun"}


# ---------------------------------------------------------------------------
# Portfolio allocation
# ---------------------------------------------------------------------------


def test_portfolio_allocation_includes_recommended_portfolio() -> None:
    r = build_portfolio_allocation_response(
        PortfolioAllocationRequest(),
        generated_at=T_GEN,
    )
    p = r["payload"]
    assert "recommended_plan" in p
    assert "candidate_pool" in p
    assert "scenario_ids" in p


# ---------------------------------------------------------------------------
# Validation failures
# ---------------------------------------------------------------------------


def test_invalid_scenario_raises_valueerror() -> None:
    with pytest.raises(ValueError):
        build_decision_packet_response(
            DecisionPacketRequest(scenario_id="invalid"),
            generated_at=T_GEN,
        )


def test_invalid_budget_raises_valueerror() -> None:
    with pytest.raises(ValueError):
        build_optimize_plan_response(
            OptimizePlanRequest(scenario_id="tennent", budget=-1.0),
            generated_at=T_GEN,
        )


def test_invalid_max_collects_raises_valueerror() -> None:
    with pytest.raises(ValueError):
        build_optimize_plan_response(
            OptimizePlanRequest(scenario_id="tennent", max_collects=0),
            generated_at=T_GEN,
        )


# ---------------------------------------------------------------------------
# Cross-endpoint invariants
# ---------------------------------------------------------------------------


def _all_responses() -> list[dict]:
    return [
        health_check(generated_at=T_GEN),
        build_decision_packet_response(
            DecisionPacketRequest(scenario_id="tennent"), generated_at=T_GEN,
        ),
        build_collect_ranking_response(
            CollectRankingRequest(scenario_id="tennent"), generated_at=T_GEN,
        ),
        build_optimize_plan_response(
            OptimizePlanRequest(scenario_id="tennent"), generated_at=T_GEN,
        ),
        build_policy_evaluation_response(
            PolicyEvaluationRequest(scenario_id="whitsun"), generated_at=T_GEN,
        ),
        build_planner_queue_response(
            PlannerQueueRequest(), generated_at=T_GEN,
        ),
        build_portfolio_allocation_response(
            PortfolioAllocationRequest(), generated_at=T_GEN,
        ),
    ]


def test_responses_are_json_serializable() -> None:
    for r in _all_responses():
        encoded = json.dumps(r)
        assert json.loads(encoded) == r


def test_generated_at_is_iso_string() -> None:
    for r in _all_responses():
        ts = r["generated_at"]
        # Round-trip via fromisoformat to verify ISO 8601.
        parsed = datetime.fromisoformat(ts)
        assert parsed.tzinfo is not None


def test_request_id_deterministic_for_same_request() -> None:
    a = build_decision_packet_response(
        DecisionPacketRequest(scenario_id="tennent"),
        generated_at=T_GEN,
    )
    b = build_decision_packet_response(
        DecisionPacketRequest(scenario_id="tennent"),
        generated_at=T_GEN,
    )
    assert a["request_id"] == b["request_id"]


# ---------------------------------------------------------------------------
# Forbidden language
# ---------------------------------------------------------------------------


_FORBIDDEN = (
    "tasking order",
    "live tasking",
    "sensor command",
    "production scheduler",
    "autonomous constellation management",
    "real mps control",
    "revenue dollars",
    "sentinel integration",
    "collection order",
    "production api",
    "zero-trust",
)


def test_no_forbidden_language_in_responses() -> None:
    """Forbidden tokens must not appear outside explicit upstream
    non-claim disclaimers.  Slice 13 / 14 caveats use the literal
    bigrams 'live tasking' / 'sensor command' inside an explicit
    'no live tasking or sensor command is issued' disclaimer; that
    phrasing is preserved verbatim via the planner_queue / portfolio
    payloads.  This test masks the non-claim phrasing first, then
    enforces the ban on the remainder."""
    blob = json.dumps(_all_responses()).lower()
    safe = blob.replace(
        "no live tasking or sensor command is issued", ""
    )
    for needle in _FORBIDDEN:
        assert needle not in safe, (
            f"forbidden token {needle!r} appears outside the explicit "
            "non-claim disclaimer in API responses"
        )


def test_source_file_no_forbidden_language() -> None:
    import custody.api.service as mod
    src = Path(mod.__file__).read_text(encoding="utf-8").lower()
    for needle in _FORBIDDEN:
        assert needle not in src, (
            f"forbidden token {needle!r} appears in service.py source"
        )


# ---------------------------------------------------------------------------
# Import-boundary scan
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Provenance integration (Slice 18)
# ---------------------------------------------------------------------------


def test_health_check_response_includes_provenance() -> None:
    r = health_check(generated_at=T_GEN)
    assert "provenance" in r
    prov = r["provenance"]
    assert prov["output_kind"] == "health-check"
    assert "run_id" in prov
    assert "git_commit" in prov


def test_decision_packet_response_includes_provenance() -> None:
    r = build_decision_packet_response(
        DecisionPacketRequest(scenario_id="tennent"), generated_at=T_GEN,
    )
    prov = r["provenance"]
    assert prov["output_kind"] == "decision-packet"
    assert prov["scenario_ids"] == ["tennent"]


def test_collect_ranking_response_includes_provenance() -> None:
    r = build_collect_ranking_response(
        CollectRankingRequest(scenario_id="whitsun"), generated_at=T_GEN,
    )
    assert r["provenance"]["output_kind"] == "collect-ranking"
    assert r["provenance"]["scenario_ids"] == ["whitsun"]


def test_optimize_plan_response_includes_provenance() -> None:
    r = build_optimize_plan_response(
        OptimizePlanRequest(scenario_id="tennent"), generated_at=T_GEN,
    )
    assert r["provenance"]["output_kind"] == "optimize-plan"


def test_policy_evaluation_response_includes_provenance() -> None:
    r = build_policy_evaluation_response(
        PolicyEvaluationRequest(scenario_id="tennent"), generated_at=T_GEN,
    )
    assert r["provenance"]["output_kind"] == "policy-evaluation"


def test_planner_queue_response_includes_provenance() -> None:
    r = build_planner_queue_response(
        PlannerQueueRequest(), generated_at=T_GEN,
    )
    prov = r["provenance"]
    assert prov["output_kind"] == "planner-queue"
    assert set(prov["scenario_ids"]) == {"tennent", "whitsun"}


def test_portfolio_allocation_response_includes_provenance() -> None:
    r = build_portfolio_allocation_response(
        PortfolioAllocationRequest(), generated_at=T_GEN,
    )
    assert r["provenance"]["output_kind"] == "portfolio-allocation"


def test_module_does_not_import_detection_or_gfw() -> None:
    import custody.api.service as mod
    tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
    forbidden_prefixes = (
        "custody.detection",
        "custody.ingest.gfw_presence",
        "sentinelhub",
        "custody.fusion.tracker",
        "custody.taskrecommendation",
    )
    offending: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if any(alias.name.startswith(p) for p in forbidden_prefixes):
                    offending.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod_name = node.module or ""
            if any(mod_name.startswith(p) for p in forbidden_prefixes):
                offending.append(mod_name)
    assert not offending, (
        f"service.py imports forbidden modules: {offending}"
    )
