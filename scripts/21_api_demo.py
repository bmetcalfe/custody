"""Local decision API demo CLI (ADR-0021 Slice 17).

Demonstrates the Custody decision API service contract by calling
service functions directly and printing JSON-serializable responses.
**No HTTP server is started.**

This is a local prototype demo.  All output is generated from the
synthetic Tennent / Whitsun narratives (the same fixtures used by
scripts 13-20).

Run::

    python scripts/21_api_demo.py --endpoint health
    python scripts/21_api_demo.py --endpoint decision-packet --scenario tennent
    python scripts/21_api_demo.py --endpoint optimize-plan --scenario whitsun --budget 1.0
    python scripts/21_api_demo.py --endpoint portfolio-allocation --scenario both --budget 1.5
    python scripts/21_api_demo.py --endpoint planner-queue --scenario both --format json
"""
from __future__ import annotations

import argparse
import json as _json
import sys
from datetime import datetime, timezone
from typing import TextIO

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


# ---------------------------------------------------------------------------
# Fixed deterministic timestamp
# ---------------------------------------------------------------------------


_DEMO_TIMESTAMP = datetime(2026, 1, 1, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Endpoint dispatch
# ---------------------------------------------------------------------------


def _scenario_ids_from_arg(value: str) -> tuple[str, ...]:
    if value == "tennent":
        return ("tennent",)
    if value == "whitsun":
        return ("whitsun",)
    return ("tennent", "whitsun")


def _dispatch_single(args) -> list[dict]:
    """Endpoints that take a single scenario_id; --scenario both -> two responses."""
    scenarios: tuple[str, ...]
    if args.scenario == "both":
        scenarios = ("tennent", "whitsun")
    else:
        scenarios = (args.scenario,)
    out: list[dict] = []
    for sid in scenarios:
        if args.endpoint == "decision-packet":
            out.append(build_decision_packet_response(
                DecisionPacketRequest(
                    scenario_id=sid,
                    include_mission_value=args.include_mission_value,
                ),
                generated_at=_DEMO_TIMESTAMP,
            ))
        elif args.endpoint == "rank-collects":
            out.append(build_collect_ranking_response(
                CollectRankingRequest(scenario_id=sid),
                generated_at=_DEMO_TIMESTAMP,
            ))
        elif args.endpoint == "optimize-plan":
            out.append(build_optimize_plan_response(
                OptimizePlanRequest(
                    scenario_id=sid,
                    budget=args.budget,
                    max_collects=args.max_collects,
                ),
                generated_at=_DEMO_TIMESTAMP,
            ))
        elif args.endpoint == "evaluate-policies":
            out.append(build_policy_evaluation_response(
                PolicyEvaluationRequest(
                    scenario_id=sid,
                    budget=args.budget,
                    max_collects=args.max_collects,
                ),
                generated_at=_DEMO_TIMESTAMP,
            ))
    return out


def _dispatch_multi(args) -> list[dict]:
    """Endpoints that natively take a list of scenarios."""
    scenarios = _scenario_ids_from_arg(args.scenario)
    if args.endpoint == "planner-queue":
        return [build_planner_queue_response(
            PlannerQueueRequest(scenario_ids=scenarios),
            generated_at=_DEMO_TIMESTAMP,
        )]
    if args.endpoint == "portfolio-allocation":
        return [build_portfolio_allocation_response(
            PortfolioAllocationRequest(
                scenario_ids=scenarios,
                budget=args.budget,
                max_collects=args.max_collects,
                max_collects_per_scenario=args.max_collects_per_scenario,
            ),
            generated_at=_DEMO_TIMESTAMP,
        )]
    return []


# ---------------------------------------------------------------------------
# Pretty-printer (avoids dumping the full payload)
# ---------------------------------------------------------------------------


def _render_pretty(out: TextIO, response: dict) -> None:
    out.write(f"Endpoint: {response['endpoint']}\n")
    out.write(f"Status:   {response['status']}\n")
    out.write(f"Request:  {response['request_id']}\n")
    out.write(f"At:       {response['generated_at']}\n")
    payload = response.get("payload", {})
    if response["endpoint"] == "/health":
        out.write(f"Service:  {payload.get('service')}\n")
        out.write(f"Scenarios: {payload.get('scenarios_available')}\n")
    elif response["endpoint"] == "/decision-packet":
        out.write(f"Scenario: {payload.get('scenario_id')}\n")
        cb = payload.get("current_belief", [])[:3]
        for entry in cb:
            out.write(
                f"  belief: {entry.get('hypothesis_id')} "
                f"({entry.get('score'):.3f})\n"
            )
        ch = payload.get("custody_health", {})
        if ch:
            out.write(
                f"  custody: {ch.get('status')} score={ch.get('score'):.2f}\n"
            )
    elif response["endpoint"] == "/rank-collects":
        out.write(f"Scenario: {payload.get('scenario_id')}\n")
        for cv in payload.get("ranked_values", [])[:3]:
            out.write(
                f"  {cv['candidate_id']:24s} score={cv['score']:.2f}\n"
            )
    elif response["endpoint"] == "/optimize-plan":
        p = payload
        out.write(f"Strategy: {p.get('strategy')}\n")
        out.write(
            f"Cost:     {p.get('total_cost'):.2f} / "
            f"{p['constraint']['budget']:.2f}\n"
        )
        out.write(f"Utility:  {p.get('total_value'):.2f}\n")
        for item in p.get("selected_items", []):
            out.write(
                f"  {item['candidate_id']:24s} cost={item['cost']:.2f} "
                f"value={item['value']:.2f}\n"
            )
    elif response["endpoint"] == "/evaluate-policies":
        out.write(f"Winner: {payload.get('winning_policy_id')}\n")
        for e in payload.get("evaluations", [])[:3]:
            out.write(
                f"  rank {e['rank']} {e['policy_id']:30s} "
                f"utility={e['planning_utility']:.2f}\n"
            )
    elif response["endpoint"] == "/planner-queue":
        for item in payload.get("items", []):
            out.write(
                f"  {item['scenario_id']:8s} {item['status']:16s} "
                f"priority={item['priority_score']:.2f}\n"
            )
    elif response["endpoint"] == "/portfolio-allocation":
        plan = payload.get("recommended_plan", {})
        out.write(f"Strategy: {plan.get('strategy')}\n")
        out.write(
            f"Cost:     {plan.get('total_cost'):.2f} / "
            f"{plan['constraint']['budget']:.2f}\n"
        )
        out.write(
            f"Score:    {plan.get('total_portfolio_score'):.2f}\n"
        )
        for item in plan.get("selected_items", []):
            out.write(
                f"  {item['scenario_id']:8s} / "
                f"{item['candidate_id']:24s} cost={item['cost']:.2f} "
                f"score={item['portfolio_score']:.2f}\n"
            )
    out.write("\nCaveats:\n")
    for c in response.get("caveats", []):
        out.write(f"  - {c}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None, *, out: TextIO | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Local decision API demo: calls custody.api.service functions "
            "directly and prints JSON-serializable responses.  No HTTP "
            "server is started."
        ),
    )
    parser.add_argument(
        "--endpoint",
        choices=(
            "health",
            "decision-packet",
            "rank-collects",
            "optimize-plan",
            "evaluate-policies",
            "planner-queue",
            "portfolio-allocation",
        ),
        default="health",
    )
    parser.add_argument(
        "--scenario",
        choices=("tennent", "whitsun", "both"),
        default="tennent",
    )
    parser.add_argument("--budget", type=float, default=1.0)
    parser.add_argument(
        "--max-collects", dest="max_collects", type=int, default=2,
    )
    parser.add_argument(
        "--max-collects-per-scenario",
        dest="max_collects_per_scenario", type=int, default=2,
    )
    parser.add_argument(
        "--format", choices=("json", "pretty"), default="pretty",
    )
    parser.add_argument(
        "--include-mission-value",
        dest="include_mission_value", action="store_true",
    )
    args = parser.parse_args(argv)

    sink = out if out is not None else sys.stdout

    if args.endpoint == "health":
        responses = [health_check(generated_at=_DEMO_TIMESTAMP)]
    elif args.endpoint in ("planner-queue", "portfolio-allocation"):
        responses = _dispatch_multi(args)
    else:
        responses = _dispatch_single(args)

    if args.format == "json":
        if len(responses) == 1:
            sink.write(_json.dumps(responses[0], indent=2) + "\n")
        else:
            sink.write(_json.dumps(responses, indent=2) + "\n")
    else:
        for i, r in enumerate(responses):
            if i > 0:
                sink.write("\n")
            _render_pretty(sink, r)

    return 0


if __name__ == "__main__":
    sys.exit(main())
