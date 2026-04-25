"""Availability-adjusted collection-plan optimizer CLI (ADR-0021 Slice 21).

Composes the base optimized plan (Slice 11) with scene-availability
feasibility (Slice 20) to produce a plan that accounts for metadata-
derived feasibility of candidate collect types.

This is decision support only.  No imagery is downloaded, no external
APIs are fetched, no execution authorizations are issued.

Run::

    python scripts/25_availability_optimized_plan.py --scenario tennent
    python scripts/25_availability_optimized_plan.py --scenario whitsun
    python scripts/25_availability_optimized_plan.py --scenario both
    python scripts/25_availability_optimized_plan.py --scenario both \\
        --budget 1.5 --max-collects 3
"""
from __future__ import annotations

import argparse
import json as _json
import sys
from pathlib import Path
from typing import TextIO

from custody.hypotheses.artifacts import (
    build_decision_from_artifacts,
    load_artifact_manifest,
)
from custody.hypotheses.availability_optimizer import (
    AvailabilityOptimizationReport,
    AvailabilityOptimizerConstraint,
    format_availability_optimizer_markdown,
    format_availability_optimizer_text,
    optimize_availability_adjusted_plan,
    report_to_json_object,
)
from custody.hypotheses.counterfactual import simulate_counterfactual_collects
from custody.hypotheses.mission_value import attribute_mission_value
from custody.hypotheses.optimizer import optimize_collection_plan
from custody.hypotheses.scene_availability import (
    adjust_recommendation_for_availability,
    load_scene_availability_catalog,
)


REPO_ROOT = Path(__file__).resolve().parents[1]

_DEFAULT_ARTIFACT_MANIFESTS: dict[str, Path] = {
    "tennent": (
        REPO_ROOT / "tests" / "fixtures" / "artifacts"
        / "tennent_artifact_manifest.json"
    ),
    "whitsun": (
        REPO_ROOT / "tests" / "fixtures" / "artifacts"
        / "whitsun_artifact_manifest.json"
    ),
}

_DEFAULT_CATALOGS: dict[str, Path] = {
    "tennent": (
        REPO_ROOT / "tests" / "fixtures" / "availability"
        / "tennent_scene_availability.json"
    ),
    "whitsun": (
        REPO_ROOT / "tests" / "fixtures" / "availability"
        / "whitsun_scene_availability.json"
    ),
}


def _scenarios_for(arg: str) -> tuple[str, ...]:
    if arg == "tennent":
        return ("tennent",)
    if arg == "whitsun":
        return ("whitsun",)
    return ("tennent", "whitsun")


def _build_report(
    scenario_id: str,
    constraint: AvailabilityOptimizerConstraint,
) -> AvailabilityOptimizationReport:
    artifact_path = _DEFAULT_ARTIFACT_MANIFESTS[scenario_id]
    catalog_path = _DEFAULT_CATALOGS[scenario_id]

    records = load_artifact_manifest(artifact_path)
    decision = build_decision_from_artifacts(records, scenario_id=scenario_id)

    mission_value = attribute_mission_value(
        decision.recommendation, decision.health,
    )
    counterfactual = simulate_counterfactual_collects(
        decision.final_state, decision.health, decision.recommendation,
    )
    base_optimization = optimize_collection_plan(
        decision.recommendation,
        mission_value,
        counterfactual,
        scenario_id=scenario_id,
    )

    catalog = load_scene_availability_catalog(
        catalog_path, scenario_id=scenario_id,
    )
    availability = adjust_recommendation_for_availability(
        decision.recommendation, catalog,
    )

    return optimize_availability_adjusted_plan(
        base_optimization, availability,
        scenario_id=scenario_id, constraint=constraint,
    )


def main(argv: list[str] | None = None, *, out: TextIO | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Availability-adjusted collection-plan optimizer: composes the "
            "base optimized plan with scene-availability feasibility.  "
            "Decision support only - no imagery, no external API fetch, "
            "no execution authorization."
        ),
    )
    parser.add_argument(
        "--scenario", choices=("tennent", "whitsun", "both"), default="both",
    )
    parser.add_argument("--budget", type=float, default=1.0)
    parser.add_argument(
        "--max-collects", dest="max_collects", type=int, default=2,
    )
    parser.add_argument(
        "--min-feasibility-score", dest="min_feasibility_score",
        type=float, default=0.0,
    )
    parser.add_argument(
        "--include-unavailable", dest="include_unavailable",
        action="store_true",
    )
    parser.add_argument(
        "--format", choices=("text", "json", "md"), default="text",
    )
    args = parser.parse_args(argv)

    sink = out if out is not None else sys.stdout

    constraint = AvailabilityOptimizerConstraint(
        budget=args.budget,
        max_collects=args.max_collects,
        min_feasibility_score=args.min_feasibility_score,
        exclude_unavailable=not args.include_unavailable,
    )

    scenarios = _scenarios_for(args.scenario)
    reports: list[AvailabilityOptimizationReport] = [
        _build_report(sid, constraint) for sid in scenarios
    ]

    if args.format == "json":
        if len(reports) == 1:
            sink.write(_json.dumps(report_to_json_object(reports[0]), indent=2))
            sink.write("\n")
        else:
            sink.write(_json.dumps(
                [report_to_json_object(r) for r in reports], indent=2,
            ))
            sink.write("\n")
    elif args.format == "md":
        for i, report in enumerate(reports):
            if i > 0:
                sink.write("\n")
            sink.write(format_availability_optimizer_markdown(report))
    else:
        for i, report in enumerate(reports):
            if i > 0:
                sink.write("\n")
            sink.write(format_availability_optimizer_text(report))

    return 0


if __name__ == "__main__":
    sys.exit(main())
