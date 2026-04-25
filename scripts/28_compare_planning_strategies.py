"""Baseline planning-strategy comparison CLI (ADR-0021 Slice 24).

Compares multiple planning strategies across scenarios using
deterministic simulation metrics.  Prototype comparison harness only -
not measured production performance, not real planner benchmarking,
not operational reporting.

Run::

    python scripts/28_compare_planning_strategies.py --scenario tennent
    python scripts/28_compare_planning_strategies.py --scenario both
    python scripts/28_compare_planning_strategies.py --scenario both \\
        --outcome-policy favorable --budget 1.5 --max-collects 3
"""
from __future__ import annotations

import argparse
import json as _json
import sys
from pathlib import Path
from typing import TextIO

from custody.hypotheses.baseline_comparison import (
    CrossScenarioComparisonReport,
    StrategyComparisonReport,
    compare_planning_strategies,
    compare_planning_strategies_across_scenarios,
    format_comparison_markdown,
    format_comparison_text,
    format_cross_scenario_markdown,
    format_cross_scenario_text,
    report_to_dict,
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

_DEFAULT_WINDOWS: dict[str, Path] = {
    "tennent": (
        REPO_ROOT / "tests" / "fixtures" / "schedule"
        / "tennent_collection_windows.json"
    ),
    "whitsun": (
        REPO_ROOT / "tests" / "fixtures" / "schedule"
        / "whitsun_collection_windows.json"
    ),
}


def _parse_strategies(value: str | None) -> tuple[str, ...] | None:
    if value is None or value == "":
        return None
    return tuple(s.strip() for s in value.split(",") if s.strip())


def main(argv: list[str] | None = None, *, out: TextIO | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Baseline planning-strategy comparison harness: evaluates "
            "manual proxy, collection-value-only, mission-value optimized, "
            "availability-adjusted, scheduler-lite, and execution-feedback "
            "strategies using deterministic prototype metrics.  Prototype "
            "comparison only - no production performance measurement, "
            "no live tasking, no real revenue."
        ),
    )
    parser.add_argument(
        "--scenario", choices=("tennent", "whitsun", "both"), default="both",
    )
    parser.add_argument(
        "--outcome-policy", dest="outcome_policy",
        choices=("favorable", "inconclusive", "adverse", "mixed"),
        default="favorable",
    )
    parser.add_argument("--budget", type=float, default=1.5)
    parser.add_argument(
        "--max-collects", dest="max_collects", type=int, default=3,
    )
    parser.add_argument(
        "--strategies", default=None,
        help=(
            "Comma-separated list of strategy IDs to evaluate; "
            "default is all six."
        ),
    )
    parser.add_argument(
        "--format", choices=("text", "json", "md"), default="text",
    )
    args = parser.parse_args(argv)

    sink = out if out is not None else sys.stdout

    strategies = _parse_strategies(args.strategies)

    report: StrategyComparisonReport | CrossScenarioComparisonReport
    if args.scenario in ("tennent", "whitsun"):
        report = compare_planning_strategies(
            scenario_id=args.scenario,
            outcome_policy=args.outcome_policy,
            budget=args.budget,
            max_collects=args.max_collects,
            strategies=strategies,
            artifact_manifest_path=_DEFAULT_ARTIFACT_MANIFESTS[args.scenario],
            availability_catalog_path=_DEFAULT_CATALOGS[args.scenario],
            collection_windows_path=_DEFAULT_WINDOWS[args.scenario],
        )
    else:
        report = compare_planning_strategies_across_scenarios(
            scenario_ids=("tennent", "whitsun"),
            outcome_policy=args.outcome_policy,
            budget=args.budget,
            max_collects=args.max_collects,
            strategies=strategies,
            artifact_manifest_paths=_DEFAULT_ARTIFACT_MANIFESTS,
            availability_catalog_paths=_DEFAULT_CATALOGS,
            collection_windows_paths=_DEFAULT_WINDOWS,
        )

    if args.format == "json":
        sink.write(_json.dumps(report_to_dict(report), indent=2))
        sink.write("\n")
    elif args.format == "md":
        if isinstance(report, CrossScenarioComparisonReport):
            sink.write(format_cross_scenario_markdown(report))
        else:
            sink.write(format_comparison_markdown(report))
    else:
        if isinstance(report, CrossScenarioComparisonReport):
            sink.write(format_cross_scenario_text(report))
        else:
            sink.write(format_comparison_text(report))

    return 0


if __name__ == "__main__":
    sys.exit(main())
