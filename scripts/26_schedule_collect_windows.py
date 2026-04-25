"""Collection-window scheduler-lite CLI (ADR-0021 Slice 22).

Composes the full decision pipeline (Slices 1-21) with provider-neutral
collection windows to simulate schedule feasibility for candidate
collect types.

This is schedule feasibility simulation only.  No imagery is downloaded,
no external APIs are fetched, no execution authorizations are issued,
and no sensor-control instructions are generated.

Run::

    python scripts/26_schedule_collect_windows.py --scenario tennent
    python scripts/26_schedule_collect_windows.py --scenario whitsun
    python scripts/26_schedule_collect_windows.py --scenario both
    python scripts/26_schedule_collect_windows.py --scenario both \\
        --max-total-capacity 1.5 --max-overlapping-collects 2
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
    AvailabilityOptimizerConstraint,
    optimize_availability_adjusted_plan,
)
from custody.hypotheses.counterfactual import simulate_counterfactual_collects
from custody.hypotheses.mission_value import attribute_mission_value
from custody.hypotheses.optimizer import optimize_collection_plan
from custody.hypotheses.scene_availability import (
    adjust_recommendation_for_availability,
    load_scene_availability_catalog,
)
from custody.hypotheses.scheduler import (
    ScheduleConstraint,
    ScheduleReport,
    format_schedule_markdown,
    format_schedule_text,
    load_collection_windows,
    schedule_collects,
    schedule_report_to_dict,
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


def _scenarios_for(arg: str) -> tuple[str, ...]:
    if arg == "tennent":
        return ("tennent",)
    if arg == "whitsun":
        return ("whitsun",)
    return ("tennent", "whitsun")


def _build_report(
    scenario_id: str,
    windows_path: Path | None,
    constraint: ScheduleConstraint,
) -> ScheduleReport:
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
        decision.recommendation, mission_value, counterfactual,
        scenario_id=scenario_id,
    )
    catalog = load_scene_availability_catalog(
        catalog_path, scenario_id=scenario_id,
    )
    availability = adjust_recommendation_for_availability(
        decision.recommendation, catalog,
    )
    avail_report = optimize_availability_adjusted_plan(
        base_optimization, availability,
        scenario_id=scenario_id,
        constraint=AvailabilityOptimizerConstraint(
            budget=1.0, max_collects=2,
        ),
    )

    wp = windows_path if windows_path is not None else _DEFAULT_WINDOWS[scenario_id]
    windows = load_collection_windows(wp)

    return schedule_collects(
        avail_report, windows,
        scenario_id=scenario_id, constraint=constraint,
    )


def main(argv: list[str] | None = None, *, out: TextIO | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Collection-window scheduler-lite: simulates placing "
            "availability-adjusted candidate collect types into "
            "provider-neutral windows.  No imagery, no external API "
            "fetch, no execution authorization."
        ),
    )
    parser.add_argument(
        "--scenario", choices=("tennent", "whitsun", "both"), default="both",
    )
    parser.add_argument(
        "--windows", default=None,
        help="Path to collection-windows JSON (overrides default fixture).",
    )
    parser.add_argument(
        "--max-total-capacity", dest="max_total_capacity",
        type=float, default=1.0,
    )
    parser.add_argument(
        "--max-overlapping-collects", dest="max_overlapping_collects",
        type=int, default=1,
    )
    parser.add_argument(
        "--format", choices=("text", "json", "md"), default="text",
    )
    args = parser.parse_args(argv)

    sink = out if out is not None else sys.stdout

    constraint = ScheduleConstraint(
        max_total_capacity=args.max_total_capacity,
        max_overlapping_collects=args.max_overlapping_collects,
    )

    windows_path = Path(args.windows) if args.windows is not None else None

    scenarios = _scenarios_for(args.scenario)
    reports: list[ScheduleReport] = [
        _build_report(sid, windows_path, constraint) for sid in scenarios
    ]

    if args.format == "json":
        if len(reports) == 1:
            sink.write(_json.dumps(schedule_report_to_dict(reports[0]), indent=2))
            sink.write("\n")
        else:
            sink.write(_json.dumps(
                [schedule_report_to_dict(r) for r in reports], indent=2,
            ))
            sink.write("\n")
    elif args.format == "md":
        for i, report in enumerate(reports):
            if i > 0:
                sink.write("\n")
            sink.write(format_schedule_markdown(report))
    else:
        for i, report in enumerate(reports):
            if i > 0:
                sink.write("\n")
            sink.write(format_schedule_text(report))

    return 0


if __name__ == "__main__":
    sys.exit(main())
