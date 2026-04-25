"""Scene-availability metadata bridge CLI (ADR-0021 Slice 20).

Loads scene-availability metadata catalogs, assesses feasibility of
candidate collect types against the existing collection-value
ranking, and emits an availability-adjusted recommendation.

This is **metadata-only**.  No imagery is downloaded, no external
APIs are fetched, no Sentinel data is ingested, and no execution
authorizations are issued.

Run::

    python scripts/24_scene_availability.py --scenario tennent
    python scripts/24_scene_availability.py --scenario whitsun
    python scripts/24_scene_availability.py --scenario both
    python scripts/24_scene_availability.py --scenario tennent \
        --catalog tests/fixtures/availability/tennent_scene_availability.json
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
from custody.hypotheses.scene_availability import (
    AvailabilityAdjustedRecommendation,
    adjust_recommendation_for_availability,
    format_availability_markdown,
    format_availability_text,
    load_scene_availability_catalog,
    recommendation_to_dict,
)


REPO_ROOT = Path(__file__).resolve().parents[1]

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


def _scenarios_for(arg: str) -> tuple[str, ...]:
    if arg == "tennent":
        return ("tennent",)
    if arg == "whitsun":
        return ("whitsun",)
    return ("tennent", "whitsun")


def _build_report(
    scenario_id: str, catalog_path: Path,
) -> tuple[AvailabilityAdjustedRecommendation, Path]:
    artifact_manifest = _DEFAULT_ARTIFACT_MANIFESTS[scenario_id]
    records = load_artifact_manifest(artifact_manifest)
    decision = build_decision_from_artifacts(records, scenario_id=scenario_id)
    catalog = load_scene_availability_catalog(
        catalog_path, scenario_id=scenario_id,
    )
    report = adjust_recommendation_for_availability(
        decision.recommendation, catalog,
    )
    return report, catalog_path


def main(argv: list[str] | None = None, *, out: TextIO | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Scene-availability metadata bridge: assesses candidate collect "
            "feasibility from provider-neutral metadata catalogs.  No "
            "imagery, no external API fetch, no execution authorization."
        ),
    )
    parser.add_argument(
        "--scenario", choices=("tennent", "whitsun", "both"), default="both",
    )
    parser.add_argument(
        "--catalog", default=None,
        help="Path to scene availability catalog JSON (overrides default).",
    )
    parser.add_argument(
        "--format", choices=("text", "json", "md"), default="text",
    )
    args = parser.parse_args(argv)

    sink = out if out is not None else sys.stdout

    scenarios = _scenarios_for(args.scenario)

    reports: list[tuple[AvailabilityAdjustedRecommendation, Path]] = []
    for sid in scenarios:
        catalog_path = (
            Path(args.catalog) if args.catalog is not None
            else _DEFAULT_CATALOGS[sid]
        )
        report, path = _build_report(sid, catalog_path)
        reports.append((report, path))

    if args.format == "json":
        if len(reports) == 1:
            sink.write(_json.dumps(recommendation_to_dict(reports[0][0]), indent=2))
            sink.write("\n")
        else:
            sink.write(_json.dumps(
                [recommendation_to_dict(r) for r, _ in reports], indent=2,
            ))
            sink.write("\n")
    elif args.format == "md":
        for i, (report, path) in enumerate(reports):
            if i > 0:
                sink.write("\n")
            sink.write(format_availability_markdown(
                report, catalog_path=str(path),
            ))
    else:
        for i, (report, path) in enumerate(reports):
            if i > 0:
                sink.write("\n")
            sink.write(format_availability_text(
                report, catalog_path=str(path),
            ))

    return 0


if __name__ == "__main__":
    sys.exit(main())
