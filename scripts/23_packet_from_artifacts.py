"""Artifact-to-decision packet CLI (ADR-0021 Slice 19).

Loads artifact manifests, converts them to HypothesisEvidence via the
artifact bridge, and runs the decision pipeline.

This is **artifact wiring**, not new ingestion.  No imagery is
processed, no VLM is run, no matcher is invoked, and no external
data sources are accessed.

Run::

    python scripts/23_packet_from_artifacts.py --scenario tennent
    python scripts/23_packet_from_artifacts.py --scenario whitsun
    python scripts/23_packet_from_artifacts.py --scenario both
    python scripts/23_packet_from_artifacts.py --scenario tennent \
        --manifest tests/fixtures/artifacts/tennent_artifact_manifest.json \
        --format json
"""
from __future__ import annotations

import argparse
import json as _json
import sys
from pathlib import Path
from typing import TextIO

from custody.hypotheses.artifacts import (
    ArtifactDecisionBundle,
    build_decision_from_artifacts,
    bundle_to_dict,
    format_artifact_decision_markdown,
    format_artifact_decision_text,
    load_artifact_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_MANIFESTS: dict[str, Path] = {
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


def _build_bundle(
    scenario_id: str, manifest: Path,
) -> tuple[ArtifactDecisionBundle, Path]:
    records = load_artifact_manifest(manifest)
    bundle = build_decision_from_artifacts(records, scenario_id=scenario_id)
    return bundle, manifest


def main(argv: list[str] | None = None, *, out: TextIO | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Artifact-to-decision packet CLI: loads small JSON artifact "
            "manifests and runs the existing decision pipeline.  No "
            "ingestion, no detection, no matcher run."
        ),
    )
    parser.add_argument(
        "--scenario", choices=("tennent", "whitsun", "both"), default="both",
    )
    parser.add_argument(
        "--manifest", default=None,
        help="Path to artifact manifest JSON (overrides default fixture).",
    )
    parser.add_argument(
        "--format", choices=("text", "json", "md"), default="text",
    )
    args = parser.parse_args(argv)

    sink = out if out is not None else sys.stdout

    scenarios = _scenarios_for(args.scenario)

    bundles: list[tuple[ArtifactDecisionBundle, Path]] = []
    for sid in scenarios:
        if args.manifest is not None:
            manifest_path = Path(args.manifest)
        else:
            manifest_path = _DEFAULT_MANIFESTS[sid]
        bundle, path = _build_bundle(sid, manifest_path)
        bundles.append((bundle, path))

    if args.format == "json":
        if len(bundles) == 1:
            sink.write(_json.dumps(bundle_to_dict(bundles[0][0]), indent=2))
            sink.write("\n")
        else:
            sink.write(_json.dumps(
                [bundle_to_dict(b) for b, _ in bundles], indent=2,
            ))
            sink.write("\n")
    elif args.format == "md":
        for i, (bundle, path) in enumerate(bundles):
            if i > 0:
                sink.write("\n")
            sink.write(format_artifact_decision_markdown(
                bundle, manifest_path=str(path),
            ))
    else:  # text
        for i, (bundle, path) in enumerate(bundles):
            if i > 0:
                sink.write("\n")
            sink.write(format_artifact_decision_text(
                bundle, manifest_path=str(path),
            ))

    return 0


if __name__ == "__main__":
    sys.exit(main())
