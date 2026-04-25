"""Generate a provenance manifest for a custody run (ADR-0021 Slice 18).

The CLI builds a :class:`custody.provenance.ProvenanceRecord` for a
synthetic fixture run and emits it as text / JSON / Markdown.  The
manifest captures the command, the scenarios, the input fixtures,
the git commit, and the default assumptions / caveats so the
artifact is auditable after the fact.

This script does **not** ingest real data, does **not** start an
HTTP server, and does **not** issue execution authorizations.

Run::

    python scripts/22_provenance_manifest.py --output-kind decision-packet \
        --scenario tennent --format text

    python scripts/22_provenance_manifest.py --output-kind portfolio \
        --scenario both --command 'python scripts/20_portfolio_allocation.py' \
        --input-ref fixture:tests/fixtures/foo.json --format json
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

from custody.provenance import (
    InputReference,
    build_provenance_record,
    file_sha256,
    write_manifest,
)


_DEMO_TIMESTAMP = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _parse_input_ref(raw: str, *, repo_root: Path) -> InputReference:
    """Parse ``role:path[:sha]`` or ``role:path`` into an :class:`InputReference`.

    If only ``role:path`` is given and the file exists under ``repo_root``,
    the sha256 is computed from disk.  Otherwise the sha is left as
    ``None`` (treated as a synthetic / in-memory input).
    """
    parts = raw.split(":", 2)
    if len(parts) < 2:
        raise argparse.ArgumentTypeError(
            f"--input-ref must be 'role:path' or 'role:path:sha', got {raw!r}"
        )
    role = parts[0]
    path = parts[1]
    sha: str | None = parts[2] if len(parts) == 3 else None
    if sha is None:
        candidate = repo_root / path
        if candidate.is_file():
            sha = file_sha256(candidate)
    return InputReference(role=role, path=path, sha256=sha)


def main(argv: list[str] | None = None, *, out: TextIO | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a provenance manifest for a custody run.  "
            "Records command, scenarios, input fixtures, git commit, "
            "and default prototype assumptions / caveats."
        ),
    )
    parser.add_argument(
        "--output-kind",
        dest="output_kind",
        default="decision-packet",
        help="Label for the kind of artifact this run produced.",
    )
    parser.add_argument(
        "--scenario",
        choices=("tennent", "whitsun", "both", "none"),
        default="tennent",
    )
    parser.add_argument(
        "--command",
        default="python scripts/22_provenance_manifest.py",
        help="Command string the run is documenting.",
    )
    parser.add_argument(
        "--input-ref",
        dest="input_refs",
        action="append",
        default=[],
        help="Input fixture reference 'role:path[:sha]'.  May be repeated.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json", "markdown"),
        default="text",
    )
    parser.add_argument(
        "--git-commit",
        dest="git_commit",
        default=None,
        help="Override git commit (default: discovered from git rev-parse HEAD).",
    )
    args = parser.parse_args(argv)

    sink = out if out is not None else sys.stdout

    if args.scenario == "tennent":
        scenario_ids: tuple[str, ...] = ("tennent",)
    elif args.scenario == "whitsun":
        scenario_ids = ("whitsun",)
    elif args.scenario == "both":
        scenario_ids = ("tennent", "whitsun")
    else:
        scenario_ids = ()

    repo_root = Path(__file__).resolve().parents[1]
    inputs: list[InputReference] = []
    for raw in args.input_refs:
        inputs.append(_parse_input_ref(raw, repo_root=repo_root))

    record = build_provenance_record(
        output_kind=args.output_kind,
        command=args.command,
        args=tuple(),  # the run being documented; CLI args are not its args
        scenario_ids=scenario_ids,
        inputs=tuple(inputs),
        git_commit=args.git_commit,
        generated_at=_DEMO_TIMESTAMP,
        repo_root=repo_root,
    )
    write_manifest(record, fmt=args.format, sink=sink)
    return 0


if __name__ == "__main__":
    sys.exit(main())
