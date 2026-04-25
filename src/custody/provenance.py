"""Run-provenance records (ADR-0021 Slice 18).

A provenance record captures *how* a deterministic decision-support
artifact was produced: which command and arguments, which scenarios,
which input fixtures (path + sha256), which git commit, and the
default assumptions / caveats that bound the prototype.  The record
is intended to be attached to API responses and CLI manifests so the
output is auditable after the fact.

This module is **standard-library only**.  It does not perform any
network I/O, does not call into ``custody.detection`` /
``custody.ingest.gfw_presence`` / ``custody.fusion.tracker``, and
does not import Sentinel SDKs.

Design constraints
------------------

- Pure stdlib (``hashlib``, ``json``, ``subprocess`` for ``git rev-parse``).
- Frozen dataclasses for value types.
- Deterministic ``stable_run_id`` keyed on command + args + inputs +
  scenarios + git commit (no wall-clock).
- ``get_git_commit`` returns ``"unknown"`` on failure (no exceptions
  leak to callers).
"""
from __future__ import annotations

import hashlib
import json as _json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO


# ---------------------------------------------------------------------------
# Default assumptions / caveats (worded to honour the language guardrails)
# ---------------------------------------------------------------------------


_DEFAULT_ASSUMPTIONS: tuple[str, ...] = (
    "synthetic Tennent / Whitsun fixture narratives only; no real "
    "SAR or AIS ingestion",
    "scenario dates and acquisition times are deterministic fixtures",
    "candidate collect types are sensor-generic prototype proxies",
    "planning utility and mission value are deterministic proxies, "
    "not financial or operational estimates",
)

_DEFAULT_CAVEATS: tuple[str, ...] = (
    "local prototype only; not deployed, not authenticated, "
    "not integrated with any external planning system",
    "decision-support output only; no execution authorization is issued",
    "candidate collect types are prototype recommendations, not "
    "execution authorizations",
    "provenance manifests describe the synthetic fixture run, not a "
    "real-data lineage",
)


# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InputReference:
    """Reference to an input artifact contributing to a run.

    ``role`` is a short label (``"fixture"``, ``"narrative"``,
    ``"config"``).  ``path`` is a string (relative paths are
    preferred).  ``sha256`` is the file's content hash, or ``None``
    if the input is synthetic (in-memory) and has no stable hash.
    """

    role: str
    path: str
    sha256: str | None = None
    description: str | None = None


@dataclass(frozen=True)
class ProvenanceRecord:
    """Auditable record of how a deterministic artifact was produced."""

    run_id: str
    output_kind: str
    command: str
    args: tuple[str, ...]
    scenario_ids: tuple[str, ...]
    inputs: tuple[InputReference, ...]
    git_commit: str
    generated_at: str
    assumptions: tuple[str, ...] = field(default_factory=lambda: _DEFAULT_ASSUMPTIONS)
    caveats: tuple[str, ...] = field(default_factory=lambda: _DEFAULT_CAVEATS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_git_commit(repo_root: Path | None = None) -> str:
    """Return the current ``HEAD`` commit hash, or ``"unknown"``.

    Never raises.  ``"unknown"`` is returned if ``git`` is unavailable,
    if the working tree is not a repository, or if the subprocess
    times out.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root) if repo_root is not None else None,
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    sha = result.stdout.strip()
    if not sha:
        return "unknown"
    return sha


def file_sha256(path: Path) -> str:
    """Return the sha256 of a file's contents (hex digest)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_run_id(
    *,
    command: str,
    args: tuple[str, ...],
    scenario_ids: tuple[str, ...],
    inputs: tuple[InputReference, ...],
    git_commit: str,
) -> str:
    """Deterministic run ID derived from invocation + inputs + commit.

    Wall-clock time is *not* part of the key, so two identical runs
    produce the same ID.  Returned as a 16-char hex prefix of sha256.
    """
    blob = _json.dumps(
        {
            "command": command,
            "args": list(args),
            "scenario_ids": list(scenario_ids),
            "inputs": [
                {"role": ir.role, "path": ir.path, "sha256": ir.sha256}
                for ir in inputs
            ],
            "git_commit": git_commit,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _generated_at_str(generated_at: datetime | None) -> str:
    dt = generated_at if generated_at is not None else datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def build_provenance_record(
    *,
    output_kind: str,
    command: str,
    args: tuple[str, ...] = (),
    scenario_ids: tuple[str, ...] = (),
    inputs: tuple[InputReference, ...] = (),
    git_commit: str | None = None,
    generated_at: datetime | None = None,
    assumptions: tuple[str, ...] | None = None,
    caveats: tuple[str, ...] | None = None,
    repo_root: Path | None = None,
) -> ProvenanceRecord:
    """Construct a :class:`ProvenanceRecord` with sensible defaults."""
    commit = git_commit if git_commit is not None else get_git_commit(repo_root)
    rid = stable_run_id(
        command=command,
        args=args,
        scenario_ids=scenario_ids,
        inputs=inputs,
        git_commit=commit,
    )
    return ProvenanceRecord(
        run_id=rid,
        output_kind=output_kind,
        command=command,
        args=tuple(args),
        scenario_ids=tuple(scenario_ids),
        inputs=tuple(inputs),
        git_commit=commit,
        generated_at=_generated_at_str(generated_at),
        assumptions=(
            tuple(assumptions) if assumptions is not None else _DEFAULT_ASSUMPTIONS
        ),
        caveats=(
            tuple(caveats) if caveats is not None else _DEFAULT_CAVEATS
        ),
    )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _input_to_dict(ir: InputReference) -> dict:
    return {
        "role": ir.role,
        "path": ir.path,
        "sha256": ir.sha256,
        "description": ir.description,
    }


def record_to_dict(record: ProvenanceRecord) -> dict:
    """Render a :class:`ProvenanceRecord` as a JSON-serializable dict."""
    return {
        "run_id": record.run_id,
        "output_kind": record.output_kind,
        "command": record.command,
        "args": list(record.args),
        "scenario_ids": list(record.scenario_ids),
        "inputs": [_input_to_dict(ir) for ir in record.inputs],
        "git_commit": record.git_commit,
        "generated_at": record.generated_at,
        "assumptions": list(record.assumptions),
        "caveats": list(record.caveats),
    }


def attach_provenance(response: dict, record: ProvenanceRecord) -> dict:
    """Return a copy of ``response`` with a top-level ``"provenance"`` key.

    Any pre-existing ``"provenance"`` key is overwritten.  Other keys
    in ``response`` are preserved verbatim (no re-ordering relied
    on).
    """
    out = dict(response)
    out["provenance"] = record_to_dict(record)
    return out


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def format_text(record: ProvenanceRecord) -> str:
    """Render a record as plain text (one field per line)."""
    lines: list[str] = []
    lines.append(f"run_id:       {record.run_id}")
    lines.append(f"output_kind:  {record.output_kind}")
    lines.append(f"command:      {record.command}")
    if record.args:
        lines.append(f"args:         {' '.join(record.args)}")
    if record.scenario_ids:
        lines.append(f"scenarios:    {', '.join(record.scenario_ids)}")
    lines.append(f"git_commit:   {record.git_commit}")
    lines.append(f"generated_at: {record.generated_at}")
    if record.inputs:
        lines.append("inputs:")
        for ir in record.inputs:
            sha = ir.sha256 if ir.sha256 is not None else "<synthetic>"
            lines.append(f"  - [{ir.role}] {ir.path} ({sha})")
    if record.assumptions:
        lines.append("assumptions:")
        for a in record.assumptions:
            lines.append(f"  - {a}")
    if record.caveats:
        lines.append("caveats:")
        for c in record.caveats:
            lines.append(f"  - {c}")
    return "\n".join(lines) + "\n"


def format_json(record: ProvenanceRecord, *, indent: int = 2) -> str:
    """Render a record as a JSON document."""
    return _json.dumps(record_to_dict(record), indent=indent) + "\n"


def format_markdown(record: ProvenanceRecord) -> str:
    """Render a record as a Markdown manifest."""
    lines: list[str] = []
    lines.append(f"# Provenance — {record.output_kind}")
    lines.append("")
    lines.append(f"- **run_id**: `{record.run_id}`")
    lines.append(f"- **command**: `{record.command}`")
    if record.args:
        lines.append(f"- **args**: `{' '.join(record.args)}`")
    if record.scenario_ids:
        lines.append(f"- **scenarios**: {', '.join(record.scenario_ids)}")
    lines.append(f"- **git_commit**: `{record.git_commit}`")
    lines.append(f"- **generated_at**: {record.generated_at}")
    lines.append("")
    if record.inputs:
        lines.append("## Inputs")
        lines.append("")
        for ir in record.inputs:
            sha = ir.sha256 if ir.sha256 is not None else "_synthetic_"
            desc = f" — {ir.description}" if ir.description else ""
            lines.append(f"- [{ir.role}] `{ir.path}` ({sha}){desc}")
        lines.append("")
    if record.assumptions:
        lines.append("## Assumptions")
        lines.append("")
        for a in record.assumptions:
            lines.append(f"- {a}")
        lines.append("")
    if record.caveats:
        lines.append("## Caveats")
        lines.append("")
        for c in record.caveats:
            lines.append(f"- {c}")
        lines.append("")
    return "\n".join(lines)


def write_manifest(record: ProvenanceRecord, *, fmt: str, sink: TextIO) -> None:
    """Render and write a record to ``sink``."""
    if fmt == "text":
        sink.write(format_text(record))
    elif fmt == "json":
        sink.write(format_json(record))
    elif fmt == "markdown":
        sink.write(format_markdown(record))
    else:
        raise ValueError(
            f"unknown format: {fmt!r}; must be one of 'text', 'json', 'markdown'"
        )
