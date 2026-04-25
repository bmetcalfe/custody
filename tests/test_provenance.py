"""Tests for :mod:`custody.provenance` (ADR-0021 Slice 18)."""
from __future__ import annotations

import ast
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import pytest

from custody.provenance import (
    InputReference,
    ProvenanceRecord,
    attach_provenance,
    build_provenance_record,
    file_sha256,
    format_json,
    format_markdown,
    format_text,
    get_git_commit,
    record_to_dict,
    stable_run_id,
    write_manifest,
)


T_GEN = datetime(2026, 1, 1, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# build_provenance_record
# ---------------------------------------------------------------------------


def test_build_provenance_record_minimal() -> None:
    rec = build_provenance_record(
        output_kind="decision-packet",
        command="python scripts/13_decision_packet.py",
        generated_at=T_GEN,
        git_commit="deadbeef",
    )
    assert rec.output_kind == "decision-packet"
    assert rec.command == "python scripts/13_decision_packet.py"
    assert rec.git_commit == "deadbeef"
    assert rec.generated_at == "2026-01-01T00:00:00+00:00"
    assert rec.run_id  # populated
    assert rec.assumptions  # default assumptions populated
    assert rec.caveats


def test_build_provenance_record_with_inputs_and_args() -> None:
    inputs = (
        InputReference(role="fixture", path="tests/fixtures/foo.json", sha256="abc"),
    )
    rec = build_provenance_record(
        output_kind="decision-packet",
        command="python scripts/13_decision_packet.py",
        args=("--scenario", "tennent"),
        scenario_ids=("tennent",),
        inputs=inputs,
        git_commit="deadbeef",
        generated_at=T_GEN,
    )
    assert rec.args == ("--scenario", "tennent")
    assert rec.scenario_ids == ("tennent",)
    assert rec.inputs == inputs


def test_build_provenance_record_uses_default_assumptions_and_caveats() -> None:
    rec = build_provenance_record(
        output_kind="x", command="cmd", git_commit="deadbeef", generated_at=T_GEN,
    )
    assert any("synthetic" in a.lower() for a in rec.assumptions)
    assert any("local prototype" in c.lower() for c in rec.caveats)


def test_build_provenance_record_overrides_assumptions_and_caveats() -> None:
    rec = build_provenance_record(
        output_kind="x", command="cmd", git_commit="deadbeef", generated_at=T_GEN,
        assumptions=("custom assumption",),
        caveats=("custom caveat",),
    )
    assert rec.assumptions == ("custom assumption",)
    assert rec.caveats == ("custom caveat",)


# ---------------------------------------------------------------------------
# stable_run_id
# ---------------------------------------------------------------------------


def test_stable_run_id_is_deterministic() -> None:
    a = stable_run_id(
        command="cmd", args=("--x",), scenario_ids=("tennent",),
        inputs=(), git_commit="deadbeef",
    )
    b = stable_run_id(
        command="cmd", args=("--x",), scenario_ids=("tennent",),
        inputs=(), git_commit="deadbeef",
    )
    assert a == b


def test_stable_run_id_changes_with_args() -> None:
    a = stable_run_id(
        command="cmd", args=("--x",), scenario_ids=(),
        inputs=(), git_commit="deadbeef",
    )
    b = stable_run_id(
        command="cmd", args=("--y",), scenario_ids=(),
        inputs=(), git_commit="deadbeef",
    )
    assert a != b


def test_stable_run_id_changes_with_git_commit() -> None:
    a = stable_run_id(
        command="cmd", args=(), scenario_ids=(),
        inputs=(), git_commit="aaa",
    )
    b = stable_run_id(
        command="cmd", args=(), scenario_ids=(),
        inputs=(), git_commit="bbb",
    )
    assert a != b


def test_stable_run_id_changes_with_inputs() -> None:
    a = stable_run_id(
        command="cmd", args=(), scenario_ids=(),
        inputs=(InputReference(role="fixture", path="a.json", sha256="x"),),
        git_commit="deadbeef",
    )
    b = stable_run_id(
        command="cmd", args=(), scenario_ids=(),
        inputs=(InputReference(role="fixture", path="b.json", sha256="x"),),
        git_commit="deadbeef",
    )
    assert a != b


def test_stable_run_id_does_not_use_wall_clock() -> None:
    """Two records built at different times share the same run_id."""
    a = build_provenance_record(
        output_kind="x", command="cmd", git_commit="deadbeef",
        generated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    b = build_provenance_record(
        output_kind="x", command="cmd", git_commit="deadbeef",
        generated_at=datetime(2030, 6, 15, tzinfo=timezone.utc),
    )
    assert a.run_id == b.run_id


# ---------------------------------------------------------------------------
# get_git_commit
# ---------------------------------------------------------------------------


def test_get_git_commit_returns_string() -> None:
    commit = get_git_commit()
    assert isinstance(commit, str)
    assert len(commit) > 0


def test_get_git_commit_returns_unknown_when_git_missing() -> None:
    with mock.patch(
        "custody.provenance.subprocess.run",
        side_effect=FileNotFoundError(),
    ):
        assert get_git_commit() == "unknown"


def test_get_git_commit_returns_unknown_on_nonzero_exit() -> None:
    fake = mock.Mock(returncode=1, stdout="", stderr="not a repo")
    with mock.patch("custody.provenance.subprocess.run", return_value=fake):
        assert get_git_commit() == "unknown"


# ---------------------------------------------------------------------------
# file_sha256
# ---------------------------------------------------------------------------


def test_file_sha256_is_deterministic(tmp_path: Path) -> None:
    p = tmp_path / "f.txt"
    p.write_bytes(b"hello world")
    a = file_sha256(p)
    b = file_sha256(p)
    assert a == b
    assert len(a) == 64  # sha256 hex


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_record_to_dict_is_json_serializable() -> None:
    rec = build_provenance_record(
        output_kind="x", command="cmd", git_commit="deadbeef", generated_at=T_GEN,
        inputs=(InputReference(role="fixture", path="a.json", sha256="abc"),),
    )
    d = record_to_dict(rec)
    encoded = json.dumps(d)
    assert json.loads(encoded) == d


def test_attach_provenance_adds_top_level_key() -> None:
    rec = build_provenance_record(
        output_kind="x", command="cmd", git_commit="deadbeef", generated_at=T_GEN,
    )
    out = attach_provenance({"payload": {}}, rec)
    assert "provenance" in out
    assert out["provenance"]["run_id"] == rec.run_id
    # Original keys preserved.
    assert "payload" in out


def test_attach_provenance_overwrites_pre_existing_key() -> None:
    rec = build_provenance_record(
        output_kind="x", command="cmd", git_commit="deadbeef", generated_at=T_GEN,
    )
    out = attach_provenance({"provenance": "old"}, rec)
    assert out["provenance"] != "old"
    assert out["provenance"]["run_id"] == rec.run_id


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def _sample_record() -> ProvenanceRecord:
    return build_provenance_record(
        output_kind="decision-packet",
        command="python scripts/13_decision_packet.py",
        args=("--scenario", "tennent"),
        scenario_ids=("tennent",),
        inputs=(InputReference(
            role="fixture", path="tests/fixtures/foo.json", sha256="abc",
            description="synthetic fixture",
        ),),
        git_commit="deadbeef",
        generated_at=T_GEN,
    )


def test_format_text_includes_essentials() -> None:
    rec = _sample_record()
    text = format_text(rec)
    assert "run_id:" in text
    assert "deadbeef" in text
    assert "tennent" in text
    assert "tests/fixtures/foo.json" in text


def test_format_json_round_trips() -> None:
    rec = _sample_record()
    blob = format_json(rec)
    parsed = json.loads(blob)
    assert parsed["run_id"] == rec.run_id
    assert parsed["git_commit"] == "deadbeef"


def test_format_markdown_has_headers() -> None:
    rec = _sample_record()
    md = format_markdown(rec)
    assert md.startswith("# Provenance")
    assert "## Inputs" in md
    assert "## Caveats" in md


def test_write_manifest_dispatches_by_format(tmp_path: Path) -> None:
    rec = _sample_record()
    import io
    for fmt in ("text", "json", "markdown"):
        buf = io.StringIO()
        write_manifest(rec, fmt=fmt, sink=buf)
        assert buf.getvalue()


def test_write_manifest_rejects_unknown_format() -> None:
    rec = _sample_record()
    import io
    with pytest.raises(ValueError):
        write_manifest(rec, fmt="yaml", sink=io.StringIO())


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


def test_no_forbidden_language_in_default_assumptions_and_caveats() -> None:
    rec = build_provenance_record(
        output_kind="x", command="cmd", git_commit="deadbeef", generated_at=T_GEN,
    )
    blob = (
        " ".join(rec.assumptions) + " " + " ".join(rec.caveats)
    ).lower()
    for needle in _FORBIDDEN:
        assert needle not in blob, (
            f"forbidden token {needle!r} appears in default assumptions / caveats"
        )


def test_source_file_no_forbidden_language() -> None:
    import custody.provenance as mod
    src = Path(mod.__file__).read_text(encoding="utf-8").lower()
    for needle in _FORBIDDEN:
        assert needle not in src, (
            f"forbidden token {needle!r} appears in provenance.py source"
        )


# ---------------------------------------------------------------------------
# Import-boundary scan
# ---------------------------------------------------------------------------


def test_module_does_not_import_detection_or_gfw() -> None:
    import custody.provenance as mod
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
        f"provenance.py imports forbidden modules: {offending}"
    )
