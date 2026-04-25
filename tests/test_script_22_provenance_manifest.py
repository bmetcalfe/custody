"""Tests for ``scripts/22_provenance_manifest.py`` (ADR-0021 Slice 18)."""
from __future__ import annotations

import ast
import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "22_provenance_manifest.py"


@pytest.fixture(scope="module")
def script_module():
    spec = importlib.util.spec_from_file_location(
        "script_22_provenance_manifest", SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["script_22_provenance_manifest"] = mod
    try:
        spec.loader.exec_module(mod)
        yield mod
    finally:
        sys.modules.pop("script_22_provenance_manifest", None)


def _run(script_module, *argv: str) -> str:
    buf = io.StringIO()
    rc = script_module.main(list(argv), out=buf)
    assert rc == 0
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Format dispatch
# ---------------------------------------------------------------------------


def test_text_format_default(script_module) -> None:
    out = _run(script_module, "--scenario", "tennent")
    assert "run_id:" in out
    assert "tennent" in out


def test_json_format_parses(script_module) -> None:
    out = _run(script_module, "--scenario", "tennent", "--format", "json")
    parsed = json.loads(out)
    assert parsed["scenario_ids"] == ["tennent"]
    assert "run_id" in parsed
    assert "git_commit" in parsed


def test_markdown_format_has_headers(script_module) -> None:
    out = _run(script_module, "--scenario", "whitsun", "--format", "markdown")
    assert out.startswith("# Provenance")
    assert "## Caveats" in out


# ---------------------------------------------------------------------------
# Scenario handling
# ---------------------------------------------------------------------------


def test_scenario_both_lists_two_scenarios(script_module) -> None:
    out = _run(
        script_module, "--scenario", "both", "--format", "json",
    )
    parsed = json.loads(out)
    assert set(parsed["scenario_ids"]) == {"tennent", "whitsun"}


def test_scenario_none_yields_empty_list(script_module) -> None:
    out = _run(
        script_module, "--scenario", "none", "--format", "json",
    )
    parsed = json.loads(out)
    assert parsed["scenario_ids"] == []


# ---------------------------------------------------------------------------
# Input refs
# ---------------------------------------------------------------------------


def test_input_ref_with_explicit_sha(script_module) -> None:
    out = _run(
        script_module,
        "--scenario", "tennent",
        "--input-ref", "fixture:some/path.json:0123abcd",
        "--format", "json",
    )
    parsed = json.loads(out)
    refs = parsed["inputs"]
    assert len(refs) == 1
    assert refs[0]["path"] == "some/path.json"
    assert refs[0]["sha256"] == "0123abcd"


def test_input_ref_repeats_accumulate(script_module) -> None:
    out = _run(
        script_module,
        "--scenario", "none",
        "--input-ref", "fixture:a.json:aa",
        "--input-ref", "config:b.json:bb",
        "--format", "json",
    )
    parsed = json.loads(out)
    assert len(parsed["inputs"]) == 2
    roles = [r["role"] for r in parsed["inputs"]]
    assert roles == ["fixture", "config"]


def test_input_ref_missing_file_keeps_sha_none(script_module) -> None:
    out = _run(
        script_module,
        "--scenario", "none",
        "--input-ref", "fixture:does/not/exist.json",
        "--format", "json",
    )
    parsed = json.loads(out)
    assert parsed["inputs"][0]["sha256"] is None


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_run_is_deterministic(script_module) -> None:
    out1 = _run(
        script_module, "--scenario", "tennent",
        "--git-commit", "deadbeef", "--format", "json",
    )
    out2 = _run(
        script_module, "--scenario", "tennent",
        "--git-commit", "deadbeef", "--format", "json",
    )
    assert out1 == out2


def test_output_kind_is_recorded(script_module) -> None:
    out = _run(
        script_module,
        "--output-kind", "portfolio-allocation",
        "--scenario", "both",
        "--git-commit", "deadbeef",
        "--format", "json",
    )
    parsed = json.loads(out)
    assert parsed["output_kind"] == "portfolio-allocation"


def test_command_string_is_recorded(script_module) -> None:
    out = _run(
        script_module,
        "--scenario", "tennent",
        "--command", "python scripts/13_decision_packet.py --scenario tennent",
        "--git-commit", "deadbeef",
        "--format", "json",
    )
    parsed = json.loads(out)
    assert parsed["command"].startswith("python scripts/13_decision_packet.py")


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


def test_output_no_forbidden_language(script_module) -> None:
    out = _run(
        script_module, "--scenario", "both",
        "--git-commit", "deadbeef", "--format", "json",
    ).lower()
    for needle in _FORBIDDEN:
        assert needle not in out, (
            f"forbidden token {needle!r} appears in CLI output"
        )


# ---------------------------------------------------------------------------
# Import-boundary scan
# ---------------------------------------------------------------------------


def test_script_does_not_import_detection_or_gfw() -> None:
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    forbidden = (
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
                if any(alias.name.startswith(p) for p in forbidden):
                    offending.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod_name = node.module or ""
            if any(mod_name.startswith(p) for p in forbidden):
                offending.append(mod_name)
    assert not offending, (
        f"script imports forbidden modules: {offending}"
    )
