"""Tests for ``scripts/24_scene_availability.py`` (ADR-0021 Slice 20)."""
from __future__ import annotations

import ast
import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "24_scene_availability.py"
TENNENT_FIXTURE = (
    REPO_ROOT / "tests" / "fixtures" / "availability"
    / "tennent_scene_availability.json"
)


@pytest.fixture(scope="module")
def script_module():
    spec = importlib.util.spec_from_file_location(
        "script_24_scene_availability", SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["script_24_scene_availability"] = mod
    try:
        spec.loader.exec_module(mod)
        yield mod
    finally:
        sys.modules.pop("script_24_scene_availability", None)


def _run(script_module, *argv: str) -> str:
    buf = io.StringIO()
    rc = script_module.main(list(argv), out=buf)
    assert rc == 0
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Return codes / scenarios
# ---------------------------------------------------------------------------


def test_main_returns_zero_tennent(script_module) -> None:
    assert _run(script_module, "--scenario", "tennent")


def test_main_returns_zero_whitsun(script_module) -> None:
    assert _run(script_module, "--scenario", "whitsun")


def test_main_returns_zero_both(script_module) -> None:
    assert _run(script_module, "--scenario", "both")


def test_main_with_explicit_catalog(script_module) -> None:
    out = _run(
        script_module, "--scenario", "tennent",
        "--catalog", str(TENNENT_FIXTURE),
    )
    assert "SCENE AVAILABILITY BRIDGE" in out


# ---------------------------------------------------------------------------
# Format dispatch
# ---------------------------------------------------------------------------


def test_format_json_parseable(script_module) -> None:
    out = _run(script_module, "--scenario", "tennent", "--format", "json")
    parsed = json.loads(out)
    assert parsed["scenario_id"] == "tennent"
    assert "adjusted_values" in parsed


def test_format_md_contains_heading(script_module) -> None:
    out = _run(script_module, "--scenario", "whitsun", "--format", "md")
    assert "# Scene Availability Bridge" in out


# ---------------------------------------------------------------------------
# Output content
# ---------------------------------------------------------------------------


def test_output_contains_scene_availability_header(script_module) -> None:
    out = _run(script_module, "--scenario", "tennent")
    assert "SCENE AVAILABILITY BRIDGE" in out


def test_output_contains_catalog_summary(script_module) -> None:
    out = _run(script_module, "--scenario", "tennent")
    assert "Catalog summary" in out


def test_output_contains_candidate_feasibility(script_module) -> None:
    out = _run(script_module, "--scenario", "tennent")
    assert "Candidate feasibility" in out


def test_output_contains_adjusted_recommendation(script_module) -> None:
    out = _run(script_module, "--scenario", "tennent")
    assert "Availability-adjusted recommendation" in out


def test_output_caveats_no_imagery(script_module) -> None:
    out = _run(script_module, "--scenario", "tennent")
    assert "no imagery was downloaded or processed" in out


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_output_is_deterministic(script_module) -> None:
    a = _run(script_module, "--scenario", "tennent", "--format", "json")
    b = _run(script_module, "--scenario", "tennent", "--format", "json")
    assert a == b


# ---------------------------------------------------------------------------
# Forbidden language
# ---------------------------------------------------------------------------


_FORBIDDEN = (
    "tasking order",
    "live tasking",
    "production scheduler",
    "sentinel integration",
    "real-time ingestion",
    "sensor command",
    "platform access",
    "autonomous execution",
    "collection order",
    "revenue dollars",
)


def test_output_no_forbidden_language(script_module) -> None:
    out = _run(script_module, "--scenario", "both", "--format", "json").lower()
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
    assert not offending, f"script imports forbidden modules: {offending}"
