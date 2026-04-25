"""Tests for ``scripts/25_availability_optimized_plan.py`` (ADR-0021 Slice 21)."""
from __future__ import annotations

import ast
import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "25_availability_optimized_plan.py"


@pytest.fixture(scope="module")
def script_module():
    spec = importlib.util.spec_from_file_location(
        "script_25_availability_optimized_plan", SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["script_25_availability_optimized_plan"] = mod
    try:
        spec.loader.exec_module(mod)
        yield mod
    finally:
        sys.modules.pop("script_25_availability_optimized_plan", None)


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


def test_main_with_budget_and_max_collects(script_module) -> None:
    out = _run(
        script_module, "--scenario", "tennent",
        "--budget", "1.5", "--max-collects", "3",
    )
    assert "AVAILABILITY-ADJUSTED OPTIMIZED PLAN" in out


# ---------------------------------------------------------------------------
# Format dispatch
# ---------------------------------------------------------------------------


def test_format_json_parseable(script_module) -> None:
    out = _run(script_module, "--scenario", "tennent", "--format", "json")
    parsed = json.loads(out)
    assert parsed["scenario_id"] == "tennent"
    assert "recommended_plan" in parsed


def test_format_md_contains_heading(script_module) -> None:
    out = _run(script_module, "--scenario", "whitsun", "--format", "md")
    assert "# Availability-Adjusted Optimized Plan" in out


# ---------------------------------------------------------------------------
# Output content
# ---------------------------------------------------------------------------


def test_output_contains_availability_adjusted_header(script_module) -> None:
    out = _run(script_module, "--scenario", "tennent")
    assert "AVAILABILITY-ADJUSTED OPTIMIZED PLAN" in out


def test_output_contains_base_plan(script_module) -> None:
    out = _run(script_module, "--scenario", "tennent")
    assert "Base optimized plan" in out


def test_output_contains_availability_feasibility(script_module) -> None:
    out = _run(script_module, "--scenario", "tennent")
    assert "Availability feasibility" in out


def test_output_contains_adjusted_pool(script_module) -> None:
    out = _run(script_module, "--scenario", "tennent")
    assert "Adjusted candidate pool" in out


def test_output_contains_recommended_plan(script_module) -> None:
    out = _run(script_module, "--scenario", "tennent")
    assert "Recommended availability-adjusted plan" in out


def test_output_caveats_decision_support(script_module) -> None:
    out = _run(script_module, "--scenario", "tennent")
    assert "decision support only" in out


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
    "live tasking",
    "tasking order",
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
