"""Tests for ``scripts/27_simulate_plan_execution.py`` (ADR-0021 Slice 23)."""
from __future__ import annotations

import ast
import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "27_simulate_plan_execution.py"


@pytest.fixture(scope="module")
def script_module():
    spec = importlib.util.spec_from_file_location(
        "script_27_simulate_plan_execution", SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["script_27_simulate_plan_execution"] = mod
    try:
        spec.loader.exec_module(mod)
        yield mod
    finally:
        sys.modules.pop("script_27_simulate_plan_execution", None)


def _run(script_module, *argv: str) -> str:
    buf = io.StringIO()
    rc = script_module.main(list(argv), out=buf)
    assert rc == 0
    return buf.getvalue()


# ---------------------------------------------------------------------------
# CLI execution
# ---------------------------------------------------------------------------


def test_main_returns_zero_tennent(script_module) -> None:
    assert _run(script_module, "--scenario", "tennent")


def test_main_returns_zero_whitsun(script_module) -> None:
    assert _run(script_module, "--scenario", "whitsun")


def test_main_returns_zero_both(script_module) -> None:
    out = _run(script_module, "--scenario", "both")
    assert "TENNENT" in out
    assert "WHITSUN" in out


def test_outcome_policy_favorable(script_module) -> None:
    out = _run(
        script_module, "--scenario", "tennent",
        "--outcome-policy", "favorable",
    )
    assert "Outcome policy: favorable" in out


def test_outcome_policy_inconclusive(script_module) -> None:
    out = _run(
        script_module, "--scenario", "tennent",
        "--outcome-policy", "inconclusive",
    )
    assert "Outcome policy: inconclusive" in out


def test_outcome_policy_adverse(script_module) -> None:
    out = _run(
        script_module, "--scenario", "tennent",
        "--outcome-policy", "adverse",
    )
    assert "Outcome policy: adverse" in out


def test_outcome_policy_mixed(script_module) -> None:
    out = _run(
        script_module, "--scenario", "tennent",
        "--outcome-policy", "mixed",
    )
    assert "Outcome policy: mixed" in out


# ---------------------------------------------------------------------------
# Format dispatch
# ---------------------------------------------------------------------------


def test_format_json_parseable(script_module) -> None:
    out = _run(
        script_module, "--scenario", "tennent", "--format", "json",
    )
    parsed = json.loads(out)
    assert parsed["scenario_id"] == "tennent"
    assert "post_recommendation" in parsed


def test_format_md_contains_heading(script_module) -> None:
    out = _run(
        script_module, "--scenario", "whitsun", "--format", "md",
    )
    assert "# Plan Execution Simulation" in out


def test_default_format_is_text(script_module) -> None:
    out = _run(script_module, "--scenario", "tennent")
    assert "PLAN EXECUTION SIMULATION" in out


# ---------------------------------------------------------------------------
# Output content
# ---------------------------------------------------------------------------


def test_output_contains_pre_execution(script_module) -> None:
    out = _run(script_module, "--scenario", "tennent")
    assert "Pre-execution" in out


def test_output_contains_scheduled_collects(script_module) -> None:
    out = _run(script_module, "--scenario", "tennent")
    assert "Scheduled collects" in out


def test_output_contains_simulated_returned_evidence(script_module) -> None:
    out = _run(script_module, "--scenario", "tennent")
    assert "Simulated returned evidence" in out


def test_output_contains_post_execution(script_module) -> None:
    out = _run(script_module, "--scenario", "tennent")
    assert "Post-execution" in out


def test_output_contains_before_after_health(script_module) -> None:
    out = _run(script_module, "--scenario", "tennent")
    assert "Before/after health" in out


def test_output_caveats_no_live_tasking(script_module) -> None:
    out = _run(script_module, "--scenario", "tennent")
    assert "no live tasking or sensor command is issued" in out


# ---------------------------------------------------------------------------
# Forbidden language
# ---------------------------------------------------------------------------


_FORBIDDEN = (
    "actual collection result",
    "tasking order",
    "production scheduler",
    "sentinel integration",
    "satellite command",
    "autonomous execution",
    "real image",
    "collection order",
    "revenue dollars",
)


def test_output_no_forbidden_language(script_module) -> None:
    out = _run(
        script_module, "--scenario", "both", "--format", "json",
    ).lower()
    for needle in _FORBIDDEN:
        assert needle not in out, (
            f"forbidden token {needle!r} appears in CLI output"
        )
    safe = out.replace(
        "no live tasking or sensor command is issued", "",
    )
    for needle in ("live tasking", "sensor command", "platform access"):
        assert needle not in safe, (
            f"forbidden token {needle!r} appears outside explicit disclaimer"
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
        "requests",
        "httpx",
        "boto3",
        "google.cloud",
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
