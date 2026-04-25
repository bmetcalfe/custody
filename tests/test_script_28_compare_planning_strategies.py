"""Tests for ``scripts/28_compare_planning_strategies.py`` (ADR-0021 Slice 24)."""
from __future__ import annotations

import ast
import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "28_compare_planning_strategies.py"


@pytest.fixture(scope="module")
def script_module():
    spec = importlib.util.spec_from_file_location(
        "script_28_compare_planning_strategies", SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["script_28_compare_planning_strategies"] = mod
    try:
        spec.loader.exec_module(mod)
        yield mod
    finally:
        sys.modules.pop("script_28_compare_planning_strategies", None)


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
    assert "CROSS-SCENARIO AGGREGATE" in out


def test_outcome_policy_favorable(script_module) -> None:
    assert _run(
        script_module, "--scenario", "tennent",
        "--outcome-policy", "favorable",
    )


def test_outcome_policy_inconclusive(script_module) -> None:
    assert _run(
        script_module, "--scenario", "tennent",
        "--outcome-policy", "inconclusive",
    )


def test_outcome_policy_adverse(script_module) -> None:
    assert _run(
        script_module, "--scenario", "tennent",
        "--outcome-policy", "adverse",
    )


def test_outcome_policy_mixed(script_module) -> None:
    assert _run(
        script_module, "--scenario", "tennent",
        "--outcome-policy", "mixed",
    )


def test_strategies_filter(script_module) -> None:
    out = _run(
        script_module, "--scenario", "tennent",
        "--strategies", "baseline_manual,execution_feedback",
    )
    assert "baseline_manual" in out
    assert "execution_feedback" in out


# ---------------------------------------------------------------------------
# Format dispatch
# ---------------------------------------------------------------------------


def test_format_json_parseable(script_module) -> None:
    out = _run(script_module, "--scenario", "tennent", "--format", "json")
    parsed = json.loads(out)
    assert parsed["scenario_id"] == "tennent"


def test_format_md_contains_heading(script_module) -> None:
    out = _run(script_module, "--scenario", "whitsun", "--format", "md")
    assert "# Planning Strategy Comparison" in out


def test_default_format_is_text(script_module) -> None:
    out = _run(script_module, "--scenario", "tennent")
    assert "PLANNING STRATEGY COMPARISON" in out


# ---------------------------------------------------------------------------
# Output content
# ---------------------------------------------------------------------------


def test_output_contains_strategies_evaluated(script_module) -> None:
    out = _run(script_module, "--scenario", "tennent")
    assert "Strategies evaluated" in out


def test_output_contains_ranked_results(script_module) -> None:
    out = _run(script_module, "--scenario", "tennent")
    assert "Ranked strategy results" in out


def test_output_contains_winning_strategy(script_module) -> None:
    out = _run(script_module, "--scenario", "tennent")
    assert "Winning strategy" in out


def test_output_contains_key_deltas(script_module) -> None:
    out = _run(script_module, "--scenario", "tennent")
    assert "Key deltas" in out


def test_output_caveats_deterministic_prototype(script_module) -> None:
    out = _run(script_module, "--scenario", "tennent")
    assert "deterministic prototype" in out


# ---------------------------------------------------------------------------
# Budget / max-collects
# ---------------------------------------------------------------------------


def test_budget_argument_accepted(script_module) -> None:
    assert _run(script_module, "--scenario", "tennent", "--budget", "2.0")


def test_max_collects_argument_accepted(script_module) -> None:
    assert _run(
        script_module, "--scenario", "tennent", "--max-collects", "4",
    )


# ---------------------------------------------------------------------------
# Forbidden language
# ---------------------------------------------------------------------------


_FORBIDDEN = (
    "proven savings",
    "actual revenue",
    "real planner adoption",
    "operational kpi",
    "production cycle-time reduction",
    "tasking order",
    "production scheduler",
    "sentinel integration",
    "satellite command",
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
