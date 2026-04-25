"""Tests for scripts/19_efficiency_metrics.py (ADR-0021 Slice 15).

CLI-level tests using importlib + io.StringIO; no subprocess.
"""
from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "19_efficiency_metrics.py"


@pytest.fixture(scope="module")
def efficiency_module():
    spec = importlib.util.spec_from_file_location(
        "script_19_efficiency", SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["script_19_efficiency"] = mod
    try:
        spec.loader.exec_module(mod)
        yield mod
    finally:
        sys.modules.pop("script_19_efficiency", None)


def _run(efficiency_module, *argv: str) -> str:
    buf = io.StringIO()
    rc = efficiency_module.main(list(argv), out=buf)
    assert rc == 0
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Return codes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", ["tennent", "whitsun", "both"])
def test_main_returns_zero_for_each_scenario(
    efficiency_module, scenario: str,
) -> None:
    out = _run(efficiency_module, "--scenario", scenario)
    assert out


def test_main_default_runs_both_scenarios(efficiency_module) -> None:
    out = _run(efficiency_module).lower()
    assert "tennent" in out
    assert "whitsun" in out


# ---------------------------------------------------------------------------
# Required section headers
# ---------------------------------------------------------------------------


def test_output_contains_workflow_efficiency_report_header(
    efficiency_module,
) -> None:
    out = _run(efficiency_module, "--scenario", "tennent")
    assert "WORKFLOW EFFICIENCY REPORT" in out


def test_output_contains_baseline_workflow_section(efficiency_module) -> None:
    out = _run(efficiency_module, "--scenario", "tennent")
    assert "Baseline workflow" in out


def test_output_contains_custody_workflow_section(efficiency_module) -> None:
    out = _run(efficiency_module, "--scenario", "tennent")
    assert "Custody-assisted workflow" in out


def test_output_contains_proxy_metrics_section(efficiency_module) -> None:
    out = _run(efficiency_module, "--scenario", "tennent")
    assert "Proxy metrics" in out


def test_output_contains_caveats_section(efficiency_module) -> None:
    out = _run(efficiency_module, "--scenario", "tennent")
    assert "Caveats" in out


def test_output_includes_prototype_proxy_caveat(efficiency_module) -> None:
    out = _run(efficiency_module, "--scenario", "tennent").lower()
    assert "proxy metric" in out
    assert "prototype" in out


# ---------------------------------------------------------------------------
# Format flags
# ---------------------------------------------------------------------------


def test_format_json_is_parseable(efficiency_module) -> None:
    out = _run(efficiency_module, "--scenario", "tennent", "--format", "json")
    parsed = json.loads(out)
    assert parsed["scenario_id"] == "tennent"
    assert "metrics" in parsed
    assert isinstance(parsed["metrics"], list)


def test_format_json_both_is_array_of_two(efficiency_module) -> None:
    out = _run(efficiency_module, "--scenario", "both", "--format", "json")
    parsed = json.loads(out)
    assert isinstance(parsed, list)
    assert len(parsed) == 2
    assert parsed[0]["scenario_id"] == "tennent"
    assert parsed[1]["scenario_id"] == "whitsun"


def test_format_md_starts_with_h1(efficiency_module) -> None:
    out = _run(efficiency_module, "--scenario", "tennent", "--format", "md")
    assert out.startswith("# Workflow Efficiency Report - tennent")


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_output_is_deterministic(efficiency_module) -> None:
    out1 = _run(efficiency_module, "--scenario", "both")
    out2 = _run(efficiency_module, "--scenario", "both")
    assert out1 == out2


# ---------------------------------------------------------------------------
# Forbidden language
# ---------------------------------------------------------------------------


_FORBIDDEN = (
    "proven savings",
    "actual revenue",
    "production cycle-time reduction",
    "real planner adoption",
    "operational kpi",
    "tasking order",
    "live tasking",
    "sensor command",
    "production scheduler",
    "autonomous execution",
    "sentinel integration",
    "collection order",
)


def test_output_does_not_contain_forbidden_language(efficiency_module) -> None:
    out = _run(efficiency_module, "--scenario", "both").lower()
    for needle in _FORBIDDEN:
        assert needle not in out, (
            f"forbidden token {needle!r} appears in CLI output"
        )


# ---------------------------------------------------------------------------
# Import-boundary scan
# ---------------------------------------------------------------------------


def test_script_does_not_import_detection_or_gfw() -> None:
    import ast
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
        f"efficiency-metrics script imports forbidden modules: {offending}"
    )
