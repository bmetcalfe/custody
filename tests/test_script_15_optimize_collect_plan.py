"""Tests for scripts/15_optimize_collect_plan.py (ADR-0021 Slice 11).

CLI-level tests using importlib + io.StringIO; no subprocess.
"""
from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "15_optimize_collect_plan.py"


@pytest.fixture(scope="module")
def opt_module():
    spec = importlib.util.spec_from_file_location(
        "script_15_optimizer", SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["script_15_optimizer"] = mod
    try:
        spec.loader.exec_module(mod)
        yield mod
    finally:
        sys.modules.pop("script_15_optimizer", None)


def _run(opt_module, *argv: str) -> str:
    buf = io.StringIO()
    rc = opt_module.main(list(argv), out=buf)
    assert rc == 0
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Return codes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", ["tennent", "whitsun", "both"])
def test_main_returns_zero_for_each_scenario(opt_module, scenario: str) -> None:
    out = _run(opt_module, "--scenario", scenario)
    assert out


def test_main_default_runs_both_scenarios(opt_module) -> None:
    out = _run(opt_module).lower()
    assert "tennent" in out
    assert "whitsun" in out


# ---------------------------------------------------------------------------
# Section headers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "header",
    [
        "OPTIMIZED COLLECTION PLAN",
        "Constraints",
        "Candidate utility inputs",
        "Recommended plan",
        "Greedy vs exhaustive comparison",
        "Caveats",
    ],
)
def test_output_contains_required_section(opt_module, header: str) -> None:
    out = _run(opt_module, "--scenario", "tennent")
    assert header in out, f"missing section header {header!r}"


def test_output_contains_planning_utility_proxy_caveat(opt_module) -> None:
    out = _run(opt_module, "--scenario", "tennent")
    lower = out.lower()
    assert "planning utility" in lower
    assert "proxy" in lower


# ---------------------------------------------------------------------------
# --budget
# ---------------------------------------------------------------------------


def test_budget_flag_changes_plan_total_cost(opt_module) -> None:
    """A small budget yields different plan output than a generous budget."""
    tight = _run(opt_module, "--scenario", "tennent", "--budget", "0.3")
    loose = _run(opt_module, "--scenario", "tennent", "--budget", "2.0")
    assert tight != loose


def test_budget_flag_constrains_total_cost(opt_module) -> None:
    """Total cost line must respect the requested budget cap."""
    out = _run(opt_module, "--scenario", "tennent", "--budget", "0.3")
    # Find the "cost: X.XX / 0.30" line in Total: block.
    import re
    m = re.search(r"cost:\s*([\d.]+)\s*/\s*0\.30", out)
    assert m is not None, "missing cost-vs-budget total line"
    total_cost = float(m.group(1))
    assert total_cost <= 0.30 + 1e-9


# ---------------------------------------------------------------------------
# --max-collects
# ---------------------------------------------------------------------------


def test_max_collects_one_returns_at_most_one_item(opt_module) -> None:
    out = _run(opt_module, "--scenario", "tennent", "--max-collects", "1")
    # The recommended-plan block lists up to N candidate-row lines (lines
    # with leading whitespace and "cost=" / "value=" tokens).
    block = out.split("Recommended plan", 1)[1].split("Total:", 1)[0]
    rows = [ln for ln in block.splitlines() if "cost=" in ln and "value=" in ln]
    assert len(rows) <= 1


# ---------------------------------------------------------------------------
# --strategy
# ---------------------------------------------------------------------------


def test_strategy_exhaustive_labels_exhaustive(opt_module) -> None:
    out = _run(
        opt_module, "--scenario", "tennent", "--strategy", "exhaustive",
    )
    # Recommended-plan header reads "Recommended plan (strategy: exhaustive)"
    rec_header = out.split("Recommended plan", 1)[1].splitlines()[0]
    assert "exhaustive" in rec_header


def test_strategy_greedy_labels_greedy(opt_module) -> None:
    out = _run(opt_module, "--scenario", "tennent", "--strategy", "greedy")
    rec_header = out.split("Recommended plan", 1)[1].splitlines()[0]
    assert "greedy" in rec_header


# ---------------------------------------------------------------------------
# Forbidden language
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "needle",
    [
        "tasking order", "live tasking", "production scheduler",
        "Sentinel integration", "actual revenue", "profit",
    ],
)
def test_output_does_not_contain_forbidden_phrase(opt_module, needle: str) -> None:
    out = _run(opt_module, "--scenario", "both")
    assert needle.lower() not in out.lower(), (
        f"forbidden phrase {needle!r} appears in CLI output"
    )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_output_is_deterministic_across_two_calls(opt_module) -> None:
    out1 = _run(opt_module, "--scenario", "both")
    out2 = _run(opt_module, "--scenario", "both")
    assert out1 == out2


# ---------------------------------------------------------------------------
# Tennent / Whitsun specific content
# ---------------------------------------------------------------------------


def test_tennent_recommended_plan_includes_optical_context(opt_module) -> None:
    out = _run(opt_module, "--scenario", "tennent")
    block = out.split("Recommended plan", 1)[1].split("Total:", 1)[0]
    assert "optical_context" in block


def test_whitsun_recommended_plan_includes_ais_coverage_query(opt_module) -> None:
    out = _run(opt_module, "--scenario", "whitsun")
    block = out.split("Recommended plan", 1)[1].split("Total:", 1)[0]
    assert "ais_coverage_query" in block


# ---------------------------------------------------------------------------
# Import-boundary scan
# ---------------------------------------------------------------------------


def test_script_does_not_import_forbidden_modules() -> None:
    import ast
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    forbidden = (
        "custody.detection",
        "custody.ingest.gfw_presence",
        "sentinelhub",
        "custody.fusion.tracker",
        "custody.taskrecommendation",
        "scipy",
        "sklearn",
        "ortools",
        "pulp",
        "networkx",
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
        f"optimizer script imports forbidden modules: {offending}"
    )
