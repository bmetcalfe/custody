"""Tests for scripts/20_portfolio_allocation.py (ADR-0021 Slice 16).

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
SCRIPT_PATH = REPO_ROOT / "scripts" / "20_portfolio_allocation.py"


@pytest.fixture(scope="module")
def portfolio_module():
    spec = importlib.util.spec_from_file_location(
        "script_20_portfolio", SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["script_20_portfolio"] = mod
    try:
        spec.loader.exec_module(mod)
        yield mod
    finally:
        sys.modules.pop("script_20_portfolio", None)


def _run(portfolio_module, *argv: str) -> str:
    buf = io.StringIO()
    rc = portfolio_module.main(list(argv), out=buf)
    assert rc == 0
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Return codes
# ---------------------------------------------------------------------------


def test_main_returns_zero_for_both(portfolio_module) -> None:
    out = _run(portfolio_module, "--scenario", "both")
    assert out


def test_main_returns_zero_for_tennent(portfolio_module) -> None:
    out = _run(portfolio_module, "--scenario", "tennent")
    assert out


def test_main_returns_zero_for_whitsun(portfolio_module) -> None:
    out = _run(portfolio_module, "--scenario", "whitsun")
    assert out


# ---------------------------------------------------------------------------
# Required section headers
# ---------------------------------------------------------------------------


def test_output_contains_portfolio_allocation_header(portfolio_module) -> None:
    out = _run(portfolio_module, "--scenario", "both")
    assert "PORTFOLIO ALLOCATION PLAN" in out


def test_output_contains_global_constraints(portfolio_module) -> None:
    out = _run(portfolio_module, "--scenario", "both")
    assert "Global constraints" in out


def test_output_contains_recommended_portfolio(portfolio_module) -> None:
    out = _run(portfolio_module, "--scenario", "both")
    assert "Recommended portfolio" in out


def test_output_contains_scenario_coverage(portfolio_module) -> None:
    out = _run(portfolio_module, "--scenario", "both").lower()
    assert "scenario coverage" in out


def test_output_contains_caveats(portfolio_module) -> None:
    out = _run(portfolio_module, "--scenario", "both")
    assert "decision support" in out.lower()
    assert "execution authorization" in out.lower() or "execution authorizations" in out.lower()


# ---------------------------------------------------------------------------
# Format flags
# ---------------------------------------------------------------------------


def test_format_json_is_parseable(portfolio_module) -> None:
    out = _run(portfolio_module, "--scenario", "both", "--format", "json")
    parsed = json.loads(out)
    assert "scenario_ids" in parsed


def test_format_json_has_required_keys(portfolio_module) -> None:
    out = _run(portfolio_module, "--scenario", "both", "--format", "json")
    parsed = json.loads(out)
    for key in (
        "scenario_ids", "constraint", "candidate_pool",
        "recommended_plan", "exhaustive_plan", "greedy_plan",
        "comparison_summary", "caveats",
    ):
        assert key in parsed, f"missing JSON key {key!r}"


def test_format_md_starts_with_h1(portfolio_module) -> None:
    out = _run(portfolio_module, "--scenario", "both", "--format", "md")
    assert out.startswith("# Portfolio Allocation Plan\n")


# ---------------------------------------------------------------------------
# Constraint flags
# ---------------------------------------------------------------------------


def test_budget_flag_changes_constraint(portfolio_module) -> None:
    out = _run(portfolio_module, "--scenario", "both", "--budget", "0.50")
    assert "budget: 0.50" in out


def test_max_collects_one_returns_at_most_one(portfolio_module) -> None:
    out = _run(
        portfolio_module, "--scenario", "both",
        "--max-collects", "1",
    )
    block = out.split("Recommended portfolio", 1)[1].split("Totals:", 1)[0]
    item_lines = [
        line for line in block.splitlines()
        if "score=" in line and line.strip().startswith(tuple("0123456789"))
    ]
    assert len(item_lines) <= 1


def test_max_collects_per_scenario_one(portfolio_module) -> None:
    out = _run(
        portfolio_module, "--scenario", "both",
        "--max-collects-per-scenario", "1",
        "--max-collects", "6",
    )
    block = out.split("Recommended portfolio", 1)[1].split("Totals:", 1)[0]
    counts: dict[str, int] = {}
    for line in block.splitlines():
        for sid in ("tennent", "whitsun"):
            if f" {sid} / " in line:
                counts[sid] = counts.get(sid, 0) + 1
    for n in counts.values():
        assert n <= 1


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_output_is_deterministic(portfolio_module) -> None:
    out1 = _run(portfolio_module, "--scenario", "both")
    out2 = _run(portfolio_module, "--scenario", "both")
    assert out1 == out2


# ---------------------------------------------------------------------------
# Forbidden language
# ---------------------------------------------------------------------------


_FORBIDDEN = (
    "tasking order",
    "live tasking",
    "production scheduler",
    "sentinel integration",
    "sensor command",
    "autonomous constellation management",
    "revenue dollars",
)


def test_output_no_forbidden_language(portfolio_module) -> None:
    out = _run(portfolio_module, "--scenario", "both").lower()
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
        f"portfolio script imports forbidden modules: {offending}"
    )
