"""Tests for scripts/16_evaluate_collect_policies.py (ADR-0021 Slice 12).

CLI-level tests using importlib + io.StringIO; no subprocess.
"""
from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "16_evaluate_collect_policies.py"


@pytest.fixture(scope="module")
def policy_module():
    spec = importlib.util.spec_from_file_location(
        "script_16_policy_eval", SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["script_16_policy_eval"] = mod
    try:
        spec.loader.exec_module(mod)
        yield mod
    finally:
        sys.modules.pop("script_16_policy_eval", None)


def _run(policy_module, *argv: str) -> str:
    buf = io.StringIO()
    rc = policy_module.main(list(argv), out=buf)
    assert rc == 0
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Return codes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", ["tennent", "whitsun", "both"])
def test_main_returns_zero_for_each_scenario(policy_module, scenario: str) -> None:
    out = _run(policy_module, "--scenario", scenario)
    assert out


def test_main_default_runs_both_scenarios(policy_module) -> None:
    out = _run(policy_module).lower()
    assert "tennent" in out
    assert "whitsun" in out


# ---------------------------------------------------------------------------
# Section headers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "header",
    [
        "COLLECTION POLICY EVALUATION",
        "Constraint",
        "Policies evaluated",
        "Ranked policy results",
        "Winning policy",
        "Caveats",
    ],
)
def test_output_contains_required_section(policy_module, header: str) -> None:
    out = _run(policy_module, "--scenario", "tennent")
    assert header in out, f"missing section header {header!r}"


def test_output_contains_rl_ready_caveat(policy_module) -> None:
    out = _run(policy_module, "--scenario", "tennent").lower()
    assert "deterministic heuristic" in out
    assert "rl-ready" in out
    assert "no learned policies" in out


# ---------------------------------------------------------------------------
# --budget
# ---------------------------------------------------------------------------


def test_budget_flag_changes_constraint_display(policy_module) -> None:
    tight = _run(policy_module, "--scenario", "tennent", "--budget", "0.3")
    loose = _run(policy_module, "--scenario", "tennent", "--budget", "2.0")
    assert "budget: 0.30" in tight
    assert "budget: 2.00" in loose
    assert tight != loose


# ---------------------------------------------------------------------------
# --max-collects
# ---------------------------------------------------------------------------


def test_max_collects_one_returns_at_most_one_candidate_per_policy(policy_module) -> None:
    """Every ranked policy line should list at most one candidate when
    max_collects=1."""
    out = _run(
        policy_module, "--scenario", "tennent", "--max-collects", "1",
    )
    block = out.split("Ranked policy results", 1)[1].split(
        "Winning policy", 1,
    )[0]
    # Each ranked-result line ends with "candidates: <list>".  Count
    # commas in the candidate list to bound the per-policy count.
    for line in block.splitlines():
        if "candidates:" not in line:
            continue
        suffix = line.split("candidates:", 1)[1].strip()
        if suffix == "(none)":
            continue
        # Comma-separated candidate count <= 1 means no comma.
        assert "," not in suffix, (
            f"max_collects=1 violated by line: {line.strip()!r}"
        )


# ---------------------------------------------------------------------------
# --policies
# ---------------------------------------------------------------------------


def test_policies_flag_limits_evaluated_set(policy_module) -> None:
    out = _run(
        policy_module, "--scenario", "tennent",
        "--policies", "value_optimized,low_cost_first",
    )
    block = out.split("Ranked policy results", 1)[1].split(
        "Winning policy", 1,
    )[0]
    # Only the two policies should appear in the ranked-results block.
    assert "value_optimized" in block
    assert "low_cost_first" in block
    assert "ambiguity_first" not in block
    assert "sar_repeat_first" not in block


# ---------------------------------------------------------------------------
# Forbidden language
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "needle",
    [
        "tasking order", "live tasking", "production scheduler",
        "Sentinel integration", "revenue dollars",
    ],
)
def test_output_does_not_contain_forbidden_phrase(policy_module, needle: str) -> None:
    out = _run(policy_module, "--scenario", "both")
    assert needle.lower() not in out.lower(), (
        f"forbidden phrase {needle!r} appears in CLI output"
    )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_output_is_deterministic_across_two_calls(policy_module) -> None:
    out1 = _run(policy_module, "--scenario", "both")
    out2 = _run(policy_module, "--scenario", "both")
    assert out1 == out2


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
        f"policy-eval script imports forbidden modules: {offending}"
    )
