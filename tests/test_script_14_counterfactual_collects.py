"""Tests for scripts/14_counterfactual_collects.py (ADR-0021 Slice 10).

CLI-level tests using importlib + io.StringIO; no subprocess.
"""
from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "14_counterfactual_collects.py"


@pytest.fixture(scope="module")
def cf_module():
    spec = importlib.util.spec_from_file_location(
        "script_14_counterfactual", SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["script_14_counterfactual"] = mod
    try:
        spec.loader.exec_module(mod)
        yield mod
    finally:
        sys.modules.pop("script_14_counterfactual", None)


def _run(cf_module, *argv: str) -> str:
    buf = io.StringIO()
    rc = cf_module.main(list(argv), out=buf)
    assert rc == 0
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Return codes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", ["tennent", "whitsun", "both"])
def test_main_returns_zero_for_each_scenario(cf_module, scenario: str) -> None:
    out = _run(cf_module, "--scenario", scenario)
    assert out


def test_main_default_runs_both_scenarios(cf_module) -> None:
    out = _run(cf_module).lower()
    assert "tennent" in out
    assert "whitsun" in out


# ---------------------------------------------------------------------------
# Section headers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "header",
    [
        "COUNTERFACTUAL COLLECT SIMULATION",
        "Current health",
        "Primary ambiguity",
        "Candidate strategy comparison",
        "Top candidate outcome paths",
        "Interpretation",
    ],
)
def test_output_contains_required_section(cf_module, header: str) -> None:
    out = _run(cf_module, "--scenario", "tennent")
    assert header in out, f"missing section header {header!r}"


def test_output_mentions_outcome_paths_block(cf_module) -> None:
    """Per the dispatch, the report must include an 'outcome paths' block."""
    out = _run(cf_module, "--scenario", "tennent")
    assert "outcome paths" in out.lower()


def test_output_includes_heuristic_weights_caveat(cf_module) -> None:
    out = _run(cf_module, "--scenario", "tennent")
    assert "heuristic resolution-potential estimates" in out
    assert "not calibrated probabilities" in out


# ---------------------------------------------------------------------------
# Forbidden language
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "needle",
    ["live tasking", "tasking order", "Sentinel integration", "production scheduler"],
)
def test_output_does_not_contain_forbidden_phrase(cf_module, needle: str) -> None:
    out = _run(cf_module, "--scenario", "both")
    assert needle.lower() not in out.lower(), (
        f"forbidden phrase {needle!r} appears in CLI output"
    )


# ---------------------------------------------------------------------------
# Determinism + max-results
# ---------------------------------------------------------------------------


def test_output_is_deterministic_across_two_calls(cf_module) -> None:
    out1 = _run(cf_module, "--scenario", "both")
    out2 = _run(cf_module, "--scenario", "both")
    assert out1 == out2


def test_max_results_limits_candidates(cf_module) -> None:
    out_full = _run(cf_module, "--scenario", "tennent")
    out_top3 = _run(cf_module, "--scenario", "tennent", "--max-results", "3")
    # Top-3 output must be shorter and the candidate strategy comparison
    # block must contain at most 3 data rows.
    assert len(out_top3) < len(out_full)
    block = out_top3.split("Candidate strategy comparison", 1)[1].split(
        "Top candidate outcome paths", 1,
    )[0]
    # Count score-row lines (lines with leading whitespace and an
    # underscore-style candidate_id).
    rows = [
        line for line in block.splitlines()
        if line and not line.startswith("---") and "candidate_id" not in line
    ]
    rows = [r for r in rows if r.strip()]
    assert len(rows) <= 3


# ---------------------------------------------------------------------------
# Tennent + Whitsun specific content
# ---------------------------------------------------------------------------


def test_tennent_primary_ambiguity_names_construction_or_fixed(cf_module) -> None:
    out = _run(cf_module, "--scenario", "tennent")
    block = out.split("Primary ambiguity", 1)[1].split(
        "Candidate strategy comparison", 1,
    )[0]
    assert (
        "fixed_reclamation_or_structure" in block
        or "construction_or_reclamation_activity" in block
    )


def test_whitsun_primary_ambiguity_names_known_pair(cf_module) -> None:
    out = _run(cf_module, "--scenario", "whitsun")
    block = out.split("Primary ambiguity", 1)[1].split(
        "Candidate strategy comparison", 1,
    )[0]
    assert (
        "ais_dark_or_poorly_observed_vessels" in block
        or "transient_anchorage_or_fishing_presence" in block
        or "vessel_cluster_activity" in block
    )


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
    assert not offending, f"counterfactual script imports forbidden modules: {offending}"
