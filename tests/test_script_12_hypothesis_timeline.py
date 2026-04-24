"""Tests for scripts/12_hypothesis_timeline.py (ADR-0021 Slice 3).

These tests exercise the CLI by importing main() directly — no subprocess.
Assertions check stable key lines (scenario headers, scene dates, "Top:"
and "Uncertainty:" presence, final-state block) rather than exact floats,
so minor scoring tweaks within the generators don't force test churn.
"""
from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "12_hypothesis_timeline.py"


@pytest.fixture(scope="module")
def timeline_module():
    """Load scripts/12_hypothesis_timeline.py as a module (no CLI side effects).

    The ``if __name__ == '__main__'`` guard prevents argparse from running
    on import; we then reach ``main`` and other helpers directly.
    """
    spec = importlib.util.spec_from_file_location(
        "script_12_hypothesis_timeline", SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["script_12_hypothesis_timeline"] = mod
    try:
        spec.loader.exec_module(mod)
        yield mod
    finally:
        sys.modules.pop("script_12_hypothesis_timeline", None)


# ---------------------------------------------------------------------------
# Return codes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", ["tennent", "whitsun", "both"])
def test_main_returns_zero_for_each_scenario(timeline_module, scenario: str) -> None:
    buf = io.StringIO()
    rc = timeline_module.main(["--scenario", scenario], out=buf)
    assert rc == 0
    assert buf.getvalue(), "CLI must produce output"


def test_main_default_runs_both_scenarios(timeline_module) -> None:
    buf = io.StringIO()
    rc = timeline_module.main([], out=buf)
    assert rc == 0
    out = buf.getvalue()
    assert "tennent" in out.lower()
    assert "whitsun" in out.lower()


# ---------------------------------------------------------------------------
# Content assertions (stable key lines only)
# ---------------------------------------------------------------------------


TENNENT_DATES = ("2023-07-02", "2023-07-23", "2023-08-07", "2023-08-09", "2023-08-13")
WHITSUN_DATES = ("2023-12-06", "2024-03-20")


def _run(timeline_module, scenario: str) -> str:
    buf = io.StringIO()
    rc = timeline_module.main(["--scenario", scenario], out=buf)
    assert rc == 0
    return buf.getvalue()


def test_tennent_output_contains_all_scene_dates(timeline_module) -> None:
    out = _run(timeline_module, "tennent")
    for d in TENNENT_DATES:
        assert d in out, f"missing Tennent scene date {d!r}"


def test_tennent_output_mentions_both_scene_ids_for_same_date_is_fine(timeline_module) -> None:
    out = _run(timeline_module, "tennent")
    # Real scene_ids from the dispatch
    for sid in (
        "tennent_20230702_umbra-01",
        "tennent_20230723_umbra-02",
        "tennent_20230807_umbra-03",
        "tennent_20230809_umbra-04",
        "tennent_20230813_umbra-05",
    ):
        assert sid in out


def test_whitsun_output_contains_scene_dates_and_ids(timeline_module) -> None:
    out = _run(timeline_module, "whitsun")
    for d in WHITSUN_DATES:
        assert d in out
    # The two 2023-12-06 scenes have distinct sensor suffixes
    assert "whitsun_20231206_umbra-04" in out
    assert "whitsun_20231206_umbra-07" in out
    assert "whitsun_20240320_umbra-05" in out


def test_output_has_top_and_uncertainty_lines_per_scene(timeline_module) -> None:
    out = _run(timeline_module, "tennent")
    # At least one "Top:" line per scene → 5 scenes for Tennent
    assert out.count("Top:") >= 5
    # Uncertainty reported to 2dp at least once per scene
    import re
    assert len(re.findall(r"uncertainty=\d\.\d{2}", out)) >= 5


def test_output_has_final_state_block_per_scenario(timeline_module) -> None:
    out = _run(timeline_module, "both")
    assert "=== Final state: tennent ===" in out
    assert "=== Final state: whitsun ===" in out


def test_output_is_deterministic_across_consecutive_invocations(timeline_module) -> None:
    """Two back-to-back invocations must produce byte-identical output."""
    out1 = _run(timeline_module, "both")
    out2 = _run(timeline_module, "both")
    assert out1 == out2


def test_tennent_top_lands_on_reclamation_or_construction(timeline_module) -> None:
    """By the end, narrative signals push Tennent toward fixed/construction."""
    out = _run(timeline_module, "tennent")
    # Final-state block lists scores sorted; the top hypothesis is one of
    # the two reclamation-family ids.
    tail = out.split("=== Final state: tennent ===", 1)[1]
    # Extract the "Top: <id>" line from the final-state block.
    top_line = next(
        line for line in tail.splitlines()
        if line.strip().startswith("Top:") and "<no evidence yet>" not in line
    )
    assert (
        "fixed_reclamation_or_structure" in top_line
        or "construction_or_reclamation_activity" in top_line
    ), top_line


def test_whitsun_top_lands_on_cluster_or_transient(timeline_module) -> None:
    out = _run(timeline_module, "whitsun")
    tail = out.split("=== Final state: whitsun ===", 1)[1]
    top_line = next(
        line for line in tail.splitlines()
        if line.strip().startswith("Top:") and "<no evidence yet>" not in line
    )
    assert (
        "vessel_cluster_activity" in top_line
        or "transient_anchorage_or_fishing_presence" in top_line
        or "ais_dark_or_poorly_observed_vessels" in top_line
    ), top_line


# ---------------------------------------------------------------------------
# Import-boundary sanity
# ---------------------------------------------------------------------------


def test_script_does_not_import_detection_or_gfw() -> None:
    """Timeline script must not pull heavy detector/ingest deps."""
    import ast
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    forbidden = ("custody.detection", "custody.ingest.gfw_presence")
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
    assert not offending, f"timeline script imports forbidden modules: {offending}"
