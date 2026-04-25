"""Tests for scripts/18_planner_queue.py (ADR-0021 Slice 14).

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
SCRIPT_PATH = REPO_ROOT / "scripts" / "18_planner_queue.py"


@pytest.fixture(scope="module")
def queue_module():
    spec = importlib.util.spec_from_file_location(
        "script_18_queue", SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["script_18_queue"] = mod
    try:
        spec.loader.exec_module(mod)
        yield mod
    finally:
        sys.modules.pop("script_18_queue", None)


def _run(queue_module, *argv: str) -> str:
    buf = io.StringIO()
    rc = queue_module.main(list(argv), out=buf)
    assert rc == 0
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Return codes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", ["tennent", "whitsun", "both"])
def test_main_returns_zero_for_each_scenario(queue_module, scenario: str) -> None:
    out = _run(queue_module, "--scenario", scenario)
    assert out


def test_main_default_runs_both_scenarios(queue_module) -> None:
    out = _run(queue_module).lower()
    assert "tennent" in out
    assert "whitsun" in out


# ---------------------------------------------------------------------------
# Required section headers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "header",
    [
        "PLANNER WORK QUEUE",
        "Generated at:",
        "Ranked work items",
        "Status:",
        "Priority:",
        "Queue summary",
        "Caveats",
    ],
)
def test_text_output_contains_required_block(queue_module, header: str) -> None:
    out = _run(queue_module, "--scenario", "both")
    assert header in out, f"missing block {header!r}"


def test_caveat_explicitly_disclaims_live_tasking(queue_module) -> None:
    out = _run(queue_module, "--scenario", "both").lower()
    assert "decision support only" in out
    assert "no live tasking or sensor command is issued" in out


# ---------------------------------------------------------------------------
# Format flags
# ---------------------------------------------------------------------------


def test_format_json_is_parseable(queue_module) -> None:
    out = _run(queue_module, "--scenario", "both", "--format", "json")
    parsed = json.loads(out)
    assert "items" in parsed
    assert isinstance(parsed["items"], list)
    assert len(parsed["items"]) == 2


def test_format_md_includes_h1(queue_module) -> None:
    out = _run(queue_module, "--scenario", "both", "--format", "md")
    assert out.startswith("# Planner Work Queue\n")


# ---------------------------------------------------------------------------
# --review-ledger
# ---------------------------------------------------------------------------


def _build_approve_jsonl_for_tennent(tmp_path) -> Path:
    """Build a tiny JSONL ledger by invoking scripts/17 once with
    --ledger-path, so the queue picks up an APPROVE record for Tennent."""
    review_script = REPO_ROOT / "scripts" / "17_review_decision_packet.py"
    spec = importlib.util.spec_from_file_location(
        "script_17_review_for_q_test", review_script,
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["script_17_review_for_q_test"] = mod
    try:
        spec.loader.exec_module(mod)
        ledger = tmp_path / "reviews.jsonl"
        rc = mod.main(
            [
                "--scenario", "tennent",
                "--action", "approve",
                "--reason", "ok",
                "--reviewed-at", "2026-04-24T12:00:00+00:00",
                "--ledger-path", str(ledger),
            ],
            out=io.StringIO(),
        )
        assert rc == 0
    finally:
        sys.modules.pop("script_17_review_for_q_test", None)
    return ledger


def test_review_ledger_changes_tennent_status_to_approved(
    queue_module, tmp_path,
) -> None:
    ledger = _build_approve_jsonl_for_tennent(tmp_path)
    parsed = json.loads(_run(
        queue_module, "--scenario", "both",
        "--review-ledger", str(ledger),
        "--format", "json",
    ))
    by_scenario = {i["scenario_id"]: i for i in parsed["items"]}
    assert by_scenario["tennent"]["status"] == "approved"
    # Whitsun has no review record -> remains pending_review (saturated AMBIGUOUS).
    assert by_scenario["whitsun"]["status"] == "pending_review"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_output_is_deterministic_across_two_calls(queue_module) -> None:
    out1 = _run(queue_module, "--scenario", "both")
    out2 = _run(queue_module, "--scenario", "both")
    assert out1 == out2


# ---------------------------------------------------------------------------
# Forbidden language
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "needle",
    [
        "tasking order", "production scheduler",
        "Sentinel integration", "collection order",
        "revenue dollars", "autonomous execution",
    ],
)
def test_output_does_not_contain_forbidden_phrase(queue_module, needle: str) -> None:
    out = _run(queue_module, "--scenario", "both").lower()
    assert needle.lower() not in out


def test_live_tasking_and_sensor_command_only_appear_in_caveat(queue_module) -> None:
    out = _run(queue_module, "--scenario", "both").lower()
    safe = out.replace(
        "no live tasking or sensor command is issued", "",
    )
    assert "live tasking" not in safe
    assert "sensor command" not in safe


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
    assert not offending, (
        f"queue script imports forbidden modules: {offending}"
    )
