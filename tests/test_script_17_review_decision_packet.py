"""Tests for scripts/17_review_decision_packet.py (ADR-0021 Slice 13).

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
SCRIPT_PATH = REPO_ROOT / "scripts" / "17_review_decision_packet.py"


@pytest.fixture(scope="module")
def review_module():
    spec = importlib.util.spec_from_file_location(
        "script_17_review", SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["script_17_review"] = mod
    try:
        spec.loader.exec_module(mod)
        yield mod
    finally:
        sys.modules.pop("script_17_review", None)


def _run(review_module, *argv: str) -> str:
    buf = io.StringIO()
    rc = review_module.main(list(argv), out=buf)
    assert rc == 0
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Action coverage
# ---------------------------------------------------------------------------


def test_approve_runs_and_returns_zero(review_module) -> None:
    out = _run(
        review_module,
        "--scenario", "tennent",
        "--action", "approve",
        "--reason", "Best ambiguity reduction under budget",
        "--reviewed-at", "2026-04-24T12:00:00+00:00",
    )
    assert "HUMAN-IN-THE-LOOP REVIEW" in out
    assert "APPROVE" in out


def test_reject_runs_and_returns_zero(review_module) -> None:
    out = _run(
        review_module,
        "--scenario", "whitsun",
        "--action", "reject",
        "--reason", "Need stronger AIS coverage context",
        "--reviewed-at", "2026-04-24T12:00:00+00:00",
    )
    assert "REJECT" in out


def test_defer_runs_and_returns_zero(review_module) -> None:
    out = _run(
        review_module,
        "--scenario", "tennent",
        "--action", "defer",
        "--reason", "Wait for weather window",
        "--reviewed-at", "2026-04-24T12:00:00+00:00",
    )
    assert "DEFER" in out


def test_override_with_candidate_runs_and_returns_zero(review_module) -> None:
    out = _run(
        review_module,
        "--scenario", "whitsun",
        "--action", "override",
        "--override-candidate", "ais_coverage_query",
        "--reason", "AIS ambiguity is operationally decisive",
        "--reviewed-at", "2026-04-24T12:00:00+00:00",
    )
    assert "OVERRIDE" in out
    assert "Override candidates" in out
    assert "ais_coverage_query" in out


# ---------------------------------------------------------------------------
# Validation failures
# ---------------------------------------------------------------------------


def test_override_without_candidate_exits_nonzero(review_module) -> None:
    buf = io.StringIO()
    with pytest.raises((SystemExit, ValueError)):
        review_module.main(
            [
                "--scenario", "tennent",
                "--action", "override",
                "--reason", "ok",
                "--reviewed-at", "2026-04-24T12:00:00+00:00",
            ],
            out=buf,
        )


def test_empty_reason_raises(review_module) -> None:
    buf = io.StringIO()
    with pytest.raises((SystemExit, ValueError)):
        review_module.main(
            [
                "--scenario", "tennent",
                "--action", "approve",
                "--reason", "",
                "--reviewed-at", "2026-04-24T12:00:00+00:00",
            ],
            out=buf,
        )


# ---------------------------------------------------------------------------
# Format flags
# ---------------------------------------------------------------------------


def test_format_json_is_parseable(review_module) -> None:
    out = _run(
        review_module,
        "--scenario", "tennent",
        "--action", "approve",
        "--reason", "ok",
        "--reviewed-at", "2026-04-24T12:00:00+00:00",
        "--format", "json",
    )
    parsed = json.loads(out)
    assert parsed["action"] == "approve"
    assert parsed["scenario_id"] == "tennent"


def test_format_md_includes_h1(review_module) -> None:
    out = _run(
        review_module,
        "--scenario", "tennent",
        "--action", "approve",
        "--reason", "ok",
        "--reviewed-at", "2026-04-24T12:00:00+00:00",
        "--format", "md",
    )
    assert out.startswith("# Human-in-the-Loop Review - tennent")


# ---------------------------------------------------------------------------
# Ledger append
# ---------------------------------------------------------------------------


def test_ledger_path_appends_jsonl(review_module, tmp_path) -> None:
    ledger = tmp_path / "reviews.jsonl"
    out = _run(
        review_module,
        "--scenario", "tennent",
        "--action", "approve",
        "--reason", "ok",
        "--reviewed-at", "2026-04-24T12:00:00+00:00",
        "--ledger-path", str(ledger),
    )
    assert "Review record appended" in out
    assert ledger.exists()
    lines = ledger.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["action"] == "approve"


def test_two_sequential_appends_produce_two_lines(review_module, tmp_path) -> None:
    ledger = tmp_path / "reviews.jsonl"
    _run(
        review_module,
        "--scenario", "tennent",
        "--action", "approve",
        "--reason", "first",
        "--reviewed-at", "2026-04-24T12:00:00+00:00",
        "--ledger-path", str(ledger),
    )
    _run(
        review_module,
        "--scenario", "whitsun",
        "--action", "reject",
        "--reason", "second",
        "--reviewed-at", "2026-04-24T13:00:00+00:00",
        "--ledger-path", str(ledger),
    )
    lines = ledger.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["action"] == "approve"
    assert parsed[0]["scenario_id"] == "tennent"
    assert parsed[1]["action"] == "reject"
    assert parsed[1]["scenario_id"] == "whitsun"


# ---------------------------------------------------------------------------
# Section headers + audit fields
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "needle",
    [
        "HUMAN-IN-THE-LOOP REVIEW",
        "Operator action",
        "Human reason",
        "review_id",
        "packet_hash",
        "reviewed_at",
    ],
)
def test_text_output_contains_required_block(review_module, needle: str) -> None:
    out = _run(
        review_module,
        "--scenario", "tennent",
        "--action", "approve",
        "--reason", "ok",
        "--reviewed-at", "2026-04-24T12:00:00+00:00",
    )
    assert needle in out


def test_caveat_explicitly_mentions_no_live_tasking(review_module) -> None:
    out = _run(
        review_module,
        "--scenario", "tennent",
        "--action", "approve",
        "--reason", "ok",
        "--reviewed-at", "2026-04-24T12:00:00+00:00",
    ).lower()
    assert "review record only" in out
    assert "no live tasking or sensor command was issued" in out


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_output_is_deterministic_with_fixed_reviewed_at(review_module) -> None:
    out1 = _run(
        review_module,
        "--scenario", "tennent",
        "--action", "approve",
        "--reason", "ok",
        "--reviewed-at", "2026-04-24T12:00:00+00:00",
    )
    out2 = _run(
        review_module,
        "--scenario", "tennent",
        "--action", "approve",
        "--reason", "ok",
        "--reviewed-at", "2026-04-24T12:00:00+00:00",
    )
    assert out1 == out2


# ---------------------------------------------------------------------------
# Forbidden-language scan (excluding the explicit non-claim caveat)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "needle",
    [
        "tasking order", "production scheduler",
        "Sentinel integration", "collection order",
    ],
)
def test_output_does_not_contain_forbidden_phrase(review_module, needle: str) -> None:
    out = _run(
        review_module,
        "--scenario", "tennent",
        "--action", "approve",
        "--reason", "ok",
        "--reviewed-at", "2026-04-24T12:00:00+00:00",
    )
    assert needle.lower() not in out.lower()


def test_live_tasking_and_sensor_command_only_appear_in_non_claim_caveat(review_module) -> None:
    """The phrases 'live tasking' and 'sensor command' only appear inside
    the explicit non-claim caveat: 'no live tasking or sensor command was issued'."""
    out = _run(
        review_module,
        "--scenario", "tennent",
        "--action", "approve",
        "--reason", "ok",
        "--reviewed-at", "2026-04-24T12:00:00+00:00",
    )
    safe = out.lower().replace(
        "no live tasking or sensor command was issued", "",
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
        f"review script imports forbidden modules: {offending}"
    )
