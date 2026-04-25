"""Tests for scripts/21_api_demo.py (ADR-0021 Slice 17).

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
SCRIPT_PATH = REPO_ROOT / "scripts" / "21_api_demo.py"


@pytest.fixture(scope="module")
def demo_module():
    spec = importlib.util.spec_from_file_location(
        "script_21_api_demo", SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["script_21_api_demo"] = mod
    try:
        spec.loader.exec_module(mod)
        yield mod
    finally:
        sys.modules.pop("script_21_api_demo", None)


def _run(demo_module, *argv: str) -> str:
    buf = io.StringIO()
    rc = demo_module.main(list(argv), out=buf)
    assert rc == 0
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Return codes per endpoint
# ---------------------------------------------------------------------------


def test_main_returns_zero_for_health(demo_module) -> None:
    out = _run(demo_module, "--endpoint", "health")
    assert out


def test_main_returns_zero_for_decision_packet(demo_module) -> None:
    out = _run(
        demo_module, "--endpoint", "decision-packet",
        "--scenario", "tennent",
    )
    assert out


def test_main_returns_zero_for_optimize_plan(demo_module) -> None:
    out = _run(
        demo_module, "--endpoint", "optimize-plan",
        "--scenario", "whitsun",
    )
    assert out


def test_main_returns_zero_for_portfolio_allocation(demo_module) -> None:
    out = _run(
        demo_module, "--endpoint", "portfolio-allocation",
        "--scenario", "both",
    )
    assert out


# ---------------------------------------------------------------------------
# Format handling
# ---------------------------------------------------------------------------


def test_format_json_produces_parseable_json(demo_module) -> None:
    out = _run(
        demo_module, "--endpoint", "health", "--format", "json",
    )
    parsed = json.loads(out)
    assert parsed["status"] == "ok"
    assert parsed["endpoint"] == "/health"


def test_format_json_for_decision_packet(demo_module) -> None:
    out = _run(
        demo_module, "--endpoint", "decision-packet",
        "--scenario", "tennent", "--format", "json",
    )
    parsed = json.loads(out)
    assert parsed["payload"]["scenario_id"] == "tennent"


# ---------------------------------------------------------------------------
# Output content
# ---------------------------------------------------------------------------


def test_output_includes_prototype_caveat(demo_module) -> None:
    out = _run(demo_module, "--endpoint", "health").lower()
    assert "local prototype api only" in out


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_output_is_deterministic(demo_module) -> None:
    out1 = _run(
        demo_module, "--endpoint", "decision-packet",
        "--scenario", "tennent", "--format", "json",
    )
    out2 = _run(
        demo_module, "--endpoint", "decision-packet",
        "--scenario", "tennent", "--format", "json",
    )
    assert out1 == out2


# ---------------------------------------------------------------------------
# Forbidden language
# ---------------------------------------------------------------------------


_FORBIDDEN = (
    "tasking order",
    "production scheduler",
    "sentinel integration",
    "autonomous constellation management",
    "revenue dollars",
    "production api",
    "zero-trust",
)


def test_output_no_forbidden_language(demo_module) -> None:
    """Forbidden tokens not in the explicit non-claim disclaimer list."""
    out = _run(
        demo_module, "--endpoint", "portfolio-allocation",
        "--scenario", "both", "--format", "json",
    ).lower()
    safe = out.replace(
        "no live tasking or sensor command is issued", "",
    )
    for needle in _FORBIDDEN:
        assert needle not in safe, (
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
        f"api-demo script imports forbidden modules: {offending}"
    )
