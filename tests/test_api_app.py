"""Tests for :mod:`custody.api.app` (ADR-0021 Slice 17).

The app module is a stub that raises ``RuntimeError`` from
``create_app()`` because FastAPI is not installed.  These tests pin
that contract and verify no business logic leaked into ``app.py``.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest


def test_app_module_imports_safely() -> None:
    import custody.api.app  # noqa: F401


def test_create_app_raises_runtime_error() -> None:
    from custody.api.app import create_app
    with pytest.raises(RuntimeError) as excinfo:
        create_app()
    assert "FastAPI" in str(excinfo.value)


def test_list_routes_returns_expected_routes() -> None:
    from custody.api.app import list_routes
    routes = list_routes()
    paths = {path for _method, path in routes}
    assert "/health" in paths
    assert "/decision-packet" in paths
    assert "/rank-collects" in paths
    assert "/optimize-plan" in paths
    assert "/evaluate-policies" in paths
    assert "/planner-queue" in paths
    assert "/portfolio-allocation" in paths


def test_app_has_no_business_logic() -> None:
    """Static scan: app.py must not call hypothesis-layer pipeline functions."""
    import custody.api.app as mod
    tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
    forbidden_calls = {
        "assess_custody_health",
        "rank_collection_candidates",
        "optimize_collection_plan",
        "attribute_mission_value",
        "simulate_counterfactual_collects",
        "evaluate_collection_policies",
        "create_queue_item",
        "build_planner_queue",
        "allocate_portfolio",
        "build_decision_packet",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "id", None)
            if name in forbidden_calls:
                pytest.fail(
                    f"app.py contains business-logic call to {name!r}; "
                    "all logic belongs in service.py"
                )
