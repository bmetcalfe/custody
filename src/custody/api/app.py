"""FastAPI application stub (ADR-0021 Slice 17).

FastAPI is not in the project's dependencies.  This module defines a
``create_app()`` factory that raises :class:`RuntimeError` with a clear
message.  The service layer (``custody.api.service``) is fully
functional and testable without FastAPI -- call service functions
directly or via ``scripts/21_api_demo.py``.

If FastAPI is added in a future slice, this module can be expanded to
expose real HTTP routes that delegate to ``service.py``.
"""
from __future__ import annotations


_ROUTES: tuple[tuple[str, str], ...] = (
    ("GET", "/health"),
    ("POST", "/decision-packet"),
    ("POST", "/rank-collects"),
    ("POST", "/optimize-plan"),
    ("POST", "/evaluate-policies"),
    ("POST", "/planner-queue"),
    ("POST", "/portfolio-allocation"),
)


def create_app():
    """Create the FastAPI application.

    Raises :class:`RuntimeError` because FastAPI is not installed.  The
    service layer remains fully usable without an HTTP wrapper.
    """
    raise RuntimeError(
        "FastAPI is not installed.  The Custody decision API service "
        "layer is fully functional without it -- call "
        "custody.api.service functions directly or use "
        "scripts/21_api_demo.py.  To enable HTTP routes, add "
        "'fastapi' and 'uvicorn' to pyproject.toml dependencies."
    )


def list_routes() -> tuple[tuple[str, str], ...]:
    """Return the planned route table for documentation purposes."""
    return _ROUTES
