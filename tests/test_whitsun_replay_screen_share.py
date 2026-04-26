"""Tests for the Whitsun replay screen-share polish pass.

Pins the five narrative + ordinal corrections so a regression in any
of them shows up in CI before the next demo:

  * ev-04 summary references VLM (not Sentinel — Sentinel hasn't
    arrived yet at trace ordinal 4).
  * Operator review (ev-11) does not show the APPROVE record.
  * Human approval (ev-12) does show the APPROVE record.
  * Score breakdown at ev-10+ has a consolidated
    "Recommendation: SAT-B | score 0.91 | rank 1" callout with ASCII
    separators.
  * Decision-inspector body shows the human-readable summary only,
    not the developer-facing event_id / kind / timestamp rows.

No live HTTP, no Dash server is started.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
TRACE_PATH = REPO_ROOT / "data" / "demo" / "whitsun_decision_trace.fixture.json"


# ---------------------------------------------------------------------------
# Fixture-only checks: ev-04 summary
# ---------------------------------------------------------------------------


def test_event_04_summary_references_vlm_not_sentinel() -> None:
    """ev-04 (Candidate tracks initialized) sits BEFORE Sentinel
    cueing context arrives at ev-05/06.  The summary must not refer
    to Sentinel; tracks come from the VLM-derived detections off the
    Umbra tile."""
    payload = json.loads(TRACE_PATH.read_text(encoding="utf-8"))
    ev04 = next(ev for ev in payload["events"] if ev["ordinal"] == 4)
    summary = (ev04.get("summary") or "").lower()
    assert "vlm" in summary, summary
    assert "sentinel" not in summary, (
        f"ev-04 summary still references Sentinel before it has "
        f"arrived (ev-05/06): {summary!r}"
    )


# ---------------------------------------------------------------------------
# Callback rendering: human action review/approve split
# ---------------------------------------------------------------------------


def _import_replay_callbacks():
    src_app = REPO_ROOT / "src" / "app"
    src = REPO_ROOT / "src"
    for p in (str(src), str(src_app)):
        if p not in sys.path:
            sys.path.insert(0, p)
    for k in list(sys.modules):
        if k.startswith("layout.") or k.startswith("callbacks."):
            del sys.modules[k]
    import callbacks.whitsun_replay as mod
    return mod


def test_human_action_at_event_11_is_review_pending() -> None:
    """ev-11 (Operator reviews): the panel must NOT show the
    APPROVE badge.  It must show a review-pending state so the
    audience sees the dramatic beat."""
    rep = _import_replay_callbacks()
    rendered = str(rep._render_human_action(rep._TRACE, 11))
    assert "REVIEW IN PROGRESS" in rendered or "review in progress" in rendered.lower()
    assert "Awaiting operator approval" in rendered or "awaiting operator approval" in rendered.lower()
    assert "APPROVE" not in rendered, (
        f"ev-11 must not surface the APPROVE badge yet; got: {rendered}"
    )
    # Also: no decided_at line at ev-11 (that comes at ev-12).
    assert "decided_at" not in rendered


def test_human_action_at_event_12_shows_approve_record() -> None:
    """ev-12 (Human approves): full APPROVE record visible."""
    rep = _import_replay_callbacks()
    rendered = str(rep._render_human_action(rep._TRACE, 12))
    assert "APPROVE" in rendered
    assert "decided_at" in rendered
    # Operator + simulated badge survive.
    assert "operator-fixture-01" in rendered or "operator_id" in rendered


def test_human_action_at_event_10_still_hidden() -> None:
    """Before ev-11 nothing about the operator should appear."""
    rep = _import_replay_callbacks()
    rendered = str(rep._render_human_action(rep._TRACE, 10)).lower()
    assert "approve" not in rendered
    assert "review in progress" not in rendered


def test_reveal_ordinals_split_human_action() -> None:
    rep = _import_replay_callbacks()
    assert rep.REVEAL_ORDINALS["human_action_review"] == 11
    assert rep.REVEAL_ORDINALS["human_action"] == 12


# ---------------------------------------------------------------------------
# Recommendation callout in the score breakdown
# ---------------------------------------------------------------------------


def test_score_breakdown_at_event_10_has_recommendation_callout() -> None:
    rep = _import_replay_callbacks()
    rendered = str(rep._render_score_breakdown(rep._TRACE, 10))
    # Audience-facing label, not the internal opt-sat-b id.
    assert "Recommendation: SAT-B" in rendered, rendered
    # Numeric score (committed in the trace as 0.91).
    assert "0.91" in rendered
    # Rank surfaces explicitly.
    assert "rank 1" in rendered


def test_score_breakdown_callout_uses_ascii_separator() -> None:
    """Screen-share font fallback dictates ASCII pipes, not the
    Unicode middle-dot."""
    rep = _import_replay_callbacks()
    rendered = str(rep._render_score_breakdown(rep._TRACE, 10))
    assert " | " in rendered, "expected ASCII pipe separator"
    # The Unicode middle-dot should not appear in the consolidated
    # callout line.  (It may legitimately appear elsewhere on the
    # page — this test is scoped to the score breakdown only.)
    callout_zone = rendered.split("Recommendation:")[1].split("\n", 1)[0]
    assert "·" not in callout_zone


def test_score_breakdown_components_still_visible() -> None:
    """The new callout doesn't replace the breakdown — it sits on top."""
    rep = _import_replay_callbacks()
    rendered = str(rep._render_score_breakdown(rep._TRACE, 10))
    for component in (
        "ambiguity_resolution",
        "custody_health_improvement",
        "mission_relevance",
        "timeliness",
        "feasibility",
        "cost_penalty",
        "total_score",
    ):
        assert component in rendered, f"missing component {component}"


# ---------------------------------------------------------------------------
# Event-summary panel: dev cruft stripped
# ---------------------------------------------------------------------------


def test_event_summary_strips_event_id_kind_timestamp_rows() -> None:
    """Decision-inspector body must show only the human-readable
    summary; event_id / kind / timestamp belong in the JSON, not on
    screen for the demo audience."""
    rep = _import_replay_callbacks()
    ev = rep._TRACE.get_event("ev-03") or {}
    rendered = str(rep._render_event_summary(ev))
    # The summary text itself must remain.
    assert "VLM" in rendered or "vlm" in rendered.lower()
    # Developer-facing labelled rows must be gone.
    assert "event_id:" not in rendered
    assert "kind:" not in rendered
    # The literal label row "timestamp:" must be gone.  (The summary
    # text itself doesn't contain that token.)
    assert "timestamp:" not in rendered


def test_event_summary_empty_event_renders_placeholder() -> None:
    rep = _import_replay_callbacks()
    rendered = str(rep._render_event_summary({})).lower()
    assert "not available" in rendered


# ---------------------------------------------------------------------------
# Initial step counter is clean ASCII
# ---------------------------------------------------------------------------


def test_initial_step_counter_is_clean_ascii() -> None:
    """The default step counter rendered before any event click must
    not contain glyphs that screen-share fonts may mojibake."""
    src_app = REPO_ROOT / "src" / "app"
    if str(src_app) not in sys.path:
        sys.path.insert(0, str(src_app))
    for k in list(sys.modules):
        if k.startswith("layout."):
            del sys.modules[k]
    import layout.whitsun_replay as mod
    rendered = str(mod.build_whitsun_replay_layout())
    # The default "Step 01 / 14" text should appear; the em-dash
    # placeholder must not.
    assert "Step 01 / 14" in rendered
    assert "Step — / 14" not in rendered, (
        "em-dash placeholder leaked into the layout"
    )


def test_callback_step_counter_fallback_is_ascii() -> None:
    rep = _import_replay_callbacks()
    # Re-derive the step_counter via the trace loader; the callback
    # uses an ASCII fallback "Step -- / 14" when no event is
    # selected.  We can't easily fire the callback, so validate the
    # source code substring instead.
    src = (
        REPO_ROOT / "src" / "app" / "callbacks" / "whitsun_replay.py"
    ).read_text(encoding="utf-8")
    assert '"Step -- / 14"' in src
    assert '"Step — / 14"' not in src
