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
    """ev-12 (Human approves): APPROVE badge + audience-facing
    sentence (operator id, option label, friendly UTC stamp)."""
    rep = _import_replay_callbacks()
    rendered = str(rep._render_human_action(rep._TRACE, 12))
    assert "APPROVE" in rendered
    # Audience-facing sentence form, not raw key:value rows.
    assert "Operator operator-fixture-01 approves SAT-B" in rendered
    assert "12 Dec 2023 | 04:35 UTC" in rendered
    # The legacy raw labelled rows must not appear in the new render.
    assert "decided_at:" not in rendered
    assert "tasking_option_id:" not in rendered


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


# ---------------------------------------------------------------------------
# Header polish: drop kind, friendly UTC timestamp
# ---------------------------------------------------------------------------


def test_header_drops_kind_label() -> None:
    """Audience-facing header must not surface the internal ``kind``
    enum string."""
    rep = _import_replay_callbacks()
    for ev_id in ("ev-01", "ev-05", "ev-10", "ev-13"):
        ev = rep._TRACE.get_event(ev_id) or {}
        rendered = str(rep._render_header(ev))
        assert "kind:" not in rendered, (
            f"{ev_id} header still renders kind: prefix"
        )


def test_header_uses_friendly_utc_timestamp() -> None:
    """Header timestamp must read like ``06 Dec 2023 | 10:35 UTC``,
    not the raw ISO string."""
    rep = _import_replay_callbacks()
    ev = rep._TRACE.get_event("ev-03") or {}
    rendered = str(rep._render_header(ev))
    assert "06 Dec 2023 | 10:35 UTC" in rendered
    # The raw ISO timestamp must not appear in the audience header.
    assert "2023-12-06T10:35:00+00:00" not in rendered


def test_format_friendly_utc_helper() -> None:
    rep = _import_replay_callbacks()
    assert rep._format_friendly_utc("2023-12-13T10:35:00+00:00") == (
        "13 Dec 2023 | 10:35 UTC"
    )
    assert rep._format_friendly_utc("2023-12-12T04:35:00Z") == (
        "12 Dec 2023 | 04:35 UTC"
    )
    # Falsy / unparseable input falls back gracefully.
    assert rep._format_friendly_utc("") == ""
    assert rep._format_friendly_utc(None) == ""


# ---------------------------------------------------------------------------
# Custody state card
# ---------------------------------------------------------------------------


def test_custody_state_card_hidden_before_first_snapshot() -> None:
    """ev-01 / ev-02 sit before cs-snap-001 reveals; the card must
    surface the placeholder, not a phantom 0.00 score."""
    rep = _import_replay_callbacks()
    for ord_ in (1, 2):
        rendered = str(rep._render_custody_state(rep._TRACE, ord_)).lower()
        assert "not been observed" in rendered or "not available" in rendered


def test_custody_state_card_shows_healthy_at_event_3() -> None:
    """cs-snap-001 reveals at ev-03 (added to refs in this slice)."""
    rep = _import_replay_callbacks()
    rendered = str(rep._render_custody_state(rep._TRACE, 3))
    assert "HEALTHY" in rendered
    assert "0.78" in rendered


def test_custody_state_card_shows_ambiguous_at_event_7() -> None:
    rep = _import_replay_callbacks()
    rendered = str(rep._render_custody_state(rep._TRACE, 7))
    assert "AMBIGUOUS" in rendered
    assert "0.42" in rendered
    # Delta vs the previous healthy state must surface.
    assert "-0.36" in rendered


def test_custody_state_card_shows_full_arc_at_event_13() -> None:
    """At ev-13 the outcome lands; the card should show all three
    snapshots (HEALTHY 0.78, AMBIGUOUS 0.42, HEALTHY 0.81) in arc
    order."""
    rep = _import_replay_callbacks()
    rendered = str(rep._render_custody_state(rep._TRACE, 13))
    # All three states shown.
    assert rendered.count("HEALTHY") >= 2  # initial + recovered
    assert "AMBIGUOUS" in rendered
    # All three scores.
    for score in ("0.78", "0.42", "0.81"):
        assert score in rendered, f"missing score {score}"
    # Recovery delta vs ambiguous baseline.
    assert "+0.39" in rendered


# ---------------------------------------------------------------------------
# Approval timestamp + sentence form
# ---------------------------------------------------------------------------


def test_human_action_decided_at_matches_ev12_split() -> None:
    """The trace's human_action.decided_at must match ev-12's
    timestamp (04:35) after the review/approval split."""
    payload = json.loads(TRACE_PATH.read_text(encoding="utf-8"))
    ev12 = next(ev for ev in payload["events"] if ev["ordinal"] == 12)
    ha = payload["human_action"]
    assert ha["decided_at"] == ev12["timestamp"]
    assert ha["decided_at"] == "2023-12-12T04:35:00+00:00"


def test_human_action_renders_sat_b_label_not_internal_id() -> None:
    rep = _import_replay_callbacks()
    rendered = str(rep._render_human_action(rep._TRACE, 12))
    # Audience-facing label.
    assert "SAT-B" in rendered
    # Internal id must not leak into the rendered tree.
    assert "opt-sat-b" not in rendered


# ---------------------------------------------------------------------------
# Outcome panel: prose form with explicit before/after
# ---------------------------------------------------------------------------


def test_outcome_renders_as_audience_sentence() -> None:
    rep = _import_replay_callbacks()
    rendered = str(rep._render_outcome(rep._TRACE, 13))
    # Sentence form with track id + before/after scores.
    assert "Follow-up Umbra collection reacquires trk-002" in rendered
    assert "0.42" in rendered
    assert "0.81" in rendered
    # Realised delta still visible.
    assert "+0.39" in rendered
    # The legacy labelled rows must be gone.
    assert "track_id:" not in rendered
    assert "recorded_at:" not in rendered
    assert "custody_score_delta:" not in rendered


# ---------------------------------------------------------------------------
# Follow-up recommendation panel: sentence form
# ---------------------------------------------------------------------------


def test_followup_renders_as_audience_sentence() -> None:
    rep = _import_replay_callbacks()
    rendered = str(rep._render_followup(rep._TRACE, 14))
    assert "Recommended next step:" in rendered
    # Trace's summary text already contains "72 hours" and "0.55".
    assert "72 hours" in rendered
    assert "0.55" in rendered
    # The legacy review-thresholds key:value row must be gone.
    assert "review thresholds:" not in rendered
    assert "custody_score_minimum=" not in rendered
    assert "ais_dark_persistence_hours=" not in rendered


# ---------------------------------------------------------------------------
# Counterfactuals: chosen-row addition
# ---------------------------------------------------------------------------


def test_counterfactuals_top_row_is_chosen_sat_b() -> None:
    rep = _import_replay_callbacks()
    rendered = str(rep._render_counterfactuals(rep._TRACE, 14))
    # CHOSEN badge + SAT-B + realised +0.39 above the rejected
    # alternatives.
    assert "CHOSEN" in rendered
    assert "SAT-B" in rendered
    chosen_idx = rendered.find("CHOSEN")
    rejected_idx = rendered.find("opt-sat-a") if "opt-sat-a" in rendered \
        else rendered.find("SAT-A")
    if rejected_idx == -1:
        # Counterfactual table relabels via _option_label_for; check
        # that one of the rejected option labels appears below the
        # CHOSEN row.
        for label in ("SAT-A", "Wait", "Optical", "Expand Search"):
            label_idx = rendered.find(label)
            if label_idx > 0:
                rejected_idx = label_idx
                break
    assert rejected_idx > chosen_idx, (
        "CHOSEN row must render above the rejected alternatives"
    )
    # Realised delta (+0.39) is the headline number.
    assert "+0.39" in rendered


def test_counterfactual_caption_explains_chosen_row() -> None:
    rep = _import_replay_callbacks()
    rendered = str(rep._render_counterfactuals(rep._TRACE, 14))
    assert "CHOSEN row" in rendered or "realised" in rendered.lower()


# ---------------------------------------------------------------------------
# Score breakdown: drop muted opt-sat-b line
# ---------------------------------------------------------------------------


def test_score_breakdown_drops_muted_selected_option_line() -> None:
    """The "selected option: opt-sat-b" muted line is now redundant
    with the new "Recommendation: SAT-B" callout and must not
    render."""
    rep = _import_replay_callbacks()
    rendered = str(rep._render_score_breakdown(rep._TRACE, 10))
    assert "selected option: opt-sat-b" not in rendered
    # The audience-facing callout still surfaces the same fact.
    assert "Recommendation: SAT-B" in rendered


# ---------------------------------------------------------------------------
# Evidence viewer: 3-vs-115 detection promotion note
# ---------------------------------------------------------------------------


def test_evidence_viewer_includes_promotion_note_at_event_03() -> None:
    """At ev-03 the annotated chip surfaces 115 raw VLM detections;
    the trace pins 3 of them as candidate-track seeds.  A short note
    must explain the split so the audience isn't confused."""
    src_app = REPO_ROOT / "src" / "app"
    src = REPO_ROOT / "src"
    for p in (str(src), str(src_app)):
        if p not in sys.path:
            sys.path.insert(0, p)
    for k in list(sys.modules):
        if k.startswith("layout.") or k.startswith("callbacks."):
            del sys.modules[k]
    import callbacks.evidence_viewer as ev_cb

    scene = ev_cb.evidence_scene_for_whitsun_event_ordinal(
        ev_cb._SCENES, 3,
    )
    assert scene is not None and scene.scenario_id == "whitsun"
    # Build the detection_block as the callback would.
    promoted = len(ev_cb._TRACE.detections or ())
    expected_note = (
        f"{int(scene.detection_count)} raw VLM detections; "
        f"{promoted} promoted to candidate tracks for custody planning."
    )
    # The full Whitsun callback isn't directly callable; verify the
    # source code constructs that exact note prefix.
    src_text = (
        REPO_ROOT / "src" / "app" / "callbacks" / "evidence_viewer.py"
    ).read_text(encoding="utf-8")
    assert "raw VLM detections" in src_text
    assert "promoted to candidate tracks" in src_text
    assert "custody planning" in src_text
    # And confirm the inputs that would fill the note: the scene has
    # 115 detections and the trace has 3 promoted seeds.
    assert int(scene.detection_count) == 115
    assert promoted == 3
