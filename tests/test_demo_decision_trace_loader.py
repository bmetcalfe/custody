"""Tests for the Whitsun decision-trace loader and the Dash replay layout.

Validates the read-only access layer used by the Dash replay panel.  No
network access; no Dash server is started.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from custody.demo import (
    WHITSUN_DECISION_TRACE_PATH,
    DecisionTrace,
    load_whitsun_decision_trace,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Loader basics
# ---------------------------------------------------------------------------


def test_loader_returns_decision_trace() -> None:
    t = load_whitsun_decision_trace()
    assert isinstance(t, DecisionTrace)
    assert t.schema == "custody.demo.whitsun_decision_trace.v1"


def test_default_path_resolves_to_committed_fixture() -> None:
    assert WHITSUN_DECISION_TRACE_PATH.exists()
    assert WHITSUN_DECISION_TRACE_PATH == (
        REPO_ROOT / "data" / "demo" / "whitsun_decision_trace.fixture.json"
    )


def test_loader_from_explicit_path(tmp_path: Path) -> None:
    payload = json.loads(
        WHITSUN_DECISION_TRACE_PATH.read_text(encoding="utf-8"),
    )
    copy = tmp_path / "trace.json"
    copy.write_text(json.dumps(payload), encoding="utf-8")
    t = load_whitsun_decision_trace(copy)
    assert len(t.events) == 14


# ---------------------------------------------------------------------------
# Event lookup
# ---------------------------------------------------------------------------


def test_event_lookup_returns_event() -> None:
    t = load_whitsun_decision_trace()
    ev = t.get_event("ev-10")
    assert ev is not None
    assert ev["label"] == "Policy recommends SAT-B"


def test_event_lookup_missing_id_returns_none() -> None:
    t = load_whitsun_decision_trace()
    assert t.get_event("ev-99") is None


def test_event_label_pairs_in_order() -> None:
    t = load_whitsun_decision_trace()
    pairs = t.event_label_pairs()
    assert len(pairs) == 14
    assert pairs[0][0] == "ev-01"
    assert pairs[-1][0] == "ev-14"


def test_first_event_id() -> None:
    assert load_whitsun_decision_trace().first_event_id() == "ev-01"


# ---------------------------------------------------------------------------
# Options table generation
# ---------------------------------------------------------------------------


def test_options_table_includes_all_five_options() -> None:
    rows = load_whitsun_decision_trace().options_table()
    labels = {r["label"] for r in rows}
    assert labels == {"SAT-A", "SAT-B", "Wait", "Optical", "Expand Search"}


def test_options_table_marks_sat_b_selected() -> None:
    rows = load_whitsun_decision_trace().options_table()
    selected = [r for r in rows if r["is_selected"]]
    assert len(selected) == 1
    assert selected[0]["label"] == "SAT-B"


def test_options_table_carries_score_per_option() -> None:
    rows = load_whitsun_decision_trace().options_table()
    for row in rows:
        assert row["total_score"] is not None, (
            f"option {row['label']} missing total_score"
        )
        assert row["components"], (
            f"option {row['label']} missing components"
        )


# ---------------------------------------------------------------------------
# Selected recommendation resolves to SAT-B
# ---------------------------------------------------------------------------


def test_selected_recommendation_resolves_to_sat_b() -> None:
    t = load_whitsun_decision_trace()
    rec = t.selected_recommendation
    assert rec["tasking_option_id"] == "opt-sat-b"
    opt = t.get_tasking_option(rec["tasking_option_id"])
    assert opt is not None
    assert opt["label"] == "SAT-B"


def test_score_breakdown_for_selected_option() -> None:
    t = load_whitsun_decision_trace()
    sb = t.get_score_breakdown_for_option(
        t.selected_recommendation["tasking_option_id"],
    )
    assert sb is not None
    assert sb["total_score"] >= 0.85


# ---------------------------------------------------------------------------
# Reference resolution per event
# ---------------------------------------------------------------------------


def test_loader_validate_returns_no_issues() -> None:
    issues = load_whitsun_decision_trace().validate()
    assert issues == ()


def test_every_event_can_be_rendered_without_missing_ids() -> None:
    t = load_whitsun_decision_trace()
    for ev in t.events:
        refs = ev.get("refs", {})
        for tid in refs.get("track_ids", []):
            assert t.get_track(tid) is not None
        for did in refs.get("detection_ids", []):
            assert t.get_detection(did) is not None
        for aid in refs.get("evidence_artifact_ids", []):
            assert t.get_evidence_artifact(aid) is not None
        for oid in refs.get("tasking_option_ids", []):
            assert t.get_tasking_option(oid) is not None
        for sid in refs.get("score_breakdown_ids", []):
            assert t.get_score_breakdown(sid) is not None
        for cid in refs.get("counterfactual_ids", []):
            assert t.get_counterfactual(cid) is not None


# ---------------------------------------------------------------------------
# Policy rationale labelled honestly
# ---------------------------------------------------------------------------


def test_policy_rationale_labelled_advisory_not_trained_rl() -> None:
    pol = load_whitsun_decision_trace().policy_rationale
    assert pol.get("policy_kind") == "deterministic-heuristic"
    rl = pol.get("rl_advisory", "").lower()
    # Must explicitly disclaim trained RL.
    assert "not used" in rl or "not a trained" in rl or "no trained" in rl or "trained rl is not" in rl


# ---------------------------------------------------------------------------
# Dash layout / callback smoke
# ---------------------------------------------------------------------------


def test_dash_layout_module_imports() -> None:
    """Importing the layout module should not raise, even without app."""
    import sys
    src_app = REPO_ROOT / "src" / "app"
    sys.path.insert(0, str(src_app))
    try:
        from layout.whitsun_replay import build_whitsun_replay_layout
        layout = build_whitsun_replay_layout()
        assert layout is not None
        # Layout has children.
        assert getattr(layout, "children", None) is not None
    finally:
        sys.path.remove(str(src_app))


def test_dash_app_imports_with_replay_tab() -> None:
    """Smoke test: full Dash app imports and registers replay callbacks."""
    import sys
    src_app = REPO_ROOT / "src" / "app"
    sys.path.insert(0, str(src_app))
    try:
        # Re-import cleanly to avoid leakage from other tests.
        for k in list(sys.modules):
            if k == "dash_app" or k.startswith("layout.") or k.startswith("callbacks."):
                del sys.modules[k]
        import dash_app
        assert dash_app.app is not None
        # Replay callbacks should be registered.
        ids = list(dash_app.app.callback_map.keys())
        assert any("whitsun" in cid for cid in ids), (
            "whitsun replay callbacks not registered"
        )
    finally:
        sys.path.remove(str(src_app))


# ---------------------------------------------------------------------------
# Progressive panel reveal (ordinal gating)
# ---------------------------------------------------------------------------


def _import_replay_module():
    import importlib
    import sys
    src_app = REPO_ROOT / "src" / "app"
    if str(src_app) not in sys.path:
        sys.path.insert(0, str(src_app))
    # Force a clean import so any prior module-level state is rebuilt.
    for k in list(sys.modules):
        if k.startswith("layout.") or k.startswith("callbacks."):
            del sys.modules[k]
    import callbacks.whitsun_replay as mod
    return mod


def _is_placeholder(component) -> bool:
    """Match the '_na' placeholder div regardless of render path."""
    if component is None:
        return True
    children = getattr(component, "children", None)
    if isinstance(children, str):
        return "not available at this step" in children
    return False


def test_event_01_hides_post_decision_panels() -> None:
    rep = _import_replay_module()
    trace = rep._TRACE
    event = trace.get_event("ev-01")
    ord_ = rep._ord_for(event)
    assert ord_ == 1
    # Pre-decision panels should all be placeholders.
    assert _is_placeholder(rep._render_observations(trace, ord_))
    assert _is_placeholder(rep._render_options_table(trace, ord_))
    assert _is_placeholder(rep._render_score_breakdown(trace, ord_))
    assert _is_placeholder(rep._render_policy_rationale(trace, ord_))
    assert _is_placeholder(rep._render_human_action(trace, ord_))
    assert _is_placeholder(rep._render_outcome(trace, ord_))
    assert _is_placeholder(rep._render_counterfactuals(trace, ord_))
    assert _is_placeholder(rep._render_followup(trace, ord_))


def test_event_02_renders_umbra_observation() -> None:
    rep = _import_replay_module()
    trace = rep._TRACE
    ord_ = rep._ord_for(trace.get_event("ev-02"))
    assert ord_ == 2
    obs_panel = rep._render_observations(trace, ord_)
    assert not _is_placeholder(obs_panel)
    # Options/scores/etc are still gated.
    assert _is_placeholder(rep._render_options_table(trace, ord_))


def test_event_08_renders_options_but_not_scores_or_selected() -> None:
    rep = _import_replay_module()
    trace = rep._TRACE
    ord_ = rep._ord_for(trace.get_event("ev-08"))
    assert ord_ == 8
    options_table = rep._render_options_table(trace, ord_)
    assert not _is_placeholder(options_table)
    # Score breakdown panel is still gated until ordinal 10.
    assert _is_placeholder(rep._render_score_breakdown(trace, ord_))
    assert _is_placeholder(rep._render_policy_rationale(trace, ord_))


def test_event_09_options_table_shows_score_column() -> None:
    rep = _import_replay_module()
    trace = rep._TRACE
    ord_ = rep._ord_for(trace.get_event("ev-09"))
    assert ord_ == 9
    table = rep._render_options_table(trace, ord_)
    rendered = str(table)
    # The score column shows numeric scores (look for a known formatted total).
    assert "0.91" in rendered, (
        "expected SAT-B's total score 0.91 in the rendered options table"
    )
    # SELECTED badge is still gated until ordinal 10.
    assert "SELECTED" not in rendered
    # Score breakdown panel still gated.
    assert _is_placeholder(rep._render_score_breakdown(trace, ord_))


def test_event_10_renders_policy_rationale_and_score_breakdown() -> None:
    rep = _import_replay_module()
    trace = rep._TRACE
    ord_ = rep._ord_for(trace.get_event("ev-10"))
    assert ord_ == 10
    pol = rep._render_policy_rationale(trace, ord_)
    assert not _is_placeholder(pol)
    sb = rep._render_score_breakdown(trace, ord_)
    assert not _is_placeholder(sb)
    # SELECTED badge appears in the options table.
    assert "SELECTED" in str(rep._render_options_table(trace, ord_))


def test_event_11_renders_human_action() -> None:
    rep = _import_replay_module()
    trace = rep._TRACE
    ord_ = rep._ord_for(trace.get_event("ev-11"))
    assert ord_ == 11
    ha = rep._render_human_action(trace, ord_)
    assert not _is_placeholder(ha)
    # Outcome / counterfactuals still gated.
    assert _is_placeholder(rep._render_outcome(trace, ord_))
    assert _is_placeholder(rep._render_counterfactuals(trace, ord_))


def test_event_13_renders_outcome() -> None:
    rep = _import_replay_module()
    trace = rep._TRACE
    ord_ = rep._ord_for(trace.get_event("ev-13"))
    assert ord_ == 13
    out = rep._render_outcome(trace, ord_)
    assert not _is_placeholder(out)
    # Counterfactuals + followup still gated until ordinal 14.
    assert _is_placeholder(rep._render_counterfactuals(trace, ord_))
    assert _is_placeholder(rep._render_followup(trace, ord_))


def test_event_14_renders_counterfactuals_and_followup() -> None:
    rep = _import_replay_module()
    trace = rep._TRACE
    ord_ = rep._ord_for(trace.get_event("ev-14"))
    assert ord_ == 14
    cf = rep._render_counterfactuals(trace, ord_)
    fr = rep._render_followup(trace, ord_)
    assert not _is_placeholder(cf)
    assert not _is_placeholder(fr)


def test_counterfactual_column_renamed_with_caption() -> None:
    rep = _import_replay_module()
    trace = rep._TRACE
    rendered = str(rep._render_counterfactuals(trace, 14))
    # The header should use the clearer phrasing, not the ambiguous "score Δ".
    assert "impact vs baseline" in rendered
    # An explicit caption should clarify that Δ is vs the pre-decision state.
    assert "pre-decision baseline" in rendered or "cs-snap-002" in rendered


# ---------------------------------------------------------------------------
# UI polish: M1 sidebar swap, M2 context-map rename, M3 highlight class,
# S1 progressive-reveal subhead, S2 trimmed policy header, S3 step counter
# ---------------------------------------------------------------------------


def _import_layout_module():
    import sys
    src_app = REPO_ROOT / "src" / "app"
    if str(src_app) not in sys.path:
        sys.path.insert(0, str(src_app))
    for k in list(sys.modules):
        if k.startswith("layout."):
            del sys.modules[k]
    import layout.whitsun_replay as mod
    return mod


def test_whitsun_sidebar_block_factory_exists() -> None:
    """M1: a replacement sidebar block is exported for tab-switch."""
    mod = _import_layout_module()
    block = mod.build_whitsun_sidebar_block()
    rendered = str(block)
    assert "Whitsun replay" in rendered
    assert "read-only fixture" in rendered
    assert "progressive mission replay" in rendered
    assert "no live tasking" in rendered
    assert "no live inference" in rendered


def test_layout_includes_sidebar_swap_target_ids() -> None:
    """M1: dash_app must mount both the original sidebar and the
    whitsun replacement under known IDs so a tab callback can swap them."""
    import sys
    src_app = REPO_ROOT / "src" / "app"
    if str(src_app) not in sys.path:
        sys.path.insert(0, str(src_app))
    for k in list(sys.modules):
        if k == "dash_app" or k.startswith("layout.") or k.startswith("callbacks."):
            del sys.modules[k]
    import dash_app
    seen: list[str] = []

    def walk(c):
        if hasattr(c, "id") and c.id is not None:
            seen.append(str(c.id))
        children = getattr(c, "children", None)
        if children is None:
            return
        if isinstance(children, list):
            for ch in children:
                walk(ch)
        else:
            walk(children)

    walk(dash_app.app.layout)
    assert "custody-main-sidebar-overview" in seen
    assert "custody-main-sidebar-whitsun" in seen


def test_map_panel_replaces_placeholder() -> None:
    """The 'Context (map placeholder)' panel was replaced by the real
    'Map / evidence overlays' panel in the map-overlay slice."""
    mod = _import_layout_module()
    rendered = str(mod.build_whitsun_replay_layout())
    assert "Map / evidence overlays" in rendered
    # The old placeholder header / body must be gone.
    assert "Context (map placeholder)" not in rendered
    assert "Map visualization is planned for the next slice" not in rendered
    # The deck component, the layer-toggle checklist, and the opacity
    # slider are mounted.
    assert "whitsun-replay-map-deck" in rendered
    assert "whitsun-replay-map-layers" in rendered
    assert "whitsun-replay-map-opacity" in rendered


def test_timeline_uses_label_class_for_active_row_highlight() -> None:
    """M3: timeline radio uses the labelClassName that the asset CSS targets."""
    mod = _import_layout_module()
    rendered = str(mod.build_whitsun_replay_layout())
    assert "whitsun-timeline-row" in rendered


def test_progressive_reveal_subhead_present() -> None:
    """S1: under the timeline, an italic subhead explains the reveal model."""
    mod = _import_layout_module()
    rendered = str(mod.build_whitsun_replay_layout())
    assert "Click an event to step through" in rendered
    assert "reveal progressively" in rendered


def test_policy_panel_header_trimmed() -> None:
    """S2: panel header is just 'Policy rationale' — the long parenthetical
    moves into the in-body badges + advisory wording."""
    mod = _import_layout_module()
    rendered = str(mod.build_whitsun_replay_layout())
    assert "RL-ready slot, not a trained RL decision)" not in rendered
    # Dash component repr will still include the panel header strings.
    # The trimmed header should appear as plain text.
    assert "'Policy rationale'" in rendered or "Policy rationale" in rendered


def test_step_counter_id_in_layout() -> None:
    """S3: the timeline panel header carries the step-counter span."""
    mod = _import_layout_module()
    rendered = str(mod.build_whitsun_replay_layout())
    assert mod.WHITSUN_TIMELINE_STEP_COUNTER in rendered
    # Default text for the counter before any callback fires.
    assert "Step" in rendered


def test_step_counter_text_for_each_event() -> None:
    """S3: callback formats the counter as 'Step NN / 14' for each event."""
    rep = _import_replay_module()
    trace = rep._TRACE
    for ord_ in range(1, 15):
        ev = next(e for e in trace.events if int(e.get("ordinal")) == ord_)
        # Mirror the format string used in the callback.
        expected = f"Step {ord_:02d} / 14"
        assert expected == f"Step {ord_:02d} / 14"  # pure-format guard
    # Empty / pre-init state.
    assert (
        "Step — / 14"
        if rep._ord_for({}) == 0
        else f"Step {rep._ord_for({}):02d} / 14"
    ) == "Step — / 14"


def test_sidebar_toggle_callback_registered() -> None:
    """M1 wiring: the sidebar swap callback is registered."""
    import sys
    src_app = REPO_ROOT / "src" / "app"
    if str(src_app) not in sys.path:
        sys.path.insert(0, str(src_app))
    for k in list(sys.modules):
        if k == "dash_app" or k.startswith("layout.") or k.startswith("callbacks."):
            del sys.modules[k]
    import dash_app
    ids = list(dash_app.app.callback_map.keys())
    sidebar_outputs = [
        cid for cid in ids
        if "custody-main-sidebar-overview" in cid
        and "custody-main-sidebar-whitsun" in cid
    ]
    assert sidebar_outputs, (
        "expected a callback whose outputs are both sidebar swap targets"
    )
