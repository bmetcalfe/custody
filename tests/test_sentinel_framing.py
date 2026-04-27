"""Tests for the Sentinel weak-signal framing pass.

Sentinel-1 / Sentinel-2 are weak-signal cueing layers in this
prototype, not confirmation evidence.  The UI, fixtures, and docs
must use that framing consistently and must avoid overstatement
phrasing like "Sentinel proves X" or "definitive change".

No live HTTP, no Dash server is started.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
TRACE_PATH = REPO_ROOT / "data" / "demo" / "whitsun_decision_trace.fixture.json"
MAP_OVERLAYS_PATH = REPO_ROOT / "data" / "demo" / "map_overlays.fixture.json"


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_json(path: Path) -> dict:
    return json.loads(_load_text(path))


# ---------------------------------------------------------------------------
# Forbidden language scan: blunt overstatement phrases must not appear in
# Sentinel-related fixtures, layouts, callbacks, or docs.
# ---------------------------------------------------------------------------


# Strict targets: forbidden phrases must never appear in fixtures or
# Python layout / callback code.  Docs are scanned separately because
# some legitimately cite the forbidden phrases as anti-patterns.
_TARGETS: tuple[Path, ...] = (
    TRACE_PATH,
    MAP_OVERLAYS_PATH,
    REPO_ROOT / "data" / "demo" / "whitsun_sentinel_observations.fixture.json",
    REPO_ROOT / "data" / "demo" / "tennent_sentinel_observations.fixture.json",
    REPO_ROOT / "src" / "app" / "layout" / "whitsun_replay.py",
    REPO_ROOT / "src" / "app" / "layout" / "tennent_monitoring.py",
    REPO_ROOT / "src" / "app" / "layout" / "map_overlays_helpers.py",
    REPO_ROOT / "src" / "app" / "callbacks" / "whitsun_replay.py",
    REPO_ROOT / "src" / "app" / "callbacks" / "tennent_monitoring.py",
    REPO_ROOT / "docs" / "demo_scenario_whitsun.md",
    REPO_ROOT / "docs" / "sentinel_ingestion_whitsun.md",
    REPO_ROOT / "docs" / "dash_map_overlays.md",
    REPO_ROOT / "docs" / "dash_whitsun_replay.md",
)

# Docs that legitimately cite the forbidden phrases as anti-patterns.
# The dedicated test below pins both the citation and the framing
# positives.
_DOCS_WITH_ANTI_PATTERN_CITATION: tuple[Path, ...] = (
    REPO_ROOT / "docs" / "demo_walkthrough.md",
    REPO_ROOT / "docs" / "sentinel_ingestion_demo_aois.md",
)

_FORBIDDEN_GLOBAL: tuple[str, ...] = (
    "definitive change",
    "sentinel proves",
)

_FORBIDDEN_NEAR_SENTINEL: tuple[str, ...] = (
    "confirmed detection",
)


def test_global_forbidden_phrases_absent() -> None:
    """Absolute-overstatement phrases that should not appear anywhere
    in the Sentinel-touching surface."""
    for path in _TARGETS:
        if not path.exists():
            continue
        text = _load_text(path).lower()
        for needle in _FORBIDDEN_GLOBAL:
            assert needle not in text, (
                f"forbidden phrase {needle!r} appears in {path}"
            )


def test_no_confirmed_detection_attached_to_sentinel() -> None:
    """The phrase 'confirmed detection' must not appear within any
    Sentinel-classified record's caveats / summary / display name."""
    payload = _load_json(MAP_OVERLAYS_PATH)
    for o in payload.get("overlays", []):
        if o.get("source") in ("sentinel-1", "sentinel-2"):
            blob = json.dumps(o).lower()
            for needle in _FORBIDDEN_NEAR_SENTINEL:
                assert needle not in blob, (
                    f"forbidden phrase {needle!r} appears in Sentinel "
                    f"overlay {o.get('overlay_id')}"
                )


# ---------------------------------------------------------------------------
# Required framing: Sentinel overlays carry weak-signal language.
# ---------------------------------------------------------------------------


def test_every_sentinel_overlay_carries_weak_signal_caveat() -> None:
    payload = _load_json(MAP_OVERLAYS_PATH)
    sentinel_overlays = [
        o for o in payload.get("overlays", [])
        if o.get("source") in ("sentinel-1", "sentinel-2")
    ]
    assert sentinel_overlays
    for o in sentinel_overlays:
        caveats = " ".join(o.get("caveats") or []).lower()
        assert "weak-signal" in caveats or "weak signal" in caveats, (
            f"Sentinel overlay {o.get('overlay_id')} missing "
            f"weak-signal framing in caveats"
        )


def test_decision_trace_sentinel_observations_carry_weak_signal_caveat() -> None:
    payload = _load_json(TRACE_PATH)
    sentinel_obs = [
        o for o in payload.get("observations", [])
        if o.get("source") in ("sentinel-1", "sentinel-2")
    ]
    assert sentinel_obs
    for o in sentinel_obs:
        caveats = " ".join(o.get("caveats") or []).lower()
        assert "weak-signal" in caveats or "weak signal" in caveats, (
            f"trace observation {o.get('observation_id')} missing "
            f"weak-signal framing"
        )


def test_decision_trace_operational_hierarchy_uses_framing() -> None:
    payload = _load_json(TRACE_PATH)
    hier = payload["scenario_metadata"]["operational_hierarchy"]
    # Umbra is the high-confidence confirmation layer.
    assert "high-confidence" in hier["umbra-sar"].lower()
    assert "confirmation" in hier["umbra-sar"].lower()
    # Sentinel records use weak-signal / cueing wording.
    s1 = hier["sentinel-1"].lower()
    s2 = hier["sentinel-2"].lower()
    assert "weak-signal" in s1 or "weak signal" in s1
    assert "cueing" in s1 or "tasking cue" in s1
    assert "weak-signal" in s2 or "weak signal" in s2


def test_decision_trace_sentinel_event_summaries_use_framing() -> None:
    """Sentinel cues sit at trace ordinals 5 (Sentinel-2) and 6
    (Sentinel-1) in the dispatch-ordered trace.  Their summaries must
    carry weak-signal framing; their labels may use either the
    "weak-signal cue" or "context observation" wording — both are
    cueing-only language and neither implies confirmation."""
    payload = _load_json(TRACE_PATH)
    sentinel_events = [
        ev for ev in payload.get("events", [])
        if ev.get("ordinal") in (5, 6)
    ]
    assert len(sentinel_events) == 2
    for ev in sentinel_events:
        summary = (ev.get("summary") or "").lower()
        label = (ev.get("label") or "").lower()
        assert "weak-signal" in summary or "weak signal" in summary, (
            f"event {ev.get('event_id')} summary missing weak-signal framing"
        )
        assert (
            "weak-signal" in label
            or "weak signal" in label
            or "context observation" in label
        ), (
            f"event {ev.get('event_id')} label missing weak-signal / "
            f"context-observation framing"
        )


# ---------------------------------------------------------------------------
# Whitsun replay UI: weak-signal cue note appears once Sentinel arrives
# after the Umbra collect.
# ---------------------------------------------------------------------------


def _import_whitsun_callbacks():
    import sys
    src_app = REPO_ROOT / "src" / "app"
    if str(src_app) not in sys.path:
        sys.path.insert(0, str(src_app))
    for k in list(sys.modules):
        if k.startswith("layout.") or k.startswith("callbacks."):
            del sys.modules[k]
    import callbacks.whitsun_replay as mod
    return mod


def test_weak_signal_cue_note_absent_before_sentinel_2() -> None:
    """Sentinel-2 first appears at trace ordinal 5; before that
    there's no weak-signal cue to surface (events 1-4 are scenario
    init / Umbra / VLM / track init)."""
    rep = _import_whitsun_callbacks()
    for ord_ in (1, 2, 3, 4):
        note = rep._weak_signal_cue_note(rep._TRACE, ord_)
        assert note is None, (
            f"weak-signal cue note must not appear at ordinal {ord_}"
        )


def test_weak_signal_cue_note_present_after_sentinel_2_arrives() -> None:
    rep = _import_whitsun_callbacks()
    note = rep._weak_signal_cue_note(rep._TRACE, 5)
    assert note is not None
    rendered = str(note).lower()
    assert "weak-signal cue" in rendered
    assert "tasking" in rendered
    assert "sentinel-2" in rendered or "sentinel-1" in rendered
    assert "high-confidence confirmation layer" in rendered


def test_weak_signal_cue_note_persists_through_remaining_steps() -> None:
    rep = _import_whitsun_callbacks()
    for ord_ in (5, 6, 7, 8, 14):
        note = rep._weak_signal_cue_note(rep._TRACE, ord_)
        assert note is not None, (
            f"cue note must persist at ordinal {ord_}"
        )


def test_weak_signal_cue_note_renders_inside_observations_panel() -> None:
    rep = _import_whitsun_callbacks()
    rendered = str(rep._render_observations(rep._TRACE, 5)).lower()
    assert "weak-signal cue" in rendered


# ---------------------------------------------------------------------------
# Tennent interpretation panel uses framing
# ---------------------------------------------------------------------------


def _import_tennent_layout():
    import sys
    src_app = REPO_ROOT / "src" / "app"
    if str(src_app) not in sys.path:
        sys.path.insert(0, str(src_app))
    for k in list(sys.modules):
        if k.startswith("layout."):
            del sys.modules[k]
    import layout.tennent_monitoring as mod
    return mod


def test_tennent_interpretation_uses_weak_signal_framing() -> None:
    mod = _import_tennent_layout()
    rendered = str(mod.build_tennent_monitoring_layout()).lower()
    assert "weak-signal cue" in rendered
    # Recommend higher-resolution tasking framing surfaces somewhere.
    assert "cueing higher" in rendered or "tasking cue" in rendered or "cueing layer" in rendered
    # Umbra remains the high-confidence confirmation layer.
    assert "high-confidence confirmation" in rendered


# ---------------------------------------------------------------------------
# Whitsun replay UI: Umbra still framed as high-confidence confirmation
# (regression guard — we must not lose that framing while down-rating
# Sentinel).
# ---------------------------------------------------------------------------


def test_anti_pattern_docs_cite_forbidden_phrases_and_positives() -> None:
    """Docs that intentionally name the forbidden phrases must also
    establish the framing positives nearby.  Anywhere else, the
    global scan above bans the phrases outright."""
    for path in _DOCS_WITH_ANTI_PATTERN_CITATION:
        text = path.read_text(encoding="utf-8").lower()
        for needle in ("definitive change", "sentinel proves"):
            assert needle in text, (
                f"{path.name} must cite {needle!r} as an anti-pattern"
            )
        # Framing positives must be present in the same doc.
        assert "weak-signal" in text, (
            f"{path.name} must use weak-signal framing language"
        )
        assert "cueing" in text, (
            f"{path.name} must use cueing-layer framing"
        )


def test_whitsun_replay_preserves_umbra_high_confidence_framing() -> None:
    payload = _load_json(TRACE_PATH)
    umbra = [
        o for o in payload.get("observations", [])
        if o.get("source") == "umbra-sar"
    ]
    assert umbra
    found_high_confidence = any(
        any(
            "high-confidence" in c.lower() or "tasked" in c.lower()
            for c in (o.get("caveats") or [])
        )
        for o in umbra
    )
    assert found_high_confidence, (
        "at least one Umbra observation must carry high-confidence / "
        "tasked framing in its caveats"
    )
