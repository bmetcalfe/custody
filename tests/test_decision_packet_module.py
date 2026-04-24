"""Tests for :mod:`custody.hypotheses.decision_packet` (ADR-0021 Slice 8).

Unit-level tests exercising the module directly (no CLI).  CLI-level
tests live in tests/test_script_13_decision_packet.py.
"""
from __future__ import annotations

import importlib.util
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "decision_packet"
SCRIPT_PATH = REPO_ROOT / "scripts" / "13_decision_packet.py"


from custody.hypotheses.custody_health import (
    CustodyHealthStatus,
    assess_custody_health,
)
from custody.hypotheses.collection_value import rank_collection_candidates
from custody.hypotheses.decision_packet import (
    SCHEMA_VERSION,
    BeliefEntry,
    CollectRecommendation,
    DecisionPacket,
    build_decision_packet,
    format_as_json,
    format_as_markdown,
    format_as_text,
    format_many_as_json,
    what_not_to_do,
)
from custody.hypotheses.registry import SCENARIO_TENNENT, SCENARIO_WHITSUN
from custody.hypotheses.types import HypothesisEvidence
from custody.hypotheses.update import update_state


# ---------------------------------------------------------------------------
# Shared fixtures — load the CLI script as a module so we can reuse its
# _build_tennent_state / _build_whitsun_state helpers for the canonical
# inputs.  The CLI script already owns the scenario narratives.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def script_module():
    spec = importlib.util.spec_from_file_location("s13_mod_tests", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["s13_mod_tests"] = mod
    try:
        spec.loader.exec_module(mod)
        yield mod
    finally:
        sys.modules.pop("s13_mod_tests", None)


@pytest.fixture()
def tennent_packet(script_module) -> DecisionPacket:
    state = script_module._build_tennent_state()
    health = assess_custody_health(state)
    rec = rank_collection_candidates(state, health)
    return build_decision_packet(
        scenario_id=SCENARIO_TENNENT, state=state, health=health, recommendation=rec,
    )


@pytest.fixture()
def whitsun_packet(script_module) -> DecisionPacket:
    state = script_module._build_whitsun_state()
    health = assess_custody_health(state)
    rec = rank_collection_candidates(state, health)
    return build_decision_packet(
        scenario_id=SCENARIO_WHITSUN, state=state, health=health, recommendation=rec,
    )


# ---------------------------------------------------------------------------
# build_decision_packet
# ---------------------------------------------------------------------------


def test_schema_version_is_one(tennent_packet) -> None:
    assert tennent_packet.schema_version == SCHEMA_VERSION == "1"


def test_scenario_id_passes_through(tennent_packet) -> None:
    assert tennent_packet.scenario_id == "tennent"


def test_current_belief_top_three(tennent_packet) -> None:
    assert len(tennent_packet.current_belief) == 3
    scores = [b.score for b in tennent_packet.current_belief]
    assert scores == sorted(scores, reverse=True)


def test_current_belief_scores_rounded_to_three_decimals(tennent_packet) -> None:
    for b in tennent_packet.current_belief:
        # round to 3dp is a no-op if already rounded
        assert round(b.score, 3) == b.score


def test_custody_status_is_enum_value_not_name(tennent_packet) -> None:
    """Dataclass stores status.value ('ambiguous'), never the Python name."""
    assert tennent_packet.custody_status == "ambiguous"
    assert tennent_packet.custody_status == CustodyHealthStatus.AMBIGUOUS.value


def test_custody_score_in_unit_interval(tennent_packet) -> None:
    assert 0.0 <= tennent_packet.custody_score <= 1.0


def test_recommended_collects_preserves_rank_order(tennent_packet) -> None:
    scores = [r.score for r in tennent_packet.recommended_collects]
    assert scores == sorted(scores, reverse=True)


def test_ambiguity_pairs_canonical(tennent_packet) -> None:
    # Tennent saturates FIXED + CONSTRUCTION — canonical pair is ordered.
    for a, b in tennent_packet.ambiguity_pairs:
        assert a < b


def test_generated_at_equals_state_timestamp_not_wall_clock(script_module) -> None:
    """No datetime.now() in the packet: generated_at comes from state.timestamp."""
    state = script_module._build_tennent_state()
    health = assess_custody_health(state)
    rec = rank_collection_candidates(state, health)
    packet = build_decision_packet(
        scenario_id=SCENARIO_TENNENT, state=state, health=health, recommendation=rec,
    )
    assert packet.generated_at == state.timestamp
    # State timestamps for the synthetic narrative land in 2023-2024; if
    # something ever wires wall-clock in, this would surface as a huge gap.
    assert packet.generated_at is not None
    assert packet.generated_at.year <= 2024


def test_what_not_to_do_tennent_ambiguous_has_vlm_and_sentinel() -> None:
    state = update_state(
        SCENARIO_TENNENT,
        evidence=(
            HypothesisEvidence(
                evidence_id="e", source_ref=None, source_kind="synthetic",
                scenario_id=SCENARIO_TENNENT,
                timestamp=datetime(2023, 8, 13, tzinfo=timezone.utc),
                supports=("fixed_reclamation_or_structure",),
                contradicts=(),
                confidence=1.0, weight=1.0, reason="r",
            ),
            HypothesisEvidence(
                evidence_id="e2", source_ref=None, source_kind="synthetic",
                scenario_id=SCENARIO_TENNENT,
                timestamp=datetime(2023, 8, 13, tzinfo=timezone.utc),
                supports=("construction_or_reclamation_activity",),
                contradicts=(),
                confidence=1.0, weight=1.0, reason="r",
            ),
        ),
    )
    health = assess_custody_health(state, as_of=datetime(2023, 8, 13, tzinfo=timezone.utc))
    assert health.status is CustodyHealthStatus.AMBIGUOUS
    bullets = what_not_to_do(SCENARIO_TENNENT, health)
    joined = "\n".join(bullets).lower()
    assert "vlm tuning" in joined
    assert "sentinel-1/2" in joined
    assert "ais absence" not in joined  # Tennent → no Whitsun bullet


def test_what_not_to_do_whitsun_ambiguous_has_all_three_bullets() -> None:
    state = update_state(
        SCENARIO_WHITSUN,
        evidence=(
            HypothesisEvidence(
                evidence_id="e", source_ref=None, source_kind="synthetic",
                scenario_id=SCENARIO_WHITSUN,
                timestamp=datetime(2023, 12, 6, tzinfo=timezone.utc),
                supports=("vessel_cluster_activity",),
                contradicts=(),
                confidence=1.0, weight=1.0, reason="r",
            ),
            HypothesisEvidence(
                evidence_id="e2", source_ref=None, source_kind="synthetic",
                scenario_id=SCENARIO_WHITSUN,
                timestamp=datetime(2023, 12, 6, tzinfo=timezone.utc),
                supports=("transient_anchorage_or_fishing_presence",),
                contradicts=(),
                confidence=1.0, weight=1.0, reason="r",
            ),
        ),
    )
    health = assess_custody_health(state, as_of=datetime(2023, 12, 6, tzinfo=timezone.utc))
    assert health.status is CustodyHealthStatus.AMBIGUOUS
    bullets = what_not_to_do(SCENARIO_WHITSUN, health)
    assert len(bullets) == 3
    joined = "\n".join(bullets).lower()
    assert "vlm tuning" in joined
    assert "ais absence" in joined
    assert "sentinel-1/2" in joined


# ---------------------------------------------------------------------------
# format_as_text — byte-identical to Slice 6 baseline
# ---------------------------------------------------------------------------


def test_format_as_text_matches_tennent_fixture(tennent_packet) -> None:
    expected = (FIXTURE_DIR / "tennent_text.txt").read_text(encoding="utf-8")
    assert format_as_text(tennent_packet) == expected


def test_format_as_text_matches_whitsun_fixture(whitsun_packet) -> None:
    expected = (FIXTURE_DIR / "whitsun_text.txt").read_text(encoding="utf-8")
    assert format_as_text(whitsun_packet) == expected


# ---------------------------------------------------------------------------
# format_as_json
# ---------------------------------------------------------------------------


_EXPECTED_JSON_TOP_LEVEL_KEYS = [
    "schema_version",
    "scenario_id",
    "generated_at",
    "current_belief",
    "custody_health",
    "ambiguity_pairs",
    "recommended_collects",
    "do_not_yet",
]

_EXPECTED_CUSTODY_HEALTH_KEYS = {
    "status", "score", "top_hypothesis", "top_score",
    "second_hypothesis", "second_score", "top_two_margin",
    "drivers", "reason", "latest_evidence_at",
}


def test_format_as_json_is_valid_json(tennent_packet) -> None:
    out = format_as_json(tennent_packet)
    parsed = json.loads(out)
    assert isinstance(parsed, dict)


def test_format_as_json_ends_with_trailing_newline(tennent_packet) -> None:
    assert format_as_json(tennent_packet).endswith("\n")


def test_json_top_level_keys_fixed_order(tennent_packet) -> None:
    out = format_as_json(tennent_packet)
    parsed = json.loads(out)
    assert list(parsed.keys()) == _EXPECTED_JSON_TOP_LEVEL_KEYS


def test_json_schema_is_closed(tennent_packet) -> None:
    """No extra top-level keys beyond the documented schema."""
    parsed = json.loads(format_as_json(tennent_packet))
    assert set(parsed.keys()) == set(_EXPECTED_JSON_TOP_LEVEL_KEYS)


def test_json_custody_health_keys_are_closed(tennent_packet) -> None:
    parsed = json.loads(format_as_json(tennent_packet))
    assert set(parsed["custody_health"].keys()) == _EXPECTED_CUSTODY_HEALTH_KEYS


def test_json_status_is_lowercase_value_not_python_name(tennent_packet) -> None:
    parsed = json.loads(format_as_json(tennent_packet))
    assert parsed["custody_health"]["status"] == "ambiguous"


def test_json_ambiguity_pairs_are_canonical_arrays(tennent_packet) -> None:
    parsed = json.loads(format_as_json(tennent_packet))
    for pair in parsed["ambiguity_pairs"]:
        assert isinstance(pair, list)
        assert len(pair) == 2
        assert pair[0] < pair[1]


def test_json_datetime_is_iso_with_offset(tennent_packet) -> None:
    parsed = json.loads(format_as_json(tennent_packet))
    ts = parsed["generated_at"]
    assert ts is not None
    # tz-aware ISO strings contain "+" or explicit "Z"
    assert "+" in ts or ts.endswith("Z")


def test_json_is_deterministic(tennent_packet) -> None:
    out1 = format_as_json(tennent_packet)
    out2 = format_as_json(tennent_packet)
    assert out1 == out2


def test_format_many_as_json_is_valid_array(tennent_packet, whitsun_packet) -> None:
    out = format_many_as_json((tennent_packet, whitsun_packet))
    parsed = json.loads(out)
    assert isinstance(parsed, list)
    assert len(parsed) == 2
    assert parsed[0]["scenario_id"] == "tennent"
    assert parsed[1]["scenario_id"] == "whitsun"


# ---------------------------------------------------------------------------
# format_as_markdown
# ---------------------------------------------------------------------------


def test_format_as_markdown_matches_tennent_fixture(tennent_packet) -> None:
    expected = (FIXTURE_DIR / "tennent.md").read_text(encoding="utf-8")
    assert format_as_markdown(tennent_packet) == expected


def test_format_as_markdown_matches_whitsun_fixture(whitsun_packet) -> None:
    expected = (FIXTURE_DIR / "whitsun.md").read_text(encoding="utf-8")
    assert format_as_markdown(whitsun_packet) == expected


def test_markdown_has_expected_h1(tennent_packet) -> None:
    md = format_as_markdown(tennent_packet)
    assert md.startswith("# Custody Decision Packet - Tennent Reef\n")


def test_markdown_uses_ascii_only(tennent_packet, whitsun_packet) -> None:
    for md in (format_as_markdown(tennent_packet), format_as_markdown(whitsun_packet)):
        # The em-dash is U+2014; explicitly excluded per dispatch.
        assert "—" not in md


def test_markdown_ends_with_trailing_newline(tennent_packet) -> None:
    assert format_as_markdown(tennent_packet).endswith("\n")


def test_markdown_top_bolded_entries_match_max_score(tennent_packet) -> None:
    md = format_as_markdown(tennent_packet)
    # Tennent: FIXED + CONSTRUCTION both at 1.000 → both bolded.
    assert "- **construction_or_reclamation_activity** - 1.000" in md
    assert "- **fixed_reclamation_or_structure** - 1.000" in md
    # no_meaningful_activity at 0.000 → NOT bolded.
    assert "- no_meaningful_activity - 0.000" in md


def test_markdown_omits_primary_ambiguity_when_no_pairs(script_module) -> None:
    """Construct a synthetic HEALTHY state and assert no '### Primary ambiguity' heading."""
    state = update_state(
        SCENARIO_TENNENT,
        evidence=(
            HypothesisEvidence(
                evidence_id="e", source_ref=None, source_kind="synthetic",
                scenario_id=SCENARIO_TENNENT,
                timestamp=datetime(2023, 8, 13, tzinfo=timezone.utc),
                supports=("fixed_reclamation_or_structure",),
                contradicts=(),
                confidence=0.85, weight=1.0, reason="r",
            ),
        ),
    )
    health = assess_custody_health(state, as_of=datetime(2023, 8, 13, tzinfo=timezone.utc))
    assert health.status is CustodyHealthStatus.HEALTHY
    rec = rank_collection_candidates(state, health)
    packet = build_decision_packet(
        scenario_id=SCENARIO_TENNENT, state=state, health=health, recommendation=rec,
    )
    md = format_as_markdown(packet)
    assert "### Primary ambiguity" not in md


# ---------------------------------------------------------------------------
# No wall-clock / no detection / no gfw imports in decision_packet.py
# ---------------------------------------------------------------------------


def test_decision_packet_module_stdlib_only() -> None:
    import ast
    import custody.hypotheses.decision_packet as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    # Allowed runtime deps: stdlib + sibling hypothesis-layer modules.
    allowed_prefixes = (
        "custody.hypotheses.",
        "json", "dataclasses", "datetime", "enum", "pathlib",
        "typing", "__future__", "collections.abc",
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod_name = node.module or ""
            assert any(mod_name.startswith(p) for p in allowed_prefixes), (
                f"disallowed import: from {mod_name}"
            )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                assert any(alias.name.startswith(p) for p in allowed_prefixes), (
                    f"disallowed import: import {alias.name}"
                )


def test_decision_packet_source_has_no_datetime_now() -> None:
    import custody.hypotheses.decision_packet as mod
    src = Path(mod.__file__).read_text(encoding="utf-8")
    # Guard against accidental wall-clock reintroduction.
    assert "datetime.now" not in src
    assert "datetime.utcnow" not in src
