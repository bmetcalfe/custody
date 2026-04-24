"""Tests for scripts/13_decision_packet.py (ADR-0021 Slice 6).

These tests exercise the CLI by importing main() directly (no subprocess),
the same pattern as tests/test_script_12_hypothesis_timeline.py.  Asserts
stable key lines — section headers, editorial text, scenario-specific
ambiguity language — rather than exact floats.
"""
from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "13_decision_packet.py"


@pytest.fixture(scope="module")
def packet_module():
    spec = importlib.util.spec_from_file_location(
        "script_13_decision_packet", SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["script_13_decision_packet"] = mod
    try:
        spec.loader.exec_module(mod)
        yield mod
    finally:
        sys.modules.pop("script_13_decision_packet", None)


def _run(packet_module, scenario: str) -> str:
    buf = io.StringIO()
    rc = packet_module.main(["--scenario", scenario], out=buf)
    assert rc == 0
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Return codes / scenario coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", ["tennent", "whitsun", "both"])
def test_main_returns_zero_for_each_scenario(packet_module, scenario: str) -> None:
    buf = io.StringIO()
    rc = packet_module.main(["--scenario", scenario], out=buf)
    assert rc == 0
    assert buf.getvalue()


def test_main_default_runs_both_scenarios(packet_module) -> None:
    buf = io.StringIO()
    rc = packet_module.main([], out=buf)
    assert rc == 0
    out = buf.getvalue().lower()
    assert "tennent" in out
    assert "whitsun" in out


# ---------------------------------------------------------------------------
# Section-header assertions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "header",
    [
        "CUSTODY DECISION PACKET",
        "Current belief",
        "Custody health",
        "Primary ambiguity",
        "Recommended candidate collects",
        "Why these collects",
        "What not to do yet",
    ],
)
def test_output_contains_each_required_section(packet_module, header: str) -> None:
    out = _run(packet_module, "tennent")
    assert header in out, f"missing section header {header!r}"


# ---------------------------------------------------------------------------
# Scenario-specific ambiguity content
# ---------------------------------------------------------------------------


def test_tennent_primary_ambiguity_names_construction_or_fixed_structure(packet_module) -> None:
    """After the narrative, Tennent saturates FIXED + CONSTRUCTION — the primary
    ambiguity block should name at least one of these ids."""
    out = _run(packet_module, "tennent")
    ambiguity_block = out.split("Primary ambiguity", 1)[1].split("Recommended", 1)[0]
    assert (
        "fixed_reclamation_or_structure" in ambiguity_block
        or "construction_or_reclamation_activity" in ambiguity_block
    )


def test_whitsun_primary_ambiguity_names_ais_dark_or_cluster(packet_module) -> None:
    out = _run(packet_module, "whitsun")
    ambiguity_block = out.split("Primary ambiguity", 1)[1].split("Recommended", 1)[0]
    assert (
        "ais_dark_or_poorly_observed_vessels" in ambiguity_block
        or "vessel_cluster_activity" in ambiguity_block
        or "transient_anchorage_or_fishing_presence" in ambiguity_block
    )


# ---------------------------------------------------------------------------
# Language invariants
# ---------------------------------------------------------------------------


def test_output_uses_candidate_collect_language(packet_module) -> None:
    out = _run(packet_module, "both")
    assert "candidate collect" in out.lower()


def test_output_does_not_say_tasking_order(packet_module) -> None:
    out = _run(packet_module, "both")
    assert "tasking order" not in out.lower()


def test_output_flags_sentinel_as_roadmap(packet_module) -> None:
    out = _run(packet_module, "both")
    lower = out.lower()
    assert "sentinel-1/2" in lower or "sentinel" in lower
    assert "roadmap" in lower and "not part of this demo" in lower


def test_vlm_tuning_caution_appears_when_ambiguous(packet_module) -> None:
    """Tennent final state is AMBIGUOUS after the narrative — the 'What not to
    do yet' block should include the VLM tuning caution."""
    out = _run(packet_module, "tennent")
    tail = out.split("What not to do yet", 1)[1]
    assert "vlm tuning" in tail.lower()
    assert "hypothesis-level" in tail.lower() or "detector-confidence-level" in tail.lower()


def test_whitsun_ais_absence_caveat_appears(packet_module) -> None:
    out = _run(packet_module, "whitsun")
    tail = out.split("What not to do yet", 1)[1]
    assert "ais absence" in tail.lower() or "ais absence alone" in tail.lower()


def test_what_not_to_do_capped_at_three_bullets(packet_module) -> None:
    """Dispatch constrains the editorial to at most three bullets."""
    out = _run(packet_module, "whitsun")
    section = out.split("What not to do yet", 1)[1]
    # Counting bullets by leading "- " markers until the next section break
    # (end of output or a blank line followed by a header-like divider).
    bullets = 0
    for line in section.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            bullets += 1
    assert bullets <= 3


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_output_is_deterministic(packet_module) -> None:
    out1 = _run(packet_module, "both")
    out2 = _run(packet_module, "both")
    assert out1 == out2


# ---------------------------------------------------------------------------
# Current-belief block: top 3 only
# ---------------------------------------------------------------------------


def test_current_belief_lists_at_most_three_hypotheses_per_scenario(packet_module) -> None:
    """Current belief block shows the top three; not the full registry."""
    out = _run(packet_module, "tennent")
    belief_block = out.split("Current belief", 1)[1].split("Custody health", 1)[0]
    # Count lines that look like "  0.XX  <hypothesis_id>"
    import re
    hits = re.findall(r"\b\d\.\d{2}\b\s+[a-z_]+", belief_block)
    assert 1 <= len(hits) <= 3


# ---------------------------------------------------------------------------
# Import-boundary sanity
# ---------------------------------------------------------------------------


def test_script_does_not_import_detection_or_gfw() -> None:
    import ast
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    forbidden = (
        "custody.detection",
        "custody.ingest.gfw_presence",
        "sentinelhub",
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
    assert not offending, f"decision-packet script imports forbidden modules: {offending}"


# ---------------------------------------------------------------------------
# Slice 8: --format branches
# ---------------------------------------------------------------------------


FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "decision_packet"


def test_default_format_is_text_and_matches_baseline(packet_module) -> None:
    """No --format flag defaults to text; output matches the Slice 6 snapshot."""
    buf = io.StringIO()
    rc = packet_module.main(["--scenario", "tennent"], out=buf)
    assert rc == 0
    expected = (FIXTURE_DIR / "tennent_text.txt").read_text(encoding="utf-8")
    assert buf.getvalue() == expected


def test_explicit_format_text_matches_tennent_baseline(packet_module) -> None:
    buf = io.StringIO()
    rc = packet_module.main(
        ["--scenario", "tennent", "--format", "text"], out=buf,
    )
    assert rc == 0
    expected = (FIXTURE_DIR / "tennent_text.txt").read_text(encoding="utf-8")
    assert buf.getvalue() == expected


def test_explicit_format_text_matches_whitsun_baseline(packet_module) -> None:
    buf = io.StringIO()
    rc = packet_module.main(
        ["--scenario", "whitsun", "--format", "text"], out=buf,
    )
    assert rc == 0
    expected = (FIXTURE_DIR / "whitsun_text.txt").read_text(encoding="utf-8")
    assert buf.getvalue() == expected


def test_format_json_is_parseable_and_has_closed_schema(packet_module) -> None:
    import json as _json
    buf = io.StringIO()
    rc = packet_module.main(
        ["--scenario", "tennent", "--format", "json"], out=buf,
    )
    assert rc == 0
    parsed = _json.loads(buf.getvalue())
    assert parsed["schema_version"] == "1"
    assert parsed["scenario_id"] == "tennent"
    # Top-level keys must match the documented closed schema.
    assert set(parsed.keys()) == {
        "schema_version", "scenario_id", "generated_at", "current_belief",
        "custody_health", "ambiguity_pairs", "recommended_collects", "do_not_yet",
    }


def test_format_json_is_deterministic_across_invocations(packet_module) -> None:
    buf1, buf2 = io.StringIO(), io.StringIO()
    packet_module.main(["--scenario", "tennent", "--format", "json"], out=buf1)
    packet_module.main(["--scenario", "tennent", "--format", "json"], out=buf2)
    assert buf1.getvalue() == buf2.getvalue()


def test_format_json_both_is_single_array_of_two(packet_module) -> None:
    import json as _json
    buf = io.StringIO()
    rc = packet_module.main(["--scenario", "both", "--format", "json"], out=buf)
    assert rc == 0
    parsed = _json.loads(buf.getvalue())
    assert isinstance(parsed, list)
    assert len(parsed) == 2
    assert parsed[0]["scenario_id"] == "tennent"
    assert parsed[1]["scenario_id"] == "whitsun"


def test_format_md_starts_with_expected_h1(packet_module) -> None:
    buf = io.StringIO()
    rc = packet_module.main(["--scenario", "tennent", "--format", "md"], out=buf)
    assert rc == 0
    assert buf.getvalue().startswith("# Custody Decision Packet - Tennent Reef\n")


def test_format_md_matches_tennent_fixture(packet_module) -> None:
    buf = io.StringIO()
    packet_module.main(["--scenario", "tennent", "--format", "md"], out=buf)
    expected = (FIXTURE_DIR / "tennent.md").read_text(encoding="utf-8")
    assert buf.getvalue() == expected


def test_format_md_both_contains_both_h1(packet_module) -> None:
    buf = io.StringIO()
    rc = packet_module.main(["--scenario", "both", "--format", "md"], out=buf)
    assert rc == 0
    out = buf.getvalue()
    assert "# Custody Decision Packet - Tennent Reef" in out
    assert "# Custody Decision Packet - Whitsun Reef" in out


def test_invalid_format_value_exits_nonzero(packet_module) -> None:
    buf = io.StringIO()
    with pytest.raises(SystemExit) as excinfo:
        packet_module.main(
            ["--scenario", "tennent", "--format", "csv"], out=buf,
        )
    assert excinfo.value.code != 0
