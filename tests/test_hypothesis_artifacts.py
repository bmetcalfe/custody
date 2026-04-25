"""Tests for :mod:`custody.hypotheses.artifacts` (ADR-0021 Slice 19)."""
from __future__ import annotations

import ast
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from custody.hypotheses.artifacts import (
    ArtifactDecisionBundle,
    ArtifactEvidenceBundle,
    ArtifactRecord,
    artifact_record_from_mapping,
    artifact_record_to_evidence,
    artifact_records_to_evidence,
    build_decision_from_artifacts,
    bundle_to_dict,
    bundle_to_json,
    format_artifact_decision_markdown,
    format_artifact_decision_text,
    load_artifact_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TENNENT_FIXTURE = (
    REPO_ROOT / "tests" / "fixtures" / "artifacts" / "tennent_artifact_manifest.json"
)
WHITSUN_FIXTURE = (
    REPO_ROOT / "tests" / "fixtures" / "artifacts" / "whitsun_artifact_manifest.json"
)


def _base_record(**overrides) -> dict:
    base = {
        "artifact_id": "test-1",
        "scenario_id": "tennent",
        "kind": "scene_signal",
        "source_ref": "test-source",
        "timestamp": "2023-07-02T14:00:00+00:00",
        "quality_flag": "yellow",
        "confidence": None,
        "payload": {},
        "notes": None,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_artifact_record_from_mapping_parses_required_fields() -> None:
    rec = artifact_record_from_mapping(_base_record())
    assert rec.artifact_id == "test-1"
    assert rec.scenario_id == "tennent"
    assert rec.kind == "scene_signal"
    assert rec.source_ref == "test-source"


def test_artifact_record_from_mapping_missing_artifact_id_raises() -> None:
    with pytest.raises(ValueError):
        artifact_record_from_mapping(_base_record(artifact_id=""))


def test_artifact_record_from_mapping_missing_source_ref_raises() -> None:
    with pytest.raises(ValueError):
        artifact_record_from_mapping(_base_record(source_ref=""))


def test_artifact_record_from_mapping_invalid_scenario_raises() -> None:
    with pytest.raises(ValueError):
        artifact_record_from_mapping(_base_record(scenario_id="invalid"))


def test_timestamp_iso_string_becomes_timezone_aware_utc() -> None:
    rec = artifact_record_from_mapping(
        _base_record(timestamp="2023-07-02T14:00:00"),
    )
    assert rec.timestamp is not None
    assert rec.timestamp.tzinfo is not None


def test_confidence_validation_rejects_above_1() -> None:
    with pytest.raises(ValueError):
        artifact_record_from_mapping(_base_record(confidence=1.5))


def test_confidence_validation_rejects_below_0() -> None:
    with pytest.raises(ValueError):
        artifact_record_from_mapping(_base_record(confidence=-0.1))


def test_load_artifact_manifest_tennent() -> None:
    records = load_artifact_manifest(TENNENT_FIXTURE)
    assert len(records) == 5
    assert all(isinstance(r, ArtifactRecord) for r in records)
    assert all(r.scenario_id == "tennent" for r in records)


# ---------------------------------------------------------------------------
# Explicit semantic path
# ---------------------------------------------------------------------------


def test_explicit_supports_contradicts_produces_evidence() -> None:
    rec = artifact_record_from_mapping(_base_record(
        kind="manual_label",
        payload={
            "supports": ["fixed_reclamation_or_structure"],
            "contradicts": ["no_meaningful_activity"],
            "reason": "analyst label",
        },
    ))
    evs = artifact_record_to_evidence(rec)
    assert len(evs) == 1
    assert "fixed_reclamation_or_structure" in evs[0].supports


def test_explicit_invalid_hypothesis_id_raises() -> None:
    rec = artifact_record_from_mapping(_base_record(
        kind="manual_label",
        payload={
            "supports": ["nonexistent_id"],
            "contradicts": [],
            "reason": "x",
        },
    ))
    with pytest.raises(ValueError):
        artifact_record_to_evidence(rec)


# ---------------------------------------------------------------------------
# Scenario signal path - Tennent
# ---------------------------------------------------------------------------


def test_tennent_scene_signal_persistent_scatterer() -> None:
    rec = artifact_record_from_mapping(_base_record(
        payload={"persistent_scatterer_detected": True},
    ))
    evs = artifact_record_to_evidence(rec)
    assert len(evs) >= 1


def test_tennent_scene_signal_change_strength() -> None:
    rec = artifact_record_from_mapping(_base_record(
        payload={
            "persistent_scatterer_detected": False,
            "change_signal_strength": 0.30,
        },
    ))
    evs = artifact_record_to_evidence(rec)
    assert len(evs) >= 1
    assert any(
        "construction_or_reclamation_activity" in e.supports for e in evs
    )


def test_tennent_matcher_displacement() -> None:
    rec = artifact_record_from_mapping(_base_record(
        kind="matcher_persistence",
        payload={
            "displacement_m": 5.0,
            "obs_a_id": "t-a",
            "obs_b_id": "t-b",
        },
    ))
    evs = artifact_record_to_evidence(rec)
    assert len(evs) == 1
    assert "fixed_reclamation_or_structure" in evs[0].supports


# ---------------------------------------------------------------------------
# Scenario signal path - Whitsun
# ---------------------------------------------------------------------------


def test_whitsun_vessel_count_scene_signal() -> None:
    rec = artifact_record_from_mapping(_base_record(
        scenario_id="whitsun",
        payload={"vessel_count": 6, "ais_coverage_flag": "sparse"},
    ))
    evs = artifact_record_to_evidence(rec)
    assert len(evs) >= 1
    assert any("vessel_cluster_activity" in e.supports for e in evs)


def test_whitsun_ais_coverage_flag_signal() -> None:
    rec = artifact_record_from_mapping(_base_record(
        scenario_id="whitsun",
        payload={"vessel_count": 6, "ais_coverage_flag": "sparse"},
    ))
    evs = artifact_record_to_evidence(rec)
    assert any(
        "ais_dark_or_poorly_observed_vessels" in e.supports for e in evs
    )


def test_whitsun_matched_cluster() -> None:
    rec = artifact_record_from_mapping(_base_record(
        scenario_id="whitsun",
        kind="matcher_persistence",
        payload={
            "matched_cluster": True,
            "obs_a_id": "w-a",
            "obs_b_id": "w-b",
            "distance_m": 15.0,
        },
    ))
    evs = artifact_record_to_evidence(rec)
    assert len(evs) == 1
    assert "vessel_cluster_activity" in evs[0].supports


# ---------------------------------------------------------------------------
# No-evidence path
# ---------------------------------------------------------------------------


def test_unknown_kind_no_semantics_produces_no_evidence() -> None:
    rec = artifact_record_from_mapping(_base_record(
        kind="unknown",
        payload={"description": "no semantics"},
    ))
    assert artifact_record_to_evidence(rec) == ()


def test_quality_flag_only_produces_no_evidence() -> None:
    rec = artifact_record_from_mapping(_base_record(
        kind="quality_flag",
        payload={},
    ))
    assert artifact_record_to_evidence(rec) == ()


# ---------------------------------------------------------------------------
# Bundle / decision
# ---------------------------------------------------------------------------


def test_artifact_records_to_evidence_counts_correctly() -> None:
    records = load_artifact_manifest(TENNENT_FIXTURE)
    bundle = artifact_records_to_evidence(records, scenario_id="tennent")
    assert isinstance(bundle, ArtifactEvidenceBundle)
    assert bundle.scenario_id == "tennent"
    assert len(bundle.records) == 5
    assert len(bundle.evidence) >= 4  # at least scene-persist, scene-change, match, label


def test_artifact_records_to_evidence_caveats_for_no_evidence() -> None:
    records = load_artifact_manifest(TENNENT_FIXTURE)
    bundle = artifact_records_to_evidence(records, scenario_id="tennent")
    joined = " ".join(bundle.caveats).lower()
    assert "no evidence generated" in joined


def test_build_decision_from_artifacts_returns_state_health_recommendation() -> None:
    records = load_artifact_manifest(TENNENT_FIXTURE)
    bundle = build_decision_from_artifacts(records, scenario_id="tennent")
    assert isinstance(bundle, ArtifactDecisionBundle)
    assert bundle.final_state.scenario_id == "tennent"
    assert bundle.health is not None
    assert bundle.recommendation is not None


def test_bundle_to_dict_json_serializable() -> None:
    records = load_artifact_manifest(TENNENT_FIXTURE)
    decision = build_decision_from_artifacts(records, scenario_id="tennent")
    d = bundle_to_dict(decision)
    encoded = json.dumps(d)
    assert json.loads(encoded) == d


def test_bundle_to_json_parseable() -> None:
    records = load_artifact_manifest(WHITSUN_FIXTURE)
    decision = build_decision_from_artifacts(records, scenario_id="whitsun")
    blob = bundle_to_json(decision)
    parsed = json.loads(blob)
    assert parsed["scenario_id"] == "whitsun"
    assert "decision_packet" in parsed


def test_scenario_mismatch_filtered() -> None:
    records = load_artifact_manifest(TENNENT_FIXTURE)
    bundle = artifact_records_to_evidence(records, scenario_id="whitsun")
    assert bundle.records == ()
    assert bundle.evidence == ()


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------


def test_no_forbidden_imports() -> None:
    import custody.hypotheses.artifacts as mod
    tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
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
    assert not offending, f"forbidden imports: {offending}"


def test_source_file_no_forbidden_language() -> None:
    import custody.hypotheses.artifacts as mod
    src = Path(mod.__file__).read_text(encoding="utf-8").lower()
    forbidden = (
        "live data ingestion",
        "real-time ingestion",
        "sentinel integration",
        "tasking order",
        "live tasking",
        "sensor command",
        "autonomous execution",
        "collection order",
        "revenue dollars",
    )
    for needle in forbidden:
        assert needle not in src, f"forbidden token {needle!r} appears in source"


def test_text_format_contains_required_sections() -> None:
    records = load_artifact_manifest(TENNENT_FIXTURE)
    decision = build_decision_from_artifacts(records, scenario_id="tennent")
    text = format_artifact_decision_text(decision)
    assert "ARTIFACT DECISION PACKET" in text
    assert "Records loaded" in text
    assert "Evidence generated" in text
    assert "Custody health" in text
    assert "Caveats" in text


def test_markdown_format_contains_heading() -> None:
    records = load_artifact_manifest(WHITSUN_FIXTURE)
    decision = build_decision_from_artifacts(records, scenario_id="whitsun")
    md = format_artifact_decision_markdown(decision)
    assert md.startswith("# Artifact Decision Packet")
