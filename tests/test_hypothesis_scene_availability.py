"""Tests for :mod:`custody.hypotheses.scene_availability` (ADR-0021 Slice 20)."""
from __future__ import annotations

import ast
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from custody.hypotheses.collection_value import (
    CollectionCandidate,
    CollectionRecommendation,
    CollectionValue,
)
from custody.hypotheses.custody_health import CustodyHealthStatus
from custody.hypotheses.scene_availability import (
    AvailabilityAdjustedRecommendation,
    AvailabilityStatus,
    SceneAvailabilityCatalog,
    SceneAvailabilityRecord,
    adjust_recommendation_for_availability,
    assess_collect_feasibility,
    format_availability_markdown,
    format_availability_text,
    load_scene_availability_catalog,
    recommendation_to_dict,
    recommendation_to_json,
    scene_record_from_mapping,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TENNENT_FIXTURE = (
    REPO_ROOT / "tests" / "fixtures" / "availability"
    / "tennent_scene_availability.json"
)
WHITSUN_FIXTURE = (
    REPO_ROOT / "tests" / "fixtures" / "availability"
    / "whitsun_scene_availability.json"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_record(**overrides) -> dict:
    base = {
        "record_id": "test-1",
        "scenario_id": "tennent",
        "sensor_type": "sar",
        "source_ref": "test-source",
        "collection_time": "2023-07-02T14:00:00+00:00",
        "quality_flag": "yellow",
        "coverage_score": 0.7,
        "cloud_cover": None,
        "resolution_m": 0.5,
        "look_geometry": "ascending",
        "latency_hours": 6.0,
        "payload": {},
        "notes": None,
    }
    base.update(overrides)
    return base


def _candidate(cid: str, label: str = "x", cost: float = 0.5) -> CollectionCandidate:
    return CollectionCandidate(
        candidate_id=cid, label=label, modality="generic",
        description="x", relative_cost=cost, latency_class="medium",
    )


def _rec(values: tuple[CollectionValue, ...]) -> CollectionRecommendation:
    return CollectionRecommendation(
        scenario_id="tennent",
        health_status=CustodyHealthStatus.AMBIGUOUS,
        primary_ambiguity=None,
        ranked_values=values,
        summary="x",
    )


def _value(cid: str, score: float) -> CollectionValue:
    return CollectionValue(
        candidate=_candidate(cid),
        score=score,
        disambiguates=None,
        reason="x",
        caveats=(),
    )


def _catalog_from(records: tuple[SceneAvailabilityRecord, ...]) -> SceneAvailabilityCatalog:
    return SceneAvailabilityCatalog(
        scenario_id="tennent",
        records=records,
        summary=f"{len(records)} records",
        caveats=(),
    )


def _make_record(**kwargs) -> SceneAvailabilityRecord:
    return scene_record_from_mapping(_base_record(**kwargs))


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_scene_record_from_mapping_parses_valid_record() -> None:
    rec = scene_record_from_mapping(_base_record())
    assert rec.record_id == "test-1"
    assert rec.sensor_type == "sar"
    assert rec.coverage_score == 0.7


def test_timestamp_iso_string_becomes_timezone_aware_utc() -> None:
    rec = scene_record_from_mapping(
        _base_record(collection_time="2023-07-02T14:00:00"),
    )
    assert rec.collection_time is not None
    assert rec.collection_time.tzinfo is not None


def test_invalid_scenario_raises() -> None:
    with pytest.raises(ValueError):
        scene_record_from_mapping(_base_record(scenario_id="invalid"))


def test_invalid_sensor_type_raises() -> None:
    with pytest.raises(ValueError):
        scene_record_from_mapping(_base_record(sensor_type="lidar"))


def test_empty_record_id_raises() -> None:
    with pytest.raises(ValueError):
        scene_record_from_mapping(_base_record(record_id=""))


def test_empty_source_ref_raises() -> None:
    with pytest.raises(ValueError):
        scene_record_from_mapping(_base_record(source_ref=""))


def test_invalid_coverage_score_above_1_raises() -> None:
    with pytest.raises(ValueError):
        scene_record_from_mapping(_base_record(coverage_score=1.5))


def test_invalid_coverage_score_below_0_raises() -> None:
    with pytest.raises(ValueError):
        scene_record_from_mapping(_base_record(coverage_score=-0.1))


def test_invalid_cloud_cover_raises() -> None:
    with pytest.raises(ValueError):
        scene_record_from_mapping(_base_record(cloud_cover=1.5))


def test_invalid_resolution_m_raises() -> None:
    with pytest.raises(ValueError):
        scene_record_from_mapping(_base_record(resolution_m=-1.0))


def test_invalid_latency_hours_raises() -> None:
    with pytest.raises(ValueError):
        scene_record_from_mapping(_base_record(latency_hours=-1.0))


def test_load_scene_availability_catalog_tennent() -> None:
    catalog = load_scene_availability_catalog(TENNENT_FIXTURE)
    assert isinstance(catalog, SceneAvailabilityCatalog)
    assert catalog.scenario_id == "tennent"
    assert len(catalog.records) == 6


def test_catalog_scenario_mismatch_raises(tmp_path: Path) -> None:
    mixed = tmp_path / "mixed.json"
    mixed.write_text(json.dumps([
        _base_record(record_id="t1", scenario_id="tennent"),
        _base_record(record_id="w1", scenario_id="whitsun"),
    ]))
    with pytest.raises(ValueError):
        load_scene_availability_catalog(mixed)


# ---------------------------------------------------------------------------
# Feasibility - optical_context
# ---------------------------------------------------------------------------


def test_optical_context_feasible_with_low_cloud() -> None:
    catalog = _catalog_from((
        _make_record(
            record_id="o1", sensor_type="optical", cloud_cover=0.2,
            coverage_score=0.7, look_geometry=None, resolution_m=3.0,
        ),
    ))
    rec = _rec((_value("optical_context", 0.85),))
    assessments = assess_collect_feasibility(rec, catalog)
    a = next(a for a in assessments if a.candidate_id == "optical_context")
    assert a.status == AvailabilityStatus.FEASIBLE


def test_optical_context_partial_with_high_cloud() -> None:
    catalog = _catalog_from((
        _make_record(
            record_id="o1", sensor_type="optical", cloud_cover=0.5,
            coverage_score=0.7, look_geometry=None, resolution_m=3.0,
        ),
    ))
    rec = _rec((_value("optical_context", 0.85),))
    assessments = assess_collect_feasibility(rec, catalog)
    a = next(a for a in assessments if a.candidate_id == "optical_context")
    assert a.status == AvailabilityStatus.PARTIAL


def test_optical_context_unavailable_no_optical() -> None:
    catalog = _catalog_from(())
    rec = _rec((_value("optical_context", 0.85),))
    assessments = assess_collect_feasibility(rec, catalog)
    a = next(a for a in assessments if a.candidate_id == "optical_context")
    assert a.status == AvailabilityStatus.UNAVAILABLE
    assert a.feasibility_score == 0.0


# ---------------------------------------------------------------------------
# Feasibility - repeat_sar
# ---------------------------------------------------------------------------


def test_repeat_sar_feasible_with_sar_metadata() -> None:
    catalog = _catalog_from((_make_record(coverage_score=0.7),))
    rec = _rec((_value("repeat_sar", 0.6),))
    assessments = assess_collect_feasibility(rec, catalog)
    a = next(a for a in assessments if a.candidate_id == "repeat_sar")
    assert a.status == AvailabilityStatus.FEASIBLE


def test_repeat_sar_partial_with_weak_coverage() -> None:
    catalog = _catalog_from((_make_record(coverage_score=0.3),))
    rec = _rec((_value("repeat_sar", 0.6),))
    assessments = assess_collect_feasibility(rec, catalog)
    a = next(a for a in assessments if a.candidate_id == "repeat_sar")
    assert a.status == AvailabilityStatus.PARTIAL


def test_repeat_sar_unavailable_no_sar() -> None:
    catalog = _catalog_from(())
    rec = _rec((_value("repeat_sar", 0.6),))
    assessments = assess_collect_feasibility(rec, catalog)
    a = next(a for a in assessments if a.candidate_id == "repeat_sar")
    assert a.status == AvailabilityStatus.UNAVAILABLE


# ---------------------------------------------------------------------------
# Feasibility - cross_geometry_sar
# ---------------------------------------------------------------------------


def test_cross_geometry_feasible_with_distinct_geometries() -> None:
    catalog = _catalog_from((
        _make_record(record_id="s1", look_geometry="ascending"),
        _make_record(record_id="s2", look_geometry="descending"),
    ))
    rec = _rec((_value("cross_geometry_sar", 0.7),))
    assessments = assess_collect_feasibility(rec, catalog)
    a = next(a for a in assessments if a.candidate_id == "cross_geometry_sar")
    assert a.status == AvailabilityStatus.FEASIBLE


def test_cross_geometry_partial_unknown_geometry() -> None:
    catalog = _catalog_from((
        _make_record(record_id="s1", look_geometry=None),
        _make_record(record_id="s2", look_geometry=None),
    ))
    rec = _rec((_value("cross_geometry_sar", 0.7),))
    assessments = assess_collect_feasibility(rec, catalog)
    a = next(a for a in assessments if a.candidate_id == "cross_geometry_sar")
    assert a.status == AvailabilityStatus.PARTIAL


# ---------------------------------------------------------------------------
# Feasibility - higher_resolution_sar
# ---------------------------------------------------------------------------


def test_higher_resolution_feasible_sub_meter() -> None:
    catalog = _catalog_from((_make_record(resolution_m=0.5),))
    rec = _rec((_value("higher_resolution_sar", 0.5),))
    assessments = assess_collect_feasibility(rec, catalog)
    a = next(a for a in assessments if a.candidate_id == "higher_resolution_sar")
    assert a.status == AvailabilityStatus.FEASIBLE


def test_higher_resolution_partial_above_meter() -> None:
    catalog = _catalog_from((_make_record(resolution_m=1.5),))
    rec = _rec((_value("higher_resolution_sar", 0.5),))
    assessments = assess_collect_feasibility(rec, catalog)
    a = next(a for a in assessments if a.candidate_id == "higher_resolution_sar")
    assert a.status == AvailabilityStatus.PARTIAL


# ---------------------------------------------------------------------------
# Feasibility - ais_coverage_query
# ---------------------------------------------------------------------------


def test_ais_feasible_with_strong_coverage() -> None:
    catalog = _catalog_from((
        _make_record(
            record_id="a1", sensor_type="ais", coverage_score=0.75,
            quality_flag="green", look_geometry=None, resolution_m=None,
        ),
    ))
    rec = _rec((_value("ais_coverage_query", 0.5),))
    assessments = assess_collect_feasibility(rec, catalog)
    a = next(a for a in assessments if a.candidate_id == "ais_coverage_query")
    assert a.status == AvailabilityStatus.FEASIBLE


def test_ais_partial_with_weak_coverage() -> None:
    catalog = _catalog_from((
        _make_record(
            record_id="a1", sensor_type="ais", coverage_score=0.40,
            quality_flag="yellow", look_geometry=None, resolution_m=None,
        ),
    ))
    rec = _rec((_value("ais_coverage_query", 0.5),))
    assessments = assess_collect_feasibility(rec, catalog)
    a = next(a for a in assessments if a.candidate_id == "ais_coverage_query")
    assert a.status == AvailabilityStatus.PARTIAL


# ---------------------------------------------------------------------------
# Feasibility - wait_or_monitor
# ---------------------------------------------------------------------------


def test_wait_or_monitor_always_feasible_low_score() -> None:
    catalog = _catalog_from(())
    rec = _rec((_value("wait_or_monitor", 0.15),))
    assessments = assess_collect_feasibility(rec, catalog)
    a = next(a for a in assessments if a.candidate_id == "wait_or_monitor")
    assert a.status == AvailabilityStatus.FEASIBLE
    assert a.feasibility_score == 0.25


# ---------------------------------------------------------------------------
# Adjusted values
# ---------------------------------------------------------------------------


def test_adjusted_score_equals_original_times_feasibility() -> None:
    catalog = load_scene_availability_catalog(TENNENT_FIXTURE)
    rec = _rec((
        _value("optical_context", 0.85),
        _value("repeat_sar", 0.60),
        _value("ais_coverage_query", 0.15),
    ))
    report = adjust_recommendation_for_availability(rec, catalog)
    for v in report.adjusted_values:
        assert abs(v.adjusted_score - v.original_score * v.feasibility_score) < 1e-9


def test_adjusted_values_sorted_by_adjusted_score_desc() -> None:
    catalog = load_scene_availability_catalog(TENNENT_FIXTURE)
    rec = _rec((
        _value("optical_context", 0.85),
        _value("repeat_sar", 0.60),
        _value("wait_or_monitor", 0.15),
        _value("ais_coverage_query", 0.15),
    ))
    report = adjust_recommendation_for_availability(rec, catalog)
    scores = [v.adjusted_score for v in report.adjusted_values]
    assert scores == sorted(scores, reverse=True)


def test_recommendation_not_mutated() -> None:
    catalog = load_scene_availability_catalog(TENNENT_FIXTURE)
    rec = _rec((_value("optical_context", 0.85),))
    snapshot = (rec.scenario_id, rec.health_status, rec.ranked_values)
    adjust_recommendation_for_availability(rec, catalog)
    assert (rec.scenario_id, rec.health_status, rec.ranked_values) == snapshot


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_recommendation_to_dict_json_serializable() -> None:
    catalog = load_scene_availability_catalog(TENNENT_FIXTURE)
    rec = _rec((_value("optical_context", 0.85),))
    report = adjust_recommendation_for_availability(rec, catalog)
    d = recommendation_to_dict(report)
    encoded = json.dumps(d)
    assert json.loads(encoded) == d


def test_recommendation_to_json_parseable() -> None:
    catalog = load_scene_availability_catalog(WHITSUN_FIXTURE)
    rec = _rec((_value("ais_coverage_query", 0.75),))
    report = adjust_recommendation_for_availability(rec, catalog)
    parsed = json.loads(recommendation_to_json(report))
    assert "adjusted_values" in parsed


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------


def test_no_forbidden_imports() -> None:
    import custody.hypotheses.scene_availability as mod
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
    import custody.hypotheses.scene_availability as mod
    src = Path(mod.__file__).read_text(encoding="utf-8").lower()
    forbidden = (
        "sentinel integration",
        "real-time ingestion",
        "live data ingestion",
        "tasking order",
        "live tasking",
        "sensor command",
        "platform access",
        "production scheduler",
        "autonomous execution",
        "collection order",
        "revenue dollars",
    )
    for needle in forbidden:
        assert needle not in src, f"forbidden token {needle!r} in source"


def test_text_format_contains_required_sections() -> None:
    catalog = load_scene_availability_catalog(TENNENT_FIXTURE)
    rec = _rec((
        _value("optical_context", 0.85),
        _value("repeat_sar", 0.60),
    ))
    report = adjust_recommendation_for_availability(rec, catalog)
    text = format_availability_text(report)
    assert "SCENE AVAILABILITY BRIDGE" in text
    assert "Catalog summary" in text
    assert "Candidate feasibility" in text
    assert "Availability-adjusted recommendation" in text
    assert "Caveats" in text


def test_markdown_format_contains_heading() -> None:
    catalog = load_scene_availability_catalog(WHITSUN_FIXTURE)
    rec = _rec((_value("ais_coverage_query", 0.75),))
    report = adjust_recommendation_for_availability(rec, catalog)
    md = format_availability_markdown(report)
    assert md.startswith("# Scene Availability Bridge")
