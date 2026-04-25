"""Scene-availability metadata bridge (ADR-0021 Slice 20).

Assesses whether recommended candidate collect TYPES are feasible
given provider-neutral scene/collection availability metadata, and
adjusts candidate collect recommendations by feasibility.

This is **metadata-only**.  No imagery is downloaded, no external APIs
are fetched, no Sentinel data is ingested, no pixels are processed,
no VLM or matcher is run, and no execution authorizations are issued.

Core distinction
----------------

- Artifact bridge (Slice 19) = existing/generated outputs become
  ``HypothesisEvidence``.
- Scene-availability bridge (Slice 20) = provider-neutral collection
  metadata informs whether candidate collect types are feasible.

Availability metadata does **not** change target hypotheses directly.
It adjusts recommendation scores only.

Design constraints
------------------

- Deterministic: no wall-clock, no randomness.
- Pure stdlib: no new runtime dependencies.
- No imports from ``custody.detection.*``, ``custody.ingest.gfw_presence``,
  matcher runtime, or external SDKs.
"""
from __future__ import annotations

import json as _json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from custody.hypotheses.collection_value import (
    CollectionRecommendation,
    CollectionValue,
)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class AvailabilityStatus(Enum):
    FEASIBLE = "feasible"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SceneAvailabilityRecord:
    record_id: str
    scenario_id: str
    sensor_type: str
    source_ref: str
    collection_time: datetime | None
    quality_flag: str | None
    coverage_score: float | None
    cloud_cover: float | None
    resolution_m: float | None
    look_geometry: str | None
    latency_hours: float | None
    payload: Mapping[str, object]
    notes: str | None


@dataclass(frozen=True)
class SceneAvailabilityCatalog:
    scenario_id: str
    records: tuple[SceneAvailabilityRecord, ...]
    summary: str
    caveats: tuple[str, ...]


@dataclass(frozen=True)
class CollectFeasibilityAssessment:
    candidate_id: str
    status: AvailabilityStatus
    feasibility_score: float
    supporting_record_ids: tuple[str, ...]
    reason: str
    caveats: tuple[str, ...]


@dataclass(frozen=True)
class AvailabilityAdjustedValue:
    candidate_id: str
    label: str
    original_score: float
    feasibility_score: float
    adjusted_score: float
    availability_status: AvailabilityStatus
    reason: str
    caveats: tuple[str, ...]


@dataclass(frozen=True)
class AvailabilityAdjustedRecommendation:
    scenario_id: str
    assessments: tuple[CollectFeasibilityAssessment, ...]
    adjusted_values: tuple[AvailabilityAdjustedValue, ...]
    summary: str
    caveats: tuple[str, ...]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


_VALID_SENSOR_TYPES = frozenset({"sar", "optical", "ais", "rf", "unknown"})
_VALID_SCENARIOS = ("tennent", "whitsun")
_VALID_QUALITY = ("green", "yellow", "red")

_BASE_CAVEATS: tuple[str, ...] = (
    "metadata-only bridge; no imagery was downloaded or processed",
    "feasibility scores are derived from availability metadata, not "
    "executable plans",
    "availability metadata does not change target hypotheses",
)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse_timestamp(raw: object) -> datetime | None:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        dt = raw
    elif isinstance(raw, str):
        dt = datetime.fromisoformat(raw)
    else:
        raise ValueError(
            f"collection_time must be ISO 8601 string, datetime, or null; "
            f"got {type(raw).__name__}"
        )
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _validate_unit_interval(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0.0, 1.0]; got {value}")


def scene_record_from_mapping(
    mapping: Mapping[str, object],
) -> SceneAvailabilityRecord:
    """Parse one :class:`SceneAvailabilityRecord` from a dict."""
    record_id = mapping.get("record_id")
    if not isinstance(record_id, str) or not record_id:
        raise ValueError("record_id is required and must be a non-empty string")

    scenario_id = mapping.get("scenario_id")
    if scenario_id not in _VALID_SCENARIOS:
        raise ValueError(
            f"scenario_id must be one of {_VALID_SCENARIOS}; got {scenario_id!r}"
        )

    sensor_type = mapping.get("sensor_type")
    if sensor_type not in _VALID_SENSOR_TYPES:
        raise ValueError(
            f"sensor_type must be one of {sorted(_VALID_SENSOR_TYPES)}; "
            f"got {sensor_type!r}"
        )

    source_ref = mapping.get("source_ref")
    if not isinstance(source_ref, str) or not source_ref:
        raise ValueError("source_ref is required and must be a non-empty string")

    collection_time = _parse_timestamp(mapping.get("collection_time"))

    quality_flag = mapping.get("quality_flag")
    if quality_flag is not None and quality_flag not in _VALID_QUALITY:
        raise ValueError(
            f"quality_flag must be one of {_VALID_QUALITY} or null; "
            f"got {quality_flag!r}"
        )

    coverage_raw = mapping.get("coverage_score")
    coverage_score: float | None
    if coverage_raw is None:
        coverage_score = None
    else:
        coverage_score = float(coverage_raw)
        _validate_unit_interval("coverage_score", coverage_score)

    cloud_raw = mapping.get("cloud_cover")
    cloud_cover: float | None
    if cloud_raw is None:
        cloud_cover = None
    else:
        cloud_cover = float(cloud_raw)
        _validate_unit_interval("cloud_cover", cloud_cover)

    res_raw = mapping.get("resolution_m")
    resolution_m: float | None
    if res_raw is None:
        resolution_m = None
    else:
        resolution_m = float(res_raw)
        if resolution_m <= 0:
            raise ValueError(
                f"resolution_m must be > 0 if present; got {resolution_m}"
            )

    look_raw = mapping.get("look_geometry")
    if look_raw is not None and not isinstance(look_raw, str):
        raise ValueError("look_geometry must be a string or null")

    latency_raw = mapping.get("latency_hours")
    latency_hours: float | None
    if latency_raw is None:
        latency_hours = None
    else:
        latency_hours = float(latency_raw)
        if latency_hours < 0:
            raise ValueError(
                f"latency_hours must be >= 0 if present; got {latency_hours}"
            )

    payload = mapping.get("payload", {})
    if not isinstance(payload, Mapping):
        raise ValueError("payload must be an object/mapping")

    notes_raw = mapping.get("notes")
    if notes_raw is not None and not isinstance(notes_raw, str):
        raise ValueError("notes must be a string or null")

    return SceneAvailabilityRecord(
        record_id=record_id,
        scenario_id=scenario_id,  # type: ignore[arg-type]
        sensor_type=sensor_type,  # type: ignore[arg-type]
        source_ref=source_ref,
        collection_time=collection_time,
        quality_flag=quality_flag,  # type: ignore[arg-type]
        coverage_score=coverage_score,
        cloud_cover=cloud_cover,
        resolution_m=resolution_m,
        look_geometry=look_raw,  # type: ignore[arg-type]
        latency_hours=latency_hours,
        payload=dict(payload),
        notes=notes_raw,
    )


def _summarize_catalog(records: tuple[SceneAvailabilityRecord, ...]) -> str:
    if not records:
        return "0 metadata records loaded"
    counts: dict[str, int] = {}
    for r in records:
        counts[r.sensor_type] = counts.get(r.sensor_type, 0) + 1
    parts = sorted(counts.items())
    breakdown = ", ".join(f"{name}={n}" for name, n in parts)
    return f"{len(records)} metadata records loaded ({breakdown})"


def load_scene_availability_catalog(
    path: str | Path,
    *,
    scenario_id: str | None = None,
) -> SceneAvailabilityCatalog:
    """Load scene-availability records from a JSON file (list of record dicts)."""
    with open(path, encoding="utf-8") as f:
        raw = _json.load(f)
    if not isinstance(raw, list):
        raise ValueError(
            f"availability catalog at {path} must be a JSON list of records"
        )
    records = tuple(scene_record_from_mapping(r) for r in raw)

    if scenario_id is not None:
        records = tuple(r for r in records if r.scenario_id == scenario_id)
        catalog_scenario = scenario_id
    else:
        if not records:
            raise ValueError(
                f"availability catalog at {path} is empty; "
                "cannot infer scenario_id"
            )
        seen = {r.scenario_id for r in records}
        if len(seen) > 1:
            raise ValueError(
                f"availability catalog at {path} mixes scenarios "
                f"{sorted(seen)}; pass scenario_id to filter"
            )
        catalog_scenario = next(iter(seen))

    return SceneAvailabilityCatalog(
        scenario_id=catalog_scenario,
        records=records,
        summary=_summarize_catalog(records),
        caveats=_BASE_CAVEATS,
    )


# ---------------------------------------------------------------------------
# Feasibility assessment
# ---------------------------------------------------------------------------


def _records_by_sensor(
    catalog: SceneAvailabilityCatalog, sensor_type: str,
) -> tuple[SceneAvailabilityRecord, ...]:
    return tuple(r for r in catalog.records if r.sensor_type == sensor_type)


def _assess_optical_context(
    catalog: SceneAvailabilityCatalog,
) -> CollectFeasibilityAssessment:
    optical = _records_by_sensor(catalog, "optical")
    if not optical:
        return CollectFeasibilityAssessment(
            candidate_id="optical_context",
            status=AvailabilityStatus.UNAVAILABLE,
            feasibility_score=0.0,
            supporting_record_ids=(),
            reason="no optical metadata available in catalog",
            caveats=(),
        )
    feasible_ids: list[str] = []
    partial_ids: list[str] = []
    for r in optical:
        cloud_ok = r.cloud_cover is None or r.cloud_cover <= 0.35
        coverage_ok = r.coverage_score is None or r.coverage_score >= 0.6
        quality_ok = r.quality_flag in (None, "green", "yellow")
        if cloud_ok and coverage_ok and quality_ok:
            feasible_ids.append(r.record_id)
        else:
            partial_ids.append(r.record_id)
    if feasible_ids:
        return CollectFeasibilityAssessment(
            candidate_id="optical_context",
            status=AvailabilityStatus.FEASIBLE,
            feasibility_score=0.85,
            supporting_record_ids=tuple(feasible_ids),
            reason=(
                "optical metadata available with acceptable cloud cover "
                "and coverage"
            ),
            caveats=(),
        )
    return CollectFeasibilityAssessment(
        candidate_id="optical_context",
        status=AvailabilityStatus.PARTIAL,
        feasibility_score=0.40,
        supporting_record_ids=tuple(partial_ids),
        reason="optical metadata exists but cloud cover or coverage is weak",
        caveats=(),
    )


def _assess_repeat_sar(
    catalog: SceneAvailabilityCatalog,
) -> CollectFeasibilityAssessment:
    sar = _records_by_sensor(catalog, "sar")
    if not sar:
        return CollectFeasibilityAssessment(
            candidate_id="repeat_sar",
            status=AvailabilityStatus.UNAVAILABLE,
            feasibility_score=0.0,
            supporting_record_ids=(),
            reason="no SAR metadata available in catalog",
            caveats=(),
        )
    feasible_ids: list[str] = []
    partial_ids: list[str] = []
    for r in sar:
        coverage_ok = r.coverage_score is None or r.coverage_score >= 0.5
        quality_ok = r.quality_flag != "red"
        if coverage_ok and quality_ok:
            feasible_ids.append(r.record_id)
        else:
            partial_ids.append(r.record_id)
    if feasible_ids:
        return CollectFeasibilityAssessment(
            candidate_id="repeat_sar",
            status=AvailabilityStatus.FEASIBLE,
            feasibility_score=0.70,
            supporting_record_ids=tuple(feasible_ids),
            reason="SAR metadata available for repeat confirmation",
            caveats=(),
        )
    return CollectFeasibilityAssessment(
        candidate_id="repeat_sar",
        status=AvailabilityStatus.PARTIAL,
        feasibility_score=0.35,
        supporting_record_ids=tuple(partial_ids),
        reason="SAR metadata exists but coverage or quality is weak",
        caveats=(),
    )


def _assess_cross_geometry_sar(
    catalog: SceneAvailabilityCatalog,
) -> CollectFeasibilityAssessment:
    sar = _records_by_sensor(catalog, "sar")
    if not sar:
        return CollectFeasibilityAssessment(
            candidate_id="cross_geometry_sar",
            status=AvailabilityStatus.UNAVAILABLE,
            feasibility_score=0.0,
            supporting_record_ids=(),
            reason="no SAR metadata available in catalog",
            caveats=(),
        )
    geometries = {r.look_geometry for r in sar if r.look_geometry is not None}
    cross_records = tuple(r for r in sar if r.look_geometry == "cross")
    if cross_records:
        return CollectFeasibilityAssessment(
            candidate_id="cross_geometry_sar",
            status=AvailabilityStatus.FEASIBLE,
            feasibility_score=0.80,
            supporting_record_ids=tuple(r.record_id for r in cross_records),
            reason="SAR metadata includes a cross-geometry record",
            caveats=(),
        )
    if len(geometries) >= 2:
        supporting = tuple(r.record_id for r in sar if r.look_geometry is not None)
        return CollectFeasibilityAssessment(
            candidate_id="cross_geometry_sar",
            status=AvailabilityStatus.FEASIBLE,
            feasibility_score=0.80,
            supporting_record_ids=supporting,
            reason="SAR metadata contains distinct look geometries",
            caveats=(),
        )
    return CollectFeasibilityAssessment(
        candidate_id="cross_geometry_sar",
        status=AvailabilityStatus.PARTIAL,
        feasibility_score=0.35,
        supporting_record_ids=tuple(r.record_id for r in sar),
        reason="SAR metadata exists but geometry diversity is unknown",
        caveats=(),
    )


def _assess_higher_resolution_sar(
    catalog: SceneAvailabilityCatalog,
) -> CollectFeasibilityAssessment:
    sar = _records_by_sensor(catalog, "sar")
    if not sar:
        return CollectFeasibilityAssessment(
            candidate_id="higher_resolution_sar",
            status=AvailabilityStatus.UNAVAILABLE,
            feasibility_score=0.0,
            supporting_record_ids=(),
            reason="no SAR metadata available in catalog",
            caveats=(),
        )
    feasible: list[str] = []
    for r in sar:
        if r.resolution_m is not None and r.resolution_m <= 1.0:
            coverage_ok = r.coverage_score is None or r.coverage_score >= 0.5
            if coverage_ok:
                feasible.append(r.record_id)
    if feasible:
        return CollectFeasibilityAssessment(
            candidate_id="higher_resolution_sar",
            status=AvailabilityStatus.FEASIBLE,
            feasibility_score=0.75,
            supporting_record_ids=tuple(feasible),
            reason="SAR metadata has sub-meter resolution available",
            caveats=(),
        )
    return CollectFeasibilityAssessment(
        candidate_id="higher_resolution_sar",
        status=AvailabilityStatus.PARTIAL,
        feasibility_score=0.35,
        supporting_record_ids=tuple(r.record_id for r in sar),
        reason="SAR metadata exists but resolution is unknown or above 1 m",
        caveats=(),
    )


def _assess_ais_coverage_query(
    catalog: SceneAvailabilityCatalog,
) -> CollectFeasibilityAssessment:
    ais = _records_by_sensor(catalog, "ais")
    if not ais:
        return CollectFeasibilityAssessment(
            candidate_id="ais_coverage_query",
            status=AvailabilityStatus.UNAVAILABLE,
            feasibility_score=0.0,
            supporting_record_ids=(),
            reason="no AIS metadata available in catalog",
            caveats=(),
        )
    feasible: list[str] = []
    for r in ais:
        coverage_ok = r.coverage_score is not None and r.coverage_score >= 0.6
        quality_ok = r.quality_flag == "green"
        if coverage_ok or quality_ok:
            feasible.append(r.record_id)
    if feasible:
        return CollectFeasibilityAssessment(
            candidate_id="ais_coverage_query",
            status=AvailabilityStatus.FEASIBLE,
            feasibility_score=0.70,
            supporting_record_ids=tuple(feasible),
            reason="AIS metadata indicates good coverage for the area",
            caveats=(),
        )
    return CollectFeasibilityAssessment(
        candidate_id="ais_coverage_query",
        status=AvailabilityStatus.PARTIAL,
        feasibility_score=0.35,
        supporting_record_ids=tuple(r.record_id for r in ais),
        reason="AIS metadata exists but coverage quality is weak",
        caveats=(),
    )


def _assess_wait_or_monitor(
    catalog: SceneAvailabilityCatalog,
) -> CollectFeasibilityAssessment:
    return CollectFeasibilityAssessment(
        candidate_id="wait_or_monitor",
        status=AvailabilityStatus.FEASIBLE,
        feasibility_score=0.25,
        supporting_record_ids=(),
        reason="no immediate external collect required",
        caveats=(),
    )


_ASSESSORS = {
    "optical_context": _assess_optical_context,
    "repeat_sar": _assess_repeat_sar,
    "cross_geometry_sar": _assess_cross_geometry_sar,
    "higher_resolution_sar": _assess_higher_resolution_sar,
    "ais_coverage_query": _assess_ais_coverage_query,
    "wait_or_monitor": _assess_wait_or_monitor,
}


def assess_collect_feasibility(
    recommendation: CollectionRecommendation,
    catalog: SceneAvailabilityCatalog,
) -> tuple[CollectFeasibilityAssessment, ...]:
    """Assess feasibility of each candidate collect against availability metadata."""
    seen: set[str] = set()
    assessments: list[CollectFeasibilityAssessment] = []
    for cv in recommendation.ranked_values:
        cid = cv.candidate.candidate_id
        if cid in seen:
            continue
        seen.add(cid)
        assessor = _ASSESSORS.get(cid)
        if assessor is None:
            assessments.append(CollectFeasibilityAssessment(
                candidate_id=cid,
                status=AvailabilityStatus.UNKNOWN,
                feasibility_score=0.5,
                supporting_record_ids=(),
                reason=f"no feasibility assessor for candidate {cid!r}",
                caveats=(),
            ))
        else:
            assessments.append(assessor(catalog))
    assessments.sort(key=lambda a: (-a.feasibility_score, a.candidate_id))
    return tuple(assessments)


# ---------------------------------------------------------------------------
# Adjusted recommendation
# ---------------------------------------------------------------------------


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def adjust_recommendation_for_availability(
    recommendation: CollectionRecommendation,
    catalog: SceneAvailabilityCatalog,
) -> AvailabilityAdjustedRecommendation:
    """Combine collection-value scores with feasibility into an adjusted report."""
    assessments = assess_collect_feasibility(recommendation, catalog)
    by_id = {a.candidate_id: a for a in assessments}

    adjusted: list[AvailabilityAdjustedValue] = []
    for cv in recommendation.ranked_values:
        cid = cv.candidate.candidate_id
        a = by_id.get(cid)
        if a is None:
            feasibility = 0.5
            status = AvailabilityStatus.UNKNOWN
            reason = f"no feasibility assessor for candidate {cid!r}"
        else:
            feasibility = a.feasibility_score
            status = a.status
            reason = a.reason
        adjusted_score = _clamp01(cv.score * feasibility)
        adjusted.append(AvailabilityAdjustedValue(
            candidate_id=cid,
            label=cv.candidate.label,
            original_score=cv.score,
            feasibility_score=feasibility,
            adjusted_score=adjusted_score,
            availability_status=status,
            reason=reason,
            caveats=(),
        ))

    adjusted.sort(key=lambda v: (-v.adjusted_score, v.candidate_id))

    summary = (
        f"availability adjustment for {recommendation.scenario_id}: "
        f"{len(catalog.records)} records, "
        f"{len(adjusted)} candidate(s) assessed"
    )
    caveats = catalog.caveats + (
        "scores combine collection-value ranking with metadata-derived "
        "feasibility; no executable plan is implied",
    )
    return AvailabilityAdjustedRecommendation(
        scenario_id=recommendation.scenario_id,
        assessments=assessments,
        adjusted_values=tuple(adjusted),
        summary=summary,
        caveats=caveats,
    )


# ---------------------------------------------------------------------------
# Serializers
# ---------------------------------------------------------------------------


def recommendation_to_dict(
    report: AvailabilityAdjustedRecommendation,
) -> dict:
    return {
        "scenario_id": report.scenario_id,
        "assessments": [
            {
                "candidate_id": a.candidate_id,
                "status": a.status.value,
                "feasibility_score": a.feasibility_score,
                "supporting_record_ids": list(a.supporting_record_ids),
                "reason": a.reason,
                "caveats": list(a.caveats),
            }
            for a in report.assessments
        ],
        "adjusted_values": [
            {
                "candidate_id": v.candidate_id,
                "label": v.label,
                "original_score": v.original_score,
                "feasibility_score": v.feasibility_score,
                "adjusted_score": v.adjusted_score,
                "availability_status": v.availability_status.value,
                "reason": v.reason,
                "caveats": list(v.caveats),
            }
            for v in report.adjusted_values
        ],
        "summary": report.summary,
        "caveats": list(report.caveats),
    }


def recommendation_to_json(
    report: AvailabilityAdjustedRecommendation,
) -> str:
    return _json.dumps(recommendation_to_dict(report), indent=2)


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def _hr(title: str) -> str:
    underline = "-" * max(len(title), 3)
    return f"\n{title}\n{underline}\n"


def format_availability_text(
    report: AvailabilityAdjustedRecommendation,
    *,
    catalog_path: str | None = None,
) -> str:
    parts: list[str] = []
    title = f"SCENE AVAILABILITY BRIDGE - {report.scenario_id.upper()}"
    parts.append(title + "\n")
    parts.append("=" * len(title) + "\n")

    parts.append(_hr("Catalog summary"))
    if catalog_path:
        parts.append(f"Catalog: {catalog_path}\n")
    parts.append(report.summary + "\n")

    parts.append(_hr("Candidate feasibility"))
    for a in report.assessments:
        parts.append(
            f"  {a.candidate_id:22s} {a.status.value.upper():12s} "
            f"{a.feasibility_score:4.2f}  {a.reason}\n"
        )

    parts.append(_hr("Availability-adjusted recommendation"))
    for v in report.adjusted_values:
        parts.append(
            f"  {v.adjusted_score:4.2f}  {v.candidate_id:22s} "
            f"({v.availability_status.value})\n"
        )

    parts.append(_hr("Caveats"))
    for c in report.caveats:
        parts.append(f"  - {c}\n")

    return "".join(parts)


def format_availability_markdown(
    report: AvailabilityAdjustedRecommendation,
    *,
    catalog_path: str | None = None,
) -> str:
    lines: list[str] = []
    name = report.scenario_id.capitalize()
    lines.append(f"# Scene Availability Bridge - {name}")
    lines.append("")
    if catalog_path:
        lines.append(f"- **Catalog**: `{catalog_path}`")
    lines.append(f"- {report.summary}")
    lines.append("")
    lines.append("## Candidate feasibility")
    lines.append("")
    for a in report.assessments:
        lines.append(
            f"- `{a.candidate_id}` - **{a.status.value.upper()}** "
            f"({a.feasibility_score:.2f}) - {a.reason}"
        )
    lines.append("")
    lines.append("## Availability-adjusted recommendation")
    lines.append("")
    for v in report.adjusted_values:
        lines.append(
            f"- `{v.candidate_id}` ({v.adjusted_score:.2f}) - "
            f"{v.label} - status `{v.availability_status.value}`"
        )
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    for c in report.caveats:
        lines.append(f"- {c}")
    lines.append("")
    return "\n".join(lines)
