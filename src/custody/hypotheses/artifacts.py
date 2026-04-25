"""Artifact-manifest evidence bridge (ADR-0021 Slice 19).

Converts existing artifact-style outputs - scene metadata, SAR/VLM
summaries, matcher outputs, AIS/GFW presence summaries, quality flags,
and manual labels - into :class:`HypothesisEvidence` and runs the
existing decision pipeline.

This is **artifact wiring**, not new ingestion.  No imagery is
processed, no VLM is invoked, no matcher is run, no Sentinel-1 or
Sentinel-2 data is fetched, and no external data sources are accessed.

Two mapping paths
-----------------

1. **Explicit semantic path**: artifact payload contains ``supports``
   and ``contradicts`` fields -> route to ``evidence.from_mapping()``.

2. **Scenario signal path**: artifact payload contains recognised
   observable fields (e.g. ``persistent_scatterer_detected``,
   ``vessel_count``) -> construct a lightweight synthetic Scene/Match
   and route to the existing scenario evidence generators in
   ``scenarios.py``.

If neither path matches, the artifact produces no evidence and a
caveat is emitted.

Design constraints
------------------

- Deterministic: no wall-clock, no randomness.
- Pure stdlib + numpy (for Scene construction only).
- No imports from ``custody.detection.*``, ``custody.ingest.gfw_presence``,
  matcher runtime, or external SDKs.
"""
from __future__ import annotations

import json as _json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import numpy as np

from custody.fusion.observations import PositionObservation
from custody.fusion.scenes import Scene
from custody.fusion.temporal import Match
from custody.hypotheses.collection_value import (
    CollectionRecommendation,
    rank_collection_candidates,
)
from custody.hypotheses.custody_health import (
    HypothesisCustodyHealth,
    assess_custody_health,
)
from custody.hypotheses.decision_packet import (
    build_decision_packet,
    format_as_markdown as _packet_format_as_markdown,
    packet_to_json_object,
)
from custody.hypotheses.evidence import from_mapping
from custody.hypotheses.registry import (
    SCENARIO_TENNENT,
    SCENARIO_WHITSUN,
    get_hypothesis_ids,
)
from custody.hypotheses.scenarios import (
    AisCoverage,
    tennent_evidence_from_match,
    tennent_evidence_from_scene,
    whitsun_evidence_from_match,
    whitsun_evidence_from_scene,
)
from custody.hypotheses.types import HypothesisEvidence, HypothesisState
from custody.hypotheses.update import update_state


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class ArtifactKind(Enum):
    SCENE_SIGNAL = "scene_signal"
    SAR_DETECTION_SUMMARY = "sar_detection_summary"
    VLM_SUMMARY = "vlm_summary"
    MATCHER_PERSISTENCE = "matcher_persistence"
    AIS_PRESENCE_SUMMARY = "ais_presence_summary"
    QUALITY_FLAG = "quality_flag"
    MANUAL_LABEL = "manual_label"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ArtifactRecord:
    artifact_id: str
    scenario_id: str
    kind: str
    source_ref: str
    timestamp: datetime | None
    quality_flag: str | None
    confidence: float | None
    payload: Mapping[str, object]
    notes: str | None


@dataclass(frozen=True)
class ArtifactEvidenceBundle:
    scenario_id: str
    records: tuple[ArtifactRecord, ...]
    evidence: tuple[HypothesisEvidence, ...]
    summary: str
    caveats: tuple[str, ...]


@dataclass(frozen=True)
class ArtifactDecisionBundle:
    scenario_id: str
    artifact_bundle: ArtifactEvidenceBundle
    final_state: HypothesisState
    health: HypothesisCustodyHealth
    recommendation: CollectionRecommendation
    summary: str
    caveats: tuple[str, ...]


# ---------------------------------------------------------------------------
# Scenario centers (lightweight; matches scripts 13-21)
# ---------------------------------------------------------------------------


_TENNENT_CENTER = (8.856, 114.665)
_WHITSUN_CENTER = (9.98, 114.63)


def _scenario_center(scenario_id: str) -> tuple[float, float]:
    if scenario_id == SCENARIO_TENNENT:
        return _TENNENT_CENTER
    return _WHITSUN_CENTER


_VALID_QUALITY = ("green", "yellow", "red")
_VALID_SCENARIOS = (SCENARIO_TENNENT, SCENARIO_WHITSUN)


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
            f"timestamp must be ISO 8601 string, datetime, or null; "
            f"got {type(raw).__name__}"
        )
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def artifact_record_from_mapping(mapping: Mapping[str, object]) -> ArtifactRecord:
    """Parse one :class:`ArtifactRecord` from a dict (e.g. loaded JSON)."""
    artifact_id = mapping.get("artifact_id")
    if not isinstance(artifact_id, str) or not artifact_id:
        raise ValueError("artifact_id is required and must be a non-empty string")

    scenario_id = mapping.get("scenario_id")
    if scenario_id not in _VALID_SCENARIOS:
        raise ValueError(
            f"scenario_id must be one of {_VALID_SCENARIOS}; got {scenario_id!r}"
        )

    kind = mapping.get("kind")
    if not isinstance(kind, str) or not kind:
        raise ValueError("kind is required and must be a non-empty string")

    source_ref = mapping.get("source_ref")
    if not isinstance(source_ref, str) or not source_ref:
        raise ValueError("source_ref is required and must be a non-empty string")

    timestamp = _parse_timestamp(mapping.get("timestamp"))

    quality_flag = mapping.get("quality_flag")
    if quality_flag is not None:
        if quality_flag not in _VALID_QUALITY:
            raise ValueError(
                f"quality_flag must be one of {_VALID_QUALITY} or null; "
                f"got {quality_flag!r}"
            )

    confidence_raw = mapping.get("confidence")
    confidence: float | None
    if confidence_raw is None:
        confidence = None
    else:
        confidence = float(confidence_raw)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0.0, 1.0]; got {confidence}"
            )

    payload = mapping.get("payload", {})
    if not isinstance(payload, Mapping):
        raise ValueError("payload must be an object/mapping")

    notes_raw = mapping.get("notes")
    if notes_raw is not None and not isinstance(notes_raw, str):
        raise ValueError("notes must be a string or null")

    return ArtifactRecord(
        artifact_id=artifact_id,
        scenario_id=scenario_id,
        kind=kind,
        source_ref=source_ref,
        timestamp=timestamp,
        quality_flag=quality_flag,  # type: ignore[arg-type]
        confidence=confidence,
        payload=dict(payload),
        notes=notes_raw,
    )


def load_artifact_manifest(path: str | Path) -> tuple[ArtifactRecord, ...]:
    """Load artifact records from a JSON file (list of record dicts)."""
    with open(path, encoding="utf-8") as f:
        raw = _json.load(f)
    if not isinstance(raw, list):
        raise ValueError(
            f"artifact manifest at {path} must be a JSON list of records"
        )
    return tuple(artifact_record_from_mapping(r) for r in raw)


# ---------------------------------------------------------------------------
# Synthetic object helpers
# ---------------------------------------------------------------------------


def _make_artifact_obs(record: ArtifactRecord) -> PositionObservation:
    center = _scenario_center(record.scenario_id)
    acq_time = record.timestamp.timestamp() if record.timestamp else 0.0
    return PositionObservation(
        obs_id=f"artifact-obs-{record.artifact_id}",
        source_id=record.source_ref,
        modality="SAR",  # type: ignore[arg-type]
        acquisition_time=acq_time,
        ingestion_time=acq_time + 1.0,
        lat=center[0],
        lon=center[1],
        cov_pos=np.eye(2) * 100.0,
        raw_ref=f"artifact://{record.artifact_id}",
    )


def _scene_id_for_artifact(record: ArtifactRecord) -> str:
    """Build a Scene-id pattern compliant with ``{case_study}_{yyyymmdd}_{sensor}``."""
    if record.timestamp is not None:
        yyyymmdd = record.timestamp.strftime("%Y%m%d")
    else:
        yyyymmdd = "19700101"
    safe_id = "".join(
        ch.lower() if ch.isalnum() or ch == "-" else "-"
        for ch in record.artifact_id
    ).strip("-") or "artifact"
    return f"{record.scenario_id}_{yyyymmdd}_artifact-{safe_id}"


def _make_artifact_scene(
    record: ArtifactRecord,
    obs_tuple: tuple[PositionObservation, ...] = (),
) -> Scene:
    center = _scenario_center(record.scenario_id)
    acq_time = record.timestamp.timestamp() if record.timestamp else 0.0
    quality = record.quality_flag or "yellow"
    return Scene(
        scene_id=_scene_id_for_artifact(record),
        sensor="artifact",
        acquisition_time=acq_time,
        pixel_size_m=0.25,
        center_lat=center[0],
        center_lon=center[1],
        footprint_latlon=(
            (center[0] - 0.02, center[1] - 0.02),
            (center[0] - 0.02, center[1] + 0.02),
            (center[0] + 0.02, center[1] + 0.02),
            (center[0] + 0.02, center[1] - 0.02),
        ),
        raw_scene_path=None,
        quality_flag=quality,  # type: ignore[arg-type]
        observations=obs_tuple,
    )


def _make_artifact_match(record: ArtifactRecord) -> Match:
    payload = record.payload
    obs_a = payload.get("obs_a_id", f"artifact-a-{record.artifact_id}")
    obs_b = payload.get("obs_b_id", f"artifact-b-{record.artifact_id}")
    dist_raw = payload.get("distance_m", payload.get("displacement_m", 15.0))
    return Match(
        obs_a_id=str(obs_a), obs_b_id=str(obs_b), distance_m=float(dist_raw),
    )


# ---------------------------------------------------------------------------
# Evidence conversion
# ---------------------------------------------------------------------------


def _has_explicit_semantics(payload: Mapping[str, object]) -> bool:
    return (
        "supports" in payload
        and "contradicts" in payload
        and isinstance(payload["supports"], list)
        and isinstance(payload["contradicts"], list)
    )


def _explicit_semantic_evidence(
    record: ArtifactRecord,
) -> tuple[HypothesisEvidence, ...]:
    payload = record.payload
    supports_raw = payload["supports"]
    contradicts_raw = payload["contradicts"]
    supports = tuple(str(x) for x in supports_raw)  # type: ignore[arg-type]
    contradicts = tuple(str(x) for x in contradicts_raw)  # type: ignore[arg-type]

    valid_ids = set(get_hypothesis_ids(record.scenario_id))
    for hid in (*supports, *contradicts):
        if hid not in valid_ids:
            raise ValueError(
                f"artifact {record.artifact_id!r} references hypothesis_id "
                f"{hid!r} not registered for scenario {record.scenario_id!r}; "
                f"valid: {sorted(valid_ids)}"
            )

    reason_raw = payload.get("reason") or record.notes or (
        f"artifact {record.artifact_id}"
    )
    reason = str(reason_raw)

    confidence = record.confidence if record.confidence is not None else 0.6

    ev = from_mapping(
        dict(payload),
        source_kind=record.kind,
        scenario_id=record.scenario_id,
        supports=supports,
        contradicts=contradicts,
        confidence=confidence,
        reason=reason,
        evidence_id=record.artifact_id,
        source_ref=record.source_ref,
        timestamp=record.timestamp,
    )
    return (ev,)


def artifact_record_to_evidence(
    record: ArtifactRecord,
) -> tuple[HypothesisEvidence, ...]:
    """Convert one :class:`ArtifactRecord` to zero or more evidence items.

    Returns an empty tuple (without raising) if the record has no
    recognised mapping path.  Callers collect those as caveats.
    """
    payload = record.payload

    # Path 1: explicit semantic.
    if _has_explicit_semantics(payload):
        return _explicit_semantic_evidence(record)

    # Path 2: scenario signal path.
    quality = record.quality_flag or "yellow"

    if record.scenario_id == SCENARIO_TENNENT:
        if (
            "persistent_scatterer_detected" in payload
            or "change_signal_strength" in payload
        ):
            obs = _make_artifact_obs(record)
            scene = _make_artifact_scene(record, obs_tuple=(obs,))
            return tennent_evidence_from_scene(
                scene,
                persistent_scatterer_detected=bool(
                    payload.get("persistent_scatterer_detected", False),
                ),
                change_signal_strength=float(
                    payload.get("change_signal_strength", 0.0),
                ),
                quality_flag=quality,
            )
        if "displacement_m" in payload:
            match = _make_artifact_match(record)
            ev = tennent_evidence_from_match(
                match,
                displacement_m=float(payload["displacement_m"]),
                quality_flag=quality,
            )
            return (ev,)

    elif record.scenario_id == SCENARIO_WHITSUN:
        if "vessel_count" in payload:
            obs = _make_artifact_obs(record)
            scene = _make_artifact_scene(record, obs_tuple=(obs,))
            ais_flag_raw = payload.get("ais_coverage_flag", "sparse")
            ais_flag: AisCoverage = str(ais_flag_raw)  # type: ignore[assignment]
            return whitsun_evidence_from_scene(
                scene,
                vessel_count=int(payload["vessel_count"]),
                ais_coverage_flag=ais_flag,
                quality_flag=quality,
            )
        if "matched_cluster" in payload:
            match = _make_artifact_match(record)
            ev = whitsun_evidence_from_match(
                match,
                matched_cluster=bool(payload["matched_cluster"]),
                quality_flag=quality,
            )
            return (ev,)

    return ()


# ---------------------------------------------------------------------------
# Bundle builders
# ---------------------------------------------------------------------------


_BASE_CAVEATS: tuple[str, ...] = (
    "artifact bridge consumes existing artifact-style outputs; it does "
    "not run detection or ingest live data",
    "artifact evidence is candidate evidence, not ground truth",
)


def artifact_records_to_evidence(
    records: Iterable[ArtifactRecord],
    *,
    scenario_id: str,
) -> ArtifactEvidenceBundle:
    """Convert a collection of artifact records to evidence for one scenario."""
    recs = tuple(r for r in records if r.scenario_id == scenario_id)
    all_evidence: list[HypothesisEvidence] = []
    caveats: list[str] = list(_BASE_CAVEATS)
    produced_count = 0
    no_evidence_count = 0
    for rec in recs:
        evs = artifact_record_to_evidence(rec)
        if evs:
            all_evidence.extend(evs)
            produced_count += 1
        else:
            no_evidence_count += 1
            caveats.append(
                f"artifact {rec.artifact_id!r} did not contain recognized "
                f"semantic fields; no evidence generated"
            )
    summary = (
        f"{len(recs)} artifact records for {scenario_id}: "
        f"{produced_count} produced evidence, {no_evidence_count} produced none"
    )
    return ArtifactEvidenceBundle(
        scenario_id=scenario_id,
        records=recs,
        evidence=tuple(all_evidence),
        summary=summary,
        caveats=tuple(caveats),
    )


def build_decision_from_artifacts(
    records: Iterable[ArtifactRecord],
    *,
    scenario_id: str,
) -> ArtifactDecisionBundle:
    """Build a full decision bundle from artifact records."""
    bundle = artifact_records_to_evidence(records, scenario_id=scenario_id)
    state = update_state(scenario_id, bundle.evidence)
    health = assess_custody_health(state)
    rec = rank_collection_candidates(state, health)
    summary = (
        f"artifact decision for {scenario_id}: health={health.status.value} "
        f"({health.score:.2f}), top={state.top_hypothesis}"
    )
    caveats = bundle.caveats + (
        "decision produced from artifact evidence only; no live data ingested",
    )
    return ArtifactDecisionBundle(
        scenario_id=scenario_id,
        artifact_bundle=bundle,
        final_state=state,
        health=health,
        recommendation=rec,
        summary=summary,
        caveats=caveats,
    )


# ---------------------------------------------------------------------------
# Serializers
# ---------------------------------------------------------------------------


def bundle_to_dict(
    bundle: ArtifactEvidenceBundle | ArtifactDecisionBundle,
) -> dict:
    """JSON-serializable dict for an evidence or decision bundle."""
    if isinstance(bundle, ArtifactDecisionBundle):
        packet = build_decision_packet(
            scenario_id=bundle.scenario_id,
            state=bundle.final_state,
            health=bundle.health,
            recommendation=bundle.recommendation,
        )
        return {
            "scenario_id": bundle.scenario_id,
            "records_loaded": len(bundle.artifact_bundle.records),
            "evidence_generated": len(bundle.artifact_bundle.evidence),
            "decision_packet": packet_to_json_object(packet),
            "summary": bundle.summary,
            "caveats": list(bundle.caveats),
        }
    return {
        "scenario_id": bundle.scenario_id,
        "records_loaded": len(bundle.records),
        "evidence_generated": len(bundle.evidence),
        "evidence_ids": [e.evidence_id for e in bundle.evidence],
        "summary": bundle.summary,
        "caveats": list(bundle.caveats),
    }


def bundle_to_json(
    bundle: ArtifactEvidenceBundle | ArtifactDecisionBundle,
) -> str:
    return _json.dumps(bundle_to_dict(bundle), indent=2)


# ---------------------------------------------------------------------------
# Text / Markdown formatters
# ---------------------------------------------------------------------------


def _hr(title: str) -> str:
    underline = "-" * max(len(title), 3)
    return f"\n{title}\n{underline}\n"


def format_artifact_decision_text(
    bundle: ArtifactDecisionBundle,
    *,
    manifest_path: str | None = None,
) -> str:
    """Human-readable artifact decision packet."""
    parts: list[str] = []
    title = f"ARTIFACT DECISION PACKET - {bundle.scenario_id.upper()}"
    parts.append(title + "\n")
    parts.append("=" * len(title) + "\n")

    parts.append(_hr("Artifact manifest"))
    if manifest_path:
        parts.append(f"Manifest: {manifest_path}\n")
    parts.append(f"Records loaded:     {len(bundle.artifact_bundle.records)}\n")
    parts.append(
        f"Evidence generated: {len(bundle.artifact_bundle.evidence)}\n"
    )

    parts.append(_hr("Custody health"))
    health = bundle.health
    parts.append(
        f"Status: {health.status.value.upper()} (score {health.score:.2f})\n"
    )
    parts.append(f"Reason: {health.reason}\n")

    parts.append(_hr("Primary ambiguity"))
    if health.ambiguity_pairs:
        a, b = health.ambiguity_pairs[0]
        parts.append(f"{a}  vs  {b}\n")
    else:
        parts.append("None surfaced by custody-health assessment\n")

    parts.append(_hr("Recommended candidate collects"))
    rec = bundle.recommendation
    if rec.ranked_values:
        for cv in rec.ranked_values:
            parts.append(
                f"  {cv.score:4.2f}  {cv.candidate.label} "
                f"({cv.candidate.candidate_id})\n"
            )
    else:
        parts.append("None\n")

    parts.append(_hr("Caveats"))
    for c in bundle.caveats:
        parts.append(f"  - {c}\n")

    return "".join(parts)


def format_artifact_decision_markdown(
    bundle: ArtifactDecisionBundle,
    *,
    manifest_path: str | None = None,
) -> str:
    """Markdown artifact decision packet."""
    lines: list[str] = []
    name = bundle.scenario_id.capitalize()
    lines.append(f"# Artifact Decision Packet - {name}")
    lines.append("")
    if manifest_path:
        lines.append(f"- **Manifest**: `{manifest_path}`")
    lines.append(
        f"- **Records loaded**: {len(bundle.artifact_bundle.records)}"
    )
    lines.append(
        f"- **Evidence generated**: {len(bundle.artifact_bundle.evidence)}"
    )
    lines.append("")
    health = bundle.health
    lines.append("## Custody health")
    lines.append("")
    lines.append(
        f"- Status: **{health.status.value.upper()}** "
        f"(score {health.score:.2f})"
    )
    lines.append(f"- Reason: {health.reason}")
    lines.append("")
    lines.append("## Primary ambiguity")
    lines.append("")
    if health.ambiguity_pairs:
        a, b = health.ambiguity_pairs[0]
        lines.append(f"- {a} vs {b}")
    else:
        lines.append("- None surfaced by custody-health assessment")
    lines.append("")
    lines.append("## Recommended candidate collects")
    lines.append("")
    rec = bundle.recommendation
    if rec.ranked_values:
        for cv in rec.ranked_values:
            lines.append(
                f"- `{cv.candidate.candidate_id}` ({cv.score:.2f}) - "
                f"{cv.candidate.label}"
            )
    else:
        lines.append("- None")
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    for c in bundle.caveats:
        lines.append(f"- {c}")
    lines.append("")
    return "\n".join(lines)


# Re-export used by the script.
format_packet_markdown = _packet_format_as_markdown
