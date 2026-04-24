"""Source-object → :class:`HypothesisEvidence` adapters (ADR-0021, Slice 2).

These adapters are a thin bridge only.  They take an existing repo source
object — an observation, scene, matcher result, or dict-shaped artifact —
plus caller-supplied supports / contradicts / confidence / weight / reason,
and return a validated :class:`HypothesisEvidence`.

Adapters do not decide what an artifact *means*.  Scenario-specific
"which hypothesis does this artifact support?" logic belongs to Slice 3
scenario generators, not here.  Similarly, there is no quality_flag →
confidence mapping, no modality weighting, no derivation of supports
from object content: the caller supplies the interpretation.

Validation happens at construction time (fail fast):

- unknown ``scenario_id`` raises :class:`ValueError` (via the registry);
- every id in ``supports`` / ``contradicts`` must belong to the
  scenario's registered hypothesis set — otherwise :class:`ValueError`
  names the bad id and lists the valid ids for that scenario.

The module deliberately does not import ``custody.detection.*`` or
``custody.ingest.gfw_presence``: heavy detector/ingest deps have no
business in the hypothesis layer.  VLM detections flow in as
:class:`PositionObservation` via the existing detector pipeline; GFW
records flow in through :func:`from_mapping` with
``source_kind="gfw_presence"``.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from custody.fusion.observations import (
    Observation,
    PositionObservation,
    PositionVelocityObservation,
)
from custody.fusion.scenes import Scene
from custody.fusion.temporal import Match
from custody.hypotheses.registry import get_hypotheses, get_hypothesis_ids
from custody.hypotheses.types import HypothesisEvidence


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _to_datetime(x: Any) -> datetime | None:
    """Coerce ``float`` epoch seconds | :class:`datetime` | ``None`` → datetime | None.

    Floats are interpreted as UTC epoch seconds and return a timezone-aware
    datetime.  Datetime instances pass through unchanged.  ``None`` returns
    ``None``.  Any other type raises :class:`TypeError`.
    """
    if x is None:
        return None
    if isinstance(x, datetime):
        return x
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return datetime.fromtimestamp(float(x), tz=timezone.utc)
    raise TypeError(
        f"cannot coerce {type(x).__name__} to datetime; "
        "expected float epoch seconds, datetime, or None"
    )


def _first_key(m: Mapping[str, Any], keys: Sequence[str]) -> Any | None:
    """Return the first non-None value at any key in ``keys`` that appears in ``m``."""
    for k in keys:
        if k in m:
            v = m[k]
            if v is not None:
                return v
    return None


def _stable_fallback_id(m: Mapping[str, Any]) -> str:
    """Deterministic short digest over a mapping's sorted items."""
    payload = repr(sorted((str(k), repr(v)) for k, v in m.items()))
    return hashlib.md5(payload.encode("utf-8"), usedforsecurity=False).hexdigest()[:10]


def _validate_scenario_and_ids(
    scenario_id: str,
    supports: Sequence[str],
    contradicts: Sequence[str],
) -> None:
    """Validate scenario_id and every supports / contradicts hypothesis_id.

    Raises :class:`ValueError` with the valid hypothesis ids in the error
    message when any id is not registered for ``scenario_id``.  Unknown
    ``scenario_id`` is validated via :func:`get_hypotheses`.
    """
    # Validate scenario_id first — get_hypotheses raises on unknown.
    get_hypotheses(scenario_id)
    valid_ids = set(get_hypothesis_ids(scenario_id))
    for role, ids in (("supports", supports), ("contradicts", contradicts)):
        for hid in ids:
            if hid not in valid_ids:
                valid_list = ", ".join(sorted(valid_ids))
                raise ValueError(
                    f"{role} hypothesis_id {hid!r} is not registered for "
                    f"scenario {scenario_id!r}; valid ids: {valid_list}"
                )


def _build(
    *,
    evidence_id: str,
    source_ref: str | None,
    source_kind: str,
    scenario_id: str,
    timestamp: datetime | None,
    supports: Sequence[str],
    contradicts: Sequence[str],
    confidence: float,
    weight: float,
    reason: str,
) -> HypothesisEvidence:
    _validate_scenario_and_ids(scenario_id, supports, contradicts)
    return HypothesisEvidence(
        evidence_id=evidence_id,
        source_ref=source_ref,
        source_kind=source_kind,
        scenario_id=scenario_id,
        timestamp=timestamp,
        supports=tuple(supports),
        contradicts=tuple(contradicts),
        confidence=float(confidence),
        weight=float(weight),
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


def from_observation(
    obs: Observation,
    *,
    scenario_id: str,
    supports: tuple[str, ...] | list[str],
    contradicts: tuple[str, ...] | list[str] = (),
    confidence: float | None = None,
    weight: float = 1.0,
    reason: str,
    evidence_id: str | None = None,
    source_ref: str | None = None,
    timestamp: datetime | None = None,
) -> HypothesisEvidence:
    """Build :class:`HypothesisEvidence` from a :class:`PositionObservation`
    or :class:`PositionVelocityObservation`.

    Defaults:
      ``source_ref``  → ``f"observation:{obs.obs_id}"``
      ``evidence_id`` → ``obs.obs_id``
      ``timestamp``   → ``_to_datetime(obs.acquisition_time)``
      ``confidence``  → ``obs.classification_conf`` if present, else ``1.0``
      ``source_kind`` → ``"observation"``
    """
    if not isinstance(obs, (PositionObservation, PositionVelocityObservation)):
        raise TypeError(
            f"from_observation: expected PositionObservation or "
            f"PositionVelocityObservation, got {type(obs).__name__}"
        )
    resolved_evidence_id = evidence_id if evidence_id is not None else obs.obs_id
    resolved_source_ref = (
        source_ref if source_ref is not None else f"observation:{obs.obs_id}"
    )
    resolved_timestamp = (
        timestamp if timestamp is not None else _to_datetime(obs.acquisition_time)
    )
    if confidence is not None:
        resolved_confidence = confidence
    else:
        obj_conf = getattr(obs, "classification_conf", None)
        resolved_confidence = obj_conf if obj_conf is not None else 1.0

    return _build(
        evidence_id=resolved_evidence_id,
        source_ref=resolved_source_ref,
        source_kind="observation",
        scenario_id=scenario_id,
        timestamp=resolved_timestamp,
        supports=supports,
        contradicts=contradicts,
        confidence=resolved_confidence,
        weight=weight,
        reason=reason,
    )


def from_scene(
    scene: Scene,
    *,
    scenario_id: str,
    supports: tuple[str, ...] | list[str],
    contradicts: tuple[str, ...] | list[str] = (),
    confidence: float | None = None,
    weight: float = 1.0,
    reason: str,
    evidence_id: str | None = None,
    source_ref: str | None = None,
    timestamp: datetime | None = None,
) -> HypothesisEvidence:
    """Build :class:`HypothesisEvidence` from a :class:`Scene`.

    Defaults:
      ``source_ref``  → ``f"scene:{scene.scene_id}"``
      ``evidence_id`` → ``scene.scene_id``
      ``timestamp``   → ``_to_datetime(scene.acquisition_time)``
      ``confidence``  → ``1.0``  (quality_flag → confidence is Slice 3)
      ``source_kind`` → ``"scene"``
    """
    if not isinstance(scene, Scene):
        raise TypeError(
            f"from_scene: expected Scene, got {type(scene).__name__}"
        )
    resolved_evidence_id = evidence_id if evidence_id is not None else scene.scene_id
    resolved_source_ref = (
        source_ref if source_ref is not None else f"scene:{scene.scene_id}"
    )
    resolved_timestamp = (
        timestamp if timestamp is not None else _to_datetime(scene.acquisition_time)
    )
    resolved_confidence = confidence if confidence is not None else 1.0

    return _build(
        evidence_id=resolved_evidence_id,
        source_ref=resolved_source_ref,
        source_kind="scene",
        scenario_id=scenario_id,
        timestamp=resolved_timestamp,
        supports=supports,
        contradicts=contradicts,
        confidence=resolved_confidence,
        weight=weight,
        reason=reason,
    )


def from_match(
    match: Match,
    *,
    scenario_id: str,
    supports: tuple[str, ...] | list[str],
    contradicts: tuple[str, ...] | list[str] = (),
    confidence: float | None = None,
    weight: float = 1.0,
    reason: str,
    evidence_id: str | None = None,
    source_ref: str | None = None,
    timestamp: datetime | None = None,
) -> HypothesisEvidence:
    """Build :class:`HypothesisEvidence` from a :class:`Match`.

    Defaults:
      ``source_ref``  → ``f"matcher_result:{match.obs_a_id}->{match.obs_b_id}"``
      ``evidence_id`` → ``f"match-{match.obs_a_id}-{match.obs_b_id}"``
      ``timestamp``   → ``None``  (Match carries no timestamp; supply if needed)
      ``confidence``  → ``1.0``
      ``source_kind`` → ``"matcher_result"``
    """
    if not isinstance(match, Match):
        raise TypeError(
            f"from_match: expected Match, got {type(match).__name__}"
        )
    resolved_evidence_id = (
        evidence_id if evidence_id is not None
        else f"match-{match.obs_a_id}-{match.obs_b_id}"
    )
    resolved_source_ref = (
        source_ref if source_ref is not None
        else f"matcher_result:{match.obs_a_id}->{match.obs_b_id}"
    )
    resolved_confidence = confidence if confidence is not None else 1.0

    return _build(
        evidence_id=resolved_evidence_id,
        source_ref=resolved_source_ref,
        source_kind="matcher_result",
        scenario_id=scenario_id,
        timestamp=timestamp,  # Match has no intrinsic timestamp — None unless caller supplies
        supports=supports,
        contradicts=contradicts,
        confidence=resolved_confidence,
        weight=weight,
        reason=reason,
    )


_ID_KEYS = ("id", "event_id", "obs_id", "scene_id")
_TIMESTAMP_KEYS = ("timestamp", "acquisition_time", "time", "acquired_at")
_CONFIDENCE_KEYS = ("confidence", "classification_conf", "raw_confidence", "score")


def from_mapping(
    m: Mapping[str, Any],
    *,
    source_kind: str,
    scenario_id: str,
    supports: tuple[str, ...] | list[str],
    contradicts: tuple[str, ...] | list[str] = (),
    confidence: float | None = None,
    weight: float = 1.0,
    reason: str,
    evidence_id: str | None = None,
    source_ref: str | None = None,
    timestamp: datetime | None = None,
) -> HypothesisEvidence:
    """Build :class:`HypothesisEvidence` from a dict-shaped artifact.

    Escape hatch for VLM raw outputs, GFW presence records, or scratch
    intermediates.  ``source_kind`` is a REQUIRED keyword — no default.

    Fallback lookups on ``m``:
      id         → first of ("id", "event_id", "obs_id", "scene_id")
      timestamp  → first of ("timestamp", "acquisition_time", "time",
                              "acquired_at"); coerced via :func:`_to_datetime`
      confidence → first of ("confidence", "classification_conf",
                              "raw_confidence", "score"), else ``1.0``

    When no id key is present, ``evidence_id`` falls back to
    ``f"{source_kind}-<short-digest>"`` derived deterministically from the
    mapping contents, and ``source_ref`` falls back to
    ``f"{source_kind}:<no-id>"``.
    """
    if not isinstance(m, Mapping):
        raise TypeError(
            f"from_mapping: expected Mapping, got {type(m).__name__}"
        )

    raw_id = _first_key(m, _ID_KEYS)
    resolved_evidence_id = (
        evidence_id if evidence_id is not None
        else (str(raw_id) if raw_id is not None
              else f"{source_kind}-{_stable_fallback_id(m)}")
    )
    resolved_source_ref = (
        source_ref if source_ref is not None
        else f"{source_kind}:{raw_id if raw_id is not None else '<no-id>'}"
    )
    if timestamp is not None:
        resolved_timestamp = timestamp
    else:
        resolved_timestamp = _to_datetime(_first_key(m, _TIMESTAMP_KEYS))
    if confidence is not None:
        resolved_confidence = confidence
    else:
        raw_conf = _first_key(m, _CONFIDENCE_KEYS)
        resolved_confidence = float(raw_conf) if raw_conf is not None else 1.0

    return _build(
        evidence_id=resolved_evidence_id,
        source_ref=resolved_source_ref,
        source_kind=source_kind,
        scenario_id=scenario_id,
        timestamp=resolved_timestamp,
        supports=supports,
        contradicts=contradicts,
        confidence=resolved_confidence,
        weight=weight,
        reason=reason,
    )
