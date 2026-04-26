"""Map overlay manifest loader for the Whitsun replay and Tennent
monitoring tabs.

A single committed fixture under
``data/demo/map_overlays.fixture.json`` carries one ``OverlayArtifact``
per AOI polygon and per observation referenced by the Whitsun decision
trace and the Tennent monitoring fixture.  No georeferenced imagery is
committed today, so every record is currently ``image_kind:
"footprint-only"``; the renderer surfaces an explicit
``missing_asset_reason`` rather than pretending a footprint is
imagery.

This module is read-only.  It does not contact any external service
and does not modify decision-layer runtime behaviour.
"""
from __future__ import annotations

import json as _json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
WHITSUN_DECISION_TRACE_DEFAULT_OVERLAY_PATH = (
    REPO_ROOT / "data" / "demo" / "map_overlays.fixture.json"
)
MAP_OVERLAYS_PATH = WHITSUN_DECISION_TRACE_DEFAULT_OVERLAY_PATH


_VALID_SOURCES = (
    "umbra", "sentinel-1", "sentinel-2", "simulated",
)
# ``png`` and ``png_preview`` are equivalent; ``footprint-only`` and
# ``footprint_only`` are equivalent.  The newer underscored names are
# what the dynamic overlay manager prefers; the older names remain
# valid so existing fixture entries do not need a flag-day rename.
_IMAGE_KIND_RASTERS = ("png", "png_preview", "sentinel_preview",
                       "geotiff", "cog", "thumbnail")
_IMAGE_KIND_FOOTPRINTS = ("footprint-only", "footprint_only")
_VALID_IMAGE_KINDS = _IMAGE_KIND_RASTERS + _IMAGE_KIND_FOOTPRINTS
_VALID_DATA_MODES = ("real", "fixture", "simulated")


@dataclass(frozen=True)
class OverlayArtifact:
    overlay_id: str
    scenario_id: str
    observation_id: str | None
    source: str
    sensor_type: str | None
    display_name: str
    data_mode: str
    image_path: str | None
    asset_url: str | None
    image_kind: str
    bounds: tuple[float, float, float, float] | None  # W, S, E, N
    geometry: Mapping[str, Any] | None
    opacity_default: float
    visible_from_event_ordinal: int
    z_index: int
    confidence_weight: float | None
    usable_for_detection: bool | None
    usable_for_context: bool | None
    caveats: tuple[str, ...]
    missing_asset_reason: str | None
    # Fields added for the dynamic overlay manager (ADR-0021 demo
    # surface).  All default to None / sensible falsy values so older
    # fixture entries remain loadable without a migration.
    collection_time: str | None = None
    default_visible: bool = False
    resolution_m: float | None = None
    cloud_coverage: float | None = None
    weak_signal: bool = False
    confirmation_layer: bool = False


def _coerce_bounds(raw) -> tuple[float, float, float, float] | None:
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) < 4:
        raise ValueError(
            f"bounds must be a 4-element [W, S, E, N] sequence; got {raw!r}"
        )
    return (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))


def _overlay_from_mapping(rec: Mapping[str, Any]) -> OverlayArtifact:
    overlay_id = rec.get("overlay_id")
    if not isinstance(overlay_id, str) or not overlay_id:
        raise ValueError("overlay_id is required and must be non-empty")
    scenario_id = rec.get("scenario_id")
    if scenario_id not in ("whitsun", "tennent"):
        raise ValueError(
            f"scenario_id must be 'whitsun' or 'tennent'; got {scenario_id!r}"
        )
    source = rec.get("source")
    if source not in _VALID_SOURCES:
        raise ValueError(
            f"source must be one of {_VALID_SOURCES}; got {source!r}"
        )
    image_kind = rec.get("image_kind")
    if image_kind not in _VALID_IMAGE_KINDS:
        raise ValueError(
            f"image_kind must be one of {_VALID_IMAGE_KINDS}; "
            f"got {image_kind!r}"
        )
    data_mode = rec.get("data_mode") or "fixture"
    if data_mode not in _VALID_DATA_MODES:
        raise ValueError(
            f"data_mode must be one of {_VALID_DATA_MODES}; got {data_mode!r}"
        )
    return OverlayArtifact(
        overlay_id=overlay_id,
        scenario_id=str(scenario_id),
        observation_id=rec.get("observation_id"),
        source=str(source),
        sensor_type=rec.get("sensor_type"),
        display_name=str(rec.get("display_name") or overlay_id),
        data_mode=str(data_mode),
        image_path=rec.get("image_path"),
        asset_url=rec.get("asset_url"),
        image_kind=str(image_kind),
        bounds=_coerce_bounds(rec.get("bounds")),
        geometry=rec.get("geometry"),
        opacity_default=float(rec.get("opacity_default", 0.5)),
        visible_from_event_ordinal=int(
            rec.get("visible_from_event_ordinal", 0),
        ),
        z_index=int(rec.get("z_index", 0)),
        confidence_weight=(
            float(rec["confidence_weight"])
            if rec.get("confidence_weight") is not None else None
        ),
        usable_for_detection=rec.get("usable_for_detection"),
        usable_for_context=rec.get("usable_for_context"),
        caveats=tuple(rec.get("caveats") or ()),
        missing_asset_reason=rec.get("missing_asset_reason"),
        collection_time=rec.get("collection_time"),
        default_visible=bool(rec.get("default_visible", False)),
        resolution_m=(
            float(rec["resolution_m"])
            if rec.get("resolution_m") is not None else None
        ),
        cloud_coverage=(
            float(rec["cloud_coverage"])
            if rec.get("cloud_coverage") is not None else None
        ),
        weak_signal=bool(rec.get("weak_signal", False)),
        confirmation_layer=bool(rec.get("confirmation_layer", False)),
    )


def load_map_overlays(
    path: str | Path | None = None,
) -> tuple[OverlayArtifact, ...]:
    """Load the committed map-overlay manifest into ``OverlayArtifact``s."""
    p = Path(path) if path is not None else MAP_OVERLAYS_PATH
    payload = _json.loads(p.read_text(encoding="utf-8"))
    overlays = payload.get("overlays") if isinstance(payload, Mapping) else None
    if overlays is None:
        raise ValueError(
            f"map overlay manifest at {p} missing 'overlays' list"
        )
    return tuple(_overlay_from_mapping(o) for o in overlays)


def overlays_for_scenario(
    overlays: Iterable[OverlayArtifact],
    scenario_id: str,
) -> tuple[OverlayArtifact, ...]:
    return tuple(o for o in overlays if o.scenario_id == scenario_id)


def available_overlays_for(
    overlays: Iterable[OverlayArtifact],
    *,
    scenario_id: str,
    current_ordinal: int,
) -> tuple[OverlayArtifact, ...]:
    """Return the subset visible at or before ``current_ordinal``."""
    return tuple(
        o for o in overlays
        if o.scenario_id == scenario_id
        and o.visible_from_event_ordinal <= current_ordinal
    )


def overlay_by_id(
    overlays: Iterable[OverlayArtifact],
    overlay_id: str,
) -> OverlayArtifact | None:
    for o in overlays:
        if o.overlay_id == overlay_id:
            return o
    return None


def has_image_asset(overlay: OverlayArtifact) -> bool:
    return (
        overlay.image_kind in _IMAGE_KIND_RASTERS
        and (overlay.image_path is not None or overlay.asset_url is not None)
    )


def is_observation_overlay(overlay: OverlayArtifact) -> bool:
    """True if the overlay is bound to an observation (i.e., not an AOI)."""
    return overlay.observation_id is not None


_SOURCE_LABELS = {
    "umbra": "Umbra SAR",
    "sentinel-1": "Sentinel-1",
    "sentinel-2": "Sentinel-2",
    "simulated": "Simulated",
}


def _date_only(timestamp: str | None) -> str | None:
    """Trim an ISO timestamp to its YYYY-MM-DD prefix when possible."""
    if not timestamp:
        return None
    s = str(timestamp)
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:10]
    return s


def overlay_kind_label(overlay: OverlayArtifact) -> str:
    """Audience-facing kind tag used in checklist labels and badges.

    - Sentinel + raster present  → ``weak-signal image``
    - Sentinel + footprint only  → ``weak-signal footprint``
    - Non-Sentinel + raster      → ``image overlay``
    - Non-Sentinel + footprint   → ``footprint``

    Sentinel imagery is always tagged as a weak signal, never as a
    confirmation/evidence layer; the ``image overlay`` term is reserved
    for the high-confidence Umbra previews.
    """
    is_sentinel = overlay.source in ("sentinel-1", "sentinel-2")
    if has_image_asset(overlay):
        return "weak-signal image" if is_sentinel else "image overlay"
    return "weak-signal footprint" if is_sentinel else "footprint"


def format_overlay_label(overlay: OverlayArtifact) -> str:
    """Build the per-overlay label used in the dynamic toggle checklist.

    Matches the dispatch examples:
      ``Umbra SAR 2023-12-06 · 0.5 m · image overlay · confidence 1.00``
      ``Sentinel-2 2023-12-12 · 8% cloud · weak-signal image/footprint · confidence 0.30``

    AOI overlays (no observation_id) fall back to ``display_name``.
    """
    if not is_observation_overlay(overlay):
        return overlay.display_name
    parts: list[str] = []
    parts.append(_SOURCE_LABELS.get(overlay.source, overlay.source))
    date = _date_only(overlay.collection_time)
    if date:
        parts.append(date)
    if overlay.source == "sentinel-2" and overlay.cloud_coverage is not None:
        parts.append(f"{int(round(overlay.cloud_coverage))}% cloud")
    elif overlay.resolution_m is not None:
        parts.append(_format_resolution(overlay.resolution_m))
    parts.append(overlay_kind_label(overlay))
    if overlay.confidence_weight is not None:
        parts.append(f"confidence {overlay.confidence_weight:.2f}")
    return " · ".join(parts)


def _format_resolution(meters: float) -> str:
    """Render a SAR ground-resolution value (e.g. 0.25 → ``0.25 m``)."""
    if abs(meters - round(meters)) < 1e-6:
        return f"{int(round(meters))} m"
    return f"{meters:g} m"
