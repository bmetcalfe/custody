"""Scene — first-class entity per ADR-0018.

A :class:`Scene` is an immutable snapshot of one SAR acquisition with the
per-acquisition metadata (sensor, pixel size, footprint, quality flag) and
the list of :class:`PositionObservation` instances produced from it.
Cross-scene analysis operates on :class:`Scene` instances explicitly;
within-scene work continues to use the observation list.

Loader fallback chain
---------------------

:func:`load_scene_from_parquet` resolves each Scene field via a priority
chain:

1. **Loader kwargs** — explicit caller assertion wins.
2. **``notes_json`` on observations** — future canonical path.  Empty on
   all parquets committed before the sub-decision in ADR-0018 lands in the
   detection pipeline; becomes the primary path once the pipeline writes
   scene metadata per observation.
3. **Sidecar JSON** at ``day0/scratch/<label>_vlm_summary.json`` — the
   existing per-run summary JSONs that accompany every committed VLM
   detection parquet.  Provides sensor (via ``scene`` filename parse),
   pixel size, and target lat/lon for all six committed scenes.
4. **Parse from ``raw_scene_path`` filename** — sensor only, pattern
   ``_UMBRA-NN_`` (extensible to other sensor prefixes).
5. **Observation-derived approximation** — for ``center_lat/lon`` and
   ``footprint_latlon`` only; falls back to mean and axis-aligned envelope
   of observation positions.  Emits a DEBUG log so future debugging can
   find where the approximate value came from.
6. **:class:`SceneLoaderError`** — raised when none of the above resolve a
   required field, with an informative message listing what was checked.

The workflow expectation is that new scenes produced after the detection
pipeline update populate ``notes_json`` directly (so step 2 supplants step
3), while the existing six committed parquets continue to load via the
sidecar fallback.  Kwargs remain available as an override path.

See ADR-0018 for the architectural rationale.
"""
from __future__ import annotations

import json
import logging
import re
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from custody.fusion.observations import PositionObservation


_log = logging.getLogger(__name__)


# Acquisition-time sanity bounds — reject times before 1970 or after 2100.
_MIN_ACQ_TIME = 0.0
_MAX_ACQ_TIME = 4102444800.0  # 2100-01-01T00:00:00Z

# scene_id format: "{case_study}_{yyyymmdd}_{sensor_lower}"
_SCENE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9]*_\d{8}_[a-z0-9\-]+$")

# Sensor prefixes we recognize in filenames.  Extend for Sentinel-1 etc.
_SENSOR_IN_FILENAME = re.compile(
    r"_(UMBRA-\d+|SENTINEL-\d+)", re.IGNORECASE,
)


class SceneLoaderError(Exception):
    """Raised when a Scene cannot be loaded from a parquet due to missing metadata."""


@dataclass(frozen=True, eq=False)
class Scene:
    """Immutable snapshot of one SAR scene + its derived observations."""

    scene_id: str
    sensor: str
    acquisition_time: float           # UTC epoch seconds; matches observations
    pixel_size_m: float
    center_lat: float
    center_lon: float
    footprint_latlon: tuple[tuple[float, float], ...]  # >= 3 corners
    raw_scene_path: str | None
    quality_flag: Literal["green", "yellow", "red"]
    observations: tuple[PositionObservation, ...] = ()
    notes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.scene_id, str) or not self.scene_id:
            raise ValueError("scene_id must be a non-empty string")
        if not _SCENE_ID_PATTERN.match(self.scene_id):
            raise ValueError(
                f"scene_id {self.scene_id!r} must match "
                f"'{{case_study}}_{{yyyymmdd}}_{{sensor_lower}}' pattern"
            )
        if not (_MIN_ACQ_TIME < self.acquisition_time < _MAX_ACQ_TIME):
            raise ValueError(
                f"acquisition_time {self.acquisition_time} outside valid range "
                f"({_MIN_ACQ_TIME}, {_MAX_ACQ_TIME})"
            )
        if not (0 < self.pixel_size_m <= 10):
            raise ValueError(
                f"pixel_size_m {self.pixel_size_m} outside (0, 10] m — "
                "scene pixel sizes outside this range are almost certainly a bug"
            )
        if not (-90 <= self.center_lat <= 90):
            raise ValueError(f"center_lat {self.center_lat} outside [-90, 90]")
        if not (-180 <= self.center_lon <= 180):
            raise ValueError(f"center_lon {self.center_lon} outside [-180, 180]")
        if not isinstance(self.footprint_latlon, tuple):
            raise ValueError(
                f"footprint_latlon must be a tuple (got {type(self.footprint_latlon).__name__})"
            )
        if len(self.footprint_latlon) < 3:
            raise ValueError(
                f"footprint_latlon must have >= 3 corners; got {len(self.footprint_latlon)}"
            )
        if self.quality_flag not in ("green", "yellow", "red"):
            raise ValueError(
                f"quality_flag {self.quality_flag!r} must be 'green' | 'yellow' | 'red'"
            )
        if not isinstance(self.observations, tuple):
            raise ValueError(
                f"observations must be a tuple (got {type(self.observations).__name__}) — "
                "Scene is frozen; pass tuple(...) not list(...)"
            )
        for i, obs in enumerate(self.observations):
            if not isinstance(obs, PositionObservation):
                raise ValueError(
                    f"observations[{i}] is {type(obs).__name__}, "
                    "expected PositionObservation"
                )
            if abs(obs.acquisition_time - self.acquisition_time) > 1.0:
                raise ValueError(
                    f"observations[{i}] acquisition_time {obs.acquisition_time} "
                    f"differs from scene acquisition_time {self.acquisition_time} "
                    "by more than 1 second"
                )

    def __hash__(self) -> int:
        return hash(self.scene_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Scene):
            return NotImplemented
        return self.scene_id == other.scene_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_sensor_from_filename(filename: str) -> str | None:
    """Extract sensor identifier from a SAR scene filename as lowercase.

    Matches ``_UMBRA-NN_`` and ``_SENTINEL-NN_`` patterns in the filename.
    Returns e.g. ``'umbra-05'`` or ``None`` if no pattern matches.
    """
    m = _SENSOR_IN_FILENAME.search(filename)
    return m.group(1).lower() if m else None


def _compute_scene_id(case_study: str, acquisition_time: float, sensor: str) -> str:
    """Compose scene_id as ``{case_study}_{yyyymmdd}_{sensor_lower}``."""
    dt = datetime.fromtimestamp(acquisition_time, tz=timezone.utc)
    return f"{case_study.lower()}_{dt.strftime('%Y%m%d')}_{sensor.lower()}"


def _resolve_sidecar_path(parquet_path: Path) -> Path | None:
    """Resolve conventional sidecar JSON path from a VLM-detections parquet.

    Convention:

    - ``data/processed/vlm_detections/{label}_position.parquet``
      → ``day0/scratch/{label}_vlm_summary.json``
    - Legacy: ``data/processed/vlm_detections/position.parquet`` (original
      Tennent 07-02 file, no date prefix in the filename) →
      ``day0/scratch/tennent_20230702_vlm_summary.json``

    Returns ``None`` if the expected sidecar does not exist on disk.
    """
    parquet_path = Path(parquet_path).resolve()
    stem = parquet_path.stem
    if stem == "position":
        label = "tennent_20230702"
    elif stem.endswith("_position"):
        label = stem[: -len("_position")]
    else:
        label = stem

    for ancestor in parquet_path.parents:
        candidate_dir = ancestor / "day0" / "scratch"
        if candidate_dir.exists():
            candidate = candidate_dir / f"{label}_vlm_summary.json"
            return candidate if candidate.exists() else None
    return None


def _load_observations(parquet_path: Path | str) -> list[PositionObservation]:
    """Load PositionObservations from a parquet via duckdb + the fusion row decoder."""
    import duckdb  # deferred — duckdb import is ~100ms

    from custody.fusion.index import _position_from_row

    p = str(Path(parquet_path).resolve().as_posix())
    cols = [c[0] for c in duckdb.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{p}')"
    ).fetchall()]
    rows = duckdb.execute(f"SELECT * FROM read_parquet('{p}')").fetchall()
    return [_position_from_row(dict(zip(cols, r))) for r in rows]


def _sidecar_center_latlon(sidecar: dict) -> tuple[float, float] | None:
    """Sidecar layouts differ between Tennent and Whitsun templates — try known keys."""
    for key in ("target_latlon", "scene_center_latlon",
                "aoi_center_latlon_from_helper"):
        val = sidecar.get(key)
        if val and len(val) >= 2:
            return float(val[0]), float(val[1])
    return None


def _raise_loader_error(
    field_name: str,
    parquet_path: Path,
    sidecar_path: Path | None,
    raw_scene_path_provided: bool,
    note: str = "",
) -> None:
    """Raise SceneLoaderError with a specific message about what's missing."""
    lines = [
        f"Could not resolve Scene.{field_name} for {parquet_path}.",
        "  Checked:",
        f"    kwarg:            not provided",
        f"    notes_json:       empty or missing field",
        f"    sidecar:          {sidecar_path if sidecar_path else 'not found at conventional path'}",
    ]
    if field_name == "sensor":
        lines.append(
            f"    raw_scene_path:   "
            f"{'provided but no sensor pattern matched' if raw_scene_path_provided else 'not provided'}"
        )
    if note:
        lines.append(f"  Note: {note}")
    lines.append(f"  Supply {field_name} via loader kwarg or update the source.")
    raise SceneLoaderError("\n".join(lines))


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_scene_from_parquet(
    parquet_path: Path | str,
    *,
    case_study: str,
    sensor: str | None = None,
    pixel_size_m: float | None = None,
    center_lat: float | None = None,
    center_lon: float | None = None,
    footprint_latlon: tuple[tuple[float, float], ...] | None = None,
    raw_scene_path: Path | str | None = None,
    quality_flag: Literal["green", "yellow", "red"] | None = None,
    sidecar_path: Path | str | None = None,
) -> Scene:
    """Load a :class:`Scene` from a committed observations parquet.

    Applies the fallback chain documented at module top.  ``case_study`` is
    required because it cannot reliably be inferred from lat/lon alone.
    """
    parquet_path_obj = Path(parquet_path)
    obs_list = _load_observations(parquet_path_obj)
    if not obs_list:
        raise SceneLoaderError(
            f"No observations in parquet {parquet_path_obj}; cannot build a Scene"
        )

    acq = obs_list[0].acquisition_time
    for i, o in enumerate(obs_list):
        if abs(o.acquisition_time - acq) > 1.0:
            raise SceneLoaderError(
                f"Observations in {parquet_path_obj} do not share a single "
                f"acquisition_time (obs[0]={acq}, obs[{i}]={o.acquisition_time}). "
                "A Scene represents one acquisition; multi-acquisition parquets "
                "need to be split before loading."
            )

    sidecar_path_resolved = (
        Path(sidecar_path) if sidecar_path is not None
        else _resolve_sidecar_path(parquet_path_obj)
    )
    sidecar: dict[str, Any] = {}
    if sidecar_path_resolved is not None and sidecar_path_resolved.exists():
        sidecar = json.loads(sidecar_path_resolved.read_text(encoding="utf-8"))

    obs_notes = obs_list[0].notes or {}

    # --- sensor ---
    resolved_sensor = sensor
    if resolved_sensor is None:
        resolved_sensor = obs_notes.get("sensor")
    if resolved_sensor is None:
        resolved_sensor = sidecar.get("sensor")
    if resolved_sensor is None and "scene" in sidecar:
        resolved_sensor = _parse_sensor_from_filename(sidecar["scene"])
    if resolved_sensor is None and raw_scene_path is not None:
        resolved_sensor = _parse_sensor_from_filename(Path(raw_scene_path).name)
    if resolved_sensor is None:
        _raise_loader_error(
            "sensor", parquet_path_obj, sidecar_path_resolved,
            raw_scene_path is not None,
        )
    resolved_sensor = resolved_sensor.lower()

    # --- pixel_size_m ---
    resolved_pixel: float | None = pixel_size_m
    if resolved_pixel is None:
        resolved_pixel = obs_notes.get("pixel_size_m")
    if resolved_pixel is None and "pixel_size_m" in sidecar:
        resolved_pixel = float(sidecar["pixel_size_m"])
    if resolved_pixel is None:
        # Whitsun run summaries (scripts/06 / 11) never wrote pixel_size_m
        # directly, but they did write approx_extent_km + full_scene_shape_px.
        # Derive a serviceable approximation: the axis-aligned envelope extent
        # divided by the scene pixel count gives roughly the pixel size (within
        # the √2 grid-convergence caveat for rotated GEC rasters — acceptable
        # for quality / cost-gate purposes, not for geodetic calculations).
        extent = sidecar.get("approx_extent_km")
        shape = sidecar.get("full_scene_shape_px") or sidecar.get("aoi_shape_px")
        if extent and shape and len(extent) >= 2 and len(shape) >= 2:
            try:
                resolved_pixel = (float(extent[0]) * 1000.0) / float(shape[0])
                _log.debug(
                    "pixel_size_m derived from sidecar extent/shape for %s: %.3f m/px",
                    parquet_path_obj, resolved_pixel,
                )
            except (TypeError, ValueError, ZeroDivisionError):
                resolved_pixel = None
    if resolved_pixel is None:
        _raise_loader_error(
            "pixel_size_m", parquet_path_obj, sidecar_path_resolved,
            raw_scene_path is not None,
            note="raw_scene_path could in principle be probed via rasterio, "
                 "but that is not implemented here to keep the loader dependency-light.",
        )

    # --- center_lat / center_lon ---
    resolved_lat = center_lat
    resolved_lon = center_lon
    if resolved_lat is None and "center_lat" in obs_notes:
        resolved_lat = float(obs_notes["center_lat"])
    if resolved_lon is None and "center_lon" in obs_notes:
        resolved_lon = float(obs_notes["center_lon"])
    if resolved_lat is None or resolved_lon is None:
        sidecar_ll = _sidecar_center_latlon(sidecar)
        if sidecar_ll is not None:
            if resolved_lat is None:
                resolved_lat = sidecar_ll[0]
            if resolved_lon is None:
                resolved_lon = sidecar_ll[1]
    if resolved_lat is None or resolved_lon is None:
        resolved_lat = (
            resolved_lat if resolved_lat is not None
            else statistics.fmean(o.lat for o in obs_list)
        )
        resolved_lon = (
            resolved_lon if resolved_lon is not None
            else statistics.fmean(o.lon for o in obs_list)
        )
        _log.debug(
            "center_lat/lon resolved from observation mean for %s (%d obs)",
            parquet_path_obj, len(obs_list),
        )

    # --- footprint_latlon ---
    resolved_footprint: tuple[tuple[float, float], ...] | None = None
    if footprint_latlon is not None:
        resolved_footprint = tuple(tuple(p) for p in footprint_latlon)  # type: ignore[assignment]
    elif "footprint_latlon" in obs_notes:
        resolved_footprint = tuple(tuple(p) for p in obs_notes["footprint_latlon"])  # type: ignore[assignment]
    elif "footprint_latlon" in sidecar:
        resolved_footprint = tuple(tuple(p) for p in sidecar["footprint_latlon"])  # type: ignore[assignment]
    else:
        lats = [o.lat for o in obs_list]
        lons = [o.lon for o in obs_list]
        la_lo, la_hi = min(lats), max(lats)
        lo_lo, lo_hi = min(lons), max(lons)
        resolved_footprint = (
            (la_lo, lo_lo), (la_lo, lo_hi),
            (la_hi, lo_hi), (la_hi, lo_lo),
        )
        _log.debug(
            "footprint_latlon approximated from observation envelope for %s",
            parquet_path_obj,
        )

    # --- raw_scene_path ---
    resolved_raw_path: str | None = (
        str(raw_scene_path) if raw_scene_path is not None
        else obs_notes.get("raw_scene_path")
    )

    # --- quality_flag ---
    resolved_quality: Literal["green", "yellow", "red"] = (
        quality_flag
        or obs_notes.get("quality_flag")
        or sidecar.get("quality_flag")
        or "green"
    )  # type: ignore[assignment]

    # --- notes ---
    out_notes: dict[str, Any] = dict(obs_notes)
    if sidecar:
        out_notes["_sidecar"] = sidecar

    scene_id = _compute_scene_id(case_study, acq, resolved_sensor)
    sorted_obs = tuple(sorted(obs_list, key=lambda o: o.obs_id))

    return Scene(
        scene_id=scene_id,
        sensor=resolved_sensor,
        acquisition_time=acq,
        pixel_size_m=float(resolved_pixel),
        center_lat=float(resolved_lat),
        center_lon=float(resolved_lon),
        footprint_latlon=resolved_footprint,
        raw_scene_path=resolved_raw_path,
        quality_flag=resolved_quality,
        observations=sorted_obs,
        notes=out_notes,
    )
