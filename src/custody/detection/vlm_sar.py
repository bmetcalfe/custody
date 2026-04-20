"""Backend-agnostic VLM-based SAR detection API (ADR-0015).

This module is the integration layer between :mod:`custody.detection.vlm_backends`
(Anthropic/OpenAI/future local models) and the fusion layer's
:class:`PositionObservation`.  Callers pass in a tile and a concrete
:class:`VLMBackend`; the module handles prompt selection, dispatch, and the
conversion of raw :class:`VLMDetection` bounding boxes into fully-populated
:class:`PositionObservation` instances with geographic coordinates.

Scene-level tiling (walking a multi-km Umbra GEC scene, dispatching each
tile, fusing results, NMS across tile boundaries) is deferred to Phase F.
The :func:`detect_vessels_in_scene` entry point is stubbed so callers can
pin the API signature but raises ``NotImplementedError`` until implemented.

Prompt variants
---------------
Three prompt variants are exposed, all from the Week 2 spike on Whitsun
tile 1 (see ``day0/scratch/vlm_detect.py`` and the cross-model spike in
Phase D.5):

- ``direct_v1`` — asks for vessel bboxes with no SAR-specific context.
- ``contextualized_v1`` — supplies sensor/band/resolution/scene context;
  the variant that produced SAR-aware reasoning in the cross-model spike
  and is the Phase E default.
- ``reasoning_first_v1`` — asks the model to describe the scene in words
  before listing detections; generates the scene-description preamble used
  by the qualitative review narrative.
"""
from __future__ import annotations

import logging
import time as _time
from typing import Callable, Iterator, Literal, Optional

import numpy as np
from rasterio.transform import Affine

from custody.detection.sar_common import pixel_to_latlon
from custody.detection.tiling import TileInfo, count_tiles, iter_tiles
from custody.detection.vlm_backends.base import VLMBackend, VLMDetection, VLMResponse
from custody.fusion.observations import PositionObservation


_log = logging.getLogger(__name__)


def _compute_rate_limit_exceptions() -> tuple[type, ...]:
    """Build the tuple of rate-limit exception classes to retry on.

    Imports are lazy-best-effort: both ``anthropic`` and ``openai`` are listed
    as project deps today, but we don't want the module to fail to import if a
    future consumer strips one out.  The computed tuple is captured once at
    module import.  Tests may monkey-patch :data:`_RATE_LIMIT_EXCEPTIONS` to
    inject a custom class for rate-limit simulation.
    """
    classes: list[type] = []
    try:
        import anthropic  # type: ignore[import-not-found]
        classes.append(anthropic.RateLimitError)
    except (ImportError, AttributeError):
        pass
    try:
        import openai  # type: ignore[import-not-found]
        classes.append(openai.RateLimitError)
    except (ImportError, AttributeError):
        pass
    return tuple(classes)


class _NeverRaised(Exception):
    """Sentinel class used when no SDK rate-limit classes could be imported.

    ``except _NeverRaised`` in the retry loop becomes a no-op, delegating all
    errors to the generic-error branch.  Prevents ``except ()`` (a SyntaxError
    at the AST level, though a TypeError at except-match time) edge cases.
    """


_RATE_LIMIT_EXCEPTIONS: tuple[type, ...] = _compute_rate_limit_exceptions() or (_NeverRaised,)


PromptVariant = Literal["direct_v1", "contextualized_v1", "reasoning_first_v1"]


PROMPTS: dict[str, str] = {
    "direct_v1": (
        "This image is from a synthetic aperture radar (SAR) satellite. Identify all "
        "ships or vessels visible in the image. For each vessel, provide approximate "
        "bounding box coordinates as (x1, y1, x2, y2) in pixels, where (0,0) is the "
        "top-left corner and the image is 640x640 pixels. Format your answer as a JSON "
        "list of objects with keys 'vessel_id', 'bbox', and 'confidence' (high/medium/low)."
    ),
    "contextualized_v1": (
        "This is a synthetic aperture radar (SAR) image from Umbra-04 satellite, X-band "
        "(~9.6 GHz) at approximately 0.25 m/pixel resolution. The scene covers Whitsun "
        "Reef in the South China Sea. Ships typically appear as compact bright returns on "
        "the darker water background, sometimes with wake signatures trailing behind them "
        "(horizontal streaks due to azimuth smearing of moving targets). The image is "
        "640x640 pixels. Identify all vessels visible in this image. For each, provide: "
        "(1) pixel coordinates of the bounding box as (x1, y1, x2, y2), (2) confidence "
        "level (high/medium/low), (3) brief reasoning for why you identified it as a "
        "vessel. Format as JSON."
    ),
    "reasoning_first_v1": (
        "This is a 640x640 pixel SAR satellite image. Take your time to carefully examine "
        "the image. First, describe what you see in the image in general terms. Then, "
        "identify any regions that contain vessels or ships. For each vessel, describe "
        "its location in words (e.g., 'upper left', 'center-right'), then provide "
        "approximate pixel bounding box coordinates (x1, y1, x2, y2). Format your final "
        "answer as JSON with descriptive text first, then a 'detections' list."
    ),
}


_CONF_TO_SCORE: dict[str, float] = {
    "high":   0.85,
    "medium": 0.60,
    "low":    0.35,
}


def detect_vessels_in_tile(
    tile: np.ndarray,
    backend: VLMBackend,
    prompt_variant: PromptVariant = "contextualized_v1",
) -> list[VLMDetection]:
    """Dispatch a single tile to ``backend`` with the selected prompt variant.

    Returns the raw :class:`VLMDetection` list from the backend.  Conversion to
    :class:`PositionObservation` is a separate step (:func:`vlm_detection_to_observation`)
    because callers may want to filter, re-rank, or de-duplicate detections
    before projecting them to geographic coordinates.
    """
    if prompt_variant not in PROMPTS:
        raise ValueError(
            f"unknown prompt_variant {prompt_variant!r}; "
            f"valid options: {sorted(PROMPTS.keys())}"
        )
    prompt = PROMPTS[prompt_variant]
    response = backend.detect_tile(tile, prompt=prompt)
    return list(response.detections)


def vlm_detection_to_observation(
    detection: VLMDetection,
    transform: Affine,
    crs_wkt: str,
    tile_origin: tuple[int, int],
    *,
    acquisition_time: float,
    source_id: str,
    backend: VLMBackend,
    sigma_m: float = 20.0,
) -> PositionObservation:
    """Convert one :class:`VLMDetection` into a fully-populated :class:`PositionObservation`.

    ``transform`` and ``crs_wkt`` are the *scene-level* rasterio transform and
    CRS; ``tile_origin`` is the ``(row, col)`` of the tile's top-left in the
    scene raster.  The bbox center is translated to scene pixel coords, then
    projected to WGS84 lat/lon via :func:`custody.detection.sar_common.pixel_to_latlon`.

    The ``classification_conf`` field is populated by mapping the VLM's
    coarse ``high``/``medium``/``low`` label to ``0.85``/``0.60``/``0.35``.
    ``detector_reasoning`` carries the VLM's per-detection rationale string
    through to the fusion layer for downstream audit and ADR-0016 narrative.
    """
    x1, y1, x2, y2 = detection.bbox
    # bbox pixels are (x=col, y=row) within the tile; translate to scene.
    tile_row_origin, tile_col_origin = tile_origin
    center_col = (x1 + x2) / 2.0 + tile_col_origin
    center_row = (y1 + y2) / 2.0 + tile_row_origin

    lat, lon = pixel_to_latlon(center_row, center_col, transform, crs_wkt)
    cov_pos = np.array([[sigma_m ** 2, 0.0], [0.0, sigma_m ** 2]], dtype=float)

    conf = detection.confidence.lower() if isinstance(detection.confidence, str) else "medium"
    classification_conf = _CONF_TO_SCORE.get(conf, _CONF_TO_SCORE["medium"])

    return PositionObservation(
        obs_id=(
            f"{source_id}-{int(acquisition_time)}-"
            f"{int(round(center_row))}-{int(round(center_col))}"
        ),
        source_id=source_id,
        modality="SAR",
        acquisition_time=acquisition_time,
        ingestion_time=_time.time(),
        lat=float(lat),
        lon=float(lon),
        cov_pos=cov_pos,
        raw_ref=source_id,
        detector_version=f"vlm_{backend.model_name}",
        classification_conf=classification_conf,
        detector_reasoning=detection.reasoning or None,
    )


def _detect_tiles(
    scene: np.ndarray,
    backend: VLMBackend,
    prompt: str,
    *,
    tile_size: int = 640,
    tile_overlap: int = 64,
    aoi_bounds: Optional[tuple[int, int, int, int]] = None,
    skip_empty: bool = True,
    empty_threshold: float = 0.05,
    max_retries: int = 3,
    retry_backoff_base: float = 2.0,
    progress_callback: Optional[Callable[[TileInfo, Optional[VLMResponse], int, int], None]] = None,
) -> Iterator[tuple[TileInfo, VLMResponse]]:
    """Iterate tiles through ``backend.detect_tile`` with rate-limit retry.

    Yields ``(tile_info, response)`` for each tile whose detection call
    succeeded.  Tiles that fail after ``max_retries`` rate-limit retries, or
    that raise any non-rate-limit exception, are logged and skipped — one bad
    tile does not abort the scene.

    ``progress_callback`` is invoked once per attempted tile (after the retry
    loop resolves), receiving ``(tile_info, response_or_none, tiles_done,
    tiles_total_estimate)``.  The total is a pre-flight geometric count; actual
    processed count may be lower when ``skip_empty`` drops tiles.
    """
    total_estimate = count_tiles(scene.shape[:2], tile_size, tile_overlap, aoi_bounds)
    tiles_done = 0
    for tile_info in iter_tiles(
        scene,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        aoi_bounds=aoi_bounds,
        skip_empty=skip_empty,
        empty_threshold=empty_threshold,
    ):
        response = _detect_one_with_retry(
            tile_info, backend, prompt, max_retries, retry_backoff_base,
        )
        tiles_done += 1
        if progress_callback is not None:
            progress_callback(tile_info, response, tiles_done, total_estimate)
        if response is not None:
            yield (tile_info, response)


def _detect_one_with_retry(
    tile_info: TileInfo,
    backend: VLMBackend,
    prompt: str,
    max_retries: int,
    retry_backoff_base: float,
) -> Optional[VLMResponse]:
    """Call ``backend.detect_tile`` with exponential-backoff rate-limit retry.

    Returns the :class:`VLMResponse` on success, or ``None`` if the call failed
    after all retries or raised a non-rate-limit exception.
    """
    for attempt in range(max_retries + 1):
        try:
            return backend.detect_tile(tile_info.tile, prompt=prompt)
        except _RATE_LIMIT_EXCEPTIONS as exc:
            if attempt >= max_retries:
                _log.warning(
                    "tile %d at %s: rate-limit retries exhausted (%d attempts); skipping. err=%s",
                    tile_info.index, tile_info.origin, attempt + 1, exc,
                )
                return None
            delay = retry_backoff_base ** attempt
            _log.info(
                "tile %d at %s: rate-limited on attempt %d; sleeping %.2fs before retry",
                tile_info.index, tile_info.origin, attempt + 1, delay,
            )
            _time.sleep(delay)
        except Exception as exc:
            _log.warning(
                "tile %d at %s: non-rate-limit error, skipping. err=%s: %s",
                tile_info.index, tile_info.origin, type(exc).__name__, exc,
            )
            return None
    return None


def detect_vessels_in_scene(
    *args,
    **kwargs,
) -> list[PositionObservation]:
    """Scene-level tiling entry point — not yet implemented.

    Phase F.3 wires :func:`_detect_tiles` into a cost-gated public API that
    returns :class:`PositionObservation` lists with NMS across tile
    boundaries.  Until then, callers can drive the tile loop directly via
    :func:`_detect_tiles` + :func:`vlm_detection_to_observation` per tile.
    """
    raise NotImplementedError(
        "detect_vessels_in_scene: full scene API lands in Phase F.3. "
        "Use _detect_tiles + vlm_detection_to_observation per tile until then."
    )
