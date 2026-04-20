"""End-to-end integration test for the scene-level VLM detection pipeline (Phase F.5).

Validates that Phase F.1 (tiling) + F.2 (rate-limit-retrying tile loop) + F.3
(cost gate + observation conversion) + F.4 (bbox_px propagation + NMS) compose
correctly into :func:`detect_vessels_in_scene`.

Uses a synthetic 1300×1300 scene with a UTM zone 50N transform representative
of the Tennent/Whitsun AOI.  Real GeoTIFF data isn't loaded — coordinate math
is already validated against a live pyproj round-trip in Phase E's
``test_observation_integration_with_real_transform_roundtrip``; this test
focuses on pipeline plumbing.

Scene geometry (tile_size=500, tile_overlap=100 → stride=400):
    tile 0: origin (  0,   0)
    tile 1: origin (  0, 400)
    tile 2: origin (  0, 800)
    tile 3: origin (400,   0)
    tile 4: origin (400, 400)
    tile 5: origin (400, 800)
    tile 6: origin (800,   0)
    tile 7: origin (800, 400)
    tile 8: origin (800, 800)

Scripted per-tile behavior:
    tile 0: 2 detections — one normal, one duplicated into tile 1's overlap region
    tile 1: 1 detection  — shared with tile 0 at the same scene bbox (higher confidence)
    tile 2: empty response (valid, zero detections)
    tile 3: rate-limit error on first attempt, succeeds on retry with 1 detection
    tile 4: 1 detection
    tile 5: 1 detection
    tile 6: ValueError — tile skipped, iteration continues
    tile 7: 1 detection
    tile 8: 1 detection

Expected: post-NMS, tile-0/tile-1 overlap merges to 1 observation (tile 1's
high-confidence wins).  Seven distinct observations total; tile 6 skipped
yields nothing.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from pyproj import CRS
from rasterio.transform import Affine

from custody.detection.vlm_backends.base import VLMBackend, VLMDetection, VLMResponse
from custody.detection.vlm_sar import (
    CostExceededError,
    detect_vessels_in_scene,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


TILE_SIZE = 500
TILE_OVERLAP = 100
SCENE_SIDE = 1300
EXPECTED_TILE_ORIGINS = [
    (0,   0), (0,   400), (0,   800),
    (400, 0), (400, 400), (400, 800),
    (800, 0), (800, 400), (800, 800),
]


def _utm50n_scene_transform() -> tuple[Affine, str]:
    """Affine + CRS_WKT for a synthetic scene anchored over the Tennent AOI.

    Places scene pixel (0, 0) at UTM50N easting 683_000, northing 983_000 — a
    point near Tennent Reef (8.86°N, 114.66°E ≈ UTM50N (683000, 980000)) — and
    uses a 1 m/pixel resolution with standard "y grows south" orientation.
    """
    transform = Affine.translation(683_000.0, 983_000.0) * Affine.scale(1.0, -1.0)
    crs_wkt = CRS.from_epsg(32650).to_wkt()
    return transform, crs_wkt


def _bright_scene() -> np.ndarray:
    """Synthetic uniform-bright scene so skip_empty never fires."""
    return np.full((SCENE_SIDE, SCENE_SIDE), 200, dtype=np.uint8)


# Scripted detections ------------------------------------------------------
#
# Tile 0's ``shared_bbox_tile0_local`` is at tile-local (420, 100, 480, 160),
# which sits inside the overlap column range [400, 500).  Tile 1 (origin
# col=400) scripts a detection at tile-local (20, 100, 80, 160), which is the
# same real-world bounding box in scene coordinates:
#
#   tile 0 scene: (420 + 0,   100 + 0,   480 + 0,   160 + 0)   = (420, 100, 480, 160)
#   tile 1 scene: (20  + 400, 100 + 0,   80  + 400, 160 + 0)   = (420, 100, 480, 160)
#
# Identical scene bbox → IoU = 1.0 → NMS merges to tile 1's "high" detection.


def _det(bbox, confidence="medium", reasoning="r") -> VLMDetection:
    return VLMDetection(bbox=bbox, confidence=confidence, reasoning=reasoning)


def _resp(dets: list[VLMDetection]) -> VLMResponse:
    return VLMResponse(
        detections=list(dets),
        tokens_used={"input": 100, "output": 50},
        wall_time_seconds=0.1,
    )


class _SimulatedRateLimitError(Exception):
    """Stand-in for anthropic/openai RateLimitError, injected via monkeypatch."""


class _ScriptedMockBackend(VLMBackend):
    """Backend that scripts responses by call-attempt sequence (not tile index).

    Rate-limit retries count as additional attempts — script them explicitly.
    """

    def __init__(self, scripts: list, model: str = "mock-scene-backend", cost: float = 0.001):
        self._scripts = list(scripts)
        self._model = model
        self._cost = cost
        self.call_count = 0
        self.tile_shapes_seen: list[tuple[int, ...]] = []

    def detect_tile(self, tile, prompt):
        self.call_count += 1
        self.tile_shapes_seen.append(tile.shape)
        if not self._scripts:
            raise AssertionError(
                f"Scripted backend exhausted on attempt {self.call_count}; "
                "too many backend calls — check tile layout or retry count."
            )
        entry = self._scripts.pop(0)
        if isinstance(entry, Exception):
            raise entry
        return entry

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def approx_cost_per_tile_usd(self) -> float:
        return self._cost


def _build_scripted_scene() -> tuple[list, dict]:
    """Build the per-attempt script and the expected post-NMS observation map.

    Returns:
        (scripts, expected): the attempt-ordered script list and a dict of
        ``{label: (scene_bbox_tuple, confidence_score)}`` for the 7 detections
        that should survive NMS.
    """
    # --- tile 0 (origin 0, 0): 2 detections ---
    tile0_normal_local = (10, 10, 50, 50)         # scene: (10, 10, 50, 50)
    tile0_shared_local = (420, 100, 480, 160)     # scene: (420, 100, 480, 160)
    # --- tile 1 (origin 0, 400): duplicate of shared, HIGH confidence ---
    tile1_shared_local = (20, 100, 80, 160)       # scene: (420, 100, 480, 160)
    # --- tile 3 (origin 400, 0): retry success, 1 det ---
    tile3_local = (100, 100, 140, 140)            # scene: (100, 500, 140, 540)
    # --- tile 4 (origin 400, 400): 1 det ---
    tile4_local = (200, 200, 240, 240)            # scene: (600, 600, 640, 640)
    # --- tile 5 (origin 400, 800): 1 det ---
    tile5_local = (150, 150, 190, 190)            # scene: (950, 550, 990, 590)
    # --- tile 7 (origin 800, 400): 1 det ---
    tile7_local = (50, 50, 90, 90)                # scene: (450, 850, 490, 890)
    # --- tile 8 (origin 800, 800): 1 det ---
    tile8_local = (250, 250, 290, 290)            # scene: (1050, 1050, 1090, 1090)

    scripts = [
        # tile 0
        _resp([_det(tile0_normal_local, confidence="medium", reasoning="t0 normal"),
               _det(tile0_shared_local, confidence="medium", reasoning="t0 shared")]),
        # tile 1
        _resp([_det(tile1_shared_local, confidence="high", reasoning="t1 shared")]),
        # tile 2 — empty response
        _resp([]),
        # tile 3 — first attempt rate-limited, retry succeeds
        _SimulatedRateLimitError("simulated rate limit"),
        _resp([_det(tile3_local, confidence="medium", reasoning="t3 retried")]),
        # tile 4
        _resp([_det(tile4_local, confidence="medium", reasoning="t4")]),
        # tile 5
        _resp([_det(tile5_local, confidence="medium", reasoning="t5")]),
        # tile 6 — non-rate-limit error, tile skipped
        ValueError("simulated parse error"),
        # tile 7
        _resp([_det(tile7_local, confidence="medium", reasoning="t7")]),
        # tile 8
        _resp([_det(tile8_local, confidence="medium", reasoning="t8")]),
    ]

    # Expected post-NMS survivors, keyed by reasoning string for easy assertion.
    # Confidence scores use the _CONF_TO_SCORE mapping (high=0.85, medium=0.60).
    expected = {
        "t0 normal": ((10, 10, 50, 50), 0.60),
        # t0 shared is dropped by NMS; t1 shared kept (higher conf)
        "t1 shared": ((420, 100, 480, 160), 0.85),
        "t3 retried": ((100, 500, 140, 540), 0.60),
        "t4": ((600, 600, 640, 640), 0.60),
        "t5": ((950, 550, 990, 590), 0.60),
        "t7": ((450, 850, 490, 890), 0.60),
        "t8": ((1050, 1050, 1090, 1090), 0.60),
    }
    return scripts, expected


def _patch_sdk_rate_limit_and_sleep(monkeypatch):
    import custody.detection.vlm_sar as vlm_sar

    monkeypatch.setattr(vlm_sar, "_RATE_LIMIT_EXCEPTIONS", (_SimulatedRateLimitError,))
    monkeypatch.setattr(vlm_sar._time, "sleep", lambda *_: None)


# ---------------------------------------------------------------------------
# End-to-end happy-path test
# ---------------------------------------------------------------------------


def test_end_to_end_scripted_scene_produces_expected_observations(monkeypatch):
    _patch_sdk_rate_limit_and_sleep(monkeypatch)
    transform, crs_wkt = _utm50n_scene_transform()
    scripts, expected = _build_scripted_scene()
    backend = _ScriptedMockBackend(scripts, cost=0.001)
    scene = _bright_scene()

    progress_events = []

    def progress_cb(tile_info, response, done, total):
        progress_events.append({
            "origin": tile_info.origin,
            "had_response": response is not None,
            "done": done,
            "total": total,
        })

    obs_list = detect_vessels_in_scene(
        scene,
        transform=transform, crs_wkt=crs_wkt,
        acquisition_time=1_688_306_455.0,
        source_id="integration-mock",
        backend=backend,
        prompt_variant="contextualized_v1",
        confidence_threshold="low",
        tile_size=TILE_SIZE, tile_overlap=TILE_OVERLAP,
        skip_empty=False,
        max_cost_usd=1.00,
        progress_callback=progress_cb,
    )

    # ---- Observation count ----
    assert len(obs_list) == len(expected), (
        f"expected {len(expected)} post-NMS observations; got {len(obs_list)}"
    )

    # ---- Per-observation structural integrity ----
    by_reasoning = {o.detector_reasoning: o for o in obs_list}
    assert set(by_reasoning.keys()) == set(expected.keys()), (
        f"reasoning labels mismatch: expected {set(expected)}, got {set(by_reasoning)}"
    )

    for label, (expected_bbox, expected_conf) in expected.items():
        o = by_reasoning[label]
        # bbox_px present and correct (scene-coord aware)
        assert o.bbox_px == expected_bbox, (
            f"{label}: bbox_px mismatch. expected {expected_bbox}, got {o.bbox_px}"
        )
        # confidence mapping (medium=0.60, high=0.85)
        assert o.classification_conf == pytest.approx(expected_conf), (
            f"{label}: confidence mismatch"
        )
        # detector_version embeds backend model_name
        assert o.detector_version == "vlm_mock-scene-backend"
        # source_id and modality
        assert o.source_id == "integration-mock"
        assert o.modality == "SAR"
        # lat/lon derived from transform — must be sensible (positive N lat,
        # E lon inside the broader South China Sea window).  The synthetic UTM50N
        # anchor (683000 E, 983000 N) back-projects to ~(8.9°N, 118.7°E), which
        # is inside the expected envelope.
        assert 5.0 < o.lat < 15.0, f"{label}: lat {o.lat} outside expected range"
        assert 110.0 < o.lon < 125.0, f"{label}: lon {o.lon} outside expected range"

    # ---- NMS: the low-confidence duplicate ("t0 shared") should be gone ----
    assert "t0 shared" not in by_reasoning, (
        "NMS failed: tile-0's shared detection should have been merged away"
    )

    # ---- Coordinate spread sanity: bboxes should span the full scene, not cluster ----
    xs = [o.bbox_px[0] for o in obs_list]
    ys = [o.bbox_px[1] for o in obs_list]
    assert max(xs) - min(xs) > 800, (
        f"bbox_px x spread too narrow ({max(xs)-min(xs)}); "
        "coordinate math likely not applying tile_origin correctly"
    )
    assert max(ys) - min(ys) > 800, (
        f"bbox_px y spread too narrow ({max(ys)-min(ys)})"
    )

    # ---- Rate-limit retry: tile 3's detection ('t3 retried') is in the result ----
    assert "t3 retried" in by_reasoning, (
        "rate-limit retry failed: tile 3's detection should have appeared on the 2nd attempt"
    )

    # ---- Error tolerance: tile 6's ValueError did not abort the scene ----
    # tile 7 and tile 8 come AFTER tile 6 in iteration order; their detections must be present.
    assert "t7" in by_reasoning and "t8" in by_reasoning, (
        "non-rate-limit error on tile 6 aborted iteration; subsequent tiles missing"
    )

    # ---- Backend call count: 9 tiles + 1 retry = 10 attempts ----
    assert backend.call_count == 10, (
        f"expected 10 backend attempts (9 tiles + 1 retry); got {backend.call_count}"
    )

    # ---- Progress callback: one event per tile iteration (9 attempts) ----
    # The retry on tile 3 is handled inside _detect_one_with_retry; the callback
    # fires once per tile-iteration, after the retry loop resolves.
    assert len(progress_events) == 9, (
        f"progress_callback should fire once per tile; got {len(progress_events)} events"
    )
    origins_in_order = [e["origin"] for e in progress_events]
    assert origins_in_order == EXPECTED_TILE_ORIGINS, (
        f"progress events not in row-major order: {origins_in_order}"
    )
    # Tile 6 (origin (800, 0)) produced no response → had_response=False.
    tile6_event = next(e for e in progress_events if e["origin"] == (800, 0))
    assert tile6_event["had_response"] is False
    # Tile 3 (origin (400, 0)) succeeded on retry → had_response=True.
    tile3_event = next(e for e in progress_events if e["origin"] == (400, 0))
    assert tile3_event["had_response"] is True


# ---------------------------------------------------------------------------
# Cost-gate check (second pass, low cap)
# ---------------------------------------------------------------------------


def test_end_to_end_cost_gate_fires_before_any_backend_call(monkeypatch):
    """With max_cost_usd set below the projected total, no tile should be processed."""
    _patch_sdk_rate_limit_and_sleep(monkeypatch)
    transform, crs_wkt = _utm50n_scene_transform()
    scripts, _ = _build_scripted_scene()
    # Bump per-tile cost so 9 × cost exceeds the low cap.
    backend = _ScriptedMockBackend(scripts, cost=2.00)  # 9 × $2 = $18 projected
    scene = _bright_scene()

    with pytest.raises(CostExceededError) as exc_info:
        detect_vessels_in_scene(
            scene,
            transform=transform, crs_wkt=crs_wkt,
            acquisition_time=1_688_306_455.0,
            source_id="integration-mock",
            backend=backend,
            tile_size=TILE_SIZE, tile_overlap=TILE_OVERLAP,
            skip_empty=False,
            max_cost_usd=5.00,
        )
    assert exc_info.value.estimated_cost == pytest.approx(18.0)
    assert backend.call_count == 0, "cost gate must fire before any backend call"
