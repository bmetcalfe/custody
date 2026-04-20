"""Tests for vlm_sar — the backend-agnostic VLM detection API (Phase E).

Three test surfaces:

- Prompt dispatch: correct variant passed to backend, unknown variant rejected.
- Detection → observation conversion: coordinate math, confidence mapping,
  provenance fields, reasoning propagation.
- Scene stub: detect_vessels_in_scene raises NotImplementedError until Phase F.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from rasterio.transform import Affine

from custody.detection.vlm_backends.base import VLMBackend, VLMDetection, VLMResponse
from custody.detection.vlm_sar import (
    PROMPTS,
    detect_vessels_in_scene,
    detect_vessels_in_tile,
    vlm_detection_to_observation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeBackend(VLMBackend):
    """Minimal VLMBackend stub for dispatch tests.

    Captures the last prompt it was called with so tests can assert prompt
    selection without going near a real API.
    """

    def __init__(self, detections=None, model="fake-model"):
        self._detections = detections or []
        self._model = model
        self.last_prompt: str | None = None
        self.last_tile: np.ndarray | None = None

    def detect_tile(self, tile, prompt):
        self.last_prompt = prompt
        self.last_tile = tile
        return VLMResponse(
            detections=list(self._detections),
            tokens_used={"input": 100, "output": 50},
            wall_time_seconds=0.1,
        )

    @property
    def model_name(self):
        return self._model

    @property
    def approx_cost_per_tile_usd(self):
        return 0.001


def _make_detection(bbox=(10, 20, 30, 40), confidence="medium", reasoning="because"):
    return VLMDetection(bbox=bbox, confidence=confidence, reasoning=reasoning)


def _utm50n_wkt() -> str:
    """UTM zone 50N — covers the Whitsun/Tennent AOI in the South China Sea."""
    from pyproj import CRS

    return CRS.from_epsg(32650).to_wkt()


# ---------------------------------------------------------------------------
# PROMPTS registry
# ---------------------------------------------------------------------------


def test_prompts_registry_has_three_variants():
    assert set(PROMPTS.keys()) == {"direct_v1", "contextualized_v1", "reasoning_first_v1"}
    for key, text in PROMPTS.items():
        assert isinstance(text, str) and len(text) > 50, f"prompt {key!r} is suspiciously short"


def test_contextualized_prompt_includes_sar_specific_context():
    """contextualized_v1 is the Phase E default; must reference SAR physics."""
    text = PROMPTS["contextualized_v1"]
    assert "SAR" in text or "synthetic aperture radar" in text.lower()
    assert "azimuth" in text.lower()


# ---------------------------------------------------------------------------
# detect_vessels_in_tile
# ---------------------------------------------------------------------------


def test_detect_vessels_in_tile_dispatches_default_prompt():
    det = _make_detection()
    backend = _FakeBackend(detections=[det])
    tile = np.zeros((64, 64), dtype=np.uint8)
    out = detect_vessels_in_tile(tile, backend)
    assert out == [det]
    assert backend.last_prompt == PROMPTS["contextualized_v1"]


def test_detect_vessels_in_tile_dispatches_named_variant():
    backend = _FakeBackend()
    detect_vessels_in_tile(np.zeros((32, 32), dtype=np.uint8), backend, prompt_variant="direct_v1")
    assert backend.last_prompt == PROMPTS["direct_v1"]


def test_detect_vessels_in_tile_rejects_unknown_variant():
    backend = _FakeBackend()
    with pytest.raises(ValueError, match="unknown prompt_variant"):
        detect_vessels_in_tile(
            np.zeros((32, 32), dtype=np.uint8), backend, prompt_variant="nonsense_v99"  # type: ignore[arg-type]
        )


def test_detect_vessels_in_tile_returns_list_copy_not_backend_internal_ref():
    """Caller mutating returned list must not affect backend state."""
    backing = [_make_detection()]
    backend = _FakeBackend(detections=backing)
    tile = np.zeros((32, 32), dtype=np.uint8)
    out = detect_vessels_in_tile(tile, backend)
    out.append(_make_detection(bbox=(0, 0, 1, 1)))
    # Re-dispatch and check backing is unchanged
    out2 = detect_vessels_in_tile(tile, backend)
    assert len(out2) == 1


# ---------------------------------------------------------------------------
# vlm_detection_to_observation — coordinate math
# ---------------------------------------------------------------------------


def test_observation_uses_bbox_center_plus_tile_origin():
    """bbox (10,20,30,40) in tile-local + tile_origin (500, 600) → scene pixel (600, 610).

    center_col = (10+30)/2 = 20; center_row = (20+40)/2 = 30.
    scene_col = 20 + 600 = 620; scene_row = 30 + 500 = 530.
    """
    det = _make_detection(bbox=(10, 20, 30, 40))
    backend = _FakeBackend(model="test-model")
    with patch("custody.detection.vlm_sar.pixel_to_latlon") as mock_p2l:
        mock_p2l.return_value = (9.98, 114.63)
        vlm_detection_to_observation(
            det,
            transform=Affine.identity(),
            crs_wkt="IRRELEVANT-IN-MOCK",
            tile_origin=(500, 600),
            acquisition_time=1_688_306_455.0,
            source_id="probe",
            backend=backend,
        )
    args, kwargs = mock_p2l.call_args
    scene_row, scene_col = args[0], args[1]
    assert scene_row == pytest.approx(530.0)
    assert scene_col == pytest.approx(620.0)


def test_observation_returns_lat_lon_from_projection():
    det = _make_detection()
    backend = _FakeBackend()
    with patch("custody.detection.vlm_sar.pixel_to_latlon") as mock_p2l:
        mock_p2l.return_value = (9.98, 114.63)
        obs = vlm_detection_to_observation(
            det,
            transform=Affine.identity(),
            crs_wkt="IRRELEVANT-IN-MOCK",
            tile_origin=(0, 0),
            acquisition_time=1_688_306_455.0,
            source_id="probe",
            backend=backend,
        )
    assert obs.lat == pytest.approx(9.98)
    assert obs.lon == pytest.approx(114.63)


def test_observation_integration_with_real_transform_roundtrip():
    """End-to-end with a real UTM transform: pixel math meets real pyproj.

    Use an Affine that places pixel (0,0) at the scene top-left projected
    origin, 1 m/pixel.  Confirm the observation's lat/lon agrees with a
    direct pixel_to_latlon call on the same scene pixel.
    """
    from custody.detection.sar_common import pixel_to_latlon

    # South China Sea UTM 50N, arbitrary origin near Whitsun Reef.
    # 500000 E, 1100000 N is inside UTM 50N extent.
    transform = Affine.translation(500_000, 1_100_000) * Affine.scale(1.0, -1.0)
    crs_wkt = _utm50n_wkt()

    det = _make_detection(bbox=(100, 100, 200, 200))  # center at (150, 150)
    backend = _FakeBackend()
    tile_origin = (1000, 2000)  # scene pixel center = (1150, 2150) row,col

    obs = vlm_detection_to_observation(
        det, transform=transform, crs_wkt=crs_wkt, tile_origin=tile_origin,
        acquisition_time=1_688_306_455.0, source_id="probe", backend=backend,
    )
    expected_lat, expected_lon = pixel_to_latlon(1150, 2150, transform, crs_wkt)
    assert obs.lat == pytest.approx(expected_lat, abs=1e-9)
    assert obs.lon == pytest.approx(expected_lon, abs=1e-9)


# ---------------------------------------------------------------------------
# vlm_detection_to_observation — field population
# ---------------------------------------------------------------------------


def test_observation_confidence_mapping():
    backend = _FakeBackend()
    cases = [("high", 0.85), ("medium", 0.60), ("low", 0.35)]
    for label, score in cases:
        det = _make_detection(confidence=label)
        with patch("custody.detection.vlm_sar.pixel_to_latlon", return_value=(9.0, 114.0)):
            obs = vlm_detection_to_observation(
                det, transform=Affine.identity(), crs_wkt="x", tile_origin=(0, 0),
                acquisition_time=1.0, source_id="probe", backend=backend,
            )
        assert obs.classification_conf == pytest.approx(score), f"label {label!r}"


def test_observation_unrecognized_confidence_defaults_to_medium():
    det = _make_detection(confidence="definitely-not-a-real-level")
    backend = _FakeBackend()
    with patch("custody.detection.vlm_sar.pixel_to_latlon", return_value=(9.0, 114.0)):
        obs = vlm_detection_to_observation(
            det, transform=Affine.identity(), crs_wkt="x", tile_origin=(0, 0),
            acquisition_time=1.0, source_id="probe", backend=backend,
        )
    assert obs.classification_conf == pytest.approx(0.60)


def test_observation_detector_version_embeds_model_name():
    det = _make_detection()
    backend = _FakeBackend(model="claude-sonnet-4-6")
    with patch("custody.detection.vlm_sar.pixel_to_latlon", return_value=(9.0, 114.0)):
        obs = vlm_detection_to_observation(
            det, transform=Affine.identity(), crs_wkt="x", tile_origin=(0, 0),
            acquisition_time=1.0, source_id="probe", backend=backend,
        )
    assert obs.detector_version == "vlm_claude-sonnet-4-6"


def test_observation_propagates_reasoning():
    det = _make_detection(reasoning="Bright compact return; azimuth smear consistent with metal hull.")
    backend = _FakeBackend()
    with patch("custody.detection.vlm_sar.pixel_to_latlon", return_value=(9.0, 114.0)):
        obs = vlm_detection_to_observation(
            det, transform=Affine.identity(), crs_wkt="x", tile_origin=(0, 0),
            acquisition_time=1.0, source_id="probe", backend=backend,
        )
    assert obs.detector_reasoning == "Bright compact return; azimuth smear consistent with metal hull."


def test_observation_empty_reasoning_becomes_none():
    """Empty-string reasoning should round-trip as None, not ``''``, in the fusion schema."""
    det = _make_detection(reasoning="")
    backend = _FakeBackend()
    with patch("custody.detection.vlm_sar.pixel_to_latlon", return_value=(9.0, 114.0)):
        obs = vlm_detection_to_observation(
            det, transform=Affine.identity(), crs_wkt="x", tile_origin=(0, 0),
            acquisition_time=1.0, source_id="probe", backend=backend,
        )
    assert obs.detector_reasoning is None


def test_observation_default_sigma_m_sets_covariance():
    det = _make_detection()
    backend = _FakeBackend()
    with patch("custody.detection.vlm_sar.pixel_to_latlon", return_value=(9.0, 114.0)):
        obs = vlm_detection_to_observation(
            det, transform=Affine.identity(), crs_wkt="x", tile_origin=(0, 0),
            acquisition_time=1.0, source_id="probe", backend=backend,
        )
    # Default sigma_m=20.0 → variance 400.0
    assert obs.cov_pos[0, 0] == pytest.approx(400.0)
    assert obs.cov_pos[1, 1] == pytest.approx(400.0)
    assert obs.cov_pos[0, 1] == 0.0 and obs.cov_pos[1, 0] == 0.0


def test_observation_custom_sigma_m():
    det = _make_detection()
    backend = _FakeBackend()
    with patch("custody.detection.vlm_sar.pixel_to_latlon", return_value=(9.0, 114.0)):
        obs = vlm_detection_to_observation(
            det, transform=Affine.identity(), crs_wkt="x", tile_origin=(0, 0),
            acquisition_time=1.0, source_id="probe", backend=backend,
            sigma_m=5.0,
        )
    assert obs.cov_pos[0, 0] == pytest.approx(25.0)


def test_observation_modality_is_sar():
    det = _make_detection()
    backend = _FakeBackend()
    with patch("custody.detection.vlm_sar.pixel_to_latlon", return_value=(9.0, 114.0)):
        obs = vlm_detection_to_observation(
            det, transform=Affine.identity(), crs_wkt="x", tile_origin=(0, 0),
            acquisition_time=1.0, source_id="probe", backend=backend,
        )
    assert obs.modality == "SAR"


def test_observation_obs_id_unique_across_bboxes_in_same_scene():
    """Different bboxes in the same scene → different obs_ids (stable against collision)."""
    backend = _FakeBackend()
    with patch("custody.detection.vlm_sar.pixel_to_latlon", return_value=(9.0, 114.0)):
        obs_a = vlm_detection_to_observation(
            _make_detection(bbox=(0, 0, 10, 10)),
            transform=Affine.identity(), crs_wkt="x", tile_origin=(0, 0),
            acquisition_time=1.0, source_id="probe", backend=backend,
        )
        obs_b = vlm_detection_to_observation(
            _make_detection(bbox=(100, 100, 110, 110)),
            transform=Affine.identity(), crs_wkt="x", tile_origin=(0, 0),
            acquisition_time=1.0, source_id="probe", backend=backend,
        )
    assert obs_a.obs_id != obs_b.obs_id


# ---------------------------------------------------------------------------
# detect_vessels_in_scene stub
# ---------------------------------------------------------------------------


def test_detect_vessels_in_scene_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="Phase F"):
        detect_vessels_in_scene()


# ---------------------------------------------------------------------------
# _detect_tiles — per-tile retry loop (Phase F.2)
# ---------------------------------------------------------------------------


class _SimulatedRateLimitError(Exception):
    """Stand-in for anthropic/openai RateLimitError classes in tests.

    Injected via monkeypatch onto vlm_sar._RATE_LIMIT_EXCEPTIONS so we can
    simulate rate-limit behavior without constructing real SDK exceptions
    (which require httpx.Response objects and other plumbing).
    """


class _ScriptedBackend(VLMBackend):
    """Backend that returns scripted responses or raises scripted exceptions.

    Accepts a list of per-call scripts: each entry is either a VLMResponse (to
    return) or an Exception instance (to raise).  The backend fires through
    the script in order; one scripted entry = one call attempt.
    """

    def __init__(self, scripts: list, model: str = "scripted"):
        self._scripts = list(scripts)
        self._model = model
        self.call_count = 0

    def detect_tile(self, tile, prompt):
        self.call_count += 1
        if not self._scripts:
            raise AssertionError("Scripted backend exhausted; unexpected extra call")
        entry = self._scripts.pop(0)
        if isinstance(entry, Exception):
            raise entry
        return entry

    @property
    def model_name(self):
        return self._model

    @property
    def approx_cost_per_tile_usd(self):
        return 0.001


def _trivial_response(n: int = 0) -> VLMResponse:
    dets = [_make_detection(bbox=(i, i, i + 5, i + 5)) for i in range(n)]
    return VLMResponse(
        detections=dets,
        tokens_used={"input": 100, "output": 50},
        wall_time_seconds=0.05,
    )


def _patch_rate_limit_and_sleep(monkeypatch):
    """Swap the live rate-limit tuple for our simulated class and no-op sleep."""
    import custody.detection.vlm_sar as vlm_sar

    monkeypatch.setattr(vlm_sar, "_RATE_LIMIT_EXCEPTIONS", (_SimulatedRateLimitError,))
    monkeypatch.setattr(vlm_sar._time, "sleep", lambda *_: None)


def test_detect_tiles_happy_path_yields_every_tile(monkeypatch):
    """All tiles succeed; every iter_tiles yield produces a (tile, response) result."""
    from custody.detection.vlm_sar import _detect_tiles

    _patch_rate_limit_and_sleep(monkeypatch)
    scene = np.full((1000, 1000), 200, dtype=np.uint8)  # bright everywhere; skip_empty won't drop
    # 1000 / (500 - 0) = 4 tiles with tile_size=500, overlap=0
    backend = _ScriptedBackend([_trivial_response(n=1) for _ in range(4)])
    results = list(_detect_tiles(
        scene, backend, prompt="test",
        tile_size=500, tile_overlap=0, skip_empty=False,
    ))
    assert len(results) == 4
    assert backend.call_count == 4
    for tile_info, response in results:
        assert len(response.detections) == 1


def test_detect_tiles_rate_limit_retry_succeeds_on_second_attempt(monkeypatch):
    """First call rate-limited; retry succeeds."""
    from custody.detection.vlm_sar import _detect_tiles

    _patch_rate_limit_and_sleep(monkeypatch)
    scene = np.full((500, 500), 200, dtype=np.uint8)
    backend = _ScriptedBackend([
        _SimulatedRateLimitError("rate limited"),
        _trivial_response(n=1),
    ])
    results = list(_detect_tiles(
        scene, backend, prompt="test",
        tile_size=500, tile_overlap=0, skip_empty=False, max_retries=3,
    ))
    assert len(results) == 1
    assert backend.call_count == 2  # one retry


def test_detect_tiles_rate_limit_exhaustion_skips_tile_without_crashing(monkeypatch):
    """All rate-limit attempts fail; tile skipped, no crash, iteration continues."""
    from custody.detection.vlm_sar import _detect_tiles

    _patch_rate_limit_and_sleep(monkeypatch)
    # 2 tiles total: first exhausts retries, second succeeds.
    scene = np.full((500, 1000), 200, dtype=np.uint8)
    backend = _ScriptedBackend([
        _SimulatedRateLimitError("rl 1"),
        _SimulatedRateLimitError("rl 2"),
        _SimulatedRateLimitError("rl 3"),
        _SimulatedRateLimitError("rl 4"),  # max_retries=3 → 4 attempts total
        _trivial_response(n=1),             # second tile succeeds first try
    ])
    results = list(_detect_tiles(
        scene, backend, prompt="test",
        tile_size=500, tile_overlap=0, skip_empty=False, max_retries=3,
    ))
    # Only second tile yields a result.
    assert len(results) == 1
    assert backend.call_count == 5
    assert results[0][0].origin == (0, 500)  # the second tile


def test_detect_tiles_non_rate_limit_error_skips_tile_and_continues(monkeypatch):
    """ValueError / other transient error: tile skipped, iteration continues, no retry."""
    from custody.detection.vlm_sar import _detect_tiles

    _patch_rate_limit_and_sleep(monkeypatch)
    scene = np.full((500, 1000), 200, dtype=np.uint8)
    backend = _ScriptedBackend([
        ValueError("transient parse error"),
        _trivial_response(n=2),
    ])
    results = list(_detect_tiles(
        scene, backend, prompt="test",
        tile_size=500, tile_overlap=0, skip_empty=False,
    ))
    assert len(results) == 1
    assert backend.call_count == 2  # no retry for non-rate-limit error
    assert len(results[0][1].detections) == 2


def test_detect_tiles_progress_callback_fires_per_tile(monkeypatch):
    """Callback fires once per attempted tile with (tile_info, response_or_none, done, total)."""
    from custody.detection.vlm_sar import _detect_tiles

    _patch_rate_limit_and_sleep(monkeypatch)
    scene = np.full((500, 1000), 200, dtype=np.uint8)  # 2 tiles at tile=500, overlap=0
    backend = _ScriptedBackend([
        ValueError("fail"),
        _trivial_response(n=1),
    ])
    events = []

    def cb(tile_info, response, done, total):
        events.append({
            "origin": tile_info.origin,
            "response_is_none": response is None,
            "done": done,
            "total": total,
        })

    list(_detect_tiles(
        scene, backend, prompt="test",
        tile_size=500, tile_overlap=0, skip_empty=False,
        progress_callback=cb,
    ))
    assert len(events) == 2
    assert events[0] == {"origin": (0, 0),   "response_is_none": True,  "done": 1, "total": 2}
    assert events[1] == {"origin": (0, 500), "response_is_none": False, "done": 2, "total": 2}


def test_detect_tiles_respects_skip_empty(monkeypatch):
    """skip_empty=True drops uniform-zero tiles; backend is not called for them."""
    from custody.detection.vlm_sar import _detect_tiles

    _patch_rate_limit_and_sleep(monkeypatch)
    scene = np.zeros((500, 1000), dtype=np.uint8)
    scene[:, :500] = 200  # left tile bright, right tile empty
    backend = _ScriptedBackend([_trivial_response(n=1)])  # only one call expected
    results = list(_detect_tiles(
        scene, backend, prompt="test",
        tile_size=500, tile_overlap=0, skip_empty=True,
    ))
    assert len(results) == 1
    assert backend.call_count == 1
    assert results[0][0].origin == (0, 0)


def test_detect_tiles_real_sdk_rate_limit_classes_are_loaded():
    """Sanity: the module-level tuple should contain the real SDK classes at import time.

    Guards against future SDK reshuffles that might rename RateLimitError.
    """
    import custody.detection.vlm_sar as vlm_sar
    from anthropic import RateLimitError as AnthropicRateLimit
    from openai import RateLimitError as OpenAIRateLimit

    classes = vlm_sar._compute_rate_limit_exceptions()
    assert AnthropicRateLimit in classes
    assert OpenAIRateLimit in classes


# ---------------------------------------------------------------------------
# count_tiles helper (Phase F.2)
# ---------------------------------------------------------------------------


def test_count_tiles_matches_iter_tiles_on_several_configurations():
    """count_tiles is the fast estimator for progress totals — must match iter_tiles."""
    from custody.detection.tiling import count_tiles, iter_tiles

    cases = [
        ((1000, 1000), 500, 0, None),
        ((1800, 1800), 500, 50, None),
        ((2000, 2000), 500, 50, None),
        ((2000, 2000), 500, 50, (500, 1500, 500, 1500)),
        ((17602, 17602), 640, 64, None),
        ((640, 640), 640, 64, None),   # exact fit → 1 tile
        ((100, 150), 640, 64, None),   # scene smaller than tile → 1 tile
    ]
    for shape, ts, ov, aoi in cases:
        scene = np.zeros(shape, dtype=np.uint8)
        counted = count_tiles(shape, tile_size=ts, tile_overlap=ov, aoi_bounds=aoi)
        iterated = sum(1 for _ in iter_tiles(
            scene, tile_size=ts, tile_overlap=ov, aoi_bounds=aoi,
        ))
        assert counted == iterated, f"mismatch at {(shape, ts, ov, aoi)}: {counted} != {iterated}"
