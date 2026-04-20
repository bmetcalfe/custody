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
