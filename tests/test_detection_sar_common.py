"""Tests for custody.detection.sar_common -- shared SAR detection helpers.

Synthetic-only: a 256x256 uint8 GeoTIFF is written to a tmp_path fixture
with a controlled affine transform (straight + rotated variants) and
EPSG:32650 CRS.  No real Umbra files required.
"""
from __future__ import annotations

from pathlib import Path

import math
import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

from custody.detection.sar_common import (
    read_geotiff,
    pixel_to_latlon,
    latlon_to_pixel,
    detection_to_observation,
    detections_to_observations,
)
from custody.fusion.observations import PositionObservation


EPSG_UTM50N = "EPSG:32650"


def _straight_transform(origin_x: float = 240000.0, origin_y: float = 980000.0,
                        pixel_size: float = 1.0) -> Affine:
    # Standard north-up transform in meters
    return Affine.translation(origin_x, origin_y) * Affine.scale(pixel_size, -pixel_size)


def _rotated_transform(origin_x: float = 240000.0, origin_y: float = 980000.0,
                       pixel_size: float = 1.0, rot_deg: float = 43.0) -> Affine:
    theta = math.radians(rot_deg)
    c, s = math.cos(theta), math.sin(theta)
    # Affine = [a b tx; d e ty] where (x, y) = A * (col, row)
    # a = s*cos, b = s*sin  (for rotation with scale s and angle theta)
    return Affine(pixel_size * c, pixel_size * s, origin_x,
                  pixel_size * s, -pixel_size * c, origin_y)


def _write_synthetic_tif(path: Path, img: np.ndarray, transform: Affine,
                         crs: str = EPSG_UTM50N) -> None:
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=img.shape[0], width=img.shape[1],
        count=1, dtype=img.dtype,
        crs=crs, transform=transform,
    ) as dst:
        dst.write(img, 1)


# ---------------------------------------------------------------------------
# 1. read_geotiff
# ---------------------------------------------------------------------------


def test_read_geotiff_returns_image_transform_crs(tmp_path: Path):
    img = (np.random.RandomState(42).rand(256, 256) * 255).astype(np.uint8)
    tfm = _straight_transform()
    p = tmp_path / "synth.tif"
    _write_synthetic_tif(p, img, tfm)

    got_img, got_tfm, got_crs = read_geotiff(p)
    assert got_img.shape == (256, 256)
    assert got_img.dtype == np.uint8
    np.testing.assert_array_equal(got_img, img)
    # Transform values match
    assert got_tfm.a == pytest.approx(tfm.a)
    assert got_tfm.e == pytest.approx(tfm.e)
    assert "32650" in got_crs  # EPSG code present in the WKT


# ---------------------------------------------------------------------------
# 2-4. pixel_to_latlon / latlon_to_pixel roundtrip
# ---------------------------------------------------------------------------


def test_pixel_to_latlon_roundtrip_straight(tmp_path: Path):
    img = np.zeros((256, 256), dtype=np.uint8)
    tfm = _straight_transform()
    p = tmp_path / "synth.tif"
    _write_synthetic_tif(p, img, tfm)
    _, t, crs = read_geotiff(p)

    # A few test pixels inside the image
    for row, col in [(0, 0), (128, 128), (255, 255), (10, 200)]:
        lat, lon = pixel_to_latlon(row, col, t, crs)
        back_row, back_col = latlon_to_pixel(lat, lon, t, crs)
        assert back_row == pytest.approx(row, abs=1e-6)
        assert back_col == pytest.approx(col, abs=1e-6)


def test_pixel_to_latlon_corner_inside_wgs84_bounds(tmp_path: Path):
    img = np.zeros((256, 256), dtype=np.uint8)
    tfm = _straight_transform()
    p = tmp_path / "synth.tif"
    _write_synthetic_tif(p, img, tfm)
    _, t, crs = read_geotiff(p)

    for row, col in [(0, 0), (255, 255), (0, 255), (255, 0)]:
        lat, lon = pixel_to_latlon(row, col, t, crs)
        assert -90.0 < lat < 90.0
        assert -180.0 < lon < 180.0


def test_pixel_to_latlon_rotated_transform_preserves_roundtrip(tmp_path: Path):
    img = np.zeros((256, 256), dtype=np.uint8)
    tfm = _rotated_transform(rot_deg=43.0)  # match Umbra's rotation
    p = tmp_path / "synth_rot.tif"
    _write_synthetic_tif(p, img, tfm)
    _, t, crs = read_geotiff(p)

    corners = [(0, 0), (0, 255), (255, 255), (255, 0)]
    latlons = [pixel_to_latlon(r, c, t, crs) for r, c in corners]
    # All four corners must map to distinct (lat, lon) pairs
    assert len({(round(lat, 6), round(lon, 6)) for lat, lon in latlons}) == 4
    # Roundtrip each
    for (r, c), (lat, lon) in zip(corners, latlons):
        br, bc = latlon_to_pixel(lat, lon, t, crs)
        assert br == pytest.approx(r, abs=1e-5)
        assert bc == pytest.approx(c, abs=1e-5)


# ---------------------------------------------------------------------------
# 5-9. detection_to_observation
# ---------------------------------------------------------------------------


def _tfm_crs(tmp_path: Path, rot: bool = False):
    img = np.zeros((256, 256), dtype=np.uint8)
    tfm = _rotated_transform() if rot else _straight_transform()
    p = tmp_path / ("synth_rot.tif" if rot else "synth.tif")
    _write_synthetic_tif(p, img, tfm)
    _, t, crs = read_geotiff(p)
    return t, crs


def test_detection_to_observation_produces_valid_position_observation(tmp_path: Path):
    tfm, crs = _tfm_crs(tmp_path)
    obs = detection_to_observation(
        row=100, col=150,
        transform=tfm, crs_wkt=crs,
        obs_id_prefix="umbra-test",
        acquisition_time=1_688_306_455.0,
        source_id="umbra",
    )
    assert isinstance(obs, PositionObservation)
    assert obs.modality == "SAR"
    assert obs.source_id == "umbra"
    # Default σ=10m → diag(100, 100) m²
    np.testing.assert_allclose(obs.cov_pos, np.diag([100.0, 100.0]), atol=1e-9)
    assert obs.detector_version == "sar_cfar_v1"


def test_detection_to_observation_covariance_passes_psd_validation(tmp_path: Path):
    # PositionObservation.__post_init__ runs the PSD check; constructing the
    # obs without exception is the test.
    tfm, crs = _tfm_crs(tmp_path)
    obs = detection_to_observation(
        row=50, col=50, transform=tfm, crs_wkt=crs,
        obs_id_prefix="p", acquisition_time=1.0, source_id="s",
    )
    eigs = np.linalg.eigvalsh(obs.cov_pos)
    assert eigs.min() > 0


def test_obs_id_format_and_uniqueness(tmp_path: Path):
    tfm, crs = _tfm_crs(tmp_path)
    a = detection_to_observation(row=10, col=20, transform=tfm, crs_wkt=crs,
                                 obs_id_prefix="umbra-p", acquisition_time=42.0,
                                 source_id="umbra")
    b = detection_to_observation(row=10, col=21, transform=tfm, crs_wkt=crs,
                                 obs_id_prefix="umbra-p", acquisition_time=42.0,
                                 source_id="umbra")
    assert a.obs_id == "umbra-p-42-10-20"
    assert b.obs_id == "umbra-p-42-10-21"
    assert a.obs_id != b.obs_id


def test_detection_to_observation_rejects_nonpositive_acquisition_time(tmp_path: Path):
    tfm, crs = _tfm_crs(tmp_path)
    with pytest.raises(ValueError, match="acquisition_time"):
        detection_to_observation(row=0, col=0, transform=tfm, crs_wkt=crs,
                                 obs_id_prefix="p", acquisition_time=0.0,
                                 source_id="s")
    with pytest.raises(ValueError, match="acquisition_time"):
        detection_to_observation(row=0, col=0, transform=tfm, crs_wkt=crs,
                                 obs_id_prefix="p", acquisition_time=-1.0,
                                 source_id="s")


def test_detections_to_observations_empty_input(tmp_path: Path):
    tfm, crs = _tfm_crs(tmp_path)
    out = detections_to_observations([], transform=tfm, crs_wkt=crs,
                                     obs_id_prefix="p", acquisition_time=1.0,
                                     source_id="s")
    assert out == []


def test_detections_to_observations_batch(tmp_path: Path):
    tfm, crs = _tfm_crs(tmp_path)
    rcs = [(10, 20), (30, 40), (50, 60)]
    out = detections_to_observations(rcs, transform=tfm, crs_wkt=crs,
                                     obs_id_prefix="p", acquisition_time=1.0,
                                     source_id="s")
    assert len(out) == 3
    assert all(isinstance(o, PositionObservation) for o in out)
    assert len({o.obs_id for o in out}) == 3
