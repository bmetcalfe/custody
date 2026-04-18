"""
Tests for custody.fusion.geo -- lat/lon <-> tangent-plane projection (ADR-0009).

Azimuthal Equidistant anchored at (AOI_ANCHOR_LAT, AOI_ANCHOR_LON) from config.
"""
from __future__ import annotations

import numpy as np
import pytest

from custody import config
from custody.fusion.geo import (
    to_tangent_plane,
    from_tangent_plane,
    to_tangent_plane_array,
    from_tangent_plane_array,
)


ANCHOR_LAT = config.AOI_ANCHOR_LAT
ANCHOR_LON = config.AOI_ANCHOR_LON


# ---------------------------------------------------------------------------
# 1. Roundtrip at anchor: (anchor_lat, anchor_lon) -> (0, 0) -> (anchor_lat, anchor_lon)
# ---------------------------------------------------------------------------


def test_anchor_forward_maps_to_origin():
    x, y = to_tangent_plane(ANCHOR_LAT, ANCHOR_LON)
    assert abs(x) < 1e-6
    assert abs(y) < 1e-6


def test_anchor_inverse_maps_origin_to_anchor():
    lat, lon = from_tangent_plane(0.0, 0.0)
    assert lat == pytest.approx(ANCHOR_LAT, abs=1e-10)
    assert lon == pytest.approx(ANCHOR_LON, abs=1e-10)


def test_roundtrip_at_anchor():
    x, y = to_tangent_plane(ANCHOR_LAT, ANCHOR_LON)
    lat, lon = from_tangent_plane(x, y)
    assert lat == pytest.approx(ANCHOR_LAT, abs=1e-10)
    assert lon == pytest.approx(ANCHOR_LON, abs=1e-10)


# ---------------------------------------------------------------------------
# 2. Roundtrip at AOI corner: 1 cm tolerance both directions
# ---------------------------------------------------------------------------


def test_roundtrip_at_aoi_corner():
    corner_lat, corner_lon = 11.0, 117.5  # NE corner of Spratly AOI
    x, y = to_tangent_plane(corner_lat, corner_lon)
    lat2, lon2 = from_tangent_plane(x, y)
    # 1 cm = 1e-2 m at the tangent plane; inverse converts to degrees.
    # 1 cm in lat = 1e-2 / 111_320 deg ≈ 9e-8 deg
    assert lat2 == pytest.approx(corner_lat, abs=1e-7)
    assert lon2 == pytest.approx(corner_lon, abs=1e-7)
    # And a direct meters roundtrip via the other direction
    x2, y2 = to_tangent_plane(lat2, lon2)
    assert x2 == pytest.approx(x, abs=1e-2)  # 1 cm
    assert y2 == pytest.approx(y, abs=1e-2)


# ---------------------------------------------------------------------------
# 3 & 4. Known-distance sanity: 1 km in lat or lon ~ 1000 m in y or x (±1 m)
# ---------------------------------------------------------------------------


def test_north_south_distance_one_km():
    # AEQD preserves distances from the anchor.  Place a point 1000 m due
    # north of the anchor using WGS84 geodesic arithmetic and verify the
    # projection gives y = 1000, x = 0.
    from pyproj import Geod
    geod = Geod(ellps="WGS84")
    lon1, lat1, _back_az = geod.fwd(ANCHOR_LON, ANCHOR_LAT, 0.0, 1000.0)
    x0, y0 = to_tangent_plane(ANCHOR_LAT, ANCHOR_LON)
    x1, y1 = to_tangent_plane(lat1, lon1)
    assert abs(x1 - x0) < 1.0  # no east-west drift along a meridian
    assert (y1 - y0) == pytest.approx(1000.0, abs=1.0)


def test_east_west_distance_one_km():
    # 1 km due east of the anchor via WGS84 geodesic; AEQD preserves the distance.
    from pyproj import Geod
    geod = Geod(ellps="WGS84")
    lon1, lat1, _back_az = geod.fwd(ANCHOR_LON, ANCHOR_LAT, 90.0, 1000.0)
    x0, y0 = to_tangent_plane(ANCHOR_LAT, ANCHOR_LON)
    x1, y1 = to_tangent_plane(lat1, lon1)
    assert abs(y1 - y0) < 1.0  # no north-south drift along a parallel (near anchor)
    assert (x1 - x0) == pytest.approx(1000.0, abs=1.0)


# ---------------------------------------------------------------------------
# 5. Vectorized equivalence: array version matches looped scalar
# ---------------------------------------------------------------------------


def test_vectorized_matches_scalar():
    lats = np.array([ANCHOR_LAT, 8.5, 11.0, 9.0, 10.5])
    lons = np.array([ANCHOR_LON, 114.5, 117.5, 115.5, 116.5])
    xs_arr, ys_arr = to_tangent_plane_array(lats, lons)
    for i in range(len(lats)):
        x_s, y_s = to_tangent_plane(float(lats[i]), float(lons[i]))
        assert xs_arr[i] == pytest.approx(x_s, abs=1e-9)
        assert ys_arr[i] == pytest.approx(y_s, abs=1e-9)

    # And inverse
    lats2, lons2 = from_tangent_plane_array(xs_arr, ys_arr)
    for i in range(len(lats)):
        lat_s, lon_s = from_tangent_plane(float(xs_arr[i]), float(ys_arr[i]))
        assert lats2[i] == pytest.approx(lat_s, abs=1e-9)
        assert lons2[i] == pytest.approx(lon_s, abs=1e-9)


# ---------------------------------------------------------------------------
# 6. Array shape preservation
# ---------------------------------------------------------------------------


def test_shape_preservation_forward():
    lats = np.linspace(8.5, 11.0, 7)
    lons = np.linspace(114.5, 117.5, 7)
    xs, ys = to_tangent_plane_array(lats, lons)
    assert xs.shape == (7,)
    assert ys.shape == (7,)


def test_shape_preservation_inverse():
    xs = np.linspace(-100_000.0, 100_000.0, 5)
    ys = np.linspace(-100_000.0, 100_000.0, 5)
    lats, lons = from_tangent_plane_array(xs, ys)
    assert lats.shape == (5,)
    assert lons.shape == (5,)


# ---------------------------------------------------------------------------
# 7. Empty-array edge case
# ---------------------------------------------------------------------------


def test_empty_forward():
    lats = np.array([], dtype=float)
    lons = np.array([], dtype=float)
    xs, ys = to_tangent_plane_array(lats, lons)
    assert xs.shape == (0,)
    assert ys.shape == (0,)


def test_empty_inverse():
    xs = np.array([], dtype=float)
    ys = np.array([], dtype=float)
    lats, lons = from_tangent_plane_array(xs, ys)
    assert lats.shape == (0,)
    assert lons.shape == (0,)
