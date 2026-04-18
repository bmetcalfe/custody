"""Lat/lon <-> tangent-plane conversion for the EKF's meters-basis frame.

Azimuthal Equidistant projection anchored at the AOI center
(``AOI_ANCHOR_LAT``, ``AOI_ANCHOR_LON`` in :mod:`custody.config`).  Per ADR-0009:
a single global anchor is accurate to ~41 m at the worst corner of a 300 km AOI,
which is negligible versus the 5 km position-sigma floor the EKF carries.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from pyproj import Transformer

from custody import config


_WGS84 = "EPSG:4326"


def _aeqd_proj_string(lat: float, lon: float) -> str:
    # Azimuthal Equidistant on a WGS84 sphere, anchored at (lat, lon) in degrees.
    return f"+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"


@lru_cache(maxsize=4)
def _forward_transformer(anchor_lat: float, anchor_lon: float) -> Transformer:
    """lat/lon -> (x_m, y_m). Cached per anchor."""
    return Transformer.from_crs(
        _WGS84,
        _aeqd_proj_string(anchor_lat, anchor_lon),
        always_xy=True,
    )


@lru_cache(maxsize=4)
def _inverse_transformer(anchor_lat: float, anchor_lon: float) -> Transformer:
    """(x_m, y_m) -> lat/lon. Cached per anchor."""
    return Transformer.from_crs(
        _aeqd_proj_string(anchor_lat, anchor_lon),
        _WGS84,
        always_xy=True,
    )


def _fwd() -> Transformer:
    return _forward_transformer(config.AOI_ANCHOR_LAT, config.AOI_ANCHOR_LON)


def _inv() -> Transformer:
    return _inverse_transformer(config.AOI_ANCHOR_LAT, config.AOI_ANCHOR_LON)


# ---------------------------------------------------------------------------
# Scalar API
# ---------------------------------------------------------------------------


def to_tangent_plane(lat: float, lon: float) -> tuple[float, float]:
    """Project `(lat, lon)` degrees to `(x_m, y_m)` tangent-plane meters."""
    x, y = _fwd().transform(lon, lat)
    return float(x), float(y)


def from_tangent_plane(x_m: float, y_m: float) -> tuple[float, float]:
    """Inverse: project `(x_m, y_m)` tangent-plane meters back to `(lat, lon)` degrees."""
    lon, lat = _inv().transform(x_m, y_m)
    return float(lat), float(lon)


# ---------------------------------------------------------------------------
# Vectorized API
# ---------------------------------------------------------------------------


def to_tangent_plane_array(
    lats: np.ndarray, lons: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized forward projection. Shapes of `lats` and `lons` must match."""
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    if lats.size == 0:
        return lats.copy(), lons.copy()
    xs, ys = _fwd().transform(lons, lats)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def from_tangent_plane_array(
    xs: np.ndarray, ys: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized inverse projection."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.size == 0:
        return xs.copy(), ys.copy()
    lons, lats = _inv().transform(xs, ys)
    return np.asarray(lats, dtype=float), np.asarray(lons, dtype=float)
