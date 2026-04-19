"""Shared helpers for SAR detection: GeoTIFF I/O, pixel ↔ lat/lon, PositionObservation bridge.

All routines handle rotated GEC transforms (as Umbra spotlight scenes carry)
transparently via :class:`rasterio.transform.Affine`'s standard operations.
"""
from __future__ import annotations

import time as _time
from pathlib import Path
from typing import Iterable, Sequence, Union

import numpy as np
import rasterio
from rasterio.transform import Affine, rowcol, xy
from pyproj import Transformer

from custody.fusion.observations import PositionObservation


# Cache one Transformer per (crs_wkt, direction) pair — pyproj transformer
# construction is the hot cost, not the transform itself.
_TFMS: dict[tuple[str, str], Transformer] = {}


def _forward_transformer(crs_wkt: str) -> Transformer:
    key = (crs_wkt, "forward")
    if key not in _TFMS:
        _TFMS[key] = Transformer.from_crs(crs_wkt, "EPSG:4326", always_xy=True)
    return _TFMS[key]


def _inverse_transformer(crs_wkt: str) -> Transformer:
    key = (crs_wkt, "inverse")
    if key not in _TFMS:
        _TFMS[key] = Transformer.from_crs("EPSG:4326", crs_wkt, always_xy=True)
    return _TFMS[key]


# ---------------------------------------------------------------------------
# GeoTIFF I/O
# ---------------------------------------------------------------------------


def read_geotiff(path: Union[str, Path]) -> tuple[np.ndarray, Affine, str]:
    """Read the first band of a GeoTIFF and return (image, transform, crs_wkt)."""
    with rasterio.open(Path(path)) as src:
        img = src.read(1)
        return img, src.transform, src.crs.to_wkt() if src.crs else ""


# ---------------------------------------------------------------------------
# Pixel ↔ lat/lon
# ---------------------------------------------------------------------------


def pixel_to_latlon(
    row: Union[int, float, np.ndarray],
    col: Union[int, float, np.ndarray],
    transform: Affine,
    crs_wkt: str,
) -> tuple[float, float]:
    """Convert a pixel (row, col) to (lat, lon) in degrees.

    Accepts scalar or array-like row/col.  Returns scalars when inputs are
    scalar, numpy arrays when inputs are arrays.
    """
    x, y = xy(transform, row, col)
    fwd = _forward_transformer(crs_wkt)
    lon, lat = fwd.transform(x, y)
    return lat, lon


def latlon_to_pixel(
    lat: Union[float, np.ndarray],
    lon: Union[float, np.ndarray],
    transform: Affine,
    crs_wkt: str,
) -> tuple[float, float]:
    """Convert (lat, lon) degrees to fractional (row, col) in the raster frame."""
    inv = _inverse_transformer(crs_wkt)
    x, y = inv.transform(lon, lat)
    # rowcol with op=float returns pixel-edge coordinates; subtract 0.5 to
    # align with rasterio.transform.xy's pixel-centre convention used in
    # pixel_to_latlon, so the pair roundtrips cleanly.
    row, col = rowcol(transform, x, y, op=float)
    return row - 0.5, col - 0.5


# ---------------------------------------------------------------------------
# PositionObservation construction
# ---------------------------------------------------------------------------


def detection_to_observation(
    row: int,
    col: int,
    transform: Affine,
    crs_wkt: str,
    *,
    obs_id_prefix: str,
    acquisition_time: float,
    source_id: str,
    sigma_m: float = 10.0,
    detector_version: str = "sar_cfar_v1",
) -> PositionObservation:
    """Bridge a pixel-space detection into a fully-validated PositionObservation.

    Default position σ is 10 m — reasonable for Umbra GEC centroids under a
    CFAR + connected-components detector.  Override if a sensor or algorithm
    warrants a different uncertainty.
    """
    if acquisition_time <= 0:
        raise ValueError(f"acquisition_time must be > 0 (got {acquisition_time})")
    lat, lon = pixel_to_latlon(row, col, transform, crs_wkt)
    cov_pos = np.array([[sigma_m ** 2, 0.0], [0.0, sigma_m ** 2]], dtype=float)
    return PositionObservation(
        obs_id=f"{obs_id_prefix}-{int(acquisition_time)}-{int(row)}-{int(col)}",
        source_id=source_id,
        modality="SAR",
        acquisition_time=acquisition_time,
        ingestion_time=_time.time(),
        lat=float(lat),
        lon=float(lon),
        cov_pos=cov_pos,
        raw_ref=obs_id_prefix,
        detector_version=detector_version,
        classification_conf=None,
        vessel_length_est_m=None,
        heading_est_deg=None,
        notes={},
    )


def detections_to_observations(
    detections: Iterable[Sequence[int]],
    transform: Affine,
    crs_wkt: str,
    *,
    obs_id_prefix: str,
    acquisition_time: float,
    source_id: str,
    sigma_m: float = 10.0,
    detector_version: str = "sar_cfar_v1",
) -> list[PositionObservation]:
    """Batch wrapper over :func:`detection_to_observation`."""
    out: list[PositionObservation] = []
    for rc in detections:
        r, c = rc[0], rc[1]
        out.append(detection_to_observation(
            r, c, transform, crs_wkt,
            obs_id_prefix=obs_id_prefix,
            acquisition_time=acquisition_time,
            source_id=source_id,
            sigma_m=sigma_m,
            detector_version=detector_version,
        ))
    return out
