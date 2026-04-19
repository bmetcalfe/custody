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
from rasterio.windows import Window, transform as window_transform
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


def crop_to_aoi(
    img: np.ndarray,
    transform: Affine,
    target_lat: float,
    target_lon: float,
    crs_wkt: str,
    box_half_km: float = 1.0,
) -> tuple[np.ndarray, Affine, dict]:
    """Crop ``img`` to a (2 × box_half_km) × (2 × box_half_km) box around a target lat/lon.

    Returns ``(cropped_img, new_transform, bounds_dict)``.  The ``bounds_dict``
    contains ``row_start``, ``row_end``, ``col_start``, ``col_end``,
    ``target_row`` (in the original image), ``target_col`` (original), plus the
    target's fractional pixel in the *cropped* image as ``crop_target_row``
    and ``crop_target_col``.

    Raises :class:`ValueError` if the target lat/lon projects outside the
    image bounds.  When the box extends past the image edge, the crop is
    clipped to the image extent (no crash, partial crop returned).
    """
    h, w = img.shape[:2]
    tgt_row_f, tgt_col_f = latlon_to_pixel(target_lat, target_lon, transform, crs_wkt)
    if not (0.0 <= tgt_row_f <= h - 1) or not (0.0 <= tgt_col_f <= w - 1):
        raise ValueError(
            f"target ({target_lat}, {target_lon}) projects to "
            f"(row={tgt_row_f:.2f}, col={tgt_col_f:.2f}) which is outside "
            f"image bounds ({h}×{w})"
        )

    # Pixel size (meters per pixel-step) from the transform's linear block.
    # For a rotated GEC this is sqrt(a² + b²); equals the `scale` factor.
    pixel_size_m = float(np.hypot(transform.a, transform.b))
    if pixel_size_m <= 0.0:
        raise ValueError(f"transform has zero pixel size: {transform}")
    half_px = int(np.ceil(box_half_km * 1000.0 / pixel_size_m))

    tgt_row = int(round(tgt_row_f))
    tgt_col = int(round(tgt_col_f))
    r0 = max(0, tgt_row - half_px)
    r1 = min(h, tgt_row + half_px + 1)
    c0 = max(0, tgt_col - half_px)
    c1 = min(w, tgt_col + half_px + 1)

    cropped = img[r0:r1, c0:c1]
    win = Window(c0, r0, c1 - c0, r1 - r0)
    new_tfm = window_transform(win, transform)

    bounds = {
        "row_start": r0,
        "row_end": r1,
        "col_start": c0,
        "col_end": c1,
        "target_row": tgt_row,
        "target_col": tgt_col,
        "crop_target_row": tgt_row_f - r0,
        "crop_target_col": tgt_col_f - c0,
        "pixel_size_m": pixel_size_m,
        "half_px": half_px,
    }
    return cropped, new_tfm, bounds


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
