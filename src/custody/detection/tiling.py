"""Tile iterator for scene-level VLM detection (Phase F).

:func:`iter_tiles` walks a 2D scene array in row-major order yielding
overlapping tiles with scene-coordinate origins.  Foundational building
block for :func:`custody.detection.vlm_sar.detect_vessels_in_scene`.

Design
------
For ``tile_size=T`` and ``tile_overlap=O`` the stride is ``S = T - O``.
Tile positions along each axis are ``0, S, 2S, ...`` until a tile reaches
the scene (or AOI) edge, at which point that final tile is included and
iteration stops.  Edge tiles may be smaller than ``tile_size`` when the
scene dimension is not a multiple of ``S``; downstream detection backends
handle variable-size input, so we yield the partial tile rather than
padding with zeros or reflecting.

The ``skip_empty`` flag suppresses tiles whose 95th-percentile pixel
value is below ``empty_threshold`` on a normalized [0, 1] scale.  Useful
for Umbra scenes with large no-data margins or open-water areas where
a VLM call would waste cost and tokens.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, Optional

import numpy as np

if TYPE_CHECKING:
    from rasterio.transform import Affine


@dataclass(eq=False)
class TileInfo:
    """One tile yielded by :func:`iter_tiles`."""

    tile: np.ndarray
    origin: tuple[int, int]       # (row, col) of tile's top-left in scene coords
    index: int                    # 0-based iteration index (only counts yielded tiles)
    tile_shape: tuple[int, int]   # (rows, cols); may be smaller than tile_size at edges


def iter_tiles(
    scene: np.ndarray,
    tile_size: int = 640,
    tile_overlap: int = 64,
    aoi_bounds: Optional[tuple[int, int, int, int]] = None,
    skip_empty: bool = False,
    empty_threshold: float = 0.05,
) -> Iterator[TileInfo]:
    """Yield :class:`TileInfo` tuples covering ``scene`` in row-major order.

    Args:
        scene: 2D or 3D numpy array; iteration uses its first two axes.
        tile_size: target tile edge length in pixels.
        tile_overlap: overlap between adjacent tiles in pixels; must be < tile_size.
        aoi_bounds: optional ``(row_start, row_end, col_start, col_end)`` half-open
            window; when given, tiles are yielded only inside the AOI, with
            origins reported in scene coordinates.
        skip_empty: if True, skip tiles where the 95th percentile of the
            normalized pixel values is below ``empty_threshold``.
        empty_threshold: threshold on the normalized [0, 1] pixel scale (uint8
            inputs are divided by 255 before the check).
    """
    if scene.ndim < 2:
        raise ValueError(f"scene must be at least 2D; got shape {scene.shape}")
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive; got {tile_size}")
    if tile_overlap < 0:
        raise ValueError(f"tile_overlap must be non-negative; got {tile_overlap}")
    if tile_overlap >= tile_size:
        raise ValueError(
            f"tile_overlap ({tile_overlap}) must be < tile_size ({tile_size})"
        )

    H, W = int(scene.shape[0]), int(scene.shape[1])
    if aoi_bounds is not None:
        r0, r1, c0, c1 = aoi_bounds
        if not (0 <= r0 < r1 <= H and 0 <= c0 < c1 <= W):
            raise ValueError(
                f"aoi_bounds {aoi_bounds} out of range for scene shape {scene.shape}"
            )
    else:
        r0, r1, c0, c1 = 0, H, 0, W

    row_starts = _starts_along_axis(r0, r1, tile_size, tile_overlap)
    col_starts = _starts_along_axis(c0, c1, tile_size, tile_overlap)

    idx = 0
    for rs in row_starts:
        re = min(rs + tile_size, r1)
        for cs in col_starts:
            ce = min(cs + tile_size, c1)
            tile = scene[rs:re, cs:ce]
            if skip_empty and _is_empty(tile, empty_threshold):
                continue
            yield TileInfo(
                tile=tile,
                origin=(int(rs), int(cs)),
                index=idx,
                tile_shape=(int(re - rs), int(ce - cs)),
            )
            idx += 1


def count_tiles(
    scene_shape: tuple[int, int],
    tile_size: int = 640,
    tile_overlap: int = 64,
    aoi_bounds: Optional[tuple[int, int, int, int]] = None,
) -> int:
    """Return the number of tiles :func:`iter_tiles` would yield (ignoring ``skip_empty``).

    Fast, allocation-free geometric count — suitable for progress-bar totals and
    pre-flight cost estimation when the full tile array materialization would
    be prohibitive on large Umbra scenes.
    """
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive; got {tile_size}")
    if tile_overlap < 0 or tile_overlap >= tile_size:
        raise ValueError(
            f"tile_overlap must be in [0, tile_size); got {tile_overlap} vs {tile_size}"
        )
    H, W = int(scene_shape[0]), int(scene_shape[1])
    if aoi_bounds is not None:
        r0, r1, c0, c1 = aoi_bounds
        if not (0 <= r0 < r1 <= H and 0 <= c0 < c1 <= W):
            raise ValueError(
                f"aoi_bounds {aoi_bounds} out of range for scene shape {scene_shape}"
            )
    else:
        r0, r1, c0, c1 = 0, H, 0, W
    return (
        len(_starts_along_axis(r0, r1, tile_size, tile_overlap))
        * len(_starts_along_axis(c0, c1, tile_size, tile_overlap))
    )


def _starts_along_axis(lo: int, hi: int, tile_size: int, overlap: int) -> list[int]:
    """Return tile start positions along one axis covering [lo, hi)."""
    stride = tile_size - overlap
    if hi - lo <= tile_size:
        return [lo]
    starts: list[int] = []
    s = lo
    while s < hi:
        starts.append(s)
        if s + tile_size >= hi:
            break
        s += stride
    return starts


def describe_aoi_bounds(
    aoi_bounds: tuple[int, int, int, int],
    transform: "Affine",
    crs_wkt: str,
) -> dict:
    """Describe what geographic region an ``aoi_bounds`` tuple covers.

    Intended as a pre-flight sanity check before
    :func:`custody.detection.vlm_sar.detect_vessels_in_scene` runs on real
    imagery — avoids surprises from mental UTM↔WGS84 estimates (see F.5
    integration-test discovery notes; UTM50N easting 683000 back-projects
    to ~118.7°E, not the ~114.7°E I initially guessed).

    Args:
        aoi_bounds: ``(row_start, row_end, col_start, col_end)`` half-open
            scene-pixel window, matching :func:`iter_tiles`' ``aoi_bounds``
            parameter.
        transform: rasterio Affine for the scene (as returned by
            :func:`custody.detection.sar_common.read_geotiff`).
        crs_wkt: scene CRS as WKT.

    Returns:
        ``{
            "pixel_bounds": (r0, r1, c0, c1),
            "corner_latlon": [(lat, lon), (lat, lon), (lat, lon), (lat, lon)],
                # NW, NE, SE, SW — row/col ordering mirrors the raster's
                # top-left origin convention.
            "center_latlon": (lat, lon),
            "approx_extent_km": (north_south_km, east_west_km),
        }``
    """
    # Deferred import — keeps tiling.py free of rasterio/pyproj at module
    # load while still allowing callers to use this helper when they have
    # already constructed an Affine via read_geotiff or similar.
    from custody.detection.sar_common import pixel_to_latlon

    r0, r1, c0, c1 = aoi_bounds
    if r1 < r0 or c1 < c0:
        raise ValueError(
            f"aoi_bounds must satisfy row_end >= row_start and col_end >= col_start; got {aoi_bounds}"
        )

    # Corner pixels in (row, col) raster convention.  Half-open bounds —
    # subtract 1 from the ends so we sample the last in-range pixel, not
    # one past it.
    last_row = max(r0, r1 - 1)
    last_col = max(c0, c1 - 1)
    nw = pixel_to_latlon(r0,       c0,       transform, crs_wkt)
    ne = pixel_to_latlon(r0,       last_col, transform, crs_wkt)
    se = pixel_to_latlon(last_row, last_col, transform, crs_wkt)
    sw = pixel_to_latlon(last_row, c0,       transform, crs_wkt)
    corners = [nw, ne, se, sw]
    center_row = (r0 + last_row) / 2.0
    center_col = (c0 + last_col) / 2.0
    center = pixel_to_latlon(center_row, center_col, transform, crs_wkt)

    lats = [lat for lat, _ in corners]
    lons = [lon for _, lon in corners]
    # Approximate great-circle distance: 111 km per degree lat; lon varies
    # with cos(lat).  Good enough for a "what size is my AOI" sanity check.
    ns_km = (max(lats) - min(lats)) * 111.0
    mean_lat = sum(lats) / len(lats)
    ew_km = (max(lons) - min(lons)) * 111.0 * float(np.cos(np.radians(mean_lat)))

    return {
        "pixel_bounds": (int(r0), int(r1), int(c0), int(c1)),
        "corner_latlon": [(float(lat), float(lon)) for lat, lon in corners],
        "center_latlon": (float(center[0]), float(center[1])),
        "approx_extent_km": (float(ns_km), float(ew_km)),
    }


def _is_empty(tile: np.ndarray, threshold: float) -> bool:
    """True if the tile's 95th-percentile pixel is below ``threshold`` on [0, 1]."""
    arr = tile
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    elif arr.dtype != np.float32 and arr.dtype != np.float64:
        arr = arr.astype(np.float32)
    if arr.size == 0:
        return True
    p95 = float(np.percentile(arr, 95))
    return p95 < threshold
