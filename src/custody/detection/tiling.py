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
from typing import Iterator, Optional

import numpy as np


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
