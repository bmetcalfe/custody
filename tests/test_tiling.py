"""Tests for the tile iterator (Phase F.1)."""
from __future__ import annotations

import numpy as np
import pytest

from custody.detection.tiling import TileInfo, iter_tiles


def _count_tiles(scene: np.ndarray, **kwargs) -> int:
    return sum(1 for _ in iter_tiles(scene, **kwargs))


# ---------------------------------------------------------------------------
# Basic grid geometry
# ---------------------------------------------------------------------------


def test_iter_tiles_no_overlap_produces_exact_grid():
    """1000x1000 scene, tile=500, overlap=0 → 4 tiles at (0,0), (0,500), (500,0), (500,500)."""
    scene = np.zeros((1000, 1000), dtype=np.uint8)
    tiles = list(iter_tiles(scene, tile_size=500, tile_overlap=0))
    assert len(tiles) == 4
    origins = [t.origin for t in tiles]
    assert origins == [(0, 0), (0, 500), (500, 0), (500, 500)]
    for t in tiles:
        assert t.tile_shape == (500, 500)
        assert t.tile.shape == (500, 500)


def test_iter_tiles_1800_tile500_overlap50_yields_16_tiles():
    """1800x1800 scene, tile=500, overlap=50 → stride=450. Starts per axis:
    0, 450, 900, 1350 (4 starts, last tile [1350, 1800]).  Result: 4x4 = 16 tiles.
    Edge tiles are 450x500 (right column) / 500x450 (bottom row) / 450x450 (corner)."""
    scene = np.zeros((1800, 1800), dtype=np.uint8)
    tiles = list(iter_tiles(scene, tile_size=500, tile_overlap=50))
    assert len(tiles) == 16
    row_origins = sorted({t.origin[0] for t in tiles})
    col_origins = sorted({t.origin[1] for t in tiles})
    assert row_origins == [0, 450, 900, 1350]
    assert col_origins == [0, 450, 900, 1350]


def test_iter_tiles_edge_tile_shape_smaller_when_scene_not_divisible():
    """2000x2000, tile=500, overlap=50 → stride=450, starts 0,450,900,1350,1800.
    The last start (1800) produces a partial tile of shape (200, 200) at the
    bottom-right corner."""
    scene = np.zeros((2000, 2000), dtype=np.uint8)
    tiles = list(iter_tiles(scene, tile_size=500, tile_overlap=50))
    assert len(tiles) == 25  # 5 x 5
    # Corner tile at (1800, 1800) spans [1800, 2000) x [1800, 2000) → 200x200.
    corner = [t for t in tiles if t.origin == (1800, 1800)]
    assert len(corner) == 1
    assert corner[0].tile_shape == (200, 200)
    assert corner[0].tile.shape == (200, 200)


def test_iter_tiles_overlap_region_shared_between_adjacent_tiles():
    """Adjacent tiles' array slices should cover the overlap region identically."""
    rng = np.random.default_rng(0)
    scene = rng.integers(0, 255, size=(1000, 1000), dtype=np.uint8)
    tiles = list(iter_tiles(scene, tile_size=500, tile_overlap=100))
    # First two tiles in the top row are at (0, 0) and (0, 400). The overlap
    # region is cols [400, 500) — that's the last 100 cols of tile0 and the
    # first 100 cols of tile1, and they must match.
    t0 = next(t for t in tiles if t.origin == (0, 0))
    t1 = next(t for t in tiles if t.origin == (0, 400))
    np.testing.assert_array_equal(t0.tile[:, 400:500], t1.tile[:, 0:100])


def test_iter_tiles_scene_smaller_than_tile_yields_single_tile():
    """Scene smaller than tile_size yields exactly one tile covering the whole scene."""
    scene = np.zeros((100, 150), dtype=np.uint8)
    tiles = list(iter_tiles(scene, tile_size=640, tile_overlap=64))
    assert len(tiles) == 1
    assert tiles[0].origin == (0, 0)
    assert tiles[0].tile_shape == (100, 150)


# ---------------------------------------------------------------------------
# AOI restriction
# ---------------------------------------------------------------------------


def test_iter_tiles_aoi_bounds_restricts_iteration():
    """AOI (500, 1500, 500, 1500) on a 2000x2000 scene, tile=500, overlap=50.
    AOI is 1000x1000 → 3 starts per axis (500, 950, 1400) → 9 tiles total.
    Origins are in scene coords, not AOI-local coords."""
    scene = np.zeros((2000, 2000), dtype=np.uint8)
    tiles = list(iter_tiles(
        scene, tile_size=500, tile_overlap=50,
        aoi_bounds=(500, 1500, 500, 1500),
    ))
    assert len(tiles) == 9
    origins = {t.origin for t in tiles}
    assert origins == {
        (500, 500),  (500, 950),  (500, 1400),
        (950, 500),  (950, 950),  (950, 1400),
        (1400, 500), (1400, 950), (1400, 1400),
    }
    # Last tile (1400, 1400) extends [1400, 1500) in both axes → partial 100x100.
    last = next(t for t in tiles if t.origin == (1400, 1400))
    assert last.tile_shape == (100, 100)


def test_iter_tiles_aoi_bounds_out_of_range_raises():
    scene = np.zeros((100, 100), dtype=np.uint8)
    with pytest.raises(ValueError, match="aoi_bounds"):
        list(iter_tiles(scene, tile_size=50, tile_overlap=0, aoi_bounds=(0, 200, 0, 50)))


# ---------------------------------------------------------------------------
# skip_empty behavior
# ---------------------------------------------------------------------------


def test_iter_tiles_skip_empty_removes_zero_tiles():
    """Scene with a zero-region and a signal region.  skip_empty should skip the zero tiles."""
    scene = np.zeros((1000, 1000), dtype=np.uint8)
    # Put bright signal in the top-left quadrant only.
    scene[0:500, 0:500] = 200
    # Without skip_empty: 4 tiles at 500/0 overlap.
    assert _count_tiles(scene, tile_size=500, tile_overlap=0) == 4
    # With skip_empty: only the top-left tile survives (the other three are uniform 0).
    kept = list(iter_tiles(scene, tile_size=500, tile_overlap=0, skip_empty=True))
    assert len(kept) == 1
    assert kept[0].origin == (0, 0)


def test_iter_tiles_skip_empty_keeps_nonzero_tiles_even_with_noise():
    """Water-noise tiles (moderate brightness speckle) should NOT be skipped."""
    rng = np.random.default_rng(7)
    scene = rng.integers(30, 80, size=(1000, 1000), dtype=np.uint8)  # speckle, mean ~55
    kept = list(iter_tiles(scene, tile_size=500, tile_overlap=0, skip_empty=True))
    assert len(kept) == 4  # p95 of ~80/255 = 0.31 >> 0.05 threshold


def test_iter_tiles_skip_empty_respects_custom_threshold():
    scene = np.full((1000, 1000), 20, dtype=np.uint8)  # uniform dim tile, p95 = 20/255 ≈ 0.078
    # Default threshold 0.05 → kept.
    assert _count_tiles(scene, tile_size=500, tile_overlap=0, skip_empty=True) == 4
    # Threshold 0.1 → all skipped.
    assert _count_tiles(
        scene, tile_size=500, tile_overlap=0, skip_empty=True, empty_threshold=0.1,
    ) == 0


# ---------------------------------------------------------------------------
# Iteration invariants
# ---------------------------------------------------------------------------


def test_iter_tiles_deterministic_row_major_order():
    """Tiles come out in row-major order: all cols of row 0, then all cols of row 1, etc."""
    scene = np.zeros((1500, 1500), dtype=np.uint8)
    tiles = list(iter_tiles(scene, tile_size=500, tile_overlap=0))
    origins = [t.origin for t in tiles]
    assert origins == sorted(origins, key=lambda rc: (rc[0], rc[1]))


def test_iter_tiles_index_starts_at_zero_and_is_monotonic():
    scene = np.zeros((1500, 1500), dtype=np.uint8)
    tiles = list(iter_tiles(scene, tile_size=500, tile_overlap=0))
    indices = [t.index for t in tiles]
    assert indices == list(range(len(tiles)))


def test_iter_tiles_index_counts_only_yielded_tiles_with_skip_empty():
    """When skip_empty drops tiles, remaining tiles still have contiguous indices 0..N-1."""
    scene = np.zeros((1000, 1000), dtype=np.uint8)
    scene[0:500, 0:500] = 200
    kept = list(iter_tiles(scene, tile_size=500, tile_overlap=0, skip_empty=True))
    assert [t.index for t in kept] == [0]


def test_iter_tiles_origins_are_unique():
    scene = np.zeros((1800, 1800), dtype=np.uint8)
    tiles = list(iter_tiles(scene, tile_size=500, tile_overlap=50))
    origins = [t.origin for t in tiles]
    assert len(origins) == len(set(origins))


def test_iter_tiles_tile_array_and_tile_shape_agree():
    scene = np.zeros((2000, 2000), dtype=np.uint8)
    for t in iter_tiles(scene, tile_size=500, tile_overlap=50):
        assert t.tile.shape == t.tile_shape


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_iter_tiles_rejects_non_2d_scene():
    with pytest.raises(ValueError, match="at least 2D"):
        list(iter_tiles(np.zeros(100, dtype=np.uint8), tile_size=50))


def test_iter_tiles_rejects_overlap_ge_tile_size():
    scene = np.zeros((100, 100), dtype=np.uint8)
    with pytest.raises(ValueError, match="tile_overlap"):
        list(iter_tiles(scene, tile_size=50, tile_overlap=50))


def test_iter_tiles_rejects_non_positive_tile_size():
    scene = np.zeros((100, 100), dtype=np.uint8)
    with pytest.raises(ValueError, match="tile_size"):
        list(iter_tiles(scene, tile_size=0))


def test_iter_tiles_3d_scene_ignores_channel_axis_for_geometry():
    """HxWx3 scenes are tiled by their first two axes; channel axis passes through."""
    scene = np.zeros((1000, 1000, 3), dtype=np.uint8)
    tiles = list(iter_tiles(scene, tile_size=500, tile_overlap=0))
    assert len(tiles) == 4
    for t in tiles:
        assert t.tile.shape == (500, 500, 3)
        assert t.tile_shape == (500, 500)
