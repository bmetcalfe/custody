"""Annular-window percentile filter for OS-CFAR (CPU).

For each pixel in a 2D image, computes the k-th percentile of pixels in an
annular (ring-shaped) neighborhood around it.  The annulus is defined by:

    inner half-width = guard      (excluded from percentile window)
    outer half-width = guard + reference

Pixels in the guard region around the center are excluded because they may
contain the target being tested.  Pixels in the reference ring around the
guard are used for the percentile estimate.  This is the correct OS-CFAR
formulation per Rohling (1983), resolving the "annular-window
implementation deferred" caveat in ADR-0014.

Implementation
--------------
numba ``@njit(parallel=True)`` with row-wise ``prange``.  Per-row buffer
is allocated inside the parallel loop so each thread has its own.  Reflect
boundary handling matches scipy's default (``mode='reflect'``).

First call JIT-compiles (~10-30s).  ``cache=True`` persists the compiled
kernel on disk so subsequent process invocations skip compilation.

The GPU variant lives in :mod:`custody.detection.annular_percentile_gpu`
so that ``import custody.detection`` (and therefore the VLM-backend pipeline)
remains importable on machines without ``numba-cuda`` installed.
"""
from __future__ import annotations

import numba
import numpy as np


@numba.njit(parallel=True, cache=True)
def _annular_percentile_2d(
    img: np.ndarray,
    guard: int,
    reference: int,
    percentile: float,
    out: np.ndarray,
) -> None:
    outer = guard + reference
    H, W = img.shape
    n_ring = (2 * outer + 1) * (2 * outer + 1) - (2 * guard + 1) * (2 * guard + 1)

    for r in numba.prange(H):
        buffer = np.empty(n_ring, dtype=img.dtype)
        for c in range(W):
            idx = 0
            for dy in range(-outer, outer + 1):
                ady = dy if dy >= 0 else -dy
                for dx in range(-outer, outer + 1):
                    adx = dx if dx >= 0 else -dx
                    # Skip the guard square (inner region)
                    if ady <= guard and adx <= guard:
                        continue
                    # Reflect boundary handling
                    yy = r + dy
                    if yy < 0:
                        yy = -yy - 1
                    elif yy >= H:
                        yy = 2 * H - yy - 1
                    xx = c + dx
                    if xx < 0:
                        xx = -xx - 1
                    elif xx >= W:
                        xx = 2 * W - xx - 1
                    buffer[idx] = img[yy, xx]
                    idx += 1
            buffer[:idx].sort()
            # Linear-interpolated percentile matching numpy default.
            pct_float = percentile / 100.0 * (idx - 1)
            pct_lo = int(pct_float)
            pct_hi = pct_lo + 1
            if pct_hi >= idx:
                pct_hi = idx - 1
            frac = pct_float - pct_lo
            out[r, c] = buffer[pct_lo] * (1.0 - frac) + buffer[pct_hi] * frac


def annular_percentile_filter(
    img: np.ndarray,
    guard: int,
    reference: int,
    percentile: float,
) -> np.ndarray:
    """Compute per-pixel percentile over an annular window (CPU).

    Args:
        img: 2D float32 or float64 array.
        guard: inner ring half-width (excluded from percentile).
        reference: outer ring half-width beyond the guard.
        percentile: float in [0, 100].
    """
    if img.ndim != 2:
        raise ValueError(f"annular_percentile_filter expects 2D input; got shape {img.shape}")
    if img.dtype not in (np.float32, np.float64):
        raise ValueError(
            f"annular_percentile_filter requires float32/float64; got {img.dtype}"
        )
    if guard < 0 or reference < 1:
        raise ValueError(f"need guard >= 0 and reference >= 1; got guard={guard}, reference={reference}")
    if not 0.0 <= percentile <= 100.0:
        raise ValueError(f"percentile must be in [0, 100]; got {percentile}")
    out = np.empty_like(img)
    _annular_percentile_2d(img, guard, reference, float(percentile), out)
    return out
