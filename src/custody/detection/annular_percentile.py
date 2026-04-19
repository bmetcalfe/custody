"""Annular-window percentile filter for OS-CFAR.

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
"""
from __future__ import annotations

import numba
import numpy as np
from numba import cuda


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


# ---------------------------------------------------------------------------
# GPU (CUDA) variant — first-pass implementation
# ---------------------------------------------------------------------------
#
# One CUDA thread per output pixel.  Each thread gathers its annulus values
# into a row of a device-resident workspace array (sized for the number of
# *concurrent* threads, not total threads queued), then performs a partial
# selection sort to land the k-th order statistic.
#
# Selection-sort-from-front: O(pct_hi+1 * n_ring) per pixel.  Slow but
# straightforward and correct; baseline against which more sophisticated
# selection algorithms (quickselect, bitonic-partial-sort) can be measured.


@cuda.jit
def _annular_percentile_cuda_kernel(
    img, workspace, out, guard, reference, percentile_frac,
):
    r, c = cuda.grid(2)
    H = img.shape[0]
    W = img.shape[1]
    if r >= H or c >= W:
        return

    outer = guard + reference
    thread_id_flat = r * W + c
    ws_row = thread_id_flat % workspace.shape[0]

    idx = 0
    for dy in range(-outer, outer + 1):
        ady = dy if dy >= 0 else -dy
        for dx in range(-outer, outer + 1):
            adx = dx if dx >= 0 else -dx
            if ady <= guard and adx <= guard:
                continue
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
            workspace[ws_row, idx] = img[yy, xx]
            idx += 1

    pct_float = percentile_frac * (idx - 1)
    pct_lo = int(pct_float)
    pct_hi = pct_lo + 1
    if pct_hi >= idx:
        pct_hi = idx - 1
    frac = pct_float - pct_lo

    # Place the smallest (pct_hi+1) elements in sorted order at the front of
    # the workspace row.  After this loop, workspace[ws_row, pct_lo] and
    # workspace[ws_row, pct_hi] hold the two order statistics we need.
    limit = pct_hi + 1
    for target in range(limit):
        min_idx = target
        min_val = workspace[ws_row, target]
        for j in range(target + 1, idx):
            v = workspace[ws_row, j]
            if v < min_val:
                min_val = v
                min_idx = j
        if min_idx != target:
            workspace[ws_row, min_idx] = workspace[ws_row, target]
            workspace[ws_row, target] = min_val

    v_lo = workspace[ws_row, pct_lo]
    v_hi = workspace[ws_row, pct_hi]
    out[r, c] = v_lo * (1.0 - frac) + v_hi * frac


def annular_percentile_filter_gpu(
    img: np.ndarray,
    guard: int,
    reference: int,
    percentile: float,
    workspace_threads: int | None = None,
) -> np.ndarray:
    """GPU-backed annular percentile filter.  Falls back to CPU if CUDA unavailable.

    Args:
        img: 2D float32 array (float64 will be downcast for GPU path).
        guard, reference: annulus half-widths (see CPU variant).
        percentile: float in [0, 100].
        workspace_threads: size of the device workspace's first axis.  Each
            row is ``n_ring * 4`` bytes.  Default picks ``min(H*W, 262144)``,
            which is enough rows to cover the max-concurrent-thread capacity
            on current Blackwell-class GPUs without collisions; collisions
            cause corruption when two concurrent threads hash to the same
            row, hence the safety floor.
    """
    if img.ndim != 2:
        raise ValueError(f"expected 2D input; got shape {img.shape}")
    if not 0.0 <= percentile <= 100.0:
        raise ValueError(f"percentile must be in [0, 100]; got {percentile}")

    if not cuda.is_available():
        return annular_percentile_filter(img, guard, reference, percentile)

    outer = guard + reference
    n_ring = (2 * outer + 1) ** 2 - (2 * guard + 1) ** 2
    percentile_frac = float(percentile) / 100.0

    img_f32 = np.ascontiguousarray(img, dtype=np.float32)
    H, W = img_f32.shape

    if workspace_threads is None:
        workspace_threads = min(H * W, 262144)

    d_img = cuda.to_device(img_f32)
    d_out = cuda.device_array_like(img_f32)
    d_ws = cuda.device_array((workspace_threads, n_ring), dtype=np.float32)

    tpb_y, tpb_x = 16, 16
    blocks_y = (H + tpb_y - 1) // tpb_y
    blocks_x = (W + tpb_x - 1) // tpb_x

    _annular_percentile_cuda_kernel[(blocks_y, blocks_x), (tpb_y, tpb_x)](
        d_img, d_ws, d_out, guard, reference, percentile_frac,
    )
    cuda.synchronize()
    return d_out.copy_to_host()
