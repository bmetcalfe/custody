"""GPU (CUDA) annular-window percentile filter for OS-CFAR.

Split out from :mod:`custody.detection.annular_percentile` so that importing
``custody.detection`` does not pull ``numba.cuda`` — the CUDA package is
heavy (~450 modules, ~hundreds of MB of native bindings) and fails to import
entirely on machines without a CUDA runtime, which would otherwise break
unrelated functionality such as the VLM-backend pipeline.

Import contract:
- ``custody.detection.__init__`` does NOT import this module.
- ``custody.detection.annular_percentile`` does NOT import this module.
- Callers that want the GPU path import this module explicitly and must
  have ``numba-cuda`` plus a CUDA runtime installed.  A missing runtime
  surfaces as an informative ImportError at module load.

One CUDA thread per output pixel.  Each thread gathers its annulus values
into a row of a device-resident workspace array (sized for the number of
*concurrent* threads, not total threads queued), then performs a partial
selection sort to land the k-th order statistic.

Selection-sort-from-front: O((pct_hi+1) * n_ring) per pixel.  Slow but
straightforward and correct; baseline against which more sophisticated
selection algorithms (quickselect, bitonic-partial-sort) can be measured.
"""
from __future__ import annotations

import numpy as np

try:
    from numba import cuda
except ImportError as e:  # pragma: no cover — exercised only on CUDA-free installs
    raise ImportError(
        "custody.detection.annular_percentile_gpu requires numba-cuda and a "
        "CUDA runtime. Install the numba-cuda package and the NVIDIA CUDA "
        "runtime (cudart) to use the GPU variant, or call "
        "custody.detection.annular_percentile.annular_percentile_filter (CPU) "
        f"instead. Original import error: {e}"
    ) from e

from custody.detection.annular_percentile import annular_percentile_filter


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
