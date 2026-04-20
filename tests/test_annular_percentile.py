"""Tests for custody.detection.annular_percentile.

Tests 2 and 3 are the correctness gates — they validate that the annular
window is actually annular (guard excluded) and returns the right
percentile on a scene with known answer.
"""
from __future__ import annotations

import time

import numpy as np
import pytest
from scipy.ndimage import percentile_filter
from scipy.stats import spearmanr

from custody.detection.annular_percentile import annular_percentile_filter

try:
    from custody.detection.annular_percentile_gpu import annular_percentile_filter_gpu
    import custody.detection.annular_percentile_gpu as _ap_gpu_mod
    _cuda_available = _ap_gpu_mod.cuda.is_available()
except ImportError:
    annular_percentile_filter_gpu = None  # type: ignore[assignment]
    _ap_gpu_mod = None  # type: ignore[assignment]
    _cuda_available = False

_requires_cuda = pytest.mark.skipif(not _cuda_available, reason="CUDA not available")


def test_annular_correlates_with_rectangular_on_random_scene():
    """Annular and rectangular percentile filters should be rank-correlated
    on a random scene (both track the local distribution).  Not pixel-identical
    because annular excludes the guard.
    """
    rng = np.random.default_rng(7)
    img = rng.standard_normal((256, 256)).astype(np.float32)
    guard, reference = 10, 20
    a = annular_percentile_filter(img, guard=guard, reference=reference, percentile=75.0)
    r = percentile_filter(img, percentile=75.0, size=2 * (guard + reference) + 1, mode="reflect")
    # Trim boundary to avoid reflect-mode corner effects
    margin = guard + reference + 1
    a_i = a[margin:-margin, margin:-margin].ravel()
    r_i = r[margin:-margin, margin:-margin].ravel()
    rho, _ = spearmanr(a_i, r_i)
    # Sanity threshold: rectangular window includes the guard region (here
    # 441 / 3721 ~= 12% of cells), annular excludes it, so they disagree
    # on roughly that many pixels per window.  A rank correlation well
    # above 0.8 confirms both track the local distribution.
    assert rho > 0.85, f"annular vs rectangular rank correlation {rho:.3f} below 0.85"


def test_annular_uniform_bg_with_single_bright_point_has_75pct_at_bg():
    """Background=10 uniformly + single bright point at center.  For pixels
    far from center, the annulus contains only background.  For pixels near
    the center, the bright point is a single outlier in a large annulus,
    well below the 75th percentile.  So output should be ~10 everywhere.
    """
    img = np.full((128, 128), 10.0, dtype=np.float32)
    img[64, 64] = 1000.0
    out = annular_percentile_filter(img, guard=5, reference=10, percentile=75.0)
    # Every output pixel should equal 10 (the background uniform value) —
    # the single bright outlier in a ~420-pixel annulus can't drag the
    # 75th percentile off the background.
    assert np.allclose(out, 10.0, atol=1e-4), (
        f"expected uniform output at bg=10; got min={out.min()} max={out.max()}"
    )


def test_annular_excludes_guard_region():
    """Guard region has value 100; reference ring has value 1; center has
    value 0.  At the center pixel, if the guard were INCLUDED in the
    percentile window the output would be elevated (toward 100).  With
    correct annular exclusion, only the reference ring (value 1) counts,
    so the 75th percentile should be 1.
    """
    H = W = 41
    cy, cx = H // 2, W // 2
    guard, reference = 5, 10
    img = np.zeros((H, W), dtype=np.float32)
    # Reference ring value
    for dy in range(-(guard + reference), guard + reference + 1):
        for dx in range(-(guard + reference), guard + reference + 1):
            if max(abs(dy), abs(dx)) > guard:
                img[cy + dy, cx + dx] = 1.0
    # Guard region value — larger than reference
    for dy in range(-guard, guard + 1):
        for dx in range(-guard, guard + 1):
            img[cy + dy, cx + dx] = 100.0

    out = annular_percentile_filter(img, guard=guard, reference=reference, percentile=75.0)
    assert out[cy, cx] == pytest.approx(1.0, abs=1e-4), (
        f"guard region leaked into percentile: center output = {out[cy, cx]} "
        "(expected 1.0 from reference ring only)"
    )


def test_annular_percentile_monotonic_in_percentile_arg():
    rng = np.random.default_rng(19)
    img = rng.standard_normal((128, 128)).astype(np.float32)
    p25 = annular_percentile_filter(img, guard=5, reference=10, percentile=25.0)
    p50 = annular_percentile_filter(img, guard=5, reference=10, percentile=50.0)
    p75 = annular_percentile_filter(img, guard=5, reference=10, percentile=75.0)
    tol = 1e-5
    assert np.all(p25 <= p50 + tol), "p25 > p50 in some pixel"
    assert np.all(p50 <= p75 + tol), "p50 > p75 in some pixel"


def test_annular_dtype_preservation():
    rng = np.random.default_rng(23)
    img32 = rng.standard_normal((64, 64)).astype(np.float32)
    img64 = rng.standard_normal((64, 64)).astype(np.float64)
    out32 = annular_percentile_filter(img32, guard=3, reference=5, percentile=75.0)
    out64 = annular_percentile_filter(img64, guard=3, reference=5, percentile=75.0)
    assert out32.dtype == np.float32
    assert out64.dtype == np.float64


# ---------------------------------------------------------------------------
# GPU (CUDA) variant — same correctness gates via the GPU function
# ---------------------------------------------------------------------------


@_requires_cuda
def test_annular_gpu_matches_cpu_on_random_scene():
    """GPU and CPU implementations should agree element-wise on the interior
    to float32 precision.  Boundary pixels may differ in the last ULP due to
    sort instability — compare the core with generous tolerance.
    """
    rng = np.random.default_rng(41)
    img = rng.standard_normal((512, 512)).astype(np.float32)
    guard, reference = 10, 20
    cpu_out = annular_percentile_filter(img, guard=guard, reference=reference, percentile=75.0)
    gpu_out = annular_percentile_filter_gpu(img, guard=guard, reference=reference, percentile=75.0)
    assert gpu_out.shape == cpu_out.shape
    assert gpu_out.dtype == cpu_out.dtype
    np.testing.assert_allclose(gpu_out, cpu_out, atol=1e-4, rtol=1e-4)


@_requires_cuda
def test_annular_gpu_uniform_bg_with_single_bright_point():
    img = np.full((128, 128), 10.0, dtype=np.float32)
    img[64, 64] = 1000.0
    out = annular_percentile_filter_gpu(img, guard=5, reference=10, percentile=75.0)
    assert np.allclose(out, 10.0, atol=1e-4), (
        f"expected uniform 10.0 output; got min={out.min()} max={out.max()}"
    )


@_requires_cuda
def test_annular_gpu_excludes_guard_region():
    H = W = 41
    cy, cx = H // 2, W // 2
    guard, reference = 5, 10
    img = np.zeros((H, W), dtype=np.float32)
    for dy in range(-(guard + reference), guard + reference + 1):
        for dx in range(-(guard + reference), guard + reference + 1):
            if max(abs(dy), abs(dx)) > guard:
                img[cy + dy, cx + dx] = 1.0
    for dy in range(-guard, guard + 1):
        for dx in range(-guard, guard + 1):
            img[cy + dy, cx + dx] = 100.0
    out = annular_percentile_filter_gpu(img, guard=guard, reference=reference, percentile=75.0)
    assert out[cy, cx] == pytest.approx(1.0, abs=1e-4), (
        f"GPU guard exclusion failed: center output = {out[cy, cx]}"
    )


@_requires_cuda
def test_annular_gpu_dtype_preservation():
    rng = np.random.default_rng(47)
    img32 = rng.standard_normal((64, 64)).astype(np.float32)
    out32 = annular_percentile_filter_gpu(img32, guard=3, reference=5, percentile=75.0)
    assert out32.dtype == np.float32


@pytest.mark.skipif(_ap_gpu_mod is None, reason="numba.cuda not installed; nothing to fall back from")
def test_annular_gpu_fallback_when_cuda_unavailable(monkeypatch):
    """If cuda.is_available() is False, the GPU wrapper should fall through
    to the CPU implementation and still produce correct output.
    """
    def fake_is_available():
        return False
    monkeypatch.setattr(_ap_gpu_mod.cuda, "is_available", fake_is_available)

    rng = np.random.default_rng(53)
    img = rng.standard_normal((128, 128)).astype(np.float32)
    cpu_out = annular_percentile_filter(img, guard=5, reference=10, percentile=75.0)
    wrapper_out = annular_percentile_filter_gpu(img, guard=5, reference=10, percentile=75.0)
    np.testing.assert_allclose(wrapper_out, cpu_out, atol=1e-5)


def test_annular_jit_cache_second_call_not_slower():
    """First call JIT-compiles (from scratch or from disk cache).  Second
    call with same signature hits in-memory cache and must not be slower.
    We don't assert a hard speedup ratio because on a warm disk cache the
    first call itself is already fast and the relative difference gets
    noisy; the semantic we care about is that cache=True doesn't hurt.
    """
    rng = np.random.default_rng(29)
    img = rng.standard_normal((128, 128)).astype(np.float32)
    t0 = time.perf_counter()
    _ = annular_percentile_filter(img, guard=5, reference=10, percentile=75.0)
    t1 = time.perf_counter() - t0
    t0 = time.perf_counter()
    _ = annular_percentile_filter(img, guard=5, reference=10, percentile=75.0)
    t2 = time.perf_counter() - t0
    assert t2 <= t1 + 0.05, f"second call materially slower than first: first={t1:.3f}s second={t2:.3f}s"
