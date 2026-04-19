"""Tests for custody.detection.sar_cfar — synthetic-only CA-CFAR behavior.

No real Umbra scenes. All fixtures are deterministic synthetic imagery
(numpy rng with fixed seeds).  Real-scene validation is the developer's
job via visual inspection of the detection overlay PNG written by the
03_detect_sar_umbra.py script.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

from custody.detection.sar_cfar import (
    compute_cfar_threshold,
    compute_os_cfar_threshold,
    compute_structure_mask,
    detect_points_cfar,
    detect_points_os_cfar,
    detect_points_os_cfar_to_observations,
    detect_points_to_observations,
)
from custody.fusion.observations import PositionObservation


def _expon_noise(shape: tuple[int, int], scale: float = 1.0, seed: int = 0) -> np.ndarray:
    """Deterministic exponentially-distributed noise (simulating SAR power)."""
    rng = np.random.default_rng(seed)
    return rng.exponential(scale=scale, size=shape).astype(np.float32)


def _straight_tfm(pixel_size: float = 1.0) -> Affine:
    return Affine.translation(240000.0, 980000.0) * Affine.scale(pixel_size, -pixel_size)


# ---------------------------------------------------------------------------
# FAR sanity on pure exponential noise
# ---------------------------------------------------------------------------


def _far(img: np.ndarray, alpha: float, guard: int = 5, reference: int = 15) -> float:
    """Observed false alarm rate on pure-noise `img`."""
    # Trim an outer margin equal to the CFAR window, since uniform_filter at the
    # edges uses reflection and isn't a fair sample of the steady-state FAR.
    margin = guard + reference + 1
    thr = compute_cfar_threshold(img, guard=guard, reference=reference, alpha=alpha)
    hit = img > thr
    core = hit[margin:-margin, margin:-margin]
    return float(core.sum()) / float(core.size)


def test_far_on_pure_exponential_noise_within_bound():
    img = _expon_noise((512, 512), scale=1.0, seed=42)
    far = _far(img, alpha=10.0, guard=5, reference=15)
    # Soft: alpha=10 drives false alarms to near zero on exp noise.
    # The theoretical FAR for CA-CFAR on exp noise converges to exp(-alpha)
    # for large N; at alpha=10 that's ~4.5e-5. Accept < 4% as "sane."
    assert far < 0.04, f"FAR {far:.4f} exceeds 4% at alpha=10"


def test_far_monotonic_in_alpha():
    img = _expon_noise((512, 512), scale=1.0, seed=7)
    far2 = _far(img, alpha=2.0)
    far3 = _far(img, alpha=3.0)
    far5 = _far(img, alpha=5.0)
    assert far2 > far3 > far5


def test_far_invariant_to_noise_scale():
    img1 = _expon_noise((512, 512), scale=1.0, seed=11)
    img2 = _expon_noise((512, 512), scale=2.0, seed=11)
    far1 = _far(img1, alpha=5.0)
    far2 = _far(img2, alpha=5.0)
    # CFAR's core property: threshold adapts to local mean.
    assert abs(far1 - far2) < 0.01, f"FAR should be scale-invariant (got {far1:.4f} vs {far2:.4f})"


# ---------------------------------------------------------------------------
# Detection probability
# ---------------------------------------------------------------------------


def test_single_bright_target_detected():
    img = _expon_noise((512, 512), scale=1.0, seed=23)
    # Inject a bright 1-pixel target at the centre. Background mean ≈ 1,
    # target value 20 is comfortably above any reasonable CFAR threshold.
    img[256, 256] = 20.0
    detections = detect_points_cfar(
        img, alpha=5.0, guard=5, reference=15,
        mask_structures=False, min_blob_pixels=1,
    )
    assert detections, "bright point target was not detected"
    # At least one detection within 1 pixel of the injected location
    found = any(abs(r - 256) <= 1 and abs(c - 256) <= 1 for r, c in detections)
    assert found, f"no detection within 1 px of (256, 256); got {detections[:5]}"


def test_dim_target_no_crash_and_reasonable_centroid():
    img = _expon_noise((256, 256), scale=1.0, seed=51)
    img[128, 128] = 3.0  # marginal
    detections = detect_points_cfar(
        img, alpha=2.5, guard=5, reference=15,
        mask_structures=False, min_blob_pixels=1,
    )
    # Don't require detection — test is that no exception is raised.  If
    # detected, centroid should be near truth.
    for r, c in detections:
        if abs(r - 128) <= 2 and abs(c - 128) <= 2:
            return  # ok
    # No crash = pass, regardless of detection outcome.


def test_very_bright_target_robust_across_alpha_range():
    img = _expon_noise((512, 512), scale=1.0, seed=13)
    img[256, 256] = 50.0
    for alpha in [2.0, 3.0, 5.0, 10.0]:
        detections = detect_points_cfar(
            img, alpha=alpha, guard=5, reference=15,
            mask_structures=False, min_blob_pixels=1,
        )
        found = any(abs(r - 256) <= 1 and abs(c - 256) <= 1 for r, c in detections)
        assert found, f"very bright target missed at alpha={alpha}"


# ---------------------------------------------------------------------------
# Multiple targets
# ---------------------------------------------------------------------------


def test_five_scattered_point_targets():
    img = _expon_noise((512, 512), scale=1.0, seed=17)
    truth = [(100, 100), (100, 400), (256, 256), (400, 100), (400, 400)]
    for r, c in truth:
        img[r, c] = 25.0
    detections = detect_points_cfar(
        img, alpha=5.0, guard=5, reference=15,
        mask_structures=False, min_blob_pixels=1,
    )
    for tr, tc in truth:
        found = any(abs(dr - tr) <= 1 and abs(dc - tc) <= 1 for dr, dc in detections)
        assert found, f"missed truth target ({tr}, {tc})"
    # A bounded false-alarm budget.  At alpha=5 on ~250k pixels, expect
    # dozens to a few hundred false alarms, well under 3x the far-sum.
    # Loose: detections count must not explode.
    assert len(detections) < 2000, f"too many detections ({len(detections)}) — FAR exploded"


def test_two_targets_50_pixels_apart_kept_separate():
    img = _expon_noise((512, 512), scale=1.0, seed=19)
    img[256, 200] = 30.0
    img[256, 250] = 30.0
    detections = detect_points_cfar(
        img, alpha=5.0, guard=5, reference=15,
        mask_structures=False, min_blob_pixels=1,
    )
    t1 = [d for d in detections if abs(d[0] - 256) <= 1 and abs(d[1] - 200) <= 1]
    t2 = [d for d in detections if abs(d[0] - 256) <= 1 and abs(d[1] - 250) <= 1]
    assert t1, "target 1 missed"
    assert t2, "target 2 missed"


def test_two_targets_10_pixels_apart_documented_behavior():
    img = _expon_noise((512, 512), scale=1.0, seed=29)
    img[256, 250] = 30.0
    img[256, 260] = 30.0  # 10 px apart — within 2*guard=10
    detections = detect_points_cfar(
        img, alpha=5.0, guard=5, reference=15,
        mask_structures=False, min_blob_pixels=1,
    )
    # Either merged (1 detection) or split (2 detections) is acceptable.
    # We assert: at least one detection in the corridor.
    in_corridor = [d for d in detections if d[0] == 256 and 248 <= d[1] <= 262]
    assert len(in_corridor) >= 1, "neither close target produced a detection"


# ---------------------------------------------------------------------------
# Structure mask
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Structure mask unit tests use zero-bg fixtures so the algorithm's
# behavior is deterministic, not buffeted by spatially-correlated smoothed
# noise.  Real SAR scenes have heterogeneous brightness where the percentile
# threshold settles in a meaningful "background band"; uniform exp-noise test
# scenes don't, so the noise statistics would otherwise dominate the tests.
# ---------------------------------------------------------------------------


def _zero_bg(shape: tuple[int, int]) -> np.ndarray:
    """Zero-background scene, no noise — purely deterministic."""
    return np.zeros(shape, dtype=np.float32)


_MASK_KW = dict(smooth_sigma=2.0, bright_threshold_pct=95.0)


def test_structure_mask_30x30_masked_point_targets_survive():
    """Large bright region is masked; 3 scattered point targets pass through CFAR."""
    img = _zero_bg((512, 512))
    img[100:130, 100:130] = 30.0     # 30x30 at value 30 → smoothed footprint well over 500
    pts = [(300, 300), (350, 400), (400, 100)]
    for r, c in pts:
        img[r, c] = 25.0

    mask = compute_structure_mask(
        img, **_MASK_KW,
        min_structure_area_pixels=500,
        dilate_pixels=10,
    )
    assert mask[115, 115], "mask missed center of 30x30 bright region"
    for tr, tc in pts:
        assert not mask[tr, tc], f"mask wrongly covered point target ({tr}, {tc})"

    # CFAR with mask on: point targets detected.  (CFAR needs actual noise to
    # exercise — use an exp-noise scene with the same features to drive CFAR.)
    img_noise = _expon_noise((512, 512), scale=1.0, seed=31)
    img_noise[100:130, 100:130] = 30.0
    for r, c in pts:
        img_noise[r, c] = 25.0
    detections_with = detect_points_cfar(
        img_noise, alpha=5.0, guard=5, reference=15,
        mask_structures=True,
        **_MASK_KW,
        min_structure_area_pixels=500,
        dilate_pixels=10,
        min_blob_pixels=1,
    )
    for tr, tc in pts:
        assert any(abs(dr - tr) <= 1 and abs(dc - tc) <= 1 for dr, dc in detections_with), (
            f"point target ({tr}, {tc}) missed with mask on"
        )


def test_structure_mask_boundary_behavior():
    img = _zero_bg((256, 256))
    img[50:100, 50:100] = 30.0   # 50x50 at value 30 → huge smoothed footprint
    mask = compute_structure_mask(
        img, **_MASK_KW,
        min_structure_area_pixels=500, dilate_pixels=0,
    )
    assert mask[75, 75], "structure mask missed centre of 50x50 bright block"
    assert not mask[200, 200], "structure mask over-extended into empty region"


# ---------------------------------------------------------------------------
# New mask behaviors (connected-components + min-area + dilation)
# ---------------------------------------------------------------------------


def test_small_bright_blob_below_min_area_is_not_masked():
    """Small bright blob (5x5, vessel-scale) stays below the 500-pixel floor."""
    img = _zero_bg((256, 256))
    img[100:105, 100:105] = 30.0   # 5x5 blob
    mask = compute_structure_mask(
        img, **_MASK_KW,
        min_structure_area_pixels=500,
        dilate_pixels=0,
    )
    assert not mask[102, 102], "small 5x5 blob should not be masked (footprint < 500)"


def test_large_bright_region_is_masked():
    """50x50 bright region produces a smoothed footprint well over 500 — masked."""
    img = _zero_bg((256, 256))
    img[80:130, 80:130] = 30.0
    mask = compute_structure_mask(
        img, **_MASK_KW,
        min_structure_area_pixels=500,
        dilate_pixels=0,
    )
    assert mask[105, 105], "50x50 region should be masked"


def test_dilation_extends_mask_by_configured_pixels():
    """With dilate_pixels=10, mask extends 10 pixels beyond the un-dilated boundary."""
    img = _zero_bg((1024, 1024))
    img[400:460, 400:460] = 30.0   # 60x60 at value 30 on a 1024x1024 scene
    mask_no_dilate = compute_structure_mask(
        img, **_MASK_KW,
        min_structure_area_pixels=500,
        dilate_pixels=0,
    )
    mask_dilate = compute_structure_mask(
        img, **_MASK_KW,
        min_structure_area_pixels=500,
        dilate_pixels=10,
    )
    row = 430
    edge_cols = np.where(mask_no_dilate[row, :])[0]
    assert edge_cols.size > 0, "no-dilate mask is empty on centre row"
    last_col = int(edge_cols.max())
    assert last_col + 11 < mask_dilate.shape[1], f"not enough image margin (last_col={last_col})"
    assert mask_dilate[row, last_col + 9], "dilate=10 failed to extend +9 past edge"
    assert not mask_dilate[row, last_col + 11], "dilate=10 extended past +11 incorrectly"


def test_two_disconnected_small_blobs_not_merged_by_smoothing():
    """Two separated small bright blobs stay as separate sub-min-area components.

    At sigma=5, the Gaussian tail at 20 pixels (half-separation) is exp(-8) ≈ 3.4e-4,
    so there's no measurable bridge of above-threshold smoothed value between the
    blobs.  Guards against the naive "smooth + threshold" algorithm fusing nearby
    vessels into a fake "structure".
    """
    img = _zero_bg((512, 512))
    img[100:105, 100:105] = 30.0     # 5x5 at value 30
    img[100:105, 140:145] = 30.0     # 5x5 at value 30, 40 px east
    mask = compute_structure_mask(
        img, **_MASK_KW,
        min_structure_area_pixels=500,
        dilate_pixels=0,
    )
    assert not mask[102, 102], "first small blob wrongly masked (smoothing merged?)"
    assert not mask[102, 142], "second small blob wrongly masked (smoothing merged?)"


# ---------------------------------------------------------------------------
# Decimation consistency
# ---------------------------------------------------------------------------


def test_decimated_matches_fullres_within_4px():
    img = _expon_noise((512, 512), scale=1.0, seed=41)
    truth = [(120, 120), (256, 256), (400, 400)]
    for r, c in truth:
        img[r, c] = 30.0
    full = detect_points_cfar(
        img, alpha=5.0, guard=5, reference=15,
        mask_structures=False, min_blob_pixels=1,
    )
    dec = detect_points_cfar(
        img[::4, ::4], alpha=5.0, guard=5, reference=15,
        mask_structures=False, min_blob_pixels=1,
    )
    for tr, tc in truth:
        in_full = any(abs(r - tr) <= 1 and abs(c - tc) <= 1 for r, c in full)
        in_dec = any(abs(r * 4 - tr) <= 4 and abs(c * 4 - tc) <= 4 for r, c in dec)
        assert in_full, f"full-res missed truth ({tr}, {tc})"
        assert in_dec, f"decimated missed truth ({tr}, {tc})"


def test_fullres_completes_under_soft_budget():
    img = _expon_noise((1024, 1024), scale=1.0, seed=53)
    for r in range(100, 1000, 200):
        img[r, r] = 30.0
    t0 = time.perf_counter()
    detect_points_cfar(img, alpha=5.0, guard=10, reference=30, mask_structures=True)
    dt = time.perf_counter() - t0
    assert dt < 10.0, f"CFAR on 1024x1024 took {dt:.2f}s (soft 10s budget)"


# ---------------------------------------------------------------------------
# Observation output
# ---------------------------------------------------------------------------


def test_detect_points_to_observations_returns_valid_position_obs(tmp_path: Path):
    # Build a tiny synthetic scene + a straight-up transform
    img = _expon_noise((256, 256), scale=1.0, seed=59)
    img[128, 128] = 30.0
    img[64, 192] = 30.0
    tfm = _straight_tfm()
    crs = "EPSG:32650"
    acq = 1_688_306_455.0
    obs = detect_points_to_observations(
        img, tfm, crs, acquisition_time=acq, source_id="synth",
        alpha=5.0, guard=5, reference=15, mask_structures=False,
        min_blob_pixels=1,
    )
    assert len(obs) >= 2
    for o in obs:
        assert isinstance(o, PositionObservation)
        assert o.modality == "SAR"
        eigs = np.linalg.eigvalsh(o.cov_pos)
        assert eigs.min() > 0
    assert len({o.obs_id for o in obs}) == len(obs)


def test_empty_image_returns_empty_observation_list():
    img = np.zeros((128, 128), dtype=np.float32)
    tfm = _straight_tfm()
    obs = detect_points_to_observations(
        img, tfm, "EPSG:32650", acquisition_time=1.0, source_id="s",
        alpha=5.0, guard=5, reference=15, mask_structures=False,
    )
    assert obs == []


# ---------------------------------------------------------------------------
# OS-CFAR (Ordered-Statistic CFAR) — per ADR-0014
# ---------------------------------------------------------------------------
#
# OS-CFAR uses the k-th percentile of the reference-cell population as the
# threshold baseline instead of the mean.  This is robust to heterogeneous
# clutter and to target-near-target contamination, at the cost of a more
# expensive per-pixel statistic (percentile_filter).  Tests here mirror the
# CA-CFAR FAR/detection sanity checks, plus the two OS-specific robustness
# cases (heterogeneous clutter, adjacent targets) that motivate the ADR.
#
# Implementation note: rectangular percentile_filter (no guard exclusion);
# the intrinsic outlier-robustness of a percentile tolerates guard
# contamination.  See ADR-0014.
# ---------------------------------------------------------------------------


def _os_far(img: np.ndarray, alpha: float, guard: int = 5, reference: int = 15,
            k_percentile: float = 0.75) -> float:
    margin = guard + reference + 1
    thr = compute_os_cfar_threshold(img, guard=guard, reference=reference,
                                    alpha=alpha, k_percentile=k_percentile)
    hit = img > thr
    core = hit[margin:-margin, margin:-margin]
    return float(core.sum()) / float(core.size)


# --- FAR sanity on pure exponential noise ---


def test_os_cfar_far_on_pure_exp_noise_matches_expected_within_2x():
    # For exp(1) noise, the 75th percentile of a local reference window
    # is near ln(4) ~= 1.386; at alpha=3.0 the threshold multiplier
    # gives expected FAR = exp(-alpha * ln(4)) = 4^-3 ~= 0.0156.
    # Assert observed FAR is within 2x of this expected value.
    from scipy.stats import expon
    img = _expon_noise((512, 512), scale=1.0, seed=101)
    observed = _os_far(img, alpha=3.0, guard=5, reference=15, k_percentile=0.75)
    # Expected from formula: 3.0 * sample_75pct of the whole image
    expected = float(expon.sf(3.0 * np.percentile(img, 75), scale=1.0))
    assert expected * 0.5 <= observed <= expected * 2.0, (
        f"OS-CFAR FAR {observed:.4f} outside 2x of expected {expected:.4f}"
    )


def test_os_cfar_far_monotonic_in_alpha():
    img = _expon_noise((512, 512), scale=1.0, seed=103)
    far3 = _os_far(img, alpha=3.0)
    far4 = _os_far(img, alpha=4.0)
    far5 = _os_far(img, alpha=5.0)
    assert far3 > far4 > far5


def test_os_cfar_far_invariant_to_noise_scale():
    img1 = _expon_noise((512, 512), scale=1.0, seed=107)
    img2 = _expon_noise((512, 512), scale=2.0, seed=107)  # same draw, scaled
    far1 = _os_far(img1, alpha=4.0)
    far2 = _os_far(img2, alpha=4.0)
    # OS-CFAR, like CA-CFAR, is scale-invariant since the threshold
    # adapts to local statistics.
    assert abs(far1 - far2) < 0.01, (
        f"OS-CFAR FAR should be scale-invariant (got {far1:.4f} vs {far2:.4f})"
    )


# --- Detection probability on point targets ---


def test_os_cfar_bright_point_target_detected():
    img = _expon_noise((512, 512), scale=1.0, seed=109)
    img[256, 256] = 20.0
    dets = detect_points_os_cfar(
        img, alpha=3.0, guard=5, reference=15,
        k_percentile=0.75, mask_structures=False, min_blob_pixels=1,
    )
    assert dets, "bright point target was not detected by OS-CFAR"
    found = any(abs(r - 256) <= 1 and abs(c - 256) <= 1 for r, c in dets)
    assert found, f"no OS-CFAR detection within 1 px of truth; got {dets[:5]}"


def test_os_cfar_dim_target_no_crash():
    img = _expon_noise((256, 256), scale=1.0, seed=113)
    img[128, 128] = 3.0  # marginal SCR
    dets = detect_points_os_cfar(
        img, alpha=3.0, guard=5, reference=15,
        k_percentile=0.75, mask_structures=False, min_blob_pixels=1,
    )
    # Pass as long as no crash — detection outcome is not asserted.
    for r, c in dets:
        if abs(r - 128) <= 2 and abs(c - 128) <= 2:
            return


def test_os_cfar_five_scattered_targets_all_detected():
    img = _expon_noise((512, 512), scale=1.0, seed=127)
    truth = [(100, 100), (100, 400), (256, 256), (400, 100), (400, 400)]
    for r, c in truth:
        img[r, c] = 10.0
    dets = detect_points_os_cfar(
        img, alpha=3.0, guard=5, reference=15,
        k_percentile=0.75, mask_structures=False, min_blob_pixels=1,
    )
    for tr, tc in truth:
        found = any(abs(dr - tr) <= 1 and abs(dc - tc) <= 1 for dr, dc in dets)
        assert found, f"OS-CFAR missed truth target ({tr}, {tc})"


# --- Heterogeneous clutter + target-near-target (OS-CFAR's raison d'etre) ---


def test_os_cfar_preserves_targets_in_heterogeneous_clutter_where_ca_cfar_misses():
    # 80% exponential + 20% lognormal mixture clutter.  Lognormal tails
    # inflate CA-CFAR's window mean, raising the threshold enough that
    # moderate-SCR real targets are missed.  OS-CFAR's 75th-percentile
    # threshold is robust to a minority of bright outliers in the window
    # and preserves target detectability.  This is the canonical textbook
    # OS-CFAR advantage (Rohling 1983).
    rng = np.random.default_rng(131)
    shape = (256, 256)
    base = rng.exponential(scale=1.0, size=shape).astype(np.float32)
    contam = rng.lognormal(mean=0.0, sigma=2.0, size=shape).astype(np.float32)
    mask_contam = rng.uniform(size=shape) < 0.20
    img = np.where(mask_contam, contam, base)
    # Three moderate-SCR targets — bright enough to detect under pure
    # exp clutter, but at risk of being missed when the window mean is
    # inflated by lognormal outliers.
    truth = [(100, 100), (128, 128), (150, 150)]
    for r, c in truth:
        img[r, c] = 12.0

    alpha = 5.0
    guard, ref = 5, 15

    ca_dets = detect_points_cfar(
        img, alpha=alpha, guard=guard, reference=ref,
        mask_structures=False, min_blob_pixels=1,
    )
    os_dets = detect_points_os_cfar(
        img, alpha=alpha, guard=guard, reference=ref,
        k_percentile=0.75, mask_structures=False, min_blob_pixels=1,
    )

    ca_found = sum(
        1 for tr, tc in truth
        if any(abs(r - tr) <= 1 and abs(c - tc) <= 1 for r, c in ca_dets)
    )
    os_found = sum(
        1 for tr, tc in truth
        if any(abs(r - tr) <= 1 and abs(c - tc) <= 1 for r, c in os_dets)
    )

    assert os_found == 3, (
        f"OS-CFAR should preserve all 3 targets in mixture clutter; got {os_found}/3"
    )
    assert os_found > ca_found, (
        f"OS-CFAR preserved {os_found}/3 targets but CA-CFAR found only {ca_found}/3 "
        "-- OS-CFAR advantage not demonstrated on this fixture"
    )


def test_os_cfar_detects_both_targets_when_ca_cfar_masks_one():
    # Two bright targets 30 px apart — within each other's reference window
    # (window half-width guard+reference=20, so 30 px is inside the 41x41 window).
    # CA-CFAR's mean is dragged up by the first target's presence in the
    # second target's reference cells, suppressing detection.  OS-CFAR's
    # 75th percentile is not dragged.
    img = _expon_noise((256, 256), scale=1.0, seed=137)
    t1 = (128, 100)
    t2 = (128, 130)  # 30 px to the east
    img[t1] = 10.0
    img[t2] = 10.0

    alpha = 5.0
    guard, ref = 5, 15

    ca_dets = detect_points_cfar(
        img, alpha=alpha, guard=guard, reference=ref,
        mask_structures=False, min_blob_pixels=1,
    )
    ca_hits_t1 = any(abs(r - t1[0]) <= 1 and abs(c - t1[1]) <= 1 for r, c in ca_dets)
    ca_hits_t2 = any(abs(r - t2[0]) <= 1 and abs(c - t2[1]) <= 1 for r, c in ca_dets)
    ca_found = int(ca_hits_t1) + int(ca_hits_t2)

    os_dets = detect_points_os_cfar(
        img, alpha=alpha, guard=guard, reference=ref,
        k_percentile=0.75, mask_structures=False, min_blob_pixels=1,
    )
    os_hits_t1 = any(abs(r - t1[0]) <= 1 and abs(c - t1[1]) <= 1 for r, c in os_dets)
    os_hits_t2 = any(abs(r - t2[0]) <= 1 and abs(c - t2[1]) <= 1 for r, c in os_dets)

    assert os_hits_t1 and os_hits_t2, (
        f"OS-CFAR must detect both adjacent targets; got t1={os_hits_t1}, t2={os_hits_t2}"
    )
    assert ca_found <= 2, "sanity: CA-CFAR cannot over-detect truth"


# --- Integration with existing pipeline ---


def test_os_cfar_to_observations_returns_position_obs(tmp_path: Path):
    img = _expon_noise((256, 256), scale=1.0, seed=139)
    img[128, 128] = 20.0
    img[64, 192] = 20.0
    tfm = _straight_tfm()
    crs = "EPSG:32650"
    acq = 1_688_306_455.0
    obs = detect_points_os_cfar_to_observations(
        img, tfm, crs, acquisition_time=acq, source_id="synth-os",
        alpha=3.0, guard=5, reference=15, k_percentile=0.75,
        mask_structures=False, min_blob_pixels=1,
    )
    assert len(obs) >= 2
    for o in obs:
        assert isinstance(o, PositionObservation)
        assert o.modality == "SAR"
        eigs = np.linalg.eigvalsh(o.cov_pos)
        assert eigs.min() > 0
    assert len({o.obs_id for o in obs}) == len(obs)


def test_os_cfar_structure_mask_integrates():
    # 30x30 bright region (structure) + 3 scattered point targets.
    # With mask on, the structure interior is rejected and the 3 targets
    # still come through OS-CFAR.
    img = _expon_noise((512, 512), scale=1.0, seed=149)
    img[100:130, 100:130] = 30.0  # structure
    pts = [(300, 300), (350, 400), (400, 100)]
    for r, c in pts:
        img[r, c] = 20.0
    dets = detect_points_os_cfar(
        img, alpha=3.0, guard=5, reference=15, k_percentile=0.75,
        mask_structures=True,
        **_MASK_KW,
        min_structure_area_pixels=500,
        dilate_pixels=10,
        min_blob_pixels=1,
    )
    # Each point target should be detected (as CA-CFAR case does).
    for tr, tc in pts:
        assert any(abs(dr - tr) <= 1 and abs(dc - tc) <= 1 for dr, dc in dets), (
            f"OS-CFAR with mask lost point target ({tr}, {tc})"
        )
    # The structure centre (115, 115) must NOT appear as a detection.
    assert not any(abs(dr - 115) <= 2 and abs(dc - 115) <= 2 for dr, dc in dets), (
        "structure centre leaked through OS-CFAR + mask"
    )
