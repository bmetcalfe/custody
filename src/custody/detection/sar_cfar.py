"""CA-CFAR point detection on SAR imagery.

Operates in the amplitude-squared (power) domain, computing a local threshold
per pixel from an annular reference region around each candidate.  Optional
post-CFAR structure masking suppresses hits that fall inside large bright
structures (land reclamation, piers) which otherwise light up the detector.

Outputs are either pixel-space (row, col) centroids or
:class:`PositionObservation` instances already projected to the AEQD tangent
plane via :mod:`custody.detection.sar_common` + :mod:`custody.fusion.geo`.

Algorithm
---------
For each pixel P at (r, c), compute the mean of the reference ring — the
annulus of cells between the guard window (side 2g+1) and the total window
(side 2(g+R)+1).  A pixel passes CFAR when

    P  >  alpha * mean(reference ring)

Input is expected to be in the power / intensity domain (e.g., amplitude²
for Rayleigh-distributed SAR magnitude, or the raw display-domain uint8
bumped to power by the caller via ``img**2``).  The implementation uses
two ``scipy.ndimage.uniform_filter`` passes (fast-mean over the total and
guard windows respectively) and differences them to get the ring mean
without an explicit ring convolution.
"""
from __future__ import annotations

from typing import Union

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    center_of_mass,
    gaussian_filter,
    label,
    uniform_filter,
)
from rasterio.transform import Affine

from custody.detection.annular_percentile import annular_percentile_filter
from custody.detection.sar_common import detections_to_observations
from custody.fusion.observations import PositionObservation


# ---------------------------------------------------------------------------
# CFAR threshold
# ---------------------------------------------------------------------------


def compute_cfar_threshold(
    img_power: np.ndarray,
    guard: int,
    reference: int,
    alpha: float,
) -> np.ndarray:
    """Return the per-pixel CA-CFAR threshold at factor ``alpha``.

    ``guard`` is the half-width of the guard window (the inner region that is
    excluded from the reference average); ``reference`` is the half-width of
    the reference ring beyond the guard.  The total window has half-width
    ``guard + reference``.
    """
    img = img_power.astype(np.float32, copy=False)
    total = guard + reference
    total_size = 2 * total + 1
    guard_size = 2 * guard + 1

    mean_total = uniform_filter(img, size=total_size, mode="reflect")
    mean_guard = uniform_filter(img, size=guard_size, mode="reflect")

    n_total = total_size * total_size
    n_guard = guard_size * guard_size
    n_ref = n_total - n_guard
    if n_ref <= 0:
        raise ValueError("CFAR reference ring is empty; increase `reference` or decrease `guard`")

    ref_mean = (mean_total * n_total - mean_guard * n_guard) / n_ref
    return alpha * ref_mean


# ---------------------------------------------------------------------------
# Structure mask
# ---------------------------------------------------------------------------


def compute_structure_mask(
    img: np.ndarray,
    smooth_sigma: float = 15.0,
    bright_threshold_pct: float = 75.0,
    min_structure_area_pixels: int = 500,
    dilate_pixels: int = 20,
) -> np.ndarray:
    """Mask large bright structures (land, reclamation, reef, large platforms).

    Returns a bool mask where True marks a pixel inside (or in the dilation
    halo of) a contiguous bright region whose area meets the minimum-area
    threshold.  Isolated bright pixels — vessels, speckle peaks — are NOT
    masked, so CFAR can still fire on them.

    Pipeline:

    1. Gaussian smoothing with ``smooth_sigma`` to reduce speckle.
    2. Threshold at the ``bright_threshold_pct`` percentile of the smoothed image.
    3. Connected-component labeling (``scipy.ndimage.label``).
    4. Keep only components whose pixel count ≥ ``min_structure_area_pixels``.
    5. Dilate the surviving mask by ``dilate_pixels`` iterations to cover edge
       artifacts that CFAR would otherwise fire on just outside the boundary.
    """
    smoothed = gaussian_filter(img.astype(np.float32, copy=False), sigma=smooth_sigma)
    threshold = np.percentile(smoothed, bright_threshold_pct)
    bright = smoothed > threshold

    labels, n_components = label(bright)
    if n_components == 0:
        return np.zeros_like(bright, dtype=bool)

    sizes = np.bincount(labels.ravel())  # sizes[0] is the background
    keep_component = sizes >= min_structure_area_pixels
    keep_component[0] = False  # never keep the background label
    mask = keep_component[labels]

    if dilate_pixels > 0 and mask.any():
        mask = binary_dilation(mask, iterations=int(dilate_pixels))

    return mask


# ---------------------------------------------------------------------------
# Point detection
# ---------------------------------------------------------------------------


def detect_points_cfar(
    img: np.ndarray,
    *,
    alpha: float = 3.0,
    guard: int = 5,
    reference: int = 15,
    mask_structures: bool = True,
    smooth_sigma: float = 15.0,
    bright_threshold_pct: float = 75.0,
    min_structure_area_pixels: int = 500,
    dilate_pixels: int = 20,
    min_blob_pixels: int = 2,
    max_blob_pixels: int = 500,
) -> list[tuple[int, int]]:
    """Run CA-CFAR, optionally suppress structure hits, return pixel-space centroids.

    ``img`` is taken to be already in the power/intensity domain.  For raw
    amplitude imagery, square the input first: ``detect_points_cfar(img**2, ...)``.

    Returns a list of ``(row, col)`` integer centroids, one per connected
    component whose pixel count is in ``[min_blob_pixels, max_blob_pixels]``.
    """
    power = img.astype(np.float32, copy=False)
    threshold = compute_cfar_threshold(power, guard=guard, reference=reference, alpha=alpha)
    hits = power > threshold

    if mask_structures:
        mask = compute_structure_mask(
            img,
            smooth_sigma=smooth_sigma,
            bright_threshold_pct=bright_threshold_pct,
            min_structure_area_pixels=min_structure_area_pixels,
            dilate_pixels=dilate_pixels,
        )
        hits &= ~mask

    if not hits.any():
        return []

    labels, n = label(hits)
    if n == 0:
        return []

    sizes = np.bincount(labels.ravel())  # sizes[0] is the background
    centroids = center_of_mass(hits, labels, range(1, n + 1))

    out: list[tuple[int, int]] = []
    for i, (r, c) in enumerate(centroids):
        size = sizes[i + 1]  # component labels are 1-indexed
        if min_blob_pixels <= size <= max_blob_pixels:
            out.append((int(round(r)), int(round(c))))
    return out


def detect_points_to_observations(
    img: np.ndarray,
    transform: Affine,
    crs_wkt: str,
    *,
    acquisition_time: float,
    source_id: str,
    sigma_m: float = 10.0,
    detector_version: str = "sar_cfar_v1",
    **cfar_kwargs,
) -> list[PositionObservation]:
    """End-to-end helper: image → detections → PositionObservation list."""
    detections = detect_points_cfar(img, **cfar_kwargs)
    return detections_to_observations(
        detections,
        transform,
        crs_wkt,
        obs_id_prefix=source_id,
        acquisition_time=acquisition_time,
        source_id=source_id,
        sigma_m=sigma_m,
        detector_version=detector_version,
    )


# ---------------------------------------------------------------------------
# OS-CFAR (Ordered-Statistic CFAR) — per ADR-0014
# ---------------------------------------------------------------------------
#
# OS-CFAR replaces CA-CFAR's reference-cell mean with a fixed percentile
# (default 75th).  This is robust to heterogeneous clutter (exponential +
# lognormal mixtures, bright speckle clumps, bathymetry bleed-through) and
# to target-near-target masking (a bright neighbour doesn't drag the
# percentile the way it drags the mean).
#
# Window implementation: annular (ring-shaped) window with the guard region
# explicitly excluded — the correct formulation per Rohling (1983).  The
# numba JIT implementation in custody.detection.annular_percentile makes
# this tractable at full Umbra scene scale (17602x17602).  Resolves the
# "annular-window implementation deferred" caveat in ADR-0014.


def compute_os_cfar_threshold(
    img_power: np.ndarray,
    guard: int,
    reference: int,
    alpha: float,
    k_percentile: float = 0.75,
) -> np.ndarray:
    """Return the per-pixel OS-CFAR threshold over an annular reference window.

    ``k_percentile`` is a fraction in (0, 1] — 0.75 picks the 75th percentile
    of the reference ring (the annulus from guard+1 to guard+reference half-width,
    centre guard square excluded).
    """
    img = img_power.astype(np.float32, copy=False)
    if not 0.0 < k_percentile <= 1.0:
        raise ValueError(f"k_percentile must be in (0, 1]; got {k_percentile}")
    ref_quantile = annular_percentile_filter(
        img, guard=guard, reference=reference, percentile=k_percentile * 100.0,
    )
    return alpha * ref_quantile


def detect_points_os_cfar(
    img: np.ndarray,
    *,
    alpha: float = 3.0,
    guard: int = 20,
    reference: int = 60,
    k_percentile: float = 0.75,
    mask_structures: bool = True,
    smooth_sigma: float = 15.0,
    bright_threshold_pct: float = 75.0,
    min_structure_area_pixels: int = 500,
    dilate_pixels: int = 20,
    min_blob_pixels: int = 2,
    max_blob_pixels: int = 500,
) -> list[tuple[int, int]]:
    """Run OS-CFAR, optionally suppress structure hits, return pixel-space centroids.

    ``img`` is taken to be already in the power/intensity domain.  For raw
    amplitude imagery, square the input first: ``detect_points_os_cfar(img**2, ...)``.
    """
    power = img.astype(np.float32, copy=False)
    threshold = compute_os_cfar_threshold(
        power, guard=guard, reference=reference,
        alpha=alpha, k_percentile=k_percentile,
    )
    hits = power > threshold

    if mask_structures:
        mask = compute_structure_mask(
            img,
            smooth_sigma=smooth_sigma,
            bright_threshold_pct=bright_threshold_pct,
            min_structure_area_pixels=min_structure_area_pixels,
            dilate_pixels=dilate_pixels,
        )
        hits &= ~mask

    if not hits.any():
        return []

    labels, n = label(hits)
    if n == 0:
        return []

    sizes = np.bincount(labels.ravel())
    centroids = center_of_mass(hits, labels, range(1, n + 1))

    out: list[tuple[int, int]] = []
    for i, (r, c) in enumerate(centroids):
        size = sizes[i + 1]
        if min_blob_pixels <= size <= max_blob_pixels:
            out.append((int(round(r)), int(round(c))))
    return out


def detect_points_os_cfar_to_observations(
    img: np.ndarray,
    transform: Affine,
    crs_wkt: str,
    *,
    acquisition_time: float,
    source_id: str,
    sigma_m: float = 10.0,
    detector_version: str = "sar_os_cfar_v1",
    **cfar_kwargs,
) -> list[PositionObservation]:
    """End-to-end helper (OS-CFAR): image → detections → PositionObservation list."""
    detections = detect_points_os_cfar(img, **cfar_kwargs)
    return detections_to_observations(
        detections,
        transform,
        crs_wkt,
        obs_id_prefix=source_id,
        acquisition_time=acquisition_time,
        source_id=source_id,
        sigma_m=sigma_m,
        detector_version=detector_version,
    )
