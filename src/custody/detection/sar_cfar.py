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
    center_of_mass,
    gaussian_filter,
    label,
    uniform_filter,
)
from rasterio.transform import Affine

from custody.detection.sar_common import detections_to_observations
from custody.fusion.observations import PositionObservation


# CA-CFAR threshold calibration for Umbra uint8 GEC imagery squared to the
# power domain.  See ADR-less Phase 2A note:  uint8-amplitude-squared has
# heavier tails than the exponential distribution CA-CFAR theory assumes, so
# alpha on this input is substantially higher than a textbook Rayleigh/exp
# setting.  Tuned against the 2023-07-02 Tennent scene 2-km AOI crop via the
# alpha sweep in scripts/03_detect_sar_umbra.py.  Override per-scene when
# needed.
CFAR_DEFAULT_ALPHA_UMBRA_UINT8: float = 8.0


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
    structure_sigma: float = 20.0,
    structure_threshold_pct: float = 95.0,
) -> np.ndarray:
    """Return a bool mask: True where the image sits inside a large bright structure.

    Gaussian-smoothed amplitude exceeding the ``structure_threshold_pct`` quantile
    of the smoothed image.  Larger ``structure_sigma`` means the mask only fires
    on genuinely region-scale bright structures, not individual pixels.
    """
    smoothed = gaussian_filter(img.astype(np.float32, copy=False), sigma=structure_sigma)
    threshold = np.percentile(smoothed, structure_threshold_pct)
    return smoothed > threshold


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
    structure_sigma: float = 20.0,
    structure_threshold_pct: float = 95.0,
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
            structure_sigma=structure_sigma,
            structure_threshold_pct=structure_threshold_pct,
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
