"""Run CA-CFAR on a 2-km AOI crop around each Umbra scene's target coordinates.

Rationale (per Phase 2A post-mortem):
  - Full-scene CFAR over 11700x11700 on uint8 GEC returned ~50k false alarms at
    alpha=3.0 due to uint8-amplitude-squared heavy tails and a scene-wide
    radiometric gradient (~43% top-to-bottom).
  - Cropping to a 2-km box around the target (from METADATA) cuts the pixel
    count by ~350x, makes the gradient negligible (<8% across 2 km), and
    lets alpha be calibrated on the AOI where detections actually matter.

This script:
  1. Reads scene + METADATA.
  2. Crops to 2-km box around sceneCenterPointLla.
  3. Squares to power domain.
  4. Sweeps alpha over a user-supplied list, saving per-alpha preview PNGs.
  5. Picks the final alpha (CFAR_DEFAULT_ALPHA_UMBRA_UINT8) and writes Parquet
     via fusion.index.

Run::

    uv run python scripts/03_detect_sar_umbra.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from scipy.ndimage import binary_dilation  # noqa: E402

from custody.detection.sar_cfar import (  # noqa: E402
    CFAR_DEFAULT_ALPHA_UMBRA_UINT8,
    compute_structure_mask,
    detect_points_cfar,
    detect_points_to_observations,
)
from custody.detection.sar_common import crop_to_aoi, read_geotiff  # noqa: E402
from custody.fusion.index import index_observations  # noqa: E402


SCENE_DIR = (
    REPO_ROOT
    / "data/raw/umbra/sar-data/tasks/ship_detection_testdata/"
    / "f0730a1d-2bf7-4193-b2fe-ec8bfb2e1aef/2023-07-02-14-00-55_UMBRA-05"
)
GEC_PATH = SCENE_DIR / "2023-07-02-14-00-55_UMBRA-05_GEC.tif"
META_PATH = SCENE_DIR / "2023-07-02-14-00-55_UMBRA-05_METADATA.json"

OUT_DIR = REPO_ROOT / "data/processed/sar_detections"
SCRATCH = REPO_ROOT / "day0/scratch"

# Alpha sweep values (Phase 2A remediation)
ALPHA_SWEEP = [4.0, 5.0, 6.0, 7.0, 8.0]

# CFAR fixed params for the sweep
GUARD = 20
REFERENCE = 60
STRUCT_SIGMA = 80.0
STRUCT_THR_PCT = 95.0
MIN_BLOB = 4
MAX_BLOB = 500


def _save_preview(
    cropped_img: np.ndarray,
    detections: list[tuple[int, int]],
    out_path: Path,
    mask: np.ndarray | None = None,
    thumb: int = 1100,
) -> None:
    valid = cropped_img[cropped_img > 0]
    if valid.size > 0:
        lo, hi = np.percentile(valid, [2, 98])
    else:
        lo, hi = 0, 255
    stretched = np.clip(
        (cropped_img.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0.0, 1.0
    )
    img8 = (stretched * 255).astype(np.uint8)
    rgb = np.stack([img8, img8, img8], axis=-1)
    if mask is not None:
        edge = binary_dilation(mask) & ~mask
        rgb[edge] = [220, 180, 60]
    pil = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(pil)
    for r, c in detections:
        draw.ellipse((c - 2, r - 2, c + 2, r + 2), fill=(220, 30, 30))
    pil.thumbnail((thumb, thumb))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(out_path, optimize=True)


def main():
    # --- Load scene + metadata
    with META_PATH.open() as f:
        meta = json.load(f)
    collect = meta["collects"][0]
    scene_center = collect["sceneCenterPointLla"]["coordinates"]
    target_lon, target_lat = scene_center[0], scene_center[1]
    start_utc = collect["startAtUTC"]
    acq_epoch = datetime.fromisoformat(start_utc.replace("Z", "+00:00")).timestamp()
    print(f"Scene: {GEC_PATH.name}  ({GEC_PATH.stat().st_size/1e6:.1f} MB)")
    print(f"  target: ({target_lat:.6f}, {target_lon:.6f})  acq: {start_utc}")

    img, transform, crs = read_geotiff(GEC_PATH)
    print(f"  full image: {img.shape}  dtype: {img.dtype}")

    # --- Crop to 2 km AOI (box_half_km = 1.0)
    cropped, new_tfm, bounds = crop_to_aoi(
        img, transform, target_lat, target_lon, crs, box_half_km=1.0
    )
    print(f"  AOI crop: {cropped.shape}  pixel_size={bounds['pixel_size_m']:.3f} m")
    crop_power = cropped.astype(np.float32) ** 2
    mask = compute_structure_mask(
        cropped.astype(np.float32),
        structure_sigma=STRUCT_SIGMA,
        structure_threshold_pct=STRUCT_THR_PCT,
    )

    # --- Alpha sweep
    print()
    print(f"Alpha sweep on 2-km AOI crop (guard={GUARD}, ref={REFERENCE}, mask on):")
    sweep_results: list[tuple[float, int, float]] = []
    for alpha in ALPHA_SWEEP:
        t0 = time.perf_counter()
        detections = detect_points_cfar(
            crop_power,
            alpha=alpha,
            guard=GUARD,
            reference=REFERENCE,
            mask_structures=True,
            structure_sigma=STRUCT_SIGMA,
            structure_threshold_pct=STRUCT_THR_PCT,
            min_blob_pixels=MIN_BLOB,
            max_blob_pixels=MAX_BLOB,
        )
        dt = time.perf_counter() - t0
        print(f"  alpha={alpha:>4.1f}   detections={len(detections):>5}   {dt*1000:.1f} ms")
        sweep_results.append((alpha, len(detections), dt))
        preview_path = SCRATCH / f"umbra_2023_07_02_alpha{int(alpha)}_detections.png"
        _save_preview(cropped, detections, preview_path, mask=mask)

    # --- Final run at CFAR_DEFAULT_ALPHA_UMBRA_UINT8
    print()
    chosen = CFAR_DEFAULT_ALPHA_UMBRA_UINT8
    print(f"Final run at default alpha = {chosen}:")
    obs = detect_points_to_observations(
        crop_power,
        new_tfm,
        crs,
        acquisition_time=acq_epoch,
        source_id="umbra-2023-07-02",
        alpha=chosen,
        guard=GUARD,
        reference=REFERENCE,
        mask_structures=True,
        structure_sigma=STRUCT_SIGMA,
        structure_threshold_pct=STRUCT_THR_PCT,
        min_blob_pixels=MIN_BLOB,
        max_blob_pixels=MAX_BLOB,
    )
    print(f"  detections: {len(obs)}")

    # Write Parquet via fusion index
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Clear prior parquet so we don't accumulate stale rows across re-runs.
    for stale in (OUT_DIR / "position.parquet", OUT_DIR / "posvel.parquet"):
        if stale.exists():
            stale.unlink()
    index_observations(obs, out_dir=OUT_DIR)
    pq = OUT_DIR / "position.parquet"
    if pq.exists():
        print(f"  wrote {pq.name} ({pq.stat().st_size/1024:.1f} KB)")

    # Final overlay PNG
    final_detections = []
    for o in obs:
        parts = o.obs_id.rsplit("-", 2)
        final_detections.append((int(parts[-2]), int(parts[-1])))
    final_preview = SCRATCH / "umbra_2023_07_02_detections.png"
    _save_preview(cropped, final_detections, final_preview, mask=mask)
    print(f"  wrote preview {final_preview.name} ({final_preview.stat().st_size/1024:.1f} KB)")

    # Top-20 detections with lat/lon for spot-check
    from custody.detection.sar_common import pixel_to_latlon
    print()
    print("Top 20 detection centroids (lat, lon) in the 2-km AOI crop:")
    for i, (r, c) in enumerate(final_detections[:20]):
        lat, lon = pixel_to_latlon(r, c, new_tfm, crs)
        print(f"  [{i+1:>2}] row={r:>4} col={c:>4}  -> ({lat:.6f}, {lon:.6f})")


if __name__ == "__main__":
    main()
