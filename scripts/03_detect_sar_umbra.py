"""Run CA-CFAR over the 2023-07-02 Umbra Tennent scene.

Performs a 4× decimated tuning pass, a full-resolution final pass, and
writes the full-resolution detections to Parquet via
:mod:`custody.fusion.index`.  A preview PNG with detection centroids and
the structure-mask outline is saved to ``day0/scratch/`` for visual
inspection (gitignored).

Run::

    uv run python scripts/03_detect_sar_umbra.py
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from rasterio.transform import Affine  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

from custody.detection.sar_cfar import (  # noqa: E402
    compute_structure_mask,
    detect_points_cfar,
    detect_points_to_observations,
)
from custody.detection.sar_common import read_geotiff  # noqa: E402
from custody.fusion.index import index_observations  # noqa: E402


SCENE_PATH = (
    REPO_ROOT
    / "data/raw/umbra/sar-data/tasks/ship_detection_testdata/"
    / "f0730a1d-2bf7-4193-b2fe-ec8bfb2e1aef/2023-07-02-14-00-55_UMBRA-05"
    / "2023-07-02-14-00-55_UMBRA-05_GEC.tif"
)
OUT_DIR = REPO_ROOT / "data/processed/sar_detections"
PREVIEW_PNG = REPO_ROOT / "day0/scratch/umbra_2023_07_02_detections.png"

# 2023-07-02T14:00:55.344Z from the scene METADATA.
ACQUISITION_EPOCH = datetime(
    2023, 7, 2, 14, 0, 55, 344_000, tzinfo=timezone.utc
).timestamp()


def main():
    print(f"Loading: {SCENE_PATH.name}  ({SCENE_PATH.stat().st_size / 1e6:.1f} MB)")
    img, transform, crs = read_geotiff(SCENE_PATH)
    print(f"  dims={img.shape}, dtype={img.dtype}")

    # --- Decimated tuning pass (4x) ---
    print()
    print("Decimated pass (4x, uint8 squared to power domain)...")
    img_decim = (img[::4, ::4].astype(np.float32)) ** 2
    transform_decim = transform * Affine.scale(4)
    t0 = time.perf_counter()
    obs_decim = detect_points_to_observations(
        img_decim, transform_decim, crs,
        acquisition_time=ACQUISITION_EPOCH,
        source_id="umbra-2023-07-02",
        alpha=3.0, guard=5, reference=15,
        mask_structures=True,
        structure_sigma=20.0, structure_threshold_pct=95.0,
        min_blob_pixels=2, max_blob_pixels=500,
    )
    dt_decim = time.perf_counter() - t0
    print(f"  detections: {len(obs_decim)}   wall: {dt_decim:.2f}s")

    # --- Full-resolution final pass ---
    print()
    print("Full-resolution pass (~145 Mpx squared to power domain)...")
    img_full_power = img.astype(np.float32) ** 2
    t0 = time.perf_counter()
    obs_full = detect_points_to_observations(
        img_full_power, transform, crs,
        acquisition_time=ACQUISITION_EPOCH,
        source_id="umbra-2023-07-02",
        alpha=3.0, guard=20, reference=60,
        mask_structures=True,
        structure_sigma=80.0, structure_threshold_pct=95.0,
        min_blob_pixels=4, max_blob_pixels=500,
    )
    dt_full = time.perf_counter() - t0
    print(f"  detections: {len(obs_full)}   wall: {dt_full:.2f}s")

    # --- Persist to Parquet via fusion index ---
    print()
    print(f"Writing Parquet to {OUT_DIR} ...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index_observations(obs_full, out_dir=OUT_DIR)
    parquet_path = OUT_DIR / "position.parquet"
    if parquet_path.exists():
        print(f"  {parquet_path.name}: {parquet_path.stat().st_size / 1024:.1f} KB")

    # --- Preview PNG ---
    print()
    print(f"Writing preview {PREVIEW_PNG} ...")
    PREVIEW_PNG.parent.mkdir(parents=True, exist_ok=True)
    decim = 12
    preview_h = img.shape[0] // decim
    preview_w = img.shape[1] // decim
    preview = img[::decim, ::decim]
    lo, hi = np.percentile(preview[preview > 0], [2, 98]) if (preview > 0).any() else (0, 255)
    preview_8b = np.clip(
        (preview.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0.0, 1.0
    )
    preview_8b = (preview_8b * 255).astype(np.uint8)
    rgb = np.stack([preview_8b, preview_8b, preview_8b], axis=-1)
    # Structure mask outline: binary edge of the mask in red.
    mask = compute_structure_mask(img.astype(np.float32), structure_sigma=80.0,
                                  structure_threshold_pct=95.0)
    mask_decim = mask[::decim, ::decim]
    # Edge detection on the decimated mask
    from scipy.ndimage import binary_dilation
    mask_edge = binary_dilation(mask_decim) & ~mask_decim
    rgb[mask_edge] = [220, 180, 60]  # warm edge
    img_pil = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(img_pil)
    # Red dot for every full-res detection
    for obs in obs_full:
        # Recover pixel coords from obs_id: "umbra-2023-07-02-<epoch>-<row>-<col>"
        parts = obs.obs_id.rsplit("-", 2)
        r = int(parts[-2]); c = int(parts[-1])
        pr = r // decim
        pc = c // decim
        draw.ellipse((pc - 2, pr - 2, pc + 2, pr + 2), fill=(220, 30, 30))
    img_pil.thumbnail((1100, 1100))
    img_pil.save(PREVIEW_PNG, optimize=True)
    print(f"  {PREVIEW_PNG.name}: {PREVIEW_PNG.stat().st_size / 1024:.1f} KB")

    print()
    print(f"Done. decimated={len(obs_decim)}   full={len(obs_full)}")


if __name__ == "__main__":
    main()
