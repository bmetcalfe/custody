"""Run CA-CFAR on the Whitsun Reef 2023-12-06 (earlier) Umbra scene.

Same defaults as scripts/03_detect_sar_umbra.py (Tennent):
  - 2-km AOI crop around sceneCenterPointLla
  - alpha = CFAR_DEFAULT_ALPHA_UMBRA_UINT8 (7.0)
  - structure mask with sigma=80, pct=95, min_area=500, dilate=20
  - guard=20, reference=60, min_blob=4, max_blob=500

Whitsun is open water with discrete vessel-like bright returns rather than a
single large reclamation structure.  The structure mask is expected to produce
a near-empty bool mask; detection count should reflect vessel-like clutter.

Run::

    uv run python scripts/04_detect_sar_whitsun.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from scipy.ndimage import binary_dilation  # noqa: E402

from custody.detection.sar_cfar import (  # noqa: E402
    CFAR_DEFAULT_ALPHA_UMBRA_UINT8,
    compute_structure_mask,
    detect_points_to_observations,
)
from custody.detection.sar_common import crop_to_aoi, pixel_to_latlon, read_geotiff  # noqa: E402
from custody.fusion.index import index_observations  # noqa: E402


SCENE_DIR = (
    REPO_ROOT
    / "data/raw/umbra/sar-data/tasks/ship_detection_testdata/"
    / "1e1f051d-4c81-4640-997d-1a03e967ad6a/2023-12-06-02-06-24_UMBRA-04"
)
GEC_PATH = SCENE_DIR / "2023-12-06-02-06-24_UMBRA-04_GEC.tif"
META_PATH = SCENE_DIR / "2023-12-06-02-06-24_UMBRA-04_METADATA.json"

OUT_DIR = REPO_ROOT / "data/processed/sar_detections"
SCRATCH = REPO_ROOT / "day0/scratch"

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
    if mask is not None and mask.any():
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

    cropped, new_tfm, bounds = crop_to_aoi(
        img, transform, target_lat, target_lon, crs, box_half_km=1.0
    )
    print(f"  AOI crop: {cropped.shape}  pixel_size={bounds['pixel_size_m']:.3f} m")
    crop_power = cropped.astype(np.float32) ** 2

    mask = compute_structure_mask(
        cropped.astype(np.float32),
        smooth_sigma=STRUCT_SIGMA,
        bright_threshold_pct=STRUCT_THR_PCT,
    )
    print(f"  structure mask: {int(mask.sum())} px ({100*mask.sum()/mask.size:.2f}% of crop)")

    chosen = CFAR_DEFAULT_ALPHA_UMBRA_UINT8
    print()
    print(f"Running CFAR at alpha = {chosen}:")
    obs = detect_points_to_observations(
        crop_power,
        new_tfm,
        crs,
        acquisition_time=acq_epoch,
        source_id="umbra-2023-12-06-early",
        alpha=chosen,
        guard=GUARD,
        reference=REFERENCE,
        mask_structures=True,
        smooth_sigma=STRUCT_SIGMA,
        bright_threshold_pct=STRUCT_THR_PCT,
        min_blob_pixels=MIN_BLOB,
        max_blob_pixels=MAX_BLOB,
    )
    print(f"  detections: {len(obs)}")

    scene_out_dir = OUT_DIR / "whitsun_2023_12_06_early"
    scene_out_dir.mkdir(parents=True, exist_ok=True)
    for stale in (scene_out_dir / "position.parquet", scene_out_dir / "posvel.parquet"):
        if stale.exists():
            stale.unlink()
    index_observations(obs, out_dir=scene_out_dir)
    pq = scene_out_dir / "position.parquet"
    if pq.exists():
        print(f"  wrote {pq.relative_to(REPO_ROOT)} ({pq.stat().st_size/1024:.1f} KB)")

    final_detections = []
    for o in obs:
        parts = o.obs_id.rsplit("-", 2)
        final_detections.append((int(parts[-2]), int(parts[-1])))
    final_preview = SCRATCH / "umbra_2023_12_06_whitsun_detections.png"
    _save_preview(cropped, final_detections, final_preview, mask=mask)
    print(f"  wrote preview {final_preview.name} ({final_preview.stat().st_size/1024:.1f} KB)")

    # Spatial distribution summary
    H = cropped.shape[0]
    mid = H // 2
    pixel_size_m = bounds["pixel_size_m"]
    halo_px = int(round(200.0 / pixel_size_m))
    if mask.any():
        mask_plus_200m = binary_dilation(mask, iterations=halo_px)
    else:
        mask_plus_200m = mask
    north = sum(1 for r, c in final_detections if r < mid)
    south = sum(1 for r, c in final_detections if r >= mid)
    near = sum(1 for r, c in final_detections if mask_plus_200m[r, c])
    far = len(final_detections) - near
    print()
    print(f"Spatial distribution ({len(final_detections)} detections):")
    print(f"  North (row < {mid}): {north}")
    print(f"  South (row >= {mid}): {south}")
    print(f"  Within 200m of structure mask: {near}")
    print(f"  Beyond 200m (open water): {far}")

    print()
    print("Top 20 detection centroids (lat, lon):")
    for i, (r, c) in enumerate(final_detections[:20]):
        lat, lon = pixel_to_latlon(r, c, new_tfm, crs)
        print(f"  [{i+1:>2}] row={r:>4} col={c:>4}  -> ({lat:.6f}, {lon:.6f})")


if __name__ == "__main__":
    main()
