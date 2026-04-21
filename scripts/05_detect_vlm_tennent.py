"""Real-scene VLM detection run on Tennent 2023-07-02 AOI crop.

Uses Claude Sonnet 4.6 via the Anthropic backend.  121 tiles (11x11) over a
~2 km square AOI centered on Đá Tiên Nữ (8.856°N, 114.665°E).  Estimated
cost ~$1.69 (measured rate $0.014/tile).

Outputs (diagnostic, not committed):
    data/processed/vlm_detections/position.parquet
    day0/scratch/tennent_20230702_vlm_progress.jsonl  (one line per tile)
    day0/scratch/tennent_20230702_vlm_summary.json
    day0/scratch/tennent_20230702_vlm_detections.png  (preview overlay)
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from custody.detection.sar_common import crop_to_aoi, read_geotiff  # noqa: E402
from custody.detection.vlm_backends import AnthropicBackend  # noqa: E402
from custody.detection.vlm_sar import (  # noqa: E402
    detect_vessels_in_scene,
    estimate_scene_cost,
)
from custody.fusion.index import index_observations  # noqa: E402


SCENE_DIR = (
    REPO_ROOT
    / "data/raw/umbra/sar-data/tasks/ship_detection_testdata"
    / "f0730a1d-2bf7-4193-b2fe-ec8bfb2e1aef/2023-07-02-14-00-55_UMBRA-05"
)
GEC_PATH = SCENE_DIR / "2023-07-02-14-00-55_UMBRA-05_GEC.tif"
META_PATH = SCENE_DIR / "2023-07-02-14-00-55_UMBRA-05_METADATA.json"

AOI_CROP_KM_HALF_WIDTH = 1.0
TILE_SIZE = 640
TILE_OVERLAP = 64
CONFIDENCE_THRESHOLD = "medium"
NMS_IOU_THRESHOLD = 0.5
MAX_COST_USD = 3.00

OUT_DIR = REPO_ROOT / "data/processed/vlm_detections"
SCRATCH = REPO_ROOT / "day0/scratch"
PROGRESS_LOG = SCRATCH / "tennent_20230702_vlm_progress.jsonl"
SUMMARY_JSON = SCRATCH / "tennent_20230702_vlm_summary.json"
PREVIEW_PNG = SCRATCH / "tennent_20230702_vlm_detections.png"


def main() -> None:
    t_start = time.perf_counter()

    # ---- Load scene + metadata ----
    with META_PATH.open() as f:
        meta = json.load(f)
    collect = meta["collects"][0]
    scene_center = collect["sceneCenterPointLla"]["coordinates"]
    target_lon, target_lat = scene_center[0], scene_center[1]
    start_utc = collect["startAtUTC"]
    acq_epoch = datetime.fromisoformat(start_utc.replace("Z", "+00:00")).timestamp()

    print(f"Scene: {GEC_PATH.name}")
    print(f"  startAtUTC: {start_utc}")
    print(f"  target (lat, lon): ({target_lat:.6f}, {target_lon:.6f})")

    img, transform, crs_wkt = read_geotiff(GEC_PATH)
    print(f"  full image: {img.shape}  dtype={img.dtype}")

    cropped, aoi_transform, bounds = crop_to_aoi(
        img, transform, target_lat, target_lon, crs_wkt,
        box_half_km=AOI_CROP_KM_HALF_WIDTH,
    )
    print(f"  AOI crop: {cropped.shape}  pixel_size={bounds['pixel_size_m']:.3f} m")
    print(f"    scene pixel bounds: rows [{bounds['row_start']}, {bounds['row_end']}),"
          f" cols [{bounds['col_start']}, {bounds['col_end']})")
    del img  # 11700x11700 uint8 = 137 MB; free before the long run

    # ---- Backend + pre-flight cost ----
    backend = AnthropicBackend(model="claude-sonnet-4-6")
    est = estimate_scene_cost(
        cropped, backend,
        tile_size=TILE_SIZE, tile_overlap=TILE_OVERLAP,
        skip_empty=True,
    )
    print(f"\nPre-flight estimate: {est['tile_count']} tiles, "
          f"${est['estimated_cost_usd']:.4f} ({est['estimate_type']})")
    print(f"Max cost USD: ${MAX_COST_USD:.2f}")

    # ---- Progress callback: JSONL line per tile + stdout ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    # Fresh progress log for this run (overwrite any previous partial log)
    if PROGRESS_LOG.exists():
        PROGRESS_LOG.unlink()
    progress_fh = PROGRESS_LOG.open("w", encoding="utf-8")

    tile_records: list[dict] = []

    def progress_cb(tile_info, response, done, total):
        rec = {
            "tile_index": int(tile_info.index),
            "origin_row": int(tile_info.origin[0]),
            "origin_col": int(tile_info.origin[1]),
            "tile_shape": [int(tile_info.tile_shape[0]), int(tile_info.tile_shape[1])],
            "errored": response is None,
            "wall_time_seconds": None if response is None else float(response.wall_time_seconds or 0.0),
            "tokens_in":  None if response is None else int(response.tokens_used["input"]),
            "tokens_out": None if response is None else int(response.tokens_used["output"]),
            "detection_count": 0 if response is None else len(response.detections),
        }
        progress_fh.write(json.dumps(rec) + "\n")
        progress_fh.flush()
        tile_records.append(rec)
        wt = rec["wall_time_seconds"] if rec["wall_time_seconds"] is not None else float("nan")
        tokens = (
            f"{rec['tokens_in']}/{rec['tokens_out']}"
            if rec["tokens_in"] is not None else "err/err"
        )
        status = "ERROR" if rec["errored"] else "ok"
        print(
            f"[tile {done:>3}/{total}] {status}  origin=({rec['origin_row']:>5},{rec['origin_col']:>5})  "
            f"wall={wt:>5.1f}s  tokens={tokens}  detections={rec['detection_count']}"
        )

    # ---- Run detection ----
    print(f"\n=== Running detect_vessels_in_scene (prompt=contextualized_v1, "
          f"confidence>={CONFIDENCE_THRESHOLD}, NMS IoU>={NMS_IOU_THRESHOLD}) ===\n")

    t_det_start = time.perf_counter()
    try:
        observations = detect_vessels_in_scene(
            cropped,
            transform=aoi_transform,
            crs_wkt=crs_wkt,
            acquisition_time=acq_epoch,
            source_id="umbra-2023-07-02-vlm",
            backend=backend,
            prompt_variant="contextualized_v1",
            confidence_threshold=CONFIDENCE_THRESHOLD,
            tile_size=TILE_SIZE,
            tile_overlap=TILE_OVERLAP,
            skip_empty=True,
            max_cost_usd=MAX_COST_USD,
            nms_iou_threshold=NMS_IOU_THRESHOLD,
            progress_callback=progress_cb,
        )
    finally:
        progress_fh.close()

    det_wall = time.perf_counter() - t_det_start
    print(f"\nDetection loop complete: {len(observations)} observations "
          f"(post-NMS, post-threshold) in {det_wall:.1f}s")

    # ---- Parquet output ----
    for stale in (OUT_DIR / "position.parquet", OUT_DIR / "posvel.parquet"):
        if stale.exists():
            stale.unlink()
    index_observations(observations, out_dir=OUT_DIR)
    pq_path = OUT_DIR / "position.parquet"
    print(f"Wrote: {pq_path.relative_to(REPO_ROOT)} "
          f"({pq_path.stat().st_size/1024:.1f} KB)")

    # ---- Aggregate stats ----
    n_tiles_attempted = len(tile_records)
    n_tiles_errored = sum(1 for r in tile_records if r["errored"])
    pre_nms_detection_count = sum(r["detection_count"] for r in tile_records)
    total_tokens_in = sum((r["tokens_in"] or 0) for r in tile_records)
    total_tokens_out = sum((r["tokens_out"] or 0) for r in tile_records)
    wall_times = [r["wall_time_seconds"] for r in tile_records if r["wall_time_seconds"] is not None]
    mean_wall = sum(wall_times) / len(wall_times) if wall_times else 0.0

    # Actual cost from tokens (Claude Sonnet 4.6 published rates)
    INPUT_PER_M = 3.0
    OUTPUT_PER_M = 15.0
    actual_cost = total_tokens_in * INPUT_PER_M / 1e6 + total_tokens_out * OUTPUT_PER_M / 1e6

    # Top-5 tiles by detection count
    top5 = sorted(
        (r for r in tile_records if not r["errored"]),
        key=lambda r: -r["detection_count"],
    )[:5]
    top5_report = [
        {
            "tile_index": r["tile_index"],
            "origin": [r["origin_row"], r["origin_col"]],
            "detection_count": r["detection_count"],
        }
        for r in top5
    ]

    errored = [
        {"tile_index": r["tile_index"], "origin": [r["origin_row"], r["origin_col"]]}
        for r in tile_records if r["errored"]
    ]

    summary = {
        "scene": GEC_PATH.name,
        "acquisition_time_utc": start_utc,
        "target_latlon": [target_lat, target_lon],
        "aoi_shape_px": list(cropped.shape),
        "pixel_size_m": float(bounds["pixel_size_m"]),
        "backend_model": backend.model_name,
        "prompt_variant": "contextualized_v1",
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "nms_iou_threshold": NMS_IOU_THRESHOLD,
        "tile_size": TILE_SIZE,
        "tile_overlap": TILE_OVERLAP,
        "tiles_total_estimate": est["tile_count"],
        "tiles_attempted": n_tiles_attempted,
        "tiles_errored": n_tiles_errored,
        "errored_tile_origins": errored,
        "detections_pre_nms": pre_nms_detection_count,
        "detections_post_nms_post_threshold": len(observations),
        "total_tokens_in": total_tokens_in,
        "total_tokens_out": total_tokens_out,
        "estimated_cost_usd": est["estimated_cost_usd"],
        "actual_cost_usd_from_tokens": actual_cost,
        "wall_time_total_s": det_wall,
        "wall_time_mean_per_tile_s": mean_wall,
        "top5_tiles_by_detection_count": top5_report,
        "parquet_path": str(pq_path.relative_to(REPO_ROOT).as_posix()),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote: {SUMMARY_JSON.relative_to(REPO_ROOT)}")

    # ---- Preview overlay ----
    print("Rendering preview...")
    render_preview(cropped, observations, bounds, aoi_transform, crs_wkt, PREVIEW_PNG,
                   thumb_side=1600)
    print(f"Wrote: {PREVIEW_PNG.relative_to(REPO_ROOT)} "
          f"({PREVIEW_PNG.stat().st_size/1024:.1f} KB)")

    total_wall = time.perf_counter() - t_start
    print()
    print("=== RUN COMPLETE ===")
    print(f"  total wall (incl. I/O):  {total_wall:.1f}s")
    print(f"  detection wall:          {det_wall:.1f}s")
    print(f"  observations (post-NMS): {len(observations)}")
    print(f"  detections (pre-NMS):    {pre_nms_detection_count}")
    print(f"  errored tiles:           {n_tiles_errored}/{n_tiles_attempted}")
    print(f"  tokens (in/out):         {total_tokens_in} / {total_tokens_out}")
    print(f"  estimated cost:          ${est['estimated_cost_usd']:.4f}")
    print(f"  actual cost (from tokens): ${actual_cost:.4f}")


def render_preview(
    cropped, observations, bounds, aoi_transform, crs_wkt, out_path, thumb_side=1600,
):
    """Render the cropped AOI with detection centroid markers (hollow red circles)."""
    # Contrast-stretch p2-p98 on non-zero pixels, same pattern as 03_detect_sar_umbra.
    valid = cropped[cropped > 0]
    if valid.size > 0:
        lo, hi = np.percentile(valid, [2, 98])
    else:
        lo, hi = 0, 255
    stretched = np.clip(
        (cropped.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0.0, 1.0
    )
    img8 = (stretched * 255).astype(np.uint8)
    rgb = np.stack([img8, img8, img8], axis=-1)
    pil = Image.fromarray(rgb, mode="RGB")

    # Scale factor for the thumbnail; we'll render markers in downsampled space
    # so circle radii stay consistent regardless of source pixel count.
    H, W = cropped.shape[:2]
    scale = thumb_side / max(H, W)
    scaled_w = int(round(W * scale))
    scaled_h = int(round(H * scale))
    pil_small = pil.resize((scaled_w, scaled_h), Image.LANCZOS)
    draw = ImageDraw.Draw(pil_small)

    # Convert each observation back to AOI-local pixel centroid.  The
    # bbox_px field stores scene-space pixels relative to the AOI crop (since
    # we passed aoi_transform and the AOI-local tile_origin).
    from custody.detection.sar_common import latlon_to_pixel

    r_marker = 6
    marker_width = 2
    for obs in observations:
        if obs.bbox_px is not None:
            x1, y1, x2, y2 = obs.bbox_px
            cx_aoi = (x1 + x2) / 2.0
            cy_aoi = (y1 + y2) / 2.0
        else:
            # Fall back to latlon → AOI-pixel projection.
            cy_aoi, cx_aoi = latlon_to_pixel(obs.lat, obs.lon, aoi_transform, crs_wkt)
        sx = cx_aoi * scale
        sy = cy_aoi * scale
        draw.ellipse(
            (sx - r_marker, sy - r_marker, sx + r_marker, sy + r_marker),
            outline=(220, 30, 30), width=marker_width,
        )
    pil_small.save(out_path, optimize=True)


if __name__ == "__main__":
    main()
