"""Tennent VLM run: 2023-08-07. UMBRA-04 at ~0.52 m/px (cross-resolution test, 2-day pair with 0809)

Same AOI geometry + backend + pipeline config as scripts/05 / 07.

Outputs (uncommitted pending review):
    data/processed/vlm_detections/tennent_20230807_position.parquet
    day0/scratch/tennent_20230807_vlm_progress.jsonl
    day0/scratch/tennent_20230807_vlm_summary.json
    day0/scratch/tennent_20230807_vlm_detections.png
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

from custody.detection.quality import assess_scene_quality  # noqa: E402
from custody.detection.sar_common import crop_to_aoi, read_geotiff  # noqa: E402
from custody.detection.tiling import describe_aoi_bounds  # noqa: E402
from custody.detection.vlm_backends import AnthropicBackend  # noqa: E402
from custody.detection.vlm_sar import (  # noqa: E402
    detect_vessels_in_scene,
    estimate_scene_cost,
)
from custody.fusion.index import index_observations  # noqa: E402


SCENE_DIR = (
    REPO_ROOT
    / "data/raw/umbra/sar-data/tasks/ship_detection_testdata"
    / "00dee081-76ee-413a-8af2-e6be1c6ab8d0/2023-08-07-13-53-00_UMBRA-04"
)
GEC_PATH = SCENE_DIR / "2023-08-07-13-53-00_UMBRA-04_GEC.tif"
META_PATH = SCENE_DIR / "2023-08-07-13-53-00_UMBRA-04_METADATA.json"

AOI_CROP_KM_HALF_WIDTH = 1.0
TILE_SIZE = 640
TILE_OVERLAP = 64
CONFIDENCE_THRESHOLD = "medium"
NMS_IOU_THRESHOLD = 0.5
MAX_COST_USD = 3.00

# Auto-approval gates per task spec
GATE_TILE_COUNT = (30, 200)
GATE_EST_COST = (0.3, 3.0)
GATE_CENTER_LAT = (8.80, 8.90)
GATE_CENTER_LON = (114.61, 114.71)

OUT_DIR = REPO_ROOT / "data/processed/vlm_detections"
SCRATCH = REPO_ROOT / "day0/scratch"
PARQUET_PATH = OUT_DIR / "tennent_20230807_position.parquet"
PROGRESS_LOG = SCRATCH / "tennent_20230807_vlm_progress.jsonl"
SUMMARY_JSON = SCRATCH / "tennent_20230807_vlm_summary.json"
PREVIEW_PNG = SCRATCH / "tennent_20230807_vlm_detections.png"


def main() -> None:
    t_start = time.perf_counter()

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
    del img

    # ---- AOI geographic description ----
    aoi_local_bounds = (0, cropped.shape[0], 0, cropped.shape[1])
    aoi_info = describe_aoi_bounds(
        aoi_bounds=aoi_local_bounds,
        transform=aoi_transform, crs_wkt=crs_wkt,
    )
    center_lat, center_lon = aoi_info["center_latlon"]
    ns_km, ew_km = aoi_info["approx_extent_km"]
    print()
    print("=== AOI geographic description ===")
    print(f"  center latlon: ({center_lat:.6f}, {center_lon:.6f})")
    print(f"  approx extent: {ns_km:.2f} km N-S x {ew_km:.2f} km E-W")

    # ---- cost estimate ----
    backend = AnthropicBackend(model="claude-sonnet-4-6")
    est = estimate_scene_cost(
        cropped, backend,
        tile_size=TILE_SIZE, tile_overlap=TILE_OVERLAP,
        skip_empty=True,
    )
    print()
    print("=== Cost estimate (claude-sonnet-4-6, tile=640, overlap=64) ===")
    print(f"  tile_count:         {est['tile_count']}")
    print(f"  estimated_cost_usd: ${est['estimated_cost_usd']:.4f}")
    print(f"  estimate_type:      {est['estimate_type']}")

    # ---- Auto-approval gate ----
    gate_fail: list[str] = []
    if not (GATE_TILE_COUNT[0] <= est["tile_count"] <= GATE_TILE_COUNT[1]):
        gate_fail.append(f"tile_count {est['tile_count']} outside {GATE_TILE_COUNT}")
    if not (GATE_EST_COST[0] <= est["estimated_cost_usd"] <= GATE_EST_COST[1]):
        gate_fail.append(f"estimated_cost ${est['estimated_cost_usd']:.2f} outside [${GATE_EST_COST[0]}, ${GATE_EST_COST[1]}]")
    if not (GATE_CENTER_LAT[0] <= center_lat <= GATE_CENTER_LAT[1]):
        gate_fail.append(f"center lat {center_lat:.4f} outside {GATE_CENTER_LAT}")
    if not (GATE_CENTER_LON[0] <= center_lon <= GATE_CENTER_LON[1]):
        gate_fail.append(f"center lon {center_lon:.4f} outside {GATE_CENTER_LON}")

    print()
    if gate_fail:
        print("=== AUTO-APPROVAL GATE FAILED ===")
        for reason in gate_fail:
            print(f"  FAIL: {reason}")
        print("\nAborting before any backend calls.")
        sys.exit(2)
    print("=== Auto-approval gates all pass; proceeding to real run ===")
    print(f"  tile count {est['tile_count']} in {GATE_TILE_COUNT}")
    print(f"  estimated cost ${est['estimated_cost_usd']:.2f} in [${GATE_EST_COST[0]}, ${GATE_EST_COST[1]}]")
    print(f"  center ({center_lat:.4f}, {center_lon:.4f}) in "
          f"({GATE_CENTER_LAT[0]}-{GATE_CENTER_LAT[1]} N, {GATE_CENTER_LON[0]}-{GATE_CENTER_LON[1]} E)")

    # ---- Scene quality screen (ADR-0018 sub-decision) ----
    aoi_bounds = (
        bounds["row_start"], bounds["row_end"],
        bounds["col_start"], bounds["col_end"],
    )
    quality = assess_scene_quality(GEC_PATH, aoi_bounds=aoi_bounds)
    print()
    print("=== Scene quality screen ===")
    print(f"  full-scene  p99={quality.full_scene_stats.p99:.1f}  "
          f"dynamic_range={quality.full_scene_stats.dynamic_range:.1f}")
    if quality.aoi_stats is not None:
        print(f"  AOI         p99={quality.aoi_stats.p99:.1f}  "
              f"dynamic_range={quality.aoi_stats.dynamic_range:.1f}")
    print(f"  quality_flag: {quality.quality_flag.upper()}")
    for reason in quality.flag_reasons:
        print(f"    - {reason}")
    if quality.quality_flag == "red":
        if "--force" in sys.argv:
            print("  NOTE: --force flag set; proceeding despite red quality flag.")
        else:
            print("\nAborting: scene quality flagged red.  Re-run with --force to proceed anyway.")
            sys.exit(3)
    elif quality.quality_flag == "yellow":
        print("  WARNING: proceeding despite yellow flag.")

    # ---- Progress callback ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
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
        tokens = (f"{rec['tokens_in']}/{rec['tokens_out']}" if rec["tokens_in"] is not None else "err/err")
        status = "ERROR" if rec["errored"] else "ok"
        print(
            f"[tennent_20230807] [tile {done:>3}/{total}] {status}  origin=({rec['origin_row']:>5},{rec['origin_col']:>5})  "
            f"wall={wt:>5.1f}s  tokens={tokens}  detections={rec['detection_count']}"
        )

    print(f"\n=== Running detect_vessels_in_scene (prompt=contextualized_v1, "
          f"confidence>={CONFIDENCE_THRESHOLD}, NMS IoU>={NMS_IOU_THRESHOLD}) ===\n")

    t_det_start = time.perf_counter()
    try:
        observations = detect_vessels_in_scene(
            cropped,
            transform=aoi_transform,
            crs_wkt=crs_wkt,
            acquisition_time=acq_epoch,
            source_id="umbra-2023-08-07-vlm",
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

    # ---- Parquet output — per-scene filename ----
    if PARQUET_PATH.exists():
        PARQUET_PATH.unlink()
    tmp_dir = OUT_DIR / "_tmp_tennent_20230807"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for stale in (tmp_dir / "position.parquet", tmp_dir / "posvel.parquet"):
        if stale.exists():
            stale.unlink()
    index_observations(observations, out_dir=tmp_dir)
    (tmp_dir / "position.parquet").rename(PARQUET_PATH)
    posvel = tmp_dir / "posvel.parquet"
    if posvel.exists():
        posvel.unlink()
    try:
        tmp_dir.rmdir()
    except OSError:
        pass
    print(f"Wrote: {PARQUET_PATH.relative_to(REPO_ROOT)} ({PARQUET_PATH.stat().st_size/1024:.1f} KB)")

    # ---- Aggregate stats ----
    n_tiles_attempted = len(tile_records)
    n_tiles_errored = sum(1 for r in tile_records if r["errored"])
    pre_nms_detection_count = sum(r["detection_count"] for r in tile_records)
    total_tokens_in = sum((r["tokens_in"] or 0) for r in tile_records)
    total_tokens_out = sum((r["tokens_out"] or 0) for r in tile_records)
    wall_times = [r["wall_time_seconds"] for r in tile_records if r["wall_time_seconds"] is not None]
    mean_wall = sum(wall_times) / len(wall_times) if wall_times else 0.0

    INPUT_PER_M = 3.0
    OUTPUT_PER_M = 15.0
    actual_cost = total_tokens_in * INPUT_PER_M / 1e6 + total_tokens_out * OUTPUT_PER_M / 1e6

    top5 = sorted(
        (r for r in tile_records if not r["errored"]),
        key=lambda r: -r["detection_count"],
    )[:5]
    top5_report = [
        {"tile_index": r["tile_index"], "origin": [r["origin_row"], r["origin_col"]],
         "detection_count": r["detection_count"]}
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
        "parquet_path": str(PARQUET_PATH.relative_to(REPO_ROOT).as_posix()),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote: {SUMMARY_JSON.relative_to(REPO_ROOT)}")

    print("Rendering preview...")
    render_preview(cropped, observations, aoi_transform, crs_wkt, PREVIEW_PNG, thumb_side=1600)
    print(f"Wrote: {PREVIEW_PNG.relative_to(REPO_ROOT)} ({PREVIEW_PNG.stat().st_size/1024:.1f} KB)")

    total_wall = time.perf_counter() - t_start
    print()
    print("=== RUN COMPLETE ===")
    print(f"  total wall (incl. I/O):  {total_wall:.1f}s")
    print(f"  detection wall:          {det_wall:.1f}s ({det_wall/60:.1f} min)")
    print(f"  observations (post-NMS): {len(observations)}")
    print(f"  detections (pre-NMS):    {pre_nms_detection_count}")
    print(f"  errored tiles:           {n_tiles_errored}/{n_tiles_attempted}")
    print(f"  tokens (in/out):         {total_tokens_in} / {total_tokens_out}")
    print(f"  estimated cost:          ${est['estimated_cost_usd']:.4f}")
    print(f"  actual cost (from tokens): ${actual_cost:.4f}")


def render_preview(cropped, observations, transform, crs_wkt, out_path, thumb_side=1600):
    valid = cropped[cropped > 0]
    lo, hi = np.percentile(valid, [2, 98]) if valid.size else (0, 255)
    stretched = np.clip((cropped.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    img8 = (stretched * 255).astype(np.uint8)
    del stretched
    H, W = cropped.shape[:2]
    scale = thumb_side / max(H, W)
    scaled_w, scaled_h = int(round(W * scale)), int(round(H * scale))
    pil_L = Image.fromarray(img8, mode="L").resize((scaled_w, scaled_h), Image.LANCZOS)
    del img8
    pil_small = pil_L.convert("RGB")
    del pil_L
    draw = ImageDraw.Draw(pil_small)

    from custody.detection.sar_common import latlon_to_pixel

    r_marker = 6
    marker_width = 2
    for obs in observations:
        if obs.bbox_px is not None:
            x1, y1, x2, y2 = obs.bbox_px
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        else:
            cy, cx = latlon_to_pixel(obs.lat, obs.lon, transform, crs_wkt)
        sx, sy = cx * scale, cy * scale
        draw.ellipse((sx - r_marker, sy - r_marker, sx + r_marker, sy + r_marker),
                     outline=(220, 30, 30), width=marker_width)
    pil_small.save(out_path, optimize=True)


if __name__ == "__main__":
    main()
