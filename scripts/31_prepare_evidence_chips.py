"""Generate dashboard Evidence Viewer assets from real Umbra GEC tiles +
committed VLM detection parquets.

For each scene that has a VLM detection parquet:

  1. Read the source GEC TIFF.
  2. Apply the same AOI crop the VLM driver applied (Tennent: 1 km half-box
     around ``sceneCenterPointLla``; Whitsun: full scene).
  3. Log-stretch + percentile-stretch to a grayscale RGBA PNG (R=G=B,
     nodata alpha=0).
  4. Write:
       src/app/assets/evidence/<scene_id>/raw.png
       src/app/assets/evidence/<scene_id>/annotated.png
     where annotated.png has red bbox rectangles + a small confidence
     label per detection drawn on the same downsampled base.
  5. Emit ``data/demo/evidence_manifest.fixture.json`` with one
     ``EvidenceScene`` record per VLM scene (scenario_id, collection_time,
     image paths, detection list with id / lat / lon / confidence /
     reasoning / bbox / vessel_length_est_m).

This is a deterministic build step.  No live HTTP, no inference, no
decision-layer mutation.  Sentinel scenes are not touched here — the
Evidence Viewer is Umbra-only because Sentinel previews are not
committed.

Run::

    uv run python scripts/31_prepare_evidence_chips.py
    uv run python scripts/31_prepare_evidence_chips.py --force
    uv run python scripts/31_prepare_evidence_chips.py --max-width 1280
"""
from __future__ import annotations

import argparse
import json as _json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from custody.detection.sar_common import crop_to_aoi, read_geotiff  # noqa: E402

PARQUET_DIR = REPO_ROOT / "data" / "processed" / "vlm_detections"
RAW_ROOT = (
    REPO_ROOT / "data" / "raw" / "umbra" / "sar-data"
    / "tasks" / "ship_detection_testdata"
)
EVIDENCE_DIR = REPO_ROOT / "src" / "app" / "assets" / "evidence"
MANIFEST_PATH = REPO_ROOT / "data" / "demo" / "evidence_manifest.fixture.json"


# ---------------------------------------------------------------------------
# Per-scene config: VLM driver convention drives the asset reproduction.
# ---------------------------------------------------------------------------


# Each entry pins:
#   parquet_filename  — relative to PARQUET_DIR
#   gec_glob          — globbed under RAW_ROOT/**/<scene_dir>/<glob>
#   scenario_id       — whitsun | tennent
#   scene_id          — short stable id used in URLs / asset paths
#   collection_time   — YYYY-MM-DD string (display)
#   crop              — "aoi" | "full" — Tennent VLM scripts crop to a 1 km
#                       half-box around sceneCenterPointLla; Whitsun runs
#                       the full scene.
#   whitsun_event_ordinal — for Whitsun replay ordinal gating; None for
#                       Tennent / off-trace Whitsun scenes.
SCENE_CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "scene_id": "tennent-2023-07-02",
        "scenario_id": "tennent",
        "collection_time": "2023-07-02",
        "parquet_filename": "position.parquet",
        "scene_subdir": "f0730a1d-2bf7-4193-b2fe-ec8bfb2e1aef/2023-07-02-14-00-55_UMBRA-05",
        "gec_filename": "2023-07-02-14-00-55_UMBRA-05_GEC.tif",
        "metadata_filename": "2023-07-02-14-00-55_UMBRA-05_METADATA.json",
        "crop": "aoi",
        "aoi_half_km": 1.0,
        "whitsun_event_ordinal": None,
    },
    {
        "scene_id": "tennent-2023-07-23",
        "scenario_id": "tennent",
        "collection_time": "2023-07-23",
        "parquet_filename": "tennent_20230723_position.parquet",
        "scene_subdir": "a9fceda5-7fa5-4849-b76b-dd6d7a50dae9/2023-07-23-14-02-49_UMBRA-05",
        "gec_filename": "2023-07-23-14-02-49_UMBRA-05_GEC.tif",
        "metadata_filename": "2023-07-23-14-02-49_UMBRA-05_METADATA.json",
        "crop": "aoi",
        "aoi_half_km": 1.0,
        "whitsun_event_ordinal": None,
    },
    {
        "scene_id": "tennent-2023-08-07",
        "scenario_id": "tennent",
        "collection_time": "2023-08-07",
        "parquet_filename": "tennent_20230807_position.parquet",
        "scene_subdir": "00dee081-76ee-413a-8af2-e6be1c6ab8d0/2023-08-07-13-53-00_UMBRA-04",
        "gec_filename": "2023-08-07-13-53-00_UMBRA-04_GEC.tif",
        "metadata_filename": "2023-08-07-13-53-00_UMBRA-04_METADATA.json",
        "crop": "aoi",
        "aoi_half_km": 1.0,
        "whitsun_event_ordinal": None,
    },
    {
        "scene_id": "tennent-2023-08-13",
        "scenario_id": "tennent",
        "collection_time": "2023-08-13",
        "parquet_filename": "tennent_20230813_position.parquet",
        "scene_subdir": "645d903f-ced5-40d9-94ed-6e4e91977c97/2023-08-13-15-00-09_UMBRA-06",
        "gec_filename": "2023-08-13-15-00-09_UMBRA-06_GEC.tif",
        "metadata_filename": "2023-08-13-15-00-09_UMBRA-06_METADATA.json",
        "crop": "aoi",
        "aoi_half_km": 1.0,
        "whitsun_event_ordinal": None,
    },
    {
        # Whitsun VLM driver used the EARLIER 2023-12-06 pass (02-06-24);
        # the dashboard map uses the LATER 02-07-18 pass.  The Evidence
        # Viewer must show the same scene the VLM analysed, otherwise
        # the bboxes wouldn't line up.
        "scene_id": "whitsun-2023-12-06",
        "scenario_id": "whitsun",
        "collection_time": "2023-12-06",
        "parquet_filename": "whitsun_20231206_position.parquet",
        "scene_subdir": "1e1f051d-4c81-4640-997d-1a03e967ad6a/2023-12-06-02-06-24_UMBRA-04",
        "gec_filename": "2023-12-06-02-06-24_UMBRA-04_GEC.tif",
        "metadata_filename": "2023-12-06-02-06-24_UMBRA-04_METADATA.json",
        "crop": "full",
        "aoi_half_km": None,
        "whitsun_event_ordinal": 2,
    },
    {
        "scene_id": "whitsun-2024-03-20",
        "scenario_id": "whitsun",
        "collection_time": "2024-03-20",
        "parquet_filename": "whitsun_20240320_position.parquet",
        "scene_subdir": "92c2b446-415c-4d26-88bc-70fdb652e852/2024-03-20-02-09-55_UMBRA-05",
        "gec_filename": "2024-03-20-02-09-55_UMBRA-05_GEC.tif",
        "metadata_filename": "2024-03-20-02-09-55_UMBRA-05_METADATA.json",
        "crop": "full",
        "aoi_half_km": None,
        "whitsun_event_ordinal": None,  # off-trace; not surfaced in Whitsun replay
    },
)


# ---------------------------------------------------------------------------
# SAR → grayscale RGBA helpers (mirrors scripts/30_prepare_demo_overlays.py).
# ---------------------------------------------------------------------------


def _stretch_to_grayscale_rgba(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Log-stretch SAR amplitude to grayscale uint8 + binary alpha mask.

    Returns ``(gray_uint8, alpha_uint8)``.  Valid pixels are alpha=255 and
    R=G=B=stretched_uint8; nodata is alpha=0.
    """
    arr = arr.astype("float32")
    valid = np.isfinite(arr) & (arr > 0)
    log_arr = np.zeros_like(arr)
    log_arr[valid] = np.log10(arr[valid] + 1.0)
    valid_values = log_arr[valid]
    if valid_values.size == 0:
        gray = np.zeros(arr.shape, dtype="uint8")
        alpha = np.zeros(arr.shape, dtype="uint8")
        return gray, alpha
    p2 = float(np.nanpercentile(valid_values, 2.0))
    p98 = float(np.nanpercentile(valid_values, 98.0))
    if p98 > p2:
        stretched = np.clip((log_arr - p2) / (p98 - p2), 0.0, 1.0)
    else:
        stretched = np.zeros_like(log_arr)
    gray = (stretched * 255.0).astype("uint8")
    gray = np.where(valid, gray, 0)
    alpha = np.where(valid, 255, 0).astype("uint8")
    return gray, alpha


def _downsample(arr: np.ndarray, scale: int) -> np.ndarray:
    """Block-mean downsample by an integer factor.  Preserves dtype."""
    if scale <= 1:
        return arr
    h, w = arr.shape
    nh, nw = h // scale, w // scale
    if nh == 0 or nw == 0:
        return arr
    trimmed = arr[: nh * scale, : nw * scale]
    return (
        trimmed.reshape(nh, scale, nw, scale)
        .mean(axis=(1, 3))
        .astype(arr.dtype)
    )


def _downsample_alpha(alpha: np.ndarray, scale: int) -> np.ndarray:
    """Downsample a binary alpha by max-pooling so any valid sub-pixel
    keeps the destination pixel valid."""
    if scale <= 1:
        return alpha
    h, w = alpha.shape
    nh, nw = h // scale, w // scale
    if nh == 0 or nw == 0:
        return alpha
    trimmed = alpha[: nh * scale, : nw * scale]
    return (
        trimmed.reshape(nh, scale, nw, scale)
        .max(axis=(1, 3))
        .astype(alpha.dtype)
    )


# ---------------------------------------------------------------------------
# Detection annotation helpers.
# ---------------------------------------------------------------------------


def _draw_detection_annotations(
    base_rgba: np.ndarray,
    detections: list[dict[str, Any]],
    scale: int,
) -> Image.Image:
    """Return an annotated PIL Image with red bbox rectangles + small
    confidence labels.  Coordinates in ``detections`` are scene/AOI-space
    pixels at the original resolution; we divide by ``scale`` to map onto
    the downsampled base.
    """
    out = Image.fromarray(base_rgba, mode="RGBA").convert("RGB")
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("arial.ttf", size=12)
    except (OSError, IOError):
        font = ImageFont.load_default()
    for det in detections:
        x1 = float(det["bbox_x1"]) / scale
        y1 = float(det["bbox_y1"]) / scale
        x2 = float(det["bbox_x2"]) / scale
        y2 = float(det["bbox_y2"]) / scale
        conf = det.get("confidence")
        # Red outlines at 2 px so they survive screenshare.
        draw.rectangle((x1, y1, x2, y2), outline=(239, 68, 68), width=2)
        if conf is not None:
            label = f"{float(conf):.2f}"
            tw = draw.textlength(label, font=font)
            th = 12
            tx = x1
            ty = max(0.0, y1 - th - 2)
            draw.rectangle(
                (tx, ty, tx + tw + 4, ty + th + 2),
                fill=(0, 0, 0),
            )
            draw.text((tx + 2, ty), label, fill=(239, 68, 68), font=font)
    return out


def _build_detection_records(
    df: pd.DataFrame,
    crop_offset: tuple[int, int],
    scenario_id: str,
    scene_id: str,
) -> list[dict[str, Any]]:
    """Project parquet rows into Evidence-Viewer-ready dicts.

    ``crop_offset = (col_start, row_start)`` translates AOI-cropped
    coordinates back to scene coordinates if needed.  For our purposes
    we expose AOI-local bbox values because the Evidence Viewer renders
    on the AOI-cropped image — which is the same image the VLM saw.
    """
    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        # Pandas / numpy types -> JSON-friendly Python scalars.
        def _maybe_float(v):
            if v is None or pd.isna(v):
                return None
            return float(v)

        records.append({
            "detection_id": str(row.get("obs_id") or ""),
            "source_id": str(row.get("source_id") or ""),
            "scene_id": scene_id,
            "scenario_id": scenario_id,
            "lat": _maybe_float(row.get("lat")),
            "lon": _maybe_float(row.get("lon")),
            "confidence": _maybe_float(row.get("classification_conf")),
            "vessel_length_est_m": _maybe_float(row.get("vessel_length_est_m")),
            "heading_est_deg": _maybe_float(row.get("heading_est_deg")),
            "bbox_x1": _maybe_float(row.get("bbox_x1")),
            "bbox_y1": _maybe_float(row.get("bbox_y1")),
            "bbox_x2": _maybe_float(row.get("bbox_x2")),
            "bbox_y2": _maybe_float(row.get("bbox_y2")),
            "reasoning": str(row.get("detector_reasoning") or ""),
            "detector_version": str(row.get("detector_version") or ""),
        })
    return records


# ---------------------------------------------------------------------------
# Per-scene driver.
# ---------------------------------------------------------------------------


def _resolve_scene_paths(cfg: dict[str, Any]) -> tuple[Path, Path]:
    base = RAW_ROOT / cfg["scene_subdir"]
    return base / cfg["gec_filename"], base / cfg["metadata_filename"]


def _resolve_aoi_target(meta_path: Path) -> tuple[float, float]:
    meta = _json.loads(meta_path.read_text(encoding="utf-8"))
    coords = meta["collects"][0]["sceneCenterPointLla"]["coordinates"]
    target_lon, target_lat = float(coords[0]), float(coords[1])
    return target_lat, target_lon


def _build_scene(
    cfg: dict[str, Any],
    *,
    parquet_path: Path,
    max_width: int,
    force: bool,
) -> dict[str, Any] | None:
    """Generate raw + annotated PNGs for one scene; return its manifest dict."""
    gec_path, meta_path = _resolve_scene_paths(cfg)
    if not gec_path.exists():
        print(f"  {cfg['scene_id']}: GEC missing at {gec_path}")
        return None
    if not parquet_path.exists():
        print(f"  {cfg['scene_id']}: parquet missing at {parquet_path}")
        return None

    out_dir = EVIDENCE_DIR / cfg["scene_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw.png"
    annotated_path = out_dir / "annotated.png"

    df = pd.read_parquet(parquet_path)
    if cfg["crop"] == "aoi":
        target_lat, target_lon = _resolve_aoi_target(meta_path)
        img, transform, crs_wkt = read_geotiff(gec_path)
        cropped, _, bounds = crop_to_aoi(
            img, transform, target_lat, target_lon, crs_wkt,
            box_half_km=float(cfg["aoi_half_km"]),
        )
        del img
        crop_offset = (bounds["col_start"], bounds["row_start"])
        full_arr = cropped
    else:
        # Whitsun VLM driver did not crop — read the raw band directly.
        with rasterio.open(gec_path) as src:
            full_arr = src.read(1)
        crop_offset = (0, 0)

    full_h, full_w = full_arr.shape[:2]
    scale = max(1, full_w // max_width)
    while max(full_h // scale, full_w // scale) > max_width:
        scale += 1
    print(
        f"  {cfg['scene_id']}: full={full_h}x{full_w}, "
        f"downsample {scale}x -> {full_h // scale}x{full_w // scale}"
    )

    gray_full, alpha_full = _stretch_to_grayscale_rgba(full_arr)
    del full_arr
    gray = _downsample(gray_full, scale)
    alpha = _downsample_alpha(alpha_full, scale)
    rgba = np.stack([gray, gray, gray, alpha], axis=-1)
    out_h, out_w = gray.shape

    if force or not raw_path.exists():
        # Save as ``LA`` (grayscale + alpha) — about half the disk
        # footprint of RGBA, identical pixel content because R=G=B.
        Image.fromarray(
            np.stack([gray, alpha], axis=-1), mode="LA",
        ).save(str(raw_path), format="PNG", optimize=True)
    detections = _build_detection_records(
        df, crop_offset, cfg["scenario_id"], cfg["scene_id"],
    )
    if force or not annotated_path.exists():
        annotated = _draw_detection_annotations(rgba, detections, scale)
        annotated.save(str(annotated_path), format="PNG", optimize=True)

    asset_base = f"/assets/evidence/{cfg['scene_id']}"
    return {
        "scene_id": cfg["scene_id"],
        "scenario_id": cfg["scenario_id"],
        "collection_time": cfg["collection_time"],
        "whitsun_event_ordinal": cfg["whitsun_event_ordinal"],
        "raw_image_path": str(raw_path.relative_to(REPO_ROOT).as_posix()),
        "raw_asset_url": f"{asset_base}/raw.png",
        "annotated_image_path": str(
            annotated_path.relative_to(REPO_ROOT).as_posix()
        ),
        "annotated_asset_url": f"{asset_base}/annotated.png",
        "image_width_px": out_w,
        "image_height_px": out_h,
        "downsample_scale": scale,
        "source_parquet": str(parquet_path.relative_to(REPO_ROOT).as_posix()),
        "source_gec": str(gec_path.relative_to(REPO_ROOT).as_posix()),
        "detector_version": (
            str(detections[0].get("detector_version") or "")
            if detections else ""
        ),
        "detection_count": len(detections),
        "detections": detections,
    }


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate dashboard Evidence Viewer PNG assets from Umbra GEC "
            "tiles + committed VLM detection parquets."
        ),
    )
    parser.add_argument(
        "--max-width", dest="max_width", type=int, default=1280,
        help="Downsampled preview width in pixels (default: 1280).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Regenerate PNG assets even when they already exist.",
    )
    args = parser.parse_args(argv)

    if not RAW_ROOT.is_dir():
        print(
            f"raw Umbra inventory not found at {RAW_ROOT}",
            file=sys.stderr,
        )
        return 2

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    scenes: list[dict[str, Any]] = []
    print(f"discovered {len(SCENE_CONFIGS)} configured Evidence Viewer scenes:")
    for cfg in SCENE_CONFIGS:
        parquet_path = PARQUET_DIR / cfg["parquet_filename"]
        rec = _build_scene(
            cfg, parquet_path=parquet_path,
            max_width=args.max_width, force=args.force,
        )
        if rec is not None:
            scenes.append(rec)

    payload = {
        "schema": "custody.demo.evidence.v1",
        "metadata": {
            "data_mode": "fixture",
            "description": (
                "Evidence Viewer manifest.  One record per VLM-analysed "
                "Umbra scene: raw grayscale RGBA preview, an annotated "
                "preview with bbox rectangles + confidence labels, and "
                "the per-detection metadata pulled from the VLM "
                "parquet outputs in data/processed/vlm_detections/.  "
                "Generated deterministically from real GEC tiles by "
                "scripts/31_prepare_evidence_chips.py."
            ),
            "regenerate_command": (
                "uv run python scripts/31_prepare_evidence_chips.py"
            ),
            "caveats": [
                "previews are log-stretched amplitude grayscale RGBA PNGs (R=G=B, nodata alpha=0)",
                "bbox rectangles are exactly the boxes the VLM emitted; no resampling, no fake boxes",
                "vessel_length_est_m / heading_est_deg are NULL in current parquets",
                "Sentinel scenes are NOT included; Sentinel remains weak-signal cueing/context",
                "Umbra is the high-confidence confirmation/evidence layer",
            ],
        },
        "scenes": scenes,
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        _json.dumps(payload, indent=2) + "\n", encoding="utf-8",
    )
    print(
        f"\nwrote {MANIFEST_PATH.relative_to(REPO_ROOT)}: "
        f"{len(scenes)} scenes "
        f"({sum(s['detection_count'] for s in scenes)} detections total)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
