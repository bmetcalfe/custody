"""Pre-flight for 4-scene batch dispatch.

Reports per-scene tile count, cost estimate, AOI center, and runs the
auto-approval gate.  No API calls made.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from custody.detection.sar_common import crop_to_aoi, read_geotiff  # noqa: E402
from custody.detection.tiling import describe_aoi_bounds  # noqa: E402
from custody.detection.vlm_backends import AnthropicBackend  # noqa: E402
from custody.detection.vlm_sar import estimate_scene_cost  # noqa: E402


SCENE_ROOT = REPO_ROOT / "data/raw/umbra/sar-data/tasks/ship_detection_testdata"

# Tennent scenes: 2 km AOI crop; Whitsun: full scene.
SCENES = [
    {
        "label": "tennent_20230807",
        "uuid": "00dee081-76ee-413a-8af2-e6be1c6ab8d0",
        "scene_dir_name": "2023-08-07-13-53-00_UMBRA-04",
        "reef": "tennent", "sensor": "UMBRA-04", "aoi_half_km": 1.0,
    },
    {
        "label": "tennent_20230809",
        "uuid": "b422de0d-b6fc-4340-aa85-6c87f3805175",
        "scene_dir_name": "2023-08-09-02-23-25_UMBRA-06",
        "reef": "tennent", "sensor": "UMBRA-06", "aoi_half_km": 1.0,
    },
    {
        "label": "tennent_20230813",
        "uuid": "645d903f-ced5-40d9-94ed-6e4e91977c97",
        "scene_dir_name": "2023-08-13-15-00-09_UMBRA-06",
        "reef": "tennent", "sensor": "UMBRA-06", "aoi_half_km": 1.0,
    },
    {
        "label": "whitsun_20240320",
        "uuid": "92c2b446-415c-4d26-88bc-70fdb652e852",
        "scene_dir_name": "2024-03-20-02-09-55_UMBRA-05",
        "reef": "whitsun", "sensor": "UMBRA-05", "aoi_half_km": None,  # full scene
    },
]


def preflight_scene(s: dict) -> dict:
    scene_dir = SCENE_ROOT / s["uuid"] / s["scene_dir_name"]
    gec = next(scene_dir.glob("*_GEC.tif"))
    meta_path = next(scene_dir.glob("*_METADATA.json"))
    with meta_path.open() as f:
        meta = json.load(f)
    collect = meta["collects"][0]
    target_lon, target_lat = collect["sceneCenterPointLla"]["coordinates"][:2]

    img, transform, crs_wkt = read_geotiff(gec)
    full_shape = img.shape

    if s["aoi_half_km"] is not None:
        cropped, aoi_transform, bounds = crop_to_aoi(
            img, transform, target_lat, target_lon, crs_wkt,
            box_half_km=s["aoi_half_km"],
        )
        use_img = cropped
        use_transform = aoi_transform
        pixel_size_m = bounds["pixel_size_m"]
    else:
        use_img = img
        use_transform = transform
        pixel_size_m = float(abs(transform.a))

    H, W = use_img.shape[:2]
    aoi_info = describe_aoi_bounds(
        aoi_bounds=(0, H, 0, W),
        transform=use_transform, crs_wkt=crs_wkt,
    )

    backend = AnthropicBackend(model="claude-sonnet-4-6", api_key="dummy")
    est = estimate_scene_cost(
        use_img, backend,
        tile_size=640, tile_overlap=64, skip_empty=True,
    )

    return {
        "label": s["label"],
        "reef": s["reef"],
        "sensor": s["sensor"],
        "gec_size_mb": round(gec.stat().st_size / 1e6, 1),
        "acq_utc": collect["startAtUTC"],
        "scene_center_latlon": [target_lat, target_lon],
        "full_scene_shape": list(full_shape),
        "aoi_shape": list(use_img.shape),
        "pixel_size_m": pixel_size_m,
        "aoi_half_km": s["aoi_half_km"],
        "center_latlon_via_helper": aoi_info["center_latlon"],
        "approx_extent_km": aoi_info["approx_extent_km"],
        "tile_count": est["tile_count"],
        "est_cost_usd": est["estimated_cost_usd"],
    }


def main() -> None:
    results = [preflight_scene(s) for s in SCENES]
    total_cost = sum(r["est_cost_usd"] for r in results)

    print()
    print(f'{"scene":<20} {"sensor":<8} {"size_mb":>8} {"aoi_shape":<14} {"tiles":>6} {"est$":>7}  center')
    print("-" * 100)
    for r in results:
        shape_str = f"{r['aoi_shape'][0]}x{r['aoi_shape'][1]}"
        print(f'{r["label"]:<20} {r["sensor"]:<8} {r["gec_size_mb"]:>8} {shape_str:<14} '
              f'{r["tile_count"]:>6} {r["est_cost_usd"]:>7.2f}  ({r["center_latlon_via_helper"][0]:.4f}, '
              f'{r["center_latlon_via_helper"][1]:.4f})')
    print("-" * 100)
    print(f'{"TOTAL":<20} {"":<8} {"":<8} {"":<14} {sum(r["tile_count"] for r in results):>6} {total_cost:>7.2f}')

    # Gate check
    print()
    print("=== Auto-approval gates ===")
    failures: list[str] = []
    for r in results:
        if r["reef"] == "tennent":
            if not (100 <= r["tile_count"] <= 150):
                failures.append(f"{r['label']}: tile count {r['tile_count']} outside [100, 150]")
            if not (1.0 <= r["est_cost_usd"] <= 3.0):
                failures.append(f"{r['label']}: cost ${r['est_cost_usd']:.2f} outside [$1, $3]")
            lat, lon = r["center_latlon_via_helper"]
            if not (8.8 <= lat <= 8.9):
                failures.append(f"{r['label']}: lat {lat:.4f} outside [8.8, 8.9]")
            if not (114.6 <= lon <= 114.7):
                failures.append(f"{r['label']}: lon {lon:.4f} outside [114.6, 114.7]")
        elif r["reef"] == "whitsun":
            if not (800 <= r["tile_count"] <= 1400):
                failures.append(f"{r['label']}: tile count {r['tile_count']} outside [800, 1400]")
            if not (10.0 <= r["est_cost_usd"] <= 25.0):
                failures.append(f"{r['label']}: cost ${r['est_cost_usd']:.2f} outside [$10, $25]")
            lat, lon = r["center_latlon_via_helper"]
            if not (9.5 <= lat <= 10.5):
                failures.append(f"{r['label']}: lat {lat:.4f} outside [9.5, 10.5]")
            if not (114.5 <= lon <= 115.5):
                failures.append(f"{r['label']}: lon {lon:.4f} outside [114.5, 115.5]")

    if not (12.0 <= total_cost <= 35.0):
        failures.append(f"TOTAL cost ${total_cost:.2f} outside [$12, $35]")

    if failures:
        print("GATE FAIL:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(2)
    print(f"ALL GATES PASS (total cost estimate ${total_cost:.2f})")

    # Dump machine-readable summary for the dispatcher
    out = REPO_ROOT / "day0/scratch/batch_preflight_summary.json"
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"Saved: {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
