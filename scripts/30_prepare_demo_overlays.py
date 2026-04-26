"""Generate dashboard map overlay assets from real Umbra GEC TIFFs.

Walks ``data/raw/umbra/sar-data/tasks/ship_detection_testdata/``, reads
each scene's ``*_METADATA.json`` and ``*_GEC.tif``, classifies it as
Whitsun or Tennent by ``sceneCenterPointLla`` proximity, and emits:

  - a downsampled, log-stretched PNG preview at
    ``src/app/assets/overlays/<scene_id>.png`` (Dash auto-serves it
    under ``/assets/overlays/<scene_id>.png``)
  - an updated ``data/demo/map_overlays.fixture.json`` that combines
    real Umbra image overlays + AOI polygons + the existing
    Sentinel / simulated entries (still footprint-only)

This is a deterministic build step.  No live HTTP, no remote fetch,
no inference, no decision-layer mutation.

Run::

    uv run python scripts/30_prepare_demo_overlays.py
    uv run python scripts/30_prepare_demo_overlays.py --force
    uv run python scripts/30_prepare_demo_overlays.py --max-width 1024
"""
from __future__ import annotations

import argparse
import json as _json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.image as mpimg
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = (
    REPO_ROOT / "data" / "raw" / "umbra" / "sar-data" / "tasks"
    / "ship_detection_testdata"
)
ASSETS_DIR = REPO_ROOT / "src" / "app" / "assets" / "overlays"
MANIFEST_PATH = REPO_ROOT / "data" / "demo" / "map_overlays.fixture.json"

WHITSUN_CENTER = (9.98, 114.63)
TENNENT_CENTER = (8.8583, 114.6561)

WHITSUN_AOI_BBOX = [114.45, 9.75, 114.85, 10.25]
TENNENT_AOI_BBOX = [114.55, 8.78, 114.75, 8.93]


# ---------------------------------------------------------------------------
# Per-Whitsun-event ordinal mapping (from data/demo/whitsun_decision_trace.fixture.json)
# ---------------------------------------------------------------------------


WHITSUN_EVENT_ORDINAL = {
    # Real Umbra observation revealed at event 02.
    "20231206": 2,
}


# ---------------------------------------------------------------------------
# Sentinel + simulated entries that the script preserves verbatim because
# no real image asset exists for them in this slice.
# ---------------------------------------------------------------------------


_SHARED_AOI_GEOMETRY_WHITSUN = {
    "type": "Polygon",
    "coordinates": [[
        [114.45, 9.75], [114.85, 9.75],
        [114.85, 10.25], [114.45, 10.25],
        [114.45, 9.75],
    ]],
}

_SHARED_AOI_GEOMETRY_TENNENT = {
    "type": "Polygon",
    "coordinates": [[
        [114.55, 8.78], [114.75, 8.78],
        [114.75, 8.93], [114.55, 8.93],
        [114.55, 8.78],
    ]],
}


_NON_UMBRA_OVERLAYS: list[dict[str, Any]] = [
    {
        "overlay_id": "whitsun-aoi",
        "scenario_id": "whitsun",
        "observation_id": None,
        "source": "simulated",
        "sensor_type": None,
        "display_name": "Whitsun AOI",
        "data_mode": "fixture",
        "image_path": None,
        "asset_url": None,
        "image_kind": "footprint-only",
        "bounds": WHITSUN_AOI_BBOX,
        "geometry": _SHARED_AOI_GEOMETRY_WHITSUN,
        "opacity_default": 0.6,
        "visible_from_event_ordinal": 1,
        "z_index": 10,
        "confidence_weight": None,
        "usable_for_detection": None,
        "usable_for_context": None,
        "caveats": ["AOI polygon, not imagery"],
        "missing_asset_reason": None,
    },
    {
        "overlay_id": "whitsun-sentinel-2-low-cloud-20231212",
        "scenario_id": "whitsun",
        "observation_id": "fixture-s2-l2a-whitsun-20231212-low-cloud",
        "source": "sentinel-2",
        "sensor_type": "optical",
        "display_name": "Sentinel-2 L2A 2023-12-12 (8% cloud)",
        "data_mode": "fixture",
        "image_path": None,
        "asset_url": None,
        "image_kind": "footprint-only",
        "bounds": WHITSUN_AOI_BBOX,
        "geometry": _SHARED_AOI_GEOMETRY_WHITSUN,
        "opacity_default": 0.55,
        "visible_from_event_ordinal": 4,
        "z_index": 25,
        "confidence_weight": 0.30,
        "usable_for_detection": True,
        "usable_for_context": True,
        "caveats": ["low cloud Sentinel-2; lower confidence than Umbra"],
        "missing_asset_reason": (
            "no committed Sentinel-2 RGB preview; footprint-only"
        ),
    },
    {
        "overlay_id": "whitsun-sentinel-1-grd-20231210",
        "scenario_id": "whitsun",
        "observation_id": "fixture-s1-grd-whitsun-20231210",
        "source": "sentinel-1",
        "sensor_type": "sar",
        "display_name": "Sentinel-1 GRD 2023-12-10",
        "data_mode": "fixture",
        "image_path": None,
        "asset_url": None,
        "image_kind": "footprint-only",
        "bounds": WHITSUN_AOI_BBOX,
        "geometry": _SHARED_AOI_GEOMETRY_WHITSUN,
        "opacity_default": 0.55,
        "visible_from_event_ordinal": 5,
        "z_index": 20,
        "confidence_weight": 0.45,
        "usable_for_detection": True,
        "usable_for_context": True,
        "caveats": ["public SAR context; lower confidence than Umbra"],
        "missing_asset_reason": (
            "no committed Sentinel-1 GRD raster; footprint-only"
        ),
    },
    {
        "overlay_id": "whitsun-sentinel-2-cloudy-20231215",
        "scenario_id": "whitsun",
        "observation_id": "fixture-s2-l2a-whitsun-20231215-cloudy",
        "source": "sentinel-2",
        "sensor_type": "optical",
        "display_name": "Sentinel-2 L2A 2023-12-15 (72% cloud)",
        "data_mode": "fixture",
        "image_path": None,
        "asset_url": None,
        "image_kind": "footprint-only",
        "bounds": WHITSUN_AOI_BBOX,
        "geometry": _SHARED_AOI_GEOMETRY_WHITSUN,
        "opacity_default": 0.40,
        "visible_from_event_ordinal": 4,
        "z_index": 15,
        "confidence_weight": 0.10,
        "usable_for_detection": False,
        "usable_for_context": True,
        "caveats": ["high cloud cover; usable_for_detection is false"],
        "missing_asset_reason": (
            "no committed Sentinel-2 RGB preview; footprint-only"
        ),
    },
    {
        "overlay_id": "whitsun-umbra-followup-20231213",
        "scenario_id": "whitsun",
        "observation_id": "fixture-umbra-whitsun-20231213-followup",
        "source": "umbra",
        "sensor_type": "sar",
        "display_name": "Umbra SAR 2023-12-13 (simulated follow-up)",
        "data_mode": "simulated",
        "image_path": None,
        "asset_url": None,
        "image_kind": "footprint-only",
        "bounds": WHITSUN_AOI_BBOX,
        "geometry": _SHARED_AOI_GEOMETRY_WHITSUN,
        "opacity_default": 0.65,
        "visible_from_event_ordinal": 12,
        "z_index": 35,
        "confidence_weight": 1.0,
        "usable_for_detection": True,
        "usable_for_context": True,
        "caveats": ["simulated follow-up Umbra collect; demo scaffolding"],
        "missing_asset_reason": "simulated record; no upstream raster",
    },
    {
        "overlay_id": "tennent-aoi",
        "scenario_id": "tennent",
        "observation_id": None,
        "source": "simulated",
        "sensor_type": None,
        "display_name": "Tennent AOI",
        "data_mode": "fixture",
        "image_path": None,
        "asset_url": None,
        "image_kind": "footprint-only",
        "bounds": TENNENT_AOI_BBOX,
        "geometry": _SHARED_AOI_GEOMETRY_TENNENT,
        "opacity_default": 0.6,
        "visible_from_event_ordinal": 0,
        "z_index": 10,
        "confidence_weight": None,
        "usable_for_detection": None,
        "usable_for_context": None,
        "caveats": ["AOI polygon, not imagery"],
        "missing_asset_reason": None,
    },
    {
        "overlay_id": "tennent-sentinel-1-grd-20230715",
        "scenario_id": "tennent",
        "observation_id": "fixture-s1-grd-tennent-20230715",
        "source": "sentinel-1",
        "sensor_type": "sar",
        "display_name": "Sentinel-1 GRD 2023-07-15",
        "data_mode": "fixture",
        "image_path": None,
        "asset_url": None,
        "image_kind": "footprint-only",
        "bounds": TENNENT_AOI_BBOX,
        "geometry": _SHARED_AOI_GEOMETRY_TENNENT,
        "opacity_default": 0.55,
        "visible_from_event_ordinal": 0,
        "z_index": 20,
        "confidence_weight": 0.45,
        "usable_for_detection": True,
        "usable_for_context": True,
        "caveats": ["public SAR context"],
        "missing_asset_reason": (
            "no committed Sentinel-1 GRD raster; footprint-only"
        ),
    },
    {
        "overlay_id": "tennent-sentinel-2-low-cloud-20230718",
        "scenario_id": "tennent",
        "observation_id": "fixture-s2-l2a-tennent-20230718-low-cloud",
        "source": "sentinel-2",
        "sensor_type": "optical",
        "display_name": "Sentinel-2 L2A 2023-07-18 (12% cloud)",
        "data_mode": "fixture",
        "image_path": None,
        "asset_url": None,
        "image_kind": "footprint-only",
        "bounds": TENNENT_AOI_BBOX,
        "geometry": _SHARED_AOI_GEOMETRY_TENNENT,
        "opacity_default": 0.55,
        "visible_from_event_ordinal": 0,
        "z_index": 25,
        "confidence_weight": 0.30,
        "usable_for_detection": True,
        "usable_for_context": True,
        "caveats": ["low cloud Sentinel-2"],
        "missing_asset_reason": (
            "no committed Sentinel-2 RGB preview; footprint-only"
        ),
    },
    {
        "overlay_id": "tennent-sentinel-2-cloudy-20230728",
        "scenario_id": "tennent",
        "observation_id": "fixture-s2-l2a-tennent-20230728-cloudy",
        "source": "sentinel-2",
        "sensor_type": "optical",
        "display_name": "Sentinel-2 L2A 2023-07-28 (85% cloud)",
        "data_mode": "fixture",
        "image_path": None,
        "asset_url": None,
        "image_kind": "footprint-only",
        "bounds": TENNENT_AOI_BBOX,
        "geometry": _SHARED_AOI_GEOMETRY_TENNENT,
        "opacity_default": 0.40,
        "visible_from_event_ordinal": 0,
        "z_index": 15,
        "confidence_weight": 0.10,
        "usable_for_detection": False,
        "usable_for_context": True,
        "caveats": ["high cloud cover; usable_for_detection is false"],
        "missing_asset_reason": (
            "no committed Sentinel-2 RGB preview; footprint-only"
        ),
    },
]


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _near(lat: float, lon: float, target: tuple[float, float], tol: float = 0.30) -> bool:
    return abs(lat - target[0]) < tol and abs(lon - target[1]) < tol


def discover_umbra_scenes() -> list[dict[str, Any]]:
    """Return scenario-classified Umbra scenes from the raw inventory."""
    if not RAW_ROOT.is_dir():
        return []
    matched: list[dict[str, Any]] = []
    for task_dir in sorted(RAW_ROOT.iterdir()):
        if not task_dir.is_dir():
            continue
        for scene_dir in task_dir.iterdir():
            if not scene_dir.is_dir():
                continue
            metas = list(scene_dir.glob("*_METADATA.json"))
            gecs = list(scene_dir.glob("*_GEC.tif"))
            if not metas or not gecs:
                continue
            try:
                meta = _json.loads(metas[0].read_text(encoding="utf-8"))
            except _json.JSONDecodeError:
                continue
            collects = meta.get("collects") or []
            if not collects:
                continue
            c0 = collects[0]
            center = c0.get("sceneCenterPointLla", {}).get("coordinates")
            if not center or len(center) < 2:
                continue
            lon, lat = float(center[0]), float(center[1])
            if _near(lat, lon, WHITSUN_CENTER):
                scenario = "whitsun"
            elif _near(lat, lon, TENNENT_CENTER):
                scenario = "tennent"
            else:
                continue
            matched.append({
                "scenario_id": scenario,
                "scene_id": scene_dir.name,
                "task_id": task_dir.name,
                "metadata_path": metas[0],
                "gec_path": gecs[0],
                "metadata": meta,
                "collect": c0,
            })
    matched.sort(key=lambda s: (
        s["scenario_id"],
        s["collect"].get("startAtUTC", ""),
    ))
    return matched


# ---------------------------------------------------------------------------
# Preview generation
# ---------------------------------------------------------------------------


def generate_preview(
    gec_path: Path,
    out_path: Path,
    *,
    max_width: int = 768,
    force: bool = False,
) -> tuple[float, float, float, float]:
    """Reproject to EPSG:4326, downsample, log-stretch, write PNG.

    Returns the EPSG:4326 axis-aligned bounds ``(west, south, east, north)``.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(gec_path) as src:
        with WarpedVRT(src, crs="EPSG:4326") as vrt:
            full_w, full_h = vrt.width, vrt.height
            scale = max(1, full_w // max_width)
            out_w = max(1, full_w // scale)
            out_h = max(1, full_h // scale)
            west, south, east, north = vrt.bounds
            bounds = (
                float(west), float(south),
                float(east), float(north),
            )
            if out_path.exists() and not force:
                return bounds
            data = vrt.read(
                1,
                out_shape=(out_h, out_w),
                resampling=Resampling.average,
            )

    arr = data.astype("float32")
    # SAR amplitude — log-stretch to compress dynamic range.
    arr = np.where(arr > 0, np.log10(arr + 1.0), 0.0)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        # Empty / all-masked tile — fall back to mid-grey.
        img = np.full((out_h, out_w), 128, dtype="uint8")
    else:
        p2 = float(np.nanpercentile(finite, 2.0))
        p98 = float(np.nanpercentile(finite, 98.0))
        if p98 > p2:
            stretched = np.clip((arr - p2) / (p98 - p2), 0.0, 1.0)
        else:
            stretched = np.zeros_like(arr)
        img = (stretched * 255.0).astype("uint8")
    mpimg.imsave(str(out_path), img, cmap="gray", format="png")
    return bounds


# ---------------------------------------------------------------------------
# Manifest assembly
# ---------------------------------------------------------------------------


def _whitsun_overlay_for(
    scene: dict[str, Any], bounds: tuple[float, float, float, float],
    asset_url: str, image_path: str,
) -> dict[str, Any]:
    collect = scene["collect"]
    start = str(collect.get("startAtUTC") or "")
    yyyymmdd = "".join(start[:10].split("-")) if len(start) >= 10 else "00000000"
    polygon = collect.get("footprintPolygonLla", {}).get("coordinates")
    geometry = (
        {"type": "Polygon",
         "coordinates": [
             [[float(p[0]), float(p[1])] for p in polygon[0]]
         ]}
        if polygon and polygon[0] else None
    )
    sat = str(scene["metadata"].get("umbraSatelliteName") or "")
    ipr = (
        scene["metadata"].get("baseIpr")
        or scene["metadata"].get("targetIpr")
    )
    label_ipr = f", {float(ipr):.2f} m" if ipr else ""
    overlay_id = f"whitsun-umbra-{yyyymmdd}"
    visible_from = WHITSUN_EVENT_ORDINAL.get(yyyymmdd, 0)
    return {
        "overlay_id": overlay_id,
        "scenario_id": "whitsun",
        "observation_id": f"fixture-umbra-whitsun-{yyyymmdd}",
        "source": "umbra",
        "sensor_type": "sar",
        "display_name": (
            f"Umbra SAR {start[:10]} ({sat.replace('UMBRA_', 'UMBRA-')}{label_ipr})"
        ),
        "data_mode": "real",
        "image_path": image_path,
        "asset_url": asset_url,
        "image_kind": "png",
        "bounds": list(bounds),
        "geometry": geometry,
        "opacity_default": 0.85,
        "visible_from_event_ordinal": visible_from,
        "z_index": 30,
        "confidence_weight": 1.0,
        "usable_for_detection": True,
        "usable_for_context": True,
        "caveats": [
            "real Umbra GEC tile, log-stretched and downsampled for dashboard preview",
        ],
        "missing_asset_reason": None,
    }


def _tennent_overlay_for(
    scene: dict[str, Any], bounds: tuple[float, float, float, float],
    asset_url: str, image_path: str,
) -> dict[str, Any]:
    collect = scene["collect"]
    start = str(collect.get("startAtUTC") or "")
    yyyymmdd = "".join(start[:10].split("-")) if len(start) >= 10 else "00000000"
    polygon = collect.get("footprintPolygonLla", {}).get("coordinates")
    geometry = (
        {"type": "Polygon",
         "coordinates": [
             [[float(p[0]), float(p[1])] for p in polygon[0]]
         ]}
        if polygon and polygon[0] else None
    )
    sat = str(scene["metadata"].get("umbraSatelliteName") or "")
    ipr = (
        scene["metadata"].get("baseIpr")
        or scene["metadata"].get("targetIpr")
    )
    label_ipr = f", {float(ipr):.2f} m" if ipr else ""
    return {
        "overlay_id": f"tennent-umbra-{yyyymmdd}",
        "scenario_id": "tennent",
        "observation_id": f"real-umbra-tennent-{yyyymmdd}",
        "source": "umbra",
        "sensor_type": "sar",
        "display_name": (
            f"Umbra SAR {start[:10]} ({sat.replace('UMBRA_', 'UMBRA-')}{label_ipr})"
        ),
        "data_mode": "real",
        "image_path": image_path,
        "asset_url": asset_url,
        "image_kind": "png",
        "bounds": list(bounds),
        "geometry": geometry,
        "opacity_default": 0.85,
        "visible_from_event_ordinal": 0,
        "z_index": 30,
        "confidence_weight": 1.0,
        "usable_for_detection": True,
        "usable_for_context": True,
        "caveats": [
            "real Umbra GEC tile, log-stretched and downsampled for dashboard preview",
        ],
        "missing_asset_reason": None,
    }


def build_manifest(
    real_overlays: list[dict[str, Any]],
    *,
    discovered_count: int,
) -> dict[str, Any]:
    payload = {
        "schema": "custody.demo.map_overlays.v1",
        "metadata": {
            "data_mode": "fixture",
            "description": (
                "Map overlay manifest for the Whitsun replay and Tennent "
                "monitoring tabs.  Real Umbra GEC tiles drive the Umbra "
                "overlays via downsampled PNG previews under "
                "src/app/assets/overlays/.  Sentinel and simulated "
                "entries remain footprint-only until imagery is "
                "committed; the renderer surfaces the missing-asset "
                "reason verbatim.  Regenerate with "
                "scripts/30_prepare_demo_overlays.py."
            ),
            "regenerate_command": (
                "uv run python scripts/30_prepare_demo_overlays.py"
            ),
            "discovered_umbra_scenes": discovered_count,
            "imagery_acquisition_needs": [
                "Whitsun Sentinel-1 GRD 2023-12-10 — PNG preview + bounds, or COG",
                "Whitsun Sentinel-2 L2A 2023-12-12 (low cloud) — RGB PNG preview + bounds, or COG",
                "Whitsun Sentinel-2 L2A 2023-12-15 (cloudy) — same",
                "Whitsun Umbra repeat 2023-12-13 (simulated follow-up) — optional",
                "Tennent Sentinel-1 GRD 2023-07-15 — PNG preview + bounds, or COG",
                "Tennent Sentinel-2 L2A 2023-07-18 (low cloud) — same",
                "Tennent Sentinel-2 L2A 2023-07-28 (cloudy) — same",
            ],
            "caveats": [
                "deterministic demo manifest, regenerated from raw Umbra inventory",
                "Umbra previews are log-stretched amplitude PNGs, not science-grade calibrated SAR products",
                "confidence_weight values are demo heuristics, not calibrated reliability",
                "Sentinel observations are public lower-confidence context, not equivalent to Umbra",
            ],
        },
        "overlays": _NON_UMBRA_OVERLAYS + real_overlays,
    }
    return payload


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate dashboard map overlay PNG previews from the real "
            "Umbra GEC inventory and rewrite the overlay manifest."
        ),
    )
    parser.add_argument(
        "--max-width", dest="max_width", type=int, default=768,
        help="Downsampled preview width in pixels (default: 768).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Regenerate PNG previews even when they already exist.",
    )
    args = parser.parse_args(argv)

    if not RAW_ROOT.is_dir():
        print(
            f"raw Umbra inventory not found at {RAW_ROOT}",
            file=sys.stderr,
        )
        return 2

    scenes = discover_umbra_scenes()
    if not scenes:
        print("no Whitsun or Tennent Umbra scenes discovered", file=sys.stderr)
        return 1

    # Dedup same-day collects deterministically: keep the LAST collect
    # of any same-(scenario, yyyymmdd) pair so the better-centered
    # second 2023-12-06 Whitsun pass wins over the earlier one.
    keepers: dict[tuple[str, str], dict[str, Any]] = {}
    for s in scenes:
        start = str(s["collect"].get("startAtUTC") or "")
        yyyymmdd = (
            "".join(start[:10].split("-")) if len(start) >= 10 else "00000000"
        )
        key = (s["scenario_id"], yyyymmdd)
        prev = keepers.get(key)
        if prev is None or str(prev["collect"].get("startAtUTC") or "") < start:
            keepers[key] = s
    selected = list(keepers.values())
    skipped = len(scenes) - len(selected)

    real_overlays: list[dict[str, Any]] = []
    print(
        f"discovered {len(scenes)} Umbra scenes "
        f"({len(selected)} kept after dedup, {skipped} duplicates):"
    )
    for s in selected:
        scene_id = s["scene_id"]
        scenario = s["scenario_id"]
        start = str(s["collect"].get("startAtUTC") or "")
        yyyymmdd = (
            "".join(start[:10].split("-")) if len(start) >= 10 else "00000000"
        )
        # Whitsun: only emit overlays that map to a trace event ordinal.
        # Dates not in the December 2023 trace (e.g. the March 2024
        # Whitsun-area collect) are honest real data but not part of the
        # current Whitsun replay; they would clutter the timeline.
        if scenario == "whitsun" and yyyymmdd not in WHITSUN_EVENT_ORDINAL:
            skipped += 1
            print(
                f"  {scenario:8s} {scene_id} -> skipped "
                f"(date {yyyymmdd} not referenced by the Whitsun trace)"
            )
            continue
        out_name = f"{scenario}-{scene_id}.png"
        out_path = ASSETS_DIR / out_name
        try:
            bounds = generate_preview(
                s["gec_path"], out_path,
                max_width=args.max_width, force=args.force,
            )
        except Exception as exc:
            print(f"  {scenario:8s} {scene_id} -> FAILED: {exc}")
            continue
        rel = out_path.relative_to(REPO_ROOT).as_posix()
        asset_url = f"/assets/overlays/{out_name}"
        if scenario == "whitsun":
            entry = _whitsun_overlay_for(s, bounds, asset_url, rel)
        else:
            entry = _tennent_overlay_for(s, bounds, asset_url, rel)
        real_overlays.append(entry)
        print(
            f"  {scenario:8s} {scene_id} -> {out_path.name} "
            f"(bounds {bounds[0]:.4f},{bounds[1]:.4f} -> "
            f"{bounds[2]:.4f},{bounds[3]:.4f})"
        )

    manifest = build_manifest(real_overlays, discovered_count=len(scenes))
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        _json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )
    print(
        f"\nwrote {MANIFEST_PATH.relative_to(REPO_ROOT)}: "
        f"{len(manifest['overlays'])} overlays "
        f"({len(real_overlays)} real Umbra, "
        f"{len(_NON_UMBRA_OVERLAYS)} footprint-only / simulated; "
        f"{skipped} duplicate Whitsun collect skipped)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
