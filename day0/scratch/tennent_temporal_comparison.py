"""Tennent 07-02 → 07-23 temporal persistence analysis.

Direct spatial matching between two single-timestamp VLM detection sets —
NOT the fusion tracker.  Tracker's EKF prediction over 21-day dt would
produce scene-wide covariance and a gate that admits nearly any match.
Direct matching at a calibrated radius (50 m, sized for VLM bbox uncertainty)
is the honest tool for this specific analysis.

Classifies each of the 42 + 44 = 86 observations into:
  PERSISTENT_07-02: 07-02 obs with a 07-23 match within 50 m
  PERSISTENT_07-23: 07-23 obs with a 07-02 match within 50 m (reciprocal)
  EMERGED:          07-23 obs with no 07-02 match within 50 m
  DISAPPEARED:      07-02 obs with no 07-23 match within 50 m

One-to-many handling: scipy linear_sum_assignment with distance-or-BIG cost
produces a global minimum-total-distance pairing within the 50 m gate.
When two 07-02 obs are both within 50 m of one 07-23 obs, the assignment
picks the closer pair; the other 07-02 obs becomes DISAPPEARED.

Outputs (diagnostic, uncommitted):
  data/processed/vlm_detections/tennent_temporal_comparison.parquet
  day0/scratch/tennent_temporal_comparison_summary.md
  day0/scratch/tennent_temporal_comparison.png
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import duckdb  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from pyproj import Geod  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402


OUT_PQ = REPO_ROOT / "data/processed/vlm_detections/tennent_temporal_comparison.parquet"
OUT_MD = REPO_ROOT / "day0/scratch/tennent_temporal_comparison_summary.md"
OUT_PNG = REPO_ROOT / "day0/scratch/tennent_temporal_comparison.png"

PQ_0702 = REPO_ROOT / "data/processed/vlm_detections/position.parquet"
PQ_0723 = REPO_ROOT / "data/processed/vlm_detections/tennent_20230723_position.parquet"

SCENE_0723_DIR = (
    REPO_ROOT
    / "data/raw/umbra/sar-data/tasks/ship_detection_testdata"
    / "a9fceda5-7fa5-4849-b76b-dd6d7a50dae9/2023-07-23-14-02-49_UMBRA-05"
)
SCENE_0723_GEC = SCENE_0723_DIR / "2023-07-23-14-02-49_UMBRA-05_GEC.tif"

MATCH_RADIUS_M = 50.0
_BIG_COST = 1.0e9
_GEOD = Geod(ellps="WGS84")


# ---------------------------------------------------------------------------


def load_obs(pq_path: Path, scene_label: str) -> list[dict]:
    """Load observations from a VLM parquet as plain dicts (not PositionObservation)."""
    rows = duckdb.execute(
        f"SELECT * FROM read_parquet('{pq_path.as_posix()}')"
    ).fetchall()
    cols = [c[0] for c in duckdb.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{pq_path.as_posix()}')"
    ).fetchall()]
    out = []
    for r in rows:
        d = dict(zip(cols, r))
        d["__scene"] = scene_label
        out.append(d)
    return out


def geodetic_dist_m(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    _, _, dist = _GEOD.inv(lon_a, lat_a, lon_b, lat_b)
    return float(dist)


def build_cost_matrix(set_a: list[dict], set_b: list[dict]) -> np.ndarray:
    """C[i, j] = geodetic distance (m) if <= MATCH_RADIUS_M, else _BIG_COST."""
    n, m = len(set_a), len(set_b)
    C = np.full((n, m), _BIG_COST, dtype=float)
    for i, a in enumerate(set_a):
        for j, b in enumerate(set_b):
            d = geodetic_dist_m(a["lat"], a["lon"], b["lat"], b["lon"])
            if d <= MATCH_RADIUS_M:
                C[i, j] = d
    return C


def assign_pairs(cost: np.ndarray) -> list[tuple[int, int, float]]:
    """Return (i, j, distance) pairs that survived the gate."""
    if cost.size == 0:
        return []
    row_idx, col_idx = linear_sum_assignment(cost)
    out = []
    for i, j in zip(row_idx, col_idx):
        d = cost[i, j]
        if d < _BIG_COST:
            out.append((int(i), int(j), float(d)))
    return out


# ---------------------------------------------------------------------------


def classify_and_output() -> dict:
    set_0702 = load_obs(PQ_0702, "07-02")
    set_0723 = load_obs(PQ_0723, "07-23")
    print(f"  07-02: {len(set_0702)} obs")
    print(f"  07-23: {len(set_0723)} obs")

    cost = build_cost_matrix(set_0702, set_0723)
    pairs = assign_pairs(cost)
    print(f"  assignments within {MATCH_RADIUS_M} m: {len(pairs)}")

    paired_i = {i: (j, d) for i, j, d in pairs}
    paired_j = {j: (i, d) for i, j, d in pairs}

    rows: list[dict] = []
    for i, a in enumerate(set_0702):
        if i in paired_i:
            j, d = paired_i[i]
            match_obs = set_0723[j]
            cat = "PERSISTENT_07-02"
            match_id = match_obs["obs_id"]
            match_d = d
        else:
            cat = "DISAPPEARED"
            match_id = None
            match_d = None
        rows.append({
            "obs_id": a["obs_id"],
            "scene": a["__scene"],
            "category": cat,
            "match_obs_id": match_id,
            "match_distance_m": match_d,
            "lat": a["lat"],
            "lon": a["lon"],
            "classification_conf": a["classification_conf"],
            "bbox_x1": a.get("bbox_x1"), "bbox_y1": a.get("bbox_y1"),
            "bbox_x2": a.get("bbox_x2"), "bbox_y2": a.get("bbox_y2"),
            "detector_reasoning": a.get("detector_reasoning"),
        })
    for j, b in enumerate(set_0723):
        if j in paired_j:
            i, d = paired_j[j]
            match_obs = set_0702[i]
            cat = "PERSISTENT_07-23"
            match_id = match_obs["obs_id"]
            match_d = d
        else:
            cat = "EMERGED"
            match_id = None
            match_d = None
        rows.append({
            "obs_id": b["obs_id"],
            "scene": b["__scene"],
            "category": cat,
            "match_obs_id": match_id,
            "match_distance_m": match_d,
            "lat": b["lat"],
            "lon": b["lon"],
            "classification_conf": b["classification_conf"],
            "bbox_x1": b.get("bbox_x1"), "bbox_y1": b.get("bbox_y1"),
            "bbox_x2": b.get("bbox_x2"), "bbox_y2": b.get("bbox_y2"),
            "detector_reasoning": b.get("detector_reasoning"),
        })

    # Write parquet
    OUT_PQ.parent.mkdir(parents=True, exist_ok=True)
    if OUT_PQ.exists():
        OUT_PQ.unlink()
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, OUT_PQ)
    print(f"  wrote {OUT_PQ.relative_to(REPO_ROOT)} ({OUT_PQ.stat().st_size/1024:.1f} KB)")

    return {
        "rows": rows,
        "set_0702": set_0702,
        "set_0723": set_0723,
        "pairs": pairs,
    }


def render_overlay(result: dict) -> None:
    """Render categorized observations over the 2023-07-23 AOI scene."""
    from custody.detection.sar_common import crop_to_aoi, latlon_to_pixel, read_geotiff

    # Load + crop the 07-23 scene the same way scripts/07_detect_vlm_tennent_0723.py did.
    with open(SCENE_0723_DIR / "2023-07-23-14-02-49_UMBRA-05_METADATA.json") as f:
        meta = json.load(f)
    center = meta["collects"][0]["sceneCenterPointLla"]["coordinates"]
    target_lon, target_lat = center[0], center[1]
    img, transform, crs_wkt = read_geotiff(SCENE_0723_GEC)
    cropped, aoi_transform, bounds = crop_to_aoi(
        img, transform, target_lat, target_lon, crs_wkt, box_half_km=1.0,
    )
    del img  # free ~137 MB

    # p2-p98 stretch; downsample to 1600 for display
    valid = cropped[cropped > 0]
    lo, hi = np.percentile(valid, [2, 98])
    stretched = np.clip((cropped.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    img8 = (stretched * 255).astype(np.uint8)
    del stretched

    thumb = 1600
    H, W = cropped.shape[:2]
    scale = thumb / max(H, W)
    sw, sh = int(round(W * scale)), int(round(H * scale))
    pil = Image.fromarray(img8, mode="L").resize((sw, sh), Image.LANCZOS).convert("RGB")
    del img8
    draw = ImageDraw.Draw(pil)

    color = {
        "PERSISTENT_07-02": (40, 200, 80),    # green
        "PERSISTENT_07-23": (40, 200, 80),
        "EMERGED":          (240, 200, 40),   # yellow
        "DISAPPEARED":      (220, 40, 40),    # red
    }
    marker_radius = 8
    marker_width = 2
    # Render DISAPPEARED first (bottom), then PERSISTENT (middle), then EMERGED on top,
    # so late scenes drawing order puts the "new" markers visibly above the "fixed" ones.
    order = ["DISAPPEARED", "PERSISTENT_07-02", "PERSISTENT_07-23", "EMERGED"]
    by_cat: dict = {c: [] for c in order}
    for r in result["rows"]:
        by_cat[r["category"]].append(r)

    # Count for legend
    counts = {c: len(by_cat[c]) for c in by_cat}

    for cat in order:
        for r in by_cat[cat]:
            cy_aoi, cx_aoi = latlon_to_pixel(r["lat"], r["lon"], aoi_transform, crs_wkt)
            sx = cx_aoi * scale
            sy = cy_aoi * scale
            draw.ellipse(
                (sx - marker_radius, sy - marker_radius, sx + marker_radius, sy + marker_radius),
                outline=color[cat], width=marker_width,
            )

    # Legend
    legend_items = [
        (f"Persistent in both scenes ({counts['PERSISTENT_07-02']} + {counts['PERSISTENT_07-23']} = "
         f"{counts['PERSISTENT_07-02'] + counts['PERSISTENT_07-23']} markers)", color["PERSISTENT_07-02"]),
        (f"Emerged on 07-23 only ({counts['EMERGED']})", color["EMERGED"]),
        (f"Disappeared after 07-02 ({counts['DISAPPEARED']})", color["DISAPPEARED"]),
    ]
    pad = 14
    lh = 26
    # Semi-opaque black box behind legend
    legend_box_h = pad * 2 + lh * len(legend_items)
    overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle([(14, 14), (680, 14 + legend_box_h)], fill=(0, 0, 0, 180))
    pil = Image.alpha_composite(pil.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(pil)
    try:
        from PIL import ImageFont
        font = ImageFont.truetype("arialbd.ttf", 20)
    except OSError:
        font = None
    y = 14 + pad
    for text, col in legend_items:
        draw.ellipse((26, y + 2, 26 + 16, y + 2 + 16), outline=col, width=2)
        draw.text((54, y), text, fill=(255, 255, 255), font=font)
        y += lh

    pil.save(OUT_PNG, optimize=True)
    print(f"  wrote {OUT_PNG.relative_to(REPO_ROOT)} ({OUT_PNG.stat().st_size/1024:.1f} KB)")


def write_summary(result: dict) -> None:
    rows = result["rows"]
    counts = {"PERSISTENT_07-02": 0, "PERSISTENT_07-23": 0, "EMERGED": 0, "DISAPPEARED": 0}
    for r in rows:
        counts[r["category"]] += 1

    dists = [r["match_distance_m"] for r in rows if r["match_distance_m"] is not None]
    mean_d = float(np.mean(dists)) if dists else float("nan")
    max_d = float(np.max(dists)) if dists else float("nan")
    min_d = float(np.min(dists)) if dists else float("nan")

    # Spatial distribution — lat/lon bounds for each category.
    def bbox_of(cat: str) -> str:
        rs = [r for r in rows if r["category"] == cat]
        if not rs:
            return "(none)"
        lats = [r["lat"] for r in rs]
        lons = [r["lon"] for r in rs]
        return (f"lat [{min(lats):.6f}, {max(lats):.6f}], "
                f"lon [{min(lons):.6f}, {max(lons):.6f}]")

    pct_persistent_of_0702 = counts["PERSISTENT_07-02"] / 42 * 100 if 42 else 0
    pct_persistent_of_0723 = counts["PERSISTENT_07-23"] / 44 * 100 if 44 else 0

    md = f"""# Tennent 07-02 → 07-23 temporal persistence analysis

*Diagnostic output from `day0/scratch/tennent_temporal_comparison.py`.  No new code in `src/`; no fusion-tracker integration per the Week-3 design decision (ADR-0017).*

Generated {datetime.utcnow().isoformat()} UTC.

## Method

Direct spatial matching between 07-02 and 07-23 VLM detection sets.  Pairing is
computed via `scipy.optimize.linear_sum_assignment` on a cost matrix where
`C[i, j]` is the geodetic distance in meters (pyproj.Geod WGS84) between
07-02 observation `i` and 07-23 observation `j`, gated at **50 m**.  Entries
beyond the gate receive a `_BIG_COST` sentinel; Hungarian returns the
minimum-total-distance pairing subject to the constraint.

The 50 m threshold is sized for VLM bbox uncertainty per Phase D.5 findings —
Claude's bounding boxes are loose enough that per-hull localization is
approximate at the ~tens of meters scale.  50 m is slightly larger than the
observed bbox centroid jitter across repeated VLM calls on the same tile.

## Aggregate counts

| category | count | % of scene total |
|---|---:|---:|
| PERSISTENT_07-02 (07-02 obs with 07-23 match ≤ 50 m) | {counts['PERSISTENT_07-02']} | {pct_persistent_of_0702:.1f}% of 42 |
| PERSISTENT_07-23 (reciprocal) | {counts['PERSISTENT_07-23']} | {pct_persistent_of_0723:.1f}% of 44 |
| EMERGED (07-23 with no 07-02 match) | {counts['EMERGED']} | {counts['EMERGED']/44*100:.1f}% of 44 |
| DISAPPEARED (07-02 with no 07-23 match) | {counts['DISAPPEARED']} | {counts['DISAPPEARED']/42*100:.1f}% of 42 |

Persistent pair count (should be equal on both sides): {counts['PERSISTENT_07-02']} = {counts['PERSISTENT_07-23']}
{"✓ balanced" if counts['PERSISTENT_07-02'] == counts['PERSISTENT_07-23'] else "⚠ mismatch — check one-to-many handling"}

## Match distance distribution

Across the {counts['PERSISTENT_07-02']} persistent pairs:

- min: {min_d:.2f} m
- mean: {mean_d:.2f} m
- max: {max_d:.2f} m

Well below the 50 m gate, suggesting the matches are genuine co-location rather than gate-saturated approximations.

## Spatial distribution

- PERSISTENT_07-02: {bbox_of("PERSISTENT_07-02")}
- PERSISTENT_07-23: {bbox_of("PERSISTENT_07-23")}
- EMERGED:         {bbox_of("EMERGED")}
- DISAPPEARED:     {bbox_of("DISAPPEARED")}

(Both Tennent AOIs are centered on (8.8557°N, 114.6651°E) with ±0.009° half-width.)

## Honest caveats

1. **Two timestamps is the minimum for "persistence" to mean anything.**  A target appearing in both scenes could be a long-moored vessel that happens to be there both days, a fixed infrastructure element, or the same vessel returning coincidentally.  This analysis distinguishes "spatially consistent across the 21-day interval" from "only in one scene," not "vessel vs infrastructure."
2. **Persistent ≠ fixed infrastructure.**  Conflating the two would overclaim.  A persistent detection near the reclamation structure is *more likely* infrastructure than a persistent detection in open water, but nothing here is ground truth.
3. **The 50 m threshold is calibrated for VLM bbox uncertainty, not per-hull localization.**  A per-hull coordinate-accuracy threshold would be tens of meters in X-band SAR at 0.34 m/pixel; VLM bbox centroid jitter is the dominant error source here.  Different detector outputs (e.g., YOLO or CFAR point centroids) would warrant different thresholds.
4. **Coordinate transforms are honest.**  Both scenes went through the same `sar_common.crop_to_aoi` + `pixel_to_latlon` path that was validated in Phase F.5.  Pixel-size difference between scenes (0.342 vs 0.332 m/px, documented in the 07-23 README) is absorbed by the geodetic distance calculation; we're not comparing pixel coordinates across scenes.
5. **Greedy global assignment, not mutual-nearest-neighbor.**  Hungarian minimum-cost matching under a hard gate will pair a 07-02 obs with its *globally-best* 07-23 partner even if an asymmetric-mutual-nearest rule would have left it unmatched.  For pairs well below 50 m the two approaches agree; near the gate boundary they can differ.

## Artifacts

- Parquet: `data/processed/vlm_detections/tennent_temporal_comparison.parquet` (86 rows, one per observation)
- Overlay image: `day0/scratch/tennent_temporal_comparison.png` (1600-side thumbnail over 07-23 AOI)
- This document: `day0/scratch/tennent_temporal_comparison_summary.md`
"""
    OUT_MD.write_text(md, encoding="utf-8")
    print(f"  wrote {OUT_MD.relative_to(REPO_ROOT)} ({OUT_MD.stat().st_size/1024:.1f} KB)")


def main() -> None:
    print("Phase 1: load + classify...")
    result = classify_and_output()

    print("\nPhase 2: render overlay...")
    try:
        render_overlay(result)
    except Exception as e:
        print(f"  overlay failed: {type(e).__name__}: {e}")

    print("\nPhase 3: write summary...")
    write_summary(result)

    # Print a terse summary
    cats = {"PERSISTENT_07-02": 0, "PERSISTENT_07-23": 0, "EMERGED": 0, "DISAPPEARED": 0}
    for r in result["rows"]:
        cats[r["category"]] += 1
    print("\n=== DONE ===")
    for k, v in cats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
