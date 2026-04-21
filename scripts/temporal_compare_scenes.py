"""Tennent 07-02 → 07-23 temporal persistence classifier — Scene-based driver.

Migrated from the original ``day0/scratch/tennent_temporal_comparison.py`` to
use the ADR-0018 :class:`Scene` abstraction.  The underlying classifier logic
lives in :mod:`custody.fusion.temporal`; this script is a thin driver that:

1. Loads two scenes via :func:`custody.fusion.scenes.load_scene_from_parquet`.
2. Calls :func:`custody.fusion.temporal.temporal_persistence` with
   :class:`DirectSpatialMatcher` at the Week-3 50 m gate.
3. Writes the legacy output parquet schema at
   ``data/processed/vlm_detections/tennent_temporal_comparison.parquet``.

The output parquet is row-equivalent (same obs_id → same category → same
match_obs_id) to the committed one; match_distance_m values differ by
sub-millimeter because the new implementation uses AEQD tangent-plane
instead of the original's pyproj.Geod.  That's expected — AEQD is what the
rest of the fusion layer uses (see ADR-0009, ADR-0010) and tangent-plane is
what ADR-0018 specified.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from custody.fusion.scenes import Scene, load_scene_from_parquet  # noqa: E402
from custody.fusion.temporal import (  # noqa: E402
    DirectSpatialMatcher,
    TemporalComparisonResult,
    temporal_persistence,
)


PQ_A = REPO_ROOT / "data/processed/vlm_detections/position.parquet"
PQ_B = REPO_ROOT / "data/processed/vlm_detections/tennent_20230723_position.parquet"
OUT_PQ = REPO_ROOT / "data/processed/vlm_detections/tennent_temporal_comparison.parquet"
MATCH_RADIUS_M = 50.0


def _short_label(scene: Scene) -> str:
    """Legacy scene label for the output parquet's ``scene`` column (e.g. '07-02')."""
    return datetime.fromtimestamp(
        scene.acquisition_time, tz=timezone.utc,
    ).strftime("%m-%d")


def result_to_parquet_rows(
    result: TemporalComparisonResult,
    scene_a: Scene,
    scene_b: Scene,
) -> list[dict]:
    """Translate a :class:`TemporalComparisonResult` into the legacy row schema.

    Row ordering preserves the original driver's convention: all scene-A rows
    first (in Scene observation order), then all scene-B rows.  Scene
    observations are sorted by obs_id inside the Scene, so this produces a
    deterministic row order.
    """
    label_a = _short_label(scene_a)
    label_b = _short_label(scene_b)

    # obs_id → match-data lookups built from the TemporalComparisonResult.
    a_to_match = {m.obs_a_id: m for m in result.matches}
    b_to_match = {m.obs_b_id: m for m in result.matches}

    obs_a_by_id = {o.obs_id: o for o in scene_a.observations}
    obs_b_by_id = {o.obs_id: o for o in scene_b.observations}

    rows: list[dict] = []
    for o in scene_a.observations:
        if o.obs_id in a_to_match:
            m = a_to_match[o.obs_id]
            rows.append({
                "obs_id": o.obs_id,
                "scene": label_a,
                "category": f"PERSISTENT_{label_a}",
                "match_obs_id": m.obs_b_id,
                "match_distance_m": m.distance_m,
                "lat": o.lat, "lon": o.lon,
                "classification_conf": o.classification_conf,
                "bbox_x1": o.bbox_px[0] if o.bbox_px else None,
                "bbox_y1": o.bbox_px[1] if o.bbox_px else None,
                "bbox_x2": o.bbox_px[2] if o.bbox_px else None,
                "bbox_y2": o.bbox_px[3] if o.bbox_px else None,
                "detector_reasoning": o.detector_reasoning,
            })
        else:
            rows.append({
                "obs_id": o.obs_id,
                "scene": label_a,
                "category": "DISAPPEARED",
                "match_obs_id": None,
                "match_distance_m": None,
                "lat": o.lat, "lon": o.lon,
                "classification_conf": o.classification_conf,
                "bbox_x1": o.bbox_px[0] if o.bbox_px else None,
                "bbox_y1": o.bbox_px[1] if o.bbox_px else None,
                "bbox_x2": o.bbox_px[2] if o.bbox_px else None,
                "bbox_y2": o.bbox_px[3] if o.bbox_px else None,
                "detector_reasoning": o.detector_reasoning,
            })
    for o in scene_b.observations:
        if o.obs_id in b_to_match:
            m = b_to_match[o.obs_id]
            rows.append({
                "obs_id": o.obs_id,
                "scene": label_b,
                "category": f"PERSISTENT_{label_b}",
                "match_obs_id": m.obs_a_id,
                "match_distance_m": m.distance_m,
                "lat": o.lat, "lon": o.lon,
                "classification_conf": o.classification_conf,
                "bbox_x1": o.bbox_px[0] if o.bbox_px else None,
                "bbox_y1": o.bbox_px[1] if o.bbox_px else None,
                "bbox_x2": o.bbox_px[2] if o.bbox_px else None,
                "bbox_y2": o.bbox_px[3] if o.bbox_px else None,
                "detector_reasoning": o.detector_reasoning,
            })
        else:
            rows.append({
                "obs_id": o.obs_id,
                "scene": label_b,
                "category": "EMERGED",
                "match_obs_id": None,
                "match_distance_m": None,
                "lat": o.lat, "lon": o.lon,
                "classification_conf": o.classification_conf,
                "bbox_x1": o.bbox_px[0] if o.bbox_px else None,
                "bbox_y1": o.bbox_px[1] if o.bbox_px else None,
                "bbox_x2": o.bbox_px[2] if o.bbox_px else None,
                "bbox_y2": o.bbox_px[3] if o.bbox_px else None,
                "detector_reasoning": o.detector_reasoning,
            })
    return rows


def main() -> None:
    print("Loading scenes via Scene API...")
    scene_a = load_scene_from_parquet(PQ_A, case_study="tennent")
    scene_b = load_scene_from_parquet(PQ_B, case_study="tennent")
    print(f"  {scene_a.scene_id}: {len(scene_a.observations)} obs")
    print(f"  {scene_b.scene_id}: {len(scene_b.observations)} obs")

    print(f"\nRunning temporal_persistence (DirectSpatialMatcher, gate={MATCH_RADIUS_M} m)...")
    result = temporal_persistence(
        scene_a, scene_b,
        matcher=DirectSpatialMatcher(gate_m=MATCH_RADIUS_M),
    )
    print(f"  matches:     {len(result.matches)}")
    print(f"  emerged:     {len(result.emerged)}")
    print(f"  disappeared: {len(result.disappeared)}")

    rows = result_to_parquet_rows(result, scene_a, scene_b)
    print(f"\nWriting {len(rows)} rows to {OUT_PQ.relative_to(REPO_ROOT)}...")
    OUT_PQ.parent.mkdir(parents=True, exist_ok=True)
    if OUT_PQ.exists():
        OUT_PQ.unlink()
    pq.write_table(pa.Table.from_pylist(rows), OUT_PQ)
    print(f"  wrote {OUT_PQ.stat().st_size/1024:.1f} KB")


if __name__ == "__main__":
    main()
