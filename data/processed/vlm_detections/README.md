# VLM Detections — Tennent Reef 2023-07-02

First real-scene run of the VLM detection pipeline (`custody.detection.vlm_sar.detect_vessels_in_scene`).  Produced via `scripts/05_detect_vlm_tennent.py` on 2026-04-20.

## Provenance

| | |
|---|---|
| Source scene | `data/raw/umbra/sar-data/tasks/ship_detection_testdata/f0730a1d-2bf7-4193-b2fe-ec8bfb2e1aef/2023-07-02-14-00-55_UMBRA-05/2023-07-02-14-00-55_UMBRA-05_GEC.tif` |
| Scene acquisition | 2023-07-02 14:00:55 UTC, UMBRA-05 X-band SAR |
| Scene center (lat, lon) | (8.855687°N, 114.665145°E) — Đá Tiên Nữ / Tennent Reef |
| AOI crop | half-width 1.0 km (~2 km × 2 km square) centered on scene center; 5849 × 5849 pixels at 0.342 m/pixel |
| Backend | Claude Sonnet 4.6 via `AnthropicBackend` |
| Prompt variant | `contextualized_v1` (Umbra-04 / X-band / Whitsun Reef context — the scene's location overridden to be SAR-generic; AOI context is Tennent but the prompt still references Whitsun per the Phase E PROMPTS registry) |
| Run date | 2026-04-20 |
| Driver | `scripts/05_detect_vlm_tennent.py` |

## Pipeline configuration

- `tile_size=640`, `tile_overlap=64` — 11 × 11 = **121 tiles** (stride 576)
- `skip_empty=True` (empty threshold 0.05) — 0 tiles skipped; all 121 attempted
- `confidence_threshold="medium"` — `low`-confidence detections filtered out
- `nms_iou_threshold=0.5` — cross-tile duplicate merging
- `max_cost_usd=3.00`

## Run results

| | |
|---|---:|
| Tiles processed | 121 / 121 (0 errored) |
| Detections pre-NMS | 53 |
| **Detections post-NMS + threshold** | **42** |
| Wall time (total) | 20 min 19 s |
| Mean wall per tile | 10.1 s |
| Tokens (in / out) | 76,899 / 52,629 |
| **Actual cost (from tokens)** | **$1.02** |
| Estimated cost (upper bound) | $1.69 |
| Rate-limit retries | 0 |

## Spatial distribution

Detections concentrate in three regions (see `day0/scratch/tennent_20230702_vlm_detections.png`):

1. **Along the Tennent reclamation structure** — the dominant cluster; detections along the north-south extent of the reclaimed feature, with clumps at the northern tip, central mid-section, and southern causeway junction.
2. **Just south of the southern tip** — a tight secondary cluster on small bright scatterers immediately south of the reef.
3. **Scattered faint detections in open water south of the reef** — a handful of isolated targets, possibly vessels or drifting debris.

The three non-central quadrants of the AOI (north, east, west of the structure) are almost empty — consistent with the Phase D.5 finding that Claude does not hallucinate detections on pure SAR speckle.

Top five tiles by detection count:

| tile_index | origin (row, col) | count |
|---:|---|---:|
| 38 | (1728, 2880) | 5 |
| 39 | (1728, 3456) | 5 |
| 26 | (1152, 2304) | 4 |
| 27 | (1152, 2880) | 4 |
| 49 | (2304, 2880) | 4 |

All five sit in the central latitude band of the AOI — i.e., on the reclamation structure itself.

## Known limitations

1. **Structure-vs-vessel disambiguation.** Claude fires on bright point scatterers *within* the reclamation structure: cranes, containers, buildings, and any vessels moored at the reef are visually indistinguishable via SAR alone at this resolution.  Some "on-structure" detections may be real vessels berthed at Tennent; others are fixed infrastructure.  Downstream fusion with AIS and temporal comparison (2023-07-02 vs 2023-07-23) is the intended disambiguation path — a real vessel will move or broadcast; a crane will not.
2. **Single-scene snapshot.**  Temporal change analysis is not yet applied.  These detections are a Day-0 baseline; the 21-day delta to 2023-07-23 is a Phase-2 work item.
3. **Prompt scene context mismatch.**  The `contextualized_v1` prompt mentions Whitsun Reef by name; the AOI is Tennent Reef.  Prompt's claim of "Whitsun Reef" context is inaccurate for this run but the SAR-physics content (azimuth smearing, metal hull signatures) transfers directly.  A prompt variant parameterizing the reef name is a small follow-up.
4. **Confidence calibration is coarse.**  The `high`/`medium`/`low` → `0.85`/`0.60`/`0.35` mapping in `vlm_detection_to_observation` is an initial heuristic.  Expect revision once we have cross-backend calibration data.
5. **No ground truth validation yet.**  These detections have not been compared against AIS presence, vessel registries, or human-annotated reference data.  Treat as the detector's raw output, not as verified vessel positions.

## Intended downstream use

- **Cross-correlation with AIS presence data** (`data/processed/ais_presence/`) — identify which VLM detections correspond to AIS-broadcasting vessels vs AIS-dark targets.  AIS-dark detections near the reclamation structure are the signal of interest for the Tennent case study.
- **Temporal comparison with 2023-07-23 Tennent scene** — 21-day change-detection over the reclamation feature and its surroundings.  VLM-derived observations will serve as the detection layer; the tracker in `custody.fusion.tracker` does the temporal association.
- **Fusion-layer ingestion** — these observations are schema-compatible with the fusion-layer `PositionObservation` path and are already indexable via `custody.fusion.index.index_observations`.

## Schema

Standard `custody.fusion.observations.PositionObservation` Parquet layout (see `custody.fusion.index._position_row`), with the Phase A / F.4 additions:

- `detector_reasoning` (str) — VLM's per-detection rationale, populated for all 42 rows
- `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2` (int) — scene-space pixel bbox in the AOI-local coordinate frame, populated for all 42 rows

## Reproducibility

```bash
uv run python scripts/05_detect_vlm_tennent.py
```

Requires `ANTHROPIC_API_KEY` environment variable.  Cost: ~$1 per run.  Outputs overwrite the parquet at `data/processed/vlm_detections/position.parquet`.
