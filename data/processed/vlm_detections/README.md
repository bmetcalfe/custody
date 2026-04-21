# VLM Detections

Vessel-detection Parquet outputs produced by `custody.detection.vlm_sar.detect_vessels_in_scene` running Claude Sonnet 4.6 via the Anthropic VLM backend against Umbra SAR scenes.  One scene = one Parquet file, with per-scene provenance and pipeline config documented below.

## Contents

| file | scene | rows | driver |
|---|---|---:|---|
| `position.parquet` | Tennent 2023-07-02 (2 km AOI crop) | 42 | `scripts/05_detect_vlm_tennent.py` |
| `whitsun_20231206_position.parquet` | Whitsun 2023-12-06 (full scene) | 115 | `scripts/06_detect_vlm_whitsun.py` |

## Schema

Standard `custody.fusion.observations.PositionObservation` Parquet layout (see `custody.fusion.index._position_row`), with the Phase A / F.4 additions:

- `detector_reasoning` (str) — VLM's per-detection rationale
- `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2` (int) — scene-space pixel bbox in the scene's own coordinate frame (AOI-local for Tennent, full-scene for Whitsun); populated for all VLM-origin rows

## Pipeline configuration (common to both scenes)

- `tile_size=640`, `tile_overlap=64` (stride 576)
- `skip_empty=True`, `empty_threshold=0.05`
- `prompt_variant="contextualized_v1"` — SAR-context prompt from the Phase E PROMPTS registry
- `confidence_threshold="medium"` — `low`-confidence detections filtered out
- `nms_iou_threshold=0.5` — cross-tile duplicate merging

---

## Tennent Reef — 2023-07-02

### Provenance

| | |
|---|---|
| Source scene | `data/raw/umbra/.../f0730a1d-2bf7-4193-b2fe-ec8bfb2e1aef/2023-07-02-14-00-55_UMBRA-05/2023-07-02-14-00-55_UMBRA-05_GEC.tif` |
| Acquisition | 2023-07-02 14:00:55 UTC, UMBRA-05 X-band SAR |
| Scene center | (8.855687°N, 114.665145°E) — Đá Tiên Nữ / Tennent Reef |
| AOI | half-width 1.0 km crop centered on scene center; 5849 × 5849 px at 0.342 m/pixel |
| Backend | Claude Sonnet 4.6 |
| Run date | 2026-04-20 |

### Run results

| | |
|---|---:|
| Tiles processed | 121 / 121 (0 errored) |
| Detections pre-NMS | 53 |
| **Detections post-NMS + threshold** | **42** |
| Wall time | 20 min 19 s |
| Tokens (in / out) | 76,899 / 52,629 |
| **Actual cost** | **$1.02** ($1.69 estimated upper bound) |

### Spatial distribution

Three clusters (see `day0/scratch/tennent_20230702_vlm_detections.png`):

1. Along the Tennent reclamation structure — the dominant cluster, detections along the structure's N-S extent.
2. A tight secondary cluster immediately south of the southern tip.
3. A handful of scattered faint detections in open water south of the reef.

Three non-central quadrants of the AOI are almost empty — consistent with the Phase D.5 finding that Claude doesn't hallucinate on SAR speckle.

### Known limitations (Tennent)

1. **Structure-vs-vessel disambiguation.** Claude fires on bright point scatterers within the reclamation structure; SAR alone can't distinguish moored vessels from cranes, containers, and fixed infrastructure.  Disambiguation path: downstream fusion with AIS + temporal comparison against 2023-07-23.
2. **Prompt scene context mismatch.** `contextualized_v1` mentions Whitsun Reef by name; the AOI is Tennent.  The SAR-physics content transfers directly, but a prompt variant parameterizing the reef name is a small follow-up.
3. **No ground truth validation.** Detections are raw detector output, not verified vessel positions.

### Intended downstream use (Tennent)

- Cross-correlate with AIS presence data to identify AIS-dark detections near the reclamation structure.
- Temporal comparison with 2023-07-23 Tennent scene (21-day delta) for change-detection.
- Fusion-layer ingestion via `custody.fusion.index.index_observations`.

---

## Whitsun Reef — 2023-12-06

### Provenance

| | |
|---|---|
| Source scene | `data/raw/umbra/.../1e1f051d-4c81-4640-997d-1a03e967ad6a/2023-12-06-02-06-24_UMBRA-04/2023-12-06-02-06-24_UMBRA-04_GEC.tif` |
| Acquisition | 2023-12-06 02:06:25 UTC, UMBRA-04 X-band SAR |
| Scene center | (9.969°N, 114.632°E) — Đá Ba Đầu / Whitsun Reef |
| AOI | **full scene — no crop.** 17602 × 17602 px |
| Backend | Claude Sonnet 4.6 |
| Run date | 2026-04-20 |

No AOI crop here because Whitsun's interesting content (distributed vessel flotillas) spreads across the scene rather than concentrating on a single feature.

### Run results

| | |
|---|---:|
| Tiles processed | 961 / 961 (31 × 31; 0 errored) |
| Detections pre-NMS | 124 |
| **Detections post-NMS + threshold** | **115** (9 merged by NMS) |
| Wall time | **2 h 41 min** |
| Mean per-tile wall | 10.1 s |
| Tokens (in / out) | 668,667 / 407,328 |
| **Actual cost** | **$8.12** ($13.45 estimated upper bound — same 60% ratio as Tennent) |
| Rate-limit retries | 0 |

### Spatial distribution

**Distributed vessel field across open water** (see `day0/scratch/whitsun_20231206_vlm_detections.png`):

- **Upper-right cluster** — densest region; tiles 19/20/80 (origins around row 0–1152, col 10368–11520) alone account for 20 of 115 detections.  Scene-geographic region most consistent with the documented Whitsun maritime militia flotilla behavior (AMTI March-2021 reports).
- **Middle and lower halves are sparser** — isolated detections spread across the central and southern parts of the scene, with distinct azimuth-smear tails visible on many, indicating moving or drifting vessels.
- **No false positives on empty water.**  ~850 of the 961 tiles produced zero detections; negative-control behavior holds at full-scene scale.

### Top 5 tiles by detection count

| tile_index | origin (row, col) | count |
|---:|---|---:|
| 20 | (0, 11520) | 8 |
| 80 | (1152, 10368) | 7 |
| 19 | (0, 10944) | 5 |
| 269 | (4608, 12096) | 5 |
| 395 | (6912, 13248) | 5 |

### Known limitations (Whitsun)

1. **Tile-boundary duplicates not fully merged by NMS.**  Visual inspection of the preview shows near-duplicate detections at tile-boundary overlap regions that NMS at `iou_threshold=0.5` did not merge.  This happens when the VLM produces loose or differently-shaped bboxes around the same real vessel across two neighboring tiles — their IoU falls below 0.5 even though the underlying target is the same.  **True vessel count is likely ~70–85 after manual deduplication; the 115 post-NMS figure overstates by ~30–40%.**
2. **NMS threshold tuning is a follow-up item.**  Options: lower the IoU threshold to ~0.25 for VLM-bbox inputs (trades some risk of merging genuinely-adjacent vessels), or switch to centroid-distance-based NMS (merge observations whose centers are within one tile-overlap).  Either approach is straightforward.
3. **Duplicates naturally resolve during track association.**  For the intended downstream use — fusion-layer track formation — tile-boundary duplicates merge automatically when the Hungarian assigner sees two near-coincident observations in the same timestep.  So this is a count-quality issue, not a pipeline-correctness issue.
4. **No ground truth validation.**  Same limitation as Tennent — detections are raw VLM output.
5. **Prompt scene context matches for this run.**  `contextualized_v1` references Whitsun Reef by name, which is correct for this scene.

### Intended downstream use (Whitsun)

- Cross-correlate with AIS presence data to identify which vessels are AIS-dark.  Whitsun's case-study narrative centers on AIS-dark flotilla detection, so AIS-negative VLM detections are the primary signal.
- Feed into the fusion-layer tracker as a dense-observation layer; tile-boundary duplicates self-resolve during track association.
- Cross-compare against CFAR detections on the same scene (`scripts/04_detect_sar_whitsun.py`) to map VLM-vs-CFAR recall / precision on maritime-militia-scale clutter.

---

## Reproducibility

```bash
# Tennent
uv run python scripts/05_detect_vlm_tennent.py
# Whitsun (~2.5 hour wall time)
uv run python scripts/06_detect_vlm_whitsun.py
```

Both require `ANTHROPIC_API_KEY`.  Tennent ~$1/run, Whitsun ~$8/run.  Outputs overwrite the per-scene Parquet files above.
