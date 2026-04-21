# VLM Detections

Vessel-detection Parquet outputs produced by `custody.detection.vlm_sar.detect_vessels_in_scene` running Claude Sonnet 4.6 via the Anthropic VLM backend against Umbra SAR scenes.  One scene = one Parquet file, with per-scene provenance and pipeline config documented below.

## Contents

| file | scene | rows | driver |
|---|---|---:|---|
| `position.parquet` | Tennent 2023-07-02 UMBRA-05 @ 0.34 m/px (2 km AOI crop) | 42 | `scripts/05_detect_vlm_tennent.py` |
| `tennent_20230723_position.parquet` | Tennent 2023-07-23 UMBRA-05 @ 0.33 m/px (2 km AOI crop) | 44 | `scripts/07_detect_vlm_tennent_0723.py` |
| `tennent_20230807_position.parquet` | Tennent 2023-08-07 UMBRA-04 @ 0.52 m/px (2 km AOI crop) | 35 | `scripts/08_detect_vlm_tennent_20230807.py` |
| `tennent_20230813_position.parquet` | Tennent 2023-08-13 UMBRA-06 @ 0.41 m/px (2 km AOI crop) | 27 | `scripts/10_detect_vlm_tennent_20230813.py` |
| `whitsun_20231206_position.parquet` | Whitsun 2023-12-06 UMBRA-04 @ 0.34 m/px (full scene) | 115 | `scripts/06_detect_vlm_whitsun.py` |
| `whitsun_20240320_position.parquet` | Whitsun 2024-03-20 UMBRA-05 @ 0.61 m/px (full scene) | 80 | `scripts/11_detect_vlm_whitsun_20240320.py` |
| `tennent_temporal_comparison.parquet` | Tennent 07-02 ↔ 07-23 per-observation classification | 86 | `day0/scratch/tennent_temporal_comparison.py` |

Four Tennent scenes span 42 days (2023-07-02 → 2023-08-13) across three Umbra sensors at four distinct pixel sizes.  Two Whitsun scenes span 3.5 months (2023-12-06 → 2024-03-20) across two sensors.  **A fifth Tennent scene, 2023-08-09 UMBRA-06 @ 0.33 m/px, was acquired and processed but is deliberately excluded from this directory pending diagnostic investigation — the run produced zero post-NMS observations across 121 tiles, with a featureless-speckle AOI preview that needs footprint / geometry diagnosis before the result can be trusted.  Diagnostic artifacts live under `day0/scratch/tennent_20230809_*`; the Parquet file is withheld until the null result is explained.**

## Schema

Standard `custody.fusion.observations.PositionObservation` Parquet layout (see `custody.fusion.index._position_row`), with the Phase A / F.4 additions:

- `detector_reasoning` (str) — VLM's per-detection rationale
- `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2` (int) — scene-space pixel bbox in the scene's own coordinate frame (AOI-local for Tennent, full-scene for Whitsun); populated for all VLM-origin rows

## Pipeline configuration (common to all scenes)

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

## Tennent Reef — 2023-07-23

Second Tennent acquisition, 21 days after 2023-07-02.  Same AOI geometry, same backend, same pipeline config — an apples-to-apples input to temporal change detection over the reclamation feature.

### Provenance

| | |
|---|---|
| Source scene | `data/raw/umbra/.../a9fceda5-7fa5-4849-b76b-dd6d7a50dae9/2023-07-23-14-02-49_UMBRA-05/2023-07-23-14-02-49_UMBRA-05_GEC.tif` |
| Acquisition | 2023-07-23 14:02:50 UTC, UMBRA-05 X-band SAR |
| Scene center | (8.855687°N, 114.665145°E) — identical to 07-02 to 6 decimals |
| AOI | half-width 1.0 km crop centered on scene center; 6021 × 6021 px at 0.332 m/pixel |
| Backend | Claude Sonnet 4.6 |
| Run date | 2026-04-21 |

Note: the 07-23 AOI is 172 px wider on each side than the 07-02 AOI.  Same 2 km box in meters, slightly different pixel-size (0.332 vs 0.342 m/px) produces the size delta.  The number of tiles from `iter_tiles` is identical (121) because the tile geometry absorbs the small AOI dimension difference.

### Run results

| | |
|---|---:|
| Tiles processed | 121 / 121 (0 errored) |
| Detections pre-NMS | 59 |
| **Detections post-NMS + threshold** | **44** |
| Wall time | 20 min 37 s |
| Tokens (in / out) | 79,743 / 54,610 |
| **Actual cost** | **$1.06** ($1.69 estimated upper bound) |

### Top 5 tiles by detection count (07-23)

| tile_index | origin (row, col) | count |
|---:|---|---:|
| 39 | (1728, 3456) | 6 |
| 70 | (3456, 2304) | 6 |
| 26 | (1152, 2304) | 5 |
| 38 | (1728, 2880) | 4 |
| 49 | (2304, 2880) | 4 |

### Known limitations (07-23)

Same structure-vs-vessel disambiguation issue as 07-02; same lack of ground-truth validation.  The cross-scene temporal comparison (below) provides a partial disambiguation heuristic.

---

## Tennent Reef — 2023-08-07

Third Tennent acquisition, 15 days after 07-23.  First Tennent scene at coarse resolution (UMBRA-04 at 0.52 m/px — roughly half the linear resolution of 07-02/07-23).  Useful as a cross-resolution reference against the higher-resolution July scenes.

### Provenance

| | |
|---|---|
| Source scene | `data/raw/umbra/.../00dee081-76ee-413a-8af2-e6be1c6ab8d0/2023-08-07-13-53-00_UMBRA-04/...GEC.tif` |
| Acquisition | 2023-08-07 13:53:00 UTC, UMBRA-04 X-band SAR |
| Scene center | (8.855687°N, 114.665145°E) |
| AOI | half-width 1.0 km crop; **3823 × 3823 px at 0.52 m/pixel** |
| Backend | Claude Sonnet 4.6 |
| Run date | 2026-04-21 |

### Run results

| | |
|---|---:|
| Tiles processed | 49 / 49 (0 errored) |
| Detections pre-NMS | 55 |
| **Detections post-NMS + threshold** | **35** |
| Wall time | 10 min 17 s |
| Tokens (in / out) | 32,071 / 29,697 |
| **Actual cost** | **$0.54** ($0.69 estimated upper bound) |

### Spatial pattern

Vessel-weighted detection at coarser resolution with visible wake signatures and multiple open-water vessels south of the reef.  The reclamation structure appears as a darker-outlined elongated feature rather than the bright-return presentation seen in 07-02/07-23 — at this resolution individual bright scatterers on the structure are aliased into the larger-pixel background.  **Fewer structure-interior detections than the higher-resolution July scenes**; more open-water detections with wake signatures.  The VLM is responding to large features (hull-scale vessels, major structure edges) but not to the fine scatterer population that drove the 42-detection counts at 0.34 m/px.

### Known limitations (08-07)

Resolution-limited — features below ~2–3 m won't register as distinct bright returns at 0.52 m/px.  Cross-resolution comparison against 07-02/07-23 is the natural next step; not attempted in this commit.

---

## Tennent Reef — 2023-08-13

Fourth Tennent acquisition.  Coarser-than-July at 0.41 m/px (UMBRA-06), between 07-23 and 08-07 in resolution.  **Orbit geometry substantially different from 07-02/07-23** — the reclamation structure appears rotated ~90° relative to the earlier scenes' presentation.

### Provenance

| | |
|---|---|
| Source scene | `data/raw/umbra/.../645d903f-ced5-40d9-94ed-6e4e91977c97/2023-08-13-15-00-09_UMBRA-06/...GEC.tif` |
| Acquisition | 2023-08-13 15:00:09 UTC, UMBRA-06 X-band SAR |
| Scene center | (8.855687°N, 114.665145°E) |
| AOI | half-width 1.0 km crop; **4921 × 4921 px at 0.41 m/pixel** |
| Backend | Claude Sonnet 4.6 |
| Run date | 2026-04-21 |

### Run results

| | |
|---|---:|
| Tiles processed | 81 / 81 (0 errored) |
| Detections pre-NMS | 37 |
| **Detections post-NMS + threshold** | **27** |
| Wall time | 15 min 19 s |
| Tokens (in / out) | 53,239 / 38,839 |
| **Actual cost** | **$0.74** ($1.13 estimated upper bound) |

### Spatial pattern

Reclamation structure presents as a horizontally-elongated feature with visible bright scatterers at its southern edge.  **A new convoy / cluster of vessels trails south-southeast** from the structure — clearly resolvable individual bright returns with azimuth-smear tails, absent from 07-02 and 07-23 at the corresponding geographic area.  This scene's detections are **NOT directly comparable** to 07-02/07-23 via the existing 50 m-radius matching approach: the structure's apparent orientation and the scatterer positions shift with viewing geometry, so a naive persistence match across these scenes would systematically fail at fixed-feature locations that the July scenes labelled stable.

### Known limitations (08-13)

Not safely comparable to 07-02/07-23 via the existing temporal-persistence approach.  A geometry-aware matcher or a reconciliation layer between scene-space observations would be required.  See the "Cross-scene heterogeneity" section below.

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

## Whitsun Reef — 2024-03-20

Second Whitsun acquisition, 3.5 months after 2023-12-06.  **Coarse resolution (0.61 m/px, UMBRA-05)** — roughly half the linear resolution of the December scene.  Useful for temporal change detection over the maritime-militia flotilla but with the caveat that finer features are invisible at this resolution.

### Provenance

| | |
|---|---|
| Source scene | `data/raw/umbra/.../92c2b446-415c-4d26-88bc-70fdb652e852/2024-03-20-02-09-55_UMBRA-05/...GEC.tif` |
| Acquisition | 2024-03-20 02:09:56 UTC, UMBRA-05 X-band SAR |
| Scene center | (9.985499°N, 114.641600°E) — ~1.8 km northeast of 12-06's scene center |
| AOI | **full scene — no crop.** 14660 × 14659 px at 0.61 m/pixel |
| Backend | Claude Sonnet 4.6 |
| Run date | 2026-04-21 |

### Run results

| | |
|---|---:|
| Tiles processed | 676 / 676 (0 errored) |
| Detections pre-NMS | 98 |
| **Detections post-NMS + threshold** | **80** |
| Wall time | 117 min 27 s |
| Tokens (in / out) | 465,933 / 284,570 |
| **Actual cost** | **$5.67** ($9.46 estimated upper bound) |

### Spatial pattern

**Bimodal tight-cluster pattern differs from 12-06's distributed upper-right flotilla.**  Two distinct vessel clusters are visible in the preview — one upper-center and one lower-center — each with 20–30 tight detections on bright features with extensive azimuth smearing indicating motion.  Quieter water between the clusters and across the rest of the scene.  The reef itself is visible as a diagonal bright feature in the upper-right quadrant.

Top-5 tiles by detection count:

| tile_index | origin (row, col) | count |
|---:|---|---:|
| 454 | (9792, 6912) | 7 |
| 429 | (9216, 7488) | 5 |
| 110 | (2304, 3456) | 4 |
| 430 | (9216, 8064) | 4 |
| 639 | (13824, 8640) | 4 |

The clustering is qualitatively different from 12-06: 12-06 had detections distributed in a broad upper-right region; 03-20 has two compact clusters in different parts of the scene.  Some of the count drop (115 → 80) is attributable to coarser resolution masking fine detail; the spatial relocation is a genuine behavioral change.

### Known limitations (2024-03-20)

Scene center is ~1.8 km northeast of 12-06's; the two scenes cover overlapping but not identical geographic footprints.  Direct spatial persistence matching across the 3.5-month interval would need to account for this offset in addition to resolution differences.  See the "Cross-scene heterogeneity" section below.

---

## Temporal comparison: Tennent 07-02 → 07-23

21-day interval at the same AOI and with the same pipeline.  First pass at the temporal-change signal that motivated running Tennent twice.

### Aggregate counts

| | 07-02 | 07-23 | Δ |
|---|---:|---:|---:|
| AOI shape (px) | 5849×5849 | 6021×6021 | +172 |
| Pixel size (m) | 0.342 | 0.332 | –3% |
| Pre-NMS detections | 53 | 59 | **+6** |
| Post-NMS detections | **42** | **44** | +2 |
| Actual cost (USD) | 1.02 | 1.06 | +0.04 |
| Wall time | 20m 19s | 20m 37s | +18s |

Aggregate detection count is nearly identical (42 → 44 post-NMS).  The interesting signal is not in the total count but in **where** detections persist vs change.

### Persistent zones (likely fixed infrastructure)

Four tiles appear in both scenes' top-5 by detection count.  Expected behaviour if the reclamation feature's bright scatterers (cranes, containers, platform edges) are stable between acquisitions.

| origin (row, col) | 07-02 count | 07-23 count |
|---|---:|---:|
| (1152, 2304) | 4 | 5 |
| (1728, 2880) | 5 | 4 |
| (1728, 3456) | 5 | 6 |
| (2304, 2880) | 4 | 4 |

These tiles tile the centerline of the reclaimed structure.  Detection persistence at the ~tile-scale here is a weak but useful signal: detections in these tiles are more likely fixed infrastructure than vessels, because a moored vessel departing and a new vessel arriving in the same 576 m × 576 m cell over 21 days is less likely than the same crane/container being imaged twice.

### Change zones (likely mobile targets)

- **New top-5 tile in 07-23: `(3456, 2304)` with 6 detections.** That's ~240 m south of any 07-02 top-5 tile (07-02's southernmost top-5 was row 2304).  Entire small-vessel cluster that wasn't present on 07-02.
- **Elongated bright target east of the structure with strong horizontal azimuth smear.**  Classic SAR moving-vessel signature — bright compact core plus a horizontal tail.  Not present on 07-02.  Probably moored or drifting vessel at the east flank of the reef.
- **Additional azimuth-smear detections in the lower-right quadrant** — more candidate moving targets south of the reef than 07-02 showed.
- **Wake-like linear feature in the lower-center** with several detections along it — possibly a vessel caught mid-transit with a clean azimuth-smear tail.

Detections that appear on one scene but not the other, at positions well separated from persistent structure-adjacent clusters, are the primary change signal for the temporal narrative.

### Structure-centric change signal (visible without a detector)

Side-by-side visual inspection of the AOI preview images (`day0/scratch/tennent_20230702_vlm_detections.png` and `.../tennent_20230723_vlm_detections.png`) shows that **the reclamation structure itself changed shape during the 21-day interval.**  Sharpened edges, new internal geometry, and more defined internal features are visible in raw amplitude imagery — no detector needed.  This is consistent with ongoing construction activity at Tennent Reef during summer 2023 and matches the documented AMTI reporting pattern.

The structure-change signal is independent of VLM detection output; it comes from the SAR amplitude differences directly.  For a production change-detection pipeline, coherent change detection (CCD, see `day0/scratch/ccd_spike_*.py` investigations) on this scene pair is the more direct route to quantifying structure-evolution.

### Disambiguation heuristic

The 07-02 README flagged moored-vessel-vs-fixed-equipment as an unsolved disambiguation problem for individual detections.  Temporal persistence across the 21-day pair provides a partial disambiguator:

- **Same location on both 07-02 and 07-23** → more likely fixed infrastructure.
- **Only one of 07-02 or 07-23 (or notably different location)** → more likely mobile.

Limitation: "same location" is fuzzy at the tile scale.  Two vessels anchoring at the same mooring 21 days apart would be indistinguishable from one fixed scatterer under this rule.  Proper disambiguation needs either AIS cross-correlation (real vessels broadcast; cranes don't) or pixel-level co-registered change detection (CCD on aligned SICD complex data).  Both are tracked as downstream work.

---

## Temporal persistence classification (Tennent 07-02 ↔ 07-23)

First temporal analysis exercising the custody architecture against real multi-timestamp SAR data.  Classifies each of the 86 combined observations (42 on 07-02 + 44 on 07-23) into one of four categories based on spatial matching across the 21-day interval.

Output: `data/processed/vlm_detections/tennent_temporal_comparison.parquet` — one row per observation with fields `obs_id`, `scene` ("07-02" or "07-23"), `category`, `match_obs_id` (nullable), `match_distance_m` (nullable), `lat`, `lon`, `classification_conf`, `bbox_x1..y2`, `detector_reasoning`.  Driver: `day0/scratch/tennent_temporal_comparison.py`.

### Methodology

Direct spatial matching, **not the fusion tracker**.

- **Distance**: pyproj.Geod WGS84 geodetic distance between each 07-02 observation's `(lat, lon)` and each 07-23 observation's `(lat, lon)`.
- **Gate**: 50 m threshold — slightly larger than Claude Sonnet 4.6's observed VLM bbox centroid jitter on identical features (per Phase D.5 findings).
- **Assignment**: `scipy.optimize.linear_sum_assignment` (Hungarian) on a cost matrix where `C[i, j]` is the geodetic distance if within the gate and a `_BIG_COST` sentinel otherwise.  Produces the global minimum-total-distance pairing subject to the 50 m constraint.  Observations left unmatched after assignment become `DISAPPEARED` (07-02 side) or `EMERGED` (07-23 side).

### Why direct matching and not the tracker

`src/custody/fusion/tracker.py` is built for continuous observation streams — the EKF predicts track state forward between observations using a velocity prior (ADR-0007) with a covariance that grows as `σ²_v · dt²`.  Over a 21-day `dt`, that prediction covariance is effectively scene-wide: the gate would admit nearly any match and the tracker would produce noise instead of signal.

Direct matching at a fixed distance is the honest tool for two single-timestamp SAR scenes 21 days apart.  The tracker's continuous-regime design fits multi-scene streams where the observation cadence is close to the target's dynamics timescale; that isn't what we have here.  See ADR-0017 for the Week-3 design decision and the investigation that motivated it.

### Aggregate counts

| category | count | % of scene |
|---|---:|---:|
| PERSISTENT_07-02 (07-02 obs with 07-23 match ≤ 50 m) | 22 | 52.4% of 42 |
| PERSISTENT_07-23 (reciprocal) | 22 | 50.0% of 44 |
| EMERGED (07-23 only) | 22 | 50.0% of 44 |
| DISAPPEARED (07-02 only) | 20 | 47.6% of 42 |

Persistent pair count is balanced on both sides (22 = 22), as expected for a one-to-one assignment under a symmetric gate.

### Match-distance distribution

Across the 22 persistent pairs:

- min: **6.6 m**
- mean: **21.4 m**
- max: **44.3 m**

All well below the 50 m gate, which matters: it rules out the "the gate is too permissive and Hungarian is fishing for matches near the boundary" failure mode.  The mean of ~21 m is consistent with the VLM bbox centroid jitter Phase D.5 measured on identical features (different prompts on the same tile produced bbox centers shifted by a few pixels, which at 0.34 m/pixel corresponds to tens of meters in geographic coordinates).

### Interpretive findings

These are observations about the spatial patterns in the overlay image (`day0/scratch/tennent_temporal_comparison.png`), not ground-truth-validated classifications.

1. **Persistent markers cluster on the reclamation structure centerline and southern platform.**  The spatial pattern is consistent with fixed reclamation infrastructure (cranes, containers, platform edges) producing repeatable SAR returns across both acquisitions.  Persistence at matched locations across 21 days is stronger evidence of fixed infrastructure than position alone.
2. **Disappeared markers form localized clusters, not random scatter.**  The ~20 disappeared 07-02 observations concentrate in two zones: an east-side line along the pier and a northern-tip cluster.  If the disappeared category were pure VLM detection noise, we'd expect it to scatter randomly; the observed clustering suggests **localized changes in the structure itself** between scenes — plausibly active construction, equipment relocation, or transient objects (vessels, containers) that were at specific work areas on 07-02 and had moved by 07-23.
3. **Emerged markers include clearly-new vessels.**  Two prominent features visible on 07-23 only are flagged emerged: the strong azimuth-smear starburst target east of the structure (already called out in the 07-23 run README as a probable moving vessel) and a lower-right cluster of bright point scatterers with azimuth-smear tails.  Classic moving-vessel SAR signatures, arriving during the 21-day interval.
4. **Southern open-water persistent detections suggest long-duration anchored vessels.**  Several persistent pairs sit in open water south of the reef at positions that aren't near any visible structure.  Most consistent interpretation: moored or anchored vessels that remained at the same location across both scenes — common for work vessels supporting reef construction or support tenders.

### Caveats

1. **Two timestamps is the minimum for "persistence" to mean anything.**  With only two observations per spatial cell, we can distinguish "matched across 21 days" from "only in one scene," but not "vessel vs infrastructure."  A third scene would materially tighten the infrastructure-vs-long-moored-vessel distinction.
2. **Persistence classification is subject to VLM detection variance.**  Mean match distance of ~21 m is on the order of Claude's bbox centroid jitter on identical features.  A target that is genuinely at the same position across both scenes can still appear at slightly different VLM-reported positions; the Hungarian match absorbs that noise up to 50 m.  Beyond that gate, a genuinely-persistent target with high bbox jitter could be misclassified as disappeared + emerged.  The mean distance staying well inside the gate (21 m vs 50 m) suggests this failure mode is rare in practice here.
3. **The 50 m threshold is calibrated for VLM localization, not per-hull tracking.**  A per-hull-accurate threshold for moving-target SAR tracking would be tens of meters; we're not doing that.  A different detector output (YOLO, CFAR point centroids) would warrant a different threshold.
4. **Interpretation as "construction activity" or "moored vessels" is hypothesis-level.**  The spatial patterns support those narratives, but nothing here is ground-truth validated — there's no AIS to cross-reference (see ADR-0017 findings) and no human annotation of the SAR imagery.
5. **Greedy global assignment, not mutual-nearest-neighbor.**  Hungarian pairs A with its globally-best partner under the gate even if an asymmetric-mutual-nearest rule would have left A unmatched.  For pairs well below the gate the two approaches agree; near the gate boundary they can differ.

### Future work: external validation via GFW

The GFW `public-global-fixed-infrastructure:latest` dataset provides pre-computed fixed-vs-mobile classification from Sentinel-1 time-series analysis and would provide independent validation of which structure-interior detections represent fixed features.  The dataset is tier-locked on our current GFW API key (see `docs/investigations/ais_coverage_investigation.md`); upgrading the key via a research-partner application is a future track.  Not blocking — this temporal-persistence classifier is a useful signal on its own — but would materially strengthen the interpretive claims above if available.

---

## Cross-scene heterogeneity and analytic implications

The seven-scene dataset now spans three Umbra sensors, four pixel sizes, and multiple orbit geometries.  Two findings from the 2023-04-21 batch run (see `day0/scratch/batch_preflight.py` and the run logs) shape how this data can and can't be used for downstream analysis.

### Finding 1 — Pixel size is per-acquisition, not per-sensor

Sensor | Scene | Pixel size
|---|---|---|
UMBRA-04 | Whitsun 2023-12-06 | 0.34 m/px
UMBRA-04 | Tennent 2023-08-07 | 0.52 m/px
UMBRA-05 | Tennent 2023-07-02 | 0.34 m/px
UMBRA-05 | Tennent 2023-07-23 | 0.33 m/px
UMBRA-05 | Whitsun 2024-03-20 | 0.61 m/px
UMBRA-06 | Tennent 2023-08-09 | 0.33 m/px (excluded)
UMBRA-06 | Tennent 2023-08-13 | 0.41 m/px

The same UMBRA-06 produced 0.33 m/px on 08-09 and 0.41 m/px on 08-13.  UMBRA-05 covers 0.33–0.61 m/px across its scenes.  Resolution is a **scene-level attribute**, not a sensor-level one — driven by collection mode / tasked resolution / orbit-geometry / range-to-target combinations on each individual acquisition.

For downstream analysis this means: do not attribute detection-count or feature-visibility differences to "sensor" alone.  Resolution is the dominant free variable, and it varies within-sensor more than a naive "sensor brand" hypothesis would assume.

### Finding 2 — Orbit geometry substantially changes apparent feature presentation

Visual inspection of the Tennent AOI previews across acquisitions shows the reclamation structure rendering in dramatically different orientations:

- 07-02 and 07-23: structure appears as a **vertically-elongated** bright feature in the center.
- 08-07: structure appears vertical but with softer contrast (coarser pixel size).
- 08-13: structure appears **rotated ~90° — horizontally-elongated** with different bright-scatterer positions.

Same physical reef.  Different orbit geometries (ascending vs descending pass, different incidence angles, different look directions) produce SAR signatures that are not trivially comparable.  A bright scatterer at pixel (X, Y) in one scene is not necessarily at (X, Y) in another scene even if the underlying target is stationary.

### Implication: the 50 m temporal-persistence classifier does not extend cleanly

The temporal-persistence classifier landed in an earlier commit (section "Temporal persistence classification" above) assumes **stable scatterer positions across scenes**.  That assumption holds for:

- **Near-identical orbit geometry** — 07-02 and 07-23 share sensor, incidence angle, pass direction, acquisition mode.  Same feature appears at near-identical positions across them.  50 m direct matching works.
- **Same reef, similar pixel size** — the July scenes match closely enough that per-hull accuracy is within the 50 m gate.

It breaks for:

- **Different orbit geometries** — 08-13 vs 07-02.  The reclamation structure is in different apparent positions; a 50 m match would fail for many genuinely-persistent scatterers and succeed accidentally for unrelated ones.
- **Different resolutions** — 08-07 (0.52 m/px) vs 07-02 (0.34 m/px).  Coarser resolution merges features that higher resolution separated; NMS and detection-confidence thresholds behave differently.
- **Different scene centers** — Whitsun 12-06 vs 03-20 (1.8 km offset).  Even though the scenes overlap, "persistence" across the non-overlapping region is ill-defined.

### Implication for multi-sensor fusion

Heterogeneity between scenes is a **first-class problem, not an edge case**.  The Tennent 5-scene timeline cannot be run through the existing 50 m classifier as a single pipeline; it needs either:

- **Geometry-aware matching** — use the scene's projected bbox, look-direction, and incidence to compute expected feature displacement between scenes and widen the gate accordingly.
- **Reconciliation layer** — abstract observations from scene-space to feature-space (per-scatterer identity, cross-scene association via multi-feature signatures rather than position alone).
- **Scene-pair analysis** — only compare scene pairs with near-identical geometry (the 07-02 / 07-23 pair qualifies; most others do not).

The existing temporal-persistence classifier remains valid for 07-02 ↔ 07-23 specifically, but extending it to the full 4-scene Tennent timeline (or the 5-scene set once 08-09 is resolved) is Week-4+ work, not a drop-in application.

### The 08-09 exclusion

Tennent 2023-08-09 UMBRA-06 was processed on the same batch but produced **zero post-NMS observations across 121 tiles**, with an AOI preview showing pure SAR speckle and no visible reclamation structure.  Three working hypotheses: (a) scene footprint doesn't cover the reef despite scene-center-lat/lon agreement; (b) the night-time (02:23 UTC) acquisition's orbit geometry produces extreme incidence that renders the reef near-invisible; (c) coordinate-transform bug in the AOI crop.  Diagnostic artifacts in `day0/scratch/tennent_20230809_footprint_*.{png,md}` address these; the scene's Parquet is withheld from this directory until diagnosis clears it.

---

## Reproducibility

```bash
# Tennent 2023-07-02 baseline
uv run python scripts/05_detect_vlm_tennent.py
# Tennent 2023-07-23 temporal comparison (same AOI, 21 days later)
uv run python scripts/07_detect_vlm_tennent_0723.py
# Whitsun 2023-12-06 (~2.5 hour wall time)
uv run python scripts/06_detect_vlm_whitsun.py
# Tennent 2023-08-07 (coarser resolution reference)
uv run python scripts/08_detect_vlm_tennent_20230807.py
# Tennent 2023-08-13 (different orbit geometry)
uv run python scripts/10_detect_vlm_tennent_20230813.py
# Whitsun 2024-03-20 (second Whitsun acquisition, ~2 hour wall time)
uv run python scripts/11_detect_vlm_whitsun_20240320.py
# Tennent temporal persistence comparison for 07-02 ↔ 07-23 (reads both parquets; no API calls)
uv run python day0/scratch/tennent_temporal_comparison.py
```

Both require `ANTHROPIC_API_KEY`.  Tennent ~$1/run, Whitsun ~$8/run.  Outputs overwrite the per-scene Parquet files above.
