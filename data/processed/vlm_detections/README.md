# VLM Detections

Vessel-detection Parquet outputs produced by `custody.detection.vlm_sar.detect_vessels_in_scene` running Claude Sonnet 4.6 via the Anthropic VLM backend against Umbra SAR scenes.  One scene = one Parquet file, with per-scene provenance and pipeline config documented below.

## Contents

| file | scene | rows | driver |
|---|---|---:|---|
| `position.parquet` | Tennent 2023-07-02 (2 km AOI crop) | 42 | `scripts/05_detect_vlm_tennent.py` |
| `tennent_20230723_position.parquet` | Tennent 2023-07-23 (2 km AOI crop, same geometry as 07-02) | 44 | `scripts/07_detect_vlm_tennent_0723.py` |
| `whitsun_20231206_position.parquet` | Whitsun 2023-12-06 (full scene) | 115 | `scripts/06_detect_vlm_whitsun.py` |
| `tennent_temporal_comparison.parquet` | Tennent 07-02 ↔ 07-23 per-observation classification | 86 | `day0/scratch/tennent_temporal_comparison.py` |

The two Tennent scenes are acquired 21 days apart over the same AOI for temporal change detection.  The comparison parquet carries one row per observation (42 + 44 = 86) with a `category` label of `PERSISTENT_07-02`, `PERSISTENT_07-23`, `EMERGED`, or `DISAPPEARED`; see the persistence-classification section below.

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

## Reproducibility

```bash
# Tennent 2023-07-02 baseline
uv run python scripts/05_detect_vlm_tennent.py
# Tennent 2023-07-23 temporal comparison (same AOI, 21 days later)
uv run python scripts/07_detect_vlm_tennent_0723.py
# Whitsun (~2.5 hour wall time)
uv run python scripts/06_detect_vlm_whitsun.py
# Tennent temporal persistence comparison (reads both parquets above; no API calls)
uv run python day0/scratch/tennent_temporal_comparison.py
```

Both require `ANTHROPIC_API_KEY`.  Tennent ~$1/run, Whitsun ~$8/run.  Outputs overwrite the per-scene Parquet files above.
