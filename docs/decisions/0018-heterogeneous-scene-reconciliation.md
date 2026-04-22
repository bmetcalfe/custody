---
id: 0018
title: Heterogeneous-scene reconciliation — architecture for pooling detections across non-uniform SAR acquisitions
date: 2026-04-21
status: accepted
---

## Context

The VLM detection work in Weeks 2–4 produced a 7-scene dataset of real VLM-derived observations across two case-study reefs:

| scene | sensor | pixel size | orbit / time | post-NMS obs |
|---|---|---:|---|---:|
| Tennent 2023-07-02 | UMBRA-05 | 0.34 m/px | 14:00 UTC | 42 |
| Tennent 2023-07-23 | UMBRA-05 | 0.33 m/px | 14:02 UTC | 44 |
| Tennent 2023-08-07 | UMBRA-04 | 0.52 m/px | 13:53 UTC | 35 |
| Tennent 2023-08-09 | UMBRA-06 | 0.33 m/px | 02:23 UTC | **0** (excluded — low-SNR acquisition) |
| Tennent 2023-08-13 | UMBRA-06 | 0.41 m/px | 15:00 UTC | 27 |
| Whitsun 2023-12-06 | UMBRA-04 | 0.34 m/px | 02:06 UTC | 115 |
| Whitsun 2024-03-20 | UMBRA-05 | 0.61 m/px | 02:09 UTC | 80 |

Cross-scene heterogeneity is substantially larger than the custody fusion architecture currently accommodates.  Specific phenomena observed in this dataset:

1. **Pixel size varies 0.33–0.61 m/px per acquisition, not per sensor.**  UMBRA-06 produced 0.33 m/px on 08-09 and 0.41 m/px on 08-13.  UMBRA-05 covers 0.33–0.61 m/px across its three appearances.  Resolution is a scene-level property driven by collection mode, tasked resolution, and range geometry — not a sensor identity.
2. **Orbit geometry dramatically changes apparent feature presentation.**  Tennent 08-13's reclamation structure appears rotated roughly 90 degrees relative to 07-02/07-23 (horizontally elongated vs vertically elongated in the rendered AOI imagery).  Same physical feature, different SAR signature.  A bright scatterer at AOI pixel (X, Y) in 07-02 is not at (X, Y) in 08-13 even if the underlying target is stationary.
3. **Acquisition quality varies — sometimes catastrophically.**  Tennent 08-09 produced zero observations across 121 tiles.  Post-hoc diagnosis (`day0/scratch/tennent_20230809_diagnosis.md`) showed the scene has ~3× narrower dynamic range than 07-02 (scene-wide intensity std 4.3 vs 11.5; p99 58 vs 102 on 0–255 scale) — a low-SNR collection where neither VLM nor human can distinguish the reef from surrounding water.  The pipeline correctly returned zero detections; the acquisition itself contributed no signal.  Cost: ~$1 spent before the null was visible.
4. **Time separation varies from days to months.**  Tennent pairs span 21 days to 42 days; the two Whitsun scenes are 3.5 months apart.  The 50 m direct-matching temporal-persistence classifier landed for 07-02↔07-23 (Week 3) works at that gate precisely because those scenes share sensor, incidence, and pass direction.  Extending it naively to the 5-scene Tennent timeline fails because the geometric shifts between scenes exceed the gate.

These findings build on but aren't fully handled by existing decisions:

- **ADR-0008 (polymorphic observations)** introduces `PositionObservation` and `PositionVelocityObservation`.  The polymorphic design accommodates different sensor types but assumes within-type observations are geometrically comparable.  ADR-0008 does not address scene-level heterogeneity within a single observation type.
- **ADR-0016 (Story 2 scope)** commits to Sentinel-1 integration in Weeks 5–6, which multiplies the heterogeneity problem — Sentinel-1 adds a fourth sensor class with its own geometry, resolution, and orbit characteristics.
- **ADR-0017 (Week 3 refocus)** established that single-scene SAR doesn't exercise the tracker's core value and pivoted to direct spatial matching for the Tennent 07-02↔07-23 pair.  The heterogeneity question was acknowledged but not resolved.
- The VLM detections README's "Cross-scene heterogeneity" section documents the findings operationally but doesn't commit to an architectural response.

**Problem statement**: SAR observations from heterogeneous scenes cannot be naively pooled into the existing fusion tracker — or into direct-matching classifiers — without a reconciliation layer.  The 5-scene Tennent temporal timeline, the 2-scene Whitsun cross-resolution pair, and the upcoming Sentinel-1 integration all require this layer.  We need to decide its shape before the next batch of cross-scene analysis work.

## Options

### Option A — Per-scene normalization (pre-fusion)

Geometrically correct each scene to a canonical frame before observations flow into the fusion layer.  Each scene runs through a pipeline that resamples / rectifies / quality-normalizes to a common geometry and intensity calibration; downstream code sees a homogeneous stream.

**Assessment.**  SAR geometry correction is a deep subject in its own right — proper rectification needs a DEM, precise sensor metadata, and handles terrain-induced geometric distortions that UMBRA ODP data doesn't provide the metadata to resolve.  Canonical-frame resampling also discards information from higher-resolution scenes (you'd downsample 0.33 m/px to match 0.61 m/px).  Moderate-to-high implementation effort (weeks); high architectural risk if the correction pipeline isn't robust (garbage-in-garbage-out propagates everywhere); poor fit to current state because it imposes a large standalone component that has to work correctly before any downstream work can proceed.

### Option B — Per-observation covariance encoding

Each `PositionObservation` carries geometry / quality metadata encoded in its covariance.  Low-quality scenes produce larger `cov_pos` values; coarser-resolution scenes produce larger `cov_pos` along the appropriate axis; rotated-geometry scenes get direction-aware covariance.  The fusion tracker's Kalman gain already respects per-observation covariance; downstream classifiers that consume `PositionObservation` lists would weight observations by their covariance determinant or similar.

**Assessment.**  Matches the existing polymorphic schema without new types.  Minimal new infrastructure — all the data structures are already in place.  Honest about uncertainty in a way that propagates naturally through the fusion math.  The hard part is **methodology for computing per-scene covariance**: turning "this scene is at 0.52 m/px under high incidence" into a 2×2 covariance matrix requires a calibration procedure we don't have.  Low-to-moderate effort (single module, a few hundred LOC); low architectural risk; good fit to current state, but insufficient on its own because covariance encodes uncertainty around an unbiased estimate — and the cross-scene problem is partly bias, not uncertainty.  A bright scatterer that appears at systematically different positions in 07-02 vs 08-13 due to orbit geometry isn't "uncertain" about its position; it's at different projected positions under different acquisitions.  Bias cannot be represented by wider covariance.  Option B captures the uncertainty half of the heterogeneity problem while leaving the systematic-shift half unaddressed.

### Option C — Scene as first-class entity

Introduce `Scene` as a distinct concept alongside `Observation`.  A Scene represents a single SAR acquisition with its own metadata (sensor, timestamp, pixel size, incidence, footprint, quality flags) and owns a collection of observations that share that scene's geometric properties.  Cross-scene analysis becomes explicit operations over scene objects — "compute temporal persistence between Scene A and Scene B using geometry-appropriate matching"— rather than implicit pooling of observations.  The existing tracker remains focused on continuous streams (AIS when available, per-scene observations within a single Scene); cross-scene work is a separate layer.

**Assessment.**  Clean separation between within-scene (where existing tools like the tracker and NMS work) and cross-scene (where explicit reconciliation is needed) operations.  Matches how MDA analysts actually reason about the data ("look at 07-02, look at 07-23, compare them").  Does not force sparse SAR acquisitions into a continuous-stream abstraction that never fit.  Moderate effort to introduce the `Scene` type and refactor the existing drivers to populate it; moderate architectural risk because "observation" stops being the universal abstraction, but the risk is contained — everything downstream that already consumes `PositionObservation` keeps working, and Scene-level operations are additive.  Good fit to the current state: the existing driver scripts already treat scenes as units; this ADR would formalize what's de-facto happening.

### Option D — Hybrid: continuous streams feed tracker, scene events are discrete tracker updates

AIS (when it exists) drives the tracker in its continuous-stream mode.  Each SAR scene becomes a single discrete "scene event" that injects all of the scene's observations into the tracker simultaneously at the acquisition timestamp.  Between scenes, the tracker dead-reckons existing tracks forward; a scene event can update existing tracks, spawn new ones, or leave them unassociated.

**Assessment.**  Respects both regimes — continuous and discrete — without picking one.  Minimal schema change if the scene event is just a batch of timestamped observations (which is what scenes already produce).  Requires careful design of `Tracker.step()` semantics for scene-event timing, particularly how to handle a 21-day gap between SAR scenes when AIS is absent (the tracker's covariance would balloon, making association meaningless — exactly the problem ADR-0017 flagged).  Moderate effort to modify tracker lifecycle; moderate-to-high architectural risk because the hybrid semantics has several corner cases (scene-event before AIS arrives, AIS between two scene-events, scene-event during a track-coasting period) that would need careful test coverage.  Uneven fit to current state: solves the future AIS+SAR problem the project doesn't have today per ADR-0017, while partially addressing the sparse-scene problem the project actually has.

Option D is not wrong so much as it's early.  The hybrid semantics it proposes fits a future project state where continuous AIS streams exist alongside periodic SAR scene events; ADR-0017 documented that state is not present in the Spratly AOIs today.  Reconsidering Option D becomes appropriate if AIS coverage materializes — whether via upgraded GFW access (ADR-0011's future-work hook), a shipping-lane extension, or a different AOI where cross-modality is naturally present.

## Decision

**Adopt Option C: Scene as first-class entity.**

Introduce a `Scene` type (location TBD, likely `custody.fusion.scenes` or `custody.detection.scenes`) that carries per-acquisition metadata — sensor, acquisition time, pixel size, scene-center lat/lon, scene footprint, estimated incidence, and a quality flag — plus owns a list of the `PositionObservation` instances produced from that scene.  Cross-scene analysis routines take `Scene` instances as explicit inputs and return either observation-level classifications (as the Week 3 temporal-persistence Parquet does today) or scene-pair metrics.

**Justification given current project state:**

1. **The custody project's observed data is scene-structured, not stream-structured.**  7 scenes across 9 months is the real cadence.  ADR-0017 already established that forcing single-scene SAR into the continuous-stream tracker abstraction produces no value.  The project's empirical reality is batches of scene-bound observations separated by long gaps.
2. **Driver scripts already treat scenes as the operational unit.**  Each `scripts/NN_detect_vlm_*.py` loads one scene, produces one Parquet.  Making `Scene` explicit in code formalizes what's already happening.  Low architectural risk because the refactor follows an existing boundary.
3. **It doesn't preclude Option B.**  Per-scene covariance calibration (Option B's core insight) is a natural operation on a `Scene` object — the Scene's metadata is exactly what you need to compute "what's the appropriate `cov_pos` for observations from this scene."  Option C creates a home for Option B's logic; the two compose.
4. **It doesn't preclude Option D.**  If future work brings AIS + SAR into the same fusion picture (per the deferred work in ADR-0017), the tracker can consume a `Scene` as a batch update exactly the way Option D proposes — but only when continuous AIS streams exist alongside.  For now, the tracker focuses on within-scene association; cross-scene work is explicit.
5. **Sentinel-1 integration (ADR-0016) benefits from Option C directly.**  Adding Sentinel-1 adds more scenes with their own metadata; `Scene` as a type gives us a uniform abstraction that handles "Umbra scene" and "Sentinel-1 scene" via the same interface, without committing to a particular within-scene observation-level correspondence across sensor types.

**What this ADR does not commit to:**

- The specific field set on `Scene` (likely iterative; start with sensor, timestamp, pixel size, center, footprint, quality flag, observations list).
- The specific cross-scene matching algorithms used by downstream classifiers.  The Week 3 direct-50 m-matcher remains valid for scenes with near-identical geometry; geometry-aware matchers for heterogeneous pairs are future work.
- The specific Python module location of `Scene`.  `custody.fusion.scenes` vs `custody.detection.scenes` depends on whether Scene is a fusion-layer concept (how observations get consumed) or a detection-layer concept (where observations come from).  Both locations are defensible; decide at implementation time.
- Any refactor of existing Parquet files.  Committed VLM-detection Parquets stay as-is; `Scene` gets populated from them at read time via a loader helper.

## Separate sub-decision: scene quality screen

Independent of options A–D, **adopt a minimal scene quality screen before VLM dispatch.**

**Motivation**: Tennent 08-09 burned ~$1 on 121 VLM calls that returned zero detections, against a scene whose featurelessness was visible in the first 1% of a decimated scene preview.  A 30-second pre-flight intensity check on the AOI-crop (or full scene) would have flagged the acquisition for review before dispatch.

**Specification:**

- New helper in the detection module (likely `src/custody/detection/quality.py` or similar): compute scene-wide intensity std and p99 on a decimated read of the scene.  Return a `SceneQuality` dataclass with `std`, `p99`, and a boolean `low_snr_flag`.
- **Threshold calibrated from observed data**: flag `low_snr_flag=True` if scene-wide p99 < 75 on uint8 GEC amplitude.  Derived from the 08-09 (p99 = 58, failed) vs 07-02 (p99 = 102, succeeded) contrast — 75 is a provisional threshold selected before the implementation measures p99 on all 7 committed scenes.  Finalize the threshold empirically during implementation: run the quality computation on all committed scenes, confirm the six known-good scenes cluster above threshold and only 08-09 falls below, and adjust if that's not the case.  Threshold may also need to be AOI-dependent rather than scene-wide (reef AOI stats may differ systematically from full-scene stats).  These calibrations happen at implementation time, not in this ADR.
- Called from the pre-flight code path in each scene driver (`scripts/NN_detect_vlm_*.py`), either inline or via the `estimate_scene_cost` path.  Flagged scenes get a warning-level log message and an `sys.exit(3)` with an informative message; operator confirms via a `--force` flag or re-runs after diagnosing the flag.
- Scope: ~50 LOC for the quality computation, ~10 LOC to wire into each driver (or a single line if we centralize pre-flight in a helper).

**Not committing to:**

- The per-scene quality metadata propagating into the `Scene` type chosen above.  Natural fit; implement after Option C lands.
- Any automatic triage of flagged scenes (auto-retry, auto-exclude from canonical dataset, etc.) — human-in-the-loop review is fine for this cadence of scenes.

## Consequences

### What this unlocks

1. **A clean path for the 5-scene Tennent temporal timeline.**  With `Scene` as a type, the temporal-persistence classifier becomes `temporal_persistence(scene_a, scene_b, matcher=...)` — the matcher can be swapped between the existing 50 m direct matcher (for near-identical geometry pairs) and future geometry-aware matchers without re-architecting the classifier.
2. **Future geometry-aware matching has a home.**  Whatever algorithm ends up working for heterogeneous scene pairs — covariance-weighted matching, geometric-registration-then-match, signature-based cross-correlation — lives as a `Matcher` implementation that a `Scene` pair feeds.  Clean extension point.
3. **Sentinel-1 integration (ADR-0016) has a clean onramp.**  Adding Sentinel-1 means adding a new scene producer and possibly new matchers; the `Scene` abstraction and the existing within-scene VLM pipeline keep working.
4. **Scene quality screens become first-class.**  The sub-decision above slots into `Scene.quality` naturally once the type exists, enabling operator dashboards and pre-flight gates to be consistent across all scene-producing pipelines.
5. **Tracker's role narrows to within-scene association (plus future continuous-stream work).**  The current `Tracker.step()` is designed for continuous observation arrivals; post-Option-C the tracker's cross-scene work is confined to within a single Scene's observations, with cross-scene analysis handled by dedicated Matcher implementations operating on Scene pairs.  This is not a refactor — the tracker code stays as-is — but its conceptual scope narrows, and the architectural expectation is that the continuous-stream code path activates when AIS coverage materializes (per ADR-0017 deferred work, and Option D becomes relevant then).

### What this defers

1. **Per-scene covariance calibration methodology (Option B's core deliverable)** is deferred.  Option C creates the home for it but doesn't require it to be filled in immediately — the existing `cov_pos` values (from `sigma_m=20.0` defaults in `vlm_detection_to_observation`) remain valid for within-scene work; per-scene calibration lands when a downstream classifier actually needs it.
2. **Cross-sensor pooling** (Umbra + Sentinel-1 observations feeding a single fusion track) is deferred.  The Scene type will accommodate both, but the algorithmic question of "when can observations from two different sensors be associated" is independent and not addressed here.
3. **AIS + SAR fusion in the tracker** is deferred per ADR-0017.  Option D's hybrid remains the likely future design when AIS coverage materializes, with `Scene` events feeding the tracker as batch updates.  Not blocking; not today's problem.

### Downstream ADR candidates

- **ADR-TBD — geometry-aware matcher for heterogeneous scene pairs.**  The specific algorithm chosen for matching 08-13 against 07-02 (or whatever cross-geometry pair we tackle first) warrants its own ADR, including the failure modes it's designed to handle.
- **ADR-TBD — per-scene covariance calibration methodology.**  If / when we commit to populating scene-derived covariances rigorously, the calibration procedure (incidence-angle lookup, pixel-size scaling, look-direction resolution-anisotropy, etc.) needs its own decision record.
- **ADR-TBD — Sentinel-1 Scene subtype or shared Scene.**  When Sentinel-1 integration lands, a decision on whether Scene is a union type with per-sensor subtypes or a single type with polymorphic metadata.

## Implementation (2026-04-21)

Accepted and implemented same day as drafted:

- `src/custody/fusion/scenes.py` (commit `b25bf91`): Scene dataclass + `load_scene_from_parquet` with pragmatic fallback chain (kwargs → notes_json → sidecar → filename → observation-derived approximation → `SceneLoaderError`).  All 6 committed parquets load cleanly via sidecar fallback.
- `src/custody/detection/quality.py` (commit `02958c7`): Scene quality screen per the sub-decision.  Dynamic-range metric (p99 − p50) instead of the provisional p99-alone threshold — empirical calibration across all 7 scenes showed Whitsun's dim-water characteristic defeats p99-alone; dynamic-range discriminates cleanly.  Thresholds: AOI-mode red < 15, yellow < 25, green ≥ 25; full-scene-mode red < 10, yellow < 12, green ≥ 12.
- `src/custody/fusion/temporal.py` (commit `63fb4a8`): first downstream consumer of Scene.  `Matcher` Protocol + `DirectSpatialMatcher` + `temporal_persistence(scene_a, scene_b, *, matcher)`.  Reproduces Week 3 Tennent 07-02↔07-23 result (22 persistent / 22 emerged / 20 disappeared) via Scene pairs.

### Calibration

The initial sub-decision proposed `p99 < 75` on uint8 GEC amplitude
as the low-SNR flag. Implementation measurement across all 7 scenes
showed this rule fails on Whitsun: healthy Whitsun full-scene p99
(60-61) is nearly identical to the 08-09 null's full-scene p99 (58).
A p99-only threshold either false-flags healthy Whitsun or misses
the confirmed 08-09 null.

Dynamic range (p99 − p50) discriminates cleanly across the 7-scene
dataset:

| scene | DR | mode | flag |
|---|---:|---|---|
| tennent_20230702 | 80 | AOI | green |
| tennent_20230723 | 69 | AOI | green |
| tennent_20230807 | 35 | AOI | green (tightest Tennent margin) |
| tennent_20230809 |  9 | AOI | **red** |
| tennent_20230813 | 77 | AOI | green |
| whitsun_20231206 | 12 | full-scene | green (zero-slack boundary) |
| whitsun_20240320 | 12 | full-scene | green (zero-slack boundary) |

Final thresholds:

- AOI mode: red < 15, yellow < 25, green ≥ 25
- Full-scene mode: red < 10, yellow < 12, green ≥ 12

Two calibration anchors worth naming explicitly because each drives
a threshold choice:

1. **Tennent 08-07 at AOI DR=35** is the floor of the healthy Tennent
   AOI distribution (other scenes sit at DR=69-80). It was collected
   under the coarsest pixel size in the Tennent set (UMBRA-04, 0.52
   m/px), which aligns with its compressed dynamic range. The AOI
   yellow threshold of 25 leaves 08-07 with 10 points of margin.
2. **Whitsun at full-scene DR=12 exactly** for both scenes is
   zero-slack against the strict-less-than-12 yellow threshold. The
   thresholds are fit to the two observed Whitsun scenes; a future
   Whitsun-like scene measuring DR=11 would trip yellow. This is
   deliberate, not accidental — the calibration is tuned to the
   present dataset and assumes future Whitsun-like captures will
   cluster near the observed values.

The AOI-vs-full-scene threshold asymmetry reflects a distribution
difference, not a threshold inconsistency: AOI crops concentrate the
bright-scatterer population into a smaller frame, producing
systematically higher DR than full-scene mode over the same
underlying scene. The same physical signal quality produces different
DR values at different crop extents.

Full per-scene stats (mean / std / p50 / p90 / p99 / DR, full-scene
and AOI) and the interpretation behind each anchor are preserved in
`docs/calibration/scene_quality_7scene_20260421.md`. Re-run via
`day0/scratch/quality_calibration_run.py`.

Part D diagnostic (`day0/scratch/tennent_all_pairs_persistence.md`, uncommitted) ran all Tennent scene pairs through the migrated classifier.  Confirms: same-geometry pairs (07-02↔07-23) cluster matches well inside the 50 m gate; cross-geometry pairs (anything involving 08-07, 08-13) produce fewer matches with mean distance pushed toward the gate boundary.  Mean distance inflation is the signal of forced Hungarian couplings across genuine scatterer displacement.  This empirically confirms the need for a geometry-aware Matcher implementation (ADR-TBD) and validates that the Matcher Protocol extension point is the right architectural location for it.

Deferred to future ADR-TBD work: per-scene covariance calibration methodology, geometry-aware matcher, Sentinel-1 Scene subtype.

## Status

**Accepted** — implementation landed in commits `b25bf91`, `02958c7`, `63fb4a8` on 2026-04-21.
