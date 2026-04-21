---
id: 0019
title: Geometry-aware matching direction for heterogeneous SAR scene pairs
date: 2026-04-21
status: proposed
---

## Context

ADR-0018 established `Matcher` as a Protocol extension point in `custody.fusion.temporal` and landed `DirectSpatialMatcher` as the V1 implementation — nearest-neighbor matching within a fixed meter gate.  The Week 3 Tennent 07-02↔07-23 temporal classifier uses it at `gate_m=50.0` and produces 22 persistent / 22 emerged / 20 disappeared with mean match distance 21.3 m.

Part D of the Scene-migration work (`day0/scratch/tennent_all_pairs_persistence.md`, uncommitted) ran `DirectSpatialMatcher(gate_m=50.0)` across every chronologically-ordered pair of the 4 committed Tennent scenes (6 pair runs; 4 pairs involving the excluded 08-09 scene marked N/A).  Results expose two degradation signals on cross-geometry pairs:

| | same-geom baseline (07-02↔07-23) | cross-geom range (5 pairs) |
|---|---:|---:|
| persistent count | 22 | 6 – 11 |
| mean match distance | 21.3 m | 30.8 – 37.6 m |

The mean-distance inflation is the cleaner signal.  Same-geom matches cluster well inside the 50 m gate; cross-geom "matches" are pushed toward the gate boundary — signature of Hungarian forcing couplings despite systematic positional shifts, not genuine same-scatterer co-location.  Tellingly, **08-07↔08-13 (only 6 days apart, same reef, different orbit)** performs worst (6 matches, mean 30.8 m) — confirming the issue is geometric presentation, not physical change over time.

`DirectSpatialMatcher` has no mechanism to distinguish "same scatterer projected to a shifted pixel position under different orbit geometry" from "different scatterer that happens to sit nearby."  Both present as "close enough under the spatial gate."

ADR-0018 already committed to the Matcher protocol as the architectural extension point.  **ADR-0019's scope is the specific matcher implementation direction**, not the architecture.

## Problem statement

For cross-geometry SAR scene pairs (different sensor, pixel size, orbit direction, or acquisition time), observations of the same physical object appear at systematically different projected positions.  `DirectSpatialMatcher` treats these shifts as "different object" and produces degraded persistence classifications.

A geometry-aware matcher needs a principled way to decide: given two observations at displaced positions, are they the same physical object or different objects that happen to be near?

## Options

Three approaches, evaluated against what we know today.

### Option A — Scene-metadata-driven offset compensation

Compute a scene-pair registration shift from SAR metadata (incidence angles, look directions, orbit positions) and apply as a bulk correction to scene B's observations before matching.  Matcher then runs `DirectSpatialMatcher` on compensated coordinates.

**Assessment.**  Cheap, uses metadata we already have (`Scene.sensor`, `Scene.pixel_size_m`, footprint).  First-order correction — handles bulk offset between scenes well, fails on second-order effects (geometric distortion varying across the scene, terrain-induced displacement without a DEM).  Adequate for pairs where geometry-driven shift is approximately uniform; inadequate where shift has significant spatial structure.

### Option B — Feature signatures alongside position

Augment each `PositionObservation` with a signature: a vector capturing object-level attributes (bbox dimensions, detection reasoning embedding, local intensity profile, confidence, classification label).  Cross-scene matching combines spatial gating with signature similarity; two observations "match" when they're close enough in both spatial and signature spaces.

**Assessment.**  Uses information the VLM already produces.  Additive — doesn't require new data pipelines.  Works without geometric correction because signature similarity is invariant to position shift.  Failure mode: signatures may not carry enough distinctive signal for all scenes (e.g., a flotilla of similar small vessels may look similar to each other).  Incremental — V1 can use simple bbox-based signatures; V2 can add VLM reasoning embeddings.

### Option C — Pixel-level scene registration

Register scene images to a common coordinate system via image-to-image alignment (phase correlation, feature-based registration, etc.) before running detection.  Downstream observations are in a unified reference frame; matching becomes the current 50 m gate trivially.

**Assessment.**  Most accurate in principle.  Handles geometric distortion at all spatial scales.  Significant implementation complexity — SAR registration is its own research problem; image-pair registration can fail silently on heterogeneous acquisitions (different look directions, different speckle realizations).  Expensive — re-detection on registered scenes instead of using existing detections.  Correct long-term direction; premature for current project state.

## Decision

Commit to **Option B (feature signatures)** as the V1 geometry-aware matcher direction.

Justification:

1. **Lowest-risk path to measurable improvement over `DirectSpatialMatcher`.**  Additive to the existing data pipeline; existing committed VLM detections already carry the V1 signature fields (`bbox_px`, `classification_conf`).
2. **Uses information already present in VLM detections** (bbox dimensions, reasoning text).  No new data generation required for V1.
3. **Part D's baseline provides a measurable success criterion**: improve cross-geometry persistent counts toward the same-geometry 22 baseline without sacrificing same-geometry accuracy.
4. **Extensibility**: signature representation can be enriched (add reasoning embeddings, intensity profiles, classification labels) without redesigning the matcher.
5. **Option A and Option C remain viable future options** if Option B plateaus.  The Matcher protocol supports multiple concurrent implementations; the fusion layer stays clean as matchers accumulate.

### V1 signature fields

Commit to a minimal V1 signature comprising:

- bbox width and height (from `bbox_px` in `PositionObservation`)
- bbox aspect ratio
- `classification_conf` (present on `PositionObservation`)

Defer to future versions:

- VLM reasoning text embeddings (requires embedding model choice)
- Local intensity profile at detection point (requires re-reading scene rasters)
- Vessel length estimate (`vessel_length_est_m` — present on some observations, absent on others)

### V1 matching rule

`SignatureMatcher` candidate design:

- Compute pairwise spatial distance (same tangent-plane AEQD as `DirectSpatialMatcher`) — `gate_m` parameter retained.
- Compute pairwise signature distance (L2 on normalized feature vector) — new `sig_gate` parameter.
- Match allowed only when spatial distance ≤ `gate_m` **AND** signature distance ≤ `sig_gate`.
- Hungarian assignment on combined cost (weighted sum or product of the two distances — tune at implementation time).
- Produces same `Match` / `TemporalComparisonResult` output as `DirectSpatialMatcher`.  Drop-in for existing `temporal_persistence` callers.

## Consequences

- **Next implementation work**: `SignatureMatcher` as a new entry in `fusion/temporal.py` alongside `DirectSpatialMatcher`.  Same `Matcher` protocol, same output type.
- **Measurement**: run the same Part D pair matrix with `SignatureMatcher`; compare persistent counts and mean distances.  Expected outcome: cross-geometry pairs recover toward same-geometry baseline.  If so, ADR-0019 flips to `accepted`.
- **V1 may be insufficient** for pairs with minimal signature diversity (e.g., flotilla of similar small vessels).  Empirical — V2 signature enrichment triggered by that finding.
- ADR-0018's Scene abstraction already provides the per-scene metadata (`pixel_size_m`, `sensor`, `footprint_latlon`) that future Option A or Option C matchers would need.  **No Scene schema change required for any matcher direction.**

## Downstream ADR candidates

- **ADR-TBD on signature enrichment strategy** — if V1 plateaus and reasoning embeddings / intensity profiles become necessary.
- **ADR-TBD on scene registration pipeline** — if Options A/C become needed (e.g., for Sentinel-1 vs Umbra cross-sensor pairs where bbox geometry differs fundamentally).
- **ADR-TBD on per-scene covariance calibration** — already flagged in ADR-0018; integrates with whatever matcher is chosen.

## Status

Proposed.
