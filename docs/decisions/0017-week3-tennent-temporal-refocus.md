---
id: 0017
title: Week 3 refocus from AIS+SAR fusion to Tennent temporal analysis
date: 2026-04-21
status: accepted
---

## Context

Week 3 was originally planned as AIS+SAR cross-modality fusion work: feeding real VLM SAR detections and real GFW AIS observations through the existing `custody.fusion.tracker.Tracker` to produce correlated multi-modality tracks.  The target output was a classifier distinguishing AIS-correlated vessels from AIS-dark vessels near our case-study reefs — the headline narrative in `docs/positioning.md` and the orchestration layer's tip-and-cue design.

A pre-implementation integration pilot (see `docs/investigations/ais_coverage_investigation.md`) checked the actual temporal overlap between the AIS parquet and our three ingested VLM SAR scenes before any new code was written.  The finding:

**Zero AIS observations fall within ±7 days of any SAR acquisition inside any AOI.**

Not sparse — zero.  Windowed at ±30 minutes, ±6 hours, ±24 hours, or ±7 days, the counts are identical (0).  The GFW presence parquet contains 39,337 rows across 80 days (2023-06-01 to 2023-08-19) at 1-hour quantization and covers the right geographic box (8.5–11°N, 114.5–117.5°E).  What it does not contain is AIS activity near Tennent Reef or Whitsun Reef during SAR acquisition windows:

| scene | AIS in ±7 days of SAR acquisition, inside AOI | AIS anywhere in AOI across 80-day parquet |
|---|---:|---:|
| Tennent 2023-07-02 | 0 | 4 |
| Tennent 2023-07-23 | 0 | 4 |
| Whitsun 2023-12-06 | 0 | 344 (all outside parquet temporal range) |

The Tennent AOI sees 4 AIS transits *in total* across the entire 80-day window.  None coincide with either SAR acquisition.  The Whitsun AOI sees more transits, but every one of them falls outside the parquet's temporal range relative to the December SAR scene (parquet ends 2023-08-19; scene is 2023-12-06).

This is the case-study scenario expressing itself quantitatively.  Contested-reef activity — reclamation at Tennent, maritime-militia flotillas at Whitsun — is known from AMTI reporting to be predominantly AIS-dark.  The GFW data confirms this: the reefs simply do not emit AIS.  The planned "correlate tracks across modalities" Week-3 framing has no signal to work with in this dataset, because the contested-reef scenarios we chose are precisely the cases where AIS-SAR correlation fails by design.

This was not evident at Week 2 close because AIS coverage near the reefs had not been measured; the assumption was "sparse but non-zero."  It is zero.

## Decision

**Week 3 refocuses on Tennent 07-02 → 07-23 temporal analysis using the existing tracker.**

Two SAR acquisitions over the same AOI 21 days apart — the pair we already ingested as `data/processed/vlm_detections/position.parquet` and `.../tennent_20230723_position.parquet` — feed into `Tracker.step()` as a two-timestep sequence.  This exercises prediction + Hungarian association + N-of-M confirmation against real data (the tracker's previously untested path — `tests/test_fusion_tracker.py` is all synthetic).  Track persistence across the 21-day interval becomes the primary classifier:

- **Present in both scenes at spatially consistent positions** → probable fixed infrastructure (cranes, containers, platform edges on the reclamation structure).
- **Present in only one scene, or with materially different positions across the pair** → candidate mobile target (vessels moored transiently or in transit).

This directly answers the moored-vessel-vs-fixed-equipment disambiguation problem flagged in the Tennent 07-02 README and elaborated in the 07-23 README's temporal-comparison section.  The question was already asked in prose; Week 3's job is to answer it with a tracker-driven classifier.

Concrete Week-3 scope:

1. Feed Tennent 07-02 + 07-23 VLM detections into `Tracker.step()` as two-timestep sequence.  Calibrate `n_of_m`, `gate_sigma`, and `max_coast_steps` for two-scene persistence (defaults tuned on synthetic multi-step data will not fit).
2. Classify resulting tracks as "persistent" (both scenes) vs "transient" (one scene).  Report counts against the 42 + 44 = 86 input detections.
3. Write a small persistence-classifier module under `src/custody/fusion/` or similar (location TBD at implementation time).  This is a new module; there is no existing "persistent-vs-transient" detector in the codebase.
4. Produce a side-by-side overlay image showing persistent detections (fixed infrastructure) vs transient detections (mobile candidates) across the two Tennent scenes.  This is the demo-grade visual the temporal-analysis narrative has been pointing at since ADR-0012.

## Consequences

- **AIS+SAR correlation work is deferred** to a future phase contingent on either (a) a denser AIS dataset — a shipping-lane-adjacent AOI where commercial traffic actually broadcasts, or (b) a synthetic AIS injection scenario constructed to make the tracker's cross-modality path visible for demo purposes.  Neither is urgent; both are straightforward when the data exists.
- **The "100% AIS-dark activity at contested reefs" finding is a project result, not a failure mode.**  It quantitatively confirms the contested-reef scenario framing from ADR-0012 and ADR-0013.  `docs/investigations/ais_coverage_investigation.md` captures the numbers for future reference and for any external artifact (post, whitepaper, demo narration).
- **No code deletion.**  `src/custody/ingest/gfw_presence.py`, `PositionObservation.modality="AIS"` (ADR-0011), and the tracker's AIS-handling path all remain in place.  They work — the pilot showed end-to-end plumbing is fine.  What's missing is AIS data that temporally overlaps the SAR windows, not code that processes it.
- **Narrative implication for `docs/positioning.md` and the demo script.**  The "AIS-dark activity detection" framing remains valid and becomes *stronger*: we are not merely identifying AIS-dark contacts amid a busy correlation landscape, we are documenting that the contested reefs are structurally AIS-dark across multi-month datasets.  The supporting evidence shifts from "SAR-detections-without-AIS-correlates" to "SAR-detections-in-AOI-windows-with-zero-AIS-coverage."  Positioning.md text update is a follow-up; not landing in this ADR.
- **Tracker finally gets exercised against real data.**  `tests/test_fusion_tracker.py`'s 16 tests are all synthetic (pyproj.Geod-generated tracks with controlled noise).  The temporal pilot forces the tracker to associate real VLM bounding-box centroids with their ~20 m σ across a 21-day dt — the actual operating regime it was designed for.  Whatever gate / lifecycle tuning that surfaces is work we need to do anyway.
- **Persistence classifier is a new module, not a shim.**  `custody.anomalies` is a compatibility re-export over `custody.behavior.detectors`, which takes timeline-of-labeled-records inputs; `custody.dark_vessel` marks already-identified entities as dark after a decision-layer judgement.  Neither fits the "classify tracker output by cross-scene persistence" shape.  Expect a small new module; implementation location decided at Week-3 implementation time.

## What this doesn't change

- **Tracker architecture.**  `Tracker.step()`, `TrackState`, N-of-M lifecycle, Hungarian association — all unchanged.  Week 3 drives this code rather than modifying it.
- **Detection pipeline.**  VLM detection as implemented in Phase F (`custody.detection.vlm_sar.detect_vessels_in_scene`) is unchanged.  The three committed VLM parquets are the inputs.
- **Case study framing.**  Tennent and Whitsun as dual case studies (ADR-0013) stand.  Zero AIS near the reefs is a finding *within* that framing, not a revision of it.
- **Fusion layer schema.**  `PositionObservation`, `PositionVelocityObservation`, and the Phase F `bbox_px` extension stay as-is.  ADR-0011's AIS-as-position-only path remains the contract even though no AIS rows exercise it in the contested-reef windows.
- **Budget envelope.**  No additional API spend required for the Week-3 pivot; the Tennent parquets are already committed.  The Week-3 work is tracker + classifier code, not more VLM runs.

## Supersedes

Nothing.  **Supplements ADR-0016** ("Story 2 scope") with the empirically-grounded Week-3 direction, and **extends the temporal-comparison methodology** sketched in the multi-scene README (`data/processed/vlm_detections/README.md`) by committing to it as the Week-3 deliverable rather than a prose observation.
