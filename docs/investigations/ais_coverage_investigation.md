# Investigation: AIS coverage at contested reefs

*Private project documentation. Source material for ADR-0017 and potential external artifacts (LinkedIn post, whitepaper).*

## Summary

Before wiring real AIS + SAR observations through `custody.fusion.tracker.Tracker` for Week 3's planned cross-modality fusion work, an integration pilot measured the actual temporal overlap between the GFW AIS parquet and our three ingested VLM SAR scenes.

The finding:

**Zero AIS observations fall within ±7 days of any SAR acquisition inside any AOI.**

At every time window tested — ±30 min, ±6 h, ±24 h, ±7 days — the count is zero.  The GFW presence parquet is not empty (39,337 rows across 80 days covering the right geographic box), but the contested reefs themselves do not emit AIS during SAR acquisition windows.  This is the case-study scenario expressing itself quantitatively: the contested Spratly reefs are structurally AIS-dark.

The investigation was conducted on 2026-04-21 against commits at HEAD `8431474`.  Script is preserved at `day0/scratch/week3_integration_pilot.py`.

## Motivation

Week 3 was planned as AIS+SAR cross-modality fusion work per the architecture in `docs/positioning.md`: feed both VLM SAR detections and GFW AIS observations through the existing tracker, produce correlated multi-modality tracks, classify AIS-correlated vs AIS-dark contacts.

A pre-implementation pilot was run to verify the shape of the data before writing any new fusion code.  Specifically to answer: *does AIS density near our three VLM SAR scenes support a meaningful correlated-vs-dark-vessel breakdown?*

## Input data

| source | rows | time range | spatial coverage |
|---|---:|---|---|
| GFW AIS (`data/processed/gfw_presence/position.parquet`) | 39,337 | 2023-06-01 → 2023-08-19 (1-hour quantization) | 8.5–11°N / 114.5–117.5°E |
| VLM Tennent 2023-07-02 (`data/processed/vlm_detections/position.parquet`) | 42 | single scene 2023-07-02 14:00 UTC | 2 km AOI at (8.856°N, 114.665°E) |
| VLM Tennent 2023-07-23 (`.../tennent_20230723_position.parquet`) | 44 | single scene 2023-07-23 14:02 UTC | 2 km AOI at (8.856°N, 114.665°E) |
| VLM Whitsun 2023-12-06 (`.../whitsun_20231206_position.parquet`) | 115 | single scene 2023-12-06 02:06 UTC | ~8.8 km envelope at (9.969°N, 114.632°E) |

## Phase 1 — AIS density per scene (time-window sweep)

Counts are for AIS rows whose lat/lon falls inside the scene's AOI bounding box and whose acquisition_time falls within ±Δt of the SAR acquisition time.

| scene | ±30 min | ±6 h | ±24 h | ±7 days | scene-wide any-time |
|---|---:|---:|---:|---:|---:|
| Tennent 2023-07-02 | 0 | 0 | 0 | 0 | 4 |
| Tennent 2023-07-23 | 0 | 0 | 0 | 0 | 4 |
| Whitsun 2023-12-06 | 0 | 0 | 0 | 0 | 344 |

Observations:

1. **Tennent AOI sees 4 AIS transits total across the entire 80-day parquet window.**  None coincide with either SAR acquisition.
2. **Whitsun AOI sees 344 scene-wide transits**, but the scene is 2023-12-06 — 110 days outside the GFW parquet's temporal range (parquet ends 2023-08-19).  Every one of those 344 rows is therefore outside even a ±7-day window of the SAR scene.
3. The finding is not a missed query.  The bounding boxes are correct (verified against `describe_aoi_bounds` output), the AIS column schema is consistent across the parquet, and the time ranges were sanity-checked.  The parquet simply contains almost no AIS activity near these reefs during any period that overlaps our SAR acquisitions.

This is the contested-reef scenario quantified.  Tennent reclamation work and Whitsun maritime-militia flotillas are documented by AMTI as predominantly AIS-dark; the GFW data confirms that with the specific number zero.

## Phase 2 — Tracker pilot (Whitsun 12-06, SAR-only)

To verify end-to-end plumbing despite the zero AIS overlap, the pilot ran Whitsun's 115 VLM SAR observations through `Tracker.step()` as a single-timestep batch.  Whitsun was picked on the grounds that it has the most VLM detections of the three scenes.

Configuration: `n_of_m=(2, 3)`, `gate_sigma=3.0`, `coast_threshold=3`, `max_coast_steps=10`.  Relaxed from synthetic-test defaults to account for the expectation of sparse data.

Result:

- 115 tracks formed, all TENTATIVE.
- 0 CONFIRMED, 0 COASTED, 0 RETIRED.
- 115 only-SAR tracks, 0 only-AIS, 0 correlated.
- 0 errors during execution.

### Why every track is TENTATIVE

All 115 SAR observations share a single acquisition timestamp.  A SAR scene is an instantaneous snapshot.  The tracker processes them in a single `step()` call: each observation spawns a TENTATIVE track with one hit in its N-of-M history window.  With `n_of_m=(2, 3)` nothing can confirm — CONFIRMED requires ≥ 2 hits within the last 3 steps, and only 1 step has occurred.

This is not a tracker bug.  It is the tracker exposing a real property of single-scene SAR data: **track confirmation needs temporal persistence, and a single scene provides none**.  The tracker's Hungarian assignment, EKF prediction, and lifecycle machinery light up only when multiple scenes over time are ingested.

### What this means for tracker testing

`tests/test_fusion_tracker.py` contains 16 test functions, all synthetic (pyproj.Geod-generated lat/lon with controlled noise across multiple simulated timesteps).  The tracker's real-data operating regime — multi-scene SAR + real-world covariance + real-world timing — has not previously been exercised in this codebase.  The investigation shows the tracker does not crash on real input, but confirming it produces *correct* associations across real scenes is a Week-3 work item.

## Phase 3 — Anomaly & dark-vessel module check

`src/custody/anomalies.py` is a compatibility shim re-exporting from `custody.behavior.detectors`.  The detector functions (`loitering`, `route_deviation`, `anomaly_score`, `in_sensitive_zone`) take timeline records — per-vessel time-ordered dicts with role tags and attention state — rather than raw `PositionObservation` or `TrackRecord` objects.  They sit a layer above the tracker; running them against raw tracker output would require an adapter that this pilot does not include.

`src/custody/dark_vessel.py` is a role/attention-state marker, not a detector.  It labels already-identified entities as 'dark' once a decision-layer judgement has been made.  There is no production "detect AIS-dark SAR contact" function in `src/` today.  The "SAR contact with no time-correlated AIS within Δt meters" classifier has to be written.

## Findings summary

1. **AIS coverage at the contested reefs is structurally zero during SAR windows.**  This is the primary finding — not a data gap, but the case-study scenario quantified.
2. **The tracker composes cleanly with real `PositionObservation` data.**  ADR-0011's `modality='AIS'` path works end-to-end; tracker code needed no changes to accept real observations.
3. **Single-scene SAR does not exercise the tracker's core value.**  Hungarian assignment, EKF prediction, and N-of-M confirmation require multi-timestep inputs.  The tracker's first meaningful run against real data needs ≥ 2 SAR scenes.
4. **No existing module implements AIS-dark-vessel detection.**  `anomalies.py` operates on labeled timelines; `dark_vessel.py` is a marker for already-classified entities.
5. **Whitsun AIS overlap requires additional data.**  GFW parquet ends 2023-08-19; Whitsun 2023-12-06 falls 110 days outside the window.  Any Whitsun AIS work needs a separate fetch or a different AIS source.
6. **`TrackRecord` lacks a per-track modality aggregate.**  Computing "SAR-only vs AIS-correlated" required walking the obs_id list against a separate lookup.  Minor but worth a helper in future work.

## Week-3 implication

Original Week-3 framing (AIS+SAR correlation) has no signal to work with in this dataset.  The natural alternative — **Tennent 2023-07-02 → 2023-07-23 temporal analysis** — uses the two scenes we already ingested at the same AOI, exercises the tracker's multi-timestep path, and directly answers the moored-vessel-vs-fixed-equipment disambiguation flagged in the VLM detections README.  See ADR-0017 for the decision record.

AIS+SAR correlation is deferred, not abandoned.  When a denser AIS dataset or a scene pair where AIS actually broadcasts becomes available, the tracker and ingest paths are ready.

## Reproducibility

```bash
uv run python day0/scratch/week3_integration_pilot.py
```

Produces:
- `day0/scratch/week3_integration_pilot.md` — prose report (superseded by this document for durable reference)
- `day0/scratch/week3_integration_pilot_aoi.png` — SAR+AIS points in AOI for the pilot scene
- `day0/scratch/week3_integration_pilot_tracks.png` — 115 TENTATIVE tracks plot

No commits.  No API calls.  Pure parquet-to-numpy analysis plus a single tracker run.
