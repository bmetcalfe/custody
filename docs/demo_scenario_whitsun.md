# Whitsun Decision Trace — Demo Scenario

*Companion to [docs/positioning.md](positioning.md), [docs/scenario.md](scenario.md), [docs/sentinel_ingestion_whitsun.md](sentinel_ingestion_whitsun.md), and [docs/sentinel_ingestion_demo_aois.md](sentinel_ingestion_demo_aois.md). This document describes the canonical Whitsun custody / reacquisition decision trace fixture.*

---

## What this is

A single deterministic JSON document — `data/demo/whitsun_decision_trace.fixture.json` — that captures one full custody / reacquisition walkthrough over the Whitsun AOI. It is the canonical scenario that the future Dash replay will render. It is intentionally a **fixture**, not a live event, and it is intentionally **JSON only**, not code that executes the decision pipeline.

The trace combines:

- **Umbra SAR** as the high-confidence tasked evidence source.
- **Sentinel-1 GRD** as lower-confidence public SAR context.
- **Sentinel-2 L2A** as optical / multispectral context with cloud-cover caveats.
- **Simulated candidate tracks**, **simulated tasking options**, and **simulated planner / RL rationale** as demo scaffolding.
- A 14-step event sequence that the future replay can step through.

Every field is labelled `data_mode: "fixture"` (records that are committed reference data) or `data_mode: "simulated"` (records that exist only as deterministic demo scaffolding).

## What this is not

- **Not a live decision.** Nothing in this fixture was produced by a real planner, a trained RL agent, or an operational tasking pipeline.
- **Not a real Sentinel query result.** The trace cross-references the existing committed Sentinel observation fixture at `data/demo/whitsun_sentinel_observations.fixture.json`; no live CDSE STAC call is performed.
- **Not a dashboard.** No Dash code is touched in this slice. The fixture is the data contract the dashboard will consume in a later step.
- **Not a Tennent decision trace.** Tennent shares the Sentinel ingestion AOI extension but is not part of this slice.
- **Not a tasking order, sensor command, or platform-access claim.** The selected recommendation, human approval, follow-up collection, and outcome are all simulated demo records.
- **Not a calibrated probability or measured KPI.** Confidence weights and scores are demo heuristics.

## Operational hierarchy preserved by the trace

The trace explicitly distinguishes four classes of record so the future replay UI cannot accidentally treat Sentinel context as equivalent to tasked Umbra evidence:

| Source class       | `confidence_weight` (demo heuristic) | Role                                             |
| ------------------ | ------------------------------------ | ------------------------------------------------ |
| `umbra-sar`        | 1.00                                 | High-confidence tasked evidence                  |
| `sentinel-1`       | 0.45                                 | Lower-confidence public SAR context              |
| `sentinel-2` low-cloud | 0.30                             | Optical / multispectral context, usable for both detection and context |
| `sentinel-2` cloudy   | 0.10                              | Optical context only (`usable_for_detection: false`) |
| simulated planner / RL records | n/a                       | Demo scaffolding; deterministic, not learned     |

This is the same hierarchy used by `src/custody/ingest/sentinel.py` and by the Sentinel ingestion fixtures.

## Multi-source pipeline proof, not Sentinel detection equivalence

The intent of the Sentinel layer in this scenario is to demonstrate that the system **handles multiple imagery / source pipelines according to quality and trust level**, not that Sentinel can substitute for Umbra:

- **Umbra** — high-confidence tasked SAR imagery; the confirmation / evidence layer.  Driven by `scripts/30_prepare_demo_overlays.py` over real Umbra GEC tiles.
- **Sentinel-2** — public optical context imagery when a valid preview exists.  Process API previews are fetched by `scripts/32_fetch_sentinel_previews.py` and rendered on the map at the Sentinel-2 reveal step.
- **Sentinel-1** — public SAR observation pipeline with graceful fallback.  When Sentinel Hub returns an empty placeholder for an acquisition window, the overlay stays footprint-only and the dashboard surfaces the reason verbatim.  Sentinel-1 imagery is **not** forced when no valid preview is returned.
- **Invalid / duplicate previews** — caught by quality gates in the fetcher (size < 5 KB, near-zero pixel variance, repeated sha256 within a run) and explicitly **not promoted**; the manifest stays footprint-only with a clear `missing_asset_reason`.

The committed `data/demo/map_overlays.fixture.json` typically lands in **mixed state** after a real fetch: some Sentinel-2 overlays promoted with `image_kind: "sentinel_preview"`, Sentinel-1 overlays footprint-only because the live API returned empty, and any duplicate Sentinel-2 acquisition footprint-only with `"Duplicate preview of <obs_id>; ..."`.  That mixed state is the proof of pipeline-aware handling.

## Event sequence (14 events)

Stable IDs `ev-01` … `ev-14`. Each event has a `kind`, a `summary`, a `timestamp`, a `data_mode`, and a `refs` block whose IDs are guaranteed to resolve into one of the normalized sections of the trace.

| ID    | Label                                              | Kind                                  |
| ----- | -------------------------------------------------- | ------------------------------------- |
| ev-01 | Scenario initialized                               | `scenario_init`                       |
| ev-02 | Umbra SAR tile received                            | `observation_received`                |
| ev-03 | VLM analysis complete                              | `evidence_artifact_produced`          |
| ev-04 | Candidate tracks initialized                       | `tracks_initialized`                  |
| ev-05 | Sentinel-2 context observation available           | `context_observation_available`       |
| ev-06 | Sentinel-1 context observation available           | `context_observation_available`       |
| ev-07 | Custody risk increases                             | `custody_risk_increase`               |
| ev-08 | Planner generates options                          | `options_generated`                   |
| ev-09 | Options scored                                     | `options_scored`                      |
| ev-10 | Policy recommends SAT-B                            | `policy_recommendation`               |
| ev-11 | Operator reviews                                   | `human_review`                        |
| ev-12 | Human approves                                     | `human_action`                        |
| ev-13 | Follow-up collection / outcome recorded            | `followup_collection_and_outcome`     |
| ev-14 | Counterfactual and follow-up recommendation shown  | `counterfactual_and_followup`         |

## Normalized sections

The trace JSON has these top-level keys:

| Section                       | What it carries                                                                |
| ----------------------------- | ------------------------------------------------------------------------------ |
| `scenario_metadata`           | Scenario id, role, AOI references, time window, hierarchy table, caveats       |
| `events`                      | The 14-event sequence above                                                    |
| `observations`                | Umbra (real-shape fixture + simulated follow-up) and Sentinel (cross-referenced from the Sentinel ingestion fixture) |
| `evidence_artifacts`          | VLM-derived cluster / isolate artifacts                                        |
| `detections`                  | Per-vessel detection records over the Umbra tile                              |
| `candidate_tracks`            | Three tracks: cluster A, cluster B (the reacquisition target), cluster C       |
| `custody_state_snapshots`     | Pre-event healthy, mid-event ambiguous, post-event healthy                     |
| `candidate_tasking_options`   | SAT-A, SAT-B, Wait, Optical, Expand Search                                     |
| `score_breakdowns`            | One score breakdown per option, components + total                             |
| `selected_recommendation`     | SAT-B (rank 1, total score 0.91)                                               |
| `policy_rationale`            | Demo planner / RL rationale; explicitly not a trained agent                    |
| `human_action`                | Operator approve record for SAT-B                                              |
| `outcome`                     | Reacquisition result and custody-score delta                                   |
| `counterfactuals`             | One per unselected option with expected result + score delta                   |
| `followup_recommendation`     | Deterministic guidance for the next 72-hour window                             |

## Why SAT-B wins

Given the simulated trk-002 AIS-dark separation event, SAT-B (matched-geometry Umbra repeat SAR) dominates on:

- **Ambiguity resolution** (0.85): direct high-confidence reacquisition signal.
- **Custody-health improvement** (0.78): largest expected score delta.
- **Timeliness** (0.85): lowest latency among high-confidence options.
- **Feasibility** (0.85): matched geometry, no cross-track or weather penalty.

SAT-A (cross-geometry SAR) is geometry-mismatched. Wait does not resolve the AIS-dark ambiguity. Optical depends on cloud cover. Expand-Search dilutes per-target reacquisition probability.

These component values are demo heuristics, exposed in the `score_breakdowns` section so the dashboard can render the breakdown verbatim.

## How the future Dash replay should consume this

1. Load `data/demo/whitsun_decision_trace.fixture.json` into memory.
2. Step through `events` in `ordinal` order.
3. For each event, resolve `refs` into the corresponding entries in `observations`, `evidence_artifacts`, `detections`, `candidate_tracks`, `custody_state_snapshots`, `candidate_tasking_options`, `score_breakdowns`, `selected_recommendation`, `policy_rationale`, `human_action`, `outcome`, `counterfactuals`, or `followup_recommendation`.
4. Render the operational hierarchy explicitly: Umbra records first, Sentinel records as context, simulated records labelled as such.
5. Show the `confidence_weight` per record, never collapse it into a single "confidence" badge that hides the source distinction.

The dashboard refactor itself is **out of scope** for this slice.

## File layout

```
data/demo/
  whitsun_aoi.fixture.geojson
  whitsun_sentinel_observations.fixture.json     # cross-referenced from the trace
  whitsun_decision_trace.fixture.json            # this slice's deliverable

docs/
  demo_scenario_whitsun.md                       # this document
  sentinel_ingestion_whitsun.md
  sentinel_ingestion_demo_aois.md

tests/
  test_demo_whitsun_decision_trace.py            # validates the trace
```

## Tests

`tests/test_demo_whitsun_decision_trace.py` validates:

- The fixture loads as valid JSON and matches the documented schema marker.
- All `events[*].refs.*` IDs resolve into the corresponding section.
- All four Sentinel observations referenced by the trace exist in the Sentinel ingestion fixture.
- Sentinel observations carry strictly lower `confidence_weight` than Umbra; no Sentinel record is treated as equivalent to Umbra.
- The five tasking options are exactly `SAT-A`, `SAT-B`, `Wait`, `Optical`, `Expand Search`.
- `SAT-B` is the selected recommendation.
- Every option has a score breakdown.
- `policy_rationale`, `human_action`, `outcome`, `counterfactuals`, and `followup_recommendation` exist.
- Every fixture / simulated record carries an explicit `data_mode`.

No test contacts the live CDSE STAC endpoint. No dashboard code is touched.
