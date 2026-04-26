# Whitsun Replay — Dash View

*Companion to [docs/demo_scenario_whitsun.md](demo_scenario_whitsun.md), [docs/sentinel_ingestion_whitsun.md](sentinel_ingestion_whitsun.md), and [docs/sentinel_ingestion_demo_aois.md](sentinel_ingestion_demo_aois.md). This document describes the Dash replay view that renders the Whitsun decision trace fixture.*

---

## What this is

A read-only Dash panel mounted as the **Whitsun replay (fixture)** tab in the existing Custody dashboard. It renders the canonical custody / reacquisition trace at `data/demo/whitsun_decision_trace.fixture.json` step-by-step.

It is intentionally:

- **Fixture-backed.** The trace is loaded once at module import; no live query is performed.
- **Self-contained.** The replay tab does not share state with the Custody overview / entity-detail tabs. It mounts alongside them but reads only the trace fixture.
- **Read-only.** No control inputs other than the timeline event selector. No tasking, no human-action mutation, no Sentinel live refresh, no decision-layer runtime is exercised.
- **Visually disciplined.** Source class (Umbra / Sentinel-1 / Sentinel-2 / simulated) and `data_mode` (fixture / simulated) are always badged. Confidence weights are surfaced explicitly. Policy rationale is labelled "heuristic advisory / RL-ready slot, not a trained RL decision".
- **Progressively revealed.** Panels show only what has been revealed by the selected event's ordinal. Event 01 hides post-decision panels; observations appear at event 02; tasking options appear at event 08; scores at event 09; selected recommendation + policy rationale + score breakdown at event 10; human approval at event 11; outcome at event 13; counterfactuals + follow-up recommendation at event 14. Panels with nothing yet to show render a small "not available at this step" placeholder rather than future-state spoilers.

## How to run

```bash
uv sync
uv run python src/app/dash_app.py
```

Then open the dashboard and switch to the **Whitsun replay (fixture)** tab.

(Existing dashboard tabs — Custody overview and entity detail — are untouched and continue to work.)

## What the panel renders

| Panel                              | Content                                                                                                                                                                |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Header                             | Scenario title, scenario id, AOI metadata, primary evidence source badge, data-mode badge, currently selected event ordinal + label + timestamp.                        |
| Event timeline (14 steps)          | Radio selector listing events `01. … 14. …` in ordinal order. Selection drives every other panel.                                                                     |
| Map / evidence overlays            | Real `dash_deck` (`pydeck`) map. Renders the AOI polygon, observation footprints (Umbra / Sentinel-1 / Sentinel-2, gated by event ordinal), candidate-track markers, and the latest custody snapshot as an on-map text annotation. Layer toggles let the operator turn each layer on/off; an opacity slider applies to the footprint fills. Source / image-kind / confidence-weight badges sit below the map. No georeferenced imagery is committed yet, so footprints render as outlines and a `image asset not available for:` block lists every overlay still missing a raster. See [docs/dash_map_overlays.md](dash_map_overlays.md). |
| Decision inspector                 | Selected event's summary, kind, timestamp.                                                                                                                              |
| Observations and evidence          | Observations / evidence artifacts / detections referenced by the current event. Each observation row shows source badge, data-mode badge, confidence weight, usability flags, and caveats. |
| Candidate tasking options          | Five-row table (SAT-A, SAT-B, Wait, Optical, Expand Search) with collect type, expected latency / cost, total score, and a `SELECTED` badge on SAT-B.                   |
| Score breakdown                    | Per-component breakdown of the selected option (`opt-sat-b`): ambiguity_resolution, custody_health_improvement, mission_relevance, timeliness, feasibility, cost_penalty, and the total. |
| Policy rationale                   | The deterministic-heuristic advisory text. Header is explicit: **"Heuristic advisory; RL-ready slot, not a trained RL decision."**                                       |
| Human action                       | Operator approval record for SAT-B with `decided_at`, reason, and operator id.                                                                                          |
| Outcome                            | Reacquisition result, custody score delta, summary.                                                                                                                     |
| Counterfactuals (rejected options) | Per unselected option: expected result, expected impact vs the pre-decision baseline (cs-snap-002), and a one-line summary. The column is labelled "impact vs baseline (Δ score)" with a caption clarifying that the values are deltas relative to the pre-decision custody state, not relative to the realised SAT-B outcome. |
| Follow-up recommendation           | Deterministic next-window guidance (`next_candidate_collect_type`, review thresholds).                                                                                  |

If the selected event does not reference a given artifact (for example, the initial step before any observation is received), the affected panels show a small "not available at this step" placeholder rather than empty / broken UI.

## What is fixture and what is simulated

Every record on screen is either `fixture` (a committed reference / cross-reference, including the Sentinel observations cross-loaded from `data/demo/whitsun_sentinel_observations.fixture.json`) or `simulated` (deterministic demo scaffolding for the planner / RL rationale, candidate tracks, custody snapshots, tasking options, scores, human action, outcome, counterfactuals, and follow-up recommendation). Both classes carry an explicit `data_mode` badge in the UI.

The trace JSON itself is the contract; the dashboard does not invent any record on the fly.

## Honest labelling for the policy rationale

The trace's `policy_rationale.policy_kind` is `"deterministic-heuristic"` and `policy_rationale.rl_advisory` explicitly disclaims trained RL. The Dash panel surfaces both, with two badges in the panel header:

- **HEURISTIC ADVISORY**
- **RL-READY SLOT (NO TRAINED RL)**

Tests assert this labelling so any future change that overstates the rationale will fail.

## What this view does not yet do

- **No real map.** The context panel is a placeholder list of AOI, tracks, and custody snapshot. A real map layer (PyDeck / Mapbox) is a follow-up; this slice does not refactor the existing map plumbing.
- **No live Sentinel query.** Sentinel observations rendered here come from the cross-referenced ingestion fixture, not from a CDSE STAC call.
- **No Tennent replay.** Tennent is a secondary AOI for the Sentinel ingestion layer only. A Tennent replay would require a separate Tennent decision trace; that is not in scope here.
- **No write actions.** The operator-approve record is a fixture; the panel does not let the user approve, reject, or mutate anything.
- **No planner / RL runtime change.** This view is purely a renderer over the trace fixture.

## Files added in this slice

```
src/custody/demo/
  __init__.py
  decision_trace.py                    # frozen DecisionTrace + helpers

src/app/layout/
  whitsun_replay.py                    # tab layout factory and component IDs

src/app/callbacks/
  whitsun_replay.py                    # selected_event_id -> all panels

src/app/dash_app.py                    # mounts a Tabs wrapper around
                                       # the existing main column

tests/
  test_demo_decision_trace_loader.py   # loader + smoke imports

docs/
  dash_whitsun_replay.md               # this document
```

## Tests

`tests/test_demo_decision_trace_loader.py` validates:

- Loader returns a valid `DecisionTrace` with the expected schema marker.
- Default path resolves to the committed fixture; explicit-path loading works.
- Event lookup by id; missing id returns `None`; first-event id is `ev-01`; ordered label pairs cover all 14 events.
- Options table contains exactly the five expected option labels, marks SAT-B selected, and carries a score breakdown for every option.
- Selected recommendation resolves to SAT-B with `total_score >= 0.85`.
- Built-in `validate()` returns no issues.
- Every event's referenced ids resolve into the corresponding section.
- Policy rationale is labelled `deterministic-heuristic` with an explicit no-trained-RL disclaimer.
- The Dash layout module imports cleanly and produces a non-empty layout tree.
- The full Dash app imports and registers the Whitsun replay callbacks.

No test contacts CDSE STAC, starts a Dash server, or writes to disk.
