# Custody Dashboard — Screen-Share Demo Script

*Companion to [docs/positioning.md](positioning.md), [docs/demo_scenario_whitsun.md](demo_scenario_whitsun.md), [docs/dash_whitsun_replay.md](dash_whitsun_replay.md), and [docs/sentinel_ingestion_demo_aois.md](sentinel_ingestion_demo_aois.md).  This is the spoken script for an end-to-end live demo of the current dashboard.*

---

## Pre-flight (off-camera, before the call)

```bash
uv sync
uv run python src/app/dash_app.py
```

Open the URL the dev server prints (usually `http://127.0.0.1:8050`). Three tabs across the top: **Custody overview** (default), **Whitsun replay (fixture)**, **Tennent monitoring (fixture)**.

Verify before you go live:

- The Whitsun tab opens and the timeline shows events 01 → 14.
- The Tennent tab loads three Sentinel observations.
- Browser zoom set to a level the screen-share viewer can read (typically 100–110% on a 1080p share).

If anything breaks, the fall-back is to drive the JSON fixtures from a terminal:

```bash
uv run python scripts/13_decision_packet.py --scenario whitsun
uv run python scripts/29_ingest_sentinel_whitsun.py --aoi tennent
```

Total target runtime: **8–10 minutes** of dashboard time.

---

## 1. Custody overview tab — 60 seconds

**Goal:** establish that there is a broader shell around the demo. Do not dwell here.

**Click path:** the dashboard opens on this tab. Don't click anything.

**Say (≈45 seconds):**

> "When the dashboard opens, you land on the Custody overview tab. This is the broader operational shell — a portfolio table of entities, KPI cards across the top, and an entity-detail panel below. It comes from the earlier fusion-and-tasking phase of the project and feeds on a different scenario file than the rest of today's demo.
>
> I'm not going to dwell here. The interesting work for this conversation is in the next two tabs, which run on small committed JSON fixtures and are deliberately kept honest about what's real and what's scaffolding."

**Click path:** click the **Whitsun replay (fixture)** tab. The left sidebar swaps to a small read-only Whitsun block.

**Do not oversell:**

- Don't claim the overview tab is hooked up to live data — it's running on a synthetic scenario file.
- Don't promise that any of the overview's numbers will move during the demo. They won't.

---

## 2. Whitsun replay tab — 5–7 minutes

**Goal:** walk through the 14-event custody / reacquisition replay, demonstrating that the dashboard renders the operational hierarchy honestly (Umbra > Sentinel > simulated), reveals artifacts progressively, and doesn't overstate the policy rationale.

### Frame the tab — 30 seconds

**Look at:** the left sidebar block, the header, the timeline, and the Step counter. Don't click yet.

**Say:**

> "The left sidebar tells you what this tab is: a read-only fixture, a progressive mission replay, no live tasking, no live inference. I want that to be visible the whole time.
>
> The replay walks through one custody / reacquisition scenario for Whitsun Reef in December 2023. There are 14 events in the timeline on the left. Every panel on the right reveals progressively as the events unfold — that's a deliberate choice so the operator never sees the recommendation before the evidence."

### Event 01 — Scenario initialized — 15 seconds

**Click:** **`01. Scenario initialized`** in the timeline. (It's already selected on tab open.)

**Look at:** the right column. Most panels show "not available at this step".

**Say:**

> "On step one, almost everything is empty. The scenario is initialized — there's an AOI, a time window, no observations yet, no candidate options, no recommendation. That's correct. If we showed the recommendation here it would defeat the point of the replay."

**Do not oversell:** don't say "the system is waiting for data". The system isn't doing anything — it's a fixture, and the panels are gated on event ordinal.

### Events 02–05 — Observations open up — 60 seconds

**Click:** `02. Umbra SAR tile received`.

**Look at:** the **Observations and evidence** panel.

**Say:**

> "Step two is the tasked Umbra SAR collect. Notice the badge: source UMBRA-SAR, confidence weight one-point-zero, fixture data mode. This is the high-confidence tasked evidence anchor for the scenario."

**Click:** `03. VLM analysis complete`.

**Say:**

> "Step three adds VLM-derived evidence artifacts and three vessel detections off the same Umbra tile. The evidence carries an explicit caveat — VLM output is candidate evidence, not ground truth — and that string is in the panel, verbatim."

**Click:** `04. Candidate tracks initialized`.

**Look at:** the **Context (map placeholder)** panel.

**Say:**

> "Step four initializes three candidate tracks straight off the VLM detections. Note the panel header is `Context (map placeholder)`, and the body says map visualization is planned for the next slice. We don't have a real map layer yet, and we're not pretending we do."

**Click:** `05. Sentinel-2 context observation available`. Then `06. Sentinel-1 context observation available`.

**Say:**

> "Steps five and six bring in the Sentinel layer.  This is where we prove **multi-source pipeline handling**, not detection equivalence with Umbra.
>
> Sentinel-2 gives us a real public optical preview — that's the colour image you can see on the map at step five.  Sentinel-1 demonstrates the public SAR observation pipeline: the metadata is here, the AOI footprint is rendered, but the live Process API call returned an empty placeholder for these acquisition windows, so the dashboard gracefully falls back to footprint-only.  We don't fake a Sentinel-1 image we don't have.
>
> Look at the confidence weights: Sentinel-1 is point-four-five, Sentinel-2 low-cloud is point-three-zero.  Neither is treated as equivalent to Umbra's one-point-zero.  The dashboard surfaces the source class in a badge for every record.  Sentinel here is a weak-signal cueing layer — useful for change candidates and tasking cues, not a substitute for tasked SAR.
>
> Notice the WEAK-SIGNAL CUE callout that just appeared at the top of the observations panel.  The UI says it explicitly: *Sentinel-N provides a possible change cue.  Recommend higher-resolution tasking if mission priority warrants.  Umbra remains the high-confidence confirmation layer.*  That callout is on screen so anyone watching can read the operational hierarchy directly, without me needing to disclaim it from the script."

**Multi-source pipeline framing (one-liner if asked):**

> The demo proves the system handles **three distinct imagery pipelines** according to their trust level: Umbra is the **confirmation layer** with tasked SAR; Sentinel-2 is **public optical context** when a valid preview exists; Sentinel-1 is the **public SAR observation pipeline with graceful fallback** — when Sentinel Hub returns an empty preview, we keep the overlay footprint-only and surface the reason rather than render a fake image.

**Do not oversell:**

- Don't claim every Sentinel layer has imagery on every run.  In this committed state two Sentinel-2 acquisitions are real previews; the rest stay footprint-only after quality gates rejected empty placeholders and within-run duplicates.
- Don't claim the system pulled Sentinel data live during the demo.  It didn't — `scripts/32_fetch_sentinel_previews.py` is a deterministic build-time refresh; the running dashboard only reads committed PNGs.
- Don't claim the confidence weights are calibrated.  They're demo heuristics.
- **Don't say "Sentinel confirms", "Sentinel proves", "definitive change", "Sentinel reacquired the target".**  Sentinel here is a weak-signal cueing layer, not confirmation evidence.  Umbra is the confirmation layer.

### Event 07 — Custody risk — 30 seconds

**Click:** `07. Custody risk increases`.

**Say:**

> "Step seven is the inflection point. Track trk-002 separates from the cluster and goes AIS-dark. The custody snapshot in the context panel goes from healthy at score point-seven-eight to ambiguous at point-four-two. That's the trigger for the planner step."

### Events 08–10 — Options, scores, policy — 90 seconds

**Click:** `08. Planner generates options`.

**Look at:** the **Candidate tasking options** panel.

**Say:**

> "Step eight populates the five candidate tasking options: SAT-A, SAT-B, Wait, Optical, Expand Search. Note the options table doesn't show scores yet — that's deliberate. At step eight the planner has named the choices, but hasn't scored them."

**Click:** `09. Options scored`.

**Say:**

> "Step nine fills in the score column. SAT-B comes out at point-nine-one. SAT-A at point-six-two. Wait at point-two-zero. Optical at point-four-zero. Expand-Search at point-five-zero. There's no SELECTED badge yet — scoring isn't selecting."

**Click:** `10. Policy recommends SAT-B`.

**Look at:** the **Score breakdown** panel and the **Policy rationale** panel.

**Say:**

> "Step ten is the recommendation. SAT-B gets the SELECTED badge in the options table, the score-breakdown panel decomposes its zero-point-nine-one into six named components — ambiguity resolution, custody-health improvement, mission relevance, timeliness, feasibility, cost penalty — and the policy rationale panel shows exactly two badges: `HEURISTIC ADVISORY` and `RL-READY SLOT (NO TRAINED RL)`. That second badge is doing real work. There is no trained reinforcement-learning agent in this prototype. The harness is RL-ready as an evaluation substrate, not an agent.
>
> The summary text below says the same thing in plain English. We've designed it so any future change that overstates the rationale would fail a test."

**Do not oversell:**

- **Do not call the policy a "trained model" or "RL decision".** It is a deterministic heuristic.
- **Do not call the score components "calibrated".** They are demo heuristics.

### Events 11–13 — Review, approve, follow-up + outcome — 60 seconds

**Click:** `11. Operator reviews`.

**Say:**

> "Step eleven is the operator review. The recommendation, the score breakdown, and the policy/RL rationale are all on screen for a deliberate beat — the operator reads them before deciding."

**Click:** `12. Human approves`.

**Say:**

> "Step twelve is the human action. Operator approve, with a one-line reason. The badge says `SIMULATED` because there's no real operator in the loop here — this is a demo record."

**Click:** `13. Follow-up collection / outcome recorded`.

**Say:**

> "Step thirteen rolls the follow-up collect and the outcome together — the simulated Umbra repeat executes over the same AOI and the result lands. The observations panel grows by one row (note the `SIMULATED` data-mode badge — this is the demo's hypothetical execution, not a real-shape committed record), and the outcome panel shows trk-002 reacquired with custody score recovering from point-four-two to point-eight-one — a delta of plus zero-point-three-nine."

### Event 14 — Counterfactuals and follow-up — 45 seconds

**Click:** `14. Counterfactual and follow-up recommendation shown`.

**Look at:** the **Counterfactuals** panel and the **Follow-up recommendation** panel.

**Say:**

> "The last step shows what would have happened with the other four options. The column is labelled `impact vs baseline (Δ score)` — that's deltas relative to the pre-decision custody state, the score zero-point-four-two we saw earlier. Wait would have lost five hundredths. Optical context would have gained eight hundredths. Cross-geometry SAR fifteen hundredths. Expand-search twelve. SAT-B's actual realized delta was plus zero-point-three-nine, which is why SAT-B won.
>
> The follow-up recommendation panel proposes a seventy-two-hour Sentinel-only watch window with a re-task threshold. That's deterministic guidance, not a tasking order."

### Wrap the Whitsun tab — 15 seconds

**Say:**

> "Quick recap of the Whitsun tab: 14 events, every panel honest about its source class and whether it's fixture or simulated, no trained RL anywhere on screen, and a Sentinel context layer that never collapses into the Umbra confidence number. Now the second scenario."

**Click:** the **Tennent monitoring (fixture)** tab. The sidebar swaps to a Tennent block.

---

## 3. Tennent monitoring tab — 90 seconds

**Goal:** show that the Sentinel observation schema is reusable beyond Whitsun, and that the team has not faked a custody decision trace where one isn't honest.

### Frame the tab — 20 seconds

**Look at:** the left sidebar.

**Say:**

> "The Tennent tab is intentionally smaller. The sidebar tells you why: this is a fixed-site monitoring scenario — Sentinel context only, no live inference, no live change detection. There is no decision replay here, because we haven't built one."

### AOI, observations, interpretation — 45 seconds

**Look at:** the **AOI / context** panel.

**Say:**

> "Tennent Reef center, eight-point-eight-five-eight degrees north, one-fourteen-point-six-five-six east — the AMTI-published coordinates. The scenario type is `fixed_site_monitoring`, not custody-and-reacquisition. Data mode fixture, with an explicit `replace before operational use` caveat on the AOI."

**Look at:** the **Sentinel observation timeline** panel.

**Say:**

> "Three Sentinel observations in the table. One Sentinel-1 GRD, one low-cloud Sentinel-2 at twelve-percent cloud, one cloudy Sentinel-2 at eighty-five-percent cloud. Look at the `for detection` column on the cloudy row: `no`. The eighty-five-percent record is `usable_for_context: true` but `usable_for_detection: false`. Same observation schema, same confidence-weight buckets — the cloudy row gets point-one-zero, the low-cloud row gets point-three-zero. Same Sentinel framing as Whitsun: these are weak-signal cueing layers — useful for change candidates and tasking cues, not standalone proof. Umbra, when tasked, remains the high-confidence confirmation layer. The Site-monitoring interpretation panel below carries that callout verbatim."

### Not yet implemented — 25 seconds

**Look at:** the **Not yet implemented** panel.

**Say:**

> "And here is the part I want to be loud about. The Tennent tab does not yet have: live imagery rendering, a site-change classifier, construction-change detection, a policy or tasking decision loop, an operator approval workflow. That's five explicit `no`s, on screen, on every demo. The point of this tab is to prove the observation layer generalizes beyond a single cherry-picked maritime case. It is not the point of this tab to claim a fixed-site monitoring product exists."

**Do not oversell:**

- **Do not say** "we just need to wire up the model" or "the change-detection part is straightforward". It isn't, and we haven't.
- **Do not say** "Tennent is on the roadmap for the next sprint" unless you have a concrete plan.
- **Do not gesture vaguely at imagery.** There is no imagery on screen.

---

## Likely questions and honest answers

> **"Is any of this connected to live data?"**

> No. The Whitsun tab is a deterministic JSON fixture. The Tennent tab reads two committed JSON fixtures (an AOI and a Sentinel observation cache). The only path that touches live CDSE STAC is the `scripts/29_ingest_sentinel_whitsun.py` CLI when you pass `--live`, and tests deliberately never use that path.

> **"Why is the policy rationale labelled 'heuristic' and 'RL-ready'?"**

> Because it is a deterministic heuristic, and the harness around it is RL-ready in the sense that it produces ranked options and proxy reward components — but no policy has been trained, fitted, or evaluated yet. The wording exists so the demo can never accidentally drift into "we trained an RL model".

> **"What's the difference between the two `data_mode` values?"**

> `fixture` means a committed reference record — the Sentinel observations cross-loaded from the ingestion fixture, the Tennent AOI, the Umbra SAR tile that anchors the scenario. `simulated` means deterministic demo scaffolding produced for the replay — candidate tracks, custody snapshots, options, scores, the operator-approve record, the follow-up Umbra collect, the outcome, the counterfactuals, the follow-up recommendation. Both classes carry an explicit badge.

> **"How would real Sentinel data flow in?"**

> The CLI `scripts/29_ingest_sentinel_whitsun.py --live` issues a single POST to the CDSE STAC v1 endpoint and normalizes the result into the same `ObservationArtifact` shape the dashboard already renders. The dashboard would need a "Refresh from CDSE" control wired to that path; we haven't built it. See [docs/sentinel_ingestion_whitsun.md](sentinel_ingestion_whitsun.md).

> **"What about Tennent — when does the change-detection pipeline arrive?"**

> No commitment. Today the Tennent tab proves the observation layer generalizes; the next-natural slice would be metadata-only change detection over a longer Sentinel time series, but that is unscoped and we will not promise it during the demo.

> **"What is the map placeholder going to become?"**

> Today the context panel surfaces the AOI center, AOI fixture path, revealed tracks, and the latest custody snapshot — text only. A real map layer would render the AOI polygon, candidate-track markers, and the Sentinel / Umbra footprint outlines. We have not picked between PyDeck, Mapbox, and a static GeoPandas-backed plot.

> **"Is the dashboard production-ready?"**

> No. It is a deterministic prototype renderer over committed JSON. There is no auth, no deploy story, no real-time refresh, no platform integration, and the existing Custody-overview tab runs on a separate synthetic scenario file. See [docs/security.md](security.md) and [docs/devsecops.md](devsecops.md).

> **"Can I use this for an operational decision today?"**

> No. Every screen explicitly carries fixture / simulated badges and caveats; the decision loop is a fixture replay, not a live system. The architecture and the disciplined source-class hierarchy are what's worth looking at — not the numbers.

---

## Current limitations to call out before being asked

If the audience is technical and detail-oriented, surface these proactively:

- **No real map.** The "Context (map placeholder)" panel is a list, not a map.
- **No live data.** Sentinel observations are fixtures cross-referenced from `data/demo/whitsun_sentinel_observations.fixture.json` and `data/demo/tennent_sentinel_observations.fixture.json`.
- **No trained RL.** The strategy comparison harness (Slice 24) is a deterministic evaluation substrate; no agent has been trained.
- **No site-change classifier.** Tennent is observation context only.
- **Confidence weights are demo heuristics.** Not calibrated reliability values.
- **The Custody overview tab uses a different scenario file than the Whitsun replay.** They do not share state.
- **The placeholder AOIs are placeholders.** Both `*.fixture.geojson` files are explicitly labelled "replace before operational use".

---

## Timing budget

| Segment                              | Target | Hard cap |
| ------------------------------------ | ------ | -------- |
| Pre-flight (off-camera)              | —      | —        |
| Custody overview                     | 60 s   | 90 s     |
| Whitsun replay total                 | 5–7 min | 8 min   |
| &nbsp;&nbsp;Frame the tab            | 30 s   |          |
| &nbsp;&nbsp;Event 01                 | 15 s   |          |
| &nbsp;&nbsp;Events 02–05             | 60 s   |          |
| &nbsp;&nbsp;Events 06–07             | 45 s   |          |
| &nbsp;&nbsp;Events 08–10             | 90 s   |          |
| &nbsp;&nbsp;Events 11–13             | 60 s   |          |
| &nbsp;&nbsp;Event 14                 | 45 s   |          |
| &nbsp;&nbsp;Wrap                     | 15 s   |          |
| Tennent monitoring                   | 90 s   | 2 min    |
| Q&A buffer                           | 2 min  | —        |

If you go over, the safest cuts are: shorten the Custody overview frame to 30 seconds, skip the events 02–05 narration (single-sentence summary), and skip the Tennent observation table walk (point at it without reading rows aloud).

---

## What to do if something breaks live

- **Tab swaps but a panel is empty:** click another event, then back. The fan-out callback re-renders.
- **Sidebar shows the wrong block:** hard-refresh the browser (Ctrl/Cmd + Shift + R).
- **App will not start:** the fall-back narrative is to walk a `scripts/13_decision_packet.py --scenario whitsun` text output. The same evidence / health / candidate-collect ranking is in that text.
- **Anyone asks for a live Sentinel query:** `scripts/29_ingest_sentinel_whitsun.py --live --aoi whitsun`. Be ready for it to fail behind a corporate proxy. Default to the offline path.
