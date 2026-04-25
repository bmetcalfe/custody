# Custody — Positioning

*This is the honest-scoping document. If any section of it embarrasses you to read, that section is wrong.*

Custody is a prototype **uncertainty-to-tasking / mission-planning decision engine for maritime GEOINT**. It models how imperfect SAR, AIS, scene-quality, and matcher evidence updates competing hypotheses over time, surfaces custody health and remaining ambiguity, ranks candidate collect *types* by expected uncertainty reduction, and runs that ranking through optimization, availability metadata, scheduling simulation, plan execution simulation, and a strategy comparison harness — closing a deterministic decision loop end to end.

The project reframed around this thesis in [ADR-0021](decisions/0021-custody-as-uncertainty-to-tasking-engine.md). This document is aligned with [README.md](../README.md) and the [pivot audit](pivot_audit_uncertainty_to_tasking.md); if it disagrees with README, README is authoritative.

---

## What this is

Shipped, runnable today, grouped:

**Evidence and belief**

- Scenario-specific hypothesis registries for Tennent Reef and Whitsun Reef
- Thin evidence adapters over existing observation, scene, matcher, VLM, and GFW-style outputs
- Deterministic hypothesis-state updates with auditable support / contradiction traces
- Synthetic scenario timelines using real Tennent / Whitsun scene dates
- Custody-health scoring distinguishing healthy / degraded / ambiguous / stale / lost states

**Decision packet and explainability**

- Composed decision packet (belief, custody health, primary ambiguity, candidate collects) with text / JSON / Markdown export formats and versioned JSON (`schema_version` "1")
- `do-not-yet` actions surfaced when custody is too uncertain to act on

**Mission value, counterfactuals, optimization, policy evaluation**

- Mission-value attribution proxy decomposing collect value into ambiguity reduction, custody-health improvement, mission relevance, timeliness, and cost / latency tradeoffs (deterministic, not a financial model)
- Counterfactual collect simulation comparing candidate strategies under deterministic heuristic outcome weights (not calibrated sensor probabilities, not live tasking)
- Constrained collection-plan optimization (exhaustive + greedy) under budget / max-collect / required / excluded constraints; planning utility is a deterministic proxy, not live tasking, not platform scheduling, not an RL policy
- Heuristic policy evaluation across six named deterministic strategies under a shared constraint; the harness is an RL-ready evaluation substrate, not a learned or trained policy

**Human-in-the-loop and planner workflow**

- Review ledger producing approve / reject / defer / override records with deterministic packet and review hashes; review-only — no live tasking or sensor commands are issued
- Planner-facing work queue ranking scenarios by custody health, ambiguity, mission-value proxy, planning utility, and human review status; decision-support system, not live tasking
- Workflow complexity map and efficiency proxy metrics comparing baseline / manual triage with the Custody-assisted flow (manual-step count, audit-artifact count, ambiguity focus, planner attention, traceability, candidate-strategy comparison count); prototype proxies only — no production timing measurement, no monetary value, no organisational performance claim

**Portfolio, API, and provenance**

- Cross-scenario portfolio allocation under shared budget, max-collects, and per-scenario constraints; deterministic exhaustive + greedy baselines; planning utility and mission value remain prototype proxies, not financial estimates; decision-support output only — no execution authorization or platform schedule is issued
- Local decision API service contract exposing decision packet / collect ranking / optimized plan / policy evaluation / planner queue / portfolio allocation as JSON-serializable responses; FastAPI is intentionally not a dependency — `create_app()` is a stub and the service layer runs entirely on the standard library; this is a local prototype API, not deployed, not authenticated, and not wired to any external planning system
- Auditable run provenance attached to every API response (deterministic run ID, command, scenarios, input fixtures, git commit, default assumptions / caveats), with a standalone provenance-manifest CLI and a documented prototype DevSecOps posture (CI test workflow, dependency hygiene, source-file vocabulary scans, AST-based import-boundary scans); manifests describe synthetic / fixture runs only — they are not a real-data lineage record

**Artifact and availability metadata bridges**

- Artifact evidence bridge that takes existing artifact-style outputs (scene metadata, SAR / VLM summaries, matcher persistence, AIS / GFW presence summaries, scene-quality flags, manual labels) as decision-layer inputs and runs the existing decision pipeline against them — explicit separation of detection from decision-making, two mapping paths (explicit semantic + scenario signal), small committed JSON manifests as fixtures; artifact evidence is candidate evidence, not ground truth, and the bridge does not run detection, fetch external data, or perform real-time ingestion
- Scene-availability metadata bridge that adjusts candidate collect recommendations using provider-neutral SAR / optical / AIS collection availability metadata without downloading imagery, issuing execution authorizations, or claiming platform access; deterministic feasibility rules over committed JSON catalogs; feasibility scores are metadata-derived decision support, not executable plans; candidate collect feasibility, not real-time ingestion

**Scheduling, execution simulation, and strategy comparison**

- Availability-adjusted collection-plan optimization that integrates scene-availability feasibility into the constrained optimizer so plans account for whether candidate collect types are feasible in the current metadata window; adjusted utility combines planning utility with metadata-derived feasibility; exhaustive + greedy baselines under budget / max-collect / min-feasibility / required / excluded constraints; decision support only — no executable plan, no platform scheduling, no live tasking
- Collection-window scheduler-lite that simulates placing availability-adjusted candidate collect types into provider-neutral collection windows under simple capacity, timing, and conflict constraints; deterministic exhaustive search over windows produces a schedule-feasibility simulation; schedule feasibility simulation only — not orbital scheduling, not platform tasking, not live tasking, not sensor-control instructions
- Plan execution simulation feedback loop that applies synthetic returned evidence from scheduled candidate collects through the belief-update engine and recomputes custody health, ambiguity, and the next recommendation under deterministic outcome policies (favorable / inconclusive / adverse / mixed); closed-loop decision-support simulation only — synthetic returned evidence only, post-collect belief update is deterministic and not a calibrated probability, not real collection, not live tasking, not platform access
- Deterministic strategy comparison harness that evaluates six planning strategies (manual baseline proxy, collection-value-only, mission-value optimized, availability-adjusted, scheduler-lite, execution feedback) on prototype proxy metrics (planning utility, mission-value proxy, ambiguity resolution, custody-health delta, schedule feasibility, traceability artifacts, review burden) and reports per-scenario and aggregate winners; future RL-ready evaluation substrate, not a trained agent; baseline proxy is not measured manual planner performance; no production KPI, no real revenue claim, no operational outcome attribution

---

## What this is not

- Not a standalone SAR ship detector. VLM / CFAR outputs are candidate evidence, not ground truth.
- Not wired to real SAR / AIS parquet data, real imagery, or any external API. The core demo runs on deterministic synthetic narratives; the artifact, availability, and scheduling layers run on small committed JSON fixtures.
- Not integrated with Sentinel-1, Sentinel-2, or any specific platform. Those are roadmap items.
- Not a live tasking system. Candidate collect recommendations are sensor-generic collect *types*, not tasking orders or platform-specific schedules.
- Not autonomous scheduling, real-time ingestion, or production-ready.
- Not a claim that AIS absence alone means dark vessel activity. The Whitsun generator encodes an explicit coverage guardrail.
- Not a replacement for existing mission-planning tools.
- Not a trained RL agent. The strategy-comparison harness is an evaluation substrate; no policy is learned or trained.

---

## The decision loop

The whole point of the pivot is that the following pipeline now runs end to end on committed fixtures:

```
evidence / artifacts -> hypothesis state -> custody health -> collection value
        -> mission value -> counterfactuals -> optimizer -> policy eval
        -> HITL review -> planner queue -> portfolio
        -> API + provenance -> availability metadata -> availability-adjusted optimizer
        -> scheduler-lite -> execution simulation -> strategy comparison
```

`scripts/13_decision_packet.py` runs the core decision loop; `scripts/28_compare_planning_strategies.py` runs the closed-loop pipeline through the strategy-comparison harness. See the [README](../README.md) for the full table of CLIs.

---

## The two scenarios

Custody uses two co-equal case studies, documented in [`docs/scenario.md`](scenario.md) and [ADR-0012](decisions/0012-scenario-reframe-tennent-whitsun.md) / [ADR-0013](decisions/0013-dual-case-studies.md):

- **Tennent Reef (Case Study A)** — Vietnamese land reclamation, five Umbra SAR scenes over 41 days in 2023. Structure / construction / clutter ambiguity.
- **Whitsun Reef (Case Study B)** — Chinese maritime militia flotilla, three Umbra scenes over four months in 2023–2024. Cluster / anchorage / AIS-dark / clutter ambiguity.

The pipeline is scenario-agnostic; scenario-specific interpretation lives in `src/custody/hypotheses/scenarios.py`. The demo does not identify specific flagged vessels, does not make legal or sovereignty claims, and does not take a position on any claimant's position.

---

## Current capabilities vs roadmap

**Current (shipped):** the grouped bullets in **What this is** above. All groups are runnable from CLIs `12`–`28` and are covered by the test suite.

**Roadmap (not yet):**

- Scaling portfolio allocation beyond two demo scenarios
- Real-data wiring from processed SAR / AIS artifacts, beyond fixture manifests
- Provider / catalog metadata integration beyond committed fixtures
- Sentinel-1 / Sentinel-2 evidence integration
- Platform-specific sensor access and scheduling
- Live tasking integration
- Real-time ingestion or production deployment
- Trained RL policy on top of the strategy-comparison harness, only if its evaluation results justify it

This split is load-bearing. A reader should be able to tell, at a glance, which bullets the repo can back up today and which are honestly on the backlog.

---

## Try it

See the [README's "Run this first" and "Explore modules" sections](../README.md#run-this-first) for the full command table covering scripts `12`–`28`. Every CLI is deterministic, runs against committed JSON fixtures, supports `--scenario {tennent,whitsun,both}`, and (where relevant) `--format {text,json,md}`.

---

## Why this shape

The project spent Week 1 and most of Week 2 on detection work — SAR / CFAR, then a VLM-assisted hybrid. That work exposed the actual product problem: raw detections are not the interesting output. The interesting output is a disciplined answer to *what should we believe, how sure are we, and what collect would reduce that uncertainty the most* — and, downstream, *can that collect actually fit in the next planning window, and what would the world look like after we ran it?*

So the pivot moved the product layer up. The hypothesis layer consumes existing detection, scene, matcher, and AIS / presence outputs as **candidate evidence**, maintains competing hypotheses with auditable traces, classifies custody health honestly (including the cases where scores are saturated but ambiguity is real), recommends sensor-generic collect types, runs them through optimization, availability metadata, scheduling simulation, and execution simulation, and benchmarks the whole stack against simpler baselines. Detection work is preserved as an input to this layer, not the product.

For the full rationale, see [ADR-0021](decisions/0021-custody-as-uncertainty-to-tasking-engine.md).
