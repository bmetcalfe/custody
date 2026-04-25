# Custody — Positioning

*This is the honest-scoping document. If any section of it embarrasses you to read, that section is wrong.*

Custody is a prototype **uncertainty-to-tasking engine for maritime GEOINT scenarios**. It models how imperfect SAR, AIS, scene-quality, and matcher evidence updates competing hypotheses over time, then surfaces custody health and remaining ambiguity so future collection can be prioritized by uncertainty reduction rather than raw detector confidence.

The project reframed around this thesis in [ADR-0021](decisions/0021-custody-as-uncertainty-to-tasking-engine.md). This document is aligned with [README.md](../README.md) and the [pivot audit](pivot_audit_uncertainty_to_tasking.md); if it disagrees with README, README is authoritative.

---

## What this is

Shipped, runnable today:

- Scenario-specific hypothesis registries for Tennent Reef and Whitsun Reef
- Thin evidence adapters over existing observation, scene, matcher, VLM, and GFW-style outputs
- Deterministic hypothesis-state updates with auditable support/contradiction traces
- Synthetic scenario timelines using real Tennent/Whitsun scene dates
- Custody-health scoring that distinguishes healthy, degraded, ambiguous, stale, and lost states
- Collection-value ranking that recommends candidate collect *types* by expected hypothesis-disambiguation value
- Decision packet CLI composing belief, custody health, primary ambiguity, and candidate collects
- Mission-value attribution proxy decomposing candidate collect value into ambiguity reduction, custody-health improvement, mission relevance, timeliness, and cost / latency tradeoffs; deterministic and not a financial model
- Counterfactual collect simulation comparing candidate collect strategies by expected ambiguity resolution and custody-health-score impact, using deterministic heuristic outcome weights (not calibrated sensor probabilities, not live tasking)
- Constrained collection-plan optimization over candidate collect types using deterministic exhaustive and greedy baselines under budget / max-collect / required / excluded constraints; planning utility is a deterministic proxy, not live tasking, not platform scheduling, not an RL policy
- Heuristic collection-policy evaluation comparing six named deterministic strategies (value-optimized, ambiguity-first, low-cost-first, SAR-first, optical-first, AIS-context-first) under one shared constraint; the harness is an RL-ready evaluation substrate, not a learned or trained policy
- Human-in-the-loop review ledger producing auditable approve / reject / defer / override records with deterministic review and packet hashes; semi-autonomous planning control layer, review-only — no live tasking or sensor commands are issued
- Planner-facing work queue ranking scenarios by custody health, ambiguity, mission-value proxy, planning utility, and human review status; decision-support system, not live tasking and not sensor commands
- Workflow complexity map and efficiency proxy metrics comparing a baseline / manual triage workflow against the Custody-assisted decision-support flow (manual-step count, audit-artifact count, ambiguity focus, planner attention, traceability, candidate-strategy comparison count); prototype proxies only — no production timing measurement, no monetary value, no organisational performance claim
- Cross-scenario portfolio allocation that selects candidate collect types across scenario work items under shared budget, max-collects, and per-scenario constraints; deterministic exhaustive + greedy baselines; planning utility and mission value remain prototype proxies, not financial estimates; decision-support output only — no execution authorization or platform schedule is issued
- Local decision API service contract exposing decision packet / collect ranking / optimized plan / policy evaluation / planner queue / portfolio allocation as JSON-serializable responses suitable for downstream planning-tool integration; FastAPI is intentionally not a dependency — `create_app()` is a stub and the service layer runs entirely on the standard library; this is a local prototype API, not deployed, not authenticated, and not wired to any external planning system
- Auditable run provenance attached to every API response (deterministic run ID, command, scenarios, input fixtures, git commit, default assumptions / caveats), with a standalone provenance-manifest CLI and a documented prototype DevSecOps posture (CI test workflow, dependency hygiene, source-file vocabulary scans, AST-based import-boundary scans); manifests describe synthetic-fixture runs only — they are not a real-data lineage record

---

## What this is not

- Not a standalone SAR ship detector. VLM/CFAR outputs are candidate evidence, not ground truth.
- Not wired to real SAR/AIS parquet data yet. The timeline and decision-packet CLIs run on synthetic narratives constructed from caller-supplied flags on real scenario dates.
- Not integrated with Sentinel-1, Sentinel-2, or any specific platform. Those are roadmap items.
- Not a live tasking system. Candidate collect recommendations are sensor-generic collect *types*, not tasking orders or platform-specific schedules.
- Not autonomous scheduling, real-time ingestion, or production-ready.
- Not a claim that AIS absence alone means dark vessel activity. The Whitsun generator encodes an explicit coverage guardrail.
- Not a replacement for existing mission-planning tools.

---

## The decision loop

The whole point of the pivot is that the following five steps now run end to end in one command:

```
evidence  ->  hypothesis state  ->  custody health  ->  primary ambiguity  ->  candidate collect ranking
```

`scripts/13_decision_packet.py` runs this loop and prints a readable packet. `scripts/12_hypothesis_timeline.py` shows the scene-by-scene trace that leads to the final state.

---

## The two scenarios

Custody uses two co-equal case studies, documented in [`docs/scenario.md`](scenario.md) and [ADR-0012](decisions/0012-scenario-reframe-tennent-whitsun.md) / [ADR-0013](decisions/0013-dual-case-studies.md):

- **Tennent Reef (Case Study A)** — Vietnamese land reclamation, five Umbra SAR scenes over 41 days in 2023. Structure / construction / clutter ambiguity.
- **Whitsun Reef (Case Study B)** — Chinese maritime militia flotilla, three Umbra scenes over four months in 2023–2024. Cluster / anchorage / AIS-dark / clutter ambiguity.

The pipeline is scenario-agnostic; scenario-specific interpretation lives in `src/custody/hypotheses/scenarios.py`. The demo does not identify specific flagged vessels, does not make legal or sovereignty claims, and does not take a position on any claimant's position.

---

## Current capabilities vs roadmap

**Current (shipped):** the bullets in **What this is** above. All seven run from the CLIs listed under **Try it** and are covered by the test suite.

**Roadmap (not yet):**

- Real-data wiring from processed SAR/AIS artifacts into the scenario generators
- Sentinel-1 / Sentinel-2 evidence integration
- Portfolio-level prioritization across multiple regions or targets
- Live tasking integration
- Platform-specific sensor access and scheduling
- Real-time ingestion or production deployment

This split is load-bearing. A reader should be able to tell, at a glance, which bullets the repo can back up today and which are honestly on the backlog.

---

## Try it

```bash
python scripts/13_decision_packet.py --scenario tennent
python scripts/13_decision_packet.py --scenario whitsun
python scripts/13_decision_packet.py --scenario both
```

Both CLIs are deterministic and produce human-readable output. See README for the timeline-only variant.

---

## Why this shape

The project spent Week 1 and most of Week 2 on detection work — SAR/CFAR, then a VLM-assisted hybrid. That work exposed the actual product problem: raw detections are not the interesting output. The interesting output is a disciplined answer to *what should we believe, how sure are we, and what collect would reduce that uncertainty the most*.

So the pivot moved the product layer up: the hypothesis layer consumes existing detection, scene, matcher, and AIS/presence outputs as **candidate evidence**, maintains competing hypotheses with auditable traces, classifies custody health honestly (including the cases where scores are saturated but ambiguity is real), and recommends sensor-generic collect types that would most reduce the remaining ambiguity. Detection work is preserved as an input to this layer, not the product.

For the full rationale, see [ADR-0021](decisions/0021-custody-as-uncertainty-to-tasking-engine.md).
