# Custody

**Custody is a prototype uncertainty-to-tasking / mission-planning decision engine for maritime GEOINT.**

Custody models how imperfect SAR, AIS, scene-quality, and matcher evidence updates competing hypotheses over time, surfaces custody health and remaining ambiguity, ranks candidate collect *types* by expected uncertainty reduction, and runs that ranking through optimization, availability metadata, scheduling simulation, plan execution simulation, and a strategy comparison harness — closing a deterministic decision loop end to end.

Two case studies drive the demo: Vietnamese land reclamation at [Tennent Reef](https://amti.csis.org/vietnam-ramps-up-spratly-island-dredging/) and [Chinese maritime militia activity](https://amti.csis.org/caught-on-camera-two-dozen-militia-boats-at-whitsun-reef-identified/) at Whitsun Reef. The architectural framing draws on the Space Development Agency's [Custody Layer capability vectors](https://www.sda.mil/custody/) — applied to the maritime domain, where open data enables public validation.

See [ADR-0021](docs/decisions/0021-custody-as-uncertainty-to-tasking-engine.md) for the product thesis and the pivot rationale, and [`docs/positioning.md`](docs/positioning.md) for the honest-scoping document.

---

## Run this first

```bash
uv sync
python scripts/13_decision_packet.py --scenario both
python scripts/28_compare_planning_strategies.py --scenario both --outcome-policy favorable
```

The first command renders the composed decision packet (belief, custody health, primary ambiguity, candidate collect ranking) for both scenarios. The second command runs the full closed-loop pipeline through the deterministic strategy comparison harness.

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/).

---

## Explore modules

Every CLI is deterministic, runs against committed JSON fixtures, supports `--scenario {tennent,whitsun,both}`, and (where relevant) `--format {text,json,md}`.

| Capability                                  | Script                                          | What it demonstrates                                                                  |
| ------------------------------------------- | ----------------------------------------------- | ------------------------------------------------------------------------------------- |
| Per-scene hypothesis timeline               | `scripts/12_hypothesis_timeline.py`             | Scene-by-scene belief, custody health, and reasoning trace                            |
| Decision packet                             | `scripts/13_decision_packet.py`                 | Composed belief / health / ambiguity / candidate collect packet (text / JSON / MD)    |
| Counterfactual collect simulation           | `scripts/14_counterfactual_collects.py`         | Per-candidate ambiguity-resolution and health-delta proxies                           |
| Constrained plan optimization               | `scripts/15_optimize_collect_plan.py`           | Exhaustive + greedy collection-plan baselines under budget / max-collect              |
| Heuristic policy evaluation                 | `scripts/16_evaluate_collect_policies.py`       | Six named deterministic strategies compared under shared constraints                  |
| Human-in-the-loop review ledger             | `scripts/17_review_decision_packet.py`          | Append-only JSONL approval / rejection / deferral / override records                  |
| Planner work queue                          | `scripts/18_planner_queue.py`                   | Cross-scenario priority queue with custody-health and review status                   |
| Workflow efficiency proxy metrics           | `scripts/19_efficiency_metrics.py`              | Baseline-vs-Custody manual-step / artifact / planner-attention proxies                |
| Portfolio allocation                        | `scripts/20_portfolio_allocation.py`            | Cross-scenario optimization under shared budget and per-scenario constraints          |
| Local decision API demo                     | `scripts/21_api_demo.py`                        | JSON-serializable service responses (no HTTP server, no auth)                         |
| Provenance manifest                         | `scripts/22_provenance_manifest.py`             | Deterministic run record (run-id, command, scenarios, git commit, caveats)            |
| Decision packet from artifact manifests     | `scripts/23_packet_from_artifacts.py`           | Artifact bridge: existing outputs → `HypothesisEvidence` → decision packet            |
| Scene-availability metadata bridge          | `scripts/24_scene_availability.py`              | Provider-neutral SAR / optical / AIS metadata → feasibility-adjusted recommendation   |
| Availability-adjusted optimized plan        | `scripts/25_availability_optimized_plan.py`     | Optimizer composed with feasibility multiplier                                        |
| Collection-window scheduler-lite            | `scripts/26_schedule_collect_windows.py`        | Schedule-feasibility simulation under capacity / timing / overlap constraints         |
| Plan execution simulation feedback loop     | `scripts/27_simulate_plan_execution.py`         | Synthetic returned evidence → updated belief → next recommendation                    |
| Baseline strategy comparison harness        | `scripts/28_compare_planning_strategies.py`     | Six-strategy deterministic ranking on prototype proxy metrics                         |

The decision packet is the "run this first" path; the timeline shows the scene-by-scene reasoning that leads to the final packet; the strategy comparison harness ranks the full stack against simpler baselines.

---

## Current capabilities

Grouped, not exhaustive — see [`docs/positioning.md`](docs/positioning.md) for full per-capability disclaimers.

**Evidence and belief**
- Scenario-specific hypothesis registries for Tennent and Whitsun
- Thin evidence adapters over existing observation, scene, matcher, VLM, and GFW-style outputs
- Deterministic weighted belief update with auditable support / contradiction traces
- Custody-health scoring distinguishing healthy / degraded / ambiguous / stale / lost states

**Decision packet and explainability**
- Decision packet composing belief, custody health, primary ambiguity, candidate collects (versioned JSON, `schema_version` "1")
- Human-readable text, structured JSON, and Markdown export formats
- `do-not-yet` actions surfaced when custody is too uncertain to act on

**Mission value, counterfactuals, optimization**
- Mission-value attribution proxy decomposing collect value into named components
- Counterfactual collect simulation comparing candidate strategies under heuristic outcome weights
- Constrained collection-plan optimization (exhaustive + greedy) under budget / max-collect / required / excluded
- Heuristic policy evaluation across six named deterministic strategies under shared constraints

**Human-in-the-loop and planner workflow**
- Review ledger producing approve / reject / defer / override records with deterministic packet + review hashes
- Planner work queue ranking scenarios by custody health, ambiguity, mission-value proxy, planning utility, and review status
- Workflow complexity map and efficiency proxy metrics comparing baseline / manual triage with the Custody-assisted flow

**Portfolio, API, and provenance**
- Cross-scenario portfolio allocation under shared budget, max-collects, and per-scenario constraints
- Local decision API service contract (JSON-serializable; no HTTP server, no auth) with seven endpoints
- Run-provenance records attached to every API response and emitted as standalone manifests

**Artifact and availability metadata bridges**
- Artifact-manifest bridge converting existing SAR / VLM / AIS / matcher-style outputs into `HypothesisEvidence`
- Scene-availability metadata bridge adjusting candidate recommendations using provider-neutral SAR / optical / AIS metadata
- Availability-adjusted optimization composing the constrained optimizer with metadata feasibility

**Scheduling, execution simulation, and strategy comparison**
- Collection-window scheduler-lite placing candidates into provider-neutral windows under simple constraints
- Plan execution simulation feedback loop producing synthetic returned evidence and a post-collect recommendation
- Baseline planning-strategy comparison harness ranking six strategies on deterministic prototype proxy metrics

---

## What this is not

- Not a standalone SAR ship detector
- Not a claim that AIS absence always means dark activity
- Not a production tasking system or live MPS integration
- Not a replacement for existing mission-planning tools
- VLM/SAR detection is treated as candidate evidence, not ground truth
- Candidate collect recommendations are sensor-generic collect *types*, not tasking orders or platform-specific schedules
- Mission-value, planning-utility, schedule-score, and comparison-score numbers are deterministic proxies, not measured KPIs and not financial estimates
- Plan execution simulation uses synthetic returned evidence under a deterministic outcome policy, not real sensor returns

See [`docs/positioning.md`](docs/positioning.md) for the full honest-scoping document.

---

## Roadmap

These items are explicitly future work, not part of the current demo:

- Scaling portfolio allocation beyond two demo scenarios
- Real-data wiring from processed SAR / AIS artifacts, beyond fixture manifests
- Provider / catalog metadata integration beyond committed fixtures
- Sentinel-1 / Sentinel-2 evidence integration
- Platform-specific sensor access and scheduling
- Live tasking integration
- Real-time ingestion or production deployment
- Trained RL policy on top of the strategy-comparison harness, only if its evaluation results justify it

---

## Demo scenarios

**Area of interest:** Spratly Islands, bbox 114.5°E–117.5°E × 8.5°N–11.0°N.

**Case Study A — Tennent Reef, Vietnamese reclamation.** Five Umbra SAR scenes across 41 days (July–August 2023) imaging active dredging at 8.856°N / 114.665°E. CSIS AMTI documents 62 acres of new artificial land added between end-of-2022 and late-2023. AIS-dark in GFW presence data — a canonical case for reasoning about persistent structure that lacks a cooperative-broadcast explanation.

**Case Study B — Whitsun Reef, vessel flotilla.** Three Umbra SAR scenes across four months (December 2023 – March 2024) at 9.98°N / 114.63°E. The site of the [March 2021 Chinese maritime militia swarm](https://amti.csis.org/caught-on-camera-two-dozen-militia-boats-at-whitsun-reef-identified/) of approximately 220 vessels. Our scenes show multiple AIS-dark vessel clusters, reasoned about as competing hypotheses (cluster activity vs transient anchorage vs detector clutter).

The pipeline is scenario-agnostic; scenario-specific interpretation lives in `src/custody/hypotheses/scenarios.py`. The demo does not identify specific flagged vessels and does not make legal or sovereignty claims.

See [`docs/scenario.md`](docs/scenario.md) for the locked AOI, full Umbra scene inventories, and case-by-case narrative.

---

## Architecture

The hypothesis layer sits on top of preserved lower-level components.

**Hypothesis and planning-decision layer** (`src/custody/hypotheses/`):

- `types.py` — `HypothesisEvidence`, `Hypothesis`, `HypothesisState`
- `registry.py` — scenario-specific hypothesis sets for Tennent and Whitsun
- `evidence.py` — source-object → evidence adapters
- `update.py` — deterministic weighted belief update with explanation traces
- `scenarios.py` — scenario-specific signal → evidence catalog
- `custody_health.py` — health classification and canonical ambiguity pairs
- `collection_value.py` — sensor-generic candidate collect ranker
- `mission_value.py` — mission-value attribution proxy
- `counterfactual.py` — deterministic counterfactual collect simulation
- `optimizer.py` — constrained collection-plan optimization
- `policy_eval.py` — heuristic policy comparison
- `planner_review.py` — human-in-the-loop review ledger
- `planner_queue.py` — planner work queue
- `efficiency_metrics.py` — workflow efficiency proxy metrics
- `portfolio.py` — cross-scenario portfolio allocation
- `decision_packet.py` — composed decision packet (text / JSON / MD)
- `explain.py` — human-readable state rendering
- `artifacts.py` — artifact-manifest evidence bridge (Slice 19)
- `scene_availability.py` — provider-neutral availability metadata bridge (Slice 20)
- `availability_optimizer.py` — availability-adjusted constrained optimization (Slice 21)
- `scheduler.py` — collection-window scheduler-lite (Slice 22)
- `execution_sim.py` — plan execution simulation feedback loop (Slice 23)
- `baseline_comparison.py` — strategy comparison harness (Slice 24)

Provenance lives at `src/custody/provenance.py` (Slice 18) — frozen `ProvenanceRecord`, deterministic `stable_run_id`, best-effort git-commit capture, and text / JSON / Markdown manifest formatters. The local decision API contract lives at `src/custody/api/` (Slice 17): schemas, service functions, and a deliberate FastAPI stub.

**CLIs** (`scripts/`):

- `12_hypothesis_timeline.py` through `28_compare_planning_strategies.py` — see the Explore modules table above

Earlier numbered scripts (`00`–`11`) cover detector / ingestion / SAR / VLM workflows that are preserved as inputs but are not part of the active decision-layer demo.

**Supporting components** — preserved as inputs, not the product:

- `fusion/` — polymorphic `Observation` types, AEQD tangent-plane geometry, H3 + DuckDB spatial index, Hungarian + EKF tracker
- `detection/` — SAR and VLM candidate detection (candidate evidence, not ground truth)
- `ingest/` — GFW presence ingestion as position-only AIS evidence
- `orchestration/`, `taskrecommendation.py` — portfolio attention and tasking machinery, to be re-coupled to custody-health outputs

See [`docs/custody_fusion_implementation_guide_v3.md`](docs/custody_fusion_implementation_guide_v3.md) for the broader architecture, and [`docs/decisions/`](docs/decisions/) for numbered ADRs — ADR-0021 is the pivot.

### Alignment with SDA Custody Layer capability vectors

| SDA Custody Layer capability vector | Custody implementation |
|---|---|
| Automated processing and fusion of data from traditional space-based sensing payloads | Multi-modal evidence adaptation over SAR, optical, and AIS/presence through a unified evidence contract |
| Multi-phenomenology fusion architecture supporting agile incorporation of new algorithms | Pluggable evidence adapters, per-source decoupling, deterministic update layer |
| Memory management and target hypothesis distribution across nodes | Frozen, serializable `HypothesisState` with auditable trace; node-to-node distribution flagged as a stretch goal |
| Reduction in latency of processing, exploitation, and dissemination | Offline pipeline demonstrates architectural patterns; production latency is explicitly out of scope |

---

## Honest limitations

This is a solo applied-research project. Limitations are documented in more detail in [`docs/positioning.md`](docs/positioning.md) and the relevant ADRs.

- The belief update is weighted-additive with deterministic tie-break, not Bayesian. Scores are bounded heuristic values. Hypothesis priors, evidence weights, and thresholds are documented in `scenarios.py` and ADR-0021.
- Custody-health classification, collection-value ranking, mission-value attribution, counterfactual outcome weights, scheduler-lite scoring, and the strategy-comparison composite score are all deterministic strategy-table lookups, not learned models. The tradeoff is legibility over adaptivity.
- Detection (VLM, CFAR) is treated strictly as candidate evidence generation. It is not the product.
- AIS absence is only treated as informative under documented coverage conditions; the Whitsun generator encodes a guardrail against "no AIS = dark vessel" overclaim.
- The core demo runs on deterministic synthetic narratives (`12_hypothesis_timeline.py`, `13_decision_packet.py`) and on committed artifact / availability / window fixtures (`23`–`28`). Both bridges are fixture / manifest based — no live external data fetch, real-time ingestion, imagery processing, or platform tasking is performed.
- Candidate collect recommendations name sensor-generic collect types (e.g. `repeat_sar`, `optical_context`, `ais_coverage_query`). Platform-specific scheduling and access are explicitly out of scope.
- Plan execution simulation uses synthetic returned evidence under a deterministic outcome policy, not real sensor returns.
- The demo does not identify specific flagged vessels and does not make legal or sovereignty claims.

---

## References

- [`docs/positioning.md`](docs/positioning.md) — public-facing "why this exists"
- [`docs/scenario.md`](docs/scenario.md) — AOI, windows, Umbra scene inventories
- [`docs/pivot_audit_uncertainty_to_tasking.md`](docs/pivot_audit_uncertainty_to_tasking.md) — slice-by-slice implementation status and the Week 0 audit history
- [`docs/security.md`](docs/security.md) and [`docs/devsecops.md`](docs/devsecops.md) — prototype security posture and local development workflow
- [`docs/decisions/`](docs/decisions/) — numbered architecture decision records; ADR-0021 locks the uncertainty-to-tasking pivot
- [SDA Custody Layer](https://www.sda.mil/custody/) — architectural reference
- [CSIS Asia Maritime Transparency Initiative](https://amti.csis.org/) — methodology grounding for both case studies
  - Tennent Reef: [Dec 2022](https://amti.csis.org/vietnams-major-spratly-expansion/) and [Nov 2023](https://amti.csis.org/vietnam-ramps-up-spratly-island-dredging/) reports
  - Whitsun Reef: [March 2021 swarm coverage](https://amti.csis.org/caught-on-camera-two-dozen-militia-boats-at-whitsun-reef-identified/)

---

## License

Code: MIT. Data: respective source licenses — Umbra ODP is CC BY 4.0, Sentinel is Copernicus open, GFW is a research license, NOAA AIS is public domain. See `LICENSE` for full terms.

---

*Custody is an open-source applied-research prototype. If the architectural patterns are useful, fork the repo. If they aren't, the postmortem is itself useful.*
