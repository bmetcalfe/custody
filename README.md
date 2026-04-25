# Custody

**Prototype uncertainty-to-tasking engine for maritime GEOINT scenarios.**

Custody models how imperfect SAR, AIS, scene-quality, and matcher evidence updates competing hypotheses over time, then surfaces custody health and remaining ambiguity so future collection can be prioritized by uncertainty reduction rather than raw detector confidence.

Custody now demonstrates a full prototype decision loop:
**evidence → hypothesis state → custody health → primary ambiguity → candidate collect ranking.**

Two case studies drive the demo: Vietnamese land reclamation at [Tennent Reef](https://amti.csis.org/vietnam-ramps-up-spratly-island-dredging/) and [Chinese maritime militia activity](https://amti.csis.org/caught-on-camera-two-dozen-militia-boats-at-whitsun-reef-identified/) at Whitsun Reef. The architectural framing draws on the Space Development Agency's [Custody Layer capability vectors](https://www.sda.mil/custody/) — applied to the maritime domain, where open data enables public validation.

See [ADR-0021](docs/decisions/0021-custody-as-uncertainty-to-tasking-engine.md) for the project's product thesis and the pivot rationale.

---

## Why it matters

Modern GEOINT workflows operate with incomplete, noisy, and sometimes contradictory evidence. The useful question is not only "what did the detector see?" but "what should we believe, how uncertain are we, and what collect would reduce that uncertainty?" Custody prototypes that reasoning layer.

---

## Try it

The primary demo is the **decision packet CLI**, which composes belief, custody health, primary ambiguity, and candidate collect recommendations into a structured artifact. Three export formats:

```bash
# Text (default, human-readable)
python scripts/13_decision_packet.py --scenario tennent

# JSON (structured, consumable by downstream planners; schema_version "1")
python scripts/13_decision_packet.py --scenario tennent --format json

# Markdown (shareable / vault / interview prep)
python scripts/13_decision_packet.py --scenario tennent --format md

# Append the mission-value attribution proxy (opt-in, off by default)
python scripts/13_decision_packet.py --scenario tennent --mission-value

# Counterfactual collect simulation
python scripts/14_counterfactual_collects.py --scenario tennent

# Optimized collection plan
python scripts/15_optimize_collect_plan.py --scenario tennent --budget 1.0 --max-collects 2

# Policy evaluation across heuristic strategies
python scripts/16_evaluate_collect_policies.py --scenario both --budget 1.0 --max-collects 2

# Human-in-the-loop review (review record only; no live tasking or sensor commands)
python scripts/17_review_decision_packet.py --scenario tennent --action approve --reason "Best ambiguity reduction under budget"

# Planner work queue (decision support only)
python scripts/18_planner_queue.py --scenario both

# Workflow efficiency proxy metrics (prototype proxies only)
python scripts/19_efficiency_metrics.py --scenario both

# Cross-scenario portfolio allocation (decision support only)
python scripts/20_portfolio_allocation.py --scenario both --budget 1.5 --max-collects 3

# Local decision API demo (JSON-serializable service responses; no HTTP server)
python scripts/21_api_demo.py --endpoint decision-packet --scenario tennent --format json

# Provenance manifest for a deterministic run (text, JSON, or Markdown)
python scripts/22_provenance_manifest.py --output-kind decision-packet --scenario tennent --format json

# Decision packet from artifact manifests (no detection or live ingestion)
python scripts/23_packet_from_artifacts.py --scenario both
```

The artifact bridge consumes existing artifact-style outputs (scene metadata, SAR/VLM summaries, matcher outputs, AIS/GFW presence summaries, quality flags, manual labels) committed as small JSON manifests under `tests/fixtures/artifacts/`. It does not run VLM, run the matcher, fetch Sentinel data, or pull from real data pipelines. Artifact evidence is candidate evidence, not ground truth. See [`docs/artifact_bridge.md`](docs/artifact_bridge.md).

The decision API is a local prototype service contract. It does not implement deployment, authentication, or downstream integration — service responses are JSON-serializable for tool composition only.

Every API response carries a `provenance` block (run ID, command, scenarios, input fixtures, git commit, default assumptions, default caveats). The same record can be emitted as a standalone manifest via `scripts/22_provenance_manifest.py` in text / JSON / Markdown. See [`docs/security.md`](docs/security.md) and [`docs/devsecops.md`](docs/devsecops.md) for the full prototype security and dev-workflow posture.

The portfolio plan is decision support only. It does not issue execution authorizations or platform-specific schedules. Portfolio score and mission value are prototype proxies, not financial estimates.

The efficiency report compares a baseline / manual triage workflow against the Custody-assisted decision-support workflow. Metrics are prototype proxies only; they do not claim measured workflow timing, real planner adoption, or any monetary value. See [`docs/mps_complexity_map.md`](docs/mps_complexity_map.md) for the workflow model the metrics are built on.

`--scenario` accepts `tennent`, `whitsun`, or `both`; `--scenario both --format json` emits a single JSON array of two packets. The JSON schema is versioned (`schema_version: "1"`), closed for that version, and the output is fully deterministic (`generated_at` is pinned to the evidence timestamp, not wall-clock).

`--mission-value` appends a per-candidate attribution block decomposing each recommended collect into named components (ambiguity reduction, custody-health improvement, mission relevance, timeliness, cost / latency tradeoffs, false-positive risk). It is a **proxy model**, not revenue attribution and not a financial model: no dollar figures, no production cost accounting. Leaving the flag off keeps the default text / JSON / Markdown outputs byte-identical to their Slice 8 baselines.

The per-scene hypothesis timeline is also runnable directly:

```bash
python scripts/12_hypothesis_timeline.py --scenario tennent
python scripts/12_hypothesis_timeline.py --scenario whitsun
python scripts/12_hypothesis_timeline.py --scenario both
```

Both CLIs are deterministic and produce human-readable output. The decision packet is the "run this first" path; the timeline shows the scene-by-scene reasoning that leads to the final packet.

---

## Current capabilities

- Scenario-specific hypothesis registries for Tennent Reef and Whitsun Reef
- Thin evidence adapters over existing observation, scene, matcher, VLM, and GFW-style outputs
- Deterministic hypothesis-state updates with auditable support/contradiction traces
- Synthetic scenario timelines using real Tennent/Whitsun scene dates
- Custody-health scoring that distinguishes healthy, degraded, ambiguous, stale, and lost states
- Collection-value ranking that recommends candidate collect types by expected hypothesis-disambiguation value
- Decision packet CLI composing belief, custody health, primary ambiguity, and candidate collects, with text, JSON, and Markdown export formats (versioned JSON, `schema_version` "1")
- Mission-value attribution proxy decomposing candidate collect value into ambiguity reduction, custody-health improvement, mission relevance, timeliness, and cost / latency tradeoffs (deterministic, not a financial model)
- Counterfactual collect simulation comparing candidate collect strategies by expected ambiguity resolution and custody-health impact using deterministic heuristic outcome weights (not calibrated probabilities)
- Constrained collection-plan optimization selecting candidate collect types under budget and max-collect constraints using deterministic exhaustive and greedy baselines
- Heuristic collection-policy evaluation comparing value-optimized, ambiguity-first, low-cost-first, SAR-first, optical-first, and AIS-context-first strategies under shared constraints
- Human-in-the-loop review ledger for approving, rejecting, deferring, or overriding candidate collection recommendations with auditable review records (review-only — no live tasking or sensor commands)
- Planner work queue that ranks scenario decision packets by custody health, ambiguity, mission-value proxy, planning utility, and human review status
- Workflow complexity and efficiency proxy metrics comparing a baseline / manual triage workflow with the Custody-assisted decision-support flow (prototype proxies only — no production timing or operational performance claim)
- Portfolio-level allocation that selects candidate collect types across scenario work items under shared budget, max-collect, and per-scenario constraints (decision-support output only — no execution authorization, no platform scheduling)
- Local decision API/service layer exposing decision packets, collect ranking, optimized plans, policy evaluation, planner queues, and portfolio allocation as JSON-serializable responses (local prototype service contract — no deployment, no authentication, no live integration)
- Run-provenance records attached to every API response (deterministic run ID, command, scenarios, input fixtures, git commit, default assumptions / caveats); standalone provenance-manifest CLI; documented prototype security posture and CI workflow (auditable artifacts only — no real-data lineage, no production controls)
- Artifact-manifest bridge that converts existing SAR / VLM / AIS / matcher-style outputs into HypothesisEvidence and runs the decision packet stack without rerunning detection or live ingestion (small committed JSON manifests; explicit semantic and scenario signal mapping paths; artifacts are candidate evidence, not ground truth)

---

## What this is not

- Not a standalone SAR ship detector
- Not a claim that AIS absence always means dark activity
- Not a production tasking system
- Not a replacement for existing mission-planning tools
- VLM/SAR detection is treated as candidate evidence, not ground truth
- Candidate collect recommendations are sensor-generic collect *types*, not tasking orders or platform-specific schedules

See [`docs/positioning.md`](docs/positioning.md) for the full honest-scoping document.

---

## Roadmap

These items are explicitly future work, not part of the current demo:

- Real-data wiring from processed SAR/AIS artifacts into the scenario generators
- Sentinel-1 / Sentinel-2 evidence integration
- Portfolio-level prioritization across multiple regions or targets
- Live tasking integration
- Platform-specific sensor access and scheduling
- Real-time ingestion or production deployment

---

## Demo scenarios

**Area of interest:** Spratly Islands, bbox 114.5°E–117.5°E × 8.5°N–11.0°N.

**Case Study A — Tennent Reef, Vietnamese reclamation.** Five Umbra SAR scenes across 41 days (July–August 2023) imaging active dredging at 8.856°N / 114.665°E. CSIS AMTI documents 62 acres of new artificial land added between end-of-2022 and late-2023. AIS-dark in GFW presence data — a canonical case for reasoning about persistent structure that lacks a cooperative-broadcast explanation.

**Case Study B — Whitsun Reef, vessel flotilla.** Three Umbra SAR scenes across four months (December 2023 – March 2024) at 9.98°N / 114.63°E. The site of the [March 2021 Chinese maritime militia swarm](https://amti.csis.org/caught-on-camera-two-dozen-militia-boats-at-whitsun-reef-identified/) of approximately 220 vessels. Our scenes show multiple AIS-dark vessel clusters, reasoned about as competing hypotheses (cluster activity vs transient anchorage vs detector clutter).

The pipeline is scenario-agnostic; the scenario-specific interpretation lives in `src/custody/hypotheses/scenarios.py`. The demo does not identify specific flagged vessels and does not make legal or sovereignty claims.

See [`docs/scenario.md`](docs/scenario.md) for the locked AOI, full Umbra scene inventories, and case-by-case narrative.

---

## Architecture

The hypothesis layer sits on top of preserved lower-level components.

**Hypothesis layer** (`src/custody/hypotheses/`) — the product layer:
- `types.py` — `HypothesisEvidence`, `Hypothesis`, `HypothesisState` dataclasses
- `registry.py` — scenario-specific hypothesis sets for Tennent and Whitsun
- `evidence.py` — thin adapters converting existing source objects to evidence annotations
- `update.py` — deterministic weighted belief update with explanation traces
- `scenarios.py` — scenario-specific signal → evidence mappings (the catalog)
- `custody_health.py` — classifies state into healthy / degraded / ambiguous / stale / lost with canonical ambiguity pairs
- `collection_value.py` — ranks sensor-generic collect types by expected disambiguation value
- `mission_value.py` — mission-value attribution proxy decomposing candidate collect value into named components
- `counterfactual.py` — deterministic counterfactual simulation of candidate collect outcomes
- `optimizer.py` — constrained collection-plan optimization over candidate collect types
- `policy_eval.py` — heuristic policy evaluation comparing collection strategies under shared constraints
- `planner_review.py` — human-in-the-loop review ledger for operator approval, rejection, deferral, or override of recommendations
- `planner_queue.py` — ranked planner work queue for scenario-level decision support
- `efficiency_metrics.py` — workflow efficiency proxy metrics comparing baseline and Custody-assisted workflows
- `portfolio.py` — cross-scenario portfolio allocation under shared budget / max-collects / per-scenario constraints
- `explain.py` — human-readable state rendering

**CLIs** (`scripts/`):
- `12_hypothesis_timeline.py` — per-scene belief, custody health, and reasoning trace
- `13_decision_packet.py` — composed belief / health / ambiguity / candidate-collect packet
- `14_counterfactual_collects.py` — counterfactual outcome comparison across candidate collects
- `15_optimize_collect_plan.py` — constrained plan optimization with exhaustive and greedy baselines
- `16_evaluate_collect_policies.py` — heuristic policy evaluation and strategy comparison
- `17_review_decision_packet.py` — human-in-the-loop review of decision packets and optimized plans
- `18_planner_queue.py` — planner work queue ranking scenarios by urgency and review status
- `19_efficiency_metrics.py` — workflow efficiency proxy metrics comparing baseline / manual triage with the Custody-assisted flow
- `20_portfolio_allocation.py` — cross-scenario portfolio allocation under shared budget and max-collect constraints
- `21_api_demo.py` — local decision API demo over `custody.api.service` (no HTTP server)

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

## Quickstart

```bash
uv sync
uv run pytest                                    # full suite
uv run python scripts/13_decision_packet.py --scenario both
```

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/).

---

## Honest limitations

This is a solo applied-research project. Limitations are documented in more detail in [`docs/positioning.md`](docs/positioning.md) and the relevant ADRs.

- The belief update is weighted-additive with deterministic tie-break, not Bayesian. Scores are bounded heuristic values. Hypothesis priors, evidence weights, and thresholds are documented in `scenarios.py` and ADR-0021.
- Custody-health classification and collection-value ranking are deterministic strategy-table lookups, not learned models. The tradeoff is legibility over adaptivity.
- Detection (VLM, CFAR) is treated strictly as candidate evidence generation. It is not the product.
- AIS absence is only treated as informative under documented coverage conditions; the Whitsun generator encodes a guardrail against "no AIS = dark vessel" overclaim.
- The timeline and decision-packet CLIs run on synthetic scenario narratives constructed from caller-supplied flags. Real-data wiring from processed SAR/AIS artifacts is on the roadmap.
- Candidate collect recommendations name sensor-generic collect types (e.g. `repeat_sar`, `optical_context`, `ais_coverage_query`). Platform-specific scheduling and access are explicitly out of scope.
- The demo does not identify specific flagged vessels and does not make legal or sovereignty claims.

---

## References

- [`docs/positioning.md`](docs/positioning.md) — public-facing "why this exists"
- [`docs/scenario.md`](docs/scenario.md) — AOI, windows, Umbra scene inventories
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
