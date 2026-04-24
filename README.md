# Custody

**Prototype uncertainty-to-tasking engine for maritime GEOINT scenarios.**

Custody models how imperfect SAR, AIS, scene-quality, and matcher evidence updates competing hypotheses over time, then surfaces custody health and remaining ambiguity so future collection can be prioritized by uncertainty reduction rather than raw detector confidence.

Two case studies drive the demo: Vietnamese land reclamation at [Tennent Reef](https://amti.csis.org/vietnam-ramps-up-spratly-island-dredging/) and [Chinese maritime militia activity](https://amti.csis.org/caught-on-camera-two-dozen-militia-boats-at-whitsun-reef-identified/) at Whitsun Reef. The architectural framing draws on the Space Development Agency's [Custody Layer capability vectors](https://www.sda.mil/custody/) — applied to the maritime domain, where open data enables public validation.

See [ADR-0021](docs/decisions/0021-custody-as-uncertainty-to-tasking-engine.md) for the project's product thesis and the pivot rationale.

---

## Why it matters

Modern GEOINT workflows operate with incomplete, noisy, and sometimes contradictory evidence. The useful question is not only "what did the detector see?" but "what should we believe, how uncertain are we, and what collect would reduce that uncertainty?" Custody prototypes that reasoning layer.

---

## Current capabilities

- Scenario-specific hypothesis registries for Tennent Reef and Whitsun Reef
- Thin evidence adapters over existing observation, scene, matcher, VLM, and GFW-style outputs
- Deterministic hypothesis-state updates with auditable support/contradiction traces
- Synthetic scenario timelines using real Tennent/Whitsun dates
- Custody-health scoring that distinguishes healthy, degraded, ambiguous, stale, and lost states

---

## What this is not

- Not a standalone SAR ship detector
- Not a claim that AIS absence always means dark activity
- Not a production tasking system
- Not a replacement for existing mission-planning tools
- VLM/SAR detection is treated as candidate evidence, not ground truth

See [`docs/positioning.md`](docs/positioning.md) for the full honest-scoping document.

---

## Try the hypothesis timeline demo

```bash
python scripts/12_hypothesis_timeline.py --scenario tennent
python scripts/12_hypothesis_timeline.py --scenario whitsun
python scripts/12_hypothesis_timeline.py --scenario both
```

Each invocation prints a deterministic, human-readable trace of evidence updates, top-hypothesis scores, and per-scene reasoning across the real Umbra scene dates for the chosen case study.

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
- `explain.py` — human-readable state rendering

**Supporting components** — preserved as inputs, not the product:
- `fusion/` — polymorphic `Observation` types, AEQD tangent-plane geometry, H3 + DuckDB spatial index, Hungarian + EKF tracker
- `detection/` — SAR and VLM candidate detection (candidate evidence, not ground truth)
- `ingest/` — GFW presence ingestion as position-only AIS evidence
- `orchestration/`, `taskrecommendation.py` — portfolio attention and tasking machinery, to be re-coupled to custody-health outputs

See [`docs/custody_fusion_implementation_guide_v3.md`](docs/custody_fusion_implementation_guide_v3.md) for the broader architecture, and [`docs/decisions/`](docs/decisions/) for numbered ADRs — ADR-0021 is the pivot.

### Alignment with SDA Custody Layer capability vectors

| SDA Custody Layer capability vector | Custody implementation |
|---|---|
| Automated processing and fusion of data from traditional space-based sensing payloads | Multi-modal evidence adaptation over SAR (Umbra + Sentinel-1), EO, AIS/GFW through a unified evidence contract |
| Multi-phenomenology fusion architecture supporting agile incorporation of new algorithms | Pluggable evidence adapters, per-source decoupling, deterministic update layer |
| Memory management and target hypothesis distribution across nodes | Frozen, serializable `HypothesisState` with auditable trace; node-to-node distribution flagged as a stretch goal |
| Reduction in latency of processing, exploitation, and dissemination | Offline pipeline demonstrates architectural patterns; production latency is explicitly out of scope |

---

## Quickstart

```bash
uv sync
uv run pytest                                    # full suite
uv run python scripts/12_hypothesis_timeline.py --scenario both
```

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/).

---

## Honest limitations

This is a solo applied-research project. Limitations are documented in more detail in [`docs/positioning.md`](docs/positioning.md) and relevant ADRs.

- The belief update is weighted-additive with deterministic tie-break, not Bayesian. Scores are bounded heuristic values. Hypothesis priors, evidence weights, and thresholds are documented in `scenarios.py` and ADR-0021.
- Detection (VLM, CFAR) is treated strictly as candidate evidence generation. It is not the product.
- AIS absence is only treated as informative under documented coverage conditions; the Whitsun generator encodes a guardrail against "no AIS = dark vessel" overclaim.
- The timeline demo runs on synthetic scenario narratives constructed from caller-supplied flags. End-to-end runs over real parquet fixtures are future work.
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
