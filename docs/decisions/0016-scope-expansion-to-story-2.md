---
id: 0016
title: Project scope expansion to Story 2 applied research framing
date: 2026-04-20
status: accepted
---

## Context

The original Custody project scoping (Week 1 positioning.md) framed the project as a 10-week portfolio-grade reference implementation of multi-sensor fusion and explainable tip-and-cue orchestration for maritime domain awareness. Target audience was broadly "GeoInt engineers" with the output being a polished demo + documentation suitable for career portfolio use.

Week 2 detection work surfaced findings that both change the project's scope and clarify what makes it interesting:

1. **SAR detection on open-data commercial SAR is harder than the literature suggests.** Four approaches failed before the working approach was identified. Each failure is diagnosable and documentable.

2. **Vision-Language Models handle SAR domain-gap problems that narrow detectors cannot.** The VLM detection approach (see ADR-0015) produces meaningful results where pre-trained YOLO fails completely. Integration of VLMs into traditional SAR analysis pipelines is a genuinely novel pattern as of 2026, with thin published literature.

3. **Coherent Change Detection on Umbra Open Data Program imagery has fundamental geometric limits** for many AOIs including both Custody case studies. This is itself a publishable finding.

4. **The polymorphic observation architecture + covariance-aware fusion + legible tip-and-cue orchestration** — our intended "hero capability" from positioning.md — remain differentiating. With VLM-integrated detection feeding the fusion layer, the integration pattern becomes the project's novel contribution rather than any single algorithmic component.

5. **The project is now private.** Originally scoped as public reference implementation, the decision to stay private (with demo walkthroughs and potential whitepaper as the audience-facing artifacts) was made based on recognition that (a) most audience interest is in the demo, not the code, and (b) the research findings benefit from protection until we've extracted their value.

## Decision

**Reframe Custody from "10-week portfolio reference implementation" to "20-24 week applied research project."**

Three substantive changes from the Week 1 framing:

### Framing shift: Story 2 (applied research)

Previous framing ("Story 1"): A well-built reference implementation demonstrating engineering excellence in maritime fusion. Audience values clean code and clear architecture.

New framing ("Story 2"): An applied research project producing (a) a working VLM-integrated SAR/AIS fusion system, (b) documented negative results on CCD viability for Umbra ODP data, (c) a comparative evaluation of SAR ship detection approaches, (d) a reference architecture for LLM-augmented remote sensing analysis. Audience values honest engineering narrative plus novel integration patterns.

### Novel contribution: VLM-integrated fusion architecture with legible reasoning traces

The project's claim to "novel contribution" is not any single new algorithm. It is the integration pattern:

- **VLMs as SAR candidate generators** — handling domain-gap problems that narrow detectors fail on
- **Traditional detectors (CFAR) as refinement** — producing tight localization within VLM-identified regions
- **Polymorphic observation infrastructure** — with VLM reasoning traces attached to detections
- **Covariance-aware fusion** — handling heteroscedastic uncertainty across modalities including LLM-origin detections
- **Legible tip-and-cue orchestration** — natural-language reasoning chains from sensor observation through fusion to action recommendation

This pattern appears in 2024-2025 research literature as direction but has no mature public reference implementation.

### Timeline: 20-24 weeks (was 10)

Original plan: Weeks 1-10.
Revised plan:

- **Weeks 1-2 (complete):** Scenario framing, ingestion infrastructure, fusion architecture, detection investigation, VLM approach validation
- **Weeks 3-4:** VLM detection integration, CFAR refinement pattern, per-case-study configuration, batch processing all 8 scenes
- **Weeks 5-6:** Sentinel-1/2 integration, cross-modal fusion, anomaly scoring layer
- **Weeks 7-10:** Tip-and-cue orchestration layer with legible reasoning traces
- **Weeks 11-14:** Frontend (map visualization + reasoning timeline + scenario playback)
- **Weeks 15-16:** Polish, demo scripting, voiceover prep
- **Weeks 17-18:** Whitepaper drafting, LinkedIn content, conference abstract if IGARSS submission timing works
- **Weeks 19-24:** Buffer for timeline slippage, scope adjustments, or extensions based on findings

Budget acknowledgment: 18 weeks is the happy path. 20-24 is the realistic target. Project is side-time on evenings and weekends.

### Audience and deliverables

- **Primary audience:** Technical reviewers (engineers and researchers in maritime domain awareness, remote sensing ML, SAR analysis, LLM applications). Narrow but high-engagement.
- **Primary deliverable:** Demo walkthrough video + technical whitepaper. Not a public code repository.
- **Secondary deliverables:** LinkedIn content series covering investigation findings. Potential IGARSS 2027 workshop abstract. Potential conference talk or invited presentation.
- **Tertiary deliverable:** The project code itself, available via private-link invite to specific interested parties during job-search or collaboration conversations.

## Consequences

- **Private repository throughout development.** Already implemented. No fork-friendliness obligations, no license purity requirements, pragmatic tool choices allowed.
- **VLM-based detection inherits API dependency.** Production system requires network access to Anthropic's API. Budget ~$300-500 in API costs across full 20-24 week development. Acceptable.
- **Writing effort is significant.** Whitepaper quality requires real documentation investment. ADRs accumulate throughout; investigation document (`docs/investigations/detection-approaches.md`) is maintained as source material for eventual whitepaper.
- **Research rigor expectation elevated.** Claims about what works and what doesn't need specific numbers and reproducible setups. This ADR and the detection-approaches investigation document are examples.
- **Pivot flexibility preserved.** If later work surfaces findings that change scope (either narrowing or expanding), additional ADRs document the reasoning. The framework supports honest evolution.
- **Success criteria revised.** "Working demo" is still the minimum bar. "Credible engineering work visible to technical reviewers" is the stronger target. "Genuinely new contribution to the field" is aspirational — we'll see at Week 14-16 whether the integration patterns we're developing rise to that level.
- **Honest framing maintained.** The project is not "novel scientific research." It is "applied engineering work at the 2025-2026 frontier of LLM-augmented remote sensing analysis, integrating existing algorithms in patterns that don't have mature reference implementations." That's real and defensible; "novel science" would be overselling.

## What this doesn't change

- Core architecture (polymorphic observations, fusion, tipcue) is unchanged.
- Case study framing (dual Tennent + Whitsun) is unchanged (ADR-0012, ADR-0013).
- Observation schema, covariance model, and downstream reasoning layers are unchanged.
- GFW/AIS ingestion is complete and unaffected.
- Sentinel-1/2 integration plan is unchanged.

## What happens next (Week 3 and forward)

**Week 3:** VLM integration module (`src/custody/detection/vlm_sar.py`), tiling/batching logic, CFAR refinement pattern, unit tests with mocked API. Run end-to-end on Tennent and Whitsun scene 1 as gate-check. If the gate passes, proceed to batch processing all 8 scenes.

**Week 4:** Per-case-study configuration refinement based on Week 3 batch results. If certain case studies need prompt-variant tuning, capture in config. Observation schema updated to include `detector_reasoning` field for VLM traces.

**Week 5-6:** Cross-modal fusion with AIS (GFW presence data already processed). First end-to-end scenarios producing fused observations with reasoning chains.

**Weeks 7+:** per the revised timeline above.

## Supersedes

Supplements (does not supersede) positioning.md Week 1 framing. positioning.md is updated to reflect this expanded scope in a separate commit. ADR-0013 (dual case study framing) is unchanged. ADR-0014 (OS-CFAR implementation) remains valid; OS-CFAR code persists as refinement option.
