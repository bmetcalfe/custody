# Custody — Positioning

*This is the honest-scoping document. It exists to say publicly what this project is, what it is not, who it is for, and what it answers. If any section of it embarrasses you to read, that section is wrong.*

---

## What this is

An **open-source reference implementation of multi-phenomenology fusion and legible tip-and-cue orchestration for maritime domain awareness**, aligned with the Space Development Agency's publicly published [Custody Layer capability vectors](https://www.sda.mil/custody/).

Built end-to-end on open commercial and public data:

- Umbra SAR Open Data Program (CC BY 4.0) for high-resolution SAR
- Sentinel-1 GRD and Sentinel-2 L2A (Copernicus) for baseline coverage
- Global Fishing Watch API for AIS-derived vessel presence data

The hero capability is **covariance-aware, explainable tip-and-cue orchestration**: every cueing decision the system emits is accompanied by a reasoning trace — what was selected, what alternatives were considered, what the expected information gain was, why rejected candidates were rejected, and a plain-language justification.

Commercial MDA systems generally expose the decision. Custody exposes the reasoning.

## What this answers

The SDA Custody Layer capability call describes architectural needs — multi-phenomenology fusion, hypothesis management, low-latency exploitation, handoff between satellite nodes — that are not specific to any single threat domain. They are *architectural patterns*. Custody applies those patterns to a domain where open data enables public validation: maritime persistence monitoring at contested features.

A reader of this repository can:

1. Clone the codebase
2. Sign up for a standard GFW research token, AWS open data access, and Copernicus Sentinel access — all free
3. Reproduce the entire demo pipeline end-to-end against the same data

That reproducibility is itself a property. No privileged data access, no private SDK, no institutional credentials.

## The demo scenario

**Primary case: Vietnamese land reclamation at Tennent Reef** (Vietnamese: Đá Tiên Nữ), 8.856°N / 114.665°E, Spratly Islands. Five Umbra SAR scenes over 41 days in summer 2023 capture active dredging and island-building at the eastern artificial island (Tiên Nữ B). The CSIS Asia Maritime Transparency Initiative has documented this specific feature's expansion in their [December 2022](https://amti.csis.org/vietnams-major-spratly-expansion/) and [November 2023](https://amti.csis.org/vietnam-ramps-up-spratly-island-dredging/) reports, identifying Tennent as one of Vietnam's four most significantly developed Spratly outposts. AMTI's November 2023 report documents 62 acres (25 hectares) of new artificial land added at Tennent between end-of-2022 and late-2023 — a period that encompasses our demo window.

The pipeline detects this reclamation activity as a **persistent AIS-dark SAR return**: GFW presence data shows no vessels broadcasting AIS at the reef, while Umbra SAR shows clear coherent structure that grows scene-over-scene across the demo window. The architecture does not need to know in advance what is being built; it detects *persistent activity inconsistent with cooperative vessel broadcasts* and surfaces it as a track worthy of attention.

**Supplementary case: Chinese maritime militia activity at Whitsun Reef** (Vietnamese: Đá Ba Đầu; Filipino: Julian Felipe Reef), 9.98°N / 114.63°E. Three Umbra scenes over three months (December 2023 – March 2024) covering the site of the canonical [March 2021 Chinese maritime militia swarm event](https://amti.csis.org/chinese-ships-swarm-philippines-eez/). Presented as a case study demonstrating the same architecture applied to a different narrative — longer-baseline change detection at an unoccupied contested reef. Not included in the 90-second demo cut.

Both cases are documented honestly with AMTI references to the actual published analysis. Neither claims to reveal anything not already in AMTI's public reporting. The contribution is the **architecture**: the pipeline does this work starting from pixels and AIS pings, without being told what it's looking for.

## What this is not

**Not a hypersonic or missile tracking system.** The SDA Custody Layer capability call primarily addresses ballistic and hypersonic threat kinematics that do not translate to maritime vessels or land reclamation. Custody applies the *architectural patterns* from those capability vectors to a domain where open data supports public validation. The threat model here is persistent AIS-dark activity at contested features, not missile defense. This distinction is deliberate and is stated up front so nobody reads the SDA alignment as claiming something it doesn't.

**Not a production system.** The demo runs against pre-computed artifacts produced by a one-shot preprocessing pipeline. The architecture supports live operation; the demonstration does not. Production latency, hardening, scale, and reliability work are out of scope for a 10-week evenings-and-weekends reference implementation.

**Not a commercial product clone.** Vantor Sentry, BlackSky, Satellogic, and other commercial maritime domain awareness products have significant internal capabilities, proprietary constellations, commercial archives, and customer-tier features that are not publicly documented. Custody focuses on the *publicly visible gap*: making planning decisions and their uncertainty explicit and auditable. Whether similar functionality exists inside commercial products is not a claim made either way. If it exists, good — the architecture is still useful as an open reference.

**Not an intelligence product.** The demo does not identify specific named vessels, does not make legal claims about sovereignty or lawful presence, and does not name any actor as having committed any specific act. It detects activity consistent with publicly reported open-source analytic methodology — the same kind of analysis AMTI publishes monthly against the same public imagery sources. Sovereignty and legal questions are outside the scope of what a technical reference implementation can or should address.

**Not a political statement.** Six nations claim overlapping sovereignty over features in the South China Sea. This project takes no position on any of those claims. The demo features (Tennent, Whitsun) were selected because Umbra Open Data Program coverage existed there during a time window for which we also had Sentinel baseline and GFW AIS coverage. The selection is data-driven, not claim-driven. Both Chinese and Vietnamese activity are documented in the demo and in the supplementary case study; neither is presented as more or less legitimate than the other.

## What this ships in 10 weeks

- Week 1: architecture, fusion package, EKF, observation types, spatial index, tracker *(done)*
- Week 2: detection — CFAR on SAR, AIS passthrough, opportunistic EO *(in progress)*
- Week 3: end-to-end fusion run over real demo-window data
- Week 4: anomaly scoring — four multi-INT anomalies including the hero "persistent AIS-dark" case
- Week 5: the tipcue layer — covariance-aware candidate scoring with full reasoning trace
- Week 6-7: frontend — React + Mapbox + deck.gl + FastAPI, belief-state visualization, orchestration trace panel
- Week 8: integration, polish
- Week 9: voiceover, documentation
- Week 10: publish

## Honest limitations

- **Scenario selection is data-driven, not narrative-first.** The primary feature in the demo (Tennent Reef) was confirmed during Week 2 reconnaissance against the actual Umbra imagery rather than during initial Day 0 scoping. See ADR-0012 for the specifics of how the scenario was locked. The architectural pipeline is scenario-agnostic.
- **Umbra coverage is sparse.** Eight AOI scenes over 9 months at two features. Sentinel-1 provides continuous 10-meter fill-in but at coarser resolution.
- **Detection uses classical methods.** CA-CFAR for SAR, pretrained CNN for EO (if cloud-free scenes exist). No fine-tuning on the specific AOI. Modern detectors would improve false-alarm rates at the cost of reproducibility.
- **Motion model is constant-velocity.** Appropriate for commercial vessel transits; approximate for dredgers and construction barges operating in place; not intended for high-maneuver targets.
- **Association is Hungarian with Mahalanobis gating.** Multiple Hypothesis Tracking (MHT) is a documented future-work item, not implemented.
- **Feasibility priors in the tipcue layer are simplified.** Binary for SAR grazing angle constraints; probabilistic for cloud forecasts. Production systems would extend these significantly.
- **Sensor fingerprinting, re-identification from imagery, and long-archive pattern-of-life** are commercial capabilities this reference does not reproduce.

## Who this is for

**Primary audience: the GeoInt engineering community responding to SDA's Custody Layer capability calls**, and engineers at commercial space and defense companies building toward similar architectures. The ADR structure, the observation-type design, the covariance-aware cueing — these are working-engineer artifacts, not marketing material.

**Secondary audience: the SDA Custody cell itself.** The capability vectors are public; a public reference implementation that applies those patterns to an open data scenario is useful in the same way any open-source reference implementation is useful — as a starting point, a counter-proposal, a teaching tool, or a sanity check.

Employment-adjacent outcomes are incidental. This project exists because the architecture is interesting and the data is public, not because a job application needs a portfolio piece.

## SDA Custody Layer capability vector alignment

| Capability vector | Custody implementation |
|---|---|
| Automated processing and fusion of data from traditional space-based sensing payloads (visible, infrared, RF, SAR, multispectral) | Multi-modal fusion of SAR (Umbra + Sentinel-1), EO (Sentinel-2), and AIS presence through unified `Observation` schema with polymorphic types (ADR-0008) |
| Design of a multi-phenomenology fusion architecture supporting agile incorporation of new algorithms | Pluggable detector interface, per-source STAC adapters, polymorphic Observation type, anomaly scorer registry, tipcue candidate scorer extension points |
| Reduction in latency of processing, exploitation, and dissemination | The offline pipeline demonstrates the architectural patterns; production latency work is documented as out of scope for this reference |
| Memory management and target hypothesis distribution from one satellite node to the next | Covariance-preserving track state serialization through the fusion index; inter-node handoff flagged as a stretch goal |

## One-line statement

*Custody is a 10-week open-source reference implementation demonstrating covariance-aware, explainable multi-sensor fusion and tip-and-cue orchestration over public maritime data, aligned with SDA Custody Layer capability vectors.*
