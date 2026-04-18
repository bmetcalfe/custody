# Custody — Positioning

*This document describes why Custody exists, who it's for, and what it is and isn't.*

## The community call

In early 2024, the Space Development Agency published a Broad Agency Announcement soliciting industry proposals for its Proliferated Warfighter Space Architecture. One capability layer, the Custody Layer, sets out a public list of technical areas where the agency is looking for contributions. Several of them are architectural rather than domain-specific:

- Automated processing and fusion of data from traditional space-based sensing payloads across visible, infrared, RF, synthetic aperture radar, and multispectral modalities
- Design of a multi-phenomenology fusion architecture that enables agile incorporation of new algorithms
- Reduction in latency of processing, exploitation, and dissemination
- Memory management and target hypothesis distribution across satellite nodes

These are general patterns. They appear in commercial maritime domain awareness products (Vantor's Sentry, BlackSky's analytics, Satellogic's AI-first constellation, Ursa's SAR analytics) and in classified programs we have no visibility into. They also appear in the academic tracking literature going back decades.

What's harder to find, publicly, is a **reference implementation** that shows these patterns working end to end on open data. Commercial products are closed. Academic work usually studies one layer in isolation. Classified work is classified.

Custody is an attempt to fill that gap for the maritime domain.

## What Custody is

An open-source reference implementation of multi-sensor fusion and legible tip-and-cue orchestration, built against open commercial and public data sources:

- Umbra SAR Open Data (CC BY 4.0)
- Sentinel-1 and Sentinel-2 via Earth Search STAC
- Global Fishing Watch AIS (research license)

The demo centers on a real geopolitical setting — the Spratly Islands in the South China Sea — where published analytic methodology (CSIS Asia Maritime Transparency Initiative) documents that a large fraction of vessels of interest are systematically AIS-dark. This makes the region the canonical real-world test case for AIS/SAR fusion.

The architecture implements:

- A federated STAC catalog across five open-data sources
- A unified `Observation` model with covariance and full source provenance
- H3 + DuckDB spatiotemporal indexing
- Hungarian assignment and per-track Extended Kalman Filters
- Multi-source anomaly scoring (AIS/SAR disagreement, dark-vessel detection, loitering in militia-trawler length band, cross-source class mismatch)
- A covariance-aware planner that scores candidate collections by expected information gain
- Natural-language reasoning traces for every cueing decision

The hero capability is the last two together. When a track enters watch state, the orchestration layer doesn't just pick the next available pass — it evaluates every candidate collection across every available constellation, scores each by expected posterior uncertainty reduction against the growing belief-state covariance, weighs modality-specific feasibility priors (cloud forecast for EO, grazing angle for SAR, pass geometry), and explains the decision. The output of the planner is both a tasking action and a reasoning trace an analyst or reviewer can audit.

## What Custody is not

**Not a hypersonic or missile tracking system.** SDA's Custody Layer primarily addresses ballistic and hypersonic threats with kinematics and timescales that do not translate to ships. Custody applies architectural patterns described in the capability vectors to a domain where the kinematics are smoothly-moving commercial and militia vessel traffic. The *architecture* is aligned with the capability call; the *threat model* is maritime. Any suggestion of direct domain applicability to missile defense would be wrong.

**Not a production system.** The demo runs against pre-computed Parquet artifacts produced by a one-shot preprocessing pipeline. The architecture supports live operation in principle; the demo does not implement it.

**Not a Vantor Sentry clone.** Sentry is a commercial product with significant internal capabilities and a constellation and archive we don't have access to. Custody focuses on the publicly visible gap: making planning decisions and their uncertainty explicit and auditable. Whether or how similar functionality exists inside Sentry or competing products is not a claim we make either way.

**Not an intelligence product.** The demo detects AIS-dark vessel activity in a widely-reported geopolitical flashpoint following a published analytic methodology. It does not identify specific flagged vessels and does not make legal or sovereignty claims. The contested-waters framing is the one for which open data and public methodology exist; the choice is driven by data availability, not advocacy.

## What it's designed to demonstrate

Three things, specifically:

1. **Legible tasking decisions.** Every cue the orchestration layer emits is accompanied by a reasoning trace — what was selected, what was considered, what the expected information gain was, why alternatives were rejected. Commercial products generally expose the decision; we expose the reasoning.

2. **Covariance-aware planning.** The planner reasons about belief state as a first-class object. It doesn't pick the next available pass over the target's last known position; it scores passes by their effect on the covariance and picks the one that reduces uncertainty the most under feasibility constraints.

3. **Open architecture.** Every boundary in the system — observation schema, track representation, sensor interface, cue policy, planner — is a clean interface that could accept a new source or a new algorithm without ripple effects. The reference implementation shows the patterns working end to end in a way closed commercial systems structurally cannot.

## Who this is for

**Engineers working on multi-sensor tasking and fusion systems**, whether at Vantor, BlackSky, Satellogic, Ursa, HawkEye 360, Planet, or any of the primes bidding into SDA's BAA. If the patterns are useful, fork the repo. If they're not, the postmortem on why is itself useful.

**Researchers working on sensor scheduling under uncertainty.** The planner implementation is a clean, readable Python reference you can use as a baseline to compare more sophisticated approaches (MDP-based, reinforcement learning, etc.).

**Hiring managers looking for people who can build in this space.** The repo's commit history, design decisions, and honest scoping are the interview.

## Honest scoping

This is a 10-week evenings-and-weekends project by one engineer. Limitations:

- Umbra coverage over the chosen AOI is 8 scenes at 2 features over 9 months. Sentinel-1 provides continuous fill-in at 10m.
- Detection uses classical CFAR plus (optionally) a pretrained CNN for optical. No fine-tuning on the specific AOI.
- The tracker uses a constant-velocity motion model. Appropriate for the maritime domain studied; not for high-maneuver targets.
- Association uses Hungarian with Mahalanobis gating. MHT (Multiple Hypothesis Tracking) is documented as future work, not implemented.
- The planner's feasibility priors are simplified — binary for SAR grazing angle, probabilistic for cloud forecast. A production system would extend these.
- Sensor fingerprinting, vessel re-identification from imagery, and 20+ year archive pattern-of-life are commercial capabilities we do not reproduce.

These limitations are called out in the demo narration and the README. The goal is not to look impressive; the goal is to be correct about what was built and why.

## References

- Space Development Agency, Custody Layer and STEC BAA. https://www.sda.mil/custody/
- CSIS Asia Maritime Transparency Initiative, annual reports on Chinese maritime militia presence in the Spratlys.
- Umbra Open Data Program, AWS Open Data Registry.
- Global Fishing Watch, research API documentation.
- Copernicus / ESA, Sentinel-1 and Sentinel-2 documentation via Earth Search STAC.
