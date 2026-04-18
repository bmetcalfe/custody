# Custody — Project Context for Claude Code

## What this project is
Custody is an open-source reference implementation of multi-sensor fusion and legible tip-and-cue orchestration for maritime domain awareness. It applies architectural patterns described in the Space Development Agency's publicly released Custody Layer capability vectors — multi-phenomenology fusion, hypothesis management, low-latency exploitation — to the maritime domain, where open data enables building against the same principles without classified constraints.

Primary audiences for the public artifact, in order:
1. The broader GeoInt engineering community responding to the SDA STEC BAA and similar capability calls (Vantor/Sentry, BlackSky, Satellogic, Ursa, HawkEye 360, and integrating primes).
2. SDA's Custody & Emerging Capabilities Cell as an open-source reference for the architectural patterns they're soliciting.
3. Hiring managers at the above commercial entities as a side effect of doing the work honestly.

This is NOT a pitch to any single company. It's a demo answering a publicly published community call.

## What this project is NOT
- Not a hypersonic or missile tracking system. SDA's Custody Layer operates on ballistic/hypersonic threats with very different kinematics, timescales, and threat models. We apply *architectural patterns* from their capability vectors to the maritime domain. We do not claim domain equivalence.
- Not a production system. Everything runs from pre-computed Parquet in the demo.
- Not a live inference engine. Detectors run once, offline, in scripts/.
- Not a generic geospatial toolkit. Every architectural choice serves the demo narrative.
- Not a Sentry clone. Vantor Sentry is a commercial product with internal capabilities we don't have visibility into. Custody is a reference implementation focused on the *publicly visible gap*: covariance-aware planning and legible reasoning traces.

## Current phase
Day 0 complete. Data reconnaissance confirmed Umbra Spratly coverage, GFW token in hand, 56 GB Umbra mirror staged locally (219 files across 103 scenes). Sentinel-1/2 and GFW AIS pulls are Week 2 work. Week 1 starting: foundations — observation model, spatial index, EKF-based tracker.

## Authoritative references — ALWAYS read these before starting work
1. `docs/custody_fusion_implementation_guide_v3.md` — the North Star. Architecture, design decisions, implementation sequence.
2. `docs/scenario.md` — locked AOI, time window, Umbra scene inventory, demo narrative.
3. `docs/positioning.md` — the public-facing "why this exists" document. Voice differs from the implementation guide; this is what outsiders read.
4. `README.md` — may be outdated; prefer docs/ for decisions.

## Locked scenario
- AOI: bbox (114.5, 8.5, 117.5, 11.0) — Spratly hotspot
- Demo time window: June 1 – August 20, 2023 (driven by Umbra scene availability)
- Hero feature: Cuarteron-area target at 114.665°E / 8.856°N, 5 Umbra scenes over 41 days
- Secondary feature: Union Banks area at 114.63°E / 9.98°N, 3 Umbra scenes (supplementary arc)
- AIS: Global Fishing Watch research API (token in .env)
- Baseline SAR: Sentinel-1 GRD via Earth Search STAC
- Optical: Sentinel-2 L2A, opportunistic (SCS ~60% cloudy)

## The hero capability
**Covariance-aware, explainable tip-and-cue orchestration.** This is what distinguishes Custody from a ship detector or a track visualizer. The tipcue layer maintains a belief state (position mean + covariance) for every active track, evaluates candidate future collections across all constellations, scores each candidate by expected information gain, and emits both a decision and a natural-language reasoning trace.

The tipcue layer's output on every cue is required to include:
- Selected collect (sensor, time, geometry)
- Expected info gain, quantified as posterior log-determinant reduction
- Feasibility priors (cloud probability for EO, grazing angle for SAR)
- Rejected alternatives with their scores
- Plain-language justification

This is the centerpiece. Every other capability supports it.

## Two-layer orchestration: portfolio + tipcue
Custody has two orchestration-adjacent modules that compose cleanly rather than compete.

The **portfolio layer** (`src/custody/orchestration/`) operates at portfolio scope — given N active tracks, it produces the attention ranking: who is in `ACTIVE_CUSTODY`, who is `WATCHLIST`, who can remain `BACKGROUND`. It exists today (738 LOC, Phase 2 build, used by `dark_vessel.py` and `simulation/timeline.py`).

The **tipcue layer** (`src/custody/tipcue/`, new in v3) operates at decision scope — given a single track whose attention tier justifies a collect, it selects which sensor pass to cue and why.

Portfolio picks the *who*; tipcue picks the *what*. The demo's hero moment happens at the tipcue layer, but it's the portfolio layer that decides the track is worth a cue in the first place. They join on `PortfolioItem.entity_id` → `tracker.get_track(entity_id)`.

## Architectural non-negotiables
- Observation-level fusion (not track-level)
- Hungarian assignment + EKF per track (not nearest-neighbor)
- Observations carry 2×2 covariance (σ_xx, σ_yy, σ_xy) in meters²; never scalar uncertainty radius in new code
- H3 resolution 8 spatial index + DuckDB over Parquet
- Provenance chain is a first-class UI feature — every track traces to raw data URIs
- WGS84 + UTC epoch seconds internally, always
- Every cueing decision must emit a reasoning trace; opaque decisions are a bug

## Vocabulary alignment
Use these terms in code, comments, and documentation. Sloppy language is a credibility leak.

- **Tip-and-cue** — the industry term. Never "cueing loop" in public docs.
- **Multi-phenomenology fusion** — SDA's phrase. Use in architectural descriptions.
- **Custody** — maintaining continuous awareness of a target across sensor handoffs. Already the project name.
- **Orchestration** — umbrella industry term used externally (Vantor calls their version "Cortex"). Inside the repo we split it into two layers (below) and name the modules after what they actually do, not after the marketing word.
- **Portfolio layer** — `src/custody/orchestration/` (Phase 2 module, retained). Fleet-wide attention allocation: who is `ACTIVE_CUSTODY` / `WATCHLIST` / `BACKGROUND`. Picks the *who*.
- **Tipcue layer** — `src/custody/tipcue/` (v3 module). Per-track collect selection via belief-state info-gain scoring. Picks the *what*.
- **Pattern of life (PoL)** — industry standard, use as-is.
- **Dark vessel** — a ship that should be transmitting AIS but isn't. Specific term, don't substitute.
- **Belief state** — the `(mean, covariance)` of a track.
- **Reasoning trace** — the explainability output of the tipcue layer.

## Code style
- Python 3.11+
- Dataclasses preferred over classes where possible
- Type hints throughout
- Functions under ~30 lines
- No Docker, no database server, no cloud services for demo runtime (DuckDB + Parquet only)
- Frontend: React + Mapbox + deck.gl, dark intel-console aesthetic, CSS tokens before components

## Known limitations — be explicit about these in docs and voiceover
- Umbra coverage is 8 scenes on 2 features over 9 months — not continuous. Sentinel-1 fills gaps.
- Sentinel-2 cloud cover in SCS ~60% — treat EO as opportunistic.
- The demo does not identify specific flagged vessels. It detects AIS-dark activity consistent with published AMTI methodology.
- CFAR detector has false positives near reefs; length-band filter (45–65 m) mitigates for the militia-specific anomaly.
- Covariance propagation uses constant-velocity motion model with Gaussian process noise. Works for smoothly-moving commercial and militia trawler traffic; not appropriate for high-maneuver targets.

## Day 0 artifacts
- `day0/ship_detection_centroids.csv` — all 995 Umbra scene centroids from the public S3 catalog
- `day0/scan_output.txt`, `day0/download_log.txt` — reconnaissance + download logs
- `day0/scan_umbra.py`, `day0/download_scenes.py` — reproducer scripts (idempotent)
- `day0/legacy/` — archived patches from decommissioned worktree branches
- `data/raw/umbra/` — 56 GB, 219 files, 103 scenes (Tier 1 AOI: 29 files / 22 GB; Tier 2 SCS-broad: 190 files / 38 GB). Gitignored.
- Sentinel-1, Sentinel-2, and GFW AIS staged data: not yet fetched — Week 2 pipeline.

## Before starting any task
1. Is this aligned with the hero capability or directly supporting it?
2. Does this touch the belief-state math? If yes, write the test first. Uncertainty bugs are silent and catastrophic.
3. Does this add vocabulary drift from the list above?
4. Am I making a claim the demo can't back up? If yes, soften the claim.
