# CLAUDE.md — Operating Guide for Claude Code CLI Sessions

*CC's entry point. Read this first every session. Contains only what CC needs to stay aligned with the project's state and conventions. Full architecture lives in docs/custody_fusion_implementation_guide_v3.md; full positioning lives in docs/positioning.md.*

---

## One-line project definition

Custody is a 10-week open-source reference implementation of multi-sensor fusion and covariance-aware tip-and-cue orchestration for maritime domain awareness, aligned with the SDA Custody Layer capability vectors, built on open data.

## What's the scenario

**Primary case, in the 90-second demo cut:** Vietnamese land reclamation at Tennent Reef (Đá Tiên Nữ, 8.856°N / 114.665°E) during June–August 2023. Five Umbra SAR scenes over 41 days document active dredging at the eastern artificial island (Tiên Nữ B). The pipeline detects this as a persistent AIS-dark SAR return. CSIS AMTI has documented this feature's expansion in their December 2022 and November 2023 reports.

**Supplementary case, documented but not in the demo cut:** Chinese maritime militia activity at Whitsun Reef (Đá Ba Đầu, 9.98°N / 114.63°E), December 2023 – March 2024, three Umbra scenes. Site of the March 2021 Chinese militia swarm event. Long-baseline change-detection case study.

**Not in scenario:** Cuarteron Reef (8.85°N / 112.85°E, Chinese-controlled, ~180 km west of our primary target) is a reference feature only. It is sometimes cited in Spratly reporting but is not in our Umbra inventory and is not part of the Custody demo. See ADR-0012 for how the primary and supplementary features were selected.

The pipeline is scenario-agnostic. CC should not make semantic assumptions about vessels, militias, or claimants in code — the architecture operates on AIS observations, SAR detections, and tracks, and the scenario-layer meaning comes from documentation and voiceover.

## What's built

**Week 1 (done):** src/custody/fusion/ package — geo.py (AEQD projection), observations.py (polymorphic Observation sum type), index.py (H3+DuckDB spatial index), tracker.py (Hungarian + lifecycle). EKF in src/custody/models.py. 884 LOC, 75 tests.

**Week 2 (in progress):** src/custody/ingest/gfw_presence.py landed with AIS passthrough via GFW presence data as PositionObservation (ADR-0011). Full 11-week AIS data processed: 39,337 observations, 601 unique MMSIs. SAR CFAR detection pending.

**Phase 2 foundations (preserved, composed):** src/custody/orchestration/ (portfolio attention engine), src/custody/anomalies.py, src/custody/compounds.py. Feeds the v3 tipcue layer per ADR-0001.

## Key conventions

- **Internal coordinates:** AEQD tangent-plane meters anchored at AOI center (9.75°N, 116.0°E) per ADR-0009. `src/custody/fusion/geo.py` is the only place lat/lon ↔ meters conversion happens for fusion math.
- **Track state:** 4-dimensional [x_east_m, y_north_m, v_north, v_east] in tangent-plane meters per ADR-0010. `TrackState.lat` and `TrackState.lon` are properties that derive via from_tangent_plane.
- **Observation types:** Polymorphic sum type per ADR-0008. `PositionObservation` for position-only sensors (SAR, EO, and GFW presence-derived AIS per ADR-0011). `PositionVelocityObservation` defined but unused in v1 — preserved for future sensors.
- **Modality literal:** `Literal["SAR", "EO", "AIS"]` on `PositionObservation`. AIS is a valid modality for `PositionObservation` post-ADR-0011.
- **Tests first:** every new module lands with tests written before implementation. Tests must confirm import failures before the module exists (guards against silent passes).
- **ADRs:** any architectural decision with blast radius beyond a single file goes into docs/decisions/ as a numbered ADR before implementation. 12 ADRs as of this writing.

## Pre-task checklist (run every session)

1. `git status` — check for uncommitted work
2. `git log --oneline -10` — verify you know where HEAD is
3. `pytest --co -q | tail -1` — verify test count matches expectations (~2173 as of end-Week-2-Day-2)
4. Read any ADRs drafted since the last session (`git log --oneline docs/decisions/`)
5. Verify understanding of the current scenario from this file — primary target is Tennent Reef, Vietnamese reclamation, summer 2023

## Anti-patterns CC should refuse

- **Using np.inf in scipy.optimize.linear_sum_assignment cost matrices** — use a _BIG_COST sentinel instead
- **Silent test passes** — a new test module must fail with ImportError before the module exists. Confirm failures are import errors, not silent returns
- **Observation mutations** — all Observation instances are frozen dataclasses. Any code attempting mutation is wrong
- **Geographic identification requires verification.** Never identify a feature by coordinate range alone — always cross-check coordinates against an authoritative reference (Wikipedia, AMTI island tracker, official gazetteer) before writing scenario-relevant labels into documentation or code
- **Live API calls in pytest** — all tests requiring external data use fixtures under tests/fixtures/. The fetch scripts (scripts/01_*, scripts/02_*) are developer actions, not test infrastructure

## Three-Claude workflow

CC is the source of truth for repo state. The other two Claudes are adjacent but not authoritative.

- **Web Claude (Opus 4.7):** strategic planning, writing, external research, ADR drafting. Produces documents that CC applies. Runs in a chat context with browsing tools but no filesystem access to the repo.
- **CC (Opus 4.7):** exclusive repo authority. Only actor that writes files, runs tests, commits, pushes.
- **Cowork (Opus 4.6):** sandboxed desktop agent with Chrome. Reference gathering, visual QA, file organization in scratch directories. Not repo state.

If CC encounters disagreement between what another Claude said and what the repo contains, the repo wins. Web Claude and Cowork cannot modify repo state; their outputs are proposals that CC evaluates.

## What to do when uncertain

1. If an ADR would resolve the question, draft one and ask the user before implementing
2. If a test would disambiguate, write the test first and see what it reveals
3. If in doubt about scenario framing, re-read `docs/scenario.md` and ADR-0012. Do not extrapolate from CLAUDE.md's compressed version.
4. If in doubt about whether a change is backward-compatible, grep for callers and report before changing

## Authoritative docs

- docs/positioning.md — public-facing what-this-is
- docs/scenario.md — AOI, window, Umbra inventory, demo narrative
- docs/custody_fusion_implementation_guide_v3.md — architecture, weekly plan
- docs/decisions/ — 12 ADRs numbered 0001–0012

Anything stated in CLAUDE.md that contradicts one of those files is stale. The four numbered docs are the source of truth. CLAUDE.md is a quick-access index only.
