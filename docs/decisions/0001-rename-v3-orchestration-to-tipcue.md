---
id: 0001
title: Rename v3 orchestration module to tipcue
status: accepted
date: 2026-04-18
---

## Status

Accepted. Landed in commit `0615806` (2026-04-18).

## Context

The v3 implementation guide initially named the new per-track collect-selection module `src/custody/orchestration/`. That name collided with an existing Phase 2 module of the same path that already ships 738 LOC of fleet-wide attention allocation (`rank_portfolio`, `ACTIVE_CUSTODY` / `WATCHLIST` / `BACKGROUND` tiers, `CustodyHealth`). The Phase 2 module is used in production simulations by `dark_vessel.py` and `simulation/timeline.py`.

A single name covering both layers conflates two distinct responsibilities: portfolio-wide attention allocation ("who do we watch?") versus per-track cue scoring ("which sensor do we pull for this track?"). It also overloads an industry term — "orchestration" is the umbrella marketing word used externally (Vantor calls their version "Cortex"), which makes it a poor module name.

## Decision

Rename the v3 module from `orchestration/` to `tipcue/`. Leave the existing Phase 2 `orchestration/` module intact. Document the two layers as a composition rather than a replacement:

- **Portfolio layer** (`src/custody/orchestration/`, retained) — picks the *who*.
- **Tipcue layer** (`src/custody/tipcue/`, new in v3) — picks the *what*.

The two layers join on `PortfolioItem.entity_id` → `tracker.get_track(entity_id)`, with a ~20 LOC adapter added to `fusion/tracker.py` as the first task of Week 5.

"Orchestration" remains the external umbrella term used in README, positioning, and voiceover. Internal module names use the sharper words — `orchestration/` (portfolio) and `tipcue/` (per-track).

## Consequences

- Week 5 builds `tipcue/` from scratch against a stable, already-working input (`PortfolioAssessment` filtered to `ACTIVE_CUSTODY`), lowering risk versus reinventing the attention layer.
- Vocabulary split between internal module naming and external umbrella term is now documented in `CLAUDE.md` and `docs/custody_fusion_implementation_guide_v3.md` §1.5 / §2.10 / §4.2.
- Scripts renamed accordingly: `12_orchestrate_decisions.py` → `12_tipcue_decisions.py`.
- UI panel name (`OrchestrationTrace.tsx`) intentionally keeps the external umbrella term — the panel shows both portfolio context and tipcue decisions.
- A separate follow-up ADR will capture what subset of the portfolio output `tipcue` actually consumes in v1 (see ADR-0002).
