---
id: 0002
title: Tipcue v1 consumes ACTIVE_CUSTODY only
status: accepted
date: 2026-04-18
---

## Status

Accepted.

## Context

The Phase 2 portfolio layer (`src/custody/orchestration/`) partitions tracks into three attention tiers: `ACTIVE_CUSTODY`, `WATCHLIST`, and `BACKGROUND`. V3's tipcue layer (`src/custody/tipcue/`, per ADR-0001) scores candidate collects against individual tracks and emits a cueing decision with reasoning trace. The composition code from §4.2 of the v3 guide is tier-agnostic — it filters `PortfolioAssessment.items` by `attention_state` before handing tracks to `decide_collect`, and the filter predicate is a single line.

The question: should the v1 tipcue run against `ACTIVE_CUSTODY` only, or should it also score passes against `WATCHLIST` tracks?

Arguments for scoring `WATCHLIST` too:
- The composition code handles it trivially — `attention_state in ("ACTIVE_CUSTODY", "WATCHLIST")`.
- Shows the architecture scales across tiers.

Arguments for `ACTIVE_CUSTODY` only:
- `WATCHLIST` is explicitly "we're watching but haven't committed to cueing." Scoring passes against tracks we haven't committed to watching adds noise to the demo — reasoning traces for collects we will never actually cue.
- The demo's hero moment (Act 3) is a single confident, well-scoped cue. Flooding the trace panel with `WATCHLIST` speculation undermines the legibility claim that is the whole point.
- Budget realism: real operational systems don't spend compute scoring every tier at equal rigor.

## Decision

V1 tipcue consumes `ACTIVE_CUSTODY` tracks only. `WATCHLIST` scoring is a documented extension point, not a v1 feature.

## Consequences

- The demo narrative stays tight: portfolio ranks the fleet, tipcue fires on the small set that crossed into `ACTIVE_CUSTODY`. One clear story, not two half-stories.
- The composition code in §4.2 already filters by `attention_state == "ACTIVE_CUSTODY"` — enabling `WATCHLIST` scoring later is a one-line change to the predicate, so this is genuinely deferral, not a design debt.
- README and voiceover should not claim tipcue "scores all tracks" — it scores tracks the portfolio has elevated to `ACTIVE_CUSTODY`. This is a legibility feature, not a limitation.
- §4.2 of the v3 guide gets a short scope note referencing this ADR.
- If post-demo feedback asks for `WATCHLIST` reasoning traces, a new ADR captures that reversal and the predicate changes in one place.
