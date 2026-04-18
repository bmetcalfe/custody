---
id: 0003
title: Rename fusion.py to belief_assessment.py to free the fusion/ namespace
status: accepted
date: 2026-04-18
---

## Status

Accepted. Rename not yet executed — blocked on confirmation before Week 1 code moves.

## Context

v3 §6 plans a `src/custody/fusion/` **package** containing `observations.py`, `index.py`, and `tracker.py` — covering the observation model, H3 + DuckDB spatial index, and Hungarian + EKF tracker respectively. These are the foundations of Week 1.

The existing `src/custody/fusion.py` is a 419 LOC single-file module emitting a `FusionAssessment` — a belief-evidence summarization whose purpose is "given anomaly, compound signals, custody confidence, and planner context, produce a fused score + uncertainty + source agreement + missing evidence + recommended confirming source." It is Layer 4 in the existing reasoning architecture: *belief summarization* for the downstream decision engine. It does not handle raw observations, spatial indexing, or tracking.

Two unrelated concepts have collided on the same name. Python cannot have both a `fusion.py` file and a `fusion/` package in the same directory, so one must move.

## Decision

Rename `src/custody/fusion.py` → `src/custody/belief_assessment.py`. The new name matches the module's actual responsibility (its primary export is `FusionAssessment`, and its conceptual job is belief-evidence summarization). The `fusion` name is freed for the v3 package per §6.

Associated test file rename: `tests/test_fusion.py` → `tests/test_belief_assessment.py`.

Per-ADR policy, this rename crosses the module-boundary threshold because imports across the repo must update.

## Consequences

- `from custody.fusion import FusionAssessment, build_fusion_assessment, fusion_for_timeline` → `from custody.belief_assessment import ...` across every caller.
- `tests/test_fusion.py` moves to `tests/test_belief_assessment.py`; internal import paths update.
- The v3 `src/custody/fusion/` package can now be created without collision.
- The `FusionAssessment` class name itself is retained (renaming the class would ripple further; the class name still reads cleanly even from a module named `belief_assessment`).
- Vocabulary note: "belief state" in v3 / CLAUDE.md means `(mean, covariance)` of a track (per the tipcue layer). "Belief assessment" (this module) means the higher-level evidence summary. The two live at different layers and should not be conflated in docs or voiceover — the new module name documents that layer separation explicitly.
- No public API of the existing module changes; only the import path.
