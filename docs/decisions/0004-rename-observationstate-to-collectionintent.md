---
id: 0004
title: Rename ObservationState to CollectionIntent; reserve Observation for raw detections
status: accepted
date: 2026-04-18
---

## Status

Accepted. Rename not yet executed — blocked on confirmation before Week 1 code moves.

## Context

v3 §2.3 introduces a frozen `Observation` dataclass for **raw sensor detections**: `obs_id`, `source_id`, `modality` (SAR / EO / AIS), `acquisition_time`, `lat`, `lon`, a 2×2 covariance `(σ_xx, σ_yy, σ_xy)` in meters², vessel length / heading / SOG estimates, provenance (`raw_ref`, `detector_version`), and so on. This is the central concept for observation-level fusion: every sensor reading becomes one `Observation`, and the EKF tracker consumes them to update track belief state. The name "Observation" is the standard term in the tracking literature and will be the more central abstraction going forward.

The existing `src/custody/observation.py` (279 LOC) exports an `ObservationState` dataclass — a **derived collection-intent bookkeeping record**: `last_observation_time`, `last_sensor`, `hours_since`, `collection_intent` (SEARCH / CONFIRM / CHARACTERIZE / MONITOR), `needs_cross_sensor`, `preferred_confirmation_sensor`, `rationale`. Its purpose is to track what the system has collected for an entity and drive tasking decisions — it's not a raw observation record at all.

The module name (`observation.py`) and the class name (`ObservationState`) both drift from what the code actually does. The v3 `Observation` has a stronger claim to the name.

## Decision

Rename `ObservationState` → `CollectionIntent`. The new name matches the dataclass fields: the `collection_intent` field is the primary payload, and "collection intent" is what the class represents (what we *intend* to collect based on what we've already collected). This frees the `Observation` name for the v3 raw-detection type in `src/custody/fusion/observations.py`.

The existing module file (`observation.py`) is renamed to `collection_intent.py` for symmetry with the class name. The existing helper `derive_observation_state()` renames to `derive_collection_intent()`; `apply_observation_to_policy()` renames to `apply_collection_intent_to_policy()`.

## Consequences

- Larger refactor than ADR-0003 because the module, class, and top-level functions all rename. Imports update across callers (`from custody.observation import ObservationState, derive_observation_state, apply_observation_to_policy` → `from custody.collection_intent import CollectionIntent, derive_collection_intent, apply_collection_intent_to_policy`).
- `tests/test_observation.py` moves to `tests/test_collection_intent.py`.
- The existing constants `SEARCH`, `CONFIRM`, `CHARACTERIZE`, `MONITOR` and `STALE_OBSERVATION_HOURS` / `RECENT_OBSERVATION_HOURS` remain at module scope in the renamed `collection_intent.py`. The `CollectionIntent` dataclass is introduced alongside them. Future consolidation of the string constants into a `StrEnum` is deferred to a separate ADR if needed.
- The v3 `Observation` type can now land in `src/custody/fusion/observations.py` with the correct name, documenting its role as "raw sensor detection with covariance and provenance" free of collision.
- Vocabulary note: "observation" in v3 and CLAUDE.md refers to the raw detection. "Collection intent" (this module) refers to the tasking-layer bookkeeping. The rename documents that distinction in code, not just in prose.
- Clearer reviewer experience: someone reading `from custody.fusion.observations import Observation` and `from custody.collection_intent import CollectionIntent` can tell the two apart immediately.
