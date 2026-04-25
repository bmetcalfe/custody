# Collection-Window Scheduler-Lite

*Companion to [docs/positioning.md](positioning.md), [docs/artifact_bridge.md](artifact_bridge.md), and [docs/scene_availability_bridge.md](scene_availability_bridge.md). Documents the schedule-feasibility simulator added in [ADR-0021](decisions/0021-custody-as-uncertainty-to-tasking-engine.md) Slice 22.*

---

## What scheduler-lite is

Scheduler-lite is a **schedule feasibility simulator**. It consumes the availability-adjusted optimized plan from [Slice 21](pivot_audit_uncertainty_to_tasking.md) plus a small JSON catalog of provider-neutral collection windows, and asks one question:

> Can the recommended candidate collect *types* fit into these windows under simple capacity, timing, and conflict constraints?

The output names which (candidate, window) pairs would form a feasible plan and which candidates have no eligible window. Each scheduled pair carries a deterministic schedule score derived from the candidate's adjusted utility, the window's quality score, and a latency factor.

## What scheduler-lite is not

- **Not orbital scheduling.** No ephemerides, no access calculations, no orbit propagation.
- **Not platform tasking.** It does not address a specific satellite, ground station, or provider.
- **Not live tasking.** No execution authorization is issued; no sensor-control instructions are generated.
- **Not constellation management.** No autonomous resource arbitration across a fleet.
- **Not MPS integration.** No mission-planning-system protocol, no real-time rescheduling, no priority queues.

If the prototype ever needs to reach a real scheduler, this slice is the *interface contract* the bridge will be built against, not the bridge itself.

## How candidate collect types map to windows

A `CollectionWindow` carries a `candidate_id` matching one of the six sensor-generic collect types from [`collection_value.py`](../src/custody/hypotheses/collection_value.py): `repeat_sar`, `cross_geometry_sar`, `higher_resolution_sar`, `optical_context`, `ais_coverage_query`, `wait_or_monitor`. The scheduler matches by `candidate_id` only — provider, platform, and ground-segment identity are intentionally absent.

Each window also carries:

- `start_time` / `end_time` (UTC, ISO 8601)
- `capacity_cost` (positive float; abstract resource units, not minutes or megabytes)
- `resource_type` (free-form string; recommended values: `sar`, `optical`, `ais`, `generic`)
- `quality_score` (0.0–1.0; provider-neutral data-quality proxy)
- `latency_hours` (>= 0; downstream availability latency proxy)

## Constraints modeled

| Constraint                  | What it does                                                                                |
| --------------------------- | ------------------------------------------------------------------------------------------- |
| `max_total_capacity`        | Bounds the sum of `capacity_cost` across all scheduled windows.                             |
| `max_overlapping_collects`  | Bounds how many scheduled windows may overlap in time at any boundary point.                |
| `earliest_start`            | Drops windows starting before this UTC instant.                                              |
| `latest_end`                | Drops windows ending after this UTC instant.                                                 |
| `allowed_resource_types`    | If non-empty, restricts windows to these resource types.                                     |
| `excluded_window_ids`       | Excludes specific window IDs from consideration.                                             |

The scheduler also enforces invariants that a schedule must satisfy:

- Each candidate is scheduled at most once.
- Each window is used at most once.

## Constraints not modeled

- Orbital mechanics (ephemerides, access geometry, sun angles).
- Weather, cloud cover at acquisition time.
- Ground-station contacts, downlink budget, or processing chains.
- Real-time priority queues or pre-emption.
- Human approval gating.
- Latency past 96 hours (folded into a coarse latency factor).
- Inter-window setup or slewing time.

## Schedule score

For a (candidate, window) pair:

```
schedule_score = clamp01( adjusted_utility * quality_score * latency_factor )

latency_factor:
  latency_hours <= 6   -> 1.0
  latency_hours <= 24  -> 0.8
  latency_hours <= 72  -> 0.6
  latency_hours >  72  -> 0.4
```

The optimizer picks the assignment that maximises total `schedule_score` across all selected pairs. Tie-breakers, in order:

1. Higher total schedule score.
2. Lower total capacity used.
3. Earlier maximum end time across scheduled windows.
4. Lexicographic tuple of sorted `window_id`s.

The search is **deterministic exhaustive** over subsets of selected candidates and per-candidate window choices.

## Why this is not orbital scheduling or platform tasking

Real orbital scheduling needs ephemerides, access geometry, ground-station contacts, downlink budgets, weather, payload calibration windows, and platform-specific operating constraints. Real platform tasking adds authorisation, accounting, queue priority, and a control protocol. Scheduler-lite has none of those; its job is to surface, in metadata terms, whether a recommended candidate collect type has a viable home in the next planning horizon.

## Future work

- Real access-window metadata sourced from a STAC catalog or provider scheduler.
- Platform-specific scheduler integration (publish-side only, with explicit credentials and approval gating).
- Resource calendars and contention modelling (downlink, processing chains).
- Conflict resolution under priority and pre-emption.
- Latency and processing-chain modelling beyond the coarse factor.
- Human approval gating before any external dispatch.
- Round-trip lineage: provenance manifests for scheduled artifacts, not just decision packets.

None of these are in scope today. Scheduler-lite is intentionally a small, deterministic, stdlib-only simulator over committed JSON fixtures.

## Downstream feedback (Slice 23)

The plan execution simulator ([Slice 23](execution_simulation.md), `src/custody/hypotheses/execution_sim.py`) consumes the `SchedulePlan` produced by scheduler-lite, generates synthetic returned evidence under a deterministic outcome policy, feeds that evidence back through the belief-update engine, and reports the resulting custody-health change and updated recommendation. That closes the prototype decision loop end-to-end.
