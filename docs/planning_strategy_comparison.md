# Planning Strategy Comparison

*Companion to [docs/positioning.md](positioning.md), [docs/scheduler_lite.md](scheduler_lite.md), [docs/scene_availability_bridge.md](scene_availability_bridge.md), and [docs/execution_simulation.md](execution_simulation.md). Documents the deterministic strategy comparison harness added in [ADR-0021](decisions/0021-custody-as-uncertainty-to-tasking-engine.md) Slice 24.*

---

## Why compare strategies

Earlier slices each layer one decision-support capability on top of the last. The harness in Slice 24 asks the question that ought to come back at every prototype gate review:

> Is each layer earning its place against simpler approaches?

The comparison harness builds the same upstream pipeline every other slice uses, then evaluates six strategies of increasing decision-support depth. It scores each one on deterministic prototype metrics (planning utility, mission-value proxy, ambiguity resolution, custody-health delta, schedule feasibility, traceability, review burden) and ranks them. It is not a benchmark of real planner performance; it is a sanity check on whether the prototype's added complexity translates into observable proxy gains.

## Strategies compared

| ID                          | Strategy                                                                              | Stack used                                                                  |
| --------------------------- | ------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| `baseline_manual`           | Manual analyst review proxy with no decision-stack support                            | None                                                                        |
| `collection_value_only`     | Top-N candidates from the collection-value ranker, filtered by budget                 | Slice 5                                                                     |
| `mission_value_optimized`   | Constrained optimizer using mission-value attribution and counterfactual              | Slices 5 + 7 + 8 + 11                                                       |
| `availability_adjusted`     | Availability-adjusted constrained optimizer                                           | Slices 5 + 7 + 8 + 11 + 20 + 21                                             |
| `scheduler_lite`            | Availability-adjusted optimizer + schedule feasibility simulation                     | Slices 5 + 7 + 8 + 11 + 20 + 21 + 22                                        |
| `execution_feedback`        | Closed loop with synthetic returned evidence                                           | Slices 5 + 7 + 8 + 11 + 20 + 21 + 22 + 23                                   |

`baseline_manual` is intentionally weak: no candidate selection, no scheduling, no execution simulation. It exists so the comparison can show whether each downstream layer has visible incremental value.

## Metrics and proxies

Each `StrategyEvaluation` carries:

| Metric                            | Source                                                                                   |
| --------------------------------- | ---------------------------------------------------------------------------------------- |
| `selected_candidate_ids`          | Per-strategy plan output                                                                 |
| `total_cost`                      | Sum of selected candidate `relative_cost`                                                |
| `planning_utility`                | Strategy-specific (collection-value sum, optimizer total, or adjusted utility)           |
| `mission_value_proxy`             | Sum of mission-value `total_value` for selected candidates                               |
| `expected_ambiguity_resolution`   | Per-candidate counterfactual proxy summed for selected candidates                         |
| `schedule_feasible_count`         | Count of (candidate, window) pairs scheduler-lite was able to place                      |
| `schedule_unscheduled_count`      | Count of selected candidates with no eligible window                                     |
| `pre_health_score` / `post_health_score` | Custody health before / after execution simulation                                |
| `health_score_delta`              | Post minus pre (clamped to 0 in the comparison score)                                    |
| `ambiguity_resolved`              | Whether the primary ambiguity was removed by simulated execution                          |
| `traceability_artifact_count`     | Deterministic proxy: 1 (manual) -> 7 (closed loop)                                        |
| `review_actions_required`         | Deterministic proxy: 3 (manual) -> 1 (closed loop)                                        |

The comparison score is a fixed-weight combination:

```
score =
    0.25 * planning_utility_normalized
  + 0.20 * mission_value_proxy_normalized
  + 0.20 * expected_ambiguity_resolution
  + 0.15 * positive_health_delta
  + 0.10 * schedule_feasibility_ratio
  + 0.10 * traceability_score
```

Tie-breakers, in order: `ambiguity_resolved`, `health_score_delta`, `total_cost`, `strategy_id`.

## What improves from baseline to full loop

For both Tennent and Whitsun under the `favorable` outcome policy:

- `baseline_manual` ends with no candidate selection and no health change; trace count 1, review burden 3.
- `collection_value_only` selects candidates but produces no mission-value attribution and no scheduling.
- `mission_value_optimized` introduces utility weighted by mission relevance and counterfactual ambiguity-resolution.
- `availability_adjusted` reduces planning utility for candidates whose feasibility is weak in the metadata window.
- `scheduler_lite` adds a non-zero schedule-feasibility count.
- `execution_feedback` provides a custody-health delta and (under `favorable`) resolves the primary ambiguity.

The deterministic ranking under `favorable` typically picks `execution_feedback` as the winner. Under `inconclusive`, `scheduler_lite` and `availability_adjusted` rise relative to `execution_feedback` because the closed loop returns no useful health delta.

## What is NOT claimed

- **No production KPI.** The metrics are deterministic proxies, not measurements.
- **No real planner performance comparison.** The baseline is a static ad-hoc proxy, not a stopwatch on a human analyst.
- **No live tasking or sensor command.** The same disclaimer that applies to every other Slice 1-23 module applies here.
- **No real revenue claim.** Mission value and planning utility are proxy-only.
- **No trained RL agent.** The harness is an evaluation substrate; no policy is learned, fitted, or trained.
- **No operational outcome attribution.** This slice does not assert that any operational outcome was changed by the prototype.

## Future work

The harness lays groundwork for - but does not implement - any of the following:

- A trained policy that picks among strategies given a state / health snapshot, evaluated against this harness.
- Calibration of the comparison-score weights against a labelled corpus.
- Per-strategy confidence intervals across many simulated outcomes.
- Operator-in-the-loop strategy selection with audit trails feeding back into the planner queue.
- Real-data replay of past planning cycles to anchor proxy metrics against historical planner choices.

None of these are in scope today. The harness is intentionally a small, deterministic, stdlib-only evaluation substrate over committed JSON fixtures.
