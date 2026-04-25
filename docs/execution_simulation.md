# Plan Execution Simulation

*Companion to [docs/positioning.md](positioning.md), [docs/scheduler_lite.md](scheduler_lite.md), and [docs/scene_availability_bridge.md](scene_availability_bridge.md). Documents the closed-loop feedback simulator added in [ADR-0021](decisions/0021-custody-as-uncertainty-to-tasking-engine.md) Slice 23.*

---

## What execution simulation is

Execution simulation closes the prototype decision loop. Given a `SchedulePlan` from [scheduler-lite](scheduler_lite.md) and a deterministic outcome policy, it:

1. Generates a synthetic returned-evidence record for each scheduled candidate collect.
2. Feeds that evidence back through `update_state` to produce a new `HypothesisState`.
3. Recomputes `assess_custody_health` and `rank_collection_candidates` against the updated state.
4. Reports pre/post belief, custody-health change, ambiguity resolution, and the next recommendation.

The full chain answered by Slices 1–23 is:

```
artifact evidence -> hypothesis state -> custody health -> ambiguity ->
collect ranking -> mission value -> counterfactual -> optimization ->
availability adjustment -> schedule feasibility -> simulated collect result ->
updated belief -> next recommendation
```

## How it closes the decision loop

Earlier slices stop at "what would we recommend?". Execution simulation extends to "given we recommended these collects, scheduled them, and they returned synthetic evidence under the named outcome policy, what does the updated belief look like?". This makes the loop end-to-end runnable on a laptop without ever leaving the deterministic, fixture-only prototype envelope.

## Outcome policies

Four deterministic outcome policies drive the simulation:

| Policy          | Behaviour                                                                 | Confidence (default) |
| --------------- | ------------------------------------------------------------------------- | -------------------- |
| `favorable`     | Returned evidence supports the **first** hypothesis of the primary ambiguity pair and contradicts the second. | `schedule_score` (or 0.75 fallback) |
| `adverse`       | Returned evidence supports the **second** hypothesis of the primary pair and contradicts the first.          | `schedule_score` (or 0.75 fallback) |
| `inconclusive`  | Returned evidence is empty (no supports / no contradicts), low confidence; primary ambiguity is preserved.   | 0.05 |
| `mixed`         | Across multiple scheduled collects, alternate `favorable`, `inconclusive`, `favorable`, ... in order. With a single collect, behaves like inconclusive at slightly higher confidence (0.15). | per alternation |

If `health.ambiguity_pairs` is empty, the simulator returns an empty result with `result_quality = 0.0` and a caveat — execution simulation is most informative when custody has surfaced an ambiguity to drive against.

## Synthetic evidence shape

Every simulated result becomes a `HypothesisEvidence` with:

- `source_kind = "execution_simulation"`
- `source_ref = f"schedule:{window_id}:{candidate_id}"`
- `evidence_id = f"exec_sim_{scenario}_{candidate_id}_{window_id}"` (deterministic)
- `timestamp = scheduled_collect.end_time`
- `weight = 1.0`
- `supports`, `contradicts`, `confidence`, `reason` per outcome policy.

The evidence is fed to `update_state` exactly the same way artifact-bridge evidence is — the simulator does not have a privileged path through the belief-update engine.

## Difference from counterfactual simulation (Slice 10)

| Concern                        | Counterfactual (Slice 10)                                         | Execution simulation (Slice 23)                                            |
| ------------------------------ | ----------------------------------------------------------------- | -------------------------------------------------------------------------- |
| When it runs                   | Before plan selection                                             | After scheduling                                                            |
| Question answered              | "What would happen if we chose collect X vs Y?"                   | "Given we scheduled these, what does the updated belief look like?"        |
| Output                         | Per-candidate ambiguity-resolution proxy                          | Updated state, health, ambiguity, and next recommendation                  |
| Acts on belief                 | No (informs ranking only)                                         | Yes (calls `update_state`)                                                  |
| Inputs                         | `state`, `health`, `recommendation`                               | `state`, `SchedulePlan`, `outcome_policy`                                  |

Both are deterministic, both are stdlib-only, and neither talks to an external service.

## What is NOT claimed

- No live tasking or sensor command is issued.
- No imagery is downloaded or processed.
- No execution authorization is issued.
- No platform-access decisions are claimed.
- The outcome policy is **deterministic**, not a calibrated probability.
- The synthetic evidence is **synthetic** — it does not represent an actual collection result, and the simulator does not claim to model real sensor returns.

## Future work

- Real artifact ingestion after a collect closes (replace synthetic evidence with provenance-linked real artifacts).
- Operator-approved execution path: planner approval gates between scheduling and execution.
- Real-result metadata: provenance manifests for executed collects feed back into the state update.
- Uncertainty calibration: outcome confidence informed by historical detection / matcher performance.
- Multi-round iterative loops: chain Slice 23 across consecutive planning cycles.
- Confidence decay over time so old simulated evidence ages out predictably.

None of these are in scope today. The current simulation is intentionally a small, deterministic, stdlib-only feedback loop over committed JSON fixtures.
