# Mission-Planning Complexity Map

A short orientation to the two workflows the Slice 15 efficiency report compares, and where each shipped Custody module fits.

> **Custody is not measuring real Vantor MPS performance.** Metrics are prototype proxy estimates produced by a hand-authored workflow model. Nothing here claims production savings, real planner adoption, or measured operational performance.

---

## Baseline / manual workflow

The baseline path is the unaided triage loop a human analyst + planner pair would walk without the Custody decision-support layer. It has four steps; all four are manual, and only the final step produces an audit artifact:

1. **Detector output review** — analyst inspects raw detector output without a structured ambiguity model.
2. **Analyst ambiguity triage** — analyst manually weighs competing explanations for the observed activity.
3. **Ad hoc collect discussion** — free-form discussion of which collection options to pursue, without ranked recommendations.
4. **Planner review** — planner records a decision in a free-text artifact.

This is a **proxy model of an unaided workflow**, not a calibrated description of any specific organisation's process.

---

## Custody-assisted workflow

The Custody-assisted path replaces three of the four manual steps with deterministic, auditable computations and pushes the human attention toward a single review step:

| Step | Module / slice | Manual? | Audit artifact? |
|---|---|---|---|
| 1. Hypothesis state update | `hypotheses/types.py`, `hypotheses/update.py`, `hypotheses/scenarios.py` (Slices 1-3) | no | yes |
| 2. Custody health assessment | `hypotheses/custody_health.py` (Slice 4) | no | yes |
| 3. Collection-value ranking | `hypotheses/collection_value.py` (Slice 5) | no | yes |
| 4. Mission-value proxy | `hypotheses/mission_value.py` (Slice 9) | no | yes |
| 5. Optimized plan | `hypotheses/optimizer.py` (Slice 11) | no | yes |
| 6. Human review queue | `hypotheses/planner_review.py` + `hypotheses/planner_queue.py` (Slices 13-14) | yes | yes |

Adjacent capabilities that feed the same loop without adding workflow steps:

- **Decision packet** (`hypotheses/decision_packet.py`, Slices 6-8) — composes the upstream outputs into a single text / JSON / Markdown artifact.
- **Counterfactual simulation** (`hypotheses/counterfactual.py`, Slice 10) — three-outcome decomposition under AMBIGUOUS custody; informs both the optimizer's planning utility and the queue's expected ambiguity resolution.
- **Heuristic policy evaluation** (`hypotheses/policy_eval.py`, Slice 12) — RL-ready evaluation substrate comparing six named strategies; surfaces in the optional `--target winning_policy` review path.
- **Workflow efficiency report** (`hypotheses/efficiency_metrics.py`, Slice 15) — composes the queue item into the comparison this document explains.

---

## Proxy metrics produced by Slice 15

`compute_efficiency_metrics` emits seven proxy metrics. None of these is a measured operational KPI; each is a deterministic comparison between the two workflow models above.

| Metric | Direction | What it measures |
|---|---|---|
| `manual_triage_steps` | lower is better | Count of steps requiring manual attention |
| `audit_artifact_count` | higher is better | Count of steps producing an auditable artifact |
| `expected_decision_cycle_minutes_proxy` | lower is better | Sum of hand-authored proxy minutes across steps |
| `ambiguity_focus_score` | higher is better | Whether the workflow surfaces a primary ambiguity pair + ranked candidates explicitly |
| `planner_attention_focus` | higher is better | Custody queue priority score; baseline has no analogue |
| `decision_traceability` | higher is better | Fraction of steps that produce an audit artifact |
| `collect_strategy_comparison_count` | higher is better | Number of candidate collect types compared (baseline = 1 ad-hoc path) |

The deltas in the report are always `custody_value − baseline_value`.

---

## What Slice 15 is *not*

- It is not a measurement of any real planning organisation's process.
- It is not a claim that any specific number of minutes will be saved in any specific deployment.
- It is not a financial model and does not assert any monetary value.
- It does not issue, command, schedule, or execute any collection.
- It does not represent calibrated planner adoption data.

The metrics are honest proxy comparisons of the two hand-authored workflow models. They are useful for explaining where the Custody decision-support layer reduces manual attention and increases auditability of the planning loop — and nothing more.

---

## Try it

```bash
python scripts/19_efficiency_metrics.py --scenario both
python scripts/19_efficiency_metrics.py --scenario tennent --format json
python scripts/19_efficiency_metrics.py --scenario whitsun --format md
```

Output is fully deterministic: identical inputs produce byte-identical results across runs.
