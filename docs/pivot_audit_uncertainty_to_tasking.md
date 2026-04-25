# Pivot Audit: Uncertainty-to-Tasking Engine

**Date:** 2026-04-23  
**Related ADR:** `docs/decisions/0021-custody-as-uncertainty-to-tasking-engine.md`  
**Purpose:** Map existing Custody repo assets onto the new Evidence → Hypothesis → Custody Health → Collection Value frame.

## Summary

This pivot is not a rewrite. The repo already contains the lower-level foundations for observations, scenes, tracking, SAR/VLM detection, fusion, portfolio attention, prediction, and task recommendation.

The missing product layer is a hypothesis layer above those components. That layer should maintain competing explanations, expose custody health, and recommend collection actions based on expected uncertainty reduction.

## Preserved components

| File / ADR | Existing role | New role under pivot | Action |
|---|---|---|---|
| `docs/decisions/0008-polymorphic-observation-types.md` | Defines observation variants for position-only and position+velocity measurements | Measurement substrate for hypothesis evidence references | Preserve |
| `src/custody/fusion/observations.py` | `PositionObservation`, `PositionVelocityObservation`, `Observation` union | Source objects referenced by `HypothesisEvidence` annotations | Preserve |
| `src/custody/models.py` | `TrackState` with EKF/covariance and uncertainty shim | Low-level track uncertainty input to custody health when entity tracking applies | Preserve |
| `src/custody/fusion/tracker.py` | Associates observations into tracks using EKF/covariance | Produces track-level uncertainty and association context for hypothesis layer | Preserve |
| `docs/decisions/0018-heterogeneous-scene-reconciliation.md` | Makes SAR scenes first-class across heterogeneous acquisitions | Scene context and quality source for hypothesis evidence | Preserve |
| `src/custody/fusion/scenes.py` | `Scene` abstraction for SAR acquisition metadata | Referenced by evidence annotations; supports scene-to-scene hypothesis updates | Preserve |
| `src/custody/detection/quality.py` | Scene/detection quality calibration | Weight modifier for hypothesis evidence and custody health | Preserve |
| `docs/calibration/scene_quality_7scene_20260421.md` | Calibration note for scene quality | Documentation input for weighting evidence | Preserve |
| `src/custody/ingest/gfw_presence.py` | GFW/AIS presence ingest | Evidence source for AIS presence/absence; absence only meaningful with coverage context | Preserve |
| `docs/investigations/ais_coverage_investigation.md` | AIS/GFW limitations and coverage investigation | Guardrail against naive "no AIS = dark vessel" claims | Preserve |
| `src/custody/orchestration/portfolio.py` | Portfolio-level assessment and prioritization | Seed for later portfolio-level custody health and collection prioritization | Preserve |
| `src/custody/orchestration/attention.py` | Attention/watchlist style prioritization | Input concept for collection priority, not replacement for hypothesis layer | Preserve |
| `src/custody/taskrecommendation.py` | Builds ranked task recommendations with expected value | Existing tasking machinery to be extended with hypothesis-disambiguation value | Preserve |
| `src/custody/tasking_policy.py` | Converts vessel/record state into tasking tier | Lower-level policy layer; may consume custody health later | Preserve |
| `src/custody/prediction/` | Trajectory, risk forecast, zone probability | Useful for future moving-target scenarios; not central to Tennent static-activity case | Preserve |
| `tests/test_track_ekf.py` and fusion tests | Regression coverage for existing math/fusion | Must remain green during pivot | Preserve |
| `tests/test_taskrecommendation.py` / `tests/test_tasking_policy.py` | Regression coverage for tasking behavior | Must remain green; extend rather than rewrite | Preserve |

## Recontextualized components

| File / ADR | Old framing | New framing | Action |
|---|---|---|---|
| `docs/decisions/0015-detection-pivot-to-vlm-cfar-hybrid.md` | VLM/CFAR as detection pivot | Detection as candidate evidence generation | Recontextualize |
| `src/custody/detection/vlm_sar.py` | SAR VLM detection pipeline | Produces candidate evidence annotations, not truth | Recontextualize |
| `src/custody/detection/vlm_backends/` | Model backend machinery for VLM detection | Support layer for evidence generation and qualitative scene interpretation | Recontextualize |
| `src/custody/detection/sar_cfar.py` | Classical detector path | Alternative candidate evidence generator | Recontextualize |
| `src/custody/detection/annular_percentile.py` | Detection tuning approach | Evidence generator / diagnostic tool, not product centerpiece | Recontextualize |
| `day0/scratch/*vlm*` outputs | Detector debugging artifacts | Candidate evidence examples and failure cases | Recontextualize; do not make central |
| `day0/scratch/gfw_coverage_*` outputs | AIS/GFW endpoint exploration | Evidence coverage/absence context | Recontextualize as guardrail |
| `docs/investigations/detection-approaches.md` | Survey of detector options | Evidence-source comparison note | Recontextualize |
| `docs/positioning.md` | Existing detector/fusion-heavy narrative | Needs rewrite around uncertainty-to-tasking | Update after ADR lands |
| `README.md` | Current repo/project narrative | Needs short thesis rewrite: detector work is foundation, not destination | Update after Week 2 proof |
| `CLAUDE.md` | Project hot-cache / operating notes | Needs pivot summary and current scope once ADR lands | Update after ADR lands |

## Paused components

| File / ADR | Current state | Pause rule | Resume trigger |
|---|---|---|---|
| `docs/decisions/0019-geometry-aware-matcher-direction.md` | Directional matcher design | Keep as background direction, not active Week 1-2 work | Resume only if hypothesis timeline identifies cross-geometry association as the highest-value ambiguity reducer |
| `docs/decisions/0020-v2-signature-design-local-intensity.md` | Proposed V2 signature design | Pause at current state | Resume only with a concrete hypothesis-disambiguation test |
| `day0/scratch/signature_matcher_calibration.py` | Matcher calibration exploration | Pause | Resume only to answer a named ambiguity, e.g. structure vs transient vessel persistence |
| `day0/scratch/signature_matcher_calibration.md` | Matcher calibration notes | Pause | Use as reference if V2 resumes |
| `day0/scratch/sig_matcher_rejected_pairs_inspection.py` | Rejected-pair investigation | Pause | Use only if matcher-specific ambiguity emerges |
| `day0/scratch/sig_matcher_rejected_pairs_inspection.md` | Rejected-pair notes | Pause | Use only if matcher-specific ambiguity emerges |
| Untracked cross-geometry labeling sheet (`tests/fixtures/ground_truth/tennent_0702_0807.{md,csv}` + `chips/tennent_0702_0807/`) | V2 matcher support artifact | Leave untracked | Regenerate if V2 resumes with concrete test |
| Additional VLM prompt/model tuning | Detector improvement loop | Pause as primary work | Resume only if a specific hypothesis needs a better evidence extraction pass |
| Full Sentinel ingestion | New data source integration | Pause as dependency | Resume after collection-value engine proves Sentinel is high-value for a scenario |

## New components required

| New file | Purpose | Notes |
|---|---|---|
| `docs/decisions/0021-custody-as-uncertainty-to-tasking-engine.md` | ADR locking the pivot and scope | First Week 0 deliverable |
| `docs/pivot_audit_uncertainty_to_tasking.md` | Maps existing repo to new frame | This document |
| `src/custody/hypotheses/__init__.py` | Package marker and exports | Keep small |
| `src/custody/hypotheses/types.py` | `Hypothesis`, `HypothesisState`, status enums | Pure dataclasses/enums first |
| `src/custody/hypotheses/evidence.py` | Thin-wrapper `HypothesisEvidence` annotation type | References existing source objects by ID/ref, does not replace them |
| `src/custody/hypotheses/registry.py` | Scenario-specific hypothesis sets | Tennent and Whitsun must differ |
| `src/custody/hypotheses/update.py` | Deterministic belief update from evidence annotations | Weighted additive first version is acceptable |
| `src/custody/hypotheses/timeline.py` | Builds hypothesis state over ordered scenes/events | CLI-friendly output |
| `src/custody/hypotheses/custody_health.py` | Converts belief state to surfaced product metric | Distinguish healthy/degraded/ambiguous/stale/lost |
| `src/custody/hypotheses/collection_value.py` | Scores candidate collects by expected ambiguity reduction | Can score not-yet-ingested sensors as proposed collects |
| `src/custody/hypotheses/explain.py` | Human-readable reasons for state and recommendations | Critical for product/demo value |
| `scripts/12_hypothesis_timeline.py` | CLI report for Tennent / Whitsun timeline | Week 1 proof |
| `scripts/13_recommend_collect.py` | CLI report with custody health + next-best collect | Week 2 proof |
| `tests/test_hypothesis_types.py` | Type/validation tests | Keep deterministic |
| `tests/test_hypothesis_registry.py` | Scenario-specific hypothesis sets | Ensure Tennent/Whitsun are not generic copies |
| `tests/test_hypothesis_update.py` | Belief update tests | Explainability / scoring behavior |
| `tests/test_hypothesis_timeline.py` | Timeline state transition tests | Use tiny synthetic fixtures first |
| `tests/test_hypothesis_custody_health.py` | Custody health status tests | Ambiguous vs lost distinction matters |
| `tests/test_hypothesis_collection_value.py` | Collection value ranking tests | Ensure recommendations respond to top ambiguity |

## Implementation guardrails

1. Do not rename existing observation, scene, tracker, or tasking types in Week 1-2.
2. Do not create a heavyweight normalized evidence schema yet.
3. Do not require Sentinel ingestion before Sentinel can appear as a proposed collect.
4. Do not resume V2 matcher work without a named hypothesis ambiguity and a test.
5. Do not tune VLM as a general detector-improvement project.
6. Keep hypothesis scoring deterministic and explainable.
7. Keep Tennent and Whitsun scenario hypotheses separate.
8. Existing tests should stay green.

## Interview-safe language

Use this phrasing externally:

> The detector work exposed the real system problem: raw detections are not enough. I added a product layer that maintains competing hypotheses under uncertain sensor evidence and recommends the next collect that would reduce uncertainty the most.

Avoid this phrasing:

> I restarted the project.

Avoid overclaiming:

> The system performs full autonomous multi-sensor tasking on real Sentinel/Umbra/AIS data end to end.

Accurate stronger claim after Week 2:

> The prototype ingests existing SAR/VLM/AIS/scene-quality outputs as evidence, maintains scenario-specific hypotheses, surfaces custody health, and recommends next collection actions based on expected uncertainty reduction.

## Implementation status (as of 2026-04-24)

The "New components required" table above was the Week 0 plan. Slices 1–16 of that plan have landed. Each slice is one focused commit on `main`:

- **Slice 1** — hypothesis layer contract and scenario registries (types, registry, update, explain). Landed in `5c89f6c`.
- **Slice 2** — source-object evidence adapters (`from_observation`, `from_scene`, `from_match`, `from_mapping`). Landed in `1bcdaa4`.
- **Slice 3** — scenario evidence generators and per-scene timeline CLI (`scripts/12_hypothesis_timeline.py`). Landed in `4e48be2`.
- **Slice 4** — custody-health assessment (`CustodyHealthStatus`, `HypothesisCustodyHealth`, `assess_custody_health`, cascade LOST > STALE > AMBIGUOUS > HEALTHY > DEGRADED, canonical `ambiguity_pairs`). Landed in `f81a7b6`.
- **Slice 5** — collection-value ranking over hypothesis ambiguity (`rank_collection_candidates`, eight-pair strategy table, sensor-generic candidate types). Landed in `28aa188`.
- **Slice 6** — decision-packet CLI composing belief, custody health, primary ambiguity, and candidate-collect recommendations (`scripts/13_decision_packet.py`). Landed in `083d04b`.
- **Slice 7** — public-truth hardening: positioning.md rewritten around the new thesis; implementation guide banner-marked as historical; `pyproject.toml` description updated. Landed in `05f6959`.
- **Slice 8** — decision-packet JSON and Markdown export (`DecisionPacket` dataclass, `build_decision_packet`, `format_as_text` / `format_as_json` / `format_as_markdown`, `--format` flag; closed JSON schema `"1"`, deterministic `generated_at` from `state.timestamp`). Landed in `4d3bbed`.
- **Slice 9** — mission-value attribution proxy (`attribute_mission_value`, seven-component decomposition, opt-in `--mission-value` flag preserving Slice 8 byte-identical baselines). Landed in `b6e853f`.
- **Slice 10** — counterfactual collect simulation (`simulate_counterfactual_collects`, three-outcome decomposition under AMBIGUOUS custody, deterministic heuristic outcome weights, `scripts/14_counterfactual_collects.py`). Landed in `ac1d523`.
- **Slice 11** — constrained collection-plan optimizer (`optimize_collection_plan`, exhaustive + greedy baselines, `PlanConstraint` over budget / max-collects / required / excluded, `scripts/15_optimize_collect_plan.py`). Landed in `0cc5552`.
- **Slice 12** — heuristic collection-policy evaluator (`evaluate_collection_policies`, six named deterministic policies, RL-ready evaluation substrate, `scripts/16_evaluate_collect_policies.py`). Landed in `f386b58`.
- **Slice 13** — human-in-the-loop review ledger (`ReviewAction`, `PlannerReviewRecord`, deterministic `review_id` / `packet_hash`, JSONL append, `scripts/17_review_decision_packet.py`). Landed in `934a619`.
- **Slice 14** — planner work queue (`PlannerQueueStatus`, `PlannerQueueItem`, `PlannerQueue`, priority formula combining custody health / planning utility / ambiguity resolution / mission-value proxy / review modifier, JSONL ledger ingestion, `scripts/18_planner_queue.py`). Landed in `114ec11`.
- **Slice 15** — workflow efficiency proxy metrics (`WorkflowMode`, `EfficiencyReport`, seven deterministic proxy metrics comparing baseline / manual workflow vs Custody-assisted workflow, `scripts/19_efficiency_metrics.py`, `docs/mps_complexity_map.md`). Landed in `d15c00f`.
- **Slice 16** — portfolio-level collection allocation (`PortfolioConstraint`, `PortfolioCandidate`, `PortfolioAllocationReport`, exhaustive + greedy cross-scenario optimization under budget / max-collect / per-scenario constraints, `scripts/20_portfolio_allocation.py`). Pending commit hash on this slice.

The **Preserved**, **Recontextualized**, and **Paused** tables at the top of this document remain accurate; no component has moved between those categories. The V2 matcher (ADR-0019 / ADR-0020) stays paused unless hypothesis-layer ambiguity demands it.

Not yet shipped, still on the roadmap per README:

- Real-data wiring from processed SAR/AIS artifacts into the scenario generators
- Sentinel-1 / Sentinel-2 evidence integration
- Portfolio-level prioritization across multiple regions or targets
- Live tasking integration, platform-specific sensor access/scheduling
- Real-time ingestion or production deployment
