"""Baseline planning-strategy comparison harness (ADR-0021 Slice 24).

Compares multiple planning strategies across scenarios using
deterministic simulation metrics: custody-health improvement, ambiguity
resolution, planning utility, cost, schedule feasibility, traceability,
and review burden.

This is a deterministic prototype comparison harness.  Every metric is
a proxy estimate based on the deterministic simulation stack; nothing
here measures planner performance, claims any monetary impact, or
asserts any operational outcome.

Design constraints
------------------

- Deterministic: no wall-clock, no randomness.
- Pure stdlib: no new runtime dependencies.
- No imports from ``custody.detection.*``, ``custody.ingest.gfw_presence``,
  matcher runtime, or external SDKs.
"""
from __future__ import annotations

import json as _json
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path

from custody.hypotheses.artifacts import (
    build_decision_from_artifacts,
    load_artifact_manifest,
)
from custody.hypotheses.availability_optimizer import (
    AvailabilityOptimizerConstraint,
    optimize_availability_adjusted_plan,
)
from custody.hypotheses.collection_value import (
    CollectionRecommendation,
    CollectionValue,
)
from custody.hypotheses.counterfactual import simulate_counterfactual_collects
from custody.hypotheses.custody_health import (
    CustodyHealthStatus,
    HypothesisCustodyHealth,
)
from custody.hypotheses.execution_sim import simulate_plan_execution
from custody.hypotheses.mission_value import (
    MissionValueReport,
    attribute_mission_value,
)
from custody.hypotheses.optimizer import (
    PlanConstraint,
    optimize_collection_plan,
)
from custody.hypotheses.scene_availability import (
    adjust_recommendation_for_availability,
    load_scene_availability_catalog,
)
from custody.hypotheses.scheduler import (
    load_collection_windows,
    schedule_collects,
)
from custody.hypotheses.types import HypothesisState


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class PlanningStrategyId(Enum):
    BASELINE_MANUAL = "baseline_manual"
    COLLECTION_VALUE_ONLY = "collection_value_only"
    MISSION_VALUE_OPTIMIZED = "mission_value_optimized"
    AVAILABILITY_ADJUSTED = "availability_adjusted"
    SCHEDULER_LITE = "scheduler_lite"
    EXECUTION_FEEDBACK = "execution_feedback"


@dataclass(frozen=True)
class StrategyEvaluation:
    strategy_id: str
    label: str
    scenario_id: str
    selected_candidate_ids: tuple[str, ...]
    total_cost: float
    planning_utility: float
    mission_value_proxy: float
    expected_ambiguity_resolution: float
    schedule_feasible_count: int
    schedule_unscheduled_count: int
    pre_health_status: CustodyHealthStatus
    post_health_status: CustodyHealthStatus | None
    pre_health_score: float
    post_health_score: float | None
    health_score_delta: float
    ambiguity_resolved: bool
    traceability_artifact_count: int
    review_actions_required: int
    rank: int
    reason: str
    caveats: tuple[str, ...]


@dataclass(frozen=True)
class StrategyComparisonReport:
    scenario_id: str
    evaluations: tuple[StrategyEvaluation, ...]
    winning_strategy_id: str
    summary: str
    caveats: tuple[str, ...]


@dataclass(frozen=True)
class CrossScenarioComparisonReport:
    scenario_ids: tuple[str, ...]
    scenario_reports: tuple[StrategyComparisonReport, ...]
    aggregate_evaluations: tuple[StrategyEvaluation, ...]
    winning_strategy_id: str
    summary: str
    caveats: tuple[str, ...]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


_BASE_CAVEATS: tuple[str, ...] = (
    "comparison uses deterministic prototype metrics",
    "baseline is a proxy, not measured manual performance",
    "no live tasking or sensor command is issued",
    "mission value and planning utility are proxies, not revenue",
    "execution feedback uses synthetic returned evidence",
)

_ALL_STRATEGY_IDS: tuple[str, ...] = tuple(s.value for s in PlanningStrategyId)

_STRATEGY_LABELS: dict[str, str] = {
    "baseline_manual": "Manual baseline proxy",
    "collection_value_only": "Collection-value only",
    "mission_value_optimized": "Mission-value optimized",
    "availability_adjusted": "Availability-adjusted",
    "scheduler_lite": "Scheduler-lite",
    "execution_feedback": "Execution feedback",
}

_TRACEABILITY_COUNTS: dict[str, int] = {
    "baseline_manual": 1,
    "collection_value_only": 2,
    "mission_value_optimized": 4,
    "availability_adjusted": 5,
    "scheduler_lite": 6,
    "execution_feedback": 7,
}

_REVIEW_ACTIONS: dict[str, int] = {
    "baseline_manual": 3,
    "collection_value_only": 2,
    "mission_value_optimized": 2,
    "availability_adjusted": 2,
    "scheduler_lite": 1,
    "execution_feedback": 1,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _round4(x: float) -> float:
    return round(float(x), 4)


def _filtered_collection_values(
    recommendation: CollectionRecommendation,
    *,
    budget: float,
    max_collects: int,
) -> tuple[CollectionValue, ...]:
    """Top-N collection values excluding wait_or_monitor and over-budget items."""
    selected: list[CollectionValue] = []
    spent = 0.0
    for cv in recommendation.ranked_values:
        if cv.candidate.candidate_id == "wait_or_monitor":
            continue
        if len(selected) >= max_collects:
            break
        cost = cv.candidate.relative_cost
        if spent + cost > budget + 1e-9:
            continue
        selected.append(cv)
        spent += cost
    return tuple(selected)


def _mission_value_for_ids(
    mv_report: MissionValueReport,
    candidate_ids: tuple[str, ...],
) -> float:
    by_id = {a.candidate_id: a for a in mv_report.ranked_assessments}
    total = 0.0
    for cid in candidate_ids:
        a = by_id.get(cid)
        if a is not None:
            total += a.total_value
    return _round4(total)


# ---------------------------------------------------------------------------
# Pipeline build helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PipelineContext:
    """Pre-computed upstream artifacts shared across strategies for one scenario."""
    scenario_id: str
    state: HypothesisState
    health: HypothesisCustodyHealth
    recommendation: CollectionRecommendation
    mission_value: MissionValueReport
    counterfactual: object  # CounterfactualReport
    base_optimization: object  # PlanOptimizationReport
    availability: object  # AvailabilityAdjustedRecommendation
    avail_report: object  # AvailabilityOptimizationReport
    schedule_report: object  # ScheduleReport
    outcome_policy: str


def _build_pipeline(
    *,
    scenario_id: str,
    outcome_policy: str,
    budget: float,
    max_collects: int,
    artifact_manifest_path: Path,
    availability_catalog_path: Path,
    collection_windows_path: Path,
) -> _PipelineContext:
    records = load_artifact_manifest(artifact_manifest_path)
    decision = build_decision_from_artifacts(records, scenario_id=scenario_id)
    mission_value = attribute_mission_value(
        decision.recommendation, decision.health,
    )
    counterfactual = simulate_counterfactual_collects(
        decision.final_state, decision.health, decision.recommendation,
    )
    base_optimization = optimize_collection_plan(
        decision.recommendation, mission_value, counterfactual,
        scenario_id=scenario_id,
        constraint=PlanConstraint(budget=budget, max_collects=max_collects),
    )
    catalog = load_scene_availability_catalog(
        availability_catalog_path, scenario_id=scenario_id,
    )
    availability = adjust_recommendation_for_availability(
        decision.recommendation, catalog,
    )
    avail_report = optimize_availability_adjusted_plan(
        base_optimization, availability,
        scenario_id=scenario_id,
        constraint=AvailabilityOptimizerConstraint(
            budget=budget, max_collects=max_collects,
        ),
    )
    windows = load_collection_windows(collection_windows_path)
    schedule_report = schedule_collects(
        avail_report, windows, scenario_id=scenario_id,
    )

    return _PipelineContext(
        scenario_id=scenario_id,
        state=decision.final_state,
        health=decision.health,
        recommendation=decision.recommendation,
        mission_value=mission_value,
        counterfactual=counterfactual,
        base_optimization=base_optimization,
        availability=availability,
        avail_report=avail_report,
        schedule_report=schedule_report,
        outcome_policy=outcome_policy,
    )


# ---------------------------------------------------------------------------
# Strategy evaluators
# ---------------------------------------------------------------------------


def _eval_baseline_manual(ctx: _PipelineContext) -> StrategyEvaluation:
    sid = "baseline_manual"
    return StrategyEvaluation(
        strategy_id=sid,
        label=_STRATEGY_LABELS[sid],
        scenario_id=ctx.scenario_id,
        selected_candidate_ids=(),
        total_cost=0.0,
        planning_utility=0.0,
        mission_value_proxy=0.0,
        expected_ambiguity_resolution=0.0,
        schedule_feasible_count=0,
        schedule_unscheduled_count=0,
        pre_health_status=ctx.health.status,
        post_health_status=None,
        pre_health_score=_round4(ctx.health.score),
        post_health_score=None,
        health_score_delta=0.0,
        ambiguity_resolved=False,
        traceability_artifact_count=_TRACEABILITY_COUNTS[sid],
        review_actions_required=_REVIEW_ACTIONS[sid],
        rank=0,
        reason=(
            "manual baseline proxy: ad hoc analyst review with no decision-stack "
            "support; no candidate selection, no scheduling"
        ),
        caveats=(
            "baseline is a deterministic proxy, not measured planner performance",
        ),
    )


def _eval_collection_value_only(
    ctx: _PipelineContext, *, budget: float, max_collects: int,
) -> StrategyEvaluation:
    sid = "collection_value_only"
    selected = _filtered_collection_values(
        ctx.recommendation, budget=budget, max_collects=max_collects,
    )
    candidate_ids = tuple(cv.candidate.candidate_id for cv in selected)
    total_cost = _round4(sum(cv.candidate.relative_cost for cv in selected))
    planning_utility = _round4(sum(cv.score for cv in selected))
    return StrategyEvaluation(
        strategy_id=sid,
        label=_STRATEGY_LABELS[sid],
        scenario_id=ctx.scenario_id,
        selected_candidate_ids=candidate_ids,
        total_cost=total_cost,
        planning_utility=planning_utility,
        mission_value_proxy=0.0,
        expected_ambiguity_resolution=0.0,
        schedule_feasible_count=0,
        schedule_unscheduled_count=0,
        pre_health_status=ctx.health.status,
        post_health_status=None,
        pre_health_score=_round4(ctx.health.score),
        post_health_score=None,
        health_score_delta=0.0,
        ambiguity_resolved=False,
        traceability_artifact_count=_TRACEABILITY_COUNTS[sid],
        review_actions_required=_REVIEW_ACTIONS[sid],
        rank=0,
        reason=(
            "selects top candidates from collection-value ranker only; no "
            "mission-value, counterfactual, or scheduling"
        ),
        caveats=(),
    )


def _eval_mission_value_optimized(ctx: _PipelineContext) -> StrategyEvaluation:
    sid = "mission_value_optimized"
    plan = ctx.base_optimization.recommended_plan
    candidate_ids = tuple(p.candidate_id for p in plan.selected_items)
    return StrategyEvaluation(
        strategy_id=sid,
        label=_STRATEGY_LABELS[sid],
        scenario_id=ctx.scenario_id,
        selected_candidate_ids=candidate_ids,
        total_cost=_round4(plan.total_cost),
        planning_utility=_round4(plan.total_value),
        mission_value_proxy=_mission_value_for_ids(
            ctx.mission_value, candidate_ids,
        ),
        expected_ambiguity_resolution=_round4(
            plan.total_expected_ambiguity_resolution,
        ),
        schedule_feasible_count=0,
        schedule_unscheduled_count=0,
        pre_health_status=ctx.health.status,
        post_health_status=None,
        pre_health_score=_round4(ctx.health.score),
        post_health_score=None,
        health_score_delta=0.0,
        ambiguity_resolved=False,
        traceability_artifact_count=_TRACEABILITY_COUNTS[sid],
        review_actions_required=_REVIEW_ACTIONS[sid],
        rank=0,
        reason=(
            "constrained optimizer using mission-value attribution and "
            "counterfactual ambiguity resolution"
        ),
        caveats=(),
    )


def _eval_availability_adjusted(ctx: _PipelineContext) -> StrategyEvaluation:
    sid = "availability_adjusted"
    plan = ctx.avail_report.recommended_plan
    candidate_ids = tuple(p.candidate_id for p in plan.selected_items)
    # Pull expected_ambiguity_resolution for selected ids from base optimization.
    base_items_by_id = {
        p.candidate_id: p for p in ctx.base_optimization.all_candidates
    }
    expected_amb = _round4(sum(
        base_items_by_id[cid].expected_ambiguity_resolution
        for cid in candidate_ids if cid in base_items_by_id
    ))
    return StrategyEvaluation(
        strategy_id=sid,
        label=_STRATEGY_LABELS[sid],
        scenario_id=ctx.scenario_id,
        selected_candidate_ids=candidate_ids,
        total_cost=_round4(plan.total_cost),
        planning_utility=_round4(plan.total_adjusted_utility),
        mission_value_proxy=_mission_value_for_ids(
            ctx.mission_value, candidate_ids,
        ),
        expected_ambiguity_resolution=expected_amb,
        schedule_feasible_count=0,
        schedule_unscheduled_count=0,
        pre_health_status=ctx.health.status,
        post_health_status=None,
        pre_health_score=_round4(ctx.health.score),
        post_health_score=None,
        health_score_delta=0.0,
        ambiguity_resolved=False,
        traceability_artifact_count=_TRACEABILITY_COUNTS[sid],
        review_actions_required=_REVIEW_ACTIONS[sid],
        rank=0,
        reason=(
            "constrained optimizer with metadata-derived feasibility "
            "adjustment over candidate collect types"
        ),
        caveats=(),
    )


def _eval_scheduler_lite(ctx: _PipelineContext) -> StrategyEvaluation:
    sid = "scheduler_lite"
    plan = ctx.avail_report.recommended_plan
    candidate_ids = tuple(p.candidate_id for p in plan.selected_items)
    base_items_by_id = {
        p.candidate_id: p for p in ctx.base_optimization.all_candidates
    }
    expected_amb = _round4(sum(
        base_items_by_id[cid].expected_ambiguity_resolution
        for cid in candidate_ids if cid in base_items_by_id
    ))
    schedule_plan = ctx.schedule_report.plan
    return StrategyEvaluation(
        strategy_id=sid,
        label=_STRATEGY_LABELS[sid],
        scenario_id=ctx.scenario_id,
        selected_candidate_ids=candidate_ids,
        total_cost=_round4(plan.total_cost),
        planning_utility=_round4(plan.total_adjusted_utility),
        mission_value_proxy=_mission_value_for_ids(
            ctx.mission_value, candidate_ids,
        ),
        expected_ambiguity_resolution=expected_amb,
        schedule_feasible_count=len(schedule_plan.scheduled_collects),
        schedule_unscheduled_count=len(schedule_plan.unscheduled_candidate_ids),
        pre_health_status=ctx.health.status,
        post_health_status=None,
        pre_health_score=_round4(ctx.health.score),
        post_health_score=None,
        health_score_delta=0.0,
        ambiguity_resolved=False,
        traceability_artifact_count=_TRACEABILITY_COUNTS[sid],
        review_actions_required=_REVIEW_ACTIONS[sid],
        rank=0,
        reason=(
            "availability-adjusted optimizer + window-feasibility simulation"
        ),
        caveats=(),
    )


def _eval_execution_feedback(ctx: _PipelineContext) -> StrategyEvaluation:
    sid = "execution_feedback"
    plan = ctx.avail_report.recommended_plan
    candidate_ids = tuple(p.candidate_id for p in plan.selected_items)
    base_items_by_id = {
        p.candidate_id: p for p in ctx.base_optimization.all_candidates
    }
    expected_amb = _round4(sum(
        base_items_by_id[cid].expected_ambiguity_resolution
        for cid in candidate_ids if cid in base_items_by_id
    ))
    schedule_plan = ctx.schedule_report.plan
    exec_report = simulate_plan_execution(
        ctx.state, schedule_plan,
        scenario_id=ctx.scenario_id, outcome_policy=ctx.outcome_policy,
    )
    return StrategyEvaluation(
        strategy_id=sid,
        label=_STRATEGY_LABELS[sid],
        scenario_id=ctx.scenario_id,
        selected_candidate_ids=candidate_ids,
        total_cost=_round4(plan.total_cost),
        planning_utility=_round4(plan.total_adjusted_utility),
        mission_value_proxy=_mission_value_for_ids(
            ctx.mission_value, candidate_ids,
        ),
        expected_ambiguity_resolution=expected_amb,
        schedule_feasible_count=len(schedule_plan.scheduled_collects),
        schedule_unscheduled_count=len(schedule_plan.unscheduled_candidate_ids),
        pre_health_status=exec_report.pre_health_status,
        post_health_status=exec_report.post_health_status,
        pre_health_score=_round4(exec_report.pre_health_score),
        post_health_score=_round4(exec_report.post_health_score),
        health_score_delta=_round4(exec_report.health_score_delta),
        ambiguity_resolved=exec_report.ambiguity_resolved,
        traceability_artifact_count=_TRACEABILITY_COUNTS[sid],
        review_actions_required=_REVIEW_ACTIONS[sid],
        rank=0,
        reason=(
            f"closed-loop simulation through synthetic returned evidence "
            f"under {ctx.outcome_policy} outcome policy"
        ),
        caveats=(),
    )


_STRATEGY_EVALUATORS = {
    "baseline_manual": _eval_baseline_manual,
    "collection_value_only": None,  # special: needs budget/max_collects
    "mission_value_optimized": _eval_mission_value_optimized,
    "availability_adjusted": _eval_availability_adjusted,
    "scheduler_lite": _eval_scheduler_lite,
    "execution_feedback": _eval_execution_feedback,
}


def _evaluate_strategy(
    strategy_id: str,
    ctx: _PipelineContext,
    *,
    budget: float,
    max_collects: int,
) -> StrategyEvaluation:
    if strategy_id == "collection_value_only":
        return _eval_collection_value_only(
            ctx, budget=budget, max_collects=max_collects,
        )
    fn = _STRATEGY_EVALUATORS.get(strategy_id)
    if fn is None:
        raise ValueError(
            f"unknown strategy_id: {strategy_id!r}; valid: {_ALL_STRATEGY_IDS}"
        )
    return fn(ctx)


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def _comparison_score(
    eval_: StrategyEvaluation,
    *,
    max_planning_utility: float,
    max_mission_value: float,
) -> float:
    pu_norm = (
        eval_.planning_utility / max_planning_utility
        if max_planning_utility > 0 else 0.0
    )
    mv_norm = (
        eval_.mission_value_proxy / max_mission_value
        if max_mission_value > 0 else 0.0
    )
    amb = _clamp01(eval_.expected_ambiguity_resolution)
    health_delta = max(0.0, eval_.health_score_delta)
    schedule_ratio = (
        eval_.schedule_feasible_count / max(1, len(eval_.selected_candidate_ids))
        if eval_.selected_candidate_ids else 0.0
    )
    schedule_ratio = _clamp01(schedule_ratio)
    trace = _clamp01(eval_.traceability_artifact_count / 7.0)

    return (
        0.25 * _clamp01(pu_norm)
        + 0.20 * _clamp01(mv_norm)
        + 0.20 * amb
        + 0.15 * _clamp01(health_delta)
        + 0.10 * schedule_ratio
        + 0.10 * trace
    )


def _rank_evaluations(
    evals: tuple[StrategyEvaluation, ...],
) -> tuple[tuple[StrategyEvaluation, ...], dict[str, float]]:
    """Re-rank evaluations and return (ranked, score_map)."""
    if not evals:
        return (), {}
    max_pu = max(e.planning_utility for e in evals)
    max_mv = max(e.mission_value_proxy for e in evals)
    score_map: dict[str, float] = {}
    for e in evals:
        score_map[e.strategy_id] = _comparison_score(
            e, max_planning_utility=max_pu, max_mission_value=max_mv,
        )

    def sort_key(e: StrategyEvaluation) -> tuple:
        return (
            -score_map[e.strategy_id],
            0 if e.ambiguity_resolved else 1,
            -e.health_score_delta,
            e.total_cost,
            e.strategy_id,
        )

    ordered = sorted(evals, key=sort_key)
    ranked = tuple(
        replace(e, rank=i + 1) for i, e in enumerate(ordered)
    )
    return ranked, score_map


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def _validate_strategies(strategies: tuple[str, ...] | None) -> tuple[str, ...]:
    if strategies is None:
        return _ALL_STRATEGY_IDS
    invalid = [s for s in strategies if s not in _ALL_STRATEGY_IDS]
    if invalid:
        raise ValueError(
            f"unknown strategy id(s): {invalid}; valid: {_ALL_STRATEGY_IDS}"
        )
    return tuple(strategies)


def compare_planning_strategies(
    *,
    scenario_id: str,
    outcome_policy: str = "favorable",
    budget: float = 1.5,
    max_collects: int = 3,
    strategies: tuple[str, ...] | None = None,
    artifact_manifest_path: Path,
    availability_catalog_path: Path,
    collection_windows_path: Path,
) -> StrategyComparisonReport:
    """Compare planning strategies for a single scenario."""
    selected_strategies = _validate_strategies(strategies)
    ctx = _build_pipeline(
        scenario_id=scenario_id,
        outcome_policy=outcome_policy,
        budget=budget,
        max_collects=max_collects,
        artifact_manifest_path=artifact_manifest_path,
        availability_catalog_path=availability_catalog_path,
        collection_windows_path=collection_windows_path,
    )
    evals: list[StrategyEvaluation] = []
    for sid in selected_strategies:
        evals.append(_evaluate_strategy(
            sid, ctx, budget=budget, max_collects=max_collects,
        ))

    ranked, score_map = _rank_evaluations(tuple(evals))
    winner = ranked[0].strategy_id if ranked else ""
    summary = (
        f"compared {len(ranked)} strategies for {scenario_id}; "
        f"winning strategy: {winner}"
    )
    return StrategyComparisonReport(
        scenario_id=scenario_id,
        evaluations=ranked,
        winning_strategy_id=winner,
        summary=summary,
        caveats=_BASE_CAVEATS,
    )


def _aggregate(
    per_scenario: tuple[StrategyComparisonReport, ...],
) -> tuple[StrategyEvaluation, ...]:
    """Average per-scenario metrics by strategy_id."""
    if not per_scenario:
        return ()
    # Index strategy_id -> list of evaluations across scenarios.
    by_strategy: dict[str, list[StrategyEvaluation]] = {}
    for report in per_scenario:
        for e in report.evaluations:
            by_strategy.setdefault(e.strategy_id, []).append(e)

    n = len(per_scenario)
    out: list[StrategyEvaluation] = []
    for sid, evs in by_strategy.items():
        if len(evs) != n:
            # Strategy did not run on every scenario; skip.
            continue
        avg_planning = _round4(sum(e.planning_utility for e in evs) / n)
        avg_mv = _round4(sum(e.mission_value_proxy for e in evs) / n)
        avg_amb = _round4(sum(e.expected_ambiguity_resolution for e in evs) / n)
        avg_cost = _round4(sum(e.total_cost for e in evs) / n)
        avg_delta = _round4(sum(e.health_score_delta for e in evs) / n)
        avg_pre_score = _round4(sum(e.pre_health_score for e in evs) / n)
        post_scores = [e.post_health_score for e in evs if e.post_health_score is not None]
        avg_post_score = (
            _round4(sum(post_scores) / len(post_scores)) if post_scores else None
        )
        sched_feasible = sum(e.schedule_feasible_count for e in evs)
        sched_unscheduled = sum(e.schedule_unscheduled_count for e in evs)
        candidate_id_set: list[str] = []
        for e in evs:
            for cid in e.selected_candidate_ids:
                if cid not in candidate_id_set:
                    candidate_id_set.append(cid)
        all_resolved = all(e.ambiguity_resolved for e in evs)
        pre_status = evs[0].pre_health_status
        post_status_set = {
            e.post_health_status for e in evs
            if e.post_health_status is not None
        }
        post_status: CustodyHealthStatus | None
        if not post_status_set:
            post_status = None
        elif len(post_status_set) == 1:
            post_status = next(iter(post_status_set))
        else:
            post_status = None  # mixed

        out.append(StrategyEvaluation(
            strategy_id=sid,
            label=_STRATEGY_LABELS.get(sid, sid),
            scenario_id="aggregate",
            selected_candidate_ids=tuple(candidate_id_set),
            total_cost=avg_cost,
            planning_utility=avg_planning,
            mission_value_proxy=avg_mv,
            expected_ambiguity_resolution=avg_amb,
            schedule_feasible_count=sched_feasible,
            schedule_unscheduled_count=sched_unscheduled,
            pre_health_status=pre_status,
            post_health_status=post_status,
            pre_health_score=avg_pre_score,
            post_health_score=avg_post_score,
            health_score_delta=avg_delta,
            ambiguity_resolved=all_resolved,
            traceability_artifact_count=evs[0].traceability_artifact_count,
            review_actions_required=evs[0].review_actions_required,
            rank=0,
            reason=f"aggregate of {n} scenarios for {sid}",
            caveats=(),
        ))
    return tuple(out)


def compare_planning_strategies_across_scenarios(
    *,
    scenario_ids: tuple[str, ...] = ("tennent", "whitsun"),
    outcome_policy: str = "favorable",
    budget: float = 1.5,
    max_collects: int = 3,
    strategies: tuple[str, ...] | None = None,
    artifact_manifest_paths: dict[str, Path],
    availability_catalog_paths: dict[str, Path],
    collection_windows_paths: dict[str, Path],
) -> CrossScenarioComparisonReport:
    """Compare planning strategies across multiple scenarios."""
    per_scenario: list[StrategyComparisonReport] = []
    for sid in scenario_ids:
        per_scenario.append(compare_planning_strategies(
            scenario_id=sid,
            outcome_policy=outcome_policy,
            budget=budget,
            max_collects=max_collects,
            strategies=strategies,
            artifact_manifest_path=artifact_manifest_paths[sid],
            availability_catalog_path=availability_catalog_paths[sid],
            collection_windows_path=collection_windows_paths[sid],
        ))

    aggregate = _aggregate(tuple(per_scenario))
    ranked_aggregate, _ = _rank_evaluations(aggregate)
    winner = ranked_aggregate[0].strategy_id if ranked_aggregate else ""
    summary = (
        f"aggregated {len(scenario_ids)} scenarios; "
        f"winning strategy: {winner}"
    )
    return CrossScenarioComparisonReport(
        scenario_ids=tuple(scenario_ids),
        scenario_reports=tuple(per_scenario),
        aggregate_evaluations=ranked_aggregate,
        winning_strategy_id=winner,
        summary=summary,
        caveats=_BASE_CAVEATS,
    )


# ---------------------------------------------------------------------------
# Serializers
# ---------------------------------------------------------------------------


def _eval_to_dict(e: StrategyEvaluation) -> dict:
    return {
        "strategy_id": e.strategy_id,
        "label": e.label,
        "scenario_id": e.scenario_id,
        "selected_candidate_ids": list(e.selected_candidate_ids),
        "total_cost": e.total_cost,
        "planning_utility": e.planning_utility,
        "mission_value_proxy": e.mission_value_proxy,
        "expected_ambiguity_resolution": e.expected_ambiguity_resolution,
        "schedule_feasible_count": e.schedule_feasible_count,
        "schedule_unscheduled_count": e.schedule_unscheduled_count,
        "pre_health_status": e.pre_health_status.value,
        "post_health_status": (
            e.post_health_status.value if e.post_health_status is not None else None
        ),
        "pre_health_score": e.pre_health_score,
        "post_health_score": e.post_health_score,
        "health_score_delta": e.health_score_delta,
        "ambiguity_resolved": e.ambiguity_resolved,
        "traceability_artifact_count": e.traceability_artifact_count,
        "review_actions_required": e.review_actions_required,
        "rank": e.rank,
        "reason": e.reason,
        "caveats": list(e.caveats),
    }


def _comparison_to_dict(r: StrategyComparisonReport) -> dict:
    return {
        "scenario_id": r.scenario_id,
        "evaluations": [_eval_to_dict(e) for e in r.evaluations],
        "winning_strategy_id": r.winning_strategy_id,
        "summary": r.summary,
        "caveats": list(r.caveats),
    }


def _cross_to_dict(r: CrossScenarioComparisonReport) -> dict:
    return {
        "scenario_ids": list(r.scenario_ids),
        "scenario_reports": [
            _comparison_to_dict(sr) for sr in r.scenario_reports
        ],
        "aggregate_evaluations": [
            _eval_to_dict(e) for e in r.aggregate_evaluations
        ],
        "winning_strategy_id": r.winning_strategy_id,
        "summary": r.summary,
        "caveats": list(r.caveats),
    }


def report_to_dict(
    report: StrategyComparisonReport | CrossScenarioComparisonReport,
) -> dict:
    if isinstance(report, CrossScenarioComparisonReport):
        return _cross_to_dict(report)
    return _comparison_to_dict(report)


def report_to_json(
    report: StrategyComparisonReport | CrossScenarioComparisonReport,
) -> str:
    return _json.dumps(report_to_dict(report), indent=2)


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def _hr(title: str) -> str:
    underline = "-" * max(len(title), 3)
    return f"\n{title}\n{underline}\n"


def _baseline_eval(evals: tuple[StrategyEvaluation, ...]) -> StrategyEvaluation | None:
    for e in evals:
        if e.strategy_id == "baseline_manual":
            return e
    return None


def _winning_eval(report: StrategyComparisonReport) -> StrategyEvaluation | None:
    for e in report.evaluations:
        if e.strategy_id == report.winning_strategy_id:
            return e
    return None


def _format_score_map(
    evals: tuple[StrategyEvaluation, ...],
) -> dict[str, float]:
    if not evals:
        return {}
    max_pu = max(e.planning_utility for e in evals)
    max_mv = max(e.mission_value_proxy for e in evals)
    return {
        e.strategy_id: round(_comparison_score(
            e, max_planning_utility=max_pu, max_mission_value=max_mv,
        ), 4)
        for e in evals
    }


def format_comparison_text(report: StrategyComparisonReport) -> str:
    parts: list[str] = []
    title = f"PLANNING STRATEGY COMPARISON - {report.scenario_id.upper()}"
    parts.append(title + "\n")
    parts.append("=" * len(title) + "\n")

    parts.append(_hr("Strategies evaluated"))
    for e in sorted(report.evaluations, key=lambda x: x.strategy_id):
        parts.append(f"  {e.strategy_id}\n")

    parts.append(_hr("Ranked strategy results"))
    score_map = _format_score_map(report.evaluations)
    for e in report.evaluations:
        amb = "yes" if e.ambiguity_resolved else "no"
        parts.append(
            f"  {e.rank}. {e.strategy_id:25s}  "
            f"score={score_map[e.strategy_id]:.2f}  "
            f"health_delta={e.health_score_delta:+.2f}  "
            f"ambiguity_resolved={amb}  cost={e.total_cost:.2f}\n"
        )

    parts.append(_hr("Winning strategy"))
    parts.append(f"  {report.winning_strategy_id}\n")

    baseline = _baseline_eval(report.evaluations)
    winner = _winning_eval(report)
    parts.append(_hr("Key deltas"))
    if baseline is not None and winner is not None:
        parts.append(
            f"  Best strategy improves custody health by "
            f"{winner.health_score_delta - baseline.health_score_delta:+.2f} "
            f"over baseline proxy.\n"
        )
        parts.append(
            f"  Traceability artifacts increase from "
            f"{baseline.traceability_artifact_count} to "
            f"{winner.traceability_artifact_count}.\n"
        )
        parts.append(
            f"  Review burden proxy decreases from "
            f"{baseline.review_actions_required} to "
            f"{winner.review_actions_required}.\n"
        )
    else:
        parts.append("  (baseline or winner not available)\n")

    parts.append(_hr("Caveats"))
    for c in report.caveats:
        parts.append(f"  - {c}\n")
    return "".join(parts)


def format_comparison_markdown(report: StrategyComparisonReport) -> str:
    lines: list[str] = []
    name = report.scenario_id.capitalize()
    lines.append(f"# Planning Strategy Comparison - {name}")
    lines.append("")
    lines.append("## Strategies evaluated")
    lines.append("")
    for e in sorted(report.evaluations, key=lambda x: x.strategy_id):
        lines.append(f"- `{e.strategy_id}`")
    lines.append("")

    lines.append("## Ranked strategy results")
    lines.append("")
    lines.append(
        "| rank | strategy | score | health delta | ambiguity resolved | cost |"
    )
    lines.append("|---|---|---|---|---|---|")
    score_map = _format_score_map(report.evaluations)
    for e in report.evaluations:
        amb = "yes" if e.ambiguity_resolved else "no"
        lines.append(
            f"| {e.rank} | `{e.strategy_id}` | "
            f"{score_map[e.strategy_id]:.2f} | "
            f"{e.health_score_delta:+.2f} | {amb} | {e.total_cost:.2f} |"
        )
    lines.append("")

    lines.append("## Winning strategy")
    lines.append("")
    lines.append(f"- `{report.winning_strategy_id}`")
    lines.append("")

    baseline = _baseline_eval(report.evaluations)
    winner = _winning_eval(report)
    lines.append("## Key deltas")
    lines.append("")
    if baseline is not None and winner is not None:
        lines.append(
            f"- Best strategy improves custody health by "
            f"{winner.health_score_delta - baseline.health_score_delta:+.2f} "
            f"over baseline proxy."
        )
        lines.append(
            f"- Traceability artifacts increase from "
            f"{baseline.traceability_artifact_count} to "
            f"{winner.traceability_artifact_count}."
        )
        lines.append(
            f"- Review burden proxy decreases from "
            f"{baseline.review_actions_required} to "
            f"{winner.review_actions_required}."
        )
    else:
        lines.append("- (baseline or winner not available)")
    lines.append("")

    lines.append("## Caveats")
    lines.append("")
    for c in report.caveats:
        lines.append(f"- {c}")
    lines.append("")
    return "\n".join(lines)


def format_cross_scenario_text(report: CrossScenarioComparisonReport) -> str:
    parts: list[str] = []
    for sub in report.scenario_reports:
        parts.append(format_comparison_text(sub))
        parts.append("\n")

    title = "CROSS-SCENARIO AGGREGATE - " + ", ".join(
        s.upper() for s in report.scenario_ids
    )
    parts.append(title + "\n")
    parts.append("=" * len(title) + "\n")

    parts.append(_hr("Aggregate ranked strategy results"))
    score_map = _format_score_map(report.aggregate_evaluations)
    for e in report.aggregate_evaluations:
        amb = "yes" if e.ambiguity_resolved else "no"
        parts.append(
            f"  {e.rank}. {e.strategy_id:25s}  "
            f"score={score_map[e.strategy_id]:.2f}  "
            f"avg_health_delta={e.health_score_delta:+.2f}  "
            f"all_scenarios_resolved={amb}  avg_cost={e.total_cost:.2f}\n"
        )

    parts.append(_hr("Aggregate winning strategy"))
    parts.append(f"  {report.winning_strategy_id}\n")

    parts.append(_hr("Caveats"))
    for c in report.caveats:
        parts.append(f"  - {c}\n")
    return "".join(parts)


def format_cross_scenario_markdown(report: CrossScenarioComparisonReport) -> str:
    lines: list[str] = []
    for sub in report.scenario_reports:
        lines.append(format_comparison_markdown(sub))
        lines.append("")

    name = ", ".join(s.capitalize() for s in report.scenario_ids)
    lines.append(f"# Cross-Scenario Aggregate - {name}")
    lines.append("")
    lines.append("## Aggregate ranked strategy results")
    lines.append("")
    lines.append(
        "| rank | strategy | score | avg health delta | "
        "all scenarios resolved | avg cost |"
    )
    lines.append("|---|---|---|---|---|---|")
    score_map = _format_score_map(report.aggregate_evaluations)
    for e in report.aggregate_evaluations:
        amb = "yes" if e.ambiguity_resolved else "no"
        lines.append(
            f"| {e.rank} | `{e.strategy_id}` | "
            f"{score_map[e.strategy_id]:.2f} | "
            f"{e.health_score_delta:+.2f} | {amb} | {e.total_cost:.2f} |"
        )
    lines.append("")
    lines.append("## Aggregate winning strategy")
    lines.append("")
    lines.append(f"- `{report.winning_strategy_id}`")
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    for c in report.caveats:
        lines.append(f"- {c}")
    lines.append("")
    return "\n".join(lines)
