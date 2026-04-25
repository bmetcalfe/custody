"""Service functions for the local decision API (Slice 17).

Each function accepts a request dataclass, runs the deterministic
hypothesis-layer pipeline on synthetic scenario fixtures, and returns
a JSON-serializable :class:`ApiResponse` rendered as a plain dict.

This is a **local prototype service**.  It runs on synthetic fixtures
only (the same Tennent / Whitsun narratives used by scripts 13-20).
No real data ingestion, no execution authorization, no external
service calls.

Design constraints
------------------

- Deterministic: no wall-clock unless the caller omits ``generated_at``.
- Pure standard library: no new runtime dependencies.
- No imports from ``custody.detection.*``, ``custody.ingest.gfw_presence``,
  matcher runtime, Sentinel SDKs, or real-data loaders.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

import numpy as np

from custody.api.schemas import (
    ApiResponse,
    CollectRankingRequest,
    DecisionPacketRequest,
    OptimizePlanRequest,
    PlannerQueueRequest,
    PolicyEvaluationRequest,
    PortfolioAllocationRequest,
    _make_request_id,
    response_to_dict,
    validate_positive_float,
    validate_positive_int,
    validate_scenario_id,
    validate_scenario_ids,
)
from custody.provenance import (
    build_provenance_record,
    record_to_dict,
)
from custody.fusion.observations import PositionObservation
from custody.fusion.scenes import Scene
from custody.fusion.temporal import Match
from custody.hypotheses import (
    SCENARIO_TENNENT,
    SCENARIO_WHITSUN,
    assess_custody_health,
    build_decision_packet,
    packet_to_json_object,
    rank_collection_candidates,
    update_state,
)
from custody.hypotheses.counterfactual import simulate_counterfactual_collects
from custody.hypotheses.mission_value import (
    attribute_mission_value,
    mission_value_to_json_object,
)
from custody.hypotheses.optimizer import (
    PlanConstraint,
    optimize_collection_plan,
)
from custody.hypotheses.planner_queue import (
    build_planner_queue,
    create_queue_item,
)
from custody.hypotheses.policy_eval import evaluate_collection_policies
from custody.hypotheses.portfolio import (
    PortfolioConstraint,
    allocate_portfolio,
    build_portfolio_candidates,
    portfolio_report_to_json_object,
)
from custody.hypotheses.scenarios import (
    AisCoverage,
    tennent_evidence_from_match,
    tennent_evidence_from_scene,
    whitsun_evidence_from_match,
    whitsun_evidence_from_scene,
)
from custody.hypotheses.types import HypothesisEvidence, HypothesisState


# ---------------------------------------------------------------------------
# Default caveats (worded to honour the Slice 17 language guardrails)
# ---------------------------------------------------------------------------


_DEFAULT_CAVEATS: tuple[str, ...] = (
    "local prototype API only",
    "decision-support output only; no execution authorization is issued",
    "candidate collect types are prototype recommendations, not "
    "execution authorizations",
    "planning utility and mission value are prototype proxies, not "
    "financial estimates",
)


# ---------------------------------------------------------------------------
# Synthetic fixture builders (duplicated privately; see scripts 13-20)
# ---------------------------------------------------------------------------


_TENNENT_CENTER = (8.856, 114.665)
_WHITSUN_CENTER = (9.98, 114.63)


def _epoch(y: int, m: int, d: int, hh: int = 12, mm: int = 0, ss: int = 0) -> float:
    return datetime(y, m, d, hh, mm, ss, tzinfo=timezone.utc).timestamp()


def _make_obs(
    *, scene_id: str, obs_id: str, modality: str,
    acquisition_time: float, lat: float, lon: float,
) -> PositionObservation:
    return PositionObservation(
        obs_id=obs_id, source_id=scene_id, modality=modality,  # type: ignore[arg-type]
        acquisition_time=acquisition_time,
        ingestion_time=acquisition_time + 1.0,
        lat=lat, lon=lon,
        cov_pos=np.eye(2) * 100.0,
        raw_ref=f"synthetic://{scene_id}/{obs_id}",
    )


def _make_scene(
    *, case_study: str, yyyymmdd: str, sensor_suffix: str,
    acquisition_time: float, center_lat: float, center_lon: float,
    quality_flag: Literal["green", "yellow", "red"],
    obs_tuple: tuple[PositionObservation, ...] = (),
) -> Scene:
    scene_id = f"{case_study}_{yyyymmdd}_umbra-{sensor_suffix}"
    return Scene(
        scene_id=scene_id, sensor=f"umbra-{sensor_suffix}",
        acquisition_time=acquisition_time, pixel_size_m=0.25,
        center_lat=center_lat, center_lon=center_lon,
        footprint_latlon=(
            (center_lat - 0.02, center_lon - 0.02),
            (center_lat - 0.02, center_lon + 0.02),
            (center_lat + 0.02, center_lon + 0.02),
            (center_lat + 0.02, center_lon - 0.02),
        ),
        raw_scene_path=None, quality_flag=quality_flag,
        observations=obs_tuple,
    )


def _make_match(obs_a_id: str, obs_b_id: str, distance_m: float) -> Match:
    return Match(obs_a_id=obs_a_id, obs_b_id=obs_b_id, distance_m=distance_m)


@dataclass(frozen=True)
class _TennentStep:
    yyyymmdd: str
    sensor_suffix: str
    acquisition_time: float
    persistent: bool
    change: float
    quality: Literal["green", "yellow", "red"]
    matches: tuple[tuple[str, str, float], ...] = ()


@dataclass(frozen=True)
class _WhitsunStep:
    yyyymmdd: str
    sensor_suffix: str
    acquisition_time: float
    vessel_count: int
    ais_coverage: AisCoverage
    quality: Literal["green", "yellow", "red"]
    matches: tuple[tuple[str, str, bool], ...] = ()


def _tennent_narrative() -> tuple[_TennentStep, ...]:
    return (
        _TennentStep("20230702", "01", _epoch(2023, 7, 2, 14, 0, 55),
                     persistent=True, change=0.00, quality="yellow"),
        _TennentStep("20230723", "02", _epoch(2023, 7, 23, 14, 0, 0),
                     persistent=True, change=0.30, quality="green",
                     matches=(("t-obs-0702", "t-obs-0723", 5.0),)),
        _TennentStep("20230807", "03", _epoch(2023, 8, 7, 14, 0, 0),
                     persistent=True, change=0.35, quality="yellow",
                     matches=(("t-obs-0723", "t-obs-0807", 25.0),)),
        _TennentStep("20230809", "04", _epoch(2023, 8, 9, 14, 0, 0),
                     persistent=True, change=0.10, quality="yellow",
                     matches=(("t-obs-0807", "t-obs-0809", 8.0),)),
        _TennentStep("20230813", "05", _epoch(2023, 8, 13, 14, 0, 0),
                     persistent=True, change=0.40, quality="green",
                     matches=(("t-obs-0809", "t-obs-0813", 22.0),)),
    )


def _whitsun_narrative() -> tuple[_WhitsunStep, ...]:
    return (
        _WhitsunStep("20231206", "04", _epoch(2023, 12, 6, 1, 0, 0),
                     vessel_count=6, ais_coverage="sparse", quality="yellow"),
        _WhitsunStep("20231206", "07", _epoch(2023, 12, 6, 10, 0, 0),
                     vessel_count=5, ais_coverage="sparse", quality="yellow",
                     matches=(
                         ("w-obs-1206a", "w-obs-1206b", True),
                         ("w-obs-1206a2", "w-obs-1206b2", True),
                     )),
        _WhitsunStep("20240320", "05", _epoch(2024, 3, 20, 1, 0, 0),
                     vessel_count=2, ais_coverage="solid", quality="green"),
    )


def _build_tennent_state() -> HypothesisState:
    state: HypothesisState | None = None
    for step in _tennent_narrative():
        obs_id = f"t-obs-{step.yyyymmdd}"
        obs = _make_obs(
            scene_id=f"tennent_{step.yyyymmdd}_umbra-{step.sensor_suffix}",
            obs_id=obs_id, modality="SAR",
            acquisition_time=step.acquisition_time,
            lat=_TENNENT_CENTER[0], lon=_TENNENT_CENTER[1],
        )
        scene = _make_scene(
            case_study="tennent", yyyymmdd=step.yyyymmdd,
            sensor_suffix=step.sensor_suffix,
            acquisition_time=step.acquisition_time,
            center_lat=_TENNENT_CENTER[0], center_lon=_TENNENT_CENTER[1],
            quality_flag=step.quality, obs_tuple=(obs,),
        )
        scene_evs = tennent_evidence_from_scene(
            scene, persistent_scatterer_detected=step.persistent,
            change_signal_strength=step.change, quality_flag=step.quality,
        )
        match_evs: list[HypothesisEvidence] = []
        for obs_a, obs_b, disp in step.matches:
            match_evs.append(tennent_evidence_from_match(
                _make_match(obs_a, obs_b, disp),
                displacement_m=disp, quality_flag=step.quality,
            ))
        state = update_state(
            SCENARIO_TENNENT,
            tuple(scene_evs) + tuple(match_evs),
            prior_state=state,
        )
    assert state is not None
    return state


def _build_whitsun_state() -> HypothesisState:
    state: HypothesisState | None = None
    for step in _whitsun_narrative():
        obs_id = f"w-obs-{step.yyyymmdd}-{step.sensor_suffix}"
        obs = _make_obs(
            scene_id=f"whitsun_{step.yyyymmdd}_umbra-{step.sensor_suffix}",
            obs_id=obs_id, modality="SAR",
            acquisition_time=step.acquisition_time,
            lat=_WHITSUN_CENTER[0], lon=_WHITSUN_CENTER[1],
        )
        scene = _make_scene(
            case_study="whitsun", yyyymmdd=step.yyyymmdd,
            sensor_suffix=step.sensor_suffix,
            acquisition_time=step.acquisition_time,
            center_lat=_WHITSUN_CENTER[0], center_lon=_WHITSUN_CENTER[1],
            quality_flag=step.quality, obs_tuple=(obs,),
        )
        scene_evs = whitsun_evidence_from_scene(
            scene, vessel_count=step.vessel_count,
            ais_coverage_flag=step.ais_coverage, quality_flag=step.quality,
        )
        match_evs: list[HypothesisEvidence] = []
        for obs_a, obs_b, matched in step.matches:
            match_evs.append(whitsun_evidence_from_match(
                _make_match(obs_a, obs_b, 15.0),
                matched_cluster=matched, quality_flag=step.quality,
            ))
        state = update_state(
            SCENARIO_WHITSUN,
            tuple(scene_evs) + tuple(match_evs),
            prior_state=state,
        )
    assert state is not None
    return state


def _build_state(scenario_id: str) -> HypothesisState:
    if scenario_id == SCENARIO_TENNENT:
        return _build_tennent_state()
    if scenario_id == SCENARIO_WHITSUN:
        return _build_whitsun_state()
    raise ValueError(f"unknown scenario_id: {scenario_id!r}")


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------


def _generated_at_str(generated_at: datetime | None) -> str:
    dt = generated_at if generated_at is not None else datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _wrap(
    *,
    endpoint: str,
    request_dict: dict,
    payload: dict,
    generated_at: datetime | None,
    output_kind: str,
    scenario_ids: tuple[str, ...] = (),
) -> dict:
    resp = ApiResponse(
        request_id=_make_request_id(endpoint, request_dict),
        generated_at=_generated_at_str(generated_at),
        endpoint=endpoint,
        status="ok",
        payload=payload,
        caveats=_DEFAULT_CAVEATS,
    )
    out = response_to_dict(resp)
    prov_args = tuple(
        f"{k}={v}" for k, v in sorted(request_dict.items())
    )
    record = build_provenance_record(
        output_kind=output_kind,
        command=f"custody.api.service{endpoint}",
        args=prov_args,
        scenario_ids=scenario_ids,
        generated_at=generated_at,
    )
    out["provenance"] = record_to_dict(record)
    return out


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def health_check(*, generated_at: datetime | None = None) -> dict:
    """Simple health-check endpoint."""
    payload = {
        "service": "custody-decision-api",
        "version": "prototype",
        "scenarios_available": ["tennent", "whitsun"],
    }
    resp = ApiResponse(
        request_id="health",
        generated_at=_generated_at_str(generated_at),
        endpoint="/health",
        status="ok",
        payload=payload,
        caveats=_DEFAULT_CAVEATS,
    )
    out = response_to_dict(resp)
    record = build_provenance_record(
        output_kind="health-check",
        command="custody.api.service/health",
        generated_at=generated_at,
    )
    out["provenance"] = record_to_dict(record)
    return out


def build_decision_packet_response(
    request: DecisionPacketRequest,
    *,
    generated_at: datetime | None = None,
) -> dict:
    validate_scenario_id(request.scenario_id)
    state = _build_state(request.scenario_id)
    health = assess_custody_health(state)
    rec = rank_collection_candidates(state, health)
    packet = build_decision_packet(
        scenario_id=request.scenario_id,
        state=state,
        health=health,
        recommendation=rec,
    )
    payload = packet_to_json_object(packet)
    if request.include_mission_value:
        mv = attribute_mission_value(rec, health)
        payload["mission_value"] = mission_value_to_json_object(mv)

    return _wrap(
        endpoint="/decision-packet",
        request_dict={
            "scenario_id": request.scenario_id,
            "include_mission_value": request.include_mission_value,
        },
        payload=payload,
        generated_at=generated_at,
        output_kind="decision-packet",
        scenario_ids=(request.scenario_id,),
    )


def build_collect_ranking_response(
    request: CollectRankingRequest,
    *,
    generated_at: datetime | None = None,
) -> dict:
    validate_scenario_id(request.scenario_id)
    state = _build_state(request.scenario_id)
    health = assess_custody_health(state)
    rec = rank_collection_candidates(
        state, health, max_results=request.max_results,
    )
    payload = {
        "scenario_id": rec.scenario_id,
        "health_status": rec.health_status.value,
        "primary_ambiguity": (
            list(rec.primary_ambiguity)
            if rec.primary_ambiguity is not None else None
        ),
        "ranked_values": [
            {
                "candidate_id": cv.candidate.candidate_id,
                "label": cv.candidate.label,
                "score": cv.score,
                "disambiguates": (
                    list(cv.disambiguates)
                    if cv.disambiguates is not None else None
                ),
                "reason": cv.reason,
                "caveats": list(cv.caveats),
            }
            for cv in rec.ranked_values
        ],
        "summary": rec.summary,
    }
    return _wrap(
        endpoint="/rank-collects",
        request_dict={
            "scenario_id": request.scenario_id,
            "max_results": request.max_results,
        },
        payload=payload,
        generated_at=generated_at,
        output_kind="collect-ranking",
        scenario_ids=(request.scenario_id,),
    )


def build_optimize_plan_response(
    request: OptimizePlanRequest,
    *,
    generated_at: datetime | None = None,
) -> dict:
    validate_scenario_id(request.scenario_id)
    validate_positive_float("budget", request.budget)
    validate_positive_int("max_collects", request.max_collects)

    state = _build_state(request.scenario_id)
    health = assess_custody_health(state)
    rec = rank_collection_candidates(state, health)
    mv = attribute_mission_value(rec, health)
    cf = simulate_counterfactual_collects(state, health, rec)
    constraint = PlanConstraint(
        budget=request.budget, max_collects=request.max_collects,
    )
    opt = optimize_collection_plan(
        rec, mv, cf, scenario_id=request.scenario_id, constraint=constraint,
    )
    if request.strategy == "exhaustive":
        plan = opt.exhaustive_plan
    elif request.strategy == "greedy":
        plan = opt.greedy_plan
    else:
        plan = opt.recommended_plan

    payload = {
        "scenario_id": opt.scenario_id,
        "strategy": plan.strategy,
        "selected_items": [
            {
                "candidate_id": p.candidate_id,
                "label": p.label,
                "cost": p.cost,
                "value": p.value,
                "expected_ambiguity_resolution": p.expected_ambiguity_resolution,
                "expected_health_score_delta": p.expected_health_score_delta,
                "reason": p.reason,
            }
            for p in plan.selected_items
        ],
        "total_cost": plan.total_cost,
        "total_value": plan.total_value,
        "constraint": {
            "budget": constraint.budget,
            "max_collects": constraint.max_collects,
        },
        "summary": plan.summary,
        "caveats": list(plan.caveats),
        "comparison_summary": opt.comparison_summary,
    }
    return _wrap(
        endpoint="/optimize-plan",
        request_dict={
            "scenario_id": request.scenario_id,
            "budget": request.budget,
            "max_collects": request.max_collects,
            "strategy": request.strategy,
        },
        payload=payload,
        generated_at=generated_at,
        output_kind="optimize-plan",
        scenario_ids=(request.scenario_id,),
    )


def build_policy_evaluation_response(
    request: PolicyEvaluationRequest,
    *,
    generated_at: datetime | None = None,
) -> dict:
    validate_scenario_id(request.scenario_id)
    validate_positive_float("budget", request.budget)
    validate_positive_int("max_collects", request.max_collects)

    state = _build_state(request.scenario_id)
    health = assess_custody_health(state)
    rec = rank_collection_candidates(state, health)
    mv = attribute_mission_value(rec, health)
    cf = simulate_counterfactual_collects(state, health, rec)
    constraint = PlanConstraint(
        budget=request.budget, max_collects=request.max_collects,
    )
    opt = optimize_collection_plan(
        rec, mv, cf, scenario_id=request.scenario_id, constraint=constraint,
    )
    policy = evaluate_collection_policies(
        rec, mv, cf, opt,
        scenario_id=request.scenario_id,
        constraint=constraint,
        policies=request.policies,
    )

    payload = {
        "scenario_id": policy.scenario_id,
        "winning_policy_id": policy.winning_policy_id,
        "evaluations": [
            {
                "policy_id": e.policy_id,
                "label": e.label,
                "selected_candidate_ids": list(e.selected_candidate_ids),
                "total_cost": e.total_cost,
                "planning_utility": e.planning_utility,
                "rank": e.rank,
                "reason": e.reason,
                "caveats": list(e.caveats),
            }
            for e in policy.evaluations
        ],
        "summary": policy.summary,
    }
    return _wrap(
        endpoint="/evaluate-policies",
        request_dict={
            "scenario_id": request.scenario_id,
            "budget": request.budget,
            "max_collects": request.max_collects,
        },
        payload=payload,
        generated_at=generated_at,
        output_kind="policy-evaluation",
        scenario_ids=(request.scenario_id,),
    )


def build_planner_queue_response(
    request: PlannerQueueRequest,
    *,
    generated_at: datetime | None = None,
) -> dict:
    validate_scenario_ids(request.scenario_ids)

    items = []
    for sid in request.scenario_ids:
        state = _build_state(sid)
        health = assess_custody_health(state)
        rec = rank_collection_candidates(state, health)
        mv = attribute_mission_value(rec, health)
        cf = simulate_counterfactual_collects(state, health, rec)
        opt = optimize_collection_plan(rec, mv, cf, scenario_id=sid)
        policy = evaluate_collection_policies(
            rec, mv, cf, opt, scenario_id=sid,
        )
        items.append(create_queue_item(
            scenario_id=sid, health=health, recommendation=rec,
            mission_value_report=mv, counterfactual_report=cf,
            optimization_report=opt, policy_report=policy,
        ))

    queue_dt = (
        generated_at if generated_at is not None
        else datetime.now(timezone.utc)
    )
    queue = build_planner_queue(items, generated_at=queue_dt)

    payload = {
        "generated_at": queue.generated_at.isoformat(),
        "items": [
            {
                "item_id": qi.item_id,
                "scenario_id": qi.scenario_id,
                "status": qi.status.value,
                "priority_score": qi.priority_score,
                "health_status": qi.health_status.value,
                "recommended_candidate_ids": list(qi.recommended_candidate_ids),
                "next_action": qi.next_action,
                "reason": qi.reason,
                "caveats": list(qi.caveats),
            }
            for qi in queue.items
        ],
        "summary": queue.summary,
        "caveats": list(queue.caveats),
    }
    return _wrap(
        endpoint="/planner-queue",
        request_dict={"scenario_ids": list(request.scenario_ids)},
        payload=payload,
        generated_at=generated_at,
        output_kind="planner-queue",
        scenario_ids=tuple(request.scenario_ids),
    )


def build_portfolio_allocation_response(
    request: PortfolioAllocationRequest,
    *,
    generated_at: datetime | None = None,
) -> dict:
    validate_scenario_ids(request.scenario_ids)
    validate_positive_float("budget", request.budget)
    validate_positive_int("max_collects", request.max_collects)
    validate_positive_int(
        "max_collects_per_scenario", request.max_collects_per_scenario,
    )

    queue_items: list = []
    opt_reports: list = []
    mv_reports: list = []
    cf_reports: list = []
    for sid in request.scenario_ids:
        state = _build_state(sid)
        health = assess_custody_health(state)
        rec = rank_collection_candidates(state, health)
        mv = attribute_mission_value(rec, health)
        cf = simulate_counterfactual_collects(state, health, rec)
        opt = optimize_collection_plan(rec, mv, cf, scenario_id=sid)
        policy = evaluate_collection_policies(
            rec, mv, cf, opt, scenario_id=sid,
        )
        qi = create_queue_item(
            scenario_id=sid, health=health, recommendation=rec,
            mission_value_report=mv, counterfactual_report=cf,
            optimization_report=opt, policy_report=policy,
        )
        queue_items.append(qi)
        opt_reports.append(opt)
        mv_reports.append(mv)
        cf_reports.append(cf)

    candidates = build_portfolio_candidates(
        queue_items=tuple(queue_items),
        optimization_reports=tuple(opt_reports),
        mission_value_reports=tuple(mv_reports),
        counterfactual_reports=tuple(cf_reports),
    )
    constraint = PortfolioConstraint(
        budget=request.budget,
        max_collects=request.max_collects,
        max_collects_per_scenario=request.max_collects_per_scenario,
    )
    report = allocate_portfolio(candidates, constraint=constraint)
    payload = portfolio_report_to_json_object(report)

    return _wrap(
        endpoint="/portfolio-allocation",
        request_dict={
            "scenario_ids": list(request.scenario_ids),
            "budget": request.budget,
            "max_collects": request.max_collects,
            "max_collects_per_scenario": request.max_collects_per_scenario,
        },
        payload=payload,
        generated_at=generated_at,
        output_kind="portfolio-allocation",
        scenario_ids=tuple(request.scenario_ids),
    )
