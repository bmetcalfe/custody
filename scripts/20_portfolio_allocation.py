"""Portfolio-level collection allocation CLI (ADR-0021 Slice 16).

Composes the full hypothesis-layer pipeline (Slices 3-14) for the
requested scenarios, joins per-scenario plans / mission-value /
counterfactual outputs into a flat candidate pool, and runs the
Slice 16 portfolio optimizer under shared cross-scenario constraints.

This is a **prototype portfolio allocation engine**.  It does not issue
execution authorizations or command sensors.  Synthetic narrative
builders are duplicated privately from ``scripts/19_efficiency_metrics.py``
so the ten CLIs stay independently runnable.

Run::

    python scripts/20_portfolio_allocation.py
    python scripts/20_portfolio_allocation.py --scenario tennent
    python scripts/20_portfolio_allocation.py --scenario both --budget 1.0 --max-collects 2
    python scripts/20_portfolio_allocation.py --scenario both --strategy greedy --format json
"""
from __future__ import annotations

import argparse
import json as _json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal, TextIO

import numpy as np

from custody.fusion.observations import PositionObservation
from custody.fusion.scenes import Scene
from custody.fusion.temporal import Match
from custody.hypotheses import (
    SCENARIO_TENNENT,
    SCENARIO_WHITSUN,
    assess_custody_health,
    rank_collection_candidates,
    update_state,
)
from custody.hypotheses.counterfactual import simulate_counterfactual_collects
from custody.hypotheses.mission_value import attribute_mission_value
from custody.hypotheses.optimizer import optimize_collection_plan
from custody.hypotheses.planner_queue import create_queue_item
from custody.hypotheses.policy_eval import evaluate_collection_policies
from custody.hypotheses.portfolio import (
    PortfolioAllocationReport,
    PortfolioConstraint,
    allocate_portfolio,
    build_portfolio_candidates,
    format_portfolio_markdown,
    format_portfolio_text,
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
# Synthetic fixture builders (duplicated from scripts/19; private here).
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


# ---------------------------------------------------------------------------
# Per-scenario pipeline
# ---------------------------------------------------------------------------


def _compute_scenario(scenario_id: str):
    """Returns ``(queue_item, optimization_report, mission_value, counterfactual)``."""
    if scenario_id == SCENARIO_TENNENT:
        state = _build_tennent_state()
    elif scenario_id == SCENARIO_WHITSUN:
        state = _build_whitsun_state()
    else:
        raise ValueError(f"unknown scenario_id: {scenario_id!r}")
    health = assess_custody_health(state)
    rec = rank_collection_candidates(state, health)
    mv = attribute_mission_value(rec, health)
    cf = simulate_counterfactual_collects(state, health, rec)
    opt = optimize_collection_plan(rec, mv, cf, scenario_id=scenario_id)
    policy = evaluate_collection_policies(
        rec, mv, cf, opt, scenario_id=scenario_id,
    )
    qi = create_queue_item(
        scenario_id=scenario_id, health=health, recommendation=rec,
        mission_value_report=mv, counterfactual_report=cf,
        optimization_report=opt, policy_report=policy,
    )
    return qi, opt, mv, cf


def _override_strategy(
    report: PortfolioAllocationReport,
    *,
    strategy: str,
) -> PortfolioAllocationReport:
    """Build a new report whose ``recommended_plan`` is the chosen strategy."""
    if strategy == "exhaustive":
        chosen = report.exhaustive_plan
        comparison = (
            f"strategy override: showing exhaustive plan "
            f"(score {chosen.total_portfolio_score:.2f})"
        )
    elif strategy == "greedy":
        chosen = report.greedy_plan
        comparison = (
            f"strategy override: showing greedy plan "
            f"(score {chosen.total_portfolio_score:.2f})"
        )
    else:
        return report
    return PortfolioAllocationReport(
        scenario_ids=report.scenario_ids,
        constraint=report.constraint,
        candidate_pool=report.candidate_pool,
        exhaustive_plan=report.exhaustive_plan,
        greedy_plan=report.greedy_plan,
        recommended_plan=chosen,
        comparison_summary=comparison,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None, *, out: TextIO | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Portfolio-level collection allocation across scenarios. "
            "Decision-support output only; no execution authorization or "
            "platform scheduling is issued."
        ),
    )
    parser.add_argument(
        "--scenario", choices=("tennent", "whitsun", "both"), default="both",
    )
    parser.add_argument(
        "--format", choices=("text", "json", "md"), default="text",
    )
    parser.add_argument("--budget", type=float, default=1.5)
    parser.add_argument("--max-collects", dest="max_collects", type=int, default=3)
    parser.add_argument(
        "--max-collects-per-scenario",
        dest="max_collects_per_scenario", type=int, default=2,
    )
    parser.add_argument(
        "--strategy",
        choices=("recommended", "exhaustive", "greedy"), default="recommended",
    )
    parser.add_argument(
        "--include-closed-items",
        dest="include_closed_items", action="store_true",
    )
    args = parser.parse_args(argv)

    sink = out if out is not None else sys.stdout

    scenarios: tuple[str, ...] = (
        (SCENARIO_TENNENT,) if args.scenario == "tennent"
        else (SCENARIO_WHITSUN,) if args.scenario == "whitsun"
        else (SCENARIO_TENNENT, SCENARIO_WHITSUN)
    )
    computed = tuple(_compute_scenario(s) for s in scenarios)
    queue_items = tuple(c[0] for c in computed)
    opt_reports = tuple(c[1] for c in computed)
    mv_reports = tuple(c[2] for c in computed)
    cf_reports = tuple(c[3] for c in computed)

    candidates = build_portfolio_candidates(
        queue_items=queue_items,
        optimization_reports=opt_reports,
        mission_value_reports=mv_reports,
        counterfactual_reports=cf_reports,
    )

    constraint = PortfolioConstraint(
        budget=args.budget,
        max_collects=args.max_collects,
        max_collects_per_scenario=args.max_collects_per_scenario,
        include_closed_items=args.include_closed_items,
    )
    report = allocate_portfolio(candidates, constraint=constraint)
    report = _override_strategy(report, strategy=args.strategy)

    if args.format == "json":
        sink.write(_json.dumps(
            portfolio_report_to_json_object(report), indent=2,
        ) + "\n")
    elif args.format == "md":
        sink.write(format_portfolio_markdown(report))
    else:
        sink.write(format_portfolio_text(report))

    return 0


if __name__ == "__main__":
    sys.exit(main())
