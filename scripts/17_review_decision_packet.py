"""Human-in-the-loop decision-packet review CLI (ADR-0021 Slice 13).

Builds the full hypothesis-layer pipeline (Slices 3-12) for a single
scenario and renders an auditable review record reflecting an operator
action (approve / reject / defer / override).

This CLI is a **review/audit interface only** — it does not execute any
collect, issue tasking orders, or command sensors.  Records can
optionally be appended to a JSONL ledger for persistent history.

Run::

    python scripts/17_review_decision_packet.py --scenario tennent --action approve --reason "..."
    python scripts/17_review_decision_packet.py --scenario whitsun --action reject --reason "..."
    python scripts/17_review_decision_packet.py --scenario tennent --action defer  --reason "..."
    python scripts/17_review_decision_packet.py --scenario whitsun --action override \\
        --override-candidate ais_coverage_query --reason "..."
    python scripts/17_review_decision_packet.py --scenario tennent --action approve \\
        --reason "..." --format json --ledger-path reviews.jsonl
"""
from __future__ import annotations

import argparse
import json as _json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
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
from custody.hypotheses.optimizer import (
    PlanConstraint,
    optimize_collection_plan,
)
from custody.hypotheses.planner_review import (
    ReviewAction,
    append_review_record_jsonl,
    create_planner_review_record,
    create_review_subject_from_optimized_plan,
    create_review_subject_from_policy_evaluation,
    format_review_markdown,
    format_review_text,
    record_to_dict,
    record_to_json,
)
from custody.hypotheses.policy_eval import evaluate_collection_policies
from custody.hypotheses.scenarios import (
    AisCoverage,
    tennent_evidence_from_match,
    tennent_evidence_from_scene,
    whitsun_evidence_from_match,
    whitsun_evidence_from_scene,
)
from custody.hypotheses.types import HypothesisEvidence, HypothesisState


# ---------------------------------------------------------------------------
# Synthetic fixture builders (duplicated from scripts/16; private here).
# ---------------------------------------------------------------------------


_TENNENT_CENTER = (8.856, 114.665)
_WHITSUN_CENTER = (9.98, 114.63)


def _epoch(y: int, m: int, d: int, hh: int = 12, mm: int = 0, ss: int = 0) -> float:
    return datetime(y, m, d, hh, mm, ss, tzinfo=timezone.utc).timestamp()


def _make_obs(
    *,
    scene_id: str,
    obs_id: str,
    modality: str,
    acquisition_time: float,
    lat: float,
    lon: float,
) -> PositionObservation:
    return PositionObservation(
        obs_id=obs_id,
        source_id=scene_id,
        modality=modality,  # type: ignore[arg-type]
        acquisition_time=acquisition_time,
        ingestion_time=acquisition_time + 1.0,
        lat=lat,
        lon=lon,
        cov_pos=np.eye(2) * 100.0,
        raw_ref=f"synthetic://{scene_id}/{obs_id}",
    )


def _make_scene(
    *,
    case_study: str,
    yyyymmdd: str,
    sensor_suffix: str,
    acquisition_time: float,
    center_lat: float,
    center_lon: float,
    quality_flag: Literal["green", "yellow", "red"],
    obs_tuple: tuple[PositionObservation, ...] = (),
) -> Scene:
    scene_id = f"{case_study}_{yyyymmdd}_umbra-{sensor_suffix}"
    return Scene(
        scene_id=scene_id,
        sensor=f"umbra-{sensor_suffix}",
        acquisition_time=acquisition_time,
        pixel_size_m=0.25,
        center_lat=center_lat,
        center_lon=center_lon,
        footprint_latlon=(
            (center_lat - 0.02, center_lon - 0.02),
            (center_lat - 0.02, center_lon + 0.02),
            (center_lat + 0.02, center_lon + 0.02),
            (center_lat + 0.02, center_lon - 0.02),
        ),
        raw_scene_path=None,
        quality_flag=quality_flag,
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
            scene,
            persistent_scatterer_detected=step.persistent,
            change_signal_strength=step.change,
            quality_flag=step.quality,
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
            scene,
            vessel_count=step.vessel_count,
            ais_coverage_flag=step.ais_coverage,
            quality_flag=step.quality,
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
# Per-scenario rendering
# ---------------------------------------------------------------------------


def _build_subject(args, scenario_id: str):
    if scenario_id == SCENARIO_TENNENT:
        state = _build_tennent_state()
    elif scenario_id == SCENARIO_WHITSUN:
        state = _build_whitsun_state()
    else:
        raise ValueError(f"unknown scenario_id: {scenario_id!r}")

    health = assess_custody_health(state)
    rec = rank_collection_candidates(state, health)
    mv = attribute_mission_value(rec, health)
    cf = simulate_counterfactual_collects(
        state, health, rec, mission_value_report=mv,
    )
    constraint = PlanConstraint(
        budget=args.budget, max_collects=args.max_collects,
    )
    opt = optimize_collection_plan(
        rec, mv, cf, scenario_id=scenario_id, constraint=constraint,
    )

    if args.target == "winning_policy":
        policy = evaluate_collection_policies(
            rec, mv, cf, opt, scenario_id=scenario_id, constraint=constraint,
        )
        subject = create_review_subject_from_policy_evaluation(
            policy, policy_id=args.policy_id,
        )
    else:
        subject = create_review_subject_from_optimized_plan(
            opt, strategy=args.strategy,
        )
    return subject


def _emit(out: TextIO, record, fmt: str) -> None:
    if fmt == "json":
        out.write(record_to_json(record))
    elif fmt == "md":
        out.write(format_review_markdown(record))
    else:
        out.write(format_review_text(record))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None, *, out: TextIO | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Human-in-the-loop review of a decision packet or policy "
            "evaluation.  Review record only; this CLI does not execute "
            "any collect or command any sensor."
        ),
    )
    parser.add_argument(
        "--scenario", choices=("tennent", "whitsun"), required=True,
    )
    parser.add_argument(
        "--action",
        choices=("approve", "reject", "defer", "override"),
        required=True,
    )
    parser.add_argument("--reason", required=True)
    parser.add_argument(
        "--target",
        choices=("optimized_plan", "winning_policy"),
        default="optimized_plan",
    )
    parser.add_argument(
        "--strategy",
        choices=("recommended", "exhaustive", "greedy"),
        default="recommended",
    )
    parser.add_argument("--policy-id", dest="policy_id", default=None)
    parser.add_argument(
        "--override-candidate",
        dest="override_candidates",
        action="append",
        default=[],
    )
    parser.add_argument("--operator-id", dest="operator_id", default=None)
    parser.add_argument(
        "--reviewed-at", dest="reviewed_at", default=None,
        help="ISO 8601 datetime; defaults to wall-clock UTC.",
    )
    parser.add_argument(
        "--budget", type=float, default=1.0,
    )
    parser.add_argument(
        "--max-collects", dest="max_collects", type=int, default=2,
    )
    parser.add_argument(
        "--format", choices=("text", "json", "md"), default="text",
    )
    parser.add_argument(
        "--ledger-path", dest="ledger_path", default=None,
        help="Optional path to a JSONL ledger; record is appended after display.",
    )
    args = parser.parse_args(argv)

    sink = out if out is not None else sys.stdout

    subject = _build_subject(args, args.scenario)
    record = create_planner_review_record(
        subject,
        action=args.action,
        operator_reason=args.reason,
        reviewed_at=args.reviewed_at,
        operator_id=args.operator_id,
        override_candidate_ids=tuple(args.override_candidates),
    )

    _emit(sink, record, args.format)

    if args.ledger_path is not None:
        result = append_review_record_jsonl(args.ledger_path, record)
        sink.write(
            f"\nReview record appended to {result.path} "
            f"(record_id: {result.record_id})\n"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
