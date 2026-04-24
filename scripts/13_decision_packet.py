"""Custody decision-packet CLI (ADR-0021 Slices 6 + 8).

Composes the hypothesis layer (Slices 3-5) into one readable command:

    evidence -> hypothesis state -> custody health -> ambiguity ->
    candidate collect recommendation

This script is a thin argparse wrapper around
``custody.hypotheses.decision_packet``.  It owns only the synthetic
scenario-fixture construction (the same path as
``scripts/12_hypothesis_timeline.py``) and dispatch to text / json / md
formatters.  The packet dataclass and rendering logic live in the
library module so downstream consumers can import them directly.

Run::

    python scripts/13_decision_packet.py                                   # both, text
    python scripts/13_decision_packet.py --scenario tennent
    python scripts/13_decision_packet.py --scenario whitsun --format json
    python scripts/13_decision_packet.py --scenario both    --format md
"""
from __future__ import annotations

import argparse
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
import json as _json

from custody.hypotheses.decision_packet import (
    DecisionPacket,
    build_decision_packet,
    format_as_json,
    format_as_markdown,
    format_as_text,
    format_many_as_json,
    packet_to_json_object,
)
from custody.hypotheses.mission_value import (
    MissionValueReport,
    attribute_mission_value,
    format_mission_value_markdown,
    format_mission_value_text,
    mission_value_to_json_object,
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
# Synthetic fixture builders (mirror scripts/12; kept private to this script).
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


# ---------------------------------------------------------------------------
# State construction
# ---------------------------------------------------------------------------


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


def _compute_scenario(scenario_id: str):
    """Build the packet + the upstream health/recommendation.

    Returns ``(packet, health, recommendation)``.  Upstream objects are
    retained so ``--mission-value`` can call
    :func:`attribute_mission_value` without rebuilding state.
    """
    if scenario_id == SCENARIO_TENNENT:
        state = _build_tennent_state()
    elif scenario_id == SCENARIO_WHITSUN:
        state = _build_whitsun_state()
    else:  # defensive; argparse already restricts choices
        raise ValueError(f"unknown scenario_id: {scenario_id!r}")

    # as_of defaults to state.timestamp inside assess_custody_health, keeping
    # the packet deterministic.
    health = assess_custody_health(state)
    recommendation = rank_collection_candidates(state, health)
    packet = build_decision_packet(
        scenario_id=scenario_id,
        state=state,
        health=health,
        recommendation=recommendation,
    )
    return packet, health, recommendation


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None, *, out: TextIO | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Print a composed Custody decision packet in text, JSON, or "
            "Markdown for Tennent / Whitsun."
        ),
    )
    parser.add_argument(
        "--scenario",
        choices=("tennent", "whitsun", "both"),
        default="both",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json", "md"),
        default="text",
    )
    parser.add_argument(
        "--mission-value",
        action="store_true",
        dest="mission_value",
        help=(
            "Append a mission-value attribution proxy (not revenue, not a "
            "financial model) to each scenario's output."
        ),
    )
    args = parser.parse_args(argv)

    sink = out if out is not None else sys.stdout

    scenarios: tuple[str, ...] = (
        (SCENARIO_TENNENT,) if args.scenario == "tennent"
        else (SCENARIO_WHITSUN,) if args.scenario == "whitsun"
        else (SCENARIO_TENNENT, SCENARIO_WHITSUN)
    )

    computed = tuple(_compute_scenario(s) for s in scenarios)
    packets = tuple(c[0] for c in computed)
    mission_reports: tuple[MissionValueReport | None, ...] = tuple(
        attribute_mission_value(rec, health) if args.mission_value else None
        for (_p, health, rec) in computed
    )

    if args.format == "json":
        # --scenario both is a JSON array of packets (single valid document).
        # When --mission-value is on, each packet object carries a
        # top-level "mission_value" key; base schema stays closed otherwise.
        objs = []
        for packet, report in zip(packets, mission_reports):
            obj = packet_to_json_object(packet)
            if report is not None:
                obj["mission_value"] = mission_value_to_json_object(report)
            objs.append(obj)
        if len(objs) == 1:
            sink.write(_json.dumps(objs[0], indent=2) + "\n")
        else:
            sink.write(_json.dumps(objs, indent=2) + "\n")
    elif args.format == "md":
        for i, (p, report) in enumerate(zip(packets, mission_reports)):
            if i > 0:
                sink.write("\n")
            sink.write(format_as_markdown(p))
            if report is not None:
                sink.write("\n")
                sink.write(format_mission_value_markdown(report))
    else:  # text — byte-identical to Slice 6 when --mission-value is off
        for i, (p, report) in enumerate(zip(packets, mission_reports)):
            if i > 0:
                sink.write("\n")
            sink.write(format_as_text(p))
            if report is not None:
                sink.write(format_mission_value_text(report))

    return 0


if __name__ == "__main__":
    sys.exit(main())
