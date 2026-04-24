"""Deterministic hypothesis-state timeline CLI (ADR-0021 Slice 3).

Builds synthetic Scene / PositionObservation / Match fixtures for the
Tennent and Whitsun case studies, generates scenario evidence via the
Slice 3 generators, updates the hypothesis state chronologically, and
prints a readable per-scene + final-state timeline.

This is a reasoning-layer demo — not a data-loader.  Nothing here reads
parquet, touches a network, or consults real detector output.  The
caller-supplied signal flags encode the scenario narrative.

Run::

    python scripts/12_hypothesis_timeline.py                 # both scenarios
    python scripts/12_hypothesis_timeline.py --scenario tennent
    python scripts/12_hypothesis_timeline.py --scenario whitsun
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal, Sequence, TextIO

import numpy as np

from custody.fusion.observations import PositionObservation
from custody.fusion.scenes import Scene
from custody.fusion.temporal import Match
from custody.hypotheses import (
    SCENARIO_TENNENT,
    SCENARIO_WHITSUN,
    format_state,
    update_state,
)
from custody.hypotheses.registry import get_hypotheses, get_hypothesis_ids
from custody.hypotheses.scenarios import (
    AisCoverage,
    tennent_evidence_from_match,
    tennent_evidence_from_scene,
    whitsun_evidence_from_match,
    whitsun_evidence_from_scene,
)
from custody.hypotheses.types import HypothesisEvidence, HypothesisState


# ---------------------------------------------------------------------------
# Synthetic fixture builders (private — not in src/)
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


# ---------------------------------------------------------------------------
# Scenario narratives (caller-supplied signals encode the story)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _TennentStep:
    yyyymmdd: str
    sensor_suffix: str
    acquisition_time: float
    persistent: bool
    change: float
    quality: Literal["green", "yellow", "red"]
    matches: tuple[tuple[str, str, float], ...] = ()  # (obs_a_id, obs_b_id, displacement_m)


@dataclass(frozen=True)
class _WhitsunStep:
    yyyymmdd: str
    sensor_suffix: str
    acquisition_time: float
    vessel_count: int
    ais_coverage: AisCoverage
    quality: Literal["green", "yellow", "red"]
    matches: tuple[tuple[str, str, bool], ...] = ()  # (obs_a_id, obs_b_id, matched_cluster)


def _tennent_narrative() -> tuple[_TennentStep, ...]:
    return (
        _TennentStep("20230702", "01", _epoch(2023, 7, 2, 14, 0, 55),
                     persistent=True,  change=0.00, quality="yellow"),
        _TennentStep("20230723", "02", _epoch(2023, 7, 23, 14, 0, 0),
                     persistent=True,  change=0.30, quality="green",
                     matches=(("t-obs-0702", "t-obs-0723", 5.0),)),
        _TennentStep("20230807", "03", _epoch(2023, 8, 7, 14, 0, 0),
                     persistent=True,  change=0.35, quality="yellow",
                     matches=(("t-obs-0723", "t-obs-0807", 25.0),)),
        _TennentStep("20230809", "04", _epoch(2023, 8, 9, 14, 0, 0),
                     persistent=True,  change=0.10, quality="yellow",
                     matches=(("t-obs-0807", "t-obs-0809", 8.0),)),
        _TennentStep("20230813", "05", _epoch(2023, 8, 13, 14, 0, 0),
                     persistent=True,  change=0.40, quality="green",
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
# Timeline printing
# ---------------------------------------------------------------------------


def _top_line(
    state: HypothesisState,
    prior_value: float,
) -> str:
    scores = list(state.scores.values())
    top_score = max(scores)
    if abs(top_score - prior_value) < 1e-6:
        return f"Top:      <no evidence yet>  uncertainty={state.uncertainty:.2f}"
    sorted_scores = sorted(scores, reverse=True)
    margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else sorted_scores[0]
    return (
        f"Top:      {state.top_hypothesis}   "
        f"score={top_score:.2f}   "
        f"uncertainty={state.uncertainty:.2f}   "
        f"margin={margin:.2f}"
    )


def _reason_line(
    state: HypothesisState,
    this_step_evidence: Sequence[HypothesisEvidence],
) -> str:
    """Pick a short dominant-reason line for this scene step.

    Prefers the first this-step evidence item whose supports contains the
    current top hypothesis; falls back to the first this-step evidence
    item; falls back to an em-dash when nothing fired this step.
    """
    if not this_step_evidence:
        return "Reason:   -"
    top = state.top_hypothesis
    for ev in this_step_evidence:
        if top is not None and top in ev.supports:
            return f"Reason:   {ev.reason}"
    return f"Reason:   {this_step_evidence[0].reason}"


def _run_tennent(out: TextIO) -> None:
    out.write("Scenario: tennent\n")
    prior_value = 1.0 / len(get_hypotheses(SCENARIO_TENNENT))
    state: HypothesisState | None = None
    prev_obs_id: str | None = None
    last_acq: float | None = None

    for step in _tennent_narrative():
        obs_id = f"t-obs-{step.yyyymmdd}"
        obs = _make_obs(
            scene_id=f"tennent_{step.yyyymmdd}_umbra-{step.sensor_suffix}",
            obs_id=obs_id,
            modality="SAR",
            acquisition_time=step.acquisition_time,
            lat=_TENNENT_CENTER[0],
            lon=_TENNENT_CENTER[1],
        )
        scene = _make_scene(
            case_study="tennent",
            yyyymmdd=step.yyyymmdd,
            sensor_suffix=step.sensor_suffix,
            acquisition_time=step.acquisition_time,
            center_lat=_TENNENT_CENTER[0],
            center_lon=_TENNENT_CENTER[1],
            quality_flag=step.quality,
            obs_tuple=(obs,),
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
                displacement_m=disp,
                quality_flag=step.quality,
            ))
        step_evs = tuple(scene_evs) + tuple(match_evs)
        state = update_state(SCENARIO_TENNENT, step_evs, prior_state=state)

        iso_date = datetime.fromtimestamp(step.acquisition_time, tz=timezone.utc).date().isoformat()
        out.write("\n")
        out.write(f"Scene:    {scene.scene_id}  ({iso_date})\n")
        out.write(_top_line(state, prior_value) + "\n")
        out.write(_reason_line(state, step_evs) + "\n")

        prev_obs_id = obs_id
        last_acq = step.acquisition_time

    out.write("\n=== Final state: tennent ===\n")
    out.write(format_state(state) + "\n")


def _run_whitsun(out: TextIO) -> None:
    out.write("Scenario: whitsun\n")
    prior_value = 1.0 / len(get_hypotheses(SCENARIO_WHITSUN))
    state: HypothesisState | None = None

    for step in _whitsun_narrative():
        obs_id = f"w-obs-{step.yyyymmdd}-{step.sensor_suffix}"
        obs = _make_obs(
            scene_id=f"whitsun_{step.yyyymmdd}_umbra-{step.sensor_suffix}",
            obs_id=obs_id,
            modality="SAR",
            acquisition_time=step.acquisition_time,
            lat=_WHITSUN_CENTER[0],
            lon=_WHITSUN_CENTER[1],
        )
        scene = _make_scene(
            case_study="whitsun",
            yyyymmdd=step.yyyymmdd,
            sensor_suffix=step.sensor_suffix,
            acquisition_time=step.acquisition_time,
            center_lat=_WHITSUN_CENTER[0],
            center_lon=_WHITSUN_CENTER[1],
            quality_flag=step.quality,
            obs_tuple=(obs,),
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
                matched_cluster=matched,
                quality_flag=step.quality,
            ))
        step_evs = tuple(scene_evs) + tuple(match_evs)
        state = update_state(SCENARIO_WHITSUN, step_evs, prior_state=state)

        iso_date = datetime.fromtimestamp(step.acquisition_time, tz=timezone.utc).date().isoformat()
        out.write("\n")
        out.write(f"Scene:    {scene.scene_id}  ({iso_date})\n")
        out.write(_top_line(state, prior_value) + "\n")
        out.write(_reason_line(state, step_evs) + "\n")

    out.write("\n=== Final state: whitsun ===\n")
    out.write(format_state(state) + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None, *, out: TextIO | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print a deterministic hypothesis-state timeline for "
                    "Tennent and/or Whitsun.",
    )
    parser.add_argument(
        "--scenario",
        choices=("tennent", "whitsun", "both"),
        default="both",
    )
    args = parser.parse_args(argv)

    sink = out if out is not None else sys.stdout

    if args.scenario in ("tennent", "both"):
        _run_tennent(sink)
        if args.scenario == "both":
            sink.write("\n")
    if args.scenario in ("whitsun", "both"):
        _run_whitsun(sink)

    return 0


if __name__ == "__main__":
    sys.exit(main())
