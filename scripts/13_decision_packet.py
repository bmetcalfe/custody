"""Custody decision-packet CLI (ADR-0021 Slice 6).

Composes the hypothesis layer (Slices 3-5) into one readable command:

    evidence -> hypothesis state -> custody health -> ambiguity ->
    candidate collect recommendation

This script is presentation only.  It adds no new domain types, no new
scoring rules, no new strategy tables, and no capability claims.  The
synthetic scenario narratives are duplicated privately from
``scripts/12_hypothesis_timeline.py`` so the two scripts stay
independently runnable without cross-script imports.

Run::

    python scripts/13_decision_packet.py                     # both scenarios
    python scripts/13_decision_packet.py --scenario tennent
    python scripts/13_decision_packet.py --scenario whitsun
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
    CollectionRecommendation,
    CustodyHealthStatus,
    HypothesisCustodyHealth,
    HypothesisState,
    assess_custody_health,
    rank_collection_candidates,
    update_state,
)
from custody.hypotheses.scenarios import (
    AisCoverage,
    tennent_evidence_from_match,
    tennent_evidence_from_scene,
    whitsun_evidence_from_match,
    whitsun_evidence_from_scene,
)
from custody.hypotheses.types import HypothesisEvidence


# ---------------------------------------------------------------------------
# Synthetic fixture builders (mirror scripts/12; kept private to this script
# per Slice 6 dispatch guidance).
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
# State construction — replay the scenario narrative and return the final
# state + evidence stream.  No new scoring; update_state is imported.
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


# ---------------------------------------------------------------------------
# Editorial helper — scenario-conditional 'what not to do yet' bullets.
# ---------------------------------------------------------------------------


_VLM_TUNING_CAUTION = (
    "Additional VLM tuning has low expected value - current uncertainty is "
    "hypothesis-level, not detector-confidence-level."
)
_WHITSUN_AIS_CAVEAT = (
    "AIS absence alone is not proof of dark vessel activity; only meaningful "
    "when coverage is known."
)
_SENTINEL_ROADMAP = (
    "Sentinel-1/2 ingestion is roadmap, not part of this demo."
)


def _what_not_to_do(
    scenario_id: str,
    health: HypothesisCustodyHealth,
    recommendations: CollectionRecommendation,
) -> tuple[str, ...]:
    """Scenario-conditional editorial.  Capped at 3 bullets per the dispatch."""
    bullets: list[str] = []
    if health.status is CustodyHealthStatus.AMBIGUOUS:
        bullets.append(_VLM_TUNING_CAUTION)
    if scenario_id == SCENARIO_WHITSUN:
        bullets.append(_WHITSUN_AIS_CAVEAT)
    bullets.append(_SENTINEL_ROADMAP)
    return tuple(bullets[:3])


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _hr(title: str) -> str:
    underline = "-" * max(len(title), 3)
    return f"\n{title}\n{underline}\n"


def _top_three_belief_lines(state: HypothesisState) -> list[str]:
    """Sorted by score descending, ties broken by hypothesis_id ascending."""
    items = sorted(state.scores.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
    return [f"  {score:4.2f}  {hid}" for hid, score in items]


def _render_packet(
    out: TextIO,
    *,
    scenario_id: str,
    state: HypothesisState,
    health: HypothesisCustodyHealth,
    recommendations: CollectionRecommendation,
) -> None:
    out.write(f"CUSTODY DECISION PACKET - {scenario_id.upper()}\n")
    out.write("=" * (len("CUSTODY DECISION PACKET - ") + len(scenario_id)) + "\n")

    out.write(_hr("Current belief"))
    out.write("Top 3 hypotheses:\n")
    for line in _top_three_belief_lines(state):
        out.write(line + "\n")

    out.write(_hr("Custody health"))
    out.write(f"Status: {health.status.name} (score {health.score:.2f})\n")
    out.write(f"Reason: {health.reason}\n")
    if health.drivers:
        out.write("Drivers:\n")
        for d in health.drivers[:3]:
            out.write(f"  - {d}\n")

    out.write(_hr("Primary ambiguity"))
    if recommendations.primary_ambiguity is not None:
        a, b = recommendations.primary_ambiguity
        out.write(f"{a}  vs  {b}\n")
    else:
        out.write("None surfaced by custody-health assessment\n")

    out.write(_hr("Recommended candidate collects"))
    out.write("Top 3 candidate collects:\n")
    for v in recommendations.ranked_values[:3]:
        out.write(f"  {v.score:4.2f}  {v.candidate.candidate_id:22s}  {v.candidate.label}\n")
        out.write(f"        Reason: {v.reason}\n")
        if v.caveats:
            for c in v.caveats:
                out.write(f"        Caveat: {c}\n")

    out.write(_hr("Why these collects"))
    out.write(recommendations.summary + "\n")
    if recommendations.primary_ambiguity is not None and recommendations.ranked_values:
        lead = recommendations.ranked_values[0]
        out.write(
            f"Leading candidate {lead.candidate.candidate_id!r} addresses this ambiguity "
            f"because {lead.reason}\n"
        )

    out.write(_hr("What not to do yet"))
    for bullet in _what_not_to_do(scenario_id, health, recommendations):
        out.write(f"- {bullet}\n")


def _run_scenario(out: TextIO, scenario_id: str) -> None:
    if scenario_id == SCENARIO_TENNENT:
        state = _build_tennent_state()
    elif scenario_id == SCENARIO_WHITSUN:
        state = _build_whitsun_state()
    else:  # defensive; argparse already restricts choices
        raise ValueError(f"unknown scenario_id: {scenario_id!r}")

    # as_of defaults to state.timestamp inside assess_custody_health, keeping
    # the packet deterministic.
    health = assess_custody_health(state)
    recommendations = rank_collection_candidates(state, health)
    _render_packet(
        out,
        scenario_id=scenario_id,
        state=state,
        health=health,
        recommendations=recommendations,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None, *, out: TextIO | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print a composed Custody decision packet for Tennent / Whitsun.",
    )
    parser.add_argument(
        "--scenario",
        choices=("tennent", "whitsun", "both"),
        default="both",
    )
    args = parser.parse_args(argv)

    sink = out if out is not None else sys.stdout

    if args.scenario in ("tennent", "both"):
        _run_scenario(sink, SCENARIO_TENNENT)
        if args.scenario == "both":
            sink.write("\n")
    if args.scenario in ("whitsun", "both"):
        _run_scenario(sink, SCENARIO_WHITSUN)

    return 0


if __name__ == "__main__":
    sys.exit(main())
