"""
Multi-target simulation timeline engine.

run_multi_target_simulation(scenario) advances all vessels in the scenario
through time using the same three-phase loop as simulate.py:

  Phase 1 — Advance all vessel states using BehaviorProfile
  Phase 2 — Per-vessel sensor-access filtering and vessel ranking
  Phase 3 — Sensor assignment (arbitration) and record emission

Output record schema is identical to simulate.run_simulation() so all
downstream pipeline code (fusion, decision, UI) works unchanged.
"""
from __future__ import annotations

import math
import random
from datetime import timedelta
from typing import Optional

from custody.models import Vessel, TrackState
from custody.tracks import update_position, update_uncertainty
from custody.anomalies import anomaly_score, anomaly_breakdown
from custody.planner import plan_collection, compute_target_priority
from custody.behavior.state_machine import infer_state
from custody.compounds import evaluate_compounds
from custody.decision_trace import build_decision_trace
from custody.sensors import get_sensor_opportunities

from custody.simulation.profiles import BehaviorProfile, PROFILE_TO_MODE
from custody.simulation.scenarios import ScenarioConfig, VesselSpec, ProfilePhase, DEFAULT_SCENARIO
from custody.simulation.generator import build_vessel_list


def _current_phase(spec: VesselSpec, hour_index: int) -> ProfilePhase:
    """Return the active phase for the given simulation hour."""
    active = spec.phases[0]
    for phase in spec.phases:
        if phase.start_hour <= hour_index:
            active = phase
    return active


def _apply_profile(
    vessel: Vessel,
    phase: ProfilePhase,
    rng: random.Random,
) -> Vessel:
    """Update vessel speed and heading according to the active phase.

    If phase.target_lat/lon are set, vessel steers toward that point.
    If phase.heading_override is set, heading is fixed to that value.
    Otherwise, heading drifts by a Gaussian sample from the profile.
    """
    profile = phase.profile

    # Speed: base + small random variance
    variance = profile.speed_variance_kmh * rng.uniform(-1.0, 1.0)
    vessel.speed_kmh = max(0.5, profile.base_speed_kmh + variance)

    if phase.heading_override is not None:
        vessel.heading_deg = phase.heading_override % 360.0

    elif phase.target_lat is not None and phase.target_lon is not None:
        # Steer toward target: compute bearing each step
        dlat = phase.target_lat - vessel.lat
        dlon = phase.target_lon - vessel.lon
        bearing = math.degrees(math.atan2(dlon, dlat)) % 360.0
        # Blend: 80% toward target, 20% current heading (avoids perfectly straight lines)
        vessel.heading_deg = (0.8 * bearing + 0.2 * vessel.heading_deg) % 360.0

    else:
        # Random heading drift
        drift = rng.gauss(0.0, profile.heading_drift_deg)
        vessel.heading_deg = (vessel.heading_deg + drift) % 360.0

    return vessel


def run_multi_target_simulation(scenario: Optional[ScenarioConfig] = None) -> list[dict]:
    """Run the multi-target simulation and return a flat list of timeline records.

    Uses the same record schema as custody.simulate.run_simulation() so all
    downstream consumers work unchanged.

    Args:
        scenario: ScenarioConfig to run.  Defaults to DEFAULT_SCENARIO.

    Returns:
        List of record dicts, one per (vessel, timestep), interleaved by time.
    """
    if scenario is None:
        scenario = DEFAULT_SCENARIO

    rng = random.Random(scenario.seed)

    # Build ordered vessel list (scripted + background)
    vessel_specs = build_vessel_list(scenario, rng)

    # Initialise state for each vessel
    vessels: dict[str, Vessel] = {}
    tracks: dict[str, TrackState] = {}
    timelines: dict[str, list[dict]] = {}
    spec_by_id: dict[str, VesselSpec] = {}

    for spec in vessel_specs:
        vid = spec.vessel_id
        vessels[vid] = Vessel(
            id=vid,
            lat=spec.start_lat,
            lon=spec.start_lon,
            speed_kmh=spec.phases[0].profile.base_speed_kmh,
            heading_deg=spec.start_heading_deg,
            last_seen=scenario.start_time,
        )
        tracks[vid] = TrackState()
        timelines[vid] = []
        spec_by_id[vid] = spec

    # ── Main simulation loop ─────────────────────────────────────────────────
    current_time = scenario.start_time
    end_time = scenario.start_time + timedelta(hours=scenario.duration_hours)

    while current_time <= end_time:
        hour_index = int(
            (current_time - scenario.start_time).total_seconds() // 3600
        )

        # ── Phase 1: advance all vessel states ───────────────────────────────
        vessel_states: dict[str, dict] = {}

        for spec in vessel_specs:
            vid = spec.vessel_id
            vessel = vessels[vid]
            track = tracks[vid]

            phase = _current_phase(spec, hour_index)
            vessel = _apply_profile(vessel, phase, rng)
            vessel = update_position(vessel, hours=scenario.dt_hours)
            vessel.last_seen = current_time
            track.uncertainty_km = update_uncertainty(
                track.uncertainty_km, hours=scenario.dt_hours
            )

            confidence = track.confidence
            breakdown = anomaly_breakdown(vessel)
            score = anomaly_score(vessel)
            inferred_state, state_confidence = infer_state(vessel, vessel.history)

            mini_record = {
                "time":                current_time,
                "anomaly_score":       score,
                "custody_confidence":  confidence,
                "sensitive_zone":      breakdown["sensitive_zone"].score,
                "loitering":           breakdown["loitering"].score,
                "vessel_proximity_score": 0.0,
            }
            compounds = evaluate_compounds(mini_record)
            compound_boost = max((s.confidence for s in compounds), default=0.0)

            vessel_states[vid] = {
                "vessel":          vessel,
                "track":           track,
                "mode":            PROFILE_TO_MODE.get(phase.profile.name, "transit"),
                "score":           score,
                "confidence":      confidence,
                "breakdown":       breakdown,
                "inferred_state":  inferred_state,
                "state_confidence": state_confidence,
                "compound_boost":  compound_boost,
            }
            vessels[vid] = vessel

        # ── Phase 2: sensor-access filtering and vessel ranking ───────────────
        for vid, state in vessel_states.items():
            v = state["vessel"]
            state["initial_opportunities"] = get_sensor_opportunities(
                current_time, v.lat, v.lon
            )

        ranked_vessels = sorted(
            vessel_states.items(),
            key=lambda kv: compute_target_priority(
                kv[1]["score"], kv[1]["confidence"], kv[1]["compound_boost"]
            ),
            reverse=True,
        )

        # ── Phase 3: sensor arbitration and record emission ───────────────────
        claimed_sensor_ids: set[str] = set()

        for vid, state in ranked_vessels:
            track = state["track"]
            score = state["score"]
            confidence = state["confidence"]
            breakdown = state["breakdown"]
            vessel = state["vessel"]
            vessel_initial = state["initial_opportunities"]
            compound_boost = state["compound_boost"]

            claimed_before = set(claimed_sensor_ids)
            vessel_remaining = [
                o for o in vessel_initial if o.sensor_id not in claimed_sensor_ids
            ]
            preempted = bool(vessel_initial) and not vessel_remaining

            decision = plan_collection(
                track, score, confidence, breakdown, current_time,
                vessel_remaining, preempted=preempted,
                compound_boost=compound_boost,
                observer_lat=vessel.lat,
                observer_lon=vessel.lon,
            )
            if decision.action == "TASK" and decision.sensor_id:
                claimed_sensor_ids.add(decision.sensor_id)

            trace = build_decision_trace(
                timestamp=current_time,
                vessel_id=vid,
                score=score,
                confidence=confidence,
                compound_boost=compound_boost,
                track=track,
                accessible_opportunities=vessel_initial,
                claimed_sensor_ids=claimed_before,
                remaining_opportunities=vessel_remaining,
                decision_action=decision.action,
                decision_sensor_id=decision.sensor_id,
                lookahead_boost=decision.lookahead_boost,
                nearest_pass_tts=decision.nearest_pass_tts,
                hold_reason=decision.hold_reason,
            )

            timelines[vid].append({
                "target_id":          vid,
                "time":               current_time,
                "lat":                vessel.lat,
                "lon":                vessel.lon,
                "uncertainty_km":     track.uncertainty_km,
                "custody_confidence": confidence,
                "history_length":     len(vessel.history),
                "anomaly_score":      score,
                "speed_kmh":          vessel.speed_kmh,
                "heading_deg":        vessel.heading_deg,
                "behavior_mode":      state["mode"],
                "action":             decision.action,
                "action_reason":      decision.action_reason,
                "sensor_id":          decision.sensor_id,
                "sensor_type":        decision.sensor_type,
                "collection_result":  decision.collection_result,
                "sensitive_zone":     breakdown["sensitive_zone"].score,
                "loitering":          breakdown["loitering"].score,
                "route_deviation":    breakdown["route_deviation"].score,
                "behavior_state":     state["inferred_state"].value,
                "state_confidence":   state["state_confidence"],
                "sensor_access_count": len(vessel_initial),
                "decision_trace":     trace,
            })

        current_time += timedelta(hours=scenario.dt_hours)

    # Return records interleaved chronologically across all vessels
    return sorted(
        (r for vid in timelines for r in timelines[vid]),
        key=lambda r: (r["time"], r["target_id"]),
    )
