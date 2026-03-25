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
import random as _random_module
from datetime import timedelta
from typing import Optional

from custody.models import Vessel, TrackState
from custody.tracks import update_position, update_uncertainty
from custody.anomalies import anomaly_score, anomaly_breakdown
from custody.planner import plan_collection, compute_target_priority
from custody.behavior.state_machine import infer_state
from custody.compounds import evaluate_compounds
from custody.decision_trace import build_decision_trace
from custody.sensors import get_sensor_opportunities, next_pass_window
from custody.features.proximity_features import haversine_km
from custody.config import PROXIMITY_CRITICAL_KM, PROXIMITY_WARNING_KM
from custody.interactions import detect_rendezvous_events, RendezvousEvent
from custody.dark_vessel import mark_dark, build_dark_reason

from custody.simulation.profiles import BehaviorProfile, PROFILE_TO_MODE
from custody.simulation.scenarios import ScenarioConfig, VesselSpec, ProfilePhase, PhaseTrigger, DEFAULT_SCENARIO
from custody.simulation.generator import build_vessel_list
from custody.orchestration.portfolio import rank_portfolio
from custody.prediction import predict_entity
from custody.config import ZONES
from custody.fusion import build_fusion_assessment
from custody.decision import build_decision


def _trigger_satisfied(
    trigger: PhaseTrigger,
    vessel_state: dict,
) -> bool:
    """Evaluate a PhaseTrigger against current vessel state.

    Args:
        trigger:      The condition to check.
        vessel_state: Dict with keys: anomaly_score, custody_confidence,
                      sensitive_zone, consecutive_no_task.
    """
    cond = trigger.condition
    thr = trigger.threshold

    if cond == "anomaly_above":
        return vessel_state.get("anomaly_score", 0.0) > thr
    if cond == "anomaly_below":
        return vessel_state.get("anomaly_score", 0.0) < thr
    if cond == "custody_below":
        return vessel_state.get("custody_confidence", 1.0) < thr
    if cond == "custody_above":
        return vessel_state.get("custody_confidence", 1.0) > thr
    if cond == "missed_collections":
        return vessel_state.get("consecutive_no_task", 0) >= thr
    if cond == "zone_entry":
        return vessel_state.get("sensitive_zone", 0.0) > 0.0
    if cond == "zone_exit":
        return vessel_state.get("sensitive_zone", 0.0) == 0.0
    return False


def _current_phase(
    spec: VesselSpec,
    hour_index: int,
    vessel_state: Optional[dict] = None,
    latched_triggers: Optional[set] = None,
) -> ProfilePhase:
    """Return the active phase for the given simulation hour.

    Phases are evaluated in order; the last phase whose start_hour has been
    reached AND whose trigger (if any) is satisfied wins.  Once a trigger
    fires it is recorded in *latched_triggers* (mutated in place) so the
    phase stays active even if the condition becomes false later.

    Args:
        spec:              Vessel specification with ordered phases.
        hour_index:        Current simulation hour offset.
        vessel_state:      Dict of current signals for trigger evaluation.
                           Pass None when no triggers are used (backward compat).
        latched_triggers:  Mutable set of phase indices whose triggers have
                           already fired.  Pass None for time-only behavior.
    """
    active = spec.phases[0]
    for idx, phase in enumerate(spec.phases):
        if phase.start_hour > hour_index:
            continue
        if phase.trigger is None:
            active = phase
        elif latched_triggers is not None and idx in latched_triggers:
            # Previously fired — stays active
            active = phase
        elif vessel_state is not None and _trigger_satisfied(phase.trigger, vessel_state):
            if latched_triggers is not None:
                latched_triggers.add(idx)
            active = phase
    return active


def _apply_profile(
    vessel: Vessel,
    phase: ProfilePhase,
    rng: _random_module.Random,
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

    # Seed the global random module so callers of random.random() (e.g.
    # collection.py success rolls) are deterministic for the same scenario.
    _random_module.seed(scenario.seed)
    rng = _random_module.Random(scenario.seed)

    # Build ordered vessel list (scripted + background)
    vessel_specs = build_vessel_list(scenario, rng)

    # Initialise state for each vessel
    vessels: dict[str, Vessel] = {}
    tracks: dict[str, TrackState] = {}
    timelines: dict[str, list[dict]] = {}
    spec_by_id: dict[str, VesselSpec] = {}
    # Per-vessel trigger latch state and consecutive-no-task counter
    latched: dict[str, set[int]] = {}
    consecutive_no_task: dict[str, int] = {}

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
        latched[vid] = set()
        consecutive_no_task[vid] = 0

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

            # Build trigger state from the previous timestep's record
            _prev = timelines[vid][-1] if timelines[vid] else None
            _trig_state: Optional[dict] = None
            if _prev is not None:
                _trig_state = {
                    "anomaly_score":       _prev.get("anomaly_score", 0.0),
                    "custody_confidence":  _prev.get("custody_confidence", 1.0),
                    "sensitive_zone":      _prev.get("sensitive_zone", 0.0),
                    "consecutive_no_task": consecutive_no_task[vid],
                }

            phase = _current_phase(spec, hour_index, _trig_state, latched[vid])
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

            # ── AIS dropout (dark vessel) ─────────────────────────────────────
            is_dropout = (
                spec.ais_dropout_hour is not None
                and hour_index >= spec.ais_dropout_hour
            )
            if is_dropout:
                if not track.is_dark:
                    mark_dark(track, vessel, current_time, score)
                # Override record values with frozen last-known state
                rec_lat   = track.last_known_lat
                rec_lon   = track.last_known_lon
                rec_score = track.last_known_anomaly
                dark_vessel_flag   = True
                dark_vessel_reason = build_dark_reason(track.dark_since, current_time)
            else:
                rec_lat   = vessel.lat
                rec_lon   = vessel.lon
                rec_score = score
                dark_vessel_flag   = False
                dark_vessel_reason = None

            mini_record = {
                "time":                current_time,
                "anomaly_score":       rec_score,
                "custody_confidence":  confidence,
                "sensitive_zone":      breakdown["sensitive_zone"].score,
                "loitering":           breakdown["loitering"].score,
                "vessel_proximity_score": 0.0,
            }
            compounds = evaluate_compounds(mini_record)
            compound_boost = max((s.confidence for s in compounds), default=0.0)

            vessel_states[vid] = {
                "vessel":              vessel,
                "track":               track,
                "mode":                PROFILE_TO_MODE.get(phase.profile.name, "transit"),
                "profile_name":        phase.profile.name,
                "score":               rec_score,
                "confidence":          confidence,
                "breakdown":           breakdown,
                "inferred_state":      inferred_state,
                "state_confidence":    state_confidence,
                "compound_boost":      compound_boost,
                "rec_lat":             rec_lat,
                "rec_lon":             rec_lon,
                "dark_vessel_flag":    dark_vessel_flag,
                "dark_vessel_reason":  dark_vessel_reason,
                "dark_since":          track.dark_since if is_dropout else None,
                "last_known_lat":      track.last_known_lat if is_dropout else None,
                "last_known_lon":      track.last_known_lon if is_dropout else None,
                "last_known_time":     track.last_known_time if is_dropout else None,
            }
            vessels[vid] = vessel

        # ── Phase 1.5: pairwise proximity and rendezvous detection ───────────
        # Runs after all vessel positions are updated (Phase 1 complete) so
        # pairwise distances reflect the current timestep.  Updates each
        # vessel's compound_boost with real proximity scores and rendezvous
        # confidence before Phase 2 uses compound_boost for sensor ranking.

        current_positions = {
            vid: (vessels[vid].lat, vessels[vid].lon) for vid in vessel_states
        }
        rendezvous_events = detect_rendezvous_events(
            current_positions=current_positions,
            vessel_histories=timelines,   # prior records only — current not yet appended
            timestamp=current_time,
            dt_hours=scenario.dt_hours,
        )
        # vessel_id → event lookup; both members of each pair get the same event
        rendezvous_by_vessel: dict[str, RendezvousEvent] = {}
        for _evt in rendezvous_events:
            rendezvous_by_vessel[_evt.vessel_a] = _evt
            rendezvous_by_vessel[_evt.vessel_b] = _evt

        for vid, state in vessel_states.items():
            v = state["vessel"]
            # Compute nearest-vessel distance and map to [0,1] proximity score
            nearest_km = min(
                (haversine_km(v.lat, v.lon, vessels[ov].lat, vessels[ov].lon)
                 for ov in vessels if ov != vid),
                default=math.inf,
            )
            if nearest_km <= PROXIMITY_CRITICAL_KM:
                prox_score = 1.0
            elif nearest_km <= PROXIMITY_WARNING_KM:
                prox_score = 0.5
            else:
                prox_score = 0.0
            state["vessel_proximity_score"] = prox_score

            # Re-evaluate compound signals with the real proximity score so
            # proximity-based compounds (LOITERING_WITH_PROXIMITY, etc.) can fire.
            if prox_score > 0.0:
                prox_mini = {
                    "time":                   current_time,
                    "anomaly_score":          state["score"],
                    "custody_confidence":     state["confidence"],
                    "sensitive_zone":         state["breakdown"]["sensitive_zone"].score,
                    "loitering":              state["breakdown"]["loitering"].score,
                    "vessel_proximity_score": prox_score,
                }
                prox_compounds = evaluate_compounds(prox_mini)
                prox_boost = max((s.confidence for s in prox_compounds), default=0.0)
                state["compound_boost"] = max(state["compound_boost"], prox_boost)

            # Rendezvous events provide an additional compound_boost
            if vid in rendezvous_by_vessel:
                state["compound_boost"] = max(
                    state["compound_boost"],
                    rendezvous_by_vessel[vid].confidence,
                )

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
        sensor_claimer:     dict[str, str] = {}   # sensor_id → entity_id that claimed it

        current_timestep_records: list[dict] = []

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
                sensor_claimer[decision.sensor_id] = vid

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

            # ── Prediction layer ──────────────────────────────────────────────
            # Convert speed from km/h to knots (1 knot = 1.852 km/h)
            speed_kts = vessel.speed_kmh / 1.852
            # Dynamic horizon: match to nearest orbital pass (SAR or OPTICAL)
            _SAT_IDS = ["SAR-1", "SAR-2", "EO-MIO-1", "EO-MIO-2", "EO-SSO-1", "EO-SSO-2"]
            _best_tts: Optional[float] = None
            for _sid in _SAT_IDS:
                _pw = next_pass_window(_sid, state["rec_lat"], state["rec_lon"], current_time)
                if _pw is not None:
                    _tts_h = _pw.time_to_start_seconds / 3600.0
                    if _best_tts is None or _tts_h < _best_tts:
                        _best_tts = _tts_h
            _horizon_hours = max(1.0, min(_best_tts, 12.0)) if _best_tts is not None else 6.0

            _pred = predict_entity(
                entity_id=vid,
                lat=state["rec_lat"],
                lon=state["rec_lon"],
                speed_knots=speed_kts,
                heading_deg=vessel.heading_deg,
                current_anomaly=score,
                custody_confidence=confidence,
                zones=ZONES,
                horizon_hours=_horizon_hours,
            )

            _rdv = rendezvous_by_vessel.get(vid)
            record = {
                "target_id":               vid,
                "time":                    current_time,
                "lat":                     state["rec_lat"],
                "lon":                     state["rec_lon"],
                "uncertainty_km":          track.uncertainty_km,
                "custody_confidence":      confidence,
                "history_length":          len(vessel.history),
                "anomaly_score":           score,
                "speed_kmh":               vessel.speed_kmh,
                "heading_deg":             vessel.heading_deg,
                "behavior_mode":           state["mode"],
                "action":                  decision.action,
                "action_reason":           decision.action_reason,
                "sensor_id":               decision.sensor_id,
                "sensor_type":             decision.sensor_type,
                "collection_result":       decision.collection_result,
                "sensitive_zone":          breakdown["sensitive_zone"].score,
                "loitering":               breakdown["loitering"].score,
                "route_deviation":         breakdown["route_deviation"].score,
                "behavior_state":          state["inferred_state"].value,
                "state_confidence":        state["state_confidence"],
                "sensor_access_count":     len(vessel_initial),
                "decision_trace":          trace,
                "profile":                 state["profile_name"],
                "is_scripted":             spec_by_id[vid].is_anomalous,
                "scenario_tags":           list(spec_by_id[vid].tags),
                # hours_since_collection: None if never collected; 0.0 if just tasked
                "hours_since_collection":  track.hours_since_collection(current_time),
                # Operator tracking directive from vessel spec (NONE | MAINTAIN_CUSTODY)
                "tracking_directive":      spec_by_id[vid].tracking_directive,
                # Real pairwise proximity score (0.0 / 0.5 / 1.0)
                "vessel_proximity_score":  state.get("vessel_proximity_score", 0.0),
                # Rendezvous fields — additive, backward-compatible (None/False/0 when absent)
                "rendezvous_flag":         _rdv is not None,
                "counterpart_id":          (
                    _rdv.vessel_b if _rdv and _rdv.vessel_a == vid
                    else _rdv.vessel_a if _rdv else None
                ),
                "rendezvous_confidence":   _rdv.confidence if _rdv else 0.0,
                "rendezvous_dwell_hours":  _rdv.dwell_hours if _rdv else 0.0,
                "min_pair_distance_km":    _rdv.min_separation_km if _rdv else None,
                # Dark-vessel fields — None/False when AIS is active
                "dark_vessel_flag":        state["dark_vessel_flag"],
                "dark_vessel_reason":      state["dark_vessel_reason"],
                "dark_since":              state["dark_since"],
                "last_known_lat":          state["last_known_lat"],
                "last_known_lon":          state["last_known_lon"],
                "last_known_time":         state["last_known_time"],
                # Prediction fields — advisory; None values stored as float('nan')
                "future_lat":              _pred.future_lat,
                "future_lon":              _pred.future_lon,
                "zone_probability":        _pred.zone_probability,
                "time_to_zone_hours":      (
                    _pred.time_to_zone_hours
                    if _pred.time_to_zone_hours is not None
                    else float("nan")
                ),
                "future_anomaly":          _pred.future_anomaly,
                "prediction_confidence":   _pred.prediction_confidence,
                "prediction_reason":       _pred.prediction_reason,
                "prediction_horizon_hours": _horizon_hours,
            }
            # ── Fusion + Decision (canonical, used by portfolio & UI) ──────
            # Evaluate compounds from the full record with the history prefix
            # (timelines[vid] does not yet include the current record).
            _fusion_compounds = evaluate_compounds(record, window=timelines[vid])
            _fa = build_fusion_assessment(
                record, _fusion_compounds, track, planner_context=trace,
            )
            _mission_dec = build_decision(
                _fa, record, track, _fusion_compounds, planner_context=trace,
            )
            record["fusion_assessment"] = _fa
            record["mission_decision"]  = _mission_dec

            # Update consecutive-no-task counter for trigger evaluation
            if decision.action == "TASK":
                consecutive_no_task[vid] = 0
            else:
                consecutive_no_task[vid] += 1

            timelines[vid].append(record)
            current_timestep_records.append(record)

        # ── Portfolio orchestration ───────────────────────────────────────────
        # Rank all entities at this timestep and annotate each record in-place.
        # entity_histories excludes the current timestep (timelines[vid][:-1]).
        portfolio = rank_portfolio(
            timestamp        = current_time,
            timestep_records = current_timestep_records,
            scenario_start   = scenario.start_time,
            sensor_claimer   = sensor_claimer,
        )
        portfolio_by_entity = portfolio.by_entity()

        for record in current_timestep_records:
            item = portfolio_by_entity.get(record["target_id"])
            if item:
                record["portfolio_rank"]     = item.portfolio_rank
                record["portfolio_score"]    = item.portfolio_score
                record["custody_health"]     = item.custody_health
                record["neglect_flag"]       = item.neglect_flag
                record["neglect_hours"]      = item.neglect_hours
                record["portfolio_reason"]   = item.portfolio_reason
                record["deferred_for"]       = item.deferred_for
                record["attention_state"]    = item.attention_state
                record["attention_basis"]    = item.attention_basis

        current_time += timedelta(hours=scenario.dt_hours)

    # Return records interleaved chronologically across all vessels
    return sorted(
        (r for vid in timelines for r in timelines[vid]),
        key=lambda r: (r["time"], r["target_id"]),
    )
