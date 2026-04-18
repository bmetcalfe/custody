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
from custody.belief_assessment import build_fusion_assessment
from custody.decision import build_decision
from custody.reasoning import enrich_record as _enrich_reasoning
from custody.solar import sensor_suitability as _sensor_suitability
from custody.swath import assess_swath as _assess_swath
from custody.tasking_policy import compute_policy as _compute_policy
from custody.confidence import (
    compute_overall_confidence as _compute_confidence,
    apply_confidence_to_priority as _apply_conf_priority,
    confidence_action_bias as _conf_action_bias,
)
from custody.collection_intent import (
    derive_observation_state as _derive_obs_state,
    apply_observation_to_policy as _apply_obs_policy,
)


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
    # Per-vessel trigger latch state, consecutive-no-task counter, and anomaly state
    latched: dict[str, set[int]] = {}
    consecutive_no_task: dict[str, int] = {}
    anomaly_states: dict[str, str | None] = {}

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
        anomaly_states[vid] = None

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
            vessel = vessel.step(hours=scenario.dt_hours)
            vessel.last_seen = current_time
            track.predict(dt_seconds=scenario.dt_hours * 3600.0)

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
                # Consecutive failed collection attempts (0 after success)
                "consecutive_failures":    track.consecutive_failures,
                # ML anomaly score (0.0 in simulation; populated by AIS ML pipeline)
                "ml_anomaly_score":        0.0,
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

            # ── Temporal anomaly reasoning ──────────────────────────────
            # Enrich with agreement, persistence, escalation, and state.
            # timelines[vid] does not yet include the current record.
            _enriched = _enrich_reasoning(
                record, timelines[vid], previous_state=anomaly_states[vid],
            )
            for _k in ("anomaly_agreement", "ml_anomaly_duration_hours",
                        "fused_anomaly_duration_hours", "is_sustained_anomaly",
                        "anomaly_onset_timestamp", "escalation_boost", "anomaly_state"):
                record[_k] = _enriched[_k]
            anomaly_states[vid] = record["anomaly_state"]

            # ── Solar suitability + tasking policy ──────────────────────
            _solar = _sensor_suitability(current_time, state["rec_lat"], state["rec_lon"])
            record["sun_elevation_deg"]  = _solar["sun_elevation_deg"]
            record["solar_condition"]    = _solar["solar_condition"]
            record["eo_suitability"]     = _solar["eo_suitability"]
            record["sar_suitability"]    = _solar["sar_suitability"]

            _policy = _compute_policy(record)
            record["tasking_tier"]                = _policy.tasking_tier
            record["desired_revisit_hours"]       = _policy.desired_revisit_hours
            record["monitoring_action"]           = _policy.monitoring_action
            record["sensor_preference"]           = _policy.sensor_preference
            record["effective_sensor_preference"] = _policy.effective_sensor_preference
            record["tasking_rationale"]           = _policy.rationale
            record["sensor_rationale"]            = _policy.sensor_rationale

            # ── Confidence modeling ─────────────────────────────────────
            _conf = _compute_confidence(record, timelines[vid][:-1] if timelines[vid] else [])
            record["history_confidence"]    = _conf["history_confidence"]
            record["baseline_confidence"]   = _conf["baseline_confidence"]
            record["sensor_confidence"]     = _conf["sensor_confidence"]
            record["overall_confidence"]    = _conf["overall_confidence"]
            record["confidence_category"]   = _conf["confidence_category"]
            record["confidence_rationale"]  = _conf["confidence_rationale"]

            # Confidence-adjusted monitoring action
            record["monitoring_action"] = _conf_action_bias(
                record["monitoring_action"], _conf["overall_confidence"]
            )

            # ── Observation-aware reasoning ─────────────────────────────
            # Derives collection intent from observation history and adjusts
            # sensor preference for cross-sensor confirmation.
            _obs = _derive_obs_state(record, timelines[vid])
            _obs_adjusted = _apply_obs_policy(record, _obs)
            for _ok in ("collection_intent", "observation_rationale",
                         "needs_cross_sensor", "last_sensor_type",
                         "hours_since_observation"):
                record[_ok] = _obs_adjusted[_ok]
            # Apply cross-sensor and stale-observation adjustments
            if _obs_adjusted.get("effective_sensor_preference") != record.get("effective_sensor_preference"):
                record["effective_sensor_preference"] = _obs_adjusted["effective_sensor_preference"]
            if _obs_adjusted.get("sensor_rationale") != record.get("sensor_rationale"):
                record["sensor_rationale"] = _obs_adjusted["sensor_rationale"]
            if _obs_adjusted.get("desired_revisit_hours") != record.get("desired_revisit_hours"):
                record["desired_revisit_hours"] = _obs_adjusted["desired_revisit_hours"]

            # Update consecutive-no-task counter for trigger evaluation.
            # Only a *successful* TASK resets the counter; a failed TASK still
            # counts as "no successful task" for trigger purposes.
            if decision.action == "TASK" and decision.success:
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

        # ── Swath assessment (post-portfolio) ────────────────────────────────
        # Runs after portfolio has set portfolio_score on all records, so
        # grouped value uses real priority information.
        for record in current_timestep_records:
            if record.get("action") == "TASK" and record.get("sensor_type"):
                _sw = _assess_swath(record, record["sensor_type"], current_timestep_records)
                record["swath_width_km"]        = _sw.swath_width_km
                record["swath_length_km"]       = _sw.swath_length_km
                record["covered_target_count"]  = _sw.covered_target_count
                record["covered_targets"]       = ",".join(_sw.covered_target_ids)
                record["covered_priority_sum"]      = _sw.covered_priority_sum
                record["mean_covered_confidence"]  = _sw.mean_covered_confidence
                record["swath_task_value"]         = _sw.swath_task_value
                record["swath_value_uplift"]    = _sw.value_uplift
                record["swath_rationale"]       = _sw.rationale
            else:
                record["swath_width_km"]        = None
                record["swath_length_km"]       = None
                record["covered_target_count"]  = 0
                record["covered_targets"]       = ""
                record["covered_priority_sum"]      = 0.0
                record["mean_covered_confidence"]  = 0.0
                record["swath_task_value"]         = 0.0
                record["swath_value_uplift"]    = 0.0
                record["swath_rationale"]       = ""

        # ── Effective task value (post-swath) ────────────────────────────────
        # Combines the planner's base task value with swath uplift to produce
        # the final decision-bearing value and an effective ranking.
        _task_this_step = [r for r in current_timestep_records if r.get("action") == "TASK"]
        for record in current_timestep_records:
            trace = record.get("decision_trace")
            base_tv = trace.task_value.total if trace else 0.0
            uplift = float(record.get("swath_value_uplift", 0.0))
            eff = round(base_tv + uplift, 4)
            record["base_task_value"]     = round(base_tv, 4)
            record["effective_task_value"] = eff

            if uplift > 0.01:
                record["effective_task_rationale"] = (
                    f"grouped swath uplift +{uplift:.3f} "
                    f"(covers {record.get('covered_target_count', 1)} targets) "
                    f"raised effective value from {base_tv:.3f} to {eff:.3f}"
                )
            else:
                record["effective_task_rationale"] = ""

        # Effective rank among TASK records (1 = highest effective value)
        if _task_this_step:
            _sorted_tasks = sorted(_task_this_step,
                                    key=lambda r: r["effective_task_value"],
                                    reverse=True)
            for rank, r in enumerate(_sorted_tasks, 1):
                r["effective_task_rank"] = rank
        for record in current_timestep_records:
            if "effective_task_rank" not in record:
                record["effective_task_rank"] = None

        # ── Confidence-adjusted priority ─────────────────────────────────
        for record in current_timestep_records:
            ps = float(record.get("portfolio_score", 0.0))
            oc = float(record.get("overall_confidence", 1.0))
            record["confidence_adjusted_priority"] = _apply_conf_priority(ps, oc)

        current_time += timedelta(hours=scenario.dt_hours)

    # Return records interleaved chronologically across all vessels
    return sorted(
        (r for vid in timelines for r in timelines[vid]),
        key=lambda r: (r["time"], r["target_id"]),
    )
