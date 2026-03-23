from datetime import datetime, UTC, timedelta
import random

from custody.models import TrackState, Vessel
from custody.tracks import update_position, update_uncertainty
from custody.anomalies import anomaly_score, anomaly_breakdown
from custody.planner import plan_collection
from custody.behavior import apply_behavior_mode, get_behavior_mode
from custody.behavior.state_machine import infer_state

random.seed(42)


def simulate_target(vessel, start_time, end_time, behavior_schedule):
    current_time = start_time
    track = TrackState()
    timeline = []

    while current_time <= end_time:
        hour_index = int((current_time - start_time).total_seconds() // 3600)

        mode = get_behavior_mode(hour_index, behavior_schedule)
        vessel = apply_behavior_mode(vessel, mode)

        vessel = update_position(vessel, hours=1)
        vessel.last_seen = current_time

        track.uncertainty_km = update_uncertainty(track.uncertainty_km, hours=1)
        confidence = track.confidence

        breakdown = anomaly_breakdown(vessel)
        score = anomaly_score(vessel)
        inferred_state, state_confidence = infer_state(vessel, vessel.history)

        decision = plan_collection(track, score, confidence, breakdown, current_time)

        timeline.append(
            {
                "target_id": vessel.id,
                "time": current_time,
                "lat": vessel.lat,
                "lon": vessel.lon,
                "uncertainty_km": track.uncertainty_km,
                "custody_confidence": track.confidence,
                "history_length": len(vessel.history),
                "anomaly_score": score,
                "speed_kmh": vessel.speed_kmh,
                "heading_deg": vessel.heading_deg,
                "behavior_mode": mode,
                "action": decision.action,
                "action_reason": decision.action_reason,
                "sensor_id": decision.sensor_id,
                "sensor_type": decision.sensor_type,
                "collection_result": decision.collection_result,
                "sensitive_zone": breakdown["sensitive_zone"].score,
                "loitering": breakdown["loitering"].score,
                "route_deviation": breakdown["route_deviation"].score,
                "behavior_state": inferred_state.value,
                "state_confidence": state_confidence,
            }
        )

        current_time += timedelta(hours=1)

    return timeline


def run_simulation():
    start_time = datetime(2026, 3, 23, 10, 0, tzinfo=UTC)
    end_time = datetime(2026, 3, 23, 19, 0, tzinfo=UTC)

    vessel_1 = Vessel(
        id="V001",
        lat=0.0,
        lon=0.0,
        speed_kmh=30,
        heading_deg=45,
        last_seen=start_time,
    )

    vessel_2 = Vessel(
        id="V002",
        lat=0.2,
        lon=0.1,
        speed_kmh=24,
        heading_deg=35,
        last_seen=start_time,
    )

    behavior_schedule_1 = [
        ((0, 3), "transit"),
        ((4, 5), "approach"),
        ((6, 8), "loiter"),
        ((9, 12), "egress"),
    ]

    behavior_schedule_2 = [
        ((0, 4), "transit"),
        ((5, 6), "approach"),
        ((7, 9), "egress"),
    ]

    timeline_1 = simulate_target(vessel_1, start_time, end_time, behavior_schedule_1)
    timeline_2 = simulate_target(vessel_2, start_time, end_time, behavior_schedule_2)

    return timeline_1 + timeline_2


if __name__ == "__main__":
    records = run_simulation()
    for row in records[:12]:
        print(
            row["target_id"],
            row["time"].isoformat(),
            row["behavior_mode"],
            row["lat"],
            row["lon"],
            row["anomaly_score"],
            row["action"],
        )