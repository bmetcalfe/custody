from dataclasses import dataclass
from datetime import datetime


@dataclass
class SensorOpportunity:
    sensor_id: str
    sensor_type: str
    success_prob: float
    resolution: str
    cost: float
    available_from: datetime
    available_to: datetime


def get_sensor_opportunities(current_time):
    opportunities = []
    hour = current_time.hour

    # A1: fast revisit, cheap, good for reacquisition
    if hour % 2 == 0:
        opportunities.append(
            SensorOpportunity(
                sensor_id="A1",
                sensor_type="fast_revisit",
                success_prob=0.85,
                resolution="medium",
                cost=1.0,
                available_from=current_time,
                available_to=current_time,
            )
        )

    # B1: high-res, less frequent, best for characterization
    if hour in [13, 17]:
        opportunities.append(
            SensorOpportunity(
                sensor_id="B1",
                sensor_type="high_resolution",
                success_prob=0.65,
                resolution="high",
                cost=2.0,
                available_from=current_time,
                available_to=current_time,
            )
        )

    # C1: all-weather, medium revisit, resilient fallback
    if hour % 3 == 0:
        opportunities.append(
            SensorOpportunity(
                sensor_id="C1",
                sensor_type="all_weather",
                success_prob=0.75,
                resolution="medium",
                cost=1.5,
                available_from=current_time,
                available_to=current_time,
            )
        )

    return opportunities