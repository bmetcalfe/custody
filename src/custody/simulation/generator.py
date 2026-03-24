"""Generate background vessel population for a scenario."""
from __future__ import annotations
import random
import math

from custody.simulation.profiles import NORMAL_TRANSIT, SLOW_TRANSIT
from custody.simulation.scenarios import VesselSpec, ProfilePhase, ScenarioConfig

# Eight compass directions for background traffic spread
_HEADING_CLASSES = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]


def generate_background_vessels(
    n: int,
    region: tuple[float, float, float, float],
    rng: random.Random,
) -> list[VesselSpec]:
    """Create n background vessels with seeded-random positions and headings.

    Vessels are spread across 8 heading classes so traffic comes from all
    directions.  Each vessel uses NORMAL_TRANSIT or SLOW_TRANSIT (2:1 ratio).
    """
    lat_min, lat_max, lon_min, lon_max = region
    vessels: list[VesselSpec] = []

    per_class = n // len(_HEADING_CLASSES)
    remainder = n % len(_HEADING_CLASSES)

    counter = 1
    for i, base_heading in enumerate(_HEADING_CLASSES):
        count = per_class + (1 if i < remainder else 0)
        for _ in range(count):
            lat = rng.uniform(lat_min, lat_max)
            lon = rng.uniform(lon_min, lon_max)
            heading = (base_heading + rng.uniform(-15.0, 15.0)) % 360.0
            profile = NORMAL_TRANSIT if rng.random() < 0.7 else SLOW_TRANSIT
            vessels.append(VesselSpec(
                vessel_id=f"BG-{counter:03d}",
                start_lat=lat,
                start_lon=lon,
                start_heading_deg=heading,
                phases=[ProfilePhase(start_hour=0, profile=profile)],
                is_anomalous=False,
            ))
            counter += 1

    return vessels


def build_vessel_list(scenario: ScenarioConfig, rng: random.Random) -> list[VesselSpec]:
    """Return all vessels: scripted anomalous actors first, then background."""
    background = generate_background_vessels(scenario.n_background, scenario.region, rng)
    return list(scenario.scripted_vessels) + background
