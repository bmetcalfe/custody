"""Generate background vessel population for a scenario."""
from __future__ import annotations
import random
import math

from custody.simulation.profiles import NORMAL_TRANSIT, SLOW_TRANSIT, ZONE_APPROACH, LOITERING
from custody.simulation.scenarios import VesselSpec, ProfilePhase, ScenarioConfig

# Eight compass directions for background traffic spread
_HEADING_CLASSES = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]


def _build_archetype_phases(
    archetype: str,
    zone_center,
    duration_hours: int,
    rng: random.Random,
) -> tuple[list[ProfilePhase], list[str]]:
    """Return (phases, tags) for a given background archetype."""
    if archetype == "transit":
        return [ProfilePhase(start_hour=0, profile=NORMAL_TRANSIT)], ["transit"]

    elif archetype == "slow_transit":
        return [ProfilePhase(start_hour=0, profile=SLOW_TRANSIT)], ["slow_transit"]

    elif archetype == "patrol":
        loiter_start = rng.randint(max(4, duration_hours // 6), duration_hours // 3)
        loiter_end = loiter_start + rng.randint(4, 8)
        phases = [
            ProfilePhase(start_hour=0, profile=NORMAL_TRANSIT),
            ProfilePhase(start_hour=loiter_start, profile=LOITERING),
            ProfilePhase(start_hour=loiter_end, profile=NORMAL_TRANSIT),
        ]
        return phases, ["patrol", "loiter_cycle"]

    elif archetype == "approach":
        approach_start = rng.randint(max(4, duration_hours // 6), duration_hours // 2)
        zc = zone_center if zone_center is not None else (0.0, 0.0)
        phases = [
            ProfilePhase(start_hour=0, profile=NORMAL_TRANSIT),
            ProfilePhase(
                start_hour=approach_start,
                profile=ZONE_APPROACH,
                target_lat=zc[0],
                target_lon=zc[1],
            ),
        ]
        return phases, ["approach"]

    else:
        # Fallback: plain transit
        return [ProfilePhase(start_hour=0, profile=NORMAL_TRANSIT)], ["transit"]


def _sample_archetype(archetype_weights: dict, rng: random.Random) -> str:
    """Sample an archetype name from a weight dict."""
    archetypes = list(archetype_weights.keys())
    weights = [archetype_weights[a] for a in archetypes]
    total = sum(weights)
    r = rng.random() * total
    cumulative = 0.0
    for archetype, weight in zip(archetypes, weights):
        cumulative += weight
        if r < cumulative:
            return archetype
    return archetypes[-1]


def generate_background_vessels(
    n: int,
    region: tuple,
    rng,
    archetype_weights=None,
    zone_center=None,
    duration_hours: int = 24,
) -> list:
    """Create n background vessels with seeded-random positions and headings.

    Vessels are spread across 8 heading classes so traffic comes from all
    directions.

    When archetype_weights is None, uses the legacy NORMAL_TRANSIT/SLOW_TRANSIT
    70/30 split so DEFAULT_SCENARIO output is unchanged.

    When archetype_weights is provided, samples from the named archetypes using
    the given weights and builds multi-phase profiles accordingly.
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

            if archetype_weights is None:
                # Legacy path: preserve exact 70/30 split behaviour
                profile = NORMAL_TRANSIT if rng.random() < 0.7 else SLOW_TRANSIT
                phases = [ProfilePhase(start_hour=0, profile=profile)]
                tags: list = []
            else:
                archetype = _sample_archetype(archetype_weights, rng)
                phases, tags = _build_archetype_phases(
                    archetype, zone_center, duration_hours, rng
                )

            vessels.append(VesselSpec(
                vessel_id=f"BG-{counter:03d}",
                start_lat=lat,
                start_lon=lon,
                start_heading_deg=heading,
                phases=phases,
                is_anomalous=False,
                tags=tags,
            ))
            counter += 1

    return vessels


def build_vessel_list(scenario: ScenarioConfig, rng: random.Random) -> list[VesselSpec]:
    """Return all vessels: scripted anomalous actors first, then background."""
    zone_center = getattr(scenario, 'zone_center', None)
    archetype_weights = getattr(scenario, 'background_archetype_weights', None)
    background = generate_background_vessels(
        scenario.n_background,
        scenario.region,
        rng,
        archetype_weights=archetype_weights,
        zone_center=zone_center,
        duration_hours=scenario.duration_hours,
    )
    return list(scenario.scripted_vessels) + background
