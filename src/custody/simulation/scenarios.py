"""Scenario configuration and scripted vessel specs."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from custody.simulation.profiles import (
    BehaviorProfile, NORMAL_TRANSIT, SLOW_TRANSIT,
    LOITERING, ZONE_APPROACH, EVASIVE,
)


@dataclass(frozen=True)
class ProfilePhase:
    """One behavior phase in a vessel's scripted timeline."""
    start_hour: int
    profile: BehaviorProfile
    # If both set, vessel steers toward this point each step
    target_lat: Optional[float] = None
    target_lon: Optional[float] = None
    # If set, override heading to this fixed value
    heading_override: Optional[float] = None


@dataclass
class VesselSpec:
    vessel_id: str
    start_lat: float
    start_lon: float
    start_heading_deg: float
    phases: list[ProfilePhase]   # must be sorted by start_hour, first phase start_hour=0
    is_anomalous: bool = False


@dataclass
class ScenarioConfig:
    name: str
    seed: int
    n_background: int
    start_time: datetime
    duration_hours: int
    dt_hours: int
    # (lat_min, lat_max, lon_min, lon_max) — bounding box for background placement
    region: tuple[float, float, float, float]
    scripted_vessels: list[VesselSpec]


# ── Default 24-hour, 30-vessel scenario ───────────────────────────────────────
#
# 3 scripted anomalous actors + 27 background vessels.
# Scenario date: 2026-03-23 (matches calibrated TLE passes in sensors.py).
# Key action window: 10:00-18:00z (orbital passes available).
#
# SIGMA-1: SW approach → loiters in ZONE_ALPHA 14:00-20:00z
# SIGMA-2: N transit + route deviation → approaches zone 18:00-24:00z
# SIGMA-3: E transit + evasive → reaches zone boundary 20:00-24:00z

DEFAULT_SCENARIO = ScenarioConfig(
    name="multi_target_24h",
    seed=2026,
    n_background=27,
    start_time=datetime(2026, 3, 23, 0, 0, tzinfo=timezone.utc),
    duration_hours=24,
    dt_hours=1,
    region=(-3.0, 5.0, -3.0, 5.0),
    scripted_vessels=[
        # SIGMA-1: Approaches zone from SW, loiters inside
        VesselSpec(
            vessel_id="SIGMA-1",
            start_lat=-0.5, start_lon=-0.5,
            start_heading_deg=45.0,
            phases=[
                ProfilePhase(0,  NORMAL_TRANSIT),
                ProfilePhase(8,  ZONE_APPROACH, target_lat=1.2, target_lon=0.75),
                ProfilePhase(14, LOITERING),
                ProfilePhase(20, EVASIVE, heading_override=120.0),
            ],
            is_anomalous=True,
        ),
        # SIGMA-2: Route deviation from N, reaches zone late
        VesselSpec(
            vessel_id="SIGMA-2",
            start_lat=3.0, start_lon=2.0,
            start_heading_deg=200.0,
            phases=[
                ProfilePhase(0,  NORMAL_TRANSIT),
                ProfilePhase(8,  NORMAL_TRANSIT, heading_override=230.0),
                ProfilePhase(14, ZONE_APPROACH, target_lat=1.3, target_lon=0.9),
                ProfilePhase(19, LOITERING),
            ],
            is_anomalous=True,
        ),
        # SIGMA-3: Evasive from E, approaches zone boundary
        VesselSpec(
            vessel_id="SIGMA-3",
            start_lat=1.5, start_lon=3.0,
            start_heading_deg=270.0,
            phases=[
                ProfilePhase(0,  SLOW_TRANSIT),
                ProfilePhase(6,  EVASIVE),
                ProfilePhase(14, ZONE_APPROACH, target_lat=1.1, target_lon=0.6),
                ProfilePhase(20, LOITERING),
            ],
            is_anomalous=True,
        ),
    ],
)
