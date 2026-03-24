"""Behavior profile archetypes for multi-target simulation."""
from dataclasses import dataclass


@dataclass(frozen=True)
class BehaviorProfile:
    name: str
    base_speed_kmh: float
    speed_variance_kmh: float
    heading_drift_deg: float   # std dev of per-step random heading change


NORMAL_TRANSIT = BehaviorProfile("normal_transit", 14.0, 1.5, 4.0)
SLOW_TRANSIT   = BehaviorProfile("slow_transit",    7.0, 1.0, 6.0)
LOITERING      = BehaviorProfile("loitering",       2.0, 0.5, 20.0)
ZONE_APPROACH  = BehaviorProfile("zone_approach",  10.0, 0.5, 2.0)
EVASIVE        = BehaviorProfile("evasive",        20.0, 3.0, 25.0)

# Map profile name → behavior_mode string for pipeline compatibility
PROFILE_TO_MODE: dict[str, str] = {
    "normal_transit": "transit",
    "slow_transit":   "transit",
    "zone_approach":  "approach",
    "loitering":      "loiter",
    "evasive":        "egress",
}
