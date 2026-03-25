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
PATROL         = BehaviorProfile("patrol",         10.0, 1.5, 8.0)

# Demo-scenario profiles — zero variance so positions are fully deterministic.
# Speeds and headings mirror the hardcoded values in behavior/modes.py so that
# DEMO_SCENARIO produces the same trajectory as the original simulate.py loop.
DEMO_TRANSIT  = BehaviorProfile("demo_transit",  28.0, 0.0, 0.0)
DEMO_APPROACH = BehaviorProfile("demo_approach", 20.0, 0.0, 0.0)
DEMO_LOITER   = BehaviorProfile("demo_loiter",    2.0, 0.0, 15.0)
DEMO_EGRESS   = BehaviorProfile("demo_egress",   26.0, 0.0, 0.0)

# Map profile name → behavior_mode string for pipeline compatibility
PROFILE_TO_MODE: dict[str, str] = {
    "normal_transit": "transit",
    "slow_transit":   "transit",
    "zone_approach":  "approach",
    "loitering":      "loiter",
    "evasive":        "egress",
    "patrol":         "transit",
    "demo_transit":   "transit",
    "demo_approach":  "approach",
    "demo_loiter":    "loiter",
    "demo_egress":    "egress",
}
