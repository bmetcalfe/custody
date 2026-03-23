"""
Central configuration for the custody package.

All constants that influence scoring, inference, or sensor-zone geometry
live here.  custody.models is the only custody module imported here (for
the Zone type); everything else in the package sits below this in the
import graph with no risk of circular imports.
"""

from custody.models import Zone

# ---------------------------------------------------------------------------
# Sensitive zone geometry
# ---------------------------------------------------------------------------

ZONES: list[Zone] = [
    Zone(
        name="ZONE_ALPHA",
        min_lat=1.0,
        max_lat=1.6,
        min_lon=0.55,
        max_lon=0.95,
        halo=0.1,  # ~11 km at equatorial latitudes
    ),
]

# Zone proximity threshold used by state inference (degrees).
# A history entry within this distance counts as "near zone" for egress confirmation.
NEAR_ZONE_DEG = 0.2

# ---------------------------------------------------------------------------
# Speed thresholds (km/h)
# ---------------------------------------------------------------------------

IDLE_SPEED_KMH = 1.0   # below → vessel is effectively stationary
SLOW_SPEED_KMH = 5.0   # below → vessel may be loitering

# ---------------------------------------------------------------------------
# Collection planning thresholds
# ---------------------------------------------------------------------------

# Anomaly score above which tasking is considered, regardless of confidence.
ANOMALY_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Alerting thresholds
# ---------------------------------------------------------------------------

# Anomaly score above which a WARNING-level alert fires.
HIGH_ANOMALY_THRESHOLD = 0.8

# Anomaly score above which a CRITICAL-level alert fires.
CRITICAL_ANOMALY_THRESHOLD = 1.5

# Custody confidence below which a LOW_CUSTODY WARNING fires.
LOW_CUSTODY_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Inter-vessel proximity thresholds
# ---------------------------------------------------------------------------

# Distance in km at which a vessel-proximity CRITICAL score (1.0) fires.
PROXIMITY_CRITICAL_KM = 5.0

# Distance in km at which a vessel-proximity WARNING score (0.5) fires.
PROXIMITY_WARNING_KM = 15.0

# ---------------------------------------------------------------------------
# Compound behavior thresholds
# ---------------------------------------------------------------------------

# Minimum sensitive_zone detector score for zone-proximity compound rules.
# Below this value the vessel is considered too far from a zone to warrant
# a compound signal.
ZONE_COMPOUND_MIN_SCORE = 0.5
