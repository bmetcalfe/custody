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

# Custody confidence below which tasking is considered, regardless of anomaly score.
CUSTODY_TASK_CONFIDENCE_THRESHOLD = 0.7

# Freshness-decay task-value model for revisit suppression.
# See custody.planner.compute_task_value for the full formula.
REVISIT_DECAY_HOURS = 3.0      # exponential decay time constant (hours)
FRESHNESS_SUPPRESSION = 0.5    # peak freshness suppression applied immediately after collection
WORSENING_BOOST = 0.5          # task-value increase per unit of anomaly worsening since last collection
TASK_VALUE_THRESHOLD = 0.2     # minimum task value to proceed past HOLD

# Orbital-pass lookahead bias for HOLD decisions.
# When no sensor is available and the nearest upcoming orbital pass starts
# within HOLD_LOOKAHEAD_THRESHOLD_SECONDS, HOLD_LOOKAHEAD_BOOST is added to
# the effective task value.  This converts NO_SENSOR → HOLD when an orbital
# window is imminent, signalling that waiting is preferable to declaring no
# sensor available.
HOLD_LOOKAHEAD_THRESHOLD_SECONDS = 1800.0  # 30 min: boost fires if pass ≤ 30 min away
HOLD_LOOKAHEAD_BOOST = 0.15               # added to task-value total when boost fires

# Minimum satellite elevation above the observer's horizon for orbital sensor access.
# Sensors below this angle are considered out of view (degrees).
SENSOR_MIN_ELEVATION_DEG = 10.0

# ---------------------------------------------------------------------------
# AIS staleness confidence decay
# ---------------------------------------------------------------------------

# Observation gaps shorter than this are treated as normal; no penalty applied.
AIS_STALE_GAP_SECONDS = 3600.0

# Exponential decay rate (per second) applied to confidence for gaps that
# exceed AIS_STALE_GAP_SECONDS.  At 1e-4/s: a 1-hour excess ≈ 70% of base,
# 3-hour excess ≈ 34% of base, 7-hour excess ≈ 8% of base (hits floor).
AIS_STALE_CONFIDENCE_DECAY_RATE = 1e-4

# Confidence floor — decay never pushes custody confidence below this value.
AIS_MIN_STALE_CONFIDENCE = 0.1

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
# Rendezvous detection thresholds
# ---------------------------------------------------------------------------

# Both vessels must be within this distance (km) to count as a dwell step.
RENDEZVOUS_PROXIMITY_KM = 5.0

# Minimum consecutive dwell steps (including the current step) required to
# emit a RendezvousEvent.  With dt_hours=1, this means at least 2 consecutive
# hours of close proximity — enough to distinguish sustained contact from a
# momentary passing encounter.
RENDEZVOUS_MIN_DWELL_STEPS = 2

# ---------------------------------------------------------------------------
# Compound behavior thresholds
# ---------------------------------------------------------------------------

# Minimum sensitive_zone detector score for zone-proximity compound rules.
# Below this value the vessel is considered too far from a zone to warrant
# a compound signal.
ZONE_COMPOUND_MIN_SCORE = 0.5
