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
# Collection failure feedback
# ---------------------------------------------------------------------------

# Uncertainty penalty (km) added after the first failed collection attempt.
# Subsequent consecutive failures receive diminishing penalties:
#   penalty = FAILURE_UNCERTAINTY_PENALTY_KM / (1 + consecutive_failures)
# A successful fast_revisit reduces uncertainty by ~50% (e.g. 20 km → 10 km);
# this penalty is less than half that (~5 km), so a single failure is moderate.
FAILURE_UNCERTAINTY_PENALTY_KM = 5.0

# Hard cap on failure-driven uncertainty (km).  Failures alone can push an
# entity to STALE (_UNC_STALE_KM = 50 km) but never reach LOST (90 km).
FAILURE_UNCERTAINTY_CAP_KM = 120.0

# Per-failure additive boost to task_value, capped at 3 failures (= 0.24).
# One failure (0.08) does not break HOLD on its own (threshold 0.2);
# two consecutive failures (0.16) make HOLD very unlikely when any anomaly
# is present; three (0.24) overcome HOLD for low-anomaly entities.
FAILURE_URGENCY_BOOST = 0.08

# Maximum number of consecutive failures that contribute to urgency boost.
FAILURE_URGENCY_MAX_COUNT = 3

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

# ---------------------------------------------------------------------------
# Swath / grouped collection
# ---------------------------------------------------------------------------

# Weight applied to covered-target bonus when computing swath_task_value.
# Each covered non-center target contributes its priority_score × this weight.
SWATH_BONUS_WEIGHT = 0.3

# Maximum swath uplift that can be added to a candidate's effective value.
# Prevents grouped value from completely dominating single-target scoring.
SWATH_UPLIFT_CAP = 0.25

# ---------------------------------------------------------------------------
# Fusion layer weights
# ---------------------------------------------------------------------------
#
# fused_score = W_HEURISTIC * anomaly_norm
#             + W_ML        * ml_anomaly_score
#             + W_COMPOUND  * top_compound_confidence
#             + W_CUSTODY   * (1 - custody_confidence)
#
# When W_ML = 0.0 the system behaves identically to the pre-ML baseline.
# The remaining weights (0.45 + 0.35 + 0.20 = 1.0) match the original
# formula exactly.  Enabling ML (e.g. W_ML = 0.15) should be accompanied
# by reducing W_HEURISTIC proportionally to keep the sum near 1.0.

FUSION_W_HEURISTIC = 0.45     # heuristic anomaly contribution (original: 0.45)
FUSION_W_ML        = 0.0      # ML anomaly contribution (default OFF)
FUSION_W_COMPOUND  = 0.35     # compound signal contribution (original: 0.35)
FUSION_W_CUSTODY   = 0.20     # custody weakness contribution (original: 0.20)

# ---------------------------------------------------------------------------
# Anomaly reasoning thresholds
# ---------------------------------------------------------------------------

# ML score threshold for "high" classification in agreement logic.
ML_ANOMALY_HIGH_THRESHOLD = 0.8

# Per-vessel relative score threshold (used when ML_SIGNAL_MODE includes
# relative scoring).
RELATIVE_ML_HIGH_THRESHOLD = 0.95

# Which ML signal to use for agreement/persistence/escalation reasoning.
#   "absolute"  — ml_anomaly_score >= ML_ANOMALY_HIGH_THRESHOLD
#   "relative"  — ml_anomaly_relative >= RELATIVE_ML_HIGH_THRESHOLD
#   "combined"  — max(ml_anomaly_score, ml_anomaly_relative) >= COMBINED_ML_HIGH_THRESHOLD
# Default "absolute" preserves pre-normalization behavior.
ML_SIGNAL_MODE = "absolute"

# Threshold for the combined signal (max of absolute and relative).
COMBINED_ML_HIGH_THRESHOLD = 0.8

# Legacy alias — kept for backward compatibility with existing tests.
# True is equivalent to ML_SIGNAL_MODE = "relative".
USE_RELATIVE_ML_THRESHOLD = False

# Heuristic anomaly score (normalised by CRITICAL_ANOMALY_THRESHOLD)
# threshold for "high" classification in agreement logic.
HEURISTIC_ANOMALY_HIGH_THRESHOLD = 0.5

# Minimum consecutive hours above ML threshold to qualify as "sustained".
SUSTAINED_ANOMALY_MIN_HOURS = 3

# Escalation boost added to fused_score per sustained hour (capped).
ESCALATION_BOOST_PER_HOUR = 0.02

# Maximum escalation boost from persistence alone.
ESCALATION_BOOST_CAP = 0.10
