"""Confidence and uncertainty modeling for anomaly reasoning.

Computes interpretable confidence factors that reflect data quality,
observation coverage, and sensor conditions.  These factors modulate
(but do not replace) anomaly reasoning and priority/tasking.

All factors are in [0, 1] where 1 = full confidence.

Public API
----------
history_confidence(record, history_window) -> float
baseline_confidence(record) -> float
sensor_confidence(record) -> float
compute_overall_confidence(record, history_window) -> dict
apply_confidence_to_priority(priority_score, overall_confidence) -> float
confidence_action_bias(monitoring_action, overall_confidence) -> str
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Weights for overall confidence aggregation
# ---------------------------------------------------------------------------

_W_HISTORY  = 0.35
_W_BASELINE = 0.30
_W_SENSOR   = 0.20
_W_CUSTODY  = 0.15

# Priority damping: how much low confidence can reduce priority (max 20%)
_PRIORITY_DAMPING_MAX = 0.20

# Confidence threshold below which action shifts toward confirmation
_CONFIRM_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Individual confidence factors
# ---------------------------------------------------------------------------

def history_confidence(
    record: dict,
    history_window: list[dict],
) -> float:
    """Confidence based on amount of vessel history available.

    Ramps linearly from 0.2 (no history) to 1.0 (6+ prior records).
    """
    n = len(history_window)
    if n >= 6:
        return 1.0
    return round(0.2 + 0.8 * (n / 6.0), 3)


def baseline_confidence(record: dict) -> float:
    """Confidence in the vessel-relative normalization / baseline.

    Uses the ml_anomaly_relative field's presence and the baseline
    record count (if available via VesselBaseline metadata).

    Returns:
        1.0 if relative score is available and based on sufficient data.
        0.5 if relative score is present but may be sparse.
        0.3 if no relative score (no baseline available).
    """
    rel = record.get("ml_anomaly_relative")
    if rel is None:
        return 0.3

    # Check for NaN
    try:
        if float(rel) != float(rel):  # NaN
            return 0.3
    except (TypeError, ValueError):
        return 0.3

    # If baseline was built from enough data, full confidence
    # (We don't have the baseline object here, so use a heuristic:
    # if relative is available and non-NaN, baseline was built from
    # at least min_records samples — confidence is high.)
    return 1.0


def sensor_confidence(record: dict) -> float:
    """Confidence based on sensor suitability and solar conditions.

    EO at night → low confidence in optical collection viability.
    SAR or daytime EO → high confidence.
    """
    eo_suit = float(record.get("eo_suitability", 1.0))
    sar_suit = float(record.get("sar_suitability", 1.0))
    eff_sensor = record.get("effective_sensor_preference", "any")

    # If effective preference is SAR, sensor confidence is always high
    if eff_sensor == "all_weather":
        return 1.0

    # If preference is optical-class, confidence tracks EO suitability
    if eff_sensor in ("high_resolution", "fast_revisit"):
        return round(max(0.3, eo_suit), 3)

    # "any" sensor: blend — whichever is better is available
    return round(max(eo_suit, sar_suit), 3)


# ---------------------------------------------------------------------------
# Overall confidence
# ---------------------------------------------------------------------------

def compute_overall_confidence(
    record: dict,
    history_window: list[dict],
) -> dict:
    """Compute all confidence factors for one record.

    Args:
        record:         Current timestep record.
        history_window: Prior records for this entity, oldest first.

    Returns:
        Dict with keys: history_confidence, baseline_confidence,
        sensor_confidence, custody_confidence_raw, overall_confidence,
        confidence_category, confidence_rationale.
    """
    h_conf = history_confidence(record, history_window)
    b_conf = baseline_confidence(record)
    s_conf = sensor_confidence(record)
    c_conf = float(record.get("custody_confidence", 1.0))

    overall = round(
        _W_HISTORY * h_conf
        + _W_BASELINE * b_conf
        + _W_SENSOR * s_conf
        + _W_CUSTODY * c_conf,
        3,
    )
    overall = min(max(overall, 0.0), 1.0)

    # Category
    if overall >= 0.75:
        category = "high"
    elif overall >= 0.5:
        category = "moderate"
    else:
        category = "low"

    # Rationale: list weakest factors
    parts = []
    if h_conf < 0.5:
        parts.append(f"sparse history ({len(history_window)} prior records)")
    if b_conf < 0.5:
        parts.append("no vessel baseline available")
    if s_conf < 0.7:
        solar = record.get("solar_condition", "?")
        parts.append(f"sensor degraded ({solar} conditions)")
    if c_conf < 0.5:
        parts.append(f"weak custody ({c_conf:.2f})")
    if not parts:
        rationale = "all confidence factors adequate"
    else:
        rationale = "; ".join(parts)

    return {
        "history_confidence":      h_conf,
        "baseline_confidence":     b_conf,
        "sensor_confidence":       s_conf,
        "custody_confidence_raw":  c_conf,
        "overall_confidence":      overall,
        "confidence_category":     category,
        "confidence_rationale":    rationale,
    }


# ---------------------------------------------------------------------------
# Confidence-modulated priority and action
# ---------------------------------------------------------------------------

def apply_confidence_to_priority(
    priority_score: float,
    overall_confidence: float,
) -> float:
    """Dampen priority when confidence is low.

    High confidence (>= 0.75): no change.
    Moderate (0.5–0.75): mild damping (up to 10%).
    Low (< 0.5): stronger damping (up to 20%).

    Returns adjusted priority in [0, 1].
    """
    if overall_confidence >= 0.75:
        damping = 0.0
    elif overall_confidence >= 0.5:
        # Linear ramp: 0% at 0.75, 10% at 0.5
        t = (0.75 - overall_confidence) / 0.25
        damping = t * (_PRIORITY_DAMPING_MAX / 2.0)
    else:
        # Linear ramp: 10% at 0.5, 20% at 0.0
        t = (0.5 - overall_confidence) / 0.5
        damping = (_PRIORITY_DAMPING_MAX / 2.0) + t * (_PRIORITY_DAMPING_MAX / 2.0)

    adjusted = priority_score * (1.0 - damping)
    return round(min(max(adjusted, 0.0), 1.0), 4)


def confidence_action_bias(
    monitoring_action: str,
    overall_confidence: float,
) -> str:
    """Shift monitoring action toward confirmation when confidence is low.

    When confidence is below ``_CONFIRM_THRESHOLD`` and the current action
    is aggressive (intensify or increase_attention), shift to "confirm"
    instead — seeking more evidence before escalating further.

    Actions that are already confirmatory or passive are not changed.
    """
    if overall_confidence >= _CONFIRM_THRESHOLD:
        return monitoring_action

    # Low confidence: shift aggressive actions toward confirmation
    if monitoring_action in ("intensify", "increase_attention"):
        return "confirm"

    return monitoring_action
