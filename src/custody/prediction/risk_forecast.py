"""
Risk (anomaly score) forecasting.

Projects a future anomaly score from current evidence and predicted
zone-approach signals.  Intentionally simple: the current anomaly score
persists or grows when a zone approach is imminent; it decays slightly
when no new threat signal is present.
"""
from __future__ import annotations


def forecast_anomaly_score(
    current_anomaly: float,
    zone_probability: float,
    time_to_zone_hours: float | None,
    horizon_hours: float = 6.0,
) -> float:
    """Forecast the future anomaly score at the end of the planning horizon.

    Args:
        current_anomaly:    Current anomaly score (typically in [0, 3] but
                            the forecast is always clamped to [0, 1]).
        zone_probability:   Probability of zone crossing within horizon [0, 1].
        time_to_zone_hours: Estimated time to zone entry (None if no approach).
        horizon_hours:      Planning horizon in hours.

    Returns:
        Forecast anomaly score in [0, 1].
    """
    if zone_probability > 0.3 and time_to_zone_hours is not None:
        # Urgency grows as time to entry shrinks relative to the horizon
        urgency = 1.0 - (time_to_zone_hours / horizon_hours)  # 0..1; higher = sooner
        urgency = max(0.0, min(1.0, urgency))
        zone_boost = zone_probability * urgency * 0.8
        future_anomaly = min(1.0, current_anomaly + zone_boost)
    else:
        # No strong zone signal — anomaly decays slightly without new evidence
        future_anomaly = current_anomaly * 0.85

    return round(min(1.0, max(0.0, future_anomaly)), 4)
