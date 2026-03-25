"""
Entity-level prediction: dead-reckoning + zone probability + risk forecast.

Composes the trajectory, zone_probability, and risk_forecast modules into a
single predict_entity() call that returns a Prediction dataclass.
"""
from __future__ import annotations

from dataclasses import dataclass

from custody.prediction.trajectory import project_position
from custody.prediction.zone_probability import compute_zone_crossing_probability
from custody.prediction.risk_forecast import forecast_anomaly_score


@dataclass(frozen=True)
class Prediction:
    """Advisory prediction for one entity at one timestep.

    All fields are deterministic for a given input.  This object is
    intentionally narrow: it supplements current-evidence scores rather
    than replacing them.

    Attributes:
        entity_id:             Entity identifier.
        horizon_hours:         Planning horizon used (hours).
        future_lat:            Dead-reckoned latitude at horizon.
        future_lon:            Dead-reckoned longitude at horizon.
        zone_probability:      Probability of zone crossing in [0, 1].
        time_to_zone_hours:    Estimated hours to zone entry; None if not
                               approaching any zone.
        future_anomaly:        Forecast anomaly score in [0, 1].
        prediction_confidence: Confidence in this prediction in [0, 1];
                               lower when custody is poor or horizon is long.
        prediction_reason:     Human-readable explanation.
    """
    entity_id: str
    horizon_hours: float
    future_lat: float
    future_lon: float
    zone_probability: float
    time_to_zone_hours: float | None
    future_anomaly: float
    prediction_confidence: float
    prediction_reason: str


def predict_entity(
    entity_id: str,
    lat: float,
    lon: float,
    speed_knots: float,
    heading_deg: float,
    current_anomaly: float,
    custody_confidence: float,
    zones: list,
    horizon_hours: float = 6.0,
) -> Prediction:
    """Generate an advisory prediction for one entity.

    Runs dead-reckoning, zone-crossing probability estimation, and anomaly
    forecasting.  When multiple zones are configured the zone with the
    highest crossing probability is used for the prediction reason.

    Args:
        entity_id:          Entity identifier.
        lat:                Current latitude (degrees).
        lon:                Current longitude (degrees).
        speed_knots:        Current speed (knots).
        heading_deg:        Current heading (degrees true).
        current_anomaly:    Current anomaly score.
        custody_confidence: Track confidence in [0, 1].
        zones:              List of Zone objects (from custody.config).
        horizon_hours:      Planning horizon in hours.

    Returns:
        Prediction dataclass with all advisory fields populated.
    """
    # Dead-reckoning projection to end of horizon
    future_lat, future_lon = project_position(
        lat, lon, speed_knots, heading_deg, horizon_hours
    )

    # prediction_confidence decays with longer horizon and lower custody
    prediction_confidence = round(
        custody_confidence * min(1.0, 2.0 / max(horizon_hours, 1.0)),
        4,
    )
    prediction_confidence = max(0.0, min(1.0, prediction_confidence))

    # Evaluate all zones; keep the one with highest probability
    best_zone_name: str | None = None
    best_prob: float = 0.0
    best_tte: float | None = None

    for zone in zones:
        prob, tte = compute_zone_crossing_probability(
            lat, lon, speed_knots, heading_deg, zone,
            horizon_hours=horizon_hours,
            custody_confidence=custody_confidence,
        )
        if prob > best_prob:
            best_prob = prob
            best_tte = tte
            best_zone_name = zone.name

    # Forecast anomaly score
    future_anomaly = forecast_anomaly_score(
        current_anomaly=current_anomaly,
        zone_probability=best_prob,
        time_to_zone_hours=best_tte,
        horizon_hours=horizon_hours,
    )

    # Build human-readable reason
    reason = _build_reason(
        speed_knots=speed_knots,
        zone_name=best_zone_name,
        zone_probability=best_prob,
        time_to_zone_hours=best_tte,
        future_anomaly=future_anomaly,
        current_anomaly=current_anomaly,
    )

    return Prediction(
        entity_id=entity_id,
        horizon_hours=horizon_hours,
        future_lat=round(future_lat, 6),
        future_lon=round(future_lon, 6),
        zone_probability=round(best_prob, 4),
        time_to_zone_hours=best_tte,
        future_anomaly=future_anomaly,
        prediction_confidence=prediction_confidence,
        prediction_reason=reason,
    )


def _build_reason(
    speed_knots: float,
    zone_name: str | None,
    zone_probability: float,
    time_to_zone_hours: float | None,
    future_anomaly: float,
    current_anomaly: float,
) -> str:
    """Build a human-readable prediction reason string."""
    if speed_knots < 0.5:
        return "Stationary or slow; limited trajectory confidence."

    if zone_name and time_to_zone_hours is not None and zone_probability > 0.5:
        if time_to_zone_hours <= 0.1:
            return (
                f"Currently inside or entering {zone_name}; "
                f"future risk elevated."
            )
        return (
            f"Projected to enter {zone_name} in {time_to_zone_hours:.1f}h; "
            f"future risk elevated."
        )

    if zone_name and zone_probability > 0.3:
        return (
            f"On course toward {zone_name} "
            f"(zone probability {zone_probability:.0%})."
        )

    if future_anomaly > current_anomaly * 1.1:
        return (
            f"Anomaly trending up; no direct zone approach detected."
        )

    return "No zone approach detected; future anomaly near current level."
