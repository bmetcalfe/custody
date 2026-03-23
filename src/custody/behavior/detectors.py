from custody.config import PROXIMITY_CRITICAL_KM, PROXIMITY_WARNING_KM, ZONES, SLOW_SPEED_KMH
from custody.features.motion_features import (
    consecutive_slow_steps,
    heading_deviation,
    is_slow,
)
from custody.features.proximity_features import nearest_vessel_from_records
from custody.features.zone_features import distance_to_zone, is_inside_zone
from custody.models import DetectorResult

# Approximate km per degree at equatorial latitudes.
# Used only for human-readable evidence strings; not used in scoring.
_KM_PER_DEG = 111.0


def _zone_dict(zone) -> dict:
    """Convert a Zone dataclass to the dict format expected by zone_features."""
    return {
        "min_lat": zone.min_lat,
        "max_lat": zone.max_lat,
        "min_lon": zone.min_lon,
        "max_lon": zone.max_lon,
    }


def in_sensitive_zone(vessel) -> DetectorResult:
    """Return the highest-scoring sensitive-zone DetectorResult across all ZONES.

    Evaluates each zone independently using its own halo.  Returns the result
    with the maximum score; ties are broken by list order in ZONES.  If ZONES
    is empty, returns score 0.0.
    """
    if not ZONES:
        return DetectorResult(
            name="sensitive_zone",
            score=0.0,
            evidence="no zones configured",
        )

    best: DetectorResult | None = None

    for zone in ZONES:
        zd = _zone_dict(zone)
        if is_inside_zone(vessel, zd):
            result = DetectorResult(
                name="sensitive_zone",
                score=1.5,
                evidence=f"inside {zone.name}",
                metadata={"zone_name": zone.name},
            )
        else:
            dist = distance_to_zone(vessel, zd)
            if dist < zone.halo:
                # Linear ramp: 0.75 at the boundary, 0.0 at halo distance.
                score = 0.75 * (1.0 - dist / zone.halo)
                dist_km = dist * _KM_PER_DEG
                result = DetectorResult(
                    name="sensitive_zone",
                    score=score,
                    evidence=f"{dist_km:.1f} km from {zone.name} boundary",
                    metadata={"distance_to_zone_km": round(dist_km, 1), "zone_name": zone.name},
                )
            else:
                result = DetectorResult(
                    name="sensitive_zone",
                    score=0.0,
                    evidence=f"outside {zone.name} and halo",
                )

        if best is None or result.score > best.score:
            best = result

    return best  # type: ignore[return-value]  # guaranteed non-None: ZONES is non-empty


def loitering(vessel) -> DetectorResult:
    if not is_slow(vessel):
        return DetectorResult(
            name="loitering",
            score=0.0,
            evidence=f"speed {vessel.speed_kmh:.1f} km/h; above slow threshold",
        )

    # Count consecutive recent history steps that also showed slow speed.
    # Each entry is a HistoryEntry; speed_kmh is the instantaneous speed recorded
    # at observation time, independent of step size.
    slow_steps = consecutive_slow_steps(vessel.history, threshold=SLOW_SPEED_KMH, max_lookback=3)

    # Base 0.5 for any slow tick; +0.1 per additional confirmed slow hour, capped at 1.0.
    score = min(0.5 + slow_steps * 0.1, 1.0)

    if slow_steps == 0:
        evidence = (
            f"speed {vessel.speed_kmh:.1f} km/h below threshold; "
            "no confirmed slow history"
        )
    else:
        steps_label = f"{slow_steps} consecutive slow step{'s' if slow_steps != 1 else ''}"
        evidence = f"{steps_label}; speed {vessel.speed_kmh:.1f} km/h"

    return DetectorResult(
        name="loitering",
        score=score,
        evidence=evidence,
        metadata={"slow_steps": slow_steps},
    )


def route_deviation(vessel, baseline_heading=45) -> DetectorResult:
    # Use the vessel's commanded heading on its first observed step as the baseline.
    # This is the stored heading_deg from the HistoryEntry, not a derived course-over-ground.
    if vessel.history:
        baseline_heading = vessel.history[0].heading_deg

    # Shortest angular distance, handling the 0°/360° wraparound.
    diff = heading_deviation(vessel, baseline_heading)

    if diff > 60:
        score = 1.0
    elif diff > 30:
        score = 0.5
    else:
        score = 0.0

    return DetectorResult(
        name="route_deviation",
        score=score,
        evidence=f"heading deviation {diff:.1f}° from baseline {baseline_heading:.1f}°",
        metadata={
            "heading_deviation_deg": round(diff, 1),
            "baseline_heading_deg": round(baseline_heading, 1),
        },
    )


def vessel_proximity(
    lat: float,
    lon: float,
    vessel_id: str,
    others: list[dict],
) -> DetectorResult:
    """Score how close this vessel is to the nearest other vessel.

    Operates on raw lat/lon floats and plain record dicts — no Vessel objects
    required.  Self-comparison is excluded inside ``nearest_vessel_from_records``
    by matching on ``target_id``.

    This detector is intentionally separate from ``anomaly_breakdown`` so that
    the existing single-vessel pipeline is unaffected.  Callers that have
    multi-vessel context invoke it explicitly.

    Args:
        lat, lon:  Current position of the vessel being scored.
        vessel_id: ID of the vessel being scored (used to exclude self).
        others:    Co-temporal timeline records for all other known vessels.

    Returns:
        DetectorResult with name ``"vessel_proximity"``.
    """
    nearest_id, nearest_km = nearest_vessel_from_records(lat, lon, vessel_id, others)

    if nearest_id is None:
        return DetectorResult(
            name="vessel_proximity",
            score=0.0,
            evidence="no other vessels in context",
        )

    if nearest_km < PROXIMITY_CRITICAL_KM:
        return DetectorResult(
            name="vessel_proximity",
            score=1.0,
            evidence=f"within {nearest_km:.1f} km of vessel {nearest_id}",
            metadata={
                "nearest_vessel_id": nearest_id,
                "distance_km": round(nearest_km, 2),
                "threshold_km": PROXIMITY_CRITICAL_KM,
            },
        )

    if nearest_km < PROXIMITY_WARNING_KM:
        return DetectorResult(
            name="vessel_proximity",
            score=0.5,
            evidence=f"within {nearest_km:.1f} km of vessel {nearest_id} (warning range)",
            metadata={
                "nearest_vessel_id": nearest_id,
                "distance_km": round(nearest_km, 2),
                "threshold_km": PROXIMITY_WARNING_KM,
            },
        )

    return DetectorResult(
        name="vessel_proximity",
        score=0.0,
        evidence="no vessels within proximity threshold",
    )


def anomaly_breakdown(vessel) -> dict[str, DetectorResult]:
    return {
        "sensitive_zone": in_sensitive_zone(vessel),
        "loitering": loitering(vessel),
        "route_deviation": route_deviation(vessel),
    }


def anomaly_score(vessel) -> float:
    breakdown = anomaly_breakdown(vessel)
    return sum(r.score for r in breakdown.values())
