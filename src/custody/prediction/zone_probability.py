"""
Zone-crossing probability estimation.

Uses dead-reckoning path sampling and heading-alignment heuristics to
estimate the probability that a vessel will enter a given zone bounding box
within a planning horizon.
"""
from __future__ import annotations

import math

from custody.prediction.trajectory import project_position


# Time resolution for path sampling (hours between projection steps)
_STEP_HOURS = 0.5


def _point_in_zone(lat: float, lon: float, zone) -> bool:
    """Return True if (lat, lon) is inside zone's bounding box."""
    return (
        zone.min_lat <= lat <= zone.max_lat
        and zone.min_lon <= lon <= zone.max_lon
    )


def _zone_center(zone) -> tuple[float, float]:
    """Return the geographic centre of the zone bounding box."""
    return (
        (zone.min_lat + zone.max_lat) / 2.0,
        (zone.min_lon + zone.max_lon) / 2.0,
    )


def _heading_alignment(
    lat: float,
    lon: float,
    heading_deg: float,
    zone_center_lat: float,
    zone_center_lon: float,
) -> float:
    """Dot product of heading unit vector vs bearing-to-zone-centre unit vector.

    Returns a value in [0.3, 1.0]:
      1.0 = heading directly at zone centre
      0.5 = heading 90° off
      0.3 = heading away (clamped floor)
    """
    # Bearing from current position to zone centre (flat-earth)
    dlat = zone_center_lat - lat
    dlon = zone_center_lon - lon
    if abs(dlat) < 1e-10 and abs(dlon) < 1e-10:
        # Already at zone centre
        return 1.0

    bearing_rad = math.atan2(dlon, dlat)  # 0 = north, positive = clockwise
    heading_rad = math.radians(heading_deg)

    # Dot product of unit vectors
    dot = math.cos(heading_rad) * math.cos(bearing_rad) + math.sin(heading_rad) * math.sin(bearing_rad)

    # Clamp to [0.3, 1.0]: ensures a residual even for moderate misalignments
    return max(0.3, min(1.0, dot))


def compute_zone_crossing_probability(
    lat: float,
    lon: float,
    speed_knots: float,
    heading_deg: float,
    zone,
    horizon_hours: float = 6.0,
    custody_confidence: float = 1.0,
) -> tuple[float, float | None]:
    """Estimate the probability that a vessel enters *zone* within *horizon_hours*.

    Uses path sampling at _STEP_HOURS intervals.  When the projected path
    intersects the zone bounding box, a heuristic probability is computed from:
      - proximity/timing within the horizon (base probability)
      - heading alignment toward the zone centre
      - custody confidence (uncertainty scales down probability)

    Args:
        lat:                Current latitude (degrees).
        lon:                Current longitude (degrees).
        speed_knots:        Current speed (knots).
        heading_deg:        Current heading (degrees true).
        zone:               Zone object with min_lat, max_lat, min_lon, max_lon.
        horizon_hours:      Planning horizon in hours.
        custody_confidence: Track confidence in [0, 1].

    Returns:
        (probability, time_to_zone_hours) where time_to_zone_hours is None
        if no path intersection was detected within the horizon.
    """
    # If already inside zone
    if _point_in_zone(lat, lon, zone):
        return min(1.0, 0.95 * custody_confidence), 0.0

    zone_clat, zone_clon = _zone_center(zone)
    alignment = _heading_alignment(lat, lon, heading_deg, zone_clat, zone_clon)

    # Stationary or very slow vessel — limited trajectory confidence
    if speed_knots < 0.5:
        # Compute distance to zone (rough flat-earth)
        dlat = max(0.0, zone.min_lat - lat, lat - zone.max_lat)
        dlon = max(0.0, zone.min_lon - lon, lon - zone.max_lon)
        dist_deg = math.hypot(dlat, dlon)
        # Convert degrees to km (rough)
        dist_km = dist_deg * 111.12
        residual = max(0.05, 0.10 - dist_km * 0.002)
        return residual * custody_confidence, None

    # Sample projected positions at each time step
    time_to_entry: float | None = None
    steps = max(1, int(horizon_hours / _STEP_HOURS))

    for i in range(1, steps + 1):
        t = i * _STEP_HOURS
        plat, plon = project_position(lat, lon, speed_knots, heading_deg, t)
        if _point_in_zone(plat, plon, zone):
            time_to_entry = t
            break

    if time_to_entry is not None:
        # Path intersects zone within horizon
        time_fraction = time_to_entry / horizon_hours  # 0 = imminent, 1 = at horizon

        if time_fraction <= 2.0 / horizon_hours:
            # Within first 2 hours
            base = 0.9
        elif time_fraction <= 1.0:
            base = 0.7
        else:
            base = 0.4

        probability = base * alignment * custody_confidence
        return round(min(1.0, max(0.0, probability)), 4), time_to_entry

    # No intersection detected — residual probability based on distance
    # Compute approximate distance from current position to nearest zone edge (km)
    dlat = max(0.0, zone.min_lat - lat, lat - zone.max_lat)
    dlon = max(0.0, zone.min_lon - lon, lon - zone.max_lon)
    dist_deg = math.hypot(dlat, dlon)
    dist_km = dist_deg * 111.12

    # Residual: 0.15 at 0 km separation, 0.05 at large distances
    residual = max(0.05, min(0.15, 0.15 - dist_km * 0.001))
    # Scale down if heading strongly away
    residual *= alignment
    residual *= custody_confidence

    return round(min(1.0, max(0.0, residual)), 4), None
