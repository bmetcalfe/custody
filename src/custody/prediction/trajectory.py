"""
Dead-reckoning trajectory projection.

Uses a simple flat-earth approximation suitable for short horizons at
maritime scales (up to a few hundred km).
"""
from __future__ import annotations

import math


def project_position(
    lat: float,
    lon: float,
    speed_knots: float,
    heading_deg: float,
    dt_hours: float,
) -> tuple[float, float]:
    """Linear dead-reckoning projection.

    Projects a vessel forward in time assuming constant speed and heading.
    Uses a flat-earth approximation: accurate enough for horizons up to ~6h
    at typical maritime speeds.

    Args:
        lat:          Current latitude in degrees.
        lon:          Current longitude in degrees.
        speed_knots:  Speed in knots.
        heading_deg:  Heading in degrees true (0 = north, 90 = east).
        dt_hours:     Time to project forward in hours.

    Returns:
        (future_lat, future_lon) clamped/wrapped to valid geographic ranges.
    """
    if dt_hours == 0.0 or speed_knots == 0.0:
        return lat, lon

    heading_rad = math.radians(heading_deg)
    lat_rad = math.radians(lat)

    # Distance travelled in km
    dist_km = speed_knots * 1.852 * dt_hours

    # Flat-earth offsets
    # North component: dist * cos(heading) in km → degrees latitude
    lat_offset = (dist_km * math.cos(heading_rad)) / 111.12

    # East component: dist * sin(heading) in km → degrees longitude
    # Divide by cos(lat) to account for meridian convergence
    cos_lat = math.cos(lat_rad)
    if abs(cos_lat) < 1e-10:
        lon_offset = 0.0
    else:
        lon_offset = (dist_km * math.sin(heading_rad)) / (111.12 * cos_lat)

    future_lat = lat + lat_offset
    future_lon = lon + lon_offset

    # Clamp latitude to [-90, 90]
    future_lat = max(-90.0, min(90.0, future_lat))

    # Wrap longitude to [-180, 180]
    future_lon = ((future_lon + 180.0) % 360.0) - 180.0

    return future_lat, future_lon
