"""Lightweight solar position and sensor suitability.

Computes sun elevation from timestamp + observer lat/lon using a
simplified astronomical algorithm (accurate to ~1 degree, sufficient
for day/twilight/night classification).

No external dependencies beyond the standard library and math.

Public API
----------
sun_elevation_deg(timestamp, lat, lon) -> float
solar_condition(timestamp, lat, lon) -> str
eo_suitability(sun_elev) -> float
sar_suitability() -> float
sensor_suitability(timestamp, lat, lon) -> dict
"""
from __future__ import annotations

import math
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Thresholds (degrees above horizon)
# ---------------------------------------------------------------------------

_DAY_THRESHOLD       =  6.0   # sun > 6° → full daylight (civil twilight ends)
_TWILIGHT_THRESHOLD  = -6.0   # sun in [-6°, 6°] → civil twilight
# sun < -6° → night


# ---------------------------------------------------------------------------
# Solar position (simplified)
# ---------------------------------------------------------------------------

def sun_elevation_deg(timestamp: datetime, lat: float, lon: float) -> float:
    """Approximate sun elevation angle in degrees.

    Uses the simplified solar position algorithm from the US Naval
    Observatory.  Accuracy is ~1° — sufficient for day/twilight/night
    classification but not for precise shadow analysis.

    Args:
        timestamp: UTC-aware datetime.
        lat:       Observer latitude in degrees (+N).
        lon:       Observer longitude in degrees (+E).

    Returns:
        Sun elevation in degrees.  Positive = above horizon.
    """
    # Ensure UTC
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    else:
        timestamp = timestamp.astimezone(timezone.utc)

    # Julian day number
    y = timestamp.year
    m = timestamp.month
    d = timestamp.day
    h = timestamp.hour + timestamp.minute / 60.0 + timestamp.second / 3600.0

    # Julian date (Meeus, Astronomical Algorithms)
    if m <= 2:
        y -= 1
        m += 12
    A = int(y / 100)
    B = 2 - A + int(A / 4)
    JD = int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + h / 24.0 + B - 1524.5

    # Days since J2000.0
    n = JD - 2451545.0

    # Mean longitude and anomaly of the Sun
    L = (280.460 + 0.9856474 * n) % 360.0    # mean longitude (deg)
    g = math.radians((357.528 + 0.9856003 * n) % 360.0)  # mean anomaly (rad)

    # Ecliptic longitude
    lam = math.radians(L + 1.915 * math.sin(g) + 0.020 * math.sin(2 * g))

    # Obliquity of ecliptic
    eps = math.radians(23.439 - 0.0000004 * n)

    # Right ascension and declination
    sin_lam = math.sin(lam)
    cos_lam = math.cos(lam)
    sin_eps = math.sin(eps)
    cos_eps = math.cos(eps)

    # Declination
    sin_dec = sin_eps * sin_lam
    dec = math.asin(sin_dec)

    # Right ascension
    ra = math.atan2(cos_eps * sin_lam, cos_lam)

    # Greenwich Mean Sidereal Time (hours)
    GMST = (18.697374558 + 24.06570982441908 * n) % 24.0

    # Local hour angle
    lst = GMST + lon / 15.0  # local sidereal time (hours)
    ha = math.radians((lst * 15.0 - math.degrees(ra)) % 360.0)

    # Elevation
    lat_rad = math.radians(lat)
    sin_elev = (math.sin(lat_rad) * math.sin(dec)
                + math.cos(lat_rad) * math.cos(dec) * math.cos(ha))
    elev = math.degrees(math.asin(max(-1.0, min(1.0, sin_elev))))

    return round(elev, 2)


# ---------------------------------------------------------------------------
# Solar condition
# ---------------------------------------------------------------------------

def solar_condition(timestamp: datetime, lat: float, lon: float) -> str:
    """Classify lighting conditions.

    Returns:
        ``"day"`` if sun > 6°,
        ``"twilight"`` if sun in [-6°, 6°],
        ``"night"`` if sun < -6°.
    """
    elev = sun_elevation_deg(timestamp, lat, lon)
    if elev >= _DAY_THRESHOLD:
        return "day"
    if elev >= _TWILIGHT_THRESHOLD:
        return "twilight"
    return "night"


# ---------------------------------------------------------------------------
# Sensor suitability
# ---------------------------------------------------------------------------

def eo_suitability(sun_elev: float) -> float:
    """EO/optical sensor suitability in [0, 1] based on sun elevation.

    - Full daylight (sun > 6°): 1.0
    - Civil twilight (sun in [-6°, 6°]): linear ramp from 0.2 to 1.0
    - Night (sun < -6°): 0.0
    """
    if sun_elev >= _DAY_THRESHOLD:
        return 1.0
    if sun_elev >= _TWILIGHT_THRESHOLD:
        # Linear interpolation: -6° → 0.2, +6° → 1.0
        t = (sun_elev - _TWILIGHT_THRESHOLD) / (_DAY_THRESHOLD - _TWILIGHT_THRESHOLD)
        return round(0.2 + 0.8 * t, 3)
    return 0.0


def sar_suitability() -> float:
    """SAR sensor suitability.  Day/night independent → always 1.0."""
    return 1.0


def sensor_suitability(timestamp: datetime, lat: float, lon: float) -> dict:
    """Compute all suitability fields for one observation.

    Returns:
        Dict with sun_elevation_deg, solar_condition, eo_suitability,
        sar_suitability.
    """
    elev = sun_elevation_deg(timestamp, lat, lon)
    return {
        "sun_elevation_deg": elev,
        "solar_condition":   solar_condition(timestamp, lat, lon),
        "eo_suitability":    eo_suitability(elev),
        "sar_suitability":   sar_suitability(),
    }
