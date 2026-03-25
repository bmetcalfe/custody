"""
Orbital mechanics helpers for the custody package.

Provides TLE parsing, SGP4 propagation, and satellite-visibility checks
from a ground observer.  This module is intentionally self-contained:
it does not import any other custody module and has no side effects on
the rest of the pipeline.

Coordinate conventions
----------------------
- TLE propagation (via sgp4) yields position in the TEME frame (km).
- TEME is rotated to ECEF using Greenwich Mean Sidereal Time (GMST).
- Observer coordinates are geodetic latitude / longitude (degrees).
- Elevation is measured from the observer's local horizon (positive up).
- Azimuth is measured clockwise from geographic North (0°–360°).

Prototype assumptions
---------------------
- Earth is modelled as a sphere (R = 6371 km).  Oblateness is ignored.
- GMST is computed from the standard IAU 1982 formula; accuracy is
  ~0.1° for dates within a few years of J2000.0.
- UTC and UT1 are treated as equal (difference < 1 second).
- TLE epoch validity is the caller's responsibility; no range check is
  performed before propagation.
"""

import math
from dataclasses import dataclass
from datetime import datetime

from sgp4.api import Satrec

# Mean Earth radius used for observer ECEF conversion (km).
_EARTH_RADIUS_KM = 6371.0


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SatellitePass:
    """Describes one satellite pass over a fixed ground observer.

    Attributes:
        rise_time:         UTC time when the satellite rises above
                           the configured minimum elevation.
        set_time:          UTC time when the satellite sets below the
                           configured minimum elevation.
        max_elevation_deg: Peak elevation angle during the pass (degrees).
    """
    rise_time: datetime
    set_time: datetime
    max_elevation_deg: float


# ---------------------------------------------------------------------------
# TLE loading
# ---------------------------------------------------------------------------

def load_tle(name: str, line1: str, line2: str) -> Satrec:
    """Parse a two-line element set and return an SGP4 satellite record.

    Args:
        name:  Satellite name or identifier (informational; not used by SGP4).
        line1: TLE line 1, exactly 69 characters, starting with "1 ".
        line2: TLE line 2, exactly 69 characters, starting with "2 ".

    Returns:
        sgp4.api.Satrec object ready for propagation via satellite_azel()
        or is_in_view().
    """
    return Satrec.twoline2rv(line1, line2)


# ---------------------------------------------------------------------------
# Internal coordinate helpers
# ---------------------------------------------------------------------------

def _jd_from_datetime(t: datetime) -> tuple[float, float]:
    """Convert a UTC datetime to a Julian Date split into (whole, fraction).

    Splitting into whole + fractional parts preserves numerical precision
    for the SGP4 propagator, which expects this two-part representation.

    Args:
        t: UTC-aware datetime.  If timezone-naive, UTC is assumed.

    Returns:
        (jd_whole, jd_frac) where jd_whole + jd_frac == full Julian Date.
    """
    # J2000.0 epoch: 2000-01-01 12:00:00 UTC = JD 2451545.0
    j2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=t.tzinfo)
    delta_days = (t - j2000).total_seconds() / 86400.0
    jd_full = 2451545.0 + delta_days
    # Split: whole part is the nearest 0.5-day boundary (noon or midnight)
    jd_whole = math.floor(jd_full) + 0.5
    jd_frac = jd_full - jd_whole
    return jd_whole, jd_frac


def _gmst_rad(jd: float) -> float:
    """Compute Greenwich Mean Sidereal Time (radians) from a Julian Date.

    Uses the IAU 1982 formula.  Accuracy is ~0.1° for dates within a few
    decades of J2000.0, which is sufficient for prototype visibility checks.

    Args:
        jd: Julian Date (UT1, treated as UTC for prototype purposes).

    Returns:
        GMST in radians, in the range [0, 2π).
    """
    t = (jd - 2451545.0) / 36525.0          # Julian centuries since J2000.0
    gmst_deg = (
        280.46061837
        + 360.98564736629 * (jd - 2451545.0)
        + 0.000387933 * t ** 2
        - t ** 3 / 38710000.0
    )
    return math.radians(gmst_deg % 360.0)


def _teme_to_ecef(
    r_teme: tuple[float, float, float],
    gmst: float,
) -> tuple[float, float, float]:
    """Rotate a position vector from TEME to ECEF frame.

    The rotation is a simple Z-axis rotation by the Greenwich Sidereal
    Angle.  The polar-motion correction is omitted (prototype accuracy).

    Args:
        r_teme: (x, y, z) position in TEME frame (km).
        gmst:   Greenwich Mean Sidereal Time in radians.

    Returns:
        (x, y, z) position in ECEF frame (km).
    """
    x, y, z = r_teme
    cos_g = math.cos(gmst)
    sin_g = math.sin(gmst)
    return (
        x * cos_g + y * sin_g,
        -x * sin_g + y * cos_g,
        z,
    )


def _ecef_to_azel(
    observer_lat_deg: float,
    observer_lon_deg: float,
    r_ecef: tuple[float, float, float],
) -> tuple[float, float]:
    """Compute azimuth and elevation from an ECEF observer to an ECEF point.

    Observer position is computed from geodetic lat/lon using a spherical
    Earth model (R = _EARTH_RADIUS_KM).

    Args:
        observer_lat_deg: Observer geodetic latitude (degrees, +N).
        observer_lon_deg: Observer geodetic longitude (degrees, +E).
        r_ecef:           Satellite position in ECEF frame (km).

    Returns:
        (azimuth_deg, elevation_deg) where azimuth is clockwise from North,
        in [0°, 360°), and elevation is in [-90°, +90°].
    """
    lat = math.radians(observer_lat_deg)
    lon = math.radians(observer_lon_deg)

    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)

    # Observer position in ECEF (spherical Earth)
    obs_x = _EARTH_RADIUS_KM * cos_lat * cos_lon
    obs_y = _EARTH_RADIUS_KM * cos_lat * sin_lon
    obs_z = _EARTH_RADIUS_KM * sin_lat

    # Range vector: satellite minus observer
    dx = r_ecef[0] - obs_x
    dy = r_ecef[1] - obs_y
    dz = r_ecef[2] - obs_z

    range_km = math.sqrt(dx * dx + dy * dy + dz * dz)
    if range_km < 1e-9:
        # Observer is at the satellite position
        return 0.0, 90.0

    # Project range vector onto topocentric (North, East, Up) unit vectors.
    #
    # North = (-sin_lat·cos_lon, -sin_lat·sin_lon,  cos_lat)
    # East  = (-sin_lon,          cos_lon,           0      )
    # Up    = ( cos_lat·cos_lon,  cos_lat·sin_lon,  sin_lat )
    north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    east = -sin_lon * dx + cos_lon * dy
    up = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz

    elevation_deg = math.degrees(math.asin(up / range_km))
    azimuth_deg = math.degrees(math.atan2(east, north)) % 360.0

    return azimuth_deg, elevation_deg


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def satellite_azel(
    satrec: Satrec,
    observer_lat_deg: float,
    observer_lon_deg: float,
    t: datetime,
) -> tuple[float, float]:
    """Compute azimuth and elevation of a satellite from a ground observer.

    Propagates the satellite to time *t* using SGP4, converts its TEME
    position to ECEF, then computes the topocentric azimuth and elevation
    as seen from the observer.

    Args:
        satrec:           SGP4 satellite record from load_tle().
        observer_lat_deg: Observer geodetic latitude in degrees (+N).
        observer_lon_deg: Observer geodetic longitude in degrees (+E).
        t:                UTC observation time.

    Returns:
        (azimuth_deg, elevation_deg) tuple.

        Returns (0.0, -90.0) as a safe below-horizon fallback when SGP4
        propagation fails (error code != 0), which can happen if the
        satellite has decayed or the requested time is far outside the
        TLE's valid range.
    """
    jd, fr = _jd_from_datetime(t)
    error_code, r_teme, _ = satrec.sgp4(jd, fr)

    if error_code != 0:
        # Propagation failed — return a safe below-horizon sentinel
        return 0.0, -90.0

    gmst = _gmst_rad(jd + fr)
    r_ecef = _teme_to_ecef(r_teme, gmst)
    return _ecef_to_azel(observer_lat_deg, observer_lon_deg, r_ecef)


def is_in_view(
    satrec: Satrec,
    observer_lat_deg: float,
    observer_lon_deg: float,
    t: datetime,
    min_elevation_deg: float = 10.0,
) -> bool:
    """Return True when a satellite is above the minimum elevation threshold.

    Args:
        satrec:             SGP4 satellite record from load_tle().
        observer_lat_deg:   Observer geodetic latitude in degrees (+N).
        observer_lon_deg:   Observer geodetic longitude in degrees (+E).
        t:                  UTC observation time.
        min_elevation_deg:  Horizon mask angle in degrees (default 10°).
                            Satellites at or below this elevation are
                            considered not in view.

    Returns:
        True if elevation strictly exceeds min_elevation_deg; False otherwise.
        Always False when SGP4 propagation fails (fallback elevation = -90°).
    """
    _, elevation_deg = satellite_azel(satrec, observer_lat_deg, observer_lon_deg, t)
    return elevation_deg > min_elevation_deg


def satellite_subpoint(
    satrec: Satrec,
    t: datetime,
) -> tuple[float, float] | None:
    """Return the sub-satellite point (lat_deg, lon_deg) at time t.

    Computes the geocentric latitude and longitude of the point on Earth's
    surface directly below the satellite.  Uses the same spherical Earth
    model as the rest of this module.

    Args:
        satrec: SGP4 satellite record from load_tle().
        t:      UTC observation time.

    Returns:
        (latitude_deg, longitude_deg) in degrees, or None if SGP4
        propagation fails.
    """
    jd, fr = _jd_from_datetime(t)
    error_code, r_teme, _ = satrec.sgp4(jd, fr)
    if error_code != 0:
        return None
    gmst = _gmst_rad(jd + fr)
    x, y, z = _teme_to_ecef(r_teme, gmst)
    lat_deg = math.degrees(math.atan2(z, math.sqrt(x * x + y * y)))
    lon_deg = math.degrees(math.atan2(y, x))
    return lat_deg, lon_deg
