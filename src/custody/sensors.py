"""
Sensor opportunity scheduling for the custody package.

Two kinds of sensor are supported:

  Schedule-based (satellite_id=None)
    Always returned when the time-of-day schedule condition is met.
    Orbit access is never checked.  A1, B1, and C1 are schedule-based;
    their behaviour is identical to pre-Step-20 versions.

  Orbital (satellite_id=<str>)
    Only included in the returned list when observer_lat and observer_lon
    are both provided AND the satellite is above SENSOR_MIN_ELEVATION_DEG
    as seen from the observer at current_time.  When no observer position
    is given, orbital sensors are omitted entirely so that the returned
    list matches the pre-Step-20 result exactly.

Satellite catalog
-----------------
Six synthetic orbital assets are bundled; no live TLE fetch is performed.
Their orbital parameters were chosen to give meaningfully different coverage
patterns over the default simulation date (2026-03-23):

  EO-MIO-1  Medium-inclination LEO (51.6°, ~400 km).  High-resolution optical.
             RAAN 0°.  Passes equatorial observers ~11:33z.

  EO-MIO-2  Medium-inclination LEO (51.6°, ~400 km).  High-resolution optical.
             RAAN 220°.  Passes equatorial observers ~13:51z.

  EO-SSO-1  Sun-synchronous LEO (97.8°, ~600 km).  High-resolution optical.
             RAAN 320°.  Passes equatorial observers ~10:00z.

  EO-SSO-2  Sun-synchronous LEO (97.8°, ~600 km).  High-resolution optical.
             RAAN 100°.  Passes equatorial observers ~18:00z.

  SAR-1     Sun-synchronous LEO (97.8°, ~700 km).  All-weather (SAR).
             RAAN 200°.  Two passes: ~12:21z and ~13:59z.

  SAR-2     Sun-synchronous LEO (97.8°, ~700 km).  All-weather (SAR).
             RAAN 80°.  Two passes: ~16:29z and ~18:08z.
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from sgp4.api import Satrec

from custody.config import SENSOR_MIN_ELEVATION_DEG
from custody.orbit import is_in_view, load_tle


# ---------------------------------------------------------------------------
# Satellite TLE catalog
# ---------------------------------------------------------------------------

# EO-MIO-1: Medium-inclination optical asset (NORAD 99910).
# High-resolution optical LEO; inclination 51.6°, ~400 km.  RAAN 0°.
# Passes equatorial observers at ~11:33z on 2026-03-23.
_TLE_EO_MIO_1 = (
    "EO-MIO-1",
    "1 99910U 98067B   26082.50000000  .00000034  00000-0  25000-4 0  0002",
    "2 99910 51.6000   0.0000 0001000  90.0000   0.0000 15.50103472000000",
)

# EO-MIO-2: Second medium-inclination optical asset (NORAD 99911).
# High-resolution optical LEO; inclination 51.6°, ~400 km.  RAAN 220°.
# Passes equatorial observers at ~13:51z on 2026-03-23.
_TLE_EO_MIO_2 = (
    "EO-MIO-2",
    "1 99911U 98067C   26082.50000000  .00000034  00000-0  25000-4 0  0003",
    "2 99911 51.6000 220.0000 0001000  90.0000   0.0000 15.50103472000005",
)

# EO-SSO-1: First sun-synchronous optical asset (NORAD 99920).
# High-resolution optical LEO; inclination 97.8°, ~600 km.  RAAN 320°.
# Passes equatorial observers at ~10:00z on 2026-03-23.
_TLE_EO_SSO_1 = (
    "EO-SSO-1",
    "1 99920U 26010A   26082.50000000  .00000034  00000-0  25000-4 0  0002",
    "2 99920 97.8000 320.0000 0001000  90.0000   0.0000 14.88000000000001",
)

# EO-SSO-2: Second sun-synchronous optical asset (NORAD 99921).
# High-resolution optical LEO; inclination 97.8°, ~600 km.  RAAN 100°.
# Passes equatorial observers at ~18:00z on 2026-03-23.
_TLE_EO_SSO_2 = (
    "EO-SSO-2",
    "1 99921U 26010B   26082.50000000  .00000034  00000-0  25000-4 0  0003",
    "2 99921 97.8000 100.0000 0001000  90.0000   0.0000 14.88000000000008",
)

# SAR-1: First SAR asset (NORAD 99930).
# All-weather SSO LEO; inclination 97.8°, ~700 km.  RAAN 200°.
# Two passes over equatorial observers on 2026-03-23: ~12:21z and ~13:59z.
_TLE_SAR_1 = (
    "SAR-1",
    "1 99930U 26002A   26082.50000000  .00000034  00000-0  25000-4 0  0004",
    "2 99930 97.8000 200.0000 0001000  90.0000   0.0000 14.57650000000006",
)

# SAR-2: Second SAR asset (NORAD 99931).
# All-weather SSO LEO; inclination 97.8°, ~700 km.  RAAN 80°.
# Two passes over equatorial observers on 2026-03-23: ~16:29z and ~18:08z.
_TLE_SAR_2 = (
    "SAR-2",
    "1 99931U 26003A   26082.50000000  .00000034  00000-0  25000-4 0  0006",
    "2 99931 97.8000  80.0000 0001000  90.0000   0.0000 14.57650000000003",
)

# Parse TLEs once at import time.  Satrec objects are immutable and safe to
# share across threads.
_SATREC_CACHE: dict[str, Satrec] = {
    "EO-MIO-1": load_tle(*_TLE_EO_MIO_1),
    "EO-MIO-2": load_tle(*_TLE_EO_MIO_2),
    "EO-SSO-1": load_tle(*_TLE_EO_SSO_1),
    "EO-SSO-2": load_tle(*_TLE_EO_SSO_2),
    "SAR-1":    load_tle(*_TLE_SAR_1),
    "SAR-2":    load_tle(*_TLE_SAR_2),
}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class SensorOpportunity:
    """A single sensor opportunity available at a particular time.

    Attributes:
        sensor_id:      Unique sensor identifier (e.g. "A1", "SAT-A").
        sensor_type:    Sensor class string ("fast_revisit", "high_resolution",
                        "all_weather").
        success_prob:   Probability that a collection attempt succeeds [0, 1].
        resolution:     Qualitative resolution label ("medium", "high").
        cost:           Relative tasking cost (lower is cheaper).
        available_from: Start of the availability window.
        available_to:   End of the availability window.
        satellite_id:   Identifier of the satellite carrying this sensor, or
                        None for ground-based / schedule-only sensors.
                        Used internally to look up the orbital access filter;
                        None means the sensor is always available if the
                        schedule condition is satisfied.
    """
    sensor_id: str
    sensor_type: str
    success_prob: float
    resolution: str
    cost: float
    available_from: datetime
    available_to: datetime
    satellite_id: Optional[str] = None


@dataclass(frozen=True)
class PassWindow:
    """A forecast orbital access window for one satellite over one observer.

    Attributes:
        satellite_id:          Catalog identifier of the satellite.
        start_time:            UTC time when the satellite rises above
                               SENSOR_MIN_ELEVATION_DEG.  Equal to the
                               ``from_time`` argument when the satellite is
                               already in view at the start of the search.
        end_time:              UTC time when the satellite first drops below
                               SENSOR_MIN_ELEVATION_DEG (or the end of the
                               search horizon if it remains in view throughout).
        duration_seconds:      Length of the access window in seconds
                               (end_time - start_time).
        time_to_start_seconds: Seconds from the search start (``from_time``)
                               until the window opens.  Zero when already
                               in view.
    """
    satellite_id: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    time_to_start_seconds: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _satellite_in_view(
    satellite_id: str,
    observer_lat: float,
    observer_lon: float,
    t: datetime,
) -> bool:
    """Return True if the named satellite is above the minimum elevation at t."""
    satrec = _SATREC_CACHE.get(satellite_id)
    if satrec is None:
        # Unknown satellite → fail safe: not available
        return False
    return is_in_view(satrec, observer_lat, observer_lon, t,
                      min_elevation_deg=SENSOR_MIN_ELEVATION_DEG)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def orbital_satrecs() -> dict[str, "Satrec"]:
    """Return the catalog of orbital satellite records keyed by satellite ID.

    The returned dict is the module-level cache parsed at import time.
    Callers must not modify it.
    """
    return _SATREC_CACHE


def get_sensor_opportunities(
    current_time: datetime,
    observer_lat: Optional[float] = None,
    observer_lon: Optional[float] = None,
) -> list[SensorOpportunity]:
    """Return sensor opportunities available at current_time.

    Schedule-based sensors (satellite_id=None) are returned whenever their
    time-of-day schedule condition is satisfied, regardless of observer
    position.  Their behaviour is unchanged from pre-Step-20 versions.

    Orbital sensors (satellite_id != None) are only added to the result when
    *both* observer_lat and observer_lon are provided and the satellite is
    currently above SENSOR_MIN_ELEVATION_DEG from the observer.

    When observer_lat or observer_lon is None (the default), orbital sensors
    are omitted entirely, preserving exact backward-compatible behaviour.

    Args:
        current_time:  Timestamp of the current planning step.
        observer_lat:  Observer geodetic latitude in degrees (+N), or None.
        observer_lon:  Observer geodetic longitude in degrees (+E), or None.

    Returns:
        List of SensorOpportunity objects.
    """
    opportunities: list[SensorOpportunity] = []
    hour = current_time.hour

    # ── Schedule-based sensors ────────────────────────────────────────────
    # These are never orbit-filtered; they behave identically to pre-Step-20.

    # A1: fast revisit, cheap, good for rapid reacquisition
    if hour % 2 == 0:
        opportunities.append(
            SensorOpportunity(
                sensor_id="A1",
                sensor_type="fast_revisit",
                success_prob=0.85,
                resolution="medium",
                cost=1.0,
                available_from=current_time,
                available_to=current_time,
            )
        )

    # B1: high-resolution, less frequent, best for characterisation
    if hour in [13, 17]:
        opportunities.append(
            SensorOpportunity(
                sensor_id="B1",
                sensor_type="high_resolution",
                success_prob=0.65,
                resolution="high",
                cost=2.0,
                available_from=current_time,
                available_to=current_time,
            )
        )

    # C1: all-weather, medium revisit, resilient fallback
    if hour % 3 == 0:
        opportunities.append(
            SensorOpportunity(
                sensor_id="C1",
                sensor_type="all_weather",
                success_prob=0.75,
                resolution="medium",
                cost=1.5,
                available_from=current_time,
                available_to=current_time,
            )
        )

    # ── Orbital sensors ───────────────────────────────────────────────────
    # Only evaluated when observer position is provided.  When omitted, this
    # block is skipped entirely so the return value matches pre-Step-20.

    if observer_lat is not None and observer_lon is not None:

        # EO-MIO-1: medium-inclination optical asset (~400 km, 51.6° inc., RAAN 0°).
        if _satellite_in_view("EO-MIO-1", observer_lat, observer_lon, current_time):
            opportunities.append(
                SensorOpportunity(
                    sensor_id="EO-MIO-1",
                    sensor_type="high_resolution",
                    success_prob=0.70,
                    resolution="high",
                    cost=2.5,
                    available_from=current_time,
                    available_to=current_time,
                    satellite_id="EO-MIO-1",
                )
            )

        # EO-MIO-2: medium-inclination optical asset (~400 km, 51.6° inc., RAAN 220°).
        if _satellite_in_view("EO-MIO-2", observer_lat, observer_lon, current_time):
            opportunities.append(
                SensorOpportunity(
                    sensor_id="EO-MIO-2",
                    sensor_type="high_resolution",
                    success_prob=0.70,
                    resolution="high",
                    cost=2.5,
                    available_from=current_time,
                    available_to=current_time,
                    satellite_id="EO-MIO-2",
                )
            )

        # EO-SSO-1: sun-synchronous optical asset (~600 km, 97.8° inc., RAAN 320°).
        if _satellite_in_view("EO-SSO-1", observer_lat, observer_lon, current_time):
            opportunities.append(
                SensorOpportunity(
                    sensor_id="EO-SSO-1",
                    sensor_type="high_resolution",
                    success_prob=0.68,
                    resolution="high",
                    cost=2.8,
                    available_from=current_time,
                    available_to=current_time,
                    satellite_id="EO-SSO-1",
                )
            )

        # EO-SSO-2: sun-synchronous optical asset (~600 km, 97.8° inc., RAAN 100°).
        if _satellite_in_view("EO-SSO-2", observer_lat, observer_lon, current_time):
            opportunities.append(
                SensorOpportunity(
                    sensor_id="EO-SSO-2",
                    sensor_type="high_resolution",
                    success_prob=0.68,
                    resolution="high",
                    cost=2.8,
                    available_from=current_time,
                    available_to=current_time,
                    satellite_id="EO-SSO-2",
                )
            )

        # SAR-1: all-weather SSO asset (~700 km, 97.8° inc., RAAN 200°).
        # Complements EO assets: usable regardless of cloud cover.
        if _satellite_in_view("SAR-1", observer_lat, observer_lon, current_time):
            opportunities.append(
                SensorOpportunity(
                    sensor_id="SAR-1",
                    sensor_type="all_weather",
                    success_prob=0.80,
                    resolution="medium",
                    cost=2.0,
                    available_from=current_time,
                    available_to=current_time,
                    satellite_id="SAR-1",
                )
            )

        # SAR-2: all-weather SSO asset (~700 km, 97.8° inc., RAAN 80°).
        # Provides afternoon/evening coverage complementing SAR-1.
        if _satellite_in_view("SAR-2", observer_lat, observer_lon, current_time):
            opportunities.append(
                SensorOpportunity(
                    sensor_id="SAR-2",
                    sensor_type="all_weather",
                    success_prob=0.80,
                    resolution="medium",
                    cost=2.0,
                    available_from=current_time,
                    available_to=current_time,
                    satellite_id="SAR-2",
                )
            )

    return opportunities


def next_pass_window(
    satellite_id: str,
    observer_lat: float,
    observer_lon: float,
    from_time: datetime,
    horizon_minutes: int = 180,
    step_seconds: int = 30,
) -> Optional[PassWindow]:
    """Find the next visibility window for a satellite above a ground observer.

    Scans forward from ``from_time`` in steps of ``step_seconds``, looking for
    a continuous window during which the satellite is above
    SENSOR_MIN_ELEVATION_DEG as seen from the observer.

    When the satellite is already in view at ``from_time``, the returned
    window opens at ``from_time`` (time_to_start_seconds == 0) and extends
    to the first step at which the satellite drops below the elevation mask.

    Args:
        satellite_id:     Catalog identifier (e.g. "EO-MIO-1", "SAR-1").
                          Returns None for unknown identifiers.
        observer_lat:     Observer geodetic latitude in degrees (+N).
        observer_lon:     Observer geodetic longitude in degrees (+E).
        from_time:        UTC start of the forward search.
        horizon_minutes:  Maximum look-ahead in minutes.  Default 180.
        step_seconds:     Time resolution of the scan in seconds.  Default 30.
                          Smaller values increase accuracy at the cost of more
                          SGP4 evaluations.

    Returns:
        A PassWindow describing the next access window, or None if no pass
        exists within the horizon.
    """
    satrec = _SATREC_CACHE.get(satellite_id)
    if satrec is None:
        return None

    step = timedelta(seconds=step_seconds)
    end_horizon = from_time + timedelta(minutes=horizon_minutes)

    # ── Find start of visibility window ───────────────────────────────────────
    if is_in_view(satrec, observer_lat, observer_lon, from_time,
                  min_elevation_deg=SENSOR_MIN_ELEVATION_DEG):
        # Already overhead: window starts now.
        start_time = from_time
    else:
        start_time = None
        t = from_time + step
        while t <= end_horizon:
            if is_in_view(satrec, observer_lat, observer_lon, t,
                          min_elevation_deg=SENSOR_MIN_ELEVATION_DEG):
                start_time = t
                break
            t += step

    if start_time is None:
        return None  # No pass within the search horizon

    # ── Find end of visibility window ─────────────────────────────────────────
    end_time = end_horizon  # default: still in view at horizon boundary
    t = start_time + step
    while t <= end_horizon:
        if not is_in_view(satrec, observer_lat, observer_lon, t,
                          min_elevation_deg=SENSOR_MIN_ELEVATION_DEG):
            end_time = t
            break
        t += step

    return PassWindow(
        satellite_id=satellite_id,
        start_time=start_time,
        end_time=end_time,
        duration_seconds=(end_time - start_time).total_seconds(),
        time_to_start_seconds=(start_time - from_time).total_seconds(),
    )


def nearest_orbital_pass(
    observer_lat: float,
    observer_lon: float,
    from_time: datetime,
    horizon_minutes: int = 180,
) -> Optional[PassWindow]:
    """Return the soonest upcoming pass across all orbital satellites, or None.

    Inspects only orbital assets registered in _SATREC_CACHE (EO-MIO-1,
    EO-MIO-2, EO-SSO-1, EO-SSO-2, SAR-1, SAR-2).  Schedule-based sensors
    (A1, B1, C1) are not considered — they
    have no orbital geometry and are always available on a fixed timetable.

    Args:
        observer_lat:    Observer geodetic latitude in degrees (+N).
        observer_lon:    Observer geodetic longitude in degrees (+E).
        from_time:       UTC start of the forward search.
        horizon_minutes: Maximum look-ahead in minutes.  Default 180.

    Returns:
        The PassWindow with the smallest time_to_start_seconds across all
        known orbital satellites, or None if no pass exists within the
        horizon for any satellite.
    """
    candidates: list[PassWindow] = []
    for sat_id in _SATREC_CACHE:
        pw = next_pass_window(sat_id, observer_lat, observer_lon, from_time,
                              horizon_minutes=horizon_minutes)
        if pw is not None:
            candidates.append(pw)
    if not candidates:
        return None
    return min(candidates, key=lambda w: w.time_to_start_seconds)
