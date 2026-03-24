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

Prototype satellite catalog
---------------------------
Two synthetic prototype orbital assets are bundled; no live TLE fetch is
performed.  Their orbital parameters were chosen to give meaningfully
different coverage patterns:

  SAT-A  ISS-like LEO orbit (inclination 51.6°, ~400 km altitude).
         Sensor type: high_resolution.  Covers mid-latitude ground tracks.
         TLE epoch: 2026-03-23 (day 82), RAAN 220°.

  SAT-B  Sun-synchronous LEO orbit (inclination 97.8°, ~700 km altitude).
         Sensor type: all_weather (SAR-like — complements SAT-A's optical).
         TLE epoch: 2026-03-23 (day 82), RAAN 40° (~180° offset from SAT-A),
         ensuring the two satellites occupy different parts of the sky and
         produce non-overlapping visibility windows over the default
         simulation date.
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from sgp4.api import Satrec

from custody.config import SENSOR_MIN_ELEVATION_DEG
from custody.orbit import is_in_view, load_tle


# ---------------------------------------------------------------------------
# Prototype satellite TLE catalog
# ---------------------------------------------------------------------------

# SAT-A: ISS-like LEO asset (epoch 2026-03-23, NORAD 25544).
# Prototype high-resolution optical LEO asset; inclination 51.6°, ~400 km.
# RAAN 220° places visible passes over equatorial observers at ~14:16-14:21 UTC
# on the default simulation date (2026-03-23).
_TLE_SAT_A = (
    "SAT-A",
    "1 25544U 98067A   26082.50000000  .00001764  00000-0  38792-4 0  0007",
    "2 25544  51.6000 220.0000 0001000  90.0000 270.0000 15.50103472000016",
)

# SAT-A2: Second optical LEO asset (NORAD 25545, RAAN 0°).
# Same orbital plane family as SAT-A but RAAN offset to produce passes over
# equatorial observers at ~11:57-12:04 UTC on 2026-03-23, filling the morning
# gap in the simulation window.
_TLE_SAT_A2 = (
    "SAT-A2",
    "1 25545U 98067B   26082.50000000  .00001764  00000-0  38792-4 0  0008",
    "2 25545  51.6000   0.0000 0001000  90.0000 270.0000 15.50103472000013",
)

# SAT-A3: Third optical LEO asset (NORAD 25546, RAAN 280°).
# RAAN 280° produces passes over equatorial observers at ~18:56-19:02 UTC
# on 2026-03-23, covering the evening portion of the simulation window.
_TLE_SAT_A3 = (
    "SAT-A3",
    "1 25546U 98067C   26082.50000000  .00001764  00000-0  38792-4 0  0009",
    "2 25546  51.6000 280.0000 0001000  90.0000 270.0000 15.50103472000014",
)

# SAT-B: Synthetic sun-synchronous orbit (epoch 2026-03-23, NORAD 99901).
# Prototype all-weather (SAR-like) asset; inclination 97.8°, ~700 km.
# RAAN 40° produces passes at ~14:49-14:57 UTC on 2026-03-23.
_TLE_SAT_B = (
    "SAT-B",
    "1 99901U 26001A   26082.50000000  .00000034  00000-0  25000-4 0  0001",
    "2 99901  97.8000  40.0000 0001000  90.0000   0.0000 14.57650000000017",
)

# SAT-B2: Second SAR asset (NORAD 99902, RAAN 200°).
# Produces two passes over equatorial observers on 2026-03-23: ~12:22-12:29 UTC
# and ~14:00-14:07 UTC, covering the late-morning portion of the window.
_TLE_SAT_B2 = (
    "SAT-B2",
    "1 99902U 26002A   26082.50000000  .00000034  00000-0  25000-4 0  0003",
    "2 99902  97.8000 200.0000 0001000  90.0000   0.0000 14.57650000000016",
)

# SAT-B3: Third SAR asset (NORAD 99903, RAAN 80°).
# Produces two passes over equatorial observers on 2026-03-23: ~16:29-16:37 UTC
# and ~18:08-18:13 UTC, covering the afternoon/evening portion of the window.
_TLE_SAT_B3 = (
    "SAT-B3",
    "1 99903U 26003A   26082.50000000  .00000034  00000-0  25000-4 0  0005",
    "2 99903  97.8000  80.0000 0001000  90.0000   0.0000 14.57650000000013",
)

# Parse TLEs once at import time.  Satrec objects are immutable and safe to
# share across threads.
_SATREC_CACHE: dict[str, Satrec] = {
    "SAT-A":  load_tle(*_TLE_SAT_A),
    "SAT-A2": load_tle(*_TLE_SAT_A2),
    "SAT-A3": load_tle(*_TLE_SAT_A3),
    "SAT-B":  load_tle(*_TLE_SAT_B),
    "SAT-B2": load_tle(*_TLE_SAT_B2),
    "SAT-B3": load_tle(*_TLE_SAT_B3),
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

        # SAT-A: prototype LEO high-resolution optical asset (~400 km, 51.6° inc.)
        if _satellite_in_view("SAT-A", observer_lat, observer_lon, current_time):
            opportunities.append(
                SensorOpportunity(
                    sensor_id="SAT-A",
                    sensor_type="high_resolution",
                    success_prob=0.70,
                    resolution="high",
                    cost=2.5,
                    available_from=current_time,
                    available_to=current_time,
                    satellite_id="SAT-A",
                )
            )

        # SAT-B: prototype sun-synchronous all-weather (SAR-like) asset
        # (~700 km, 97.8° inc., RAAN ~138° offset from SAT-A).
        # Complements SAT-A: visible from different ground tracks at the
        # same time, and usable regardless of cloud cover.
        if _satellite_in_view("SAT-B", observer_lat, observer_lon, current_time):
            opportunities.append(
                SensorOpportunity(
                    sensor_id="SAT-B",
                    sensor_type="all_weather",
                    success_prob=0.80,
                    resolution="medium",
                    cost=2.0,
                    available_from=current_time,
                    available_to=current_time,
                    satellite_id="SAT-B",
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
        satellite_id:     Catalog identifier ("SAT-A" or "SAT-B").
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

    Inspects only orbital assets registered in _SATREC_CACHE (SAT-A and
    SAT-B).  Schedule-based sensors (A1, B1, C1) are not considered — they
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
