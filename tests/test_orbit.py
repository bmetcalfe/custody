"""
Tests for custody/orbit.py.

Covers:
  - TLE parsing succeeds and returns a usable Satrec
  - satellite_azel returns values in physically plausible ranges
  - An observer on the opposite side of the Earth from the satellite
    does not see the satellite
  - An observer placed directly below the satellite (sub-satellite point)
    sees near-zenith elevation
  - Elevation decreases as the observer moves laterally away from the
    sub-satellite point
  - Over a 24-hour window, a LEO satellite produces at least one pass
    above a mid-latitude observer
  - Propagation failure returns the (0.0, -90.0) safe fallback

All tests use the same ISS TLE (epoch 2019-343), so behaviour is
deterministic and independent of network access or live ephemeris.
"""

import math
from datetime import UTC, datetime, timedelta

import pytest

from custody.orbit import (
    SatellitePass,
    _ecef_to_azel,
    _gmst_rad,
    _jd_from_datetime,
    _teme_to_ecef,
    is_in_view,
    load_tle,
    satellite_azel,
)

# ---------------------------------------------------------------------------
# Test fixtures — ISS TLE, epoch 2019-343.69339541 (2019-12-09 16:38:29 UTC)
# Taken verbatim from the sgp4 README example; checksums verified.
# ---------------------------------------------------------------------------

ISS_NAME = "ISS (ZARYA)"
ISS_LINE1 = "1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991"
ISS_LINE2 = "2 25544  51.6439 211.2001 0007417  17.6667 342.4956 15.50103472202482"

# Reference time: ~21 minutes after TLE epoch — propagation error is minimal.
T_EPOCH = datetime(2019, 12, 9, 17, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Helper — compute sub-satellite point (lat, lon) at time t
# ---------------------------------------------------------------------------

def _subsat_point(sat, t: datetime) -> tuple[float, float]:
    """Return (lat_deg, lon_deg) of the point on Earth directly below the satellite."""
    jd, fr = _jd_from_datetime(t)
    error_code, r_teme, _ = sat.sgp4(jd, fr)
    assert error_code == 0, f"SGP4 propagation failed with error {error_code}"
    gmst = _gmst_rad(jd + fr)
    r_ecef = _teme_to_ecef(r_teme, gmst)
    x, y, z = r_ecef
    r = math.sqrt(x ** 2 + y ** 2 + z ** 2)
    lat_deg = math.degrees(math.asin(z / r))
    lon_deg = math.degrees(math.atan2(y, x))
    return lat_deg, lon_deg


# ---------------------------------------------------------------------------
# TLE parsing
# ---------------------------------------------------------------------------

class TestLoadTle:
    def test_load_tle_returns_satrec(self):
        """load_tle must return a Satrec that can be propagated without error."""
        from sgp4.api import Satrec
        sat = load_tle(ISS_NAME, ISS_LINE1, ISS_LINE2)
        assert isinstance(sat, Satrec)

    def test_satrec_propagates_without_error(self):
        """Satrec returned by load_tle must produce error_code == 0 near epoch."""
        sat = load_tle(ISS_NAME, ISS_LINE1, ISS_LINE2)
        jd, fr = _jd_from_datetime(T_EPOCH)
        error_code, r, v = sat.sgp4(jd, fr)
        assert error_code == 0
        # Position vector should be non-zero and in LEO altitude range (~400 km)
        r_km = math.sqrt(sum(x ** 2 for x in r))
        assert 6500 < r_km < 7000  # Earth radius ~6371 km + ~200–600 km altitude


# ---------------------------------------------------------------------------
# Plausible output ranges
# ---------------------------------------------------------------------------

class TestAzelRanges:
    def test_azimuth_in_range(self):
        sat = load_tle(ISS_NAME, ISS_LINE1, ISS_LINE2)
        az, _ = satellite_azel(sat, 51.5, -0.1, T_EPOCH)  # London, approx
        assert 0.0 <= az < 360.0

    def test_elevation_in_range(self):
        sat = load_tle(ISS_NAME, ISS_LINE1, ISS_LINE2)
        _, el = satellite_azel(sat, 51.5, -0.1, T_EPOCH)
        assert -90.0 <= el <= 90.0

    def test_satellite_azel_values_are_float(self):
        sat = load_tle(ISS_NAME, ISS_LINE1, ISS_LINE2)
        az, el = satellite_azel(sat, 0.0, 0.0, T_EPOCH)
        assert isinstance(az, float)
        assert isinstance(el, float)


# ---------------------------------------------------------------------------
# Geometry: antipodal observer cannot see the satellite
# ---------------------------------------------------------------------------

class TestAntipodal:
    def test_antipodal_observer_not_in_view(self):
        """An observer on the opposite side of the Earth should never see the satellite."""
        sat = load_tle(ISS_NAME, ISS_LINE1, ISS_LINE2)
        sub_lat, sub_lon = _subsat_point(sat, T_EPOCH)

        # Antipodal point
        anti_lat = -sub_lat
        anti_lon = (sub_lon + 180.0) % 360.0 - 180.0

        assert not is_in_view(sat, anti_lat, anti_lon, T_EPOCH)

    def test_antipodal_elevation_is_negative(self):
        """Elevation from the antipodal point must be below the horizon."""
        sat = load_tle(ISS_NAME, ISS_LINE1, ISS_LINE2)
        sub_lat, sub_lon = _subsat_point(sat, T_EPOCH)
        anti_lat = -sub_lat
        anti_lon = (sub_lon + 180.0) % 360.0 - 180.0

        _, el = satellite_azel(sat, anti_lat, anti_lon, T_EPOCH)
        assert el < 0.0


# ---------------------------------------------------------------------------
# Geometry: observer at sub-satellite point sees near-zenith elevation
# ---------------------------------------------------------------------------

class TestSubSatPoint:
    def test_subsat_observer_sees_high_elevation(self):
        """An observer directly below the satellite should see elevation > 80°."""
        sat = load_tle(ISS_NAME, ISS_LINE1, ISS_LINE2)
        sub_lat, sub_lon = _subsat_point(sat, T_EPOCH)
        _, el = satellite_azel(sat, sub_lat, sub_lon, T_EPOCH)
        assert el > 80.0

    def test_subsat_observer_is_in_view(self):
        """is_in_view must return True for an observer at the sub-satellite point."""
        sat = load_tle(ISS_NAME, ISS_LINE1, ISS_LINE2)
        sub_lat, sub_lon = _subsat_point(sat, T_EPOCH)
        assert is_in_view(sat, sub_lat, sub_lon, T_EPOCH)


# ---------------------------------------------------------------------------
# Geometry: elevation decreases with lateral distance from sub-satellite point
# ---------------------------------------------------------------------------

class TestElevationDecreaseWithDistance:
    def test_elevation_decreases_as_observer_moves_away(self):
        """Moving the observer away from the sub-satellite point reduces elevation."""
        sat = load_tle(ISS_NAME, ISS_LINE1, ISS_LINE2)
        sub_lat, sub_lon = _subsat_point(sat, T_EPOCH)

        # Sample elevations at increasing angular offsets along the equatorial direction
        offsets_deg = [0.0, 5.0, 15.0, 30.0, 60.0]
        elevations = [
            satellite_azel(sat, sub_lat, sub_lon + offset, T_EPOCH)[1]
            for offset in offsets_deg
        ]

        # Each successive elevation must be strictly less than the previous
        for i in range(1, len(elevations)):
            assert elevations[i] < elevations[i - 1], (
                f"Elevation at offset {offsets_deg[i]}° ({elevations[i]:.2f}°) "
                f"is not less than at {offsets_deg[i-1]}° ({elevations[i-1]:.2f}°)"
            )

    def test_large_offset_below_horizon(self):
        """Observer 70° away from sub-satellite point (in longitude) is below horizon."""
        sat = load_tle(ISS_NAME, ISS_LINE1, ISS_LINE2)
        sub_lat, sub_lon = _subsat_point(sat, T_EPOCH)
        _, el = satellite_azel(sat, sub_lat, sub_lon + 70.0, T_EPOCH)
        assert el < 0.0


# ---------------------------------------------------------------------------
# Pass behavior over 24-hour window
# ---------------------------------------------------------------------------

class TestPassBehavior:
    def test_at_least_one_pass_in_24_hours(self):
        """A LEO satellite should be visible at least once in 24 hours from any
        mid-latitude site within its inclination band."""
        sat = load_tle(ISS_NAME, ISS_LINE1, ISS_LINE2)
        # Observer at 51.5°N (within ISS inclination band of 51.6°)
        obs_lat, obs_lon = 51.5, -0.1

        # Sample every 2 minutes for 24 hours
        step = timedelta(minutes=2)
        count = int(24 * 60 / 2)
        in_view_count = sum(
            1 for i in range(count)
            if is_in_view(sat, obs_lat, obs_lon, T_EPOCH + i * step)
        )
        assert in_view_count > 0, (
            "No passes found in 24 hours — expected at least 2–5 for a LEO orbit"
        )

    def test_multiple_passes_in_24_hours(self):
        """A LEO satellite at ~92-minute period should give several passes per day."""
        sat = load_tle(ISS_NAME, ISS_LINE1, ISS_LINE2)
        obs_lat, obs_lon = 51.5, -0.1

        step = timedelta(minutes=2)
        count = int(24 * 60 / 2)
        in_view_states = [
            is_in_view(sat, obs_lat, obs_lon, T_EPOCH + i * step)
            for i in range(count)
        ]

        # Count pass transitions (False→True)
        passes = sum(
            1 for i in range(1, len(in_view_states))
            if in_view_states[i] and not in_view_states[i - 1]
        )
        assert passes >= 2, f"Expected at least 2 passes in 24 hours, got {passes}"

    def test_passes_have_finite_duration(self):
        """Each pass should last more than 1 sample interval (> 2 minutes) but
        less than 30 minutes (typical LEO pass duration)."""
        sat = load_tle(ISS_NAME, ISS_LINE1, ISS_LINE2)
        obs_lat, obs_lon = 51.5, -0.1

        step = timedelta(minutes=2)
        count = int(24 * 60 / 2)

        # Find the first pass: collect consecutive in-view samples
        in_pass = False
        pass_lengths = []
        current_length = 0

        for i in range(count):
            vis = is_in_view(sat, obs_lat, obs_lon, T_EPOCH + i * step)
            if vis:
                in_pass = True
                current_length += 1
            elif in_pass:
                pass_lengths.append(current_length * 2)  # minutes
                in_pass = False
                current_length = 0

        assert pass_lengths, "No completed passes found in 24-hour window"
        for length_min in pass_lengths:
            assert 2 < length_min < 30, (
                f"Pass duration {length_min} min outside expected 2–30 min range"
            )


# ---------------------------------------------------------------------------
# Propagation failure fallback
# ---------------------------------------------------------------------------

class TestPropagationFailure:
    def test_failure_returns_safe_fallback(self):
        """When SGP4 returns error_code != 0, satellite_azel returns (0.0, -90.0)."""

        class _FailSatrec:
            """Duck-type stand-in for Satrec that always reports propagation error."""
            def sgp4(self, jd, fr):
                return 6, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

        az, el = satellite_azel(_FailSatrec(), 0.0, 0.0, T_EPOCH)
        assert az == 0.0
        assert el == -90.0

    def test_failure_is_not_in_view(self):
        """A propagation failure must never be treated as an in-view contact."""

        class _FailSatrec:
            def sgp4(self, jd, fr):
                return 5, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

        assert not is_in_view(_FailSatrec(), 0.0, 0.0, T_EPOCH)

    def test_failure_not_in_view_with_zero_threshold(self):
        """Even with min_elevation_deg=0, a failed propagation is not in view."""

        class _FailSatrec:
            def sgp4(self, jd, fr):
                return 1, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

        assert not is_in_view(_FailSatrec(), 0.0, 0.0, T_EPOCH, min_elevation_deg=0.0)


# ---------------------------------------------------------------------------
# SatellitePass dataclass
# ---------------------------------------------------------------------------

class TestSatellitePassDataclass:
    def test_satellite_pass_is_frozen(self):
        """SatellitePass must be immutable (frozen dataclass)."""
        t0 = datetime(2024, 1, 1, tzinfo=UTC)
        t1 = datetime(2024, 1, 1, 0, 10, tzinfo=UTC)
        sp = SatellitePass(rise_time=t0, set_time=t1, max_elevation_deg=45.0)
        with pytest.raises((AttributeError, TypeError)):
            sp.max_elevation_deg = 90.0  # type: ignore[misc]

    def test_satellite_pass_fields(self):
        t0 = datetime(2024, 1, 1, tzinfo=UTC)
        t1 = datetime(2024, 1, 1, 0, 8, tzinfo=UTC)
        sp = SatellitePass(rise_time=t0, set_time=t1, max_elevation_deg=32.5)
        assert sp.rise_time == t0
        assert sp.set_time == t1
        assert sp.max_elevation_deg == pytest.approx(32.5)
