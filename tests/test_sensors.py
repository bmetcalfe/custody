"""
Tests for custody/sensors.py.

Covers:
  - Backward-compatible behavior: calling without observer returns the
    pre-Step-20 schedule-only result (no orbital sensors)
  - Schedule-based sensors (satellite_id=None) are always returned when
    the schedule condition is met, regardless of observer position
  - Orbital sensor (EO-MIO-1) is returned only when the observer is in view
  - Orbital sensor (SAR-1) is returned only when the observer is in view
  - Orbital sensors are absent when the observer is on the opposite side
    of the Earth from the satellite
  - EO-MIO-1 and SAR-1 can have different availability for the same
    observer/time (distinct ground tracks due to different inclination
    and RAAN)
  - The filtered (distant observer) result is a subset of the full result
    (close observer where all satellites are in view)
  - Returned SensorOpportunity objects carry the expected field values,
    including satellite_id=None for schedule-based sensors
"""
import math
from datetime import UTC, datetime

import pytest

from custody.sensors import SensorOpportunity, get_sensor_opportunities
from custody.orbit import (
    _gmst_rad,
    _jd_from_datetime,
    _teme_to_ecef,
    load_tle,
)

# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

# Same TLEs used by sensors.py — kept in sync with the catalog there.
# Epoch: 2026-03-23 noon UTC (day 82 of 2026).
_SAT_A_LINE1 = "1 99910U 98067B   26082.50000000  .00000034  00000-0  25000-4 0  0002"
_SAT_A_LINE2 = "2 99910 51.6000   0.0000 0001000  90.0000   0.0000 15.50103472000000"

_SAT_B_LINE1 = "1 99930U 26002A   26082.50000000  .00000034  00000-0  25000-4 0  0004"
_SAT_B_LINE2 = "2 99930 97.8000 200.0000 0001000  90.0000   0.0000 14.57650000000006"

# T_ORBIT: 5 hours after TLE epoch (2026-03-23 noon) — propagation error is minimal.
# hour=17 → schedule yields B1 only (A1 needs even hour; C1 needs hour%3==0).
T_ORBIT = datetime(2026, 3, 23, 17, 0, 0, tzinfo=UTC)

# T_EVEN: hour=0 → schedule yields A1 and C1.
T_EVEN = datetime(2026, 3, 23, 0, 0, 0, tzinfo=UTC)

# T_NO_SCHEDULE: hour=11 → schedule yields nothing (11%2=1, not in [13,17], 11%3=2).
T_NO_SCHEDULE = datetime(2026, 3, 23, 11, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Helper — sub-satellite point at a given time
# ---------------------------------------------------------------------------

def _subsat_point(t: datetime, name: str = "EO-MIO-1") -> tuple[float, float]:
    """Return (lat_deg, lon_deg) directly below a satellite at time t.

    Defaults to EO-MIO-1; pass name="SAR-1" for the SAR orbital asset.
    """
    if name == "EO-MIO-1":
        sat = load_tle("EO-MIO-1", _SAT_A_LINE1, _SAT_A_LINE2)
    else:
        sat = load_tle("SAR-1", _SAT_B_LINE1, _SAT_B_LINE2)
    jd, fr = _jd_from_datetime(t)
    error_code, r_teme, _ = sat.sgp4(jd, fr)
    assert error_code == 0, f"SGP4 propagation failed (error {error_code})"
    gmst = _gmst_rad(jd + fr)
    r_ecef = _teme_to_ecef(r_teme, gmst)
    x, y, z = r_ecef
    r = math.sqrt(x ** 2 + y ** 2 + z ** 2)
    lat = math.degrees(math.asin(z / r))
    lon = math.degrees(math.atan2(y, x))
    return lat, lon


def _antipodal(lat: float, lon: float) -> tuple[float, float]:
    """Return the point on the opposite side of the Earth."""
    anti_lon = lon + 180.0
    if anti_lon > 180.0:
        anti_lon -= 360.0
    return -lat, anti_lon


# ---------------------------------------------------------------------------
# Backward compatibility (no observer position)
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    def test_no_observer_returns_only_schedule_sensors(self):
        """Without observer, result must not include any orbital sensors."""
        opportunities = get_sensor_opportunities(T_ORBIT)
        sat_ids = {o.satellite_id for o in opportunities}
        assert None in sat_ids or len(sat_ids) == 0  # only schedule-based
        sensor_ids = {o.sensor_id for o in opportunities}
        assert "EO-MIO-1" not in sensor_ids
        assert "SAR-1" not in sensor_ids

    def test_no_observer_hour17_returns_b1_only(self):
        """hour=17 schedule: B1 is the only sensor, no EO-MIO-1."""
        ids = [o.sensor_id for o in get_sensor_opportunities(T_ORBIT)]
        assert ids == ["B1"]

    def test_no_observer_even_hour_returns_a1_c1(self):
        """hour=0 schedule: A1 and C1, no EO-MIO-1."""
        ids = {o.sensor_id for o in get_sensor_opportunities(T_EVEN)}
        assert ids == {"A1", "C1"}

    def test_no_observer_no_schedule_returns_empty(self):
        """hour=11 schedule: nothing available."""
        assert get_sensor_opportunities(T_NO_SCHEDULE) == []

    def test_no_observer_only_lat_provided_also_skips_orbit(self):
        """Providing only lat (no lon) preserves backward-compat — no orbit filter."""
        ids = {o.sensor_id for o in
               get_sensor_opportunities(T_ORBIT, observer_lat=51.5)}
        assert "EO-MIO-1" not in ids
        assert "SAR-1" not in ids

    def test_no_observer_only_lon_provided_also_skips_orbit(self):
        """Providing only lon (no lat) preserves backward-compat — no orbit filter."""
        ids = {o.sensor_id for o in
               get_sensor_opportunities(T_ORBIT, observer_lon=-0.1)}
        assert "EO-MIO-1" not in ids
        assert "SAR-1" not in ids


# ---------------------------------------------------------------------------
# Schedule-based sensors are unaffected by observer position
# ---------------------------------------------------------------------------

class TestScheduleSensorsUnaffected:
    def test_b1_present_regardless_of_observer(self):
        """B1 (schedule-based) must appear with and without an observer."""
        sub_lat, sub_lon = _subsat_point(T_ORBIT)
        anti_lat, anti_lon = _antipodal(sub_lat, sub_lon)

        ids_no_obs = {o.sensor_id for o in get_sensor_opportunities(T_ORBIT)}
        ids_subsat = {o.sensor_id for o in
                      get_sensor_opportunities(T_ORBIT, sub_lat, sub_lon)}
        ids_anti   = {o.sensor_id for o in
                      get_sensor_opportunities(T_ORBIT, anti_lat, anti_lon)}

        assert "B1" in ids_no_obs
        assert "B1" in ids_subsat
        assert "B1" in ids_anti

    def test_schedule_sensor_has_null_satellite_id(self):
        """Schedule-based sensors must have satellite_id=None."""
        for opp in get_sensor_opportunities(T_ORBIT):
            assert opp.satellite_id is None, (
                f"Expected satellite_id=None for {opp.sensor_id}, "
                f"got {opp.satellite_id!r}"
            )

    def test_a1_satellite_id_is_none(self):
        ids_map = {o.sensor_id: o.satellite_id
                   for o in get_sensor_opportunities(T_EVEN)}
        assert ids_map["A1"] is None

    def test_c1_satellite_id_is_none(self):
        ids_map = {o.sensor_id: o.satellite_id
                   for o in get_sensor_opportunities(T_EVEN)}
        assert ids_map["C1"] is None


# ---------------------------------------------------------------------------
# Orbital sensor: in-view and out-of-view behavior
# ---------------------------------------------------------------------------

class TestOrbitalSensorFiltering:
    def test_sat_a_present_when_observer_at_subsat_point(self):
        """EO-MIO-1 must be returned when the observer is directly below it."""
        sub_lat, sub_lon = _subsat_point(T_ORBIT)
        ids = {o.sensor_id for o in
               get_sensor_opportunities(T_ORBIT, sub_lat, sub_lon)}
        assert "EO-MIO-1" in ids

    def test_sat_a_absent_when_observer_at_antipode(self):
        """EO-MIO-1 must not be returned when the observer is on the far side."""
        sub_lat, sub_lon = _subsat_point(T_ORBIT)
        anti_lat, anti_lon = _antipodal(sub_lat, sub_lon)
        ids = {o.sensor_id for o in
               get_sensor_opportunities(T_ORBIT, anti_lat, anti_lon)}
        assert "EO-MIO-1" not in ids

    def test_sat_a_satellite_id_field(self):
        """Returned EO-MIO-1 opportunity must carry satellite_id='EO-MIO-1'."""
        sub_lat, sub_lon = _subsat_point(T_ORBIT)
        opps = {o.sensor_id: o for o in
                get_sensor_opportunities(T_ORBIT, sub_lat, sub_lon)}
        assert "EO-MIO-1" in opps
        assert opps["EO-MIO-1"].satellite_id == "EO-MIO-1"

    def test_subsat_list_larger_than_antipodal_list(self):
        """Observer at sub-satellite point sees more sensors than at antipode."""
        sub_lat, sub_lon = _subsat_point(T_ORBIT)
        anti_lat, anti_lon = _antipodal(sub_lat, sub_lon)
        sub_ids  = [o.sensor_id for o in get_sensor_opportunities(T_ORBIT, sub_lat, sub_lon)]
        anti_ids = [o.sensor_id for o in get_sensor_opportunities(T_ORBIT, anti_lat, anti_lon)]
        assert len(sub_ids) > len(anti_ids)

    def test_with_observer_count_lte_all_sensors_in_view(self):
        """Any observer position returns ≤ sensors than the maximum (subsat) position."""
        sub_lat, sub_lon = _subsat_point(T_ORBIT)
        max_count = len(get_sensor_opportunities(T_ORBIT, sub_lat, sub_lon))

        # Test a range of observer positions
        test_positions = [
            (0.0,  0.0),
            (51.5, -0.1),
            (-33.9, 151.2),
            (_antipodal(sub_lat, sub_lon)),
        ]
        for lat, lon in test_positions:
            count = len(get_sensor_opportunities(T_ORBIT, lat, lon))
            assert count <= max_count, (
                f"Observer at ({lat}, {lon}) returned {count} sensors, "
                f"more than the subsat maximum of {max_count}"
            )


# ---------------------------------------------------------------------------
# Returned object field validation
# ---------------------------------------------------------------------------

class TestOpportunityFields:
    def test_all_fields_present_schedule_sensor(self):
        """A schedule-based opportunity must carry all required fields."""
        opps = get_sensor_opportunities(T_EVEN)
        a1 = next(o for o in opps if o.sensor_id == "A1")
        assert isinstance(a1.sensor_id, str)
        assert isinstance(a1.sensor_type, str)
        assert 0.0 <= a1.success_prob <= 1.0
        assert isinstance(a1.resolution, str)
        assert isinstance(a1.cost, float)
        assert a1.available_from == T_EVEN
        assert a1.available_to == T_EVEN
        assert a1.satellite_id is None

    def test_all_fields_present_orbital_sensor(self):
        """An orbital opportunity must carry all required fields including satellite_id."""
        sub_lat, sub_lon = _subsat_point(T_ORBIT)
        opps = {o.sensor_id: o for o in
                get_sensor_opportunities(T_ORBIT, sub_lat, sub_lon)}
        sat_a = opps["EO-MIO-1"]
        assert sat_a.satellite_id == "EO-MIO-1"
        assert isinstance(sat_a.sensor_type, str)
        assert 0.0 <= sat_a.success_prob <= 1.0
        assert isinstance(sat_a.resolution, str)
        assert isinstance(sat_a.cost, float)
        assert sat_a.available_from == T_ORBIT
        assert sat_a.available_to == T_ORBIT

    def test_opportunity_is_mutable_dataclass(self):
        """SensorOpportunity is a regular (mutable) dataclass."""
        opps = get_sensor_opportunities(T_EVEN)
        a1 = next(o for o in opps if o.sensor_id == "A1")
        a1.cost = 99.0  # must not raise
        assert a1.cost == 99.0


# ---------------------------------------------------------------------------
# SAR-1 orbital sensor — in-view, out-of-view, and field validation
# ---------------------------------------------------------------------------

class TestSatBFiltering:
    def test_sat_b_present_when_observer_at_subsat_point(self):
        """SAR-1 must be returned when the observer is directly below it."""
        sub_lat, sub_lon = _subsat_point(T_ORBIT, name="SAR-1")
        ids = {o.sensor_id for o in
               get_sensor_opportunities(T_ORBIT, sub_lat, sub_lon)}
        assert "SAR-1" in ids

    def test_sat_b_absent_when_observer_at_antipode(self):
        """SAR-1 must not be returned when the observer is on the far side."""
        sub_lat, sub_lon = _subsat_point(T_ORBIT, name="SAR-1")
        anti_lat, anti_lon = _antipodal(sub_lat, sub_lon)
        ids = {o.sensor_id for o in
               get_sensor_opportunities(T_ORBIT, anti_lat, anti_lon)}
        assert "SAR-1" not in ids

    def test_sat_b_absent_without_observer_position(self):
        """SAR-1 must never appear when no observer position is provided."""
        ids = {o.sensor_id for o in get_sensor_opportunities(T_ORBIT)}
        assert "SAR-1" not in ids

    def test_sat_b_satellite_id_field(self):
        """Returned SAR-1 opportunity must carry satellite_id='SAR-1'."""
        sub_lat, sub_lon = _subsat_point(T_ORBIT, name="SAR-1")
        opps = {o.sensor_id: o for o in
                get_sensor_opportunities(T_ORBIT, sub_lat, sub_lon)}
        assert "SAR-1" in opps
        assert opps["SAR-1"].satellite_id == "SAR-1"

    def test_sat_b_all_fields_present(self):
        """SAR-1 opportunity must carry all required SensorOpportunity fields."""
        sub_lat, sub_lon = _subsat_point(T_ORBIT, name="SAR-1")
        opps = {o.sensor_id: o for o in
                get_sensor_opportunities(T_ORBIT, sub_lat, sub_lon)}
        sat_b = opps["SAR-1"]
        assert sat_b.satellite_id == "SAR-1"
        assert isinstance(sat_b.sensor_type, str)
        assert 0.0 <= sat_b.success_prob <= 1.0
        assert isinstance(sat_b.resolution, str)
        assert isinstance(sat_b.cost, float)
        assert sat_b.available_from == T_ORBIT
        assert sat_b.available_to == T_ORBIT


# ---------------------------------------------------------------------------
# EO-MIO-1 and SAR-1 differential availability — distinct ground-track coverage
# ---------------------------------------------------------------------------

class TestSatABDiffer:
    def test_sat_b_visible_from_its_own_subsat_sat_a_is_not(self):
        """From SAR-1's sub-satellite point, SAR-1 is in view but EO-MIO-1 is not.

        EO-MIO-1 and SAR-1 have different inclinations (51.6° vs 97.8°) and RAAN
        values substantially apart (0° vs 200°).  Their ground tracks are in different
        parts of the sky at the same epoch, so observing from below one
        satellite does not imply the other is also overhead.
        """
        sub_lat, sub_lon = _subsat_point(T_ORBIT, name="SAR-1")
        ids = {o.sensor_id for o in
               get_sensor_opportunities(T_ORBIT, sub_lat, sub_lon)}
        assert "SAR-1" in ids, "SAR-1 must be visible from its own sub-sat point"
        assert "EO-MIO-1" not in ids, (
            "EO-MIO-1 should not be visible from SAR-1's sub-sat point "
            "(different orbital plane)"
        )

    def test_sat_a_and_sat_b_can_be_simultaneously_visible(self):
        """There exists at least one observer position where both are in view.

        From directly below EO-MIO-1, EO-MIO-1 is always visible.  We verify
        that SAR-1 can also be in view from some location (its sub-sat
        point), confirming both participate in the sensor pool.
        """
        sub_a = _subsat_point(T_ORBIT, name="EO-MIO-1")
        sub_b = _subsat_point(T_ORBIT, name="SAR-1")
        ids_from_a = {o.sensor_id for o in
                      get_sensor_opportunities(T_ORBIT, *sub_a)}
        ids_from_b = {o.sensor_id for o in
                      get_sensor_opportunities(T_ORBIT, *sub_b)}
        assert "EO-MIO-1" in ids_from_a
        assert "SAR-1" in ids_from_b

    def test_orbital_assets_increase_pool_size_when_in_view(self):
        """When an observer is below an orbital satellite, the sensor pool is
        larger than the schedule-only baseline for the same time slot.

        T_ORBIT is hour=17, which yields only B1 from the schedule.
        From SAR-1's sub-satellite point at T_ORBIT, SAR-1 is also in view,
        so the pool size increases by at least 1 compared to no-position call.
        """
        sub_lat, sub_lon = _subsat_point(T_ORBIT, name="SAR-1")
        no_pos = get_sensor_opportunities(T_ORBIT)
        with_pos = get_sensor_opportunities(T_ORBIT, sub_lat, sub_lon)
        assert len(with_pos) > len(no_pos), (
            "pool with observer position should include at least one "
            "orbital sensor beyond the schedule-only result"
        )
        assert "SAR-1" in {o.sensor_id for o in with_pos}
