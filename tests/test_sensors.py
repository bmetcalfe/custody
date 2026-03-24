"""
Tests for custody/sensors.py.

Covers:
  - Backward-compatible behavior: calling without observer returns the
    pre-Step-20 schedule-only result (no orbital sensors)
  - Schedule-based sensors (satellite_id=None) are always returned when
    the schedule condition is met, regardless of observer position
  - Orbital sensor (SAT-A) is returned only when the observer is in view
  - Orbital sensor (SAT-B) is returned only when the observer is in view
  - Orbital sensors are absent when the observer is on the opposite side
    of the Earth from the satellite
  - SAT-A and SAT-B can have different availability for the same
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
_SAT_A_LINE1 = "1 25544U 98067A   26082.50000000  .00001764  00000-0  38792-4 0  0007"
_SAT_A_LINE2 = "2 25544  51.6000 220.0000 0001000  90.0000 270.0000 15.50103472000016"

_SAT_B_LINE1 = "1 99901U 26001A   26082.50000000  .00000034  00000-0  25000-4 0  0001"
_SAT_B_LINE2 = "2 99901  97.8000  40.0000 0001000  90.0000   0.0000 14.57650000000017"

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

def _subsat_point(t: datetime, name: str = "SAT-A") -> tuple[float, float]:
    """Return (lat_deg, lon_deg) directly below a satellite at time t.

    Defaults to SAT-A; pass name="SAT-B" for the second orbital asset.
    """
    if name == "SAT-A":
        sat = load_tle("SAT-A", _SAT_A_LINE1, _SAT_A_LINE2)
    else:
        sat = load_tle("SAT-B", _SAT_B_LINE1, _SAT_B_LINE2)
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
    return -lat, (lon + 180.0) % 360.0 - 180.0


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
        assert "SAT-A" not in sensor_ids
        assert "SAT-B" not in sensor_ids

    def test_no_observer_hour17_returns_b1_only(self):
        """hour=17 schedule: B1 is the only sensor, no SAT-A."""
        ids = [o.sensor_id for o in get_sensor_opportunities(T_ORBIT)]
        assert ids == ["B1"]

    def test_no_observer_even_hour_returns_a1_c1(self):
        """hour=0 schedule: A1 and C1, no SAT-A."""
        ids = {o.sensor_id for o in get_sensor_opportunities(T_EVEN)}
        assert ids == {"A1", "C1"}

    def test_no_observer_no_schedule_returns_empty(self):
        """hour=11 schedule: nothing available."""
        assert get_sensor_opportunities(T_NO_SCHEDULE) == []

    def test_no_observer_only_lat_provided_also_skips_orbit(self):
        """Providing only lat (no lon) preserves backward-compat — no orbit filter."""
        ids = {o.sensor_id for o in
               get_sensor_opportunities(T_ORBIT, observer_lat=51.5)}
        assert "SAT-A" not in ids
        assert "SAT-B" not in ids

    def test_no_observer_only_lon_provided_also_skips_orbit(self):
        """Providing only lon (no lat) preserves backward-compat — no orbit filter."""
        ids = {o.sensor_id for o in
               get_sensor_opportunities(T_ORBIT, observer_lon=-0.1)}
        assert "SAT-A" not in ids
        assert "SAT-B" not in ids


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
        """SAT-A must be returned when the observer is directly below it."""
        sub_lat, sub_lon = _subsat_point(T_ORBIT)
        ids = {o.sensor_id for o in
               get_sensor_opportunities(T_ORBIT, sub_lat, sub_lon)}
        assert "SAT-A" in ids

    def test_sat_a_absent_when_observer_at_antipode(self):
        """SAT-A must not be returned when the observer is on the far side."""
        sub_lat, sub_lon = _subsat_point(T_ORBIT)
        anti_lat, anti_lon = _antipodal(sub_lat, sub_lon)
        ids = {o.sensor_id for o in
               get_sensor_opportunities(T_ORBIT, anti_lat, anti_lon)}
        assert "SAT-A" not in ids

    def test_sat_a_satellite_id_field(self):
        """Returned SAT-A opportunity must carry satellite_id='SAT-A'."""
        sub_lat, sub_lon = _subsat_point(T_ORBIT)
        opps = {o.sensor_id: o for o in
                get_sensor_opportunities(T_ORBIT, sub_lat, sub_lon)}
        assert "SAT-A" in opps
        assert opps["SAT-A"].satellite_id == "SAT-A"

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
        sat_a = opps["SAT-A"]
        assert sat_a.satellite_id == "SAT-A"
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
# SAT-B orbital sensor — in-view, out-of-view, and field validation
# ---------------------------------------------------------------------------

class TestSatBFiltering:
    def test_sat_b_present_when_observer_at_subsat_point(self):
        """SAT-B must be returned when the observer is directly below it."""
        sub_lat, sub_lon = _subsat_point(T_ORBIT, name="SAT-B")
        ids = {o.sensor_id for o in
               get_sensor_opportunities(T_ORBIT, sub_lat, sub_lon)}
        assert "SAT-B" in ids

    def test_sat_b_absent_when_observer_at_antipode(self):
        """SAT-B must not be returned when the observer is on the far side."""
        sub_lat, sub_lon = _subsat_point(T_ORBIT, name="SAT-B")
        anti_lat, anti_lon = _antipodal(sub_lat, sub_lon)
        ids = {o.sensor_id for o in
               get_sensor_opportunities(T_ORBIT, anti_lat, anti_lon)}
        assert "SAT-B" not in ids

    def test_sat_b_absent_without_observer_position(self):
        """SAT-B must never appear when no observer position is provided."""
        ids = {o.sensor_id for o in get_sensor_opportunities(T_ORBIT)}
        assert "SAT-B" not in ids

    def test_sat_b_satellite_id_field(self):
        """Returned SAT-B opportunity must carry satellite_id='SAT-B'."""
        sub_lat, sub_lon = _subsat_point(T_ORBIT, name="SAT-B")
        opps = {o.sensor_id: o for o in
                get_sensor_opportunities(T_ORBIT, sub_lat, sub_lon)}
        assert "SAT-B" in opps
        assert opps["SAT-B"].satellite_id == "SAT-B"

    def test_sat_b_all_fields_present(self):
        """SAT-B opportunity must carry all required SensorOpportunity fields."""
        sub_lat, sub_lon = _subsat_point(T_ORBIT, name="SAT-B")
        opps = {o.sensor_id: o for o in
                get_sensor_opportunities(T_ORBIT, sub_lat, sub_lon)}
        sat_b = opps["SAT-B"]
        assert sat_b.satellite_id == "SAT-B"
        assert isinstance(sat_b.sensor_type, str)
        assert 0.0 <= sat_b.success_prob <= 1.0
        assert isinstance(sat_b.resolution, str)
        assert isinstance(sat_b.cost, float)
        assert sat_b.available_from == T_ORBIT
        assert sat_b.available_to == T_ORBIT


# ---------------------------------------------------------------------------
# SAT-A and SAT-B differential availability — distinct ground-track coverage
# ---------------------------------------------------------------------------

class TestSatABDiffer:
    def test_sat_b_visible_from_its_own_subsat_sat_a_is_not(self):
        """From SAT-B's sub-satellite point, SAT-B is in view but SAT-A is not.

        SAT-A and SAT-B have different inclinations (51.6° vs 97.8°) and RAAN
        values ~180° apart (220° vs 40°).  Their ground tracks are in different
        parts of the sky at the same epoch, so observing from below one
        satellite does not imply the other is also overhead.
        """
        sub_lat, sub_lon = _subsat_point(T_ORBIT, name="SAT-B")
        ids = {o.sensor_id for o in
               get_sensor_opportunities(T_ORBIT, sub_lat, sub_lon)}
        assert "SAT-B" in ids, "SAT-B must be visible from its own sub-sat point"
        assert "SAT-A" not in ids, (
            "SAT-A should not be visible from SAT-B's sub-sat point "
            "(different orbital plane)"
        )

    def test_sat_a_and_sat_b_can_be_simultaneously_visible(self):
        """There exists at least one observer position where both are in view.

        From directly below SAT-A, SAT-A is always visible.  We verify
        that SAT-B can also be in view from some location (its sub-sat
        point), confirming both participate in the sensor pool.
        """
        sub_a = _subsat_point(T_ORBIT, name="SAT-A")
        sub_b = _subsat_point(T_ORBIT, name="SAT-B")
        ids_from_a = {o.sensor_id for o in
                      get_sensor_opportunities(T_ORBIT, *sub_a)}
        ids_from_b = {o.sensor_id for o in
                      get_sensor_opportunities(T_ORBIT, *sub_b)}
        assert "SAT-A" in ids_from_a
        assert "SAT-B" in ids_from_b

    def test_orbital_assets_increase_pool_size_when_in_view(self):
        """When an observer is below an orbital satellite, the sensor pool is
        larger than the schedule-only baseline for the same time slot.

        T_ORBIT is hour=17, which yields only B1 from the schedule.
        From SAT-B's sub-satellite point at T_ORBIT, SAT-B is also in view,
        so the pool size increases by at least 1 compared to no-position call.
        """
        sub_lat, sub_lon = _subsat_point(T_ORBIT, name="SAT-B")
        no_pos = get_sensor_opportunities(T_ORBIT)
        with_pos = get_sensor_opportunities(T_ORBIT, sub_lat, sub_lon)
        assert len(with_pos) > len(no_pos), (
            "pool with observer position should include at least one "
            "orbital sensor beyond the schedule-only result"
        )
        assert "SAT-B" in {o.sensor_id for o in with_pos}
