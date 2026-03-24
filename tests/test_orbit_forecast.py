"""
Tests for next_pass_window() orbital forecasting.

Covers:
  - Pass found within horizon: returns PassWindow, fields valid
  - No pass within horizon: returns None
  - Already in view at from_time: start_time == from_time, time_to_start == 0
  - start_time < end_time for every returned window
  - duration_seconds == (end_time - start_time).total_seconds()
  - time_to_start_seconds == (start_time - from_time).total_seconds()
  - SAT-A and SAT-B return different windows from the same observer/time
  - Unknown satellite_id returns None
  - Observer-location sensitivity: same satellite, different positions → different windows
  - PassWindow.satellite_id matches the requested satellite_id
"""
from datetime import UTC, datetime, timedelta

import pytest

from custody.sensors import PassWindow, next_pass_window

# ---------------------------------------------------------------------------
# Observer and time constants
#
# The 2026-epoch TLEs were tuned so that from observer (0.5, 0.5):
#   SAT-A: visible ~14:15–14:22 UTC on 2026-03-23  (step-quantised to 30s)
#   SAT-B: visible ~14:49–14:57 UTC on 2026-03-23
#
# All tests that check exact timestamps allow ±60 seconds of tolerance
# (two scan steps) to stay robust against minor TLE-propagation differences.
# ---------------------------------------------------------------------------

_OBS_LAT = 0.5
_OBS_LON = 0.5
_T13 = datetime(2026, 3, 23, 13, 0, tzinfo=UTC)  # search start, before both passes
_T_IN_VIEW = datetime(2026, 3, 23, 14, 52, tzinfo=UTC)  # mid-pass for SAT-B

_TOL = timedelta(seconds=60)  # ±1 step tolerance for timestamp assertions


# ---------------------------------------------------------------------------
# Pass found — SAT-A
# ---------------------------------------------------------------------------

class TestPassFound:
    def setup_method(self):
        self.w_a = next_pass_window("SAT-A", _OBS_LAT, _OBS_LON, _T13)
        self.w_b = next_pass_window("SAT-B", _OBS_LAT, _OBS_LON, _T13)

    def test_sat_a_returns_pass_window(self):
        assert self.w_a is not None
        assert isinstance(self.w_a, PassWindow)

    def test_sat_b_returns_pass_window(self):
        assert self.w_b is not None
        assert isinstance(self.w_b, PassWindow)

    def test_satellite_id_matches_request(self):
        assert self.w_a.satellite_id == "SAT-A"
        assert self.w_b.satellite_id == "SAT-B"

    def test_start_before_end(self):
        assert self.w_a.start_time < self.w_a.end_time
        assert self.w_b.start_time < self.w_b.end_time

    def test_duration_consistent_with_times(self):
        expected_a = (self.w_a.end_time - self.w_a.start_time).total_seconds()
        assert self.w_a.duration_seconds == pytest.approx(expected_a)
        expected_b = (self.w_b.end_time - self.w_b.start_time).total_seconds()
        assert self.w_b.duration_seconds == pytest.approx(expected_b)

    def test_time_to_start_consistent_with_times(self):
        expected_a = (self.w_a.start_time - _T13).total_seconds()
        assert self.w_a.time_to_start_seconds == pytest.approx(expected_a)
        expected_b = (self.w_b.start_time - _T13).total_seconds()
        assert self.w_b.time_to_start_seconds == pytest.approx(expected_b)

    def test_sat_a_starts_around_1415_utc(self):
        expected = datetime(2026, 3, 23, 14, 15, tzinfo=UTC)
        assert abs(self.w_a.start_time - expected) <= _TOL, (
            f"SAT-A start {self.w_a.start_time} not within {_TOL} of {expected}"
        )

    def test_sat_b_starts_around_1448_utc(self):
        expected = datetime(2026, 3, 23, 14, 48, tzinfo=UTC)
        assert abs(self.w_b.start_time - expected) <= _TOL, (
            f"SAT-B start {self.w_b.start_time} not within {_TOL} of {expected}"
        )

    def test_duration_is_positive(self):
        assert self.w_a.duration_seconds > 0
        assert self.w_b.duration_seconds > 0

    def test_time_to_start_is_nonnegative(self):
        assert self.w_a.time_to_start_seconds >= 0
        assert self.w_b.time_to_start_seconds >= 0

    def test_start_time_within_horizon(self):
        horizon = timedelta(minutes=180)
        assert self.w_a.start_time <= _T13 + horizon
        assert self.w_b.start_time <= _T13 + horizon


# ---------------------------------------------------------------------------
# No pass within horizon
# ---------------------------------------------------------------------------

class TestNoPass:
    def test_short_horizon_returns_none_for_sat_a(self):
        # SAT-A pass is ~75 min away; a 60-min horizon must miss it.
        w = next_pass_window("SAT-A", _OBS_LAT, _OBS_LON, _T13, horizon_minutes=60)
        assert w is None

    def test_short_horizon_returns_none_for_sat_b(self):
        # SAT-B pass is ~115 min away; a 60-min horizon must miss it.
        w = next_pass_window("SAT-B", _OBS_LAT, _OBS_LON, _T13, horizon_minutes=60)
        assert w is None

    def test_zero_horizon_returns_none_when_not_in_view(self):
        # at 13:00 neither satellite is overhead; horizon=0 → no scan at all
        w = next_pass_window("SAT-A", _OBS_LAT, _OBS_LON, _T13, horizon_minutes=0)
        assert w is None

    def test_unknown_satellite_id_returns_none(self):
        assert next_pass_window("UNKNOWN", _OBS_LAT, _OBS_LON, _T13) is None

    def test_nonexistent_satellite_returns_none(self):
        assert next_pass_window("SAT-Z", _OBS_LAT, _OBS_LON, _T13) is None


# ---------------------------------------------------------------------------
# Already in view at from_time
# ---------------------------------------------------------------------------

class TestAlreadyInView:
    def setup_method(self):
        # _T_IN_VIEW (14:52 UTC) is mid-pass for SAT-B from (0.5, 0.5)
        self.w = next_pass_window("SAT-B", _OBS_LAT, _OBS_LON, _T_IN_VIEW,
                                   horizon_minutes=30)

    def test_returns_pass_window(self):
        assert self.w is not None

    def test_start_time_equals_from_time(self):
        assert self.w.start_time == _T_IN_VIEW

    def test_time_to_start_is_zero(self):
        assert self.w.time_to_start_seconds == pytest.approx(0.0)

    def test_end_time_after_start(self):
        assert self.w.end_time > self.w.start_time

    def test_satellite_id_correct(self):
        assert self.w.satellite_id == "SAT-B"


# ---------------------------------------------------------------------------
# SAT-A and SAT-B produce distinct windows
# ---------------------------------------------------------------------------

class TestSatABDistinct:
    def setup_method(self):
        self.w_a = next_pass_window("SAT-A", _OBS_LAT, _OBS_LON, _T13)
        self.w_b = next_pass_window("SAT-B", _OBS_LAT, _OBS_LON, _T13)

    def test_different_start_times(self):
        assert self.w_a.start_time != self.w_b.start_time

    def test_sat_a_pass_earlier_than_sat_b(self):
        # From (0.5, 0.5) on 2026-03-23, SAT-A passes ~14:15, SAT-B ~14:49
        assert self.w_a.start_time < self.w_b.start_time

    def test_windows_do_not_overlap(self):
        # Both passes are distinct and non-overlapping
        assert self.w_a.end_time <= self.w_b.start_time or \
               self.w_b.end_time <= self.w_a.start_time, (
            f"Passes overlap: A={self.w_a.start_time}-{self.w_a.end_time} "
            f"B={self.w_b.start_time}-{self.w_b.end_time}"
        )


# ---------------------------------------------------------------------------
# Observer-location sensitivity
# ---------------------------------------------------------------------------

class TestObserverSensitivity:
    def test_different_observers_see_sat_b_at_different_times(self):
        # Two observers separated by 10 degrees longitude see the same satellite
        # at a slightly different time (or possibly not at all in one case).
        w1 = next_pass_window("SAT-B", _OBS_LAT, _OBS_LON, _T13)
        w2 = next_pass_window("SAT-B", _OBS_LAT, _OBS_LON + 10.0, _T13)
        # At least one should find a pass; if both do, start times should differ.
        assert w1 is not None or w2 is not None
        if w1 is not None and w2 is not None:
            assert w1.start_time != w2.start_time or w1.end_time != w2.end_time

    def test_distant_observer_gets_different_window(self):
        # An observer 90° of longitude away sees the satellite's passes at
        # a systematically different local time.  Use a 720-minute horizon
        # (≈7 orbital periods for SAT-B) to guarantee a pass is found from
        # either location.
        w_near = next_pass_window("SAT-B", _OBS_LAT, _OBS_LON, _T13,
                                   horizon_minutes=720)
        w_far = next_pass_window("SAT-B", _OBS_LAT, _OBS_LON + 90.0, _T13,
                                  horizon_minutes=720)
        assert w_near is not None
        assert w_far is not None
        assert w_near.start_time != w_far.start_time

    def test_sat_a_not_visible_from_subsat_sat_b_point_same_time(self):
        """From SAT-B's sub-satellite point, SAT-A may not be in its same window.

        This is the ground-track separation test: the two satellites occupy
        different orbital planes, so being under one doesn't guarantee the
        other is nearby.  We verify by checking that next_pass_window for SAT-A
        and SAT-B from SAT-B's sub-satellite point return different start times.
        """
        from custody.orbit import _gmst_rad, _jd_from_datetime, _teme_to_ecef
        from custody.sensors import _SATREC_CACHE
        import math

        # Compute SAT-B's sub-satellite point at the start of SAT-B's pass
        t_mid = datetime(2026, 3, 23, 14, 53, tzinfo=UTC)  # mid-pass
        satrec_b = _SATREC_CACHE["SAT-B"]
        jd, fr = _jd_from_datetime(t_mid)
        _, r_teme, _ = satrec_b.sgp4(jd, fr)
        gmst = _gmst_rad(jd + fr)
        r_ecef = _teme_to_ecef(r_teme, gmst)
        x, y, z = r_ecef
        r = math.sqrt(x**2 + y**2 + z**2)
        sub_lat = math.degrees(math.asin(z / r))
        sub_lon = math.degrees(math.atan2(y, x))

        w_b = next_pass_window("SAT-B", sub_lat, sub_lon, _T13)
        w_a = next_pass_window("SAT-A", sub_lat, sub_lon, _T13)
        # Both may find passes, but at different times
        assert w_b is not None  # SAT-B is visible from its own sub-sat point
        if w_a is not None:
            assert w_a.start_time != w_b.start_time


# ---------------------------------------------------------------------------
# PassWindow dataclass properties
# ---------------------------------------------------------------------------

class TestPassWindowDataclass:
    def test_pass_window_is_frozen(self):
        w = next_pass_window("SAT-A", _OBS_LAT, _OBS_LON, _T13)
        assert w is not None
        with pytest.raises((AttributeError, TypeError)):
            w.satellite_id = "OTHER"  # type: ignore[misc]

    def test_pass_window_fields_present(self):
        w = next_pass_window("SAT-A", _OBS_LAT, _OBS_LON, _T13)
        assert w is not None
        assert hasattr(w, "satellite_id")
        assert hasattr(w, "start_time")
        assert hasattr(w, "end_time")
        assert hasattr(w, "duration_seconds")
        assert hasattr(w, "time_to_start_seconds")

    def test_duration_is_float(self):
        w = next_pass_window("SAT-A", _OBS_LAT, _OBS_LON, _T13)
        assert w is not None
        assert isinstance(w.duration_seconds, float)

    def test_time_to_start_is_float(self):
        w = next_pass_window("SAT-A", _OBS_LAT, _OBS_LON, _T13)
        assert w is not None
        assert isinstance(w.time_to_start_seconds, float)
