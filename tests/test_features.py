"""
Unit tests for custody/features/.

Covers:
  - motion_features: is_slow, heading_deviation, consecutive_slow_steps
  - zone_features: is_inside_zone, distance_to_zone

Tests are independent of detector policy.  All zone tests use a local
TEST_ZONE rather than SENSITIVE_ZONE to avoid coupling to anomaly constants.
"""
import math
from datetime import datetime, UTC

import pytest

from custody.models import HistoryEntry, Vessel
from custody.features.motion_features import (
    consecutive_slow_steps,
    heading_deviation,
    is_slow,
)
from custody.features.zone_features import distance_to_zone, is_inside_zone


_EPOCH = datetime(2026, 1, 1, tzinfo=UTC)

# Local test zone — does not depend on SENSITIVE_ZONE constants.
TEST_ZONE = {
    "min_lat": 10.0,
    "max_lat": 12.0,
    "min_lon": 20.0,
    "max_lon": 22.0,
}


def make_vessel(lat=0.0, lon=0.0, speed_kmh=30.0, heading_deg=45.0):
    return Vessel(
        id="T",
        lat=lat,
        lon=lon,
        speed_kmh=speed_kmh,
        heading_deg=heading_deg,
        last_seen=_EPOCH,
    )


def entry(speed_kmh=28.0, heading_deg=45.0):
    return HistoryEntry(0.0, 0.0, _EPOCH, speed_kmh, heading_deg)


# ---------------------------------------------------------------------------
# is_slow
# ---------------------------------------------------------------------------

class TestIsSlow:
    def test_slow_vessel_returns_true(self):
        assert is_slow(make_vessel(speed_kmh=2.0)) is True

    def test_fast_vessel_returns_false(self):
        assert is_slow(make_vessel(speed_kmh=20.0)) is False

    def test_speed_at_default_threshold_returns_false(self):
        # speed == threshold is NOT slow (< not <=)
        assert is_slow(make_vessel(speed_kmh=5.0)) is False

    def test_just_below_threshold_returns_true(self):
        assert is_slow(make_vessel(speed_kmh=4.99)) is True

    def test_zero_speed_returns_true(self):
        assert is_slow(make_vessel(speed_kmh=0.0)) is True

    def test_custom_threshold_lower(self):
        assert is_slow(make_vessel(speed_kmh=3.0), threshold=2.0) is False

    def test_custom_threshold_higher(self):
        assert is_slow(make_vessel(speed_kmh=8.0), threshold=10.0) is True


# ---------------------------------------------------------------------------
# heading_deviation
# ---------------------------------------------------------------------------

class TestHeadingDeviation:
    def test_no_deviation_returns_zero(self):
        v = make_vessel(heading_deg=45.0)
        assert heading_deviation(v, baseline_heading=45.0) == pytest.approx(0.0)

    def test_moderate_deviation(self):
        v = make_vessel(heading_deg=90.0)
        assert heading_deviation(v, baseline_heading=45.0) == pytest.approx(45.0)

    def test_large_deviation(self):
        v = make_vessel(heading_deg=120.0)
        assert heading_deviation(v, baseline_heading=45.0) == pytest.approx(75.0)

    def test_wraparound_takes_shortest_path(self):
        # 355° and 5° — shortest angular distance is 10°, not 350°
        v = make_vessel(heading_deg=355.0)
        assert heading_deviation(v, baseline_heading=5.0) == pytest.approx(10.0)

    def test_opposite_headings_returns_180(self):
        v = make_vessel(heading_deg=225.0)
        assert heading_deviation(v, baseline_heading=45.0) == pytest.approx(180.0)

    def test_result_always_in_zero_to_180(self):
        for h in range(0, 360, 15):
            v = make_vessel(heading_deg=float(h))
            d = heading_deviation(v, baseline_heading=45.0)
            assert 0.0 <= d <= 180.0, f"Out of range for heading {h}: {d}"

    def test_default_baseline_is_45_degrees(self):
        v = make_vessel(heading_deg=90.0)
        assert heading_deviation(v) == pytest.approx(45.0)

    def test_symmetric_around_baseline(self):
        # Deviation 30° above and 30° below the baseline should be equal.
        v_above = make_vessel(heading_deg=75.0)
        v_below = make_vessel(heading_deg=15.0)
        assert heading_deviation(v_above, baseline_heading=45.0) == pytest.approx(
            heading_deviation(v_below, baseline_heading=45.0)
        )


# ---------------------------------------------------------------------------
# consecutive_slow_steps
# ---------------------------------------------------------------------------

class TestConsecutiveSlowSteps:
    def test_empty_history_returns_zero(self):
        assert consecutive_slow_steps([]) == 0

    def test_all_fast_returns_zero(self):
        history = [entry(speed_kmh=20.0), entry(speed_kmh=25.0)]
        assert consecutive_slow_steps(history) == 0

    def test_single_slow_entry_returns_one(self):
        assert consecutive_slow_steps([entry(speed_kmh=2.0)]) == 1

    def test_all_slow_capped_at_max_lookback(self):
        history = [entry(speed_kmh=2.0) for _ in range(10)]
        assert consecutive_slow_steps(history, max_lookback=3) == 3

    def test_fast_most_recent_breaks_chain_immediately(self):
        # Last entry is fast — chain breaks before any slow step counted.
        history = [entry(speed_kmh=2.0), entry(speed_kmh=2.0), entry(speed_kmh=20.0)]
        assert consecutive_slow_steps(history) == 0

    def test_fast_in_middle_breaks_chain(self):
        # oldest=slow, middle=fast, newest=slow — only newest counted.
        history = [entry(speed_kmh=2.0), entry(speed_kmh=20.0), entry(speed_kmh=2.0)]
        assert consecutive_slow_steps(history) == 1

    def test_max_lookback_limits_window(self):
        history = [entry(speed_kmh=2.0) for _ in range(5)]
        assert consecutive_slow_steps(history, max_lookback=2) == 2

    def test_custom_threshold_above_entries(self):
        history = [entry(speed_kmh=8.0), entry(speed_kmh=8.0)]
        assert consecutive_slow_steps(history, threshold=10.0) == 2

    def test_custom_threshold_below_entries(self):
        history = [entry(speed_kmh=8.0), entry(speed_kmh=8.0)]
        assert consecutive_slow_steps(history, threshold=5.0) == 0

    def test_history_read_from_end(self):
        # First entry is fast, last two are slow — only last two should count.
        history = [entry(speed_kmh=20.0), entry(speed_kmh=2.0), entry(speed_kmh=2.0)]
        assert consecutive_slow_steps(history) == 2


# ---------------------------------------------------------------------------
# is_inside_zone
# ---------------------------------------------------------------------------

class TestIsInsideZone:
    def test_center_is_inside(self):
        assert is_inside_zone(make_vessel(lat=11.0, lon=21.0), TEST_ZONE) is True

    def test_far_outside_is_false(self):
        assert is_inside_zone(make_vessel(lat=0.0, lon=0.0), TEST_ZONE) is False

    def test_on_min_lat_boundary_is_inside(self):
        assert is_inside_zone(make_vessel(lat=10.0, lon=21.0), TEST_ZONE) is True

    def test_on_max_lat_boundary_is_inside(self):
        assert is_inside_zone(make_vessel(lat=12.0, lon=21.0), TEST_ZONE) is True

    def test_on_min_lon_boundary_is_inside(self):
        assert is_inside_zone(make_vessel(lat=11.0, lon=20.0), TEST_ZONE) is True

    def test_on_max_lon_boundary_is_inside(self):
        assert is_inside_zone(make_vessel(lat=11.0, lon=22.0), TEST_ZONE) is True

    def test_just_south_of_zone_is_false(self):
        assert is_inside_zone(make_vessel(lat=9.99, lon=21.0), TEST_ZONE) is False

    def test_just_north_of_zone_is_false(self):
        assert is_inside_zone(make_vessel(lat=12.01, lon=21.0), TEST_ZONE) is False

    def test_just_west_of_zone_is_false(self):
        assert is_inside_zone(make_vessel(lat=11.0, lon=19.99), TEST_ZONE) is False

    def test_just_east_of_zone_is_false(self):
        assert is_inside_zone(make_vessel(lat=11.0, lon=22.01), TEST_ZONE) is False


# ---------------------------------------------------------------------------
# distance_to_zone
# ---------------------------------------------------------------------------

class TestDistanceToZone:
    def test_inside_returns_zero(self):
        assert distance_to_zone(make_vessel(lat=11.0, lon=21.0), TEST_ZONE) == pytest.approx(0.0)

    def test_on_boundary_returns_zero(self):
        assert distance_to_zone(make_vessel(lat=10.0, lon=21.0), TEST_ZONE) == pytest.approx(0.0)

    def test_south_of_zone_returns_lat_gap(self):
        # 1 degree south of min_lat, on the lon centerline
        assert distance_to_zone(make_vessel(lat=9.0, lon=21.0), TEST_ZONE) == pytest.approx(1.0)

    def test_north_of_zone_returns_lat_gap(self):
        assert distance_to_zone(make_vessel(lat=13.0, lon=21.0), TEST_ZONE) == pytest.approx(1.0)

    def test_east_of_zone_returns_lon_gap(self):
        assert distance_to_zone(make_vessel(lat=11.0, lon=23.0), TEST_ZONE) == pytest.approx(1.0)

    def test_west_of_zone_returns_lon_gap(self):
        assert distance_to_zone(make_vessel(lat=11.0, lon=19.0), TEST_ZONE) == pytest.approx(1.0)

    def test_corner_uses_euclidean_not_manhattan(self):
        # 1 degree south and 1 degree east: Euclidean = sqrt(2), Manhattan = 2
        v = make_vessel(lat=9.0, lon=23.0)
        assert distance_to_zone(v, TEST_ZONE) == pytest.approx(math.sqrt(2.0))

    def test_distance_increases_moving_away(self):
        near = make_vessel(lat=9.5, lon=21.0)
        far = make_vessel(lat=8.0, lon=21.0)
        assert distance_to_zone(near, TEST_ZONE) < distance_to_zone(far, TEST_ZONE)

    def test_always_non_negative(self):
        positions = [
            (11.0, 21.0),  # inside
            (9.0, 21.0),   # south
            (13.0, 23.0),  # NE corner
            (0.0, 0.0),    # far away
        ]
        for lat, lon in positions:
            assert distance_to_zone(make_vessel(lat=lat, lon=lon), TEST_ZONE) >= 0.0
