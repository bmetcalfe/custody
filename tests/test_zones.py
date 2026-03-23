"""
Tests for Step 13a: Zone type, ZONES config, and best_zone_for_point helper.

Covers:
  - Zone dataclass construction and immutability
  - ZONES list in config: non-empty, all entries are Zone instances,
    geometry matches current single-zone baseline
  - best_zone_for_point: inside, outside (closest), empty list, ties
"""
import pytest

from custody.models import Zone
from custody.config import ZONES
from custody.features.zone_features import best_zone_for_point


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ALPHA = Zone(name="ZONE_ALPHA", min_lat=1.0, max_lat=1.6, min_lon=0.55, max_lon=0.95, halo=0.1)
BETA  = Zone(name="ZONE_BETA",  min_lat=3.0, max_lat=3.5, min_lon=2.0,  max_lon=2.5,  halo=0.05)


# ---------------------------------------------------------------------------
# Zone dataclass
# ---------------------------------------------------------------------------

class TestZone:
    def test_construction(self):
        z = Zone(name="TEST", min_lat=0.0, max_lat=1.0, min_lon=0.0, max_lon=1.0, halo=0.1)
        assert z.name == "TEST"
        assert z.min_lat == 0.0
        assert z.max_lat == 1.0
        assert z.min_lon == 0.0
        assert z.max_lon == 1.0
        assert z.halo == 0.1

    def test_is_frozen(self):
        z = Zone(name="TEST", min_lat=0.0, max_lat=1.0, min_lon=0.0, max_lon=1.0, halo=0.1)
        with pytest.raises((AttributeError, TypeError)):
            z.name = "CHANGED"  # type: ignore[misc]

    def test_equality(self):
        z1 = Zone(name="A", min_lat=0.0, max_lat=1.0, min_lon=0.0, max_lon=1.0, halo=0.1)
        z2 = Zone(name="A", min_lat=0.0, max_lat=1.0, min_lon=0.0, max_lon=1.0, halo=0.1)
        assert z1 == z2

    def test_inequality_on_name(self):
        z1 = Zone(name="A", min_lat=0.0, max_lat=1.0, min_lon=0.0, max_lon=1.0, halo=0.1)
        z2 = Zone(name="B", min_lat=0.0, max_lat=1.0, min_lon=0.0, max_lon=1.0, halo=0.1)
        assert z1 != z2


# ---------------------------------------------------------------------------
# ZONES config
# ---------------------------------------------------------------------------

class TestZonesConfig:
    def test_zones_is_nonempty(self):
        assert len(ZONES) >= 1

    def test_all_entries_are_zone_instances(self):
        for z in ZONES:
            assert isinstance(z, Zone), f"Expected Zone, got {type(z)}"

    def test_all_zones_have_names(self):
        for z in ZONES:
            assert isinstance(z.name, str) and z.name.strip(), \
                f"Zone has blank/missing name: {z!r}"

    def test_zone_alpha_geometry_matches_baseline(self):
        """ZONE_ALPHA must preserve the original SENSITIVE_ZONE geometry."""
        alpha = next(z for z in ZONES if z.name == "ZONE_ALPHA")
        assert alpha.min_lat == 1.0
        assert alpha.max_lat == 1.6
        assert alpha.min_lon == 0.55
        assert alpha.max_lon == 0.95
        assert alpha.halo == 0.1

    def test_zone_boundaries_are_consistent(self):
        for z in ZONES:
            assert z.min_lat < z.max_lat, f"{z.name}: min_lat >= max_lat"
            assert z.min_lon < z.max_lon, f"{z.name}: min_lon >= max_lon"
            assert z.halo > 0.0, f"{z.name}: halo must be positive"


# ---------------------------------------------------------------------------
# best_zone_for_point
# ---------------------------------------------------------------------------

class TestBestZoneForPoint:
    def test_empty_zones_returns_none(self):
        assert best_zone_for_point(1.3, 0.75, []) is None

    def test_point_inside_only_zone(self):
        assert best_zone_for_point(1.3, 0.75, [ALPHA]) is ALPHA

    def test_point_outside_only_zone_returns_it(self):
        # Outside but ALPHA is the only option
        assert best_zone_for_point(0.0, 0.0, [ALPHA]) is ALPHA

    def test_point_inside_first_zone_of_two(self):
        # Inside ALPHA, outside BETA
        result = best_zone_for_point(1.3, 0.75, [ALPHA, BETA])
        assert result is ALPHA

    def test_point_inside_second_zone_of_two(self):
        # Outside ALPHA, inside BETA
        result = best_zone_for_point(3.2, 2.2, [ALPHA, BETA])
        assert result is BETA

    def test_point_outside_both_returns_closer(self):
        # Point at (2.0, 1.5): equidistant in a symmetrical setup would be ALPHA or BETA;
        # here we just assert the returned zone is one of the two and is the closer one.
        # ALPHA boundary at lat=1.6; BETA boundary at lat=3.0.
        # Point lat=2.0 is 0.4° from ALPHA top and 1.0° from BETA bottom.
        result = best_zone_for_point(2.0, 0.75, [ALPHA, BETA])
        assert result is ALPHA

    def test_point_on_boundary_counts_as_inside(self):
        # Exactly on the min_lat boundary
        result = best_zone_for_point(1.0, 0.75, [ALPHA])
        assert result is ALPHA

    def test_single_zone_list_always_returns_that_zone(self):
        # Regardless of position, only one choice
        for lat, lon in [(0.0, 0.0), (1.3, 0.75), (10.0, 10.0)]:
            assert best_zone_for_point(lat, lon, [ALPHA]) is ALPHA

    def test_order_preserved_for_inside_tie(self):
        # Two overlapping zones, point inside both — first wins
        overlap = Zone(
            name="OVERLAP",
            min_lat=ALPHA.min_lat, max_lat=ALPHA.max_lat,
            min_lon=ALPHA.min_lon, max_lon=ALPHA.max_lon,
            halo=0.1,
        )
        result = best_zone_for_point(1.3, 0.75, [ALPHA, overlap])
        assert result is ALPHA

        result_reversed = best_zone_for_point(1.3, 0.75, [overlap, ALPHA])
        assert result_reversed is overlap
