"""
Geometric / proximity feature calculations for zone bounding boxes.

All functions accept explicit vessel and zone arguments.  They return raw
geometric measurements; scoring policy (ramp constants, halo thresholds)
lives in the detector layer.

Legacy dict-based functions (is_inside_zone, distance_to_zone) accept:
    {"min_lat": float, "max_lat": float, "min_lon": float, "max_lon": float}

The newer best_zone_for_point helper accepts Zone dataclass instances.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from custody.models import Zone


def is_inside_zone(vessel, zone: dict) -> bool:
    """Return True if the vessel's current position is inside the zone bounding box."""
    return (
        zone["min_lat"] <= vessel.lat <= zone["max_lat"]
        and zone["min_lon"] <= vessel.lon <= zone["max_lon"]
    )


def distance_to_zone(vessel, zone: dict) -> float:
    """
    Euclidean distance in degrees from the vessel to the nearest zone edge.

    Returns 0.0 if the vessel is on the boundary or inside the zone.
    Uses an axis-aligned bounding-box model consistent with is_inside_zone.

    Args:
        vessel: Vessel whose current lat/lon is measured.
        zone:   Zone bounding box dict with min/max lat and lon keys.

    Returns:
        Distance in degrees (>= 0.0).
    """
    if is_inside_zone(vessel, zone):
        return 0.0
    lat_gap = max(zone["min_lat"] - vessel.lat, vessel.lat - zone["max_lat"], 0.0)
    lon_gap = max(zone["min_lon"] - vessel.lon, vessel.lon - zone["max_lon"], 0.0)
    return math.sqrt(lat_gap ** 2 + lon_gap ** 2)


def best_zone_for_point(lat: float, lon: float, zones: list[Zone]) -> Zone | None:
    """Return the most relevant Zone for a given lat/lon point.

    Selection priority:
      1. If the point is inside one or more zones, return the first such zone
         (preserving list order from the caller — typically config.ZONES).
      2. If the point is outside all zones, return the zone with the nearest
         boundary (minimum Euclidean distance in degrees).
      3. If zones is empty, return None.

    Args:
        lat:   Latitude of the point in degrees.
        lon:   Longitude of the point in degrees.
        zones: Ordered list of Zone instances to evaluate.

    Returns:
        The most relevant Zone, or None if zones is empty.
    """
    if not zones:
        return None

    for zone in zones:
        if zone.min_lat <= lat <= zone.max_lat and zone.min_lon <= lon <= zone.max_lon:
            return zone

    def _boundary_distance(zone: Zone) -> float:
        lat_gap = max(zone.min_lat - lat, lat - zone.max_lat, 0.0)
        lon_gap = max(zone.min_lon - lon, lon - zone.max_lon, 0.0)
        return math.sqrt(lat_gap ** 2 + lon_gap ** 2)

    return min(zones, key=_boundary_distance)
