"""
Vessel-to-vessel proximity feature calculations.

All functions are pure and independent of the rest of the custody package.
No imports from custody models or config — inputs are raw floats and plain dicts.

Distance uses the Haversine formula for spherical great-circle accuracy,
consistent with maritime use cases where Euclidean degree-distance becomes
misleading beyond a few dozen kilometres.
"""
from __future__ import annotations

import math


# Earth mean radius in km (WGS-84 approximation).
_EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in kilometres.

    Uses the Haversine formula.  Returns 0.0 immediately for identical
    coordinates to avoid floating-point noise near zero.

    Args:
        lat1, lon1: Latitude and longitude of the first point (degrees).
        lat2, lon2: Latitude and longitude of the second point (degrees).

    Returns:
        Distance in kilometres (>= 0.0).
    """
    if lat1 == lat2 and lon1 == lon2:
        return 0.0

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lam = math.radians(lon2 - lon1)

    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2) ** 2
    return _EARTH_RADIUS_KM * 2 * math.asin(math.sqrt(a))


def nearest_vessel_from_records(
    lat: float,
    lon: float,
    vessel_id: str,
    others: list[dict],
) -> tuple[str | None, float]:
    """Find the nearest other vessel from a list of co-temporal timeline records.

    Self-records are excluded by matching on ``target_id``.  Records without a
    ``target_id`` key are skipped silently.

    Args:
        lat:       Latitude of the reference vessel (degrees).
        lon:       Longitude of the reference vessel (degrees).
        vessel_id: ID of the reference vessel — used to exclude self-records.
        others:    List of timeline record dicts for other vessels at the same
                   timestep.  Each dict must have ``"lat"``, ``"lon"``, and
                   ``"target_id"`` keys.

    Returns:
        ``(nearest_id, distance_km)`` where ``nearest_id`` is the ``target_id``
        of the closest vessel and ``distance_km`` is the Haversine distance.
        Returns ``(None, math.inf)`` when *others* is empty or every record
        matches *vessel_id*.
    """
    nearest_id: str | None = None
    nearest_km = math.inf

    for record in others:
        tid = record.get("target_id")
        if tid is None or tid == vessel_id:
            continue
        dist = haversine_km(lat, lon, float(record["lat"]), float(record["lon"]))
        if dist < nearest_km:
            nearest_km = dist
            nearest_id = tid

    return nearest_id, nearest_km


def group_records_by_time(
    records: list[dict],
    time_key: str = "time",
) -> dict[object, list[dict]]:
    """Group a flat list of timeline records by their timestamp value.

    Preserves insertion order within each group.  The timestamp values are
    used as dict keys directly — callers are responsible for ensuring they
    are hashable (e.g. ``datetime`` objects).

    Args:
        records:  Flat list of timeline record dicts from any replay path.
        time_key: The key used for the timestamp in each record.  Defaults
                  to ``"time"``, which is the field name emitted by both
                  ``simulate_target`` and ``ingest_ais_track``.

    Returns:
        Dict mapping each unique timestamp to the list of records at that time.
    """
    groups: dict[object, list[dict]] = {}
    for record in records:
        t = record[time_key]
        groups.setdefault(t, []).append(record)
    return groups
