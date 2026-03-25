"""Ground-track sampling for the overview map overlay.

Provides helpers to sample sub-satellite points around a reference time and
return them as PathLayer-compatible data rows.  Tracks are split at
anti-meridian crossings to prevent horizontal wrap artefacts in the renderer.

This module has no Streamlit dependency and can be tested independently.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from custody.orbit import satellite_subpoint
from custody.sensors import orbital_satrecs


# ---------------------------------------------------------------------------
# Sampling constants (callers may override per-call)
# ---------------------------------------------------------------------------

_HALF_WINDOW_MINUTES: int = 90   # ±90 min window around the reference time
_N_POINTS: int = 15              # total sample points → one every ~12 min
_ANTIMERIDIAN_THRESHOLD: float = 180.0  # longitude gap (°) that triggers a split


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ground_track_segments(
    satellite_id: str,
    t_center: datetime,
    n_points: int = _N_POINTS,
    half_window_minutes: int = _HALF_WINDOW_MINUTES,
) -> list[list[list[float]]]:
    """Sample sub-satellite points and return as contiguous path segments.

    Each segment is a list of [lon, lat] pairs.  The track is split wherever
    consecutive points have a longitude gap exceeding the anti-meridian
    threshold (180°) to avoid globe-spanning rendering artefacts.

    Args:
        satellite_id:        Orbital catalog ID (e.g. "SAR-1").
        t_center:            Reference time (centre of the sampling window).
        n_points:            Number of sample points (distributed evenly).
        half_window_minutes: Half the total window width in minutes.

    Returns:
        List of path segments; each segment has >= 2 points.
        Empty list if the satellite is unknown or all propagation calls fail.
    """
    satrec = orbital_satrecs().get(satellite_id)
    if satrec is None:
        return []

    total_minutes = half_window_minutes * 2
    step = timedelta(minutes=total_minutes / max(n_points - 1, 1))
    t0 = t_center - timedelta(minutes=half_window_minutes)

    points: list[list[float]] = []
    for i in range(n_points):
        result = satellite_subpoint(satrec, t0 + step * i)
        if result is not None:
            lat, lon = result
            points.append([lon, lat])

    return _split_antimeridian(points)


def satellite_current_positions(t_center: datetime) -> list[dict]:
    """Return the sub-satellite point for each orbital satellite at t_center.

    Used to render a small "current position" marker on the ground-track
    overlay so the viewer can see where each satellite is right now.

    Args:
        t_center: The reference time (typically the current simulation step).

    Returns:
        List of dicts with keys: lon, lat, satellite_id.
        Satellites whose SGP4 propagation fails at t_center are omitted.
    """
    positions: list[dict] = []
    for sat_id, satrec in orbital_satrecs().items():
        result = satellite_subpoint(satrec, t_center)
        if result is not None:
            lat, lon = result
            positions.append({"lon": lon, "lat": lat, "satellite_id": sat_id})
    return positions


def all_ground_track_layers_data(
    t_center: datetime,
    n_points: int = _N_POINTS,
    half_window_minutes: int = _HALF_WINDOW_MINUTES,
) -> list[dict]:
    """Return PathLayer-compatible data rows for all orbital satellites.

    Each row is a dict with:
        path:          list of [lon, lat] for one contiguous track segment
        satellite_id:  catalog ID of the satellite

    Anti-meridian splits may produce multiple rows per satellite.

    Args:
        t_center:            Reference time (centre of the sampling window).
        n_points:            Number of sample points per satellite.
        half_window_minutes: Half the sampling window in minutes.

    Returns:
        List of row dicts ready for use as a pydeck PathLayer data argument.
    """
    rows: list[dict] = []
    for sat_id in orbital_satrecs():
        for segment in ground_track_segments(sat_id, t_center, n_points, half_window_minutes):
            if len(segment) >= 2:
                rows.append({"path": segment, "satellite_id": sat_id})
    return rows


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _split_antimeridian(
    points: list[list[float]],
) -> list[list[list[float]]]:
    """Split a [lon, lat] point list at large longitude jumps.

    A jump larger than _ANTIMERIDIAN_THRESHOLD degrees indicates the track
    has crossed the anti-meridian; a new segment is started so that the
    renderer does not draw a line across the globe.

    Args:
        points: List of [lon, lat] pairs.

    Returns:
        List of segments; each segment contains >= 2 points.
        Points that would form a single-point segment are discarded.
    """
    if not points:
        return []

    segments: list[list[list[float]]] = []
    current: list[list[float]] = [points[0]]

    for pt in points[1:]:
        if abs(pt[0] - current[-1][0]) > _ANTIMERIDIAN_THRESHOLD:
            if len(current) >= 2:
                segments.append(current)
            current = [pt]
        else:
            current.append(pt)

    if len(current) >= 2:
        segments.append(current)

    return segments
