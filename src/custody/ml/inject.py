"""Anomaly injection for real AIS vessel tracks.

Composable functions that take a vessel timeline (list of dicts or
DataFrame) and return a modified copy with injected anomalous behavior.
Originals are never mutated.

Public API
----------
inject_loitering(timeline, start_hour, duration_hours, target_speed_kmh)
inject_heading_deviation(timeline, start_hour, duration_hours, mode, deviation_deg)
inject_zone_approach(timeline, start_hour, duration_hours, zone_lat, zone_lon)
"""
from __future__ import annotations

import copy
from datetime import timedelta

import numpy as np


# ---------------------------------------------------------------------------
# A. Loitering injection
# ---------------------------------------------------------------------------

def inject_loitering(
    timeline: list[dict],
    start_hour: int,
    duration_hours: int = 6,
    target_speed_kmh: float = 1.0,
    position_jitter_deg: float = 0.001,
    seed: int = 42,
) -> list[dict]:
    """Reduce speed and hold position within a time window.

    Args:
        timeline:           Vessel timeline (list of record dicts).
        start_hour:         Index of the first record to modify (0-based).
        duration_hours:     Number of consecutive records to modify.
        target_speed_kmh:   Speed to set during the loitering window.
        position_jitter_deg: Small random position noise (degrees) to
                            simulate drift rather than perfect stillness.
        seed:               Random seed for jitter.

    Returns:
        New timeline with modified records in the window.
    """
    out = [copy.deepcopy(r) for r in timeline]
    rng = np.random.default_rng(seed)
    end = min(start_hour + duration_hours, len(out))

    if start_hour >= len(out):
        return out

    anchor_lat = out[start_hour]["lat"]
    anchor_lon = out[start_hour]["lon"]

    for i in range(start_hour, end):
        out[i]["speed_kmh"] = target_speed_kmh
        out[i]["lat"] = anchor_lat + rng.normal(0, position_jitter_deg)
        out[i]["lon"] = anchor_lon + rng.normal(0, position_jitter_deg)

    return out


# ---------------------------------------------------------------------------
# B. Heading deviation injection
# ---------------------------------------------------------------------------

def inject_heading_deviation(
    timeline: list[dict],
    start_hour: int,
    duration_hours: int = 6,
    mode: str = "sustained",
    deviation_deg: float = 90.0,
    seed: int = 42,
) -> list[dict]:
    """Alter heading within a time window.

    Args:
        timeline:       Vessel timeline.
        start_hour:     First record to modify.
        duration_hours: Window length.
        mode:           "sustained" for a constant offset, or "erratic"
                        for random heading each step.
        deviation_deg:  For "sustained": offset added to original heading.
                        For "erratic": std dev of random heading changes.
        seed:           Random seed for erratic mode.

    Returns:
        New timeline with modified headings.
    """
    out = [copy.deepcopy(r) for r in timeline]
    rng = np.random.default_rng(seed)
    end = min(start_hour + duration_hours, len(out))

    for i in range(start_hour, end):
        original = out[i]["heading_deg"]
        if mode == "sustained":
            out[i]["heading_deg"] = (original + deviation_deg) % 360.0
        elif mode == "erratic":
            out[i]["heading_deg"] = rng.uniform(0, 360)
        else:
            raise ValueError(f"Unknown mode: {mode!r}. Use 'sustained' or 'erratic'.")

    return out


# ---------------------------------------------------------------------------
# C. Zone approach / intrusion injection
# ---------------------------------------------------------------------------

def inject_zone_approach(
    timeline: list[dict],
    start_hour: int,
    duration_hours: int = 8,
    zone_lat: float = 30.0,
    zone_lon: float = -88.0,
    approach_speed_kmh: float = 12.0,
) -> list[dict]:
    """Modify trajectory to approach and enter a target zone.

    Linearly interpolates position from the vessel's current location
    at ``start_hour`` toward ``(zone_lat, zone_lon)`` over the window.
    Speed is set to ``approach_speed_kmh`` and heading is computed from
    the bearing toward the zone.

    Args:
        timeline:           Vessel timeline.
        start_hour:         First record to modify.
        duration_hours:     Window length.
        zone_lat, zone_lon: Target point (zone center or entry point).
        approach_speed_kmh: Speed during approach.

    Returns:
        New timeline with modified positions, speeds, and headings.
    """
    out = [copy.deepcopy(r) for r in timeline]
    end = min(start_hour + duration_hours, len(out))

    if start_hour >= len(out):
        return out

    origin_lat = out[start_hour]["lat"]
    origin_lon = out[start_hour]["lon"]
    n_steps = end - start_hour

    for i in range(start_hour, end):
        t = (i - start_hour + 1) / n_steps  # 0→1 over the window
        out[i]["lat"] = origin_lat + t * (zone_lat - origin_lat)
        out[i]["lon"] = origin_lon + t * (zone_lon - origin_lon)
        out[i]["speed_kmh"] = approach_speed_kmh

        # Compute bearing toward zone from current position
        dlat = zone_lat - out[i]["lat"]
        dlon = zone_lon - out[i]["lon"]
        bearing = np.degrees(np.arctan2(dlon, dlat)) % 360.0
        out[i]["heading_deg"] = bearing

    return out
