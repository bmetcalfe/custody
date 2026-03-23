"""
Motion-derived feature calculations.

These functions extract reusable kinematic features from vessel state or
observation history.  They are pure and deterministic: given the same inputs
they always return the same output.

Policy decisions (scoring thresholds, weights, base values) live in the
detector layer — not here.
"""
from __future__ import annotations

from custody.models import HistoryEntry


def is_slow(vessel, threshold: float = 5.0) -> bool:
    """Return True if the vessel's current speed is strictly below *threshold*."""
    return vessel.speed_kmh < threshold


def heading_deviation(vessel, baseline_heading: float = 45.0) -> float:
    """
    Shortest angular distance in degrees between vessel heading and a baseline.

    Always returns a value in [0, 180].  Handles the 0°/360° wraparound
    correctly so that, e.g., 355° vs 5° yields 10° rather than 350°.

    Args:
        vessel:           Vessel whose current heading_deg is compared.
        baseline_heading: Reference heading in degrees (compass convention).

    Returns:
        Angular difference in degrees, in the range [0.0, 180.0].
    """
    diff = abs(vessel.heading_deg - baseline_heading) % 360
    return min(diff, 360 - diff)


def consecutive_slow_steps(
    history: list[HistoryEntry],
    threshold: float = 5.0,
    max_lookback: int = 3,
) -> int:
    """
    Count consecutive slow-speed entries from the most recent history entry
    backwards, up to *max_lookback* entries.

    History is passed explicitly so callers control which context is visible.
    The count stops as soon as a non-slow entry is encountered.

    Args:
        history:      Ordered list of HistoryEntry observations (oldest first).
        threshold:    Speed in km/h below which a step is considered slow.
        max_lookback: Maximum number of recent entries to inspect.

    Returns:
        Integer count in [0, max_lookback].
    """
    count = 0
    window = history[-max_lookback:] if len(history) > max_lookback else history
    for entry in reversed(window):
        if entry.speed_kmh < threshold:
            count += 1
        else:
            break
    return count
