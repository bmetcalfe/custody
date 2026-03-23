"""
Explicit, deterministic behavior state inference for vessel tracks.

Rules are evaluated in priority order:
    IDLE → LOITER → APPROACH → EGRESS → TRANSIT → UNKNOWN

The function is conservative: when evidence is insufficient or a speed-based
rule fires, spatial trend rules are not evaluated.  A future developer should
be able to read each rule block and predict the output for any input.

Confidence bands:
    0.2  — weak or insufficient evidence  (UNKNOWN)
    0.6  — moderate rule match (single confirmation or short trend)
    0.85 — strong consistent match (multiple confirmations or long trend)
"""
from __future__ import annotations

from custody.config import (
    IDLE_SPEED_KMH,
    NEAR_ZONE_DEG,
    SLOW_SPEED_KMH,
    ZONES,
)
from custody.features.zone_features import distance_to_zone, is_inside_zone
from custody.models import BehaviorState, HistoryEntry, Vessel

# Zone proximity thresholds (degrees)
TREND_MIN_DEG = 0.03   # minimum net distance change to count as a trend

# History window
RECENT_STEPS = 4       # number of recent HistoryEntry objects to inspect


def _zone_dict(zone) -> dict:
    """Convert a Zone dataclass to the dict format expected by zone_features."""
    return {
        "min_lat": zone.min_lat,
        "max_lat": zone.max_lat,
        "min_lon": zone.min_lon,
        "max_lon": zone.max_lon,
    }


def _zone_dist(point) -> float:
    """Minimum distance to any zone in ZONES for an object exposing .lat and .lon.

    Returns a large sentinel value when ZONES is empty so that spatial trend
    rules conservatively fall through rather than firing incorrectly.
    """
    if not ZONES:
        return 1e9
    return min(distance_to_zone(point, _zone_dict(z)) for z in ZONES)


def _in_zone(point) -> bool:
    """True if an object exposing .lat and .lon is inside any zone in ZONES."""
    return any(is_inside_zone(point, _zone_dict(z)) for z in ZONES)


def infer_state(
    vessel: Vessel,
    history: list[HistoryEntry],
) -> tuple[BehaviorState, float]:
    """
    Infer the current operational state of a vessel from its recent history.

    Both *vessel* (current snapshot) and *history* (explicit context) are
    required.  Callers control which history is passed; this function does not
    reach into vessel.history.

    Rules are evaluated in the order shown below.  Speed-based rules (IDLE,
    LOITER) are checked before spatial trend rules (APPROACH, EGRESS) so that
    a slow vessel near the zone is classified by its movement pattern rather
    than its proximity.

    Args:
        vessel:  Current vessel snapshot.  Provides speed, position, heading.
        history: Ordered observation history, oldest entry first.  Only the
                 most recent RECENT_STEPS entries are used.

    Returns:
        (BehaviorState, confidence) where confidence is one of {0.2, 0.6, 0.85}.
    """
    recent = history[-RECENT_STEPS:] if len(history) > RECENT_STEPS else list(history)

    # ── UNKNOWN: no history at all ─────────────────────────────────────────────
    if not recent:
        return BehaviorState.UNKNOWN, 0.2

    # ── IDLE: vessel is effectively stationary ─────────────────────────────────
    # Requires current speed below threshold AND at least one history entry that
    # also shows idle speed.  A single snapshot of zero speed is not sufficient.
    if vessel.speed_kmh < IDLE_SPEED_KMH:
        idle_count = sum(1 for e in recent if e.speed_kmh < IDLE_SPEED_KMH)
        if idle_count >= 1:
            confidence = 0.85 if idle_count >= 2 else 0.6
            return BehaviorState.IDLE, confidence
        # Current idle but no history confirmation — fall through to UNKNOWN.

    # ── LOITER: slow movement, above idle threshold ────────────────────────────
    # Requires current speed in [IDLE_SPEED, SLOW_SPEED) AND at least one
    # history entry in the same band.
    elif IDLE_SPEED_KMH <= vessel.speed_kmh < SLOW_SPEED_KMH:
        slow_count = sum(
            1 for e in recent
            if IDLE_SPEED_KMH <= e.speed_kmh < SLOW_SPEED_KMH
        )
        if slow_count >= 1:
            confidence = 0.85 if slow_count >= 2 else 0.6
            return BehaviorState.LOITER, confidence
        # Current slow but no history confirmation — fall through to UNKNOWN.

    # ── APPROACH / EGRESS: spatial trend analysis ─────────────────────────────
    # Requires at least 2 history entries to compute a meaningful distance trend.
    elif len(recent) >= 2:
        oldest_dist = _zone_dist(recent[0])
        current_dist = _zone_dist(vessel)
        distance_trend = current_dist - oldest_dist   # negative = approaching

        # APPROACH: moving toward zone, not yet inside it.
        if distance_trend < -TREND_MIN_DEG and current_dist > 0:
            # Confidence is higher when every consecutive step also decreases.
            consistent = all(
                _zone_dist(recent[i + 1]) <= _zone_dist(recent[i])
                for i in range(len(recent) - 1)
            )
            return BehaviorState.APPROACH, (0.85 if consistent else 0.6)

        # EGRESS: moving away from zone AND was recently near or inside it.
        if distance_trend > TREND_MIN_DEG:
            was_near = any(_in_zone(e) or _zone_dist(e) < NEAR_ZONE_DEG for e in recent)
            if was_near:
                strong = distance_trend > TREND_MIN_DEG * 3
                return BehaviorState.EGRESS, (0.85 if strong else 0.6)

        # Trend insufficient or zone not relevant — fall through to TRANSIT.

    # ── TRANSIT: default for normal-speed movement ─────────────────────────────
    if vessel.speed_kmh >= SLOW_SPEED_KMH:
        return BehaviorState.TRANSIT, 0.6

    # ── UNKNOWN: slow speed without history confirmation, or early history ──────
    return BehaviorState.UNKNOWN, 0.2
