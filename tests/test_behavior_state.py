"""
Unit tests for custody/behavior/state_machine.py.

Each test targets one clearly-named rule or edge case.  Positions are chosen
relative to SENSITIVE_ZONE (lat 1.0–1.6, lon 0.55–0.95) so the spatial rules
are easy to verify by inspection.

Structure:
  TestUnknown          — insufficient history and no-confirmation edge cases
  TestIdle             — speed below IDLE_SPEED_KMH with history confirmation
  TestLoiter           — speed in [IDLE, SLOW) with history confirmation
  TestApproach         — decreasing zone-distance trend
  TestEgress           — increasing distance after recent zone proximity
  TestTransit          — default for normal-speed movement
  TestBoundaryAndEdge  — threshold edges and ordering
"""
from datetime import datetime, UTC

import pytest

from custody.models import BehaviorState, HistoryEntry, Vessel
from custody.behavior.state_machine import (
    IDLE_SPEED_KMH,
    SLOW_SPEED_KMH,
    TREND_MIN_DEG,
    NEAR_ZONE_DEG,
    infer_state,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EPOCH = datetime(2026, 1, 1, tzinfo=UTC)

# Positions relative to SENSITIVE_ZONE (lat 1.0–1.6, lon 0.55–0.95)
INSIDE_LAT, INSIDE_LON       = 1.3, 0.75    # clearly inside
NEAR_LAT,   NEAR_LON         = 0.85, 0.75   # just outside (dist ≈ 0.15°)
FAR_LAT,    FAR_LON          = 0.0,  0.0    # far away
BOUNDARY_LAT, BOUNDARY_LON   = 1.0,  0.75   # on min_lat boundary


def vessel(lat=0.0, lon=0.0, speed=30.0, heading=45.0) -> Vessel:
    return Vessel(
        id="T", lat=lat, lon=lon,
        speed_kmh=speed, heading_deg=heading, last_seen=_EPOCH,
    )


def entry(lat=0.0, lon=0.0, speed=28.0, heading=45.0) -> HistoryEntry:
    return HistoryEntry(lat, lon, _EPOCH, speed, heading)


# ---------------------------------------------------------------------------
# TestUnknown
# ---------------------------------------------------------------------------

class TestUnknown:
    def test_empty_history_returns_unknown(self):
        v = vessel(speed=28.0)
        state, conf = infer_state(v, [])
        assert state is BehaviorState.UNKNOWN
        assert conf == pytest.approx(0.2)

    def test_idle_speed_no_history_confirmation_returns_unknown(self):
        # Current is idle but the only history entry shows fast speed.
        v = vessel(lat=FAR_LAT, lon=FAR_LON, speed=0.5)
        history = [entry(lat=FAR_LAT, lon=FAR_LON, speed=28.0)]
        state, conf = infer_state(v, history)
        assert state is BehaviorState.UNKNOWN
        assert conf == pytest.approx(0.2)

    def test_slow_speed_no_history_confirmation_returns_unknown(self):
        # Current speed in loiter band but history shows fast movement.
        v = vessel(lat=FAR_LAT, lon=FAR_LON, speed=3.0)
        history = [entry(lat=FAR_LAT, lon=FAR_LON, speed=28.0)]
        state, conf = infer_state(v, history)
        assert state is BehaviorState.UNKNOWN
        assert conf == pytest.approx(0.2)

    def test_single_entry_slow_speed_same_band_loiter(self):
        # Single slow history entry does provide one confirmation → LOITER not UNKNOWN.
        v = vessel(lat=FAR_LAT, lon=FAR_LON, speed=3.0)
        history = [entry(lat=FAR_LAT, lon=FAR_LON, speed=3.0)]
        state, conf = infer_state(v, history)
        assert state is BehaviorState.LOITER
        # Only 1 confirmation → moderate confidence.
        assert conf == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# TestIdle
# ---------------------------------------------------------------------------

class TestIdle:
    def test_idle_single_history_confirmation(self):
        v = vessel(lat=FAR_LAT, lon=FAR_LON, speed=0.5)
        history = [entry(lat=FAR_LAT, lon=FAR_LON, speed=0.5)]
        state, conf = infer_state(v, history)
        assert state is BehaviorState.IDLE
        assert conf == pytest.approx(0.6)

    def test_idle_multiple_confirmations_raises_confidence(self):
        v = vessel(lat=FAR_LAT, lon=FAR_LON, speed=0.5)
        history = [entry(speed=0.5) for _ in range(3)]
        state, conf = infer_state(v, history)
        assert state is BehaviorState.IDLE
        assert conf == pytest.approx(0.85)

    def test_idle_at_threshold_boundary_is_not_idle(self):
        # speed == IDLE_SPEED_KMH is NOT idle (< not <=).
        v = vessel(speed=IDLE_SPEED_KMH)
        history = [entry(speed=IDLE_SPEED_KMH)]
        state, _ = infer_state(v, history)
        assert state is not BehaviorState.IDLE

    def test_idle_zero_speed(self):
        v = vessel(speed=0.0)
        history = [entry(speed=0.0), entry(speed=0.0)]
        state, conf = infer_state(v, history)
        assert state is BehaviorState.IDLE
        assert conf == pytest.approx(0.85)

    def test_idle_takes_priority_over_zone_proximity(self):
        # Vessel is inside zone but completely stationary — IDLE overrides.
        v = vessel(lat=INSIDE_LAT, lon=INSIDE_LON, speed=0.5)
        history = [entry(lat=INSIDE_LAT, lon=INSIDE_LON, speed=0.5)]
        state, _ = infer_state(v, history)
        assert state is BehaviorState.IDLE


# ---------------------------------------------------------------------------
# TestLoiter
# ---------------------------------------------------------------------------

class TestLoiter:
    def test_loiter_single_history_confirmation(self):
        v = vessel(lat=FAR_LAT, lon=FAR_LON, speed=2.0)
        history = [entry(lat=FAR_LAT, lon=FAR_LON, speed=2.0)]
        state, conf = infer_state(v, history)
        assert state is BehaviorState.LOITER
        assert conf == pytest.approx(0.6)

    def test_loiter_multiple_confirmations_raises_confidence(self):
        v = vessel(lat=FAR_LAT, lon=FAR_LON, speed=2.0)
        history = [entry(speed=2.0) for _ in range(3)]
        state, conf = infer_state(v, history)
        assert state is BehaviorState.LOITER
        assert conf == pytest.approx(0.85)

    def test_loiter_takes_priority_over_zone_proximity(self):
        # Slow vessel near zone — speed rule fires before spatial trend.
        v = vessel(lat=NEAR_LAT, lon=NEAR_LON, speed=2.0)
        history = [
            entry(lat=FAR_LAT, lon=FAR_LON, speed=2.0),
            entry(lat=NEAR_LAT, lon=NEAR_LON, speed=2.0),
        ]
        state, _ = infer_state(v, history)
        assert state is BehaviorState.LOITER

    def test_loiter_at_upper_boundary_is_not_loiter(self):
        # speed == SLOW_SPEED_KMH is NOT loiter (< not <=).
        v = vessel(speed=SLOW_SPEED_KMH)
        history = [entry(speed=SLOW_SPEED_KMH), entry(speed=SLOW_SPEED_KMH)]
        state, _ = infer_state(v, history)
        assert state is not BehaviorState.LOITER


# ---------------------------------------------------------------------------
# TestApproach
# ---------------------------------------------------------------------------

class TestApproach:
    def _approaching_history(self, n=2):
        """n history entries converging toward the zone from far south."""
        # Zone min_lat = 1.0; vessel moves north one step at a time.
        # Positions: lat = 0.3, 0.5, 0.7, ... (step = 0.2°)
        return [
            entry(lat=0.3 + i * 0.2, lon=0.75, speed=20.0)
            for i in range(n)
        ]

    def test_approach_two_entry_history(self):
        # Oldest entry far from zone; current position closer.
        v = vessel(lat=0.85, lon=0.75, speed=20.0)
        history = self._approaching_history(2)   # lat 0.3, 0.5
        state, conf = infer_state(v, history)
        assert state is BehaviorState.APPROACH
        assert conf in (0.6, 0.85)

    def test_approach_consistent_trend_gives_high_confidence(self):
        # Four entries each closer than the last → consistent → 0.85.
        v = vessel(lat=0.9, lon=0.75, speed=20.0)
        history = self._approaching_history(4)   # lat 0.3, 0.5, 0.7, 0.9 → all closer
        # Re-make history to be strictly monotone toward zone.
        history = [
            entry(lat=0.2, lon=0.75, speed=20.0),
            entry(lat=0.4, lon=0.75, speed=20.0),
            entry(lat=0.6, lon=0.75, speed=20.0),
            entry(lat=0.8, lon=0.75, speed=20.0),
        ]
        state, conf = infer_state(v, history)
        assert state is BehaviorState.APPROACH
        assert conf == pytest.approx(0.85)

    def test_approach_inconsistent_trend_gives_lower_confidence(self):
        # History zigzags slightly — still net-approaching but not consistent.
        v = vessel(lat=0.85, lon=0.75, speed=20.0)
        history = [
            entry(lat=0.3, lon=0.75, speed=20.0),  # far
            entry(lat=0.6, lon=0.75, speed=20.0),  # closer
            entry(lat=0.5, lon=0.75, speed=20.0),  # stepped back
            entry(lat=0.7, lon=0.75, speed=20.0),  # closer again
        ]
        state, conf = infer_state(v, history)
        assert state is BehaviorState.APPROACH
        assert conf == pytest.approx(0.6)

    def test_approach_not_triggered_when_already_inside_zone(self):
        # Vessel is inside zone → current_dist = 0 → APPROACH guard prevents it.
        v = vessel(lat=INSIDE_LAT, lon=INSIDE_LON, speed=20.0)
        history = [
            entry(lat=0.5, lon=0.75, speed=20.0),
            entry(lat=0.8, lon=0.75, speed=20.0),
        ]
        state, _ = infer_state(v, history)
        assert state is not BehaviorState.APPROACH

    def test_approach_not_triggered_without_sufficient_trend(self):
        # Distance change is below TREND_MIN_DEG — not a real approach.
        small_delta = TREND_MIN_DEG * 0.5
        v = vessel(lat=0.5 + small_delta, lon=0.75, speed=20.0)
        history = [entry(lat=0.5, lon=0.75, speed=20.0)]
        state, _ = infer_state(v, history)
        assert state is not BehaviorState.APPROACH


# ---------------------------------------------------------------------------
# TestEgress
# ---------------------------------------------------------------------------

class TestEgress:
    def test_egress_after_being_inside_zone(self):
        # Was inside zone, now clearly outside and moving away.
        v = vessel(lat=INSIDE_LAT, lon=1.5, speed=26.0)   # lon 1.5 is outside zone (max_lon=0.95)
        history = [
            entry(lat=INSIDE_LAT, lon=INSIDE_LON, speed=2.0),   # inside zone
            entry(lat=INSIDE_LAT, lon=INSIDE_LON, speed=2.0),   # still inside
        ]
        state, conf = infer_state(v, history)
        assert state is BehaviorState.EGRESS
        assert conf in (0.6, 0.85)

    def test_egress_after_being_near_zone(self):
        # Was within NEAR_ZONE_DEG of zone boundary, now moving away.
        v = vessel(lat=0.3, lon=0.75, speed=26.0)
        history = [
            entry(lat=NEAR_LAT, lon=NEAR_LON, speed=20.0),   # near zone (dist < 0.2)
            entry(lat=0.6, lon=0.75, speed=26.0),             # still near
        ]
        state, _ = infer_state(v, history)
        assert state is BehaviorState.EGRESS

    def test_egress_strong_trend_gives_high_confidence(self):
        # Large distance increase → strong egress signal.
        v = vessel(lat=INSIDE_LAT, lon=2.5, speed=26.0)   # far east
        history = [
            entry(lat=INSIDE_LAT, lon=INSIDE_LON, speed=2.0),
            entry(lat=INSIDE_LAT, lon=INSIDE_LON, speed=2.0),
        ]
        state, conf = infer_state(v, history)
        assert state is BehaviorState.EGRESS
        assert conf == pytest.approx(0.85)

    def test_egress_not_triggered_when_not_near_zone(self):
        # Moving away from a far position — was never near zone, so not egress.
        v = vessel(lat=FAR_LAT - 0.5, lon=FAR_LON, speed=26.0)
        history = [
            entry(lat=FAR_LAT, lon=FAR_LON, speed=26.0),
            entry(lat=FAR_LAT - 0.3, lon=FAR_LON, speed=26.0),
        ]
        state, _ = infer_state(v, history)
        assert state is not BehaviorState.EGRESS

    def test_egress_not_triggered_without_sufficient_trend(self):
        # Near zone but distance barely changed — no clear egress.
        small_delta = TREND_MIN_DEG * 0.5
        v = vessel(lat=INSIDE_LAT, lon=INSIDE_LON + small_delta, speed=26.0)
        history = [
            entry(lat=INSIDE_LAT, lon=INSIDE_LON, speed=26.0),
            entry(lat=INSIDE_LAT, lon=INSIDE_LON, speed=26.0),
        ]
        state, _ = infer_state(v, history)
        assert state is not BehaviorState.EGRESS


# ---------------------------------------------------------------------------
# TestTransit
# ---------------------------------------------------------------------------

class TestTransit:
    def test_transit_default_for_fast_speed_far_from_zone(self):
        v = vessel(lat=FAR_LAT, lon=FAR_LON, speed=28.0)
        history = [entry(lat=FAR_LAT, lon=FAR_LON, speed=28.0)]
        state, conf = infer_state(v, history)
        assert state is BehaviorState.TRANSIT
        assert conf == pytest.approx(0.6)

    def test_transit_when_no_meaningful_zone_trend(self):
        # Fast speed, zone distance barely changes — no approach or egress.
        v = vessel(lat=FAR_LAT, lon=FAR_LON + 0.01, speed=28.0)
        history = [
            entry(lat=FAR_LAT, lon=FAR_LON, speed=28.0),
            entry(lat=FAR_LAT, lon=FAR_LON + 0.005, speed=28.0),
        ]
        state, _ = infer_state(v, history)
        assert state is BehaviorState.TRANSIT

    def test_transit_single_history_entry_fast_speed(self):
        v = vessel(speed=20.0)
        history = [entry(speed=20.0)]
        state, _ = infer_state(v, history)
        assert state is BehaviorState.TRANSIT


# ---------------------------------------------------------------------------
# TestBoundaryAndEdge
# ---------------------------------------------------------------------------

class TestBoundaryAndEdge:
    def test_recent_steps_window_limits_lookback(self):
        # 10 slow entries then a fast one at the end — the fast entry is within
        # the recent window so slow_count drops and, with fast current speed, we
        # should not get LOITER.
        from custody.behavior.state_machine import RECENT_STEPS
        slow_history = [entry(speed=2.0) for _ in range(RECENT_STEPS)]
        fast_history = [entry(speed=28.0)]
        history = slow_history + fast_history   # fast entry is most recent
        v = vessel(speed=28.0)
        state, _ = infer_state(v, history)
        assert state is not BehaviorState.LOITER

    def test_idle_and_loiter_are_mutually_exclusive_at_boundary(self):
        # At exactly IDLE_SPEED_KMH the vessel is in the LOITER band, not IDLE.
        v = vessel(speed=IDLE_SPEED_KMH)
        history = [entry(speed=IDLE_SPEED_KMH), entry(speed=IDLE_SPEED_KMH)]
        state, _ = infer_state(v, history)
        assert state is BehaviorState.LOITER
        assert state is not BehaviorState.IDLE

    def test_confidence_is_one_of_three_values(self):
        # Confidence must be one of the documented coarse bands.
        allowed = {0.2, 0.6, 0.85}
        cases = [
            (vessel(speed=0.5),  [entry(speed=0.5)]),
            (vessel(speed=2.0),  [entry(speed=2.0), entry(speed=2.0)]),
            (vessel(speed=28.0), [entry(speed=28.0)]),
            (vessel(speed=28.0), []),
        ]
        for v, h in cases:
            _, conf = infer_state(v, h)
            assert conf in allowed, f"Unexpected confidence {conf} for speed={v.speed_kmh}"

    def test_state_is_behavior_state_enum(self):
        v = vessel(speed=28.0)
        state, _ = infer_state(v, [entry(speed=28.0)])
        assert isinstance(state, BehaviorState)


# ---------------------------------------------------------------------------
# TestMultiZone — Step 13c
# ---------------------------------------------------------------------------

# A second zone far from ZONE_ALPHA (lat 1.0-1.6, lon 0.55-0.95).
_ZONE_B_LAT_MID = 8.0
_ZONE_B_LON_MID = 8.0


def _zone_b():
    from custody.models import Zone
    return Zone(name="ZONE_BETA", min_lat=7.5, max_lat=8.5, min_lon=7.5, max_lon=8.5, halo=0.1)


class TestMultiZone:
    """Verify that zone-proximity logic works correctly with multiple zones.

    Uses monkeypatch to replace custody.behavior.state_machine.ZONES so that
    the existing single-zone config is unaffected.
    """

    def test_single_zone_parity_approach(self, monkeypatch):
        """One-zone config must produce APPROACH identical to pre-multi-zone behavior."""
        from custody.config import ZONES
        monkeypatch.setattr("custody.behavior.state_machine.ZONES", ZONES)
        v = vessel(lat=0.85, lon=0.75, speed=20.0)
        history = [
            entry(lat=0.3, lon=0.75, speed=20.0),
            entry(lat=0.5, lon=0.75, speed=20.0),
        ]
        state, _ = infer_state(v, history)
        assert state is BehaviorState.APPROACH

    def test_single_zone_parity_egress(self, monkeypatch):
        """One-zone config must produce EGRESS identical to pre-multi-zone behavior."""
        from custody.config import ZONES
        monkeypatch.setattr("custody.behavior.state_machine.ZONES", ZONES)
        v = vessel(lat=INSIDE_LAT, lon=1.5, speed=26.0)
        history = [
            entry(lat=INSIDE_LAT, lon=INSIDE_LON, speed=2.0),
            entry(lat=INSIDE_LAT, lon=INSIDE_LON, speed=2.0),
        ]
        state, _ = infer_state(v, history)
        assert state is BehaviorState.EGRESS

    def test_egress_after_near_second_zone(self, monkeypatch):
        """Egress fires when history was near ZONE_BETA, not ZONE_ALPHA."""
        from custody.config import ZONES
        zb = _zone_b()
        monkeypatch.setattr("custody.behavior.state_machine.ZONES", [ZONES[0], zb])
        # Vessel moves away from ZONE_BETA
        v = vessel(lat=_ZONE_B_LAT_MID, lon=10.5, speed=26.0)
        history = [
            entry(lat=_ZONE_B_LAT_MID, lon=_ZONE_B_LON_MID, speed=2.0),   # inside ZONE_BETA
            entry(lat=_ZONE_B_LAT_MID, lon=_ZONE_B_LON_MID, speed=2.0),
        ]
        state, conf = infer_state(v, history)
        assert state is BehaviorState.EGRESS
        assert conf in (0.6, 0.85)

    def test_approach_toward_second_zone(self, monkeypatch):
        """Approach fires when vessel moves toward ZONE_BETA."""
        from custody.config import ZONES
        zb = _zone_b()
        monkeypatch.setattr("custody.behavior.state_machine.ZONES", [ZONES[0], zb])
        # Vessel at lat=8.8 (just outside ZONE_BETA max_lat=8.5), was at 9.5 — now closer
        v = vessel(lat=8.8, lon=_ZONE_B_LON_MID, speed=20.0)
        history = [
            entry(lat=9.5, lon=_ZONE_B_LON_MID, speed=20.0),
            entry(lat=9.2, lon=_ZONE_B_LON_MID, speed=20.0),
        ]
        state, _ = infer_state(v, history)
        assert state is BehaviorState.APPROACH

    def test_no_false_egress_when_far_from_all_zones(self, monkeypatch):
        """Moving away from a position far from every zone must not produce EGRESS."""
        from custody.config import ZONES
        zb = _zone_b()
        monkeypatch.setattr("custody.behavior.state_machine.ZONES", [ZONES[0], zb])
        # Vessel at lat=50 — far from both zones, increasing distance
        v = vessel(lat=50.5, lon=50.5, speed=26.0)
        history = [
            entry(lat=50.0, lon=50.0, speed=26.0),
            entry(lat=50.2, lon=50.2, speed=26.0),
        ]
        state, _ = infer_state(v, history)
        assert state is not BehaviorState.EGRESS

    def test_empty_zones_does_not_crash(self, monkeypatch):
        """Empty ZONES must not raise; spatial rules conservatively fall through."""
        monkeypatch.setattr("custody.behavior.state_machine.ZONES", [])
        v = vessel(lat=INSIDE_LAT, lon=INSIDE_LON, speed=26.0)
        history = [
            entry(lat=INSIDE_LAT, lon=INSIDE_LON, speed=26.0),
            entry(lat=INSIDE_LAT, lon=INSIDE_LON, speed=26.0),
        ]
        state, conf = infer_state(v, history)
        # With no zones the sentinel distance means no approach/egress fires.
        assert state in {BehaviorState.TRANSIT, BehaviorState.UNKNOWN}
        assert conf in {0.2, 0.6, 0.85}
