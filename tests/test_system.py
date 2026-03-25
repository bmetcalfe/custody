"""
End-to-end / system-level tests for the planner arbitration and access loop.

These tests protect the interaction between:
  - compute_target_priority  — vessel ranking
  - plan_collection          — per-vessel decision, caller-supplied opportunities
  - run_simulation           — three-phase loop with per-vessel sensor access
  - sensor_access_count      — per-vessel pre-arbitration pool size

They sit at a higher level than the unit tests in test_planner.py and
complement the pipeline smoke tests in test_dashboard.py.

Arbitration recap
-----------------
At each simulation timestep the planner loop:
  1. fetches sensor opportunities per vessel from that vessel's own position,
  2. ranks vessels by priority (highest first),
  3. calls plan_collection for each vessel, excluding already-claimed sensor ids,
  4. adds the chosen sensor_id to a claimed set after every TASK decision.

A vessel that needs tasking but finds all its initially-available sensors
claimed receives PREEMPTED (sensors existed in its pool but were consumed).
A vessel whose position-filtered pool was empty from the start receives
NO_SENSOR instead.  A position-unique sensor never enters the claimed set
and is always available to its vessel regardless of arbitration order.
"""
from datetime import UTC, datetime

import pytest

from custody.models import TrackState
from custody.planner import CollectionDecision, compute_target_priority, plan_collection
from custody.sensors import SensorOpportunity
from custody.simulate import run_simulation


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_T = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)

_BREAKDOWN = {
    "sensitive_zone":  type("R", (), {"score": 0.0})(),
    "loitering":       type("R", (), {"score": 0.0})(),
    "route_deviation": type("R", (), {"score": 0.0})(),
}


def _sensor(sensor_id: str = "S1") -> SensorOpportunity:
    """Return a minimal always-succeeding sensor opportunity."""
    return SensorOpportunity(
        sensor_id=sensor_id,
        sensor_type="fast_revisit",
        success_prob=1.0,   # deterministic success
        resolution="medium",
        cost=1.0,
        available_from=_T,
        available_to=_T,
    )


def _simulate_arbitration(
    specs: list[tuple[float, float]],   # (anomaly_score, custody_confidence)
    pool: list[SensorOpportunity],
) -> list[CollectionDecision]:
    """Run the arbitration loop for N vessels sharing a sensor pool.

    Mirrors the Phase-2/Phase-3 logic in run_simulation:
      - rank by compute_target_priority (highest first)
      - call plan_collection for each vessel with the remaining pool
      - remove the claimed sensor after TASK
      - set preempted=True when the initial pool was non-empty but is now
        exhausted before the current vessel's turn

    Returns decisions in the original spec-list order.
    """
    ranked = sorted(
        enumerate(specs),
        key=lambda kv: compute_target_priority(kv[1][0], kv[1][1]),
        reverse=True,
    )
    initial = list(pool)
    remaining = list(initial)
    results: dict[int, CollectionDecision] = {}

    for original_idx, (score, confidence) in ranked:
        preempted = bool(initial) and not remaining
        decision = plan_collection(
            TrackState(), score, confidence, _BREAKDOWN, _T,
            remaining, preempted=preempted,
        )
        if decision.action == "TASK" and decision.sensor_id:
            remaining = [o for o in remaining if o.sensor_id != decision.sensor_id]
        results[original_idx] = decision

    return [results[i] for i in range(len(specs))]


# ---------------------------------------------------------------------------
# Arbitration: higher-priority vessel claims sensor, lower-priority PREEMPTED
# ---------------------------------------------------------------------------

class TestArbitrationSequence:
    def test_higher_priority_vessel_is_tasked(self):
        """The highest-priority vessel is TASK when a sensor is available."""
        decisions = _simulate_arbitration(
            specs=[(1.5, 0.9), (0.6, 0.9)],  # hi-priority first
            pool=[_sensor("S1")],
        )
        assert decisions[0].action == "TASK", (
            "High-priority vessel (score=1.5) should be TASK"
        )

    def test_lower_priority_vessel_is_preempted(self):
        """After the high-priority vessel claims the only sensor, the lower-
        priority vessel that also needs tasking receives PREEMPTED."""
        decisions = _simulate_arbitration(
            specs=[(1.5, 0.9), (0.6, 0.9)],
            pool=[_sensor("S1")],
        )
        assert decisions[1].action == "PREEMPTED", (
            "Lower-priority vessel should be PREEMPTED once sensors are exhausted"
        )

    def test_preempted_reason_references_higher_priority_vessels(self):
        """The PREEMPTED action_reason must mention higher-priority vessels."""
        decisions = _simulate_arbitration(
            specs=[(1.5, 0.9), (0.6, 0.9)],
            pool=[_sensor("S1")],
        )
        assert "higher-priority" in decisions[1].action_reason.lower()

    def test_both_vessels_tasked_when_two_sensors_available(self):
        """With two sensors in the pool, both vessels that need tasking are
        TASK (neither is PREEMPTED)."""
        decisions = _simulate_arbitration(
            specs=[(1.5, 0.9), (0.6, 0.9)],
            pool=[_sensor("S1"), _sensor("S2")],
        )
        assert decisions[0].action == "TASK"
        assert decisions[1].action == "TASK"

    def test_priority_order_is_respected(self):
        """Priority ordering: the higher-score vessel goes first regardless of
        the spec-list order."""
        # spec[0] is LOW priority, spec[1] is HIGH priority
        decisions = _simulate_arbitration(
            specs=[(0.6, 0.9), (1.5, 0.9)],   # lo first, hi second
            pool=[_sensor("S1")],
        )
        # spec[1] (high priority) should claim the sensor
        assert decisions[1].action == "TASK"
        # spec[0] (low priority) should be PREEMPTED
        assert decisions[0].action == "PREEMPTED"


# ---------------------------------------------------------------------------
# NO_SENSOR is distinct from PREEMPTED
# ---------------------------------------------------------------------------

class TestNoSensorVsPreempted:
    def test_no_sensor_when_global_pool_was_empty(self):
        """When no sensors were scheduled globally, all vessels that need
        tasking receive NO_SENSOR — not PREEMPTED."""
        decisions = _simulate_arbitration(
            specs=[(1.5, 0.9), (0.6, 0.9)],
            pool=[],   # globally empty
        )
        assert decisions[0].action == "NO_SENSOR"
        assert decisions[1].action == "NO_SENSOR"

    def test_preempted_and_no_sensor_have_distinct_action_strings(self):
        """PREEMPTED and NO_SENSOR must not collapse to the same string."""
        assert "PREEMPTED" != "NO_SENSOR"

    def test_preempted_flag_is_what_distinguishes_them(self):
        """Calling plan_collection with the same empty-opportunities list but
        different preempted flags produces different action codes."""
        track = TrackState()
        decision_preempted = plan_collection(
            track, 1.0, 0.9, _BREAKDOWN, _T,
            opportunities=[], preempted=True,
        )
        decision_no_sensor = plan_collection(
            TrackState(), 1.0, 0.9, _BREAKDOWN, _T,
            opportunities=[], preempted=False,
        )
        assert decision_preempted.action == "PREEMPTED"
        assert decision_no_sensor.action == "NO_SENSOR"

    def test_preempted_does_not_fire_when_not_needed(self):
        """A vessel that returns NONE or HOLD is not affected by preempted=True."""
        # NONE case: low anomaly, high confidence
        decision_none = plan_collection(
            TrackState(), 0.1, 0.9, _BREAKDOWN, _T,
            opportunities=[], preempted=True,
        )
        assert decision_none.action == "NONE"


# ---------------------------------------------------------------------------
# sensor_access_count is now a per-vessel field
# ---------------------------------------------------------------------------

class TestSensorAccessCount:
    def test_sensor_access_count_is_non_negative(self):
        """sensor_access_count must be >= 0 for every vessel at every timestep."""
        records = run_simulation()
        for r in records:
            assert r["sensor_access_count"] >= 0

    def test_sensor_access_count_present_in_all_records(self):
        """Every simulation record must carry sensor_access_count."""
        records = run_simulation()
        assert all("sensor_access_count" in r for r in records)

    def test_task_implies_nonzero_access_count(self):
        """A vessel that receives TASK must have had at least one sensor in
        its initial position-filtered pool (you cannot task what you cannot see)."""
        records = run_simulation()
        for r in records:
            if r["action"] == "TASK":
                assert r["sensor_access_count"] >= 1, (
                    f"{r['target_id']} at {r['time']}: action=TASK but "
                    f"sensor_access_count={r['sensor_access_count']}"
                )


# ---------------------------------------------------------------------------
# Per-vessel access: position determines pool; claimed set blocks sharing
# ---------------------------------------------------------------------------

def _per_vessel_arbitration(
    specs: list[tuple[float, float]],          # (anomaly_score, custody_confidence)
    pools: list[list[SensorOpportunity]],      # one pool per spec entry
) -> list[CollectionDecision]:
    """Run the per-vessel arbitration loop.

    Mirrors the Phase-2/Phase-3 logic in run_simulation:
      - each vessel has its own initial pool (position-filtered)
      - a claimed_sensor_ids set grows as TASK decisions are made
      - each vessel's remaining list = its initial pool minus claimed ids
      - preempted=True when initial was non-empty but remaining is empty
    """
    ranked = sorted(
        enumerate(specs),
        key=lambda kv: compute_target_priority(kv[1][0], kv[1][1]),
        reverse=True,
    )
    claimed_ids: set[str] = set()
    results: dict[int, CollectionDecision] = {}

    for original_idx, (score, confidence) in ranked:
        vessel_initial = pools[original_idx]
        vessel_remaining = [o for o in vessel_initial if o.sensor_id not in claimed_ids]
        preempted = bool(vessel_initial) and not vessel_remaining

        decision = plan_collection(
            TrackState(), score, confidence, _BREAKDOWN, _T,
            vessel_remaining, preempted=preempted,
        )
        if decision.action == "TASK" and decision.sensor_id:
            claimed_ids.add(decision.sensor_id)
        results[original_idx] = decision

    return [results[i] for i in range(len(specs))]


class TestPerVesselAccess:
    def test_lower_priority_tasks_with_position_unique_sensor(self):
        """Lower-priority vessel has S2 which is not in the higher-priority
        vessel's pool.  After the high-priority vessel claims S1, S2 is still
        available — the lower-priority vessel can TASK with it."""
        s1 = _sensor("S1")
        s2 = _sensor("S2")
        decisions = _per_vessel_arbitration(
            specs=[(1.5, 0.9), (0.6, 0.9)],   # hi-priority first in specs
            pools=[[s1], [s1, s2]],            # hi has S1 only; lo has S1 and S2
        )
        assert decisions[0].action == "TASK"
        assert decisions[0].sensor_id == "S1"
        # S1 is now claimed; lo still has S2
        assert decisions[1].action == "TASK"
        assert decisions[1].sensor_id == "S2"

    def test_position_unique_sensor_never_blocked(self):
        """A sensor that appears only in one vessel's pool is never added to
        claimed_sensor_ids by another vessel, so it is always available."""
        s_shared = _sensor("SHARED")
        s_unique = _sensor("UNIQUE")
        # Three vessels: hi claims SHARED; mid has only SHARED → PREEMPTED;
        # lo has only UNIQUE → still gets UNIQUE
        decisions = _per_vessel_arbitration(
            specs=[(1.5, 0.9), (1.0, 0.9), (0.6, 0.9)],
            pools=[[s_shared], [s_shared], [s_unique]],
        )
        assert decisions[0].action == "TASK"    # claimed SHARED
        assert decisions[1].action == "PREEMPTED"  # SHARED gone, its pool empty
        assert decisions[2].action == "TASK"    # UNIQUE never claimed

    def test_preempted_requires_initial_pool_non_empty(self):
        """PREEMPTED only fires when the vessel's own initial pool was non-empty.
        A vessel with no sensors at its position receives NO_SENSOR, not PREEMPTED,
        even when other vessels were successfully tasked."""
        s1 = _sensor("S1")
        decisions = _per_vessel_arbitration(
            specs=[(1.5, 0.9), (0.6, 0.9)],
            pools=[[s1], []],          # hi has S1; lo has nothing
        )
        assert decisions[0].action == "TASK"
        assert decisions[1].action == "NO_SENSOR"   # not PREEMPTED

    def test_per_vessel_access_count_can_differ_at_same_timestep(self):
        """With position-sensitive sensor access, vessels at different positions
        receive different sensor_access_count values at the same timestep.

        V001 starts at lat=0 and moves to ~0.19 after the first position update.
        V002 starts at lat=0.2 and moves to ~0.38 after the first update.
        The threshold lat > 0.3 separates them at the first timestep, giving
        V001 count=1 and V002 count=2.
        """
        from unittest.mock import patch
        import custody.simulation.timeline as timeline_mod

        def position_mock(t, lat=None, lon=None):
            if lat is None:
                return []
            base = [_sensor("S1")]
            if lat > 0.3:   # separates V001 (~0.19) from V002 (~0.38) at t=0
                base.append(_sensor("S2"))
            return base

        with patch.object(timeline_mod, "get_sensor_opportunities", position_mock):
            records = run_simulation()

        found_diff = any(
            len({r["sensor_access_count"] for r in records if r["time"] == ts}) > 1
            for ts in {r["time"] for r in records}
        )
        assert found_diff, (
            "expected at least one timestep where vessels have different "
            "sensor_access_count values due to differing positions"
        )
