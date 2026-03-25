"""Tests for overview_events helpers."""
from __future__ import annotations

import pytest
import pandas as pd

from overview_events import (
    build_event_feed,
    detect_rank_change_events,
    detect_health_change_events,
    detect_neglect_events,
    detect_zone_entry_events,
    detect_preemption_events,
    detect_zone_approach_events,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _df(**cols) -> pd.DataFrame:
    """Build a 1-row DataFrame; target_id defaults to 'A'."""
    row = {"target_id": "A"}
    row.update(cols)
    return pd.DataFrame([row])


def _two(curr_vals: dict, prev_vals: dict, eid: str = "A") -> tuple[pd.DataFrame, pd.DataFrame]:
    curr = {"target_id": eid}
    curr.update(curr_vals)
    prev = {"target_id": eid}
    prev.update(prev_vals)
    return pd.DataFrame([curr]), pd.DataFrame([prev])


# ── detect_rank_change_events ─────────────────────────────────────────────────

class TestDetectRankChangeEvents:

    def test_large_improvement_detected(self):
        curr, prev = _two({"portfolio_rank": 2}, {"portfolio_rank": 8})
        events = detect_rank_change_events(curr, prev)
        assert len(events) == 1
        assert events[0]["event_type"] == "rank_change"
        assert events[0]["delta"] > 0  # improved
        assert "↑" in events[0]["description"]

    def test_large_decline_detected(self):
        curr, prev = _two({"portfolio_rank": 10}, {"portfolio_rank": 2})
        events = detect_rank_change_events(curr, prev)
        assert len(events) == 1
        assert events[0]["delta"] < 0  # worsened
        assert "↓" in events[0]["description"]

    def test_small_change_not_detected(self):
        curr, prev = _two({"portfolio_rank": 3}, {"portfolio_rank": 4})
        events = detect_rank_change_events(curr, prev, threshold=3)
        assert events == []

    def test_exact_threshold_detected(self):
        curr, prev = _two({"portfolio_rank": 1}, {"portfolio_rank": 4})
        events = detect_rank_change_events(curr, prev, threshold=3)
        assert len(events) == 1

    def test_missing_portfolio_rank_returns_empty(self):
        curr = _df(anomaly_score=1.0)
        prev = _df(anomaly_score=0.5)
        assert detect_rank_change_events(curr, prev) == []

    def test_empty_current_returns_empty(self):
        assert detect_rank_change_events(pd.DataFrame(), _df(portfolio_rank=1)) == []

    def test_empty_prev_returns_empty(self):
        assert detect_rank_change_events(_df(portfolio_rank=1), pd.DataFrame()) == []

    def test_entity_not_in_prev_ignored(self):
        curr = pd.DataFrame([{"target_id": "A", "portfolio_rank": 1}])
        prev = pd.DataFrame([{"target_id": "B", "portfolio_rank": 8}])
        assert detect_rank_change_events(curr, prev) == []

    def test_sorted_by_abs_delta_descending(self):
        curr = pd.DataFrame([
            {"target_id": "A", "portfolio_rank": 1},
            {"target_id": "B", "portfolio_rank": 2},
        ])
        prev = pd.DataFrame([
            {"target_id": "A", "portfolio_rank": 10},   # delta=9
            {"target_id": "B", "portfolio_rank":  5},   # delta=3
        ])
        events = detect_rank_change_events(curr, prev)
        assert events[0]["entity_id"] == "A"


# ── detect_health_change_events ───────────────────────────────────────────────

class TestDetectHealthChangeEvents:

    def test_healthy_to_degrading_detected(self):
        curr, prev = _two({"custody_health": "DEGRADING"}, {"custody_health": "HEALTHY"})
        events = detect_health_change_events(curr, prev)
        assert len(events) == 1
        assert events[0]["event_type"] == "health_worsened"
        assert events[0]["from_health"] == "HEALTHY"
        assert events[0]["to_health"] == "DEGRADING"

    def test_degrading_to_stale_detected(self):
        curr, prev = _two({"custody_health": "STALE"}, {"custody_health": "DEGRADING"})
        events = detect_health_change_events(curr, prev)
        assert len(events) == 1

    def test_stale_to_lost_detected(self):
        curr, prev = _two({"custody_health": "LOST"}, {"custody_health": "STALE"})
        events = detect_health_change_events(curr, prev)
        assert len(events) == 1

    def test_health_improvement_not_detected(self):
        curr, prev = _two({"custody_health": "HEALTHY"}, {"custody_health": "DEGRADING"})
        events = detect_health_change_events(curr, prev)
        assert events == []

    def test_same_health_not_detected(self):
        curr, prev = _two({"custody_health": "STALE"}, {"custody_health": "STALE"})
        assert detect_health_change_events(curr, prev) == []

    def test_missing_column_returns_empty(self):
        curr = _df(anomaly_score=1.0)
        prev = _df(anomaly_score=0.5)
        assert detect_health_change_events(curr, prev) == []

    def test_empty_df_returns_empty(self):
        assert detect_health_change_events(pd.DataFrame(), _df(custody_health="STALE")) == []


# ── detect_neglect_events ─────────────────────────────────────────────────────

class TestDetectNeglectEvents:

    def test_false_to_true_detected(self):
        curr, prev = _two({"neglect_flag": True}, {"neglect_flag": False})
        events = detect_neglect_events(curr, prev)
        assert len(events) == 1
        assert events[0]["event_type"] == "neglect_triggered"
        assert "A" in events[0]["description"]

    def test_already_neglected_not_repeated(self):
        curr, prev = _two({"neglect_flag": True}, {"neglect_flag": True})
        assert detect_neglect_events(curr, prev) == []

    def test_neglect_cleared_not_detected(self):
        curr, prev = _two({"neglect_flag": False}, {"neglect_flag": True})
        assert detect_neglect_events(curr, prev) == []

    def test_missing_column_returns_empty(self):
        curr = _df(anomaly_score=1.0)
        prev = _df(anomaly_score=0.5)
        assert detect_neglect_events(curr, prev) == []

    def test_empty_df_returns_empty(self):
        assert detect_neglect_events(pd.DataFrame(), _df(neglect_flag=True)) == []


# ── detect_zone_entry_events ──────────────────────────────────────────────────

class TestDetectZoneEntryEvents:

    def test_zone_entry_detected(self):
        curr, prev = _two({"sensitive_zone": 0.8}, {"sensitive_zone": 0.1})
        events = detect_zone_entry_events(curr, prev)
        assert len(events) == 1
        assert events[0]["event_type"] == "zone_entry"
        assert "A" in events[0]["description"]

    def test_already_in_zone_not_detected(self):
        curr, prev = _two({"sensitive_zone": 0.9}, {"sensitive_zone": 0.5})
        assert detect_zone_entry_events(curr, prev) == []

    def test_zone_exit_not_detected(self):
        curr, prev = _two({"sensitive_zone": 0.1}, {"sensitive_zone": 0.9})
        assert detect_zone_entry_events(curr, prev) == []

    def test_below_threshold_not_detected(self):
        curr, prev = _two({"sensitive_zone": 0.2}, {"sensitive_zone": 0.0})
        assert detect_zone_entry_events(curr, prev, threshold=0.3) == []

    def test_custom_threshold(self):
        curr, prev = _two({"sensitive_zone": 0.5}, {"sensitive_zone": 0.1})
        events = detect_zone_entry_events(curr, prev, threshold=0.4)
        assert len(events) == 1

    def test_missing_column_returns_empty(self):
        curr = _df(anomaly_score=1.0)
        prev = _df(anomaly_score=0.5)
        assert detect_zone_entry_events(curr, prev) == []


# ── detect_preemption_events ──────────────────────────────────────────────────

class TestDetectPreemptionEvents:

    def test_newly_preempted_detected(self):
        curr, prev = _two({"action": "PREEMPTED"}, {"action": "NONE"})
        events = detect_preemption_events(curr, prev)
        assert len(events) == 1
        assert events[0]["event_type"] == "preempted"

    def test_already_preempted_not_repeated(self):
        curr, prev = _two({"action": "PREEMPTED"}, {"action": "PREEMPTED"})
        assert detect_preemption_events(curr, prev) == []

    def test_deferred_for_in_description(self):
        curr = pd.DataFrame([{
            "target_id": "B", "action": "PREEMPTED", "deferred_for": "BRAVO-1"
        }])
        prev = pd.DataFrame([{"target_id": "B", "action": "NONE"}])
        events = detect_preemption_events(curr, prev)
        assert len(events) == 1
        assert "BRAVO-1" in events[0]["description"]
        assert events[0]["deferred_for"] == "BRAVO-1"

    def test_no_preempted_returns_empty(self):
        curr, prev = _two({"action": "TASK"}, {"action": "NONE"})
        assert detect_preemption_events(curr, prev) == []

    def test_missing_action_column_returns_empty(self):
        curr = _df(anomaly_score=1.0)
        prev = _df(anomaly_score=0.5)
        assert detect_preemption_events(curr, prev) == []

    def test_empty_current_returns_empty(self):
        assert detect_preemption_events(pd.DataFrame(), _df(action="NONE")) == []


# ── build_event_feed ──────────────────────────────────────────────────────────

class TestBuildEventFeed:

    def test_empty_current_returns_empty(self):
        assert build_event_feed(pd.DataFrame(), _df(portfolio_rank=1)) == []

    def test_none_prev_returns_empty(self):
        assert build_event_feed(_df(portfolio_rank=1), None) == []

    def test_empty_prev_returns_empty(self):
        assert build_event_feed(_df(portfolio_rank=1), pd.DataFrame()) == []

    def test_returns_list_of_dicts(self):
        curr, prev = _two({"portfolio_rank": 1, "neglect_flag": True,
                            "custody_health": "DEGRADING"},
                           {"portfolio_rank": 8, "neglect_flag": False,
                            "custody_health": "HEALTHY"})
        events = build_event_feed(curr, prev)
        assert isinstance(events, list)
        for e in events:
            assert isinstance(e, dict)
            assert "entity_id" in e
            assert "event_type" in e
            assert "description" in e

    def test_max_events_respected(self):
        """build_event_feed should not exceed max_events."""
        # Create many simultaneous events
        rows_curr, rows_prev = [], []
        for i in range(20):
            eid = f"E{i:02d}"
            rows_curr.append({
                "target_id": eid, "portfolio_rank": 1,
                "neglect_flag": True, "custody_health": "LOST",
                "sensitive_zone": 0.9, "action": "PREEMPTED", "deferred_for": None,
            })
            rows_prev.append({
                "target_id": eid, "portfolio_rank": 15,
                "neglect_flag": False, "custody_health": "HEALTHY",
                "sensitive_zone": 0.0, "action": "NONE",
            })
        events = build_event_feed(pd.DataFrame(rows_curr), pd.DataFrame(rows_prev), max_events=5)
        assert len(events) <= 5

    def test_zone_entry_ranked_before_rank_change(self):
        curr = pd.DataFrame([{
            "target_id": "A", "portfolio_rank": 1,
            "sensitive_zone": 0.9, "neglect_flag": False,
            "custody_health": "HEALTHY", "action": "NONE",
        }, {
            "target_id": "B", "portfolio_rank": 2,
            "sensitive_zone": 0.0, "neglect_flag": False,
            "custody_health": "HEALTHY", "action": "NONE",
        }])
        prev = pd.DataFrame([{
            "target_id": "A", "portfolio_rank": 10,
            "sensitive_zone": 0.0, "neglect_flag": False,
            "custody_health": "HEALTHY", "action": "NONE",
        }, {
            "target_id": "B", "portfolio_rank": 10,
            "sensitive_zone": 0.0, "neglect_flag": False,
            "custody_health": "HEALTHY", "action": "NONE",
        }])
        events = build_event_feed(curr, prev)
        types = [e["event_type"] for e in events]
        if "zone_entry" in types and "rank_change" in types:
            assert types.index("zone_entry") < types.index("rank_change")

    def test_health_worsened_ranked_before_preempted(self):
        curr = pd.DataFrame([{
            "target_id": "A", "custody_health": "STALE", "action": "PREEMPTED",
            "neglect_flag": False, "sensitive_zone": 0.0,
        }])
        prev = pd.DataFrame([{
            "target_id": "A", "custody_health": "HEALTHY", "action": "NONE",
            "neglect_flag": False, "sensitive_zone": 0.0,
        }])
        events = build_event_feed(curr, prev)
        types = [e["event_type"] for e in events]
        if "health_worsened" in types and "preempted" in types:
            assert types.index("health_worsened") < types.index("preempted")

    def test_no_changes_returns_empty(self):
        """Stable state between timesteps → no events."""
        curr = pd.DataFrame([{
            "target_id": "A", "portfolio_rank": 3,
            "custody_health": "HEALTHY", "neglect_flag": False,
            "sensitive_zone": 0.0, "action": "NONE",
        }])
        prev = pd.DataFrame([{
            "target_id": "A", "portfolio_rank": 3,
            "custody_health": "HEALTHY", "neglect_flag": False,
            "sensitive_zone": 0.0, "action": "NONE",
        }])
        assert build_event_feed(curr, prev) == []


# ── detect_zone_approach_events ───────────────────────────────────────────────

class TestDetectZoneApproachEvents:

    def test_approach_detected_when_crosses_threshold(self):
        curr, prev = _two(
            {"zone_probability": 0.7, "time_to_zone_hours": 3.5},
            {"zone_probability": 0.2, "time_to_zone_hours": None},
        )
        events = detect_zone_approach_events(curr, prev)
        assert len(events) == 1
        assert events[0]["event_type"] == "zone_approach"
        assert "A" in events[0]["description"]

    def test_description_includes_tte_when_available(self):
        curr, prev = _two(
            {"zone_probability": 0.8, "time_to_zone_hours": 4.2},
            {"zone_probability": 0.1, "time_to_zone_hours": None},
        )
        events = detect_zone_approach_events(curr, prev)
        assert "4.2h" in events[0]["description"]

    def test_no_event_when_already_approaching(self):
        """Both steps above threshold — not a new transition."""
        curr, prev = _two(
            {"zone_probability": 0.8, "time_to_zone_hours": 2.0},
            {"zone_probability": 0.7, "time_to_zone_hours": 3.0},
        )
        assert detect_zone_approach_events(curr, prev) == []

    def test_no_event_when_prob_low(self):
        curr, prev = _two(
            {"zone_probability": 0.3, "time_to_zone_hours": 5.0},
            {"zone_probability": 0.1, "time_to_zone_hours": None},
        )
        assert detect_zone_approach_events(curr, prev) == []

    def test_no_event_when_column_missing(self):
        curr, prev = _two({"custody_health": "HEALTHY"}, {"custody_health": "HEALTHY"})
        assert detect_zone_approach_events(curr, prev) == []

    def test_event_fields_populated(self):
        curr, prev = _two(
            {"zone_probability": 0.9, "time_to_zone_hours": 1.5},
            {"zone_probability": 0.0, "time_to_zone_hours": None},
        )
        events = detect_zone_approach_events(curr, prev)
        assert events[0]["entity_id"] == "A"
        assert "zone_probability" in events[0]
        assert "time_to_zone_hours" in events[0]

    def test_zone_approach_ranked_before_rank_change_in_feed(self):
        curr = pd.DataFrame([{
            "target_id": "A", "portfolio_rank": 2,
            "custody_health": "HEALTHY", "neglect_flag": False,
            "sensitive_zone": 0.0, "action": "NONE",
            "zone_probability": 0.8, "time_to_zone_hours": 2.0,
        }])
        prev = pd.DataFrame([{
            "target_id": "A", "portfolio_rank": 9,
            "custody_health": "HEALTHY", "neglect_flag": False,
            "sensitive_zone": 0.0, "action": "NONE",
            "zone_probability": 0.1, "time_to_zone_hours": None,
        }])
        events = build_event_feed(curr, prev)
        types = [e["event_type"] for e in events]
        if "zone_approach" in types and "rank_change" in types:
            assert types.index("zone_approach") < types.index("rank_change")
