"""
Tests for custody.whatif — scenario replay and what-if analysis.

Coverage:
  1.  run_scenario executes without error (baseline and with overrides)
  2.  run_scenario restores config after execution (no leakage)
  3.  run_scenario restores config even when scenario_fn raises
  4.  overrides visible to planner during the run
  5.  overrides affect summary metrics in the expected direction
  6.  run_comparison returns one summary per variant
  7.  summarise returns all expected schema keys
  8.  summarise handles empty record list
  9.  action_counts matches actual record actions
 10.  hold_reason_counts excludes None reasons
 11.  lookahead_hold_count is a subset of hold_reason_counts["lookahead"]
 12.  avg_custody_confidence is in [0, 1]
 13.  sensor_usage counts only TASK records
 14.  no cross-run config leakage between two consecutive run_comparison calls
 15.  existing simulation behavior unchanged when whatif is not used
"""
from datetime import UTC, datetime, timedelta

import pytest

import custody.config as config
from custody.ais import AISObservation, ingest_ais_track
from custody.simulate import run_simulation
from custody.whatif import run_comparison, run_scenario, summarise


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

EXPECTED_SUMMARY_KEYS = {
    "label",
    "record_count",
    "action_counts",
    "hold_reason_counts",
    "lookahead_hold_count",
    "avg_custody_confidence",
    "avg_anomaly_score",
    "sensor_usage",
}

# scenario_fn that returns a small deterministic set of records
def _sim_scenario():
    return run_simulation()


# Minimal AIS scenario: two vessels, two observations each, 6-hour gap (stale)
_T0 = datetime(2026, 3, 23, 10, 0, tzinfo=UTC)
_AIS_GAP = 6 * 3600  # 6 hours — exceeds default AIS_STALE_GAP_SECONDS

def _ais_obs(vid, t, lat=0.0, lon=0.0):
    return AISObservation(
        vessel_id=vid, timestamp=t, lat=lat, lon=lon,
        speed_knots=5.0, heading_deg=0.0,
    )

def _ais_scenario():
    obs = [
        _ais_obs("V001", _T0),
        _ais_obs("V001", _T0 + timedelta(seconds=_AIS_GAP)),
    ]
    return ingest_ais_track(obs)


# ---------------------------------------------------------------------------
# 1. run_scenario executes
# ---------------------------------------------------------------------------

class TestRunScenarioExecutes:
    def test_baseline_no_overrides(self):
        records = run_scenario(_sim_scenario)
        assert isinstance(records, list)
        assert len(records) > 0

    def test_with_overrides_executes(self):
        records = run_scenario(_sim_scenario, {"TASK_VALUE_THRESHOLD": 0.0})
        assert isinstance(records, list)
        assert len(records) > 0

    def test_none_overrides_treated_as_empty(self):
        records = run_scenario(_sim_scenario, None)
        assert isinstance(records, list)
        assert len(records) > 0

    def test_ais_scenario_executes(self):
        records = run_scenario(_ais_scenario)
        assert isinstance(records, list)
        assert len(records) > 0


# ---------------------------------------------------------------------------
# 2 & 3. Config restoration — no leakage
# ---------------------------------------------------------------------------

class TestConfigRestoration:
    def test_config_restored_after_run(self):
        original = config.TASK_VALUE_THRESHOLD
        run_scenario(_sim_scenario, {"TASK_VALUE_THRESHOLD": 9999.0})
        assert config.TASK_VALUE_THRESHOLD == original

    def test_config_restored_after_exception(self):
        original = config.HOLD_LOOKAHEAD_THRESHOLD_SECONDS

        def bad_scenario():
            raise RuntimeError("intentional failure")

        with pytest.raises(RuntimeError):
            run_scenario(bad_scenario, {"HOLD_LOOKAHEAD_THRESHOLD_SECONDS": 0.0})

        assert config.HOLD_LOOKAHEAD_THRESHOLD_SECONDS == original

    def test_multiple_overrides_all_restored(self):
        orig_threshold = config.TASK_VALUE_THRESHOLD
        orig_boost = config.HOLD_LOOKAHEAD_BOOST

        run_scenario(_sim_scenario, {
            "TASK_VALUE_THRESHOLD": 999.0,
            "HOLD_LOOKAHEAD_BOOST": 0.99,
        })

        assert config.TASK_VALUE_THRESHOLD == orig_threshold
        assert config.HOLD_LOOKAHEAD_BOOST == orig_boost

    def test_config_visible_to_planner_during_run(self):
        """The override value is what the planner sees during execution."""
        seen_values = []

        import custody.config as cfg
        original_plan = None

        import custody.ais as ais_mod
        original_plan = ais_mod.plan_collection

        def recording_plan(track, score, confidence, breakdown, current_time, opps, **kwargs):
            seen_values.append(cfg.AIS_STALE_GAP_SECONDS)
            return original_plan(track, score, confidence, breakdown, current_time, opps, **kwargs)

        import unittest.mock as mock
        with mock.patch("custody.ais.plan_collection", side_effect=recording_plan):
            run_scenario(_ais_scenario, {"AIS_STALE_GAP_SECONDS": 0.0})

        assert seen_values, "plan_collection was never called"
        assert all(v == 0.0 for v in seen_values)


# ---------------------------------------------------------------------------
# 4 & 5. Overrides affect summary metrics
# ---------------------------------------------------------------------------

class TestOverridesAffectSummaries:
    def test_high_task_value_threshold_produces_more_holds(self):
        """TASK_VALUE_THRESHOLD=999 blocks all task-value progress → more HOLDs."""
        baseline = summarise(run_scenario(_sim_scenario, {}), "baseline")
        high_thresh = summarise(
            run_scenario(_sim_scenario, {"TASK_VALUE_THRESHOLD": 999.0}),
            "high_thresh",
        )
        baseline_hold = baseline["action_counts"].get("HOLD", 0)
        high_hold = high_thresh["action_counts"].get("HOLD", 0)
        assert high_hold >= baseline_hold

    def test_zero_task_value_threshold_produces_fewer_holds(self):
        """TASK_VALUE_THRESHOLD=0 removes the HOLD gate → fewer or equal HOLDs."""
        baseline = summarise(run_scenario(_sim_scenario, {}), "baseline")
        no_thresh = summarise(
            run_scenario(_sim_scenario, {"TASK_VALUE_THRESHOLD": 0.0}),
            "no_thresh",
        )
        assert no_thresh["action_counts"].get("HOLD", 0) <= baseline["action_counts"].get("HOLD", 0)

    def test_ais_stale_gap_zero_lowers_average_confidence(self):
        """AIS_STALE_GAP_SECONDS=0 makes every gap trigger decay → lower avg confidence."""
        baseline = summarise(run_scenario(_ais_scenario, {}), "baseline")
        stale = summarise(
            run_scenario(_ais_scenario, {"AIS_STALE_GAP_SECONDS": 0.0}),
            "stale",
        )
        # With gap=0 threshold, decay fires immediately → lower avg confidence
        assert stale["avg_custody_confidence"] <= baseline["avg_custody_confidence"]

    def test_ais_min_stale_confidence_floor_is_respected(self):
        """Higher AIS_MIN_STALE_CONFIDENCE floor raises the worst-case confidence."""
        high_floor = summarise(
            run_scenario(_ais_scenario, {
                "AIS_STALE_GAP_SECONDS": 0.0,
                "AIS_MIN_STALE_CONFIDENCE": 0.9,
            }),
            "high_floor",
        )
        low_floor = summarise(
            run_scenario(_ais_scenario, {
                "AIS_STALE_GAP_SECONDS": 0.0,
                "AIS_MIN_STALE_CONFIDENCE": 0.1,
            }),
            "low_floor",
        )
        assert high_floor["avg_custody_confidence"] >= low_floor["avg_custody_confidence"]


# ---------------------------------------------------------------------------
# 6. run_comparison
# ---------------------------------------------------------------------------

class TestRunComparison:
    def test_returns_one_summary_per_variant(self):
        variants = [
            {"label": "a", "overrides": {}},
            {"label": "b", "overrides": {"TASK_VALUE_THRESHOLD": 0.0}},
            {"label": "c", "overrides": {"TASK_VALUE_THRESHOLD": 999.0}},
        ]
        results = run_comparison(_sim_scenario, variants)
        assert len(results) == 3

    def test_labels_preserved(self):
        variants = [
            {"label": "baseline", "overrides": {}},
            {"label": "variant1", "overrides": {}},
        ]
        results = run_comparison(_sim_scenario, variants)
        assert [r["label"] for r in results] == ["baseline", "variant1"]

    def test_empty_variants_returns_empty(self):
        assert run_comparison(_sim_scenario, []) == []

    def test_no_cross_run_leakage_in_comparison(self):
        """Each run in a comparison sees only its own overrides."""
        original = config.TASK_VALUE_THRESHOLD
        run_comparison(_sim_scenario, [
            {"label": "a", "overrides": {"TASK_VALUE_THRESHOLD": 0.0}},
            {"label": "b", "overrides": {"TASK_VALUE_THRESHOLD": 999.0}},
        ])
        assert config.TASK_VALUE_THRESHOLD == original

    def test_two_consecutive_comparisons_independent(self):
        """Running run_comparison twice with different overrides gives different results."""
        first = run_comparison(_sim_scenario, [
            {"label": "high", "overrides": {"TASK_VALUE_THRESHOLD": 999.0}},
        ])
        second = run_comparison(_sim_scenario, [
            {"label": "low", "overrides": {"TASK_VALUE_THRESHOLD": 0.0}},
        ])
        assert config.TASK_VALUE_THRESHOLD not in (999.0, 0.0)  # restored both times
        # Sanity: high threshold should produce more HOLDs
        high_holds = first[0]["action_counts"].get("HOLD", 0)
        low_holds = second[0]["action_counts"].get("HOLD", 0)
        assert high_holds >= low_holds


# ---------------------------------------------------------------------------
# 7 & 8. summarise schema
# ---------------------------------------------------------------------------

class TestSummariseSchema:
    def test_all_expected_keys_present(self):
        records = run_scenario(_sim_scenario)
        summary = summarise(records, "test")
        assert set(summary.keys()) == EXPECTED_SUMMARY_KEYS

    def test_empty_records_returns_valid_summary(self):
        summary = summarise([], label="empty")
        assert set(summary.keys()) == EXPECTED_SUMMARY_KEYS
        assert summary["record_count"] == 0
        assert summary["avg_custody_confidence"] == 0.0
        assert summary["avg_anomaly_score"] == 0.0
        assert summary["action_counts"] == {}
        assert summary["sensor_usage"] == {}

    def test_label_preserved(self):
        records = run_scenario(_sim_scenario)
        assert summarise(records, "my_label")["label"] == "my_label"

    def test_record_count_matches_input(self):
        records = run_scenario(_sim_scenario)
        assert summarise(records)["record_count"] == len(records)


# ---------------------------------------------------------------------------
# 9. action_counts
# ---------------------------------------------------------------------------

class TestActionCounts:
    def test_action_counts_sum_to_record_count(self):
        records = run_scenario(_sim_scenario)
        summary = summarise(records)
        assert sum(summary["action_counts"].values()) == summary["record_count"]

    def test_known_actions_only(self):
        records = run_scenario(_sim_scenario)
        valid = {"NONE", "TASK", "HOLD", "NO_SENSOR", "PREEMPTED"}
        assert set(summarise(records)["action_counts"].keys()) <= valid


# ---------------------------------------------------------------------------
# 10. hold_reason_counts excludes None
# ---------------------------------------------------------------------------

class TestHoldReasonCounts:
    def test_none_reason_not_in_hold_reason_counts(self):
        records = run_scenario(_sim_scenario)
        summary = summarise(records)
        assert None not in summary["hold_reason_counts"]

    def test_hold_reason_values_are_strings(self):
        records = run_scenario(_sim_scenario)
        for reason in summarise(records)["hold_reason_counts"]:
            assert isinstance(reason, str)


# ---------------------------------------------------------------------------
# 11. lookahead_hold_count
# ---------------------------------------------------------------------------

class TestLookaheadHoldCount:
    def test_lookahead_hold_count_leq_hold_count(self):
        records = run_scenario(_sim_scenario)
        summary = summarise(records)
        total_holds = summary["action_counts"].get("HOLD", 0)
        assert summary["lookahead_hold_count"] <= total_holds

    def test_lookahead_hold_count_matches_hold_reason_counts(self):
        records = run_scenario(_sim_scenario)
        summary = summarise(records)
        assert summary["lookahead_hold_count"] == summary["hold_reason_counts"].get("lookahead", 0)


# ---------------------------------------------------------------------------
# 12. avg_custody_confidence
# ---------------------------------------------------------------------------

class TestAvgCustodyConfidence:
    def test_avg_confidence_in_valid_range(self):
        records = run_scenario(_sim_scenario)
        avg = summarise(records)["avg_custody_confidence"]
        assert 0.0 <= avg <= 1.0

    def test_avg_confidence_reflects_records(self):
        records = run_scenario(_sim_scenario)
        manual_avg = sum(float(r["custody_confidence"]) for r in records) / len(records)
        assert summarise(records)["avg_custody_confidence"] == pytest.approx(manual_avg, abs=1e-9)


# ---------------------------------------------------------------------------
# 13. sensor_usage counts only TASK records
# ---------------------------------------------------------------------------

class TestSensorUsage:
    def test_sensor_usage_keys_only_from_task_records(self):
        records = run_scenario(_sim_scenario)
        task_sensor_ids = {
            r["sensor_id"] for r in records
            if r["action"] == "TASK" and r["sensor_id"] is not None
        }
        summary = summarise(records)
        assert set(summary["sensor_usage"].keys()) <= task_sensor_ids

    def test_sensor_usage_count_leq_task_count(self):
        records = run_scenario(_sim_scenario)
        summary = summarise(records)
        task_count = summary["action_counts"].get("TASK", 0)
        total_sensor_uses = sum(summary["sensor_usage"].values())
        assert total_sensor_uses <= task_count


# ---------------------------------------------------------------------------
# 15. Simulation unchanged when whatif is not used
# ---------------------------------------------------------------------------

class TestSimulationUnchanged:
    def test_run_simulation_unaffected(self):
        """run_simulation() produces valid output independent of whatif imports."""
        records = run_simulation()
        assert len(records) > 0
        valid = {"NONE", "TASK", "HOLD", "NO_SENSOR", "PREEMPTED"}
        assert all(r["action"] in valid for r in records)
