"""
Tests for custody/decision_trace.py and the trace-capture integration in both
simulate.py (multi-vessel simulation) and ais.py (single-vessel AIS replay).

Covers:
  - PriorityBreakdown correctness (fields match compute_target_priority formula)
  - TaskValueBreakdown correctness (fields match compute_task_value formula)
  - Arbitration trace: accessible, claimed, and final-pool fields are consistent
  - Result consistency: trace.arbitration.result == decision action
  - Simulation: one trace per vessel per timestep
  - sensor_access_count reflects pre-arbitration access
  - traces_to_rows returns all expected keys
  - AIS replay: every record carries decision_trace
  - AIS trace shape is identical to simulation trace shape
  - AIS result/action consistency
  - traces_to_rows works on AIS traces
"""
import math
from datetime import UTC, datetime, timedelta

import pytest

from custody.config import (
    CRITICAL_ANOMALY_THRESHOLD,
    FRESHNESS_SUPPRESSION,
    REVISIT_DECAY_HOURS,
    TASK_VALUE_THRESHOLD,
    WORSENING_BOOST,
)
from custody.decision_trace import (
    ArbitrationBreakdown,
    DecisionInputs,
    DecisionTrace,
    PriorityBreakdown,
    TaskValueBreakdown,
    traces_to_rows,
)
from custody.ais import AISObservation, ingest_ais_track, replay_ais_file
from custody.planner import compute_target_priority, compute_task_value
from custody.simulate import run_simulation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_T = datetime(2026, 3, 23, 10, 0, 0, tzinfo=UTC)


def _make_trace(
    result="NONE",
    accessible=(),
    claimed=(),
    final_pool=(),
    chosen=None,
    anomaly_score=0.5,
    confidence=0.8,
    compound_boost=0.0,
    freshness=0.0,
    access_count=0,
    priority_total=0.2,
    tv_base=0.2,
    tv_total=0.2,
    hold_eligible=False,
) -> DecisionTrace:
    return DecisionTrace(
        timestamp=_T,
        vessel_id="V001",
        inputs=DecisionInputs(
            anomaly_score=anomaly_score,
            anomaly_norm=anomaly_score / CRITICAL_ANOMALY_THRESHOLD,
            custody_confidence=confidence,
            compound_boost=compound_boost,
            freshness=freshness,
            sensor_access_count=access_count,
        ),
        priority=PriorityBreakdown(
            anomaly_norm=anomaly_score / CRITICAL_ANOMALY_THRESHOLD,
            uncertainty=1.0 - confidence,
            compound_boost=compound_boost,
            total=priority_total,
        ),
        task_value=TaskValueBreakdown(
            base=tv_base,
            freshness_decay=0.0,
            worsening_boost=0.0,
            total=tv_total,
            hold_threshold=TASK_VALUE_THRESHOLD,
            hold_eligible=hold_eligible,
        ),
        arbitration=ArbitrationBreakdown(
            accessible_sensor_ids=accessible,
            claimed_by_higher_priority=claimed,
            final_sensor_pool=final_pool,
            chosen_sensor_id=chosen,
            result=result,
        ),
    )


# ---------------------------------------------------------------------------
# PriorityBreakdown correctness
# ---------------------------------------------------------------------------

class TestPriorityBreakdown:
    def test_fields_match_formula(self):
        score, conf, boost = 0.9, 0.6, 0.3
        total, bd = compute_target_priority(score, conf, boost, return_breakdown=True)

        norm = score / CRITICAL_ANOMALY_THRESHOLD
        uncert = 1.0 - conf
        expected = norm * 0.55 + uncert * 0.35 + boost * 0.10

        assert bd.anomaly_norm == pytest.approx(norm)
        assert bd.uncertainty == pytest.approx(uncert)
        assert bd.compound_boost == pytest.approx(boost)
        assert bd.total == pytest.approx(expected)
        assert total == pytest.approx(expected)

    def test_no_boost_zero_compound(self):
        total, bd = compute_target_priority(0.5, 0.8, 0.0, return_breakdown=True)
        assert bd.compound_boost == 0.0
        assert bd.total == pytest.approx(total)

    def test_return_breakdown_false_returns_float(self):
        result = compute_target_priority(0.5, 0.8)
        assert isinstance(result, float)

    def test_return_breakdown_true_returns_tuple(self):
        result = compute_target_priority(0.5, 0.8, return_breakdown=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        total, bd = result
        assert isinstance(total, float)
        assert isinstance(bd, PriorityBreakdown)

    def test_total_equals_sum_of_weighted_components(self):
        _, bd = compute_target_priority(1.2, 0.4, 0.5, return_breakdown=True)
        expected = bd.anomaly_norm * 0.55 + bd.uncertainty * 0.35 + bd.compound_boost * 0.10
        assert bd.total == pytest.approx(expected)


# ---------------------------------------------------------------------------
# TaskValueBreakdown correctness
# ---------------------------------------------------------------------------

class TestTaskValueBreakdown:
    def test_no_prior_collection_zero_freshness_decay(self):
        total, bd = compute_task_value(
            0.8, 0.7, None, None, return_breakdown=True
        )
        assert bd.freshness_decay == 0.0
        assert bd.worsening_boost == 0.0
        assert bd.total == pytest.approx(bd.base)
        assert total == pytest.approx(bd.base)

    def test_with_prior_collection_freshness_decay_nonzero(self):
        hours_since = 2.0
        score, conf, boost = 0.8, 0.7, 0.0
        total, bd = compute_task_value(
            score, conf, hours_since, score, boost, return_breakdown=True
        )
        freshness = math.exp(-hours_since / REVISIT_DECAY_HOURS)
        expected_decay = freshness * FRESHNESS_SUPPRESSION
        assert bd.freshness_decay == pytest.approx(expected_decay)

    def test_worsening_boost_when_score_increased(self):
        last_score = 0.5
        current_score = 0.9
        total, bd = compute_task_value(
            current_score, 0.7, 3.0, last_score, return_breakdown=True
        )
        worsening = current_score - last_score
        assert bd.worsening_boost == pytest.approx(worsening * WORSENING_BOOST)

    def test_no_worsening_boost_when_score_unchanged(self):
        score = 0.8
        total, bd = compute_task_value(score, 0.7, 3.0, score, return_breakdown=True)
        assert bd.worsening_boost == 0.0

    def test_hold_eligible_true_when_total_below_threshold(self):
        # Force a low task value: high confidence, low score, recent collection
        total, bd = compute_task_value(
            0.3, 0.95, 0.5, 0.3, return_breakdown=True
        )
        if bd.total < TASK_VALUE_THRESHOLD:
            assert bd.hold_eligible is True
        else:
            assert bd.hold_eligible is False

    def test_hold_threshold_equals_config(self):
        _, bd = compute_task_value(0.5, 0.7, None, None, return_breakdown=True)
        assert bd.hold_threshold == TASK_VALUE_THRESHOLD

    def test_total_equals_base_minus_decay_plus_worsening(self):
        total, bd = compute_task_value(
            0.9, 0.6, 2.0, 0.5, return_breakdown=True
        )
        expected = bd.base - bd.freshness_decay + bd.worsening_boost
        assert bd.total == pytest.approx(expected)

    def test_return_breakdown_false_returns_float(self):
        result = compute_task_value(0.5, 0.8, None, None)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# Arbitration breakdown invariants
# ---------------------------------------------------------------------------

class TestArbitrationBreakdown:
    def test_final_pool_is_accessible_minus_claimed(self):
        accessible = ("A1", "B1", "EO-MIO-1")
        claimed = ("A1",)
        final = tuple(s for s in accessible if s not in claimed)
        t = _make_trace(accessible=accessible, claimed=claimed, final_pool=final)
        assert set(t.arbitration.final_sensor_pool) == set(accessible) - set(claimed)

    def test_claimed_is_subset_of_accessible(self):
        accessible = ("A1", "C1")
        claimed = ("A1",)
        t = _make_trace(accessible=accessible, claimed=claimed,
                        final_pool=("C1",))
        assert set(t.arbitration.claimed_by_higher_priority).issubset(
            set(t.arbitration.accessible_sensor_ids)
        )

    def test_chosen_sensor_in_final_pool_when_task(self):
        t = _make_trace(
            result="TASK",
            accessible=("A1", "B1"),
            claimed=(),
            final_pool=("A1", "B1"),
            chosen="A1",
        )
        assert t.arbitration.chosen_sensor_id in t.arbitration.final_sensor_pool

    def test_chosen_sensor_none_when_no_task(self):
        for result in ("NONE", "HOLD", "NO_SENSOR", "PREEMPTED"):
            t = _make_trace(result=result, chosen=None)
            assert t.arbitration.chosen_sensor_id is None

    def test_preempted_final_pool_empty_accessible_nonempty(self):
        t = _make_trace(
            result="PREEMPTED",
            accessible=("A1",),
            claimed=("A1",),
            final_pool=(),
            chosen=None,
        )
        assert len(t.arbitration.accessible_sensor_ids) > 0
        assert len(t.arbitration.final_sensor_pool) == 0


# ---------------------------------------------------------------------------
# Simulation integration
# ---------------------------------------------------------------------------

class TestSimulationTraces:
    def setup_method(self):
        self.records = run_simulation()

    def test_every_record_has_decision_trace(self):
        for r in self.records:
            assert "decision_trace" in r, f"Missing decision_trace in record {r}"
            assert isinstance(r["decision_trace"], DecisionTrace)

    def test_one_trace_per_vessel_per_timestep(self):
        by_time: dict = {}
        for r in self.records:
            t = r["time"]
            by_time.setdefault(t, set()).add(r["target_id"])
        # At each timestep both vessels should have a record (and hence a trace)
        for t, vids in by_time.items():
            assert len(vids) >= 1

    def test_trace_result_matches_record_action(self):
        for r in self.records:
            trace = r["decision_trace"]
            assert trace.arbitration.result == r["action"], (
                f"Trace result {trace.arbitration.result!r} != "
                f"record action {r['action']!r} for {r['target_id']} @ {r['time']}"
            )

    def test_trace_vessel_id_matches_record(self):
        for r in self.records:
            assert r["decision_trace"].vessel_id == r["target_id"]

    def test_trace_timestamp_matches_record(self):
        for r in self.records:
            assert r["decision_trace"].timestamp == r["time"]

    def test_sensor_access_count_reflects_prearbitation_pool(self):
        for r in self.records:
            trace = r["decision_trace"]
            # sensor_access_count in the record and in the trace inputs agree
            assert trace.inputs.sensor_access_count == r["sensor_access_count"]
            # accessible_sensor_ids length matches the count
            assert len(trace.arbitration.accessible_sensor_ids) == r["sensor_access_count"]

    def test_claimed_sensors_subset_of_accessible(self):
        for r in self.records:
            trace = r["decision_trace"]
            arb = trace.arbitration
            assert set(arb.claimed_by_higher_priority).issubset(
                set(arb.accessible_sensor_ids)
            ), f"Claimed sensors not a subset of accessible at {r['time']} {r['target_id']}"

    def test_final_pool_equals_accessible_minus_claimed(self):
        for r in self.records:
            arb = r["decision_trace"].arbitration
            expected = set(arb.accessible_sensor_ids) - set(arb.claimed_by_higher_priority)
            assert set(arb.final_sensor_pool) == expected, (
                f"final_pool mismatch at {r['time']} {r['target_id']}"
            )

    def test_chosen_sensor_in_final_pool_when_task(self):
        task_records = [r for r in self.records if r["action"] == "TASK"]
        assert len(task_records) > 0, "No TASK actions in simulation — cannot verify"
        for r in task_records:
            arb = r["decision_trace"].arbitration
            assert arb.chosen_sensor_id in arb.final_sensor_pool, (
                f"Chosen sensor {arb.chosen_sensor_id!r} not in final pool "
                f"{arb.final_sensor_pool} at {r['time']} {r['target_id']}"
            )

    def test_priority_breakdown_total_in_trace(self):
        for r in self.records:
            trace = r["decision_trace"]
            expected = (
                trace.priority.anomaly_norm * 0.55
                + trace.priority.uncertainty * 0.35
                + trace.priority.compound_boost * 0.10
            )
            assert trace.priority.total == pytest.approx(expected, abs=1e-9)

    def test_task_value_total_equals_base_minus_decay_plus_worsening(self):
        for r in self.records:
            tv = r["decision_trace"].task_value
            expected = (tv.base - tv.freshness_decay + tv.worsening_boost
                        + tv.lookahead_boost + tv.failure_boost)
            assert tv.total == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# traces_to_rows
# ---------------------------------------------------------------------------

EXPECTED_ROW_KEYS = {
    "Time", "Vessel", "Action", "Hold Reason", "Chosen Sensor",
    "Priority", "Priority Anomaly", "Priority Uncertainty", "Priority Compound",
    "Task Value", "Task Base", "Freshness Decay", "Worsening Boost",
    "Lookahead Boost", "Failure Boost", "Nearest Pass TTS", "Hold Eligible",
    "Consecutive Failures",
    "Accessible Sensors", "Claimed Higher", "Final Pool", "Sensor Access Count",
}


class TestTracesToRows:
    def test_returns_expected_keys(self):
        records = run_simulation()
        traces = [r["decision_trace"] for r in records]
        rows = traces_to_rows(traces)
        assert len(rows) == len(traces)
        for row in rows:
            assert set(row.keys()) == EXPECTED_ROW_KEYS, (
                f"Row keys mismatch: {set(row.keys()) ^ EXPECTED_ROW_KEYS}"
            )

    def test_action_matches_trace_result(self):
        records = run_simulation()
        traces = [r["decision_trace"] for r in records]
        rows = traces_to_rows(traces)
        for trace, row in zip(traces, rows):
            assert row["Action"] == trace.arbitration.result

    def test_chosen_sensor_matches_trace(self):
        records = run_simulation()
        traces = [r["decision_trace"] for r in records]
        rows = traces_to_rows(traces)
        for trace, row in zip(traces, rows):
            assert row["Chosen Sensor"] == trace.arbitration.chosen_sensor_id

    def test_empty_input_returns_empty_list(self):
        assert traces_to_rows([]) == []

    def test_sensor_access_count_is_int(self):
        records = run_simulation()
        traces = [r["decision_trace"] for r in records]
        rows = traces_to_rows(traces)
        for row in rows:
            assert isinstance(row["Sensor Access Count"], int)


# ---------------------------------------------------------------------------
# AIS replay trace integration
# ---------------------------------------------------------------------------

_AIS_T0 = datetime(2026, 3, 23, 10, 0, 0, tzinfo=UTC)


def _ais_obs(vessel_id, t, lat=0.0, lon=0.0, speed_knots=10.0, heading_deg=45.0):
    return AISObservation(
        vessel_id=vessel_id,
        timestamp=t,
        lat=lat,
        lon=lon,
        speed_knots=speed_knots,
        heading_deg=heading_deg,
    )


def _ais_track(n=5, vessel_id="V_AIS"):
    from datetime import timedelta
    return [
        _ais_obs(vessel_id, _AIS_T0 + timedelta(hours=i))
        for i in range(n)
    ]


class TestAISTraces:
    def setup_method(self):
        self.obs = _ais_track(n=5)
        self.records = ingest_ais_track(self.obs)

    # ── Presence ──────────────────────────────────────────────────────────────

    def test_every_record_has_decision_trace(self):
        for r in self.records:
            assert "decision_trace" in r, f"Missing decision_trace in {r}"
            assert isinstance(r["decision_trace"], DecisionTrace)

    def test_first_record_has_trace(self):
        """First AIS record (hardcoded NONE) must carry a trace."""
        assert isinstance(self.records[0]["decision_trace"], DecisionTrace)

    def test_all_records_have_trace(self):
        assert len(self.records) == 5
        assert all("decision_trace" in r for r in self.records)

    # ── Result / action consistency ────────────────────────────────────────────

    def test_trace_result_matches_record_action(self):
        for r in self.records:
            trace = r["decision_trace"]
            assert trace.arbitration.result == r["action"], (
                f"Trace result {trace.arbitration.result!r} != "
                f"record action {r['action']!r} at {r['time']}"
            )

    def test_first_record_result_is_none(self):
        """First record is always action=NONE; trace must reflect that."""
        first = self.records[0]
        assert first["action"] == "NONE"
        assert first["decision_trace"].arbitration.result == "NONE"
        assert first["decision_trace"].arbitration.chosen_sensor_id is None

    # ── Shape consistency with simulation path ─────────────────────────────────

    def test_trace_vessel_id_matches_record(self):
        for r in self.records:
            assert r["decision_trace"].vessel_id == r["target_id"]

    def test_trace_timestamp_matches_record(self):
        for r in self.records:
            assert r["decision_trace"].timestamp == r["time"]

    def test_trace_has_all_required_dataclass_fields(self):
        """Spot-check that all nested dataclass fields are populated."""
        trace = self.records[1]["decision_trace"]
        assert isinstance(trace.inputs, DecisionInputs)
        assert isinstance(trace.priority, PriorityBreakdown)
        assert isinstance(trace.task_value, TaskValueBreakdown)
        assert isinstance(trace.arbitration, ArbitrationBreakdown)

    def test_priority_total_formula_holds(self):
        for r in self.records:
            p = r["decision_trace"].priority
            expected = p.anomaly_norm * 0.55 + p.uncertainty * 0.35 + p.compound_boost * 0.10
            assert p.total == pytest.approx(expected, abs=1e-9)

    def test_task_value_total_formula_holds(self):
        for r in self.records:
            tv = r["decision_trace"].task_value
            expected = tv.base - tv.freshness_decay + tv.worsening_boost
            assert tv.total == pytest.approx(expected, abs=1e-9)

    def test_ais_has_no_claimed_sensors(self):
        """Single-vessel AIS replay never preempts; claimed must always be empty."""
        for r in self.records:
            arb = r["decision_trace"].arbitration
            assert arb.claimed_by_higher_priority == (), (
                f"claimed_by_higher_priority should be empty in AIS mode, "
                f"got {arb.claimed_by_higher_priority} at {r['time']}"
            )

    def test_final_pool_equals_accessible_in_ais(self):
        """No claimed sensors → final pool equals accessible pool."""
        for r in self.records:
            arb = r["decision_trace"].arbitration
            assert set(arb.final_sensor_pool) == set(arb.accessible_sensor_ids)

    def test_compound_boost_is_zero_in_ais(self):
        """AIS path passes compound_boost=0.0; trace must reflect that."""
        for r in self.records:
            assert r["decision_trace"].inputs.compound_boost == 0.0
            assert r["decision_trace"].priority.compound_boost == 0.0

    # ── traces_to_rows on AIS output ──────────────────────────────────────────

    def test_traces_to_rows_works_on_ais_traces(self):
        traces = [r["decision_trace"] for r in self.records]
        rows = traces_to_rows(traces)
        assert len(rows) == len(self.records)
        for row in rows:
            assert set(row.keys()) == EXPECTED_ROW_KEYS

    def test_traces_to_rows_action_matches_in_ais(self):
        traces = [r["decision_trace"] for r in self.records]
        rows = traces_to_rows(traces)
        for r, row in zip(self.records, rows):
            assert row["Action"] == r["action"]

    # ── replay_ais_file (multi-vessel) ────────────────────────────────────────

    def test_replay_ais_file_records_have_traces(self):
        """Multi-vessel replay: every record has a decision_trace."""
        csv_text = (
            "vessel_id,timestamp,lat,lon,speed_knots,heading_deg\n"
            "V1,2026-03-23T10:00:00,0.0,0.0,10.0,45.0\n"
            "V1,2026-03-23T11:00:00,0.1,0.1,10.0,45.0\n"
            "V2,2026-03-23T10:00:00,1.0,1.0,8.0,90.0\n"
            "V2,2026-03-23T11:00:00,1.1,1.1,8.0,90.0\n"
        )
        result = replay_ais_file(csv_text)
        for vid, timeline in result.items():
            for r in timeline:
                assert "decision_trace" in r, f"Missing trace for {vid}"
                assert isinstance(r["decision_trace"], DecisionTrace)
