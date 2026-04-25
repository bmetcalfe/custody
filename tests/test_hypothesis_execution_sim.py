"""Tests for :mod:`custody.hypotheses.execution_sim` (ADR-0021 Slice 23)."""
from __future__ import annotations

import ast
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from custody.hypotheses.artifacts import (
    build_decision_from_artifacts,
    load_artifact_manifest,
)
from custody.hypotheses.collection_value import CollectionRecommendation
from custody.hypotheses.custody_health import CustodyHealthStatus
from custody.hypotheses.execution_sim import (
    ExecutionOutcomePolicy,
    ExecutionSimulationReport,
    SimulatedCollectResult,
    execution_report_to_dict,
    execution_report_to_json,
    format_execution_markdown,
    format_execution_text,
    simulate_collect_result,
    simulate_plan_execution,
)
from custody.hypotheses.scheduler import (
    ScheduleConstraint,
    ScheduledCollect,
    SchedulePlan,
    load_collection_windows,
    schedule_collects,
)
from custody.hypotheses.availability_optimizer import (
    AvailabilityOptimizerConstraint,
    optimize_availability_adjusted_plan,
)
from custody.hypotheses.counterfactual import simulate_counterfactual_collects
from custody.hypotheses.mission_value import attribute_mission_value
from custody.hypotheses.optimizer import optimize_collection_plan
from custody.hypotheses.scene_availability import (
    adjust_recommendation_for_availability,
    load_scene_availability_catalog,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ART_TENNENT = (
    REPO_ROOT / "tests" / "fixtures" / "artifacts"
    / "tennent_artifact_manifest.json"
)
CAT_TENNENT = (
    REPO_ROOT / "tests" / "fixtures" / "availability"
    / "tennent_scene_availability.json"
)
WIN_TENNENT = (
    REPO_ROOT / "tests" / "fixtures" / "schedule"
    / "tennent_collection_windows.json"
)


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------


def _build_state_health_plan(scenario: str = "tennent") -> tuple:
    """Return (state, health, plan) for the named scenario."""
    art = REPO_ROOT / "tests" / "fixtures" / "artifacts" / f"{scenario}_artifact_manifest.json"
    cat = REPO_ROOT / "tests" / "fixtures" / "availability" / f"{scenario}_scene_availability.json"
    win = REPO_ROOT / "tests" / "fixtures" / "schedule" / f"{scenario}_collection_windows.json"
    records = load_artifact_manifest(art)
    decision = build_decision_from_artifacts(records, scenario_id=scenario)
    mission_value = attribute_mission_value(decision.recommendation, decision.health)
    counterfactual = simulate_counterfactual_collects(
        decision.final_state, decision.health, decision.recommendation,
    )
    base_opt = optimize_collection_plan(
        decision.recommendation, mission_value, counterfactual,
        scenario_id=scenario,
    )
    catalog = load_scene_availability_catalog(cat, scenario_id=scenario)
    avail = adjust_recommendation_for_availability(decision.recommendation, catalog)
    avail_report = optimize_availability_adjusted_plan(
        base_opt, avail,
        scenario_id=scenario,
        constraint=AvailabilityOptimizerConstraint(budget=1.0, max_collects=2),
    )
    windows = load_collection_windows(win)
    schedule_report = schedule_collects(
        avail_report, windows, scenario_id=scenario,
    )
    return decision.final_state, decision.health, schedule_report.plan


def _make_scheduled(
    candidate_id: str = "optical_context",
    window_id: str = "test-w1",
    *,
    end_time: datetime = datetime(2023, 8, 14, 15, 0, tzinfo=timezone.utc),
    schedule_score: float = 0.65,
) -> ScheduledCollect:
    return ScheduledCollect(
        candidate_id=candidate_id,
        label=candidate_id.replace("_", " "),
        window_id=window_id,
        start_time=end_time - timedelta(hours=1),
        end_time=end_time,
        capacity_cost=0.3,
        schedule_score=schedule_score,
        reason="test",
        caveats=(),
    )


# ---------------------------------------------------------------------------
# Outcome policy tests
# ---------------------------------------------------------------------------


def test_favorable_supports_first_contradicts_second() -> None:
    state, health, _ = _build_state_health_plan("tennent")
    assert health.ambiguity_pairs
    a, b = health.ambiguity_pairs[0]
    sched = _make_scheduled()
    result = simulate_collect_result(
        state, health, sched,
        scenario_id="tennent", outcome_policy="favorable",
    )
    ev = result.returned_evidence[0]
    assert a in ev.supports
    assert b in ev.contradicts


def test_adverse_supports_second_contradicts_first() -> None:
    state, health, _ = _build_state_health_plan("tennent")
    a, b = health.ambiguity_pairs[0]
    sched = _make_scheduled()
    result = simulate_collect_result(
        state, health, sched,
        scenario_id="tennent", outcome_policy="adverse",
    )
    ev = result.returned_evidence[0]
    assert b in ev.supports
    assert a in ev.contradicts


def test_inconclusive_produces_weak_evidence() -> None:
    state, health, _ = _build_state_health_plan("tennent")
    sched = _make_scheduled()
    result = simulate_collect_result(
        state, health, sched,
        scenario_id="tennent", outcome_policy="inconclusive",
    )
    ev = result.returned_evidence[0]
    assert ev.supports == ()
    assert ev.contradicts == ()
    assert ev.confidence <= 0.10


def test_invalid_policy_raises() -> None:
    state, health, _ = _build_state_health_plan("tennent")
    sched = _make_scheduled()
    with pytest.raises(ValueError):
        simulate_collect_result(
            state, health, sched,
            scenario_id="tennent", outcome_policy="bogus",
        )


def test_mixed_alternates_favorable_inconclusive() -> None:
    state, health, plan = _build_state_health_plan("tennent")
    if len(plan.scheduled_collects) < 2:
        pytest.skip("scenario has only one scheduled collect")
    report = simulate_plan_execution(
        state, plan, scenario_id="tennent", outcome_policy="mixed",
    )
    policies = [r.outcome_policy for r in report.simulated_results]
    assert policies[0] == "favorable"
    assert policies[1] == "inconclusive"


def test_mixed_single_collect_uses_mixed_label() -> None:
    state, health, _ = _build_state_health_plan("tennent")
    one_collect = (_make_scheduled(),)
    plan = SchedulePlan(
        scenario_id="tennent",
        scheduled_collects=one_collect,
        unscheduled_candidate_ids=(),
        total_capacity_used=0.3,
        total_schedule_score=0.65,
        constraint=ScheduleConstraint(),
        summary="x",
        caveats=(),
    )
    report = simulate_plan_execution(
        state, plan, scenario_id="tennent", outcome_policy="mixed",
    )
    assert report.simulated_results[0].outcome_policy == "mixed"
    assert report.simulated_results[0].returned_evidence[0].confidence > 0.10


# ---------------------------------------------------------------------------
# Evidence construction
# ---------------------------------------------------------------------------


def test_evidence_timestamp_from_scheduled_end_time() -> None:
    state, health, _ = _build_state_health_plan("tennent")
    end = datetime(2023, 8, 14, 16, 0, tzinfo=timezone.utc)
    sched = _make_scheduled(end_time=end)
    result = simulate_collect_result(
        state, health, sched,
        scenario_id="tennent", outcome_policy="favorable",
    )
    assert result.returned_evidence[0].timestamp == end


def test_result_id_deterministic() -> None:
    state, health, _ = _build_state_health_plan("tennent")
    sched = _make_scheduled()
    a = simulate_collect_result(state, health, sched, scenario_id="tennent")
    b = simulate_collect_result(state, health, sched, scenario_id="tennent")
    assert a.result_id == b.result_id


def test_evidence_source_kind() -> None:
    state, health, _ = _build_state_health_plan("tennent")
    sched = _make_scheduled()
    result = simulate_collect_result(
        state, health, sched, scenario_id="tennent",
    )
    assert result.returned_evidence[0].source_kind == "execution_simulation"


def test_evidence_source_ref_format() -> None:
    state, health, _ = _build_state_health_plan("tennent")
    sched = _make_scheduled(window_id="w-XYZ", candidate_id="optical_context")
    result = simulate_collect_result(
        state, health, sched, scenario_id="tennent",
    )
    assert result.returned_evidence[0].source_ref == (
        "schedule:w-XYZ:optical_context"
    )


def test_evidence_weight_one() -> None:
    state, health, _ = _build_state_health_plan("tennent")
    sched = _make_scheduled()
    result = simulate_collect_result(
        state, health, sched, scenario_id="tennent",
    )
    assert result.returned_evidence[0].weight == 1.0


# ---------------------------------------------------------------------------
# State update integration
# ---------------------------------------------------------------------------


def test_simulate_plan_execution_returns_post_state_changes() -> None:
    state, _, plan = _build_state_health_plan("tennent")
    report = simulate_plan_execution(
        state, plan, scenario_id="tennent", outcome_policy="favorable",
    )
    # health score delta or top hypothesis change after favorable simulation
    assert report.health_score_delta != 0 or (
        report.pre_top_hypothesis != report.post_top_hypothesis
    )


def test_simulate_plan_execution_post_recommendation_is_recommendation() -> None:
    state, _, plan = _build_state_health_plan("tennent")
    report = simulate_plan_execution(
        state, plan, scenario_id="tennent",
    )
    assert isinstance(report.post_recommendation, CollectionRecommendation)


def test_inconclusive_preserves_primary_ambiguity() -> None:
    state, _, plan = _build_state_health_plan("tennent")
    report = simulate_plan_execution(
        state, plan, scenario_id="tennent", outcome_policy="inconclusive",
    )
    assert report.ambiguity_resolved is False


def test_adverse_changes_top_or_evidence() -> None:
    state, _, plan = _build_state_health_plan("tennent")
    report = simulate_plan_execution(
        state, plan, scenario_id="tennent", outcome_policy="adverse",
    )
    # Either top hypothesis flipped or returned-evidence trace exists
    assert (
        report.pre_top_hypothesis != report.post_top_hypothesis
        or any(r.returned_evidence for r in report.simulated_results)
    )


# ---------------------------------------------------------------------------
# Ambiguity resolution
# ---------------------------------------------------------------------------


def test_ambiguity_resolved_false_when_no_pre_pairs() -> None:
    state, health, _ = _build_state_health_plan("tennent")
    # Force state with no ambiguity (use empty plan)
    empty_plan = SchedulePlan(
        scenario_id="tennent",
        scheduled_collects=(),
        unscheduled_candidate_ids=(),
        total_capacity_used=0.0,
        total_schedule_score=0.0,
        constraint=ScheduleConstraint(),
        summary="x",
        caveats=(),
    )
    report = simulate_plan_execution(
        state, empty_plan, scenario_id="tennent",
    )
    if not report.pre_ambiguity_pairs:
        assert report.ambiguity_resolved is False


# ---------------------------------------------------------------------------
# No-ambiguity edge case
# ---------------------------------------------------------------------------


def test_no_ambiguity_returns_empty_evidence() -> None:
    from custody.hypotheses.custody_health import HypothesisCustodyHealth
    state, _, _ = _build_state_health_plan("tennent")
    fake_health = HypothesisCustodyHealth(
        status=CustodyHealthStatus.HEALTHY,
        score=0.9,
        top_hypothesis="x",
        top_score=0.9,
        second_hypothesis=None,
        second_score=0.0,
        top_two_margin=0.9,
        ambiguity_pairs=(),
        latest_evidence_at=None,
        drivers=(),
        reason="no ambiguity",
    )
    sched = _make_scheduled()
    result = simulate_collect_result(
        state, fake_health, sched, scenario_id="tennent",
    )
    assert result.returned_evidence == ()
    assert result.result_quality == 0.0


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_execution_report_to_dict_json_safe() -> None:
    state, _, plan = _build_state_health_plan("tennent")
    report = simulate_plan_execution(
        state, plan, scenario_id="tennent",
    )
    d = execution_report_to_dict(report)
    encoded = json.dumps(d)
    assert json.loads(encoded) == d


def test_execution_report_to_json_parseable() -> None:
    state, _, plan = _build_state_health_plan("whitsun")
    report = simulate_plan_execution(
        state, plan, scenario_id="whitsun", outcome_policy="adverse",
    )
    parsed = json.loads(execution_report_to_json(report))
    assert parsed["scenario_id"] == "whitsun"
    assert parsed["outcome_policy"] == "adverse"


def test_datetimes_serialized_as_iso() -> None:
    state, _, plan = _build_state_health_plan("tennent")
    report = simulate_plan_execution(state, plan, scenario_id="tennent")
    d = execution_report_to_dict(report)
    if d["scheduled_collects"]:
        iso = d["scheduled_collects"][0]["start_time"]
        datetime.fromisoformat(iso)


def test_enum_serialized_as_string() -> None:
    state, _, plan = _build_state_health_plan("tennent")
    report = simulate_plan_execution(state, plan, scenario_id="tennent")
    d = execution_report_to_dict(report)
    assert isinstance(d["pre_health_status"], str)
    assert isinstance(d["post_health_status"], str)


# ---------------------------------------------------------------------------
# Caveats
# ---------------------------------------------------------------------------


def test_caveats_include_synthetic() -> None:
    state, _, plan = _build_state_health_plan("tennent")
    report = simulate_plan_execution(state, plan, scenario_id="tennent")
    joined = " ".join(report.caveats).lower()
    assert "synthetic returned evidence" in joined


def test_caveats_include_no_live_tasking() -> None:
    state, _, plan = _build_state_health_plan("tennent")
    report = simulate_plan_execution(state, plan, scenario_id="tennent")
    joined = " ".join(report.caveats).lower()
    assert "no live tasking or sensor command is issued" in joined


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def test_text_format_contains_required_sections() -> None:
    state, _, plan = _build_state_health_plan("tennent")
    report = simulate_plan_execution(state, plan, scenario_id="tennent")
    text = format_execution_text(report)
    assert "PLAN EXECUTION SIMULATION" in text
    assert "Pre-execution" in text
    assert "Scheduled collects" in text
    assert "Simulated returned evidence" in text
    assert "Post-execution" in text
    assert "Before/after health" in text
    assert "Caveats" in text


def test_markdown_format_contains_heading() -> None:
    state, _, plan = _build_state_health_plan("whitsun")
    report = simulate_plan_execution(state, plan, scenario_id="whitsun")
    md = format_execution_markdown(report)
    assert md.startswith("# Plan Execution Simulation")


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------


def test_no_forbidden_imports() -> None:
    import custody.hypotheses.execution_sim as mod
    tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
    forbidden = (
        "custody.detection",
        "custody.ingest.gfw_presence",
        "custody.ingest.sentinel",
        "custody.matcher",
        "custody.dashboard",
        "custody.fusion.tracker",
        "custody.taskrecommendation",
        "sentinelhub",
        "requests",
        "httpx",
        "urllib.request",
        "boto3",
        "google.cloud",
    )
    offending: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if any(alias.name.startswith(p) for p in forbidden):
                    offending.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod_name = node.module or ""
            if any(mod_name.startswith(p) for p in forbidden):
                offending.append(mod_name)
    assert not offending, f"forbidden imports: {offending}"


def test_source_file_no_forbidden_language() -> None:
    import custody.hypotheses.execution_sim as mod
    raw = Path(mod.__file__).read_text(encoding="utf-8").lower()
    # Collapse whitespace so docstring line wraps don't split disclaimers.
    src = " ".join(raw.split())
    forbidden = (
        "actual collection result",
        "tasking order",
        "satellite command",
        "real image",
        "production scheduler",
        "autonomous execution",
        "real mps control",
        "sentinel integration",
        "collection order",
        "revenue dollars",
    )
    for needle in forbidden:
        assert needle not in src, f"forbidden token {needle!r} in source"
    # "live tasking" / "sensor command" / "platform access" appear only inside
    # explicit non-claim disclaimers; mask before scanning.
    safe = src.replace(
        "no live tasking or sensor command is issued", "",
    ).replace(
        "no platform-access decisions are claimed", "",
    )
    for needle in ("live tasking", "sensor command", "platform access"):
        assert needle not in safe, (
            f"forbidden token {needle!r} appears outside explicit disclaimer"
        )
