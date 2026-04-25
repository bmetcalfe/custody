"""Tests for :mod:`custody.hypotheses.efficiency_metrics` (ADR-0021 Slice 15).

Pin the baseline + custody workflow shapes, the seven proxy metrics,
deterministic deltas, JSON / Markdown / text formatters, scope
guardrails (no forbidden imports), and the language guardrails (no
production-savings / monetary / live-tasking / sensor-command claims).
"""
from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from pathlib import Path

import pytest

from custody.hypotheses.custody_health import CustodyHealthStatus
from custody.hypotheses.efficiency_metrics import (
    EfficiencyMetric,
    EfficiencyReport,
    WorkflowMode,
    WorkflowModel,
    WorkflowStep,
    build_baseline_workflow,
    build_custody_workflow,
    compute_efficiency_metrics,
    efficiency_report_to_json_object,
    format_efficiency_markdown,
    format_efficiency_text,
)
from custody.hypotheses.planner_queue import (
    PlannerQueueItem,
    PlannerQueueStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_queue_item(
    *,
    scenario_id: str = "tennent",
    status: PlannerQueueStatus = PlannerQueueStatus.PENDING_REVIEW,
    priority_score: float = 0.85,
    health_status: CustodyHealthStatus = CustodyHealthStatus.AMBIGUOUS,
    health_score: float = 0.35,
    primary_ambiguity: tuple[str, str] | None = (
        "construction_or_reclamation_activity",
        "fixed_reclamation_or_structure",
    ),
    recommended_candidate_ids: tuple[str, ...] = (
        "optical_context", "repeat_sar",
    ),
    planning_utility: float = 1.05,
    mission_value_proxy: float = 0.51,
    expected_ambiguity_resolution: float = 0.85,
    expected_health_score_delta: float = 0.55,
) -> PlannerQueueItem:
    return PlannerQueueItem(
        item_id="qi-test",
        scenario_id=scenario_id,
        status=status,
        priority_score=priority_score,
        health_status=health_status,
        health_score=health_score,
        primary_ambiguity=primary_ambiguity,
        recommended_candidate_ids=recommended_candidate_ids,
        planning_utility=planning_utility,
        mission_value_proxy=mission_value_proxy,
        expected_ambiguity_resolution=expected_ambiguity_resolution,
        expected_health_score_delta=expected_health_score_delta,
        latest_review_action=None,
        latest_review_id=None,
        reason="synthetic",
        next_action="review recommended candidate collect plan",
        caveats=(),
    )


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


def test_workflow_step_is_frozen() -> None:
    step = build_baseline_workflow(scenario_id="tennent").steps[0]
    with pytest.raises(FrozenInstanceError):
        step.label = "x"  # type: ignore[misc]


def test_efficiency_report_is_frozen() -> None:
    report = compute_efficiency_metrics(
        scenario_id="tennent", queue_item=_make_queue_item(),
    )
    with pytest.raises(FrozenInstanceError):
        report.summary = "x"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Workflow shape tests
# ---------------------------------------------------------------------------


def test_baseline_workflow_has_four_manual_steps() -> None:
    wf = build_baseline_workflow(scenario_id="tennent")
    assert len(wf.steps) == 4
    assert all(s.manual_attention_required for s in wf.steps)
    assert wf.mode is WorkflowMode.BASELINE


def test_baseline_workflow_has_one_audit_artifact() -> None:
    wf = build_baseline_workflow(scenario_id="tennent")
    artifact_steps = [s for s in wf.steps if s.produces_audit_artifact]
    assert len(artifact_steps) == 1
    assert artifact_steps[0].step_id == "planner_review"


def test_custody_workflow_has_six_steps() -> None:
    wf = build_custody_workflow(_make_queue_item(), scenario_id="tennent")
    assert len(wf.steps) == 6
    assert wf.mode is WorkflowMode.CUSTODY_ASSISTED


def test_custody_workflow_has_one_manual_step() -> None:
    wf = build_custody_workflow(_make_queue_item(), scenario_id="tennent")
    manual = [s for s in wf.steps if s.manual_attention_required]
    assert len(manual) == 1
    assert manual[0].step_id == "human_review_queue"


def test_custody_workflow_has_six_audit_artifacts() -> None:
    wf = build_custody_workflow(_make_queue_item(), scenario_id="tennent")
    artifacts = [s for s in wf.steps if s.produces_audit_artifact]
    assert len(artifacts) == 6


# ---------------------------------------------------------------------------
# Per-metric tests
# ---------------------------------------------------------------------------


def _metric_by_id(report: EfficiencyReport) -> dict[str, EfficiencyMetric]:
    return {m.metric_id: m for m in report.metrics}


def test_manual_triage_steps_lower_is_better() -> None:
    report = compute_efficiency_metrics(
        scenario_id="tennent", queue_item=_make_queue_item(),
    )
    m = _metric_by_id(report)["manual_triage_steps"]
    assert m.baseline_value == 4.0
    assert m.custody_value == 1.0
    assert m.delta == -3.0
    assert m.improvement_direction == "lower_is_better"


def test_audit_artifact_count_higher_is_better() -> None:
    report = compute_efficiency_metrics(
        scenario_id="tennent", queue_item=_make_queue_item(),
    )
    m = _metric_by_id(report)["audit_artifact_count"]
    assert m.baseline_value == 1.0
    assert m.custody_value == 6.0
    assert m.delta == 5.0
    assert m.improvement_direction == "higher_is_better"


def test_decision_cycle_minutes_proxy_deterministic() -> None:
    a = compute_efficiency_metrics(
        scenario_id="tennent", queue_item=_make_queue_item(),
    )
    b = compute_efficiency_metrics(
        scenario_id="tennent", queue_item=_make_queue_item(),
    )
    am = _metric_by_id(a)["expected_decision_cycle_minutes_proxy"]
    bm = _metric_by_id(b)["expected_decision_cycle_minutes_proxy"]
    # Baseline: 30 + 45 + 60 + 30 = 165
    # Custody:  5 * 0.5 + 15 = 17.5
    assert am.baseline_value == 165.0
    assert am.custody_value == 17.5
    assert am.delta == bm.delta == -147.5


def test_ambiguity_focus_score_with_primary_ambiguity() -> None:
    report = compute_efficiency_metrics(
        scenario_id="tennent", queue_item=_make_queue_item(),
    )
    m = _metric_by_id(report)["ambiguity_focus_score"]
    assert m.baseline_value == 0.0
    assert m.custody_value == 1.0
    assert m.delta == 1.0


def test_ambiguity_focus_score_without_ambiguity() -> None:
    report = compute_efficiency_metrics(
        scenario_id="tennent",
        queue_item=_make_queue_item(
            primary_ambiguity=None,
            recommended_candidate_ids=(),
        ),
    )
    m = _metric_by_id(report)["ambiguity_focus_score"]
    assert m.custody_value == 0.0


def test_ambiguity_focus_score_pair_only_returns_half() -> None:
    """Pair present but no candidates -> score 0.5."""
    report = compute_efficiency_metrics(
        scenario_id="tennent",
        queue_item=_make_queue_item(
            primary_ambiguity=("a", "b"),
            recommended_candidate_ids=(),
        ),
    )
    m = _metric_by_id(report)["ambiguity_focus_score"]
    assert m.custody_value == 0.5


def test_planner_attention_focus_uses_priority_score() -> None:
    qi = _make_queue_item(priority_score=0.73)
    report = compute_efficiency_metrics(
        scenario_id="tennent", queue_item=qi,
    )
    m = _metric_by_id(report)["planner_attention_focus"]
    assert m.baseline_value == 0.0
    assert m.custody_value == 0.73
    assert m.delta == 0.73


def test_decision_traceability_baseline_vs_custody() -> None:
    report = compute_efficiency_metrics(
        scenario_id="tennent", queue_item=_make_queue_item(),
    )
    m = _metric_by_id(report)["decision_traceability"]
    # Baseline: 1/4 = 0.25; Custody: 6/6 = 1.0.
    assert m.baseline_value == pytest.approx(0.25)
    assert m.custody_value == pytest.approx(1.0)


def test_collect_strategy_comparison_count() -> None:
    qi = _make_queue_item(
        recommended_candidate_ids=("a", "b", "c"),
    )
    report = compute_efficiency_metrics(
        scenario_id="tennent", queue_item=qi,
    )
    m = _metric_by_id(report)["collect_strategy_comparison_count"]
    assert m.baseline_value == 1.0
    assert m.custody_value == 3.0
    assert m.delta == 2.0


def test_collect_strategy_comparison_count_falls_back_when_empty() -> None:
    qi = _make_queue_item(recommended_candidate_ids=())
    report = compute_efficiency_metrics(
        scenario_id="tennent", queue_item=qi,
    )
    m = _metric_by_id(report)["collect_strategy_comparison_count"]
    assert m.custody_value == 1.0


# ---------------------------------------------------------------------------
# Cross-metric invariants
# ---------------------------------------------------------------------------


def test_metric_deltas_are_custody_minus_baseline() -> None:
    report = compute_efficiency_metrics(
        scenario_id="tennent", queue_item=_make_queue_item(),
    )
    for m in report.metrics:
        expected = round(m.custody_value - m.baseline_value, 4)
        assert m.delta == expected, (
            f"{m.metric_id} delta {m.delta} != custody-baseline {expected}"
        )


def test_improvement_direction_values_are_valid() -> None:
    report = compute_efficiency_metrics(
        scenario_id="tennent", queue_item=_make_queue_item(),
    )
    for m in report.metrics:
        assert m.improvement_direction in {
            "lower_is_better", "higher_is_better",
        }


def test_report_includes_proxy_caveats() -> None:
    report = compute_efficiency_metrics(
        scenario_id="tennent", queue_item=_make_queue_item(),
    )
    joined = " ".join(report.caveats).lower()
    assert "prototype" in joined
    assert "proxy" in joined
    assert "decision-support" in joined or "decision support" in joined


# ---------------------------------------------------------------------------
# Language guardrails
# ---------------------------------------------------------------------------


_FORBIDDEN = (
    "proven savings",
    "actual revenue",
    "production cycle-time reduction",
    "real planner adoption",
    "operational kpi",
    "tasking order",
    "live tasking",
    "sensor command",
    "production scheduler",
    "autonomous execution",
    "collection order",
    "sentinel integration",
)


def test_summary_contains_no_forbidden_language() -> None:
    """Scan summary, interpretations, and caveats for forbidden tokens."""
    report = compute_efficiency_metrics(
        scenario_id="tennent", queue_item=_make_queue_item(),
    )
    blobs = [report.summary]
    for m in report.metrics:
        blobs.append(m.interpretation)
        blobs.append(m.label)
        blobs.extend(m.caveats)
    blobs.extend(report.caveats)
    blobs.append(report.baseline_workflow.summary)
    blobs.append(report.custody_workflow.summary)
    for s in report.baseline_workflow.steps:
        blobs.extend([s.label, s.description])
        blobs.extend(s.caveats)
    for s in report.custody_workflow.steps:
        blobs.extend([s.label, s.description])
        blobs.extend(s.caveats)
    text = " ".join(blobs).lower()
    for needle in _FORBIDDEN:
        assert needle not in text, (
            f"forbidden token {needle!r} appears in efficiency report"
        )


def test_source_file_does_not_contain_forbidden_language() -> None:
    import custody.hypotheses.efficiency_metrics as mod
    src = Path(mod.__file__).read_text(encoding="utf-8").lower()
    for needle in _FORBIDDEN:
        assert needle not in src, (
            f"forbidden token {needle!r} appears in efficiency_metrics.py source"
        )


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def test_text_format_contains_required_sections() -> None:
    text = format_efficiency_text(compute_efficiency_metrics(
        scenario_id="tennent", queue_item=_make_queue_item(),
    ))
    assert "WORKFLOW EFFICIENCY REPORT" in text
    assert "Baseline workflow" in text
    assert "Custody-assisted workflow" in text
    assert "Proxy metrics" in text
    assert "Caveats" in text


def test_json_round_trips_deterministically() -> None:
    report = compute_efficiency_metrics(
        scenario_id="tennent", queue_item=_make_queue_item(),
    )
    a = json.dumps(efficiency_report_to_json_object(report), indent=2)
    b = json.dumps(efficiency_report_to_json_object(report), indent=2)
    assert a == b
    parsed = json.loads(a)
    assert parsed["scenario_id"] == "tennent"
    assert "metrics" in parsed
    assert isinstance(parsed["metrics"], list)


def test_markdown_contains_heading() -> None:
    md = format_efficiency_markdown(compute_efficiency_metrics(
        scenario_id="tennent", queue_item=_make_queue_item(),
    ))
    assert md.startswith("# Workflow Efficiency Report - tennent")


# ---------------------------------------------------------------------------
# Import-boundary scan
# ---------------------------------------------------------------------------


def test_module_does_not_import_detection_or_gfw() -> None:
    import ast
    import custody.hypotheses.efficiency_metrics as mod
    tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
    forbidden_prefixes = (
        "custody.detection",
        "custody.ingest.gfw_presence",
        "sentinelhub",
        "custody.fusion.tracker",
        "custody.taskrecommendation",
    )
    offending: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if any(alias.name.startswith(p) for p in forbidden_prefixes):
                    offending.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod_name = node.module or ""
            if any(mod_name.startswith(p) for p in forbidden_prefixes):
                offending.append(mod_name)
    assert not offending, (
        f"efficiency_metrics.py imports forbidden modules: {offending}"
    )
