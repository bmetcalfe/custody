"""Workflow efficiency proxy metrics (ADR-0021 Slice 15).

Models two mission-planning workflows - a baseline / manual triage path
and the Custody-assisted decision-support path - and computes
deterministic proxy metrics comparing them.

This is a **prototype efficiency model**.  Every metric is a proxy
estimate based on hand-authored workflow models; nothing here measures
operational performance, claims any monetary impact, or asserts any
real-world workflow timing.

Design constraints
------------------

- Deterministic: no wall-clock, no randomness.
- Pure stdlib: no new runtime dependencies.
- No imports from ``custody.detection.*``, ``custody.ingest.gfw_presence``,
  matcher runtime, Sentinel SDKs, or real-data loaders.
- Language: "efficiency proxy", "prototype metrics".  No claim of
  measured timing, monetary value, or organisational performance.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum

from custody.hypotheses.planner_queue import PlannerQueueItem


# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------


class WorkflowMode(Enum):
    BASELINE = "baseline"
    CUSTODY_ASSISTED = "custody_assisted"


@dataclass(frozen=True)
class WorkflowStep:
    step_id: str
    label: str
    description: str
    mode: WorkflowMode
    manual_attention_required: bool
    produces_audit_artifact: bool
    expected_minutes_proxy: float
    ambiguity_reduction_proxy: float
    caveats: tuple[str, ...]


@dataclass(frozen=True)
class WorkflowModel:
    mode: WorkflowMode
    steps: tuple[WorkflowStep, ...]
    summary: str
    caveats: tuple[str, ...]


@dataclass(frozen=True)
class EfficiencyMetric:
    metric_id: str
    label: str
    baseline_value: float
    custody_value: float
    delta: float
    improvement_direction: str  # "lower_is_better" or "higher_is_better"
    interpretation: str
    caveats: tuple[str, ...]


@dataclass(frozen=True)
class EfficiencyReport:
    scenario_id: str
    baseline_workflow: WorkflowModel
    custody_workflow: WorkflowModel
    metrics: tuple[EfficiencyMetric, ...]
    summary: str
    caveats: tuple[str, ...]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


_LOWER_IS_BETTER = "lower_is_better"
_HIGHER_IS_BETTER = "higher_is_better"

# Per-metric proxy caveat (avoids the banned tokens by design).
_METRIC_PROXY_CAVEAT = "proxy metric; not measured workflow timing data"

_REPORT_CAVEATS: tuple[str, ...] = (
    "metrics are prototype proxy metrics, not measured workflow timing",
    "decision-support workflow only; no execution or scheduling commitments are made",
    "no monetary or organisational performance value is claimed",
)


# ---------------------------------------------------------------------------
# Baseline workflow
# ---------------------------------------------------------------------------


def _baseline_steps() -> tuple[WorkflowStep, ...]:
    base_caveat = ("proxy step description; not measured operational data",)
    return (
        WorkflowStep(
            step_id="detector_output_review",
            label="Detector output review",
            description=(
                "Analyst inspects raw detector outputs without a structured "
                "ambiguity model"
            ),
            mode=WorkflowMode.BASELINE,
            manual_attention_required=True,
            produces_audit_artifact=False,
            expected_minutes_proxy=30.0,
            ambiguity_reduction_proxy=0.0,
            caveats=base_caveat,
        ),
        WorkflowStep(
            step_id="analyst_ambiguity_triage",
            label="Analyst ambiguity triage",
            description=(
                "Analyst manually triages competing explanations for the "
                "observed activity"
            ),
            mode=WorkflowMode.BASELINE,
            manual_attention_required=True,
            produces_audit_artifact=False,
            expected_minutes_proxy=45.0,
            ambiguity_reduction_proxy=0.10,
            caveats=base_caveat,
        ),
        WorkflowStep(
            step_id="ad_hoc_collect_discussion",
            label="Ad hoc collect discussion",
            description=(
                "Free-form discussion of which collection options to pursue, "
                "without ranked recommendations"
            ),
            mode=WorkflowMode.BASELINE,
            manual_attention_required=True,
            produces_audit_artifact=False,
            expected_minutes_proxy=60.0,
            ambiguity_reduction_proxy=0.05,
            caveats=base_caveat,
        ),
        WorkflowStep(
            step_id="planner_review",
            label="Planner review",
            description=(
                "Planner reviews the discussion outcome and records a "
                "decision in a free-text artifact"
            ),
            mode=WorkflowMode.BASELINE,
            manual_attention_required=True,
            produces_audit_artifact=True,
            expected_minutes_proxy=30.0,
            ambiguity_reduction_proxy=0.0,
            caveats=base_caveat,
        ),
    )


def build_baseline_workflow(*, scenario_id: str) -> WorkflowModel:
    """Build the baseline / manual triage workflow proxy.

    The workflow has four manual steps and produces one audit artifact.
    All numeric values are hand-authored proxy estimates.
    """
    steps = _baseline_steps()
    return WorkflowModel(
        mode=WorkflowMode.BASELINE,
        steps=steps,
        summary=(
            f"baseline workflow for {scenario_id}: 4 manual steps, "
            f"1 audit artifact (proxy estimates only)"
        ),
        caveats=("proxy workflow model; not measured operational data",),
    )


# ---------------------------------------------------------------------------
# Custody-assisted workflow
# ---------------------------------------------------------------------------


def _custody_steps(queue_item: PlannerQueueItem) -> tuple[WorkflowStep, ...]:
    has_pair = queue_item.primary_ambiguity is not None
    base_caveat = ("proxy step description; deterministic prototype model",)
    # Slice 5 ranking gets a small bump when a primary ambiguity is present,
    # since the strategy table directly addresses such pairs.
    cv_amb_reduction = 0.25 if has_pair else 0.20
    return (
        WorkflowStep(
            step_id="hypothesis_state_update",
            label="Hypothesis state update",
            description=(
                "Slice 1-3 evidence-to-state update with auditable trace"
            ),
            mode=WorkflowMode.CUSTODY_ASSISTED,
            manual_attention_required=False,
            produces_audit_artifact=True,
            expected_minutes_proxy=0.5,
            ambiguity_reduction_proxy=0.15,
            caveats=base_caveat,
        ),
        WorkflowStep(
            step_id="custody_health_assessment",
            label="Custody health assessment",
            description=(
                "Slice 4 health classification with canonical ambiguity pairs"
            ),
            mode=WorkflowMode.CUSTODY_ASSISTED,
            manual_attention_required=False,
            produces_audit_artifact=True,
            expected_minutes_proxy=0.5,
            ambiguity_reduction_proxy=0.10,
            caveats=base_caveat,
        ),
        WorkflowStep(
            step_id="collection_value_ranking",
            label="Collection value ranking",
            description=(
                "Slice 5 strategy-table ranking of candidate collect types"
            ),
            mode=WorkflowMode.CUSTODY_ASSISTED,
            manual_attention_required=False,
            produces_audit_artifact=True,
            expected_minutes_proxy=0.5,
            ambiguity_reduction_proxy=cv_amb_reduction,
            caveats=base_caveat,
        ),
        WorkflowStep(
            step_id="mission_value_proxy",
            label="Mission value proxy",
            description=(
                "Slice 9 value attribution proxy across named components"
            ),
            mode=WorkflowMode.CUSTODY_ASSISTED,
            manual_attention_required=False,
            produces_audit_artifact=True,
            expected_minutes_proxy=0.5,
            ambiguity_reduction_proxy=0.05,
            caveats=base_caveat,
        ),
        WorkflowStep(
            step_id="optimized_plan",
            label="Optimized plan",
            description=(
                "Slice 11 constrained planner: exhaustive + greedy baselines"
            ),
            mode=WorkflowMode.CUSTODY_ASSISTED,
            manual_attention_required=False,
            produces_audit_artifact=True,
            expected_minutes_proxy=0.5,
            ambiguity_reduction_proxy=0.10,
            caveats=base_caveat,
        ),
        WorkflowStep(
            step_id="human_review_queue",
            label="Human review queue",
            description=(
                "Slice 13-14 operator review and ranked work queue with "
                "auditable record"
            ),
            mode=WorkflowMode.CUSTODY_ASSISTED,
            manual_attention_required=True,
            produces_audit_artifact=True,
            expected_minutes_proxy=15.0,
            ambiguity_reduction_proxy=0.0,
            caveats=base_caveat,
        ),
    )


def build_custody_workflow(
    queue_item: PlannerQueueItem,
    *,
    scenario_id: str,
) -> WorkflowModel:
    """Build the Custody-assisted workflow proxy.

    Six steps, only the human review step is manual; every step produces
    an audit artifact.  Slice 5 step's ambiguity reduction is bumped
    slightly when the queue item carries a primary ambiguity pair so
    the strategy table is directly applicable.
    """
    steps = _custody_steps(queue_item)
    return WorkflowModel(
        mode=WorkflowMode.CUSTODY_ASSISTED,
        steps=steps,
        summary=(
            f"custody-assisted workflow for {scenario_id}: 6 steps, "
            f"1 manual, 6 audit artifacts (proxy estimates only)"
        ),
        caveats=("proxy workflow model; actual elapsed times will vary",),
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _round4(x: float) -> float:
    return round(float(x), 4)


def _count_manual(steps: tuple[WorkflowStep, ...]) -> int:
    return sum(1 for s in steps if s.manual_attention_required)


def _count_artifacts(steps: tuple[WorkflowStep, ...]) -> int:
    return sum(1 for s in steps if s.produces_audit_artifact)


def _sum_minutes(steps: tuple[WorkflowStep, ...]) -> float:
    return sum(s.expected_minutes_proxy for s in steps)


def _ambiguity_focus(queue_item: PlannerQueueItem) -> float:
    """Proxy: 1.0 when both an ambiguity pair and ranked candidates exist,
    0.5 with a pair only, 0.0 otherwise."""
    has_pair = queue_item.primary_ambiguity is not None
    has_candidates = bool(queue_item.recommended_candidate_ids)
    if has_pair and has_candidates:
        return 1.0
    if has_pair:
        return 0.5
    return 0.0


def _metric(
    metric_id: str,
    label: str,
    baseline_value: float,
    custody_value: float,
    direction: str,
    interpretation: str,
) -> EfficiencyMetric:
    return EfficiencyMetric(
        metric_id=metric_id,
        label=label,
        baseline_value=_round4(baseline_value),
        custody_value=_round4(custody_value),
        delta=_round4(custody_value - baseline_value),
        improvement_direction=direction,
        interpretation=interpretation,
        caveats=(_METRIC_PROXY_CAVEAT,),
    )


def compute_efficiency_metrics(
    *,
    scenario_id: str,
    queue_item: PlannerQueueItem,
    baseline_workflow: WorkflowModel | None = None,
    custody_workflow: WorkflowModel | None = None,
) -> EfficiencyReport:
    """Compose the seven proxy metrics into a deterministic report."""
    baseline = (
        baseline_workflow if baseline_workflow is not None
        else build_baseline_workflow(scenario_id=scenario_id)
    )
    custody = (
        custody_workflow if custody_workflow is not None
        else build_custody_workflow(queue_item, scenario_id=scenario_id)
    )

    bs = baseline.steps
    cs = custody.steps

    manual_baseline = _count_manual(bs)
    manual_custody = _count_manual(cs)
    artifact_baseline = _count_artifacts(bs)
    artifact_custody = _count_artifacts(cs)
    minutes_baseline = _sum_minutes(bs)
    minutes_custody = _sum_minutes(cs)

    candidate_count = len(queue_item.recommended_candidate_ids)
    strategy_count_baseline = 1.0
    strategy_count_custody = (
        float(candidate_count) if candidate_count > 0 else 1.0
    )

    traceability_baseline = (
        artifact_baseline / len(bs) if bs else 0.0
    )
    traceability_custody = (
        artifact_custody / len(cs) if cs else 0.0
    )

    metrics: list[EfficiencyMetric] = []

    metrics.append(_metric(
        "manual_triage_steps",
        "Manual triage steps",
        manual_baseline,
        manual_custody,
        _LOWER_IS_BETTER,
        (
            f"baseline requires {manual_baseline} manual triage step(s); "
            f"custody-assisted requires {manual_custody}; delta "
            f"{int(manual_custody - manual_baseline):+d}"
        ),
    ))

    metrics.append(_metric(
        "audit_artifact_count",
        "Audit artifact count",
        artifact_baseline,
        artifact_custody,
        _HIGHER_IS_BETTER,
        (
            f"custody-assisted produces {artifact_custody} audit artifact(s) "
            f"vs baseline's {artifact_baseline}; delta "
            f"{int(artifact_custody - artifact_baseline):+d}"
        ),
    ))

    metrics.append(_metric(
        "expected_decision_cycle_minutes_proxy",
        "Expected decision cycle minutes (proxy)",
        minutes_baseline,
        minutes_custody,
        _LOWER_IS_BETTER,
        (
            f"baseline ~{minutes_baseline:.1f} proxy minutes; "
            f"custody-assisted ~{minutes_custody:.1f}; delta "
            f"{minutes_custody - minutes_baseline:+.1f} (proxy estimates only)"
        ),
    ))

    amb_focus_custody = _ambiguity_focus(queue_item)
    metrics.append(_metric(
        "ambiguity_focus_score",
        "Ambiguity focus score",
        0.0,
        amb_focus_custody,
        _HIGHER_IS_BETTER,
        (
            "custody surfaces a primary ambiguity and ranked candidates "
            "explicitly; baseline does not encode ambiguity in a structured way"
            if amb_focus_custody > 0.0
            else "no primary ambiguity surfaced; both workflows score 0"
        ),
    ))

    metrics.append(_metric(
        "planner_attention_focus",
        "Planner attention focus",
        0.0,
        float(queue_item.priority_score),
        _HIGHER_IS_BETTER,
        (
            f"custody assigns priority score {queue_item.priority_score:.2f} "
            f"from the planner queue; baseline has no equivalent ranking"
        ),
    ))

    metrics.append(_metric(
        "decision_traceability",
        "Decision traceability",
        traceability_baseline,
        traceability_custody,
        _HIGHER_IS_BETTER,
        (
            f"custody-assisted produces an audit artifact at every step "
            f"({traceability_custody:.2f}); baseline ratio "
            f"{traceability_baseline:.2f}"
        ),
    ))

    metrics.append(_metric(
        "collect_strategy_comparison_count",
        "Collect strategy comparison count",
        strategy_count_baseline,
        strategy_count_custody,
        _HIGHER_IS_BETTER,
        (
            f"custody-assisted compares {int(strategy_count_custody)} "
            f"candidate collect type(s); baseline considers a single ad-hoc "
            f"path"
        ),
    ))

    summary = (
        f"workflow efficiency proxy for {scenario_id}: "
        f"manual triage {manual_baseline} -> {manual_custody}, "
        f"audit artifacts {artifact_baseline} -> {artifact_custody}, "
        f"traceability {traceability_baseline:.2f} -> "
        f"{traceability_custody:.2f} (proxy estimates only)"
    )

    return EfficiencyReport(
        scenario_id=scenario_id,
        baseline_workflow=baseline,
        custody_workflow=custody,
        metrics=tuple(metrics),
        summary=summary,
        caveats=_REPORT_CAVEATS,
    )


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def _step_to_dict(step: WorkflowStep) -> dict:
    return {
        "step_id": step.step_id,
        "label": step.label,
        "description": step.description,
        "mode": step.mode.value,
        "manual_attention_required": step.manual_attention_required,
        "produces_audit_artifact": step.produces_audit_artifact,
        "expected_minutes_proxy": step.expected_minutes_proxy,
        "ambiguity_reduction_proxy": step.ambiguity_reduction_proxy,
        "caveats": list(step.caveats),
    }


def _workflow_to_dict(model: WorkflowModel) -> dict:
    return {
        "mode": model.mode.value,
        "steps": [_step_to_dict(s) for s in model.steps],
        "summary": model.summary,
        "caveats": list(model.caveats),
    }


def _metric_to_dict(metric: EfficiencyMetric) -> dict:
    return {
        "metric_id": metric.metric_id,
        "label": metric.label,
        "baseline_value": metric.baseline_value,
        "custody_value": metric.custody_value,
        "delta": metric.delta,
        "improvement_direction": metric.improvement_direction,
        "interpretation": metric.interpretation,
        "caveats": list(metric.caveats),
    }


def efficiency_report_to_json_object(report: EfficiencyReport) -> dict:
    """Plain-dict representation suitable for ``json.dumps``."""
    return {
        "scenario_id": report.scenario_id,
        "baseline_workflow": _workflow_to_dict(report.baseline_workflow),
        "custody_workflow": _workflow_to_dict(report.custody_workflow),
        "metrics": [_metric_to_dict(m) for m in report.metrics],
        "summary": report.summary,
        "caveats": list(report.caveats),
    }


def format_efficiency_text(report: EfficiencyReport) -> str:
    """Human-readable text rendering."""
    parts: list[str] = []
    title = f"WORKFLOW EFFICIENCY REPORT - {report.scenario_id.upper()}"
    parts.append(title + "\n")
    parts.append("=" * len(title) + "\n")

    parts.append("\nBaseline workflow:\n")
    for i, step in enumerate(report.baseline_workflow.steps, start=1):
        parts.append(f"  {i}. {step.step_id}\n")

    parts.append("\nCustody-assisted workflow:\n")
    for i, step in enumerate(report.custody_workflow.steps, start=1):
        parts.append(f"  {i}. {step.step_id}\n")

    parts.append("\nProxy metrics:\n")
    for m in report.metrics:
        delta_str = (
            f"{m.delta:+.2f}" if isinstance(m.delta, float) else f"{m.delta:+d}"
        )
        parts.append(
            f"  {m.metric_id:42s}  baseline={m.baseline_value:>6.2f}  "
            f"custody={m.custody_value:>6.2f}  delta={delta_str:>7s}  "
            f"{m.improvement_direction}\n"
        )

    parts.append("\nInterpretation:\n")
    parts.append(f"  {report.summary}\n")

    parts.append("\nCaveats:\n")
    for c in report.caveats:
        parts.append(f"  - {c}\n")

    return "".join(parts)


def format_efficiency_markdown(report: EfficiencyReport) -> str:
    """Markdown rendering for vault drop-in."""
    parts: list[str] = []
    parts.append(
        f"# Workflow Efficiency Report - {report.scenario_id}\n\n"
    )

    parts.append("## Baseline workflow\n\n")
    for i, step in enumerate(report.baseline_workflow.steps, start=1):
        parts.append(
            f"{i}. **{step.step_id}** - {step.label}\n"
        )
    parts.append("\n")

    parts.append("## Custody-assisted workflow\n\n")
    for i, step in enumerate(report.custody_workflow.steps, start=1):
        parts.append(
            f"{i}. **{step.step_id}** - {step.label}\n"
        )
    parts.append("\n")

    parts.append("## Proxy metrics\n\n")
    parts.append(
        "| metric | baseline | custody | delta | improvement direction |\n"
    )
    parts.append("|---|---|---|---|---|\n")
    for m in report.metrics:
        parts.append(
            f"| {m.metric_id} | {m.baseline_value:.2f} | "
            f"{m.custody_value:.2f} | {m.delta:+.2f} | "
            f"{m.improvement_direction} |\n"
        )
    parts.append("\n")

    parts.append("## Interpretation\n\n")
    parts.append(f"{report.summary}\n\n")

    parts.append("## Caveats\n\n")
    for c in report.caveats:
        parts.append(f"- {c}\n")

    return "".join(parts)
