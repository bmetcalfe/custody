"""Deterministic counterfactual collect simulation (ADR-0021 Slice 10).

For each candidate collect ranked by Slice 5, this module simulates a
small set of plausible outcome paths against the current
:class:`HypothesisState` and reports how each candidate would shift
custody-health and ambiguity *if* the collect resolved one way or
another.

Scope guardrails
----------------

- This is a **deterministic counterfactual simulation**, not a
  calibrated sensor model.  Outcome weights are heuristic resolution-
  potential estimates, not calibrated probabilities.
- No live tasking, no platform scheduling, no production scheduler.
- Pure standard library; no new runtime dependencies.
- No imports from ``custody.detection.*``, ``custody.ingest.gfw_presence``,
  ``custody.fusion.tracker``, ``custody.taskrecommendation``, or
  Sentinel SDKs.
- ``state.timestamp`` is the only "now" the module references — no
  wall-clock.  This keeps output deterministic across invocations.

Outcome model
-------------

When custody is AMBIGUOUS with a primary ambiguity pair ``(a, b)`` the
module generates three outcomes per candidate:

  - ``resolve_toward_first``  — hypothetical evidence supports ``a``,
    contradicts ``b``.
  - ``resolve_toward_second`` — hypothetical evidence supports ``b``,
    contradicts ``a``.
  - ``inconclusive``          — empty supports/contradicts (a no-op),
    ambiguity preserved.

Weights are derived from the candidate's collection-value score
(``resolution_potential``):

  - ``resolve_a`` = ``resolution_potential / 2``
  - ``resolve_b`` = ``resolution_potential / 2``
  - ``inconclusive`` = ``1 - resolution_potential``

Aggregate metrics per candidate:

  - ``expected_health_score_delta`` = weighted average of outcome deltas
  - ``expected_ambiguity_resolution`` = sum of weights for outcomes
    that no longer report AMBIGUOUS

For non-ambiguous custody, the module emits a single cautious
"non-ambiguous" outcome per candidate and a summary explaining that
counterfactual evaluation is most useful when custody is ambiguous.
"""
from __future__ import annotations

from dataclasses import dataclass

from custody.hypotheses.collection_value import (
    CollectionRecommendation,
)
from custody.hypotheses.custody_health import (
    CustodyHealthStatus,
    HypothesisCustodyHealth,
    assess_custody_health,
)
from custody.hypotheses.mission_value import MissionValueReport
from custody.hypotheses.types import HypothesisEvidence, HypothesisState
from custody.hypotheses.update import update_state


# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CounterfactualOutcome:
    """One simulated outcome path for a candidate collect.

    ``likelihood_weight`` is a heuristic resolution-potential estimate,
    not a calibrated probability.
    """
    outcome_id: str
    label: str
    likelihood_weight: float
    resulting_top_hypothesis: str | None
    resulting_health_status: CustodyHealthStatus
    resulting_health_score: float
    ambiguity_pairs: tuple[tuple[str, str], ...]
    health_score_delta: float
    ambiguity_resolved: bool
    evidence_trace: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class CounterfactualCollectAssessment:
    """Aggregate counterfactual assessment for one candidate collect."""
    candidate_id: str
    candidate_label: str
    current_health_status: CustodyHealthStatus
    primary_ambiguity: tuple[str, str] | None
    current_mission_value: float | None
    expected_health_score_delta: float
    expected_ambiguity_resolution: float
    outcomes: tuple[CounterfactualOutcome, ...]
    summary: str


@dataclass(frozen=True)
class CounterfactualReport:
    """Ranked counterfactual assessments for all candidates in a scenario."""
    scenario_id: str
    primary_ambiguity: tuple[str, str] | None
    ranked_assessments: tuple[CounterfactualCollectAssessment, ...]
    summary: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _normalize(weights: tuple[float, ...]) -> tuple[float, ...]:
    """Normalize a tuple of non-negative weights so they sum to 1.0."""
    total = sum(weights)
    if total <= 0.0:
        # Defensive: return uniform; shouldn't occur given how we build
        # weights from a clamped score.
        n = len(weights)
        return tuple(1.0 / n for _ in weights) if n else weights
    return tuple(w / total for w in weights)


def _make_evidence(
    *,
    state: HypothesisState,
    candidate_id: str,
    outcome_tag: str,
    supports: tuple[str, ...],
    contradicts: tuple[str, ...],
    confidence: float,
    reason: str,
) -> HypothesisEvidence:
    return HypothesisEvidence(
        evidence_id=f"cf-{candidate_id}-{outcome_tag}",
        source_ref=f"counterfactual:{candidate_id}",
        source_kind="counterfactual",
        scenario_id=state.scenario_id,
        timestamp=state.timestamp,
        supports=supports,
        contradicts=contradicts,
        confidence=confidence,
        weight=1.0,
        reason=reason,
    )


def _apply(
    state: HypothesisState,
    health: HypothesisCustodyHealth,
    ev: HypothesisEvidence,
) -> tuple[HypothesisState, HypothesisCustodyHealth, float, bool]:
    new_state = update_state(state.scenario_id, evidence=(ev,), prior_state=state)
    new_health = assess_custody_health(new_state)
    delta = round(new_health.score - health.score, 4)
    resolved = new_health.status is not CustodyHealthStatus.AMBIGUOUS
    return new_state, new_health, delta, resolved


def _mission_value_lookup(
    candidate_id: str,
    mission_value_report: MissionValueReport | None,
) -> float | None:
    if mission_value_report is None:
        return None
    for a in mission_value_report.ranked_assessments:
        if a.candidate_id == candidate_id:
            return a.total_value
    return None


# ---------------------------------------------------------------------------
# Outcome builders
# ---------------------------------------------------------------------------


def _build_ambiguous_outcomes(
    state: HypothesisState,
    health: HypothesisCustodyHealth,
    candidate_id: str,
    candidate_score: float,
    pair: tuple[str, str],
) -> tuple[CounterfactualOutcome, ...]:
    a, b = pair
    resolution_potential = _clamp01(candidate_score)
    raw_weights = (
        resolution_potential / 2.0,        # resolve_a
        resolution_potential / 2.0,        # resolve_b
        1.0 - resolution_potential,        # inconclusive
    )
    norm_weights = _normalize(raw_weights)
    confidence = max(0.1, candidate_score)

    # resolve_toward_first
    ev_a = _make_evidence(
        state=state, candidate_id=candidate_id, outcome_tag="resolve-a",
        supports=(a,), contradicts=(b,), confidence=confidence,
        reason=f"counterfactual: {candidate_id} resolves toward {a}",
    )
    new_state_a, new_health_a, delta_a, resolved_a = _apply(state, health, ev_a)
    out_a = CounterfactualOutcome(
        outcome_id=f"cf-{candidate_id}-resolve-a",
        label=f"resolve_toward_first ({a})",
        likelihood_weight=round(norm_weights[0], 4),
        resulting_top_hypothesis=new_state_a.top_hypothesis,
        resulting_health_status=new_health_a.status,
        resulting_health_score=round(new_health_a.score, 3),
        ambiguity_pairs=tuple(new_health_a.ambiguity_pairs),
        health_score_delta=delta_a,
        ambiguity_resolved=resolved_a,
        evidence_trace=tuple(new_state_a.explanation),
        reason=(
            f"hypothetical evidence supports {a} and contradicts {b} at "
            f"confidence {confidence:.2f}"
        ),
    )

    # resolve_toward_second
    ev_b = _make_evidence(
        state=state, candidate_id=candidate_id, outcome_tag="resolve-b",
        supports=(b,), contradicts=(a,), confidence=confidence,
        reason=f"counterfactual: {candidate_id} resolves toward {b}",
    )
    new_state_b, new_health_b, delta_b, resolved_b = _apply(state, health, ev_b)
    out_b = CounterfactualOutcome(
        outcome_id=f"cf-{candidate_id}-resolve-b",
        label=f"resolve_toward_second ({b})",
        likelihood_weight=round(norm_weights[1], 4),
        resulting_top_hypothesis=new_state_b.top_hypothesis,
        resulting_health_status=new_health_b.status,
        resulting_health_score=round(new_health_b.score, 3),
        ambiguity_pairs=tuple(new_health_b.ambiguity_pairs),
        health_score_delta=delta_b,
        ambiguity_resolved=resolved_b,
        evidence_trace=tuple(new_state_b.explanation),
        reason=(
            f"hypothetical evidence supports {b} and contradicts {a} at "
            f"confidence {confidence:.2f}"
        ),
    )

    # inconclusive — empty supports/contradicts, low confidence
    ev_none = _make_evidence(
        state=state, candidate_id=candidate_id, outcome_tag="inconclusive",
        supports=(), contradicts=(), confidence=0.05,
        reason=f"counterfactual: {candidate_id} returns inconclusive evidence",
    )
    new_state_n, new_health_n, delta_n, resolved_n = _apply(state, health, ev_none)
    out_n = CounterfactualOutcome(
        outcome_id=f"cf-{candidate_id}-inconclusive",
        label="inconclusive",
        likelihood_weight=round(norm_weights[2], 4),
        resulting_top_hypothesis=new_state_n.top_hypothesis,
        resulting_health_status=new_health_n.status,
        resulting_health_score=round(new_health_n.score, 3),
        ambiguity_pairs=tuple(new_health_n.ambiguity_pairs),
        health_score_delta=delta_n,
        ambiguity_resolved=resolved_n,
        evidence_trace=tuple(new_state_n.explanation),
        reason="hypothetical no-op evidence preserves ambiguity",
    )

    return (out_a, out_b, out_n)


def _build_non_ambiguous_outcome(
    state: HypothesisState,
    health: HypothesisCustodyHealth,
    candidate_id: str,
) -> tuple[CounterfactualOutcome, ...]:
    """Single cautious 'no-op' outcome when custody isn't AMBIGUOUS."""
    ev = _make_evidence(
        state=state, candidate_id=candidate_id, outcome_tag="non-ambiguous",
        supports=(), contradicts=(), confidence=0.05,
        reason=(
            f"counterfactual: {candidate_id} evaluated under non-ambiguous "
            f"custody ({health.status.value})"
        ),
    )
    new_state, new_health, delta, resolved = _apply(state, health, ev)
    return (CounterfactualOutcome(
        outcome_id=f"cf-{candidate_id}-non-ambiguous",
        label=f"non-ambiguous ({health.status.value})",
        likelihood_weight=1.0,
        resulting_top_hypothesis=new_state.top_hypothesis,
        resulting_health_status=new_health.status,
        resulting_health_score=round(new_health.score, 3),
        ambiguity_pairs=tuple(new_health.ambiguity_pairs),
        health_score_delta=delta,
        ambiguity_resolved=resolved,
        evidence_trace=tuple(new_state.explanation),
        reason=(
            "counterfactual evaluation is most useful when custody is "
            "AMBIGUOUS; cautious low-impact assessment under "
            f"{health.status.value}"
        ),
    ),)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _expected_metrics(
    outcomes: tuple[CounterfactualOutcome, ...],
) -> tuple[float, float]:
    """Return (expected_health_score_delta, expected_ambiguity_resolution)."""
    expected_delta = sum(o.health_score_delta * o.likelihood_weight for o in outcomes)
    expected_resolution = sum(
        o.likelihood_weight for o in outcomes if o.ambiguity_resolved
    )
    return round(expected_delta, 4), round(expected_resolution, 4)


def _build_assessment(
    candidate_id: str,
    candidate_label: str,
    candidate_score: float,
    health: HypothesisCustodyHealth,
    primary_ambiguity: tuple[str, str] | None,
    outcomes: tuple[CounterfactualOutcome, ...],
    mission_value_report: MissionValueReport | None,
) -> CounterfactualCollectAssessment:
    expected_delta, expected_resolution = _expected_metrics(outcomes)
    summary = (
        f"{candidate_id}: expected ambiguity resolution "
        f"{expected_resolution:.2f}, expected health delta "
        f"{expected_delta:+.2f}"
    )
    return CounterfactualCollectAssessment(
        candidate_id=candidate_id,
        candidate_label=candidate_label,
        current_health_status=health.status,
        primary_ambiguity=primary_ambiguity,
        current_mission_value=_mission_value_lookup(
            candidate_id, mission_value_report,
        ),
        expected_health_score_delta=expected_delta,
        expected_ambiguity_resolution=expected_resolution,
        outcomes=outcomes,
        summary=summary,
    )


def _sort_key(a: CounterfactualCollectAssessment) -> tuple[float, float, str]:
    return (
        -a.expected_ambiguity_resolution,
        -a.expected_health_score_delta,
        a.candidate_id,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def simulate_counterfactual_collects(
    state: HypothesisState,
    health: HypothesisCustodyHealth,
    recommendation: CollectionRecommendation,
    *,
    mission_value_report: MissionValueReport | None = None,
    scenario_id: str | None = None,
    max_results: int | None = None,
) -> CounterfactualReport:
    """Simulate counterfactual outcomes for each ranked candidate collect.

    Deterministic and pure; no wall-clock, no randomness.  ``state.timestamp``
    is the only "now" used.

    The returned :class:`CounterfactualReport` ranks candidates by
    expected ambiguity resolution descending, with ties broken by
    expected health-score delta and then by candidate_id.
    """
    resolved_scenario = (
        scenario_id if scenario_id is not None else state.scenario_id
    )

    primary_pair: tuple[str, str] | None = None
    if (
        health.status is CustodyHealthStatus.AMBIGUOUS
        and health.ambiguity_pairs
    ):
        primary_pair = sorted(health.ambiguity_pairs)[0]

    assessments: list[CounterfactualCollectAssessment] = []
    for cv in recommendation.ranked_values:
        if primary_pair is not None:
            outcomes = _build_ambiguous_outcomes(
                state=state,
                health=health,
                candidate_id=cv.candidate.candidate_id,
                candidate_score=cv.score,
                pair=primary_pair,
            )
        else:
            outcomes = _build_non_ambiguous_outcome(
                state=state,
                health=health,
                candidate_id=cv.candidate.candidate_id,
            )
        assessments.append(_build_assessment(
            candidate_id=cv.candidate.candidate_id,
            candidate_label=cv.candidate.label,
            candidate_score=cv.score,
            health=health,
            primary_ambiguity=primary_pair,
            outcomes=outcomes,
            mission_value_report=mission_value_report,
        ))

    assessments.sort(key=_sort_key)
    if max_results is not None:
        assessments = assessments[:max_results]

    if primary_pair is not None and assessments:
        top = assessments[0]
        summary = (
            f"counterfactual simulation under AMBIGUOUS custody ranks "
            f"{top.candidate_id!r} highest at expected ambiguity resolution "
            f"{top.expected_ambiguity_resolution:.2f}; outcome weights are "
            f"heuristic resolution-potential estimates, not calibrated "
            f"probabilities"
        )
    elif primary_pair is None:
        summary = (
            f"counterfactual evaluation under {health.status.value} custody "
            f"is low-impact; the simulation is most useful when custody is "
            f"AMBIGUOUS with a primary ambiguity pair"
        )
    else:
        summary = "no candidates to simulate"

    return CounterfactualReport(
        scenario_id=resolved_scenario,
        primary_ambiguity=primary_pair,
        ranked_assessments=tuple(assessments),
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Text formatter
# ---------------------------------------------------------------------------


def format_counterfactual_text(report: CounterfactualReport) -> str:
    """Human-readable counterfactual summary."""
    parts: list[str] = []
    title = f"COUNTERFACTUAL COLLECT SIMULATION - {report.scenario_id.upper()}"
    parts.append(title + "\n")
    parts.append("=" * len(title) + "\n")

    parts.append("\nPrimary ambiguity\n")
    parts.append("-----------------\n")
    if report.primary_ambiguity is not None:
        a, b = report.primary_ambiguity
        parts.append(f"{a}  vs  {b}\n")
    else:
        parts.append("None surfaced (custody is not AMBIGUOUS)\n")

    parts.append("\nCandidate strategy comparison\n")
    parts.append("-----------------------------\n")
    parts.append(
        f"{'candidate_id':24s}  {'exp_amb_resolution':>18s}  "
        f"{'exp_health_delta':>17s}\n"
    )
    for a in report.ranked_assessments:
        parts.append(
            f"{a.candidate_id:24s}  "
            f"{a.expected_ambiguity_resolution:18.2f}  "
            f"{a.expected_health_score_delta:+17.2f}\n"
        )

    parts.append("\nTop candidate outcome paths\n")
    parts.append("---------------------------\n")
    for a in report.ranked_assessments[:3]:
        parts.append(f"\n{a.candidate_id} ({a.candidate_label}):\n")
        for o in a.outcomes:
            parts.append(
                f"  {o.label:40s}  weight={o.likelihood_weight:.2f}  "
                f"status={o.resulting_health_status.name}  "
                f"delta={o.health_score_delta:+.2f}\n"
            )

    parts.append("\nInterpretation\n")
    parts.append("--------------\n")
    parts.append(
        "Outcome weights are heuristic resolution-potential estimates, "
        "not calibrated probabilities.\n"
    )
    parts.append(
        "Counterfactual simulation is most useful when custody is AMBIGUOUS; "
        "non-ambiguous states emit a single cautious low-impact outcome.\n"
    )
    parts.append(f"\n{report.summary}\n")

    return "".join(parts)
