"""Mission-value attribution proxy (ADR-0021 Slice 9).

Decomposes *why* a candidate collect is valuable into named, auditable
components.  This is a **proxy model**, not a financial model: there
are no dollar figures, no production cost accounting, and no claim that
mission value corresponds to any monetary quantity.

Value components (positive and negative):

    ambiguity_reduction       — how much this collect reduces the primary
                                hypothesis ambiguity (derived from the
                                strategy-table disambiguation score)
    custody_health_improvement — expected improvement to custody-health
                                status given the current tier
    mission_relevance         — scenario-level mission priority weight
    timeliness                — bonus for fast-turnaround collects when
                                custody is degraded or worse
    cost_penalty              — negative, proportional to relative_cost
    latency_penalty           — negative, scaled by latency_class
    false_positive_risk_penalty — negative, penalises collects where the
                                  primary ambiguity involves clutter /
                                  false-positive hypotheses

The total mission-value proxy is the sum of components, clamped to [0, 1].

Design constraints
------------------

- Deterministic: no wall-clock, no randomness.
- Pure stdlib: no new runtime dependencies.
- No imports from ``custody.detection.*``, ``custody.ingest.gfw_presence``,
  matcher runtime, Sentinel SDKs, or real-data loaders.
- Language: "mission-value proxy", no monetary claims, no dollar figures.
"""
from __future__ import annotations

from dataclasses import dataclass

from custody.hypotheses.collection_value import (
    CollectionRecommendation,
    CollectionValue,
)
from custody.hypotheses.custody_health import (
    CustodyHealthStatus,
    HypothesisCustodyHealth,
)


# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MissionValueComponent:
    """One named contribution to the mission-value proxy score."""
    name: str
    contribution: float
    reason: str


@dataclass(frozen=True)
class MissionValueAssessment:
    """Decomposed mission-value proxy for a single candidate collect."""
    candidate_id: str
    total_value: float
    components: tuple[MissionValueComponent, ...]
    caveats: tuple[str, ...]


@dataclass(frozen=True)
class MissionValueReport:
    """Ranked mission-value assessments for all candidates in a scenario."""
    scenario_id: str
    ranked_assessments: tuple[MissionValueAssessment, ...]
    summary: str


# ---------------------------------------------------------------------------
# Component scoring — deterministic strategy tables
# ---------------------------------------------------------------------------


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


# Latency-class penalty mapping.  Lower latency = smaller penalty.
_LATENCY_PENALTIES: dict[str, float] = {
    "same-pass": 0.00,
    "hours": -0.02,
    "days": -0.06,
}

# Health-status → expected improvement potential.
# AMBIGUOUS has the most room to improve; HEALTHY has none.
_HEALTH_IMPROVEMENT: dict[CustodyHealthStatus, float] = {
    CustodyHealthStatus.LOST: 0.08,
    CustodyHealthStatus.STALE: 0.12,
    CustodyHealthStatus.AMBIGUOUS: 0.20,
    CustodyHealthStatus.DEGRADED: 0.10,
    CustodyHealthStatus.HEALTHY: 0.02,
}

# Timeliness bonus: fast collects get a bonus when custody is degraded
# or worse.  Only non-trivial for status tiers below HEALTHY.
_TIMELINESS_ELIGIBLE: frozenset[CustodyHealthStatus] = frozenset({
    CustodyHealthStatus.LOST,
    CustodyHealthStatus.STALE,
    CustodyHealthStatus.AMBIGUOUS,
    CustodyHealthStatus.DEGRADED,
})

_TIMELINESS_BONUS: dict[str, float] = {
    "same-pass": 0.06,
    "hours": 0.04,
    "days": 0.00,
}

# Hypothesis IDs that indicate clutter / false-positive ambiguity.
# When the primary ambiguity pair includes one of these, the candidate
# incurs a small false-positive-risk penalty if the collect cannot
# directly resolve clutter (i.e. its disambiguation score is low).
_CLUTTER_HYPOTHESIS_IDS: frozenset[str] = frozenset({
    "stationary_sar_scatter_or_reef_clutter",
    "detector_clutter_false_positives",
})


def _involves_clutter(pair: tuple[str, str] | None) -> bool:
    if pair is None:
        return False
    return bool({pair[0], pair[1]} & _CLUTTER_HYPOTHESIS_IDS)


# ---------------------------------------------------------------------------
# Component builders
# ---------------------------------------------------------------------------


def _ambiguity_reduction(cv: CollectionValue) -> MissionValueComponent:
    """Scale the strategy-table disambiguation score into its contribution.

    The raw score from collection_value is in [0, 1]; we scale it into a
    narrower band [0, 0.40] so that ambiguity reduction is the largest
    positive contributor but does not dominate everything.
    """
    contribution = cv.score * 0.40
    return MissionValueComponent(
        name="ambiguity_reduction",
        contribution=round(contribution, 4),
        reason=(
            f"disambiguation score {cv.score:.2f} for candidate "
            f"{cv.candidate.candidate_id!r} scaled to {contribution:.2f}"
        ),
    )


def _custody_health_improvement(
    cv: CollectionValue,
    health: HypothesisCustodyHealth,
) -> MissionValueComponent:
    """Expected custody-health improvement from executing this collect.

    Higher improvement potential when custody is worse, modulated by
    the candidate's disambiguation score (better collects improve health
    more).
    """
    base = _HEALTH_IMPROVEMENT.get(health.status, 0.05)
    contribution = base * cv.score
    return MissionValueComponent(
        name="custody_health_improvement",
        contribution=round(contribution, 4),
        reason=(
            f"health status {health.status.value} has base improvement "
            f"potential {base:.2f}, modulated by disambiguation score "
            f"{cv.score:.2f}"
        ),
    )


def _mission_relevance_component(
    mission_relevance: float,
) -> MissionValueComponent:
    """Scenario-level mission priority, passed in by the caller.

    Scaled into a [0, 0.20] band.
    """
    clamped = _clamp01(mission_relevance)
    contribution = clamped * 0.20
    return MissionValueComponent(
        name="mission_relevance",
        contribution=round(contribution, 4),
        reason=(
            f"mission relevance weight {clamped:.2f} scaled to "
            f"{contribution:.2f}"
        ),
    )


def _timeliness(
    cv: CollectionValue,
    health: HypothesisCustodyHealth,
) -> MissionValueComponent:
    """Timeliness bonus for fast collects when custody needs attention."""
    if health.status in _TIMELINESS_ELIGIBLE:
        bonus = _TIMELINESS_BONUS.get(cv.candidate.latency_class, 0.0)
        reason = (
            f"latency class {cv.candidate.latency_class!r} earns "
            f"timeliness bonus {bonus:+.2f} under {health.status.value} "
            f"custody"
        )
    else:
        bonus = 0.0
        reason = (
            f"no timeliness bonus under {health.status.value} custody"
        )
    return MissionValueComponent(
        name="timeliness",
        contribution=round(bonus, 4),
        reason=reason,
    )


def _cost_penalty(cv: CollectionValue) -> MissionValueComponent:
    """Negative contribution proportional to relative_cost."""
    penalty = -cv.candidate.relative_cost * 0.10
    return MissionValueComponent(
        name="cost_penalty",
        contribution=round(penalty, 4),
        reason=(
            f"relative cost {cv.candidate.relative_cost:.2f} incurs "
            f"penalty {penalty:+.2f}"
        ),
    )


def _latency_penalty(cv: CollectionValue) -> MissionValueComponent:
    """Negative contribution based on latency_class."""
    penalty = _LATENCY_PENALTIES.get(cv.candidate.latency_class, -0.06)
    return MissionValueComponent(
        name="latency_penalty",
        contribution=round(penalty, 4),
        reason=(
            f"latency class {cv.candidate.latency_class!r} incurs "
            f"penalty {penalty:+.2f}"
        ),
    )


def _false_positive_risk_penalty(
    cv: CollectionValue,
    primary_ambiguity: tuple[str, str] | None,
) -> MissionValueComponent:
    """Penalty when the ambiguity pair involves clutter / false-positive
    hypotheses and the candidate has low disambiguation value for it.

    Higher-scoring candidates against clutter pairs get a smaller penalty
    because they are more likely to resolve the clutter question.
    """
    if _involves_clutter(primary_ambiguity):
        # Invert the score: low disambiguation → higher penalty.
        penalty = -(1.0 - cv.score) * 0.08
        reason = (
            f"primary ambiguity involves clutter/false-positive hypothesis; "
            f"candidate scores {cv.score:.2f} against it, penalty {penalty:+.3f}"
        )
    else:
        penalty = 0.0
        reason = "primary ambiguity does not involve clutter/false-positive hypotheses"
    return MissionValueComponent(
        name="false_positive_risk_penalty",
        contribution=round(penalty, 4),
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _sort_key(a: MissionValueAssessment) -> tuple[float, str]:
    return (-a.total_value, a.candidate_id)


def attribute_mission_value(
    recommendation: CollectionRecommendation,
    health: HypothesisCustodyHealth,
    *,
    scenario_id: str | None = None,
    mission_relevance: float = 0.5,
) -> MissionValueReport:
    """Decompose each candidate collect's value into named components.

    Returns a :class:`MissionValueReport` with assessments sorted by
    ``total_value`` descending (ties broken by ``candidate_id`` ascending).

    ``mission_relevance`` is a [0, 1] scenario-level priority weight
    supplied by the caller.  Default 0.5 (neutral).

    This is a **mission-value proxy**, not revenue attribution.
    """
    resolved_scenario = (
        scenario_id if scenario_id is not None
        else recommendation.scenario_id
    )
    primary_ambiguity = recommendation.primary_ambiguity

    assessments: list[MissionValueAssessment] = []
    for cv in recommendation.ranked_values:
        components = (
            _ambiguity_reduction(cv),
            _custody_health_improvement(cv, health),
            _mission_relevance_component(mission_relevance),
            _timeliness(cv, health),
            _cost_penalty(cv),
            _latency_penalty(cv),
            _false_positive_risk_penalty(cv, primary_ambiguity),
        )
        raw_total = sum(c.contribution for c in components)
        total = round(_clamp01(raw_total), 3)

        caveats: list[str] = [
            "mission-value proxy only - no monetary or production-cost claim",
        ]
        if cv.caveats:
            caveats.extend(cv.caveats)

        assessments.append(MissionValueAssessment(
            candidate_id=cv.candidate.candidate_id,
            total_value=total,
            components=components,
            caveats=tuple(caveats),
        ))

    assessments.sort(key=_sort_key)

    # Build summary from top candidate.
    if assessments:
        top = assessments[0]
        top_component = max(top.components, key=lambda c: c.contribution)
        summary = (
            f"mission-value proxy ranks {top.candidate_id!r} highest "
            f"at {top.total_value:.2f}; largest contributor is "
            f"{top_component.name} ({top_component.contribution:+.2f})"
        )
    else:
        summary = "no candidates to assess"

    return MissionValueReport(
        scenario_id=resolved_scenario,
        ranked_assessments=tuple(assessments),
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Text formatter — mirrors decision-packet style
# ---------------------------------------------------------------------------


def format_mission_value_text(report: MissionValueReport) -> str:
    """Human-readable mission-value summary, suitable for appending to
    the decision packet text output."""
    parts: list[str] = []
    parts.append("\nMission-value attribution proxy\n")
    parts.append("-" * 31 + "\n")
    for a in report.ranked_assessments[:3]:
        parts.append(f"\nCandidate: {a.candidate_id}\n")
        parts.append(f"Mission value proxy: {a.total_value:.2f}\n")
        parts.append("Contributions:\n")
        for c in a.components:
            parts.append(f"  {c.name:30s} {c.contribution:+.2f}\n")
    parts.append(f"\n{report.summary}\n")
    return "".join(parts)


def mission_value_to_json_object(report: MissionValueReport) -> dict:
    """Structured dict suitable for embedding in the decision packet JSON.

    Insertion order is fixed so downstream JSON output is deterministic.
    Returned object is plain Python types only (no dataclasses).
    """
    return {
        "scenario_id": report.scenario_id,
        "summary": report.summary,
        "ranked_assessments": [
            {
                "candidate_id": a.candidate_id,
                "total_value": a.total_value,
                "components": [
                    {
                        "name": c.name,
                        "contribution": c.contribution,
                        "reason": c.reason,
                    }
                    for c in a.components
                ],
                "caveats": list(a.caveats),
            }
            for a in report.ranked_assessments
        ],
    }


def format_mission_value_markdown(report: MissionValueReport) -> str:
    """Markdown section suitable for appending to the decision packet md output.

    ASCII only; opens with ``## Mission-value attribution proxy`` so it
    slots under the Slice 8 packet hierarchy.
    """
    parts: list[str] = []
    parts.append("## Mission-value attribution proxy\n")
    parts.append("\n")
    for i, a in enumerate(report.ranked_assessments[:3], start=1):
        parts.append(f"{i}. **{a.candidate_id}** - total {a.total_value:.2f}\n")
        for c in a.components:
            parts.append(f"   - {c.name}: {c.contribution:+.2f}\n")
    parts.append("\n")
    parts.append(f"_{report.summary}_\n")
    return "".join(parts)
