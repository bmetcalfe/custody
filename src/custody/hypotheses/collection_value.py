"""Collection-value ranking over hypothesis ambiguity (ADR-0021 Slice 5).

Ranks candidate collect *types* by expected hypothesis-disambiguation
value.  It does not select platforms, schedule sensors, or issue tasking
orders; platform selection and access are downstream concerns.

Candidate labels are deliberately sensor-generic:

    ``optical_context``      — opportunistic daylight optical imagery
    ``repeat_sar``           — same-geometry SAR reacquisition
    ``cross_geometry_sar``   — SAR acquisition at a different geometry
    ``higher_resolution_sar``— higher-resolution SAR take
    ``ais_coverage_query``   — AIS/presence coverage + message query
    ``wait_or_monitor``      — defer collection; watch existing state

The module does *not* name Sentinel-1, Sentinel-2, ICEYE, Umbra, or any
specific platform as an implemented collect.  This keeps the decision
layer honest: no implicit claim of Sentinel integration or platform access.

Strategy design
---------------

Deterministic strategy-table lookup, keyed on canonical hypothesis-pair
tuples (``(a, b)`` with ``a < b`` lexicographically).  No learned model,
no cost formula: each cell records score + reason, hand-authored so the
rationale is legible and the tests are stable.

Covered ambiguity pairs (canonical form):

Tennent:
  - (construction_or_reclamation_activity, fixed_reclamation_or_structure)
  - (fixed_reclamation_or_structure, stationary_sar_scatter_or_reef_clutter)
  - (construction_or_reclamation_activity, stationary_sar_scatter_or_reef_clutter)
  - (fixed_reclamation_or_structure, transient_vessel_activity)

Whitsun:
  - (detector_clutter_false_positives, vessel_cluster_activity)
  - (ais_dark_or_poorly_observed_vessels, vessel_cluster_activity)
  - (transient_anchorage_or_fishing_presence, vessel_cluster_activity)
  - (ais_dark_or_poorly_observed_vessels, transient_anchorage_or_fishing_presence)

Uncovered pairs fall through to a uniform low-confidence recommendation
that flags itself as outside the covered strategy table.

Scope guardrails
----------------

- No imports from ``custody.detection.*``, ``custody.ingest.gfw_presence``,
  matcher runtime, Sentinel SDKs, or real-data loaders.
- Recommendations are collect types, not tasking orders.
- Non-ambiguous states (HEALTHY / DEGRADED / STALE / LOST) produce
  generic, cautious recommendations, not scenario-specific collect plans.
"""
from __future__ import annotations

from dataclasses import dataclass

from custody.hypotheses.custody_health import (
    CustodyHealthStatus,
    HypothesisCustodyHealth,
)
from custody.hypotheses.registry import (
    TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
    TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
    TENNENT_STATIONARY_SAR_SCATTER_OR_REEF_CLUTTER,
    TENNENT_TRANSIENT_VESSEL_ACTIVITY,
    WHITSUN_AIS_DARK_OR_POORLY_OBSERVED_VESSELS,
    WHITSUN_DETECTOR_CLUTTER_FALSE_POSITIVES,
    WHITSUN_TRANSIENT_ANCHORAGE_OR_FISHING_PRESENCE,
    WHITSUN_VESSEL_CLUSTER_ACTIVITY,
)
from custody.hypotheses.types import HypothesisState


# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CollectionCandidate:
    """A sensor-generic collect type (not a platform)."""
    candidate_id: str
    label: str
    modality: str
    description: str
    relative_cost: float
    latency_class: str


@dataclass(frozen=True)
class CollectionValue:
    """A scored candidate with its disambiguation rationale."""
    candidate: CollectionCandidate
    score: float
    disambiguates: tuple[str, str] | None
    reason: str
    caveats: tuple[str, ...]


@dataclass(frozen=True)
class CollectionRecommendation:
    """Ranked set of collect-type values for a state + health combination."""
    scenario_id: str
    health_status: CustodyHealthStatus
    primary_ambiguity: tuple[str, str] | None
    ranked_values: tuple[CollectionValue, ...]
    summary: str


# ---------------------------------------------------------------------------
# Candidate catalog
# ---------------------------------------------------------------------------


_CANDIDATES: tuple[CollectionCandidate, ...] = (
    CollectionCandidate(
        candidate_id="repeat_sar",
        label="Repeat SAR acquisition (same geometry)",
        modality="SAR",
        description=(
            "Reacquire the same area with matched acquisition geometry. "
            "Distinguishes persistent returns from transient ones."
        ),
        relative_cost=0.50,
        latency_class="days",
    ),
    CollectionCandidate(
        candidate_id="cross_geometry_sar",
        label="Cross-geometry SAR acquisition",
        modality="SAR",
        description=(
            "Acquire the same area with a different look geometry. "
            "Separates geometry-dependent scatter from true structure."
        ),
        relative_cost=0.70,
        latency_class="days",
    ),
    CollectionCandidate(
        candidate_id="higher_resolution_sar",
        label="Higher-resolution SAR take",
        modality="SAR",
        description=(
            "Finer-resolution SAR acquisition over the target. "
            "Resolves sub-resolution ambiguity between discrete targets and clutter."
        ),
        relative_cost=0.80,
        latency_class="days",
    ),
    CollectionCandidate(
        candidate_id="optical_context",
        label="Opportunistic optical context",
        modality="EO",
        description=(
            "Daylight optical imagery of the target area. "
            "Distinguishes construction activity and man-made features from natural clutter."
        ),
        relative_cost=0.40,
        latency_class="days",
    ),
    CollectionCandidate(
        candidate_id="ais_coverage_query",
        label="AIS coverage and message query",
        modality="AIS",
        description=(
            "Query AIS presence / coverage over the time-window of interest. "
            "Confirms whether absence of AIS is informative."
        ),
        relative_cost=0.10,
        latency_class="hours",
    ),
    CollectionCandidate(
        candidate_id="wait_or_monitor",
        label="Defer collection; watch existing state",
        modality="none",
        description=(
            "No additional collect proposed; continue monitoring current belief. "
            "Appropriate when custody is healthy or when no collect would reduce uncertainty."
        ),
        relative_cost=0.00,
        latency_class="same-pass",
    ),
)


_CANDIDATE_BY_ID: dict[str, CollectionCandidate] = {
    c.candidate_id: c for c in _CANDIDATES
}


# ---------------------------------------------------------------------------
# Strategy table
# ---------------------------------------------------------------------------


# Canonical score bands used below; hand-picked so strict ordering matches
# the dispatch's "highest / high / medium-high / medium / lower / low" hints.
_HIGHEST = 0.85
_HIGH = 0.75
_MEDIUM_HIGH = 0.60
_MEDIUM = 0.45
_LOWER = 0.25
_LOW = 0.15

_FALLBACK_SCORE = 0.10


def _canonical(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


# Per-pair strategy: dict mapping candidate_id -> (score, reason).
# Every cell is hand-authored; absence means the candidate falls below the
# low band for that pair.  The build_* helpers apply the default low score
# + a pair-agnostic reason to any candidate not explicitly scored.
_STRATEGY: dict[tuple[str, str], dict[str, tuple[float, str]]] = {
    # Tennent --------------------------------------------------------------
    _canonical(
        TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
        TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
    ): {
        "optical_context": (
            _HIGHEST,
            "daylight optical best separates visible construction / dredging "
            "signatures from a static structure",
        ),
        "cross_geometry_sar": (
            _HIGH,
            "geometry diversity reveals whether the returns change with look "
            "angle, consistent with active worksite vs completed structure",
        ),
        "repeat_sar": (
            _MEDIUM_HIGH,
            "same-geometry repeat tracks whether the footprint is evolving "
            "scene-to-scene",
        ),
        "ais_coverage_query": (
            _LOW,
            "AIS coverage is weakly informative for structure-vs-activity",
        ),
        "wait_or_monitor": (
            _LOW,
            "waiting does not disambiguate structure vs active construction",
        ),
    },
    _canonical(
        TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
        TENNENT_STATIONARY_SAR_SCATTER_OR_REEF_CLUTTER,
    ): {
        "cross_geometry_sar": (
            _HIGHEST,
            "geometry-dependent scatter collapses under a different look "
            "angle; true structure does not",
        ),
        "optical_context": (
            _HIGH,
            "daylight optical directly confirms whether a built feature "
            "exists or returns are natural reef",
        ),
        "repeat_sar": (
            _MEDIUM,
            "same-geometry repeat shows persistence but cannot separate "
            "structure from stationary clutter",
        ),
        "ais_coverage_query": (
            _LOW,
            "AIS does not help distinguish structure from scatter",
        ),
    },
    _canonical(
        TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
        TENNENT_STATIONARY_SAR_SCATTER_OR_REEF_CLUTTER,
    ): {
        "optical_context": (
            _HIGHEST,
            "daylight optical reveals visible activity indicators; clutter "
            "has none",
        ),
        "cross_geometry_sar": (
            _MEDIUM_HIGH,
            "geometry change degrades clutter but leaves activity signatures",
        ),
        "repeat_sar": (
            _MEDIUM_HIGH,
            "same-geometry repeat tracks scene-to-scene change consistent "
            "with activity, not clutter",
        ),
        "ais_coverage_query": (
            _LOW,
            "AIS does not help distinguish activity from clutter",
        ),
    },
    _canonical(
        TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
        TENNENT_TRANSIENT_VESSEL_ACTIVITY,
    ): {
        "repeat_sar": (
            _HIGH,
            "persistence vs disappearance across a repeat pass is the clean "
            "separator between static structure and transient vessels",
        ),
        "optical_context": (
            _MEDIUM_HIGH,
            "daylight optical confirms whether the feature is a built "
            "structure or a vessel that moved on",
        ),
        "cross_geometry_sar": (
            _MEDIUM,
            "geometry diversity helps but persistence is more informative",
        ),
        "ais_coverage_query": (
            _LOWER,
            "AIS helps only when the vessel is cooperative, which is the "
            "bad case here",
        ),
    },
    # Whitsun --------------------------------------------------------------
    _canonical(
        WHITSUN_DETECTOR_CLUTTER_FALSE_POSITIVES,
        WHITSUN_VESSEL_CLUSTER_ACTIVITY,
    ): {
        "repeat_sar": (
            _HIGH,
            "persistence across a repeat pass separates real vessels from "
            "per-scene clutter",
        ),
        "optical_context": (
            _HIGH,
            "daylight optical directly confirms vessel presence vs clutter",
        ),
        "higher_resolution_sar": (
            _MEDIUM_HIGH,
            "finer resolution resolves vessel shape vs diffuse clutter",
        ),
        "ais_coverage_query": (
            _LOWER,
            "AIS is only weakly diagnostic when dark-vessel behaviour is "
            "expected",
        ),
    },
    _canonical(
        WHITSUN_AIS_DARK_OR_POORLY_OBSERVED_VESSELS,
        WHITSUN_VESSEL_CLUSTER_ACTIVITY,
    ): {
        "ais_coverage_query": (
            _HIGH,
            "AIS coverage / message query is the definitive test for whether "
            "observed vessels are truly AIS-dark or unobserved",
        ),
        "repeat_sar": (
            _MEDIUM_HIGH,
            "persistence across passes distinguishes a coordinated cluster "
            "from a scatter of singletons",
        ),
        "optical_context": (
            _MEDIUM,
            "daylight optical can show cluster configuration when cloud-free",
        ),
        "cross_geometry_sar": (
            _MEDIUM,
            "geometry diversity helps characterise the cluster but is not "
            "the primary separator",
        ),
    },
    _canonical(
        WHITSUN_TRANSIENT_ANCHORAGE_OR_FISHING_PRESENCE,
        WHITSUN_VESSEL_CLUSTER_ACTIVITY,
    ): {
        "repeat_sar": (
            _HIGH,
            "persistence and spatial configuration across a repeat pass "
            "distinguishes a deliberate cluster from ordinary anchorage",
        ),
        "optical_context": (
            _MEDIUM_HIGH,
            "daylight optical can disambiguate fishing vs militia posture "
            "at the cluster scale",
        ),
        "ais_coverage_query": (
            _MEDIUM,
            "AIS can corroborate ordinary fishing presence when coverage "
            "is solid",
        ),
    },
    _canonical(
        WHITSUN_AIS_DARK_OR_POORLY_OBSERVED_VESSELS,
        WHITSUN_TRANSIENT_ANCHORAGE_OR_FISHING_PRESENCE,
    ): {
        "ais_coverage_query": (
            _HIGH,
            "AIS coverage and message query distinguishes truly dark vessels "
            "from ordinary fishing traffic that simply was not captured",
        ),
        "repeat_sar": (
            _MEDIUM,
            "persistence can help characterise the population but does not "
            "answer cooperativity",
        ),
        "optical_context": (
            _MEDIUM,
            "daylight optical provides posture context but does not confirm "
            "AIS cooperativity",
        ),
    },
}


_GENERIC_LOW_REASON = "limited disambiguation value for this ambiguity pair"


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _sort_key(v: CollectionValue) -> tuple[float, str]:
    # Sort by score descending, then candidate_id ascending.
    return (-v.score, v.candidate.candidate_id)


def _build_values_for_pair(
    pair: tuple[str, str] | None,
    strategy: dict[str, tuple[float, str]] | None,
) -> list[CollectionValue]:
    """Score every candidate; fill in low defaults for candidates not in the
    strategy row."""
    values: list[CollectionValue] = []
    strategy = strategy or {}
    for cand in _CANDIDATES:
        if cand.candidate_id in strategy:
            score, reason = strategy[cand.candidate_id]
            values.append(CollectionValue(
                candidate=cand,
                score=_clamp01(score),
                disambiguates=pair,
                reason=reason,
                caveats=(),
            ))
        else:
            values.append(CollectionValue(
                candidate=cand,
                score=_clamp01(_LOW),
                disambiguates=pair,
                reason=_GENERIC_LOW_REASON,
                caveats=(),
            ))
    return values


def _fallback_values(pair: tuple[str, str]) -> list[CollectionValue]:
    """Uniform low-confidence scoring when the ambiguity pair is uncovered."""
    caveat = "outside covered strategy table"
    reason = (
        "insufficient doctrine for this ambiguity pair; treat as "
        "low-confidence recommendation"
    )
    return [
        CollectionValue(
            candidate=cand,
            score=_FALLBACK_SCORE,
            disambiguates=pair,
            reason=reason,
            caveats=(caveat,),
        )
        for cand in _CANDIDATES
    ]


def _healthy_values() -> list[CollectionValue]:
    values: list[CollectionValue] = []
    for cand in _CANDIDATES:
        if cand.candidate_id == "wait_or_monitor":
            values.append(CollectionValue(
                candidate=cand,
                score=_MEDIUM_HIGH,
                disambiguates=None,
                reason=(
                    "custody is sufficiently separated; defer additional "
                    "collection and continue monitoring"
                ),
                caveats=(),
            ))
        else:
            values.append(CollectionValue(
                candidate=cand,
                score=_LOW,
                disambiguates=None,
                reason="low-priority confirmation; healthy state does not require it",
                caveats=(),
            ))
    return values


def _lost_values() -> list[CollectionValue]:
    """Every candidate at low score — no evidence to target."""
    reason = (
        "insufficient evidence to rank targeted collects; any observation "
        "has low disambiguation value at this stage"
    )
    return [
        CollectionValue(
            candidate=cand,
            score=_LOWER if cand.candidate_id != "wait_or_monitor" else _LOW,
            disambiguates=None,
            reason=reason,
            caveats=("no useful evidence yet",),
        )
        for cand in _CANDIDATES
    ]


def _stale_values() -> list[CollectionValue]:
    values: list[CollectionValue] = []
    for cand in _CANDIDATES:
        if cand.candidate_id == "repeat_sar":
            values.append(CollectionValue(
                candidate=cand,
                score=_HIGH,
                disambiguates=None,
                reason=(
                    "latest evidence is stale; a repeat SAR take is the "
                    "generic reacquisition path"
                ),
                caveats=("generic reacquisition; not scenario-tuned",),
            ))
        elif cand.candidate_id == "optical_context":
            values.append(CollectionValue(
                candidate=cand,
                score=_MEDIUM_HIGH,
                disambiguates=None,
                reason="daylight optical provides generic confirmation of current state",
                caveats=("generic confirmation; not scenario-tuned",),
            ))
        elif cand.candidate_id == "ais_coverage_query":
            values.append(CollectionValue(
                candidate=cand,
                score=_MEDIUM,
                disambiguates=None,
                reason="AIS query is cheap context for reacquisition",
                caveats=(),
            ))
        else:
            values.append(CollectionValue(
                candidate=cand,
                score=_LOW,
                disambiguates=None,
                reason="not the primary reacquisition path",
                caveats=(),
            ))
    return values


def _degraded_values() -> list[CollectionValue]:
    values: list[CollectionValue] = []
    for cand in _CANDIDATES:
        if cand.candidate_id == "repeat_sar":
            values.append(CollectionValue(
                candidate=cand,
                score=_MEDIUM_HIGH,
                disambiguates=None,
                reason=(
                    "custody is degraded but not ambiguous; a generic "
                    "confirmation collect is warranted"
                ),
                caveats=("generic confirmation; not scenario-tuned",),
            ))
        elif cand.candidate_id == "optical_context":
            values.append(CollectionValue(
                candidate=cand,
                score=_MEDIUM,
                disambiguates=None,
                reason="daylight optical provides cautious confirmation",
                caveats=("cautious; confidence below healthy thresholds",),
            ))
        else:
            values.append(CollectionValue(
                candidate=cand,
                score=_LOW,
                disambiguates=None,
                reason="not the primary confirmation path for a degraded state",
                caveats=(),
            ))
    return values


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def rank_collection_candidates(
    state: HypothesisState,
    health: HypothesisCustodyHealth,
    *,
    scenario_id: str | None = None,
    max_results: int | None = None,
) -> CollectionRecommendation:
    """Rank candidate collect types by expected disambiguation value.

    ``scenario_id`` defaults to ``state.scenario_id``.  Ranking is fully
    deterministic: strategy-table lookup + hard-coded branches per
    non-ambiguous status.  Ties break by ``candidate_id`` ascending.
    """
    resolved_scenario = scenario_id if scenario_id is not None else state.scenario_id

    primary_pair: tuple[str, str] | None = None
    summary: str

    if health.status is CustodyHealthStatus.AMBIGUOUS:
        if health.ambiguity_pairs:
            primary_pair = sorted(health.ambiguity_pairs)[0]
            strategy = _STRATEGY.get(primary_pair)
            if strategy is not None:
                values = _build_values_for_pair(primary_pair, strategy)
                summary = (
                    f"ambiguity between {primary_pair[0]} and {primary_pair[1]}; "
                    f"ranked collects prioritise disambiguation value"
                )
            else:
                values = _fallback_values(primary_pair)
                summary = (
                    f"ambiguity pair ({primary_pair[0]}, {primary_pair[1]}) is "
                    "outside the covered strategy table; treat recommendations "
                    "as low-confidence"
                )
        else:
            # Defensive: AMBIGUOUS with no pair (should not occur per Slice 4).
            primary_pair = None
            values = _build_values_for_pair(None, None)
            summary = (
                "ambiguous custody with no canonical ambiguity pair reported; "
                "recommendations degrade to generic low-value entries"
            )

    elif health.status is CustodyHealthStatus.HEALTHY:
        values = _healthy_values()
        summary = (
            "custody appears sufficiently separated; recommend wait_or_monitor "
            "and defer additional collection"
        )

    elif health.status is CustodyHealthStatus.LOST:
        values = _lost_values()
        summary = (
            "insufficient evidence to rank targeted collects; any observation "
            "has low disambiguation value at this stage"
        )

    elif health.status is CustodyHealthStatus.STALE:
        values = _stale_values()
        summary = (
            "evidence is stale; recommend generic confirmation / reacquisition "
            "rather than scenario-specific collects"
        )

    else:  # DEGRADED
        values = _degraded_values()
        summary = (
            "custody degraded; generic confirmation candidates only, with "
            "cautious reasons"
        )

    # Sort: score descending, candidate_id ascending.
    sorted_values = sorted(values, key=_sort_key)
    if max_results is not None:
        sorted_values = sorted_values[:max_results]

    return CollectionRecommendation(
        scenario_id=resolved_scenario,
        health_status=health.status,
        primary_ambiguity=primary_pair,
        ranked_values=tuple(sorted_values),
        summary=summary,
    )
