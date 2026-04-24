"""Custody-health assessment for :class:`HypothesisState` (ADR-0021, Slice 4).

Answers the product-facing question:

    How well do we understand this situation right now?

The answer is a :class:`CustodyHealthStatus` plus rationale
(:class:`HypothesisCustodyHealth`).  The status cascade is strict:

    LOST  >  STALE  >  AMBIGUOUS  >  HEALTHY  >  DEGRADED

Each tier is only considered if the higher-precedence tiers have been
ruled out.  Scoring and drivers are deterministic — no wall-clock,
no randomness.  ``as_of=None`` defaults to ``state.timestamp`` so a
freshly updated synthetic state never reports STALE spuriously.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from itertools import combinations

from custody.hypotheses.registry import get_hypotheses
from custody.hypotheses.types import HypothesisState


_PRIORS_TOLERANCE = 1e-6


class CustodyHealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    AMBIGUOUS = "ambiguous"
    STALE = "stale"
    LOST = "lost"


@dataclass(frozen=True)
class HypothesisCustodyHealth:
    """Product-facing health assessment over a :class:`HypothesisState`.

    ``ambiguity_pairs`` is populated only when ``status`` is AMBIGUOUS.
    ``top_two_margin`` is ``None`` when fewer than two scores exist
    (defensive — registered scenarios always have >= 2 hypotheses).
    ``latest_evidence_at`` mirrors ``state.timestamp``; see note in
    :func:`assess_custody_health`.
    """
    status: CustodyHealthStatus
    score: float
    top_hypothesis: str | None
    top_score: float
    second_hypothesis: str | None
    second_score: float
    top_two_margin: float | None
    ambiguity_pairs: tuple[tuple[str, str], ...]
    latest_evidence_at: datetime | None
    drivers: tuple[str, ...]
    reason: str


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _canonical_pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def _priors_for(state: HypothesisState) -> dict[str, float]:
    return {h.hypothesis_id: h.prior for h in get_hypotheses(state.scenario_id)}


def _is_at_priors(state: HypothesisState) -> bool:
    priors = _priors_for(state)
    for hid, score in state.scores.items():
        if abs(score - priors.get(hid, 0.0)) >= _PRIORS_TOLERANCE:
            return False
    return True


def _top_two(state: HypothesisState) -> tuple[str | None, float, str | None, float, float | None]:
    """Return (top_id, top_score, second_id, second_score, margin).

    Deterministic tie-break uses ``HypothesisState.top_hypothesis`` for the
    top slot and the lowest sorted hypothesis_id among the remaining for
    the second slot.
    """
    if not state.scores:
        return None, 0.0, None, 0.0, None

    top_id = state.top_hypothesis
    if top_id is None:
        # Defensive: pick the top deterministically by sorted id.
        top_id = max(sorted(state.scores), key=lambda h: state.scores[h])
    top_score = state.scores[top_id]

    others = [(h, s) for h, s in state.scores.items() if h != top_id]
    if not others:
        return top_id, top_score, None, 0.0, None
    # Highest score among others, tie-break by sorted id.
    others.sort(key=lambda hs: (-hs[1], hs[0]))
    second_id, second_score = others[0]
    margin = top_score - second_score
    return top_id, top_score, second_id, second_score, margin


def _saturated_ids(state: HypothesisState, threshold: float) -> list[str]:
    return sorted(h for h, s in state.scores.items() if s >= threshold)


def _score_for(
    status: CustodyHealthStatus,
    *,
    top_score: float,
    top_two_margin: float | None,
    ambiguity_margin: float,
    healthy_top_score: float,
) -> float:
    """Deterministic score bounded to the per-status band documented in ADR-0021."""
    if status is CustodyHealthStatus.LOST:
        return 0.0
    if status is CustodyHealthStatus.STALE:
        return 0.20
    if status is CustodyHealthStatus.AMBIGUOUS:
        if top_two_margin is None or ambiguity_margin <= 0.0:
            return 0.40
        frac = _clamp01(top_two_margin / ambiguity_margin)
        return _clamp01(0.35 + 0.25 * frac)
    if status is CustodyHealthStatus.DEGRADED:
        return _clamp01(0.40 + 0.25 * _clamp01(top_score))
    if status is CustodyHealthStatus.HEALTHY:
        # Scale between 0.70 and 1.0 by how far top_score is above threshold.
        room = max(1e-9, 1.0 - healthy_top_score)
        lift = _clamp01((top_score - healthy_top_score) / room)
        return _clamp01(0.70 + 0.30 * lift)
    return 0.0  # pragma: no cover - enum exhaustiveness


def _reason_for(status: CustodyHealthStatus, drivers: tuple[str, ...]) -> str:
    head = status.name
    if not drivers:
        return head
    return f"{head}: {drivers[0]}"


def assess_custody_health(
    state: HypothesisState,
    *,
    as_of: datetime | None = None,
    stale_after_days: int = 30,
    ambiguity_margin: float = 0.15,
    healthy_margin: float = 0.25,
    healthy_top_score: float = 0.65,
    saturation_score: float = 0.95,
) -> HypothesisCustodyHealth:
    """Classify a :class:`HypothesisState` into a custody-health tier.

    ``as_of`` defaults to ``state.timestamp`` (not wall-clock) so
    synthetic timelines are deterministic and newly updated states never
    report STALE spuriously.  The ``latest_evidence_at`` field on the
    returned :class:`HypothesisCustodyHealth` mirrors ``state.timestamp``
    exactly — per ADR-0021 Slice 4 we do not re-scan the supporting /
    contradicting evidence mappings here.

    Cascade: LOST > STALE > AMBIGUOUS > HEALTHY > DEGRADED.
    """
    top_id, top_score, second_id, second_score, margin = _top_two(state)
    saturated = _saturated_ids(state, saturation_score)
    at_priors = _is_at_priors(state)
    effective_as_of = as_of if as_of is not None else state.timestamp
    latest_evidence_at = state.timestamp

    # --- LOST ----------------------------------------------------------------
    if at_priors:
        drivers = ("no useful evidence has moved scores beyond priors",)
        status = CustodyHealthStatus.LOST
        score = _score_for(
            status, top_score=top_score, top_two_margin=margin,
            ambiguity_margin=ambiguity_margin,
            healthy_top_score=healthy_top_score,
        )
        return HypothesisCustodyHealth(
            status=status, score=score,
            top_hypothesis=top_id, top_score=top_score,
            second_hypothesis=second_id, second_score=second_score,
            top_two_margin=margin, ambiguity_pairs=(),
            latest_evidence_at=latest_evidence_at,
            drivers=drivers, reason=_reason_for(status, drivers),
        )

    # --- STALE ---------------------------------------------------------------
    if effective_as_of is not None and latest_evidence_at is not None:
        age = effective_as_of - latest_evidence_at
        if age > timedelta(days=stale_after_days):
            age_days = age.days
            drivers = (
                f"latest evidence {age_days} days old exceeds stale threshold "
                f"{stale_after_days} days",
            )
            status = CustodyHealthStatus.STALE
            score = _score_for(
                status, top_score=top_score, top_two_margin=margin,
                ambiguity_margin=ambiguity_margin,
                healthy_top_score=healthy_top_score,
            )
            return HypothesisCustodyHealth(
                status=status, score=score,
                top_hypothesis=top_id, top_score=top_score,
                second_hypothesis=second_id, second_score=second_score,
                top_two_margin=margin, ambiguity_pairs=(),
                latest_evidence_at=latest_evidence_at,
                drivers=drivers, reason=_reason_for(status, drivers),
            )

    # --- AMBIGUOUS -----------------------------------------------------------
    margin_ambiguous = (margin is not None) and (margin < ambiguity_margin)
    saturation_ambiguous = len(saturated) >= 2
    if margin_ambiguous or saturation_ambiguous:
        drivers_list: list[str] = []
        pairs: set[tuple[str, str]] = set()

        if saturation_ambiguous:
            drivers_list.append(
                f"{len(saturated)} hypotheses saturated at >= {saturation_score:.2f}"
            )
            for a, b in combinations(saturated, 2):
                pairs.add(_canonical_pair(a, b))
        if margin_ambiguous:
            drivers_list.append(
                f"top-two margin {margin:.2f} below ambiguity threshold "
                f"{ambiguity_margin:.2f}"
            )
            if top_id is not None and second_id is not None:
                pairs.add(_canonical_pair(top_id, second_id))

        status = CustodyHealthStatus.AMBIGUOUS
        score = _score_for(
            status, top_score=top_score, top_two_margin=margin,
            ambiguity_margin=ambiguity_margin,
            healthy_top_score=healthy_top_score,
        )
        return HypothesisCustodyHealth(
            status=status, score=score,
            top_hypothesis=top_id, top_score=top_score,
            second_hypothesis=second_id, second_score=second_score,
            top_two_margin=margin,
            ambiguity_pairs=tuple(sorted(pairs)),
            latest_evidence_at=latest_evidence_at,
            drivers=tuple(drivers_list),
            reason=_reason_for(status, tuple(drivers_list)),
        )

    # --- HEALTHY -------------------------------------------------------------
    healthy_conf = top_score >= healthy_top_score
    healthy_sep = (margin is not None) and (margin >= healthy_margin)
    if healthy_conf and healthy_sep:
        drivers = (
            f"top hypothesis confidence {top_score:.2f} above threshold {healthy_top_score:.2f}",
            f"top-two margin {margin:.2f} above healthy margin {healthy_margin:.2f}",
        )
        status = CustodyHealthStatus.HEALTHY
        score = _score_for(
            status, top_score=top_score, top_two_margin=margin,
            ambiguity_margin=ambiguity_margin,
            healthy_top_score=healthy_top_score,
        )
        return HypothesisCustodyHealth(
            status=status, score=score,
            top_hypothesis=top_id, top_score=top_score,
            second_hypothesis=second_id, second_score=second_score,
            top_two_margin=margin, ambiguity_pairs=(),
            latest_evidence_at=latest_evidence_at,
            drivers=drivers, reason=_reason_for(status, drivers),
        )

    # --- DEGRADED (fallback) -------------------------------------------------
    drivers_list = []
    if not healthy_conf:
        drivers_list.append(
            f"top hypothesis confidence {top_score:.2f} below healthy "
            f"threshold {healthy_top_score:.2f}"
        )
    if margin is not None and not healthy_sep:
        drivers_list.append(
            f"top-two margin {margin:.2f} below healthy margin {healthy_margin:.2f}"
        )
    if not drivers_list:
        drivers_list.append("evidence present but below healthy thresholds")
    status = CustodyHealthStatus.DEGRADED
    score = _score_for(
        status, top_score=top_score, top_two_margin=margin,
        ambiguity_margin=ambiguity_margin,
        healthy_top_score=healthy_top_score,
    )
    return HypothesisCustodyHealth(
        status=status, score=score,
        top_hypothesis=top_id, top_score=top_score,
        second_hypothesis=second_id, second_score=second_score,
        top_two_margin=margin, ambiguity_pairs=(),
        latest_evidence_at=latest_evidence_at,
        drivers=tuple(drivers_list),
        reason=_reason_for(status, tuple(drivers_list)),
    )
