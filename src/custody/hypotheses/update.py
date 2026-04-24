"""Deterministic belief update for the hypothesis layer (ADR-0021).

Slice 1 scoring is weighted additive:

  - Start from scenario priors, or from ``prior_state.scores`` when a
    running belief is supplied.
  - For each :class:`HypothesisEvidence`:
      * Add ``confidence * weight`` to every hypothesis in ``supports``.
      * Subtract ``confidence * weight`` from every hypothesis in
        ``contradicts``, floored at 0.
  - Clamp all scores to ``[0, 1]`` (no normalisation in Slice 1).
  - ``top_hypothesis`` = argmax with deterministic tie-break on sorted
    ``hypothesis_id``.
  - ``uncertainty = 1 - top_score`` with a +0.1 ambiguity boost when the
    top-two margin is below 0.1 (clamped to 1).
  - Maintain a human-readable explanation trace, one line per evidence.

Bayesian rigour is intentionally out of scope.  Determinism and
explainability are the contract.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Sequence

from custody.hypotheses.registry import get_hypotheses
from custody.hypotheses.types import HypothesisEvidence, HypothesisState


_MARGIN_THRESHOLD = 0.1
_AMBIGUITY_BOOST = 0.1


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _seed_scores(
    scenario_id: str,
    prior_state: HypothesisState | None,
) -> dict[str, float]:
    hs = get_hypotheses(scenario_id)
    if prior_state is None:
        return {h.hypothesis_id: h.prior for h in hs}
    # Preserve the hypothesis key set for this scenario; fall back to prior
    # for any hypothesis_id not present in the incoming state.
    return {
        h.hypothesis_id: float(prior_state.scores.get(h.hypothesis_id, h.prior))
        for h in hs
    }


def _latest_timestamp(evidence: Sequence[HypothesisEvidence]) -> datetime | None:
    stamps = [ev.timestamp for ev in evidence if ev.timestamp is not None]
    return max(stamps) if stamps else None


def _format_trace_line(
    ev: HypothesisEvidence,
    deltas: list[tuple[str, float]],
) -> str:
    """Render one explanation line for an evidence item.

    ``deltas`` is a list of ``(hypothesis_id, signed_delta)`` recording
    what the evidence contributed.  Signed deltas: positive for supports,
    negative for contradicts.
    """
    parts: list[str] = []
    for hid, delta in deltas:
        sign = "+" if delta >= 0 else "-"
        parts.append(f"{sign}{abs(delta):.2f} {'to' if delta >= 0 else 'from'} {hid}")
    body = "; ".join(parts) if parts else "no-op"
    return f"{ev.evidence_id}: {body} ({ev.reason})"


def update_state(
    scenario_id: str,
    evidence: Sequence[HypothesisEvidence],
    prior_state: HypothesisState | None = None,
) -> HypothesisState:
    """Apply ``evidence`` to the belief state for ``scenario_id``.

    If ``prior_state`` is ``None``, scoring starts from scenario priors.
    If supplied, its scores seed the running belief (unknown hypothesis
    ids fall back to their prior).

    Unknown ``scenario_id`` raises :class:`ValueError` via
    :func:`custody.hypotheses.registry.get_hypotheses`.
    """
    hs = get_hypotheses(scenario_id)
    known_ids = {h.hypothesis_id for h in hs}

    scores = _seed_scores(scenario_id, prior_state)

    supporting: dict[str, list[HypothesisEvidence]] = defaultdict(list)
    contradicting: dict[str, list[HypothesisEvidence]] = defaultdict(list)
    explanation: list[str] = []

    for ev in evidence:
        magnitude = float(ev.confidence) * float(ev.weight)
        # Trace records the intended signed contribution (confidence * weight),
        # not the clamped applied delta — the intended magnitude is what a
        # reader cares about, and it doesn't depend on current score.
        deltas: list[tuple[str, float]] = []

        for hid in ev.supports:
            if hid not in known_ids:
                continue
            scores[hid] = _clamp01(scores[hid] + magnitude)
            deltas.append((hid, +magnitude))
            supporting[hid].append(ev)

        for hid in ev.contradicts:
            if hid not in known_ids:
                continue
            scores[hid] = _clamp01(scores[hid] - magnitude)
            deltas.append((hid, -magnitude))
            contradicting[hid].append(ev)

        explanation.append(_format_trace_line(ev, deltas))

    # Deterministic top selection: max score, tie-break by sorted hypothesis_id.
    ordered_ids = sorted(scores)
    top_hypothesis = max(ordered_ids, key=lambda hid: scores[hid])
    top_score = scores[top_hypothesis]

    # Uncertainty with ambiguity boost when the top-two margin is tight.
    score_values = sorted(scores.values(), reverse=True)
    second_score = score_values[1] if len(score_values) > 1 else 0.0
    uncertainty = 1.0 - top_score
    if (top_score - second_score) < _MARGIN_THRESHOLD:
        uncertainty = min(1.0, uncertainty + _AMBIGUITY_BOOST)

    return HypothesisState(
        scenario_id=scenario_id,
        timestamp=_latest_timestamp(evidence),
        scores=dict(scores),
        top_hypothesis=top_hypothesis,
        uncertainty=uncertainty,
        supporting_evidence={k: tuple(v) for k, v in supporting.items()},
        contradicting_evidence={k: tuple(v) for k, v in contradicting.items()},
        explanation=tuple(explanation),
    )
