"""Human-readable helpers for :class:`HypothesisState` (ADR-0021).

Keep logic out of this module — it exists so CLI/report surfaces and tests
can consistently render the state without re-inventing formatting.
"""
from __future__ import annotations

from custody.hypotheses.types import HypothesisState


def explain_top(state: HypothesisState) -> str:
    """One-line summary: top hypothesis, score, uncertainty, primary support."""
    if state.top_hypothesis is None:
        return (
            f"scenario={state.scenario_id} top=<none> "
            f"uncertainty={state.uncertainty:.2f}"
        )
    top_score = state.scores.get(state.top_hypothesis, 0.0)
    supporting = state.supporting_evidence.get(state.top_hypothesis, ())
    primary = supporting[0].evidence_id if supporting else "none"
    return (
        f"scenario={state.scenario_id} "
        f"top={state.top_hypothesis} "
        f"score={top_score:.2f} "
        f"uncertainty={state.uncertainty:.2f} "
        f"primary_support={primary}"
    )


def format_state(state: HypothesisState) -> str:
    """Multi-line human-readable summary of the belief state."""
    lines: list[str] = []
    lines.append(f"Scenario: {state.scenario_id}")
    if state.timestamp is not None:
        lines.append(f"As-of:    {state.timestamp.isoformat()}")
    lines.append(f"Top:      {state.top_hypothesis or '<none>'}")
    lines.append(f"Uncertainty: {state.uncertainty:.3f}")
    lines.append("")
    lines.append("Scores:")
    for hid, score in sorted(state.scores.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"  {score:5.3f}  {hid}")
    if state.explanation:
        lines.append("")
        lines.append("Trace:")
        for line in state.explanation:
            lines.append(f"  {line}")
    return "\n".join(lines)
