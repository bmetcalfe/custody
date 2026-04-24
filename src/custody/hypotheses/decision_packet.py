"""Decision packet — first-class product artifact (ADR-0021 Slice 8).

Turns the composed output of Slices 3-5 into a structured artifact that
multiple downstream consumers can serialize: human-readable text,
machine-readable JSON (closed schema, versioned), and Markdown for
notebooks / interview prep.

Scope guardrails:
- Pure stdlib; no new runtime dependencies.
- No wall-clock: ``generated_at`` derives from ``state.timestamp`` so the
  output is deterministic.
- Text output is byte-identical to the Slice 6 prose format — the Slice 8
  refactor is structural, not cosmetic.
- JSON schema is closed for version ``"1"``.  Display-only fields on the
  dataclass are intentionally excluded from the JSON surface.
- Markdown is ASCII only (no em-dash).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime

from custody.hypotheses.collection_value import CollectionRecommendation
from custody.hypotheses.custody_health import HypothesisCustodyHealth
from custody.hypotheses.types import HypothesisState


SCHEMA_VERSION = "1"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BeliefEntry:
    """One entry in the top-3 current-belief slice.

    Scores are rounded to three decimals at construction so display and
    JSON serialization stay stable.
    """
    hypothesis_id: str
    score: float


@dataclass(frozen=True)
class CollectRecommendation:
    """Flat view of one ranked candidate collect, suitable for serialization."""
    candidate_id: str
    label: str
    score: float
    reason: str
    caveats: tuple[str, ...]


@dataclass(frozen=True)
class DecisionPacket:
    """Composed ADR-0021 product artifact.

    The dataclass is deliberately wider than any single serialization
    surface.  ``why_these_collects`` and ``custody_reason`` are display-
    oriented fields used by the text / markdown formatters; the JSON
    schema omits ``why_these_collects`` because the summary is not part
    of the structured interchange contract.
    """
    schema_version: str
    scenario_id: str
    generated_at: datetime | None
    current_belief: tuple[BeliefEntry, ...]
    custody_status: str
    custody_score: float
    custody_top_hypothesis: str | None
    custody_top_score: float
    custody_second_hypothesis: str | None
    custody_second_score: float
    custody_top_two_margin: float | None
    custody_drivers: tuple[str, ...]
    custody_reason: str
    latest_evidence_at: datetime | None
    ambiguity_pairs: tuple[tuple[str, str], ...]
    recommended_collects: tuple[CollectRecommendation, ...]
    why_these_collects: str
    do_not_yet: tuple[str, ...]


# ---------------------------------------------------------------------------
# Editorial — scenario-conditional 'what not to do yet' bullets
# ---------------------------------------------------------------------------


_VLM_TUNING_CAUTION = (
    "Additional VLM tuning has low expected value - current uncertainty is "
    "hypothesis-level, not detector-confidence-level."
)
_WHITSUN_AIS_CAVEAT = (
    "AIS absence alone is not proof of dark vessel activity; only meaningful "
    "when coverage is known."
)
_SENTINEL_ROADMAP = (
    "Sentinel-1/2 ingestion is roadmap, not part of this demo."
)

_SCENARIO_TENNENT = "tennent"
_SCENARIO_WHITSUN = "whitsun"


def what_not_to_do(
    scenario_id: str,
    health: HypothesisCustodyHealth,
) -> tuple[str, ...]:
    """Scenario-conditional editorial; at most 3 bullets."""
    bullets: list[str] = []
    if health.status.value == "ambiguous":
        bullets.append(_VLM_TUNING_CAUTION)
    if scenario_id == _SCENARIO_WHITSUN:
        bullets.append(_WHITSUN_AIS_CAVEAT)
    bullets.append(_SENTINEL_ROADMAP)
    return tuple(bullets[:3])


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def _round3(x: float) -> float:
    return round(float(x), 3)


def _round3_or_none(x: float | None) -> float | None:
    return _round3(x) if x is not None else None


def build_decision_packet(
    *,
    scenario_id: str,
    state: HypothesisState,
    health: HypothesisCustodyHealth,
    recommendation: CollectionRecommendation,
) -> DecisionPacket:
    """Compose the DecisionPacket from the Slice 3/4/5 outputs.

    Pure function; no I/O, no wall-clock.  ``generated_at`` is set to
    ``state.timestamp`` so two calls against the same state produce
    identical output.
    """
    belief_items = sorted(state.scores.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
    current_belief = tuple(
        BeliefEntry(hypothesis_id=hid, score=_round3(score))
        for hid, score in belief_items
    )
    recommended = tuple(
        CollectRecommendation(
            candidate_id=v.candidate.candidate_id,
            label=v.candidate.label,
            score=_round3(v.score),
            reason=v.reason,
            caveats=tuple(v.caveats),
        )
        for v in recommendation.ranked_values
    )
    return DecisionPacket(
        schema_version=SCHEMA_VERSION,
        scenario_id=scenario_id,
        generated_at=state.timestamp,
        current_belief=current_belief,
        custody_status=health.status.value,
        custody_score=_round3(health.score),
        custody_top_hypothesis=health.top_hypothesis,
        custody_top_score=_round3(health.top_score),
        custody_second_hypothesis=health.second_hypothesis,
        custody_second_score=_round3(health.second_score),
        custody_top_two_margin=_round3_or_none(health.top_two_margin),
        custody_drivers=tuple(health.drivers),
        custody_reason=health.reason,
        latest_evidence_at=health.latest_evidence_at,
        ambiguity_pairs=tuple(health.ambiguity_pairs),
        recommended_collects=recommended,
        why_these_collects=recommendation.summary,
        do_not_yet=what_not_to_do(scenario_id, health),
    )


# ---------------------------------------------------------------------------
# Text formatter — byte-identical to Slice 6
# ---------------------------------------------------------------------------


def _hr(title: str) -> str:
    underline = "-" * max(len(title), 3)
    return f"\n{title}\n{underline}\n"


def format_as_text(packet: DecisionPacket) -> str:
    parts: list[str] = []
    parts.append(f"CUSTODY DECISION PACKET - {packet.scenario_id.upper()}\n")
    parts.append(
        "=" * (len("CUSTODY DECISION PACKET - ") + len(packet.scenario_id)) + "\n"
    )

    parts.append(_hr("Current belief"))
    parts.append("Top 3 hypotheses:\n")
    for b in packet.current_belief:
        parts.append(f"  {b.score:4.2f}  {b.hypothesis_id}\n")

    parts.append(_hr("Custody health"))
    parts.append(
        f"Status: {packet.custody_status.upper()} (score {packet.custody_score:.2f})\n"
    )
    parts.append(f"Reason: {packet.custody_reason}\n")
    if packet.custody_drivers:
        parts.append("Drivers:\n")
        for d in packet.custody_drivers[:3]:
            parts.append(f"  - {d}\n")

    parts.append(_hr("Primary ambiguity"))
    if packet.ambiguity_pairs:
        a, b = packet.ambiguity_pairs[0]
        parts.append(f"{a}  vs  {b}\n")
    else:
        parts.append("None surfaced by custody-health assessment\n")

    parts.append(_hr("Recommended candidate collects"))
    parts.append("Top 3 candidate collects:\n")
    for r in packet.recommended_collects[:3]:
        parts.append(f"  {r.score:4.2f}  {r.candidate_id:22s}  {r.label}\n")
        parts.append(f"        Reason: {r.reason}\n")
        if r.caveats:
            for c in r.caveats:
                parts.append(f"        Caveat: {c}\n")

    parts.append(_hr("Why these collects"))
    parts.append(packet.why_these_collects + "\n")
    if packet.ambiguity_pairs and packet.recommended_collects:
        lead = packet.recommended_collects[0]
        parts.append(
            f"Leading candidate {lead.candidate_id!r} addresses this ambiguity "
            f"because {lead.reason}\n"
        )

    parts.append(_hr("What not to do yet"))
    for bullet in packet.do_not_yet:
        parts.append(f"- {bullet}\n")

    return "".join(parts)


# ---------------------------------------------------------------------------
# JSON formatter — closed schema, version "1"
# ---------------------------------------------------------------------------


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt is not None else None


def packet_to_json_object(packet: DecisionPacket) -> dict:
    """Build the dict in insertion order so json.dumps preserves the schema.

    Exposed publicly so callers that need to extend the JSON object with
    optional keys (e.g. ``mission_value`` under ``--mission-value``) can
    compose around the base schema without re-implementing the mapping.
    """
    return {
        "schema_version": packet.schema_version,
        "scenario_id": packet.scenario_id,
        "generated_at": _iso(packet.generated_at),
        "current_belief": [
            {"hypothesis_id": b.hypothesis_id, "score": b.score}
            for b in packet.current_belief
        ],
        "custody_health": {
            "status": packet.custody_status,
            "score": packet.custody_score,
            "top_hypothesis": packet.custody_top_hypothesis,
            "top_score": packet.custody_top_score,
            "second_hypothesis": packet.custody_second_hypothesis,
            "second_score": packet.custody_second_score,
            "top_two_margin": packet.custody_top_two_margin,
            "drivers": list(packet.custody_drivers),
            "reason": packet.custody_reason,
            "latest_evidence_at": _iso(packet.latest_evidence_at),
        },
        "ambiguity_pairs": [[a, b] for a, b in packet.ambiguity_pairs],
        "recommended_collects": [
            {
                "candidate_id": r.candidate_id,
                "label": r.label,
                "score": r.score,
                "reason": r.reason,
                "caveats": list(r.caveats),
            }
            for r in packet.recommended_collects
        ],
        "do_not_yet": list(packet.do_not_yet),
    }


def format_as_json(packet: DecisionPacket) -> str:
    """Deterministic JSON with a trailing newline."""
    return json.dumps(packet_to_json_object(packet), indent=2) + "\n"


def format_many_as_json(packets: tuple[DecisionPacket, ...]) -> str:
    """Serialize multiple packets as a single JSON array.

    Used by the CLI for ``--scenario both --format json`` so the output is
    a single valid JSON document rather than two concatenated objects.
    """
    return json.dumps(
        [packet_to_json_object(p) for p in packets], indent=2
    ) + "\n"


# ---------------------------------------------------------------------------
# Markdown formatter — ASCII only
# ---------------------------------------------------------------------------


_SCENARIO_NAMES: dict[str, str] = {
    "tennent": "Tennent Reef",
    "whitsun": "Whitsun Reef",
}


def format_as_markdown(packet: DecisionPacket) -> str:
    scenario_name = _SCENARIO_NAMES.get(packet.scenario_id, packet.scenario_id)
    parts: list[str] = []
    parts.append(f"# Custody Decision Packet - {scenario_name}\n")
    parts.append("\n")

    # Current belief
    parts.append("## Current belief\n")
    parts.append("\n")
    if packet.current_belief:
        max_score = max(b.score for b in packet.current_belief)
        for b in packet.current_belief:
            score_str = f"{b.score:.3f}"
            if b.score == max_score:
                parts.append(f"- **{b.hypothesis_id}** - {score_str}\n")
            else:
                parts.append(f"- {b.hypothesis_id} - {score_str}\n")
    parts.append("\n")

    # Custody health
    parts.append("## Custody health\n")
    parts.append("\n")
    parts.append(
        f"**{packet.custody_status.upper()}** "
        f"({packet.custody_score:.2f}) - {packet.custody_reason}\n"
    )
    if packet.custody_drivers:
        parts.append("\n")
        parts.append("Drivers:\n")
        for d in packet.custody_drivers[:3]:
            parts.append(f"- {d}\n")
    parts.append("\n")

    # Primary ambiguity (only when pairs exist)
    if packet.ambiguity_pairs:
        parts.append("### Primary ambiguity\n")
        parts.append("\n")
        for a, b in packet.ambiguity_pairs:
            parts.append(f"- {a} vs {b}\n")
        parts.append("\n")

    # Recommended candidate collects — top 3
    parts.append("## Recommended candidate collects\n")
    parts.append("\n")
    for i, r in enumerate(packet.recommended_collects[:3], start=1):
        parts.append(f"{i}. **{r.label}** ({r.score:.2f}) - {r.reason}\n")
    parts.append("\n")

    # Decision notes
    parts.append("## Decision notes\n")
    parts.append("\n")
    for bullet in packet.do_not_yet:
        parts.append(f"- {bullet}\n")

    return "".join(parts)
