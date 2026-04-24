"""Hypothesis-layer contract types (ADR-0021).

This module defines three frozen dataclasses that form the first-cut
contract for the hypothesis layer:

- :class:`HypothesisEvidence` — a thin annotation over an existing repo
  source object.  ``source_ref`` is a *typed reference string* (e.g.
  ``"observation:obs-42"`` or ``"scene:tennent_20230702_umbra-05"``); the
  referenced object lives in its existing module and is not copied in.
- :class:`Hypothesis` — one scenario-specific competing explanation, with
  a uniform prior unless a defensible weighting is documented.
- :class:`HypothesisState` — the scenario-scoped belief snapshot that
  answers "what do we believe is happening across competing explanations?"

Relationship to existing reasoning stack
----------------------------------------
The seven-layer entity-centric stack (belief_assessment → decision →
taskrecommendation) remains intact.  ``FusionAssessment.uncertainty`` is
entity-level ("how sure about this track?"); ``HypothesisState.uncertainty``
is scenario-level ("how ambiguous is the top hypothesis among competitors?").
The two coexist on different axes and must not be consolidated.  Per
ADR-0021, Slice 1 adds this layer without modifying belief_assessment.py,
decision.py, or taskrecommendation.py.

ADR-0021 lists ``evidence.py`` as a separate file; for Slice 1 the three
tightly-coupled types live here.  The source-object → HypothesisEvidence
adapter helpers belong to Slice 2 and land in ``evidence.py`` then.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping


@dataclass(frozen=True)
class HypothesisEvidence:
    """Thin annotation referring to an existing repo source object.

    ``source_ref`` is a typed reference string, not an inline copy.
    Conventional formats:
      - ``"observation:<obs_id>"``
      - ``"scene:<scene_id>"``
      - ``"vlm_detection:<detection_id>"``
      - ``"gfw_presence:<query_id>"``
      - ``"matcher_result:<pair_id>"``

    ``source_ref`` may be ``None`` for derived or synthetic evidence that
    does not point to a single backing source object.
    """
    evidence_id: str
    source_ref: str | None
    source_kind: str
    scenario_id: str
    timestamp: datetime | None
    supports: tuple[str, ...]
    contradicts: tuple[str, ...]
    confidence: float
    weight: float
    reason: str


@dataclass(frozen=True)
class Hypothesis:
    """One scenario-specific competing explanation."""
    hypothesis_id: str
    label: str
    description: str
    prior: float
    scenario_id: str


@dataclass(frozen=True)
class HypothesisState:
    """Scenario-scoped belief snapshot over competing hypotheses.

    ``scores`` maps ``hypothesis_id → current belief score`` in [0, 1].
    ``supporting_evidence`` and ``contradicting_evidence`` group the
    HypothesisEvidence items that moved each score, keyed by hypothesis_id.
    ``explanation`` is an ordered human-readable trace — one string per
    evidence item — describing its contribution.
    """
    scenario_id: str
    timestamp: datetime | None
    scores: Mapping[str, float]
    top_hypothesis: str | None
    uncertainty: float
    supporting_evidence: Mapping[str, tuple[HypothesisEvidence, ...]] = field(
        default_factory=dict
    )
    contradicting_evidence: Mapping[str, tuple[HypothesisEvidence, ...]] = field(
        default_factory=dict
    )
    explanation: tuple[str, ...] = ()
