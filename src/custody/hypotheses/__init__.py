"""Hypothesis layer — uncertainty-to-tasking engine (ADR-0021).

Slice 1 exposes the contract types, scenario registry, deterministic
belief update, and human-readable explain helpers.  This layer sits
alongside (not inside) the seven-layer entity reasoning stack: it answers
"what do we believe is happening in this scenario across competing
explanations?" while ``belief_assessment.FusionAssessment`` continues to
answer "what do we believe about this entity right now?"
"""

from custody.hypotheses.custody_health import (
    CustodyHealthStatus,
    HypothesisCustodyHealth,
    assess_custody_health,
)
from custody.hypotheses.evidence import (
    from_mapping,
    from_match,
    from_observation,
    from_scene,
)
from custody.hypotheses.explain import explain_top, format_state
from custody.hypotheses.registry import (
    SCENARIO_TENNENT,
    SCENARIO_WHITSUN,
    get_hypotheses,
    get_hypothesis_ids,
)
from custody.hypotheses.types import (
    Hypothesis,
    HypothesisEvidence,
    HypothesisState,
)
from custody.hypotheses.update import update_state

__all__ = [
    "CustodyHealthStatus",
    "Hypothesis",
    "HypothesisCustodyHealth",
    "HypothesisEvidence",
    "HypothesisState",
    "SCENARIO_TENNENT",
    "SCENARIO_WHITSUN",
    "assess_custody_health",
    "explain_top",
    "format_state",
    "from_mapping",
    "from_match",
    "from_observation",
    "from_scene",
    "get_hypotheses",
    "get_hypothesis_ids",
    "update_state",
]
