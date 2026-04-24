"""Scenario-specific evidence generators (ADR-0021 Slice 3).

This module is the first place in the hypothesis layer where domain
heuristics live.  Upstream layers stay neutral: Slice 1 types are
generic, Slice 2 adapters never decide what an artifact *means*.  Here
we translate explicit scenario signals (caller-supplied flags) into
:class:`HypothesisEvidence`, routed via the Slice 2 adapters so every
item is validated against the scenario registry.

Heuristic catalog (Slice 3)
---------------------------

Each row below encodes one signal → (supports, contradicts) mapping.
Generators implement these rows literally; tests pin the mapping.

======== ============================================================ ========================================== ===================================== ================================================================================= ==============================
Scenario Signal                                                       Supports                                   Contradicts                           Rationale                                                                         Reference
======== ============================================================ ========================================== ===================================== ================================================================================= ==============================
Tennent  persistent bright SAR scatter near reef footprint            fixed_reclamation_or_structure             no_meaningful_activity                persistent scene signal near reef is more consistent with structure/reclamation   scenario.md / AMTI context
                                                                                                                                                       than no activity
Tennent  scene-to-scene change signal (>= 0.25)                       construction_or_reclamation_activity       stationary_sar_scatter_or_reef_clutter change over time suggests activity, not static scatter                              scenario.md
Tennent  low-displacement match (< 10 m)                              fixed_reclamation_or_structure             transient_vessel_activity             persistent low-motion return consistent with static feature                       scenario.md
Tennent  moderate-displacement match (10 – 40 m)                      construction_or_reclamation_activity       stationary_sar_scatter_or_reef_clutter slow evolution of reef footprint consistent with active reclamation                scenario.md
Tennent  high-displacement match within gate (>= 40 m)                transient_vessel_activity                  fixed_reclamation_or_structure        mover within gate is vessel-like, not structure                                   scenario.md
Whitsun  multiple vessel-like candidates / cluster signal (>= 3)      vessel_cluster_activity                    no_persistent_activity                cluster-like maritime activity is the main Whitsun narrative                      scenario.md / AMTI context
Whitsun  few vessel-like candidates (1 or 2)                          transient_anchorage_or_fishing_presence    no_persistent_activity                scattered returns consistent with ordinary anchorage / fishing                    scenario.md
Whitsun  weak/absent AIS under uncertain coverage, vessels present    ais_dark_or_poorly_observed_vessels        (none)                                AIS absence is suggestive only when coverage is sparse/none and vessels are        scenario.md
                                                                                                                                                       actually visible; do not overclaim dark vessels
Whitsun  solid AIS coverage with zero vessel returns                  no_persistent_activity                     vessel_cluster_activity               absence is informative only when coverage is solid                                scenario.md
Whitsun  matched vessel in persistent cluster pattern                 vessel_cluster_activity                    detector_clutter_false_positives      cluster persistence across scenes is the Whitsun militia-swarm signature          scenario.md / AMTI
Whitsun  isolated match, no cluster context                           transient_anchorage_or_fishing_presence    vessel_cluster_activity               scattered returns consistent with ordinary anchorage / fishing                    scenario.md
======== ============================================================ ========================================== ===================================== ================================================================================= ==============================

Scope and guardrails
--------------------

- Generators build evidence through the Slice 2 adapters (``from_scene``
  / ``from_match``); they never construct :class:`HypothesisEvidence`
  directly.  This guarantees registry validation at construction time.
- No I/O, no wall-clock, no randomness.
- No imports from ``custody.detection.*`` or ``custody.ingest.gfw_presence``.
- ``quality_flag`` controls confidence via :func:`_quality_to_confidence`;
  the Slice 2 "no quality→confidence mapping" invariant holds at the
  *adapter* layer — this module owns the mapping.
- Whitsun never claims dark vessels under ``ais_coverage_flag="solid"``
  with zero vessel returns.
"""
from __future__ import annotations

from typing import Literal

from custody.fusion.scenes import Scene
from custody.fusion.temporal import Match
from custody.hypotheses.evidence import from_match, from_scene
from custody.hypotheses.registry import (
    SCENARIO_TENNENT,
    SCENARIO_WHITSUN,
    TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
    TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
    TENNENT_NO_MEANINGFUL_ACTIVITY,
    TENNENT_STATIONARY_SAR_SCATTER_OR_REEF_CLUTTER,
    TENNENT_TRANSIENT_VESSEL_ACTIVITY,
    WHITSUN_AIS_DARK_OR_POORLY_OBSERVED_VESSELS,
    WHITSUN_DETECTOR_CLUTTER_FALSE_POSITIVES,
    WHITSUN_NO_PERSISTENT_ACTIVITY,
    WHITSUN_TRANSIENT_ANCHORAGE_OR_FISHING_PRESENCE,
    WHITSUN_VESSEL_CLUSTER_ACTIVITY,
)
from custody.hypotheses.types import HypothesisEvidence


AisCoverage = Literal["solid", "sparse", "none"]


_QUALITY_CONFIDENCE: dict[str, float] = {
    "green": 1.0,
    "yellow": 0.6,
    "red": 0.2,
}

_VALID_AIS_COVERAGE: tuple[str, ...] = ("solid", "sparse", "none")

_CHANGE_THRESHOLD = 0.25
_MATCH_LOW_GATE_M = 10.0
_MATCH_MODERATE_GATE_M = 40.0


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _quality_to_confidence(quality_flag: str) -> float:
    """Map a Scene/observation quality_flag to a confidence multiplier.

    ``green`` → 1.0, ``yellow`` → 0.6, ``red`` → 0.2.  Unknown flags raise
    :class:`ValueError` listing the valid values.
    """
    try:
        return _QUALITY_CONFIDENCE[quality_flag]
    except KeyError:
        valid = ", ".join(sorted(_QUALITY_CONFIDENCE))
        raise ValueError(
            f"unknown quality_flag {quality_flag!r}; valid values: {valid}"
        ) from None


def _ais_coverage_signal(flag: str) -> str:
    """Validate an AIS coverage flag and return it unchanged.

    Accepted: ``"solid"``, ``"sparse"``, ``"none"``.  Unknown flags raise
    :class:`ValueError` listing the valid values.  Returning the flag as-is
    lets callers use this as a validation gate inline.
    """
    if flag not in _VALID_AIS_COVERAGE:
        valid = ", ".join(_VALID_AIS_COVERAGE)
        raise ValueError(
            f"unknown ais_coverage_flag {flag!r}; valid values: {valid}"
        )
    return flag


# ---------------------------------------------------------------------------
# Tennent generators
# ---------------------------------------------------------------------------


def tennent_evidence_from_scene(
    scene: Scene,
    *,
    persistent_scatterer_detected: bool,
    change_signal_strength: float = 0.0,
    quality_flag: str = "yellow",
) -> tuple[HypothesisEvidence, ...]:
    """Emit Tennent scene-level evidence from explicit signals.

    Each signal triggers a separate :class:`HypothesisEvidence`; both may
    fire simultaneously.
    """
    confidence = _quality_to_confidence(quality_flag)
    evs: list[HypothesisEvidence] = []

    if persistent_scatterer_detected:
        evs.append(from_scene(
            scene,
            scenario_id=SCENARIO_TENNENT,
            supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,),
            contradicts=(TENNENT_NO_MEANINGFUL_ACTIVITY,),
            confidence=confidence,
            reason="persistent bright scatter near reef footprint",
            evidence_id=f"tennent-scene-{scene.scene_id}-persistent",
        ))

    if change_signal_strength >= _CHANGE_THRESHOLD:
        evs.append(from_scene(
            scene,
            scenario_id=SCENARIO_TENNENT,
            supports=(TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,),
            contradicts=(TENNENT_STATIONARY_SAR_SCATTER_OR_REEF_CLUTTER,),
            confidence=confidence,
            reason=(
                f"scene-to-scene change signal {change_signal_strength:.2f} "
                f">= {_CHANGE_THRESHOLD:.2f}"
            ),
            evidence_id=f"tennent-scene-{scene.scene_id}-change",
        ))

    return tuple(evs)


def tennent_evidence_from_match(
    match: Match,
    *,
    displacement_m: float,
    quality_flag: str = "yellow",
) -> HypothesisEvidence:
    """Emit Tennent match-level evidence based on displacement band.

    Bands:
      ``displacement_m < 10``             → supports FIXED,        contradicts TRANSIENT
      ``10 <= displacement_m < 40``       → supports CONSTRUCTION, contradicts STATIONARY_SCATTER
      ``displacement_m >= 40``            → supports TRANSIENT,    contradicts FIXED
    """
    confidence = _quality_to_confidence(quality_flag)

    if displacement_m < _MATCH_LOW_GATE_M:
        supports = (TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,)
        contradicts = (TENNENT_TRANSIENT_VESSEL_ACTIVITY,)
        band = "low"
        reason = f"low-displacement match ({displacement_m:.1f} m) consistent with static feature"
    elif displacement_m < _MATCH_MODERATE_GATE_M:
        supports = (TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,)
        contradicts = (TENNENT_STATIONARY_SAR_SCATTER_OR_REEF_CLUTTER,)
        band = "moderate"
        reason = (
            f"moderate-displacement match ({displacement_m:.1f} m) "
            f"consistent with active reclamation"
        )
    else:
        supports = (TENNENT_TRANSIENT_VESSEL_ACTIVITY,)
        contradicts = (TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,)
        band = "high"
        reason = (
            f"high-displacement match ({displacement_m:.1f} m) within gate "
            f"is vessel-like, not structure"
        )

    return from_match(
        match,
        scenario_id=SCENARIO_TENNENT,
        supports=supports,
        contradicts=contradicts,
        confidence=confidence,
        reason=reason,
        evidence_id=f"tennent-match-{match.obs_a_id}-{match.obs_b_id}-{band}",
    )


# ---------------------------------------------------------------------------
# Whitsun generators
# ---------------------------------------------------------------------------


def whitsun_evidence_from_scene(
    scene: Scene,
    *,
    vessel_count: int,
    ais_coverage_flag: AisCoverage,
    quality_flag: str = "yellow",
) -> tuple[HypothesisEvidence, ...]:
    """Emit Whitsun scene-level evidence from explicit signals.

    - ``vessel_count >= 3`` → cluster evidence.
    - ``1 <= vessel_count <= 2`` → transient-anchorage evidence.
    - ``vessel_count > 0`` under ``sparse``/``none`` coverage → additional
      AIS-dark evidence.  Never emitted under ``solid`` coverage.
    - ``vessel_count == 0`` under ``solid`` coverage → no-persistent-activity
      evidence.  Zero vessels under ``sparse``/``none`` is uninformative —
      no evidence emitted.
    """
    coverage = _ais_coverage_signal(ais_coverage_flag)
    confidence = _quality_to_confidence(quality_flag)
    evs: list[HypothesisEvidence] = []

    if vessel_count >= 3:
        evs.append(from_scene(
            scene,
            scenario_id=SCENARIO_WHITSUN,
            supports=(WHITSUN_VESSEL_CLUSTER_ACTIVITY,),
            contradicts=(WHITSUN_NO_PERSISTENT_ACTIVITY,),
            confidence=confidence,
            reason=f"vessel cluster signal ({vessel_count} candidates) at reef",
            evidence_id=f"whitsun-scene-{scene.scene_id}-cluster",
        ))
    elif vessel_count in (1, 2):
        evs.append(from_scene(
            scene,
            scenario_id=SCENARIO_WHITSUN,
            supports=(WHITSUN_TRANSIENT_ANCHORAGE_OR_FISHING_PRESENCE,),
            contradicts=(WHITSUN_NO_PERSISTENT_ACTIVITY,),
            confidence=confidence,
            reason=(
                f"{vessel_count} vessel-like candidate(s) - scattered returns "
                f"consistent with ordinary anchorage / fishing"
            ),
            evidence_id=f"whitsun-scene-{scene.scene_id}-transient",
        ))

    if coverage in ("sparse", "none") and vessel_count > 0:
        evs.append(from_scene(
            scene,
            scenario_id=SCENARIO_WHITSUN,
            supports=(WHITSUN_AIS_DARK_OR_POORLY_OBSERVED_VESSELS,),
            contradicts=(),
            confidence=confidence,
            reason=(
                f"vessels visible with {coverage} AIS coverage - "
                f"consistent with dark or poorly-observed vessels"
            ),
            evidence_id=f"whitsun-scene-{scene.scene_id}-ais-dark",
        ))

    if coverage == "solid" and vessel_count == 0:
        evs.append(from_scene(
            scene,
            scenario_id=SCENARIO_WHITSUN,
            supports=(WHITSUN_NO_PERSISTENT_ACTIVITY,),
            contradicts=(WHITSUN_VESSEL_CLUSTER_ACTIVITY,),
            confidence=confidence,
            reason="zero vessel returns under solid AIS coverage - absence is informative",
            evidence_id=f"whitsun-scene-{scene.scene_id}-no-persistent",
        ))

    return tuple(evs)


def whitsun_evidence_from_match(
    match: Match,
    *,
    matched_cluster: bool,
    quality_flag: str = "yellow",
) -> HypothesisEvidence:
    """Emit Whitsun match-level evidence.

    ``matched_cluster=True``  → supports CLUSTER,           contradicts CLUTTER_FP.
    ``matched_cluster=False`` → supports TRANSIENT_ANCHORAGE, contradicts CLUSTER.
    """
    confidence = _quality_to_confidence(quality_flag)

    if matched_cluster:
        supports = (WHITSUN_VESSEL_CLUSTER_ACTIVITY,)
        contradicts = (WHITSUN_DETECTOR_CLUTTER_FALSE_POSITIVES,)
        reason = (
            "matched vessel within persistent cluster pattern - "
            "Whitsun militia-swarm signature"
        )
        tag = "cluster"
    else:
        supports = (WHITSUN_TRANSIENT_ANCHORAGE_OR_FISHING_PRESENCE,)
        contradicts = (WHITSUN_VESSEL_CLUSTER_ACTIVITY,)
        reason = (
            "isolated match without cluster context - consistent with "
            "ordinary anchorage / fishing, not coordination"
        )
        tag = "noncluster"

    return from_match(
        match,
        scenario_id=SCENARIO_WHITSUN,
        supports=supports,
        contradicts=contradicts,
        confidence=confidence,
        reason=reason,
        evidence_id=f"whitsun-match-{match.obs_a_id}-{match.obs_b_id}-{tag}",
    )
