"""
Fusion and belief layer for custody.

Combines track state, behavioural signals, and contextual evidence into a
unified FusionAssessment per entity per timestep.  This is Layer 4 of the
seven-layer reasoning architecture: the step that answers "what do we actually
believe, and how sure are we?" before the decision engine asks "so what?"

FusionAssessment summarises fused belief state only.  It must not decide
whether to elevate, whether to task, or which task wins — those belong to
Decision and TaskRecommendation.

Public API
----------
FusionAssessment
    Frozen dataclass: entity_id, timestamp, fused_score, uncertainty,
    source_agreement, missing_evidence, recommended_confirming_source.

build_fusion_assessment(record, compounds, track, planner_context) -> FusionAssessment
    Primary builder.  Takes existing planner/track/anomaly/compound
    ingredients and packages them into a first-class output artifact.

fusion_for_timeline(timeline, track, planner_context) -> list[FusionAssessment]
    Apply build_fusion_assessment to every record in a single-vessel timeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import custody.config as config
from custody.compounds import CompoundSignal
from custody.decision_trace import DecisionTrace
from custody.models import TrackState


# ---------------------------------------------------------------------------
# Core data contract
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FusionAssessment:
    """Unified evidence picture for one entity at one moment in time.

    Attributes:
        entity_id:    Entity being assessed.
        timestamp:    Observation time this assessment corresponds to.
        fused_score:  0–1. Weighted combination of anomaly severity, strongest
                      compound signal, and custody weakness.  Higher = more
                      operationally significant.
        uncertainty:  0–1. How much we do not know right now.  Driven by
                      custody confidence, source agreement, and evidence gaps.
        source_agreement:
                      0–1. Consistency between available evidence sources.
                      High = signals point in the same direction; low = mixed
                      or conflicting picture.
        missing_evidence:
                      Short labels for what additional information would most
                      reduce uncertainty, in priority order.
        recommended_confirming_source:
                      Single best next source to reduce uncertainty: "OPTICAL",
                      "SAR", "AIS", or None.
    """
    entity_id: str
    timestamp: datetime
    fused_score: float
    uncertainty: float
    source_agreement: float
    missing_evidence: list[str]
    recommended_confirming_source: Optional[str]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _clamp01(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _compute_fused_score(
    anomaly_score: float,
    top_compound_confidence: float,
    custody_confidence: float,
    ml_anomaly_score: float = 0.0,
) -> float:
    """Weighted combination of the primary evidence dimensions.

    When ``config.FUSION_W_ML`` is 0.0 (the default), this produces
    identical output to the pre-ML formula:
        0.45 * anomaly_norm + 0.35 * compound + 0.20 * (1 - custody)

    When ML is enabled, the ML score contributes as a fourth term using
    the weights defined in config.py.

    Args:
        anomaly_score:          Composite behavioural anomaly score, clamped [0, 1].
        top_compound_confidence: Confidence of the strongest active compound signal,
                                 or 0.0 if no compounds are active.
        custody_confidence:     Current track confidence in [0, 1].
        ml_anomaly_score:       ML anomaly score in [0, 1], or 0.0 if unavailable.

    Returns:
        fused_score in [0, 1].
    """
    return round(_clamp01(
        config.FUSION_W_HEURISTIC * min(anomaly_score, 1.0)
        + config.FUSION_W_ML * _clamp01(ml_anomaly_score)
        + config.FUSION_W_COMPOUND * top_compound_confidence
        + config.FUSION_W_CUSTODY * (1.0 - custody_confidence)
    ), 3)


def _compute_source_agreement(
    record: dict,
    compounds: list[CompoundSignal],
    track: TrackState,
    planner_context=None,
) -> float:
    """Heuristic agreement score across available evidence sources.

    Only uses signals that actually exist in the data.  High when signals point
    in the same direction; low when anomaly is elevated but corroborating context
    is absent or contradictory.

    Args:
        record:          One timeline record dict.
        compounds:       CompoundSignal objects active for this record.
        track:           TrackState for this entity.
        planner_context: Optional DecisionTrace for sensor access context.

    Returns:
        source_agreement in [0, 1].
    """
    anomaly = float(record.get("anomaly_score", 0.0))
    confidence = float(record.get("custody_confidence", 1.0))
    anom_norm = min(anomaly / config.CRITICAL_ANOMALY_THRESHOLD, 1.0)
    top_compound = max((cs.confidence for cs in compounds), default=0.0)

    # Compound–anomaly alignment: how well does compound evidence corroborate
    # the raw anomaly reading?
    if compounds:
        # Both detectors agree on degree of concern → coherent
        compound_alignment = 1.0 - abs(anom_norm - top_compound)
    elif anom_norm < 0.3:
        # No compounds and low anomaly → both say "normal" → coherent
        compound_alignment = 1.0
    else:
        # Anomaly without compound corroboration → mixed picture
        compound_alignment = 0.5

    # Custody contribution: good AIS track = AIS source is reliable and fresh
    custody_contribution = confidence

    # Sensor access contribution: if a sensor was available and we still hold,
    # the planner's caution agrees with the behavioural signal — mild alignment.
    sensor_factor = 1.0
    if isinstance(planner_context, DecisionTrace):
        access = planner_context.inputs.sensor_access_count
        action = planner_context.arbitration.result
        if access == 0 and anom_norm > 0.4:
            # No sensor accessible despite elevated anomaly → source gap, reduce
            sensor_factor = 0.8

    # ML–heuristic alignment: when both ML and heuristic agree on severity,
    # agreement rises.  When they disagree, uncertainty increases.
    # Only considered when ML weight is enabled and ML score is present.
    ml_score = float(record.get("ml_anomaly_score", 0.0))
    if config.FUSION_W_ML > 0.0 and ml_score > 0.0:
        ml_alignment = 1.0 - abs(anom_norm - ml_score)
        agreement = _clamp01(
            0.45 * compound_alignment
            + 0.25 * ml_alignment
            + 0.30 * custody_contribution
        ) * sensor_factor
    else:
        agreement = _clamp01(
            0.60 * compound_alignment
            + 0.40 * custody_contribution
        ) * sensor_factor

    return round(agreement, 3)


def _compute_uncertainty(
    custody_confidence: float,
    source_agreement: float,
    evidence_gap_score: float,
) -> float:
    """Uncertainty from three components: track quality, source coherence, evidence gaps.

    Args:
        custody_confidence: Current track confidence in [0, 1].
        source_agreement:   Output of _compute_source_agreement, in [0, 1].
        evidence_gap_score: Normalised count of missing evidence items, in [0, 1].

    Returns:
        uncertainty in [0, 1].
    """
    return round(_clamp01(
        0.50 * (1.0 - custody_confidence)
        + 0.30 * (1.0 - source_agreement)
        + 0.20 * evidence_gap_score
    ), 3)


def _infer_missing_evidence(
    record: dict,
    track: TrackState,
    compounds: list[CompoundSignal],
) -> list[str]:
    """Generate short evidence-gap labels from real gaps in the data.

    Only adds items based on conditions that genuinely exist.  No imagined gaps.

    Args:
        record:    One timeline record dict.
        track:     TrackState for this entity.
        compounds: Active CompoundSignal objects.

    Returns:
        Ordered list of short evidence-gap label strings.
    """
    anomaly = float(record.get("anomaly_score", 0.0))
    confidence = float(record.get("custody_confidence", 1.0))
    action = record.get("action", "NONE")
    zone = float(record.get("sensitive_zone", 0.0))

    missing: list[str] = []

    # Stale or weak track
    if confidence < 0.5:
        missing.append("fresh AIS update")

    # High anomaly near a sensitive zone without a confirming collection
    if anomaly > config.HIGH_ANOMALY_THRESHOLD and zone > 0 and action != "TASK":
        missing.append("optical confirmation")

    # Persistent uncertainty despite behavioural signals — SAR is weather-tolerant
    if confidence < 0.5 and anomaly > config.HIGH_ANOMALY_THRESHOLD:
        missing.append("SAR confirmation")

    # Compound signals suggest a complex pattern that merits RF cross-cue
    rf_codes = {"LOITERING_NEAR_ZONE", "PROXIMITY_NEAR_ZONE", "REPEATED_ZONE_ENTRY"}
    if any(cs.code in rf_codes for cs in compounds):
        missing.append("RF cross-cue")

    # If no collection occurred and no sensor was available, flag repeat observation
    if action == "NO_SENSOR":
        missing.append("repeat observation in next access window")

    return missing


def _recommend_confirming_source(
    record: dict,
    track: TrackState,
    compounds: list[CompoundSignal],
    missing_evidence: list[str],
) -> Optional[str]:
    """Return the single best source to reduce uncertainty next.

    Simple priority rules based on actual evidence gaps:
    - "OPTICAL" when visual confirmation is needed and track is reasonable
    - "SAR"     when confirmation is needed but uncertainty is high or AIS is stale
    - "AIS"     when the primary tracking source has degraded
    - None      when no specific confirming action can be recommended

    Args:
        record:          One timeline record dict.
        track:           TrackState for this entity.
        compounds:       Active CompoundSignal objects.
        missing_evidence: Output of _infer_missing_evidence.

    Returns:
        "OPTICAL", "SAR", "AIS", or None.
    """
    confidence = float(record.get("custody_confidence", 1.0))
    anomaly = float(record.get("anomaly_score", 0.0))

    if "SAR confirmation" in missing_evidence:
        return "SAR"

    if "optical confirmation" in missing_evidence:
        # Prefer SAR if confidence is low — likely stale conditions
        if confidence < 0.4:
            return "SAR"
        return "OPTICAL"

    if "fresh AIS update" in missing_evidence:
        return "AIS"

    if "RF cross-cue" in missing_evidence:
        return "SAR"

    return None


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

def build_fusion_assessment(
    record: dict,
    compounds: list[CompoundSignal],
    track: TrackState,
    planner_context=None,
) -> FusionAssessment:
    """Build a FusionAssessment from existing planner/track/anomaly/compound ingredients.

    This is an output artifact: frozen and deterministic for the same inputs.
    It summarises fused belief state only — it does not decide whether to
    elevate, task, or choose between sensors.

    Args:
        record:          One timeline record dict as produced by run_simulation()
                         or ingest_ais_track().  Must have an ``entity_id`` or
                         ``vessel_id`` key and a ``time`` key.
        compounds:       CompoundSignal objects active for this entity at this
                         timestep.  Pass an empty list if none are active.
        track:           TrackState for this entity at this timestep.
        planner_context: Optional DecisionTrace from this timestep's planner run.
                         Used for sensor access count and orbital pass context.
                         Passing None is safe — defaults are applied.

    Returns:
        A frozen FusionAssessment.
    """
    entity_id = (
        record.get("entity_id")
        or record.get("vessel_id")
        or record.get("target_id")
        or "unknown"
    )
    timestamp: datetime = record["time"]

    anomaly_score = float(record.get("anomaly_score") or 0.0)
    custody_confidence = float(record.get("custody_confidence") or 0.0)

    top_compound_confidence = max(
        (cs.confidence for cs in compounds), default=0.0
    )

    # Build in dependency order (uncertainty needs source_agreement)
    source_agreement = _compute_source_agreement(
        record, compounds, track, planner_context
    )

    missing_evidence = _infer_missing_evidence(record, track, compounds)

    evidence_gap_score = min(len(missing_evidence) / 4.0, 1.0)

    ml_anomaly_score = float(record.get("ml_anomaly_score") or 0.0)

    fused_score = _compute_fused_score(
        anomaly_score, top_compound_confidence, custody_confidence,
        ml_anomaly_score=ml_anomaly_score,
    )

    uncertainty = _compute_uncertainty(
        custody_confidence, source_agreement, evidence_gap_score
    )

    recommended = _recommend_confirming_source(
        record, track, compounds, missing_evidence
    )

    return FusionAssessment(
        entity_id=entity_id,
        timestamp=timestamp,
        fused_score=fused_score,
        uncertainty=uncertainty,
        source_agreement=source_agreement,
        missing_evidence=missing_evidence,
        recommended_confirming_source=recommended,
    )


# ---------------------------------------------------------------------------
# Timeline convenience wrapper
# ---------------------------------------------------------------------------

def fusion_for_timeline(
    timeline: list[dict],
    track: TrackState,
    planner_context_by_time: Optional[dict] = None,
) -> list[FusionAssessment]:
    """Apply build_fusion_assessment to every record in a single-vessel timeline.

    Compounds are evaluated incrementally so window-based rules (e.g.
    REPEATED_ZONE_ENTRY) have access to the correct history prefix.

    Args:
        timeline:              Ordered list of record dicts for one vessel.
        track:                 The vessel's TrackState (shared across all steps;
                               uncertainty evolves but this function treats it
                               as a snapshot).
        planner_context_by_time:
                               Optional dict mapping datetime → DecisionTrace
                               for per-step planner context.  Pass None if
                               planner traces are not available.

    Returns:
        List of FusionAssessment in the same order as timeline.
        Returns an empty list when timeline is empty.
    """
    from custody.compounds import evaluate_compounds

    ctx_map = planner_context_by_time or {}
    result: list[FusionAssessment] = []

    for i, record in enumerate(timeline):
        window = timeline[:i]
        compounds = evaluate_compounds(record, window=window)
        ctx = ctx_map.get(record.get("time"))
        result.append(build_fusion_assessment(record, compounds, track, ctx))

    return result
