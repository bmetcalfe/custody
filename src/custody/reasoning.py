"""Temporal anomaly reasoning layer.

Enriches per-timestep records with agreement classification, persistence
tracking, escalation factors, and anomaly state transitions.  Operates
on the record dict + a window of prior records for the same entity.
Does not modify any engine scoring — purely additive.

All functions are deterministic, use only past/current data (no future
leakage), and degrade gracefully when ML fields are absent.

Public API
----------
AnomalyAgreement (str enum)
    CONFIRMED | EMERGING | RULE_TRIGGERED | NORMAL

AnomalyState (str enum)
    NORMAL | EMERGING | CONFIRMED | SUSTAINED | CRITICAL | RECOVERING

classify_agreement(record) -> AnomalyAgreement
compute_persistence(record, history_window, threshold) -> dict
compute_escalation(persistence, agreement, custody_confidence) -> float
derive_anomaly_state(agreement, persistence, custody_confidence) -> AnomalyState
enrich_record(record, history_window) -> dict
"""
from __future__ import annotations

import custody.config as config


# ---------------------------------------------------------------------------
# Agreement classification
# ---------------------------------------------------------------------------

class AnomalyAgreement:
    CONFIRMED      = "confirmed"       # ML high AND heuristic high
    EMERGING       = "emerging"         # ML high, heuristic low
    RULE_TRIGGERED = "rule_triggered"   # heuristic high, ML low
    NORMAL         = "normal"           # both low


def _safe_float(val, default: float = 0.0) -> float:
    """Convert a value to float, returning *default* for None or NaN."""
    if val is None:
        return default
    try:
        f = float(val)
        return default if f != f else f  # NaN check
    except (TypeError, ValueError):
        return default


def ml_combined_score(record: dict) -> float:
    """Compute the combined ML signal: max(absolute, relative).

    Returns a value in [0, 1].  Falls back to absolute when relative
    is unavailable.
    """
    absolute = _safe_float(record.get("ml_anomaly_score"))
    relative = _safe_float(record.get("ml_anomaly_relative"))
    return max(absolute, relative)


def _is_ml_high(record: dict) -> bool:
    """Determine if the ML signal is "high" for this record.

    Behaviour depends on ``config.ML_SIGNAL_MODE``:
      ``"absolute"``  — ml_anomaly_score >= ML_ANOMALY_HIGH_THRESHOLD
      ``"relative"``  — ml_anomaly_relative >= RELATIVE_ML_HIGH_THRESHOLD
                         (falls back to absolute when relative is absent)
      ``"combined"``  — max(absolute, relative) >= COMBINED_ML_HIGH_THRESHOLD

    The legacy ``USE_RELATIVE_ML_THRESHOLD = True`` is treated as
    ``ML_SIGNAL_MODE = "relative"`` for backward compatibility.
    """
    mode = config.ML_SIGNAL_MODE

    # Legacy compat
    if config.USE_RELATIVE_ML_THRESHOLD and mode == "absolute":
        mode = "relative"

    if mode == "relative":
        rel = _safe_float(record.get("ml_anomaly_relative"))
        if rel > 0:
            return rel >= config.RELATIVE_ML_HIGH_THRESHOLD
        # Fall back to absolute when relative is unavailable
        return _safe_float(record.get("ml_anomaly_score")) >= config.ML_ANOMALY_HIGH_THRESHOLD

    if mode == "combined":
        return ml_combined_score(record) >= config.COMBINED_ML_HIGH_THRESHOLD

    # Default: absolute
    return _safe_float(record.get("ml_anomaly_score")) >= config.ML_ANOMALY_HIGH_THRESHOLD


def classify_agreement(record: dict) -> str:
    """Classify ML/heuristic agreement from one record.

    Uses ``ml_anomaly_score`` (or ``ml_anomaly_relative`` when enabled)
    and ``anomaly_score`` (normalised by CRITICAL_ANOMALY_THRESHOLD).
    Degrades to NORMAL when ML is absent.

    Returns one of the :class:`AnomalyAgreement` constants.
    """
    heuristic = float(record.get("anomaly_score", 0.0))
    heuristic_norm = min(heuristic / config.CRITICAL_ANOMALY_THRESHOLD, 1.0)

    ml_high = _is_ml_high(record)
    heur_high = heuristic_norm >= config.HEURISTIC_ANOMALY_HIGH_THRESHOLD

    if ml_high and heur_high:
        return AnomalyAgreement.CONFIRMED
    if ml_high and not heur_high:
        return AnomalyAgreement.EMERGING
    if heur_high and not ml_high:
        return AnomalyAgreement.RULE_TRIGGERED
    return AnomalyAgreement.NORMAL


# ---------------------------------------------------------------------------
# Persistence tracking
# ---------------------------------------------------------------------------

def compute_persistence(
    record: dict,
    history_window: list[dict],
    fused_threshold: float = 0.5,
) -> dict:
    """Compute anomaly duration from the current record and prior history.

    Scans backwards through ``history_window`` counting consecutive
    timesteps where the ML signal is "high" (using :func:`_is_ml_high`,
    which respects relative thresholds when enabled).

    Args:
        record:          Current timestep record.
        history_window:  Prior records for this entity, oldest first.
        fused_threshold: Fused score threshold for fused persistence.

    Returns:
        Dict with keys: ml_anomaly_duration_hours, fused_anomaly_duration_hours,
        is_sustained_anomaly, anomaly_onset_timestamp.
    """
    # ML persistence: count consecutive hours where ML is "high"
    # Uses the same _is_ml_high() logic as agreement classification,
    # so relative thresholds are respected when enabled.
    ml_streak = 1 if _is_ml_high(record) else 0
    if ml_streak > 0:
        for r in reversed(history_window):
            if _is_ml_high(r):
                ml_streak += 1
            else:
                break

    # Fused persistence
    fused_current = float(record.get("fused_score", 0.0))
    # Also check fusion_assessment object if fused_score not directly on record
    fa = record.get("fusion_assessment")
    if fa is not None and hasattr(fa, "fused_score"):
        fused_current = max(fused_current, fa.fused_score)
    fused_streak = 1 if fused_current >= fused_threshold else 0
    if fused_streak > 0:
        for r in reversed(history_window):
            r_fused = float(r.get("fused_score", 0.0))
            r_fa = r.get("fusion_assessment")
            if r_fa is not None and hasattr(r_fa, "fused_score"):
                r_fused = max(r_fused, r_fa.fused_score)
            if r_fused >= fused_threshold:
                fused_streak += 1
            else:
                break

    is_sustained = ml_streak >= config.SUSTAINED_ANOMALY_MIN_HOURS

    # Find onset timestamp
    onset = None
    if ml_streak > 0:
        onset = record.get("time") or record.get("timestamp")
        lookback = list(reversed(history_window))
        for i in range(min(ml_streak - 1, len(lookback))):
            if _is_ml_high(lookback[i]):
                onset = lookback[i].get("time") or lookback[i].get("timestamp")

    return {
        "ml_anomaly_duration_hours":    ml_streak,
        "fused_anomaly_duration_hours": fused_streak,
        "is_sustained_anomaly":         is_sustained,
        "anomaly_onset_timestamp":      onset,
    }


# ---------------------------------------------------------------------------
# Escalation
# ---------------------------------------------------------------------------

def compute_escalation(
    persistence: dict,
    agreement: str,
    custody_confidence: float,
) -> float:
    """Compute an escalation boost from persistence + agreement + custody.

    Returns a small additive boost in [0, ESCALATION_BOOST_CAP].
    Higher when anomaly is sustained, confirmed, and custody is weak.

    Does not modify fused_score directly — the caller decides how to use it.
    """
    ml_hours = persistence.get("ml_anomaly_duration_hours", 0)
    is_sustained = persistence.get("is_sustained_anomaly", False)

    if ml_hours == 0:
        return 0.0

    # Base: hours above threshold × per-hour boost
    base = min(ml_hours, 10) * config.ESCALATION_BOOST_PER_HOUR

    # Agreement multiplier
    if agreement == AnomalyAgreement.CONFIRMED:
        multiplier = 1.5
    elif agreement == AnomalyAgreement.EMERGING:
        multiplier = 1.0
    elif agreement == AnomalyAgreement.RULE_TRIGGERED:
        multiplier = 0.5
    else:
        multiplier = 0.0

    # Custody weakness amplifier: weaker custody → stronger escalation
    custody_factor = 1.0 + 0.5 * (1.0 - min(max(custody_confidence, 0.0), 1.0))

    boost = base * multiplier * custody_factor
    return min(boost, config.ESCALATION_BOOST_CAP)


# ---------------------------------------------------------------------------
# Anomaly state transitions
# ---------------------------------------------------------------------------

class AnomalyState:
    NORMAL     = "normal"
    EMERGING   = "emerging"
    CONFIRMED  = "confirmed"
    SUSTAINED  = "sustained"
    CRITICAL   = "critical"
    RECOVERING = "recovering"


def derive_anomaly_state(
    agreement: str,
    persistence: dict,
    custody_confidence: float,
    previous_state: str | None = None,
) -> str:
    """Derive the current anomaly state from agreement + persistence + custody.

    State definitions:
        normal:     No significant anomaly signal.
        emerging:   ML high but not yet confirmed by heuristic or duration.
        confirmed:  ML and heuristic both high.
        sustained:  Anomaly confirmed for >= SUSTAINED_ANOMALY_MIN_HOURS.
        critical:   Sustained + weak custody (confidence < 0.4).
        recovering: Previously anomalous (sustained/critical/confirmed),
                    now dropping below thresholds.

    Args:
        agreement:          Output of :func:`classify_agreement`.
        persistence:        Output of :func:`compute_persistence`.
        custody_confidence: Current custody confidence [0, 1].
        previous_state:     Anomaly state from the previous timestep,
                            or None for the first timestep.

    Returns:
        One of the :class:`AnomalyState` constants.
    """
    ml_hours = persistence.get("ml_anomaly_duration_hours", 0)
    is_sustained = persistence.get("is_sustained_anomaly", False)

    # Check if currently in an elevated state
    if agreement == AnomalyAgreement.NORMAL:
        # Was previously anomalous? → recovering
        if previous_state in (
            AnomalyState.CONFIRMED, AnomalyState.SUSTAINED, AnomalyState.CRITICAL
        ):
            return AnomalyState.RECOVERING
        return AnomalyState.NORMAL

    # ML or heuristic is elevated
    if is_sustained and custody_confidence < 0.4:
        return AnomalyState.CRITICAL

    if is_sustained:
        return AnomalyState.SUSTAINED

    if agreement == AnomalyAgreement.CONFIRMED:
        return AnomalyState.CONFIRMED

    if agreement == AnomalyAgreement.EMERGING:
        return AnomalyState.EMERGING

    if agreement == AnomalyAgreement.RULE_TRIGGERED:
        # Heuristic-only without ML support: do not escalate lightly.
        # The heuristic signal is already reflected in anomaly_score and
        # fusion.  Only escalate if there is persistent ML confirmation.
        if ml_hours >= 3:
            return AnomalyState.CONFIRMED
        # Otherwise treat as normal — single-source heuristic is
        # not sufficient for reasoning-layer escalation.
        return AnomalyState.NORMAL

    return AnomalyState.NORMAL


# ---------------------------------------------------------------------------
# Record enrichment (convenience)
# ---------------------------------------------------------------------------

def enrich_record(
    record: dict,
    history_window: list[dict],
    previous_state: str | None = None,
) -> dict:
    """Add all reasoning fields to a record dict (non-mutating).

    Args:
        record:          Current timestep record.
        history_window:  Prior records for this entity, oldest first.
        previous_state:  Anomaly state from the previous timestep.

    Returns:
        New dict with all original fields plus:
        anomaly_agreement, ml_anomaly_duration_hours,
        fused_anomaly_duration_hours, is_sustained_anomaly,
        anomaly_onset_timestamp, escalation_boost, anomaly_state.
    """
    agreement = classify_agreement(record)
    persistence = compute_persistence(record, history_window)
    custody = float(record.get("custody_confidence", 1.0))
    escalation = compute_escalation(persistence, agreement, custody)
    state = derive_anomaly_state(agreement, persistence, custody, previous_state)

    enriched = dict(record)
    enriched["ml_anomaly_combined"] = ml_combined_score(record)
    enriched["anomaly_agreement"] = agreement
    enriched["ml_anomaly_duration_hours"] = persistence["ml_anomaly_duration_hours"]
    enriched["fused_anomaly_duration_hours"] = persistence["fused_anomaly_duration_hours"]
    enriched["is_sustained_anomaly"] = persistence["is_sustained_anomaly"]
    enriched["anomaly_onset_timestamp"] = persistence["anomaly_onset_timestamp"]
    enriched["escalation_boost"] = escalation
    enriched["anomaly_state"] = state
    return enriched
