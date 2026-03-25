"""
Portfolio-level orchestration layer for Custody.

Operates across all entities at each timestep to:
  1. Rank entities by portfolio urgency (anomaly + neglect + custody degradation)
  2. Classify custody health per entity (HEALTHY → DEGRADING → STALE → LOST)
  3. Flag entities that have gone unobserved beyond the neglect threshold
  4. Surface tradeoff context when one entity is serviced over another

This layer composes from existing per-entity signals already present in
timeline records (anomaly_score, custody_confidence, uncertainty_km, action,
hours_since_collection).  It does not replace or re-implement fusion,
decision, or taskrecommendation logic.

Key design rules
----------------
- Deterministic: same inputs → same outputs, no hidden state
- Explainable: every rank comes with a one-sentence rationale
- Composable: reads from existing record fields, writes to new additive fields
- Narrow: no booking, no multi-step planning, no trajectory projection

Public API
----------
CustodyHealth           Frozen dataclass: per-entity health classification.
PortfolioItem           Frozen dataclass: per-entity portfolio assessment.
PortfolioAssessment     Frozen dataclass: full portfolio at one timestep.

compute_custody_health(entity_id, uncertainty_km, custody_confidence, neglect_hours)
    -> CustodyHealth

detect_neglect(entity_id, neglect_hours, threshold_hours=NEGLECT_THRESHOLD_HOURS)
    -> bool

rank_portfolio(timestamp, timestep_records, scenario_start, sensor_claimer=None)
    -> PortfolioAssessment
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from custody.orchestration.attention import (
    derive_attention_state,
    apply_tracking_directive_floor,
    apply_dark_vessel_floor,
    compute_neglect_weight,
    explain_attention_state,
    DIRECTIVE_NONE,
)

# ---------------------------------------------------------------------------
# Tunable thresholds
# ---------------------------------------------------------------------------

NEGLECT_THRESHOLD_HOURS: float = 4.0
"""Hours without a collection action before an entity is flagged as neglected."""

# Custody confidence breakpoints (descending; worst wins)
_CONF_DEGRADING = 0.55
_CONF_STALE     = 0.35
_CONF_LOST      = 0.15

# Positional uncertainty breakpoints in km (ascending; worst wins)
_UNC_DEGRADING_KM = 20.0
_UNC_STALE_KM     = 50.0
_UNC_LOST_KM      = 90.0

# Portfolio score weights (must sum to 1.0)
_W_URGENCY = 0.30   # normalised anomaly signal
_W_CUSTODY = 0.25   # custody degradation pressure (1 – confidence)
_W_NEGLECT = 0.20   # neglect pressure (tier-gated)
_W_UNCERT  = 0.10   # positional imprecision
_W_TIER    = 0.15   # attention tier baseline (BG=0, WL=0.5, AC=1.0)

_TIER_BASELINE = {"BACKGROUND": 0.0, "WATCHLIST": 0.5, "ACTIVE_CUSTODY": 1.0}

_ANOMALY_MAX = 3.0    # anomaly_score is [0, 3]; used as normaliser
_UNC_MAX_KM  = 150.0  # uncertainty normaliser


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CustodyHealth:
    """Custody health classification for one entity at one timestep.

    Attributes:
        entity_id:          Entity identifier.
        status:             One of HEALTHY, DEGRADING, STALE, LOST.
        uncertainty_km:     Positional uncertainty at this timestep.
        custody_confidence: Track confidence in [0, 1].
        neglect_hours:      Hours elapsed since last collection action.
        reason:             Human-readable explanation of the classification.
    """
    entity_id: str
    status: str               # HEALTHY | DEGRADING | STALE | LOST
    uncertainty_km: float
    custody_confidence: float
    neglect_hours: float
    reason: str


@dataclass(frozen=True)
class PortfolioItem:
    """Portfolio-level assessment for one entity at one timestep.

    Attributes:
        entity_id:          Entity identifier.
        portfolio_rank:     1 = highest urgency in the portfolio this timestep.
        portfolio_score:    Composite urgency score in [0, 1].
        custody_health:     Status string from CustodyHealth.
        neglect_flag:       True if entity has not been collected for >= NEGLECT_THRESHOLD_HOURS.
        neglect_hours:      Hours since last successful collection (or since scenario start).
        portfolio_reason:   One-sentence explanation of this ranking.
        deferred_for:       Entity_id that consumed collection this timestep when this entity
                            was PREEMPTED.  None if not preempted.
        attention_state:    Selective-custody tier: BACKGROUND | WATCHLIST | ACTIVE_CUSTODY.
        tracking_directive: Operator override: NONE | MAINTAIN_CUSTODY.
        attention_basis:    One-sentence explanation of the attention tier.
    """
    entity_id: str
    portfolio_rank: int
    portfolio_score: float
    custody_health: str
    neglect_flag: bool
    neglect_hours: float
    portfolio_reason: str
    deferred_for: Optional[str]
    attention_state: str       # BACKGROUND | WATCHLIST | ACTIVE_CUSTODY
    tracking_directive: str    # NONE | MAINTAIN_CUSTODY
    attention_basis: str


@dataclass(frozen=True)
class PortfolioAssessment:
    """Full portfolio-wide assessment at one timestep.

    Attributes:
        timestamp:  Simulation time this assessment covers.
        items:      All entities sorted by portfolio_rank ascending (rank 1 first).
        serviced:   Entity IDs that received a TASK action this timestep.
        deferred:   Entity IDs that were PREEMPTED this timestep.
    """
    timestamp: datetime
    items: tuple[PortfolioItem, ...]
    serviced: tuple[str, ...]
    deferred: tuple[str, ...]

    def by_entity(self) -> dict[str, PortfolioItem]:
        """Return a mapping from entity_id to its PortfolioItem."""
        return {item.entity_id: item for item in self.items}


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def compute_custody_health(
    entity_id: str,
    uncertainty_km: float,
    custody_confidence: float,
    neglect_hours: float,
) -> CustodyHealth:
    """Classify custody health for one entity.

    The worst-case of the confidence and uncertainty signals determines the
    status.  Severity order: HEALTHY < DEGRADING < STALE < LOST.

    Args:
        entity_id:          Entity identifier.
        uncertainty_km:     Current positional uncertainty in km.
        custody_confidence: Current confidence in [0, 1].
        neglect_hours:      Hours since last collection action.

    Returns:
        CustodyHealth with status ∈ {HEALTHY, DEGRADING, STALE, LOST}.
    """
    if custody_confidence < _CONF_LOST or uncertainty_km > _UNC_LOST_KM:
        status = "LOST"
        reason = (
            f"Custody effectively lost: confidence={custody_confidence:.2f}, "
            f"uncertainty={uncertainty_km:.1f} km"
        )
    elif custody_confidence < _CONF_STALE or uncertainty_km > _UNC_STALE_KM:
        status = "STALE"
        reason = (
            f"Track stale: confidence={custody_confidence:.2f}, "
            f"uncertainty={uncertainty_km:.1f} km"
        )
    elif custody_confidence < _CONF_DEGRADING or uncertainty_km > _UNC_DEGRADING_KM:
        status = "DEGRADING"
        reason = (
            f"Custody degrading: confidence={custody_confidence:.2f}, "
            f"uncertainty={uncertainty_km:.1f} km"
        )
    else:
        status = "HEALTHY"
        reason = (
            f"Custody healthy: confidence={custody_confidence:.2f}, "
            f"uncertainty={uncertainty_km:.1f} km"
        )

    return CustodyHealth(
        entity_id=entity_id,
        status=status,
        uncertainty_km=uncertainty_km,
        custody_confidence=custody_confidence,
        neglect_hours=neglect_hours,
        reason=reason,
    )


def detect_neglect(
    entity_id: str,
    neglect_hours: float,
    threshold_hours: float = NEGLECT_THRESHOLD_HOURS,
) -> bool:
    """Return True if the entity has gone unobserved beyond the neglect threshold.

    Args:
        entity_id:       Entity identifier (unused; reserved for future extension).
        neglect_hours:   Hours since last collection action.
        threshold_hours: Threshold in hours (default: NEGLECT_THRESHOLD_HOURS).

    Returns:
        True if neglect_hours >= threshold_hours.
    """
    return neglect_hours >= threshold_hours


def _portfolio_score(
    anomaly_score: float,
    custody_confidence: float,
    uncertainty_km: float,
    neglect_hours: float,
    attention_state: str = "ACTIVE_CUSTODY",
) -> float:
    """Composite portfolio urgency score in [0, 1].

    Components:
      0.30 × normalised anomaly                             (mission urgency)
      0.25 × (1 – custody_conf)                            (custody pressure)
      0.20 × neglect pressure × neglect_weight(tier)       (tier-gated neglect)
      0.10 × normalised uncertainty                         (positional imprecision)
      0.15 × attention tier baseline (BG=0, WL=0.5, AC=1.0)

    Neglect is gated by attention tier:
      BACKGROUND     → weight 0.0  (no neglect pressure)
      WATCHLIST      → weight 0.3  (reduced)
      ACTIVE_CUSTODY → weight 1.0  (full pressure)

    The tier baseline ensures ACTIVE_CUSTODY vessels always score above
    BACKGROUND vessels with otherwise identical signals, even after
    background neglect is zeroed out.
    """
    urgency        = min(anomaly_score / _ANOMALY_MAX, 1.0)
    custody        = 1.0 - min(max(custody_confidence, 0.0), 1.0)
    neglect_raw    = min(neglect_hours / (NEGLECT_THRESHOLD_HOURS * 2.0), 1.0)
    neglect        = neglect_raw * compute_neglect_weight(attention_state)
    uncert         = min(uncertainty_km / _UNC_MAX_KM, 1.0)
    tier_base      = _TIER_BASELINE.get(attention_state, 1.0)
    score          = (
        _W_URGENCY * urgency
        + _W_CUSTODY * custody
        + _W_NEGLECT * neglect
        + _W_UNCERT  * uncert
        + _W_TIER    * tier_base
    )
    return round(min(score, 1.0), 4)


def _build_reason(
    anomaly_score: float,
    custody_health: str,
    neglect_flag: bool,
    neglect_hours: float,
    rank: int,
    n_entities: int,
    zone_probability: float = 0.0,
    time_to_zone_hours: float | None = None,
) -> str:
    """One-sentence human-readable explanation of this entity's portfolio rank."""
    parts: list[str] = []
    # Prediction-driven part: prepend zone-approach signal when significant
    if zone_probability > 0.5 and time_to_zone_hours is not None:
        parts.append(f"projected zone entry in {time_to_zone_hours:.1f}h")
    if anomaly_score >= 1.5:
        parts.append(f"elevated anomaly ({anomaly_score:.2f})")
    if neglect_flag:
        parts.append(f"neglected {neglect_hours:.1f}h without collection")
    if custody_health in ("STALE", "LOST"):
        parts.append(f"custody {custody_health.lower()}")
    elif custody_health == "DEGRADING":
        parts.append("custody degrading")
    if not parts:
        parts.append("routine monitoring")
    return f"Ranked {rank}/{n_entities}: " + "; ".join(parts) + "."


def rank_portfolio(
    timestamp: datetime,
    timestep_records: list[dict],
    scenario_start: datetime,
    sensor_claimer: dict[str, str] | None = None,
) -> PortfolioAssessment:
    """Evaluate the full entity portfolio at one timestep.

    Each record must contain at minimum:
        target_id            (str)
        anomaly_score        (float)
        custody_confidence   (float)
        uncertainty_km       (float)
        action               (str)
        hours_since_collection (float | None)

    When ``hours_since_collection`` is None the entity has never been
    collected; elapsed time since scenario_start is substituted so the
    neglect clock starts from the beginning of the scenario.

    Args:
        timestamp:         Current simulation time.
        timestep_records:  One record per entity emitted this timestep.
        scenario_start:    Scenario start datetime (neglect fallback for
                           entities that have never been collected).
        sensor_claimer:    Optional mapping sensor_id → entity_id.  Used to
                           explain which entity consumed collection capacity
                           when another was preempted.

    Returns:
        PortfolioAssessment with all entities ranked by urgency descending.
    """
    if not timestep_records:
        return PortfolioAssessment(
            timestamp=timestamp,
            items=(),
            serviced=(),
            deferred=(),
        )

    sensor_claimer = sensor_claimer or {}
    elapsed_h = (timestamp - scenario_start).total_seconds() / 3600.0

    # ── Pass 1: score every entity ───────────────────────────────────────────
    # Tuple: (score, entity_id, record, CustodyHealth, neglect_hours, attention_state, directive)
    scored: list[tuple[float, str, dict, CustodyHealth, float, str, str]] = []

    for record in timestep_records:
        eid       = record["target_id"]
        anomaly   = float(record.get("anomaly_score", 0.0))
        conf      = float(record.get("custody_confidence", 1.0))
        unc_km    = float(record.get("uncertainty_km", 0.0))
        zone_sc   = float(record.get("sensitive_zone", 0.0))
        directive = str(record.get("tracking_directive", DIRECTIVE_NONE))
        hsc       = record.get("hours_since_collection")
        n_hours   = float(hsc) if hsc is not None else elapsed_h

        dark_flag = bool(record.get("dark_vessel_flag", False))

        # Derive attention tier: behavior-driven → directive floor → dark floor
        attention = derive_attention_state(anomaly, conf, unc_km, zone_sc)
        attention = apply_tracking_directive_floor(attention, directive)
        attention = apply_dark_vessel_floor(attention, dark_flag, directive)

        health = compute_custody_health(eid, unc_km, conf, n_hours)
        base_score = _portfolio_score(anomaly, conf, unc_km, n_hours, attention)
        # Prediction boost: approaching zone raises urgency slightly
        zone_prob_raw = record.get("zone_probability", 0.0)
        zone_prob = float(zone_prob_raw) if zone_prob_raw is not None else 0.0
        if math.isnan(zone_prob):
            zone_prob = 0.0
        prediction_boost = zone_prob * 0.15
        score = round(min(1.0, base_score + prediction_boost), 4)
        scored.append((score, eid, record, health, n_hours, attention, directive))

    # Sort descending by score; break ties by entity_id for determinism
    scored.sort(key=lambda t: (-t[0], t[1]))
    n = len(scored)

    # ── Identify serviced / deferred sets ────────────────────────────────────
    serviced_set: set[str] = set()
    deferred_set: set[str] = set()
    for record in timestep_records:
        action = record.get("action")
        if action == "TASK":
            serviced_set.add(record["target_id"])
        elif action == "PREEMPTED":
            deferred_set.add(record["target_id"])

    # Highest-ranked serviced entity — used to explain preemption
    top_serviced: Optional[str] = None
    for _, eid, _, _, _, _, _ in scored:
        if eid in serviced_set:
            top_serviced = eid
            break

    # ── Pass 2: build PortfolioItems ─────────────────────────────────────────
    items: list[PortfolioItem] = []
    for rank_idx, (score, eid, record, health, n_hours, attention, directive) in enumerate(scored, start=1):
        anomaly      = float(record.get("anomaly_score", 0.0))
        zone_sc      = float(record.get("sensitive_zone", 0.0))
        conf         = float(record.get("custody_confidence", 1.0))
        dark_flag    = bool(record.get("dark_vessel_flag", False))
        neglect      = detect_neglect(eid, n_hours)
        # Prediction fields from record (present only when timeline has run prediction)
        _raw_zone_prob = record.get("zone_probability", 0.0)
        _zone_prob = float(_raw_zone_prob) if _raw_zone_prob == _raw_zone_prob else 0.0
        _raw_tte = record.get("time_to_zone_hours")
        if _raw_tte is None or (isinstance(_raw_tte, float) and math.isnan(_raw_tte)):  # None or NaN
            _tte: float | None = None
        else:
            _tte = float(_raw_tte)
        reason       = _build_reason(
            anomaly, health.status, neglect, n_hours, rank_idx, n,
            zone_probability=_zone_prob,
            time_to_zone_hours=_tte,
        )
        deferred_for = top_serviced if eid in deferred_set else None
        basis        = explain_attention_state(attention, directive, anomaly, zone_sc, conf, dark_flag=dark_flag)

        items.append(PortfolioItem(
            entity_id          = eid,
            portfolio_rank     = rank_idx,
            portfolio_score    = score,
            custody_health     = health.status,
            neglect_flag       = neglect,
            neglect_hours      = n_hours,
            portfolio_reason   = reason,
            deferred_for       = deferred_for,
            attention_state    = attention,
            tracking_directive = directive,
            attention_basis    = basis,
        ))

    return PortfolioAssessment(
        timestamp = timestamp,
        items     = tuple(items),
        serviced  = tuple(sorted(serviced_set)),
        deferred  = tuple(sorted(deferred_set)),
    )
