"""Adaptive tasking policy driven by anomaly reasoning.

Maps the current vessel state (priority, anomaly state, agreement,
persistence, custody) into actionable policy outputs that downstream
systems (planner, decision trace, UI) can consume.

All logic is deterministic and explainable.  No hidden state.

Public API
----------
TaskingPolicy (frozen dataclass)
    tasking_tier, desired_revisit_hours, monitoring_action,
    sensor_preference, rationale.

compute_policy(record) -> TaskingPolicy
"""
from __future__ import annotations

from dataclasses import dataclass

import custody.config as config


# ---------------------------------------------------------------------------
# Tasking tiers (ordered lowest → highest)
# ---------------------------------------------------------------------------

TIER_ROUTINE   = "routine"       # baseline monitoring
TIER_ELEVATED  = "elevated"      # increased attention
TIER_PRIORITY  = "priority"      # active tasking
TIER_URGENT    = "urgent"        # shortest revisit, best sensor
TIER_CRITICAL  = "critical"      # all-available, immediate

_TIER_ORDER = [TIER_ROUTINE, TIER_ELEVATED, TIER_PRIORITY, TIER_URGENT, TIER_CRITICAL]

# ---------------------------------------------------------------------------
# Monitoring actions
# ---------------------------------------------------------------------------

ACTION_MAINTAIN           = "maintain"           # continue current cadence
ACTION_INCREASE_ATTENTION = "increase_attention"  # shorten revisit
ACTION_CONFIRM            = "confirm"             # seek confirming observation
ACTION_INTENSIFY          = "intensify"           # maximum collection effort
ACTION_COOLDOWN           = "cooldown"            # gradually reduce attention

# ---------------------------------------------------------------------------
# Sensor preferences
# ---------------------------------------------------------------------------

SENSOR_ANY       = "any"
SENSOR_HIGH_RES  = "high_resolution"
SENSOR_SAR       = "all_weather"
SENSOR_FAST      = "fast_revisit"


# ---------------------------------------------------------------------------
# Policy output
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TaskingPolicy:
    """Actionable tasking policy for one vessel at one timestep.

    Attributes:
        tasking_tier:              One of routine/elevated/priority/urgent/critical.
        desired_revisit_hours:     How soon the vessel should be re-observed.
        monitoring_action:         What kind of operational response is warranted.
        sensor_preference:         Preferred sensor type (anomaly-driven, pre-solar).
        effective_sensor_preference:
                                   Final sensor preference after solar adjustment.
        rationale:                 Human-readable explanation of the policy.
        sensor_rationale:          Explanation of solar-adjusted sensor selection.
    """
    tasking_tier: str
    desired_revisit_hours: float
    monitoring_action: str
    sensor_preference: str
    effective_sensor_preference: str
    rationale: str
    sensor_rationale: str


# ---------------------------------------------------------------------------
# Policy computation
# ---------------------------------------------------------------------------

# Default revisit cadence per tier (hours)
_REVISIT = {
    TIER_ROUTINE:  12.0,
    TIER_ELEVATED:  6.0,
    TIER_PRIORITY:  3.0,
    TIER_URGENT:    1.0,
    TIER_CRITICAL:  0.5,
}


def compute_policy(record: dict) -> TaskingPolicy:
    """Compute the adaptive tasking policy for one enriched record.

    Args:
        record: Dict with at minimum: priority_score, anomaly_state,
                anomaly_agreement, ml_anomaly_duration_hours,
                custody_confidence.  Missing fields use safe defaults.

    Returns:
        A frozen :class:`TaskingPolicy`.
    """
    # Use anomaly-driven signals for tier gating.
    # fused_score includes custody weakness which inflates all vessels over time.
    # Require actual anomaly evidence (anomaly_score) above a meaningful level
    # before fused_score can drive tier escalation.
    fa = record.get("fusion_assessment")
    fused = fa.fused_score if fa is not None and hasattr(fa, "fused_score") else 0.0
    raw_anomaly = float(record.get("anomaly_score", 0.0))
    has_anomaly_evidence = raw_anomaly > config.ANOMALY_THRESHOLD
    priority = fused if has_anomaly_evidence else fused * 0.5
    state = record.get("anomaly_state", "normal")
    agreement = record.get("anomaly_agreement", "normal")
    duration = int(record.get("ml_anomaly_duration_hours", 0))
    custody = float(record.get("custody_confidence", 1.0))

    # ── Determine tasking tier ───────────────────────────────────────
    # Disciplined escalation: most vessels routine, narrow elevated band,
    # priority requires strong + persistent evidence.
    if state == "critical":
        tier = TIER_CRITICAL
    elif state == "sustained" or (state == "confirmed" and duration >= 5):
        tier = TIER_URGENT
    elif (state == "confirmed" and duration >= 2) or priority >= 0.86:
        tier = TIER_PRIORITY
    elif state == "confirmed" or (state == "emerging" and duration >= 2) or priority >= 0.72:
        tier = TIER_ELEVATED
    else:
        tier = TIER_ROUTINE

    # Weak custody raises the tier one step — but ONLY for vessels
    # already at priority tier with anomaly evidence.  This prevents
    # background vessels with zone proximity + time-degraded custody
    # from being promoted beyond elevated.
    # Custody-driven escalation: only promote priority → urgent when
    # custody is critically weak AND there is anomaly evidence AND
    # there is reasoning-layer confirmation (not just heuristic).
    if (custody < 0.2 and has_anomaly_evidence
            and state in ("confirmed", "sustained", "critical")
            and tier == TIER_PRIORITY):
        tier = TIER_URGENT

    # ── Determine revisit interval ───────────────────────────────────
    revisit = _REVISIT[tier]
    # Duration pressure: very long anomalies get even shorter revisit
    if duration >= 6 and revisit > 1.0:
        revisit = max(1.0, revisit * 0.5)

    # ── Determine monitoring action ──────────────────────────────────
    if state == "recovering":
        action = ACTION_COOLDOWN
    elif tier == TIER_CRITICAL:
        action = ACTION_INTENSIFY
    elif tier == TIER_URGENT:
        action = ACTION_INTENSIFY
    elif agreement == "confirmed":
        action = ACTION_CONFIRM
    elif tier in (TIER_PRIORITY, TIER_ELEVATED):
        action = ACTION_INCREASE_ATTENTION
    else:
        action = ACTION_MAINTAIN

    # ── Determine sensor preference ──────────────────────────────────
    if tier in (TIER_CRITICAL, TIER_URGENT):
        if custody < 0.4:
            sensor = SENSOR_SAR       # all-weather when custody is weak
        else:
            sensor = SENSOR_HIGH_RES  # best characterization
    elif agreement == "emerging":
        sensor = SENSOR_FAST          # quick confirmation
    elif agreement == "confirmed":
        sensor = SENSOR_HIGH_RES      # detailed characterization
    else:
        sensor = SENSOR_ANY

    # ── Solar-adjusted sensor preference ────────────────────────────
    solar_cond = record.get("solar_condition", "day")
    eo_suit = float(record.get("eo_suitability", 1.0))

    effective_sensor = sensor
    sensor_reason_parts = []

    _is_optical = sensor in (SENSOR_HIGH_RES, SENSOR_FAST)

    if _is_optical and solar_cond == "night":
        # Night: EO is not viable → switch to SAR
        effective_sensor = SENSOR_SAR
        sensor_reason_parts.append(
            f"night conditions (sun {record.get('sun_elevation_deg', '?')}deg); "
            f"EO degraded, SAR preferred"
        )
    elif _is_optical and solar_cond == "twilight":
        if tier in (TIER_CRITICAL, TIER_URGENT):
            # Twilight + urgent/critical: too important to risk degraded EO
            effective_sensor = SENSOR_SAR
            sensor_reason_parts.append(
                f"twilight (EO suitability {eo_suit:.0%}); "
                f"SAR preferred for {tier} collection under marginal light"
            )
        else:
            # Twilight + lower tiers: optical retained but noted as degraded
            sensor_reason_parts.append(
                f"twilight (EO suitability {eo_suit:.0%}); "
                f"optical viable but degraded"
            )
    elif _is_optical:
        sensor_reason_parts.append("daylight supports optical collection")
    elif sensor == SENSOR_SAR:
        sensor_reason_parts.append("SAR selected (all-weather, day/night capable)")
    else:
        sensor_reason_parts.append(f"{solar_cond} conditions; any sensor viable")

    sensor_rationale = "; ".join(sensor_reason_parts) if sensor_reason_parts else ""

    # ── Build rationale ──────────────────────────────────────────────
    parts = []
    if state != "normal":
        parts.append(f"anomaly state: {state}")
    if agreement not in ("normal", state):
        parts.append(f"agreement: {agreement}")
    if duration > 0:
        parts.append(f"duration: {duration}h")
    if custody < 0.5:
        parts.append(f"weak custody ({custody:.2f})")
    if priority >= 0.7:
        parts.append(f"high priority ({priority:.2f})")
    if not parts:
        parts.append("nominal behavior")
    rationale = f"{tier}: " + "; ".join(parts)

    return TaskingPolicy(
        tasking_tier=tier,
        desired_revisit_hours=revisit,
        monitoring_action=action,
        sensor_preference=sensor,
        effective_sensor_preference=effective_sensor,
        rationale=rationale,
        sensor_rationale=sensor_rationale,
    )
