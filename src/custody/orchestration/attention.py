"""
Selective custody attention model.

Defines three attention tiers that gate how strongly neglect and custody
pressure influence portfolio scoring.  A human operator can also impose a
minimum tier via a tracking directive, ensuring a vessel stays in active
custody even when its current behaviour is nominal.

Tiers (ordered least → most attention)
---------------------------------------
BACKGROUND     — routine traffic; no persistent custody expectation.
                 Neglect pressure is zero; background vessels do not pollute
                 the portfolio ranking simply because they have not been
                 collected recently.
WATCHLIST      — elevated interest; light revisit value.
                 Neglect contributes at a reduced weight.
ACTIVE_CUSTODY — deliberate continuity; full neglect and custody pressure
                 applies.  Used for scripted anomalous actors and manually
                 designated vessels.

Tracking directives
--------------------
NONE             — system-driven tier only.
MAINTAIN_CUSTODY — operator floor; vessel stays at least ACTIVE_CUSTODY
                   regardless of current anomaly or zone score.

Public API
----------
derive_attention_state(anomaly_score, custody_confidence, uncertainty_km, zone_score)
    -> str   (behavior-driven tier, ignoring any directive)

apply_tracking_directive_floor(attention_state, tracking_directive)
    -> str   (apply operator floor; raises tier if directive demands it)

apply_dark_vessel_floor(attention_state, dark_flag, tracking_directive)
    -> str   (raise to ACTIVE_CUSTODY for operationally relevant dark vessels)

compute_neglect_weight(attention_state)
    -> float [0, 1]   (multiplier on the raw neglect scoring component)

explain_attention_state(attention_state, tracking_directive, anomaly_score,
                        zone_score, custody_confidence, dark_flag=False)
    -> str   (one-sentence explanation)
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Tier and directive constants
# ---------------------------------------------------------------------------

BACKGROUND     = "BACKGROUND"
WATCHLIST      = "WATCHLIST"
ACTIVE_CUSTODY = "ACTIVE_CUSTODY"

_TIER_ORDER: list[str] = [BACKGROUND, WATCHLIST, ACTIVE_CUSTODY]

DIRECTIVE_NONE             = "NONE"
DIRECTIVE_MAINTAIN_CUSTODY = "MAINTAIN_CUSTODY"

# ---------------------------------------------------------------------------
# Behaviour thresholds (all tunable in one place)
# ---------------------------------------------------------------------------

# Anomaly score thresholds
_ANOMALY_ACTIVE_CUSTODY = 1.5   # >= → ACTIVE_CUSTODY
_ANOMALY_WATCHLIST      = 0.5   # >= → WATCHLIST

# Sensitive-zone score thresholds
_ZONE_ACTIVE_CUSTODY = 0.5   # >= → ACTIVE_CUSTODY
_ZONE_WATCHLIST      = 0.2   # >= → WATCHLIST

# Custody confidence thresholds (lower = worse)
_CONF_ACTIVE_CUSTODY = 0.35  # < → ACTIVE_CUSTODY
_CONF_WATCHLIST      = 0.65  # < → WATCHLIST

# ---------------------------------------------------------------------------
# Neglect weight per tier
# ---------------------------------------------------------------------------

_NEGLECT_WEIGHT: dict[str, float] = {
    BACKGROUND:     0.0,   # zero neglect pressure — background traffic is not custody-tracked
    WATCHLIST:      0.3,   # light contribution
    ACTIVE_CUSTODY: 1.0,   # full neglect pressure
}


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def derive_attention_state(
    anomaly_score: float,
    custody_confidence: float,
    uncertainty_km: float = 0.0,
    zone_score: float = 0.0,
) -> str:
    """Derive the behavior-driven attention tier for one entity.

    Does NOT apply tracking directives; call apply_tracking_directive_floor
    afterwards if a directive exists.

    Args:
        anomaly_score:       Current anomaly signal (0 – 3).
        custody_confidence:  Track confidence in [0, 1].
        uncertainty_km:      Positional uncertainty (km); not used in tier
                             logic directly but available for future extension.
        zone_score:          Sensitive-zone membership score in [0, 1].

    Returns:
        One of: BACKGROUND, WATCHLIST, ACTIVE_CUSTODY.
    """
    if (
        anomaly_score      >= _ANOMALY_ACTIVE_CUSTODY
        or zone_score      >= _ZONE_ACTIVE_CUSTODY
        or custody_confidence < _CONF_ACTIVE_CUSTODY
    ):
        return ACTIVE_CUSTODY

    if (
        anomaly_score      >= _ANOMALY_WATCHLIST
        or zone_score      >= _ZONE_WATCHLIST
        or custody_confidence < _CONF_WATCHLIST
    ):
        return WATCHLIST

    return BACKGROUND


def _raise_tier_floor(state: str, floor: str) -> str:
    """Return the higher of *state* and *floor* in the tier order."""
    current_idx = _TIER_ORDER.index(state) if state in _TIER_ORDER else 0
    floor_idx   = _TIER_ORDER.index(floor)  if floor  in _TIER_ORDER else 0
    return _TIER_ORDER[max(current_idx, floor_idx)]


def apply_tracking_directive_floor(
    attention_state: str,
    tracking_directive: str,
) -> str:
    """Apply an operator directive as a minimum attention floor.

    MAINTAIN_CUSTODY raises the floor to ACTIVE_CUSTODY if the behavior-
    driven tier would otherwise be lower.  NONE is a no-op.

    Args:
        attention_state:    Behavior-driven tier (output of derive_attention_state).
        tracking_directive: NONE or MAINTAIN_CUSTODY.

    Returns:
        Adjusted attention tier (>= input tier).
    """
    if tracking_directive == DIRECTIVE_MAINTAIN_CUSTODY:
        return _raise_tier_floor(attention_state, ACTIVE_CUSTODY)
    return attention_state


def apply_dark_vessel_floor(
    attention_state: str,
    dark_flag: bool,
    tracking_directive: str,
) -> str:
    """Raise attention to ACTIVE_CUSTODY for operationally relevant dark vessels.

    A vessel that has gone AIS-dark is a custody emergency when it is already
    being tracked (WATCHLIST / ACTIVE_CUSTODY) or under a MAINTAIN_CUSTODY
    directive.  Background vessels with no directive remain BACKGROUND even
    when dark — transient AIS gaps for untracked traffic are routine.

    Args:
        attention_state:    Current tier after directive floor has been applied.
        dark_flag:          True if the vessel's AIS has gone silent.
        tracking_directive: Operator directive (NONE | MAINTAIN_CUSTODY).

    Returns:
        Adjusted attention tier (>= input tier).
    """
    if not dark_flag:
        return attention_state
    relevant = (
        tracking_directive == DIRECTIVE_MAINTAIN_CUSTODY
        or attention_state in (WATCHLIST, ACTIVE_CUSTODY)
    )
    if relevant:
        return _raise_tier_floor(attention_state, ACTIVE_CUSTODY)
    return attention_state


def compute_neglect_weight(attention_state: str) -> float:
    """Return the neglect scoring weight [0, 1] for the given tier.

    BACKGROUND     → 0.0  (zero neglect pressure)
    WATCHLIST      → 0.3  (reduced pressure)
    ACTIVE_CUSTODY → 1.0  (full pressure)
    Unknown tiers default to 1.0 (safe/conservative).
    """
    return _NEGLECT_WEIGHT.get(attention_state, 1.0)


def explain_attention_state(
    attention_state: str,
    tracking_directive: str,
    anomaly_score: float,
    zone_score: float,
    custody_confidence: float,
    dark_flag: bool = False,
) -> str:
    """Return a one-sentence explanation of this entity's attention tier.

    Covers directive-driven, dark-vessel, and behavior-driven reasons.
    """
    parts: list[str] = []

    if dark_flag:
        parts.append("AIS dark (transponder loss)")

    if tracking_directive == DIRECTIVE_MAINTAIN_CUSTODY:
        parts.append("operator directive (MAINTAIN_CUSTODY)")

    if anomaly_score >= _ANOMALY_ACTIVE_CUSTODY:
        parts.append(f"elevated anomaly ({anomaly_score:.2f})")
    elif anomaly_score >= _ANOMALY_WATCHLIST:
        parts.append(f"anomaly ({anomaly_score:.2f})")

    if zone_score >= _ZONE_ACTIVE_CUSTODY:
        parts.append("in sensitive zone")
    elif zone_score >= _ZONE_WATCHLIST:
        parts.append("near sensitive zone")

    if custody_confidence < _CONF_ACTIVE_CUSTODY:
        parts.append(f"low confidence ({custody_confidence:.2f})")

    if not parts:
        parts.append("routine behavior")

    return f"{attention_state}: " + "; ".join(parts)
