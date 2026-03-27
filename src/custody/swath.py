"""Sensor footprint / swath geometry and grouped collection valuation.

Models collection footprints as axis-aligned rectangles centered on a
target position, with sensor-type-specific dimensions.  Uses local
planar approximation (km → degrees at the target latitude) — accurate
enough for ISR footprints (<100 km) but not for global geometry.

Public API
----------
SwathSpec (frozen dataclass)
    width_km, length_km — footprint dimensions.

SWATH_SPECS: dict[str, SwathSpec]
    Per-sensor-type default footprint dimensions.

footprint_bbox(lat, lon, spec) -> tuple[float, float, float, float]
    Compute (min_lat, max_lat, min_lon, max_lon) for a footprint.

targets_in_footprint(center_lat, center_lon, spec, targets) -> list[dict]
    Return targets that fall within the footprint.

SwathAssessment (frozen dataclass)
    Grouped collection valuation for one candidate footprint.

assess_swath(center_record, sensor_type, all_timestep_records) -> SwathAssessment
    Compute the grouped value of a candidate footprint.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import custody.config as config


# ---------------------------------------------------------------------------
# Swath specifications per sensor type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SwathSpec:
    """Footprint dimensions for one sensor class.

    Attributes:
        width_km:  Cross-track swath width in km.
        length_km: Along-track strip length in km.
    """
    width_km: float
    length_km: float


# Realistic but simplified defaults:
#   EO high-res: narrow swath (~15 km), medium strip (~50 km)
#   SAR:         medium swath (~40 km), long strip (~80 km)
#   Fast revisit: moderate swath (~25 km), moderate strip (~40 km)
SWATH_SPECS: dict[str, SwathSpec] = {
    "high_resolution": SwathSpec(width_km=15.0, length_km=50.0),
    "all_weather":     SwathSpec(width_km=40.0, length_km=80.0),
    "fast_revisit":    SwathSpec(width_km=25.0, length_km=40.0),
}

# Fallback for unknown sensor types
_DEFAULT_SPEC = SwathSpec(width_km=20.0, length_km=40.0)


# ---------------------------------------------------------------------------
# Local coordinate conversion
# ---------------------------------------------------------------------------

def _km_per_deg_lat() -> float:
    """Approximate km per degree of latitude (constant)."""
    return 111.0


def _km_per_deg_lon(lat: float) -> float:
    """Approximate km per degree of longitude at given latitude."""
    return 111.0 * math.cos(math.radians(lat))


# ---------------------------------------------------------------------------
# Footprint geometry
# ---------------------------------------------------------------------------

def footprint_bbox(
    lat: float,
    lon: float,
    spec: SwathSpec,
) -> tuple[float, float, float, float]:
    """Compute axis-aligned bounding box for a footprint.

    Args:
        lat, lon: Center of the footprint (degrees).
        spec:     Swath dimensions.

    Returns:
        (min_lat, max_lat, min_lon, max_lon) in degrees.
    """
    half_len_deg = (spec.length_km / 2.0) / _km_per_deg_lat()
    km_per_lon = _km_per_deg_lon(lat)
    if km_per_lon <= 0:
        km_per_lon = 1.0  # pole guard
    half_wid_deg = (spec.width_km / 2.0) / km_per_lon

    return (
        lat - half_len_deg,
        lat + half_len_deg,
        lon - half_wid_deg,
        lon + half_wid_deg,
    )


def targets_in_footprint(
    center_lat: float,
    center_lon: float,
    spec: SwathSpec,
    targets: list[dict],
) -> list[dict]:
    """Return target records that fall within the footprint.

    Args:
        center_lat, center_lon: Footprint center.
        spec:      Swath dimensions.
        targets:   List of record dicts with ``lat`` and ``lon`` fields.

    Returns:
        List of target dicts inside the footprint (may include the
        center target itself).
    """
    min_lat, max_lat, min_lon, max_lon = footprint_bbox(center_lat, center_lon, spec)
    return [
        t for t in targets
        if min_lat <= float(t["lat"]) <= max_lat
        and min_lon <= float(t["lon"]) <= max_lon
    ]


# ---------------------------------------------------------------------------
# Grouped collection valuation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SwathAssessment:
    """Grouped collection value for one candidate footprint.

    Attributes:
        center_target_id:     The target this footprint is centered on.
        sensor_type:          Sensor class used for swath dimensions.
        swath_width_km:       Footprint width.
        swath_length_km:      Footprint length.
        covered_target_ids:   IDs of all targets inside the footprint.
        covered_target_count: Number of targets covered.
        covered_priority_sum: Sum of priority_score for covered targets.
        single_target_value:  Value if only the center target were collected.
        swath_task_value:     Grouped collection value.
        value_uplift:         swath_task_value - single_target_value.
        rationale:            Human-readable explanation.
    """
    center_target_id: str
    sensor_type: str
    swath_width_km: float
    swath_length_km: float
    covered_target_ids: tuple[str, ...]
    covered_target_count: int
    covered_priority_sum: float
    mean_covered_confidence: float
    single_target_value: float
    swath_task_value: float
    value_uplift: float
    rationale: str


def _target_value(record: dict) -> float:
    """Extract the best available priority value from a record.

    Prefers confidence_adjusted_priority (set by confidence module),
    then portfolio_score, then priority_score, then anomaly-based
    fallback.  This ensures grouped value reflects both importance
    and confidence in that importance.
    """
    for key in ("confidence_adjusted_priority", "portfolio_score", "priority_score"):
        v = record.get(key)
        if v is not None and float(v) > 0:
            return float(v)
    return float(record.get("anomaly_score", 0.0)) / 1.5


def assess_swath(
    center_record: dict,
    sensor_type: str,
    all_timestep_records: list[dict],
) -> SwathAssessment:
    """Compute the grouped value of a candidate footprint.

    The footprint is centered on ``center_record`` and sized by the
    sensor type.  All other targets within the footprint contribute
    additional value proportional to their priority.

    Value formula:
        single_value = _target_value(center_record)
        bonus = min(
            sum(covered_other_value * SWATH_BONUS_WEIGHT),
            SWATH_UPLIFT_CAP
        )
        swath_value = single_value + bonus

    The bonus is capped by ``config.SWATH_UPLIFT_CAP`` so grouped
    value never completely overwhelms the center target logic.

    Args:
        center_record:         The primary target record.
        sensor_type:           Sensor class (determines footprint size).
        all_timestep_records:  All target records at this timestep.

    Returns:
        A frozen :class:`SwathAssessment`.
    """
    spec = SWATH_SPECS.get(sensor_type, _DEFAULT_SPEC)
    center_id = center_record.get("target_id", "unknown")
    center_lat = float(center_record.get("lat", 0.0))
    center_lon = float(center_record.get("lon", 0.0))

    covered = targets_in_footprint(center_lat, center_lon, spec, all_timestep_records)
    covered_ids = tuple(str(t.get("target_id", "?")) for t in covered)
    others = [t for t in covered if t.get("target_id") != center_id]

    # Value computation using confidence-adjusted priority when available
    single_value = _target_value(center_record)
    raw_bonus = sum(_target_value(t) * config.SWATH_BONUS_WEIGHT for t in others)
    capped_bonus = min(raw_bonus, config.SWATH_UPLIFT_CAP)
    swath_value = round(single_value + capped_bonus, 4)
    uplift = round(capped_bonus, 4)

    # Priority sum and mean confidence for all covered targets
    pri_sum = round(sum(_target_value(t) for t in covered), 4)
    conf_values = [float(t.get("overall_confidence", 1.0)) for t in covered]
    mean_conf = round(sum(conf_values) / len(conf_values), 3) if conf_values else 1.0

    # Rationale
    if len(others) > 0:
        high_pri = [t for t in others if _target_value(t) >= 0.5]
        parts = [f"swath covers {len(covered)} targets"]
        if high_pri:
            parts.append(f"including {len(high_pri)} priority vessel(s)")
        parts.append(f"value uplift +{uplift:.3f}")
        if raw_bonus > config.SWATH_UPLIFT_CAP:
            parts.append(f"(capped from {raw_bonus:.3f})")
        if mean_conf < 0.5:
            parts.append(f"low confidence across covered targets ({mean_conf:.2f})")
        elif mean_conf >= 0.75:
            parts.append(f"high-confidence grouping ({mean_conf:.2f})")
        rationale = "; ".join(parts)
    else:
        rationale = "single-target collect (no additional targets in footprint)"

    return SwathAssessment(
        center_target_id=center_id,
        sensor_type=sensor_type,
        swath_width_km=spec.width_km,
        swath_length_km=spec.length_km,
        covered_target_ids=covered_ids,
        covered_target_count=len(covered),
        covered_priority_sum=pri_sum,
        mean_covered_confidence=mean_conf,
        single_target_value=round(single_value, 4),
        swath_task_value=swath_value,
        value_uplift=uplift,
        rationale=rationale,
    )
