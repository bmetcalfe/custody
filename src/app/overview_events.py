"""Event feed helpers for the portfolio overview page.

Derives a compact list of operator-relevant changes by comparing the current
timestep DataFrame to the previous one.  No Streamlit imports.

Supported event types (listed in descending feed priority)
----------------------------------------------------------
zone_entry          — sensitive_zone score crossed entry threshold
health_worsened     — custody_health degraded (HEALTHY→DEGRADING→STALE→LOST)
neglect_triggered   — neglect_flag newly became True
zone_approach       — entity newly reached zone_probability > 0.5 (APPROACHING status)
rank_change         — portfolio_rank shifted by >= threshold (default 3)
preempted           — entity newly received action=PREEMPTED this step
"""
from __future__ import annotations

import math
from typing import Optional
import pandas as pd

# ── Thresholds ────────────────────────────────────────────────────────────────
_RANK_CHANGE_THRESHOLD: int = 3
_ZONE_ENTRY_THRESHOLD: float = 0.3
_HEALTH_ORDER: dict[str, int] = {
    "HEALTHY": 0, "DEGRADING": 1, "STALE": 2, "LOST": 3
}

# Priority for sorting (lower = higher priority in feed)
_EVENT_PRIORITY: dict[str, int] = {
    "zone_entry":        0,
    "health_worsened":   1,
    "neglect_triggered": 2,
    "zone_approach":     3,
    "rank_change":       4,
    "preempted":         5,
}

_ZONE_APPROACH_THRESHOLD: float = 0.5


def _health_rank(status: str) -> int:
    return _HEALTH_ORDER.get(str(status), -1)


# ── Individual detectors ──────────────────────────────────────────────────────

def detect_rank_change_events(
    current_df: pd.DataFrame,
    prev_df: pd.DataFrame,
    threshold: int = _RANK_CHANGE_THRESHOLD,
) -> list[dict]:
    """Return events where portfolio_rank changed by >= threshold.

    Each event dict contains:
      entity_id, event_type="rank_change", description, delta
      (delta > 0 = improved rank; delta < 0 = worsened rank)
    """
    if (
        "portfolio_rank" not in current_df.columns
        or "portfolio_rank" not in prev_df.columns
        or current_df.empty
        or prev_df.empty
    ):
        return []

    curr = current_df.set_index("target_id")["portfolio_rank"]
    prev = prev_df.set_index("target_id")["portfolio_rank"]
    common = curr.index.intersection(prev.index)

    events: list[dict] = []
    for eid in common:
        c_rank, p_rank = int(curr[eid]), int(prev[eid])
        delta = p_rank - c_rank  # positive = rank number fell = improved
        if abs(delta) >= threshold:
            arrow = "↑" if delta > 0 else "↓"
            events.append({
                "entity_id":   eid,
                "event_type":  "rank_change",
                "description": f"{eid} priority {arrow} (rank {p_rank}→{c_rank})",
                "delta":       delta,
            })

    events.sort(key=lambda e: abs(e["delta"]), reverse=True)
    return events


def detect_health_change_events(
    current_df: pd.DataFrame,
    prev_df: pd.DataFrame,
) -> list[dict]:
    """Return events where custody_health worsened.

    Each event dict contains:
      entity_id, event_type="health_worsened", description, from_health, to_health
    """
    if (
        "custody_health" not in current_df.columns
        or "custody_health" not in prev_df.columns
        or current_df.empty
        or prev_df.empty
    ):
        return []

    curr = current_df.set_index("target_id")["custody_health"]
    prev = prev_df.set_index("target_id")["custody_health"]
    common = curr.index.intersection(prev.index)

    events: list[dict] = []
    for eid in common:
        c_h, p_h = str(curr[eid]), str(prev[eid])
        if _health_rank(c_h) > _health_rank(p_h):
            events.append({
                "entity_id":   eid,
                "event_type":  "health_worsened",
                "description": f"{eid} custody {p_h} → {c_h}",
                "from_health": p_h,
                "to_health":   c_h,
            })
    return events


def detect_neglect_events(
    current_df: pd.DataFrame,
    prev_df: pd.DataFrame,
) -> list[dict]:
    """Return events where neglect_flag newly became True.

    Each event dict contains:
      entity_id, event_type="neglect_triggered", description
    """
    if (
        "neglect_flag" not in current_df.columns
        or "neglect_flag" not in prev_df.columns
        or current_df.empty
        or prev_df.empty
    ):
        return []

    curr = current_df.set_index("target_id")["neglect_flag"].astype(bool)
    prev = prev_df.set_index("target_id")["neglect_flag"].astype(bool)
    common = curr.index.intersection(prev.index)

    events: list[dict] = []
    for eid in common:
        if bool(curr[eid]) and not bool(prev[eid]):
            events.append({
                "entity_id":   eid,
                "event_type":  "neglect_triggered",
                "description": f"{eid} neglect threshold crossed",
            })
    return events


def detect_zone_entry_events(
    current_df: pd.DataFrame,
    prev_df: pd.DataFrame,
    threshold: float = _ZONE_ENTRY_THRESHOLD,
) -> list[dict]:
    """Return events where sensitive_zone score crossed the entry threshold.

    Each event dict contains:
      entity_id, event_type="zone_entry", description
    """
    if (
        "sensitive_zone" not in current_df.columns
        or "sensitive_zone" not in prev_df.columns
        or current_df.empty
        or prev_df.empty
    ):
        return []

    curr = (
        pd.to_numeric(current_df.set_index("target_id")["sensitive_zone"], errors="coerce")
        .fillna(0.0)
    )
    prev = (
        pd.to_numeric(prev_df.set_index("target_id")["sensitive_zone"], errors="coerce")
        .fillna(0.0)
    )
    common = curr.index.intersection(prev.index)

    events: list[dict] = []
    for eid in common:
        if float(curr[eid]) >= threshold and float(prev[eid]) < threshold:
            events.append({
                "entity_id":   eid,
                "event_type":  "zone_entry",
                "description": f"{eid} entered sensitive zone",
            })
    return events


def detect_zone_approach_events(
    current_df: pd.DataFrame,
    prev_df: pd.DataFrame,
    threshold: float = _ZONE_APPROACH_THRESHOLD,
) -> list[dict]:
    """Return events where zone_probability newly crossed the approach threshold.

    Each event dict contains:
      entity_id, event_type="zone_approach", description, zone_probability,
      time_to_zone_hours
    """
    if (
        "zone_probability" not in current_df.columns
        or "zone_probability" not in prev_df.columns
        or current_df.empty
        or prev_df.empty
    ):
        return []

    curr = (
        pd.to_numeric(current_df.set_index("target_id")["zone_probability"], errors="coerce")
        .fillna(0.0)
    )
    prev = (
        pd.to_numeric(prev_df.set_index("target_id")["zone_probability"], errors="coerce")
        .fillna(0.0)
    )
    common = curr.index.intersection(prev.index)

    tte_lookup: dict[str, object] = {}
    if "time_to_zone_hours" in current_df.columns:
        tte_lookup = current_df.set_index("target_id")["time_to_zone_hours"].to_dict()

    events: list[dict] = []
    for eid in common:
        c_zp, p_zp = float(curr[eid]), float(prev[eid])
        if c_zp > threshold and p_zp <= threshold:
            tte = tte_lookup.get(eid)
            try:
                tte_f = float(tte) if tte is not None else None
                tte_str = f" (est. {tte_f:.1f}h)" if tte_f is not None and not math.isnan(tte_f) else ""
            except (TypeError, ValueError):
                tte_str = ""
            events.append({
                "entity_id":        eid,
                "event_type":       "zone_approach",
                "description":      f"{eid} projected to enter zone{tte_str}",
                "zone_probability": c_zp,
                "time_to_zone_hours": tte,
            })
    return events


def detect_preemption_events(
    current_df: pd.DataFrame,
    prev_df: pd.DataFrame,
) -> list[dict]:
    """Return events where an entity is PREEMPTED this step but was not last step.

    Each event dict contains:
      entity_id, event_type="preempted", description, deferred_for
    """
    if "action" not in current_df.columns or current_df.empty:
        return []

    curr_preempted = current_df[current_df["action"] == "PREEMPTED"]
    if curr_preempted.empty:
        return []

    prev_actions: dict[str, str] = {}
    if not prev_df.empty and "action" in prev_df.columns:
        prev_actions = prev_df.set_index("target_id")["action"].to_dict()

    deferred_lookup: dict[str, object] = {}
    if "deferred_for" in current_df.columns:
        deferred_lookup = current_df.set_index("target_id")["deferred_for"].to_dict()

    events: list[dict] = []
    for _, row in curr_preempted.iterrows():
        eid = str(row["target_id"])
        if prev_actions.get(eid) == "PREEMPTED":
            continue  # not new — was already preempted last step
        deferred_for = deferred_lookup.get(eid)
        if deferred_for and str(deferred_for) not in ("None", "nan", ""):
            desc = f"{eid} deferred for {deferred_for}"
        else:
            desc = f"{eid} preempted (no sensor available)"
        events.append({
            "entity_id":   eid,
            "event_type":  "preempted",
            "description": desc,
            "deferred_for": deferred_for,
        })
    return events


# ── Public API ────────────────────────────────────────────────────────────────

def build_event_feed(
    current_df: pd.DataFrame,
    prev_df: Optional[pd.DataFrame],
    max_events: int = 10,
) -> list[dict]:
    """Build a compact event feed by comparing current and previous timesteps.

    Returns up to max_events event dicts, sorted by event priority:
      zone_entry > health_worsened > neglect_triggered > rank_change > preempted

    Each dict contains at minimum: entity_id, event_type, description.
    Returns [] when current_df is empty or prev_df is None/empty.
    """
    if current_df.empty or prev_df is None or prev_df.empty:
        return []

    all_events: list[dict] = []
    all_events.extend(detect_zone_entry_events(current_df, prev_df))
    all_events.extend(detect_health_change_events(current_df, prev_df))
    all_events.extend(detect_neglect_events(current_df, prev_df))
    all_events.extend(detect_zone_approach_events(current_df, prev_df))
    all_events.extend(detect_rank_change_events(current_df, prev_df))
    all_events.extend(detect_preemption_events(current_df, prev_df))

    all_events.sort(key=lambda e: _EVENT_PRIORITY.get(e["event_type"], 99))
    return all_events[:max_events]
