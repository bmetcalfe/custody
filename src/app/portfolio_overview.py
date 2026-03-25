"""UI helpers for the portfolio overview page. No Streamlit imports."""
from __future__ import annotations
from typing import Optional
import pandas as pd


# ── Display status ─────────────────────────────────────────────────────────────

def derive_display_status(record: dict) -> str:
    """Map portfolio record fields to an operator-readable display status.

    Priority (highest wins):
      NEEDS ACTION  — custody LOST, or anomaly_score >= 2.0
      PREEMPTED     — action == PREEMPTED or deferred_for is set
      NEGLECTED     — neglect_flag is True
      STALE         — custody_health in (STALE, LOST) [catches LOST not already caught]
      WATCH         — custody_health == DEGRADING, or anomaly_score >= 0.5
      HEALTHY       — all clear
    """
    anomaly    = float(record.get("anomaly_score", 0.0))
    health     = str(record.get("custody_health", "HEALTHY"))
    action     = str(record.get("action", "NONE"))
    neglect    = bool(record.get("neglect_flag", False))
    deferred   = record.get("deferred_for")
    attention  = str(record.get("attention_state", "ACTIVE_CUSTODY"))

    # pandas NaN / None both mean "no deferred entity"; guard against both
    try:
        _deferred_null = deferred is None or pd.isna(deferred)
    except (TypeError, ValueError):
        _deferred_null = False
    _has_deferred = not _deferred_null and str(deferred) not in ("", "None", "nan")

    if health == "LOST" or anomaly >= 2.0:
        return "NEEDS ACTION"
    if action == "PREEMPTED" or _has_deferred:
        return "PREEMPTED"
    # BACKGROUND vessels are not in active custody — suppress neglect escalation for them
    if neglect and attention != "BACKGROUND":
        return "NEGLECTED"
    if health in ("STALE", "LOST"):
        return "STALE"
    _zone_prob = float(record.get("zone_probability", 0.0))
    _sensitive = float(record.get("sensitive_zone", 0.0))
    if _zone_prob > 0.5 and _sensitive == 0.0:
        return "APPROACHING"
    if health == "DEGRADING" or anomaly >= 0.5:
        return "WATCH"
    return "HEALTHY"


# ── Overview dataframe ─────────────────────────────────────────────────────────

def build_overview_df(timestep_df: pd.DataFrame) -> pd.DataFrame:
    """Build a sorted portfolio table from a single-timestep DataFrame.

    Returns columns: Rank, Entity, Status, Anomaly, Health, Neglect h,
                     Action, Deferred For, Reason
    Sorted by Rank ascending.
    """
    df = timestep_df.copy()
    df["_display_status"] = df.apply(lambda r: derive_display_status(r.to_dict()), axis=1)

    out = pd.DataFrame()
    out["Rank"]       = df.get("portfolio_rank", range(1, len(df) + 1))
    out["Entity"]     = df.get("target_id",      pd.Series(["—"] * len(df)))
    out["Status"]     = df["_display_status"]
    out["Attention"]  = df.get("attention_state", pd.Series(["—"] * len(df)))
    out["Anomaly"]    = pd.to_numeric(df.get("anomaly_score", 0), errors="coerce").round(2)
    out["Health"]     = df.get("custody_health", "—")
    out["Neglect h"]  = pd.to_numeric(df.get("neglect_hours", 0), errors="coerce").round(1)
    out["Action"]     = df.get("action", "—")
    out["Deferred For"] = df.get("deferred_for", pd.Series([None] * len(df))).fillna("—")
    out["Reason"]     = df.get("portfolio_reason", "—").apply(
        lambda s: s[s.find(": ") + 2:] if ": " in str(s) else str(s)  # strip "Ranked N/M: " prefix
    )
    out = out.sort_values("Rank").reset_index(drop=True)
    out.index = range(1, len(out) + 1)
    return out


# ── KPI counts ────────────────────────────────────────────────────────────────

def compute_kpi_counts(timestep_df: pd.DataFrame) -> dict:
    """Compute operator KPI counts for one timestep.

    Returns dict with keys:
      total, needs_action, neglected, stale_or_lost, preempted, approaching
    """
    if timestep_df.empty:
        return {"total": 0, "needs_action": 0, "neglected": 0,
                "stale_or_lost": 0, "preempted": 0, "approaching": 0}

    statuses = timestep_df.apply(lambda r: derive_display_status(r.to_dict()), axis=1)
    actions  = timestep_df.get("action", pd.Series(dtype=str))

    return {
        "total":         len(timestep_df),
        "needs_action":  int((statuses == "NEEDS ACTION").sum()),
        "neglected":     int((statuses == "NEGLECTED").sum()),
        "stale_or_lost": int(statuses.isin(["STALE", "NEEDS ACTION"]).sum()),
        "preempted":     int((actions == "PREEMPTED").sum()),
        "approaching":   int((statuses == "APPROACHING").sum()),
    }


# ── Top-attention helper ───────────────────────────────────────────────────────

def top_attention_targets(timestep_df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Return up to n most urgent entities, sorted by portfolio_rank.

    Falls back to anomaly_score descending if portfolio_rank is absent.
    Returns: DataFrame with columns target_id, display_status, portfolio_reason,
             action, deferred_for, anomaly_score (as available).
    """
    if timestep_df.empty:
        return pd.DataFrame()

    df = timestep_df.copy()
    df["display_status"] = df.apply(lambda r: derive_display_status(r.to_dict()), axis=1)

    if "portfolio_rank" in df.columns:
        df = df.sort_values("portfolio_rank")
    elif "anomaly_score" in df.columns:
        df = df.sort_values("anomaly_score", ascending=False)

    keep = [c for c in ["target_id", "display_status", "portfolio_reason",
                         "action", "deferred_for", "anomaly_score",
                         "custody_health", "neglect_hours", "portfolio_rank"]
            if c in df.columns]
    return df[keep].head(n).reset_index(drop=True)


# ── Map focus helpers ──────────────────────────────────────────────────────────

def build_focus_view_state(
    timestep_df: pd.DataFrame,
    entity_id: Optional[str],
    focused_zoom: int = 9,
    default_zoom: int = 7,
) -> tuple[float, float, int]:
    """Return (center_lat, center_lon, zoom) for the portfolio map.

    When entity_id is provided and present in the data, the view centers on
    that entity at focused_zoom.  Otherwise the mean position of the full
    dataset is used at default_zoom.

    Args:
        timestep_df:   Single-timestep portfolio DataFrame.
        entity_id:     Entity to focus on, or None for the default full view.
        focused_zoom:  Zoom level when an entity is selected (default 9).
        default_zoom:  Zoom level when showing the full portfolio (default 7).

    Returns:
        (center_lat, center_lon, zoom) tuple.
    """
    if entity_id and not timestep_df.empty:
        row = timestep_df[timestep_df["target_id"] == entity_id]
        if not row.empty:
            return float(row.iloc[0]["lat"]), float(row.iloc[0]["lon"]), focused_zoom
    lat = float(timestep_df["lat"].mean()) if not timestep_df.empty else 0.0
    lon = float(timestep_df["lon"].mean()) if not timestep_df.empty else 0.0
    return lat, lon, default_zoom


def build_label_data(
    timestep_df: pd.DataFrame,
    selected_id: Optional[str] = None,
    top_n: int = 5,
) -> list[dict]:
    """Build TextLayer-compatible data rows for the overview map.

    Includes:
    - The explicitly selected entity (if any).
    - The top-N entities by portfolio_rank (or head order if rank absent).

    Duplicate IDs are deduplicated; no entity appears twice.

    Args:
        timestep_df: Single-timestep portfolio DataFrame with lat/lon/target_id.
        selected_id: Entity that is currently focused (always labeled if set).
        top_n:       Number of highest-rank entities to label additionally.

    Returns:
        List of dicts with keys: lon, lat, label, is_selected.
        Empty list when timestep_df is empty.
    """
    if timestep_df.empty:
        return []

    label_ids: list[str] = []
    if selected_id:
        label_ids.append(selected_id)

    sort_col = "portfolio_rank" if "portfolio_rank" in timestep_df.columns else None
    top_rows = (
        timestep_df.sort_values(sort_col).head(top_n)
        if sort_col else timestep_df.head(top_n)
    )
    for eid in top_rows["target_id"].tolist():
        if eid not in label_ids:
            label_ids.append(eid)

    label_set = set(label_ids)
    rows: list[dict] = []
    for _, row in timestep_df.iterrows():
        eid = str(row["target_id"])
        if eid in label_set:
            is_sel = eid == selected_id
            rows.append({
                "lon":         float(row["lon"]),
                "lat":         float(row["lat"]),
                "label":       eid,
                "is_selected": is_sel,
                "size":        13 if is_sel else 10,  # selected entity labeled slightly larger
            })
    return rows
