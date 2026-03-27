"""Filter helpers for the portfolio overview page.

No Streamlit imports — pure dataframe logic.
"""
from __future__ import annotations

import pandas as pd

ALL_STATUSES: list[str] = [
    "NEEDS ACTION", "APPROACHING", "WATCH", "HEALTHY"
]


def apply_overview_filters(
    df: pd.DataFrame,
    status_filter: list[str] | None = None,
    scripted_only: bool = False,
    neglected_only: bool = False,
    stale_lost_only: bool = False,
) -> pd.DataFrame:
    """Return rows matching all active filters (AND logic).

    Filter order:
      1. display_status in status_filter  (if provided and not all statuses)
      2. is_scripted == True              (if scripted_only)
      3. neglect_flag == True             (if neglected_only)
      4. custody_health in STALE/LOST     (if stale_lost_only)

    Filters referencing absent columns are silently skipped.
    Returns a copy; original index is reset.
    """
    if df.empty:
        return df.copy()

    out = df.copy()

    # Status filter
    if status_filter is not None and set(status_filter) != set(ALL_STATUSES):
        from portfolio_overview import derive_display_status  # local import avoids circular

        if "_display_status" not in out.columns:
            out["_display_status"] = out.apply(
                lambda r: derive_display_status(r.to_dict()), axis=1
            )
        out = out[out["_display_status"].isin(status_filter)]

    # Scripted toggle
    if scripted_only and "is_scripted" in out.columns:
        out = out[out["is_scripted"].astype(bool)]

    # Neglect toggle
    if neglected_only and "neglect_flag" in out.columns:
        out = out[out["neglect_flag"].astype(bool)]

    # Stale/Lost toggle
    if stale_lost_only and "custody_health" in out.columns:
        out = out[out["custody_health"].isin(["STALE", "LOST"])]

    return out.reset_index(drop=True)


def filter_top_n(df: pd.DataFrame, n: int | None) -> pd.DataFrame:
    """Return the top-n rows sorted by portfolio_rank ascending.

    If n is None or <= 0, all rows are returned (still sorted by rank).
    Falls back to existing row order when portfolio_rank is absent.
    """
    if df.empty:
        return df.copy()

    if "portfolio_rank" in df.columns:
        out = df.sort_values("portfolio_rank")
    else:
        out = df.copy()

    if n is not None and n > 0:
        out = out.head(n)

    return out.reset_index(drop=True)
