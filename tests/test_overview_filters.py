"""Tests for overview_filters helpers."""
from __future__ import annotations

import pytest
import pandas as pd

from overview_filters import apply_overview_filters, filter_top_n, ALL_STATUSES


# ── Fixtures / helpers ────────────────────────────────────────────────────────

def _make_df(n: int = 6) -> pd.DataFrame:
    """Build a minimal timestep DataFrame with varied statuses."""
    rows = []
    health_cycle = ["HEALTHY", "HEALTHY", "DEGRADING", "STALE", "LOST", "DEGRADING"]
    for i in range(n):
        rows.append({
            "target_id":      f"E{i:02d}",
            "portfolio_rank": i + 1,
            "anomaly_score":  float(i) * 0.3,
            "custody_health": health_cycle[i % len(health_cycle)],
            "neglect_flag":   i >= 4,
            "action":         "PREEMPTED" if i == 3 else "NONE",
            "deferred_for":   None,
            "is_scripted":    i < 2,
        })
    return pd.DataFrame(rows)


# ── apply_overview_filters ────────────────────────────────────────────────────

class TestApplyOverviewFilters:

    def test_no_filters_returns_all(self):
        df = _make_df(6)
        result = apply_overview_filters(df)
        assert len(result) == 6

    def test_empty_df_returns_empty(self):
        result = apply_overview_filters(pd.DataFrame())
        assert result.empty

    def test_status_filter_healthy_only(self):
        df = _make_df(6)
        result = apply_overview_filters(df, status_filter=["HEALTHY"])
        # Only rows whose derived display_status == "HEALTHY" survive
        assert len(result) > 0
        from portfolio_overview import derive_display_status
        statuses = result.apply(lambda r: derive_display_status(r.to_dict()), axis=1)
        assert set(statuses.unique()) == {"HEALTHY"}

    def test_status_filter_all_statuses_returns_all(self):
        """Passing all status values is equivalent to no filter."""
        df = _make_df(6)
        result = apply_overview_filters(df, status_filter=list(ALL_STATUSES))
        assert len(result) == len(df)

    def test_status_filter_none_returns_all(self):
        df = _make_df(6)
        result = apply_overview_filters(df, status_filter=None)
        assert len(result) == len(df)

    def test_status_filter_empty_list_returns_empty(self):
        """Empty status list matches nothing."""
        df = _make_df(6)
        result = apply_overview_filters(df, status_filter=[])
        # empty list != ALL_STATUSES, but no rows can match an empty set
        assert len(result) == 0

    def test_scripted_only(self):
        df = _make_df(6)
        result = apply_overview_filters(df, scripted_only=True)
        assert len(result) == 2  # only E00 and E01 are scripted
        assert set(result["target_id"]) == {"E00", "E01"}

    def test_scripted_only_without_column(self):
        """scripted_only is silently ignored when is_scripted column is absent."""
        df = _make_df(6).drop(columns=["is_scripted"])
        result = apply_overview_filters(df, scripted_only=True)
        assert len(result) == 6

    def test_neglected_only(self):
        df = _make_df(6)
        result = apply_overview_filters(df, neglected_only=True)
        assert len(result) == 2  # E04 and E05 have neglect_flag=True
        assert all(result["neglect_flag"])

    def test_neglected_only_without_column(self):
        df = _make_df(6).drop(columns=["neglect_flag"])
        result = apply_overview_filters(df, neglected_only=True)
        assert len(result) == 6

    def test_stale_lost_only(self):
        df = _make_df(6)
        result = apply_overview_filters(df, stale_lost_only=True)
        assert all(result["custody_health"].isin(["STALE", "LOST"]))
        assert len(result) > 0

    def test_stale_lost_only_without_column(self):
        df = _make_df(6).drop(columns=["custody_health"])
        result = apply_overview_filters(df, stale_lost_only=True)
        assert len(result) == 6

    def test_scripted_and_neglected_and_logic(self):
        """scripted_only + neglected_only = intersection (fewer rows than either alone)."""
        df = _make_df(6)
        only_scripted  = apply_overview_filters(df, scripted_only=True)
        only_neglected = apply_overview_filters(df, neglected_only=True)
        both = apply_overview_filters(df, scripted_only=True, neglected_only=True)
        assert len(both) <= len(only_scripted)
        assert len(both) <= len(only_neglected)

    def test_result_index_is_reset(self):
        df = _make_df(6)
        result = apply_overview_filters(df, neglected_only=True)
        assert list(result.index) == list(range(len(result)))

    def test_original_df_not_mutated(self):
        df = _make_df(6)
        _ = apply_overview_filters(df, status_filter=["HEALTHY"])
        assert "_display_status" not in df.columns


# ── filter_top_n ──────────────────────────────────────────────────────────────

class TestFilterTopN:

    def test_empty_df_returns_empty(self):
        assert filter_top_n(pd.DataFrame(), n=5).empty

    def test_top_n_none_returns_all(self):
        df = _make_df(8)
        result = filter_top_n(df, n=None)
        assert len(result) == 8

    def test_top_n_zero_returns_all(self):
        df = _make_df(8)
        result = filter_top_n(df, n=0)
        assert len(result) == 8

    def test_top_5_returns_5(self):
        df = _make_df(8)
        result = filter_top_n(df, n=5)
        assert len(result) == 5

    def test_top_n_larger_than_df_returns_all(self):
        df = _make_df(3)
        result = filter_top_n(df, n=10)
        assert len(result) == 3

    def test_sorted_by_rank_ascending(self):
        df = _make_df(6)
        result = filter_top_n(df, n=None)
        ranks = result["portfolio_rank"].tolist()
        assert ranks == sorted(ranks)

    def test_top_n_contains_lowest_rank_numbers(self):
        """Top 3 should have rank 1, 2, 3 (not 4, 5, 6)."""
        df = _make_df(6)
        result = filter_top_n(df, n=3)
        assert set(result["portfolio_rank"]) == {1, 2, 3}

    def test_no_portfolio_rank_column_preserves_row_order(self):
        """Falls back gracefully when portfolio_rank is absent."""
        df = _make_df(4).drop(columns=["portfolio_rank"])
        result = filter_top_n(df, n=2)
        assert len(result) == 2

    def test_result_index_is_reset(self):
        df = _make_df(8)
        result = filter_top_n(df, n=4)
        assert list(result.index) == list(range(len(result)))
