"""Tests for portfolio overview UI helpers."""
import pytest
import pandas as pd

from portfolio_overview import (
    derive_display_status,
    build_overview_df,
    compute_kpi_counts,
    top_attention_targets,
    build_focus_view_state,
    build_label_data,
)


# ── derive_display_status ─────────────────────────────────────────────────────

class TestDeriveDisplayStatus:
    def _r(self, anomaly=0.1, health="HEALTHY", action="NONE",
           neglect_flag=False, deferred_for=None):
        return {
            "anomaly_score": anomaly, "custody_health": health,
            "action": action, "neglect_flag": neglect_flag,
            "deferred_for": deferred_for,
        }

    def test_lost_custody_is_needs_action(self):
        assert derive_display_status(self._r(health="LOST")) == "NEEDS ACTION"

    def test_very_high_anomaly_is_needs_action(self):
        assert derive_display_status(self._r(anomaly=2.0)) == "NEEDS ACTION"

    def test_preempted_action(self):
        assert derive_display_status(self._r(action="PREEMPTED")) == "PREEMPTED"

    def test_deferred_for_set(self):
        assert derive_display_status(self._r(deferred_for="BRAVO-1")) == "PREEMPTED"

    def test_neglect_flag_true(self):
        assert derive_display_status(self._r(neglect_flag=True)) == "NEGLECTED"

    def test_stale_health(self):
        assert derive_display_status(self._r(health="STALE")) == "STALE"

    def test_degrading_health(self):
        assert derive_display_status(self._r(health="DEGRADING")) == "WATCH"

    def test_moderate_anomaly_is_watch(self):
        assert derive_display_status(self._r(anomaly=0.5)) == "WATCH"

    def test_healthy_baseline(self):
        assert derive_display_status(self._r()) == "HEALTHY"

    def test_priority_needs_action_over_neglect(self):
        """NEEDS ACTION should win over NEGLECTED."""
        r = self._r(health="LOST", neglect_flag=True)
        assert derive_display_status(r) == "NEEDS ACTION"

    def test_priority_preempted_over_neglected(self):
        """PREEMPTED should win over NEGLECTED when both apply."""
        r = self._r(action="PREEMPTED", neglect_flag=True)
        assert derive_display_status(r) == "PREEMPTED"

    def test_deferred_for_none_string_not_preempted(self):
        assert derive_display_status(self._r(deferred_for=None)) != "PREEMPTED"

    def test_deferred_for_float_nan_not_preempted(self):
        """pandas NaN in deferred_for must not trigger PREEMPTED status."""
        import math
        assert derive_display_status(self._r(deferred_for=float("nan"))) != "PREEMPTED"

    def test_deferred_for_nan_string_not_preempted(self):
        """String 'nan' in deferred_for must not trigger PREEMPTED status."""
        assert derive_display_status(self._r(deferred_for="nan")) != "PREEMPTED"


# ── build_overview_df ─────────────────────────────────────────────────────────

def _make_ts_df(n=5):
    return pd.DataFrame([{
        "target_id": f"E{i:02d}", "portfolio_rank": i + 1,
        "anomaly_score": float(i) / 10, "custody_health": "HEALTHY",
        "neglect_hours": float(i), "action": "NONE",
        "deferred_for": None, "portfolio_reason": f"Ranked {i+1}/5: routine monitoring.",
        "neglect_flag": i >= 4,
    } for i in range(n)])


class TestBuildOverviewDf:
    def test_returns_dataframe(self):
        df = build_overview_df(_make_ts_df())
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns_present(self):
        df = build_overview_df(_make_ts_df())
        for col in ("Rank", "Entity", "Status", "Anomaly", "Action"):
            assert col in df.columns, f"Missing column: {col}"

    def test_sorted_by_rank(self):
        df = build_overview_df(_make_ts_df())
        assert list(df["Rank"]) == sorted(df["Rank"].tolist())

    def test_row_count(self):
        df = build_overview_df(_make_ts_df(8))
        assert len(df) == 8

    def test_reason_strips_ranked_prefix(self):
        df = build_overview_df(_make_ts_df(1))
        assert not df["Reason"].iloc[0].startswith("Ranked ")


# ── compute_kpi_counts ────────────────────────────────────────────────────────

class TestComputeKpiCounts:
    def test_empty_df_returns_zeros(self):
        counts = compute_kpi_counts(pd.DataFrame())
        assert counts == {"total": 0, "needs_action": 0, "neglected": 0,
                          "stale_or_lost": 0, "preempted": 0}

    def test_total_matches_rows(self):
        counts = compute_kpi_counts(_make_ts_df(10))
        assert counts["total"] == 10

    def test_neglected_count(self):
        df = pd.DataFrame([{
            "target_id": "A", "anomaly_score": 0.1, "custody_health": "HEALTHY",
            "action": "NONE", "neglect_flag": True, "deferred_for": None,
        }, {
            "target_id": "B", "anomaly_score": 0.1, "custody_health": "HEALTHY",
            "action": "NONE", "neglect_flag": False, "deferred_for": None,
        }])
        counts = compute_kpi_counts(df)
        assert counts["neglected"] == 1

    def test_needs_action_count(self):
        df = pd.DataFrame([{
            "target_id": "A", "anomaly_score": 2.5, "custody_health": "DEGRADING",
            "action": "NONE", "neglect_flag": False, "deferred_for": None,
        }, {
            "target_id": "B", "anomaly_score": 0.1, "custody_health": "HEALTHY",
            "action": "NONE", "neglect_flag": False, "deferred_for": None,
        }])
        counts = compute_kpi_counts(df)
        assert counts["needs_action"] == 1

    def test_preempted_count(self):
        df = pd.DataFrame([{
            "target_id": f"E{i}", "anomaly_score": 0.1, "custody_health": "HEALTHY",
            "action": "PREEMPTED" if i < 2 else "NONE",
            "neglect_flag": False, "deferred_for": None,
        } for i in range(5)])
        counts = compute_kpi_counts(df)
        assert counts["preempted"] == 2


# ── top_attention_targets ─────────────────────────────────────────────────────

class TestTopAttentionTargets:
    def test_empty_df(self):
        result = top_attention_targets(pd.DataFrame())
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_returns_at_most_n(self):
        result = top_attention_targets(_make_ts_df(10), n=3)
        assert len(result) <= 3

    def test_sorted_by_rank(self):
        result = top_attention_targets(_make_ts_df(10), n=5)
        if "portfolio_rank" in result.columns:
            ranks = result["portfolio_rank"].tolist()
            assert ranks == sorted(ranks)

    def test_display_status_column_present(self):
        result = top_attention_targets(_make_ts_df(5))
        assert "display_status" in result.columns

    def test_target_id_column_present(self):
        result = top_attention_targets(_make_ts_df(5))
        assert "target_id" in result.columns

    def test_higher_anomaly_surfaces_first_without_rank(self):
        """Without portfolio_rank, sort should fall back to anomaly descending."""
        df = pd.DataFrame([{
            "target_id": f"E{i}", "anomaly_score": float(i) / 10,
            "custody_health": "HEALTHY", "action": "NONE",
            "neglect_flag": False, "deferred_for": None,
        } for i in range(5)])
        result = top_attention_targets(df, n=2)
        # E4 has highest anomaly (0.4), E3 second
        assert result.iloc[0]["target_id"] == "E4"


# ── build_focus_view_state ─────────────────────────────────────────────────────

class TestBuildFocusViewState:
    def _df(self):
        return pd.DataFrame([
            {"target_id": "A", "lat": 1.0, "lon": 2.0, "portfolio_rank": 1},
            {"target_id": "B", "lat": 3.0, "lon": 4.0, "portfolio_rank": 2},
            {"target_id": "C", "lat": 5.0, "lon": 6.0, "portfolio_rank": 3},
        ])

    def test_no_selection_returns_mean_and_default_zoom(self):
        lat, lon, zoom = build_focus_view_state(self._df(), None)
        assert zoom == 7
        assert abs(lat - 3.0) < 0.01
        assert abs(lon - 4.0) < 0.01

    def test_known_entity_returns_its_position(self):
        lat, lon, zoom = build_focus_view_state(self._df(), "B")
        assert lat == 3.0
        assert lon == 4.0
        assert zoom == 9

    def test_unknown_entity_falls_back_to_mean(self):
        lat, lon, zoom = build_focus_view_state(self._df(), "MISSING")
        assert zoom == 7

    def test_focused_zoom_is_tighter_than_default(self):
        _, _, z_sel  = build_focus_view_state(self._df(), "A")
        _, _, z_none = build_focus_view_state(self._df(), None)
        assert z_sel > z_none

    def test_empty_df_no_selection_returns_zeros(self):
        lat, lon, zoom = build_focus_view_state(pd.DataFrame(), None)
        assert lat == 0.0 and lon == 0.0

    def test_custom_zoom_levels(self):
        _, _, z = build_focus_view_state(self._df(), "A", focused_zoom=11, default_zoom=5)
        assert z == 11
        _, _, z2 = build_focus_view_state(self._df(), None, focused_zoom=11, default_zoom=5)
        assert z2 == 5


# ── build_label_data ──────────────────────────────────────────────────────────

class TestBuildLabelData:
    def _df(self, n=6):
        return pd.DataFrame([
            {
                "target_id": f"E{i}", "lat": float(i), "lon": float(i),
                "portfolio_rank": i,
            }
            for i in range(1, n + 1)
        ])

    def test_empty_df_returns_empty(self):
        assert build_label_data(pd.DataFrame()) == []

    def test_top_n_entities_labeled(self):
        rows = build_label_data(self._df(), selected_id=None, top_n=3)
        labels = {r["label"] for r in rows}
        assert labels == {"E1", "E2", "E3"}

    def test_selected_entity_always_included(self):
        rows = build_label_data(self._df(), selected_id="E6", top_n=3)
        labels = {r["label"] for r in rows}
        assert "E6" in labels

    def test_no_duplicates(self):
        rows = build_label_data(self._df(), selected_id="E1", top_n=3)
        labels = [r["label"] for r in rows]
        assert len(labels) == len(set(labels))

    def test_selected_row_has_is_selected_true(self):
        rows = build_label_data(self._df(), selected_id="E2", top_n=5)
        sel = [r for r in rows if r["label"] == "E2"]
        assert len(sel) == 1
        assert sel[0]["is_selected"] is True

    def test_non_selected_rows_have_is_selected_false(self):
        rows = build_label_data(self._df(), selected_id="E2", top_n=3)
        for r in rows:
            if r["label"] != "E2":
                assert r["is_selected"] is False

    def test_rows_have_required_keys(self):
        for r in build_label_data(self._df(), top_n=3):
            assert "lon" in r and "lat" in r and "label" in r
            assert "is_selected" in r and "size" in r

    def test_selected_entity_has_larger_size(self):
        rows = build_label_data(self._df(), selected_id="E3", top_n=5)
        sel_size     = next(r["size"] for r in rows if r["label"] == "E3")
        non_sel_size = next(r["size"] for r in rows if r["label"] != "E3")
        assert sel_size > non_sel_size

    def test_non_selected_entities_share_same_size(self):
        rows = build_label_data(self._df(), selected_id="E1", top_n=5)
        sizes = {r["size"] for r in rows if not r["is_selected"]}
        assert len(sizes) == 1   # all non-selected share the same size

    def test_no_selection_none_marked_selected(self):
        rows = build_label_data(self._df(), selected_id=None, top_n=3)
        assert all(not r["is_selected"] for r in rows)

    def test_max_rows_bounded_by_top_n_plus_selected(self):
        rows = build_label_data(self._df(n=10), selected_id="E9", top_n=3)
        # top-3 (E1, E2, E3) + E9 selected = 4
        assert len(rows) == 4
