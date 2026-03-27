"""
Tests for custody.ml.priority — vessel priority scoring.

Covers:
  1.  sustained ranks higher than brief spike
  2.  confirmed ranks higher than emerging
  3.  low custody increases priority
  4.  critical state produces highest priority
  5.  normal state produces base-level priority
  6.  duration boost increases with hours
  7.  duration boost capped at 0.10
  8.  priority clamped to [0, 1]
  9.  add_priority_scores adds column
 10.  rank_vessels returns top-N sorted descending
 11.  vessel_priority_summary includes state counts
 12.  empty DataFrame handled
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pandas as pd
import pytest

from custody.ml.priority import (
    compute_priority,
    add_priority_scores,
    rank_vessels,
    vessel_priority_summary,
)

UTC = timezone.utc
_T0 = datetime(2024, 9, 20, 0, 0, tzinfo=UTC)


def _rec(
    fused=0.5, ml_combined=0.5, state="normal", agreement="normal",
    duration=0, custody=0.8, escalation=0.0, mmsi=100, hour=0,
):
    return {
        "mmsi": mmsi,
        "timestamp": _T0 + timedelta(hours=hour),
        "fused_score": fused,
        "ml_anomaly_combined": ml_combined,
        "anomaly_state": state,
        "anomaly_agreement": agreement,
        "ml_anomaly_duration_hours": duration,
        "custody_confidence": custody,
        "escalation_boost": escalation,
    }


# ---------------------------------------------------------------------------
# compute_priority
# ---------------------------------------------------------------------------

class TestComputePriority:

    def test_sustained_higher_than_brief(self):
        brief = _rec(state="emerging", duration=1)
        sustained = _rec(state="sustained", duration=5)
        assert compute_priority(sustained) > compute_priority(brief)

    def test_confirmed_higher_than_emerging(self):
        emerging = _rec(state="emerging", agreement="emerging")
        confirmed = _rec(state="confirmed", agreement="confirmed")
        assert compute_priority(confirmed) > compute_priority(emerging)

    def test_low_custody_increases_priority(self):
        good = _rec(custody=0.9)
        weak = _rec(custody=0.2)
        assert compute_priority(weak) > compute_priority(good)

    def test_critical_highest(self):
        normal = _rec(state="normal", agreement="normal", duration=0)
        critical = _rec(state="critical", agreement="confirmed", duration=8, custody=0.2, escalation=0.08)
        assert compute_priority(critical) > compute_priority(normal)
        assert compute_priority(critical) > 0.7

    def test_normal_base_level(self):
        r = _rec(fused=0.2, ml_combined=0.1, state="normal", duration=0, custody=0.9)
        p = compute_priority(r)
        # Should be close to base (0.2) + small custody pressure
        assert p < 0.4

    def test_duration_boost_increases(self):
        short = _rec(duration=1)
        long = _rec(duration=6)
        assert compute_priority(long) > compute_priority(short)

    def test_duration_boost_capped(self):
        r10 = _rec(duration=10)
        r20 = _rec(duration=20)
        # Both above cap → same duration boost
        assert compute_priority(r10) == compute_priority(r20)

    def test_clamped_to_unit(self):
        extreme = _rec(fused=1.0, ml_combined=1.0, state="critical",
                        agreement="confirmed", duration=10, custody=0.0, escalation=0.1)
        assert compute_priority(extreme) <= 1.0

    def test_clamped_above_zero(self):
        low = _rec(fused=0.0, ml_combined=0.0, state="normal",
                    duration=0, custody=1.0, escalation=0.0)
        assert compute_priority(low) >= 0.0


# ---------------------------------------------------------------------------
# DataFrame operations
# ---------------------------------------------------------------------------

class TestDataFrameOps:

    @pytest.fixture
    def df(self):
        rows = []
        for h in range(10):
            rows.append(_rec(fused=0.3, ml_combined=0.3, state="normal", mmsi=100, hour=h))
            rows.append(_rec(fused=0.8, ml_combined=0.9, state="sustained",
                             agreement="confirmed", duration=h, custody=0.3,
                             escalation=0.05, mmsi=200, hour=h))
            rows.append(_rec(fused=0.5, ml_combined=0.5, state="emerging",
                             mmsi=300, hour=h))
        return pd.DataFrame(rows)

    def test_add_priority_scores(self, df):
        result = add_priority_scores(df)
        assert "priority_score" in result.columns
        assert len(result) == len(df)
        assert result["priority_score"].between(0, 1).all()

    def test_rank_vessels_sorted(self, df):
        scored = add_priority_scores(df)
        top = rank_vessels(scored, n=3)
        assert len(top) == 3
        assert top.iloc[0]["max_priority"] >= top.iloc[1]["max_priority"]
        # Vessel 200 (sustained/confirmed) should rank first
        assert top.iloc[0]["mmsi"] == 200

    def test_rank_vessels_top_n(self, df):
        scored = add_priority_scores(df)
        top2 = rank_vessels(scored, n=2)
        assert len(top2) == 2

    def test_summary_includes_state_counts(self, df):
        scored = add_priority_scores(df)
        summary = vessel_priority_summary(scored)
        assert "state_counts" in summary.columns
        assert "dominant_state" in summary.columns
        row200 = summary[summary["mmsi"] == 200].iloc[0]
        assert "sustained" in row200["state_counts"]

    def test_empty_dataframe(self):
        empty = pd.DataFrame(columns=["mmsi", "timestamp", "priority_score",
                                       "anomaly_state", "anomaly_agreement"])
        top = rank_vessels(empty, n=5)
        assert len(top) == 0
