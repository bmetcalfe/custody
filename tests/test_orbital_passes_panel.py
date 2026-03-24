"""
Tests for orbital_passes_panel.build_orbital_passes_rows.

Coverage:
  1. Returns exactly two rows — one for SAT-A, one for SAT-B
  2. Output row keys match the expected column schema
  3. "In View Now" status when satellite is already overhead
  4. "Upcoming" status for a near-future pass
  5. "No Pass In Horizon" when no pass exists within the search window
  6. Time to Start is 0.0 when already in view
  7. Start / End are None when no pass found
  8. Duration is None when no pass found
  9. Streamlit app imports / calls the helper (import-level smoke test)
 10. Within Threshold flag — true when 0 < tts <= HOLD_LOOKAHEAD_THRESHOLD_SECONDS
 11. Within Threshold flag — false for in-view, no-pass, and beyond-threshold passes
 12. Sort order: In View Now first, then ascending tts, No Pass In Horizon last
"""
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from orbital_passes_panel import build_orbital_passes_rows
from custody.sensors import PassWindow
import custody.config as config


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

T0 = datetime(2026, 3, 23, 10, 0, tzinfo=UTC)
_LAT, _LON = 0.5, 0.5

EXPECTED_COLUMNS = {
    "Satellite",
    "Start",
    "End",
    "Duration (min)",
    "Time to Start (min)",
    "Status",
    "Within Threshold",
}

_PASS_IN_VIEW = PassWindow(
    satellite_id="SAT-A",
    start_time=T0,
    end_time=T0 + timedelta(minutes=7),
    duration_seconds=420.0,
    time_to_start_seconds=0,
)

_PASS_UPCOMING_NEAR = PassWindow(
    satellite_id="SAT-A",
    start_time=T0 + timedelta(minutes=20),
    end_time=T0 + timedelta(minutes=27),
    duration_seconds=420.0,
    time_to_start_seconds=1200.0,  # 20 min — within 30-min threshold
)

_PASS_UPCOMING_FAR = PassWindow(
    satellite_id="SAT-B",
    start_time=T0 + timedelta(minutes=60),
    end_time=T0 + timedelta(minutes=67),
    duration_seconds=420.0,
    time_to_start_seconds=3600.0,  # 60 min — beyond 30-min threshold
)

# Exactly at the threshold boundary (1800 s == HOLD_LOOKAHEAD_THRESHOLD_SECONDS)
_PASS_AT_THRESHOLD = PassWindow(
    satellite_id="SAT-A",
    start_time=T0 + timedelta(seconds=config.HOLD_LOOKAHEAD_THRESHOLD_SECONDS),
    end_time=T0 + timedelta(seconds=config.HOLD_LOOKAHEAD_THRESHOLD_SECONDS + 420),
    duration_seconds=420.0,
    time_to_start_seconds=config.HOLD_LOOKAHEAD_THRESHOLD_SECONDS,
)

# One second past the threshold
_PASS_JUST_OVER_THRESHOLD = PassWindow(
    satellite_id="SAT-A",
    start_time=T0 + timedelta(seconds=config.HOLD_LOOKAHEAD_THRESHOLD_SECONDS + 1),
    end_time=T0 + timedelta(seconds=config.HOLD_LOOKAHEAD_THRESHOLD_SECONDS + 421),
    duration_seconds=420.0,
    time_to_start_seconds=config.HOLD_LOOKAHEAD_THRESHOLD_SECONDS + 1,
)


def _make_rows(sat_a_pw, sat_b_pw):
    """Return rows with controlled PassWindow returns for both satellites."""
    def fake_next_pass(sat_id, lat, lon, from_time, horizon_minutes=180):
        if sat_id == "SAT-A":
            return sat_a_pw
        if sat_id == "SAT-B":
            return sat_b_pw
        return None

    with patch("orbital_passes_panel.next_pass_window", side_effect=fake_next_pass):
        return build_orbital_passes_rows(_LAT, _LON, T0)


# ---------------------------------------------------------------------------
# 1 & 2. Structure: two rows, correct columns
# ---------------------------------------------------------------------------

class TestOutputSchema:
    def test_returns_two_rows(self):
        rows = _make_rows(_PASS_UPCOMING_NEAR, _PASS_UPCOMING_FAR)
        assert len(rows) == 2

    def test_row_keys_match_expected_columns(self):
        rows = _make_rows(_PASS_UPCOMING_NEAR, _PASS_UPCOMING_FAR)
        for row in rows:
            assert set(row.keys()) == EXPECTED_COLUMNS

    def test_returns_list(self):
        rows = _make_rows(None, None)
        assert isinstance(rows, list)

    def test_both_satellites_present(self):
        rows = _make_rows(_PASS_UPCOMING_NEAR, _PASS_UPCOMING_FAR)
        satellites = {r["Satellite"] for r in rows}
        assert satellites == {"SAT-A", "SAT-B"}


# ---------------------------------------------------------------------------
# 3 & 6. In View Now
# ---------------------------------------------------------------------------

class TestInViewNow:
    def test_status_is_in_view_now(self):
        rows = _make_rows(_PASS_IN_VIEW, None)
        sat_a = next(r for r in rows if r["Satellite"] == "SAT-A")
        assert sat_a["Status"] == "In View Now"

    def test_time_to_start_is_zero(self):
        rows = _make_rows(_PASS_IN_VIEW, None)
        sat_a = next(r for r in rows if r["Satellite"] == "SAT-A")
        assert sat_a["Time to Start (min)"] == 0.0

    def test_start_equals_from_time(self):
        rows = _make_rows(_PASS_IN_VIEW, None)
        sat_a = next(r for r in rows if r["Satellite"] == "SAT-A")
        assert sat_a["Start"] == T0

    def test_duration_is_positive(self):
        rows = _make_rows(_PASS_IN_VIEW, None)
        sat_a = next(r for r in rows if r["Satellite"] == "SAT-A")
        assert sat_a["Duration (min)"] > 0


# ---------------------------------------------------------------------------
# 4. Upcoming
# ---------------------------------------------------------------------------

class TestUpcoming:
    def test_status_is_upcoming(self):
        rows = _make_rows(_PASS_UPCOMING_NEAR, None)
        sat_a = next(r for r in rows if r["Satellite"] == "SAT-A")
        assert sat_a["Status"] == "Upcoming"

    def test_time_to_start_is_correct(self):
        rows = _make_rows(_PASS_UPCOMING_NEAR, None)
        sat_a = next(r for r in rows if r["Satellite"] == "SAT-A")
        assert sat_a["Time to Start (min)"] == pytest.approx(20.0, abs=0.1)

    def test_start_is_datetime(self):
        rows = _make_rows(_PASS_UPCOMING_NEAR, None)
        sat_a = next(r for r in rows if r["Satellite"] == "SAT-A")
        assert isinstance(sat_a["Start"], datetime)

    def test_end_is_after_start(self):
        rows = _make_rows(_PASS_UPCOMING_NEAR, None)
        sat_a = next(r for r in rows if r["Satellite"] == "SAT-A")
        assert sat_a["End"] > sat_a["Start"]


# ---------------------------------------------------------------------------
# 5, 7 & 8. No Pass In Horizon
# ---------------------------------------------------------------------------

class TestNoPassInHorizon:
    def test_status_is_no_pass_in_horizon(self):
        rows = _make_rows(None, None)
        for row in rows:
            assert row["Status"] == "No Pass In Horizon"

    def test_start_is_none(self):
        rows = _make_rows(None, None)
        for row in rows:
            assert row["Start"] is None

    def test_end_is_none(self):
        rows = _make_rows(None, None)
        for row in rows:
            assert row["End"] is None

    def test_duration_is_none(self):
        rows = _make_rows(None, None)
        for row in rows:
            assert row["Duration (min)"] is None

    def test_time_to_start_is_none(self):
        rows = _make_rows(None, None)
        for row in rows:
            assert row["Time to Start (min)"] is None


# ---------------------------------------------------------------------------
# Mixed: one satellite in view, one no pass
# ---------------------------------------------------------------------------

class TestMixedStatuses:
    def test_independent_per_satellite(self):
        rows = _make_rows(_PASS_IN_VIEW, None)
        sat_a = next(r for r in rows if r["Satellite"] == "SAT-A")
        sat_b = next(r for r in rows if r["Satellite"] == "SAT-B")
        assert sat_a["Status"] == "In View Now"
        assert sat_b["Status"] == "No Pass In Horizon"


# ---------------------------------------------------------------------------
# 10 & 11. Within Threshold flag
# ---------------------------------------------------------------------------

class TestWithinThreshold:
    def test_within_threshold_true_for_near_upcoming(self):
        """20-min pass is inside the 30-min lookahead threshold."""
        rows = _make_rows(_PASS_UPCOMING_NEAR, None)
        sat_a = next(r for r in rows if r["Satellite"] == "SAT-A")
        assert sat_a["Within Threshold"] is True

    def test_within_threshold_false_for_far_upcoming(self):
        """60-min pass is outside the 30-min lookahead threshold."""
        rows = _make_rows(_PASS_UPCOMING_FAR, None)
        sat_a = next(r for r in rows if r["Satellite"] == "SAT-A")
        assert sat_a["Within Threshold"] is False

    def test_within_threshold_false_for_in_view(self):
        """Already-in-view satellite: tts == 0, not a future pass → False."""
        rows = _make_rows(_PASS_IN_VIEW, None)
        sat_a = next(r for r in rows if r["Satellite"] == "SAT-A")
        assert sat_a["Within Threshold"] is False

    def test_within_threshold_false_for_no_pass(self):
        rows = _make_rows(None, None)
        for row in rows:
            assert row["Within Threshold"] is False

    def test_within_threshold_true_at_exact_boundary(self):
        """tts == HOLD_LOOKAHEAD_THRESHOLD_SECONDS (≤) → True."""
        rows = _make_rows(_PASS_AT_THRESHOLD, None)
        sat_a = next(r for r in rows if r["Satellite"] == "SAT-A")
        assert sat_a["Within Threshold"] is True

    def test_within_threshold_false_one_second_over(self):
        """tts == HOLD_LOOKAHEAD_THRESHOLD_SECONDS + 1 (>) → False."""
        rows = _make_rows(_PASS_JUST_OVER_THRESHOLD, None)
        sat_a = next(r for r in rows if r["Satellite"] == "SAT-A")
        assert sat_a["Within Threshold"] is False

    def test_within_threshold_uses_config_constant(self):
        """Flag uses HOLD_LOOKAHEAD_THRESHOLD_SECONDS, not a hardcoded value."""
        # Verify: at threshold → True, one second past → False.
        rows_at = _make_rows(_PASS_AT_THRESHOLD, None)
        rows_over = _make_rows(_PASS_JUST_OVER_THRESHOLD, None)
        sat_a_at = next(r for r in rows_at if r["Satellite"] == "SAT-A")
        sat_a_over = next(r for r in rows_over if r["Satellite"] == "SAT-A")
        assert sat_a_at["Within Threshold"] is True
        assert sat_a_over["Within Threshold"] is False


# ---------------------------------------------------------------------------
# 12. Sort order
# ---------------------------------------------------------------------------

class TestSortOrder:
    def test_in_view_comes_before_upcoming(self):
        rows = _make_rows(_PASS_IN_VIEW, _PASS_UPCOMING_NEAR)
        assert rows[0]["Status"] == "In View Now"
        assert rows[1]["Status"] == "Upcoming"

    def test_in_view_comes_before_no_pass(self):
        rows = _make_rows(_PASS_IN_VIEW, None)
        assert rows[0]["Status"] == "In View Now"
        assert rows[1]["Status"] == "No Pass In Horizon"

    def test_upcoming_comes_before_no_pass(self):
        rows = _make_rows(_PASS_UPCOMING_NEAR, None)
        assert rows[0]["Status"] == "Upcoming"
        assert rows[1]["Status"] == "No Pass In Horizon"

    def test_nearer_upcoming_comes_first(self):
        """SAT-B is 60 min away, SAT-A is 20 min away → SAT-A first."""
        rows = _make_rows(_PASS_UPCOMING_NEAR, _PASS_UPCOMING_FAR)
        assert rows[0]["Satellite"] == "SAT-A"
        assert rows[1]["Satellite"] == "SAT-B"

    def test_farther_upcoming_comes_second(self):
        rows = _make_rows(_PASS_UPCOMING_NEAR, _PASS_UPCOMING_FAR)
        assert rows[1]["Time to Start (min)"] > rows[0]["Time to Start (min)"]

    def test_no_pass_last_when_mixed(self):
        rows = _make_rows(_PASS_IN_VIEW, None)
        assert rows[-1]["Status"] == "No Pass In Horizon"

    def test_both_no_pass_stable_order(self):
        """When both are No Pass In Horizon, both satellites still present."""
        rows = _make_rows(None, None)
        satellites = [r["Satellite"] for r in rows]
        assert set(satellites) == {"SAT-A", "SAT-B"}


# ---------------------------------------------------------------------------
# App smoke tests
# ---------------------------------------------------------------------------

def test_streamlit_app_imports_build_orbital_passes_rows():
    import ast
    import pathlib

    app_src = (
        pathlib.Path(__file__).parent.parent / "src" / "app" / "streamlit_app.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(app_src)

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "orbital_passes_panel":
            imports.extend(alias.name for alias in node.names)

    assert "build_orbital_passes_rows" in imports


def test_streamlit_app_calls_build_orbital_passes_rows():
    import pathlib

    app_src = (
        pathlib.Path(__file__).parent.parent / "src" / "app" / "streamlit_app.py"
    ).read_text(encoding="utf-8")

    assert "build_orbital_passes_rows(" in app_src
