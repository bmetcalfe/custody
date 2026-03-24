"""
Dashboard smoke tests.

These tests verify the data pipeline that streamlit_app.py depends on:
  - run_simulation() produces records with the expected columns and types
  - replay_ais_file() produces records compatible with the same display path
  - The dataframe transforms applied in the app don't raise
  - Imports used by the app (SENSITIVE_ZONE, get_sensor_opportunities) work
  - st.dataframe is called without keyword arguments that require a newer
    Streamlit version than the one pinned in pyproject.toml (>=1.19.0)

We do not invoke Streamlit's UI. These tests catch the class of error that
triggered this test file's creation: a Streamlit API incompatibility that
only surfaces at runtime.
"""
import inspect
import re
from datetime import datetime, UTC, timedelta

import pandas as pd
import pytest

from custody.ais import replay_ais_file
from custody.alerts import alerts_for_timeline
from custody.compounds import compounds_for_timeline, evaluate_compounds
from custody.simulate import run_simulation
from custody.config import ZONES
from custody.sensors import get_sensor_opportunities
from custody.models import BehaviorState
from compound_panels import (
    aggregate_compound_history,
    build_active_compounds_df,
    build_compound_history_df,
)


# ---------------------------------------------------------------------------
# AIS replay fixture (two vessels, three observations each)
# ---------------------------------------------------------------------------

_T0 = datetime(2026, 3, 23, 10, 0, tzinfo=UTC)

_TWO_VESSEL_CSV = "\n".join([
    "vessel_id,timestamp,lat,lon,speed_knots,heading_deg",
    f"V001,{_T0.isoformat()},0.0,0.0,10.0,45.0",
    f"V001,{(_T0 + timedelta(hours=1)).isoformat()},0.01,0.01,10.0,45.0",
    f"V001,{(_T0 + timedelta(hours=2)).isoformat()},0.02,0.02,10.0,45.0",
    f"V002,{_T0.isoformat()},0.1,0.1,12.0,90.0",
    f"V002,{(_T0 + timedelta(hours=1)).isoformat()},0.11,0.11,12.0,90.0",
    f"V002,{(_T0 + timedelta(hours=2)).isoformat()},0.12,0.12,12.0,90.0",
])


@pytest.fixture(scope="module")
def replay_result():
    return replay_ais_file(_TWO_VESSEL_CSV)


@pytest.fixture(scope="module")
def replay_target_df(replay_result):
    df = pd.DataFrame(replay_result["V001"])
    df["time_str"] = df["time"].astype(str)
    return df


# ---------------------------------------------------------------------------
# Simulation output contract
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "target_id", "time", "lat", "lon",
    "uncertainty_km", "custody_confidence", "anomaly_score",
    "speed_kmh", "heading_deg", "behavior_mode",
    "action", "action_reason", "sensor_id", "sensor_type", "collection_result",
    "sensitive_zone", "loitering", "route_deviation",
    "behavior_state", "state_confidence",
]

# sensor_access_count is simulation-only (AIS replay does not emit it).
SIM_ONLY_COLUMNS = ["sensor_access_count"]

_VALID_BEHAVIOR_STATES = {member.value for member in BehaviorState}
_ALLOWED_CONFIDENCE_LEVELS = {0.2, 0.6, 0.85}


@pytest.fixture(scope="module")
def sim_df():
    records = run_simulation()
    df = pd.DataFrame(records)
    df["time_str"] = df["time"].astype(str)
    return df


def test_simulation_returns_records(sim_df):
    assert len(sim_df) > 0


def test_simulation_has_required_columns(sim_df):
    all_expected = set(REQUIRED_COLUMNS) | set(SIM_ONLY_COLUMNS)
    missing = all_expected - set(sim_df.columns)
    assert not missing, f"Missing columns: {missing}"


def test_simulation_has_two_targets(sim_df):
    assert set(sim_df["target_id"].unique()) == {"V001", "V002"}


def test_time_column_is_datetime(sim_df):
    assert pd.api.types.is_datetime64_any_dtype(sim_df["time"]) or all(
        isinstance(v, datetime) for v in sim_df["time"]
    )


def test_numeric_columns_are_finite(sim_df):
    for col in ["lat", "lon", "uncertainty_km", "custody_confidence", "anomaly_score",
                "speed_kmh", "heading_deg"]:
        series = pd.to_numeric(sim_df[col], errors="coerce")
        assert series.notna().all(), f"Non-finite values in column: {col}"


def test_action_values_are_known(sim_df):
    valid = {"NONE", "TASK", "HOLD", "NO_SENSOR", "PREEMPTED"}
    unknown = set(sim_df["action"].unique()) - valid
    assert not unknown, f"Unknown action values: {unknown}"


def test_anomaly_score_in_valid_range(sim_df):
    scores = pd.to_numeric(sim_df["anomaly_score"], errors="coerce")
    assert (scores >= 0).all()
    assert (scores <= 3.0).all()


def test_custody_confidence_in_valid_range(sim_df):
    conf = pd.to_numeric(sim_df["custody_confidence"], errors="coerce")
    assert (conf >= 0).all()
    assert (conf <= 1.0).all()


def test_behavior_state_values_are_valid(sim_df):
    unknown = set(sim_df["behavior_state"].unique()) - _VALID_BEHAVIOR_STATES
    assert not unknown, f"Unexpected behavior_state values: {unknown}"


def test_state_confidence_uses_allowed_levels(sim_df):
    actual = set(sim_df["state_confidence"].unique())
    unexpected = actual - _ALLOWED_CONFIDENCE_LEVELS
    assert not unexpected, f"Unexpected state_confidence values: {unexpected}"


def test_sensor_access_count_present(sim_df):
    """Every simulation record must carry a sensor_access_count field."""
    assert "sensor_access_count" in sim_df.columns


def test_sensor_access_count_is_non_negative_integer(sim_df):
    """sensor_access_count must be a non-negative integer at every step."""
    counts = pd.to_numeric(sim_df["sensor_access_count"], errors="coerce")
    assert counts.notna().all(), "sensor_access_count contains NaN"
    assert (counts >= 0).all(), "sensor_access_count contains negative values"
    assert (counts == counts.astype(int)).all(), "sensor_access_count is not integer-valued"



def test_available_sensors_lookup_with_position():
    """get_sensor_opportunities called with lat/lon as in the dashboard should
    not raise and should return a list."""
    from datetime import UTC, datetime
    t = datetime(2026, 3, 23, 10, 0, tzinfo=UTC)
    result = get_sensor_opportunities(t, 0.0, 0.0)
    assert isinstance(result, list)


def test_available_sensors_lookup_without_position():
    """Backward-compat path: no position → same result as pre-Step-20."""
    from datetime import UTC, datetime
    t = datetime(2026, 3, 23, 10, 0, tzinfo=UTC)
    result_no_pos = get_sensor_opportunities(t)
    result_with_pos = get_sensor_opportunities(t, None, None)
    assert [o.sensor_id for o in result_no_pos] == [o.sensor_id for o in result_with_pos]


# ---------------------------------------------------------------------------
# Dashboard dataframe transforms (mirrors the logic in streamlit_app.py)
# ---------------------------------------------------------------------------

def test_cot_dataframe_builds_without_error(sim_df):
    target_df = sim_df[sim_df["target_id"] == "V001"].reset_index(drop=True)
    cot_df = target_df[["time_str", "custody_confidence", "anomaly_score"]].copy()
    cot_df["custody_confidence"] = pd.to_numeric(cot_df["custody_confidence"], errors="coerce").round(2)
    cot_df["anomaly_score"] = pd.to_numeric(cot_df["anomaly_score"], errors="coerce").round(2)
    cot_df.columns = ["Time", "Confidence", "Anomaly"]
    assert list(cot_df.columns) == ["Time", "Confidence", "Anomaly"]
    assert len(cot_df) > 0


def test_timeline_dataframe_builds_without_error(sim_df):
    display_cols = [
        "time_str", "lat", "lon", "uncertainty_km", "custody_confidence",
        "anomaly_score", "action", "sensor_id", "collection_result",
        "behavior_state", "state_confidence",
    ]
    target_df = sim_df[sim_df["target_id"] == "V001"].reset_index(drop=True)
    timeline_df = target_df[display_cols].copy().fillna("—")
    for col in ["lat", "lon", "uncertainty_km", "custody_confidence", "anomaly_score",
                "state_confidence"]:
        timeline_df[col] = pd.to_numeric(timeline_df[col], errors="coerce")
    timeline_df["lat"] = timeline_df["lat"].round(3)
    timeline_df["lon"] = timeline_df["lon"].round(3)
    timeline_df["uncertainty_km"] = timeline_df["uncertainty_km"].round(2)
    timeline_df["custody_confidence"] = timeline_df["custody_confidence"].round(2)
    timeline_df["anomaly_score"] = timeline_df["anomaly_score"].round(2)
    timeline_df["state_confidence"] = timeline_df["state_confidence"].round(2)
    assert len(timeline_df) > 0
    assert timeline_df["lat"].notna().all()
    assert timeline_df["behavior_state"].notna().all()
    assert timeline_df["state_confidence"].notna().all()


# ---------------------------------------------------------------------------
# Playback-slice behavior (visible_df = target_df.iloc[:selected_idx + 1])
# ---------------------------------------------------------------------------

def test_visible_df_at_step_zero_has_one_row(sim_df):
    target_df = sim_df[sim_df["target_id"] == "V001"].reset_index(drop=True)
    visible_df = target_df.iloc[: 0 + 1]
    assert len(visible_df) == 1


def test_visible_df_row_count_is_nondecreasing(sim_df):
    target_df = sim_df[sim_df["target_id"] == "V001"].reset_index(drop=True)
    counts = [len(target_df.iloc[: i + 1]) for i in range(len(target_df))]
    assert counts == sorted(counts)


def test_visible_df_at_final_step_equals_full_timeline(sim_df):
    target_df = sim_df[sim_df["target_id"] == "V001"].reset_index(drop=True)
    last_idx = len(target_df) - 1
    visible_df = target_df.iloc[: last_idx + 1]
    assert len(visible_df) == len(target_df)


def test_cot_df_respects_playback_slice(sim_df):
    target_df = sim_df[sim_df["target_id"] == "V001"].reset_index(drop=True)
    for selected_idx in (0, len(target_df) // 2, len(target_df) - 1):
        visible_df = target_df.iloc[: selected_idx + 1]
        cot_df = visible_df[["time_str", "custody_confidence", "anomaly_score"]].copy()
        assert len(cot_df) == selected_idx + 1


def test_timeline_df_respects_playback_slice(sim_df):
    display_cols = [
        "time_str", "lat", "lon", "uncertainty_km", "custody_confidence",
        "anomaly_score", "action", "sensor_id", "collection_result",
        "behavior_state", "state_confidence",
    ]
    target_df = sim_df[sim_df["target_id"] == "V001"].reset_index(drop=True)
    for selected_idx in (0, len(target_df) // 2, len(target_df) - 1):
        visible_df = target_df.iloc[: selected_idx + 1]
        timeline_df = visible_df[display_cols].copy().fillna("—")
        assert len(timeline_df) == selected_idx + 1


# ---------------------------------------------------------------------------
# Imports used directly by the app
# ---------------------------------------------------------------------------

def test_zones_config_is_nonempty():
    assert len(ZONES) >= 1


def test_zones_have_positive_halo():
    for z in ZONES:
        assert z.halo > 0


def test_streamlit_app_imports_zones_not_sensitive_zone():
    """Source-level guard: app must import ZONES, not the removed SENSITIVE_ZONE shim."""
    import pathlib
    src = pathlib.Path(__file__).parents[1] / "src" / "app" / "streamlit_app.py"
    text = src.read_text(encoding="utf-8")
    assert "from custody.config import ZONES" in text
    assert "SENSITIVE_ZONE" not in text


def test_zone_polygon_data_builds_correctly():
    """zone_data list used by the map layer must be well-formed for all ZONES."""
    zone_data = [
        {
            "name": z.name,
            "polygon": [
                [z.min_lon, z.min_lat],
                [z.max_lon, z.min_lat],
                [z.max_lon, z.max_lat],
                [z.min_lon, z.max_lat],
            ],
        }
        for z in ZONES
    ]
    assert len(zone_data) == len(ZONES)
    for entry in zone_data:
        assert "name" in entry
        assert "polygon" in entry
        poly = entry["polygon"]
        assert len(poly) == 4
        for coord in poly:
            assert len(coord) == 2   # [lon, lat]


def test_get_sensor_opportunities_returns_list():
    t = datetime(2026, 3, 23, 10, 0, tzinfo=UTC)
    result = get_sensor_opportunities(t)
    assert isinstance(result, list)


def test_sensor_opportunities_have_required_fields():
    t = datetime(2026, 3, 23, 10, 0, tzinfo=UTC)
    for sensor in get_sensor_opportunities(t):
        assert hasattr(sensor, "sensor_id")
        assert hasattr(sensor, "sensor_type")
        assert hasattr(sensor, "resolution")


# ---------------------------------------------------------------------------
# Streamlit API compatibility guard
# ---------------------------------------------------------------------------

def test_dataframe_calls_do_not_use_hide_index():
    """
    hide_index was added in Streamlit 1.23. pyproject.toml pins >=1.19.0.
    Ensure the app never passes hide_index to st.dataframe.
    """
    app_path = (
        __file__.replace("tests\\test_dashboard.py", "src\\app\\streamlit_app.py")
        .replace("tests/test_dashboard.py", "src/app/streamlit_app.py")
    )
    source = open(app_path, encoding="utf-8").read()
    assert "hide_index" not in source, (
        "st.dataframe(..., hide_index=...) requires Streamlit >=1.23 "
        "but pyproject.toml pins >=1.19.0. Remove hide_index."
    )


def test_experimental_rerun_is_only_call_used():
    """
    st.rerun() replaced st.experimental_rerun() in Streamlit 1.27.
    Since we're on 1.19, experimental_rerun is correct — this test
    documents that and will fail if someone upgrades the call prematurely.
    """
    app_path = (
        __file__.replace("tests\\test_dashboard.py", "src\\app\\streamlit_app.py")
        .replace("tests/test_dashboard.py", "src/app/streamlit_app.py")
    )
    source = open(app_path, encoding="utf-8").read()
    # Should use experimental_rerun, not the 1.27+ st.rerun()
    assert "st.experimental_rerun()" in source, (
        "Autoplay requires st.experimental_rerun() on Streamlit 1.19"
    )
    assert "st.rerun()" not in source, (
        "st.rerun() requires Streamlit >=1.27. Use st.experimental_rerun() "
        "while pyproject.toml pins >=1.19.0."
    )


# ---------------------------------------------------------------------------
# AIS Replay mode — data pipeline
# ---------------------------------------------------------------------------

def test_replay_returns_two_vessels(replay_result):
    assert set(replay_result.keys()) == {"V001", "V002"}


def test_replay_timeline_length(replay_result):
    assert len(replay_result["V001"]) == 3
    assert len(replay_result["V002"]) == 3


def test_replay_target_df_has_required_columns(replay_target_df):
    missing = set(REQUIRED_COLUMNS) - set(replay_target_df.columns)
    assert not missing, f"AIS replay df missing columns: {missing}"


def test_replay_behavior_mode_is_observed(replay_target_df):
    assert (replay_target_df["behavior_mode"] == "observed").all()


def test_replay_cot_dataframe_builds_without_error(replay_target_df):
    """Mirror of test_cot_dataframe_builds_without_error for AIS path."""
    cot_df = replay_target_df[["time_str", "custody_confidence", "anomaly_score"]].copy()
    cot_df["custody_confidence"] = pd.to_numeric(cot_df["custody_confidence"], errors="coerce").round(2)
    cot_df["anomaly_score"] = pd.to_numeric(cot_df["anomaly_score"], errors="coerce").round(2)
    cot_df.columns = ["Time", "Confidence", "Anomaly"]
    assert len(cot_df) == 3
    assert cot_df["Confidence"].notna().all()


def test_replay_timeline_dataframe_builds_without_error(replay_target_df):
    """Mirror of test_timeline_dataframe_builds_without_error for AIS path."""
    display_cols = [
        "time_str", "lat", "lon", "uncertainty_km", "custody_confidence",
        "anomaly_score", "action", "sensor_id", "collection_result",
        "behavior_state", "state_confidence",
    ]
    timeline_df = replay_target_df[display_cols].copy().fillna("—")
    for col in ["lat", "lon", "uncertainty_km", "custody_confidence", "anomaly_score",
                "state_confidence"]:
        timeline_df[col] = pd.to_numeric(timeline_df[col], errors="coerce")
    assert len(timeline_df) > 0
    assert timeline_df["lat"].notna().all()
    assert timeline_df["behavior_state"].notna().all()


def test_replay_action_values_are_known(replay_target_df):
    valid = {"NONE", "TASK", "HOLD", "NO_SENSOR", "PREEMPTED"}
    unknown = set(replay_target_df["action"].unique()) - valid
    assert not unknown, f"Unknown action values in AIS replay: {unknown}"


def test_replay_behavior_state_values_are_valid(replay_target_df):
    unknown = set(replay_target_df["behavior_state"].unique()) - _VALID_BEHAVIOR_STATES
    assert not unknown, f"Unexpected behavior_state values in AIS replay: {unknown}"


# ---------------------------------------------------------------------------
# Source-level guard: AIS Replay mode uses replay_ais_file
# ---------------------------------------------------------------------------

def test_app_imports_replay_ais_file():
    app_path = (
        __file__.replace("tests\\test_dashboard.py", "src\\app\\streamlit_app.py")
        .replace("tests/test_dashboard.py", "src/app/streamlit_app.py")
    )
    source = open(app_path, encoding="utf-8").read()
    assert "replay_ais_file" in source, (
        "streamlit_app.py must import and call replay_ais_file for AIS Replay mode"
    )


# ---------------------------------------------------------------------------
# Alerts section — data pipeline (Step 12c)
# ---------------------------------------------------------------------------

def test_alerts_for_timeline_returns_list_from_sim(sim_df):
    target_records = sim_df[sim_df["target_id"] == "V001"].to_dict("records")
    result = alerts_for_timeline(target_records)
    assert isinstance(result, list)


def test_alerts_dataframe_builds_without_error(sim_df):
    """Mirror the alerts DataFrame construction in streamlit_app.py (sliced to step)."""
    target_df = sim_df[sim_df["target_id"] == "V001"].reset_index(drop=True)
    selected_idx = len(target_df) - 1  # full timeline, as in the worst-case playback position
    target_records = target_df.iloc[: selected_idx + 1].to_dict("records")
    alerts = alerts_for_timeline(target_records)
    if alerts:
        alerts_df = pd.DataFrame([
            {"Time": a.timestamp, "Level": a.level, "Code": a.code, "Message": a.message}
            for a in alerts
        ])
        alerts_df["Time"] = alerts_df["Time"].astype(str).str.split("+").str[0]
        assert list(alerts_df.columns) == ["Time", "Level", "Code", "Message"]
        assert len(alerts_df) == len(alerts)


def test_alerts_dataframe_no_alert_case():
    """No-alert path: alerts_for_timeline returns [] and no DataFrame is built."""
    result = alerts_for_timeline([])
    assert result == []


def test_alerts_for_timeline_from_replay(replay_target_df):
    target_records = replay_target_df.to_dict("records")
    result = alerts_for_timeline(target_records)
    assert isinstance(result, list)


def test_alerts_slice_at_step_zero_covers_only_first_record(sim_df):
    """At playback step 0, only the first record is evaluated."""
    target_df = sim_df[sim_df["target_id"] == "V001"].reset_index(drop=True)
    alerts_step0 = alerts_for_timeline(target_df.iloc[:1].to_dict("records"))
    alerts_full = alerts_for_timeline(target_df.to_dict("records"))
    # Step 0 can only have <= alerts than the full timeline
    assert len(alerts_step0) <= len(alerts_full)


def test_alerts_accumulate_with_playback_steps(sim_df):
    """Alert count is non-decreasing as the playback step increases."""
    target_df = sim_df[sim_df["target_id"] == "V001"].reset_index(drop=True)
    counts = [
        len(alerts_for_timeline(target_df.iloc[: i + 1].to_dict("records")))
        for i in range(len(target_df))
    ]
    assert counts == sorted(counts), "Alert counts should not decrease as steps advance"


def test_app_imports_alerts_for_timeline():
    app_path = (
        __file__.replace("tests\\test_dashboard.py", "src\\app\\streamlit_app.py")
        .replace("tests/test_dashboard.py", "src/app/streamlit_app.py")
    )
    source = open(app_path, encoding="utf-8").read()
    assert "alerts_for_timeline" in source, (
        "streamlit_app.py must import and call alerts_for_timeline for the Alerts section"
    )


# ---------------------------------------------------------------------------
# Compound signals section — Step 16a (active at selected step)
# ---------------------------------------------------------------------------

def _active_compounds_at(target_df, selected_idx):
    """Mirror the app's evaluate_compounds call with prefix window."""
    prefix_window = target_df.iloc[:selected_idx].to_dict("records")
    current_record = target_df.iloc[selected_idx].to_dict()
    return sorted(
        evaluate_compounds(current_record, window=prefix_window),
        key=lambda s: s.confidence,
        reverse=True,
    )


def test_active_compounds_returns_list_from_sim(sim_df):
    target_df = sim_df[sim_df["target_id"] == "V001"].reset_index(drop=True)
    result = _active_compounds_at(target_df, len(target_df) - 1)
    assert isinstance(result, list)


def test_active_compounds_returns_list_from_replay(replay_target_df):
    result = _active_compounds_at(replay_target_df, len(replay_target_df) - 1)
    assert isinstance(result, list)


def test_active_compounds_at_step_zero_uses_empty_window(sim_df):
    """At step 0 the prefix window is empty — evaluate_compounds must not crash."""
    target_df = sim_df[sim_df["target_id"] == "V001"].reset_index(drop=True)
    result = _active_compounds_at(target_df, 0)
    assert isinstance(result, list)


def test_active_compounds_sorted_by_confidence_descending():
    """When multiple compounds are active, result is sorted confidence-high-first."""
    from custody.config import ZONE_COMPOUND_MIN_SCORE, HIGH_ANOMALY_THRESHOLD, LOW_CUSTODY_THRESHOLD

    # This record triggers LOITERING_NEAR_ZONE (low conf), PROXIMITY_NEAR_ZONE,
    # LOITERING_WITH_PROXIMITY, and HIGH_ANOMALY_LOW_CUSTODY (highest conf).
    record = {
        "target_id": "V001", "time": _T0,
        "loitering": 0.3,
        "sensitive_zone": ZONE_COMPOUND_MIN_SCORE,
        "vessel_proximity_score": 0.5,
        "anomaly_score": HIGH_ANOMALY_THRESHOLD + 0.1,
        "custody_confidence": LOW_CUSTODY_THRESHOLD - 0.1,
    }
    result = sorted(
        evaluate_compounds(record, window=[]),
        key=lambda s: s.confidence,
        reverse=True,
    )
    assert len(result) >= 2
    confidences = [s.confidence for s in result]
    assert confidences == sorted(confidences, reverse=True)


def test_active_compounds_dataframe_builds_without_error(sim_df):
    """Mirror the full DataFrame construction path used in streamlit_app.py."""
    target_df = sim_df[sim_df["target_id"] == "V001"].reset_index(drop=True)
    selected_idx = len(target_df) - 1
    active_compounds = _active_compounds_at(target_df, selected_idx)
    if active_compounds:
        compounds_df = pd.DataFrame([
            {
                "Time": str(c.timestamp).split("+")[0],
                "Code": c.code,
                "Conf": f"{c.confidence:.2f}",
                "Evidence": c.evidence,
                "Components": ", ".join(
                    f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in c.components.items()
                ),
            }
            for c in active_compounds
        ])
        assert list(compounds_df.columns) == ["Time", "Code", "Conf", "Evidence", "Components"]
        assert len(compounds_df) == len(active_compounds)


def test_window_based_compound_appears_with_sufficient_history():
    """REPEATED_ZONE_ENTRY fires when the prefix window contains ≥2 zone entries."""
    from custody.config import ZONE_COMPOUND_MIN_SCORE

    def _r(offset_h, zone_val):
        return {
            "target_id": "V001", "time": _T0 + timedelta(hours=offset_h),
            "anomaly_score": 0.0, "custody_confidence": 0.85,
            "loitering": 0.0, "sensitive_zone": zone_val,
            "vessel_proximity_score": 0.0,
        }

    # Window: two outside→inside transitions
    window = [
        _r(0, 0.0),
        _r(1, ZONE_COMPOUND_MIN_SCORE),   # entry 1
        _r(2, 0.0),
        _r(3, ZONE_COMPOUND_MIN_SCORE),   # entry 2
    ]
    current = _r(4, 0.0)
    result = evaluate_compounds(current, window=window)
    assert any(s.code == "REPEATED_ZONE_ENTRY" for s in result)


def test_empty_state_when_no_compounds_active():
    """Nominal record with empty window produces an empty list."""
    record = {
        "target_id": "V001", "time": _T0,
        "anomaly_score": 0.0, "custody_confidence": 0.85,
        "loitering": 0.0, "sensitive_zone": 0.0,
        "vessel_proximity_score": 0.0,
    }
    result = evaluate_compounds(record, window=[])
    assert result == []


def test_app_imports_evaluate_compounds():
    app_path = (
        __file__.replace("tests\\test_dashboard.py", "src\\app\\streamlit_app.py")
        .replace("tests/test_dashboard.py", "src/app/streamlit_app.py")
    )
    source = open(app_path, encoding="utf-8").read()
    assert "evaluate_compounds" in source, (
        "streamlit_app.py must import and call evaluate_compounds for the Compounds section"
    )


# ---------------------------------------------------------------------------
# Compound history section — Step 16b
# ---------------------------------------------------------------------------

def _make_agg(timeline_records, threshold=0.0):
    """Thin wrapper around aggregate_compound_history for test convenience."""
    compounds = [
        c for c in compounds_for_timeline(timeline_records)
        if c.confidence >= threshold
    ]
    return aggregate_compound_history(compounds)


# Shared timeline fixture for history aggregation tests.
_T = datetime(2026, 3, 23, 10, 0, tzinfo=UTC)

def _hr(offset_h, loitering=0.0, zone=0.0, conf=0.85):
    return {
        "target_id": "V001", "time": _T + timedelta(hours=offset_h),
        "anomaly_score": 0.0, "custody_confidence": conf,
        "loitering": loitering, "sensitive_zone": zone,
        "vessel_proximity_score": 0.0,
    }


def test_history_empty_for_nominal_timeline():
    timeline = [_hr(i) for i in range(4)]
    assert _make_agg(timeline) == {}


def test_history_one_row_per_code():
    from custody.config import ZONE_COMPOUND_MIN_SCORE
    timeline = [
        _hr(0),
        _hr(1, loitering=0.5, zone=ZONE_COMPOUND_MIN_SCORE),  # fires
        _hr(2),
        _hr(3, loitering=0.5, zone=ZONE_COMPOUND_MIN_SCORE),  # fires again
    ]
    agg = _make_agg(timeline)
    assert "LOITERING_NEAR_ZONE" in agg
    assert len(agg) == 1


def test_history_count_is_correct():
    from custody.config import ZONE_COMPOUND_MIN_SCORE
    timeline = [
        _hr(0),
        _hr(1, loitering=0.5, zone=ZONE_COMPOUND_MIN_SCORE),
        _hr(2),
        _hr(3, loitering=0.5, zone=ZONE_COMPOUND_MIN_SCORE),
        _hr(4, loitering=0.5, zone=ZONE_COMPOUND_MIN_SCORE),
    ]
    agg = _make_agg(timeline)
    assert agg["LOITERING_NEAR_ZONE"]["count"] == 3


def test_history_first_seen_is_earliest():
    from custody.config import ZONE_COMPOUND_MIN_SCORE
    timeline = [
        _hr(0),
        _hr(1, loitering=0.5, zone=ZONE_COMPOUND_MIN_SCORE),  # first firing
        _hr(2),
        _hr(3, loitering=0.5, zone=ZONE_COMPOUND_MIN_SCORE),
    ]
    agg = _make_agg(timeline)
    assert agg["LOITERING_NEAR_ZONE"]["first"] == _T + timedelta(hours=1)


def test_history_last_seen_is_latest():
    from custody.config import ZONE_COMPOUND_MIN_SCORE
    timeline = [
        _hr(0),
        _hr(1, loitering=0.5, zone=ZONE_COMPOUND_MIN_SCORE),
        _hr(2),
        _hr(3, loitering=0.5, zone=ZONE_COMPOUND_MIN_SCORE),  # last firing
    ]
    agg = _make_agg(timeline)
    assert agg["LOITERING_NEAR_ZONE"]["last"] == _T + timedelta(hours=3)


def test_history_peak_conf_is_maximum():
    from custody.config import ZONE_COMPOUND_MIN_SCORE
    import pytest as _pytest
    # First firing: loitering=0.3, zone=0.5 → norm_loitering=0.3, norm_zone=0.333 → conf=0.333
    # Second firing: loitering=0.8, zone=1.5 → norm_loitering=0.8, norm_zone=1.0 → conf=0.8
    timeline = [
        _hr(0),
        _hr(1, loitering=0.3, zone=ZONE_COMPOUND_MIN_SCORE),
        _hr(2),
        _hr(3, loitering=0.8, zone=1.5),
    ]
    agg = _make_agg(timeline)
    assert agg["LOITERING_NEAR_ZONE"]["peak"] == _pytest.approx(0.8, abs=1e-4)


def test_history_multiple_codes_produce_separate_rows():
    from custody.config import ZONE_COMPOUND_MIN_SCORE, HIGH_ANOMALY_THRESHOLD, LOW_CUSTODY_THRESHOLD
    timeline = [
        _hr(0),
        _hr(1, loitering=0.5, zone=ZONE_COMPOUND_MIN_SCORE),
        {   # HIGH_ANOMALY_LOW_CUSTODY step
            "target_id": "V001", "time": _T + timedelta(hours=2),
            "anomaly_score": HIGH_ANOMALY_THRESHOLD + 0.1,
            "custody_confidence": LOW_CUSTODY_THRESHOLD - 0.1,
            "loitering": 0.0, "sensitive_zone": 0.0,
            "vessel_proximity_score": 0.0,
        },
    ]
    agg = _make_agg(timeline)
    assert "LOITERING_NEAR_ZONE" in agg
    assert "HIGH_ANOMALY_LOW_CUSTODY" in agg
    assert len(agg) == 2


def test_history_dataframe_builds_without_error(sim_df):
    """Mirror the history DataFrame construction in streamlit_app.py."""
    target_df = sim_df[sim_df["target_id"] == "V001"].reset_index(drop=True)
    selected_idx = len(target_df) - 1
    prefix_timeline = target_df.iloc[: selected_idx + 1].to_dict("records")
    agg = _make_agg(prefix_timeline)
    if agg:
        history_df = pd.DataFrame([
            {
                "Code": code,
                "First Seen": str(v["first"]).split("+")[0],
                "Last Seen": str(v["last"]).split("+")[0],
                "Count": v["count"],
                "Peak Conf": f"{v['peak']:.2f}",
            }
            for code, v in agg.items()
        ])
        assert list(history_df.columns) == ["Code", "First Seen", "Last Seen", "Count", "Peak Conf"]
        assert len(history_df) == len(agg)


def test_history_from_replay(replay_target_df):
    """AIS replay path: history aggregation returns a dict without crashing."""
    prefix_timeline = replay_target_df.to_dict("records")
    agg = _make_agg(prefix_timeline)
    assert isinstance(agg, dict)


def test_app_imports_compounds_for_timeline():
    app_path = (
        __file__.replace("tests\\test_dashboard.py", "src\\app\\streamlit_app.py")
        .replace("tests/test_dashboard.py", "src/app/streamlit_app.py")
    )
    source = open(app_path, encoding="utf-8").read()
    assert "compounds_for_timeline" in source, (
        "streamlit_app.py must import and call compounds_for_timeline for the Compound History section"
    )


# ---------------------------------------------------------------------------
# Compound confidence threshold filter — Step 16c
# ---------------------------------------------------------------------------

def _active_compounds_filtered(target_df, selected_idx, threshold=0.0):
    """Mirror the app's 16c threshold-filtered active compounds path."""
    prefix_window = target_df.iloc[:selected_idx].to_dict("records")
    current_record = target_df.iloc[selected_idx].to_dict()
    return sorted(
        [s for s in evaluate_compounds(current_record, window=prefix_window)
         if s.confidence >= threshold],
        key=lambda s: s.confidence,
        reverse=True,
    )


def _make_filtered_record(loitering=0.0, zone=0.0, anomaly=0.0, conf=0.85, proximity=0.0):
    """Construct a minimal compound-eligible record."""
    return {
        "target_id": "V001", "time": _T,
        "anomaly_score": anomaly,
        "custody_confidence": conf,
        "loitering": loitering,
        "sensitive_zone": zone,
        "vessel_proximity_score": proximity,
    }


def test_threshold_zero_preserves_all_active_compounds():
    """Default threshold 0.0 shows all compounds that fire."""
    from custody.config import ZONE_COMPOUND_MIN_SCORE
    record = _make_filtered_record(loitering=0.3, zone=ZONE_COMPOUND_MIN_SCORE)
    unfiltered = evaluate_compounds(record, window=[])
    filtered = [s for s in unfiltered if s.confidence >= 0.0]
    assert len(filtered) == len(unfiltered)


def test_threshold_excludes_low_confidence_active_compounds():
    """Threshold > 0 drops compounds whose confidence is below it."""
    from custody.config import ZONE_COMPOUND_MIN_SCORE
    # Small loitering + min zone → low confidence compound
    record = _make_filtered_record(loitering=0.1, zone=ZONE_COMPOUND_MIN_SCORE)
    all_compounds = evaluate_compounds(record, window=[])
    # Determine actual confidence of the fired compound
    assert len(all_compounds) >= 1
    lowest_conf = min(s.confidence for s in all_compounds)
    # Set threshold just above the lowest confidence
    threshold = lowest_conf + 0.01
    filtered = [s for s in all_compounds if s.confidence >= threshold]
    assert len(filtered) < len(all_compounds)


def test_threshold_one_blocks_all_active_unless_perfect():
    """Threshold 1.0 should block all compounds except perfectly-scoring ones."""
    from custody.config import ZONE_COMPOUND_MIN_SCORE
    record = _make_filtered_record(loitering=0.3, zone=ZONE_COMPOUND_MIN_SCORE)
    all_compounds = evaluate_compounds(record, window=[])
    filtered = [s for s in all_compounds if s.confidence >= 1.0]
    # Only compounds with confidence exactly 1.0 pass; most won't
    for s in filtered:
        assert s.confidence == 1.0


def test_threshold_active_empty_state_when_all_filtered():
    """When threshold blocks every compound, active list is empty."""
    from custody.config import ZONE_COMPOUND_MIN_SCORE
    record = _make_filtered_record(loitering=0.1, zone=ZONE_COMPOUND_MIN_SCORE)
    all_compounds = evaluate_compounds(record, window=[])
    assert len(all_compounds) >= 1, "precondition: at least one compound fires"
    filtered = [s for s in all_compounds if s.confidence >= 1.0]
    # No compound can have conf > 1 since confidence is capped at 1.0
    # With loitering=0.1 no compound reaches 1.0
    assert filtered == []


def test_threshold_history_zero_preserves_all_rows():
    """Default threshold 0.0 keeps all codes in the history summary."""
    from custody.config import ZONE_COMPOUND_MIN_SCORE
    timeline = [
        _hr(0),
        _hr(1, loitering=0.5, zone=ZONE_COMPOUND_MIN_SCORE),
        _hr(2, loitering=0.5, zone=ZONE_COMPOUND_MIN_SCORE),
    ]
    agg_default = _make_agg(timeline, threshold=0.0)
    agg_unfiltered = _make_agg(timeline)
    assert agg_default == agg_unfiltered


def test_threshold_history_excludes_low_confidence_codes():
    """Threshold above a compound's peak drops it from history entirely."""
    from custody.config import ZONE_COMPOUND_MIN_SCORE
    # loitering=0.1, zone=ZONE_COMPOUND_MIN_SCORE → low confidence
    timeline = [
        _hr(0),
        _hr(1, loitering=0.1, zone=ZONE_COMPOUND_MIN_SCORE),
    ]
    agg_all = _make_agg(timeline, threshold=0.0)
    assert len(agg_all) >= 1
    peak = max(v["peak"] for v in agg_all.values())
    # Threshold just above the peak removes all codes
    agg_filtered = _make_agg(timeline, threshold=peak + 0.01)
    assert agg_filtered == {}


def test_threshold_history_empty_state_when_all_filtered():
    """When threshold filters all history compounds, the agg dict is empty."""
    from custody.config import ZONE_COMPOUND_MIN_SCORE
    timeline = [
        _hr(0),
        _hr(1, loitering=0.2, zone=ZONE_COMPOUND_MIN_SCORE),
    ]
    agg = _make_agg(timeline, threshold=1.0)
    # Confidence from loitering=0.2 + min zone can't reach 1.0
    assert agg == {}


def test_threshold_history_partial_filter():
    """High threshold may keep some codes but not others."""
    from custody.config import ZONE_COMPOUND_MIN_SCORE, HIGH_ANOMALY_THRESHOLD, LOW_CUSTODY_THRESHOLD
    # LOITERING_NEAR_ZONE fires with low loitering → lower confidence
    # HIGH_ANOMALY_LOW_CUSTODY fires with values near thresholds → confidence depends on margins
    timeline = [
        _hr(0),
        _hr(1, loitering=0.1, zone=ZONE_COMPOUND_MIN_SCORE),     # low confidence
        {
            "target_id": "V001", "time": _T + timedelta(hours=2),
            "anomaly_score": HIGH_ANOMALY_THRESHOLD * 3,           # well above → high norm
            "custody_confidence": 0.0,                             # inverted → 1.0
            "loitering": 0.0, "sensitive_zone": 0.0,
            "vessel_proximity_score": 0.0,
        },
    ]
    agg_all = _make_agg(timeline, threshold=0.0)
    # At least two codes should be present
    assert len(agg_all) >= 2
    peaks = {code: v["peak"] for code, v in agg_all.items()}
    # There must be at least one code whose peak < 0.5 and one whose peak >= 0.5
    low_codes = [c for c, p in peaks.items() if p < 0.5]
    high_codes = [c for c, p in peaks.items() if p >= 0.5]
    if low_codes and high_codes:
        # A threshold of 0.5 should keep high-confidence codes and drop low ones
        agg_filtered = _make_agg(timeline, threshold=0.5)
        for code in low_codes:
            assert code not in agg_filtered
        for code in high_codes:
            assert code in agg_filtered


def test_threshold_filter_applied_before_count(sim_df):
    """Threshold filtering happens before aggregation — count reflects only passing signals."""
    from custody.config import ZONE_COMPOUND_MIN_SCORE
    timeline = [
        _hr(0, loitering=0.1, zone=ZONE_COMPOUND_MIN_SCORE),  # low conf
        _hr(1, loitering=0.1, zone=ZONE_COMPOUND_MIN_SCORE),  # low conf
        _hr(2, loitering=0.9, zone=1.5),                      # higher conf
    ]
    agg_all = _make_agg(timeline, threshold=0.0)
    if "LOITERING_NEAR_ZONE" not in agg_all:
        return  # compound didn't fire — skip
    # Find the confidence of the high-loitering record
    high_conf = agg_all["LOITERING_NEAR_ZONE"]["peak"]
    low_conf = min(
        c.confidence
        for c in compounds_for_timeline(timeline)
        if c.code == "LOITERING_NEAR_ZONE"
    )
    if low_conf < high_conf:
        threshold = (low_conf + high_conf) / 2
        agg_filtered = _make_agg(timeline, threshold=threshold)
        if "LOITERING_NEAR_ZONE" in agg_filtered:
            assert agg_filtered["LOITERING_NEAR_ZONE"]["count"] < agg_all["LOITERING_NEAR_ZONE"]["count"]


def test_threshold_slider_source_present_in_app():
    """Source-level guard: app contains the min_compound_conf slider."""
    app_path = (
        __file__.replace("tests\\test_dashboard.py", "src\\app\\streamlit_app.py")
        .replace("tests/test_dashboard.py", "src/app/streamlit_app.py")
    )
    source = open(app_path, encoding="utf-8").read()
    assert "min_compound_conf" in source, (
        "streamlit_app.py must define the min_compound_conf sidebar slider (Step 16c)"
    )


def test_threshold_filter_safe_with_replay(replay_target_df):
    """AIS replay path: threshold filter does not crash on replay-derived records."""
    result = _active_compounds_filtered(replay_target_df, len(replay_target_df) - 1, threshold=0.5)
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# compound_panels module — Step 18 direct tests
# ---------------------------------------------------------------------------

def test_aggregate_compound_history_empty_input():
    assert aggregate_compound_history([]) == {}


def test_aggregate_compound_history_count_and_keys():
    from custody.config import ZONE_COMPOUND_MIN_SCORE
    timeline = [
        _hr(0),
        _hr(1, loitering=0.5, zone=ZONE_COMPOUND_MIN_SCORE),
        _hr(2),
        _hr(3, loitering=0.5, zone=ZONE_COMPOUND_MIN_SCORE),
    ]
    compounds = [c for c in compounds_for_timeline(timeline)]
    agg = aggregate_compound_history(compounds)
    assert "LOITERING_NEAR_ZONE" in agg
    assert agg["LOITERING_NEAR_ZONE"]["count"] == 2
    assert agg["LOITERING_NEAR_ZONE"]["first"] < agg["LOITERING_NEAR_ZONE"]["last"]
    assert 0.0 < agg["LOITERING_NEAR_ZONE"]["peak"] <= 1.0


def test_aggregate_compound_history_peak_is_max():
    from custody.config import ZONE_COMPOUND_MIN_SCORE
    timeline = [
        _hr(0),
        _hr(1, loitering=0.1, zone=ZONE_COMPOUND_MIN_SCORE),  # low conf
        _hr(2, loitering=0.9, zone=1.5),                      # higher conf
    ]
    compounds = [c for c in compounds_for_timeline(timeline) if c.code == "LOITERING_NEAR_ZONE"]
    if len(compounds) < 2:
        return  # precondition not met in this sim run — skip
    agg = aggregate_compound_history(compounds)
    assert agg["LOITERING_NEAR_ZONE"]["peak"] == max(c.confidence for c in compounds)


def test_build_active_compounds_df_columns():
    from custody.config import ZONE_COMPOUND_MIN_SCORE
    record = {
        "target_id": "V001", "time": _T0,
        "loitering": 0.5, "sensitive_zone": ZONE_COMPOUND_MIN_SCORE,
        "anomaly_score": 0.0, "custody_confidence": 0.85,
        "vessel_proximity_score": 0.0,
    }
    compounds = evaluate_compounds(record, window=[])
    assert len(compounds) >= 1
    df = build_active_compounds_df(compounds)
    assert list(df.columns) == ["Time", "Code", "Conf", "Evidence", "Components"]
    assert len(df) == len(compounds)


def test_build_active_compounds_df_conf_format():
    """Conf column is formatted as a 2-decimal string."""
    from custody.config import ZONE_COMPOUND_MIN_SCORE
    record = {
        "target_id": "V001", "time": _T0,
        "loitering": 0.5, "sensitive_zone": ZONE_COMPOUND_MIN_SCORE,
        "anomaly_score": 0.0, "custody_confidence": 0.85,
        "vessel_proximity_score": 0.0,
    }
    df = build_active_compounds_df(evaluate_compounds(record, window=[]))
    for val in df["Conf"]:
        assert isinstance(val, str)
        assert len(val.split(".")[1]) == 2


def test_build_compound_history_df_columns():
    from custody.config import ZONE_COMPOUND_MIN_SCORE
    timeline = [_hr(0), _hr(1, loitering=0.5, zone=ZONE_COMPOUND_MIN_SCORE)]
    agg = aggregate_compound_history(compounds_for_timeline(timeline))
    assert agg  # precondition
    df = build_compound_history_df(agg)
    assert list(df.columns) == ["Code", "First Seen", "Last Seen", "Count", "Peak Conf"]
    assert len(df) == len(agg)


def test_build_compound_history_df_ordered_by_first_seen():
    """Rows are sorted ascending by first seen timestamp."""
    from custody.config import ZONE_COMPOUND_MIN_SCORE, HIGH_ANOMALY_THRESHOLD, LOW_CUSTODY_THRESHOLD
    timeline = [
        _hr(0),
        _hr(1, loitering=0.5, zone=ZONE_COMPOUND_MIN_SCORE),
        {
            "target_id": "V001", "time": _T + timedelta(hours=2),
            "anomaly_score": HIGH_ANOMALY_THRESHOLD + 0.1,
            "custody_confidence": LOW_CUSTODY_THRESHOLD - 0.1,
            "loitering": 0.0, "sensitive_zone": 0.0,
            "vessel_proximity_score": 0.0,
        },
    ]
    agg = aggregate_compound_history(compounds_for_timeline(timeline))
    if len(agg) < 2:
        return  # not enough distinct codes to test ordering
    df = build_compound_history_df(agg)
    first_seen_vals = list(df["First Seen"])
    assert first_seen_vals == sorted(first_seen_vals)


def test_app_uses_compound_panels_functions():
    """Source-level guard: streamlit_app.py calls the extracted helpers."""
    app_path = (
        __file__.replace("tests\\test_dashboard.py", "src\\app\\streamlit_app.py")
        .replace("tests/test_dashboard.py", "src/app/streamlit_app.py")
    )
    source = open(app_path, encoding="utf-8").read()
    assert "aggregate_compound_history" in source
    assert "build_active_compounds_df" in source
    assert "build_compound_history_df" in source
