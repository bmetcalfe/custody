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
from custody.compounds import compounds_for_timeline
from custody.simulate import run_simulation
from custody.config import ZONES
from custody.sensors import get_sensor_opportunities
from custody.models import BehaviorState


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
    missing = set(REQUIRED_COLUMNS) - set(sim_df.columns)
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
    valid = {"NONE", "TASK", "HOLD", "NO_SENSOR"}
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
    text = src.read_text()
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
    source = open(app_path).read()
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
    source = open(app_path).read()
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
    valid = {"NONE", "TASK", "HOLD", "NO_SENSOR"}
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
    source = open(app_path).read()
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
    source = open(app_path).read()
    assert "alerts_for_timeline" in source, (
        "streamlit_app.py must import and call alerts_for_timeline for the Alerts section"
    )


# ---------------------------------------------------------------------------
# Compound signals section — data pipeline (Step 15d refactor)
# ---------------------------------------------------------------------------

def test_compounds_for_timeline_returns_list_from_sim(sim_df):
    target_records = sim_df[sim_df["target_id"] == "V001"].to_dict("records")
    result = compounds_for_timeline(target_records)
    assert isinstance(result, list)


def test_compounds_dataframe_builds_without_error(sim_df):
    """Mirror the compounds DataFrame construction in streamlit_app.py."""
    target_df = sim_df[sim_df["target_id"] == "V001"].reset_index(drop=True)
    selected_idx = len(target_df) - 1
    compounds = compounds_for_timeline(
        target_df.iloc[: selected_idx + 1].to_dict("records")
    )
    if compounds:
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
            for c in compounds
        ])
        assert list(compounds_df.columns) == ["Time", "Code", "Conf", "Evidence", "Components"]
        assert len(compounds_df) == len(compounds)


def test_compounds_slice_respects_playback_position(sim_df):
    """compounds_for_timeline called with a prefix slice grows monotonically."""
    target_df = sim_df[sim_df["target_id"] == "V001"].reset_index(drop=True)
    counts = [
        len(compounds_for_timeline(target_df.iloc[: i + 1].to_dict("records")))
        for i in range(len(target_df))
    ]
    assert counts == sorted(counts), "Compound counts must not decrease as playback advances"


def test_compounds_for_timeline_returns_list_from_replay(replay_target_df):
    target_records = replay_target_df.to_dict("records")
    result = compounds_for_timeline(target_records)
    assert isinstance(result, list)


def test_app_imports_compounds_for_timeline():
    app_path = (
        __file__.replace("tests\\test_dashboard.py", "src\\app\\streamlit_app.py")
        .replace("tests/test_dashboard.py", "src/app/streamlit_app.py")
    )
    source = open(app_path).read()
    assert "compounds_for_timeline" in source, (
        "streamlit_app.py must import and call compounds_for_timeline for the Compounds section"
    )
