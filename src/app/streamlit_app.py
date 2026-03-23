import time

import pandas as pd
import pydeck as pdk
import streamlit as st

from custody.ais import replay_ais_file
from custody.alerts import alerts_for_timeline
from custody.compounds import compounds_for_timeline
from custody.simulate import run_simulation
from custody.config import ZONES
from custody.sensors import get_sensor_opportunities


def _coerce_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Convert non-object string columns to object dtype.

    Streamlit >=1.19 cannot render Arrow LargeUtf8 columns (type 20).
    pandas >=3.0 infers string columns as StringDtype or ArrowDtype by
    default, so we convert them back to plain object dtype before display.
    """
    cols = {
        c: object for c, d in df.dtypes.items()
        if pd.api.types.is_string_dtype(d) and str(d) != "object"
    }
    return df.astype(cols) if cols else df


st.set_page_config(page_title="Custody", layout="wide")

st.markdown("""
<style>
html, body, [class*='css'] { font-size: 90% !important; }

/* Tighten main container */
.block-container { padding-top: 0.75rem !important; padding-bottom: 0.75rem !important; }

/* Compact section labels */
.section-label {
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: #888;
    border-bottom: 1px solid #2d2d2d;
    padding-bottom: 3px;
    margin-top: 14px;
    margin-bottom: 6px;
}

/* Tighten metric spacing */
[data-testid="metric-container"] { padding: 0.25rem 0 !important; }
[data-testid="metric-container"] label { font-size: 0.7rem !important; color: #999 !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { font-size: 1rem !important; }

/* Compact alert banners */
.stAlert { padding: 0.4rem 0.75rem !important; margin-bottom: 0.5rem !important; }
.stAlert p { font-size: 0.8rem !important; margin: 0 !important; }

/* Compact dataframe */
.stDataFrame { font-size: 0.75rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    "<h3 style='margin:0 0 2px 0;'>Custody</h3>"
    "<p style='margin:0 0 10px 0; color:#888; font-size:0.75rem;'>"
    "Prototype for anomaly-aware target custody and collection planning</p>",
    unsafe_allow_html=True,
)

st.sidebar.header("Playback")
mode = st.sidebar.radio("Mode", ["Simulation", "AIS Replay"])

if mode == "Simulation":
    records = run_simulation()
    all_df = pd.DataFrame(records)
    all_df["time_str"] = all_df["time"].astype(str)
    target_ids = sorted(all_df["target_id"].unique().tolist())
    selected_target = st.sidebar.selectbox("Target", options=target_ids, index=0)
    target_df = all_df[all_df["target_id"] == selected_target].reset_index(drop=True)

else:  # AIS Replay
    csv_source = st.sidebar.text_area(
        "AIS CSV source",
        height=100,
        placeholder="Paste CSV text or enter a file path",
    )
    if not csv_source.strip():
        st.info("Paste an AIS CSV into the sidebar (or enter a file path) to begin replay.")
        st.stop()
    try:
        replay_result = replay_ais_file(csv_source)
    except (ValueError, OSError) as exc:
        st.error(f"Could not load AIS source: {exc}")
        st.stop()
    if not replay_result:
        st.warning("No observations found in the provided source.")
        st.stop()
    vessel_ids = sorted(replay_result.keys())
    selected_target = st.sidebar.selectbox("Vessel", options=vessel_ids)
    target_df = pd.DataFrame(replay_result[selected_target])
    target_df["time_str"] = target_df["time"].astype(str)
    # Build a unified df from all replayed vessels (used for the other-targets map layer)
    all_records = [r for tl in replay_result.values() for r in tl]
    all_df = pd.DataFrame(all_records)
    all_df["time_str"] = all_df["time"].astype(str)

if "playback_idx" not in st.session_state:
    st.session_state.playback_idx = 0

autoplay = st.sidebar.checkbox("Autoplay", value=False)

max_idx = len(target_df) - 1
default_idx = st.session_state.playback_idx if st.session_state.playback_idx <= max_idx else 0

selected_idx = st.sidebar.slider(
    "Timeline step",
    min_value=0,
    max_value=max_idx,
    value=default_idx,
    key="timeline_slider",
)

st.session_state.playback_idx = selected_idx
current = target_df.iloc[selected_idx]

action_reason = (
    current["action_reason"]
    if pd.notna(current["action_reason"]) and current["action_reason"]
    else "—"
)
sensor_id = current["sensor_id"] if pd.notna(current["sensor_id"]) else "—"
sensor_type = current["sensor_type"] if pd.notna(current["sensor_type"]) else "—"
collection_result = (
    current["collection_result"] if pd.notna(current["collection_result"]) else "—"
)

# ── Status banner ─────────────────────────────────────────────────────────────
if current["action"] == "TASK":
    st.error("TASKING SENSOR")
elif current["action"] == "HOLD":
    st.success("HOLDING CUSTODY")
elif float(current["anomaly_score"]) > 0.5:
    st.warning("ANOMALY DETECTED")
else:
    st.info("NORMAL TRACKING")

# ── Top metrics: all 6 in one row ─────────────────────────────────────────────
st.markdown("<div class='section-label'>Current Step</div>", unsafe_allow_html=True)
c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1: st.metric("Time", current["time_str"].split("+")[0])
with c2: st.metric("Anomaly Score", f'{current["anomaly_score"]:.2f}')
with c3: st.metric("Custody Confidence", f'{current["custody_confidence"]:.2f}')
with c4: st.metric("Uncertainty (km)", f'{current["uncertainty_km"]:.2f}')
with c5: st.metric("Action", current["action"])
with c6: st.metric("Collection Result", collection_result)

# ── Decision ──────────────────────────────────────────────────────────────────
st.markdown("<div class='section-label'>Decision</div>", unsafe_allow_html=True)
d1, d2 = st.columns(2)
with d1:
    st.markdown(f"<span style='color:#999;font-size:0.75rem;'>Reason</span><br>{action_reason}", unsafe_allow_html=True)
with d2:
    st.markdown(f"<span style='color:#999;font-size:0.75rem;'>Sensor</span><br>{sensor_id} ({sensor_type})", unsafe_allow_html=True)

# ── Mission summary ───────────────────────────────────────────────────────────
st.markdown("<div class='section-label'>Mission Summary</div>", unsafe_allow_html=True)
task_count = int((target_df["action"] == "TASK").sum())
hold_count = int((target_df["action"] == "HOLD").sum())
peak_anomaly = float(pd.to_numeric(target_df["anomaly_score"], errors="coerce").max())
final_confidence = float(pd.to_numeric(target_df["custody_confidence"], errors="coerce").iloc[-1])

m1, m2, m3, m4 = st.columns(4)
with m1: st.metric("Total Taskings", task_count)
with m2: st.metric("Total Holds", hold_count)
with m3: st.metric("Peak Anomaly", f"{peak_anomaly:.2f}")
with m4: st.metric("Final Confidence", f"{final_confidence:.2f}")

# ── Current state + anomaly context in one row ────────────────────────────────
st.markdown("<div class='section-label'>Current State</div>", unsafe_allow_html=True)
s1, s2, s3, s4, s5, s6, s7, s8 = st.columns(8)
with s1: st.metric("Lat", f'{current["lat"]:.3f}')
with s2: st.metric("Lon", f'{current["lon"]:.3f}')
with s3: st.metric("Speed (km/h)", f'{current["speed_kmh"]:.1f}')
with s4: st.metric("Heading", f'{current["heading_deg"]:.1f}')
with s5: st.metric("Behavior", str(current["behavior_mode"]).upper())
with s6: st.metric("Sensitive Zone", "ACTIVE" if float(current["sensitive_zone"]) > 0 else "clear")
with s7: st.metric("Loitering", "ACTIVE" if float(current["loitering"]) > 0 else "clear")
with s8: st.metric("Route Deviation", "ACTIVE" if float(current["route_deviation"]) > 0 else "clear")

# ── Inferred state ────────────────────────────────────────────────────────────
st.markdown("<div class='section-label'>Inferred State</div>", unsafe_allow_html=True)
i1, i2 = st.columns(2)
with i1: st.metric("Behavior State", str(current["behavior_state"]).upper())
with i2: st.metric("State Confidence", f'{current["state_confidence"]:.2f}')

# ── Alerts ────────────────────────────────────────────────────────────────────
st.markdown("<div class='section-label'>Alerts</div>", unsafe_allow_html=True)
alerts = alerts_for_timeline(target_df.iloc[: selected_idx + 1].to_dict("records"))
if alerts:
    alerts_df = pd.DataFrame([
        {"Time": a.timestamp, "Level": a.level, "Code": a.code, "Message": a.message}
        for a in alerts
    ])
    alerts_df["Time"] = alerts_df["Time"].astype(str).str.split("+").str[0]
    st.dataframe(_coerce_strings(alerts_df.reset_index(drop=True)), use_container_width=True)
else:
    st.markdown("<span style='color:#666;'>No alerts for this target.</span>", unsafe_allow_html=True)

# ── Compound signals ──────────────────────────────────────────────────────────
st.markdown("<div class='section-label'>Compound Signals</div>", unsafe_allow_html=True)
compounds = compounds_for_timeline(target_df.iloc[: selected_idx + 1].to_dict("records"))
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
    st.dataframe(_coerce_strings(compounds_df.reset_index(drop=True)), use_container_width=True)
else:
    st.markdown("<span style='color:#666;'>No compound signals for this target.</span>", unsafe_allow_html=True)

# ── Available sensors ─────────────────────────────────────────────────────────
st.markdown("<div class='section-label'>Available Sensors</div>", unsafe_allow_html=True)
available_sensors = get_sensor_opportunities(current["time"])
if available_sensors:
    st.markdown(
        " &nbsp;|&nbsp; ".join(
            f"<code>{s.sensor_id}</code> {s.sensor_type} · {s.resolution}"
            for s in available_sensors
        ),
        unsafe_allow_html=True,
    )
else:
    st.markdown("<span style='color:#666;'>No sensors currently available</span>", unsafe_allow_html=True)

# ── Track map ─────────────────────────────────────────────────────────────────
st.markdown("<div class='section-label'>Track Map</div>", unsafe_allow_html=True)

normal_segments = []
anomalous_segments = []

for i in range(1, selected_idx + 1):
    prev_row = target_df.iloc[i - 1]
    curr_row = target_df.iloc[i]
    segment = [
        [float(prev_row["lon"]), float(prev_row["lat"])],
        [float(curr_row["lon"]), float(curr_row["lat"])],
    ]
    if float(curr_row["anomaly_score"]) > 0.5:
        anomalous_segments.append({"path": segment})
    else:
        normal_segments.append({"path": segment})

task_points = []
hold_points = []
for i in range(selected_idx + 1):
    row = target_df.iloc[i]
    point = {"lon": float(row["lon"]), "lat": float(row["lat"])}
    if row["action"] == "TASK":
        task_points.append(point)
    elif row["action"] == "HOLD":
        hold_points.append(point)

other_targets_df = all_df[
    (all_df["target_id"] != selected_target) & (all_df["time"] == current["time"])
].copy()

current_lat = float(current["lat"])
current_lon = float(current["lon"])
uncertainty_m = float(current["uncertainty_km"]) * 1000.0

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

zone_layer = pdk.Layer("PolygonLayer", data=zone_data,
    get_polygon="polygon", get_fill_color=[255, 215, 0, 35],
    get_line_color=[255, 215, 0, 180], line_width_min_pixels=2,
    stroked=True, filled=True, pickable=True,
    tooltip=True)

other_targets_layer = pdk.Layer("ScatterplotLayer",
    data=[{"lon": float(r["lon"]), "lat": float(r["lat"])} for _, r in other_targets_df.iterrows()],
    get_position="[lon, lat]", get_radius=2200, get_fill_color=[180, 180, 180, 120], pickable=True)

line_layer_normal = pdk.Layer("PathLayer", data=normal_segments,
    get_path="path", get_width=8, width_min_pixels=3, get_color=[0, 200, 255], pickable=False)

line_layer_anomalous = pdk.Layer("PathLayer", data=anomalous_segments,
    get_path="path", get_width=12, width_min_pixels=4, get_color=[255, 140, 0], pickable=False)

task_layer = pdk.Layer("ScatterplotLayer", data=task_points,
    get_position="[lon, lat]", get_radius=2500, get_fill_color=[255, 60, 60, 220], pickable=True)

hold_layer = pdk.Layer("ScatterplotLayer", data=hold_points,
    get_position="[lon, lat]", get_radius=1800, get_fill_color=[80, 220, 120, 180], pickable=True)

current_point_layer = pdk.Layer("ScatterplotLayer",
    data=[{"lon": current_lon, "lat": current_lat}],
    get_position="[lon, lat]", get_radius=4000, get_fill_color=[255, 255, 255, 255], pickable=True)

uncertainty_layer = pdk.Layer("ScatterplotLayer",
    data=[{"lon": current_lon, "lat": current_lat, "radius": uncertainty_m}],
    get_position="[lon, lat]", get_radius="radius",
    get_fill_color=[255, 50, 50, 35], get_line_color=[255, 80, 80, 180],
    stroked=True, filled=True, line_width_min_pixels=2, pickable=False)

deck = pdk.Deck(
    map_style="mapbox://styles/mapbox/dark-v10",
    initial_view_state=pdk.ViewState(latitude=current_lat, longitude=current_lon, zoom=10),
    layers=[zone_layer, other_targets_layer, uncertainty_layer,
            line_layer_normal, line_layer_anomalous, hold_layer, task_layer, current_point_layer],
    tooltip={"text": "Lat: {lat}\nLon: {lon}"},
)

st.pydeck_chart(deck, use_container_width=True)

legend_cols = st.columns(6)
with legend_cols[0]: st.markdown("<small>🟨 Sensitive zone</small>", unsafe_allow_html=True)
with legend_cols[1]: st.markdown("<small>⚪ Other target</small>", unsafe_allow_html=True)
with legend_cols[2]: st.markdown("<small>🔵 Normal track</small>", unsafe_allow_html=True)
with legend_cols[3]: st.markdown("<small>🟠 Anomalous track</small>", unsafe_allow_html=True)
with legend_cols[4]: st.markdown("<small>🔴 TASK event</small>", unsafe_allow_html=True)
with legend_cols[5]: st.markdown("<small>🟢 HOLD event</small>", unsafe_allow_html=True)

# Slice the target timeline to the current playback position.
# All time-indexed tables below use this slice so they grow step-by-step,
# matching the map track and alert view.
visible_df = target_df.iloc[: selected_idx + 1]

# ── Custody over time ─────────────────────────────────────────────────────────
st.markdown("<div class='section-label'>Custody Over Time</div>", unsafe_allow_html=True)
cot_df = visible_df[["time_str", "custody_confidence", "anomaly_score"]].copy()
cot_df["custody_confidence"] = pd.to_numeric(cot_df["custody_confidence"], errors="coerce").round(2)
cot_df["anomaly_score"] = pd.to_numeric(cot_df["anomaly_score"], errors="coerce").round(2)
cot_df.columns = ["Time", "Confidence", "Anomaly"]
st.dataframe(_coerce_strings(cot_df.reset_index(drop=True)), use_container_width=True)

# ── Timeline ──────────────────────────────────────────────────────────────────
st.markdown("<div class='section-label'>Timeline</div>", unsafe_allow_html=True)
display_cols = ["time_str", "lat", "lon", "uncertainty_km", "custody_confidence",
                "anomaly_score", "action", "sensor_id", "collection_result",
                "behavior_state", "state_confidence"]
timeline_df = visible_df[display_cols].copy().fillna("—")
for col in ["lat", "lon", "uncertainty_km", "custody_confidence", "anomaly_score",
            "state_confidence"]:
    timeline_df[col] = pd.to_numeric(timeline_df[col], errors="coerce")
timeline_df["lat"] = timeline_df["lat"].round(3)
timeline_df["lon"] = timeline_df["lon"].round(3)
timeline_df["uncertainty_km"] = timeline_df["uncertainty_km"].round(2)
timeline_df["custody_confidence"] = timeline_df["custody_confidence"].round(2)
timeline_df["anomaly_score"] = timeline_df["anomaly_score"].round(2)
timeline_df["state_confidence"] = timeline_df["state_confidence"].round(2)
timeline_df.columns = ["Time", "Lat", "Lon", "Unc (km)", "Confidence",
                        "Anomaly", "Action", "Sensor", "Result",
                        "State", "St.Conf"]
st.dataframe(_coerce_strings(timeline_df.reset_index(drop=True)), use_container_width=True)

if autoplay:
    time.sleep(0.8)
    if st.session_state.playback_idx < len(target_df) - 1:
        st.session_state.playback_idx += 1
    else:
        st.session_state.playback_idx = 0
    st.experimental_rerun()
