import math
import time

import pandas as pd
import pydeck as pdk
import streamlit as st

import custody.config as config
from custody.ais import replay_ais_file
from custody.alerts import alerts_for_timeline
from custody.compounds import compounds_for_timeline, evaluate_compounds
from custody.decision_trace import DecisionTrace, traces_to_rows
from custody.simulation import run_multi_target_simulation
from custody.simulation.scenarios import PORTFOLIO_SCENARIO
from custody.config import ZONES
from custody.sensors import get_sensor_opportunities, next_pass_window
from custody.whatif import run_comparison
from compound_panels import (
    aggregate_compound_history,
    build_active_compounds_df,
    build_compound_history_df,
)
from orbital_passes_panel import build_orbital_passes_rows
from whatif_panel import build_variants, build_whatif_results_df
from entity_detail_panel import render_entity_detail_panel
from portfolio_overview import (
    derive_display_status,
    build_overview_df,
    compute_kpi_counts,
    top_attention_targets,
    build_focus_view_state,
    build_label_data,
)
from overview_filters import apply_overview_filters, filter_top_n, ALL_STATUSES
from overview_events import build_event_feed
from ground_track import all_ground_track_layers_data, satellite_current_positions



st.set_page_config(page_title="Custody", layout="wide")

st.markdown("""
<style>
/* Tighten main container */
.block-container { padding-bottom: 0.75rem !important; }

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
    # Resolve any pending navigation from Inspect buttons BEFORE widgets are instantiated
    if "_pending_view_mode" in st.session_state:
        st.session_state["view_mode_radio"] = st.session_state.pop("_pending_view_mode")
    if "_pending_entity_select" in st.session_state:
        st.session_state["sidebar_entity_select"] = st.session_state.pop("_pending_entity_select")

    if "view_mode_radio" not in st.session_state:
        st.session_state["view_mode_radio"] = "Overview"
    view_mode = st.sidebar.radio(
        "View", ["Overview", "Entity Detail"], key="view_mode_radio"
    )
else:
    view_mode = "Entity Detail"

if mode == "Simulation":
    records = run_multi_target_simulation(PORTFOLIO_SCENARIO)
    all_df = pd.DataFrame(records)
    all_df["time_str"] = all_df["time"].astype(str)

    # Sidebar entity filter with scripted-actor toggle
    target_ids = sorted(all_df["target_id"].unique().tolist())
    if "is_scripted" in all_df.columns:
        _show_scripted_only = st.sidebar.checkbox("Scripted actors only", value=False)
        if _show_scripted_only:
            _scripted_ids = all_df[all_df["is_scripted"]]["target_id"].unique().tolist()
            target_ids = sorted(_scripted_ids) if _scripted_ids else target_ids

    selected_target = st.sidebar.selectbox(
        "Target", options=target_ids,
        index=(target_ids.index(st.session_state["sidebar_entity_select"])
               if st.session_state.get("sidebar_entity_select") in target_ids else 0),
        key="sidebar_entity_select",
    )

    # Metadata chip — profile, scripted flag, scenario tags
    if "profile" in all_df.columns:
        _meta = all_df[all_df["target_id"] == selected_target].iloc[0]
        _tags = ", ".join(_meta.get("scenario_tags", [])) or "—"
        st.sidebar.caption(
            f"**Profile:** {_meta.get('profile', '—')}  \n"
            f"**Scripted:** {'Yes' if _meta.get('is_scripted') else 'No'}  \n"
            f"**Tags:** {_tags}"
        )
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
if "playing" not in st.session_state:
    st.session_state.playing = False

max_idx = len(target_df) - 1
if st.session_state.playback_idx > max_idx:
    st.session_state.playback_idx = 0

# ── Playback transport controls ───────────────────────────────────────────────
_tc = st.sidebar.columns(5)
with _tc[0]:
    if st.button("⏮", key="btn_to_start", help="Skip to start"):
        st.session_state.playback_idx = 0
        st.session_state.playing = False
with _tc[1]:
    if st.button("⏪", key="btn_step_back", help="Step back one frame"):
        st.session_state.playback_idx = max(0, st.session_state.playback_idx - 1)
        st.session_state.playing = False
with _tc[2]:
    _play_icon = "⏸" if st.session_state.playing else "▶"
    if st.button(_play_icon, key="btn_play_pause", help="Play / Pause"):
        st.session_state.playing = not st.session_state.playing
with _tc[3]:
    if st.button("⏩", key="btn_step_fwd", help="Step forward one frame"):
        st.session_state.playback_idx = min(max_idx, st.session_state.playback_idx + 1)
        st.session_state.playing = False
with _tc[4]:
    if st.button("⏭", key="btn_to_end", help="Skip to end"):
        st.session_state.playback_idx = max_idx
        st.session_state.playing = False

selected_idx = st.sidebar.slider(
    "Timeline step",
    min_value=0,
    max_value=max_idx,
    value=st.session_state.playback_idx,
)

min_compound_conf = st.sidebar.slider(
    "Min compound confidence",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.05,
    key="min_compound_conf",
    help="Hide compound signals below this confidence. Default 0.0 shows all.",
)

# ── What-If Analysis ──────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.header("What-If Analysis")
st.sidebar.caption("Compare planner behavior across config variants. Baseline always runs with current config.")

st.sidebar.markdown("**Variant A**")
_wf_a_threshold = st.sidebar.number_input(
    "Lookahead Threshold (s)", value=config.HOLD_LOOKAHEAD_THRESHOLD_SECONDS,
    min_value=0.0, step=60.0, key="wf_a_threshold",
)
_wf_a_boost = st.sidebar.number_input(
    "Lookahead Boost", value=config.HOLD_LOOKAHEAD_BOOST,
    min_value=0.0, max_value=2.0, step=0.05, key="wf_a_boost",
)
_wf_a_stale_gap = st.sidebar.number_input(
    "AIS Stale Gap (s)", value=config.AIS_STALE_GAP_SECONDS,
    min_value=0.0, step=60.0, key="wf_a_stale_gap",
)
_wf_a_decay_rate = st.sidebar.number_input(
    "AIS Decay Rate (1/s)", value=config.AIS_STALE_CONFIDENCE_DECAY_RATE,
    min_value=0.0, max_value=0.01, step=1e-5, format="%.5f", key="wf_a_decay_rate",
)
_wf_a_min_conf = st.sidebar.number_input(
    "AIS Min Confidence", value=config.AIS_MIN_STALE_CONFIDENCE,
    min_value=0.0, max_value=1.0, step=0.05, key="wf_a_min_conf",
)

_include_b = st.sidebar.checkbox("Include Variant B", value=False, key="wf_include_b")
if _include_b:
    st.sidebar.markdown("**Variant B**")
    _wf_b_threshold = st.sidebar.number_input(
        "Lookahead Threshold (s) ", value=config.HOLD_LOOKAHEAD_THRESHOLD_SECONDS,
        min_value=0.0, step=60.0, key="wf_b_threshold",
    )
    _wf_b_boost = st.sidebar.number_input(
        "Lookahead Boost ", value=config.HOLD_LOOKAHEAD_BOOST,
        min_value=0.0, max_value=2.0, step=0.05, key="wf_b_boost",
    )
    _wf_b_stale_gap = st.sidebar.number_input(
        "AIS Stale Gap (s) ", value=config.AIS_STALE_GAP_SECONDS,
        min_value=0.0, step=60.0, key="wf_b_stale_gap",
    )
    _wf_b_decay_rate = st.sidebar.number_input(
        "AIS Decay Rate (1/s) ", value=config.AIS_STALE_CONFIDENCE_DECAY_RATE,
        min_value=0.0, max_value=0.01, step=1e-5, format="%.5f", key="wf_b_decay_rate",
    )
    _wf_b_min_conf = st.sidebar.number_input(
        "AIS Min Confidence ", value=config.AIS_MIN_STALE_CONFIDENCE,
        min_value=0.0, max_value=1.0, step=0.05, key="wf_b_min_conf",
    )

if st.sidebar.button("Run What-If", key="btn_whatif"):
    _variant_a_overrides = {
        "HOLD_LOOKAHEAD_THRESHOLD_SECONDS": _wf_a_threshold,
        "HOLD_LOOKAHEAD_BOOST": _wf_a_boost,
        "AIS_STALE_GAP_SECONDS": _wf_a_stale_gap,
        "AIS_STALE_CONFIDENCE_DECAY_RATE": _wf_a_decay_rate,
        "AIS_MIN_STALE_CONFIDENCE": _wf_a_min_conf,
    }
    _variant_b_overrides = None
    if _include_b:
        _variant_b_overrides = {
            "HOLD_LOOKAHEAD_THRESHOLD_SECONDS": _wf_b_threshold,
            "HOLD_LOOKAHEAD_BOOST": _wf_b_boost,
            "AIS_STALE_GAP_SECONDS": _wf_b_stale_gap,
            "AIS_STALE_CONFIDENCE_DECAY_RATE": _wf_b_decay_rate,
            "AIS_MIN_STALE_CONFIDENCE": _wf_b_min_conf,
        }

    _wf_variants = build_variants(_variant_a_overrides, _variant_b_overrides)

    if mode == "Simulation":
        _wf_scenario = lambda: run_multi_target_simulation(PORTFOLIO_SCENARIO)
    else:
        _wf_csv = csv_source  # capture for closure
        _wf_scenario = lambda: [r for tl in replay_ais_file(_wf_csv).values() for r in tl]

    st.session_state.whatif_results = run_comparison(_wf_scenario, _wf_variants)

st.session_state.playback_idx = selected_idx
current = target_df.iloc[selected_idx]

# ── Overview mode ──────────────────────────────────────────────────────────────
if view_mode == "Overview":
    _ov_time = target_df.iloc[selected_idx]["time"]
    _ov_ts   = all_df[all_df["time"] == _ov_time].copy()

    _STATUS_BADGE_COLOR = {
        "NEEDS ACTION": "#d14343", "PREEMPTED": "#b85c00",
        "NEGLECTED":    "#b08000", "STALE":     "#7a5c00",
        "APPROACHING":  "#7b3fa0",
        "WATCH":        "#1f6fa8", "HEALTHY":   "#2ea043",
    }
    _STATUS_RGB = {
        "NEEDS ACTION": [230,  55,  55, 240],
        "PREEMPTED":    [215, 130,  25, 225],
        "NEGLECTED":    [215, 195,  30, 220],
        "STALE":        [175, 105,  45, 215],
        "APPROACHING":  [180,  80, 220, 225],
        "WATCH":        [ 85, 165, 235, 215],
        "HEALTHY":      [155, 165, 175, 195],
    }

    # ── Intro banner (first load only; dismissed per-session) ─────────────────
    if "show_intro_banner" not in st.session_state:
        st.session_state["show_intro_banner"] = True
    if st.session_state["show_intro_banner"]:
        _bi, _bd = st.columns([20, 1])
        with _bi:
            st.markdown(
                "<div style='background:#1a1f2e;border:1px solid #2d3550;"
                "border-radius:6px;padding:10px 14px;margin-bottom:8px;'>"
                "<span style='font-size:0.8rem;font-weight:600;color:#c8cfe0;'>"
                "Custody — Decision-led ISR reasoning system</span>"
                "<span style='font-size:0.75rem;color:#778;margin-left:12px;'>"
                "Screens maritime traffic, promotes targets of interest, and allocates "
                "limited sensor resources across competing priorities. "
                "Start with the top-ranked vessels to see what matters and why."
                "</span></div>",
                unsafe_allow_html=True,
            )
        with _bd:
            if st.button("✕", key="dismiss_intro", help="Dismiss"):
                st.session_state["show_intro_banner"] = False

    # ── KPI strip (always full portfolio) ─────────────────────────────────────
    st.markdown("<div class='section-label'>Portfolio at a Glance</div>", unsafe_allow_html=True)
    _kpis = compute_kpi_counts(_ov_ts)
    _kc = st.columns(6)
    with _kc[0]: st.metric("Tracked",      _kpis["total"])
    with _kc[1]: st.metric("Need Action",  _kpis["needs_action"])
    with _kc[2]: st.metric("Approaching",  _kpis["approaching"])
    with _kc[3]: st.metric("Neglected",    _kpis["neglected"])
    with _kc[4]: st.metric("Stale / Lost", _kpis["stale_or_lost"])
    with _kc[5]: st.metric("Preempted",    _kpis["preempted"])

    # ── Filters & Focus ────────────────────────────────────────────────────────
    with st.expander("Filters & Focus", expanded=False):
        _fc = st.columns([3, 1, 1, 1, 2])
        with _fc[0]:
            _f_status = st.multiselect(
                "Status", ALL_STATUSES, default=ALL_STATUSES, key="ov_f_status",
                help="Show only selected display-status categories",
            )
        with _fc[1]:
            _f_scripted  = st.checkbox("Scripted", value=False, key="ov_f_scripted",
                                       help="Show scripted actors only")
        with _fc[2]:
            _f_neglected = st.checkbox("Neglected", value=False, key="ov_f_neglected",
                                       help="Show neglected entities only")
        with _fc[3]:
            _f_stale     = st.checkbox("Stale/Lost", value=False, key="ov_f_stale",
                                       help="Show STALE or LOST custody tracks only")
        with _fc[4]:
            _f_topn_label = st.selectbox(
                "Top N", ["All", "Top 5", "Top 10", "Top 15"], index=0, key="ov_f_topn",
                help="Limit table and cards to N highest-urgency entities",
            )
    _f_topn_map = {"All": None, "Top 5": 5, "Top 10": 10, "Top 15": 15}
    _f_top_n = _f_topn_map[_f_topn_label]

    # Determine whether any filter is active (for map dimming)
    _any_filter = (
        set(_f_status) != set(ALL_STATUSES)
        or _f_scripted or _f_neglected or _f_stale
        or _f_top_n is not None
    )

    # Apply filters to get the focused subset
    _focused_ts = apply_overview_filters(
        _ov_ts,
        status_filter=_f_status if set(_f_status) != set(ALL_STATUSES) else None,
        scripted_only=_f_scripted,
        neglected_only=_f_neglected,
        stale_lost_only=_f_stale,
    )
    _focused_ts = filter_top_n(_focused_ts, _f_top_n)
    _focused_ids = set(_focused_ts["target_id"].tolist()) if not _focused_ts.empty else set()

    # ── Attention Now — top 5 cards (from focused set) ────────────────────────
    _top = top_attention_targets(_focused_ts, n=5)
    if not _top.empty:
        st.markdown("<div class='section-label'>Attention Now</div>", unsafe_allow_html=True)
        _card_cols = st.columns(min(len(_top), 5))
        for _ci, (_col, (_, _row)) in enumerate(zip(_card_cols, _top.iterrows())):
            _eid    = str(_row.get("target_id", "—"))
            _status = str(_row.get("display_status", "—"))
            _action = str(_row.get("action", "—"))
            _reason = str(_row.get("portfolio_reason", ""))
            _deferred = _row.get("deferred_for")
            _badge_c = _STATUS_BADGE_COLOR.get(_status, "#555")
            _short_reason = _reason[_reason.find(": ") + 2:] if ": " in _reason else _reason
            with _col:
                st.markdown(
                    f"<div style='border:1px solid #2d2d2d;border-radius:6px;"
                    f"padding:8px 10px;margin-bottom:4px;'>"
                    f"<div style='font-family:monospace;font-weight:700;"
                    f"font-size:0.9rem;margin-bottom:4px;'>{_eid}</div>"
                    f"<div style='display:inline-block;background:{_badge_c};color:#fff;"
                    f"border-radius:4px;padding:1px 7px;font-size:0.68rem;"
                    f"font-weight:600;margin-bottom:5px;'>{_status}</div>"
                    f"<div style='color:#999;font-size:0.68rem;margin-bottom:4px;"
                    f"line-height:1.3;'>{_short_reason[:80]}</div>"
                    f"<div style='color:#aaa;font-size:0.68rem;'>Action: {_action}"
                    f"{'<br>Deferred for: ' + str(_deferred) if pd.notna(_deferred) and str(_deferred) not in ('', 'None', 'nan') else ''}"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )
                if st.button("→ Inspect", key=f"btn_inspect_{_eid}_{_ci}"):
                    st.session_state["_pending_entity_select"] = _eid
                    st.session_state["_pending_view_mode"]     = "Entity Detail"
                    st.rerun()

    # ── Ranked portfolio table (focused) ──────────────────────────────────────
    _tbl_label = (
        f"Filtered Entities ({len(_focused_ts)}/{len(_ov_ts)}) — Ranked by Urgency"
        if _any_filter else
        "All Entities — Ranked by Urgency"
    )
    st.markdown(f"<div class='section-label'>{_tbl_label}</div>", unsafe_allow_html=True)
    _map_focused_id: str | None = None
    if _focused_ts.empty and _any_filter:
        st.caption("No entities match the current filters.")
    else:
        _ov_df = build_overview_df(_focused_ts)
        _tbl_sel = st.dataframe(
            _ov_df,
            use_container_width=True,
            height=min(550, 36 * len(_ov_df) + 38),
            on_select="rerun",
            selection_mode="single-row",
            key="ov_ranked_table",
        )
        # Derive focused entity from selected row (row indices are 1-based in _ov_df)
        _sel_rows = _tbl_sel.selection.rows
        if _sel_rows:
            # st.dataframe returns 0-based positional indices; _ov_df index starts at 1
            _sel_entity = _ov_df.iloc[_sel_rows[0]]["Entity"]
            _map_focused_id = str(_sel_entity) if pd.notna(_sel_entity) else None

    # ── Portfolio map — full portfolio; dim non-focused when filters active ───
    # Map header: label | labels toggle | ground-track toggle
    # (entity focus is driven by row selection in the ranked table above)
    _map_hdr_cols = st.columns([3, 2, 3])
    with _map_hdr_cols[0]:
        st.markdown("<div class='section-label'>Portfolio Map</div>", unsafe_allow_html=True)
    with _map_hdr_cols[1]:
        _show_labels = st.checkbox(
            "Labels", value=False, key="ov_show_labels",
            help="Show vessel ID labels for all entities on the map",
        )
    with _map_hdr_cols[2]:
        _show_ground_tracks = st.checkbox(
            "Sensor ground tracks", value=False, key="ov_show_ground_tracks",
            help="Overlay orbital ground-track segments (±90 min) for all sensor satellites",
        )

    # Build entity layer data; boost selected entity
    _map_rows = []
    for _, _mr in _ov_ts.iterrows():
        _eid_m      = str(_mr["target_id"])
        _ds         = derive_display_status(_mr.to_dict())
        _in_focus   = not _any_filter or _eid_m in _focused_ids
        _is_sel     = _eid_m == _map_focused_id
        if _in_focus:
            _rgb = list(_STATUS_RGB.get(_ds, [140, 140, 140, 120]))
            if _is_sel:
                _rgb[3] = 255          # full opacity for selected entity
            _radius = 5500 if _is_sel else (4000 if _ds in ("NEEDS ACTION", "NEGLECTED") else 2500)
        else:
            _rgb    = [50, 50, 50, 70]
            _radius = 1500
        _map_rows.append({
            "lon":      float(_mr["lon"]),
            "lat":      float(_mr["lat"]),
            "color":    _rgb,
            "radius":   _radius,
            "label":    _eid_m,
            "in_focus": _in_focus,
        })

    # Ghost-dot rings for approaching vessels (zone_probability > 0.5)
    _approaching_rows = []
    for _, _ar in _ov_ts.iterrows():
        try:
            _zp = float(_ar.get("zone_probability", 0.0))
        except (TypeError, ValueError):
            _zp = 0.0
        if _zp > 0.5:
            _approaching_rows.append({
                "lon": float(_ar["lon"]),
                "lat": float(_ar["lat"]),
            })

    # Future-position ghost dots + trajectory lines for approaching vessels
    _future_dots = []
    _future_lines = []
    for _, _fr in _ov_ts.iterrows():
        try:
            _zp = float(_fr.get("zone_probability", 0.0))
            _flat = float(_fr.get("future_lat", float("nan")))
            _flon = float(_fr.get("future_lon", float("nan")))
        except (TypeError, ValueError):
            continue
        if _zp > 0.4 and not math.isnan(_flat) and not math.isnan(_flon):
            _in_focus_fr = not _any_filter or str(_fr["target_id"]) in _focused_ids
            if _in_focus_fr:
                _future_dots.append({"lon": _flon, "lat": _flat})
                _future_lines.append({
                    "path": [
                        [float(_fr["lon"]), float(_fr["lat"])],
                        [_flon, _flat],
                    ]
                })

    _ov_zone_data = [
        {
            "name": z.name,
            "polygon": [
                [z.min_lon, z.min_lat], [z.max_lon, z.min_lat],
                [z.max_lon, z.max_lat], [z.min_lon, z.max_lat],
            ],
        }
        for z in ZONES
    ]
    _ov_zone_layer = pdk.Layer(
        "PolygonLayer", data=_ov_zone_data,
        get_polygon="polygon", get_fill_color=[255, 215, 0, 35],
        get_line_color=[255, 215, 0, 180], line_width_min_pixels=2,
        stroked=True, filled=True,
    )
    _ov_entity_layer = pdk.Layer(
        "ScatterplotLayer", data=_map_rows,
        get_position="[lon, lat]", get_radius="radius",
        get_fill_color="color", pickable=True,
    )

    _ov_approaching_layer = None
    if _approaching_rows:
        _ov_approaching_layer = pdk.Layer(
            "ScatterplotLayer", data=_approaching_rows,
            get_position="[lon, lat]",
            get_radius=9000,
            radius_min_pixels=10,
            get_fill_color=[0, 0, 0, 0],
            get_line_color=[180, 80, 220, 200],
            line_width_min_pixels=2,
            stroked=True, filled=False,
            pickable=False,
        )

    # Centered view: pan to selected entity at tighter zoom, or mean position
    _ov_center_lat, _ov_center_lon, _ov_zoom = build_focus_view_state(
        _ov_ts, _map_focused_id
    )

    # Assemble layers (bottom → top): ground tracks, zones, entities, trajectory, approaching rings, labels
    _ov_layers = [_ov_zone_layer, _ov_entity_layer]

    if _future_lines:
        _ov_layers.append(pdk.Layer(
            "PathLayer", data=_future_lines,
            get_path="path",
            get_color=[180, 80, 220, 100],
            get_width=800,
            width_min_pixels=1,
            pickable=False,
        ))
    if _future_dots:
        _ov_layers.append(pdk.Layer(
            "ScatterplotLayer", data=_future_dots,
            get_position="[lon, lat]",
            get_radius=3500,
            radius_min_pixels=4,
            get_fill_color=[180, 80, 220, 90],
            get_line_color=[180, 80, 220, 160],
            line_width_min_pixels=1,
            stroked=True, filled=True,
            pickable=False,
        ))
    if _ov_approaching_layer:
        _ov_layers.append(_ov_approaching_layer)

    if _show_ground_tracks:
        _gt_data = all_ground_track_layers_data(_ov_time)
        if _gt_data:
            _gt_layer = pdk.Layer(
                "PathLayer", data=_gt_data,
                get_path="path",
                get_color=[55, 190, 210, 70],   # muted teal — distinct from vessel layer
                get_width=1200,
                width_min_pixels=1,
            )
            _ov_layers.insert(0, _gt_layer)  # render below zones and entities

            # Current-position marker for each satellite — rendered on top of tracks
            _sat_pos = satellite_current_positions(_ov_time)
            if _sat_pos:
                _ov_layers.append(pdk.Layer(
                    "ScatterplotLayer", data=_sat_pos,
                    get_position="[lon, lat]",
                    get_radius=18000,
                    radius_min_pixels=7,
                    get_fill_color=[30, 220, 245, 220],
                    get_line_color=[255, 255, 255, 200],
                    stroked=True,
                    line_width_min_pixels=2,
                    pickable=False,
                ))

    # White ring around the focused entity
    if _map_focused_id:
        _sel_rows = _ov_ts[_ov_ts["target_id"] == _map_focused_id]
        if not _sel_rows.empty:
            _sel_highlight = [{
                "lon": float(_sel_rows.iloc[0]["lon"]),
                "lat": float(_sel_rows.iloc[0]["lat"]),
            }]
            _ov_layers.append(pdk.Layer(
                "ScatterplotLayer", data=_sel_highlight,
                get_position="[lon, lat]",
                get_radius=7000,
                get_fill_color=[0, 0, 0, 0],
                get_line_color=[255, 255, 255, 180],
                line_width_min_pixels=2,
                stroked=True, filled=False,
            ))

    # Vessel labels — all entities, toggled by the "Labels" checkbox.
    # Uses _map_rows directly (already has lon/lat/label for every entity).
    # Two-pass: shadow behind main for contrast on the dark basemap.
    # font_family uses inner single-quotes so pydeck serialises as a JS string
    # literal ("@@='monospace'" → "monospace") not a broken variable expression.
    if _show_labels:
        _label_rows = [r for r in _map_rows if r["in_focus"]]
    if _show_labels and _label_rows:
        _label_common = dict(
            get_position="[lon, lat]",
            get_text="label",
            get_size=11,
            font_weight=700,
            font_family="'monospace'",
            pickable=False,
        )
        _ov_layers.append(pdk.Layer(
            "TextLayer", data=_label_rows,
            **_label_common,
            get_color=[15, 15, 15, 160],
            get_pixel_offset=[1, -11],
        ))
        _ov_layers.append(pdk.Layer(
            "TextLayer", data=_label_rows,
            **_label_common,
            get_color=[245, 245, 245, 235],
            get_pixel_offset=[0, -12],
        ))

    _ov_deck = pdk.Deck(
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        initial_view_state=pdk.ViewState(
            latitude=_ov_center_lat, longitude=_ov_center_lon, zoom=_ov_zoom,
        ),
        layers=_ov_layers,
        tooltip={"text": "{label}"},
    )
    st.pydeck_chart(_ov_deck, use_container_width=True)
    _ov_legend = st.columns(6)
    with _ov_legend[0]: st.markdown("<small>🔴 Needs Action</small>", unsafe_allow_html=True)
    with _ov_legend[1]: st.markdown("<small>🟠 Preempted</small>",    unsafe_allow_html=True)
    with _ov_legend[2]: st.markdown("<small>🟡 Neglected</small>",    unsafe_allow_html=True)
    with _ov_legend[3]: st.markdown("<small>🟤 Stale</small>",        unsafe_allow_html=True)
    with _ov_legend[4]: st.markdown("<small>🔵 Watch</small>",        unsafe_allow_html=True)
    with _ov_legend[5]: st.markdown("<small>⚪ Healthy</small>",       unsafe_allow_html=True)

    # ── Event feed (changes since previous timestep) ───────────────────────────
    _prev_times = sorted(all_df[all_df["time"] < _ov_time]["time"].unique())
    _prev_ts = (
        all_df[all_df["time"] == _prev_times[-1]].copy()
        if _prev_times else pd.DataFrame()
    )
    _events = build_event_feed(_ov_ts, _prev_ts if not _prev_ts.empty else None, max_events=6)
    if _events:
        _EVENT_ICON = {
            "zone_entry":        "🚨",
            "health_worsened":   "📉",
            "neglect_triggered": "⏱",
            "rank_change":       "📊",
            "preempted":         "⛔",
        }
        st.markdown("<div class='section-label'>Recent Changes</div>", unsafe_allow_html=True)
        _ev_cols = st.columns(min(len(_events), 4))
        for _ei, _ev in enumerate(_events):
            _icon = _EVENT_ICON.get(_ev["event_type"], "•")
            with _ev_cols[_ei % len(_ev_cols)]:
                st.markdown(
                    f"<div style='border:1px solid #2d2d2d;border-radius:4px;"
                    f"padding:5px 8px;margin-bottom:4px;font-size:0.72rem;"
                    f"color:#ccc;line-height:1.35;'>"
                    f"{_icon} {_ev['description']}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<div style='border-top:1px solid #2a2a2a;padding-top:10px;"
        "font-size:0.68rem;color:#555;text-align:left;'>"
        "Custody &mdash; Decision-led ISR reasoning prototype &nbsp;·&nbsp; "
        "Built by Ben Metcalfe &nbsp;·&nbsp; 2026"
        "</div>",
        unsafe_allow_html=True,
    )

    st.stop()  # do not render entity detail content below

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

# ── Portfolio status ───────────────────────────────────────────────────────────
if "portfolio_rank" in current.index and pd.notna(current.get("portfolio_rank")):
    st.markdown("<div class='section-label'>Portfolio Status</div>", unsafe_allow_html=True)
    _n_entities = all_df["target_id"].nunique() if mode == "Simulation" else 1

    _health     = str(current.get("custody_health", "—"))
    _n_hours    = float(current.get("neglect_hours", 0.0))
    _neglected  = bool(current.get("neglect_flag", False))
    _deferred   = current.get("deferred_for")
    _reason     = str(current.get("portfolio_reason", ""))

    p1, p2, p3, p4, p5 = st.columns(5)
    with p1:
        st.metric("Portfolio Rank", f"{int(current['portfolio_rank'])} / {_n_entities}")
    with p2:
        st.metric("Portfolio Score", f"{float(current['portfolio_score']):.3f}")
    with p3:
        st.metric("Custody Health", _health)
    with p4:
        _neglect_label = f"{_n_hours:.1f} h  {'[NEGLECTED]' if _neglected else ''}"
        st.metric("Without Collection", _neglect_label.strip())
    with p5:
        st.metric("Deferred For", str(_deferred) if _deferred else "—")

    if _reason:
        st.caption(_reason)

    # ── Portfolio overview at this timestep (all entities ranked) ─────────────
    with st.expander("Portfolio overview — all entities at this step", expanded=False):
        _port_ts  = current["time"]
        _port_df  = all_df[all_df["time"] == _port_ts].copy()
        if "portfolio_rank" in _port_df.columns:
            _port_show = (
                _port_df
                .sort_values("portfolio_rank")[[
                    "portfolio_rank", "target_id", "portfolio_score",
                    "custody_health", "neglect_flag", "neglect_hours",
                    "action", "deferred_for",
                ]]
                .rename(columns={
                    "portfolio_rank":  "Rank",
                    "target_id":       "Entity",
                    "portfolio_score": "Score",
                    "custody_health":  "Health",
                    "neglect_flag":    "Neglected",
                    "neglect_hours":   "Neglect h",
                    "action":          "Action",
                    "deferred_for":    "Deferred For",
                })
                .reset_index(drop=True)
            )
            _port_show["Score"]      = _port_show["Score"].apply(lambda x: f"{x:.3f}")
            _port_show["Neglect h"]  = _port_show["Neglect h"].apply(lambda x: f"{x:.1f}")
            _port_show["Deferred For"] = _port_show["Deferred For"].fillna("—")
            st.dataframe(
                _port_show,
                use_container_width=True,
                height=min(420, 35 * len(_port_show) + 40),
            )

# ── Reasoning stack: Fusion Assessment → Decision → Task Queue ────────────────
prefix_window  = target_df.iloc[:selected_idx].to_dict("records")
current_record = target_df.iloc[selected_idx].to_dict()
render_entity_detail_panel(current_record, prefix_window)

# ── Alerts ────────────────────────────────────────────────────────────────────
st.markdown("<div class='section-label'>Alerts</div>", unsafe_allow_html=True)
alerts = alerts_for_timeline(target_df.iloc[: selected_idx + 1].to_dict("records"))
if alerts:
    alerts_df = pd.DataFrame([
        {"Time": a.timestamp, "Level": a.level, "Code": a.code, "Message": a.message}
        for a in alerts
    ])
    alerts_df["Time"] = alerts_df["Time"].astype(str).str.split("+").str[0]
    st.dataframe(alerts_df.reset_index(drop=True), use_container_width=True)
else:
    st.markdown("<span style='color:#666;'>No alerts for this target.</span>", unsafe_allow_html=True)

# ── Compound signals ──────────────────────────────────────────────────────────
# Shows only rules that fire at the selected step (not accumulated history).
st.markdown("<div class='section-label'>Compound Signals</div>", unsafe_allow_html=True)
st.caption("Active at selected step only. Adjust 'Min compound confidence' in the sidebar to filter.")
active_compounds = sorted(
    [s for s in evaluate_compounds(current_record, window=prefix_window)
     if s.confidence >= min_compound_conf],
    key=lambda s: s.confidence,
    reverse=True,
)
if active_compounds:
    compounds_df = build_active_compounds_df(active_compounds)
    st.dataframe(compounds_df.reset_index(drop=True), use_container_width=True)
else:
    st.markdown("<span style='color:#666;'>No active compound signals at this step.</span>", unsafe_allow_html=True)

# ── Compound history ──────────────────────────────────────────────────────────
# Aggregates all compound firings from step 0 through the selected step.
# Each code appears once; count, first/last seen, and peak confidence are shown.
st.markdown("<div class='section-label'>Compound History</div>", unsafe_allow_html=True)
st.caption("All compound firings from step 0 through the selected step, grouped by code.")
prefix_timeline = target_df.iloc[: selected_idx + 1].to_dict("records")
history_compounds = [
    c for c in compounds_for_timeline(prefix_timeline)
    if c.confidence >= min_compound_conf
]
if history_compounds:
    agg = aggregate_compound_history(history_compounds)
    history_df = build_compound_history_df(agg)
    st.dataframe(history_df.reset_index(drop=True), use_container_width=True)
else:
    st.markdown("<span style='color:#666;'>No compound history for this target.</span>", unsafe_allow_html=True)

# ── Available capabilities ────────────────────────────────────────────────────
st.markdown("<div class='section-label'>Available Capabilities</div>", unsafe_allow_html=True)

_cap_lat  = float(current["lat"])
_cap_lon  = float(current["lon"])
_cap_now  = current["time"]

_CAP_SATELLITES = {
    "OPTICAL": ["EO-MIO-1", "EO-MIO-2", "EO-SSO-1", "EO-SSO-2"],
    "SAR":     ["SAR-1", "SAR-2"],
}

_cap_rows = []
for _sensor in ("OPTICAL", "SAR", "AIS", "MONITOR"):
    if _sensor in _CAP_SATELLITES:
        _best = None
        for _sat in _CAP_SATELLITES[_sensor]:
            _pw = next_pass_window(_sat, _cap_lat, _cap_lon, _cap_now)
            if _pw is not None:
                if _best is None or _pw.time_to_start_seconds < _best.time_to_start_seconds:
                    _best = _pw
        if _best is None:
            _status, _window, _tts = "No Pass", "—", "—"
        elif _best.time_to_start_seconds <= 0:
            _status = "Available"
            _window = _best.start_time.strftime("%H:%Mz")
            _tts    = "Now"
        else:
            _tts_min = int(_best.time_to_start_seconds / 60)
            _status  = "Delayed"
            _window  = _best.start_time.strftime("%H:%Mz")
            _tts     = f"{_tts_min} min"
    else:
        _status = "Continuous"
        _window = "—"
        _tts    = "—"
    _cap_rows.append({"Capability": _sensor, "Status": _status,
                      "Next Window": _window, "TTS": _tts})

_STATUS_COLOR = {"Available": "#2ea043", "Delayed": "#b08000",
                 "Continuous": "#1f78b4", "No Pass": "#555"}

_cap_cols = st.columns(len(_cap_rows))
for _col, _row in zip(_cap_cols, _cap_rows):
    _sc = _STATUS_COLOR.get(_row["Status"], "#555")
    with _col:
        st.markdown(
            f"<div style='border:1px solid #2d2d2d;border-radius:6px;"
            f"padding:8px 12px;text-align:center;'>"
            f"<div style='font-family:monospace;font-weight:700;"
            f"font-size:0.9rem;margin-bottom:4px;'>{_row['Capability']}</div>"
            f"<div style='display:inline-block;background:{_sc};color:#fff;"
            f"border-radius:4px;padding:1px 8px;font-size:0.72rem;"
            f"font-weight:600;margin-bottom:4px;'>{_row['Status']}</div>"
            f"<div style='color:#aaa;font-size:0.75rem;'>"
            f"{_row['Next Window']}"
            f"{'&nbsp;&nbsp;<span style=\"color:#666\">TTS ' + _row['TTS'] + '</span>' if _row['TTS'] not in ('—', 'Now') else ''}"
            f"</div></div>",
            unsafe_allow_html=True,
        )

# ── Next Orbital Passes ───────────────────────────────────────────────────────
_orbital_rows = build_orbital_passes_rows(
    float(current["lat"]), float(current["lon"]), current["time"]
)
_orbital_rows_visible = [r for r in _orbital_rows if r["Status"] != "No Pass In Horizon"]
if _orbital_rows_visible:
    st.markdown("<div class='section-label'>Next Orbital Passes</div>", unsafe_allow_html=True)
    _orbital_df = pd.DataFrame(_orbital_rows_visible)
    for _col in ("Start", "End"):
        _orbital_df[_col] = _orbital_df[_col].apply(
            lambda v: str(v).split("+")[0] if v is not None else "—"
        )
    for _col in ("Duration (min)", "Time to Start (min)"):
        _orbital_df[_col] = _orbital_df[_col].apply(lambda v: v if v is not None else "—")
    _orbital_df["Within Threshold"] = _orbital_df["Within Threshold"].apply(
        lambda v: "✓" if v else "—"
    )
    st.dataframe(_orbital_df, use_container_width=True)
else:
    st.markdown("<span style='color:#666; font-size:0.8rem;'>No Orbital Passes In Horizon</span>", unsafe_allow_html=True)

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
    map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
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
st.dataframe(cot_df.reset_index(drop=True), use_container_width=True)

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
st.dataframe(timeline_df.reset_index(drop=True), use_container_width=True)

# ── Decision Traces ───────────────────────────────────────────────────────────
# Available in both Simulation and AIS Replay modes when traces are present.
st.markdown("<div class='section-label'>Decision Traces</div>", unsafe_allow_html=True)
st.caption("Structured breakdown of each planner decision up to the selected step.")
_trace_display_cols = [
    "Time", "Vessel", "Action", "Hold Reason", "Chosen Sensor",
    "Priority", "Task Value",
    "Accessible Sensors", "Claimed Higher", "Final Pool",
]
_trace_records = visible_df.to_dict("records")
_traces = [r["decision_trace"] for r in _trace_records if isinstance(r.get("decision_trace"), DecisionTrace)]
if _traces:
    _trace_rows = traces_to_rows(_traces)
    _traces_df = pd.DataFrame(_trace_rows)[_trace_display_cols].copy()
    _traces_df["Time"] = _traces_df["Time"].astype(str).str.split("+").str[0]
    st.dataframe(_traces_df.reset_index(drop=True), use_container_width=True)
else:
    st.markdown("<span style='color:#666;'>No traces available.</span>", unsafe_allow_html=True)

# ── What-If Results ───────────────────────────────────────────────────────────
if st.session_state.get("whatif_results"):
    st.markdown("<div class='section-label'>What-If Results</div>", unsafe_allow_html=True)
    st.caption("Comparison across variants. Baseline uses current config; variants use sidebar overrides.")
    _wf_df = build_whatif_results_df(st.session_state.whatif_results)
    st.dataframe(_wf_df.reset_index(drop=True), use_container_width=True)

if st.session_state.playing:
    time.sleep(2.5)
    if st.session_state.playback_idx < max_idx:
        st.session_state.playback_idx += 1
    else:
        st.session_state.playing = False   # stop at end rather than looping
    st.rerun()
