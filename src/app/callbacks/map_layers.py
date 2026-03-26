"""Map callback: rebuild pydeck layers on timestep or entity change."""
from __future__ import annotations

import math

import pydeck as pdk
from dash import Dash, Input, Output, no_update

import state as app_state
from adapter import records_at_timestep
from custody.config import ZONES
from ground_track import all_ground_track_layers_data, satellite_current_positions
from layout.map_panel import OVERVIEW_MAP
from portfolio_overview import build_focus_view_state, derive_display_status

# Status → RGBA for map scatter points
_STATUS_RGB = {
    "NEEDS ACTION": [230, 55, 55, 240],
    "PREEMPTED":    [215, 130, 25, 225],
    "NEGLECTED":    [215, 195, 30, 220],
    "STALE":        [175, 105, 45, 215],
    "APPROACHING":  [180, 80, 220, 225],
    "WATCH":        [85, 165, 235, 215],
    "HEALTHY":      [155, 165, 175, 195],
}

_BASEMAP = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"


def _build_deck_json(
    ts_records: list[dict],
    selected_entity: str | None,
) -> str:
    """Build the full pydeck Deck JSON from one timestep's records."""
    import pandas as pd

    ts_df = pd.DataFrame(ts_records)
    if ts_df.empty:
        deck = pdk.Deck(
            map_style=_BASEMAP,
            initial_view_state=pdk.ViewState(latitude=1.0, longitude=0.5, zoom=6),
            layers=[],
        )
        return deck.to_json()

    ts_time = ts_records[0]["time"]

    # ── Entity scatter data ──────────────────────────────────────────────
    map_rows = []
    approaching_rows = []
    future_dots = []
    future_lines = []

    for r in ts_records:
        eid = r["target_id"]
        ds = derive_display_status(r)
        is_sel = eid == selected_entity
        rgb = list(_STATUS_RGB.get(ds, [140, 140, 140, 120]))
        if is_sel:
            rgb[3] = 255
        radius = 5500 if is_sel else (4000 if ds in ("NEEDS ACTION", "NEGLECTED") else 2500)
        map_rows.append({
            "lon": float(r["lon"]),
            "lat": float(r["lat"]),
            "color": rgb,
            "radius": radius,
            "label": eid,
        })

        # Approaching rings
        try:
            zp = float(r.get("zone_probability", 0.0))
        except (TypeError, ValueError):
            zp = 0.0
        if zp > 0.5:
            approaching_rows.append({"lon": float(r["lon"]), "lat": float(r["lat"])})

        # Future trajectory
        try:
            flat = float(r.get("future_lat", float("nan")))
            flon = float(r.get("future_lon", float("nan")))
        except (TypeError, ValueError):
            continue
        if zp > 0.4 and not math.isnan(flat) and not math.isnan(flon):
            future_dots.append({"lon": flon, "lat": flat})
            future_lines.append({
                "path": [[float(r["lon"]), float(r["lat"])], [flon, flat]]
            })

    # ── Zone polygons ────────────────────────────────────────────────────
    zone_data = [
        {
            "name": z.name,
            "polygon": [
                [z.min_lon, z.min_lat], [z.max_lon, z.min_lat],
                [z.max_lon, z.max_lat], [z.min_lon, z.max_lat],
            ],
        }
        for z in ZONES
    ]

    # ── Assemble layers (bottom → top) ───────────────────────────────────
    layers = []

    # Ground tracks
    gt_data = all_ground_track_layers_data(ts_time)
    if gt_data:
        layers.append(pdk.Layer(
            "PathLayer", data=gt_data,
            get_path="path", get_color=[55, 190, 210, 70],
            get_width=1200, width_min_pixels=1,
        ))
        sat_pos = satellite_current_positions(ts_time)
        if sat_pos:
            layers.append(pdk.Layer(
                "ScatterplotLayer", data=sat_pos,
                get_position="[lon, lat]", get_radius=18000,
                radius_min_pixels=7,
                get_fill_color=[30, 220, 245, 220],
                get_line_color=[255, 255, 255, 200],
                stroked=True, line_width_min_pixels=2, pickable=False,
            ))

    # Zones
    layers.append(pdk.Layer(
        "PolygonLayer", data=zone_data,
        get_polygon="polygon", get_fill_color=[255, 215, 0, 35],
        get_line_color=[255, 215, 0, 180], line_width_min_pixels=2,
        stroked=True, filled=True,
    ))

    # Entities
    layers.append(pdk.Layer(
        "ScatterplotLayer", data=map_rows,
        get_position="[lon, lat]", get_radius="radius",
        get_fill_color="color", pickable=True,
    ))

    # Future trajectory lines
    if future_lines:
        layers.append(pdk.Layer(
            "PathLayer", data=future_lines,
            get_path="path", get_color=[180, 80, 220, 100],
            get_width=800, width_min_pixels=1, pickable=False,
        ))
    if future_dots:
        layers.append(pdk.Layer(
            "ScatterplotLayer", data=future_dots,
            get_position="[lon, lat]", get_radius=3500, radius_min_pixels=4,
            get_fill_color=[180, 80, 220, 90],
            get_line_color=[180, 80, 220, 160],
            line_width_min_pixels=1, stroked=True, filled=True, pickable=False,
        ))

    # Approaching rings
    if approaching_rows:
        layers.append(pdk.Layer(
            "ScatterplotLayer", data=approaching_rows,
            get_position="[lon, lat]", get_radius=9000, radius_min_pixels=10,
            get_fill_color=[0, 0, 0, 0],
            get_line_color=[180, 80, 220, 200],
            line_width_min_pixels=2, stroked=True, filled=False, pickable=False,
        ))

    # Selection highlight ring
    if selected_entity:
        sel = [r for r in ts_records if r["target_id"] == selected_entity]
        if sel:
            layers.append(pdk.Layer(
                "ScatterplotLayer",
                data=[{"lon": float(sel[0]["lon"]), "lat": float(sel[0]["lat"])}],
                get_position="[lon, lat]", get_radius=7000,
                get_fill_color=[0, 0, 0, 0],
                get_line_color=[255, 255, 255, 180],
                line_width_min_pixels=2, stroked=True, filled=False,
            ))

    # Labels (top-5 by rank + selected)
    label_rows = []
    ranked = sorted(ts_records, key=lambda r: r.get("portfolio_rank", 999))
    top_ids = {r["target_id"] for r in ranked[:5]}
    if selected_entity:
        top_ids.add(selected_entity)
    for r in ts_records:
        if r["target_id"] in top_ids:
            label_rows.append({
                "lon": float(r["lon"]),
                "lat": float(r["lat"]),
                "label": r["target_id"],
            })
    if label_rows:
        label_common = dict(
            get_position="[lon, lat]", get_text="label",
            get_size=11, font_weight=700, font_family="'monospace'",
            pickable=False,
        )
        layers.append(pdk.Layer(
            "TextLayer", data=label_rows, **label_common,
            get_color=[15, 15, 15, 160], get_pixel_offset=[1, -11],
        ))
        layers.append(pdk.Layer(
            "TextLayer", data=label_rows, **label_common,
            get_color=[245, 245, 245, 235], get_pixel_offset=[0, -12],
        ))

    # ── View state ───────────────────────────────────────────────────────
    center_lat, center_lon, zoom = build_focus_view_state(ts_df, selected_entity)

    deck = pdk.Deck(
        map_style=_BASEMAP,
        initial_view_state=pdk.ViewState(
            latitude=center_lat, longitude=center_lon, zoom=zoom,
        ),
        layers=layers,
        tooltip={"text": "{label}"},
    )
    return deck.to_json()


def register(app: Dash) -> None:
    """Register the map update callback."""

    @app.callback(
        Output(OVERVIEW_MAP, "data"),
        Input(app_state.SCENARIO_KEY, "data"),
        Input(app_state.TIMESTEP_INDEX, "data"),
        Input(app_state.SELECTED_ENTITY, "data"),
    )
    def update_map(scenario_key, timestep_idx, selected_entity):
        if not scenario_key or timestep_idx is None:
            return no_update
        records = app_state.get_records(scenario_key)
        ts_records = records_at_timestep(records, timestep_idx)
        if not ts_records:
            return no_update
        return _build_deck_json(ts_records, selected_entity)
