"""Shared deck-rendering helpers for the Whitsun and Tennent map panels.

Both tabs use ``dash_deck.DeckGL`` driven by a JSON spec produced from
``custody.demo.OverlayArtifact``s.  This module owns the layer
construction so the two callback modules stay thin.

No live data, no external service.  No new dependencies — uses
``pydeck`` (already in pyproject) and the dark CARTO basemap that the
existing overview map already targets.
"""
from __future__ import annotations

import json as _json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import pydeck as pdk

from custody.demo.map_overlays import OverlayArtifact, has_image_asset


_BASEMAP = (
    "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
)


# Per-source RGBA colours.  The Umbra colour is the same accent used in
# the Whitsun replay panel so the visual hierarchy survives the map.
_SOURCE_FILL = {
    "umbra": [94, 234, 212, 90],            # accent cyan
    "sentinel-1": [99, 179, 237, 70],       # blue
    "sentinel-2": [251, 191, 36, 70],       # amber
    "simulated": [156, 163, 175, 60],       # neutral grey
}

_SOURCE_STROKE = {
    "umbra": [94, 234, 212],
    "sentinel-1": [99, 179, 237],
    "sentinel-2": [251, 191, 36],
    "simulated": [156, 163, 175],
}

_AOI_STROKE = [255, 255, 255]
# Default AOI is outline-only so the imagery overlays stay readable.
# A fill is allowed but only at very low opacity (max 0.05) so the
# fallback can never wash out the evidence.
_AOI_FILL_ZERO = [255, 255, 255, 0]
_AOI_MAX_FALLBACK_FILL_ALPHA = int(round(0.05 * 255))
_TRACK_FILL = [251, 191, 36, 200]
_TRACK_STROKE = [255, 255, 255]


def _scale_color(rgba: Sequence[int], opacity: float) -> list[int]:
    """Multiply the alpha channel by ``opacity`` (0.0-1.0)."""
    base_rgba = list(rgba)
    if len(base_rgba) == 3:
        base_rgba.append(255)
    base_rgba[3] = int(max(0.0, min(1.0, opacity)) * base_rgba[3])
    return base_rgba


# ---------------------------------------------------------------------------
# Layer constructors
# ---------------------------------------------------------------------------


def _bitmap_layer_specs(
    overlays: Iterable[OverlayArtifact],
    opacity: float,
) -> list[dict[str, Any]]:
    """Raw BitmapLayer specs (deck.gl JSON shape).

    Returned as plain dicts because ``pdk.Layer("BitmapLayer", image=url)``
    wraps the URL in a ``pydeck.types.image.Image`` that base64-encodes a
    local file at serialization time — wrong for a URL that the browser
    fetches relative to the Dash assets endpoint.
    """
    sorted_overlays = sorted(
        (o for o in overlays if has_image_asset(o) and o.bounds is not None),
        key=lambda o: o.z_index,
    )
    out: list[dict[str, Any]] = []
    for o in sorted_overlays:
        url = o.asset_url or o.image_path
        if not url:
            continue
        out.append({
            "@@type": "BitmapLayer",
            "id": f"bitmap-{o.overlay_id}",
            "image": url,
            "bounds": list(o.bounds),
            "opacity": max(0.0, min(1.0, opacity * o.opacity_default)),
            "pickable": False,
        })
    return out


def _build_aoi_layer(
    overlays: Iterable[OverlayArtifact],
) -> pdk.Layer | None:
    aois = [o for o in overlays if o.observation_id is None and o.geometry]
    if not aois:
        return None
    data = []
    for o in aois:
        coords = o.geometry.get("coordinates") if o.geometry else None
        if not coords:
            continue
        data.append({
            "polygon": coords[0],
            "label": o.display_name,
            "data_mode": o.data_mode,
            "tooltip": (
                f"{o.display_name} · AOI polygon · {o.data_mode}"
            ),
        })
    if not data:
        return None
    return pdk.Layer(
        "PolygonLayer",
        data=data,
        get_polygon="polygon",
        stroked=True,
        filled=False,
        # Belt-and-suspenders: even though ``filled=False`` should
        # suppress the fill, set the colour explicitly to fully
        # transparent so deck.gl cannot fall back to a strong default.
        get_fill_color=_AOI_FILL_ZERO,
        # Thin but readable outline.  Pin the upper bound so the line
        # does not balloon at high zoom.
        line_width_min_pixels=1,
        line_width_max_pixels=2,
        get_line_color=_AOI_STROKE,
        pickable=True,
    )


def _build_footprint_layer(
    overlays: Iterable[OverlayArtifact], opacity: float,
) -> pdk.Layer | None:
    footprints = [
        o for o in overlays
        if o.observation_id is not None and o.geometry
    ]
    if not footprints:
        return None
    data = []
    for o in footprints:
        coords = o.geometry.get("coordinates") if o.geometry else None
        if not coords:
            continue
        fill = _scale_color(
            _SOURCE_FILL.get(o.source, [156, 163, 175, 60]),
            opacity * o.opacity_default,
        )
        stroke = _SOURCE_STROKE.get(o.source, [156, 163, 175])
        data.append({
            "polygon": coords[0],
            "label": o.display_name,
            "fill_color": fill,
            "line_color": stroke,
            "data_mode": o.data_mode,
            "tooltip": (
                f"{o.display_name} · {o.source} · "
                f"{o.image_kind} · {o.data_mode}"
            ),
        })
    if not data:
        return None
    return pdk.Layer(
        "PolygonLayer",
        data=data,
        get_polygon="polygon",
        get_fill_color="fill_color",
        get_line_color="line_color",
        stroked=True,
        filled=True,
        line_width_min_pixels=1,
        pickable=True,
    )


def _build_track_layer(tracks: Iterable[Mapping[str, Any]]) -> pdk.Layer | None:
    data = []
    for t in tracks:
        if t.get("lon") is None or t.get("lat") is None:
            continue
        data.append({
            "position": [float(t["lon"]), float(t["lat"])],
            "label": t.get("label", t.get("track_id", "")),
            "tooltip": (
                f"track {t.get('track_id', '')} · {t.get('label', '')}"
            ),
        })
    if not data:
        return None
    return pdk.Layer(
        "ScatterplotLayer",
        data=data,
        get_position="position",
        get_fill_color=_TRACK_FILL,
        get_line_color=_TRACK_STROKE,
        get_radius=200,
        radius_min_pixels=4,
        radius_max_pixels=10,
        line_width_min_pixels=1,
        stroked=True,
        pickable=True,
    )


def _build_custody_text_layer(
    custody_label: str | None,
    center_lon: float, center_lat: float,
) -> pdk.Layer | None:
    if not custody_label:
        return None
    return pdk.Layer(
        "TextLayer",
        data=[{
            "position": [center_lon, center_lat],
            "text": custody_label,
        }],
        get_position="position",
        get_text="text",
        get_size=14,
        get_color=[94, 234, 212],
        background=True,
        get_background_color=[26, 26, 26, 200],
        billboard=True,
    )


# ---------------------------------------------------------------------------
# Top-level deck builder
# ---------------------------------------------------------------------------


def build_deck_json(
    *,
    overlays: Iterable[OverlayArtifact],
    base_visibility: Mapping[str, bool] | None = None,
    opacity: float,
    center_lat: float,
    center_lon: float,
    zoom: float = 11.0,
    tracks: Iterable[Mapping[str, Any]] = (),
    custody_label: str | None = None,
    layer_visibility: Mapping[str, bool] | None = None,
) -> str:
    """Render a deck JSON for the chosen overlays + tracks + custody label.

    ``overlays`` is the already-filtered set the dashboard wants drawn —
    AOI overlays plus whichever observation-bound overlays the user has
    selected via the per-overlay toggle manager.  Per-source filtering
    is no longer the renderer's job; the caller owns that.

    ``base_visibility`` keys: ``aoi``, ``tracks``, ``custody``.  These
    gate the rendering of layers that are not driven by per-overlay
    toggles (the AOI polygon, track markers, the custody-state label).

    ``layer_visibility`` is a deprecated alias kept for callers that
    still pass the legacy keyset; only ``aoi`` / ``tracks`` / ``custody``
    are honoured from it.  Per-source keys (``umbra`` /
    ``sentinel_1`` / ``sentinel_2``) and ``footprints`` are ignored;
    drop those toggles before calling.
    """
    overlays = tuple(overlays)
    bv = dict(base_visibility or layer_visibility or {})

    observation_overlays = tuple(
        o for o in overlays if o.observation_id is not None
    )
    bitmap_specs = _bitmap_layer_specs(observation_overlays, opacity)

    # Build pydeck layers in two buckets so we can splice the raw
    # bitmap specs in between.  Final z-order (bottom to top):
    #   basemap, AOI outline, imagery overlays (BitmapLayers),
    #   footprints, tracks, custody label.
    aoi_layers: list[pdk.Layer] = []
    if bv.get("aoi", True):
        l = _build_aoi_layer(overlays)
        if l is not None:
            aoi_layers.append(l)

    upper_layers: list[pdk.Layer] = []
    l = _build_footprint_layer(observation_overlays, opacity)
    if l is not None:
        upper_layers.append(l)

    if bv.get("tracks", True):
        l = _build_track_layer(tracks)
        if l is not None:
            upper_layers.append(l)

    if bv.get("custody", True):
        l = _build_custody_text_layer(custody_label, center_lon, center_lat)
        if l is not None:
            upper_layers.append(l)

    deck = pdk.Deck(
        map_style=_BASEMAP,
        initial_view_state=pdk.ViewState(
            latitude=center_lat, longitude=center_lon, zoom=zoom,
        ),
        layers=aoi_layers + upper_layers,
    )
    spec = _json.loads(deck.to_json())
    serialized_layers = list(spec.get("layers") or [])
    if bitmap_specs:
        # Insert bitmap rasters between the AOI outline and the
        # footprint / track / custody layers so the AOI sits beneath
        # the imagery (no fill muddying) while operator-relevant
        # markers stay drawn on top of the imagery.
        aoi_count = len(aoi_layers)
        serialized_layers = (
            serialized_layers[:aoi_count]
            + bitmap_specs
            + serialized_layers[aoi_count:]
        )
    spec["layers"] = serialized_layers
    return _json.dumps(spec)


def overlays_with_missing_imagery(
    overlays: Iterable[OverlayArtifact],
) -> tuple[OverlayArtifact, ...]:
    return tuple(
        o for o in overlays
        if o.observation_id is not None and not has_image_asset(o)
    )


# ---------------------------------------------------------------------------
# Base-layer toggle config (per-overlay toggles are dynamic and live on
# each tab; these are the always-present base layers).
# ---------------------------------------------------------------------------


BASE_LAYER_TOGGLE_OPTIONS: tuple[tuple[str, str], ...] = (
    ("aoi", "AOI"),
    ("tracks", "Tracks"),
    ("custody", "Custody state"),
)
# Backward-compatible alias retained because external callers may still
# import the old name; the value is the new base-toggle list.
LAYER_TOGGLE_OPTIONS = BASE_LAYER_TOGGLE_OPTIONS


def default_base_visibility() -> dict[str, bool]:
    return {key: True for key, _ in BASE_LAYER_TOGGLE_OPTIONS}


# Backward-compatible alias.
default_visibility = default_base_visibility


def base_visibility_from_checked(
    values: Iterable[str] | None,
) -> dict[str, bool]:
    """Build a base-visibility dict from a Dash checklist's selected values."""
    selected = set(values or ())
    return {key: (key in selected) for key, _ in BASE_LAYER_TOGGLE_OPTIONS}


# Backward-compatible alias.
visibility_from_checked = base_visibility_from_checked
