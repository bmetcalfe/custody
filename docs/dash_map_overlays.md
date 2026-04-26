# Dashboard Map / Evidence Overlays

*Companion to [docs/dash_whitsun_replay.md](dash_whitsun_replay.md), [docs/sentinel_ingestion_demo_aois.md](sentinel_ingestion_demo_aois.md), and the slice that replaced the Whitsun "Context (map placeholder)" panel with a real map.*

---

## What this is

A real `dash_deck` (`pydeck`) map panel mounted on both the **Whitsun replay** tab and the **Tennent monitoring** tab.  Both tabs share one rendering helper (`src/app/layout/map_overlays_helpers.py`) and one committed manifest (`data/demo/map_overlays.fixture.json`).

The map renders:

- **AOI polygon** outline.
- **Observation footprints** (Umbra / Sentinel-1 / Sentinel-2 / simulated), gated by event ordinal on the Whitsun tab and static on the Tennent tab.
- **Candidate-track markers** at the lat/lon of each track's initial detection (Whitsun only).
- **Latest custody snapshot** as a small on-map text annotation (Whitsun only).
- **(Future)** real raster overlays as `BitmapLayer`s when imagery lands.

## What this is not

- **Not a full geospatial imagery pipeline.** No tile server, no PMTiles, no raster reprojection at runtime.
- **Not a live data fetch.** No CDSE STAC call, no WMS, no XYZ tile pull beyond the public CARTO basemap that the existing overview map already targets.
- **Not faked imagery.** Every overlay that lacks a real image asset says so, both in its panel badge (`image_kind: "footprint-only"`) and in the `image asset not available for:` block under the map.
- **Not a refactor of the existing overview map.** The Custody-overview tab keeps its existing `dash_deck` layout untouched; the new helper module lives next to it.

## Overlay manifest

Every overlay is one entry in `data/demo/map_overlays.fixture.json` matching the `OverlayArtifact` schema in `src/custody/demo/map_overlays.py`:

```json
{
  "overlay_id": "whitsun-umbra-20231206",
  "scenario_id": "whitsun",
  "observation_id": "fixture-umbra-whitsun-20231206",
  "source": "umbra",
  "sensor_type": "sar",
  "display_name": "Umbra SAR 2023-12-06",
  "data_mode": "fixture",
  "image_path": null,
  "asset_url": null,
  "image_kind": "footprint-only",
  "bounds": [114.45, 9.75, 114.85, 10.25],
  "geometry": { "type": "Polygon", "coordinates": [...] },
  "opacity_default": 0.65,
  "visible_from_event_ordinal": 2,
  "z_index": 30,
  "confidence_weight": 1.0,
  "usable_for_detection": true,
  "usable_for_context": true,
  "caveats": ["high-confidence tasked SAR; canonical evidence"],
  "missing_asset_reason": "no committed georeferenced Umbra raster; footprint-only"
}
```

`image_kind` is one of `png`, `geotiff`, `cog`, `thumbnail`, `footprint-only`.  The renderer special-cases `footprint-only` (outline only, with the missing-asset note); when any of the four imagery kinds appear with a populated `image_path` or `asset_url`, the renderer can be extended to add a `BitmapLayer` keyed on `bounds`.

## Layer toggles + opacity

The same checklist appears on both tabs:

- AOI · Tracks · Custody state · Footprints · Umbra SAR · Sentinel-1 · Sentinel-2

Footprints are filtered per source (the `Umbra SAR` / `Sentinel-1` / `Sentinel-2` toggles only affect their own source); the broader `Footprints` toggle hides every footprint at once.  A single opacity slider (0.0–1.0, step 0.05) scales the footprint fill alpha.

## Whitsun progressive reveal

The Whitsun map is gated by the same `selected_event_id` store as the rest of the replay tab.  An overlay is visible at the current step if `overlay.visible_from_event_ordinal <= current_ordinal`.

| Event | Map state                                                                 |
| ----- | ------------------------------------------------------------------------- |
| 01    | AOI only.  No footprints, no tracks, no custody annotation.              |
| 02    | + Umbra SAR footprint.                                                   |
| 03    | (no map change; VLM evidence renders elsewhere.)                         |
| 04    | + Sentinel-2 low-cloud footprint, + Sentinel-2 cloudy footprint.         |
| 05    | + Sentinel-1 GRD footprint.                                              |
| 06    | + candidate-track markers (trk-001, trk-002, trk-003).                   |
| 07    | + custody-state annotation goes from healthy → ambiguous.                |
| 12    | + simulated follow-up Umbra footprint.                                   |
| 13–14 | custody-state annotation goes back to healthy.                           |

The Tennent map is not gated — every Tennent overlay is visible from ordinal 0.

## Imagery acquisition needs

To upgrade footprint-only overlays to true raster overlays, the following assets are needed.  Once any of these land under `data/demo/overlays/`, only the corresponding manifest entry needs `image_kind` / `image_path` / `bounds` updated.

| Scenario | Observation | Required asset |
| --- | --- | --- |
| Whitsun | `fixture-umbra-whitsun-20231206` | Georeferenced PNG + bounds, or COG |
| Whitsun | `fixture-s1-grd-whitsun-20231210` | PNG preview + bounds, or COG |
| Whitsun | `fixture-s2-l2a-whitsun-20231212-low-cloud` | RGB PNG preview + bounds, or COG |
| Whitsun | `fixture-s2-l2a-whitsun-20231215-cloudy` | RGB PNG preview + bounds, or COG |
| Whitsun | `fixture-umbra-whitsun-20231213-followup` (simulated) | Optional placeholder, marked `data_mode: simulated` |
| Tennent | `fixture-s1-grd-tennent-20230715` | PNG preview + bounds, or COG |
| Tennent | `fixture-s2-l2a-tennent-20230718-low-cloud` | PNG preview + bounds, or COG |
| Tennent | `fixture-s2-l2a-tennent-20230728-cloudy` | PNG preview + bounds, or COG |

`rasterio>=1.5` is already a runtime dependency, so a small COG → PNG-with-bounds conversion helper can land in a follow-up without adding any new dependency.

## Files added in this slice

```
data/demo/
  map_overlays.fixture.json                      # the manifest

src/custody/demo/
  map_overlays.py                                # frozen OverlayArtifact + loader
  __init__.py                                    # exports added

src/app/layout/
  map_overlays_helpers.py                        # build_deck_json, layer toggles
  whitsun_replay.py                              # map panel mounted, IDs added
  tennent_monitoring.py                          # map panel mounted, IDs added

src/app/callbacks/
  whitsun_replay.py                              # +1 callback driving the Whitsun deck
  tennent_monitoring.py                          # +1 callback driving the Tennent deck

src/app/dash_app.py                              # registers tennent_monitoring callbacks

tests/
  test_demo_map_overlays.py                      # 21 cases
  test_demo_decision_trace_loader.py             # placeholder-removal regression test
  test_dash_phase2.py                            # callback count guard 16 → 18
  test_dash_tennent_monitoring.py                # same guard 16 → 18
  test_dash_sidebar_collapse.py                  # same guard 16 → 18

docs/
  dash_map_overlays.md                           # this document
  dash_whitsun_replay.md                         # context-map row updated
```

## Tests

`tests/test_demo_map_overlays.py` covers:

- The manifest loads via `load_map_overlays()`; explicit-path loading works; invalid scenario / image_kind values raise `ValueError`.
- Whitsun ordinal gating: ordinal 1 → AOI only; 2 → Umbra appears; 4 → Sentinel-2 (both) appears; 5 → Sentinel-1 appears; 12 → simulated follow-up appears; 14 → all six overlays.
- Tennent overlays are static (every Tennent overlay always visible) and their bounds lie inside the Tennent AOI bbox.
- No overlay silently pretends to be imagery: every observation-bearing footprint-only entry carries a non-empty `missing_asset_reason` and `has_image_asset` returns `False`.
- `build_deck_json()` returns parseable JSON; per-source toggles filter the rendered layers; `visibility_from_checked()` builds the right dict.
- The Whitsun and Tennent layouts mount all the expected map IDs and the full Dash app registers both map callbacks.

No test contacts the live CDSE STAC endpoint, fetches imagery, or starts a Dash server.
