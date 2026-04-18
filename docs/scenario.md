# Custody — Scenario

Locked as of Day 0 reconnaissance. Updates to this file are phase transitions and require updating CLAUDE.md and the implementation guide together.

## Area of Interest

Spratly Islands hotspot in the South China Sea.

- Bounding box: 114.5°E – 117.5°E, 8.5°N – 11.0°N
- Size: approximately 330 km × 280 km
- EPSG for internal use: 4326 (WGS84) throughout; project only at compute/display boundaries

### Anchor features inside the AOI

| Feature | Lat | Lon | Role in demo |
|---|---|---|---|
| Cuarteron Reef area | 8.856°N | 114.665°E | Hero target. 5 Umbra scenes, 41-day repeat window. |
| Union Banks / Sin Cowe area | 9.98°N | 114.63°E | Secondary arc. 3 Umbra scenes Dec 2023 – Mar 2024. |
| Second Thomas Shoal | 9.75°N | 115.87°E | Reference feature (not imaged by Umbra ODP in our window). |
| Sabina Shoal | 9.77°N | 116.55°E | Reference feature (2024 standoff site, outside primary window). |
| Mischief Reef | 9.90°N | 115.55°E | Militia hub reference. |
| Whitsun Reef | 9.90°N | 115.20°E | Historic militia massing reference. |

The hero target at 114.665°E / 8.856°N is approximately 25 nautical miles northeast of Cuarteron Reef, within commercial shipping reach but far enough from any single feature to be interesting in its own right.

## Time Window

Primary demo window: **June 1 – August 20, 2023** (approximately 11 weeks).

Driven by Umbra scene availability, not by choice. The Sabina Shoal 2024 standoff narrative was considered and rejected in Day 0 scoping because no Umbra coverage of the AOI exists during that period.

### Rationale for the 11-week window

- **June:** pattern-of-life baseline period. AIS + Sentinel-1 continuous, no Umbra scenes yet, commercial traffic flows through northern edge of AOI.
- **July 2:** first Umbra scene. Persistent target emerges at Cuarteron-area feature.
- **July 23:** second Umbra scene. Same feature. Anomaly score climbing.
- **August 7:** third Umbra scene. Confirms ongoing presence.
- **August 9:** fourth Umbra scene. 48 hours after prior scene. This is the money shot for the tipcue layer's demonstration — the tight revisit cadence mirrors what a real cued system would produce.
- **August 13:** fifth Umbra scene. Confirms loitering pattern persisted.
- **August 20:** buffer for post-cluster observations and demo wrap.

## Umbra Scene Inventory

All scenes confirmed present in `s3://umbra-open-data-catalog/sar-data/tasks/ship_detection_testdata/` via Day 0 centroid scan. See `day0/ship_detection_centroids.csv` for the full catalog.

### Primary cluster — Cuarteron-area feature (114.665°E / 8.856°N)

| Date (UTC) | Notes |
|---|---|
| 2023-07-02 | Scene 1. First appearance of target. |
| 2023-07-23 | Scene 2. Same feature. Length estimate in militia band. |
| 2023-08-07 | Scene 3. Sustained presence. |
| 2023-08-09 | Scene 4. 48 h after Scene 3 — the cued-pass demonstration. |
| 2023-08-13 | Scene 5. Confirms loitering pattern. |

### Secondary cluster — Union Banks / Sin Cowe area (114.63°E / 9.98°N)

| Date (UTC) | Notes |
|---|---|
| 2023-12-06 | Scene 1. |
| 2023-12-06 | Scene 2. Same date, later acquisition. |
| 2024-03-20 | Scene 3. Months-later revisit. |

These three scenes enable a supplementary "long-baseline pattern of life" case study documented in the repo but not included in the 90-second demo cut.

## Data Sources Locked

| Source | Role | Format | License |
|---|---|---|---|
| Umbra `ship_detection_testdata` | Primary high-res SAR (25 cm – 1 m) | GEC GeoTIFF + SICD/SIDD NITF + METADATA JSON | CC BY 4.0 |
| Sentinel-1 GRD | Baseline SAR (C-band, 10 m, ~12 day revisit) | GeoTIFF via Earth Search STAC | Copernicus open |
| Sentinel-2 L2A | Opportunistic optical (10 m) | COG via Earth Search STAC | Copernicus open |
| Global Fishing Watch | Primary AIS | API v3, parquet dumps | Research license (token in `.env`) |
| TLEs | Orbital pass propagation | Celestrak / Space-Track | Free |

## Demo Narrative Thread

Three Acts, mapped to the scene timing:

**Act 1 — Baseline (~25 s).** June 1 – July 1. Commercial shipping through the northern edge of the AOI. AIS and Sentinel-1 tracks flowing normally. Quiet at Cuarteron-area feature. Establishes "normal."

**Act 2 — Persistent target (~35 s).** July 2 – August 6. July 2 Umbra scene reveals SAR returns at the hero feature with no AIS explanation. Tracker spawns AIS-dark tracks. July 23 scene confirms the pattern: similar returns, length 57 m (militia-trawler band per AMTI methodology). Anomaly score climbs. Track enters WATCH state.

**Act 3 — The orchestration decision (~30 s).** August 7 – August 13. Belief-state covariance grows between observations. On WATCH entry, the tipcue layer scores candidate collects. Reasoning trace panel animates in showing top-3 candidates with info gain scores. Umbra-08 August 9 pass wins. Pass animates. Observation lands. Ellipse collapses. Provenance chain on selected track now shows 5 SAR chips across 41 days.

## Framing References

Citations to include in README and voiceover:

- **CSIS Asia Maritime Transparency Initiative**, *Dropping the Act: China's Militia in 2024* (Feb 2025). Documents that most Chinese maritime militia vessels do not transmit AIS, making the Spratlys the canonical real-world test case for AIS/SAR fusion.
- **SDA Custody Layer capability vectors**, STEC BAA (updated March 2026). Architectural reference, not threat-domain reference. See `docs/positioning.md` for full alignment table.
- **UrsaSpace commercial case study** (2023), vessel detection in SCS with Umbra SAR + AIS cross-correlation. Public validation that this use case is commercially operational.

## Storage Footprint

Post Day 0 download:

| Tier | Contents | Size |
|---|---|---|
| Tier 1 | 8 AOI Umbra scenes, all formats except CPHD | ~7 GB |
| Tier 2 | ~95 SCS-broad Umbra scenes, GEC + METADATA only | ~5.5 GB |
| Sentinel-1 | 11-week AOI coverage, GRD | ~20 GB (est, pre-download) |
| Sentinel-2 | 11-week AOI coverage, L2A cloud <40% | ~10 GB (est, pre-download) |
| GFW AIS | 11-week AOI, parquet | ~500 MB |
| **Total raw** | | **~45 GB baseline** |
| CPHD (optional, for stretch InSAR work) | up to all 8 AOI scenes | + up to 14 GB |

Actual post-download footprint will land somewhere in the **45–60 GB range** depending on final format choices. CLAUDE.md should reflect measured value once download completes.

## Pre-Week-1 Completion Criteria

Before Week 1 code starts, all of the following should be true:

- [ ] Umbra Tier 1 + Tier 2 download complete, measured size recorded in CLAUDE.md
- [ ] GFW API token works against a test query
- [ ] This document committed to repo
- [ ] `docs/positioning.md` committed to repo
- [ ] `docs/custody_fusion_implementation_guide_v3.md` committed to repo
- [ ] CLAUDE.md at repo root with v3-aligned content
- [ ] `.env.example` with `GFW_API_TOKEN` placeholder
- [ ] `.gitignore` covers `.env*`, `data/raw/`, `data/processed/`, `day0/*.csv`
