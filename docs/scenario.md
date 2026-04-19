# Custody — Scenario

*Rewritten per ADR-0013. Updates to this file are phase transitions and require updating CLAUDE.md and the implementation guide together.*

## Area of Interest

Spratly Islands hotspot in the South China Sea.

- **Bounding box:** 114.5°E – 117.5°E, 8.5°N – 11.0°N
- **Size:** approximately 330 km × 280 km
- **EPSG for internal use:** 4326 (WGS84) throughout; project only at compute/display boundaries

### Anchor features inside the AOI

| Feature | Lat | Lon | Claimant | Role in demo |
|---|---|---|---|---|
| **Tennent Reef** (Đá Tiên Nữ / Bahura ng Lopez-Jaena) | 8.856°N | 114.665°E | Vietnam | **Case Study A.** 5 Umbra scenes over 41 days imaging active land reclamation. |
| **Whitsun Reef** (Đá Ba Đầu / Julian Felipe Reef / 牛軛礁) | 9.98°N | 114.63°E | Contested (unoccupied; site of March 2021 Chinese militia swarm) | **Case Study B.** 3 Umbra scenes imaging vessel flotilla activity. |
| Cuarteron Reef (Chinese-controlled) | ~8.85°N | ~112.85°E | China | Outside AOI. Reference feature only — named in Vietnamese reporting, not in our imagery. |
| Mischief Reef | 9.90°N | 115.55°E | China | Reference feature. Fully militarized Chinese outpost. Not in our Umbra inventory. |
| Sabina Shoal | 9.77°N | 116.55°E | Contested | 2024 standoff site, outside primary demo window. Reference only. |

The demo presents two co-equal case studies. Both cases appear in the 90-second public video cut. Neither is designated primary or supplementary.

## Time Windows

**Case Study A: June 1 – August 20, 2023** (approximately 11 weeks). Driven by Umbra Open Data Program availability over Tennent Reef.

**Case Study B: December 2023 – March 2024** (approximately 15 weeks). Driven by Umbra availability over Whitsun Reef.

The two windows are disjoint. The architecture is demonstrated running over each window independently, producing separate track catalogs, separate belief states, and separate tipcue decisions. This is honest to the data and honest to the architectural claim — the pipeline is not a single monolithic run but a reusable system applied to two scenarios.

## Case Study A — Tennent Reef

### Narrative

Active Vietnamese land reclamation at the eastern artificial island (Tiên Nữ B) of Tennent Reef. Five Umbra SAR scenes across 41 days capture ongoing dredging and island-building that is AIS-dark in GFW presence data. The architecture detects the persistent structure as a multi-observation track with growing provenance chain and demonstrates tip-and-cue orchestration through a 48-hour cued revisit.

### Ground truth context

CSIS Asia Maritime Transparency Initiative has documented this specific feature's expansion in their [December 2022](https://amti.csis.org/vietnams-major-spratly-expansion/) and [November 2023](https://amti.csis.org/vietnam-ramps-up-spratly-island-dredging/) reports, identifying Tennent as one of Vietnam's four most significantly developed Spratly outposts. AMTI's November 2023 report documents 62 acres (25 hectares) of new artificial land added at Tennent between end-of-2022 and late-2023 — a period that encompasses our demo window.

### Umbra scene inventory

All scenes confirmed in `s3://umbra-open-data-catalog/sar-data/tasks/ship_detection_testdata/`.

| Date (UTC) | Mode | Grazing | Notes |
|---|---|---|---|
| 2023-07-02 14:00:55 | Spotlight | 45.4° | Scene 1. First appearance in inventory. Pier + reclamation footprint clearly visible. |
| 2023-07-23 | Spotlight | 40.9° | Scene 2. Change detection baseline. |
| 2023-08-07 | Spotlight | — | Scene 3. Sustained reclamation. |
| 2023-08-09 | Spotlight | — | Scene 4. **48 h after Scene 3 — the cued-pass demonstration.** |
| 2023-08-13 | Spotlight | — | Scene 5. Confirms sustained pattern. |

All scenes are Umbra GEC (orthorectified 8-bit GeoTIFF) + SICD (complex NITF) + SIDD (detected NITF) + METADATA JSON. Native resolution ~0.25 m; GEC ground resolution ~0.5 m.

### Act structure for this case

- **Act A1:** Baseline period (June 1 – July 1). Commercial shipping in AIS, no Umbra coverage yet. Establishes normal.
- **Act A2:** First two Umbra scenes (July 2 and July 23). Persistent bright structure at Tennent. Tracker spawns AIS-dark tracks. Anomaly scoring triggers WATCH state.
- **Act A3:** Cued revisit decision and execution (Aug 7 → Aug 9 → Aug 13). Orchestration reasoning trace animates. Scene selection shown with info-gain scores. Cued pass executes. Provenance chain on the track now includes 5 SAR chips across 41 days.

## Case Study B — Whitsun Reef

### Narrative

Vessel flotilla activity at Whitsun Reef, site of the March 2021 Chinese maritime militia swarm event (~220 vessels). Three Umbra SAR scenes across four months capture multiple AIS-dark vessel clusters visible at the reef. The architecture detects vessels as point observations, associates them into tracks through Hungarian assignment, and demonstrates cross-INT fusion where SAR detections persistently exist without corresponding GFW AIS broadcasts.

### Ground truth context

Whitsun Reef is the site of a [widely reported international incident](https://amti.csis.org/caught-on-camera-two-dozen-militia-boats-at-whitsun-reef-identified/) in March 2021 when approximately 220 Chinese maritime militia vessels gathered at the reef, prompting protest from the Philippines and Vietnam. AMTI, Reuters, BBC, and international press documented the event. Our three 2023-2024 scenes show the reef's state at three later time points, enabling the architecture to demonstrate militia-style flotilla detection at a persistently contested feature.

### Umbra scene inventory

| Date (UTC) | Notes |
|---|---|
| 2023-12-06 (earlier) | Scene 1. Multiple vessel clusters visible. |
| 2023-12-06 (later) | Scene 2. Same date, later acquisition. |
| 2024-03-20 | Scene 3. Months-later revisit. |

### Act structure for this case

- **Act B1:** Scene 1 arrives. CFAR detects 20+ point targets clustered at the reef. None are present in concurrent GFW AIS data. Tracker spawns TENTATIVE tracks; multi-target association via Hungarian resolves cluster structure.
- **Act B2:** Scene 2 (same day, hours later). Confirms cluster persistence at higher confidence. Tracks transition CONFIRMED.
- **Act B3:** Scene 3 (months later). Different cluster configuration. Tracks from scenes 1-2 have aged out; new tracks spawn. Demonstrates the architecture's lifecycle handling and the feature's continuing status as a militia activity hub.

## Data Sources Locked

| Source | Role | Format | License |
|---|---|---|---|
| Umbra ship_detection_testdata | High-resolution SAR (25 cm – 1 m) | GEC GeoTIFF + SICD/SIDD NITF + METADATA JSON | CC BY 4.0 |
| Sentinel-1 GRD | Baseline SAR (C-band, 10 m, ~12 day revisit) | GeoTIFF via Earth Search STAC | Copernicus open |
| Sentinel-2 L2A | Opportunistic optical (10 m) | COG via Earth Search STAC | Copernicus open |
| Global Fishing Watch | Presence (AIS-derived, per-cell-per-hour aggregate) | API v3, Parquet dumps | Research license (token in .env) |
| TLEs | Orbital pass propagation | Celestrak / Space-Track | Free |

## Framing References

Citations to include in README and voiceover:

### Case Study A (Tennent Reef)

- **CSIS AMTI — Vietnam Tracker.** [https://amti.csis.org/island-tracker/vietnam/](https://amti.csis.org/island-tracker/vietnam/). Canonical reference for Vietnamese Spratly outpost status.
- **AMTI, "Vietnam's Major Spratly Expansion" (Dec 2022).** Identifies Tennent as significantly developed outpost; documents reclamation trajectory.
- **AMTI, "Vietnam Ramps Up Spratly Island Dredging" (Nov 2023).** Documents the specific 62-acre expansion in our demo window.

### Case Study B (Whitsun Reef)

- **AMTI — Whitsun Reef / Julian Felipe Reef coverage.** [https://amti.csis.org/caught-on-camera-two-dozen-militia-boats-at-whitsun-reef-identified/](https://amti.csis.org/caught-on-camera-two-dozen-militia-boats-at-whitsun-reef-identified/) documenting the March 2021 swarm.
- **Broader AMTI South China Sea militia coverage** — for context on the operational pattern.

### Shared

- **Umbra Open Data Program**, AWS Open Data Registry.
- **Global Fishing Watch**, research API documentation.
- **Copernicus / ESA**, Sentinel-1 and Sentinel-2 documentation via Earth Search STAC.

## Storage Footprint (as of Week 2 Day 3)

| Tier | Contents | Size |
|---|---|---|
| Tier 1 | 8 AOI Umbra scenes (both case studies), all formats except CPHD | 7 GB |
| Tier 2 | ~95 SCS-broad Umbra scenes, GEC + METADATA only | 5.5 GB |
| Total Umbra | 219 files | 56 GB (measured) |
| GFW presence (raw JSON) | 12 weekly chunks, 11-week Case A window | 19 MB |
| GFW presence (processed Parquet) | 39,337 observations | 1.2 MB |
| SAR detections (processed Parquet) | produced per-scene-per-run | varies |
| Sentinel-1 | Not yet fetched (Week 2) | — |
| Sentinel-2 | Not yet fetched (Week 2) | — |

## Scenario Confirmation Log

Week 2 Day 2 reconnaissance confirmed the Tennent Reef feature identification and established the initial case-study framing (ADR-0012). Week 2 Day 3 reconnaissance against Whitsun imagery confirmed vessel-flotilla activity at that feature, leading to the co-equal dual-case-study framing of ADR-0013. The architectural pipeline (fusion package, EKF, observation types, spatial index, tracker) is scenario-agnostic and operates identically across both case studies.
