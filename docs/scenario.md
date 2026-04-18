# Custody — Scenario

*Rewritten per ADR-0012. Updates to this file are phase transitions and require updating CLAUDE.md and the implementation guide together.*

## Area of Interest

Spratly Islands hotspot in the South China Sea.

- **Bounding box:** 114.5°E – 117.5°E, 8.5°N – 11.0°N
- **Size:** approximately 330 km × 280 km
- **EPSG for internal use:** 4326 (WGS84) throughout; project only at compute/display boundaries

### Anchor features inside the AOI

| Feature | Lat | Lon | Claimant | Role in demo |
|---|---|---|---|---|
| **Tennent Reef** (Đá Tiên Nữ / Bahura ng Lopez-Jaena) | 8.856°N | 114.665°E | Vietnam | **Hero target.** 5 Umbra scenes, 41-day repeat window over active Vietnamese land reclamation. |
| **Whitsun Reef** (Đá Ba Đầu / Julian Felipe Reef / 牛軛礁) | 9.98°N | 114.63°E | Contested (unoccupied; site of March 2021 Chinese militia swarm) | Supplementary case study. 3 Umbra scenes Dec 2023 – Mar 2024. |
| Cuarteron Reef (Chinese-controlled) | ~8.85°N | ~112.85°E | China | Outside AOI. Reference feature only — named in Vietnamese reporting, not in our imagery. |
| Mischief Reef | 9.90°N | 115.55°E | China | Reference feature. Fully militarized Chinese outpost. Not in our Umbra inventory. |
| Sabina Shoal | 9.77°N | 116.55°E | Contested | 2024 standoff site, outside primary demo window. Reference only. |

Our primary feature is **Tennent Reef**, specifically the eastern artificial island Tiên Nữ B, at the coordinates above. Vietnam has been actively reclaiming land at this feature since December 2021. CSIS AMTI has documented the expansion extensively in their [December 2022](https://amti.csis.org/vietnams-major-spratly-expansion/) and [November 2023](https://amti.csis.org/vietnam-ramps-up-spratly-island-dredging/) reports, identifying Tennent as one of Vietnam's four most significantly developed Spratly outposts. AMTI's November 2023 report specifically documents Tennent growing by 62 acres (25 hectares) of new artificial land in the period that encompasses our demo window.

## Time Window

**Primary demo window: June 1 – August 20, 2023** (approximately 11 weeks).

Driven by Umbra scene availability over Tennent Reef, not by choice. The 2024 Sabina Shoal standoff and other candidate narratives were considered and rejected in Day 0 scoping because no Umbra coverage of the AOI existed during those periods.

### Rationale for the 11-week window

- **June:** pattern-of-life baseline period. AIS + Sentinel-1 continuous, no Umbra scenes yet, commercial shipping flows through northern edge of AOI, reclamation activity ongoing but not directly observed in our high-resolution SAR.
- **July 2:** first Umbra scene. Persistent reclamation footprint at Tiên Nữ B visible — bright coherent structure, ~1 km long pier-like feature extending north.
- **July 23:** second Umbra scene. Same feature, same structures. Slight additional expansion visible. Temporal baseline for change detection established.
- **August 7:** third Umbra scene. Sustained reclamation activity. Anomaly scoring picks up the persistent AIS-dark bright return.
- **August 9:** fourth Umbra scene. **48 hours after scene 3 — the cued revisit demonstration.** The orchestration layer reasons: WATCH tier target with growing belief-state uncertainty crosses threshold; info-gain scoring picks Umbra over Sentinel-1 baseline due to resolution advantage.
- **August 13:** fifth Umbra scene. Confirms sustained reclamation. Track's provenance chain now includes 5 SAR chips across 41 days.
- **August 20:** buffer for post-cluster observations and demo wrap.

## Umbra Scene Inventory

All scenes confirmed present in `s3://umbra-open-data-catalog/sar-data/tasks/ship_detection_testdata/` via Day 0 centroid scan. See `day0/ship_detection_centroids.csv` for the full catalog.

### Primary cluster — Tennent Reef / Tiên Nữ B (8.856°N / 114.665°E)

| Date (UTC) | Mode | Grazing | Notes |
|---|---|---|---|
| 2023-07-02 14:00:55 | Spotlight | 45.4° | Scene 1. First appearance in our inventory. Pier + reclamation footprint clearly visible. |
| 2023-07-23 | Spotlight | 40.9° | Scene 2. Same feature. Change detection baseline. |
| 2023-08-07 | Spotlight | — | Scene 3. Sustained reclamation. |
| 2023-08-09 | Spotlight | — | Scene 4. **48 h after Scene 3 — the cued-pass demonstration.** |
| 2023-08-13 | Spotlight | — | Scene 5. Confirms sustained pattern. |

All scenes are Umbra GEC (orthorectified 8-bit GeoTIFF) + SICD (complex NITF) + SIDD (detected NITF) + METADATA JSON. Approximate native resolution 0.25 m; GEC ground resolution ~0.5 m.

### Supplementary cluster — Whitsun Reef (9.98°N / 114.63°E)

| Date (UTC) | Notes |
|---|---|
| 2023-12-06 | Scene 1. |
| 2023-12-06 | Scene 2. Same date, later acquisition. |
| 2024-03-20 | Scene 3. Months-later revisit. |

Whitsun is a triangular reef approximately 10 km², submerged at high tide, contested between China, Philippines, and Vietnam. The reef is the site of the **March 2021 Chinese maritime militia swarm event** — approximately 220 Chinese militia vessels gathered at the reef, prompting international protest from the Philippines and Vietnam. AMTI, Reuters, and SpaceNews extensively documented the event and its aftermath.

These three scenes enable a supplementary case study demonstrating the same Custody architecture applied to a different narrative — Chinese militia activity at an unoccupied contested reef. Documented in the repo as a second case study. **Not included in the 90-second demo cut.**

## Data Sources Locked

| Source | Role | Format | License |
|---|---|---|---|
| Umbra ship_detection_testdata | Primary high-res SAR (25 cm – 1 m) | GEC GeoTIFF + SICD/SIDD NITF + METADATA JSON | CC BY 4.0 |
| Sentinel-1 GRD | Baseline SAR (C-band, 10 m, ~12 day revisit) | GeoTIFF via Earth Search STAC | Copernicus open |
| Sentinel-2 L2A | Opportunistic optical (10 m) | COG via Earth Search STAC | Copernicus open |
| Global Fishing Watch | Presence (AIS-derived, per-cell-per-hour aggregate) | API v3, Parquet dumps | Research license (token in .env) |
| TLEs | Orbital pass propagation | Celestrak / Space-Track | Free |

## Demo Narrative Thread

Three Acts, mapped to the scene timing:

**Act 1 — Baseline (~25 s).** June 1 – July 1. Commercial shipping transit through the northern edge of the AOI. GFW presence data shows dense vessel activity at routine shipping lanes. The AOI around Tennent Reef is quiet in AIS — the nearest consistent GFW returns are ~5 km south, at natural fishing grounds. No Umbra coverage yet. Establishes "normal."

**Act 2 — Persistent AIS-dark target (~35 s).** July 2 – August 6. July 2 Umbra scene reveals a bright coherent structure at Tennent Reef that GFW presence data doesn't explain. Tracker spawns AIS-dark tracks at the feature. July 23 scene confirms the pattern: same returns, same location, same approximate footprint. The detection is not a ship — it is land being actively reclaimed. Anomaly scoring picks up the persistent AIS-dark SAR-only return. Track enters WATCH state.

Context cue for voiceover: *"AMTI has documented Vietnamese dredging and land reclamation at Tennent Reef since 2021. The pipeline detects activity at this feature without knowing in advance what's being built there — the architecture distinguishes 'something persistent is here and not broadcasting AIS' without requiring prior knowledge of the feature's identity."*

**Act 3 — The orchestration decision (~30 s).** August 7 – August 13. Belief-state covariance grows between observations. On WATCH entry, the orchestration layer scores candidate collects. Reasoning trace panel animates in showing top-3 candidates with info-gain scores. Umbra-08 August 9 pass wins (resolution advantage for localization at a reclamation site). Pass animates. Observation lands. Ellipse collapses. Provenance chain on the selected track now shows 5 SAR chips across 41 days.

## Supplementary Case Study — Whitsun Reef

Presented in the repository as a second case demonstrating the same architecture applied to a different scenario type: contested-reef Chinese militia activity. Three scenes over 3 months enable a longer-baseline change-detection story rather than a tight-cadence cueing story. Referenced in the README and covered in a dedicated docs section; not in the 90-second demo cut.

Context for this case study: the March 2021 Chinese militia swarm event at Whitsun was a widely reported international incident. The ~220 vessel concentration at a reef inside the Philippine EEZ was documented by AMTI, Philippine government, Vietnamese government, and international press. Our three 2023-2024 scenes show the reef's state at three later time points, enabling the architecture to demonstrate pattern-of-life baselines and anomaly detection at a persistently-contested feature.

## Framing References

Citations to include in README and voiceover:

- **CSIS Asia Maritime Transparency Initiative — Vietnam Tracker.** [https://amti.csis.org/island-tracker/vietnam/](https://amti.csis.org/island-tracker/vietnam/). The canonical reference for Vietnamese Spratly outpost status, including Tennent Reef.
- **AMTI, "Vietnam's Major Spratly Expansion" (Dec 2022).** Identifies Tennent as a significantly developed outpost; documents reclamation trajectory.
- **AMTI, "Vietnam Ramps Up Spratly Island Dredging" (Nov 2023).** Documents the specific 62-acre expansion at Tennent in the period encompassing our demo window.
- **AMTI, "No Islet Left Behind" (Aug 2025).** Documents ongoing Tennent reclamation and the broader Vietnamese program.
- **AMTI Whitsun Reef coverage.** Supplementary-case reference for the March 2021 militia swarm.
- **Umbra Open Data Program**, AWS Open Data Registry.
- **Global Fishing Watch**, research API documentation.
- **Copernicus / ESA**, Sentinel-1 and Sentinel-2 documentation via Earth Search STAC.

## Storage Footprint (as of Week 2 Day 2)

Post Day 0 download and Stream A processing:

| Tier | Contents | Size |
|---|---|---|
| Tier 1 | 8 AOI Umbra scenes, all formats except CPHD | 7 GB |
| Tier 2 | ~95 SCS-broad Umbra scenes, GEC + METADATA only | 5.5 GB |
| Total Umbra | 219 files | 56 GB (measured) |
| GFW presence (raw JSON) | 12 weekly chunks, full 11-week window | 19 MB |
| GFW presence (processed Parquet) | 39,337 observations | 1.2 MB |
| Sentinel-1 | Not yet fetched (Week 2) | — |
| Sentinel-2 | Not yet fetched (Week 2) | — |

## Scenario Confirmation Log

Week 2 Day 2 reconnaissance against actual Umbra imagery confirmed the primary feature as Tennent Reef (Vietnamese-controlled, active reclamation since 2021), and established Whitsun Reef as the supplementary case study. See ADR-0012 for the full scoping record. The architectural pipeline (fusion package, EKF, observation types, spatial index, tracker) is scenario-agnostic and operates identically under either narrative framing.
