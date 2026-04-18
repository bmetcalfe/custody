# Custody Multi-Sensor Fusion — Implementation Guide (v3)

**Status:** Scenario locked. Data sources resolved. Positioning reframed around SDA capability vectors. Ready for Week 1.
**Supersedes:** v2.
**Scope:** Open-source reference implementation answering a publicly published community call. Architecture is real; scenario and data are curated.
**Builds on:** Existing `custody/` repo — `models.py`, `tracks.py`, and the Custody → Anomaly → Watch → Predict → Task → Collect loop.

---

## 0. North Star

One positioning sentence:

> *"An open-source reference implementation aligned with the Space Development Agency's Custody Layer capability vectors, demonstrating multi-phenomenology fusion and legible tip-and-cue orchestration for maritime domain awareness in contested waters."*

One hero capability:

> **Covariance-aware, explainable tip-and-cue orchestration.** Every cueing decision emits both an action and a reasoning trace — selected collect, expected information gain, feasibility priors, rejected alternatives with their scores, plain-language justification.

Two design rules:

1. Architecture is real, data is curated. We never fake the pipeline; we hand-pick the scenario so the outcomes are narratively tight.
2. Every claim in the README, voiceover, and UI must be something the demo can back up. Honest scoping beats marketing polish.

---

## 1. Strategic Framing

### 1.1 The community call

In 2024 the Space Development Agency published a Broad Agency Announcement (STEC) soliciting industry proposals for its Proliferated Warfighter Space Architecture. One capability layer — the Custody Layer — sets out a public list of architectural capability vectors that are agnostic to threat domain:

- Automated processing and fusion of data from traditional space-based sensing payloads across visible, infrared, RF, SAR, multispectral
- Design of a multi-phenomenology fusion architecture supporting agile incorporation of new algorithms
- Reduction in latency of processing, exploitation, and dissemination
- Memory management and target hypothesis distribution across satellite nodes

These patterns also describe the architectural space commercial MDA systems (Vantor Sentry, BlackSky, Satellogic, Ursa, HawkEye 360) are operating in. What's hard to find *publicly* is a reference implementation showing these patterns working end to end on open data. Commercial systems are closed; academic work studies layers in isolation; classified work is classified.

Custody fills that gap for the maritime domain. It answers the community call, not a single company's job requisition.

### 1.2 Honest domain scope

SDA's Custody Layer focuses primarily on ballistic and hypersonic missile threats. Those kinematics do not translate to vessels:

| Property | SDA hypersonic target | Maritime vessel |
|---|---|---|
| Velocity | Mach 5+ | 0–25 knots |
| Altitude | 20–80 km | Sea surface |
| Maneuvering | Aggressive | Smooth, constrained by routing |
| Timescale of custody | Seconds to minutes | Hours to days |
| Relevant covariance growth rate | km/s | km/hr |

Custody applies the *architectural patterns* from SDA's capability vectors — multi-phenomenology fusion, hypothesis management, low-latency exploitation — to a maritime problem where open data enables public validation. We do not claim direct applicability to missile defense. This distinction is called out explicitly in the README, voiceover, and positioning document.

### 1.3 Primary and incidental audiences

- **Primary:** The GeoInt engineering community responding to the SDA BAA — Vantor/Sentry, BlackSky, Satellogic, Ursa, HawkEye 360, Planet, integrating primes (Lockheed, Northrop, Leidos, L3Harris, etc.).
- **Secondary:** SDA's Custody & Emerging Capabilities Cell as an open-source reference aligned with the capability call.
- **Incidental:** Hiring managers at the above commercial entities, as a side effect of doing the work visibly.

Vantor is *the* most-probable commercial reader given the April 15, 2026 Sentry + Windward announcement and the Westminster, CO overlap, but the repo is not a Vantor pitch. It's a community-aligned artifact that Vantor will benefit from reading first.

### 1.4 The hero capability (expanded)

Most public descriptions of commercial tip-and-cue systems (Sentry, BlackSky analytics) present cueing as automation — "the system tasks the next sensor without analyst intervention." They do not show the decision logic.

Custody's differentiator is legibility. For every cue the tipcue layer emits, it produces:

```
CUED: Umbra-08 pass at 2023-08-09T09:14:22Z
  Target: Track T-0042 (belief state: 9.745N, 115.881E ± 4.2 km, 3σ)
  Modality: SAR
  Expected info gain: 0.71 (log-det reduction 8.2 km → 1.1 km post)
  Feasibility: grazing angle 48° (good), no cloud dependency
  
  Rejected alternatives:
    - Sentinel-2 at 11:22 UTC: info gain 0.23 (cloud forecast 82%)
    - Sentinel-1 at 15:48 UTC: info gain 0.58 (info gain lower; 6 h covariance growth)
    - Defer to next day: growth rate 0.9 km/hr exceeds 4 km threshold
  
  Justification: Covariance growth on T-0042 crosses the 4 km threshold by
  10:00 UTC. Umbra-08 at 09:14 provides best uncertainty reduction under
  clear-weather feasibility. Sentinel-2 alternative rejected on cloud forecast;
  Sentinel-1 alternative rejected on later arrival.
```

This is the demo's money shot. Every other capability — detection, fusion, anomaly scoring, provenance chain — is plumbing that makes this output meaningful.

### 1.5 Extension, not rewrite

The existing Custody modules stay. This guide adds `fusion/`, `catalog/`, `detection/`, a `tipcue/` subpackage (the per-track cueing decision layer — originally named `orchestration/` in early drafts, renamed to avoid a collision with the existing Phase 2 `orchestration/` portfolio module and to use sharper vocabulary), a preprocessing script pipeline, and a new `app/` frontend.

The existing `src/custody/orchestration/` from Phase 2 (738 LOC, attention tiers + portfolio ranking) is retained as-is. It composes with the new `tipcue/` module — see §4.2.

---

## 2. Design Decisions

Most decisions locked; remaining opens flagged `[OPEN]`.

### 2.1 Scenario — LOCKED

- **AOI:** Spratly Islands hotspot. Bbox (114.5°E, 8.5°N, 117.5°E, 11.0°N) — roughly 330×280 km.
- **Hero feature:** Cuarteron-area target at 114.665°E / 8.856°N. 5 Umbra scenes: 2023-07-02, 2023-07-23, 2023-08-07, 2023-08-09, 2023-08-13.
- **Secondary feature:** Union Banks area at 114.63°E / 9.98°N. 3 Umbra scenes (2023-12-06 × 2, 2024-03-20), used for supplementary long-duration PoL arc outside primary demo.
- **Time window:** June 1 – August 20, 2023 (~11 weeks). Driven by Umbra scene availability.

### 2.2 Demo Acts — LOCKED

- **Act 1 — Baseline (~25 s).** June AIS + Sentinel-1 traffic across the Spratlys. Commercial shipping flowing through the northern edge. Quiet at the reef cluster. Establishes "normal."
- **Act 2 — Persistent target (~35 s).** July 2 Umbra scene lands. SAR returns at Cuarteron-area feature that AIS doesn't explain. Tracker spawns AIS-dark tracks. July 23 scene arrives — same feature, similar returns, length in militia-trawler band. Anomaly score climbs. Track enters WATCH state.
- **Act 3 — The orchestration decision (~30 s).** This is the hero moment. Belief-state visualization: covariance ellipse grows between observations. On entering WATCH, the orchestration layer evaluates candidate collects. Reasoning trace panel animates in, scoring three candidates. Umbra-08 Aug 9 pass wins, cued. Pass animates on map. Observation lands. Ellipse collapses. Provenance chain on the track now shows 5 SAR chips across 41 days.

### 2.3 Observation model — LOCKED

```python
@dataclass(frozen=True)
class Observation:
    obs_id: str                    # ULID for ordering + uniqueness
    source_id: str                 # "umbra", "sentinel1", "sentinel2", "gfw_ais"
    modality: Literal["SAR","EO","AIS"]
    acquisition_time: float        # UTC epoch seconds
    ingestion_time: float
    lat: float
    lon: float
    cov: tuple[float, float, float]  # (σ_xx, σ_yy, σ_xy), meters²
    vessel_length_est_m: float | None
    heading_est_deg: float | None
    sog_est_kt: float | None
    mmsi: int | None
    vessel_name: str | None
    classification_conf: float | None
    raw_ref: str                   # S3 URI or local path + pixel/row index
    detector_version: str | None
    notes: dict[str, Any]
```

### 2.4 Coordinate and time discipline — LOCKED

- Internal: WGS84 + UTC epoch seconds. No exceptions.
- Display: Web Mercator for map, UTC+8 for display clock (Philippines time).
- SAR chip extraction: rasterio `Affine` per scene; never eyeball pixel→geo.
- Scene-center timestamp with documented ±5 s absorbed into covariance.

### 2.5 Spatial index — LOCKED

H3 resolution 8 (~0.46 km edge). Time buckets: 1-minute for AIS, 1-hour for satellite observations. DuckDB over Parquet. Revisit to r9 if reef clusters feel clumped.

### 2.6 Association + tracking — LOCKED

- **Association:** Hungarian (scipy `linear_sum_assignment`) over Mahalanobis-gated cost matrix. Modality-mismatch priors baked in.
- **Tracking:** EKF per track. State `[lat, lon, v_n, v_e]`. Constant-velocity motion model with Gaussian process noise tuned per-modality arrival rate.
- **Track lifecycle:** 3-of-5 N-of-M confirmation, aging-out after T seconds of no updates (T tuned per demo).

### 2.7 Detection — LOCKED

- **AIS:** passthrough from GFW parquet. No detection.
- **SAR (Umbra GEC, Sentinel-1 GRD):** custom CA-CFAR in ~100 lines. OSM coastline water mask (minimal land in this AOI — mostly reefs). Morphological cleanup. Connected components → centroids. Length estimated from oriented bounding box major axis.
- **EO (Sentinel-2):** pretrained detector (YOLOv8n fine-tuned on HRSID or SDFSD-v1.0 Umbra subset). Opportunistic only — SCS ~60% cloudy.

### 2.8 Fusion level — LOCKED

Observation-level. Each track owns an EKF. Observations arrive timestamped and update the filter state.

### 2.9 Anomaly catalog — LOCKED

Four multi-INT anomalies:

- **AIS/SAR disagreement.** AIS position + SAR detection at different positions within the same time window, outside 3σ gate.
- **SAR-only dark vessel.** SAR detection where AIS density suggests coverage but no AIS within gate. *Hero anomaly for SCS.*
- **Loitering in militia length band.** Vessel 45–65 m estimated length, stationary > N hours at reef cell, no AIS. Mirrors AMTI methodology.
- **Cross-source class mismatch.** AIS-reported vessel type inconsistent with SAR-estimated length class.

### 2.10 Tipcue layer — LOCKED (the hero, expanded)

This is where v3 invests most heavily. The module is `src/custody/tipcue/`. It is the per-track decision layer of the broader orchestration story; the portfolio side (`src/custody/orchestration/`, retained from Phase 2) feeds it. See §4.2 for how the two layers compose.

**Inputs per scoring call:**
- Track belief state: mean `(lat, lon, v_n, v_e)` + 4×4 covariance
- Process noise model: per-modality expected covariance growth rate
- Candidate collects: tuples of (sensor_id, start_time, swath_polygon, modality, expected_resolution)
- Feasibility priors per candidate:
  - Cloud probability (EO only) from ERA5 or recent cloud climatology
  - Grazing angle (SAR only) from orbital geometry
  - Pass geometry over target (does the swath actually cover the uncertainty ellipse?)

**Math:**

Expected information gain of candidate `c` against track covariance `Σ`:

```
I(c) = ½ log |Σ_prior| − ½ log |Σ_posterior(c)|
```

Posterior is computed under the sensor's modeled measurement covariance `R_c`, gated by feasibility priors. Feasibility enters multiplicatively on the expected info gain:

```
score(c) = I(c) × P_feasible(c)
```

Best candidate wins. Reasoning trace captures top-3 by score and the feasibility components of each.

**Output dataclass:**

```python
@dataclass(frozen=True)
class CueingDecision:
    decision_id: str
    track_id: str
    cued_at: float                      # UTC epoch
    selected: CandidateCollect          # winner
    expected_info_gain: float
    posterior_cov_logdet: float
    alternatives_considered: list[tuple[CandidateCollect, float, str]]
    # alternative: (candidate, score, reason_rejected)
    justification_text: str             # natural-language rendering
```

**Natural-language renderer:** templated, not LLM (pre-generated LLM briefs only for the top-level intel brief panel, per §5.2 item 7). Templating is deterministic, auditable, and keeps the demo offline-runnable.

### 2.11 Cueing loop — LOCKED (simulated demo-time, pre-planned)

Honest-framing matters here. Voiceover:

> *"A live system would evaluate candidates in real-time. For the demo, the full pass schedule and candidate scoring are pre-computed over the window. What you see is the same logic a live system would execute, replayed smoothly."*

TLE propagation via `skyfield` over the demo window. For each track, candidate collects are every pass from every constellation whose swath intersects the track's predicted position (plus uncertainty cone). Tipcue decisions are pre-computed; the UI replays them.

### 2.12 Provenance chain — LOCKED (non-negotiable feature)

Clicking a track expands to:

```
Track T-0042   (watch, confidence 0.82)
├── fused from 11 observations
├── Jun 14 02:11 UTC — Sentinel-1 SAR detection (conf 0.72)
│   raw → s3://.../S1A_IW_GRDH_..._20230614T...tif
├── Jul 02 14:22 UTC — Umbra SAR detection (conf 0.88, no AIS correlation)
│   raw → s3://.../umbra/.../GEC.tif
│   estimated length 57 m (militia trawler band) ← ANOMALY
├── Jul 23 03:45 UTC — Umbra SAR detection, loitering (ANOMALY)
├── Aug 07 09:14 UTC — Umbra SAR detection (CUED from anomaly)
│   ↳ orchestration decision: exp. info gain 0.71, 2 rejected alts
├── Aug 09 09:14 UTC — Umbra SAR detection (48h tight cue)
└── Aug 13 06:30 UTC — Umbra SAR detection (confirmed loiter pattern)
```

Every line clickable. Every `raw →` is a real S3 URI that works.

### 2.13 Visualization stack — LOCKED

React + Mapbox + deck.gl + FastAPI backend reading Parquet via DuckDB. Dark intel-console aesthetic. Cesium 3D view is in the MAYBE tier — attempted after belief-state viz is solid.

### 2.14 Aesthetic — LOCKED before coding

`app/web/src/tokens.css` written first. Semantics:
- Cyan: confirmed multi-source track
- Amber: watch, single-source, uncertain
- Red: active anomaly
- Dim grey: stale/aged track
- White (monospace): reasoning trace text

### 2.15 Storage — LOCKED

`data/raw/` → `data/processed/` (Parquet + COG) → `data/demo/` (curated, ~3 GB, versioned via git-lfs or S3).

### 2.16 Secondary arc — LOCKED

The 3 Union Banks scenes (Dec 2023 × 2, Mar 2024) are processed and documented but not in the 90-second demo. They live in the repo as a second case study showing "same architecture, longer PoL baseline." Called out in the README as a supplementary result.

### 2.17 [OPEN] Cesium 3D

Attempted in Week 7 if belief-state viz is solid by end of Week 6. Cut without apology if not.

### 2.18 [OPEN] Inter-node hypothesis passing (stretch)

SDA's BAA explicitly calls out "memory management and target hypothesis distribution from one satellite node to the next." If the belief-state planner is solid and time remains, a simulated inter-node handoff demo is the highest-value architectural showpiece we could add — sends track state across two simulated satellites with covariance preserved. Not promised; flagged as aspirational.

---

## 3. Data Sources — Resolved

| Source | Access | Role | Status |
|---|---|---|---|
| Umbra `ship_detection_testdata` | `s3://umbra-open-data-catalog/...` | Primary high-res SAR | 8 scenes confirmed over AOI |
| Umbra ODP | same bucket, other task folders | Supplementary | Port of Singapore scene for comparison |
| Sentinel-1 GRD | Earth Search STAC | Baseline SAR (10 m, ~12 day revisit) | Global |
| Sentinel-2 L2A | Earth Search STAC | Opportunistic optical (10 m) | Cloud-limited |
| Global Fishing Watch | Research API | Primary AIS | Token in hand |
| TLEs | Celestrak / Space-Track | Orchestration pass propagation | Free |

Download pipeline in §3.3 of v2 (unchanged). Tier 1 (hero 8 scenes, all formats except CPHD) + Tier 2 (rest of SCS-broad, GEC + metadata only) ≈ 13 GB.

---

## 4. Core Architecture

```
┌────────────────────────────────────────────────────────────┐
│  SOURCES                                                    │
│  Umbra · Sentinel-1 · Sentinel-2 · GFW AIS · TLEs          │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│  CATALOG  (catalog/)                                        │
│  Federated STAC client — one query surface, normalized      │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│  DETECTION  (detection/)                                    │
│  CFAR for SAR · pretrained CNN for EO · passthrough for AIS │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│  OBSERVATION MODEL  (fusion/observations.py)                │
│  Typed, provenance-bearing, covariance-carrying             │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│  SPATIOTEMPORAL INDEX  (fusion/index.py)                    │
│  H3 r8 cells × 1-minute buckets, DuckDB-backed              │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│  ASSOCIATION + TRACKING  (fusion/tracker.py)                │
│  Hungarian · EKF per track · Mahalanobis gating             │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│  ANOMALY + POL  (anomalies.py, pol.py)                      │
│  Four multi-INT anomalies · reef-cell PoL baselines         │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│  PORTFOLIO  (orchestration/)       [Phase 2, retained]      │
│  Fleet-wide attention allocation · rank_portfolio() ·       │
│  BACKGROUND / WATCHLIST / ACTIVE_CUSTODY tiers ·            │
│  neglect pressure · custody health                          │
└────────────────────────┬────────────────────────────────────┘
                         │  PortfolioItem.entity_id
                         │  (ACTIVE_CUSTODY tracks)
┌────────────────────────▼────────────────────────────────────┐
│  TIPCUE  (tipcue/)                                 ★ HERO   │
│  Per-track belief-state scoring · info-gain optimization ·  │
│  feasibility priors · natural-language reasoning trace      │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│  DEMO SERVER  (app/api/)                                    │
│  FastAPI · DuckDB over Parquet                              │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│  FRONTEND  (app/web/)                                       │
│  React · Mapbox · deck.gl · (maybe Cesium 3D)               │
└─────────────────────────────────────────────────────────────┘
```

### 4.1 Tipcue module structure

```
src/custody/tipcue/
├── __init__.py
├── passes.py          # skyfield TLE propagation → candidate passes
├── feasibility.py     # cloud priors, grazing angle, pass geometry
├── info_gain.py       # the expected log-det reduction math
├── policy.py          # top-level cue decision logic
├── justification.py   # templated natural-language renderer
└── types.py           # CandidateCollect, CueingDecision dataclasses
```

Every file < 150 lines. Every function < 30 lines where practical. Unit tests on the math first (`info_gain.py`), then the policy.

### 4.2 How portfolio and tipcue compose

Custody has two orchestration-adjacent modules that compose cleanly rather than compete. The `orchestration/` module operates at portfolio scope — given N active tracks, it produces the attention ranking: who is in `ACTIVE_CUSTODY`, who is `WATCHLIST`, who can remain `BACKGROUND`. The `tipcue/` module operates at decision scope — given a single track whose attention tier justifies a collect, it selects which sensor pass to cue and why. **Portfolio picks the *who*; tipcue picks the *what*.** The demo's hero moment happens at the tipcue layer, but it's the portfolio layer that decides the track is worth a cue in the first place.

The two layers join cleanly via `PortfolioItem.entity_id`:

```python
from custody.orchestration import rank_portfolio          # Phase 2
from custody.tipcue import decide_collect                 # v3

assessment = rank_portfolio(timestamp, records, scenario_start)
candidates = [item for item in assessment.items
              if item.attention_state == "ACTIVE_CUSTODY"]

for item in candidates:
    track = tracker.get_track(item.entity_id)             # ~20-LOC adapter
    decision = decide_collect(track, candidate_passes)
    # decision is a CueingDecision with reasoning trace
```

`PortfolioItem` already carries everything tipcue needs to filter the input set: `entity_id`, `attention_state`, `portfolio_rank`, `portfolio_score`, `custody_health`, and the rationale strings. The one piece deliberately *not* in `PortfolioItem` is the belief state (mean + 4×4 covariance) — that lives in the EKF Track maintained by `fusion/tracker.py`. The composition therefore needs a `tracker.get_track(entity_id) -> Track` accessor, written as the first task of Week 5.

This composition lowers Week 5 risk: portfolio already exists at 738 LOC and works. Week 5 builds tipcue from scratch against an input the portfolio already produces, instead of reinventing the attention layer.

**Scope — v1 tipcue consumes `ACTIVE_CUSTODY` tracks only.** `WATCHLIST` scoring is a documented extension point: the composition code handles it trivially (the filter predicate is a single line), but scoring passes against tracks we haven't committed to watching adds noise to the demo and undermines the legibility claim. Deferred to a follow-up ADR if post-demo feedback asks for it. See [ADR-0002](decisions/0002-tipcue-v1-active-custody-only.md).

---

## 5. Presentation Layer

### 5.1 Screen composition

```
┌────────────────────────────┬───────────────────────────┐
│  MAP PANE  (primary)       │  TRACK DETAIL             │
│  - fused tracks            │  - provenance chain       │
│  - uncertainty ellipses    │  - anomaly timeline       │
│  - sensor footprints       │  - SAR/EO chip gallery    │
│  - orbital ground tracks   │  - LLM intel brief        │
├────────────────────────────┼───────────────────────────┤
│  TIMELINE SCRUBBER         │  ORCHESTRATION TRACE      │
│  - demo time control       │  - last decision card     │
│  - anomaly markers         │  - info gain bar chart    │
│  - collection bars         │  - alternatives list      │
│                            │  - justification text     │
└────────────────────────────┴───────────────────────────┘
```

### 5.2 Critical visual elements (rank order, changed from v2)

1. **Belief-state visualization.** Growing covariance ellipses between observations, collapsing on collect. This is the demo's core visual.
2. **Orchestration reasoning trace panel.** Shows the most recent cueing decision. Expected info gain as bar chart. Rejected alternatives with their scores. Justification text. Auditable, click-through to per-candidate detail.
3. **Provenance chain on click.** Every track expands to observation list with raw URIs.
4. **Cueing animation.** Satellite glyph → ground track → target → collection pulse → observation drops → ellipse collapses.
5. **Multi-source chip gallery.** Actual SAR/EO chips per selected vessel in detail pane.
6. **Sensor footprint projection on map.**
7. **Alert feed** (terminal aesthetic, demo-time scrolling log of decisions).
8. **LLM intel brief** pre-generated server-side, cached as JSON.

### 5.3 Cesium 3D (MAYBE)

Small corner-panel Cesium viewer showing actual satellite positions in 3D orbit, current pass highlighted, target AOI pinged. Real aerospace feel. 3 days done well. Attempted in Week 7 only if belief-state viz complete end of Week 6.

---

## 6. Repository Structure

```
custody/
├── README.md                          # SDA framing, architecture, honest limitations
├── CLAUDE.md                          # Claude Code context
├── pyproject.toml                     # full dep list
├── src/custody/
│   ├── models.py                      # EXISTING → extended
│   ├── tracks.py                      # EXISTING → EKF-capable
│   ├── anomalies.py                   # NEW
│   ├── pol.py                         # NEW
│   ├── simulate.py                    # EXISTING → playback engine
│   ├── catalog/
│   │   ├── federated.py
│   │   ├── umbra.py
│   │   ├── sentinel.py
│   │   └── gfw_ais.py
│   ├── detection/
│   │   ├── sar_cfar.py
│   │   ├── eo_cnn.py
│   │   └── ais_passthrough.py
│   ├── fusion/
│   │   ├── observations.py
│   │   ├── index.py
│   │   └── tracker.py
│   ├── orchestration/                 # EXISTING (Phase 2) — portfolio layer
│   │   ├── attention.py               # BACKGROUND / WATCHLIST / ACTIVE_CUSTODY tiers
│   │   ├── portfolio.py               # rank_portfolio(), CustodyHealth
│   │   └── __init__.py
│   └── tipcue/                        # ★ HERO module — per-track decision layer
│       ├── passes.py
│       ├── feasibility.py
│       ├── info_gain.py
│       ├── policy.py
│       ├── justification.py
│       └── types.py
├── scripts/                           # one-shot preprocessing
│   ├── 01_fetch_umbra_scenes.py
│   ├── 02_fetch_sentinel1.py
│   ├── 03_fetch_sentinel2.py
│   ├── 04_fetch_ais_gfw.py
│   ├── 05_detect_sar.py
│   ├── 06_detect_eo.py
│   ├── 07_fuse_observations.py
│   ├── 08_associate_tracks.py
│   ├── 09_score_anomalies.py
│   ├── 10_generate_chips.py
│   ├── 11_propagate_passes.py
│   ├── 12_tipcue_decisions.py         # ★ hero pre-computation
│   └── 13_build_timeline.py
├── data/
│   ├── raw/                           # gitignored
│   ├── processed/                     # gitignored
│   └── demo/                          # curated, LFS
├── app/
│   ├── api/
│   └── web/
│       ├── src/
│       │   ├── tokens.css
│       │   ├── Map.tsx
│       │   ├── TrackDetail.tsx
│       │   ├── Timeline.tsx
│       │   ├── OrchestrationTrace.tsx  # ★ new panel
│       │   └── AlertFeed.tsx
│       └── public/
├── tests/
│   ├── test_observations.py
│   ├── test_tracker.py
│   ├── test_cfar.py
│   ├── test_info_gain.py               # ★ math correctness
│   └── test_policy.py
└── docs/
    ├── custody_fusion_implementation_guide_v3.md
    ├── scenario.md
    ├── positioning.md                  # public-facing why-this-exists
    ├── architecture.md
    └── demo_script.md                  # voiceover + Act breakdown
```

---

## 7. Implementation Sequence (10 weeks)

**Day 0 (done).** Data recon complete. Umbra S3 check confirms 8 AOI scenes at 2 repeat features. GFW token in hand. ~13 GB Umbra mirror being staged locally.

**Week 1: Foundations.**
- Scenario doc locked (`docs/scenario.md`).
- Positioning doc locked (`docs/positioning.md`).
- Observation model + covariance math in `fusion/observations.py`.
- EKF extension in `tracks.py`, backward-compat scalar radius as property.
- H3 + DuckDB spatial index in `fusion/index.py`.
- Tests first for each module.

**Week 2: Detection.**
- CFAR over one Sentinel-1 scene + one Umbra scene. Tune by hand. Document tuning.
- AIS passthrough from GFW parquet.
- EO detector only if cloud-free Sentinel-2 scenes exist.

**Week 3: Fusion.**
- Hungarian + EKF tracker in `fusion/tracker.py`.
- Unit tests on toy scenarios first.
- End-to-end run over 1 week of data, validate against expected shipping lane traffic.
- Full 11-week window.

**Week 4: Anomalies + PoL.**
- Four multi-INT anomaly scorers.
- Per-reef-cell density baselines over first 4 weeks, detection over remainder.

**Week 5: Tipcue (the hero).**
Builds `tipcue/` against the existing portfolio output — Phase 2's `orchestration/` already produces `PortfolioAssessment` objects we can filter by `attention_state == "ACTIVE_CUSTODY"`. We're not reinventing a working module, we're composing onto it.
- First: `fusion/tracker.py` exposes `get_track(entity_id) -> Track` so tipcue can pull belief state by the portfolio's join key (~20 LOC adapter).
- `passes.py`: TLE propagation, candidate generation.
- `feasibility.py`: cloud priors, grazing angle, pass geometry.
- `info_gain.py`: the log-det math. Tests first.
- `policy.py`: top-level decision logic.
- `justification.py`: templated NL renderer.
- Pre-compute all decisions over the demo window.

**Week 6: UI scaffold.**
- `tokens.css` first.
- FastAPI + DuckDB serving.
- Mapbox base + deck.gl track layer + covariance ellipses.
- Time scrubber wired to timeline.
- Provenance expansion.

**Week 7: UI hero features.**
- OrchestrationTrace panel.
- Cueing animation.
- Chip gallery.
- Alert feed.
- LLM brief integration.
- Cesium 3D *attempted here* if Week 6 landed clean. Cut without regret if not.

**Week 8: Polish.**
- Visual pass.
- Performance pass (any sluggishness in the scrubber is a bug).
- Secondary-arc documentation (Union Banks case study in repo).
- README pass.

**Week 9: Narrative.**
- Voiceover script finalized against the actually-shipped UI.
- Screen captures of all three Acts.
- Voiceover recording.

**Week 10: Publishing.**
- Final edit of 90-second cut.
- LinkedIn post drafted, reviewed, scheduled.
- Repo public. README final. docs/ final.
- Slack for slippage.

---

## 8. Hurdles and Risks

Reranked for v3:

1. **Belief-state math correctness.** The hero capability depends on getting the info-gain computation right. Silent bugs are catastrophic. Mitigation: tests first, test against hand-computed toy scenarios, sanity-check log-det outputs against a baseline.
2. **Tipcue reasoning trace credibility.** If the justification text reads like marketing, the whole demo loses credibility. Mitigation: templated, literal, numeric. No adjectives. Show your work.
3. **Frontend complexity for the trace panel.** New UI element, no clear reference implementation to copy. Mitigation: wireframe before coding, 2-pass approach (functional first, polish second).
4. **CFAR noise near reefs.** Unchanged from v2.
5. **Sentinel-2 cloud cover.** Unchanged from v2.
6. **Umbra STAC schema differences vs Element84.** Unchanged from v2.
7. **TLE freshness.** Use TLEs from the exact week of each scene.
8. **Cesium 3D scope creep.** Strict gate at end of Week 6. Cut without apology.
9. **Voiceover overreach.** Script reviewed by non-technical reader before recording. If any sentence sounds like a commercial claim, rewrite it.
10. **SDA framing overreach.** Every reference to the capability vectors has to be defensible. Positioning doc is the reference; voiceover follows it literally.

### What this demo does NOT claim

- Not hypersonic tracking.
- Not a production system.
- Not identifying specific flagged vessels.
- Not a Sentry or BlackSky clone.
- Not making legal or sovereignty claims about contested waters.
- Not asserting the orchestration layer is novel — only that its output is legible in a way public commercial artifacts aren't.

---

## 9. Open Questions

Down to 4:

1. **Cesium 3D commit.** Gated on Week 6 completion.
2. **Inter-node hypothesis passing stretch.** Gated on Week 7 completion.
3. **Voiceover recording setup.** $80 USB mic, quiet room, Audacity. Buy this week.
4. **LinkedIn post timing.** Recommend: publish Tuesday morning of Week 10 for maximum reader attention.

---

## 10. What to Say in an Interview

One-minute version, use literally:

> "I built an open-source reference implementation aligned with the Space Development Agency's Custody Layer capability vectors — multi-phenomenology fusion, hypothesis management, low-latency exploitation — applied to maritime domain awareness in the South China Sea. It federates Umbra SAR, Sentinel-1, Sentinel-2, and Global Fishing Watch AIS through a unified observation model with covariance and full provenance. The orchestration story has two layers: a portfolio layer that does attention allocation across the fleet — who's in active custody, who's on the watchlist, who's background — and a tipcue layer that scores candidate collections by expected information gain for individual tracks in active custody. The tipcue layer applies sensor-specific feasibility priors — cloud forecast for EO, grazing angle for SAR — and emits both a tasking decision and a natural-language reasoning trace for every cue. It's a 10-week reference implementation over curated open data — the architecture is real and the decisions are legible; the scenario is chosen because the AIS-dark dynamics in the Spratlys make SAR/AIS fusion the canonical real-world test case. The positioning is answering a community capability call, not a company pitch."

Every sentence is defensible against a skeptical engineer. Every claim maps to a module in the repo.

---

## 11. Positioning Against Public Capability Calls

Custody's feature set maps explicitly to published SDA Custody Layer capability vectors. This is the table that goes in `docs/positioning.md` and a condensed version in the README.

| SDA capability vector | Custody implementation |
|---|---|
| "Automated processing and fusion of data from traditional space-based sensing payloads (visible, infrared, RF, SAR, multispectral)" | Multi-modal fusion of SAR (Umbra + Sentinel-1), EO (Sentinel-2), and AIS through unified Observation schema |
| "Design of a multi-phenomenology fusion architecture that enables agile incorporation of new algorithms" | Pluggable detector interface, per-source STAC adapters, anomaly scorer registry — new sources add in ~200 LOC |
| "Reduction in latency of processing, exploitation, and dissemination" | Architectural patterns for offline pipeline; noted explicitly that production latency work is out of scope for this reference |
| "Memory management and target hypothesis distribution from one satellite node to the next" | Covariance-preserving track state serialization; inter-node handoff as flagged stretch goal |

The table appears literally in the README so readers can verify the alignment claim.

---

*End of v3 guide. Week 1 starts with the scenario doc and observation model. Belief-state math has tests first. Cesium and inter-node are earned, not promised.*
