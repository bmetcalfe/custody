# Custody

**An open-source reference implementation of multi-phenomenology fusion and legible tip-and-cue orchestration for maritime domain awareness**

**An open-source reference implementation of multi-phenomenology fusion and legible tip-and-cue orchestration for maritime domain awareness.** Demonstrated on two AIS-dark South China Sea case studies: Vietnamese land reclamation at [Tennent Reef](https://amti.csis.org/vietnam-ramps-up-spratly-island-dredging/) and [Chinese maritime militia](https://amti.csis.org/caught-on-camera-two-dozen-militia-boats-at-whitsun-reef-identified/) activity at Whitsun Reef.

Aligned with the Space Development Agency's publicly published [Custody Layer capability vectors](https://www.sda.mil/custody/).

[![Custody demo](https://github.com/bmetcalfe/custody/raw/main/docs/progression.gif)](/bmetcalfe/custody/blob/main/docs/progression.gif)
*Demo GIF above is the Phase 2 Streamlit UI. The v3 belief-state visualization is scheduled for Week 6–7 and will replace this asset.*

---

## What this is

An open-source reference implementation of multi-sensor fusion and legible tip-and-cue orchestration, built on open commercial and public data:

- **Umbra SAR Open Data** (CC BY 4.0) — high-resolution (25 cm–1 m) SAR imagery
- **Sentinel-1 GRD** — baseline 10 m SAR via Earth Search STAC
- **Sentinel-2 L2A** — opportunistic 10 m optical
- **Global Fishing Watch AIS** — research API

Custody applies architectural patterns described in SDA's Custody Layer capability vectors — **multi-phenomenology fusion**, **hypothesis management**, **low-latency exploitation** — to a domain where open data enables public validation.

### Hero capability: covariance-aware, explainable tip-and-cue orchestration

Every cueing decision the system emits is accompanied by a reasoning trace — what was selected, what alternatives were considered, what the expected information gain was, why rejected candidates were rejected, and a plain-language justification. Commercial systems generally expose the decision. Custody exposes the reasoning.

### What this is *not*

**Not a hypersonic or missile tracking system.** SDA's Custody Layer primarily addresses ballistic and hypersonic threats with kinematics that do not translate to maritime vessels. Custody applies the *architectural patterns* from those capability vectors to the maritime domain. The threat model is AIS-dark commercial and militia vessel traffic, not missile defense.

**Not a production system.** The demo runs against pre-computed artifacts produced by a one-shot preprocessing pipeline. Architecture supports live operation; the demo does not.

**Not a commercial product clone.** Vantor Sentry, BlackSky, Satellogic, and other commercial MDA products have significant internal capabilities and constellations and archives we don't have access to. Custody focuses on the publicly visible gap: making planning decisions and their uncertainty explicit and auditable. Whether or how similar functionality exists inside commercial products is not a claim made either way.

See [`docs/positioning.md`](docs/positioning.md) for the full honest-scoping document.

---

## Demo scenario

**Area of interest:** Spratly Islands hotspot, bbox 114.5°E–117.5°E × 8.5°N–11.0°N (~330 km × 280 km).

**Two co-equal case studies** drive the demo, demonstrating architectural flexibility across scenario types:

### Case Study A — Vietnamese land reclamation at Tennent Reef

**Window:** June 1 – August 20, 2023 (11 weeks).

Active Vietnamese dredging and land reclamation at Tennent Reef (Đá Tiên Nữ), 8.856°N / 114.665°E, imaged by Umbra SAR across 5 scenes spanning 41 days. The final three scenes — August 7, 9, and 13 — include a 48-hour revisit cadence that demonstrates the tip-and-cue architecture in action. CSIS [Asia Maritime Transparency Initiative](https://amti.csis.org/vietnam-ramps-up-spratly-island-dredging/) has documented active Vietnamese land reclamation at this feature since December 2021, with 62 acres of new artificial land added between end-of-2022 and late-2023. The reclamation activity is AIS-dark in GFW presence data, making this a canonical case for SAR/AIS fusion detecting persistent structure that lacks a cooperative-broadcast explanation.

### Case Study B — Chinese maritime militia activity at Whitsun Reef

**Window:** December 2023 – March 2024 (15 weeks).

Vessel flotilla activity at Whitsun Reef (Đá Ba Đầu / Julian Felipe Reef), 9.98°N / 114.63°E, imaged by Umbra SAR across 3 scenes. The site of the [March 2021 Chinese maritime militia swarm event](https://amti.csis.org/caught-on-camera-two-dozen-militia-boats-at-whitsun-reef-identified/) — approximately 220 vessels that prompted international protest from the Philippines and Vietnam. Our scenes show multiple AIS-dark vessel clusters visible in open water. The architecture detects these as point observations, associates them into vessel tracks via Hungarian assignment, and demonstrates cross-INT fusion where SAR returns persistently exist without corresponding AIS broadcasts.

Both case studies use the same architecture: same EKF, same observation types, same spatial index, same tracker. The pipeline is not specialized for either scenario. The demonstration is that the architectural patterns — covariance-aware multi-sensor fusion and explainable tip-and-cue orchestration — work across maritime domain awareness problems of different character.

The demo does not identify specific named vessels, does not take a position on claimant sovereignty, and does not make legal claims. It detects activity consistent with published open-source analytic methodology — the same work AMTI publishes monthly against the same public imagery sources.

See [`docs/scenario.md`](docs/scenario.md) for the locked AOIs, Umbra scene inventories, and Act-by-Act demo narrative for each case.

---

## Architecture — the two-layer story

Custody has two orchestration-adjacent layers that compose cleanly rather than compete:

```
                    ┌─────────────────────────────┐
                    │   N active tracks            │
                    │   with belief state          │
                    └──────────────┬───────────────┘
                                   │
                  ┌────────────────▼────────────────┐
                  │   PORTFOLIO LAYER               │
                  │   (src/custody/orchestration/)  │
                  │   "Who deserves attention?"     │
                  │                                 │
                  │   Attention tiers:              │
                  │     BACKGROUND                  │
                  │     WATCHLIST                   │
                  │     ACTIVE_CUSTODY              │
                  └────────────────┬────────────────┘
                                   │
                                   │  tracks flagged ACTIVE_CUSTODY
                                   │
                  ┌────────────────▼────────────────┐
                  │   TIPCUE LAYER                  │
                  │   (src/custody/tipcue/)         │
                  │   "Which collect to use?"       │
                  │                                 │
                  │   For each candidate pass:      │
                  │     expected info gain          │
                  │     feasibility priors          │
                  │   Emits cueing decision +       │
                  │   reasoning trace               │
                  └─────────────────────────────────┘
```

**Portfolio picks the who. Tipcue picks the what.** Portfolio operates at fleet scope over attention tiers; tipcue operates at per-track scope over individual candidate sensor collections. The demo's hero moment happens at the tipcue layer, but it's the portfolio layer that decides the track is worth a cue in the first place. See [ADR-0001](docs/decisions/0001-rename-v3-orchestration-to-tipcue.md).

### SDA capability vector alignment

| SDA Custody Layer capability vector | Custody implementation |
|---|---|
| Automated processing and fusion of data from traditional space-based sensing payloads (visible, infrared, RF, SAR, multispectral) | Multi-modal fusion of SAR (Umbra + Sentinel-1), EO (Sentinel-2), and AIS through unified Observation schema |
| Design of a multi-phenomenology fusion architecture supporting agile incorporation of new algorithms | Pluggable detector interface, per-source STAC adapters, polymorphic Observation type, anomaly scorer registry |
| Reduction in latency of processing, exploitation, and dissemination | Offline pipeline demonstrates the architectural patterns; production latency work is documented as out of scope for this reference |
| Memory management and target hypothesis distribution from one satellite node to the next | Covariance-preserving track state serialization; inter-node handoff flagged as a stretch goal |

---

## What's built (as of Week 1)

**The v3 `src/custody/fusion/` package** (~885 LOC, 75 tests):
- `fusion/geo.py` — AEQD tangent-plane projection anchored at AOI center for meters-basis EKF math
- `fusion/observations.py` — polymorphic `Observation` sum type: `PositionObservation` (SAR, EO) and `PositionVelocityObservation` (AIS) with honest covariance shapes per sensor
- `fusion/index.py` — H3 r8 + DuckDB spatial-temporal index over Parquet
- `fusion/tracker.py` — Hungarian assignment with Mahalanobis gating, N-of-M track lifecycle, EKF predict/update per track

**EKF belief state** (`src/custody/models.py`):
- 4-dimensional state `[x_east_m, y_north_m, v_north, v_east]` in tangent-plane meters basis
- 4×4 covariance with Joseph-form updates
- Process noise tuned so σ growth matches published maritime-tracking envelopes at operational time scales — not by empirical fit but as a direct consequence of CV Kalman F-coupling math. See [ADR-0007](docs/decisions/0007-ekf-velocity-prior-for-phase2-envelope.md).

**Phase 2 portfolio and attention engine** (`src/custody/orchestration/`, `src/custody/anomalies.py`, `src/custody/compounds.py`, ~2000 tests):
- Per-vessel behavioral baselines with ML (Isolation Forest trained on real NOAA AIS) and heuristic detectors
- Attention-tier classification, custody-health scoring, neglect pressure, preemption tradeoffs
- Dash + Streamlit UIs for the Phase 2 demo
- Preserved and composed with the v3 layers per ADR-0001. Not ancestral, not deprecated — a live subsystem that feeds the v3 tipcue layer.

**Day 0 data reconnaissance:**
- 56 GB Umbra SAR mirror across 219 files
- 8 AOI scenes at 2 repeat-imaged Spratly features
- Scene inventory, GFW AIS token, Sentinel STAC queries all validated

---

## What's coming

**Week 2 (now):** Detection. CA-CFAR over Umbra and Sentinel-1 SAR. AIS passthrough from GFW Parquet. Opportunistic Sentinel-2 EO if cloud-free scenes exist.

**Week 3:** Run the v3 fusion pipeline over the full 11-week window. Validate track lifecycles against known shipping lane patterns.

**Week 4:** Anomaly scoring. Four multi-INT anomalies including SAR-only-dark-vessel (the hero anomaly for the demo narrative).

**Week 5:** The tipcue layer. Covariance-aware candidate scoring via expected information gain, feasibility priors per modality, natural-language reasoning trace for every cueing decision.

**Week 6–7:** Frontend. React + Mapbox + deck.gl + FastAPI. Belief-state uncertainty ellipses, orchestration trace panel, provenance chain. Cesium 3D if belief-state viz lands on time.

**Week 8–9:** Polish and voiceover.

**Week 10:** Publish.

See [`docs/custody_fusion_implementation_guide_v3.md`](docs/custody_fusion_implementation_guide_v3.md) for the full 10-week plan, design decisions, and honest scoping.

---

## References

- [`docs/positioning.md`](docs/positioning.md) — public-facing "why this exists"
- [`docs/custody_fusion_implementation_guide_v3.md`](docs/custody_fusion_implementation_guide_v3.md) — full v3 architecture and weekly sequence
- [`docs/scenario.md`](docs/scenario.md) — locked AOI, window, Umbra inventory
- [`docs/decisions/`](docs/decisions/) — 10 architecture decision records from Week 1
- [SDA Custody Layer](https://www.sda.mil/custody/) — primary community-call reference
- [CSIS Asia Maritime Transparency Initiative](https://amti.csis.org/) — methodology grounding for both case studies:
  - Tennent Reef: [Dec 2022](https://amti.csis.org/vietnams-major-spratly-expansion/) and [Nov 2023](https://amti.csis.org/vietnam-ramps-up-spratly-island-dredging/) reports
  - Whitsun Reef: [March 2021 swarm coverage](https://amti.csis.org/caught-on-camera-two-dozen-militia-boats-at-whitsun-reef-identified/)

---

## Quickstart

```bash
uv sync
uv run pytest tests/                    # full suite, currently ~2156 tests
uv run python src/app/dash_app.py       # Phase 2 Dash UI
uv run streamlit run src/app/streamlit_app.py   # Phase 2 Streamlit UI (legacy)
```

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/).

The v3 demo UI (React + Mapbox + deck.gl) ships in Week 6–7.

---

## Honest limitations

This is a 10-week evenings-and-weekends project by one engineer. The limitations below are documented in more detail in [`docs/positioning.md`](docs/positioning.md).

- The demo scenario is data-driven rather than narrative-first: the primary feature (Tennent Reef, Vietnamese reclamation) was confirmed during Week 2 reconnaissance against actual SAR imagery. See ADR-0012 for how the scenario was locked. The architectural pipeline is scenario-agnostic.
- Umbra coverage over the chosen AOI is 8 scenes at 2 features over 9 months. Sentinel-1 provides continuous fill-in at 10 m resolution.
- Detection uses classical CFAR (SAR) and a pretrained CNN (EO). No fine-tuning on the specific AOI.
- The EKF uses a constant-velocity motion model. Appropriate for the maritime domain studied; not for high-maneuver targets.
- Association uses Hungarian with Mahalanobis gating. Multiple Hypothesis Tracking (MHT) is documented as future work, not implemented.
- Tipcue's feasibility priors are simplified: binary for SAR grazing angle, probabilistic for cloud forecast. Production systems would extend these.
- Sensor fingerprinting, vessel re-identification from imagery, and 20+ year archive pattern-of-life are commercial capabilities we do not reproduce.
- The demo does not identify specific flagged vessels and does not make legal or sovereignty claims.

---

## License

Code: MIT. Data: respective source licenses (Umbra ODP is CC BY 4.0; Sentinel is Copernicus open; GFW is research license; NOAA AIS is public domain). See `LICENSE` for full terms.

---

*Custody is an open-source reference implementation answering a publicly published community capability call. It is not a pitch, a job application, or a product. If the architectural patterns are useful, fork the repo. If they're not, the postmortem is itself useful.*
