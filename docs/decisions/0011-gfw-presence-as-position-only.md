---
id: 0011
title: AIS ingestion uses GFW public-tier presence dataset; observations are position-only
date: 2026-04-18
status: accepted
---

## Context

ADR-0008 established a polymorphic `Observation` sum type with two variants: `PositionObservation` (2×2 cov) for image-derived sensors and `PositionVelocityObservation` (4×4 cov) for sensors that directly broadcast velocity. AIS was named as the canonical example of a `PositionVelocityObservation` source, on the assumption that the AIS transponder's SOG/COG broadcasts would be preserved through the ingestion pipeline.

Phase A reconnaissance against the GFW v3 API under a standard research token demonstrated this assumption does not hold at the access tier available to the project:

- The dataset that would contain per-ping SOG/COG (`public-global-ais-position:latest`) returns HTTP 403 "Insufficient permissions."
- The `/v3/events` endpoint (encounters, port visits, loitering) also returns 403.
- `/v3/vessels/{id}/tracks` does not exist at that path.
- The only endpoint that returns per-vessel position data is `/v3/4wings/report` against the `public-global-presence:latest` dataset. This returns hourly, grid-cell-quantized per-vessel dwell records. No SOG, no COG, no heading, no speed.

The probe against a 10 km × 24 h window at Cuarteron-area (2023-07-02) returned 59 records across ~17 distinct vessels, with lat/lon quantized to ~0.01° (~1 km) cell centroids. This is sufficient density for the demo scenario; it is not raw AIS.

Three options were considered:

- **Option A — Upgrade the GFW token.** Pursue research-program access with AIS position dataset. Unknown timeline (typical: days to weeks). Not a Week 2 blocker; not pursued in parallel for this iteration.
- **Option B — Accept presence data as `PositionObservation`.** Ingest what the available endpoint returns, treat it as position-only per the sensor's actual measurement surface, document the quantization honestly. Selected.
- **Option C — Swap to a different AIS source (Spire, MarineTraffic, NOAA, self-hosted).** Spire and MarineTraffic gate access commercially. NOAA archives do not cover the Spratly AOI. Self-hosted receivers do not cover the June-August 2023 historical demo window. Rejected.

## Decision

AIS ingestion in v1 pulls from `/v3/4wings/report` against `public-global-presence:latest` at hourly temporal resolution. Each returned record becomes a `PositionObservation` (not a `PositionVelocityObservation`) with covariance reflecting the grid-cell quantization of the source data.

The ingest module is named `src/custody/ingest/gfw_presence.py` — not `gfw_ais.py` — because what we consume is GFW's processed per-cell-per-hour presence product, not raw AIS broadcasts. Using "AIS" in the module name would overstate fidelity.

Position covariance for GFW presence observations is:

```
GFW_PRESENCE_POS_SIGMA_M = 500.0   # half the ~1 km quantization cell
cov_pos = diag(500², 500²) meters²
```

This is an honest representation of the data's actual spatial resolution, not GPS-grade uncertainty.

`PositionVelocityObservation` remains in the codebase as defined by ADR-0008. No sensor in v1 produces it. The type is preserved for future sensors — an upgraded GFW access tier, Spire, MarineTraffic, Doppler radar, or any sensor that instrumentally measures velocity. Hooking a new source into the existing pipeline requires implementing a parser that emits `PositionVelocityObservation` instances; no downstream code changes are required.

## Consequences

- **`PositionVelocityObservation` is unused in v1.** The type is defined, tested, and handled by the EKF's dispatch logic, but no live pipeline produces instances. This is explicitly tolerable per ADR-0008's polymorphic design — the type system anticipates sensor variety without requiring every variant be active.
- **Velocity estimation is the EKF's job, not the sensor's.** With position-only observations, `TrackState` estimates velocity from sequential position fixes across predict/update cycles. ADR-0007's velocity prior (`σ_v = 0.83 m/s`) was tuned for exactly this case — the prior encodes initial uncertainty about heading before the filter has seen enough pings to estimate it. Convergence is slower than it would be with instrumented velocity, but the math is unchanged.
- **Position covariance is 500 m σ, not 10 m GPS-grade.** The Kalman gain on incoming GFW observations reflects the sensor's actual resolution. A track with prior σ ≈ 500 m being updated by a GFW observation with σ = 500 m will see modest covariance reduction, not collapse. This is correct behavior given the data.
- **Module name `gfw_presence.py` signals the actual source.** Any future reader searching for "AIS ingestion" should find documentation routing them here. A top-of-module docstring explicitly states: "This module ingests GFW's public-tier presence dataset, which is derived from AIS broadcasts but exposed as per-cell-per-hour aggregates. For raw AIS with SOG/COG, upgraded GFW access or a different data source is required."
- **Positioning alignment.** Using GFW public-tier access strengthens the reference-implementation framing: any reader of the repo can reproduce the pipeline with a standard GFW research account. No privileged data access, no reproduction blocker.
- **Demo narrative unchanged.** "AIS-dark at Cuarteron" means "GFW presence data shows no vessel at this feature while Umbra SAR shows persistent returns." This is exactly the cross-source disagreement the demo was designed around.
- **Upgrade path preserved.** If GFW later grants AIS position access, or a different sensor source is added, the upgrade is scoped to the ingest module alone. The Observation type system, EKF, tracker, index, anomaly scorers, and tipcue are all unaffected.

## Notes

- The `gfw_presence.py` module will drop records with the known GFW sentinel values (lat=91, lon=181, etc.) per the earlier Phase A analysis, even though the presence endpoint is less likely than the ping endpoint to emit them.
- `notes` dict on each `PositionObservation` preserves flag state, vessel class, gear type, and GFW's vessel ID for downstream anomaly scoring and provenance display.
- The `AIS_POS_SIGMA_M = 10.0` and `AIS_VEL_SIGMA_MPS = 0.5` constants originally proposed in Stream A's prompt are dropped. They were appropriate for raw AIS, not for presence-dataset data.
