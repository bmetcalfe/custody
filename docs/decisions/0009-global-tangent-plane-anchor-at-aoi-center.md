---
id: 0009
title: Global tangent-plane anchor at AOI center for lat/lon ↔ meters conversions
date: 2026-04-18
status: accepted
---

## Context

Converting `(lat, lon)` observations to meters before feeding the EKF requires an anchor point defining the origin of the local tangent plane. The tangent plane is flat; the Earth is curved; the conversion is exact at the anchor and introduces errors elsewhere proportional to `d³ / (6R²)` where `d` is distance from anchor and `R` is Earth radius.

Three options:

- **Per-track anchor** set at track initialization. Smallest error near track origin, but different tracks have different frames — covariances can't be directly compared without re-projection.
- **Global anchor at AOI center** (9.75°N, 116.0°E for the Spratly demo AOI bounded by 114.5°E–117.5°E, 8.5°N–11.0°N). One frame for all tracks. Projection error bounded by AOI extent.
- **Sliding anchor** re-derived at every predict/update step. Near-zero error always; requires covariance re-expression in each new frame at every step.

Worst-case projection error for the global-anchor option across the Spratly AOI: farthest point ~216 km from center, giving `216³ / (6 × 6371²) ≈ 41 m` error. Position σ starts at 5000 m and grows from there. Projection error is ~1% of the smallest uncertainty we ever carry; in variance terms, a factor of ~15,000 smaller. Physically negligible.

## Decision

Use a global tangent-plane anchor at AOI center: `(9.75°N, 116.0°E)` for the Spratly demo. Projection: Azimuthal Equidistant centered at the anchor, implemented via `pyproj.Transformer`. All observations convert `(lat, lon) → (x, y)` meters in this frame on ingestion; all EKF predict/update math happens in this frame; visualization converts back to lat/lon for map display.

A single module `src/custody/fusion/geo.py` holds the anchor constants and the transformer instances. Anchor values live in `src/custody/config.py` as `AOI_ANCHOR_LAT` and `AOI_ANCHOR_LON` so they're centrally discoverable and easy to override for tests.

## Consequences

- Uniform frame for all tracks. Covariance matrices across tracks are directly comparable.
- Spatial index (H3 r8) and visualization both operate in one consistent frame.
- Projection error negligible relative to position uncertainty for the demo AOI. Documented explicitly so future work deploying outside the Spratly AOI knows this is a local-scale approximation.
- Tests pin the anchor and verify a few known (lat, lon) ↔ (x, y) roundtrips to millimeter precision.
- **Out of scope for v1.** Sliding anchor (needed for continental-scale deployments) and per-track anchor (needed for tracks that span the anchor's far field) are not implemented. A future ADR introduces either if the AOI grows.
- **Config knob.** If the demo AOI moves (e.g., to Sabina Shoal standoff area or a different case study), the anchor moves with it. The 41m worst-case bound holds for any AOI of roughly the same extent (~300 km).
