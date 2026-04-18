---
id: 0010
title: TrackState mean stored in tangent-plane meters, not lat/lon radians
date: 2026-04-18
status: accepted
---

## Context

The EKF introduced in ADR-0005 stores state mean as [lat_rad, lon_rad, v_n, v_e] — a mix of angular position and linear velocity. Both predict and the original update use a hand-rolled spherical approximation (multiply by Earth radius, scale longitude by cos(lat)) to convert lat/lon changes into meters for the covariance math. ADR-0009 subsequently locked AEQD-through-pyproj as the canonical lat/lon ↔ meters projection for EKF predict/update math.

Keeping mean in radians after ADR-0009 means the EKF has two inconsistent projection paths: predict advances position via spherical approximation, update projects observations via AEQD. They agree to millimeter precision near the anchor but diverge measurably at AOI edges. Over repeated predict/update cycles the disagreement accumulates as untested drift.

## Decision

Store TrackState.mean as [x_m, y_m, v_n, v_e] in the AEQD tangent-plane frame anchored at AOI center (per ADR-0009). Expose lat and lon as properties that compute via fusion.geo.from_tangent_plane on access. TrackState.__init__ accepts lat/lon in degrees and projects internally; callers never construct mean in meters directly.

Predict becomes mean = F @ mean — no trigonometry, no projection, no Earth radius. Update projects the observation's lat/lon (degrees) once via fusion.geo.to_tangent_plane, does the Kalman math in meters, updates mean in place. The spherical approximation in the current implementation is retired.

## Consequences

- One projection path. fusion.geo is the only place lat/lon ↔ meters happens for EKF math.
- Predict is a single matrix multiply. Simpler, faster, no numerical weirdness near high latitudes.
- Mean and cov are in the same frame (meters-basis) for the first time. Simplifies info-gain math for tipcue (Week 5).
- 5 test sites in tests/test_track_ekf.py reference t.mean = np.array([lat_rad, lon_rad, ...]). These migrate to either passing lat/lon degrees to the constructor, or reading via track.lat / track.lon properties.
- Breaking change to TrackState.mean semantics. No production code reads mean today (verified via grep), so no production migration needed.
- External API (TrackState(lat=, lon=, v_n=, v_e=, ...)) remains in degrees — callers don't need to know about the tangent plane.
