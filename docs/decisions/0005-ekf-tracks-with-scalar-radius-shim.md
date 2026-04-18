---
id: 0005
title: Replace scalar-radius track model with EKF; retain uncertainty_km and add position_sigma_km as shim properties
status: accepted
date: 2026-04-18
---

## Status

Accepted. Rewrite not yet executed — blocked on confirmation before Week 1 code moves.

## Context

`TrackState` is defined in `src/custody/models.py` with a scalar `uncertainty_km` field. Update logic lives in `src/custody/tracks.py` as three free functions: `update_position`, `update_uncertainty`, and `custody_confidence`. v3 §2.6 and §2.8 require EKF with 4-dim state `[lat, lon, v_n, v_e]` and 4×4 covariance for observation-level fusion.

A 4×4 covariance is a strictly richer representation than a scalar radius, but the rest of the repo (display layer, decision trace, alerting) reads `track.uncertainty_km` as a scalar in km. Breaking every caller to consume the covariance directly is out of scope for Week 1.

## Decision

Replace the scalar `uncertainty_km` field on `TrackState` with a 4×4 covariance matrix. Retain an `uncertainty_km` property that computes `sqrt(trace(cov_pos) / 2)` as a backward-compatible read accessor — same name, same numeric meaning, zero migration cost for existing callers. Add a second property `position_sigma_km` returning the same value under the v3-aligned name, available for new code that wants the standard term. The three free functions in `tracks.py` (`update_position`, `update_uncertainty`, `custody_confidence`) are retired in favor of `TrackState.predict(dt)` and `TrackState.update(obs)` methods.

EKF specifics:

State vector `[lat, lon, v_n, v_e]` maintained as radians + m/s at rest, but the covariance's 2×2 position block is projected to a local tangent plane in meters² for prediction and update. Conversion to/from lat/lon radians happens at the predict/update boundary. Velocity block is in (m/s)². This keeps the covariance in consistent metric units for trace-based radius computations and Kalman gain calculations.

- Predict step: constant-velocity motion model with configurable process noise Q
- Update step: linearized observation model, Kalman gain, Joseph-form covariance update
- Process noise tunable per-modality arrival rate

Shim properties:

```python
@property
def uncertainty_km(self) -> float:
    # Position-block 2×2 covariance → circular-equivalent 1σ radius in km
    cov_pos = self.cov[:2, :2]
    trace = cov_pos[0, 0] + cov_pos[1, 1]
    return math.sqrt(trace / 2.0) / 1000.0

@property
def position_sigma_km(self) -> float:
    # v3-aligned name for new code; identical value
    return self.uncertainty_km
```

Constructor backward compat: `TrackState(uncertainty_km=float)` continues to work by projecting the scalar to an isotropic position covariance and default velocity covariance.

## Consequences

- `TrackState` public surface: the `uncertainty_km: float` dataclass field is replaced by a `cov: np.ndarray` field plus an `uncertainty_km: float` property (same name) and a `position_sigma_km: float` property. Reads at existing call sites are unchanged.
- Retired free functions, now methods on `TrackState`:
  - `update_position(vessel, hours)` → `TrackState.predict(dt)` (and the separate `Vessel` position step is absorbed or kept as a Vessel method — to be decided at implementation time)
  - `update_uncertainty(uncertainty, hours)` → folded into `TrackState.predict(dt)` via the process-noise model
- `custody_confidence(uncertainty)` free function is retired. `TrackState` gains a `confidence` property that computes the same `exp(-radius_km / 50)` formula from the covariance-derived `position_sigma_km`. The 2 callsites (`models.py:151`, `test_collection_feedback.py:265-267`) migrate to `track.confidence`.
- Direct callers of the retired `tracks.py` functions — `src/custody/ais.py:291`, `src/custody/simulation/timeline.py:241,243`, `tests/test_ais_stale_confidence.py:159,160,207,208` — need migration to the method API.
- The `uncertainty_km` property has a matching setter that projects a scalar km value to an isotropic 2×2 position covariance block: `cov[:2,:2] = (r_km * 1000.0)**2 * np.eye(2)`. Velocity covariance block is preserved if already set, otherwise defaults to a configurable prior. Constructor kwarg `TrackState(uncertainty_km=N)` uses the same projection. This preserves write-path backward compat — every existing `track.uncertainty_km = N` site continues working unchanged.
- Belief-state visualization in the UI becomes possible — growing/collapsing covariance ellipses are the v3 §5.2 hero visual.
- Tipcue's info-gain computation (Week 5) has a well-defined input: the 4×4 posterior covariance is what `score(c) = ½ log|Σ_prior| − ½ log|Σ_posterior(c)|` operates on.
- Per the v3 CLAUDE.md "Before starting any task" checklist: this touches belief-state math, so tests come first. Hand-computed toy scenarios for predict + update are mandatory before integration.
- **Alternative name `uncertainty_radius_km` was considered and rejected.** A shim exists to avoid migration; renaming the shim while claiming backward compat is self-contradictory, and the v3-aligned `position_sigma_km` name is available as a second property for new code. This keeps the migration surface bounded to the sites that genuinely need EKF-aware logic (assignments and retired-function calls), not every scalar read.

## Behavioral Note

The EKF's growth curve under default Q and `σ_v` is linear in `t` over operational time scales, matching the Phase 2 heuristic as a direct consequence of CV Kalman math per [ADR-0007](0007-ekf-velocity-prior-for-phase2-envelope.md). The position variance evolves as `σ²_pos(0) + σ²_v · t² + q_pos · t`; at operational time scales the `t²` F-coupling term dominates, so `σ ∝ t`. Beyond the range where the Phase 2 linear curve was itself validated (~48 h), no mission-critical behavior depends on σ.
