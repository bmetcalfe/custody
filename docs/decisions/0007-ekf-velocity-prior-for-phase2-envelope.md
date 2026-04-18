---
id: 0007
title: Default EKF velocity prior produces Phase 2 linear-σ envelope via F-coupling
date: 2026-04-18
status: accepted
---

## Status

Accepted.

## Context

The Phase 2 linear growth heuristic `σ(t) = 5 + 3t` was modeling — without labeling it as such — a target under constant-velocity motion with a non-trivial velocity uncertainty. For a CV Kalman filter with initial velocity covariance σ²_v and position process noise q_pos, the position variance evolves as:

```
var_pos(t) = σ²_pos(0)  +  σ²_v · t²  +  q_pos · t
              constant      F-coupling    Brownian
```

At operational time scales the `t²` term — the F-matrix coupling of the initial velocity uncertainty into the position block — dominates, giving `σ ∝ t` as a direct consequence of the CV Kalman math. The EKF is not tuned to approximate the Phase 2 heuristic; the heuristic was a compressed form of exactly this physics.

Two constraints in the Custody repo depend on the linear-σ shape:

- **Portfolio health thresholds** in `src/custody/orchestration/portfolio.py` — `_UNC_DEGRADING_KM = 20`, `_UNC_STALE_KM = 50`, `_UNC_LOST_KM = 90`, `_UNC_MAX_KM = 150` — were calibrated to the linear curve. A well-chosen velocity prior preserves their operational meaning.
- **Demo narrative timing** in `tests/test_multi_day_scenario.py` and the scenario doc expects hour-scale dark-vessel transitions. Linear σ growth produces exactly that.

A default build that ships `σ²_v = 0` (pure Brownian) would be wrong for the threat model — adversarial targets have heading uncertainty that compounds deterministically, not random-walk-like. The right default encodes that heading uncertainty in the initial velocity covariance, not hidden in Q.

## Decision

Set the EKF's default initial velocity covariance `σ²_v = 0.694 (m/s)²` (`σ_v ≈ 0.83 m/s ≈ 3 km/h`) and position process noise `q_pos = 8333 m²/s`. Under these defaults, `σ(1h) = 8.00 km` and `σ(24h) = 77.0 km`, matching Phase 2 at both anchors and interpolating linearly between them to within 0.04%.

Parameters in `src/custody/models.py`:

| Parameter | Old | New |
|---|---|---|
| `_DEFAULT_VEL_SIGMA_MPS` (initial velocity σ) | 1.0 | **0.833** (3 km/h) |
| `_DEFAULT_Q_POS_PER_SEC` | 10833 | **8333** |
| `_DEFAULT_Q_VEL_PER_SEC` | 0.01 | **0.0** |

Setting `q_vel = 0` keeps the velocity covariance constant across predict steps, so the F-coupling term `σ²_v · t²` stays clean. Any nonzero `q_vel` would compound velocity variance across discrete steps and amplify position variance beyond the intended envelope at long t.

Anchor verification under new defaults (1-hour discrete steps):

| t (h) | σ_EKF (km) | σ_linear (km) | err |
|---:|---:|---:|---:|
| 1 | 8.00 | 8.00 | 0.0% |
| 6 | 22.99 | 23.00 | 0.0% |
| 12 | 40.99 | 41.00 | 0.0% |
| 24 | 76.97 | 77.00 | -0.04% |
| 36 | 112.96 | 113.00 | -0.04% |
| 48 | 148.94 | 149.00 | -0.04% |

Portfolio-threshold crossing times under new defaults vs Phase 2:

| Threshold | Phase 2 | EKF-tuned | Δ |
|---:|---:|---:|---:|
| σ = 20 km (DEGRADING) | 5.00 h | 5.10 h | +2.0% |
| σ = 50 km (STALE) | 15.00 h | 15.10 h | +0.7% |
| σ = 90 km (LOST) | 28.33 h | 28.40 h | +0.2% |
| σ = 150 km (MAX) | 48.33 h | 48.40 h | +0.1% |

All thresholds fire within ±2% of Phase 2 timing.

## Consequences

- Portfolio thresholds (20/50/90 km) remain unchanged. Demo narrative timing preserved. Belief-state visualizations keep their visual calibration.
- The EKF remains mathematically correct — we're choosing process noise parameters appropriate to the threat model, not bypassing the filter.
- Regression tests pinned to the anchors:
  - `tests/test_track_ekf.py::test_default_q_anchor_at_one_hour` — σ(1h) within 2% of 8.0 km
  - `tests/test_track_ekf.py::test_default_q_anchor_at_twenty_four_hours` — σ(24h) within 2% of 77 km
  Any future change to default Q that breaks demo-narrative timing fails these tests immediately.
- Tuning scripts live under `day0/scratch/ekf_q_tune.py` and `day0/scratch/ekf_threshold_times.py` for reproducibility and for future re-tuning if the Phase 2 anchors or portfolio thresholds change.
- Fit diverges outside `[0, 48 h]` — beyond 48 h the EKF's trajectory continues to track the linear extrapolation (σ²_vel(0) × t² dominates), but the linear Phase 2 curve itself was never validated beyond ~48 h either. No mission-critical behavior depends on σ at >48 h gaps.
- **Follow-up — per-target-class velocity prior.** Per-target-class uncertainty growth can be expressed as a per-class initial velocity covariance rather than a per-class Q. Faster growth (adversarial targets, unknown heading): larger `σ_v`. Slower growth (well-characterized commercial traffic): smaller `σ_v`. This interpretation is cleaner than per-class process noise because `σ_v` maps directly to "how well do we know the target's heading" — a physical quantity rather than a tuning parameter. Not implemented in V1; tracked as an open follow-up.
