---
id: 0007
title: Tune EKF process noise to match Phase 2 heuristic in demo operational range
date: 2026-04-18
status: accepted
---

## Status

Accepted.

## Context

The EKF's default predict step produces position variance linear in time, which means σ grows proportional to √t (Brownian diffusion). The retired Phase 2 heuristic produced σ linear in t: `σ(t) = 5 + 3t`. These are different physical models — Brownian growth is mathematically correct for unbiased random-walk tracking under Gaussian white noise, while linear growth is the correct envelope for a target whose heading error compounds deterministically.

Two constraints in the Custody repo depend on the Phase 2 linear shape:

- **Portfolio health thresholds** in `src/custody/orchestration/portfolio.py` — `_UNC_DEGRADING_KM = 20`, `_UNC_STALE_KM = 50`, `_UNC_LOST_KM = 90`, `_UNC_MAX_KM = 150` — were calibrated to the linear curve. Under linear growth a vessel hits LOST at ~28 hours after AIS dropout; under default EKF Q, ~207 hours.
- **Demo narrative timing** in `tests/test_multi_day_scenario.py` and the scenario doc is calibrated to the linear curve — vessel transitions to LOST/STALE states happen on hour-scale gaps, not day-scale.

The thresholds encode operational meaning (at what σ does a track become too large to usefully task against), not a statistical property. For adversarial dark vessels, linear σ growth is also the better model — heading errors accumulate deterministically rather than diffusing.

Q should be tuned to the threat model, not defaulted to the Gaussian textbook value.

## Decision

Tune the EKF's initial velocity covariance and process-noise parameters so the discrete 1-hour predict trajectory matches `σ(t) = 5 + 3t` as closely as possible in the 0-48 h operational range. The tuning mechanism exploits the F-matrix coupling of initial velocity variance into position variance over time: with `σ²_vel(0)` held constant and non-zero, position variance grows as `σ²_pos(0) + σ²_vel(0) × t² + q_pos × t`, which matches `(5 + 3t)²` across the operational range.

Chosen parameters in `src/custody/models.py`:

| Parameter | Old | New |
|---|---|---|
| `_DEFAULT_VEL_SIGMA_MPS` (initial velocity σ) | 1.0 | **0.833** (3 km/h) |
| `_DEFAULT_Q_POS_PER_SEC` | 10833 | **8333** |
| `_DEFAULT_Q_VEL_PER_SEC` | 0.01 | **0.0** |

Setting `q_vel = 0` is essential — any nonzero q_vel causes velocity variance to compound across discrete predict steps, which then amplifies position variance through F-coupling and blows up σ at long t. Keeping σ_vel constant preserves the clean linear-σ envelope.

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
- **Follow-up — per-target-class Q.** Commercial vessels following shipping lanes have lower compounding heading error than adversarial militia targets and would be better modeled with a lower `σ_vel(0)`. A future ADR can introduce per-target Q selection. Not implemented in V1; tracked as an open follow-up.
