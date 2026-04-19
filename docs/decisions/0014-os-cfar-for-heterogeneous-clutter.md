---
id: 0014
title: OS-CFAR as second detector variant for heterogeneous-clutter scenes
date: 2026-04-19
status: accepted
---

## Context

The Custody architecture uses CA-CFAR (Cell-Averaging Constant False Alarm Rate) as its SAR point-target detector, implemented in `src/custody/detection/sar_cfar.py`. CA-CFAR computes a threshold based on the mean of reference cells around each pixel and declares a detection when the pixel exceeds threshold × α. The approach works cleanly for homogeneous clutter — i.e. backgrounds whose statistical distribution is approximately exponential in the power domain.

Week 2 Day 3 detection tuning revealed that CA-CFAR at any reasonable α fails on Umbra scenes over Whitsun Reef. The Whitsun scenes show:

- Brighter, more uniform background than Tennent scenes (mean ≈ 50, std ≈ 19 vs Tennent's heavier-tailed distribution)
- Real vessel targets with signal-to-clutter ratio (SCR) in the 2.5× to 5.8× range — lower than Tennent's vessels/structures
- A "detection cliff" behavior in CA-CFAR alpha sweeps: 412 detections at α=4.0 (mostly speckle false alarms) collapsing to 18 at α=4.5 and 0 at α≥5.5, with no α producing clean signal-on-targets

Root cause analysis: CA-CFAR's mean-based threshold assumes the reference cells are drawn from a known distribution (exponential in power domain). Whitsun's actual clutter is a mixture — smooth water + wave texture + possible residual bathymetry effects + azimuth-smearing artifacts from nearby targets. The mixture is not well-fit by an exponential. CA-CFAR's threshold consequently either admits too many false alarms (when α is low) or rejects real targets along with noise (when α is high).

This is a textbook failure mode for CA-CFAR. The standard literature solution is OS-CFAR (Ordered-Statistic CFAR), which replaces the reference cell mean with a fixed percentile (typically 75th). OS-CFAR is:

- Robust to heterogeneous clutter distributions
- Robust to target-near-reference-window contamination (a percentile is not dragged by a few bright outliers the way a mean is)
- Well-established in the SAR ship detection literature (Rohling 1983 and subsequent)

## Decision

Implement OS-CFAR as a second detector variant in `src/custody/detection/sar_cfar.py`, alongside the existing CA-CFAR. The module exposes two detection functions:

- `detect_points_ca_cfar(img, alpha, guard, reference, ...)` — existing, unchanged
- `detect_points_os_cfar(img, alpha, guard, reference, k_percentile, ...)` — new

The per-case-study configuration (introduced in ADR-0011's follow-through and now extended) gains a `detector_variant` field. Case Study A (Tennent) uses `"ca_cfar"` with α=7.0. Case Study B (Whitsun) uses `"os_cfar"` with α tuned from a fresh sweep.

The dispatch function `detect_points_to_observations()` reads `detector_variant` from config and calls the appropriate detection function.

## Consequences

- **Two detectors coexist.** CA-CFAR remains the simpler default for homogeneous-clutter scenes. OS-CFAR becomes the choice for heterogeneous-clutter scenes. The config framework supports per-case-study selection without code changes per scene.
- **Config hierarchy expanded.** `config/sar_detection_params.json` now carries `detector_variant` alongside `alpha` and `aoi_crop_km_half_width`. Same three-level resolution: scene-specific > case-study > source defaults.
- **New dependency on percentile operations.** OS-CFAR uses `scipy.ndimage.percentile_filter`. No new dependency adds; scipy is already in pyproject.
- **Implementation choice documented.** OS-CFAR's annular reference window (guard region excluded from the window) is approximated by using a rectangular percentile-filter window without guard exclusion. Guard contamination is tolerated because percentile is already robust to a small fraction of outlier pixels in the reference window — this is one of OS-CFAR's intrinsic advantages. If target detection quality is insufficient at this approximation, a second-iteration implementation would compute percentiles over explicitly-masked annular windows; that implementation is 10× slower but more precise. Out of scope for v1.
- **Honest documentation of scene-dependence.** The project now explicitly states that SAR detection is not a universal algorithm in the demo narrative — different clutter types require different detector variants. This is a real applied-algorithm engineering finding and is captured in the implementation guide.
- **Test coverage.** New OS-CFAR tests in tests/test_detection_sar_cfar.py mirror the existing CA-CFAR tests (false-alarm rate on pure noise, single target detection, multi-target detection, structure mask integration). Additional OS-CFAR-specific tests: robustness to clutter heterogeneity (inject noise with lognormal contamination and verify OS-CFAR stays accurate where CA-CFAR fails), robustness to target-near-target masking (two close targets; both detected by OS-CFAR even when CA-CFAR misses the second).
- **No new ADR for OS-CFAR parameters per se.** The k_percentile (default 0.75), guard, and reference parameters are tunable through the config but have CA-CFAR-equivalent defaults. The architectural decision is "use OS-CFAR for heterogeneous scenes;" the specific numerical values are implementation knobs.

## What's explicitly deferred

- **Annular-window implementation of OS-CFAR.** Current approximation uses rectangular windows. If Whitsun detection quality is insufficient at this approximation, the next implementation step is explicit annular percentile computation. Not required for v1.
- **Adaptive detector variant selection per scene.** A production system might auto-detect clutter heterogeneity and select the detector automatically. We explicitly do not implement this; the config specifies the variant per case study. This is honest about the manual tuning reality.
- **Third detector variant (TM-CFAR, GO-CFAR, or learned).** Only CA-CFAR and OS-CFAR are implemented. Other variants would be future work if a future scene type exhibits a failure mode neither handles.

## Supersedes

None. This ADR extends the detector layer introduced in the Week 2 Day 1 CA-CFAR implementation without replacing it. CA-CFAR remains the default for Tennent-type scenes.
