# Calibration: Scene Quality Dynamic-Range Thresholds — 7-Scene Baseline (2026-04-21)

*Private project documentation. Empirical calibration record for the
`dynamic_range` thresholds in `src/custody/detection/quality.py`.
Cited by ADR-0018.*

## Summary

Ran the scene quality screen against all 7 canonical Custody scenes to
validate the calibrated red/yellow/green thresholds in `quality.py`.
Results confirm:

1. The known-null Tennent 2023-08-09 scene trips red cleanly (DR=9,
   threshold 15 in AOI mode).
2. All six healthy scenes pass green at their applicable thresholds.
3. The calibration is tight on two axes: Tennent AOI margin against
   yellow (08-07 at DR=35, 10 points above the 25 threshold) and
   Whitsun full-scene margin against yellow (both scenes at DR=12
   exactly, zero-slack against the strict-less-than 12 threshold).

## Run context

- Script: `day0/scratch/quality_calibration_run.py`
- Run date: 2026-04-21
- Tennent mode: AOI, 2 km box centered at 8.855687°N, 114.665145°E
- Whitsun mode: full-scene (no AOI)
- Output is deterministic — stats are exact over non-zero pixels of
  the GEC amplitude rasters.

## Results

| scene | full DR | AOI DR | flag | flag basis |
|---|---:|---:|---|---|
| tennent_20230702 | 53 | 80  | green | AOI DR 80 ≥ 25 |
| tennent_20230723 | 41 | 69  | green | AOI DR 69 ≥ 25 |
| tennent_20230807 | 41 | 35  | green | AOI DR 35 ≥ 25 (tightest Tennent margin) |
| tennent_20230809 |  9 |  9  | **red** | AOI DR 9 < 15 |
| tennent_20230813 | 59 | 77  | green | AOI DR 77 ≥ 25 |
| whitsun_20231206 | 12 | n/a | green | full DR 12 ≥ 12 (strict-less-than, boundary) |
| whitsun_20240320 | 12 | n/a | green | full DR 12 ≥ 12 (strict-less-than, boundary) |

Full stats table (mean / std / p50 / p90 / p99 / DR) is produced by
re-running the script.

## Interpretation

### p99-alone would not discriminate

Full-scene p99 across scenes: tennent_20230809 is 58; whitsun_20231206
is 61; whitsun_20240320 is 60. The known-null scene and the two
healthy Whitsun scenes cluster within 3 intensity units on p99. Any
p99-only threshold either false-flags Whitsun or misses 08-09.
`dynamic_range = p99 − p50` discriminates cleanly: 08-09 at 9 versus
healthy-Whitsun at 12 versus healthy-Tennent at 41+.

### 08-09 is intrinsically low-SNR, not AOI-dependent

Full-scene DR and AOI DR are both 9 — identical. The AOI crop over
the reef is as featureless as the surrounding ocean. Confirms this
is an acquisition-quality failure, not a crop-sensitivity artifact.
Even in full-scene mode (threshold 10), the scene would still trip
red.

### Tennent 08-07 is the AOI yellow-threshold anchor

Healthy Tennent AOI DR values span 35 to 80. The 08-07 value (DR=35)
is the floor; the other four scenes sit at DR=69-80. 08-07 was
collected under the most different geometry of the Tennent set
(UMBRA-04, 0.52 m/px, coarsest pixel size), which aligns with its
compressed dynamic range. The AOI yellow threshold of 25 leaves 08-07
with 10 points of margin — the calibration's tightest Tennent point.

### Whitsun is zero-slack on the full-scene yellow threshold

Both Whitsun scenes measure DR=12 exactly. The full-scene yellow
threshold is strict-less-than 12, so DR=12 passes green. A future
Whitsun-like scene measuring DR=11 would trip yellow. This is a
deliberate calibration choice — the thresholds are fit to the present
dataset, and drift in the Whitsun-like regime is a known sensitivity.

### The AOI-vs-full-scene threshold asymmetry is empirical, not design

Healthy Tennent AOI DR (35-80) runs higher than healthy Tennent
full-scene DR (41-59) on average, because the AOI crop concentrates
the bright-scatterer population (reef returns) into a smaller frame.
Asymmetric thresholds (AOI red 15 / yellow 25; full red 10 / yellow
12) reflect that distribution difference rather than a threshold
inconsistency. The same physical signal quality produces different DR
values at different crop extents.

## Reproducibility

Re-run with:

    uv run python day0/scratch/quality_calibration_run.py

Output is a plain-text stats table. The per-scene DR values are
exact (no sampling, no random state), so any drift between runs
reflects either a change in the underlying rasters or a change in
`quality.py::_compute_stats`.

## Status

Referenced by ADR-0018 (Implementation → Calibration subsection).
