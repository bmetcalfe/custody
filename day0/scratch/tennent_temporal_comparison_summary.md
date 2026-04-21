# Tennent 07-02 → 07-23 temporal persistence analysis

*Diagnostic output from `day0/scratch/tennent_temporal_comparison.py`.  No new code in `src/`; no fusion-tracker integration per the Week-3 design decision (ADR-0017).*

Generated 2026-04-21T13:21:50.386654 UTC.

## Method

Direct spatial matching between 07-02 and 07-23 VLM detection sets.  Pairing is
computed via `scipy.optimize.linear_sum_assignment` on a cost matrix where
`C[i, j]` is the geodetic distance in meters (pyproj.Geod WGS84) between
07-02 observation `i` and 07-23 observation `j`, gated at **50 m**.  Entries
beyond the gate receive a `_BIG_COST` sentinel; Hungarian returns the
minimum-total-distance pairing subject to the constraint.

The 50 m threshold is sized for VLM bbox uncertainty per Phase D.5 findings —
Claude's bounding boxes are loose enough that per-hull localization is
approximate at the ~tens of meters scale.  50 m is slightly larger than the
observed bbox centroid jitter across repeated VLM calls on the same tile.

## Aggregate counts

| category | count | % of scene total |
|---|---:|---:|
| PERSISTENT_07-02 (07-02 obs with 07-23 match ≤ 50 m) | 22 | 52.4% of 42 |
| PERSISTENT_07-23 (reciprocal) | 22 | 50.0% of 44 |
| EMERGED (07-23 with no 07-02 match) | 22 | 50.0% of 44 |
| DISAPPEARED (07-02 with no 07-23 match) | 20 | 47.6% of 42 |

Persistent pair count (should be equal on both sides): 22 = 22
✓ balanced

## Match distance distribution

Across the 22 persistent pairs:

- min: 6.61 m
- mean: 21.35 m
- max: 44.30 m

Well below the 50 m gate, suggesting the matches are genuine co-location rather than gate-saturated approximations.

## Spatial distribution

- PERSISTENT_07-02: lat [8.850522, 8.858872], lon [114.661146, 114.671599]
- PERSISTENT_07-23: lat [8.850283, 8.858905], lon [114.661059, 114.671482]
- EMERGED:         lat [8.851850, 8.861674], lon [114.660743, 114.670713]
- DISAPPEARED:     lat [8.851222, 8.858541], lon [114.659727, 114.671787]

(Both Tennent AOIs are centered on (8.8557°N, 114.6651°E) with ±0.009° half-width.)

## Honest caveats

1. **Two timestamps is the minimum for "persistence" to mean anything.**  A target appearing in both scenes could be a long-moored vessel that happens to be there both days, a fixed infrastructure element, or the same vessel returning coincidentally.  This analysis distinguishes "spatially consistent across the 21-day interval" from "only in one scene," not "vessel vs infrastructure."
2. **Persistent ≠ fixed infrastructure.**  Conflating the two would overclaim.  A persistent detection near the reclamation structure is *more likely* infrastructure than a persistent detection in open water, but nothing here is ground truth.
3. **The 50 m threshold is calibrated for VLM bbox uncertainty, not per-hull localization.**  A per-hull coordinate-accuracy threshold would be tens of meters in X-band SAR at 0.34 m/pixel; VLM bbox centroid jitter is the dominant error source here.  Different detector outputs (e.g., YOLO or CFAR point centroids) would warrant different thresholds.
4. **Coordinate transforms are honest.**  Both scenes went through the same `sar_common.crop_to_aoi` + `pixel_to_latlon` path that was validated in Phase F.5.  Pixel-size difference between scenes (0.342 vs 0.332 m/px, documented in the 07-23 README) is absorbed by the geodetic distance calculation; we're not comparing pixel coordinates across scenes.
5. **Greedy global assignment, not mutual-nearest-neighbor.**  Hungarian minimum-cost matching under a hard gate will pair a 07-02 obs with its *globally-best* 07-23 partner even if an asymmetric-mutual-nearest rule would have left it unmatched.  For pairs well below 50 m the two approaches agree; near the gate boundary they can differ.

## Artifacts

- Parquet: `data/processed/vlm_detections/tennent_temporal_comparison.parquet` (86 rows, one per observation)
- Overlay image: `day0/scratch/tennent_temporal_comparison.png` (1600-side thumbnail over 07-23 AOI)
- This document: `day0/scratch/tennent_temporal_comparison_summary.md`
