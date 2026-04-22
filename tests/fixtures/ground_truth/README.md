# Ground truth reference data

Hand-labeled pair labels for persistence matcher evaluation.

Per-pair-file format: one Markdown file per scene pair, with chip previews
and human `same` / `different` / `ambiguous` labels inline.  Paired CSV
file provides programmatic access to matcher verdicts and bbox metadata.

## Files

- **`tennent_0702_0723.md` / `.csv`** — Tennent 2023-07-02 ↔ 2023-07-23
  (same-geometry, same-sensor pair).  43 candidate pairs, 40 scored,
  2 ambiguous, 1 unfilled.  Built during V1 SignatureMatcher evaluation
  (ADR-0019, 2026-04-21).
- **`chips/tennent_0702_0723/`** — 128×128 to 384×384 pixel PNG chips
  extracted from source SAR imagery at each observation's centroid.  Both
  observations per pair have side-by-side chips for visual inspection.

## Using this data

- **Generate / regenerate a ground-truth scene-pair sheet:**
  `python scripts/ground_truth_prepare.py`
  (parameters at top of script; safe to re-run — preserves existing labels)
- **Score a matcher against existing labels:**
  `python scripts/ground_truth_evaluate.py`

## Future ground truth sets

Planned per ADR-0020:

- **`tennent_0702_0807.md`** — cross-geometry same-reef pair (required
  before V2 validation).

Whitsun ground truth remains as future work.
