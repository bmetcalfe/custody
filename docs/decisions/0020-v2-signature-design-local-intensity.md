---
id: 0020
title: V2 signature design — local intensity profile features for SignatureMatcher
date: 2026-04-21
status: proposed
---

## Context

ADR-0019 committed to feature signatures (Option B) as the V1 direction for the geometry-aware `SignatureMatcher`.  V1 shipped at `src/custody/fusion/temporal.py` with the signature vector `[bbox_w / 200, bbox_h / 200, log2(w/h) / 3, classification_conf]` — entirely derived from VLM annotation output.

A hand-labeled ground-truth set was built specifically to break the circularity of evaluating V1 against `DirectSpatialMatcher`-as-truth: `day0/scratch/handlabel_sheet.md` enumerates all 43 candidate pairs within the 50 m spatial gate on the Tennent 2023-07-02 ↔ 2023-07-23 same-geometry pair, with chip previews, detector reasoning, and a human `same` / `different` / `ambiguous` label per pair.  40 pairs were scored (1 unfilled, 2 ambiguous excluded).  Scoring via `day0/scratch/handlabel_evaluate.py`:

| matcher | precision | recall | F1 |
|---|---:|---:|---:|
| DirectSpatialMatcher(gate_m=50)            | 0.700 | 0.875 | 0.778 |
| SignatureMatcher(gate_m=50, sig_gate=0.8)  | 0.857 | 0.750 | **0.800** |

V1 delivered the precision lift the ADR-0019 design intended (+0.157 over Direct).  But it gave back recall (-0.125), and the F1 advantage is marginal (+0.022) on this label set.  V1 is a real improvement on the precision axis and a real regression on the recall axis — not an across-the-board win.

The recall cost concentrates on a single failure mode.  Three of the four false rejects (pair_01, pair_21, pair_26 — see `day0/scratch/v1_false_reject_analysis.md`) share a consistent pattern: the 07-02 and 07-23 observations refer to the same physical feature per human label, but the VLM annotated each acquisition with substantially different bbox scope.  Bbox width/height differ by 5–15×; aspect-ratio log2 swings sign in two of the three pairs; confidence flips between 0.6 and 0.85 in the same direction the bbox grows.  The detector reasoning text reads consistently across these three pairs: the VLM described one acquisition as a "compact bright return" and the other as "elongated cluster" / "ship hull and superstructure" of the same physical site.

The diagnosis is structural.  V1's signature is built entirely from the VLM's annotation output, so any axis on which the VLM's annotation choices vary across acquisitions is an axis on which V1's signature distance is inflated for same-feature pairs.  No `sig_gate` setting can keep V1's precision gain while admitting these cases — the signal V1 uses to discriminate features is the same signal that gets defeated by VLM bbox-scope variance.

ADR-0019 framed signature enrichment as a downstream candidate ("ADR-TBD on signature enrichment strategy").  The V1 evaluation produces the empirical trigger that ADR called out.  ADR-0020 is the resulting refinement — it does not supersede ADR-0019, which committed correctly to the feature-signature direction.  V1 ships as the current `SignatureMatcher` implementation for precision-favoring use cases; V2 is the next iteration addressing V1's known failure mode.

## Problem statement

V1 signature features (bbox dimensions + classification confidence) carry useful discrimination *when VLM bbox scoping is consistent across scenes*.  Empirically, VLM bbox scoping is not consistent across scenes — the VLM makes compositional choices about what extent of a bright structure to annotate as a single detection, and those choices vary stochastically across acquisitions of the same physical feature.

V2 requires signature features that survive VLM annotation variance — preferably features derived from the scene imagery itself rather than from VLM annotation output.

## Options

Three approaches considered.

### Option A — Local intensity profile features

Sample scene imagery in a fixed-size window (e.g., 32 × 32 or 64 × 64 px) centered on each observation's centroid (lat/lon → AOI pixel via the existing `custody.detection.sar_common.latlon_to_pixel` path).  Compute intensity statistics — histogram, central moments, simple texture measures, or a normalized flat patch — as the signature vector.  Compare via L2 (or histogram distance) on the normalized representation.

**Assessment.**  Deterministic from scene data — no dependence on VLM annotation choices, by construction.  Independent of bbox scope entirely.  Cheap to compute (≪ 1 ms per observation at typical AOI scales).  Pairs naturally with existing spatial matching machinery and the `Matcher` protocol — only the signature distance computation changes.  Two specific limitations to acknowledge:

1. **Window size is a hyperparameter** that may need per-resolution tuning across the project's pixel-size range (0.33–0.61 m/px).  Mitigation: parameterize `window_size_m` (in meters) and convert to pixels per scene, so the same physical-extent window is sampled regardless of pixel size.
2. **Intensity profiles encode orbit-geometry-dependent appearance** — the same physical feature looks different under different look angles.  This is the cross-geometry problem ADR-0019 also flagged.  Mitigation acknowledged but not solved here: V2 targets same-geometry pairs first (where the intensity profile of a same-feature should be similar); cross-geometry validation is deferred until same-geometry V2 is proven.  In the cross-geometry case, intensity differences carry useful information for distinguishing different objects nearby — they are signal, not noise, for that discrimination.

### Option B — VLM reasoning text embeddings

Embed the per-observation `detector_reasoning` text via a sentence embedding model (OpenAI text-embedding-3-small, local sentence-transformers, etc.).  Signature distance is cosine distance on the resulting vectors.

**Assessment.**  Captures semantic content of what the VLM described, which can be more stable than the bbox geometry it chose.  But: adds an embedding model dependency (API cost for the OpenAI path; ~100 MB model footprint for the local path).  The reasoning text itself is VLM-variable — Claude can describe the same feature with phrasing that varies stochastically across acquisitions, in the same way bbox scope varies.  Embedding distance does not always track the semantic distinction we care about: two reasoning strings can be embedded close together while describing different objects ("small vessel at pier" vs "large barge at pier") if the surrounding context dominates the embedding.

### Option C — Hybrid (intensity profile + reasoning embedding)

V2 signature combines both: local intensity features **and** reasoning embedding distance.  Match eligibility requires passing gates on both axes.

**Assessment.**  Strongest discrimination available from this option set.  But: more gates to calibrate, higher implementation cost, and premature without first establishing whether either A or B alone is sufficient.

## Decision

Commit to **Option A — local intensity profile features** as the V2 signature direction.

Justification:

1. Addresses the specific failure mode identified in V1's hand-labeled evaluation.  Intensity features sampled in a fixed window are independent of VLM annotation choices by construction, so the bbox-scope variance that defeats V1 cannot defeat V2.
2. Deterministic from scene data.  No additional model dependencies, no API costs, no per-call network round-trips.
3. Cheap to compute and matches the project's "local prototype first" preference.
4. Option B (reasoning embeddings) remains a viable future addition.  The `Matcher` protocol accommodates signature enrichment without architectural change — V3 candidate.
5. Option C (hybrid) is the long-term target but premature without validation of either component alone.

### V2 signature specification

This ADR commits to the feature category; specific parameters are left to implementation.

- Sample scene imagery in a fixed-physical-extent window centered on each observation's `(lat, lon)` centroid.  Project to pixel coordinates via `custody.detection.sar_common.latlon_to_pixel` against the scene's AOI transform.
- Feature vector derived from window pixel intensities.  Specific representation (histogram, central moments, flat patch, or some combination) chosen at implementation, justified against the same hand-labeled ground truth.
- Window size parameterized in meters (e.g. `window_size_m=10.0`) and converted to pixels per scene from `Scene.pixel_size_m`, so the same physical extent is sampled regardless of pixel size.  Initial target ~10 m at Tennent 0.33 m/px (~30 × 30 px); rescales to ~20 px at the 0.52 m/px scenes.
- Distance metric: L2 on the normalized feature vector, with per-feature normalization appropriate to the chosen representation (e.g. histogram bins divided by window pixel count; moments scaled to typical Umbra GEC dynamic range).

### V2 matching rule

Same dual-gate structure as V1: spatial gate **AND** signature gate must both pass; Hungarian assignment cost is spatial distance among eligible pairs.  Only the signature distance computation changes.  Drop-in replacement for callers using `SignatureMatcher`.

### What this doesn't commit to

- Specific window size in meters, histogram bin count, or other representation details (implementation time, justified empirically).
- Whether V2 supersedes V1 in the default matcher or ships alongside it (decide after V2 validates against ground truth).
- Reasoning embedding signatures (deferred; potential V3).
- Cross-sensor / cross-geometry validation (deferred; V2 targets same-geometry pairs first).

## Validation target

V2 success criteria, evaluated against the hand-labeled Tennent 2023-07-02 ↔ 2023-07-23 ground truth:

- Match V1's precision (≥ 0.85).
- Improve recall toward the Direct baseline (0.875) — target ≥ 0.80.
- F1 improves over V1's 0.800.

Pre-V2 implementation, expand the ground-truth dataset to include at least one cross-geometry hand-labeled pair (e.g., 07-02 ↔ 08-07) so V2's cross-geometry behavior is evaluable against ground truth, not Direct-as-truth.

## Consequences

### What this unlocks

1. A path to a matcher that handles VLM bbox-framing variance structurally rather than via threshold tuning.
2. A framework for scene-imagery-derived features that naturally extends to future Matcher implementations (wavelet features, texture measures, local CFAR statistics).
3. The ground-truth dataset expands as part of V2's validation work — cross-geometry pair labeling becomes a tracked artifact.

### What this defers

1. Reasoning embedding signatures (V3).
2. Hybrid intensity + embedding matching (post-V2).
3. Cross-sensor validation (V2 tests on same-sensor pairs first; cross-sensor deferred until V2 is proven on same-sensor).

### Ground truth promotion

The hand-labeled dataset built for V1 evaluation is first-class reference data, not scratch.  This ADR commits to promoting it as part of V2's preparation work:

- `day0/scratch/handlabel_sheet.md` → `tests/fixtures/ground_truth/tennent_0702_0723.md`
- `day0/scratch/handlabel_sheet.csv` → `tests/fixtures/ground_truth/tennent_0702_0723.csv`
- `day0/scratch/handlabel_chips/` → `tests/fixtures/ground_truth/chips/tennent_0702_0723/`
- `day0/scratch/handlabel_prepare.py` → `scripts/ground_truth_prepare.py`
- `day0/scratch/handlabel_evaluate.py` → `scripts/ground_truth_evaluate.py`

Promotion happens in a separate commit after this ADR lands.

## Downstream ADR candidates

- **ADR-TBD on V3 reasoning embedding signatures** — if V2 plateaus on a class of pairs that intensity features cannot distinguish.
- **ADR-TBD on hybrid matcher combining intensity + reasoning** — post-V2, after either component alone is validated.
- **ADR-TBD on matcher selection policy** — precision-favoring vs recall-favoring matcher per downstream use case (e.g., V1 stays as the precision-favoring choice, V2 as the balanced choice, Direct as the recall-favoring choice).

## Status

Proposed.  Will flip to `accepted` after V2 implementation validates against the promoted ground-truth dataset per the criteria above.
