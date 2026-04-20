# Investigation: SAR Ship Detection Approaches on Umbra Open Data

*Private project documentation. Potential LinkedIn / whitepaper source material.*

## Summary

Between Week 1 and Week 2 of Custody development, the project evaluated five distinct approaches to SAR ship detection on Umbra Open Data Program imagery. Four approaches failed for various reasons — some surface-level, some fundamental. The fifth approach (Vision-Language Model candidate generation with CFAR refinement) emerged as the working path forward.

This document captures what was tried, what failed, and why — in enough detail that a future post or whitepaper can be assembled from this material.

**TL;DR for the eventual LinkedIn post:**
> "Before you reach for YOLO or CCD on open-data commercial SAR, here's what I learned the hard way."

---

## The Problem Space

The Custody project needs to detect vessels and structures in SAR imagery across two case studies:

- **Tennent Reef (Spratly Islands, Vietnamese-controlled):** 5 Umbra scenes, June-August 2023, tracking reclamation activity. Scene is dominated by a large bright reclamation structure with vessels clustered at the working edges.
- **Whitsun Reef (Spratly Islands, Chinese-controlled):** 3 Umbra scenes, December 2023 - March 2024, tracking maritime militia flotilla behavior. Scene is open water with 20-50 distinct vessel-like targets scattered across ~5 km².

Both scenarios required a detection layer producing `PositionObservation` instances with covariance, confidence, and associated metadata for downstream fusion with AIS data.

The AOI characteristics matter:
- **X-band spotlight SAR at 0.25 m GSD** (Umbra native resolution)
- **uint8 GEC amplitude products** (32-bit SICD also available)
- **Contested Spratly Island features** — mix of structural targets, distributed vessel fleets, and pure water

---

## Approach 1: Cell-Averaging CFAR (CA-CFAR)

### What it is
The textbook entry point for SAR point-target detection. For each pixel, compute the mean of reference cells in a ring around the pixel (skipping a guard region), then detect if the pixel exceeds α × mean. Alpha is the scale factor tied to desired false-alarm rate.

### What we tried
- Implemented CA-CFAR with guard=20, reference=60 pixel windows
- Added structure-mask integration via connected-component filtering
- Converted detections to `PositionObservation` with empirical covariance based on pixel spacing

### Results on Tennent (2023-07-02 scene)
- **α=7.0 produced 62 detections clustered at the reclamation structure's edges**
- 53 of 62 detections (85%) within 200m of the structure mask — reasonable behavior
- Per-scene calibration worked for this scene type

### Results on Whitsun (2023-12-06 scene)
- **Detection cliff behavior: 412 detections at α=4.0 → 18 at α=4.5 → 0 at α≥5.5**
- At α=4.0, the 412 detections were mostly speckle false alarms scattered across empty water
- At α≥5.5, real vessels disappeared into the noise along with the speckle
- **No value of α produced a clean detection-on-visible-vessels pattern**

### Root cause
CA-CFAR assumes homogeneous clutter with approximately exponential distribution in the power domain. Whitsun scenes have heterogeneous clutter (mixed wave texture + bathymetry + azimuth smear artifacts from nearby targets). The clutter mixture isn't exponential. CA-CFAR's mean-based threshold produces a cliff in detection probability where real targets and noise become indistinguishable.

### Key finding
**CA-CFAR is scene-type-dependent in ways the literature doesn't emphasize strongly enough.** A single α value that works for a "closed lagoon with central structure" scene does not transfer to an "open-water flotilla" scene, even when both are maritime SAR. Per-scene calibration is mandatory, but that breaks the "algorithm does the same thing everywhere" promise that makes CFAR attractive as a reference implementation choice.

---

## Approach 2: Order-Statistic CFAR (OS-CFAR)

### What it is
Replaces CA-CFAR's mean-based threshold with a k-th percentile (typically 75th) of the reference cells. Theoretically robust to heterogeneous clutter because percentiles aren't dragged by outliers the way means are. Rohling (1983) is the canonical reference.

### What we tried
- Implemented OS-CFAR as a second detector variant following ADR-0014
- Synthetic tests confirmed OS-CFAR advantage over CA-CFAR on heterogeneous clutter fixtures
- Ran OS-CFAR alpha sweep on Whitsun scene 1

### Results
- **Synthetic tests passed.** On lognormal-contamination clutter mixtures, OS-CFAR detected targets CA-CFAR missed.
- **Real Whitsun scene showed the same cliff behavior.** OS-CFAR at various α values produced either over-detection (noise-dominated) or under-detection (missing real vessels).
- The performance gap between CA and OS wasn't large enough to overcome the scene's fundamental clutter characteristics.

### Performance dead end
At full Umbra scene scale (17,602 × 17,602 pixels with a 161² reference window), `scipy.ndimage.percentile_filter` took multi-hour to complete a single pass. O(N × W × log W) is intractable at these dimensions without custom implementation.

### Custom numba CUDA kernel attempt
- Ported the annular percentile computation to a numba CUDA kernel for RTX 5090 Blackwell
- Per-pixel sort of 24,240 annulus values × 310M pixels = ~1.1×10^14 operations
- Even with GPU parallelism, estimated multi-hour runtime
- Bailed to evaluate simpler paths before investing in Quickselect optimization

### Key finding
**The "better algorithm" path hit physics before it hit software limits.** OS-CFAR's theoretical advantage was real on synthetic data. On real Whitsun data, the scene's fundamental clutter characteristics meant neither detector could cleanly separate signal from noise. We were tuning algorithms to solve a data-distribution problem they weren't designed for.

---

## Approach 3: Pre-trained YOLO on SAR

### What it is
Take a YOLO ship detection model pre-trained on public SAR datasets (HRSID, SSDD), run it on our scenes. The hope: ML models handle clutter variance better than hand-tuned CFAR because they learn clutter-invariant features during training.

### Candidates evaluated

| Model | Training set | License | Integration |
|-------|--------------|---------|-------------|
| Akashkalasagond YOLOv8n | HRSID | Unclear | via Ultralytics (AGPL-3.0 runtime) |
| ioEclipse YOLOv11m | SSDD | MIT weights | via Ultralytics (AGPL-3.0 runtime) |
| MultimediaTechLab/YOLO | — (for fine-tuning) | MIT | Clean base, no pretrained SAR weights |

### License wrinkle
The Ultralytics runtime library (standard `from ultralytics import YOLO` import path) is AGPL-3.0. An MIT-licensed project importing AGPL code inherits AGPL obligations on distribution. We briefly considered fine-tuning from a non-AGPL base (MultimediaTechLab/YOLO) to avoid this, then simplified by making the repository private — which sidesteps the distribution question entirely.

### Results

**Tennent scene center tile (pier region):**
- Akashkalasagond YOLOv8n produced 1 detection at 0.42 confidence
- Detection landed correctly on a vessel-near-pier
- Promising initial result

**Whitsun scene vessel-dense tile (the actual test):**
- Akashkalasagond YOLOv8n: **0 detections at conf≥0.1, max 0.029 at conf≥0.001**
- ioEclipse YOLOv11m: **0 detections at conf≥0.1, max 0.203 at conf≥0.001**
- Both checkpoints produced scattered, low-confidence predictions with no spatial correlation to visible vessels

### Visual evidence
At permissive confidence thresholds, the detection clouds scattered across empty water with no localization on actual targets. Preview images showed dozens to hundreds of bounding boxes of wildly different scales overlapping in regions where a human clearly sees 2-3 vessels.

### Root cause: domain gap
- **Training data resolution:** HRSID is 0.5-3m GSD, SSDD is 1-15m GSD
- **Our data resolution:** Umbra spotlight at 0.25m GSD
- **Scale mismatch:** Ships appear 3-12× larger in our pixels than in training pixels
- **Frequency mismatch:** HRSID mixes C-band (Sentinel-1) and X-band (TerraSAR-X) training data; Umbra is all X-band spotlight
- **Radiometric mismatch:** JPEG-compressed training imagery vs our uncompressed uint8 GeoTIFF
- **Scale of features YOLO learned isn't the scale our vessels present at**

### Key finding
**Pre-trained SAR YOLO checkpoints don't transfer to high-resolution commercial spotlight data.** The implicit assumption behind "we'll use a pre-trained model" is that someone else trained on data similar enough to yours. For X-band commercial spotlight, no such pre-trained model exists in the open-source ecosystem. Either you fine-tune on your own labeled data, or you use a different approach.

The checkpoint-hunting process burned ~3 hours of real time before this became conclusive.

---

## Approach 4: Coherent Change Detection (CCD)

### What it is
Compute phase coherence between two SAR acquisitions of the same area. Regions that didn't change between acquisitions have high coherence; regions that changed (construction, moving vessels, water surface) have low coherence. CCD is phase-domain rather than amplitude-domain detection, and theoretically should sidestep the domain-gap and scene-variance problems that killed Approaches 1-3.

### Why it looked promising
- Umbra's Open Data Program explicitly makes SICD (complex, phase-preserved) data available since August 2023
- Umbra curates "Time-Series Locations" specifically for repeat-pass analysis
- Published research (ESA Φ-lab, Rollo's InSAR tutorial) shows successful CCD on Umbra data at other locations (notably Sāqand, Iran)
- CCD produces a genuinely different signal (change detection) that could be the "novel contribution" of the project

### What we tried

**SICD ingestion:** Passed. sarpy reads Umbra SICDs cleanly, extracts 32-bit complex data with proper phase preservation. All 8 of our scenes have SICDs alongside GEC products.

**Pair selection for Tennent:**
- Checked all 10 possible pairings across our 5 Tennent scenes
- **Only viable pair: 2023-07-02 ↔ 2023-07-23 (both UMBRA-05, 21-day baseline, 0.34° dAzim, 4.45° dGraze)**
- All other pairs had azimuth differences >80° (different orbital positions) or grazing differences >20°

**Pair selection for Whitsun:**
- 54-second pair (same pass): 73° azimuth flip between collections — beam re-steered mid-pass. Not CCD-compatible.

**Registration attempts on Tennent 07-02 / 07-23 pair:**
- sarpy's built-in `register_arrays` has library bugs: typo on line 231, inverted shape contract on line 233
- Pivoted to OpenCV ECC with affine motion model on amplitude images
- Coarse registration locked at cc=0.799; fine registration at cc=0.595 (moderate)
- Single affine fit was inadequate across the full chip — amplitude differences at corners ranged from 18% to 46%

**Polynomial warp upgrade:**
- Extracted 369 tie points via phase correlation on 20×20 grid
- **Phase correlation peaks only marginally above noise floor** (median 0.033 vs noise floor 0.018)
- Polynomial fit RMS residual: 8+ pixels (ties disagree among themselves)
- Degree-2 and degree-3 fits produced near-identical coherence results

**Coherence computation:**
- Boxcar coherence at W=5 and W=11 windows
- Mean coherence across full chip: 0.20 (affine), 0.21 (polynomial degree 3)
- **Mean coherence matches the sample-estimator bias for zero true coherence** (E[γ̂] ≈ √(π/4N) ≈ 0.177 for W=5)
- Coherence does not correlate with amplitude — bright structures and water show indistinguishable coherence

**Targeted analysis:**
- Even at the brightest point scatterers (where coherence should theoretically be highest), only 1.73% of bright-in-both pixels show γ > 0.5
- 70× amplitude mismatch at the "brightest-in-both" pixel — the warp didn't align the dominant scatterer correctly

### Root cause: physical decorrelation past critical baseline

At 4.45° grazing angle difference over 713 km slant range:
- Perpendicular baseline ≈ 55 km
- Critical baseline for X-band at this geometry: ~250-500m
- **We're ~100× past critical baseline**

Physical meaning: each resolution cell samples different portions of the 3D scatterer response in each acquisition. The phase information from the two acquisitions is sampling different physical populations of scatterers within each cell. This is unrecoverable by any registration technique.

The 21-day temporal baseline in a tropical marine environment with active reclamation adds additional decorrelation.

### Key finding
**Umbra Open Data Program acquisitions are not CCD-curated for our AOIs.** The Sāqand tutorial pair works because Umbra specifically chose nearly-identical orbits for that tutorial location. General-purpose ODP acquisitions (including all 5 of our Tennent scenes and all 3 of our Whitsun scenes) have geometric differences that exceed X-band critical baseline.

This is a publishable finding in its own right: researchers planning to use Umbra ODP data for CCD should verify geometric compatibility (dAzim, dGraze) before investing in processing pipelines. For many AOIs the ODP inventory does not contain CCD-compatible pairs.

Commercial Umbra tasking can deliver CCD-compatible pairs but requires paid requests and adds license complications. Sentinel-1 SLC data has guaranteed repeat geometry via the IW mode and is a realistic alternative at lower resolution (10m vs 0.25m).

### Engineering cost
Full CCD spike including SICD inventory, geometric compatibility analysis, coregistration attempts, polynomial warp, and coherence diagnostics: approximately 4 hours of focused work. Produced definitive negative result with clear physical explanation.

---

## Approach 5: Vision-Language Model Detection (Working Approach)

### What it is
Use a frontier vision-language model (Claude via API) to identify vessels in SAR imagery by prompting it directly. VLMs leverage broader visual priors from diverse training data, so they handle out-of-distribution imagery better than narrow detectors.

### Rationale for trying this
Observation: a human can clearly identify vessels in Whitsun scene imagery. Claude (the VLM) correctly identified and counted vessels when shown the preview images during project planning. The discriminative features *are* in the pixels; narrow pre-trained detectors couldn't access them because their training distributions didn't include Umbra-like data.

VLMs draw on training covering satellite imagery, aerial photos, maritime datasets, and general object recognition. "Things that look like ships" is a much broader representation than "things labeled as ships in HRSID."

### What we tried
- Extracted Whitsun tile 1 (same tile YOLO failed on): 640×640 pixels, contrast-stretched
- Sent to Claude Sonnet 4.6 via Anthropic API with three different prompts:
  - **Direct:** "Identify all vessels, provide bounding boxes as JSON"
  - **Contextualized:** Added SAR physics explanation (azimuth smearing, wake signatures, scattering characteristics)
  - **Reasoning-first:** Asked for image description first, then localization

### Results

| Prompt | Vessels detected | Wall time | Tokens in/out |
|--------|------------------|-----------|---------------|
| Direct | 3 | 7.8s | 649 / 346 |
| Contextualized | 3 | 12.4s | 712 / 699 |
| Reasoning-first | 1 | 9.9s | 654 / 440 |

All three prompts:
- Identified the bright vessel cluster YOLO missed completely
- Produced SAR-aware reasoning (identified azimuth smearing, wake signatures, superstructure characteristics)
- Converged on the same spatial region (image center-left area)

Cost: ~$0.03 for the three API calls combined. Full-scene processing at 755 tiles per scene: ~$8/scene.

### Known limitations

**Loose bounding boxes:** The VLM produces cluster-scale boxes (~100×30-70m) rather than per-hull boxes (expected 20-40m). This is a real constraint — the VLM sees "region containing vessels" rather than "individual vessel positions."

**Ambiguous count:** One prompt said 1 vessel, others said 2-3. This partially reflects real ambiguity in the image (a raft-up of close vessels looks similar to a single larger vessel in SAR) but also means the detector isn't committing to a precise count.

**Context injection:** The contextualized prompt invoked world knowledge ("Whitsun Reef is known for Chinese maritime militia"). This bias is useful for operational awareness but may be unwanted in some applications. Worth noting for any downstream evaluation.

**Not deterministic:** Different prompt phrasings produce different results on the same image. Standard for LLM behavior but a real constraint for production pipelines that need reproducible outputs.

### Why this is the working approach
- Detects vessels where YOLO fails (domain-gap solved via broader training)
- Produces reasoning traces that support the "legible tip-and-cue" architectural goal
- Cost is affordable at project scale (~$64 for all 8 scenes)
- Can be combined with CFAR refinement inside VLM-identified regions for tighter localization

### Architecture: VLM candidate generator + CFAR refinement
The design pattern emerging from this work:
1. **VLM identifies candidate regions** with natural-language reasoning attached
2. **CFAR (or similar) refines localization** within candidate regions — the clutter distribution inside a small candidate region is much more uniform than across a full scene
3. **Fusion layer** integrates the refined detections with AIS data and downstream reasoning

This is a novel integration pattern. VLMs for SAR detection exists in 2024-2025 research; integrating them into explainable fusion architectures is less mapped territory.

---

## What This Taught Us

### About open-data commercial SAR
- Umbra's Open Data Program is excellent for general SAR analysis, research, and demonstration
- Pairs in the ODP inventory are not curated for CCD — geometry verification is mandatory before processing
- Commercial X-band spotlight at 0.25m is outside the training distribution of publicly-available pre-trained detectors
- Bring-your-own-labels or bring-your-own-VLM are the realistic detection paths

### About algorithm selection
- CFAR is a candidate generator, not a finished detector — clutter variance across scene types is real
- Pre-trained YOLO has sharp domain boundaries that aren't always documented
- CCD physics is sensitive — baseline compatibility matters more than temporal baseline alone
- VLMs meaningfully close the domain-gap problem at the cost of localization precision

### About project scoping
- Four failed approaches over ~48 hours surfaced real engineering constraints
- Each failure was diagnosable with specific numbers, not just vibes
- The negative results are more valuable documented than forgotten — they save the next person weeks of exploration
- A VLM + refinement hybrid emerged as the working approach not by design but by exhausting simpler paths

### Things worth mentioning in a LinkedIn post or whitepaper
1. **The CFAR scene-variance problem** — specific numbers showing where CA-CFAR's homogeneous-clutter assumption breaks
2. **The pre-trained-model resolution mismatch** — HRSID/SSDD don't cover X-band spotlight
3. **Umbra ODP geometry considerations** — specific dAzim/dGraze thresholds with explanation of critical baseline
4. **VLM emergent capability** — frontier models handling SAR domain despite no explicit SAR training
5. **The integration pattern** — VLM candidate generation + classical CFAR refinement as a practical hybrid

---

## Source Data for the Future Writeup

If writing this up as a LinkedIn post or longer blog:

**Hook:** "I spent a weekend trying to detect ships in commercial SAR imagery. Here's what didn't work, and why the thing that finally worked surprised me."

**Structure:** Chronological. Each approach gets a section. Each section has:
- What I tried
- Representative numbers from our actual runs
- Why it failed
- The single most important lesson for someone considering the same path

**Close:** What ended up working (VLM + CFAR refinement) and what it suggests about the 2024-2025 frontier of satellite image analysis.

**Tone:** Honest, technical, specific. The audience values someone who documents their failures with precision more than someone who presents only the polished result.

**Images available:**
- Whitsun YOLO failure: scattered-box preview at α=0.001
- CFAR cliff: alpha sweep comparison table
- CCD coherence: flat-distribution histogram at noise floor
- VLM success: same tile with meaningful (if loose) bounding boxes

**Metric summary that could anchor the post:**
- 5 approaches evaluated
- 4 failed (with specific failure modes)
- 1 worked (with honest caveats)
- Total: ~48 hours of weekend work
- Cost of the thing that worked: $0.03 per tile
