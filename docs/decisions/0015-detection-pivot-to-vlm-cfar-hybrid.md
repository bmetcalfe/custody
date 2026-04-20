---
id: 0015
title: Detection pivot from CFAR to VLM-candidate-generator + CFAR-refinement hybrid
date: 2026-04-20
status: accepted
---

## Context

Week 2 of Custody development evaluated multiple SAR ship detection approaches. Detailed investigation of each approach is documented in `docs/investigations/detection-approaches.md`. This ADR captures the architectural decision emerging from that investigation.

The evaluation summary:

- **CA-CFAR** (ADR-0014 initial implementation): scene-variance failure on Whitsun. Homogeneous-clutter assumption breaks on mixed wave-texture + bathymetry scenes, producing a detection cliff where no α value separates signal from noise cleanly.
- **OS-CFAR** (ADR-0014): theoretically better on heterogeneous clutter, but the performance gap vs CA-CFAR was insufficient to overcome Whitsun's fundamental clutter characteristics. Full-scene computation was also intractable without custom implementation.
- **Pre-trained YOLO** (Akashkalasagond HRSID, ioEclipse SSDD): domain gap between training distribution (0.5-15m GSD, mixed C/X-band, JPEG) and our data (0.25m X-band spotlight, uncompressed GeoTIFF) produced max confidences of 0.03-0.20 on visible vessels where 0.8+ would indicate working detection.
- **Coherent Change Detection**: our Umbra ODP pairs exceed X-band critical baseline by ~100×. Physical decorrelation prevents coherence estimation regardless of registration quality. Documented null result.
- **Vision-Language Model (Claude via API) detection**: identifies vessels YOLO misses, produces SAR-aware reasoning, but generates cluster-scale bounding boxes rather than per-hull precision.

## Decision

Production detection uses a two-stage hybrid:

**Stage 1 — VLM candidate generation.**
Tile each scene into 640×640 pixel tiles with overlap. For each tile, send to Claude Sonnet 4.6 via Anthropic API with a contextualized prompt describing SAR physics. The VLM returns candidate regions (bounding boxes) with natural-language reasoning explaining why each region was flagged.

**Stage 2 — CFAR refinement within candidates.**
For each VLM-identified candidate region, run CA-CFAR locally with parameters tuned for the region's clutter characteristics. The refined detections produce tighter localization (per-hull scale rather than per-cluster scale) suitable for `PositionObservation` emission.

**Stage 3 — Fusion layer (unchanged).**
Refined detections flow into the existing polymorphic observation infrastructure as `PositionObservation` instances with:
- `modality = "SAR"`
- `classification_conf` from the VLM's confidence assessment
- `detector_version = "vlm_cfar_hybrid_v1"`
- Observation metadata includes the VLM's reasoning trace (natural language) linked to the detection

The VLM reasoning trace becomes part of the legible tip-and-cue story: every detection can be explained in natural language describing what the detector noticed and why.

## Configuration

`config/sar_detection_params.json` carries per-case-study parameters. Current structure:

```json
{
  "umbra": {
    "defaults": {
      "detector_variant": "vlm_cfar_hybrid",
      "vlm": {
        "model": "claude-sonnet-4-6",
        "prompt_variant": "contextualized_v1",
        "tile_size": 640,
        "tile_overlap": 64
      },
      "cfar": {
        "alpha": 7.0,
        "guard": 10,
        "reference": 30,
        "min_blob_pixels": 2,
        "max_blob_pixels": 200
      }
    },
    "case_studies": {
      "tennent": {"comment": "Reclamation structure scenes; CFAR tuned for low-contrast vessel-near-structure detection"},
      "whitsun": {"comment": "Open-water vessel flotilla; CFAR tuned for high-contrast vessel-in-water detection"}
    },
    "scenes": {
      "2023-07-02-14-00-55_UMBRA-05": {"case_study": "tennent"},
      "2023-12-06-02-06-24_UMBRA-04": {"case_study": "whitsun"}
    }
  }
}
```

The CFAR parameters are tighter than before (guard=10, reference=30 rather than 20/60) because we're refining within VLM-identified regions, not scanning full scenes. Smaller windows are appropriate when the clutter is approximately uniform (which it is within a candidate region).

## Consequences

- **Dependency on external API.** Detection now requires network access to Anthropic's API. Offline operation is not possible for the VLM stage.
- **Cost per scene.** ~$8/scene at current pricing (755 tiles × ~$0.01/tile). Budget-compatible for project scale.
- **Latency.** ~8-12 seconds per tile sequentially; parallelizable to minutes-per-scene with concurrent API calls.
- **Rate limiting.** Anthropic API has rate limits; tiled inference on a full scene may require batching or throttling.
- **Non-deterministic outputs.** VLM responses vary across identical inputs. Production pipelines requiring reproducibility need seed-pinning or response caching.
- **Reasoning traces stored with observations.** The VLM's natural-language explanation for each detection is preserved. This is valuable for the "legible tip-and-cue" story but increases observation payload size; current `PositionObservation` schema needs an optional `detector_reasoning` field added.
- **CFAR code retained as reference.** The ADR-0014 CA-CFAR and OS-CFAR implementations stay in the codebase. They're used for Stage 2 refinement and remain available as documented reference for the algorithm comparisons in the investigation document.
- **Legal and license posture.** The project is now private and will remain so through Week 14+. The Ultralytics AGPL concern that drove earlier architecture decisions is mitigated by the private-repository stance. If the repository becomes public at any point, VLM-based detection avoids the pre-trained-YOLO license inheritance entirely (VLM is accessed via API, not code import).

## What's explicitly deferred

- **Fine-tuning our own detector on hand-labeled Umbra data.** Real option if VLM limitations become blockers. Not needed for demo scope; a day of work if it becomes necessary.
- **Multi-VLM ensemble.** Using multiple VLMs (Claude + GPT-4V + local) and voting across detections would improve reliability but multiplies cost. Deferred until single-VLM limitations are characterized in actual use.
- **VLM response caching.** Production use would benefit from caching VLM responses by image-hash. Current implementation calls VLM fresh on every inference. Cache layer is future work if rate limits or costs become constraints.
- **Deterministic mode.** Running with `temperature=0` and seeded randomness to approximate reproducibility. Not critical for current demo use; worth exploring for production applications.
- **Custom prompt variants per case study.** Current design uses a single contextualized prompt. Case-study-specific prompts (Tennent emphasizes structure detection, Whitsun emphasizes flotilla patterns) may produce better results. Experimentation opportunity.

## Supersedes

Does not supersede ADR-0014 (OS-CFAR for heterogeneous-clutter scenes). ADR-0014's implementation stays in the codebase as the Stage 2 refinement option for CFAR. The detector_variant configuration allows switching between CA-CFAR and OS-CFAR for the refinement stage.

Refines the detection layer position within the broader architecture. Upstream (sensor ingestion) and downstream (fusion, tipcue) layers are unchanged.
