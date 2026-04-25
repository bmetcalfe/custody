# Artifact Evidence Bridge

*Companion to [docs/positioning.md](positioning.md). Documents the artifact-to-decision-layer bridge added in [ADR-0021](decisions/0021-custody-as-uncertainty-to-tasking-engine.md) Slice 19.*

---

## What the artifact bridge is

The artifact bridge converts existing artifact-style outputs - scene metadata, SAR/VLM detection summaries, matcher persistence outputs, AIS/GFW presence summaries, scene-quality flags, and manual analyst labels - into `HypothesisEvidence` and runs the existing decision pipeline (state update, custody-health assessment, collection-value ranking, decision packet).

It exists because **detection and decision-making are two different jobs**. Detector pipelines produce artifacts on their own cadence; the decision layer should be able to consume those artifacts after the fact, without re-running detection or pulling in detector dependencies.

## What the artifact bridge is not

- **Not real-time data ingestion.** The bridge reads small JSON manifests, not raw imagery, parquet shards, or telemetry streams.
- **Not a VLM run.** No model is invoked, no prompt is dispatched, no GPU is touched.
- **Not a matcher run.** The matcher V2 design (ADR-0019 / ADR-0020) remains paused. Matcher persistence artifacts are consumed as-is.
- **Not Sentinel-1 / Sentinel-2 wiring.** No SDK is imported, no scene catalog is queried.
- **Not a production data pipeline.** The fixture manifests are committed to the repository as small JSON files, not pulled from object storage.

## Supported artifact kinds

Each artifact carries a `kind` label (free-form string; the [`ArtifactKind`](../src/custody/hypotheses/artifacts.py) enum lists the canonical values). The kind is recorded for audit, but **mapping to evidence is driven by payload contents**, not the kind label:

| Kind                     | Typical source                                       | Mapping path                                       |
| ------------------------ | ---------------------------------------------------- | -------------------------------------------------- |
| `scene_signal`           | Per-scene observable (persistent scatter, vessel count) | Scenario signal path                              |
| `sar_detection_summary`  | Aggregated SAR detector output                       | Scenario signal path or explicit semantic         |
| `vlm_summary`            | VLM-derived qualitative signals                      | Scenario signal path or explicit semantic         |
| `matcher_persistence`    | Matcher displacement / cluster outputs               | Scenario signal path                               |
| `ais_presence_summary`   | GFW / AIS coverage summary                           | Explicit semantic (analyst-shaped supports list)  |
| `quality_flag`           | Scene-quality only, no semantic content              | No evidence (caveat emitted)                       |
| `manual_label`           | Analyst-supplied supports / contradicts              | Explicit semantic                                  |
| `unknown`                | Anything not covered above                           | No evidence (caveat emitted)                       |

## Two mapping paths

### Path 1 - Explicit semantic

If the artifact's `payload` contains both `supports` (list of hypothesis IDs) and `contradicts` (list of hypothesis IDs), the bridge routes the artifact through `evidence.from_mapping()`. Hypothesis IDs are validated against the scenario registry; unknown IDs raise `ValueError`. A `reason` field, the artifact's `notes`, or a default fallback feeds the evidence reason string.

This is the path for analyst labels and any source that has already done the work of deciding *what the artifact means*.

### Path 2 - Scenario signal

If the artifact's `payload` contains recognised observable fields, the bridge constructs a lightweight synthetic `Scene` or `Match` and dispatches to the existing scenario evidence generators in `custody.hypotheses.scenarios`:

| Scenario | Field                              | Generator                             |
| -------- | ---------------------------------- | ------------------------------------- |
| Tennent  | `persistent_scatterer_detected`    | `tennent_evidence_from_scene`         |
| Tennent  | `change_signal_strength`           | `tennent_evidence_from_scene`         |
| Tennent  | `displacement_m`                   | `tennent_evidence_from_match`         |
| Whitsun  | `vessel_count` + `ais_coverage_flag` | `whitsun_evidence_from_scene`       |
| Whitsun  | `matched_cluster`                  | `whitsun_evidence_from_match`         |

The same heuristic table that drives the synthetic decision packet drives the artifact-derived decision packet. There is no second source of scenario semantics.

### No-match - explicit no-evidence

If neither path matches, the bridge produces an empty tuple of evidence and the bundle's caveats record which artifacts produced nothing. This is intentional: the bridge does not infer semantics from a `kind` label alone.

## Example manifest

```json
[
  {
    "artifact_id": "tennent-scene-20230702",
    "scenario_id": "tennent",
    "kind": "scene_signal",
    "source_ref": "tennent_20230702_umbra-01",
    "timestamp": "2023-07-02T14:00:55+00:00",
    "quality_flag": "yellow",
    "confidence": null,
    "payload": {
      "persistent_scatterer_detected": true,
      "change_signal_strength": 0.0
    },
    "notes": "first observation"
  },
  {
    "artifact_id": "tennent-label-reef",
    "scenario_id": "tennent",
    "kind": "manual_label",
    "source_ref": "analyst:tennent-reef-label",
    "timestamp": "2023-08-13T12:00:00+00:00",
    "quality_flag": null,
    "confidence": 0.8,
    "payload": {
      "supports": ["fixed_reclamation_or_structure"],
      "contradicts": ["no_meaningful_activity"],
      "reason": "analyst label"
    },
    "notes": "explicit analyst label"
  }
]
```

## Why artifacts are candidate evidence, not ground truth

Detector outputs - SAR / VLM detections, matcher persistence, AIS presence summaries - have known failure modes:

- VLM detections include hallucinations and false positives.
- SAR / CFAR detections are sensitive to clutter and scene-quality.
- AIS / GFW coverage is uneven; absence is informative only when coverage is solid.
- Manual analyst labels reflect one analyst's interpretation, not consensus.

The hypothesis layer accepts all of these as **candidate evidence**, weights them by quality, and surfaces remaining ambiguity through custody health. Treating any single artifact as ground truth would defeat the purpose of the layer.

## How this differs from real-data ingestion

| Concern                          | Artifact bridge (Slice 19)                  | Real-data ingestion (roadmap)            |
| -------------------------------- | ------------------------------------------- | ---------------------------------------- |
| Input format                     | Small JSON manifests, committed             | Imagery, parquet shards, STAC catalogs   |
| External dependencies            | None - stdlib + numpy                       | Sentinel SDK, Umbra SDK, GFW client      |
| Detector execution               | None                                        | VLM, CFAR, matcher                        |
| Wall-clock IO                    | None                                        | Object-storage reads, network calls       |
| Determinism                      | Byte-identical reruns                       | Best-effort                               |
| Scope                            | Decision-layer wiring                       | Full ingest pipeline                     |

## Future work

The roadmap items in [README.md](../README.md) list real-data integration ahead of the artifact bridge. Concretely:

- A real-data metadata bridge that consumes scene-metadata records produced by an ingestion pipeline (rather than committed fixture manifests).
- STAC catalog integration for Sentinel / Umbra scene discovery.
- Processed artifact discovery that scans an output directory for SAR / VLM / matcher outputs and assembles manifests automatically.
- A runtime caching layer so artifact manifests can be regenerated cheaply from a stable upstream artifact store.

None of these are in scope today. The current bridge is intentionally limited to small committed JSON manifests so that the decision layer's behaviour stays auditable and reproducible.

## Related: scene-availability bridge (Slice 20)

The artifact bridge (Slice 19) converts existing outputs into `HypothesisEvidence` - it updates hypothesis state. The scene-availability bridge (Slice 20) uses provider-neutral collection metadata to assess whether candidate collect types are feasible - it adjusts recommendation scores without changing hypothesis state. See [`docs/scene_availability_bridge.md`](scene_availability_bridge.md).
