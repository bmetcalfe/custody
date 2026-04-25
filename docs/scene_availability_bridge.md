# Scene-Availability Metadata Bridge

*Companion to [docs/positioning.md](positioning.md) and [docs/artifact_bridge.md](artifact_bridge.md). Documents the scene-availability metadata bridge added in [ADR-0021](decisions/0021-custody-as-uncertainty-to-tasking-engine.md) Slice 20.*

---

## What the scene-availability bridge is

The scene-availability bridge adjusts candidate-collect recommendations using **provider-neutral** scene/collection availability metadata. For each candidate collect type the existing collection-value ranker recommends (`repeat_sar`, `cross_geometry_sar`, `higher_resolution_sar`, `optical_context`, `ais_coverage_query`, `wait_or_monitor`), the bridge reads the metadata catalog and decides whether the candidate is **feasible**, **partial**, **unavailable**, or **unknown**, then multiplies the value-based score by a feasibility multiplier to produce an availability-adjusted ranking.

The bridge does not download imagery, does not call any external API, does not process pixels, does not run VLM or matcher, and does not issue execution authorizations. It reasons over small JSON metadata catalogs that are committed to the repository as fixtures.

## How it differs from the artifact bridge (Slice 19)

| Concern                       | Artifact bridge (Slice 19)                                 | Scene-availability bridge (Slice 20)                       |
| ----------------------------- | ---------------------------------------------------------- | ---------------------------------------------------------- |
| Input                         | Existing detector/matcher/AIS-summary outputs              | Provider-neutral collection metadata                       |
| Effect on hypothesis state    | Updates `HypothesisState` via `HypothesisEvidence`         | None                                                       |
| Effect on collection ranking  | Indirect (changes belief, which feeds the ranker)          | Direct (multiplicative feasibility adjustment)             |
| Output type                   | `ArtifactDecisionBundle` with full decision packet         | `AvailabilityAdjustedRecommendation`                       |
| Question answered             | "What do we believe given these existing outputs?"         | "Which of the recommended collect types is feasible?"      |

Both bridges read small committed JSON fixtures, both are pure stdlib, both are deterministic, and neither talks to an external service.

## Why metadata-only comes before imagery

Metadata is available before a collect is executed. Reasoning over metadata first keeps the prototype honest about what the decision layer can and cannot say with the artifacts on hand. A candidate collect that the value ranker scores highly but for which no metadata is available should be ranked lower than a candidate of similar value with strong metadata support — the bridge captures exactly that.

## Supported metadata fields

| Field             | Type                | Used by                                             |
| ----------------- | ------------------- | --------------------------------------------------- |
| `record_id`       | non-empty string    | Identification, traceability                        |
| `scenario_id`     | "tennent" / "whitsun" | Catalog scoping                                  |
| `sensor_type`     | "sar" / "optical" / "ais" / "rf" / "unknown" | Candidate matching                |
| `source_ref`      | provider-neutral string | Audit                                          |
| `collection_time` | ISO 8601 timestamp  | Audit                                               |
| `quality_flag`    | "green" / "yellow" / "red" / null | Quality gate                          |
| `coverage_score`  | float 0.0-1.0       | `repeat_sar`, `optical_context`, `ais_coverage_query` |
| `cloud_cover`     | float 0.0-1.0       | `optical_context`                                   |
| `resolution_m`    | float > 0           | `higher_resolution_sar`                             |
| `look_geometry`   | "ascending" / "descending" / "cross" / free-form / null | `cross_geometry_sar`        |
| `latency_hours`   | float >= 0          | Audit                                               |
| `payload`         | dict                | Free-form provider extras                           |
| `notes`           | string / null       | Human notes                                         |

## Feasibility rules (deterministic)

| Candidate                | FEASIBLE                                                 | PARTIAL                                              | UNAVAILABLE                |
| ------------------------ | -------------------------------------------------------- | ---------------------------------------------------- | -------------------------- |
| `optical_context`        | optical record exists, cloud_cover <= 0.35 (or null), coverage >= 0.6 (or null), quality != red | optical exists but cloud or coverage weak | no optical records         |
| `repeat_sar`             | SAR record with coverage >= 0.5 (or null) and quality != red | SAR exists but coverage/quality weak             | no SAR records             |
| `cross_geometry_sar`     | two SAR records with distinct look_geometry, OR a "cross" record | SAR exists but geometries unknown / single   | no SAR records             |
| `higher_resolution_sar`  | SAR record with resolution_m <= 1.0 and coverage acceptable | SAR exists but resolution unknown / above 1 m  | no SAR records             |
| `ais_coverage_query`     | AIS record with coverage >= 0.6 OR quality == "green"    | AIS exists but coverage and quality both weak       | no AIS records             |
| `wait_or_monitor`        | always feasible (low score, no immediate external collect required) | (n/a)                                            | (n/a)                      |

Feasibility scores: FEASIBLE 0.70-0.85, PARTIAL 0.35-0.40, UNAVAILABLE 0.0, `wait_or_monitor` 0.25.

`adjusted_score = clamp01(original_score * feasibility_score)`. Adjusted candidates are sorted by `(-adjusted_score, candidate_id)`.

## Example catalog entry

```json
{
  "record_id": "tennent-sar-20230702-asc",
  "scenario_id": "tennent",
  "sensor_type": "sar",
  "source_ref": "public_sar_metadata:tennent-20230702-asc",
  "collection_time": "2023-07-02T14:00:55+00:00",
  "quality_flag": "yellow",
  "coverage_score": 0.75,
  "cloud_cover": null,
  "resolution_m": 0.5,
  "look_geometry": "ascending",
  "latency_hours": 6.0,
  "payload": {},
  "notes": "SAR ascending pass, sub-meter resolution"
}
```

## What is NOT claimed

- No external API fetch (no Sentinel SDK, no STAC client, no GFW client, no commercial provider client).
- No imagery processing.
- No execution authorization, no platform schedule, no specific platform claim.
- The `source_ref` strings are intentionally provider-neutral - "public_sar_metadata", "public_optical_metadata", "ais_metadata", "commercial_sar_metadata" - they identify the *kind* of metadata source, not a specific platform or provider account.
- Feasibility is decision support. It does not prove a candidate can be executed; it indicates that metadata supports the candidate being considered next.

## Future work

The roadmap items in [`README.md`](../README.md) cover what would have to happen for this bridge to leave the laptop:

- STAC catalog integration to populate the metadata catalog from real data sources.
- Public scene-availability metadata queries (Sentinel public catalog, USGS, etc.) under explicit credentials and with caching.
- Commercial provider availability queries under explicit account scope.
- Geometry / footprint intersection so feasibility considers whether the metadata's footprint covers the AOI.
- Temporal availability windows (next-pass scheduling, latency budgeting) so the adjusted ranking can prefer near-term metadata.

None of those are in scope today. The bridge is intentionally a small, deterministic, stdlib-only adjustment layer over the existing decision pipeline.

## Optimizer integration (Slice 21)

The availability-adjusted optimizer ([Slice 21](decisions/0021-custody-as-uncertainty-to-tasking-engine.md), `src/custody/hypotheses/availability_optimizer.py`) composes the base optimized plan (Slice 11) with feasibility assessments from this bridge. Candidates with `UNAVAILABLE` status are excluded by default; partial and feasible candidates have their planning utility scaled by feasibility score. Both exhaustive and greedy baselines are produced; for small candidate pools (<= 10) the recommended plan uses the exhaustive strategy. The CLI is `scripts/25_availability_optimized_plan.py`.
