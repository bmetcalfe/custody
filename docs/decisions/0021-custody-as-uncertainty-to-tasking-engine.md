# ADR-0021: Custody as Uncertainty-to-Tasking Engine

**Status:** Proposed  
**Date:** 2026-04-23  
**Related:** ADR-0008, ADR-0011, ADR-0012, ADR-0015, ADR-0018, ADR-0019, ADR-0020  
**Scope:** Product thesis, architecture positioning, Week 1-2 implementation direction

## Context

Custody began as a decision-led GEOINT prototype aligned with the SDA concept of maintaining target custody across uncertain, heterogeneous sensing. The repo has since accumulated substantial scaffolding: polymorphic observations, scene abstraction, SAR/VLM/CFAR detection work, matcher direction, EKF/covariance tracking, portfolio orchestration, prediction/tasking modules, and a large test suite.

The project narrative has drifted toward detector tuning, especially around VLM-assisted SAR interpretation. That drift creates the wrong success criterion. A detector-first framing asks, "Can this model identify ships in SAR?" That is a narrow and crowded problem, and it turns the project into a model-tuning exercise.

The stronger thesis is that Custody maintains belief under bad data and recommends the next collection action that best reduces uncertainty.

Custody should not be judged by whether one detector is perfect. It should be judged by whether it can ingest imperfect evidence, maintain competing hypotheses, expose uncertainty, and select the next best collect.

## Decision

Custody is now explicitly framed as an **uncertainty-to-tasking engine**.

The system's central loop becomes:

```text
Observation / Scene / Match / AIS gap
    -> Hypothesis evidence annotation
    -> Hypothesis state and belief timeline
    -> Custody health
    -> Collection value / next-best collect
```

Detection remains useful, but it is no longer the product. Detection outputs are candidate evidence. VLM outputs are candidate evidence. AIS gaps are candidate evidence only when coverage context supports that interpretation. SAR scenes are evidence sources with quality and geometry constraints. Matcher results are evidence association aids.

The product layer is the reasoning layer above those inputs.

## Evidence structure decision

For the first implementation, **evidence is a thin wrapper around existing repo objects**, not a new normalized universal event type.

This preserves the current architecture and avoids a schema rewrite.

Evidence will be represented as a lightweight annotation object that references an existing source object, such as:

- `PositionObservation` or `PositionVelocityObservation` from `src/custody/fusion/observations.py`
- `Scene` from `src/custody/fusion/scenes.py`
- `TrackRecord` / `Tracker` outputs from `src/custody/fusion/tracker.py`
- SAR/VLM candidate outputs from `src/custody/detection/vlm_sar.py`
- scene quality outputs from `src/custody/detection/quality.py`
- AIS/GFW presence or absence from `src/custody/ingest/gfw_presence.py`
- future matcher outputs from the ADR-0019/0020 path

Suggested first-pass type:

```python
@dataclass(frozen=True)
class HypothesisEvidence:
    source_ref: str
    source_kind: str
    timestamp: datetime
    scenario_id: str
    supports: tuple[str, ...]
    contradicts: tuple[str, ...]
    confidence: float
    weight: float
    reason: str
```

This is intentionally not a replacement for observations or scenes. It is an interpretation layer over them.

A future normalized `EvidenceEvent` may be justified if multiple external feeds need durable interchange, but that is not Week 1 scope.

## Preserved foundation

The pivot preserves the existing technical foundation.

- ADR-0008 polymorphic observations remain the measurement substrate.
- ADR-0011 GFW presence-as-position-only remains the AIS evidence ingestion path; the AIS-coverage guardrail still applies.
- ADR-0018 scenes remain the acquisition/context abstraction for heterogeneous SAR scenes.
- ADR-0019/0020 matcher work remains a proposed direction for cross-geometry association, not the active priority.
- `TrackState` covariance remains the low-level track uncertainty model.
- `Tracker` remains the observation-to-track fusion component.
- `src/custody/orchestration/portfolio.py` remains the seed for portfolio-level prioritization.
- Existing task recommendation logic remains useful, but needs a hypothesis-disambiguation value function above it.
- VLM/SAR work remains useful as candidate evidence generation, not as the backbone of the system.

This is a repositioning and product-layer addition, not a restart.

## New components required

The missing layer is above observations and tracks.

Add a new module:

```text
src/custody/hypotheses/
  __init__.py
  types.py
  registry.py
  evidence.py
  update.py
  timeline.py
  custody_health.py
  collection_value.py
  explain.py
```

### Hypothesis sets

Hypotheses are scenario-specific (per ADR-0012 scenario reframe and ADR-0013 dual case studies). Do not use one generic global hypothesis list.

#### Tennent Reef

Tennent is not primarily a moving-vessel story. It is a persistent SAR activity / structure / reclamation ambiguity story.

Initial hypothesis set:

```text
fixed_reclamation_or_structure
construction_or_reclamation_activity
stationary_sar_scatter_or_reef_clutter
transient_vessel_activity
no_meaningful_activity
```

#### Whitsun Reef

Whitsun is better suited for vessel cluster / anchorage / AIS-dark ambiguity.

Initial hypothesis set:

```text
vessel_cluster_activity
transient_anchorage_or_fishing_presence
ais_dark_or_poorly_observed_vessels
detector_clutter_false_positives
no_persistent_activity
```

### Hypothesis state

Suggested first-pass type:

```python
@dataclass(frozen=True)
class HypothesisState:
    scenario_id: str
    timestamp: datetime
    scores: Mapping[str, float]
    supporting_evidence: Mapping[str, tuple[HypothesisEvidence, ...]]
    contradicting_evidence: Mapping[str, tuple[HypothesisEvidence, ...]]
    top_hypothesis: str
    uncertainty: float
    explanation: tuple[str, ...]
```

Belief updates should be simple, deterministic, and explainable. Weighted additive scoring is acceptable for the first version. Bayesian elegance is not required yet.

### Custody health

Custody health is the surfaced product metric. It should answer:

> How well do we currently understand this situation?

Suggested first-pass type:

```python
class CustodyHealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    AMBIGUOUS = "ambiguous"
    STALE = "stale"
    LOST = "lost"

@dataclass(frozen=True)
class HypothesisCustodyHealth:
    status: CustodyHealthStatus
    score: float
    drivers: tuple[str, ...]
    reason: str
```

Candidate drivers:

- top hypothesis confidence
- margin between top two hypotheses
- recency of useful evidence
- source diversity
- contradiction level
- scene quality
- track covariance where trackable entities exist

Important distinction: **AMBIGUOUS is not LOST**. Ambiguous means there is evidence, but competing explanations remain close. Lost means the system no longer has a useful belief state.

### Collection value

Collection value scores candidate collects by expected hypothesis disambiguation, not just sensor suitability.

Candidate collect types may include:

```text
same_geometry_sar
cross_geometry_sar
higher_resolution_sar
sentinel_1_wide_area_sar
sentinel_2_optical_context
ais_gfw_coverage_query
wait_no_collect
```

The collection-value engine may score sensors that are not yet ingested. That is intentional. The product demo can reason about the value of a collect before full ingestion exists. Real Sentinel ingestion is downstream implementation work, not a blocker for proving the engine.

The docs and CLI must make this explicit: a recommendation for Sentinel-1 or Sentinel-2 may be a proposed collect, not proof that the repo already ingests that product end to end.

## Demoted components

VLM is demoted from "detection backbone" to "candidate evidence generator / analyst assistant."

The lower success bar is:

- Can VLM output contribute useful evidence annotations?
- Can it help generate candidate detections or qualitative scene interpretation?
- Does it move or fail to move a hypothesis state?

The higher bar is no longer:

- Can VLM become the best possible SAR ship detector?
- Can prompt tuning solve the full detection problem?

Detector tuning is supporting work, not the main project objective.

## Paused work

The V2 matcher path pauses at its current state.

ADR-0020 remains proposed. The cross-geometry labeling sheet should remain untracked for now. It is cheap to regenerate and committing it would signal active recommitment to the V2 matcher path before the pivot is validated.

The V2 matcher resumes only if the hypothesis layer surfaces a specific ambiguity that cross-geometry matching would disambiguate.

Acceptable future trigger:

> Tennent hypothesis timeline shows uncertainty dominated by "persistent structure vs transient vessel activity," and cross-scene geometry-aware association is the highest-value evidence needed to resolve it.

Unacceptable future trigger:

> Improve matcher accuracy generally.

This rule prevents drift back into matcher swamp.

## Sentinel stance

Sentinel ingestion is paused as a Week 1-2 implementation dependency.

However, Sentinel-1 and Sentinel-2 may appear as candidate collection options in the collection-value engine. This is not a contradiction. The engine's job is to reason about what evidence would reduce uncertainty. It can recommend a sensor before the repo has full ingestion for that sensor.

The recommendation output must distinguish:

- implemented evidence source
- simulated candidate collect
- proposed downstream integration

This keeps the product thesis clear without pretending ingestion is complete.

## Interview-safe framing

This pivot should be described externally as:

> I added a product/reasoning layer above existing detection, fusion, and tasking work so the system can reason about sensor uncertainty and recommend next-best collection.

Do not describe it as restarting the project.

The accurate story is:

> The detector work exposed the real product need: not a better one-off model, but a system that maintains belief under imperfect data and decides what evidence to collect next.

This is especially relevant for AI transformation, GEOINT product strategy, and operational decision-support roles.

## Consequences

### Positive

- Re-centers Custody on the original custody/tasking thesis.
- Preserves existing implementation work and test investment.
- Stops VLM tuning from dominating the roadmap.
- Creates a product metric: custody health.
- Makes the demo legible to product/strategy/operations audiences.
- Produces a stronger story for Vantor, BlackSky, Pano, and Planet-style conversations.

### Negative / tradeoffs

- Hypothesis scores will initially be heuristic, not statistically rigorous.
- Collection value will initially be rule-based and partially simulated.
- Some existing detector/matcher work will feel unfinished.
- The repo will temporarily have both lower-level tracking uncertainty and higher-level hypothesis uncertainty, which need clear naming to avoid confusion.

### Mitigations

- Keep hypothesis evidence as a thin wrapper to avoid type churn.
- Make every score explainable and deterministic.
- Keep scenario-specific hypotheses narrow.
- Document whether a candidate collect is implemented, simulated, or future integration.
- Require that resumed V2 matcher work be tied to a concrete hypothesis-disambiguation test.

## Week 1-2 implementation sequence

### Week 0 / Day 0

- Add this ADR.
- Add `docs/pivot_audit_uncertainty_to_tasking.md`.
- Do not modify detector code.
- Do not commit the untracked V2 labeling sheet.

### Week 1

- Add `src/custody/hypotheses/` module.
- Implement scenario-specific hypothesis registry for Tennent and Whitsun.
- Implement thin-wrapper `HypothesisEvidence` annotations.
- Implement deterministic hypothesis update.
- Build CLI report for Tennent and Whitsun belief timeline.

### Week 2

- Add custody health scoring.
- Add hypothesis-disambiguation collection value.
- Wire recommendation output into CLI report.
- Update README / positioning language to reflect the new thesis.

## Success criteria

By the end of Week 2, the repo should produce a report similar to:

```text
Scenario: Tennent Reef
Custody health: AMBIGUOUS, 0.54
Top hypothesis: fixed_reclamation_or_structure, 0.61
Second hypothesis: construction_or_reclamation_activity, 0.48

Primary uncertainty:
Persistent structure vs active construction/reclamation remains unresolved.

Recommended collect:
Sentinel-2 optical context if cloud-free, otherwise cross-geometry SAR.

Why:
Optical context best separates fixed reef/structure from visible construction signatures.
Cross-geometry SAR helps test whether the bright scatterer pattern is geometry-dependent.
Additional VLM tuning has low expected value because current ambiguity is hypothesis ambiguity, not detector confidence.
```

That output is the first real proof of the pivot.
