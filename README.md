# Custody

**An ISR collection-management reasoning engine that screens a vessel population, selectively maintains custody on targets that matter, and allocates limited sensor resources through explicit portfolio-level tradeoffs.**

Custody models the full intelligence loop from raw observation to ranked collection recommendation: multi-source evidence fusion, mission-level decision-making with explainability, and collection orchestration against real orbital pass windows. The system tells an analyst not just *what* to do but *why*, *in what window*, and *what the alternatives are if it fails* — across a competing population of targets, not just a single vessel.

![Custody demo](docs/demo.gif)

---

## The problem it solves

In maritime surveillance, custody of a target degrades continuously. AIS can go dark. Behavioral anomalies compound. Sensors are constrained by orbital geometry, cloud cover, and competing priorities across a vessel population. A naive system answers "task this sensor." A good one answers: "the fused evidence picture shows elevated significance and unresolved uncertainty — task SAR in the next 12-minute access window at 14:49z; if that misses, fall back to optical at 16:29z with this expected information gain."

That problem becomes harder — and more important — at population scale. With many entities competing for the same sensor passes, the system must also answer: *which* targets warrant active custody, *why* those and not others, and what you are trading off when one is serviced over another.

Custody builds toward both kinds of answers.

---

## Operating model: selective custody

Custody does not attempt to maintain persistent tracking on every vessel. That approach fails at scale: limited sensors mean that trying to track everything produces shallow coverage of nothing.

Instead, the system screens the full population and selectively commits resources to the subset of targets that matter.

**How attention is managed across the population:**

The system continuously classifies each entity into one of three attention tiers:

| Tier | When | Neglect pressure |
|---|---|---|
| `BACKGROUND` | Routine traffic — no behavioral signal, no zone relevance, nominal confidence | None |
| `WATCHLIST` | Elevated interest — mild anomaly, zone proximity, or degrading custody | Reduced |
| `ACTIVE_CUSTODY` | Active tracking obligation — high anomaly, zone entry, low confidence, or operator directive | Full |

Promotion into a higher tier is driven by:
- **Anomaly signal** — behavioral scoring from the atomic and compound detection layers
- **Zone relevance** — proximity to or entry into sensitive areas
- **Custody degradation** — confidence below thresholds indicating observation gap risk
- **Compound signals** — co-occurring behavioral patterns with correlated risk

Only ACTIVE_CUSTODY and WATCHLIST entities accrue meaningful neglect pressure. A BACKGROUND vessel going unobserved for 100 hours does not crowd out a newly promoted ACTIVE_CUSTODY target. Neglect is a risk only *where custody is actually expected*.

---

## Key behaviors modeled

- **Selective custody at scale** — the system continuously screens the full vessel population and selectively commits observation resources only to entities that warrant active tracking. Background traffic accrues no neglect pressure, keeping limited sensor capacity focused where it matters.
- **Rendezvous and transfer detection** — pairwise and sequence-based: the system detects vessel proximity, tracks the full converge → dwell → separate sequence, and distinguishes a genuine rendezvous from a transient crossing.
- **Dark-vessel / AIS dropout handling** — when AIS transmission stops, the system freezes last-known position, grows positional uncertainty over time, and escalates custody health from HEALTHY through DEGRADING to STALE/LOST. Manually directed vessels maintain their attention tier even after going dark.
- **Manual tracking directives** — operators can designate a vessel for persistent ACTIVE_CUSTODY tracking regardless of anomaly score. The directive sets an attention-tier floor; the vessel still competes for scarce sensors on merit rather than consuming them unconditionally.
- **Explicit portfolio tradeoffs and preemption** — when one entity is serviced and another cannot be, the system records who was deferred, for whom, and why. Preemption is surfaced as a first-class event in the portfolio view, not a silent drop.

---

## Manual tracking override

Operators can explicitly designate a vessel for persistent tracking, independent of its current anomaly score. Typical cases:

- vessel departing a port of interest
- known entity of ongoing concern
- mission-directed surveillance requirement

The `MAINTAIN_CUSTODY` directive sets a floor on attention tier: the vessel is treated as ACTIVE_CUSTODY regardless of how benign its behavior appears. It remains in the portfolio, competes for sensors, and accrues neglect pressure if unobserved — but it does not automatically jump to the top of the priority stack. A genuinely anomalous vessel still outranks a nominal manually-tracked one.

In the default scenario, `PORT-1` illustrates this: a routine slow-transit vessel that an operator has designated for persistent tracking. Its behavior is unremarkable; its attention tier is ACTIVE_CUSTODY throughout.

---

## Architecture

Eight layers, each with a clean data contract. Each layer reads from the previous layer's output and writes nothing back.

```
  Raw Observations (AIS, simulation)
           │
           ▼
 ┌─────────────────────────────────┐
 │  Layer 1: Source Ingestion      │  custody/ais.py · custody/simulate.py
 │  Normalise inputs               │
 └─────────────────────────────────┘
           │
           ▼
 ┌─────────────────────────────────┐
 │  Layer 2: Custody State         │  custody/tracks.py · custody/models.py
 │  Evolving entity belief         │  TrackState: uncertainty_km,
 │  Position · uncertainty · time  │  custody_confidence, last_collection_time
 └─────────────────────────────────┘
           │
           ▼
 ┌─────────────────────────────────┐
 │  Layer 3: Behavior & Anomaly    │  custody/behavior/ · custody/compounds.py
 │  Atomic + compound signals      │  BehaviorSignal, CompoundSignal
 │  Loitering · zone proximity     │  scored multi-signal behavioral patterns
 │  Route deviation · AIS gap      │
 └─────────────────────────────────┘
           │
           ▼
 ┌─────────────────────────────────┐
 │  Layer 4: Fusion & Belief       │  custody/fusion.py
 │  Multi-source reconciliation    │  FusionAssessment:
 │                                 │    fused_score   = 0.45·anomaly
 │                                 │                  + 0.35·compound_conf
 │                                 │                  + 0.20·(1−custody)
 │                                 │    uncertainty   = 0.50·(1−custody)
 │                                 │                  + 0.30·(1−agreement)
 │                                 │                  + 0.20·evidence_gap
 │                                 │    missing_evidence[], recommended_source
 └─────────────────────────────────┘
           │
           ▼
 ┌─────────────────────────────────┐
 │  Layer 5: Mission Reasoning     │  custody/decision.py
 │  "So what, and what next?"      │  Decision:
 │                                 │    action ∈ {PASSIVE_MONITOR, ELEVATE,
 │                                 │              TASK_OPTICAL, TASK_SAR,
 │                                 │              ESCALATE}
 │                                 │    priority, confidence
 │                                 │    why[]              ← operator-grade
 │                                 │    next_best_actions[] ← ordered fallbacks
 └─────────────────────────────────┘
           │
           ▼
 ┌─────────────────────────────────┐
 │  Layer 6: Collection            │  custody/taskrecommendation.py
 │  Orchestration                  │  TaskRecommendation:
 │  "Which asset, when, and why?"  │    sensor, window_start, window_end
 │                                 │    expected_value = 0.40·priority
 │                                 │                   + 0.25·confidence
 │                                 │                   + 0.20·sensor_fit
 │                                 │                   + 0.15·timing_score
 │                                 │    reason, fallbacks[], rank
 └─────────────────────────────────┘
           │  (per-entity outputs × N entities)
           ▼
 ┌─────────────────────────────────┐
 │  Layer 7: Portfolio             │  custody/orchestration/
 │  Orchestration                  │  attention.py   — tier classification
 │  "Who matters most, and why?"   │  portfolio.py   — portfolio scoring and ranking
 │                                 │
 │  Screens the full population.   │  PortfolioItem:
 │  Assigns attention tiers.       │    portfolio_rank, portfolio_score
 │  Gates neglect by tier.         │    attention_state, attention_basis
 │  Surfaces preemption tradeoffs. │    custody_health, neglect_flag, neglect_hours
 │  Applies operator directives.   │    tracking_directive, deferred_for
 └─────────────────────────────────┘
           │
           ▼
 ┌─────────────────────────────────┐
 │  Layer 8: Analyst Interface     │  src/app/
 │  Overview-first dashboard       │  Portfolio overview: rank, tier, health,
 │                                 │  event feed, filters, attention map
 │                                 │  Entity detail: full reasoning chain
 └─────────────────────────────────┘
```

---

## The product scenario

The main scenario is a 36-hour, 24-entity portfolio (`PORTFOLIO_SCENARIO`): 20 background vessels with realistic archetypes (transit, patrol, approach) and four scripted actors with distinct, staggered roles:

| Entity | Role | Behavior | Attention trajectory |
|---|---|---|---|
| `BRAVO-1` | Zone loiterer | SW approach reaches ZONE_ALPHA ~h15 → loiters h16–28 → evasive egress | WATCHLIST → ACTIVE_CUSTODY during loiter |
| `ECHO-1` + `ECHO-2` | Rendezvous pair | Converge from opposite sides on a shared waypoint → dwell together ~h12–22 → separate | ACTIVE_CUSTODY during rendezvous dwell |
| `PORT-1` | Manual custody / dark vessel | Slow transit under MAINTAIN_CUSTODY directive → AIS dropout at h12 → frozen last-known position, growing uncertainty | ACTIVE_CUSTODY throughout; dark floor holds tier after dropout |
| background ×20 | Mixed archetypes | Routine transits, patrols, approaches | BACKGROUND at start; may rise as confidence decays without collection |

**Scenario narrative:** three distinct concerns emerge and overlap:
- **h0–11** — normal portfolio; PORT-1 visible and tracked; ECHO pair in transit; BRAVO-1 approaching zone
- **h12** — PORT-1 AIS dropout and ECHO rendezvous fire simultaneously → direct portfolio tradeoff
- **h12–22** — dark-vessel concern and active rendezvous dwell compete for sensor capacity; BRAVO-1 loitering inside zone from h16
- **h22+** — ECHO pair separates; PORT-1 track degrades; BRAVO-1 evasive egress from h28

What this demonstrates:
- Anomalous actors surface through zone scoring, pairwise rendezvous detection, and AIS-loss escalation
- PORT-1 stays ACTIVE_CUSTODY under operator directive even before going dark; dark floor maintains that tier after AIS loss
- Rendezvous detection is sequence-based (converge → dwell → separate) — not a one-frame proximity check
- When multiple concerns overlap in time, preemption tradeoffs are surfaced with explanation
- Neglect pressure is gated by tier — background vessels accumulate zero neglect weight

![Entity detail panel at 17:00z — TASK SAR decision with full reasoning chain](docs/screenshot.png)

---

## Analyst interface

The dashboard is **overview-first**: the default view is a portfolio triage surface, not a single-entity detail screen.

**Overview page:**
- **Ranked portfolio table** — all entities sorted by urgency, with display status, anomaly score, custody health, neglect flag, and a one-line reason. Click any row to focus the map on that entity.
- **Click-to-focus map** — geospatial view of the full portfolio; the selected entity is highlighted, centered, and labeled. Vessel dot color encodes display status (red = needs action → amber = preempted → yellow = neglected → blue = watch → gray = healthy).
- **Event feed** — compact stream of notable changes since the previous timestep: zone entries, custody health degradations, neglect triggers, and significant rank shifts.
- **Filters** — narrow the table and map by status category, neglect flag, scripted-actors-only, or top-N by urgency rank.
- **Optional orbital ground tracks** — toggle to overlay ±90 min sub-satellite paths for all sensor satellites, with current satellite positions marked.

**Entity detail panel (drill-down):**
- Full Evidence → Fusion → Decision → Task Queue reasoning chain, explainable at each step.
- What-if analysis, compound signal history, and orbital pass reference table.

---

## Why this matters

A simple dashboard shows everything and prioritizes nothing. A naive tracking system attempts to maintain custody on all entities and produces thin, unreliable coverage everywhere.

Custody takes a different position:

- **Focused attention.** The system allocates observation effort to the subset of entities that actually warrant it, not uniformly across the fleet.
- **Appropriate neglect sensitivity.** Lack of observation is treated as a risk only for entities in active custody. A background vessel going dark is not an alert; an active-custody vessel going dark is.
- **Operator intent as a first-class input.** Manual tracking directives plug into the same tier system as anomaly-driven promotion. The operator's designation is respected, but the vessel still competes for scarce resources rather than consuming them unconditionally.
- **Explicit tradeoffs.** When one entity is serviced and another is deferred, the system records who was preempted and why, rather than silently dropping the request.

---

## Sensor model

Three abstract sensor types are resolved against real orbital pass windows using SGP4 propagation:

| Abstract label | Satellites | Orbit | Passes on 2026-03-23 |
|---|---|---|---|
| `OPTICAL` | EO-MIO-1, EO-MIO-2 | 51.6° MIO ~400 km | 11:33z, 13:51z |
| `OPTICAL` | EO-SSO-1, EO-SSO-2 | 97.8° SSO ~600 km | 10:00z, 18:00z |
| `SAR` | SAR-1, SAR-2 | 97.8° SSO ~700 km | 12:21z, 13:59z (SAR-1) · 16:29z, 18:08z (SAR-2) |
| `AIS_REFRESH` | — | Immediate | Always (synthetic 30-min window) |
| `MONITOR` | — | Passive | Always (synthetic 2-hr window) |

The task planner searches all satellites of a type and returns the nearest upcoming pass. Sensors with no accessible window within the horizon are omitted from the queue; MONITOR ensures the queue is never empty.

---

## Quickstart

```bash
uv sync
uv run streamlit run src/app/streamlit_app.py
```

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/).

The dashboard opens in portfolio overview mode showing all entities ranked by urgency. Use the timeline slider to step through 36 hours; the event feed, attention tier assignments, and neglect flags update at each step. Click any row in the ranked table to focus the map on that entity, or drill into any entity for its full reasoning chain.

---

## Running tests

```bash
uv run pytest tests/
```

**1464 tests** across: orbital mechanics, SGP4 propagation, sensor scheduling, AIS ingestion, track state, behavior detection, anomaly scoring, compound signals, alert layer, collection planner, multi-vessel arbitration, decision trace, what-if analysis, fusion assessment, mission reasoning, task recommendation, portfolio orchestration, attention tier classification, and dashboard data pipeline.

---

## Key data contracts

```python
@dataclass(frozen=True)
class PortfolioItem:
    entity_id: str
    portfolio_rank: int             # 1 = highest urgency
    portfolio_score: float          # [0,1] composite urgency
    custody_health: str             # HEALTHY | DEGRADING | STALE | LOST
    neglect_flag: bool              # unobserved beyond threshold
    neglect_hours: float            # hours since last collection
    portfolio_reason: str           # one-sentence rationale
    deferred_for: Optional[str]     # entity that consumed this slot (if preempted)
    attention_state: str            # BACKGROUND | WATCHLIST | ACTIVE_CUSTODY
    tracking_directive: str         # NONE | MAINTAIN_CUSTODY
    attention_basis: str            # plain-English explanation of tier assignment

@dataclass(frozen=True)
class FusionAssessment:
    entity_id: str
    timestamp: datetime
    fused_score: float              # [0,1] composite significance
    uncertainty: float              # [0,1] evidence confidence gap
    source_agreement: float         # [0,1] cross-source consistency
    missing_evidence: list[str]     # what would reduce uncertainty
    recommended_confirming_source: Optional[str]   # "OPTICAL" | "SAR" | "AIS" | None

@dataclass(frozen=True)
class Decision:
    entity_id: str
    timestamp: datetime
    action: str                     # PASSIVE_MONITOR | ELEVATE | TASK_OPTICAL | TASK_SAR | ESCALATE
    priority: float                 # [0,1] mission-facing urgency
    confidence: float               # [0,1] confidence in the chosen action
    why: list[str]                  # operator-grade rationale bullets
    next_best_actions: list[str]    # ordered fallback actions

@dataclass(frozen=True)
class TaskRecommendation:
    task_id: str                    # e.g. "task_v001_sar_20260323t1449z"
    entity_id: str
    sensor: str                     # OPTICAL | SAR | AIS_REFRESH | MONITOR
    window_start: datetime
    window_end: datetime
    expected_value: float           # [0,1] estimated information gain
    reason: str                     # one-sentence operator justification
    fallbacks: list[str]            # alternative sensors if this task fails
    rank: int                       # 1 = highest expected value
```

---

## Repository layout

```
src/
  custody/
    ais.py              AIS CSV ingestion and single-vessel replay
    alerts.py           Sparse event-style alert layer
    anomalies.py        Composite anomaly scorer
    behavior/           Atomic behavior detectors and state machine
    compounds.py        Multi-signal compound behavioral patterns
    config.py           All thresholds, zone geometry, sensor constants
    decision.py         Mission reasoning layer → Decision
    decision_trace.py   Structured planner decision breakdown
    features/           Motion, zone, and proximity feature extractors
    fusion.py           Multi-source evidence fusion → FusionAssessment
    models.py           Shared dataclasses (TrackState, Vessel, Zone, …)
    orbit.py            TLE parsing, SGP4 propagation, satellite visibility
    planner.py          Legacy collection planner (HOLD/TASK/PREEMPTED gate)
    sensors.py          Orbital and schedule-based sensor catalog
    simulate.py         Smoke/regression entry-point (TWO_VESSEL_SMOKE fixture)
    taskrecommendation.py  Collection orchestration → ranked TaskRecommendation queue
    tracks.py           Track position update and uncertainty decay
    whatif.py           Config-variant comparison engine
    orchestration/
      attention.py      Attention tier classification and neglect-weight gating
      portfolio.py      Population-level ranking, health, neglect, tradeoff surface
    simulation/
      scenarios.py      ScenarioConfig, VesselSpec, PORTFOLIO_SCENARIO
      timeline.py       Multi-target simulation engine
      profiles.py       BehaviorProfile archetypes
      generator.py      Background vessel placement
  app/
    streamlit_app.py        Main dashboard (overview-first)
    portfolio_overview.py   Portfolio table, KPI counts, display status
    overview_filters.py     Status/tier/top-N filter helpers
    overview_events.py      Event feed detectors (zone entry, health change, neglect, rank shift)
    entity_detail_panel.py  Fusion → Decision → Task Queue panel
    compound_panels.py      Compound signal history and active views
    orbital_passes_panel.py Raw orbital pass reference table
    whatif_panel.py         What-if results display
    ground_track.py         Orbital ground-track sampling and satellite position helpers
tests/                  1464 tests, one file per module
```
