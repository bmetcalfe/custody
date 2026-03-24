# Custody

Anomaly-aware target custody and collection-planning prototype.

## What it does

Custody tracks one or more vessels over time, scoring each observation for
anomalous behavior and deciding whether to task a collection sensor.
It runs in two modes: a deterministic simulation of synthetic targets, and
an AIS replay mode that ingests real or test CSV tracks.
A Streamlit dashboard provides a step-by-step playback view of the full
pipeline — track state, alerts, compound signals, sensor tasking, and map.

## Quickstart

```bash
uv sync
uv run streamlit run src/app/streamlit_app.py
```

## Architecture

The pipeline is a chain of pure-function layers.  Each layer reads from
the previous layer's output and writes nothing back.

```
AIS CSV / Simulation
        │
        ▼
  Feature extraction          custody/features/
    motion_features.py        – speed, heading, route deviation
    zone_features.py          – sensitive-zone proximity score
    proximity_features.py     – nearest-vessel distance and score

        │
        ▼
  Behavior detection          custody/behavior/
    detectors.py              – per-signal detectors → anomaly_breakdown
    modes.py                  – behavior mode classification
    state_machine.py          – inferred behavior state + confidence

        │
        ▼
  Anomaly scoring             custody/behavior/detectors.py
    anomaly_score             – weighted composite of detector outputs
                                (re-exported via custody/anomalies.py)

        │
        ▼
  Orbital access              custody/orbit.py  +  custody/sensors.py
    orbit.py                  – TLE parsing, SGP4 propagation, az/el,
                                is_in_view() for a ground observer
    sensors.py                – schedule-based and orbital sensor catalog;
                                get_sensor_opportunities(t, lat, lon) returns
                                sensors that pass both schedule and orbit filter

        │
        ▼
  Collection planner          custody/planner.py
    compute_target_priority() – rank vessels by anomaly + confidence + compound
    plan_collection()         – NONE / HOLD / TASK / NO_SENSOR / PREEMPTED
                                caller supplies the pre-filtered opportunity list;
                                PREEMPTED fires when sensors existed globally but
                                were claimed by a higher-priority vessel first

        │
        ▼
  Multi-vessel arbitration    custody/simulate.py (Phase 2 + 3)
    Each timestep:
      1. fetch per-vessel sensor pools (orbit-filtered from each vessel's lat/lon)
      2. rank all vessels by compute_target_priority
      3. assign in priority order via a claimed_sensor_ids set; a vessel is
         PREEMPTED when its entire initial pool has been claimed by higher-
         priority vessels; a vessel with no accessible sensors gets NO_SENSOR

        │
        ▼
  Alert layer                 custody/alerts.py
    evaluate_alerts()         – sparse, event-style notifications
                                (fire once at a state transition, no score)
    Codes: CRITICAL_ANOMALY, HIGH_ANOMALY, SENSITIVE_ZONE_ENTRY,
           COLLECTION_FAILED, NO_SENSOR_AVAILABLE, LOW_CUSTODY,
           LOITERING_CONFIRMED, VESSEL_PROXIMITY
    Note: PREEMPTED does not trigger NO_SENSOR_AVAILABLE — sensors existed
          but were claimed by higher-priority vessels, which is an arbitration
          outcome rather than a sensor-gap.

        │
        ▼
  Compound signal layer       custody/compounds.py
    evaluate_compounds()      – scored multi-signal behavioral patterns
                                (read-only; does not affect anomaly_score)
    Codes: LOITERING_NEAR_ZONE, PROXIMITY_NEAR_ZONE,
           LOITERING_WITH_PROXIMITY, HIGH_ANOMALY_LOW_CUSTODY,
           REPEATED_ZONE_ENTRY

        │
        ▼
  Dashboard                   src/app/streamlit_app.py
    Streamlit app             – playback slider, map, alerts, compound panels
    Available Sensors section uses the selected vessel's actual lat/lon for
    per-position orbital access lookup.
```

Supporting modules:

| Module | Purpose |
|---|---|
| `custody/config.py` | All thresholds and zone geometry |
| `custody/models.py` | Shared dataclasses (TrackState, Vessel, …) |
| `custody/tracks.py` | Track position and uncertainty update |
| `custody/orbit.py` | TLE parsing, SGP4 propagation, satellite visibility |
| `custody/sensors.py` | Schedule-based and orbital sensor catalog |
| `custody/simulate.py` | Multi-vessel simulation with per-timestep arbitration |
| `custody/ais.py` | AIS CSV parsing and single-vessel replay |

## Planner action codes

| Code | Meaning |
|---|---|
| `NONE` | Tasking not warranted (low anomaly, adequate confidence) |
| `HOLD` | Task value below threshold — recent collection dampens re-task urgency |
| `TASK` | Sensor assigned; see `collection_result` for SUCCESS / FAILED |
| `NO_SENSOR` | Tasking warranted but no sensors available after orbit filtering |
| `PREEMPTED` | Tasking warranted; sensors existed but were claimed by a higher-priority vessel in the same timestep |

## Sensor access and orbit filtering

`get_sensor_opportunities(t, lat, lon)` returns two kinds of sensor:

- **Schedule-based** (`satellite_id=None`): always returned when the time-of-day
  condition is met, regardless of observer position.  Backward-compatible with
  the pre-orbit version (calling without lat/lon returns the same result).
- **Orbital** (`satellite_id != None`): returned only when the satellite is above
  `SENSOR_MIN_ELEVATION_DEG` as seen from (lat, lon) at time t.

The simulation checks orbital access **per vessel**: each vessel's sensor pool
is fetched from its own current lat/lon, so a vessel far from a satellite's
ground track will not see that sensor even if a nearby vessel can.

The dashboard **Available Sensors** panel performs the same per-position
lookup using the selected vessel's lat/lon at the selected playback step.

## `sensor_access_count` field

Each simulation record carries a `sensor_access_count` integer: the number
of sensors in *that vessel's* position-filtered pool before any arbitration.
Values can differ between vessels at the same timestep if their positions
give different orbital visibility.  This field is simulation-only; the AIS
replay path does not emit it.

## Alert vs compound signals

**Alerts** fire once, at a transition (e.g. first zone entry, first collection
failure).  They have a level (INFO / WARNING / CRITICAL) but no continuous
score.  They are appropriate for operator notifications.

**Compound signals** fire on every step where their conditions are met and
carry a normalised confidence score in [0, 1].  They combine multiple
signals — e.g. loitering *and* zone proximity simultaneously — and are
surfaced as a scored panel in the dashboard rather than as point alerts.
Compounds are a read-only interpretation layer: they never modify records
or feed back into anomaly scoring.

## Running tests

```bash
uv run pytest tests/
```

634 tests across simulation, AIS ingestion, behavior detection, anomaly
scoring, alerts, compound signals, orbital access, sensor scheduling,
planner arbitration, and dashboard data pipeline.
