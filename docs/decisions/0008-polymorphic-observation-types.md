---
id: 0008
title: Observation types split by what the sensor actually measures (polymorphic, not unified 4×4)
date: 2026-04-18
status: accepted
---

## Context

v3 §2.3 specifies an `Observation` dataclass carrying a covariance matrix alongside a position fix, to feed the EKF-based tracker's update step. The original spec said "2×2 covariance" without addressing sensors that measure velocity directly.

The question is: what shape is `Observation.cov`?

**Option A — 2×2 position covariance only; velocity fields carried separately.**
AIS course-over-ground and speed-over-ground live as standalone fields with their own variance scalars. Fusion code dispatches on source type to decide whether to use them.

**Option B — 4×4 full-state covariance for all observations.**
Position-only sensors (SAR, EO) fill the velocity block with `np.inf` on the diagonal, meaning "infinite uncertainty = no information." Kalman math cancels the infinities via zero Kalman gain, so the update works, but the matrix contains values that misrepresent what the sensor measured.

**Option C — Polymorphic `Observation` sum type: shape matches what the sensor measured.**
Two dataclass variants. `PositionObservation` carries a 2×2 position covariance. `PositionVelocityObservation` carries a 4×4 covariance consistent with the track state basis `[lat, lon, v_n, v_e]`. The EKF's update method dispatches on the variant and builds the appropriate measurement matrix H.

Sensor inventory for the Custody demo:

| Sensor | What it actually measures | Natural Observation type |
|---|---|---|
| Umbra SAR (GEC, SICD) | Position only (image-derived centroid) | `PositionObservation` |
| Sentinel-1 SAR (GRD) | Position only | `PositionObservation` |
| Sentinel-2 EO | Position only | `PositionObservation` |
| Global Fishing Watch AIS | Position + course-over-ground + speed-over-ground (broadcast directly by the vessel's AIS transponder) | `PositionVelocityObservation` |

Potential future sensors (Doppler radar, along-track SAR interferometry, track-before-detect pseudo-observations) also split naturally into these two categories.

## Decision

Implement Option C. `Observation` is a `Union[PositionObservation, PositionVelocityObservation]`, not a single dataclass with a variable-shape covariance field.

```python
@dataclass(frozen=True)
class PositionObservation:
    obs_id: str
    source_id: str
    modality: Literal["SAR", "EO"]
    acquisition_time: float       # UTC epoch seconds
    ingestion_time: float
    lat: float
    lon: float
    cov_pos: np.ndarray           # 2×2 meters² in local tangent plane
    raw_ref: str                  # provenance URI
    detector_version: str | None
    classification_conf: float | None
    vessel_length_est_m: float | None
    heading_est_deg: float | None
    notes: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class PositionVelocityObservation:
    obs_id: str
    source_id: str
    modality: Literal["AIS"]
    acquisition_time: float
    ingestion_time: float
    lat: float
    lon: float
    v_n: float                    # m/s, north component
    v_e: float                    # m/s, east component
    cov: np.ndarray               # 4×4 in (m², m², (m/s)², (m/s)²) basis
    mmsi: int | None
    vessel_name: str | None
    raw_ref: str
    notes: dict[str, Any] = field(default_factory=dict)

Observation = Union[PositionObservation, PositionVelocityObservation]
```

The EKF's `TrackState.update(obs)` method dispatches:

```python
def update(self, obs: Observation) -> None:
    if isinstance(obs, PositionObservation):
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])
        R = obs.cov_pos
        z = project_latlon_to_tangent_plane(obs.lat, obs.lon, anchor=self.anchor)
    elif isinstance(obs, PositionVelocityObservation):
        H = np.eye(4)
        R = obs.cov
        z = np.array([
            *project_latlon_to_tangent_plane(obs.lat, obs.lon, anchor=self.anchor),
            obs.v_n,
            obs.v_e,
        ])
    else:
        raise TypeError(f"Unknown observation type: {type(obs)}")
    # Standard Kalman update: innovation, gain, Joseph-form covariance update
```

The v3 implementation guide §2.3 is amended to reflect the polymorphic design.

## Consequences

- **Honest semantics.** The `Observation` type carries exactly what the sensor measured. Position-only sensors don't encode "said nothing" as "said with infinite uncertainty," avoiding a category of fiction that invites downstream bugs.
- **Clear sensor-addition path.** Adding a new sensor means picking which variant applies (or introducing a third variant if some genuinely new measurement shape arrives). No decision about how to encode missing information; the type system enforces correctness.
- **Uniform update math via `H`.** The measurement matrix `H` is a function of the observation type, which is already how the Kalman filter textbook formulates it. `H` is `[[1,0,0,0],[0,1,0,0]]` for position-only, `I_4` for full-state. No infinities, no special-case arithmetic.
- **Tipcue info-gain computation (Week 5) gets a cleaner input.** The expected posterior covariance under a candidate collect depends on what dimensions the sensor measures. With distinct observation types, the info-gain math is explicit: SAR candidates reduce position uncertainty; AIS candidates reduce position and velocity uncertainty. The formula `I(c) = ½ log |Σ_prior| − ½ log |Σ_posterior(c)|` is computed against the appropriate H per candidate.
- **Minor complexity cost.** `TrackState.update` dispatches on type. This is ~5 lines of `isinstance` dispatch. Tests cover both branches.
- **Serialization.** When observations are persisted to Parquet, the two variants become two tables (or one table with a `type` discriminator column and conditional columns). Preference: two Parquet files (`observations_position.parquet`, `observations_posvel.parquet`) keyed by `obs_id`, since the schemas genuinely differ. A unified read API joins them back into the `Observation` union at load time.
- **Rejected alternatives.**
  - *Option A (scalar velocity fields alongside 2×2 position cov)*: creates two parallel uncertainty representations. Fusion code has to know which sensor produced which fields. Accretes per-sensor special cases. Rejected.
  - *Option B (4×4 everywhere with infinities)*: mathematically viable but encodes absence as infinity, which is a fiction. Kalman gain math cancels the infinities in practice, but the matrix contents no longer represent what the sensor measured. Rejected as dishonest representation.
- **Future work — third variant.** If we later ingest sensors that measure position + single-component velocity (e.g., radar Doppler range-rate), a third observation variant is natural. Not implemented in V1.

## Notes on Numpy + frozen dataclasses

`np.ndarray` is not hashable. Frozen dataclasses auto-generate `__hash__`, which will fail on instances. Mitigations: set `eq=False` on the dataclass, or override `__hash__` to return a hash of `obs_id`. Decide at implementation time; the `obs_id` approach is cleaner since observations are uniquely identified by ID regardless of covariance content.
