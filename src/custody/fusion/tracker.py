"""Hungarian-assignment multi-target tracker wrapping the EKF TrackState.

Flow in :meth:`Tracker.step`:

1. Predict all active tracks forward by ``dt = timestamp - last_timestamp``.
2. Build a cost matrix ``C[i, j] = Mahalanobis²(track_i, obs_j)`` using each
   track's covariance and each observation's measurement covariance (via the
   observation-type-appropriate ``H`` matrix).  Entries outside the
   ``gate_sigma`` gate are set to a large sentinel so scipy's
   linear_sum_assignment avoids them but doesn't choke on ``inf``.
3. Apply Hungarian assignment (``scipy.optimize.linear_sum_assignment``).
   Assignments whose cost is above the gate are discarded.
4. For each assigned pair: run :meth:`TrackState.update`.
5. For each unassigned observation: spawn a TENTATIVE track.
6. Advance lifecycle: N-of-M confirmation via a sliding window of the last
   ``M`` step() hits/misses; COASTED after ``coast_threshold`` consecutive
   misses; RETIRED (moved to the retired list) after ``max_coast_steps``
   consecutive misses.

N-of-M confirmation semantics: a track confirms the first step on which its
hit_history deque (capacity ``M``) contains at least ``N`` True entries.
Once confirmed, status is never demoted back to TENTATIVE; later demotion
only happens via the coasting/retirement path.
"""
from __future__ import annotations

import time as _time_mod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment
from ulid import ULID

from custody.fusion.geo import to_tangent_plane
from custody.fusion.observations import (
    Observation,
    PositionObservation,
    PositionVelocityObservation,
)
from custody.models import (
    TrackState,
    _DEFAULT_VEL_SIGMA_MPS,
    _DEFAULT_POS_SIGMA_M,
)


_BIG_COST = 1.0e18
_MAX_COAST_SENTINEL = 999_999


class TrackStatus(Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    COASTED = "coasted"
    RETIRED = "retired"


# ---------------------------------------------------------------------------
# Per-track record (state + metadata)
# ---------------------------------------------------------------------------


@dataclass
class TrackRecord:
    track_id: str
    state: TrackState
    status: TrackStatus
    age_steps: int = 0
    time_last_updated: float = 0.0
    observation_history: list[str] = field(default_factory=list)
    hit_history: deque = field(default_factory=deque)
    consecutive_misses: int = 0


# ---------------------------------------------------------------------------
# Step report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StepReport:
    observations_assigned: int
    observations_unassigned_spawned: int
    tracks_confirmed: int
    tracks_coasted: int
    tracks_retired: int
    step_seconds: float


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


class Tracker:
    """Hungarian assignment + EKF-backed multi-target tracker."""

    def __init__(
        self,
        n_of_m: tuple[int, int] = (3, 5),
        gate_sigma: float = 3.0,
        coast_threshold: int = 5,
        max_coast_steps: int = 20,
        initial_track_cov: Optional[np.ndarray] = None,
    ) -> None:
        if coast_threshold > max_coast_steps:
            raise ValueError(
                "coast_threshold must be <= max_coast_steps "
                f"(got coast_threshold={coast_threshold}, max_coast_steps={max_coast_steps})"
            )
        n, m = n_of_m
        if not (0 < n <= m):
            raise ValueError(f"n_of_m must satisfy 0 < N <= M (got {n_of_m})")
        self._n = n
        self._m = m
        self._gate_sigma_sq = float(gate_sigma) ** 2
        self._coast_threshold = coast_threshold
        self._max_coast_steps = max_coast_steps
        self._initial_track_cov = initial_track_cov
        self._active: list[TrackRecord] = []
        self._retired: list[TrackRecord] = []
        self._last_timestamp: Optional[float] = None

    # -- Public accessors ---------------------------------------------------

    @property
    def active_tracks(self) -> list[TrackRecord]:
        return list(self._active)

    @property
    def retired_tracks(self) -> list[TrackRecord]:
        return list(self._retired)

    # -- Main driver --------------------------------------------------------

    def step(
        self, observations: list[Observation], timestamp: float
    ) -> StepReport:
        """Advance the tracker one step: predict, associate, update, spawn, age."""
        t_start = _time_mod.perf_counter()

        # 1. Predict forward by elapsed dt.
        if self._last_timestamp is not None:
            dt = max(0.0, float(timestamp) - float(self._last_timestamp))
            if dt > 0.0:
                for tr in self._active:
                    tr.state.predict(dt_seconds=dt)

        # 2-4. Associate assigned pairs; leave unassigned observations for spawn.
        assigned_pairs, unassigned_obs_idx = self._associate(observations)
        for track_idx, obs_idx in assigned_pairs:
            tr = self._active[track_idx]
            obs = observations[obs_idx]
            tr.state.update(obs)
            tr.observation_history.append(obs.obs_id)
            tr.time_last_updated = timestamp
            tr.consecutive_misses = 0

        # Mark per-track hits vs misses for N-of-M.
        hit_track_indices = {ti for ti, _oi in assigned_pairs}
        for i, tr in enumerate(self._active):
            hit = i in hit_track_indices
            tr.hit_history.append(hit)
            while len(tr.hit_history) > self._m:
                tr.hit_history.popleft()
            if not hit:
                tr.consecutive_misses += 1
            tr.age_steps += 1

        # 5. Spawn new TENTATIVE tracks from unassigned observations.
        spawn_count = 0
        for oi in unassigned_obs_idx:
            obs = observations[oi]
            new_track = self._spawn_track(obs, timestamp)
            self._active.append(new_track)
            spawn_count += 1

        # 6. Advance lifecycle status and partition retired tracks.
        confirmed_count, coasted_count, retired_count = self._advance_lifecycle()

        self._last_timestamp = float(timestamp)
        step_seconds = _time_mod.perf_counter() - t_start
        return StepReport(
            observations_assigned=len(assigned_pairs),
            observations_unassigned_spawned=spawn_count,
            tracks_confirmed=confirmed_count,
            tracks_coasted=coasted_count,
            tracks_retired=retired_count,
            step_seconds=step_seconds,
        )

    # -- Association --------------------------------------------------------

    def _associate(
        self, observations: list[Observation]
    ) -> tuple[list[tuple[int, int]], list[int]]:
        """Return (assigned_pairs, unassigned_obs_indices).

        Cost matrix uses Mahalanobis² (track-predicted-mean vs observation
        measurement).  Entries beyond the gate are set to ``_BIG_COST`` to
        guide Hungarian around them; any chosen pair with cost above the
        gate is discarded post-hoc.
        """
        if not observations:
            return [], []
        if not self._active:
            return [], list(range(len(observations)))

        n_tracks = len(self._active)
        n_obs = len(observations)
        cost = np.full((n_tracks, n_obs), _BIG_COST, dtype=float)
        for i, tr in enumerate(self._active):
            if tr.state.mean is None:
                continue
            for j, obs in enumerate(observations):
                mdist_sq = self._mahalanobis_sq(tr.state, obs)
                if mdist_sq <= self._gate_sigma_sq:
                    cost[i, j] = mdist_sq

        row_idx, col_idx = linear_sum_assignment(cost)
        pairs: list[tuple[int, int]] = []
        used_obs: set[int] = set()
        for i, j in zip(row_idx, col_idx):
            if cost[i, j] < _BIG_COST:
                pairs.append((int(i), int(j)))
                used_obs.add(int(j))
        unassigned = [j for j in range(n_obs) if j not in used_obs]
        return pairs, unassigned

    @staticmethod
    def _mahalanobis_sq(state: TrackState, obs: Observation) -> float:
        """Squared Mahalanobis distance of `obs` relative to `state`'s prediction."""
        mean = state.mean
        assert mean is not None  # caller filters mean=None
        if isinstance(obs, PositionObservation):
            H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
            x_m, y_m = to_tangent_plane(obs.lat, obs.lon)
            z = np.array([x_m, y_m])
            R = np.asarray(obs.cov_pos, dtype=float)
        elif isinstance(obs, PositionVelocityObservation):
            H = np.eye(4)
            x_m, y_m = to_tangent_plane(obs.lat, obs.lon)
            z = np.array([x_m, y_m, obs.v_n, obs.v_e])
            R = np.asarray(obs.cov, dtype=float)
        else:
            raise TypeError(f"Unknown Observation variant: {type(obs).__name__}")
        innovation = z - H @ mean
        S = H @ state.cov @ H.T + R
        try:
            return float(innovation @ np.linalg.solve(S, innovation))
        except np.linalg.LinAlgError:
            return _BIG_COST

    # -- Spawning -----------------------------------------------------------

    def _spawn_track(self, obs: Observation, timestamp: float) -> TrackRecord:
        """Create a TENTATIVE TrackRecord from a first observation."""
        if isinstance(obs, PositionObservation):
            cov = self._make_initial_cov_from_position(obs)
            state = TrackState(lat=obs.lat, lon=obs.lon, cov=cov)
        elif isinstance(obs, PositionVelocityObservation):
            state = TrackState(
                lat=obs.lat, lon=obs.lon,
                v_n=obs.v_n, v_e=obs.v_e,
                cov=np.asarray(obs.cov, dtype=float).copy(),
            )
        else:
            raise TypeError(f"Unknown Observation variant: {type(obs).__name__}")
        return TrackRecord(
            track_id=str(ULID()),
            state=state,
            status=TrackStatus.TENTATIVE,
            age_steps=1,
            time_last_updated=float(timestamp),
            observation_history=[obs.obs_id],
            hit_history=deque([True], maxlen=None),
            consecutive_misses=0,
        )

    def _make_initial_cov_from_position(
        self, obs: PositionObservation
    ) -> np.ndarray:
        """Build a 4×4 initial covariance from a position-only observation.

        Position block from ``obs.cov_pos`` (honest to what the sensor saw),
        velocity block from the ADR-0007 default prior (σ_v ≈ 0.83 m/s) unless
        the Tracker was constructed with an explicit ``initial_track_cov``.
        """
        if self._initial_track_cov is not None:
            return np.asarray(self._initial_track_cov, dtype=float).copy()
        cov = np.zeros((4, 4))
        cov[:2, :2] = np.asarray(obs.cov_pos, dtype=float)
        cov[2:, 2:] = (_DEFAULT_VEL_SIGMA_MPS ** 2) * np.eye(2)
        return cov

    # -- Lifecycle ----------------------------------------------------------

    def _advance_lifecycle(self) -> tuple[int, int, int]:
        """Update per-track status after association. Returns counts for StepReport."""
        confirmed_count = 0
        coasted_count = 0
        retired_count = 0

        still_active: list[TrackRecord] = []
        for tr in self._active:
            # Confirmation via N-of-M sliding window (one-way).
            if tr.status == TrackStatus.TENTATIVE:
                if sum(tr.hit_history) >= self._n:
                    tr.status = TrackStatus.CONFIRMED
                    confirmed_count += 1

            # Retirement: too many consecutive misses.
            if tr.consecutive_misses >= self._max_coast_steps:
                tr.status = TrackStatus.RETIRED
                self._retired.append(tr)
                retired_count += 1
                continue

            # COASTED after coast_threshold consecutive misses — but only
            # downgrade CONFIRMED (not TENTATIVE, which should keep trying to
            # confirm until it retires from sheer neglect).
            if (
                tr.consecutive_misses >= self._coast_threshold
                and tr.status == TrackStatus.CONFIRMED
            ):
                tr.status = TrackStatus.COASTED
                coasted_count += 1

            # A coasted track that just got a hit this step has
            # consecutive_misses == 0 — promote back to CONFIRMED.
            if tr.status == TrackStatus.COASTED and tr.consecutive_misses == 0:
                tr.status = TrackStatus.CONFIRMED

            still_active.append(tr)

        self._active = still_active
        return confirmed_count, coasted_count, retired_count
