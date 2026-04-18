from __future__ import annotations

import math
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, NamedTuple, Optional

import numpy as np


class BehaviorState(Enum):
    """Inferred operational state of a vessel."""
    TRANSIT = "transit"
    APPROACH = "approach"
    LOITER = "loiter"
    EGRESS = "egress"
    IDLE = "idle"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Zone:
    """A named geographic bounding box used for zone-based detection.

    Attributes:
        name:    Stable identifier (e.g. "ZONE_ALPHA").  Used in metadata,
                 alert context, and display labels.
        min_lat: Southern boundary (degrees).
        max_lat: Northern boundary (degrees).
        min_lon: Western boundary (degrees).
        max_lon: Eastern boundary (degrees).
        halo:    Distance in degrees outside the boundary that earns a
                 partial anomaly score.  ~0.1° ≈ 11 km at equatorial latitudes.
    """
    name: str
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float
    halo: float


@dataclass(frozen=True)
class DetectorResult:
    """Output of a single anomaly detector.

    Attributes:
        name:     Detector identifier (e.g. "sensitive_zone").
        score:    Numeric contribution to the composite anomaly score.
        evidence: Human-readable explanation of why this score was assigned.
        metadata: Optional structured data for future extension (reasoning
                  chain, sub-scores, thresholds used, etc.). Defaults to
                  empty dict; callers should treat it as read-only.
    """
    name: str
    score: float
    evidence: str
    metadata: dict[str, Any] = field(default_factory=dict)


class HistoryEntry(NamedTuple):
    lat: float
    lon: float
    timestamp: datetime
    speed_kmh: float
    heading_deg: float


@dataclass
class Vessel:
    id: str
    lat: float
    lon: float
    speed_kmh: float
    heading_deg: float
    last_seen: datetime
    history: list[HistoryEntry] = field(default_factory=list)

    def step(self, hours: float) -> "Vessel":
        """Advance position along current heading by ``hours`` at ``speed_kmh``.

        Appends the pre-step position to history. Same kinematics as the
        retired ``custody.tracks.update_position`` free function.
        """
        self.history.append(
            HistoryEntry(self.lat, self.lon, self.last_seen, self.speed_kmh, self.heading_deg)
        )
        heading_rad = math.radians(self.heading_deg)
        # Compass convention: 0° north, 90° east, clockwise.
        self.lat += math.cos(heading_rad) * self.speed_kmh * hours * 0.01
        self.lon += math.sin(heading_rad) * self.speed_kmh * hours * 0.01
        self.last_seen += timedelta(hours=hours)
        return self


# ---------------------------------------------------------------------------
# TrackState — EKF belief-state + custody metadata (ADR-0005)
# ---------------------------------------------------------------------------


# Process-noise defaults tuned per ADR-0007 to approximate the Phase 2 linear
# heuristic σ(t) = 5 + 3t km across the 0-48 h operational range.  The linear
# σ growth comes from the F-matrix coupling of the initial velocity variance
# (σ²_vel × t² in position) plus q_pos × t; q_vel is held at 0 so velocity
# variance does not compound across discrete predict steps.
#
# Anchor check at defaults:
#   σ(1 h)  ≈  8.00 km    (target 8)
#   σ(24 h) ≈ 76.97 km    (target 77)
# All portfolio health thresholds (DEGRADING=20, STALE=50, LOST=90, MAX=150)
# fire within ±2% of the Phase 2 timing.
_DEFAULT_Q_POS_PER_SEC = 8333.0   # m²/s per position-diagonal element
_DEFAULT_Q_VEL_PER_SEC = 0.0      # (m/s)²/s — see note above

# Default initial covariance: 5 km isotropic position; 0.833 m/s (3 km/h)
# isotropic velocity to drive the linear σ growth via F-coupling.
_DEFAULT_POS_SIGMA_M = 5_000.0
_DEFAULT_VEL_SIGMA_MPS = 0.833


def _default_cov() -> np.ndarray:
    cov = np.zeros((4, 4))
    cov[:2, :2] = (_DEFAULT_POS_SIGMA_M ** 2) * np.eye(2)
    cov[2:, 2:] = (_DEFAULT_VEL_SIGMA_MPS ** 2) * np.eye(2)
    return cov


def _cov_from_uncertainty_km(r_km: float) -> np.ndarray:
    cov = _default_cov()
    r_m_squared = (r_km * 1_000.0) ** 2
    cov[:2, :2] = r_m_squared * np.eye(2)
    return cov


class TrackState:
    """EKF belief-state + custody metadata (ADR-0005, ADR-0008, ADR-0010).

    State mean (when known): ``[x_east_m, y_north_m, v_n_mps, v_e_mps]`` in the
    AEQD tangent-plane frame anchored at AOI center (ADR-0009, ADR-0010).
    ``lat`` and ``lon`` (degrees) are derived properties.

    Covariance: 4×4 in SI units.  Position block m², velocity block (m/s)²,
    cross-terms m·m/s.  The F matrix couples position and velocity via the
    physical axis alignment: ``F[0, 3] = dt`` (east position += dt × east
    velocity) and ``F[1, 2] = dt`` (north position += dt × north velocity).

    Scalar-radius backward compat:
      - ``uncertainty_km`` is a read/write property; the setter projects
        a scalar km to an isotropic 2×2 position block.
      - ``position_sigma_km`` is the v3-aligned identical-value alias.
      - The constructor still accepts ``uncertainty_km=N`` for the same
        projection at init time.

    Dark-vessel fields (all None/False when AIS is active):
      is_dark, dark_since, last_known_lat/lon, last_known_time, last_known_anomaly.
    """

    def __init__(
        self,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        v_n: float = 0.0,
        v_e: float = 0.0,
        cov: Optional[np.ndarray] = None,
        uncertainty_km: Optional[float] = None,
        last_collection_time: Optional[datetime] = None,
        last_collection_anomaly_score: float = 0.0,
        consecutive_failures: int = 0,
        last_failure_time: Optional[datetime] = None,
        is_dark: bool = False,
        dark_since: Optional[datetime] = None,
        last_known_lat: Optional[float] = None,
        last_known_lon: Optional[float] = None,
        last_known_time: Optional[datetime] = None,
        last_known_anomaly: float = 0.0,
    ) -> None:
        if cov is not None:
            self.cov = np.asarray(cov, dtype=float).copy()
        elif uncertainty_km is not None:
            self.cov = _cov_from_uncertainty_km(uncertainty_km)
        else:
            self.cov = _default_cov()

        if lat is not None and lon is not None:
            # Lazy import to avoid circular-import risk: fusion.geo reads config.
            from custody.fusion.geo import to_tangent_plane
            x_m, y_m = to_tangent_plane(lat, lon)
            self.mean: Optional[np.ndarray] = np.array(
                [x_m, y_m, v_n, v_e], dtype=float
            )
        else:
            self.mean = None

        self.last_collection_time = last_collection_time
        self.last_collection_anomaly_score = last_collection_anomaly_score
        self.consecutive_failures = consecutive_failures
        self.last_failure_time = last_failure_time
        self.is_dark = is_dark
        self.dark_since = dark_since
        self.last_known_lat = last_known_lat
        self.last_known_lon = last_known_lon
        self.last_known_time = last_known_time
        self.last_known_anomaly = last_known_anomaly

    # -- Scalar-radius shim -------------------------------------------------

    @property
    def uncertainty_km(self) -> float:
        """1σ circular-equivalent position radius in km, read from covariance."""
        pos = self.cov[:2, :2]
        trace_m2 = pos[0, 0] + pos[1, 1]
        return math.sqrt(trace_m2 / 2.0) / 1_000.0

    @uncertainty_km.setter
    def uncertainty_km(self, r_km: float) -> None:
        """Project a scalar km to an isotropic 2×2 position block.

        Velocity block is preserved; cross-terms are zeroed to keep the
        block-diagonal structure (no pretend correlation between an
        externally-supplied scalar and the internal velocity state).
        """
        new_cov = self.cov.copy()
        r_m_squared = (r_km * 1_000.0) ** 2
        new_cov[:2, :2] = r_m_squared * np.eye(2)
        new_cov[:2, 2:] = 0.0
        new_cov[2:, :2] = 0.0
        self.cov = new_cov

    @property
    def position_sigma_km(self) -> float:
        """v3-aligned name for the covariance-derived 1σ position radius."""
        return self.uncertainty_km

    @property
    def confidence(self) -> float:
        """Custody confidence as exp(-radius_km / 50)."""
        return math.exp(-self.uncertainty_km / 50.0)

    @property
    def lat(self) -> Optional[float]:
        """Latitude (degrees) of the track mean, or None if mean is unset."""
        if self.mean is None:
            return None
        from custody.fusion.geo import from_tangent_plane
        lat_deg, _lon_deg = from_tangent_plane(
            float(self.mean[0]), float(self.mean[1])
        )
        return lat_deg

    @property
    def lon(self) -> Optional[float]:
        """Longitude (degrees) of the track mean, or None if mean is unset."""
        if self.mean is None:
            return None
        from custody.fusion.geo import from_tangent_plane
        _lat_deg, lon_deg = from_tangent_plane(
            float(self.mean[0]), float(self.mean[1])
        )
        return lon_deg

    # -- EKF step operators -------------------------------------------------

    def predict(
        self,
        dt_seconds: float,
        q_pos: float = _DEFAULT_Q_POS_PER_SEC,
        q_vel: float = _DEFAULT_Q_VEL_PER_SEC,
    ) -> None:
        """Advance the belief state by ``dt_seconds`` under the CV motion model.

        State order is ``[x_east, y_north, v_n, v_e]``.  F couples each
        position axis to the matching velocity component:
        ``F[0, 3] = dt`` (east position advances with east velocity v_e) and
        ``F[1, 2] = dt`` (north position advances with north velocity v_n).
        """
        F = np.eye(4)
        F[0, 3] = dt_seconds
        F[1, 2] = dt_seconds

        Q = np.zeros((4, 4))
        Q[0, 0] = q_pos * dt_seconds
        Q[1, 1] = q_pos * dt_seconds
        Q[2, 2] = q_vel * dt_seconds
        Q[3, 3] = q_vel * dt_seconds

        if self.mean is not None:
            self.mean = F @ self.mean

        self.cov = F @ self.cov @ F.T + Q

    def update(self, obs: "Observation") -> None:  # type: ignore[name-defined]
        """Kalman update dispatched on Observation variant (ADR-0008).

        :class:`PositionObservation` uses ``H = [[1,0,0,0],[0,1,0,0]]`` and
        ``R = obs.cov_pos``.  :class:`PositionVelocityObservation` uses
        ``H = I_4`` and ``R = obs.cov``.  Either variant projects its ``lat``
        and ``lon`` through :mod:`custody.fusion.geo` (ADR-0009) into the
        tangent-plane meters basis used by this track's state and covariance.
        """
        # Lazy import to break the models ↔ fusion.observations cycle.
        from custody.fusion.observations import (
            PositionObservation,
            PositionVelocityObservation,
        )
        from custody.fusion.geo import to_tangent_plane

        if isinstance(obs, PositionObservation):
            x_m, y_m = to_tangent_plane(obs.lat, obs.lon)
            z = np.array([x_m, y_m], dtype=float)
            H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
            R = np.asarray(obs.cov_pos, dtype=float)
            if self.mean is None:
                self.mean = np.array([x_m, y_m, 0.0, 0.0])
                return
        elif isinstance(obs, PositionVelocityObservation):
            x_m, y_m = to_tangent_plane(obs.lat, obs.lon)
            z = np.array([x_m, y_m, obs.v_n, obs.v_e], dtype=float)
            H = np.eye(4, dtype=float)
            R = np.asarray(obs.cov, dtype=float)
            if self.mean is None:
                self.mean = z.copy()
                return
        else:
            raise TypeError(
                "TrackState.update expected an Observation "
                "(PositionObservation or PositionVelocityObservation), "
                f"got {type(obs).__name__}"
            )

        innovation = z - H @ self.mean
        S = H @ self.cov @ H.T + R
        K = self.cov @ H.T @ np.linalg.inv(S)
        self.mean = self.mean + K @ innovation

        I4 = np.eye(4)
        IKH = I4 - K @ H
        self.cov = IKH @ self.cov @ IKH.T + K @ R @ K.T

    # -- Custody metadata operations (preserved from prior TrackState) ------

    def hours_since_collection(self, now: datetime) -> Optional[float]:
        """Return hours elapsed since the last successful collection, or None."""
        if self.last_collection_time is None:
            return None
        return (now - self.last_collection_time).total_seconds() / 3600

    def record_collection(
        self,
        now: datetime,
        anomaly_score: float,
        new_uncertainty: float,
    ) -> None:
        """Update track state after a successful collection event."""
        self.uncertainty_km = new_uncertainty  # property setter projects to cov
        self.last_collection_time = now
        self.last_collection_anomaly_score = anomaly_score
        self.consecutive_failures = 0

    def record_failure(self, now: datetime) -> None:
        """Update track state after a failed collection attempt.

        Applies a diminishing uncertainty penalty and increments the
        consecutive failure counter.  The penalty shrinks with each
        additional failure and is hard-capped to prevent runaway growth.
        """
        from custody.config import (
            FAILURE_UNCERTAINTY_PENALTY_KM,
            FAILURE_UNCERTAINTY_CAP_KM,
        )
        self.consecutive_failures += 1
        self.last_failure_time = now
        penalty = FAILURE_UNCERTAINTY_PENALTY_KM / (1 + self.consecutive_failures)
        new_radius = min(self.uncertainty_km + penalty, FAILURE_UNCERTAINTY_CAP_KM)
        self.uncertainty_km = new_radius  # property setter

    def __repr__(self) -> str:
        return (
            f"TrackState(uncertainty_km={self.uncertainty_km:.3f}, "
            f"confidence={self.confidence:.3f}, "
            f"consecutive_failures={self.consecutive_failures}, "
            f"is_dark={self.is_dark})"
        )
