"""Polymorphic Observation sum type for the fusion layer (ADR-0008).

Two variants, shape matches what the sensor actually measured:

- :class:`PositionObservation` — 2×2 position covariance, for image-derived
  sensors (SAR, EO).
- :class:`PositionVelocityObservation` — 4×4 full-state covariance, for sensors
  that instrumentally measure velocity (AIS transponder broadcasts).

Both dataclasses are frozen, hash via ``obs_id``, and validate the covariance
matrix (shape, symmetry, positive-semi-definiteness) at construction.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Union

import numpy as np


# ---------------------------------------------------------------------------
# Shared validation
# ---------------------------------------------------------------------------


_SYMMETRY_ATOL = 1e-10
_PSD_ATOL = 1e-10


def _validate_covariance(
    cov: np.ndarray, expected_shape: tuple[int, int], name: str
) -> None:
    """Raise ValueError unless ``cov`` is a symmetric, PSD matrix of the expected shape."""
    arr = np.asarray(cov, dtype=float)
    if arr.shape != expected_shape:
        raise ValueError(
            f"{name}: expected shape {expected_shape}, got {arr.shape}"
        )
    if not np.allclose(arr, arr.T, atol=_SYMMETRY_ATOL):
        raise ValueError(f"{name}: matrix is not symmetric")
    min_eig = float(np.linalg.eigvalsh(arr).min())
    if min_eig < -_PSD_ATOL:
        raise ValueError(
            f"{name}: matrix is not positive semi-definite "
            f"(min eigenvalue = {min_eig:.3e})"
        )


def _validate_obs_id(obs_id: str) -> None:
    if not isinstance(obs_id, str) or not obs_id:
        raise ValueError("obs_id must be a non-empty string")


def _validate_timestamps(acquisition_time: float, ingestion_time: float) -> None:
    if not (acquisition_time > 0):
        raise ValueError(
            f"acquisition_time must be > 0 (got {acquisition_time})"
        )
    if not (ingestion_time > 0):
        raise ValueError(
            f"ingestion_time must be > 0 (got {ingestion_time})"
        )


# ---------------------------------------------------------------------------
# PositionObservation — image-derived sensors (SAR, EO)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class PositionObservation:
    obs_id: str
    source_id: str
    modality: Literal["SAR", "EO", "AIS"]
    acquisition_time: float  # UTC epoch seconds
    ingestion_time: float
    lat: float
    lon: float
    cov_pos: np.ndarray      # 2×2 meters² in local tangent plane
    raw_ref: str             # provenance URI
    detector_version: str | None = None
    classification_conf: float | None = None
    vessel_length_est_m: float | None = None
    heading_est_deg: float | None = None
    notes: dict[str, Any] = field(default_factory=dict)
    detector_reasoning: str | None = None

    def __post_init__(self) -> None:
        _validate_obs_id(self.obs_id)
        _validate_timestamps(self.acquisition_time, self.ingestion_time)
        _validate_covariance(self.cov_pos, (2, 2), name="cov_pos")

    def __hash__(self) -> int:
        return hash(self.obs_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PositionObservation):
            return NotImplemented
        return self.obs_id == other.obs_id


# ---------------------------------------------------------------------------
# PositionVelocityObservation — sensors broadcasting velocity (AIS)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class PositionVelocityObservation:
    obs_id: str
    source_id: str
    modality: Literal["AIS"]
    acquisition_time: float
    ingestion_time: float
    lat: float
    lon: float
    v_n: float               # m/s, north component
    v_e: float               # m/s, east component
    cov: np.ndarray          # 4×4 in (m², m², (m/s)², (m/s)²) basis
    raw_ref: str
    mmsi: int | None = None
    vessel_name: str | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_obs_id(self.obs_id)
        _validate_timestamps(self.acquisition_time, self.ingestion_time)
        if not math.isfinite(self.v_n):
            raise ValueError(f"v_n must be finite (got {self.v_n})")
        if not math.isfinite(self.v_e):
            raise ValueError(f"v_e must be finite (got {self.v_e})")
        _validate_covariance(self.cov, (4, 4), name="cov")

    def __hash__(self) -> int:
        return hash(self.obs_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PositionVelocityObservation):
            return NotImplemented
        return self.obs_id == other.obs_id


# ---------------------------------------------------------------------------
# Union alias
# ---------------------------------------------------------------------------


Observation = Union[PositionObservation, PositionVelocityObservation]
