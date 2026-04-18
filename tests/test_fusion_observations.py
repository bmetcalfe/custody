"""
Tests for custody.fusion.observations -- polymorphic Observation sum type (ADR-0008).

PositionObservation (2x2 cov) + PositionVelocityObservation (4x4 cov) + Observation union.
"""
from __future__ import annotations

import typing
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from custody.fusion.observations import (
    Observation,
    PositionObservation,
    PositionVelocityObservation,
)


# ---------------------------------------------------------------------------
# Fixtures / constructors
# ---------------------------------------------------------------------------


def _pos_cov(sx_m: float = 100.0, sy_m: float = 100.0) -> np.ndarray:
    return np.diag([sx_m ** 2, sy_m ** 2]).astype(float)


def _posvel_cov() -> np.ndarray:
    return np.diag([100.0 ** 2, 100.0 ** 2, 1.0, 1.0]).astype(float)


def _pos_obs(obs_id: str = "obs-1", **kw) -> PositionObservation:
    base = dict(
        obs_id=obs_id,
        source_id="umbra",
        modality="SAR",
        acquisition_time=1_689_000_000.0,
        ingestion_time=1_689_000_060.0,
        lat=9.75,
        lon=116.0,
        cov_pos=_pos_cov(),
        raw_ref="s3://umbra-open-data-catalog/x/y/GEC.tif#row=0,col=0",
    )
    base.update(kw)
    return PositionObservation(**base)


def _posvel_obs(obs_id: str = "ais-1", **kw) -> PositionVelocityObservation:
    base = dict(
        obs_id=obs_id,
        source_id="gfw_ais",
        modality="AIS",
        acquisition_time=1_689_000_000.0,
        ingestion_time=1_689_000_030.0,
        lat=9.80,
        lon=116.1,
        v_n=2.5,
        v_e=-1.0,
        cov=_posvel_cov(),
        raw_ref="gfw://mmsi=123456789/ts=1689000000",
    )
    base.update(kw)
    return PositionVelocityObservation(**base)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_position_observation_constructs_with_all_fields():
    o = _pos_obs()
    assert o.obs_id == "obs-1"
    assert o.source_id == "umbra"
    assert o.modality == "SAR"
    assert o.lat == 9.75
    assert o.lon == 116.0
    assert o.cov_pos.shape == (2, 2)
    assert o.raw_ref.startswith("s3://")


def test_position_velocity_observation_constructs_with_all_fields():
    o = _posvel_obs()
    assert o.obs_id == "ais-1"
    assert o.modality == "AIS"
    assert o.v_n == 2.5
    assert o.v_e == -1.0
    assert o.cov.shape == (4, 4)


def test_position_observation_field_defaults():
    o = _pos_obs()
    assert o.notes == {}
    assert o.detector_version is None
    assert o.classification_conf is None
    assert o.vessel_length_est_m is None
    assert o.heading_est_deg is None


def test_position_velocity_observation_field_defaults():
    o = _posvel_obs()
    assert o.notes == {}
    assert o.mmsi is None
    assert o.vessel_name is None


# ---------------------------------------------------------------------------
# Covariance validation — PositionObservation
# ---------------------------------------------------------------------------


def test_pos_cov_symmetric_passes():
    cov = np.array([[4.0, 1.0], [1.0, 9.0]])
    _pos_obs(cov_pos=cov)


def test_pos_cov_asymmetric_rejected():
    cov = np.array([[4.0, 1.0], [2.0, 9.0]])  # σ_xy=1, σ_yx=2
    with pytest.raises(ValueError, match="symmetric"):
        _pos_obs(cov_pos=cov)


def test_pos_cov_three_by_three_rejected():
    cov = np.eye(3)
    with pytest.raises(ValueError, match="shape"):
        _pos_obs(cov_pos=cov)


def test_pos_cov_four_by_four_rejected():
    cov = np.eye(4)
    with pytest.raises(ValueError, match="shape"):
        _pos_obs(cov_pos=cov)


def test_pos_cov_negative_diagonal_rejected():
    cov = np.array([[-1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="positive semi-definite|positive-semi-definite|psd"):
        _pos_obs(cov_pos=cov)


def test_pos_cov_indefinite_rejected():
    # Symmetric but eigenvalues are 3 and -1 → indefinite
    cov = np.array([[1.0, 2.0], [2.0, 1.0]])
    with pytest.raises(ValueError, match="positive semi-definite|positive-semi-definite|psd"):
        _pos_obs(cov_pos=cov)


# ---------------------------------------------------------------------------
# Covariance validation — PositionVelocityObservation
# ---------------------------------------------------------------------------


def test_posvel_cov_symmetric_passes():
    cov = np.diag([1.0, 2.0, 3.0, 4.0]).astype(float)
    _posvel_obs(cov=cov)


def test_posvel_cov_two_by_two_rejected():
    with pytest.raises(ValueError, match="shape"):
        _posvel_obs(cov=np.eye(2))


def test_posvel_cov_three_by_three_rejected():
    with pytest.raises(ValueError, match="shape"):
        _posvel_obs(cov=np.eye(3))


def test_posvel_cov_asymmetric_rejected():
    cov = np.eye(4)
    cov[0, 1] = 0.5  # not mirrored on [1,0]
    with pytest.raises(ValueError, match="symmetric"):
        _posvel_obs(cov=cov)


def test_posvel_cov_indefinite_rejected():
    cov = np.eye(4)
    cov[0, 1] = 2.0
    cov[1, 0] = 2.0  # symmetric but eigenvalues of [[1,2],[2,1]] block = 3, -1
    with pytest.raises(ValueError, match="positive semi-definite|positive-semi-definite|psd"):
        _posvel_obs(cov=cov)


# ---------------------------------------------------------------------------
# Hashability
# ---------------------------------------------------------------------------


def test_pos_same_id_hash_equal_and_set_collision():
    a = _pos_obs(obs_id="abc")
    b = _pos_obs(obs_id="abc", lat=10.0, lon=117.0)
    assert hash(a) == hash(b)
    s = {a, b}
    assert len(s) == 1


def test_pos_different_ids_hash_differently():
    a = _pos_obs(obs_id="abc")
    b = _pos_obs(obs_id="def")
    assert hash(a) != hash(b)
    s = {a, b}
    assert len(s) == 2


def test_posvel_hashability_same_id_collides():
    a = _posvel_obs(obs_id="xyz")
    b = _posvel_obs(obs_id="xyz", lat=10.0, lon=117.0)
    assert hash(a) == hash(b)
    assert len({a, b}) == 1


def test_mixed_dict_both_retrievable():
    p = _pos_obs(obs_id="p1")
    v = _posvel_obs(obs_id="v1")
    d = {p: "alpha", v: "bravo"}
    assert d[p] == "alpha"
    assert d[v] == "bravo"


# ---------------------------------------------------------------------------
# Type dispatch
# ---------------------------------------------------------------------------


def test_isinstance_position_observation():
    p = _pos_obs()
    assert isinstance(p, PositionObservation)
    assert not isinstance(p, PositionVelocityObservation)


def test_isinstance_position_velocity_observation():
    v = _posvel_obs()
    assert isinstance(v, PositionVelocityObservation)
    assert not isinstance(v, PositionObservation)


def test_observation_union_covers_both_variants():
    args = typing.get_args(Observation)
    assert PositionObservation in args
    assert PositionVelocityObservation in args


# ---------------------------------------------------------------------------
# Frozen semantics
# ---------------------------------------------------------------------------


def test_position_observation_is_frozen():
    o = _pos_obs()
    with pytest.raises(FrozenInstanceError):
        o.lat = 0.0  # type: ignore[misc]


def test_position_velocity_observation_is_frozen():
    o = _posvel_obs()
    with pytest.raises(FrozenInstanceError):
        o.v_n = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_obs_id_rejected():
    with pytest.raises(ValueError, match="obs_id"):
        _pos_obs(obs_id="")


def test_negative_acquisition_time_rejected():
    with pytest.raises(ValueError, match="acquisition_time"):
        _pos_obs(acquisition_time=-1.0)


def test_zero_acquisition_time_rejected():
    with pytest.raises(ValueError, match="acquisition_time"):
        _pos_obs(acquisition_time=0.0)


def test_negative_ingestion_time_rejected():
    with pytest.raises(ValueError, match="ingestion_time"):
        _pos_obs(ingestion_time=-5.0)


def test_nan_velocity_north_rejected():
    with pytest.raises(ValueError, match="v_n"):
        _posvel_obs(v_n=float("nan"))


def test_inf_velocity_east_rejected():
    with pytest.raises(ValueError, match="v_e"):
        _posvel_obs(v_e=float("inf"))
