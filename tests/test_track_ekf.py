"""
Tests for the TrackState EKF (ADR-0005, ADR-0008, ADR-0010).

Belief-state math with hand-computed expected values.  State mean is stored
in the AEQD tangent-plane meters basis [x_east, y_north, v_n, v_e] per
ADR-0010; observations dispatch on PositionObservation vs
PositionVelocityObservation per ADR-0008.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from custody import config
from custody.fusion import geo as fusion_geo
from custody.fusion.observations import (
    PositionObservation,
    PositionVelocityObservation,
)
from custody.models import TrackState


UTC_EPS = 1e-9
ANCHOR_LAT = config.AOI_ANCHOR_LAT
ANCHOR_LON = config.AOI_ANCHOR_LON


def _pos_obs(
    lat: float,
    lon: float,
    cov_pos: np.ndarray,
    obs_id: str = "obs",
) -> PositionObservation:
    return PositionObservation(
        obs_id=obs_id,
        source_id="test",
        modality="SAR",
        acquisition_time=1_689_000_000.0,
        ingestion_time=1_689_000_060.0,
        lat=lat,
        lon=lon,
        cov_pos=cov_pos,
        raw_ref="test://",
    )


def _posvel_obs(
    lat: float,
    lon: float,
    v_n: float,
    v_e: float,
    cov: np.ndarray,
    obs_id: str = "ais",
) -> PositionVelocityObservation:
    return PositionVelocityObservation(
        obs_id=obs_id,
        source_id="gfw_ais",
        modality="AIS",
        acquisition_time=1_689_000_000.0,
        ingestion_time=1_689_000_030.0,
        lat=lat,
        lon=lon,
        v_n=v_n,
        v_e=v_e,
        cov=cov,
        raw_ref="test://",
    )


# ---------------------------------------------------------------------------
# 1. uncertainty_km property
# ---------------------------------------------------------------------------


def test_uncertainty_km_isotropic_block():
    """Trace/2 of an isotropic 2x2 position block in m² → √ / 1000 km."""
    t = TrackState()
    t.cov = np.zeros((4, 4))
    t.cov[:2, :2] = (3000.0 ** 2) * np.eye(2)
    assert t.uncertainty_km == pytest.approx(3.0, abs=1e-9)


def test_uncertainty_km_anisotropic_block():
    t = TrackState()
    t.cov = np.zeros((4, 4))
    t.cov[0, 0] = 9e6
    t.cov[1, 1] = 1e6
    assert t.uncertainty_km == pytest.approx(math.sqrt(5e6) / 1000.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 2. uncertainty_km setter
# ---------------------------------------------------------------------------


def test_uncertainty_km_setter_isotropic_projection():
    t = TrackState()
    t.uncertainty_km = 10.0
    expected = 1e8 * np.eye(2)
    np.testing.assert_allclose(t.cov[:2, :2], expected, atol=1e-6)


def test_uncertainty_km_setter_preserves_velocity_block():
    t = TrackState()
    t.cov[2:, 2:] = np.array([[4.0, 0.5], [0.5, 9.0]])
    v_before = t.cov[2:, 2:].copy()
    t.uncertainty_km = 7.5
    np.testing.assert_allclose(t.cov[2:, 2:], v_before, atol=1e-12)


# ---------------------------------------------------------------------------
# 3. position_sigma_km alias
# ---------------------------------------------------------------------------


def test_position_sigma_km_matches_uncertainty_km():
    t = TrackState(uncertainty_km=4.0)
    assert t.position_sigma_km == pytest.approx(t.uncertainty_km, abs=UTC_EPS)
    assert t.position_sigma_km == pytest.approx(4.0, abs=UTC_EPS)


# ---------------------------------------------------------------------------
# 4. confidence property
# ---------------------------------------------------------------------------


def test_confidence_for_known_radius():
    t = TrackState(uncertainty_km=5.0)
    assert t.confidence == pytest.approx(math.exp(-0.1), abs=UTC_EPS)
    t.uncertainty_km = 25.0
    assert t.confidence == pytest.approx(math.exp(-0.5), abs=UTC_EPS)


# ---------------------------------------------------------------------------
# 5. Setter/property roundtrip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("r_km", [0.5, 2.0, 5.0, 12.5, 40.0])
def test_roundtrip_setter_then_property(r_km: float):
    t = TrackState()
    t.uncertainty_km = r_km
    assert t.uncertainty_km == pytest.approx(r_km, abs=1e-9)


def test_constructor_kwarg_sets_isotropic_cov():
    t = TrackState(uncertainty_km=8.0)
    assert t.uncertainty_km == pytest.approx(8.0, abs=1e-9)


def test_default_constructor_gives_five_km():
    t = TrackState()
    assert t.uncertainty_km == pytest.approx(5.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 5b. lat/lon properties derived from mean
# ---------------------------------------------------------------------------


def test_lat_lon_properties_return_none_when_mean_none():
    t = TrackState()
    assert t.mean is None
    assert t.lat is None
    assert t.lon is None


def test_lat_lon_properties_roundtrip_through_projection():
    t = TrackState(lat=9.80, lon=116.10)
    assert t.lat == pytest.approx(9.80, abs=1e-9)
    assert t.lon == pytest.approx(116.10, abs=1e-9)


def test_constructor_projects_lat_lon_at_anchor_to_origin():
    t = TrackState(lat=ANCHOR_LAT, lon=ANCHOR_LON)
    assert t.mean is not None
    assert t.mean[0] == pytest.approx(0.0, abs=1e-6)
    assert t.mean[1] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 6. predict(dt)
# ---------------------------------------------------------------------------


def test_predict_cov_grows_by_Q_times_dt_on_zero_prior():
    t = TrackState()
    t.cov = np.zeros((4, 4))
    q_pos = 5000.0
    q_vel = 0.1
    t.predict(dt_seconds=10.0, q_pos=q_pos, q_vel=q_vel)
    assert t.cov[0, 0] == pytest.approx(q_pos * 10.0, abs=1e-9)
    assert t.cov[1, 1] == pytest.approx(q_pos * 10.0, abs=1e-9)
    assert t.cov[2, 2] == pytest.approx(q_vel * 10.0, abs=1e-9)
    assert t.cov[3, 3] == pytest.approx(q_vel * 10.0, abs=1e-9)


def test_predict_couples_velocity_into_position_via_F():
    """State order [x_east, y_north, v_n, v_e]: F couples x<->v_e, y<->v_n.

    Prior diag(0, 0, v_n²=4, v_e²=4), dt=3, Q=0.
    F P F.T: position diag grows as v² × dt² = 4 × 9 = 36.
    Cross terms: cov[0,3] = dt × v_e² = 3 × 4 = 12 (x with v_e);
                 cov[1,2] = dt × v_n² = 3 × 4 = 12 (y with v_n).
    """
    t = TrackState()
    t.cov = np.zeros((4, 4))
    t.cov[2, 2] = 4.0  # v_n variance (m/s)²
    t.cov[3, 3] = 4.0  # v_e variance
    t.predict(dt_seconds=3.0, q_pos=0.0, q_vel=0.0)
    assert t.cov[0, 0] == pytest.approx(36.0, abs=1e-9)
    assert t.cov[1, 1] == pytest.approx(36.0, abs=1e-9)
    assert t.cov[0, 3] == pytest.approx(12.0, abs=1e-9)
    assert t.cov[1, 2] == pytest.approx(12.0, abs=1e-9)
    # And the mirrored symmetric entries
    assert t.cov[3, 0] == pytest.approx(12.0, abs=1e-9)
    assert t.cov[2, 1] == pytest.approx(12.0, abs=1e-9)


def test_predict_advances_mean_by_velocity_dt():
    """10 m/s northward for 100 s advances mean[1] (y_north) by 1000 m."""
    t = TrackState(lat=ANCHOR_LAT, lon=ANCHOR_LON, v_n=10.0, v_e=0.0)
    assert t.mean[0] == pytest.approx(0.0, abs=1e-6)
    assert t.mean[1] == pytest.approx(0.0, abs=1e-6)
    t.predict(dt_seconds=100.0, q_pos=0.0, q_vel=0.0)
    assert t.mean[0] == pytest.approx(0.0, abs=1e-9)      # x (east) unchanged
    assert t.mean[1] == pytest.approx(1000.0, abs=1e-9)   # y (north) += v_n * dt


def test_predict_without_mean_only_grows_cov():
    t = TrackState()
    assert t.mean is None
    cov_before_trace = np.trace(t.cov)
    t.predict(dt_seconds=1.0)
    assert np.trace(t.cov) > cov_before_trace
    assert t.mean is None


# ---------------------------------------------------------------------------
# 6b. ADR-0007 anchor regression — default Q matches Phase 2 sigma(t)=5+3t
# ---------------------------------------------------------------------------


def _run_default_q_for_hours(hours: float) -> float:
    t = TrackState()
    for _ in range(int(hours)):
        t.predict(dt_seconds=3600.0)
    return t.uncertainty_km


def test_default_q_anchor_at_one_hour():
    sigma = _run_default_q_for_hours(1)
    assert sigma == pytest.approx(8.0, rel=0.02)


def test_default_q_anchor_at_twenty_four_hours():
    sigma = _run_default_q_for_hours(24)
    assert sigma == pytest.approx(77.0, rel=0.02)


# ---------------------------------------------------------------------------
# 7. update — PositionObservation branch (ADR-0008 §1-4)
# ---------------------------------------------------------------------------


def test_position_obs_update_shrinks_position_cov_preserves_velocity_cov():
    """Obs at same position: innovation ~0; position cov ~= R; velocity cov barely moves."""
    t = TrackState(lat=ANCHOR_LAT, lon=ANCHOR_LON)
    t.cov = np.zeros((4, 4))
    t.cov[0, 0] = 1e6   # (1000 m)² position
    t.cov[1, 1] = 1e6
    t.cov[2, 2] = 1.0   # velocity cov
    t.cov[3, 3] = 1.0
    vel_before = t.cov[2:, 2:].copy()
    R = (100.0 ** 2) * np.eye(2)
    obs = _pos_obs(ANCHOR_LAT, ANCHOR_LON, cov_pos=R)
    t.update(obs)
    expected_pos = 1e6 * 1e4 / (1e6 + 1e4)
    assert t.cov[0, 0] == pytest.approx(expected_pos, rel=1e-9)
    assert t.cov[1, 1] == pytest.approx(expected_pos, rel=1e-9)
    np.testing.assert_allclose(t.cov[2:, 2:], vel_before, atol=1e-6)


def test_position_obs_update_moves_mean_toward_observation():
    """With prior σ_pos ~ R, posterior mean is ~halfway between prior and obs."""
    t = TrackState(lat=ANCHOR_LAT, lon=ANCHOR_LON)
    t.cov = np.zeros((4, 4))
    t.cov[0, 0] = 1e6   # prior pos σ² = 1e6 m²
    t.cov[1, 1] = 1e6
    R = 1e6 * np.eye(2)  # obs σ² = 1e6 m² → Kalman gain = 0.5 per axis
    # Put the observation at 2 km east of anchor
    x_obs = 2000.0
    obs_lat, obs_lon = fusion_geo.from_tangent_plane(x_obs, 0.0)
    obs = _pos_obs(obs_lat, obs_lon, cov_pos=R)
    t.update(obs)
    # Posterior mean[0] = 0.5 * 2000 = 1000
    assert t.mean[0] == pytest.approx(1000.0, abs=1.0)
    assert t.mean[1] == pytest.approx(0.0, abs=1.0)


def test_position_obs_update_tiny_R_snaps_posterior_to_observation():
    t = TrackState(lat=ANCHOR_LAT, lon=ANCHOR_LON)
    t.cov = np.zeros((4, 4))
    t.cov[0, 0] = 1e8  # large prior σ² = 1e8 m²
    t.cov[1, 1] = 1e8
    R = (0.1 ** 2) * np.eye(2)  # tiny sensor σ = 10 cm
    obs_x_target = 500.0
    obs_lat, obs_lon = fusion_geo.from_tangent_plane(obs_x_target, 0.0)
    obs = _pos_obs(obs_lat, obs_lon, cov_pos=R)
    t.update(obs)
    # Posterior position cov ~ R; mean snaps to observation
    assert t.cov[0, 0] == pytest.approx(R[0, 0], rel=1e-4)
    assert t.cov[1, 1] == pytest.approx(R[1, 1], rel=1e-4)
    assert t.mean[0] == pytest.approx(obs_x_target, abs=0.01)
    assert t.mean[1] == pytest.approx(0.0, abs=0.01)


def test_position_obs_update_huge_R_leaves_posterior_near_prior():
    t = TrackState(lat=ANCHOR_LAT, lon=ANCHOR_LON)
    t.cov = np.zeros((4, 4))
    t.cov[0, 0] = 1e6   # prior σ² = 1e6
    t.cov[1, 1] = 1e6
    prior = t.cov.copy()
    R = 1e12 * np.eye(2)  # useless sensor
    obs_lat, obs_lon = fusion_geo.from_tangent_plane(5000.0, 5000.0)
    obs = _pos_obs(obs_lat, obs_lon, cov_pos=R)
    t.update(obs)
    # Posterior pos cov ≈ prior (Kalman gain ≈ 0); mean barely moves
    np.testing.assert_allclose(t.cov[:2, :2], prior[:2, :2], rtol=1e-5)
    assert abs(t.mean[0]) < 10.0
    assert abs(t.mean[1]) < 10.0


# ---------------------------------------------------------------------------
# 8. update — PositionVelocityObservation branch (ADR-0008 §5-7)
# ---------------------------------------------------------------------------


def test_posvel_obs_update_shrinks_both_position_and_velocity_cov():
    t = TrackState(lat=ANCHOR_LAT, lon=ANCHOR_LON)
    t.cov = np.zeros((4, 4))
    t.cov[0, 0] = 1e6
    t.cov[1, 1] = 1e6
    t.cov[2, 2] = 4.0  # (2 m/s)²
    t.cov[3, 3] = 4.0
    R = np.diag([1e4, 1e4, 0.25, 0.25])  # pos 100 m, vel 0.5 m/s
    obs = _posvel_obs(ANCHOR_LAT, ANCHOR_LON, v_n=0.0, v_e=0.0, cov=R)
    t.update(obs)
    # Position cov shrinks: 1e6 * 1e4 / (1e6 + 1e4)
    assert t.cov[0, 0] < 1e6
    assert t.cov[1, 1] < 1e6
    # Velocity cov shrinks: 4 * 0.25 / (4 + 0.25)
    assert t.cov[2, 2] < 4.0
    assert t.cov[3, 3] < 4.0


def test_posvel_obs_update_tiny_R_on_velocity_converges_velocity():
    t = TrackState(lat=ANCHOR_LAT, lon=ANCHOR_LON, v_n=0.0, v_e=0.0)
    t.cov = np.zeros((4, 4))
    t.cov[0, 0] = 1e6    # moderate position uncertainty
    t.cov[1, 1] = 1e6
    t.cov[2, 2] = 100.0  # v_n σ² = 100 (σ = 10 m/s)
    t.cov[3, 3] = 100.0
    R = np.diag([1e6, 1e6, 0.0001, 0.0001])  # precise velocity only
    obs = _posvel_obs(ANCHOR_LAT, ANCHOR_LON, v_n=5.0, v_e=-3.0, cov=R)
    t.update(obs)
    # Velocity state snaps to observation
    assert t.mean[2] == pytest.approx(5.0, abs=0.01)
    assert t.mean[3] == pytest.approx(-3.0, abs=0.01)
    # Position mean barely moved (R on position is huge)
    assert abs(t.mean[0]) < 10.0
    assert abs(t.mean[1]) < 10.0


def test_posvel_obs_update_tiny_R_everywhere_snaps_all_state():
    t = TrackState(lat=ANCHOR_LAT, lon=ANCHOR_LON, v_n=0.0, v_e=0.0)
    t.cov = 1e8 * np.eye(4)
    R = (0.01 ** 2) * np.eye(4)  # cm-precision everywhere
    x_obs = 1234.5
    obs_lat, obs_lon = fusion_geo.from_tangent_plane(x_obs, 0.0)
    obs = _posvel_obs(obs_lat, obs_lon, v_n=2.5, v_e=-1.0, cov=R)
    t.update(obs)
    assert t.mean[0] == pytest.approx(x_obs, abs=0.01)
    assert t.mean[1] == pytest.approx(0.0, abs=0.01)
    assert t.mean[2] == pytest.approx(2.5, abs=0.01)
    assert t.mean[3] == pytest.approx(-1.0, abs=0.01)


# ---------------------------------------------------------------------------
# 9. Dispatch: unknown type raises TypeError; variants differ in H
# ---------------------------------------------------------------------------


def test_update_with_wrong_type_raises_type_error():
    t = TrackState(lat=ANCHOR_LAT, lon=ANCHOR_LON)
    with pytest.raises(TypeError, match="Observation"):
        t.update(42)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Observation"):
        t.update({"lat": 9.75, "lon": 116.0})  # type: ignore[arg-type]


def test_different_variants_produce_different_posterior_shape():
    """PositionObservation reduces only position variance; PositionVelocityObservation
    reduces position AND velocity. Verify by comparing posterior cov diagonals."""
    # Two identical starting states
    t1 = TrackState(lat=ANCHOR_LAT, lon=ANCHOR_LON)
    t1.cov = np.diag([1e6, 1e6, 100.0, 100.0])
    t2 = TrackState(lat=ANCHOR_LAT, lon=ANCHOR_LON)
    t2.cov = np.diag([1e6, 1e6, 100.0, 100.0])
    # Observation 1: position-only with small R
    t1.update(_pos_obs(ANCHOR_LAT, ANCHOR_LON, cov_pos=1e4 * np.eye(2)))
    # Observation 2: position+velocity with same R on position, small R on velocity
    R_pv = np.diag([1e4, 1e4, 0.01, 0.01])
    t2.update(_posvel_obs(ANCHOR_LAT, ANCHOR_LON, v_n=0.0, v_e=0.0, cov=R_pv))
    # Position diagonals agree (same H rows for position)
    assert t1.cov[0, 0] == pytest.approx(t2.cov[0, 0], rel=1e-3)
    # Velocity diagonals differ: t1 unchanged from prior 100, t2 shrunk far below
    assert t1.cov[2, 2] == pytest.approx(100.0, rel=1e-6)
    assert t2.cov[2, 2] < 1.0


# ---------------------------------------------------------------------------
# 10. Geo integration — observation lat/lon projects through fusion.geo
# ---------------------------------------------------------------------------


def test_update_innovation_computed_in_meters_via_fusion_geo():
    """Build an observation 5 km east of the track; innovation magnitude should
    match fusion.geo.to_tangent_plane, not a hand-rolled spherical approximation."""
    t = TrackState(lat=ANCHOR_LAT, lon=ANCHOR_LON)
    t.cov = np.diag([1e12, 1e12, 1.0, 1.0])  # very weak prior → posterior ≈ obs
    target_x = 5000.0  # 5 km east of anchor
    obs_lat, obs_lon = fusion_geo.from_tangent_plane(target_x, 0.0)
    obs = _pos_obs(obs_lat, obs_lon, cov_pos=(10.0 ** 2) * np.eye(2))
    t.update(obs)
    # Posterior mean[0] should equal target_x (via fusion.geo roundtrip)
    assert t.mean[0] == pytest.approx(target_x, abs=0.1)
    assert t.mean[1] == pytest.approx(0.0, abs=0.1)


# ---------------------------------------------------------------------------
# 11. Joseph form preserves symmetry under non-trivial mean and cov
# ---------------------------------------------------------------------------


def test_update_cov_is_symmetric():
    t = TrackState(lat=9.78, lon=116.05, v_n=1.0, v_e=2.0)
    t.cov = np.array([
        [1e6, 1e5, 100.0, 50.0],
        [1e5, 1e6, 50.0, 100.0],
        [100.0, 50.0, 10.0, 1.0],
        [50.0, 100.0, 1.0, 10.0],
    ])
    obs = _pos_obs(9.79, 116.06, cov_pos=np.array([[2e4, 0.0], [0.0, 2e4]]))
    t.update(obs)
    np.testing.assert_allclose(t.cov, t.cov.T, atol=1e-6)


# ---------------------------------------------------------------------------
# 12. Update initialises mean when None
# ---------------------------------------------------------------------------


def test_update_initializes_mean_when_none_from_position_obs():
    t = TrackState()
    assert t.mean is None
    obs = _pos_obs(9.80, 116.10, cov_pos=(50.0 ** 2) * np.eye(2))
    t.update(obs)
    assert t.mean is not None
    x_exp, y_exp = fusion_geo.to_tangent_plane(9.80, 116.10)
    assert t.mean[0] == pytest.approx(x_exp, abs=1e-6)
    assert t.mean[1] == pytest.approx(y_exp, abs=1e-6)
    assert t.mean[2] == pytest.approx(0.0, abs=1e-12)
    assert t.mean[3] == pytest.approx(0.0, abs=1e-12)


def test_update_initializes_mean_when_none_from_posvel_obs():
    t = TrackState()
    assert t.mean is None
    obs = _posvel_obs(9.80, 116.10, v_n=3.0, v_e=-2.0, cov=np.eye(4))
    t.update(obs)
    assert t.mean is not None
    x_exp, y_exp = fusion_geo.to_tangent_plane(9.80, 116.10)
    assert t.mean[0] == pytest.approx(x_exp, abs=1e-6)
    assert t.mean[1] == pytest.approx(y_exp, abs=1e-6)
    assert t.mean[2] == pytest.approx(3.0, abs=1e-12)
    assert t.mean[3] == pytest.approx(-2.0, abs=1e-12)
