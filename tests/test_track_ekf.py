"""
Tests for the TrackState EKF (ADR-0005).

Belief-state math with hand-computed expected values.  Covers:

  1.  uncertainty_km property — known 2x2 position block → expected scalar
  2.  uncertainty_km setter  — scalar r_km → expected isotropic cov
  3.  position_sigma_km       — identical numeric alias for uncertainty_km
  4.  confidence property     — known radius → expected exp(-r/50)
  5.  setter/property roundtrip — set, then read, equal
  6.  predict(dt)             — known prior cov + dt + Q → expected posterior cov
  7.  update(obs)             — known prior + observation + R → expected posterior
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from custody.models import TrackState


UTC_EPS = 1e-9


# ---------------------------------------------------------------------------
# 1. uncertainty_km property
# ---------------------------------------------------------------------------


def test_uncertainty_km_isotropic_block():
    """Trace/2 of an isotropic 2x2 position block in m² → √ / 1000 km."""
    t = TrackState()
    # Force cov[:2, :2] to diag((3000m)², (3000m)²)  → 9e6 each
    t.cov = np.zeros((4, 4))
    t.cov[:2, :2] = (3000.0 ** 2) * np.eye(2)
    # trace = 1.8e7, trace/2 = 9e6, sqrt = 3000 m = 3.0 km
    assert t.uncertainty_km == pytest.approx(3.0, abs=1e-9)


def test_uncertainty_km_anisotropic_block():
    """trace/2 average applies to asymmetric 2x2 block."""
    t = TrackState()
    t.cov = np.zeros((4, 4))
    t.cov[0, 0] = 9e6       # (3 km)²
    t.cov[1, 1] = 1e6       # (1 km)²
    # trace = 1e7, trace/2 = 5e6, sqrt = 2236.068 m ≈ 2.236 km
    assert t.uncertainty_km == pytest.approx(math.sqrt(5e6) / 1000.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 2. uncertainty_km setter
# ---------------------------------------------------------------------------


def test_uncertainty_km_setter_isotropic_projection():
    """Setting uncertainty_km=r → cov[:2,:2] = (r*1000)² * I."""
    t = TrackState()
    t.uncertainty_km = 10.0
    # Expected: cov[:2,:2] = (10000)² * I = 1e8 * I
    expected = 1e8 * np.eye(2)
    np.testing.assert_allclose(t.cov[:2, :2], expected, atol=1e-6)


def test_uncertainty_km_setter_preserves_velocity_block():
    """Setter for position must not clobber the velocity (bottom-right) block."""
    t = TrackState()
    t.cov[2:, 2:] = np.array([[4.0, 0.5], [0.5, 9.0]])  # (m/s)²
    v_before = t.cov[2:, 2:].copy()
    t.uncertainty_km = 7.5
    np.testing.assert_allclose(t.cov[2:, 2:], v_before, atol=1e-12)


# ---------------------------------------------------------------------------
# 3. position_sigma_km — alias
# ---------------------------------------------------------------------------


def test_position_sigma_km_matches_uncertainty_km():
    """v3-aligned name returns identical numeric value."""
    t = TrackState(uncertainty_km=4.0)
    assert t.position_sigma_km == pytest.approx(t.uncertainty_km, abs=UTC_EPS)
    assert t.position_sigma_km == pytest.approx(4.0, abs=UTC_EPS)


# ---------------------------------------------------------------------------
# 4. confidence property
# ---------------------------------------------------------------------------


def test_confidence_for_known_radius():
    """confidence = exp(-uncertainty_km / 50)."""
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
    """TrackState(uncertainty_km=N) projects to isotropic cov."""
    t = TrackState(uncertainty_km=8.0)
    assert t.uncertainty_km == pytest.approx(8.0, abs=1e-9)


def test_default_constructor_gives_five_km():
    """TrackState() with no args has uncertainty_km == 5.0 (existing default)."""
    t = TrackState()
    assert t.uncertainty_km == pytest.approx(5.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 6. predict(dt)
# ---------------------------------------------------------------------------


def test_predict_cov_grows_by_Q_times_dt_on_zero_prior():
    """Starting from a zero position cov, predict grows each diagonal by Q*dt."""
    t = TrackState()
    t.cov = np.zeros((4, 4))
    q_pos = 5000.0  # m²/s
    q_vel = 0.1     # (m/s)²/s
    t.predict(dt_seconds=10.0, q_pos=q_pos, q_vel=q_vel)
    # Expected: cov = diag(50000, 50000, 1.0, 1.0)  (pure Q*dt since prior cov is 0)
    assert t.cov[0, 0] == pytest.approx(q_pos * 10.0, abs=1e-9)
    assert t.cov[1, 1] == pytest.approx(q_pos * 10.0, abs=1e-9)
    assert t.cov[2, 2] == pytest.approx(q_vel * 10.0, abs=1e-9)
    assert t.cov[3, 3] == pytest.approx(q_vel * 10.0, abs=1e-9)


def test_predict_couples_velocity_into_position_via_F():
    """F[0,2]=dt, F[1,3]=dt: velocity variance couples into position after predict.

    Prior cov: diag(0, 0, v², v²). After F P F.T with dt, position block is v²*dt².
    """
    t = TrackState()
    t.cov = np.zeros((4, 4))
    t.cov[2, 2] = 4.0  # (2 m/s)²
    t.cov[3, 3] = 4.0
    # Q=0 to isolate F @ P @ F.T behavior
    t.predict(dt_seconds=3.0, q_pos=0.0, q_vel=0.0)
    # Expected cov[0,0] = 4 * 9 = 36; cov[1,1] = 36
    # Also position-velocity cross terms: cov[0,2] = 4 * 3 = 12
    assert t.cov[0, 0] == pytest.approx(36.0, abs=1e-9)
    assert t.cov[1, 1] == pytest.approx(36.0, abs=1e-9)
    assert t.cov[0, 2] == pytest.approx(12.0, abs=1e-9)
    assert t.cov[1, 3] == pytest.approx(12.0, abs=1e-9)


def test_predict_advances_mean_by_velocity_dt():
    """When mean is set, lat/lon advance by v * dt / R_earth (CV model)."""
    earth_r = 6371000.0
    t = TrackState()
    t.mean = np.array([0.0, 0.0, 10.0, 0.0])  # 10 m/s north at equator
    t.predict(dt_seconds=100.0, q_pos=0.0, q_vel=0.0)
    # dn = 10 * 100 = 1000 m → dlat = 1000 / earth_r radians
    assert t.mean[0] == pytest.approx(1000.0 / earth_r, abs=1e-12)
    assert t.mean[1] == pytest.approx(0.0, abs=1e-12)


def test_predict_without_mean_only_grows_cov():
    """If mean is None, predict only grows cov (doesn't raise)."""
    t = TrackState()
    assert t.mean is None
    cov_before_trace = np.trace(t.cov)
    t.predict(dt_seconds=1.0)
    assert np.trace(t.cov) > cov_before_trace


# ---------------------------------------------------------------------------
# 7. update(obs_lat, obs_lon, R)
# ---------------------------------------------------------------------------


def test_update_initializes_mean_when_none():
    """First observation sets the mean to the obs position; velocity = 0."""
    t = TrackState()
    assert t.mean is None
    t.update(obs_lat_rad=0.1, obs_lon_rad=0.2)
    assert t.mean is not None
    assert t.mean[0] == pytest.approx(0.1, abs=UTC_EPS)
    assert t.mean[1] == pytest.approx(0.2, abs=UTC_EPS)
    assert t.mean[2] == pytest.approx(0.0, abs=UTC_EPS)
    assert t.mean[3] == pytest.approx(0.0, abs=UTC_EPS)


def test_update_reduces_position_uncertainty_toward_observation():
    """Kalman update with obs at predicted position → posterior cov shrinks.

    Hand-computed 1D scalar equivalent: prior σ²=P, R=observation variance,
    posterior σ² = P*R / (P+R).  For our 2D isotropic case with P=(1000m)²
    and R=(100m)² per axis, posterior diag = 1e6 * 1e4 / (1e6+1e4) = 9901.
    """
    t = TrackState()
    t.mean = np.array([0.0, 0.0, 0.0, 0.0])
    # Pure isotropic position cov, no velocity coupling
    t.cov = np.zeros((4, 4))
    t.cov[0, 0] = 1e6   # (1000 m)²
    t.cov[1, 1] = 1e6
    # Observe at the same position (zero innovation) with R = (100m)² diag
    R = (100.0 ** 2) * np.eye(2)
    t.update(obs_lat_rad=0.0, obs_lon_rad=0.0, R=R)
    expected = 1e6 * 1e4 / (1e6 + 1e4)  # = 9900.990099...
    assert t.cov[0, 0] == pytest.approx(expected, rel=1e-9)
    assert t.cov[1, 1] == pytest.approx(expected, rel=1e-9)


def test_update_zero_innovation_leaves_mean_unchanged():
    """Observation equal to predicted position must not shift the mean."""
    t = TrackState()
    t.mean = np.array([0.5, 1.5, 0.0, 0.0])
    t.update(obs_lat_rad=0.5, obs_lon_rad=1.5)
    assert t.mean[0] == pytest.approx(0.5, abs=1e-12)
    assert t.mean[1] == pytest.approx(1.5, abs=1e-12)


def test_update_cov_is_symmetric():
    """Joseph-form update preserves symmetry under non-trivial R and mean."""
    t = TrackState()
    t.mean = np.array([0.1, 0.2, 1.0, 2.0])
    t.cov = np.array([
        [1e6, 1e5, 100.0, 50.0],
        [1e5, 1e6, 50.0, 100.0],
        [100.0, 50.0, 10.0, 1.0],
        [50.0, 100.0, 1.0, 10.0],
    ])
    R = np.array([[2e4, 0.0], [0.0, 2e4]])
    t.update(obs_lat_rad=0.10001, obs_lon_rad=0.20001, R=R)
    # Symmetric to floating-point tolerance
    np.testing.assert_allclose(t.cov, t.cov.T, atol=1e-9)


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
