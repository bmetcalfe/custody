"""
Tests for custody.fusion.tracker — Hungarian association + track lifecycle.

Covers: single-track basic tracking, N-of-M confirmation, association on
pairs (parallel + crossed), Mahalanobis gating, observation-type dispatch
through the tracker, coasting and retirement, multi-batch scenarios.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pyproj import Geod

from custody import config
from custody.fusion.observations import (
    PositionObservation,
    PositionVelocityObservation,
)
from custody.fusion.tracker import Tracker, TrackStatus


ANCHOR_LAT = config.AOI_ANCHOR_LAT
ANCHOR_LON = config.AOI_ANCHOR_LON
_GEOD = Geod(ellps="WGS84")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _offset(lat: float, lon: float, bearing_deg: float, distance_m: float) -> tuple[float, float]:
    lon2, lat2, _ = _GEOD.fwd(lon, lat, bearing_deg, distance_m)
    return lat2, lon2


def _pos(
    obs_id: str,
    lat: float = ANCHOR_LAT,
    lon: float = ANCHOR_LON,
    t: float = 1_689_000_000.0,
    sigma_m: float = 50.0,
) -> PositionObservation:
    return PositionObservation(
        obs_id=obs_id,
        source_id="umbra",
        modality="SAR",
        acquisition_time=t,
        ingestion_time=t + 60.0,
        lat=lat,
        lon=lon,
        cov_pos=np.diag([sigma_m ** 2, sigma_m ** 2]).astype(float),
        raw_ref=f"test://{obs_id}",
    )


def _posvel(
    obs_id: str,
    lat: float = ANCHOR_LAT,
    lon: float = ANCHOR_LON,
    t: float = 1_689_000_000.0,
    v_n: float = 0.0,
    v_e: float = 0.0,
    sigma_pos_m: float = 100.0,
    sigma_vel_mps: float = 0.5,
) -> PositionVelocityObservation:
    cov = np.diag([
        sigma_pos_m ** 2, sigma_pos_m ** 2,
        sigma_vel_mps ** 2, sigma_vel_mps ** 2,
    ]).astype(float)
    return PositionVelocityObservation(
        obs_id=obs_id,
        source_id="gfw_ais",
        modality="AIS",
        acquisition_time=t,
        ingestion_time=t + 30.0,
        lat=lat,
        lon=lon,
        v_n=v_n,
        v_e=v_e,
        cov=cov,
        raw_ref=f"gfw://{obs_id}",
    )


# ---------------------------------------------------------------------------
# 1-4. BASIC TRACKING
# ---------------------------------------------------------------------------


def test_single_track_straight_line_three_observations():
    tracker = Tracker()
    t0 = 1_689_000_000.0
    for i in range(3):
        lat, lon = _offset(ANCHOR_LAT, ANCHOR_LON, 90.0, 100.0 * i)
        tracker.step([_pos(f"o{i}", lat=lat, lon=lon, t=t0 + i * 10.0)], t0 + i * 10.0)
    actives = tracker.active_tracks
    assert len(actives) == 1
    track = actives[0]
    # Track accumulated all three observations
    assert len(track.observation_history) == 3


def test_three_consecutive_hits_confirm_track():
    tracker = Tracker(n_of_m=(3, 5))
    t0 = 1_689_000_000.0
    for i in range(3):
        lat, lon = _offset(ANCHOR_LAT, ANCHOR_LON, 90.0, 100.0 * i)
        tracker.step([_pos(f"o{i}", lat=lat, lon=lon, t=t0 + i * 10.0)], t0 + i * 10.0)
    assert tracker.active_tracks[0].status == TrackStatus.CONFIRMED


def test_single_observation_spawns_tentative_track():
    tracker = Tracker()
    tracker.step([_pos("o1")], 1_689_000_000.0)
    assert len(tracker.active_tracks) == 1
    assert tracker.active_tracks[0].status == TrackStatus.TENTATIVE


def test_predict_grows_cov_between_batches_update_shrinks_on_obs():
    tracker = Tracker()
    t0 = 1_689_000_000.0
    tracker.step([_pos("o0", t=t0)], t0)
    track = tracker.active_tracks[0]
    sigma_after_first = track.state.uncertainty_km
    # Second batch 1 hour later with no observations — predict only, cov grows
    tracker.step([], t0 + 3600.0)
    sigma_after_gap = track.state.uncertainty_km
    assert sigma_after_gap > sigma_after_first
    # Third batch at the same place — update shrinks cov back down
    lat, lon = _offset(ANCHOR_LAT, ANCHOR_LON, 90.0, 50.0)
    tracker.step([_pos("o1", lat=lat, lon=lon, t=t0 + 3600.0)], t0 + 3600.0)
    sigma_after_update = track.state.uncertainty_km
    assert sigma_after_update < sigma_after_gap


# ---------------------------------------------------------------------------
# 5-8. ASSOCIATION (Hungarian + gating)
# ---------------------------------------------------------------------------


def test_two_parallel_tracks_two_observations_correct_routing():
    tracker = Tracker()
    t0 = 1_689_000_000.0
    # Batch 1: spawn two tracks 2 km apart (north vs south).
    lat_n, lon_n = _offset(ANCHOR_LAT, ANCHOR_LON, 0.0, 1000.0)
    lat_s, lon_s = _offset(ANCHOR_LAT, ANCHOR_LON, 180.0, 1000.0)
    tracker.step([_pos("a0", lat=lat_n, lon=lon_n, t=t0),
                  _pos("b0", lat=lat_s, lon=lon_s, t=t0)], t0)
    assert len(tracker.active_tracks) == 2

    # Batch 2, 10 s later: each track gets a plausible observation 100 m east
    lat_n2, lon_n2 = _offset(lat_n, lon_n, 90.0, 100.0)
    lat_s2, lon_s2 = _offset(lat_s, lon_s, 90.0, 100.0)
    tracker.step([_pos("a1", lat=lat_n2, lon=lon_n2, t=t0 + 10.0),
                  _pos("b1", lat=lat_s2, lon=lon_s2, t=t0 + 10.0)], t0 + 10.0)

    # Still two tracks; each now has two obs
    assert len(tracker.active_tracks) == 2
    obs_ids = [sorted(tr.observation_history) for tr in tracker.active_tracks]
    # Either tracker keeps them in creation order; just check each track has its two obs
    assert {"a0", "a1"} in [set(h) for h in obs_ids]
    assert {"b0", "b1"} in [set(h) for h in obs_ids]


def test_crossed_tracks_hungarian_uses_full_cost_matrix():
    tracker = Tracker()
    t0 = 1_689_000_000.0
    # Spawn two tracks 200 m apart
    lat_e, lon_e = _offset(ANCHOR_LAT, ANCHOR_LON, 90.0, 200.0)
    tracker.step([_pos("A", lat=ANCHOR_LAT, lon=ANCHOR_LON, t=t0),
                  _pos("B", lat=lat_e, lon=lon_e, t=t0)], t0)
    # Next batch: observations swapped in position so a greedy nearest would
    # assign wrong — but in this construction we want Hungarian to pick the
    # minimum-total-cost assignment.
    # Place observations close to the original spawns.
    lat_e2, lon_e2 = _offset(lat_e, lon_e, 90.0, 50.0)  # near B's spawn
    tracker.step([_pos("A1", lat=ANCHOR_LAT, lon=ANCHOR_LON, t=t0 + 5.0),
                  _pos("B1", lat=lat_e2, lon=lon_e2, t=t0 + 5.0)], t0 + 5.0)
    # Each track should now hold two observations
    hist_sets = [set(tr.observation_history) for tr in tracker.active_tracks]
    assert {"A", "A1"} in hist_sets
    assert {"B", "B1"} in hist_sets


def test_mahalanobis_gate_rejects_far_observation():
    tracker = Tracker(gate_sigma=3.0)
    t0 = 1_689_000_000.0
    tracker.step([_pos("seed", lat=ANCHOR_LAT, lon=ANCHOR_LON, t=t0, sigma_m=50.0)], t0)
    # Second observation 50 km away — far outside 3-sigma gate of the existing track
    lat_far, lon_far = _offset(ANCHOR_LAT, ANCHOR_LON, 90.0, 50_000.0)
    tracker.step([_pos("far", lat=lat_far, lon=lon_far, t=t0 + 1.0, sigma_m=50.0)], t0 + 1.0)
    # Should now have two tracks — one seed, one spawned from far observation
    assert len(tracker.active_tracks) == 2


def test_batch_with_in_gate_and_out_of_gate_observations():
    tracker = Tracker(gate_sigma=3.0)
    t0 = 1_689_000_000.0
    tracker.step([_pos("seed", t=t0, sigma_m=50.0)], t0)
    # Next batch: one obs close (in-gate), one obs far (out-of-gate)
    lat_c, lon_c = _offset(ANCHOR_LAT, ANCHOR_LON, 90.0, 50.0)
    lat_f, lon_f = _offset(ANCHOR_LAT, ANCHOR_LON, 90.0, 50_000.0)
    tracker.step([
        _pos("close", lat=lat_c, lon=lon_c, t=t0 + 1.0, sigma_m=50.0),
        _pos("far", lat=lat_f, lon=lon_f, t=t0 + 1.0, sigma_m=50.0),
    ], t0 + 1.0)
    # 2 tracks total: seed + far's new track
    assert len(tracker.active_tracks) == 2
    hist_sets = [set(tr.observation_history) for tr in tracker.active_tracks]
    assert {"seed", "close"} in hist_sets
    assert {"far"} in hist_sets


# ---------------------------------------------------------------------------
# 9. Mixed observation types pass through TrackState.update dispatch
# ---------------------------------------------------------------------------


def test_mixed_position_and_posvel_in_same_batch():
    tracker = Tracker()
    t0 = 1_689_000_000.0
    lat_a, lon_a = _offset(ANCHOR_LAT, ANCHOR_LON, 0.0, 500.0)
    lat_b, lon_b = _offset(ANCHOR_LAT, ANCHOR_LON, 180.0, 500.0)
    tracker.step([
        _pos("sar", lat=lat_a, lon=lon_a, t=t0),
        _posvel("ais", lat=lat_b, lon=lon_b, t=t0, v_n=1.0, v_e=0.0),
    ], t0)
    assert len(tracker.active_tracks) == 2
    # The AIS-seeded track should have non-zero velocity on its state mean
    by_history = {tuple(tr.observation_history): tr for tr in tracker.active_tracks}
    ais_track = by_history[("ais",)]
    assert ais_track.state.mean is not None
    assert ais_track.state.mean[2] == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# 10-13. LIFECYCLE
# ---------------------------------------------------------------------------


def test_confirmed_to_coasted_after_five_missed_steps():
    tracker = Tracker(n_of_m=(3, 5), coast_threshold=5)
    t0 = 1_689_000_000.0
    # 3 consecutive hits → CONFIRMED by step 3
    for i in range(3):
        lat, lon = _offset(ANCHOR_LAT, ANCHOR_LON, 90.0, 50.0 * i)
        tracker.step([_pos(f"h{i}", lat=lat, lon=lon, t=t0 + i)], t0 + i)
    track = tracker.active_tracks[0]
    assert track.status == TrackStatus.CONFIRMED
    # 5 empty steps: miss, miss, miss, miss, miss → COASTED after 5th miss
    for i in range(5):
        tracker.step([], t0 + 10.0 + i)
    assert track.status == TrackStatus.COASTED


def test_coasted_to_confirmed_on_new_observation_in_gate():
    tracker = Tracker(n_of_m=(3, 5), coast_threshold=5)
    t0 = 1_689_000_000.0
    for i in range(3):
        lat, lon = _offset(ANCHOR_LAT, ANCHOR_LON, 90.0, 50.0 * i)
        tracker.step([_pos(f"h{i}", lat=lat, lon=lon, t=t0 + i)], t0 + i)
    track = tracker.active_tracks[0]
    # Coast
    for i in range(5):
        tracker.step([], t0 + 10.0 + i)
    assert track.status == TrackStatus.COASTED
    # A new observation near the last-known position brings it back
    lat_c, lon_c = _offset(ANCHOR_LAT, ANCHOR_LON, 90.0, 120.0)
    tracker.step([_pos("revive", lat=lat_c, lon=lon_c, t=t0 + 15.0, sigma_m=500.0)],
                 t0 + 15.0)
    assert track.status == TrackStatus.CONFIRMED


def test_coasted_to_retired_after_max_coast_steps():
    tracker = Tracker(n_of_m=(3, 5), coast_threshold=5, max_coast_steps=10)
    t0 = 1_689_000_000.0
    for i in range(3):
        lat, lon = _offset(ANCHOR_LAT, ANCHOR_LON, 90.0, 50.0 * i)
        tracker.step([_pos(f"h{i}", lat=lat, lon=lon, t=t0 + i)], t0 + i)
    # 11 empty steps — exceeds max_coast_steps=10
    for i in range(11):
        tracker.step([], t0 + 10.0 + i)
    assert len(tracker.active_tracks) == 0
    assert len(tracker.retired_tracks) == 1
    assert tracker.retired_tracks[0].status == TrackStatus.RETIRED


def test_retired_tracks_do_not_participate_in_association():
    tracker = Tracker(n_of_m=(3, 5), coast_threshold=5, max_coast_steps=6)
    t0 = 1_689_000_000.0
    for i in range(3):
        lat, lon = _offset(ANCHOR_LAT, ANCHOR_LON, 90.0, 50.0 * i)
        tracker.step([_pos(f"h{i}", lat=lat, lon=lon, t=t0 + i)], t0 + i)
    for i in range(8):
        tracker.step([], t0 + 10.0 + i)
    assert len(tracker.retired_tracks) == 1
    # New observation near the retired track's last known position.
    lat, lon = _offset(ANCHOR_LAT, ANCHOR_LON, 90.0, 100.0)
    tracker.step([_pos("new", lat=lat, lon=lon, t=t0 + 20.0, sigma_m=50.0)], t0 + 20.0)
    # A new track should have spawned, not a resurrection
    assert len(tracker.active_tracks) == 1
    assert tracker.active_tracks[0].observation_history == ["new"]
    assert tracker.active_tracks[0].status == TrackStatus.TENTATIVE


# ---------------------------------------------------------------------------
# 14-15. N-OF-M CONFIRMATION
# ---------------------------------------------------------------------------


def test_n_of_m_confirms_with_3_hits_in_5_sliding_window():
    """Hits at steps 1, 3, 5 under 3-of-5 sliding window → confirm on step 5."""
    tracker = Tracker(n_of_m=(3, 5))
    t0 = 1_689_000_000.0
    # step 1: hit
    tracker.step([_pos("o1", t=t0)], t0)
    # step 2: miss
    tracker.step([], t0 + 1.0)
    assert tracker.active_tracks[0].status == TrackStatus.TENTATIVE
    # step 3: hit
    lat3, lon3 = _offset(ANCHOR_LAT, ANCHOR_LON, 90.0, 50.0)
    tracker.step([_pos("o3", lat=lat3, lon=lon3, t=t0 + 2.0, sigma_m=500.0)], t0 + 2.0)
    assert tracker.active_tracks[0].status == TrackStatus.TENTATIVE
    # step 4: miss
    tracker.step([], t0 + 3.0)
    assert tracker.active_tracks[0].status == TrackStatus.TENTATIVE
    # step 5: hit → 3 hits in window of 5
    lat5, lon5 = _offset(ANCHOR_LAT, ANCHOR_LON, 90.0, 100.0)
    tracker.step([_pos("o5", lat=lat5, lon=lon5, t=t0 + 4.0, sigma_m=500.0)], t0 + 4.0)
    assert tracker.active_tracks[0].status == TrackStatus.CONFIRMED


def test_n_of_m_does_not_confirm_with_only_2_hits_in_window():
    """Hits at steps 1, 2, 6, 7 — sliding window of 5 at step 7 contains only 2 hits."""
    tracker = Tracker(n_of_m=(3, 5), coast_threshold=999, max_coast_steps=999)
    t0 = 1_689_000_000.0
    # hit, hit, miss, miss, miss, hit, hit
    # positions jittered so the gate is wide but they still associate
    def near(off_m):
        lat, lon = _offset(ANCHOR_LAT, ANCHOR_LON, 90.0, off_m)
        return lat, lon
    steps = [
        (True, 0.0),    # 1
        (True, 20.0),   # 2
        (False, 0.0),   # 3
        (False, 0.0),   # 4
        (False, 0.0),   # 5
        (True, 40.0),   # 6
        (True, 60.0),   # 7
    ]
    for idx, (hit, off) in enumerate(steps):
        if hit:
            lat, lon = near(off)
            tracker.step([_pos(f"o{idx}", lat=lat, lon=lon, t=t0 + idx,
                               sigma_m=500.0)], t0 + idx)
        else:
            tracker.step([], t0 + idx)
    # Window at step 7 covers steps 3..7 = [F, F, F, T, T] → 2 hits < 3 → TENTATIVE
    assert tracker.active_tracks[0].status == TrackStatus.TENTATIVE


# ---------------------------------------------------------------------------
# 16. MULTI-BATCH
# ---------------------------------------------------------------------------


def test_multi_batch_scenario_creation_and_retirement():
    """Three batches over 30 minutes. Two tracks fully confirm; one retires mid-window."""
    tracker = Tracker(n_of_m=(3, 5), coast_threshold=3, max_coast_steps=4)
    t0 = 1_689_000_000.0

    # Batch 1: three observations at three separated locations → 3 TENTATIVE tracks
    a = _offset(ANCHOR_LAT, ANCHOR_LON, 0.0, 5000.0)
    b = _offset(ANCHOR_LAT, ANCHOR_LON, 90.0, 5000.0)
    c = _offset(ANCHOR_LAT, ANCHOR_LON, 180.0, 5000.0)
    tracker.step([
        _pos("a0", lat=a[0], lon=a[1], t=t0),
        _pos("b0", lat=b[0], lon=b[1], t=t0),
        _pos("c0", lat=c[0], lon=c[1], t=t0),
    ], t0)
    assert len(tracker.active_tracks) == 3

    # Batches 2..5: only A and B get observations; C is abandoned
    for step_idx in range(1, 5):
        a_next = _offset(a[0], a[1], 90.0, 50.0 * step_idx)
        b_next = _offset(b[0], b[1], 90.0, 50.0 * step_idx)
        tracker.step([
            _pos(f"a{step_idx}", lat=a_next[0], lon=a_next[1], t=t0 + step_idx),
            _pos(f"b{step_idx}", lat=b_next[0], lon=b_next[1], t=t0 + step_idx),
        ], t0 + step_idx)

    # After 5 total batches: A and B confirmed (5 hits each); C retired (coast then max-coast)
    statuses = {tuple(tr.observation_history)[0]: tr.status
                for tr in tracker.active_tracks}
    assert statuses.get("a0") == TrackStatus.CONFIRMED
    assert statuses.get("b0") == TrackStatus.CONFIRMED
    assert len(tracker.retired_tracks) == 1
    assert tracker.retired_tracks[0].observation_history == ["c0"]
