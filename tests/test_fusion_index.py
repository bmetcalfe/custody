"""
Tests for custody.fusion.index -- H3 + DuckDB + Parquet spatial/temporal index.

Covers round-trip serialization (PositionObservation and PositionVelocityObservation),
spatial queries (AOI polygon, nearest), temporal queries (time buckets, time window),
persistence (two Parquet files), append behavior, and soft performance bounds.
"""
from __future__ import annotations

import random
import time as time_mod
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Polygon

from custody import config
from custody.fusion import geo as fusion_geo
from custody.fusion.index import (
    index_observations,
    nearest_observations,
    open_index,
    query_by_aoi,
)
from custody.fusion.observations import (
    PositionObservation,
    PositionVelocityObservation,
)


ANCHOR_LAT = config.AOI_ANCHOR_LAT
ANCHOR_LON = config.AOI_ANCHOR_LON


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pos(
    obs_id: str,
    lat: float = ANCHOR_LAT,
    lon: float = ANCHOR_LON,
    t: float = 1_689_000_000.0,
    sigma_m: float = 50.0,
    notes: dict | None = None,
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
        detector_version="v1",
        classification_conf=0.88,
        vessel_length_est_m=57.0,
        heading_est_deg=270.0,
        notes=notes or {},
    )


def _posvel(
    obs_id: str,
    lat: float = ANCHOR_LAT,
    lon: float = ANCHOR_LON,
    t: float = 1_689_000_000.0,
    v_n: float = 2.5,
    v_e: float = -1.0,
    mmsi: int | None = 123456789,
) -> PositionVelocityObservation:
    cov = np.diag([100.0 ** 2, 100.0 ** 2, 0.5 ** 2, 0.5 ** 2]).astype(float)
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
        mmsi=mmsi,
        vessel_name="TEST VESSEL",
    )


def _aoi_polygon() -> Polygon:
    # Spratly AOI
    return Polygon([
        (114.5, 8.5), (117.5, 8.5), (117.5, 11.0), (114.5, 11.0), (114.5, 8.5),
    ])


# ---------------------------------------------------------------------------
# 1-5. INDEXING ROUND TRIP
# ---------------------------------------------------------------------------


def test_roundtrip_10_position_observations(tmp_path: Path):
    obs = [
        _pos(f"pos-{i}", lat=9.0 + 0.1 * i, lon=115.0 + 0.2 * i)
        for i in range(10)
    ]
    index_observations(obs, out_dir=tmp_path)
    results = query_by_aoi(tmp_path, aoi_polygon=_aoi_polygon())
    assert len(results) == 10


def test_position_type_preserved_through_serialization(tmp_path: Path):
    obs = [_pos("pos-1")]
    index_observations(obs, out_dir=tmp_path)
    results = query_by_aoi(tmp_path, aoi_polygon=_aoi_polygon())
    assert len(results) == 1
    assert isinstance(results[0], PositionObservation)
    assert results[0].obs_id == "pos-1"


def test_posvel_type_preserved_with_cov_intact(tmp_path: Path):
    obs = [_posvel("ais-1")]
    index_observations(obs, out_dir=tmp_path)
    results = query_by_aoi(tmp_path, aoi_polygon=_aoi_polygon())
    assert len(results) == 1
    assert isinstance(results[0], PositionVelocityObservation)
    np.testing.assert_allclose(results[0].cov, obs[0].cov, atol=1e-10)


def test_mixed_types_indexed_and_queried(tmp_path: Path):
    obs = [_pos(f"p-{i}") for i in range(5)] + [_posvel(f"a-{i}") for i in range(3)]
    index_observations(obs, out_dir=tmp_path)
    results = query_by_aoi(tmp_path, aoi_polygon=_aoi_polygon())
    positions = [o for o in results if isinstance(o, PositionObservation)]
    posvels = [o for o in results if isinstance(o, PositionVelocityObservation)]
    assert len(positions) == 5
    assert len(posvels) == 3


def test_covariance_roundtrip_equality_1e10(tmp_path: Path):
    cov_pos_in = np.array([[1234.5, 12.3], [12.3, 5678.9]])
    pos = PositionObservation(
        obs_id="p1", source_id="x", modality="SAR",
        acquisition_time=1_689_000_000.0, ingestion_time=1_689_000_060.0,
        lat=ANCHOR_LAT, lon=ANCHOR_LON,
        cov_pos=cov_pos_in, raw_ref="test://",
    )
    cov_in = np.array([
        [1e4, 1.1, 2.2, 3.3],
        [1.1, 2e4, 4.4, 5.5],
        [2.2, 4.4, 1.0, 0.1],
        [3.3, 5.5, 0.1, 2.0],
    ])
    posvel = PositionVelocityObservation(
        obs_id="a1", source_id="gfw_ais", modality="AIS",
        acquisition_time=1_689_000_000.0, ingestion_time=1_689_000_030.0,
        lat=ANCHOR_LAT, lon=ANCHOR_LON, v_n=1.0, v_e=2.0,
        cov=cov_in, raw_ref="test://",
    )
    index_observations([pos, posvel], out_dir=tmp_path)
    results = query_by_aoi(tmp_path, aoi_polygon=_aoi_polygon())
    by_id = {o.obs_id: o for o in results}
    np.testing.assert_allclose(by_id["p1"].cov_pos, cov_pos_in, atol=1e-10)
    np.testing.assert_allclose(by_id["a1"].cov, cov_in, atol=1e-10)


# ---------------------------------------------------------------------------
# 6-11. SPATIAL QUERIES
# ---------------------------------------------------------------------------


def test_observation_outside_aoi_not_returned(tmp_path: Path):
    inside = _pos("in-1", lat=ANCHOR_LAT, lon=ANCHOR_LON)
    outside = _pos("out-1", lat=0.0, lon=0.0)  # well outside Spratly polygon
    index_observations([inside, outside], out_dir=tmp_path)
    results = query_by_aoi(tmp_path, aoi_polygon=_aoi_polygon())
    assert {o.obs_id for o in results} == {"in-1"}


def test_two_observations_100m_apart_share_h3_r8_cell(tmp_path: Path):
    # 100 m east of anchor via pyproj.Geod is within one r8 cell (~0.53 km edge)
    from pyproj import Geod
    g = Geod(ellps="WGS84")
    lon2, lat2, _ = g.fwd(ANCHOR_LON, ANCHOR_LAT, 90.0, 100.0)
    a = _pos("a", lat=ANCHOR_LAT, lon=ANCHOR_LON)
    b = _pos("b", lat=lat2, lon=lon2)
    import h3
    assert h3.latlng_to_cell(a.lat, a.lon, 8) == h3.latlng_to_cell(b.lat, b.lon, 8)


def test_two_observations_10km_apart_do_not_share_h3_r8_cell():
    from pyproj import Geod
    g = Geod(ellps="WGS84")
    lon2, lat2, _ = g.fwd(ANCHOR_LON, ANCHOR_LAT, 90.0, 10_000.0)
    import h3
    assert h3.latlng_to_cell(ANCHOR_LAT, ANCHOR_LON, 8) != h3.latlng_to_cell(lat2, lon2, 8)


def test_nearest_observations_spatial_and_temporal_bounds(tmp_path: Path):
    # Close: same place, within time window
    near = _pos("near", lat=ANCHOR_LAT, lon=ANCHOR_LON, t=1_689_000_000.0)
    # Far spatially: > 50 km east
    far_space = _pos("far-sp", lat=ANCHOR_LAT, lon=ANCHOR_LON + 0.5, t=1_689_000_000.0)
    # Far temporally: same place, 1 hour later
    far_time = _pos("far-t", lat=ANCHOR_LAT, lon=ANCHOR_LON, t=1_689_003_600.0)
    index_observations([near, far_space, far_time], out_dir=tmp_path)
    hits = nearest_observations(
        tmp_path, lat=ANCHOR_LAT, lon=ANCHOR_LON,
        time=1_689_000_000.0, radius_km=1.0, dt_sec=300.0,
    )
    ids = {o.obs_id for o in hits}
    assert "near" in ids
    assert "far-sp" not in ids
    assert "far-t" not in ids


def test_nearest_observations_radius_zero_returns_same_cell(tmp_path: Path):
    a = _pos("a")
    # ~50 m away, solidly inside the same r8 cell (avoids edge-crossing fragility)
    from pyproj import Geod
    g = Geod(ellps="WGS84")
    lon2, lat2, _ = g.fwd(ANCHOR_LON, ANCHOR_LAT, 90.0, 50.0)
    b = _pos("b", lat=lat2, lon=lon2)
    index_observations([a, b], out_dir=tmp_path)
    hits = nearest_observations(
        tmp_path, lat=ANCHOR_LAT, lon=ANCHOR_LON,
        time=1_689_000_000.0, radius_km=0.0, dt_sec=120.0,
    )
    # Both are in the same cell; both returned
    assert {o.obs_id for o in hits} >= {"a", "b"}


def test_nearest_observations_at_aoi_corner_no_crash(tmp_path: Path):
    corner = _pos("c", lat=11.0, lon=117.5)
    index_observations([corner], out_dir=tmp_path)
    # Call near the corner — verify no exception
    hits = nearest_observations(
        tmp_path, lat=11.0, lon=117.5,
        time=1_689_000_000.0, radius_km=5.0, dt_sec=3600.0,
    )
    assert len(hits) == 1


# ---------------------------------------------------------------------------
# 12-14. TEMPORAL QUERIES
# ---------------------------------------------------------------------------


def test_observation_outside_time_window_not_returned(tmp_path: Path):
    old = _pos("old", t=1_689_000_000.0)
    new = _pos("new", t=1_689_003_600.0)  # 1 hour later
    index_observations([old, new], out_dir=tmp_path)
    results = query_by_aoi(
        tmp_path,
        aoi_polygon=_aoi_polygon(),
        time_window=(1_689_003_000.0, 1_689_010_000.0),
    )
    assert {o.obs_id for o in results} == {"new"}


def test_ais_30s_apart_same_minute_90s_apart_different_minute(tmp_path: Path):
    a = _posvel("a", t=1_689_000_000.0)
    b = _posvel("b", t=1_689_000_030.0)   # 30 s later  → same 1-min bucket
    c = _posvel("c", t=1_689_000_090.0)   # 90 s later  → different bucket
    index_observations([a, b, c], out_dir=tmp_path)
    conn = open_index(tmp_path)
    buckets = conn.execute(
        "SELECT obs_id, time_bucket_minute FROM read_parquet($p) ORDER BY obs_id",
        {"p": str(tmp_path / "posvel.parquet")},
    ).fetchall()
    by_id = dict(buckets)
    assert by_id["a"] == by_id["b"]
    assert by_id["a"] != by_id["c"]


def test_sar_30min_apart_same_hour_2h_apart_different_hour(tmp_path: Path):
    # Align the base timestamp to an hour boundary so "30 min later" stays in the same bucket.
    t_base = (1_689_000_000 // 3600) * 3600   # 1_688_997_600, exact multiple of 3600
    a = _pos("a", t=float(t_base))
    b = _pos("b", t=float(t_base + 1800))     # 30 min later  → same 1h bucket
    c = _pos("c", t=float(t_base + 7200))     # 2 h later    → different bucket
    index_observations([a, b, c], out_dir=tmp_path)
    conn = open_index(tmp_path)
    buckets = conn.execute(
        "SELECT obs_id, time_bucket_hour FROM read_parquet($p) ORDER BY obs_id",
        {"p": str(tmp_path / "position.parquet")},
    ).fetchall()
    by_id = dict(buckets)
    assert by_id["a"] == by_id["b"]
    assert by_id["a"] != by_id["c"]


# ---------------------------------------------------------------------------
# 15-17. PERSISTENCE
# ---------------------------------------------------------------------------


def test_index_writes_expected_parquet_files(tmp_path: Path):
    # Mixed → both files
    mixed_dir = tmp_path / "mixed"
    index_observations([_pos("p1"), _posvel("a1")], out_dir=mixed_dir)
    assert (mixed_dir / "position.parquet").exists()
    assert (mixed_dir / "posvel.parquet").exists()

    # Position-only → only position.parquet
    pos_dir = tmp_path / "pos-only"
    index_observations([_pos("p1")], out_dir=pos_dir)
    assert (pos_dir / "position.parquet").exists()
    assert not (pos_dir / "posvel.parquet").exists()

    # Posvel-only → only posvel.parquet
    pv_dir = tmp_path / "pv-only"
    index_observations([_posvel("a1")], out_dir=pv_dir)
    assert (pv_dir / "posvel.parquet").exists()
    assert not (pv_dir / "position.parquet").exists()


def test_index_reopens_from_disk_in_fresh_connection(tmp_path: Path):
    obs = [_pos(f"p-{i}") for i in range(4)]
    index_observations(obs, out_dir=tmp_path)
    # Simulate a fresh process by calling with a new DuckDB connection under open_index
    results = query_by_aoi(tmp_path, aoi_polygon=_aoi_polygon())
    assert len(results) == 4


def test_append_merges_without_duplicates_or_loss(tmp_path: Path):
    batch_a = [_pos(f"a-{i}") for i in range(5)]
    batch_b = [_pos(f"b-{i}") for i in range(5)]
    index_observations(batch_a, out_dir=tmp_path)
    index_observations(batch_b, out_dir=tmp_path)  # append
    results = query_by_aoi(tmp_path, aoi_polygon=_aoi_polygon())
    ids = {o.obs_id for o in results}
    assert len(ids) == 10
    assert {f"a-{i}" for i in range(5)} <= ids
    assert {f"b-{i}" for i in range(5)} <= ids


# ---------------------------------------------------------------------------
# 18. PERFORMANCE SANITY (soft bounds)
# ---------------------------------------------------------------------------


def test_performance_bounds_are_reasonable(tmp_path: Path):
    rng = random.Random(42)
    obs = []
    for i in range(10_000):
        lat = 8.5 + rng.random() * 2.5
        lon = 114.5 + rng.random() * 3.0
        t = 1_689_000_000.0 + rng.random() * 3600.0
        obs.append(_pos(f"p-{i}", lat=lat, lon=lon, t=t))

    t0 = time_mod.perf_counter()
    index_observations(obs, out_dir=tmp_path)
    idx_sec = time_mod.perf_counter() - t0

    t0 = time_mod.perf_counter()
    results = query_by_aoi(tmp_path, aoi_polygon=_aoi_polygon())
    query_sec = time_mod.perf_counter() - t0

    print(
        f"\n[perf] index 10k: {idx_sec:.2f}s | query {len(results)}: {query_sec:.2f}s"
    )
    # Soft bounds: fail only at 4× the target to catch order-of-magnitude regressions.
    assert idx_sec < 20.0, f"index 10k obs took {idx_sec:.2f}s (target <5s)"
    assert query_sec < 4.0, f"query {len(results)} obs took {query_sec:.2f}s (target <1s)"
