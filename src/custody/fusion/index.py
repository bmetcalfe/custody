"""H3 + DuckDB + Parquet spatial/temporal index for fused observations.

Backing store: two Parquet files under an ``out_dir`` — ``position.parquet``
for :class:`PositionObservation` (2×2 position covariance, 1-hour time
buckets) and ``posvel.parquet`` for :class:`PositionVelocityObservation`
(4×4 covariance serialized as upper triangle, 1-minute time buckets).

DuckDB is used as a thin query engine against the Parquet files; there is
no persistent ``.duckdb`` file.  Callers can reach through
:func:`open_index` for ad-hoc queries.

Spatial indexing uses H3 resolution 8 (~0.53 km average edge) per ADR-0009.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import duckdb
import h3
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

from custody.fusion.observations import (
    Observation,
    PositionObservation,
    PositionVelocityObservation,
)


POSITION_FILE = "position.parquet"
POSVEL_FILE = "posvel.parquet"

_H3_RES = 8
# Average r8 edge length; used to translate a radius_km request into a k-ring count.
_H3_EDGE_KM = h3.average_hexagon_edge_length(_H3_RES, unit="km")


# ---------------------------------------------------------------------------
# Row building
# ---------------------------------------------------------------------------


def _position_row(obs: PositionObservation) -> dict[str, Any]:
    cov = np.asarray(obs.cov_pos, dtype=float)
    return {
        "obs_id": obs.obs_id,
        "source_id": obs.source_id,
        "modality": obs.modality,
        "acquisition_time": obs.acquisition_time,
        "ingestion_time": obs.ingestion_time,
        "lat": obs.lat,
        "lon": obs.lon,
        "cov_xx": float(cov[0, 0]),
        "cov_xy": float(cov[0, 1]),
        "cov_yy": float(cov[1, 1]),
        "raw_ref": obs.raw_ref,
        "detector_version": obs.detector_version,
        "classification_conf": obs.classification_conf,
        "vessel_length_est_m": obs.vessel_length_est_m,
        "heading_est_deg": obs.heading_est_deg,
        "notes_json": json.dumps(obs.notes),
        "h3_cell_r8": h3.latlng_to_cell(obs.lat, obs.lon, _H3_RES),
        "time_bucket_hour": int(obs.acquisition_time // 3600),
    }


def _posvel_row(obs: PositionVelocityObservation) -> dict[str, Any]:
    cov = np.asarray(obs.cov, dtype=float)
    row = {
        "obs_id": obs.obs_id,
        "source_id": obs.source_id,
        "modality": obs.modality,
        "acquisition_time": obs.acquisition_time,
        "ingestion_time": obs.ingestion_time,
        "lat": obs.lat,
        "lon": obs.lon,
        "v_n": obs.v_n,
        "v_e": obs.v_e,
        "raw_ref": obs.raw_ref,
        "mmsi": obs.mmsi,
        "vessel_name": obs.vessel_name,
        "notes_json": json.dumps(obs.notes),
        "h3_cell_r8": h3.latlng_to_cell(obs.lat, obs.lon, _H3_RES),
        "time_bucket_minute": int(obs.acquisition_time // 60),
    }
    # Upper-triangle of the 4×4 covariance — 10 entries (i, j) with i ≤ j.
    for i in range(4):
        for j in range(i, 4):
            row[f"cov_{i}{j}"] = float(cov[i, j])
    return row


def _position_from_row(row: dict[str, Any]) -> PositionObservation:
    cov = np.array([
        [row["cov_xx"], row["cov_xy"]],
        [row["cov_xy"], row["cov_yy"]],
    ])
    return PositionObservation(
        obs_id=row["obs_id"],
        source_id=row["source_id"],
        modality=row["modality"],
        acquisition_time=row["acquisition_time"],
        ingestion_time=row["ingestion_time"],
        lat=row["lat"],
        lon=row["lon"],
        cov_pos=cov,
        raw_ref=row["raw_ref"],
        detector_version=row["detector_version"],
        classification_conf=row["classification_conf"],
        vessel_length_est_m=row["vessel_length_est_m"],
        heading_est_deg=row["heading_est_deg"],
        notes=json.loads(row["notes_json"] or "{}"),
    )


def _posvel_from_row(row: dict[str, Any]) -> PositionVelocityObservation:
    cov = np.zeros((4, 4))
    for i in range(4):
        for j in range(i, 4):
            v = row[f"cov_{i}{j}"]
            cov[i, j] = v
            cov[j, i] = v
    return PositionVelocityObservation(
        obs_id=row["obs_id"],
        source_id=row["source_id"],
        modality=row["modality"],
        acquisition_time=row["acquisition_time"],
        ingestion_time=row["ingestion_time"],
        lat=row["lat"],
        lon=row["lon"],
        v_n=row["v_n"],
        v_e=row["v_e"],
        cov=cov,
        raw_ref=row["raw_ref"],
        mmsi=row["mmsi"],
        vessel_name=row["vessel_name"],
        notes=json.loads(row["notes_json"] or "{}"),
    )


# ---------------------------------------------------------------------------
# Write side
# ---------------------------------------------------------------------------


def _append_parquet(new_rows: list[dict[str, Any]], path: Path) -> None:
    """Write ``new_rows`` to ``path``, appending to any existing Parquet content."""
    if not new_rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new_tbl = pa.Table.from_pylist(new_rows)
    if path.exists():
        existing = pq.read_table(path)
        merged = pa.concat_tables([existing, new_tbl], promote_options="default")
        pq.write_table(merged, path)
    else:
        pq.write_table(new_tbl, path)


def index_observations(
    observations: list[Observation], out_dir: Path
) -> dict[str, Any]:
    """Write observations to Parquet files under ``out_dir``.

    Returns a summary dict with per-type counts and output paths.  Appends to
    existing files when present (merge semantics — no dedup, no overwrite).
    """
    out_dir = Path(out_dir)
    pos_rows: list[dict[str, Any]] = []
    posvel_rows: list[dict[str, Any]] = []
    for obs in observations:
        if isinstance(obs, PositionObservation):
            pos_rows.append(_position_row(obs))
        elif isinstance(obs, PositionVelocityObservation):
            posvel_rows.append(_posvel_row(obs))
        else:
            raise TypeError(
                f"index_observations: unknown Observation type {type(obs).__name__}"
            )
    pos_path = out_dir / POSITION_FILE
    posvel_path = out_dir / POSVEL_FILE
    _append_parquet(pos_rows, pos_path)
    _append_parquet(posvel_rows, posvel_path)
    return {
        "out_dir": out_dir,
        "position_count": len(pos_rows),
        "posvel_count": len(posvel_rows),
        "position_path": pos_path if pos_rows or pos_path.exists() else None,
        "posvel_path": posvel_path if posvel_rows or posvel_path.exists() else None,
    }


# ---------------------------------------------------------------------------
# Read side
# ---------------------------------------------------------------------------


def open_index(out_dir: Path) -> duckdb.DuckDBPyConnection:
    """Return a fresh in-memory DuckDB connection. Callers query the Parquet
    files under ``out_dir`` directly via ``read_parquet('<path>')``."""
    conn = duckdb.connect(database=":memory:")
    return conn


def _read_parquet_rows(conn: duckdb.DuckDBPyConnection, path: Path) -> list[dict]:
    if not path.exists():
        return []
    cur = conn.execute(f"SELECT * FROM read_parquet($p)", {"p": str(path)})
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _filter_time(
    rows: list[dict], time_window: Optional[tuple[float, float]]
) -> list[dict]:
    if time_window is None:
        return rows
    t0, t1 = time_window
    return [r for r in rows if t0 <= r["acquisition_time"] <= t1]


def _filter_polygon(rows: list[dict], polygon: BaseGeometry) -> list[dict]:
    return [r for r in rows if polygon.contains(Point(r["lon"], r["lat"]))]


def query_by_aoi(
    out_dir: Path,
    aoi_polygon: BaseGeometry,
    time_window: Optional[tuple[float, float]] = None,
) -> list[Observation]:
    """Return every observation under ``out_dir`` whose lat/lon falls inside
    ``aoi_polygon`` and (optionally) whose acquisition time falls in
    ``time_window`` = ``(t_start, t_end)``."""
    out_dir = Path(out_dir)
    conn = open_index(out_dir)
    try:
        pos_rows = _filter_polygon(
            _filter_time(_read_parquet_rows(conn, out_dir / POSITION_FILE), time_window),
            aoi_polygon,
        )
        posvel_rows = _filter_polygon(
            _filter_time(_read_parquet_rows(conn, out_dir / POSVEL_FILE), time_window),
            aoi_polygon,
        )
    finally:
        conn.close()
    out: list[Observation] = [_position_from_row(r) for r in pos_rows]
    out.extend(_posvel_from_row(r) for r in posvel_rows)
    return out


def nearest_observations(
    out_dir: Path,
    lat: float,
    lon: float,
    time: float,
    radius_km: float,
    dt_sec: float,
) -> list[Observation]:
    """Return observations within ``radius_km`` of (lat, lon) AND within
    ``dt_sec`` of ``time``. H3 r8 k-ring filter plus exact time window."""
    out_dir = Path(out_dir)
    center = h3.latlng_to_cell(lat, lon, _H3_RES)
    k = max(0, int(np.ceil(radius_km / _H3_EDGE_KM)))
    cells = set(h3.grid_disk(center, k))
    t0, t1 = time - dt_sec, time + dt_sec

    conn = open_index(out_dir)
    try:
        pos_rows = [
            r for r in _read_parquet_rows(conn, out_dir / POSITION_FILE)
            if r["h3_cell_r8"] in cells and t0 <= r["acquisition_time"] <= t1
        ]
        posvel_rows = [
            r for r in _read_parquet_rows(conn, out_dir / POSVEL_FILE)
            if r["h3_cell_r8"] in cells and t0 <= r["acquisition_time"] <= t1
        ]
    finally:
        conn.close()
    out: list[Observation] = [_position_from_row(r) for r in pos_rows]
    out.extend(_posvel_from_row(r) for r in posvel_rows)
    return out
