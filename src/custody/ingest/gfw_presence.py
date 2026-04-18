"""Parser: Global Fishing Watch v3 4wings presence response → PositionObservation list.

This module ingests GFW's public-tier presence dataset, which is derived from
AIS broadcasts but exposed as per-cell-per-hour aggregates.  For raw AIS with
SOG/COG, upgraded GFW access or a different data source is required.  See
[ADR-0011](../../../docs/decisions/0011-gfw-presence-as-position-only.md).

Input: the JSON body returned by::

    POST /v3/4wings/report?datasets[0]=public-global-presence:latest
        &temporal-resolution=HOURLY&group-by=VESSEL_ID
    body: {"geojson": <polygon>}

Output: a list of :class:`PositionObservation` with ``modality="AIS"`` and
covariance reflecting the grid-cell quantization of the source data (not
GPS-grade), plus a :class:`DropReport` tallying discarded records by reason.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

import numpy as np

from custody.fusion.observations import PositionObservation


# 1σ position uncertainty for GFW presence records: half the ~1 km
# quantization cell.  See ADR-0011.
GFW_PRESENCE_POS_SIGMA_M = 500.0

# GFW / AIS sentinel values for "not available."
LAT_NOT_AVAILABLE = 91.0
LON_NOT_AVAILABLE = 181.0

# Dataset key inside the `entries[*]` objects of a 4wings/report response.
_PRESENCE_KEY_PREFIX = "public-global-presence:"


@dataclass
class DropReport:
    """Counts of records dropped by reason during parsing."""
    invalid_position: int = 0
    missing_mmsi: int = 0
    missing_timestamp: int = 0
    duplicate: int = 0

    @property
    def total(self) -> int:
        return (
            self.invalid_position
            + self.missing_mmsi
            + self.missing_timestamp
            + self.duplicate
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "invalid_position": self.invalid_position,
            "missing_mmsi": self.missing_mmsi,
            "missing_timestamp": self.missing_timestamp,
            "duplicate": self.duplicate,
            "total": self.total,
        }


# ---------------------------------------------------------------------------
# Core parsing
# ---------------------------------------------------------------------------


def _iter_presence_records(response: dict) -> Iterable[dict]:
    """Yield every per-vessel record inside ``response['entries'][*][<dataset-key>]``."""
    if "entries" not in response:
        raise ValueError("GFW response missing top-level 'entries' key")
    for entry in response["entries"]:
        if not isinstance(entry, dict):
            continue
        for key, payload in entry.items():
            if key.startswith(_PRESENCE_KEY_PREFIX) and isinstance(payload, list):
                for rec in payload:
                    if isinstance(rec, dict):
                        yield rec


def _parse_timestamp(date_str: str) -> float | None:
    """Parse GFW ``date`` strings into UTC epoch seconds.

    Accepted forms: ``"2023-07-02"``, ``"2023-07-02 17:00"``,
    ``"2023-07-02T17:00:00Z"`` (and variants with seconds).
    Returns ``None`` if the string can't be parsed.
    """
    if not date_str or not isinstance(date_str, str):
        return None
    candidates = [
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
    ]
    for fmt in candidates:
        try:
            dt = datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            continue
    return None


def _position_cov() -> np.ndarray:
    """2×2 isotropic position covariance in m² at GFW_PRESENCE_POS_SIGMA_M."""
    s2 = GFW_PRESENCE_POS_SIGMA_M ** 2
    return np.array([[s2, 0.0], [0.0, s2]], dtype=float)


def _record_to_observation(
    rec: dict, drops: DropReport, seen: set[tuple[str, int]]
) -> PositionObservation | None:
    mmsi = rec.get("mmsi")
    if not mmsi:
        drops.missing_mmsi += 1
        return None

    ts = _parse_timestamp(rec.get("date", ""))
    if ts is None:
        drops.missing_timestamp += 1
        return None

    lat = rec.get("lat")
    lon = rec.get("lon")
    if lat is None or lon is None:
        drops.invalid_position += 1
        return None
    if lat == LAT_NOT_AVAILABLE or lon == LON_NOT_AVAILABLE:
        drops.invalid_position += 1
        return None
    if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
        drops.invalid_position += 1
        return None

    key = (str(mmsi), int(ts))
    if key in seen:
        drops.duplicate += 1
        return None
    seen.add(key)

    notes: dict[str, Any] = {}
    for k in ("flag", "vesselType", "geartype", "vesselId", "shipName",
              "callsign", "imo", "hours",
              "firstTransmissionDate", "lastTransmissionDate",
              "entryTimestamp", "exitTimestamp"):
        if k in rec and rec[k] not in (None, ""):
            notes[k] = rec[k]

    return PositionObservation(
        obs_id=f"gfw-presence-{mmsi}-{int(ts)}",
        source_id="gfw_presence",
        modality="AIS",
        acquisition_time=ts,
        ingestion_time=ts,  # we do not learn ingestion time from the response body
        lat=float(lat),
        lon=float(lon),
        cov_pos=_position_cov(),
        raw_ref=f"gfw://presence/mmsi={mmsi}/ts={int(ts)}",
        detector_version=None,
        classification_conf=None,
        vessel_length_est_m=None,
        heading_est_deg=None,
        notes=notes,
    )


def parse_gfw_presence_response(
    response: dict,
) -> tuple[list[PositionObservation], DropReport]:
    """Parse a GFW 4wings/report presence response into PositionObservations.

    Drops records whose MMSI, timestamp, or position is missing/invalid, plus
    duplicates sharing the same ``(mmsi, timestamp_epoch_seconds)`` pair.
    Returns the observations list and a :class:`DropReport` summarising drops.
    """
    if not isinstance(response, dict):
        raise ValueError(f"GFW response must be a dict, got {type(response).__name__}")

    drops = DropReport()
    seen: set[tuple[str, int]] = set()
    observations: list[PositionObservation] = []
    for rec in _iter_presence_records(response):
        obs = _record_to_observation(rec, drops, seen)
        if obs is not None:
            observations.append(obs)
    return observations, drops
