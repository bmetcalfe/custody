"""Sentinel STAC observation ingestion (Sentinel ingestion sprint).

Thin metadata-search layer over the Copernicus Data Space Ecosystem
(CDSE) STAC v1 endpoint.  Returns a normalized :class:`ObservationArtifact`
per discovered Sentinel-1 / Sentinel-2 product so the system can answer:

    "What Sentinel observations are available around the Whitsun AOI
    and time window, and how should they be represented as
    lower-confidence context between Umbra SAR collections?"

Scope
-----

This module performs **metadata-only search and normalization**.  It does
not download imagery, does not fetch full Sentinel products, does not
run detection, and does not interact with any tasking pipeline.

- Endpoint: ``https://stac.dataspace.copernicus.eu/v1/`` (current CDSE
  STAC v1; not the deprecated RESTO/OpenSearch API and not the legacy
  ``catalogue.dataspace.copernicus.eu/stac`` URL).
- Collections supported initially: ``sentinel-1-grd``,
  ``sentinel-2-l2a``.
- Live HTTP is opt-in.  Tests and the offline default path read
  captured / fixture STAC items.
- No new runtime dependencies: this module uses ``requests``,
  ``json``, and ``shapely`` (all already in the project).

Confidence weights are *demo heuristics*, not calibrated reliability
numbers.  See ``docs/sentinel_ingestion_whitsun.md`` for the full
caveat list.
"""
from __future__ import annotations

import hashlib
import json as _json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


CDSE_STAC_BASE_URL = "https://stac.dataspace.copernicus.eu/v1"
CDSE_STAC_SEARCH_URL = f"{CDSE_STAC_BASE_URL}/search"

SENTINEL_1_GRD_COLLECTION = "sentinel-1-grd"
SENTINEL_2_L2A_COLLECTION = "sentinel-2-l2a"

DEFAULT_COLLECTIONS: tuple[str, ...] = (
    SENTINEL_1_GRD_COLLECTION,
    SENTINEL_2_L2A_COLLECTION,
)

# Cloud-cover threshold (percent) for treating Sentinel-2 imagery as
# usable for detection vs context-only.  Demo heuristic.
S2_LOW_CLOUD_PCT = 25.0

# Confidence weight defaults.  These are *demo heuristics*, not
# calibrated sensor reliability numbers.  Treat as starter values that
# should be replaced once calibrated weights exist.
CONFIDENCE_WEIGHTS: Mapping[str, float] = {
    "umbra": 1.00,
    "sentinel-1": 0.45,
    "sentinel-2-low-cloud": 0.30,
    "sentinel-2-cloudy": 0.10,
    "simulated": 0.50,
}

_BASE_CAVEATS: tuple[str, ...] = (
    "metadata-only ingestion; no imagery is downloaded",
    "confidence weights are demo heuristics, not calibrated sensor "
    "reliability values",
    "Sentinel observations are public lower-confidence context, not "
    "tasked Umbra collections",
)

_VALID_SOURCES: tuple[str, ...] = (
    "umbra", "sentinel-1", "sentinel-2", "simulated",
)
_VALID_DATA_MODES: tuple[str, ...] = ("real", "fixture", "simulated")
_VALID_SENSOR_TYPES: tuple[str, ...] = ("sar", "optical", "multispectral")


# ---------------------------------------------------------------------------
# Value type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObservationArtifact:
    """Normalized Sentinel / Umbra / simulated observation record."""

    observation_id: str
    source: str                    # one of _VALID_SOURCES
    provider: str                  # e.g. "cdse-stac", "umbra-odp", "fixture"
    collection: str | None
    product_id: str | None
    timestamp: str | None          # ISO 8601 acquisition midpoint, if available
    datetime_start: str | None
    datetime_end: str | None
    sensor_type: str               # one of _VALID_SENSOR_TYPES
    platform: str | None
    instrument: str | None
    resolution_m: float | None
    cloud_coverage: float | None   # percent 0-100, if applicable
    polarization: tuple[str, ...]
    orbit_direction: str | None    # "ascending" / "descending" / None
    geometry: Mapping[str, Any] | None
    bbox: tuple[float, float, float, float] | None
    asset_links: Mapping[str, str]
    thumbnail_url: str | None
    product_url: str | None
    confidence_weight: float
    usable_for_detection: bool
    usable_for_context: bool
    data_mode: str                 # one of _VALID_DATA_MODES
    caveats: tuple[str, ...]
    raw_properties: Mapping[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stable_id(prefix: str, *parts: str) -> str:
    blob = "|".join(parts)
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _classify_collection(collection: str | None) -> tuple[str, str]:
    """Return ``(source, sensor_type)`` for a STAC collection id."""
    if collection is None:
        return ("simulated", "sar")
    cid = collection.lower()
    if cid.startswith("sentinel-1") or cid.startswith("s1"):
        return ("sentinel-1", "sar")
    if cid.startswith("sentinel-2") or cid.startswith("s2"):
        return ("sentinel-2", "optical")
    if cid.startswith("umbra"):
        return ("umbra", "sar")
    return ("simulated", "sar")


def _clean_polarization(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        return (raw.upper(),)
    if isinstance(raw, Iterable):
        return tuple(str(p).upper() for p in raw)
    return ()


def _bbox_from_geometry(
    geometry: Mapping[str, Any] | None,
) -> tuple[float, float, float, float] | None:
    if geometry is None:
        return None
    coords = geometry.get("coordinates")
    if not coords:
        return None
    try:
        from shapely.geometry import shape  # type: ignore
        bounds = shape(geometry).bounds
        return (float(bounds[0]), float(bounds[1]),
                float(bounds[2]), float(bounds[3]))
    except Exception:
        return None


def _resolution_for(source: str, properties: Mapping[str, Any]) -> float | None:
    """Pick a resolution proxy from STAC properties when available."""
    for key in ("gsd", "eo:gsd", "s1:resolution", "s2:gsd"):
        v = properties.get(key)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    if source == "sentinel-1":
        return 10.0  # canonical GRD pixel spacing for IW mode
    if source == "sentinel-2":
        return 10.0  # default 10 m bands
    return None


def _decide_usability_and_weight(
    source: str,
    cloud_coverage: float | None,
) -> tuple[float, bool, bool, str]:
    """Return ``(weight, usable_for_detection, usable_for_context, label)``."""
    if source == "umbra":
        return CONFIDENCE_WEIGHTS["umbra"], True, True, "umbra"
    if source == "sentinel-1":
        return CONFIDENCE_WEIGHTS["sentinel-1"], True, True, "sentinel-1"
    if source == "sentinel-2":
        if cloud_coverage is not None and cloud_coverage > S2_LOW_CLOUD_PCT:
            return (
                CONFIDENCE_WEIGHTS["sentinel-2-cloudy"],
                False, True, "sentinel-2-cloudy",
            )
        return (
            CONFIDENCE_WEIGHTS["sentinel-2-low-cloud"],
            True, True, "sentinel-2-low-cloud",
        )
    return CONFIDENCE_WEIGHTS["simulated"], False, True, "simulated"


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def normalize_stac_item(
    item: Mapping[str, Any],
    *,
    data_mode: str = "real",
) -> ObservationArtifact:
    """Convert a CDSE STAC v1 Item dict into an :class:`ObservationArtifact`.

    ``data_mode`` should be ``"real"`` for live STAC search results,
    ``"fixture"`` for committed demo records, and ``"simulated"`` for
    synthesised records.
    """
    if data_mode not in _VALID_DATA_MODES:
        raise ValueError(
            f"data_mode must be one of {_VALID_DATA_MODES}; got {data_mode!r}"
        )

    properties: Mapping[str, Any] = item.get("properties") or {}
    collection_id = item.get("collection")
    source, sensor_type = _classify_collection(collection_id)

    raw_id = item.get("id") or properties.get("title") or properties.get("product_id")
    observation_id = (
        str(raw_id) if raw_id else _stable_id(source, _json.dumps(item, sort_keys=True, default=str))
    )

    datetime_start = (
        properties.get("start_datetime")
        or properties.get("datetime")
    )
    datetime_end = properties.get("end_datetime") or properties.get("datetime")
    timestamp = properties.get("datetime") or datetime_start

    cloud_raw = (
        properties.get("eo:cloud_cover")
        or properties.get("cloud_cover")
        or properties.get("s2:cloud_cover")
    )
    cloud_coverage: float | None
    if cloud_raw is None:
        cloud_coverage = None
    else:
        try:
            cloud_coverage = float(cloud_raw)
        except (TypeError, ValueError):
            cloud_coverage = None

    polarization = _clean_polarization(
        properties.get("sar:polarizations")
        or properties.get("polarization")
        or properties.get("polarizations")
    )

    orbit_direction_raw = (
        properties.get("sat:orbit_state")
        or properties.get("sar:orbit_state")
        or properties.get("orbit_direction")
    )
    if isinstance(orbit_direction_raw, str):
        orbit_direction: str | None = orbit_direction_raw.lower() or None
    else:
        orbit_direction = None

    geometry = item.get("geometry")
    bbox_raw = item.get("bbox")
    bbox: tuple[float, float, float, float] | None
    if isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) >= 4:
        bbox = (
            float(bbox_raw[0]), float(bbox_raw[1]),
            float(bbox_raw[2]), float(bbox_raw[3]),
        )
    else:
        bbox = _bbox_from_geometry(geometry)

    assets_raw = item.get("assets") or {}
    asset_links: dict[str, str] = {}
    thumbnail_url: str | None = None
    product_url: str | None = None
    if isinstance(assets_raw, Mapping):
        for key, asset in assets_raw.items():
            if not isinstance(asset, Mapping):
                continue
            href = asset.get("href")
            if not href:
                continue
            asset_links[str(key)] = str(href)
            roles = asset.get("roles") or []
            lower_key = str(key).lower()
            if thumbnail_url is None and (
                "thumbnail" in roles or lower_key in ("thumbnail", "preview", "quicklook")
            ):
                thumbnail_url = str(href)
            if product_url is None and (
                "data" in roles or lower_key in ("product", "data")
            ):
                product_url = str(href)

    if product_url is None:
        for link in item.get("links") or []:
            if isinstance(link, Mapping) and link.get("rel") == "self":
                href = link.get("href")
                if href:
                    product_url = str(href)
                    break

    confidence_weight, usable_det, usable_ctx, weight_label = _decide_usability_and_weight(
        source, cloud_coverage,
    )

    caveats: list[str] = list(_BASE_CAVEATS)
    caveats.append(f"confidence weight bucket: {weight_label}")
    if data_mode == "fixture":
        caveats.append("fixture record; not a live STAC query result")
    elif data_mode == "simulated":
        caveats.append("simulated record; no upstream provider")
    if source == "sentinel-2" and cloud_coverage is not None and cloud_coverage > S2_LOW_CLOUD_PCT:
        caveats.append(
            f"Sentinel-2 cloud coverage {cloud_coverage:.0f}% exceeds "
            f"{S2_LOW_CLOUD_PCT:.0f}%; usable_for_context only"
        )

    return ObservationArtifact(
        observation_id=str(observation_id),
        source=source,
        provider=str(properties.get("provider") or "cdse-stac"),
        collection=str(collection_id) if collection_id is not None else None,
        product_id=str(properties.get("product_id") or properties.get("title") or raw_id) if (
            raw_id or properties.get("product_id") or properties.get("title")
        ) else None,
        timestamp=str(timestamp) if timestamp else None,
        datetime_start=str(datetime_start) if datetime_start else None,
        datetime_end=str(datetime_end) if datetime_end else None,
        sensor_type=sensor_type,
        platform=str(properties.get("platform")) if properties.get("platform") else None,
        instrument=str(properties.get("instruments")[0])
        if isinstance(properties.get("instruments"), list) and properties.get("instruments")
        else (
            str(properties.get("instrument"))
            if properties.get("instrument") else None
        ),
        resolution_m=_resolution_for(source, properties),
        cloud_coverage=cloud_coverage,
        polarization=polarization,
        orbit_direction=orbit_direction,
        geometry=dict(geometry) if isinstance(geometry, Mapping) else None,
        bbox=bbox,
        asset_links=dict(asset_links),
        thumbnail_url=thumbnail_url,
        product_url=product_url,
        confidence_weight=confidence_weight,
        usable_for_detection=usable_det,
        usable_for_context=usable_ctx,
        data_mode=data_mode,
        caveats=tuple(caveats),
        raw_properties=dict(properties),
    )


# ---------------------------------------------------------------------------
# STAC client
# ---------------------------------------------------------------------------


def search_sentinel_observations(
    aoi_geojson: Mapping[str, Any],
    start_time: str,
    end_time: str,
    *,
    collections: Iterable[str] = DEFAULT_COLLECTIONS,
    max_items: int = 20,
    base_url: str = CDSE_STAC_BASE_URL,
    timeout_s: float = 30.0,
    session: requests.Session | None = None,
    data_mode: str = "real",
) -> tuple[ObservationArtifact, ...]:
    """Search CDSE STAC v1 for Sentinel observations and normalize the result.

    Performs **one metadata-only POST** to ``{base_url}/search``.  Does
    not download imagery.  Pagination is handled up to ``max_items`` results.

    ``aoi_geojson`` should be a GeoJSON Geometry dict (e.g. a Polygon).

    Raises :class:`requests.HTTPError` on non-2xx responses.
    """
    if data_mode not in _VALID_DATA_MODES:
        raise ValueError(
            f"data_mode must be one of {_VALID_DATA_MODES}; got {data_mode!r}"
        )

    payload = {
        "collections": list(collections),
        "intersects": dict(aoi_geojson),
        "datetime": f"{start_time}/{end_time}",
        "limit": int(max_items),
    }

    sess = session or requests.Session()
    search_url = f"{base_url.rstrip('/')}/search"
    resp = sess.post(search_url, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    body = resp.json()

    features = body.get("features") or []
    out = [normalize_stac_item(f, data_mode=data_mode) for f in features[:max_items]]
    return tuple(out)


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------


def _artifact_to_dict(artifact: ObservationArtifact) -> dict:
    return {
        "observation_id": artifact.observation_id,
        "source": artifact.source,
        "provider": artifact.provider,
        "collection": artifact.collection,
        "product_id": artifact.product_id,
        "timestamp": artifact.timestamp,
        "datetime_start": artifact.datetime_start,
        "datetime_end": artifact.datetime_end,
        "sensor_type": artifact.sensor_type,
        "platform": artifact.platform,
        "instrument": artifact.instrument,
        "resolution_m": artifact.resolution_m,
        "cloud_coverage": artifact.cloud_coverage,
        "polarization": list(artifact.polarization),
        "orbit_direction": artifact.orbit_direction,
        "geometry": artifact.geometry,
        "bbox": list(artifact.bbox) if artifact.bbox is not None else None,
        "asset_links": dict(artifact.asset_links),
        "thumbnail_url": artifact.thumbnail_url,
        "product_url": artifact.product_url,
        "confidence_weight": artifact.confidence_weight,
        "usable_for_detection": artifact.usable_for_detection,
        "usable_for_context": artifact.usable_for_context,
        "data_mode": artifact.data_mode,
        "caveats": list(artifact.caveats),
        "raw_properties": dict(artifact.raw_properties),
    }


def _dict_to_artifact(record: Mapping[str, Any]) -> ObservationArtifact:
    bbox_raw = record.get("bbox")
    bbox: tuple[float, float, float, float] | None
    if isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) >= 4:
        bbox = (
            float(bbox_raw[0]), float(bbox_raw[1]),
            float(bbox_raw[2]), float(bbox_raw[3]),
        )
    else:
        bbox = None
    return ObservationArtifact(
        observation_id=str(record["observation_id"]),
        source=str(record["source"]),
        provider=str(record["provider"]),
        collection=record.get("collection"),
        product_id=record.get("product_id"),
        timestamp=record.get("timestamp"),
        datetime_start=record.get("datetime_start"),
        datetime_end=record.get("datetime_end"),
        sensor_type=str(record.get("sensor_type") or "sar"),
        platform=record.get("platform"),
        instrument=record.get("instrument"),
        resolution_m=record.get("resolution_m"),
        cloud_coverage=record.get("cloud_coverage"),
        polarization=tuple(record.get("polarization") or ()),
        orbit_direction=record.get("orbit_direction"),
        geometry=record.get("geometry"),
        bbox=bbox,
        asset_links=dict(record.get("asset_links") or {}),
        thumbnail_url=record.get("thumbnail_url"),
        product_url=record.get("product_url"),
        confidence_weight=float(record.get("confidence_weight") or 0.0),
        usable_for_detection=bool(record.get("usable_for_detection")),
        usable_for_context=bool(record.get("usable_for_context")),
        data_mode=str(record.get("data_mode") or "real"),
        caveats=tuple(record.get("caveats") or ()),
        raw_properties=dict(record.get("raw_properties") or {}),
    )


def write_observation_cache(
    observations: Iterable[ObservationArtifact],
    output_path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Write a normalized observation list to a JSON file.

    The output file has the shape::

        {
            "metadata": { "generated_at": ..., "data_mode": ..., ... },
            "observations": [ {...}, {...}, ... ]
        }

    ``metadata`` is merged into the top-level ``metadata`` block.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    obs_list = list(observations)
    base_meta: dict[str, Any] = {
        "schema": "custody.ingest.sentinel.v1",
        "count": len(obs_list),
    }
    if metadata is not None:
        base_meta.update(dict(metadata))
    payload = {
        "metadata": base_meta,
        "observations": [_artifact_to_dict(o) for o in obs_list],
    }
    out_path.write_text(_json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def load_observation_cache(
    path: str | Path,
) -> tuple[ObservationArtifact, ...]:
    """Read a cache file written by :func:`write_observation_cache`."""
    blob = Path(path).read_text(encoding="utf-8")
    payload = _json.loads(blob)
    records = payload.get("observations") if isinstance(payload, Mapping) else None
    if records is None and isinstance(payload, list):
        records = payload
    return tuple(_dict_to_artifact(r) for r in (records or ()))


# ---------------------------------------------------------------------------
# Demo helpers
# ---------------------------------------------------------------------------


def best_low_cloud_sentinel_2(
    observations: Iterable[ObservationArtifact],
) -> ObservationArtifact | None:
    """Return the lowest-cloud Sentinel-2 record, or ``None``."""
    s2 = [
        o for o in observations
        if o.source == "sentinel-2" and o.cloud_coverage is not None
    ]
    if not s2:
        return None
    return min(s2, key=lambda o: (o.cloud_coverage or 100.0, o.observation_id))


def latest_sentinel_1(
    observations: Iterable[ObservationArtifact],
) -> ObservationArtifact | None:
    """Return the latest Sentinel-1 record by timestamp, or ``None``."""
    s1 = [o for o in observations if o.source == "sentinel-1" and o.timestamp]
    if not s1:
        return None
    return max(s1, key=lambda o: (o.timestamp or "", o.observation_id))


def date_range(
    observations: Iterable[ObservationArtifact],
) -> tuple[str | None, str | None]:
    """Return ``(min_timestamp, max_timestamp)`` ISO strings, or ``(None, None)``."""
    ts = [o.timestamp for o in observations if o.timestamp]
    if not ts:
        return (None, None)
    return (min(ts), max(ts))
