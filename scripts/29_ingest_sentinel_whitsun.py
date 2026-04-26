"""Sentinel observation ingestion CLI for the Whitsun AOI.

Runs a metadata-only search against the Copernicus Data Space Ecosystem
(CDSE) STAC v1 endpoint for Sentinel-1 GRD and Sentinel-2 L2A
observations covering the Whitsun AOI and time window, writes the
normalized observations to a JSON cache, and prints a summary.

Live HTTP is **opt-in**.  Without ``--live``, the script runs in
offline mode and reads the committed demo fixture
``data/demo/whitsun_sentinel_observations.fixture.json``.  Tests and
demo flows that should never reach out to CDSE rely on this offline
default.

This is metadata-only ingestion: no imagery is downloaded, no products
are fetched, no detection is run, and no tasking is issued.

Run::

    # Offline (default): read the committed fixture
    python scripts/29_ingest_sentinel_whitsun.py

    # Live: query CDSE STAC v1 (requires network)
    python scripts/29_ingest_sentinel_whitsun.py --live \\
        --start 2023-12-01 --end 2023-12-15 --max-items 20

    # Override AOI / output paths
    python scripts/29_ingest_sentinel_whitsun.py --live \\
        --aoi data/demo/whitsun_aoi.fixture.geojson \\
        --output data/demo/whitsun_sentinel_observations.json
"""
from __future__ import annotations

import argparse
import json as _json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

from custody.ingest.sentinel import (
    DEFAULT_COLLECTIONS,
    ObservationArtifact,
    best_low_cloud_sentinel_2,
    date_range,
    latest_sentinel_1,
    load_observation_cache,
    search_sentinel_observations,
    write_observation_cache,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AOI_PATH = REPO_ROOT / "data" / "demo" / "whitsun_aoi.fixture.geojson"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "data" / "demo" / "whitsun_sentinel_observations.json"
DEFAULT_FIXTURE_PATH = (
    REPO_ROOT / "data" / "demo" / "whitsun_sentinel_observations.fixture.json"
)


def _load_aoi_geometry(aoi_path: Path) -> dict:
    """Extract a single Polygon/MultiPolygon geometry from a GeoJSON file."""
    blob = aoi_path.read_text(encoding="utf-8")
    obj = _json.loads(blob)
    if obj.get("type") == "FeatureCollection":
        features = obj.get("features") or []
        for feat in features:
            geom = feat.get("geometry") if isinstance(feat, dict) else None
            if geom:
                return dict(geom)
        raise ValueError(f"AOI file {aoi_path} has no usable feature geometry")
    if obj.get("type") == "Feature":
        geom = obj.get("geometry")
        if geom:
            return dict(geom)
        raise ValueError(f"AOI file {aoi_path} feature has no geometry")
    if obj.get("type") in ("Polygon", "MultiPolygon"):
        return dict(obj)
    raise ValueError(
        f"AOI file {aoi_path} is not a Polygon, Feature, or FeatureCollection"
    )


def _summary_lines(
    observations: tuple[ObservationArtifact, ...],
    *,
    mode: str,
    aoi_path: Path,
    output_path: Path | None,
) -> list[str]:
    s1 = [o for o in observations if o.source == "sentinel-1"]
    s2 = [o for o in observations if o.source == "sentinel-2"]
    s2_low = [
        o for o in s2
        if o.cloud_coverage is not None and o.cloud_coverage <= 25.0
    ]
    s2_cloudy = [
        o for o in s2
        if o.cloud_coverage is not None and o.cloud_coverage > 25.0
    ]
    other = [
        o for o in observations
        if o.source not in ("sentinel-1", "sentinel-2")
    ]
    start, end = date_range(observations)
    best_s2 = best_low_cloud_sentinel_2(observations)
    latest_s1 = latest_sentinel_1(observations)

    lines: list[str] = []
    lines.append(f"Mode:                {mode}")
    lines.append(f"AOI:                 {aoi_path}")
    if output_path is not None:
        lines.append(f"Output:              {output_path}")
    lines.append(f"Total observations:  {len(observations)}")
    lines.append(f"  Sentinel-1:        {len(s1)}")
    lines.append(
        f"  Sentinel-2:        {len(s2)}  "
        f"(low-cloud {len(s2_low)}, cloudy {len(s2_cloudy)})"
    )
    if other:
        labels = ", ".join(sorted({o.source for o in other}))
        lines.append(f"  Other sources:     {len(other)} ({labels})")
    lines.append(
        f"Date range:          "
        f"{start or '(none)'}  ->  {end or '(none)'}"
    )
    if best_s2 is not None:
        lines.append(
            f"Best low-cloud S2:   {best_s2.observation_id}  "
            f"cloud={best_s2.cloud_coverage:.0f}%  "
            f"at {best_s2.timestamp or '(unknown)'}"
        )
    else:
        lines.append("Best low-cloud S2:   (none)")
    if latest_s1 is not None:
        lines.append(
            f"Latest Sentinel-1:   {latest_s1.observation_id}  "
            f"at {latest_s1.timestamp or '(unknown)'}"
        )
    else:
        lines.append("Latest Sentinel-1:   (none)")
    return lines


def main(argv: list[str] | None = None, *, out: TextIO | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Sentinel observation ingestion CLI for the Whitsun AOI.  "
            "Metadata-only search against the CDSE STAC v1 endpoint.  "
            "Live HTTP is opt-in via --live; without it, the script "
            "reads the committed demo fixture."
        ),
    )
    parser.add_argument(
        "--aoi", default=str(DEFAULT_AOI_PATH),
        help="Path to a GeoJSON Polygon / Feature / FeatureCollection.",
    )
    parser.add_argument(
        "--output", default=str(DEFAULT_OUTPUT_PATH),
        help="Output cache path for the live mode JSON.",
    )
    parser.add_argument(
        "--fixture", default=str(DEFAULT_FIXTURE_PATH),
        help="Committed demo fixture JSON used when --live is not given.",
    )
    parser.add_argument(
        "--start", default="2023-12-01",
        help="Search window start (ISO date or datetime, UTC).",
    )
    parser.add_argument(
        "--end", default="2023-12-15",
        help="Search window end (ISO date or datetime, UTC).",
    )
    parser.add_argument(
        "--max-items", dest="max_items", type=int, default=20,
        help="Maximum number of items to retrieve from CDSE STAC.",
    )
    parser.add_argument(
        "--collections", default=",".join(DEFAULT_COLLECTIONS),
        help="Comma-separated list of CDSE STAC collection ids.",
    )
    parser.add_argument(
        "--live", action="store_true",
        help=(
            "Make a live request to CDSE STAC.  Without this flag, the "
            "script reads the committed offline fixture."
        ),
    )
    parser.add_argument(
        "--format", choices=("text", "json"), default="text",
        help="Summary output format.  Default: text.",
    )
    args = parser.parse_args(argv)

    sink = out if out is not None else sys.stdout
    aoi_path = Path(args.aoi)

    if args.live:
        try:
            geom = _load_aoi_geometry(aoi_path)
        except (FileNotFoundError, ValueError) as exc:
            sink.write(f"error loading AOI: {exc}\n")
            return 2
        collections = tuple(
            c.strip() for c in args.collections.split(",") if c.strip()
        )
        try:
            observations = search_sentinel_observations(
                geom, args.start, args.end,
                collections=collections,
                max_items=args.max_items,
                data_mode="real",
            )
        except Exception as exc:  # broad: includes requests + parse errors
            sink.write(f"live CDSE STAC query failed: {exc}\n")
            return 1
        output_path = Path(args.output)
        write_observation_cache(
            observations, output_path,
            metadata={
                "data_mode": "real",
                "aoi_path": str(aoi_path),
                "start": args.start,
                "end": args.end,
                "collections": list(collections),
                "max_items": args.max_items,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        mode_label = "live (CDSE STAC v1)"
        cache_path: Path | None = output_path
    else:
        fixture_path = Path(args.fixture)
        try:
            observations = load_observation_cache(fixture_path)
        except FileNotFoundError:
            sink.write(
                f"error: fixture not found at {fixture_path}; pass --live "
                f"or supply --fixture\n"
            )
            return 2
        mode_label = f"offline fixture ({fixture_path})"
        cache_path = None

    if args.format == "json":
        s1 = [o for o in observations if o.source == "sentinel-1"]
        s2 = [o for o in observations if o.source == "sentinel-2"]
        s2_low = [
            o for o in s2
            if o.cloud_coverage is not None and o.cloud_coverage <= 25.0
        ]
        s2_cloudy = [
            o for o in s2
            if o.cloud_coverage is not None and o.cloud_coverage > 25.0
        ]
        start, end = date_range(observations)
        best_s2 = best_low_cloud_sentinel_2(observations)
        latest_s1 = latest_sentinel_1(observations)
        summary = {
            "mode": mode_label,
            "aoi": str(aoi_path),
            "output": str(cache_path) if cache_path is not None else None,
            "total": len(observations),
            "sentinel_1_count": len(s1),
            "sentinel_2_count": len(s2),
            "sentinel_2_low_cloud_count": len(s2_low),
            "sentinel_2_cloudy_count": len(s2_cloudy),
            "date_range": [start, end],
            "best_low_cloud_sentinel_2": (
                best_s2.observation_id if best_s2 is not None else None
            ),
            "latest_sentinel_1": (
                latest_s1.observation_id if latest_s1 is not None else None
            ),
        }
        sink.write(_json.dumps(summary, indent=2))
        sink.write("\n")
    else:
        for line in _summary_lines(
            observations, mode=mode_label,
            aoi_path=aoi_path, output_path=cache_path,
        ):
            sink.write(line + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
