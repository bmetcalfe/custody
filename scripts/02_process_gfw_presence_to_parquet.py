"""Process cached GFW presence JSON → Parquet + summary JSON.

Reads every ``data/raw/gfw_presence/*.json`` chunk produced by
``01_fetch_gfw_presence.py``, parses each via
:func:`custody.ingest.gfw_presence.parse_gfw_presence_response`, flushes
the combined ``PositionObservation`` list through
:func:`custody.fusion.index.index_observations` to
``data/processed/gfw_presence/`` as Parquet, and writes an
``observations_summary.json`` receipt with counts, time range, unique MMSI
count, and per-reason drops.

Idempotent over the Parquet layer via :mod:`custody.fusion.index` append
semantics — re-running is safe but will grow the Parquet file. Delete
``data/processed/gfw_presence/`` first for a clean regen.

Run::

    python scripts/02_process_gfw_presence_to_parquet.py
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make src/custody importable when invoked as a bare script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from custody.fusion.index import index_observations  # noqa: E402
from custody.ingest.gfw_presence import parse_gfw_presence_response, DropReport  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw" / "gfw_presence"
OUT_DIR = REPO_ROOT / "data" / "processed" / "gfw_presence"


def _iso(ts_epoch: float) -> str:
    return datetime.fromtimestamp(ts_epoch, tz=timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=str(RAW_DIR))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    if not raw_dir.exists():
        raise SystemExit(f"raw dir not found: {raw_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    chunk_files = sorted(raw_dir.glob("*.json"))
    if not chunk_files:
        raise SystemExit(f"no chunk JSON files found under {raw_dir}")

    all_observations: list = []
    merged_drops = DropReport()

    for chunk_path in chunk_files:
        with chunk_path.open() as f:
            response = json.load(f)
        obs_list, drops = parse_gfw_presence_response(response)
        all_observations.extend(obs_list)
        merged_drops.invalid_position += drops.invalid_position
        merged_drops.missing_mmsi += drops.missing_mmsi
        merged_drops.missing_timestamp += drops.missing_timestamp
        merged_drops.duplicate += drops.duplicate
        print(
            f"  {chunk_path.name}: +{len(obs_list)} obs, drops={drops.as_dict()}"
        )

    # Write Parquet via the v3 fusion index.
    index_observations(all_observations, out_dir=out_dir)

    mmsi_set: set[str] = set()
    earliest: float | None = None
    latest: float | None = None
    for obs in all_observations:
        mmsi = obs.notes.get("mmsi") or obs.raw_ref.split("mmsi=")[-1].split("/")[0]
        mmsi_set.add(str(mmsi))
        t = obs.acquisition_time
        earliest = t if earliest is None or t < earliest else earliest
        latest = t if latest is None or t > latest else latest

    summary = {
        "observation_count": len(all_observations),
        "unique_mmsi_count": len(mmsi_set),
        "earliest_acquisition_utc": _iso(earliest) if earliest is not None else None,
        "latest_acquisition_utc": _iso(latest) if latest is not None else None,
        "chunk_file_count": len(chunk_files),
        "drops": merged_drops.as_dict(),
        "generator": "scripts/02_process_gfw_presence_to_parquet.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = out_dir / "observations_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print()
    print(json.dumps(summary, indent=2))
    print(f"Wrote summary -> {summary_path}")


if __name__ == "__main__":
    main()
