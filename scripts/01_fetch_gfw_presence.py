"""Fetch GFW v3 presence records for the Spratly AOI + demo window.

Chunks the window by ISO week, writing each chunk's raw JSON to
``data/raw/gfw_presence/{YYYY-MM-DD}_{YYYY-MM-DD}.json``.  Idempotent:
skips any chunk whose output file already exists (safe to rerun).

Run::

    python scripts/01_fetch_gfw_presence.py

Optionally pass custom date bounds::

    python scripts/01_fetch_gfw_presence.py --start 2023-06-01 --end 2023-08-20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "raw" / "gfw_presence"

# Spratly AOI per docs/scenario.md.
AOI_GEOJSON = {
    "type": "Polygon",
    "coordinates": [[
        [114.5, 8.5], [117.5, 8.5],
        [117.5, 11.0], [114.5, 11.0],
        [114.5, 8.5],
    ]],
}

# Demo window per docs/scenario.md.
DEFAULT_START = date(2023, 6, 1)
DEFAULT_END = date(2023, 8, 20)

ENDPOINT = "https://gateway.api.globalfishingwatch.org/v3/4wings/report"


def _iter_weeks(start: date, end: date):
    """Yield (chunk_start, chunk_end) per ISO week covering [start, end]."""
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=6), end)
        yield cur, chunk_end
        cur = chunk_end + timedelta(days=1)


def fetch_chunk(token: str, chunk_start: date, chunk_end: date) -> dict:
    params = {
        "datasets[0]": "public-global-presence:latest",
        "date-range": f"{chunk_start.isoformat()},{chunk_end.isoformat()}",
        "format": "JSON",
        "spatial-resolution": "HIGH",
        "temporal-resolution": "HOURLY",
        "group-by": "VESSEL_ID",
    }
    body = {"geojson": AOI_GEOJSON}
    r = requests.post(
        ENDPOINT,
        headers={"Authorization": f"Bearer {token}"},
        params=params,
        json=body,
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=DEFAULT_START.isoformat(),
                        help="ISO start date (default: 2023-06-01)")
    parser.add_argument("--end", default=DEFAULT_END.isoformat(),
                        help="ISO end date (default: 2023-08-20)")
    parser.add_argument("--sleep", type=float, default=1.0,
                        help="Seconds to sleep between chunk calls (default: 1.0)")
    args = parser.parse_args()

    load_dotenv()
    token = os.getenv("GFW_API_TOKEN")
    if not token or token == "your_token_here":
        sys.exit("ERR: GFW_API_TOKEN not set in .env")

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    total_chunks = 0
    fetched = 0
    skipped = 0
    for chunk_start, chunk_end in _iter_weeks(start, end):
        total_chunks += 1
        fname = f"{chunk_start.isoformat()}_{chunk_end.isoformat()}.json"
        out_path = OUT_DIR / fname
        if out_path.exists():
            print(f"  skip    {fname}  ({out_path.stat().st_size} bytes)")
            skipped += 1
            continue
        print(f"  fetch   {fname} ...", end="", flush=True)
        try:
            data = fetch_chunk(token, chunk_start, chunk_end)
        except requests.HTTPError as e:
            print(f" ERR {e.response.status_code}: {e.response.text[:200]}")
            continue
        out_path.write_text(json.dumps(data))
        print(f" ok ({out_path.stat().st_size} bytes)")
        fetched += 1
        time.sleep(args.sleep)

    print()
    print(f"Done. chunks={total_chunks} fetched={fetched} skipped={skipped}")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
