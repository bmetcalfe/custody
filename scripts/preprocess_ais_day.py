#!/usr/bin/env python
"""Convenience script to preprocess a NOAA AIS daily CSV.

Usage:
    uv run python scripts/preprocess_ais_day.py \\
        --input "C:\\Users\\Ben\\Downloads\\ais-2024-09-24.csv" \\
        --output data/ais_2024_09_24_hourly.parquet \\
        --n-vessels 30

Equivalent to:
    uv run python -m custody.ingest.noaa_preprocess --input ... --output ...
"""
import os
import sys

# Ensure src is on path when running from repo root
_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_repo, "src"))

from custody.ingest.noaa_preprocess import _cli

if __name__ == "__main__":
    _cli()
