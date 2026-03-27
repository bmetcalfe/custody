"""Reprocess all 10 AIS daily CSVs with a shared 300-vessel set."""
import os, sys, time, glob
import numpy as np

_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_repo, "src"))

from custody.ingest.noaa_preprocess import load_raw, clean, resample_hourly, save_parquet

files = sorted(glob.glob("C:/Users/Ben/Documents/Custody/AIS_Data/ais-2024-09-*.csv"))
chosen = set(np.load(os.path.join(_repo, "data", "shared_mmsi_300.npy")).tolist())
print(f"Processing {len(files)} files for {len(chosen)} vessels...")

for f in files:
    base = os.path.basename(f).replace(".csv", "")
    out = os.path.join(_repo, "data", f"{base}_hourly_300.parquet")
    t0 = time.time()

    raw = load_raw(f)
    raw = raw[raw["mmsi"].isin(chosen)]
    cleaned = clean(raw)
    hourly = resample_hourly(cleaned)
    save_parquet(hourly, out)

    n_vessels = hourly["mmsi"].nunique()
    n_records = len(hourly)
    elapsed = time.time() - t0
    print(f"  {base}: {n_vessels} vessels, {n_records:,} records, {elapsed:.1f}s")

print("\nDone.")
