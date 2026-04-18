"""
Download Umbra SAR scenes from the public S3 bucket.

Two tiers:
  Tier 1 (AOI)      : full complement -- GEC, SICD, SIDD, METADATA
  Tier 2 (SCS-broad): lite            -- GEC, METADATA only
"""
import boto3
import csv
import sys
from botocore import UNSIGNED
from botocore.config import Config
from shapely.geometry import Point, box
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BUCKET = "umbra-open-data-catalog"
LOCAL_ROOT = Path("C:/dev/custody/data/raw/umbra")
CSV_PATH = Path(__file__).parent / "ship_detection_centroids.csv"
ERR_LOG = Path(__file__).parent / "download_errors.txt"

AOI = box(114.5, 8.5, 117.5, 11.0)
SCS_BROAD = box(109.0, 5.0, 121.0, 22.0)

TIER1_SUFFIXES = ("_GEC.tif", "_SICD.nitf", "_SIDD.nitf", "_METADATA.json")
TIER2_SUFFIXES = ("_GEC.tif", "_METADATA.json")

s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))


def task_prefix(meta_key: str) -> str:
    # sar-data/tasks/ship_detection_testdata/<uuid>/...
    parts = meta_key.split("/")
    return "/".join(parts[:4]) + "/"


def list_task_files(prefix: str):
    paginator = s3.get_paginator("list_objects_v2")
    out = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for o in page.get("Contents", []):
            out.append((o["Key"], o["Size"]))
    return out


def classify(lon: float, lat: float) -> int:
    p = Point(lon, lat)
    if AOI.contains(p):
        return 1
    if SCS_BROAD.contains(p):
        return 2
    return 0


def download(key: str, size: int):
    local = LOCAL_ROOT / key
    if local.exists() and local.stat().st_size > 0:
        return (key, "skip")
    local.parent.mkdir(parents=True, exist_ok=True)
    try:
        s3.download_file(BUCKET, key, str(local))
        return (key, "ok")
    except Exception as e:
        with open(ERR_LOG, "a") as f:
            f.write(f"{key}\t{e}\n")
        return (key, f"err:{e}")


def main():
    auto_yes = "--yes" in sys.argv

    rows = []
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            if row["lon"] and row["lat"]:
                rows.append((row["key"], float(row["lon"]), float(row["lat"])))

    tier1_prefixes, tier2_prefixes = set(), set()
    for key, lon, lat in rows:
        t = classify(lon, lat)
        if t == 1:
            tier1_prefixes.add(task_prefix(key))
        elif t == 2:
            tier2_prefixes.add(task_prefix(key))

    # Tier-2 must not include anything already in tier-1
    tier2_prefixes -= tier1_prefixes

    print(f"Tier 1 task folders (AOI):        {len(tier1_prefixes)}")
    print(f"Tier 2 task folders (SCS-broad):  {len(tier2_prefixes)}")

    print("\nListing files in each task folder...")
    tier1_files, tier2_files = [], []

    with ThreadPoolExecutor(max_workers=32) as ex:
        t1_futs = {ex.submit(list_task_files, p): p for p in tier1_prefixes}
        for fut in as_completed(t1_futs):
            for k, sz in fut.result():
                if k.endswith(TIER1_SUFFIXES):
                    tier1_files.append((k, sz))

        t2_futs = {ex.submit(list_task_files, p): p for p in tier2_prefixes}
        for fut in as_completed(t2_futs):
            for k, sz in fut.result():
                if k.endswith(TIER2_SUFFIXES):
                    tier2_files.append((k, sz))

    t1_bytes = sum(sz for _, sz in tier1_files)
    t2_bytes = sum(sz for _, sz in tier2_files)
    total_files = tier1_files + tier2_files
    total_bytes = t1_bytes + t2_bytes

    # Breakdown by suffix
    def by_suffix(files):
        d = {}
        for k, sz in files:
            for s in TIER1_SUFFIXES:
                if k.endswith(s):
                    d.setdefault(s, [0, 0])
                    d[s][0] += 1
                    d[s][1] += sz
                    break
        return d

    print(f"\nTier 1 files: {len(tier1_files):>5}  {t1_bytes/1e9:6.2f} GB")
    for s, (n, b) in by_suffix(tier1_files).items():
        print(f"   {s:<16} {n:>4}  {b/1e9:6.2f} GB")
    print(f"Tier 2 files: {len(tier2_files):>5}  {t2_bytes/1e9:6.2f} GB")
    for s, (n, b) in by_suffix(tier2_files).items():
        print(f"   {s:<16} {n:>4}  {b/1e9:6.2f} GB")
    print(f"\nTOTAL:        {len(total_files):>5}  {total_bytes/1e9:6.2f} GB")

    # Skip check: how many already present?
    already = sum(
        1 for k, _ in total_files
        if (LOCAL_ROOT / k).exists() and (LOCAL_ROOT / k).stat().st_size > 0
    )
    print(f"Already on disk (will skip): {already}")
    to_fetch = len(total_files) - already
    print(f"To download:                 {to_fetch}")

    if to_fetch == 0:
        print("Nothing to do.")
        return

    if auto_yes:
        resp = "y"
    else:
        resp = input("\nProceed with download? (y/N): ").strip().lower()
    if resp != "y":
        print("Aborted.")
        return

    LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
    done = 0
    ok = 0
    skip = 0
    err = 0
    with ThreadPoolExecutor(max_workers=32) as ex:
        futs = [ex.submit(download, k, sz) for k, sz in total_files]
        for fut in as_completed(futs):
            done += 1
            key, status = fut.result()
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                err += 1
            if done % 20 == 0:
                print(f"  {done}/{len(total_files)}  ok={ok} skip={skip} err={err}")
    print(f"\nDone. ok={ok} skip={skip} err={err}")
    if err:
        print(f"See {ERR_LOG}")


if __name__ == "__main__":
    main()
