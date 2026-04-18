"""
Day-0 Umbra coverage scan.

Goal: find which of the 1,000 ship_detection_testdata scenes
intersect our Spratly AOI, and also list the general tasks/ folder
to spot any human-named SCS folders.
"""
import boto3
import json
from botocore import UNSIGNED
from botocore.config import Config
from shapely.geometry import shape, box, Point
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv

BUCKET = "umbra-open-data-catalog"
SHIP_PREFIX = "sar-data/tasks/ship_detection_testdata/"
TASKS_PREFIX = "sar-data/tasks/"

AOI = box(114.5, 8.5, 117.5, 11.0)
SCS_BROAD = box(109.0, 5.0, 121.0, 22.0)

s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))


def list_prefix(prefix, delimiter="/"):
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix, Delimiter=delimiter):
        for p in page.get("CommonPrefixes", []):
            yield p["Prefix"]


def find_metadata_keys(task_prefix):
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=task_prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith("_METADATA.json"):
                keys.append(obj["Key"])
    return keys


def fetch_centroid(meta_key):
    try:
        body = s3.get_object(Bucket=BUCKET, Key=meta_key)["Body"].read()
        meta = json.loads(body)
        poly = shape(meta["collects"][0]["footprintPolygonLla"])
        c = poly.centroid
        return (meta_key, c.x, c.y)
    except Exception:
        return (meta_key, None, None)


def main():
    print("=" * 60)
    print("Step A -- top-level tasks/ folders (human names live here)")
    print("=" * 60)
    top = list(list_prefix(TASKS_PREFIX))
    print(f"Total top-level task folders: {len(top)}")
    human = [p for p in top if "ship_detection" not in p and len(p.rsplit('/', 2)[-2]) != 36]
    print(f"Human-named folders: {len(human)}")
    scs_keywords = ("sprat", "mischief", "sabina", "thomas", "scarborough",
                    "whitsun", "reed", "philippine", "palawan", "spratly",
                    "south china", "scs", "paracel", "hainan", "luzon",
                    "singapore", "malacca", "vietnam")
    candidates = [p for p in human if any(k in p.lower() for k in scs_keywords)]
    print("\nSCS-keyword matches in tasks/ folder names:")
    if candidates:
        for c in candidates:
            print("  ", c)
    else:
        print("  (none)")
    print("\nFirst 20 human-named folders for eyeballing:")
    for p in human[:20]:
        print("  ", p)

    print()
    print("=" * 60)
    print("Step B -- ship_detection_testdata scene centroids")
    print("=" * 60)
    task_folders = list(list_prefix(SHIP_PREFIX))
    print(f"Task folders under ship_detection_testdata: {len(task_folders)}")

    print("Finding metadata JSONs (one list_objects per folder)...")
    all_meta_keys = []
    with ThreadPoolExecutor(max_workers=32) as ex:
        futures = {ex.submit(find_metadata_keys, t): t for t in task_folders}
        for i, fut in enumerate(as_completed(futures)):
            all_meta_keys.extend(fut.result())
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(task_folders)} folders scanned")

    print(f"Total metadata files: {len(all_meta_keys)}")

    print("Downloading centroids (parallel)...")
    rows = []
    with ThreadPoolExecutor(max_workers=32) as ex:
        futures = [ex.submit(fetch_centroid, k) for k in all_meta_keys]
        for i, fut in enumerate(as_completed(futures)):
            rows.append(fut.result())
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(all_meta_keys)} centroids fetched")

    with open("ship_detection_centroids.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "lon", "lat"])
        for row in rows:
            w.writerow(row)
    print(f"Wrote ship_detection_centroids.csv ({len(rows)} rows)")

    in_aoi = [r for r in rows if r[1] is not None and AOI.contains(Point(r[1], r[2]))]
    in_broad = [r for r in rows if r[1] is not None and SCS_BROAD.contains(Point(r[1], r[2]))]

    print()
    print(f"Scenes inside Spratly AOI (114.5-117.5E, 8.5-11.0N): {len(in_aoi)}")
    for r in in_aoi[:20]:
        print(f"   {r[1]:.3f}, {r[2]:.3f}   {r[0]}")
    print()
    print(f"Scenes inside broader SCS bbox (109-121E, 5-22N): {len(in_broad)}")
    for r in in_broad[:20]:
        print(f"   {r[1]:.3f}, {r[2]:.3f}   {r[0]}")


if __name__ == "__main__":
    main()
