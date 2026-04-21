"""Calibration run: apply :func:`assess_scene_quality` to the 7 ingested SAR scenes.

Prints a stats table for the user to pick real red/yellow thresholds from.
Tennent scenes are assessed on a 2 km (half-width 1 km) AOI centered on
8.855687 N / 114.665145 E; Whitsun scenes use the full scene.

Run:
    uv run python day0/scratch/quality_calibration_run.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from custody.detection.quality import assess_scene_quality, IntensityStats  # noqa: E402
from custody.detection.sar_common import crop_to_aoi, read_geotiff  # noqa: E402
DATA_ROOT = REPO_ROOT / "data" / "raw" / "umbra" / "sar-data" / "tasks" / "ship_detection_testdata"

TENNENT_LAT = 8.855687
TENNENT_LON = 114.665145
BOX_HALF_KM = 1.0

SCENES = [
    # (label, relative_path, is_tennent)
    ("tennent_20230702", "f0730a1d-2bf7-4193-b2fe-ec8bfb2e1aef/2023-07-02-14-00-55_UMBRA-05/2023-07-02-14-00-55_UMBRA-05_GEC.tif", True),
    ("tennent_20230723", "a9fceda5-7fa5-4849-b76b-dd6d7a50dae9/2023-07-23-14-02-49_UMBRA-05/2023-07-23-14-02-49_UMBRA-05_GEC.tif", True),
    ("tennent_20230807", "00dee081-76ee-413a-8af2-e6be1c6ab8d0/2023-08-07-13-53-00_UMBRA-04/2023-08-07-13-53-00_UMBRA-04_GEC.tif", True),
    ("tennent_20230809", "b422de0d-b6fc-4340-aa85-6c87f3805175/2023-08-09-02-23-25_UMBRA-06/2023-08-09-02-23-25_UMBRA-06_GEC.tif", True),
    ("tennent_20230813", "645d903f-ced5-40d9-94ed-6e4e91977c97/2023-08-13-15-00-09_UMBRA-06/2023-08-13-15-00-09_UMBRA-06_GEC.tif", True),
    ("whitsun_20231206", "1e1f051d-4c81-4640-997d-1a03e967ad6a/2023-12-06-02-06-24_UMBRA-04/2023-12-06-02-06-24_UMBRA-04_GEC.tif", False),
    ("whitsun_20240320", "92c2b446-415c-4d26-88bc-70fdb652e852/2024-03-20-02-09-55_UMBRA-05/2024-03-20-02-09-55_UMBRA-05_GEC.tif", False),
]


def _fmt_stats(s: IntensityStats | None) -> str:
    if s is None:
        return f"{'(n/a — full scene)':<34}"
    return f"{s.mean:5.1f} {s.std:5.1f} {int(round(s.p50)):>4d} {int(round(s.p90)):>4d} {int(round(s.p99)):>4d} {int(round(s.dynamic_range)):>4d}"


def _tennent_aoi_bounds(scene_path: Path) -> tuple[int, int, int, int]:
    """Derive full-resolution AOI pixel bounds for a Tennent scene."""
    img, transform, crs_wkt = read_geotiff(scene_path)
    _, _, bounds = crop_to_aoi(
        img, transform, TENNENT_LAT, TENNENT_LON, crs_wkt, box_half_km=BOX_HALF_KM,
    )
    return (
        int(bounds["row_start"]),
        int(bounds["row_end"]),
        int(bounds["col_start"]),
        int(bounds["col_end"]),
    )


def main() -> None:
    header_top = (
        f"{'scene':<20} | {'full scene':<34} | {'AOI':<34} | flag"
    )
    header_sub = (
        f"{'':<20} | {'mean   std   p50   p90   p99   dr':<34} | "
        f"{'mean   std   p50   p90   p99   dr':<34} |"
    )
    sep = f"{'-' * 20}-|-{'-' * 34}-|-{'-' * 34}-|------"

    print(header_top)
    print(header_sub)
    print(sep)

    for label, rel, is_tennent in SCENES:
        scene_path = DATA_ROOT / rel
        aoi_bounds: tuple[int, int, int, int] | None = None
        if is_tennent:
            aoi_bounds = _tennent_aoi_bounds(scene_path)

        report = assess_scene_quality(scene_path, aoi_bounds=aoi_bounds)
        full_str = _fmt_stats(report.full_scene_stats)
        aoi_str = _fmt_stats(report.aoi_stats)
        print(f"{label:<20} | {full_str:<34} | {aoi_str:<34} | {report.quality_flag}")
        if report.flag_reasons:
            for r in report.flag_reasons:
                print(f"{'':<20}   reason: {r}")


if __name__ == "__main__":
    main()
