"""Ground-truth preparation — build the hand-labeling sheet.

Enumerate all observation pairs on the Tennent 07-02 <-> 07-23 pair whose
tangent-plane distance is within 50 m, extract a same-size SAR chip crop for
each observation (AOI-local crop, centered on the detector's bbox centroid),
and write:

  * tests/fixtures/ground_truth/tennent_0702_0723.md       — human reading view
  * tests/fixtures/ground_truth/tennent_0702_0723.csv      — programmatic reimport
  * tests/fixtures/ground_truth/chips/tennent_0702_0723/pair_NN_a.png
  * tests/fixtures/ground_truth/chips/tennent_0702_0723/pair_NN_b.png

Every row includes the DirectSpatialMatcher and SignatureMatcher(V1) verdicts
for that exact (obs_a_id, obs_b_id), plus a ``Human label:`` line in the MD
the user fills in manually (``same`` / ``different`` / ``ambiguous``).

Idempotent re-run
-----------------

If the output MD already exists, ``parse_existing_labels`` reads each pair's
existing ``Human label`` value and ``merge_labels`` splices those values back
into the freshly-generated MD.  Pairs that exist in the regenerated MD but
not in the old one get blank labels (new pairs since last run).  This keeps
the script safe to re-run after the file has been promoted to a labeled
ground-truth artifact.

Chip sizing
-----------

User asked for 128x128 chips originally.  Several Direct-only pairs have
bbox dimensions up to ~480 px; a strict 128x128 chip would crop most of the
feature out, making labeling harder.  We use ``chip_size = max(128, longest
bbox dim * 1.2 rounded up to multiple of 16)``, capped at 384 px, and apply
the same chip size to both observations in a pair for visual comparability.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from custody.detection.sar_common import crop_to_aoi, read_geotiff  # noqa: E402
from custody.fusion.geo import to_tangent_plane_array  # noqa: E402
from custody.fusion.scenes import load_scene_from_parquet  # noqa: E402
from custody.fusion.temporal import (  # noqa: E402
    DirectSpatialMatcher,
    SignatureMatcher,
    _signature_vector,
    temporal_persistence,
)


SCENE_A_PARQUET = REPO_ROOT / "data/processed/vlm_detections/position.parquet"
SCENE_B_PARQUET = REPO_ROOT / "data/processed/vlm_detections/tennent_20230723_position.parquet"

SCENE_A_TIF = (
    REPO_ROOT
    / "data/raw/umbra/sar-data/tasks/ship_detection_testdata"
    / "f0730a1d-2bf7-4193-b2fe-ec8bfb2e1aef/2023-07-02-14-00-55_UMBRA-05"
    / "2023-07-02-14-00-55_UMBRA-05_GEC.tif"
)
SCENE_B_TIF = (
    REPO_ROOT
    / "data/raw/umbra/sar-data/tasks/ship_detection_testdata"
    / "a9fceda5-7fa5-4849-b76b-dd6d7a50dae9/2023-07-23-14-02-49_UMBRA-05"
    / "2023-07-23-14-02-49_UMBRA-05_GEC.tif"
)

AOI_HALF_KM = 1.0          # same as detection runs
GATE_M = 50.0
SIG_GATE = 0.8
CHIP_MIN = 128
CHIP_MAX = 384

GROUND_TRUTH_DIR = REPO_ROOT / "tests/fixtures/ground_truth"
SHEET_MD = GROUND_TRUTH_DIR / "tennent_0702_0723.md"
SHEET_CSV = GROUND_TRUTH_DIR / "tennent_0702_0723.csv"
CHIP_SUBDIR = "chips/tennent_0702_0723"   # MD-relative path
CHIP_DIR = GROUND_TRUTH_DIR / CHIP_SUBDIR

# Sidecars stay in scratch — they're VLM-detection-run artifacts, not ground truth.
SIDECAR_A = REPO_ROOT / "day0/scratch/tennent_20230702_vlm_summary.json"
SIDECAR_B = REPO_ROOT / "day0/scratch/tennent_20230723_vlm_summary.json"


# ---------------------------------------------------------------------------
# Label preservation (importable for tests)
# ---------------------------------------------------------------------------


_PAIR_HEADER_RE = re.compile(r"^##\s+(pair_\d+)\b", re.MULTILINE)
_LABEL_LINE_RE = re.compile(
    r"\*\*Human label:\*\*\s*`([^`]*)`(?:\s*\([^\)]*\))?"
)


_LEADING_COMMENT_RE = re.compile(r"\A\s*<!--.*?-->\s*\n", re.DOTALL)


def parse_existing_header(md_path: Path) -> str:
    """Return any leading HTML comment block (with trailing newline) from ``md_path``.

    Allows the canonical provenance comment in the ground-truth MD to survive
    regeneration.  Returns ``""`` if the file does not exist or does not start
    with an HTML comment.
    """
    if not md_path.exists():
        return ""
    text = md_path.read_text(encoding="utf-8")
    m = _LEADING_COMMENT_RE.match(text)
    return m.group(0) if m else ""


def parse_existing_labels(md_path: Path) -> dict[str, str]:
    """Return ``{pair_id: label_str}`` for every pair section in ``md_path``.

    ``label_str`` is the raw string found between the backticks on the
    ``**Human label:** `...``` line for that pair section.  The placeholder
    ``_____`` and empty strings are returned verbatim — the caller decides
    whether to treat them as "unfilled".  Returns an empty dict if the file
    does not exist.
    """
    if not md_path.exists():
        return {}
    text = md_path.read_text(encoding="utf-8")
    out: dict[str, str] = {}
    headers = list(_PAIR_HEADER_RE.finditer(text))
    for idx, m in enumerate(headers):
        pair_id = m.group(1)
        start = m.start()
        end = headers[idx + 1].start() if idx + 1 < len(headers) else len(text)
        section = text[start:end]
        lm = _LABEL_LINE_RE.search(section)
        out[pair_id] = (lm.group(1) if lm else "").strip()
    return out


def _is_unfilled(label: str) -> bool:
    return label.strip() in {"", "_____"}


def merge_labels(new_md: str, old_labels: dict[str, str]) -> tuple[str, int, int]:
    r"""Splice ``old_labels`` into ``new_md``'s ``**Human label:**`` lines.

    Returns ``(merged_md, n_preserved, n_blank)`` where ``n_preserved`` counts
    pair sections in ``new_md`` whose pair_id had a non-blank label in
    ``old_labels``, and ``n_blank`` counts pair sections in ``new_md`` that
    received no preserved label (either the pair_id was absent from the old
    MD, or the old label was the blank placeholder).  The merged MD always
    uses ``\`<label>\` (same / different / ambiguous)`` formatting.
    """
    n_preserved = 0
    n_blank = 0
    headers = list(_PAIR_HEADER_RE.finditer(new_md))

    def _replace(match: re.Match[str], pair_id: str) -> str:
        nonlocal n_preserved, n_blank
        old = old_labels.get(pair_id, "")
        if _is_unfilled(old):
            n_blank += 1
            value = "_____"
        else:
            n_preserved += 1
            value = old
        return f"**Human label:** `{value}` (same / different / ambiguous)"

    pieces: list[str] = []
    cursor = 0
    for idx, hm in enumerate(headers):
        pair_id = hm.group(1)
        sect_start = hm.start()
        sect_end = headers[idx + 1].start() if idx + 1 < len(headers) else len(new_md)
        # Append everything up to the section start unchanged.
        pieces.append(new_md[cursor:sect_start])
        section = new_md[sect_start:sect_end]
        section = _LABEL_LINE_RE.sub(lambda m: _replace(m, pair_id), section, count=1)
        pieces.append(section)
        cursor = sect_end
    pieces.append(new_md[cursor:])
    return "".join(pieces), n_preserved, n_blank


# ---------------------------------------------------------------------------
# Pair / chip helpers
# ---------------------------------------------------------------------------


@dataclass
class PairRow:
    pair_id: str
    obs_a_id: str
    obs_b_id: str
    lat_a: float
    lon_a: float
    lat_b: float
    lon_b: float
    distance_m: float
    bbox_a: tuple[int, int, int, int]
    bbox_b: tuple[int, int, int, int]
    wh_a: tuple[int, int]
    wh_b: tuple[int, int]
    conf_a: float
    conf_b: float
    sig_distance: float
    direct_accept: bool
    signature_accept: bool
    chip_px: int
    reasoning_a: str
    reasoning_b: str
    chip_rel_a: str
    chip_rel_b: str


def _load_aoi_crop(tif_path: Path, sidecar_json: Path) -> tuple[np.ndarray, dict]:
    with sidecar_json.open() as f:
        meta = json.load(f)
    target_lat, target_lon = meta["target_latlon"]
    img, transform, crs_wkt = read_geotiff(tif_path)
    cropped, _aoi_transform, bounds = crop_to_aoi(
        img, transform, target_lat, target_lon, crs_wkt,
        box_half_km=AOI_HALF_KM,
    )
    del img
    return cropped, bounds


def _stretch_to_uint8(chip: np.ndarray) -> np.ndarray:
    valid = chip[chip > 0]
    if valid.size > 0:
        lo, hi = np.percentile(valid, [2, 98])
    else:
        lo, hi = 0, 255
    stretched = np.clip(
        (chip.astype(np.float32) - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0
    )
    img8 = (stretched * 255).astype(np.uint8)
    return np.stack([img8, img8, img8], axis=-1)


def _chip_size_for_pair(
    bbox_a: tuple[int, int, int, int], bbox_b: tuple[int, int, int, int]
) -> int:
    wa = bbox_a[2] - bbox_a[0]
    ha = bbox_a[3] - bbox_a[1]
    wb = bbox_b[2] - bbox_b[0]
    hb = bbox_b[3] - bbox_b[1]
    longest = max(wa, ha, wb, hb)
    need = int(np.ceil(longest * 1.2 / 16) * 16)
    return int(np.clip(max(CHIP_MIN, need), CHIP_MIN, CHIP_MAX))


def _extract_chip(
    cropped: np.ndarray,
    bbox: tuple[int, int, int, int],
    chip_px: int,
    out_path: Path,
) -> None:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    half = chip_px // 2

    H, W = cropped.shape[:2]
    r0_src = cy - half
    c0_src = cx - half
    r1_src = r0_src + chip_px
    c1_src = c0_src + chip_px

    chip = np.zeros((chip_px, chip_px), dtype=cropped.dtype)
    r0_clip = max(0, r0_src)
    c0_clip = max(0, c0_src)
    r1_clip = min(H, r1_src)
    c1_clip = min(W, c1_src)
    if r0_clip < r1_clip and c0_clip < c1_clip:
        dst_r0 = r0_clip - r0_src
        dst_c0 = c0_clip - c0_src
        dst_r1 = dst_r0 + (r1_clip - r0_clip)
        dst_c1 = dst_c0 + (c1_clip - c0_clip)
        chip[dst_r0:dst_r1, dst_c0:dst_c1] = cropped[r0_clip:r1_clip, c0_clip:c1_clip]

    rgb = _stretch_to_uint8(chip)
    pil = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(pil)

    bx1 = x1 - c0_src
    by1 = y1 - r0_src
    bx2 = x2 - c0_src
    by2 = y2 - r0_src
    draw.rectangle((bx1, by1, bx2, by2), outline=(0, 255, 0), width=2)
    cx_local = cx - c0_src
    cy_local = cy - r0_src
    draw.line((cx_local - 6, cy_local, cx_local + 6, cy_local), fill=(255, 0, 0), width=1)
    draw.line((cx_local, cy_local - 6, cx_local, cy_local + 6), fill=(255, 0, 0), width=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(out_path, format="PNG")


# ---------------------------------------------------------------------------
# MD assembly
# ---------------------------------------------------------------------------


def _build_md(rows: list[PairRow], n_direct: int, n_sig: int) -> str:
    lines: list[str] = []
    lines.append("# SignatureMatcher V1 — hand-labeling sheet")
    lines.append("")
    lines.append(
        "Ground-truth harness for the Tennent 2023-07-02 <-> 2023-07-23 "
        f"scene pair, gate {GATE_M:.0f} m.  Each row is one candidate pair "
        "whose tangent-plane spatial distance is within the gate.  The "
        "`direct_accept` / `signature_accept` columns are the matcher verdicts "
        "for that exact pair under Hungarian assignment; the `human_label` "
        "line is left blank for manual completion "
        "(`same` / `different` / `ambiguous`)."
    )
    lines.append("")
    lines.append(
        f"Counts: **{len(rows)}** candidate pairs; "
        f"DirectSpatialMatcher accepts **{n_direct}**, "
        f"SignatureMatcher(gate_m={GATE_M:.0f}, sig_gate={SIG_GATE}) "
        f"accepts **{n_sig}**.  "
        f"Rows sorted by spatial distance ascending."
    )
    lines.append("")
    lines.append(
        "Chips are centered on each bbox centroid (green rectangle = bbox, "
        "red crosshair = centroid).  Per-pair chip size is adaptive to the "
        "larger of the two bboxes, capped at 384 px."
    )
    lines.append("")
    lines.append(
        "Provenance: generated by `scripts/ground_truth_prepare.py`.  "
        "Companion CSV: `tests/fixtures/ground_truth/tennent_0702_0723.csv`."
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    for r in rows:
        direct_tick = "ACCEPT" if r.direct_accept else "reject"
        sig_tick = "ACCEPT" if r.signature_accept else "reject"
        lines.append(
            f"## {r.pair_id} — spatial {r.distance_m:.2f} m, signature {r.sig_distance:.3f}"
        )
        lines.append("")
        lines.append(f"- **Direct:** {direct_tick}   |   **Signature (V1):** {sig_tick}")
        lines.append(f"- **07-02 obs:** `{r.obs_a_id}`")
        lines.append(f"- **07-23 obs:** `{r.obs_b_id}`")
        lines.append(f"- **07-02 lat/lon:** ({r.lat_a:.6f}, {r.lon_a:.6f})")
        lines.append(f"- **07-23 lat/lon:** ({r.lat_b:.6f}, {r.lon_b:.6f})")
        lines.append(f"- **Bbox A (w x h):** {r.wh_a[0]} x {r.wh_a[1]} px,  conf={r.conf_a}")
        lines.append(f"- **Bbox B (w x h):** {r.wh_b[0]} x {r.wh_b[1]} px,  conf={r.conf_b}")
        lines.append(f"- **Chip size:** {r.chip_px} x {r.chip_px} px")
        lines.append("")
        lines.append(
            f"| 07-02 | 07-23 |\n|---|---|\n"
            f"| ![{r.pair_id} A]({r.chip_rel_a}) | ![{r.pair_id} B]({r.chip_rel_b}) |"
        )
        lines.append("")
        lines.append("**detector_reasoning — 07-02**")
        lines.append("")
        lines.append("> " + (r.reasoning_a or "(none)").replace("\n", "\n> "))
        lines.append("")
        lines.append("**detector_reasoning — 07-23**")
        lines.append("")
        lines.append("> " + (r.reasoning_b or "(none)").replace("\n", "\n> "))
        lines.append("")
        lines.append("**Human label:** `_____` (same / different / ambiguous)")
        lines.append("")
        lines.append("**Notes:** ")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    sa = load_scene_from_parquet(SCENE_A_PARQUET, case_study="tennent")
    sb = load_scene_from_parquet(SCENE_B_PARQUET, case_study="tennent")
    print(f"Scene A: {sa.scene_id}  (|obs|={len(sa.observations)}, px={sa.pixel_size_m:.3f})")
    print(f"Scene B: {sb.scene_id}  (|obs|={len(sb.observations)}, px={sb.pixel_size_m:.3f})")

    obs_a = sa.observations
    obs_b = sb.observations

    lats_a = np.array([o.lat for o in obs_a], dtype=float)
    lons_a = np.array([o.lon for o in obs_a], dtype=float)
    lats_b = np.array([o.lat for o in obs_b], dtype=float)
    lons_b = np.array([o.lon for o in obs_b], dtype=float)
    xs_a, ys_a = to_tangent_plane_array(lats_a, lons_a)
    xs_b, ys_b = to_tangent_plane_array(lats_b, lons_b)
    dx = xs_a[:, None] - xs_b[None, :]
    dy = ys_a[:, None] - ys_b[None, :]
    dist = np.hypot(dx, dy)

    r_direct = temporal_persistence(sa, sb, matcher=DirectSpatialMatcher(gate_m=GATE_M))
    r_sig = temporal_persistence(sa, sb, matcher=SignatureMatcher(gate_m=GATE_M, sig_gate=SIG_GATE))
    direct_pairs = {(m.obs_a_id, m.obs_b_id) for m in r_direct.matches}
    sig_pairs = {(m.obs_a_id, m.obs_b_id) for m in r_sig.matches}

    I, J = np.where(dist <= GATE_M)
    order = np.argsort(dist[I, J])
    pairs_sorted: list[tuple[int, int, float]] = [
        (int(I[k]), int(J[k]), float(dist[I[k], J[k]])) for k in order
    ]
    print(f"\nCandidate pairs within {GATE_M:.0f} m: {len(pairs_sorted)}")
    print(f"  DirectSpatialMatcher accepts: {len(direct_pairs)}")
    print(f"  SignatureMatcher accepts:     {len(sig_pairs)}")

    print("\nLoading scene A AOI crop...")
    cropped_a, bounds_a = _load_aoi_crop(SCENE_A_TIF, SIDECAR_A)
    print(f"  shape={cropped_a.shape}  dtype={cropped_a.dtype}  px={bounds_a['pixel_size_m']:.3f}")

    print("Loading scene B AOI crop...")
    cropped_b, bounds_b = _load_aoi_crop(SCENE_B_TIF, SIDECAR_B)
    print(f"  shape={cropped_b.shape}  dtype={cropped_b.dtype}  px={bounds_b['pixel_size_m']:.3f}")

    CHIP_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[PairRow] = []
    for idx, (i, j, d_m) in enumerate(pairs_sorted, start=1):
        a = obs_a[i]
        b = obs_b[j]
        va = _signature_vector(a)
        vb = _signature_vector(b)
        sig_d = (
            float(np.linalg.norm(va - vb))
            if va is not None and vb is not None
            else float("nan")
        )
        bbox_a = tuple(int(v) for v in a.bbox_px)  # type: ignore[arg-type]
        bbox_b = tuple(int(v) for v in b.bbox_px)  # type: ignore[arg-type]
        wa, ha = bbox_a[2] - bbox_a[0], bbox_a[3] - bbox_a[1]
        wb, hb = bbox_b[2] - bbox_b[0], bbox_b[3] - bbox_b[1]
        chip_px = _chip_size_for_pair(bbox_a, bbox_b)

        pair_id = f"pair_{idx:02d}"
        chip_path_a = CHIP_DIR / f"{pair_id}_a.png"
        chip_path_b = CHIP_DIR / f"{pair_id}_b.png"
        _extract_chip(cropped_a, bbox_a, chip_px, chip_path_a)
        _extract_chip(cropped_b, bbox_b, chip_px, chip_path_b)

        rows.append(PairRow(
            pair_id=pair_id,
            obs_a_id=a.obs_id,
            obs_b_id=b.obs_id,
            lat_a=a.lat, lon_a=a.lon,
            lat_b=b.lat, lon_b=b.lon,
            distance_m=d_m,
            bbox_a=bbox_a, bbox_b=bbox_b,
            wh_a=(wa, ha), wh_b=(wb, hb),
            conf_a=float(a.classification_conf) if a.classification_conf is not None else float("nan"),
            conf_b=float(b.classification_conf) if b.classification_conf is not None else float("nan"),
            sig_distance=sig_d,
            direct_accept=(a.obs_id, b.obs_id) in direct_pairs,
            signature_accept=(a.obs_id, b.obs_id) in sig_pairs,
            chip_px=chip_px,
            reasoning_a=a.detector_reasoning or "",
            reasoning_b=b.detector_reasoning or "",
            chip_rel_a=f"{CHIP_SUBDIR}/{pair_id}_a.png",
            chip_rel_b=f"{CHIP_SUBDIR}/{pair_id}_b.png",
        ))

    # CSV (no labels — labels live in the MD only).
    SHEET_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SHEET_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "pair_id",
            "obs_a_id", "obs_b_id",
            "lat_a", "lon_a", "lat_b", "lon_b",
            "distance_m",
            "bbox_a_w", "bbox_a_h", "bbox_b_w", "bbox_b_h",
            "conf_a", "conf_b",
            "signature_distance",
            "direct_accept", "signature_accept",
            "chip_px",
            "human_label",
            "notes",
        ])
        for r in rows:
            w.writerow([
                r.pair_id,
                r.obs_a_id, r.obs_b_id,
                f"{r.lat_a:.6f}", f"{r.lon_a:.6f}",
                f"{r.lat_b:.6f}", f"{r.lon_b:.6f}",
                f"{r.distance_m:.3f}",
                r.wh_a[0], r.wh_a[1], r.wh_b[0], r.wh_b[1],
                r.conf_a, r.conf_b,
                f"{r.sig_distance:.4f}",
                int(r.direct_accept), int(r.signature_accept),
                r.chip_px,
                "",
                "",
            ])

    n_direct = sum(1 for r in rows if r.direct_accept)
    n_sig = sum(1 for r in rows if r.signature_accept)
    new_md = _build_md(rows, n_direct, n_sig)

    # Preserve any leading HTML comment header (provenance) and existing labels.
    header = parse_existing_header(SHEET_MD)
    existing = parse_existing_labels(SHEET_MD)
    merged_md, n_preserved, n_blank = merge_labels(new_md, existing)
    SHEET_MD.write_text(header + merged_md, encoding="utf-8")

    print(f"\nWrote {SHEET_MD.relative_to(REPO_ROOT)} ({SHEET_MD.stat().st_size/1024:.1f} KB)")
    print(f"Wrote {SHEET_CSV.relative_to(REPO_ROOT)} ({SHEET_CSV.stat().st_size/1024:.1f} KB)")
    print(f"Wrote {len(rows)*2} chip PNGs to {CHIP_DIR.relative_to(REPO_ROOT)}")
    print(f"Preserved {n_preserved} existing labels; {n_blank} pairs left blank.")


if __name__ == "__main__":
    main()
