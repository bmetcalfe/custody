"""Fetch real Sentinel Hub PNG previews for the committed Sentinel
observation fixtures, cache them under ``src/app/assets/evidence/``, and
update ``data/demo/map_overlays.fixture.json`` so the dashboard's
dynamic-overlay manager picks the previews up automatically.

Sentinel imagery is treated as **weak-signal cueing context only** in
the dashboard.  These previews are not VLM detection evidence, are not
confirmation of change, and do not stand on their own.  Umbra remains
the high-confidence confirmation/evidence layer.

Inputs (read-only):

  - ``data/demo/whitsun_sentinel_observations.fixture.json``
  - ``data/demo/tennent_sentinel_observations.fixture.json``
  - ``data/demo/whitsun_aoi.fixture.geojson``
  - ``data/demo/tennent_aoi.fixture.geojson``
  - ``data/demo/map_overlays.fixture.json`` (also written back)

Outputs:

  - ``src/app/assets/evidence/sentinel/<scenario>/<observation_id>/preview.png``
  - in-place update of the matching overlay records in
    ``data/demo/map_overlays.fixture.json``:
      * on success: ``image_kind="sentinel_preview"``, ``image_path``
        and ``asset_url`` set, ``missing_asset_reason=null``
      * on failure: existing footprint-only record is preserved with
        a more specific ``missing_asset_reason``

Credentials are read from environment variables only:

  - ``SENTINEL_HUB_CLIENT_ID``
  - ``SENTINEL_HUB_CLIENT_SECRET``

If either is missing the script logs a single line and exits cleanly
with status 0 (the dashboard continues to use the committed
footprint-only Sentinel overlays).

This is a deterministic build / refresh step.  It does not run during
normal Dash page load.  No tests in the suite require live credentials
or live HTTP.

Run::

    SENTINEL_HUB_CLIENT_ID=... SENTINEL_HUB_CLIENT_SECRET=... \\
        uv run python scripts/32_fetch_sentinel_previews.py
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json as _json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import requests


REPO_ROOT = Path(__file__).resolve().parents[1]

# Inputs
SENTINEL_FIXTURES = (
    REPO_ROOT / "data" / "demo" / "whitsun_sentinel_observations.fixture.json",
    REPO_ROOT / "data" / "demo" / "tennent_sentinel_observations.fixture.json",
)
AOI_FIXTURES: dict[str, Path] = {
    "whitsun": REPO_ROOT / "data" / "demo" / "whitsun_aoi.fixture.geojson",
    "tennent": REPO_ROOT / "data" / "demo" / "tennent_aoi.fixture.geojson",
}
# Per-scenario observation fixtures, keyed by the same scenario id the
# overlay manifest uses.  ``--scenario`` filters this map; the default
# is intentionally narrow (``whitsun``) so an accidental run cannot
# request Tennent previews without explicit opt-in.
SCENARIO_FIXTURES: dict[str, Path] = {
    "whitsun": SENTINEL_FIXTURES[0],
    "tennent": SENTINEL_FIXTURES[1],
}


def _fixtures_for_scenario(scenario: str) -> tuple[Path, ...]:
    """Resolve the ``--scenario`` choice to a tuple of fixture paths.

    ``whitsun`` -> Whitsun fixture only (default, narrow scope)
    ``tennent`` -> Tennent fixture only
    ``all``     -> both, preserving the prior behaviour for callers
                   that explicitly want it
    """
    if scenario == "whitsun":
        return (SCENARIO_FIXTURES["whitsun"],)
    if scenario == "tennent":
        return (SCENARIO_FIXTURES["tennent"],)
    if scenario == "all":
        return (SCENARIO_FIXTURES["whitsun"], SCENARIO_FIXTURES["tennent"])
    raise ValueError(f"unknown scenario {scenario!r}")
MANIFEST_PATH = REPO_ROOT / "data" / "demo" / "map_overlays.fixture.json"

# Outputs
EVIDENCE_DIR = REPO_ROOT / "src" / "app" / "assets" / "evidence" / "sentinel"

# Sentinel Hub Process API endpoints (see https://docs.sentinel-hub.com).
TOKEN_URL = "https://services.sentinel-hub.com/oauth/token"
PROCESS_URL = "https://services.sentinel-hub.com/api/v1/process"

# Default preview size — small enough to keep the round-trip cheap and
# the checked-in PNG sub-MB.
DEFAULT_PREVIEW_PX = 512

# How wide a date window to search around the observation timestamp,
# in days.  Sentinel-2 has a 5-day revisit and Sentinel-1 a 12-day
# revisit at our latitude; ±2 days catches the matching scene without
# pulling neighbouring revisits.
TIME_WINDOW_DAYS = 2

# Content-validation thresholds.  Sentinel Hub Process API can return
# HTTP 200 with a small uniform "no scene matched" placeholder PNG —
# typically ~334 bytes of fully-zero pixels.  Any preview below
# ``MIN_PREVIEW_SIZE_BYTES`` or with pixel-std-dev under
# ``MIN_PREVIEW_STDDEV`` is rejected as an empty placeholder so the
# overlay stays footprint-only instead of rendering as an opaque
# black square on the map.
MIN_PREVIEW_SIZE_BYTES = 5 * 1024
MIN_PREVIEW_STDDEV = 0.5


def _validate_preview_png(png_bytes: bytes) -> tuple[bool, str]:
    """Return ``(ok, reason)`` after sanity-checking a Process API
    response body.  Treats sub-threshold size or near-uniform pixel
    content as a failed preview so the manifest stays footprint-only.

    Network-level failures are caught earlier in ``_process_one``;
    this function only sees byte payloads from successful HTTP 200
    responses.
    """
    if not png_bytes:
        return False, "empty body"
    if len(png_bytes) < MIN_PREVIEW_SIZE_BYTES:
        return False, (
            f"preview too small ({len(png_bytes)} B < "
            f"{MIN_PREVIEW_SIZE_BYTES} B); likely empty placeholder"
        )
    try:
        from PIL import Image
        import numpy as np
        img = Image.open(io.BytesIO(png_bytes)).convert("L")
        std = float(np.array(img, dtype="float32").std())
    except Exception as exc:  # pragma: no cover - corrupt PNG branch
        return False, f"could not parse preview PNG: {exc}"
    if std < MIN_PREVIEW_STDDEV:
        return False, (
            f"preview is uniform (pixel std {std:.2f} < "
            f"{MIN_PREVIEW_STDDEV}); likely empty placeholder"
        )
    return True, "ok"


# ---------------------------------------------------------------------------
# Evalscripts (Sentinel Hub band math).
# ---------------------------------------------------------------------------


# Sentinel-2 L2A true colour — 2.5× linear stretch on B04 / B03 / B02.
S2_TRUE_COLOR_EVALSCRIPT = """//VERSION=3
function setup() {
  return {
    input: ["B02", "B03", "B04"],
    output: { bands: 3, sampleType: "AUTO" }
  };
}
function evaluatePixel(s) {
  return [2.5 * s.B04, 2.5 * s.B03, 2.5 * s.B02];
}
"""

# Sentinel-1 GRD VV dB grayscale — log-stretched single-channel SAR.
# Maps roughly -25 dB → 0, 0 dB → 1.
S1_VV_DB_EVALSCRIPT = """//VERSION=3
function setup() {
  return {
    input: ["VV"],
    output: { bands: 1, sampleType: "AUTO" }
  };
}
function evaluatePixel(s) {
  if (s.VV <= 0) { return [0]; }
  let db = 10 * Math.log(s.VV) / Math.LN10;
  let g = (db + 25) / 25;
  return [Math.max(0, Math.min(1, g))];
}
"""


# ---------------------------------------------------------------------------
# Per-source plumbing
# ---------------------------------------------------------------------------


def _data_collection_for(observation: Mapping[str, Any]) -> str | None:
    src = observation.get("source")
    if src == "sentinel-1":
        return "sentinel-1-grd"
    if src == "sentinel-2":
        return "sentinel-2-l2a"
    return None


def _evalscript_for(observation: Mapping[str, Any]) -> str | None:
    src = observation.get("source")
    if src == "sentinel-1":
        return S1_VV_DB_EVALSCRIPT
    if src == "sentinel-2":
        return S2_TRUE_COLOR_EVALSCRIPT
    return None


def _aoi_bbox_from_geojson(path: Path) -> tuple[float, float, float, float]:
    """Return ``(W, S, E, N)`` for the first feature in an AOI GeoJSON."""
    blob = _json.loads(path.read_text(encoding="utf-8"))
    feats = blob.get("features") or []
    if not feats:
        raise ValueError(f"{path} has no features")
    geom = feats[0].get("geometry") or {}
    coords = geom.get("coordinates") or []
    ring = (coords[0] if coords else None) or []
    if not ring:
        raise ValueError(f"{path} has empty geometry")
    lons = [float(p[0]) for p in ring]
    lats = [float(p[1]) for p in ring]
    return (min(lons), min(lats), max(lons), max(lats))


def _date_window(timestamp: str) -> tuple[str, str]:
    """Build an ISO date window centred on ``timestamp``."""
    if not isinstance(timestamp, str) or len(timestamp) < 10:
        raise ValueError(f"unparseable timestamp {timestamp!r}")
    from datetime import datetime, timedelta, timezone
    # Normalise trailing 'Z' for fromisoformat compat.
    iso = timestamp.replace("Z", "+00:00") if timestamp.endswith("Z") else timestamp
    try:
        ts = datetime.fromisoformat(iso)
    except ValueError as exc:
        raise ValueError(f"unparseable timestamp {timestamp!r}: {exc}") from exc
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    lo = ts - timedelta(days=TIME_WINDOW_DAYS)
    hi = ts + timedelta(days=TIME_WINDOW_DAYS)
    return (
        lo.strftime("%Y-%m-%dT%H:%M:%SZ"),
        hi.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


# ---------------------------------------------------------------------------
# Sentinel Hub I/O.  Mockable seams: ``_get_oauth_token`` and
# ``_request_preview``.
# ---------------------------------------------------------------------------


def _get_oauth_token(client_id: str, client_secret: str) -> str:
    """Exchange a client_credentials pair for an access token."""
    resp = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()
    token = payload.get("access_token")
    if not token:
        raise RuntimeError(f"sentinel hub OAuth response missing access_token: {payload!r}")
    return str(token)


def _build_process_request(
    *,
    bbox: Sequence[float],
    time_from: str,
    time_to: str,
    data_collection: str,
    evalscript: str,
    width: int,
    height: int,
) -> dict[str, Any]:
    return {
        "input": {
            "bounds": {
                "bbox": list(bbox),
                "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"},
            },
            "data": [{
                "type": data_collection,
                "dataFilter": {
                    "timeRange": {"from": time_from, "to": time_to},
                },
            }],
        },
        "output": {
            "width": int(width),
            "height": int(height),
            "responses": [{
                "identifier": "default",
                "format": {"type": "image/png"},
            }],
        },
        "evalscript": evalscript,
    }


def _request_preview(
    *,
    token: str,
    bbox: Sequence[float],
    time_from: str,
    time_to: str,
    data_collection: str,
    evalscript: str,
    width: int,
    height: int,
) -> bytes:
    body = _build_process_request(
        bbox=bbox, time_from=time_from, time_to=time_to,
        data_collection=data_collection, evalscript=evalscript,
        width=width, height=height,
    )
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "image/png",
        "Content-Type": "application/json",
    }
    resp = requests.post(PROCESS_URL, json=body, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.content


# ---------------------------------------------------------------------------
# Manifest update
# ---------------------------------------------------------------------------


def _scenario_for(observation_id: str) -> str | None:
    if "whitsun" in observation_id:
        return "whitsun"
    if "tennent" in observation_id:
        return "tennent"
    return None


def update_manifest_with_preview(
    manifest: dict[str, Any],
    *,
    observation_id: str,
    image_path: str,
    asset_url: str,
) -> bool:
    """Mutate ``manifest`` so any overlay matching ``observation_id``
    becomes a Sentinel preview overlay.  Returns True if at least one
    overlay was updated."""
    updated = False
    for overlay in manifest.get("overlays", []):
        if overlay.get("observation_id") != observation_id:
            continue
        if overlay.get("source") not in ("sentinel-1", "sentinel-2"):
            # Only Sentinel overlays should be promoted; never touch an
            # Umbra record from this script.
            continue
        overlay["image_kind"] = "sentinel_preview"
        overlay["image_path"] = image_path
        overlay["asset_url"] = asset_url
        overlay["missing_asset_reason"] = None
        updated = True
    return updated


def update_manifest_for_failure(
    manifest: dict[str, Any],
    *,
    observation_id: str,
    reason: str,
) -> bool:
    """Mutate ``manifest`` so any matching overlay records a more
    specific footprint-only ``missing_asset_reason``.  Returns True if
    at least one overlay was updated."""
    updated = False
    for overlay in manifest.get("overlays", []):
        if overlay.get("observation_id") != observation_id:
            continue
        if overlay.get("source") not in ("sentinel-1", "sentinel-2"):
            continue
        # Preserve footprint-only as the on-disk truth; just refine the
        # reason copy so the dashboard's "asset not available" footer
        # tells the operator something useful.
        overlay["image_kind"] = "footprint-only"
        overlay["image_path"] = None
        overlay["asset_url"] = None
        overlay["missing_asset_reason"] = reason
        updated = True
    return updated


# ---------------------------------------------------------------------------
# Per-observation driver
# ---------------------------------------------------------------------------


def _load_observations(path: Path) -> list[dict[str, Any]]:
    blob = _json.loads(path.read_text(encoding="utf-8"))
    obs = blob.get("observations") or []
    return list(obs)


def _output_paths_for(scenario: str, observation_id: str) -> tuple[Path, str, str]:
    """Resolve filesystem path + manifest path + asset URL for one preview.

    The manifest's ``image_path`` is the canonical
    ``src/app/assets/evidence/...`` string regardless of where the
    actual PNG is written, so tests can redirect ``EVIDENCE_DIR`` to a
    tmp dir without breaking the path-relative-to-repo invariant.
    """
    out_dir = EVIDENCE_DIR / scenario / observation_id
    out_path = out_dir / "preview.png"
    canonical_rel = (
        f"src/app/assets/evidence/sentinel/{scenario}/{observation_id}/preview.png"
    )
    asset_url = (
        f"/assets/evidence/sentinel/{scenario}/{observation_id}/preview.png"
    )
    return out_path, canonical_rel, asset_url


def _process_one(
    observation: Mapping[str, Any],
    *,
    token: str,
    aoi_bbox: Sequence[float],
    width: int,
    height: int,
    fetcher=None,
) -> tuple[bool, str, bytes | None]:
    """Fetch one Sentinel preview.  Returns ``(success, reason, png_bytes)``.

    ``fetcher`` defaults to the module-level ``_request_preview``.  We
    resolve via the module globals at call time so tests can swap it
    out via ``monkeypatch.setattr``.
    """
    if fetcher is None:
        fetcher = _request_preview
    obs_id = str(observation.get("observation_id") or "")
    if not obs_id:
        return False, "observation has no observation_id", None
    data_collection = _data_collection_for(observation)
    evalscript = _evalscript_for(observation)
    if data_collection is None or evalscript is None:
        return False, (
            f"unsupported source {observation.get('source')!r}"
        ), None
    timestamp = observation.get("timestamp")
    try:
        time_from, time_to = _date_window(str(timestamp))
    except ValueError as exc:
        return False, str(exc), None
    try:
        png = fetcher(
            token=token, bbox=aoi_bbox,
            time_from=time_from, time_to=time_to,
            data_collection=data_collection,
            evalscript=evalscript,
            width=width, height=height,
        )
    except Exception as exc:
        return False, f"sentinel hub fetch failed: {exc}", None
    if not png:
        return False, "sentinel hub returned empty body", None
    return True, "ok", png


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch real Sentinel Hub previews for the committed Sentinel "
            "observation fixtures and update the map overlay manifest."
        ),
    )
    parser.add_argument(
        "--width", type=int, default=DEFAULT_PREVIEW_PX,
        help="Preview width in pixels (default: 512).",
    )
    parser.add_argument(
        "--height", type=int, default=DEFAULT_PREVIEW_PX,
        help="Preview height in pixels (default: 512).",
    )
    parser.add_argument(
        "--scenario",
        choices=("whitsun", "tennent", "all"),
        default="whitsun",
        help=(
            "Which Sentinel observation fixture(s) to process: "
            "'whitsun' (default, narrow scope), 'tennent', or 'all'."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help=(
            "Do not call Sentinel Hub, do not write PNGs, do not update "
            "the manifest; just print which scenarios / observations "
            "would be processed.  No OAuth, no Process API requests."
        ),
    )
    args = parser.parse_args(argv)

    selected_fixtures = _fixtures_for_scenario(args.scenario)

    # Dry-run reports the static plan without ever touching the
    # network — no OAuth, no Process API calls, no writes.
    if args.dry_run:
        print(
            f"[dry-run] scenario={args.scenario}; "
            f"would call Sentinel Hub Process API for the following "
            f"observations:"
        )
        total_would_fetch = 0
        for fixture in selected_fixtures:
            if not fixture.exists():
                print(f"  skipping missing fixture {fixture}")
                continue
            observations = _load_observations(fixture)
            for obs in observations:
                obs_id = str(obs.get("observation_id") or "")
                scenario = _scenario_for(obs_id)
                if scenario is None:
                    print(f"  skip {obs_id}: could not classify scenario")
                    continue
                _, rel_path, _ = _output_paths_for(scenario, obs_id)
                print(
                    f"  [dry-run] {obs_id} ({obs.get('source')}) "
                    f"-> would write {rel_path}"
                )
                total_would_fetch += 1
        print()
        print(
            f"[dry-run] {total_would_fetch} preview(s) would be fetched; "
            f"no live HTTP performed, no files written."
        )
        return 0

    client_id = os.environ.get("SENTINEL_HUB_CLIENT_ID")
    client_secret = os.environ.get("SENTINEL_HUB_CLIENT_SECRET")
    if not client_id or not client_secret:
        print(
            "SENTINEL_HUB_CLIENT_ID / SENTINEL_HUB_CLIENT_SECRET not set; "
            "skipping Sentinel preview fetch.  Existing footprint-only "
            "Sentinel overlays remain unchanged."
        )
        return 0

    # OAuth ------------------------------------------------------------
    try:
        token = _get_oauth_token(client_id, client_secret)
    except Exception as exc:
        print(f"sentinel hub OAuth failed: {exc}", file=sys.stderr)
        return 1
    print(f"sentinel hub OAuth token acquired (scenario={args.scenario})")

    # Manifest ---------------------------------------------------------
    manifest = _json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    fetched: list[str] = []        # HTTP 200 + non-empty body
    promoted: list[str] = []       # passed validation + unique → manifest updated
    empty: list[tuple[str, str]] = []      # 200 OK but uniform / too small
    duplicate: list[tuple[str, str]] = []  # bytes match an earlier promoted preview
    failed: list[tuple[str, str]] = []     # HTTP error / API exception

    # sha256 → first-promoted observation_id, for within-run dedup.
    seen_hashes: dict[str, str] = {}

    for fixture in selected_fixtures:
        if not fixture.exists():
            print(f"skipping missing fixture {fixture}")
            continue
        observations = _load_observations(fixture)
        for obs in observations:
            obs_id = str(obs.get("observation_id") or "")
            scenario = _scenario_for(obs_id)
            if scenario is None:
                failed.append((obs_id, "could not classify scenario"))
                continue
            aoi_bbox = _aoi_bbox_from_geojson(AOI_FIXTURES[scenario])
            ok, reason, png = _process_one(
                obs, token=token, aoi_bbox=aoi_bbox,
                width=args.width, height=args.height,
            )
            if not ok:
                failed.append((obs_id, reason))
                update_manifest_for_failure(
                    manifest, observation_id=obs_id,
                    reason=f"sentinel hub preview unavailable: {reason}",
                )
                continue

            # HTTP succeeded; fetched bytes in hand.
            fetched.append(obs_id)

            # Content validation: catch sub-threshold size + uniform tiles
            # before the manifest claims they're real previews.
            ok_content, content_reason = _validate_preview_png(png or b"")
            if not ok_content:
                empty.append((obs_id, content_reason))
                update_manifest_for_failure(
                    manifest, observation_id=obs_id,
                    reason=(
                        "Sentinel Hub preview returned empty placeholder; "
                        "keeping footprint-only overlay"
                    ),
                )
                print(f"  empty {obs_id}: {content_reason}")
                continue

            # Within-run duplicate detection by sha256.  The first
            # observation that produced a given image is kept; later
            # collisions stay footprint-only so the dashboard doesn't
            # render the same picture under multiple dates.
            digest = hashlib.sha256(png).hexdigest()
            if digest in seen_hashes:
                first_obs = seen_hashes[digest]
                duplicate.append((obs_id, first_obs))
                update_manifest_for_failure(
                    manifest, observation_id=obs_id,
                    reason=(
                        f"Duplicate preview of {first_obs}; keeping "
                        f"footprint-only overlay to avoid misleading replay"
                    ),
                )
                print(f"  duplicate {obs_id}: shares bytes with {first_obs}")
                continue
            seen_hashes[digest] = obs_id

            # Promote: write the PNG and stamp the manifest.
            out_path, rel_path, asset_url = _output_paths_for(scenario, obs_id)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(png)
            update_manifest_with_preview(
                manifest,
                observation_id=obs_id,
                image_path=rel_path,
                asset_url=asset_url,
            )
            promoted.append(obs_id)
            print(f"  promoted {obs_id} -> {rel_path}")

    MANIFEST_PATH.write_text(
        _json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )

    print()
    print(
        f"sentinel previews: {len(fetched)} fetched, "
        f"{len(promoted)} promoted, "
        f"{len(empty)} empty/placeholder, "
        f"{len(duplicate)} duplicate, "
        f"{len(failed)} failed"
    )
    for obs_id, reason in empty:
        print(f"  empty/placeholder {obs_id}: {reason}")
    for obs_id, first_obs in duplicate:
        print(f"  duplicate {obs_id}: byte-identical to {first_obs}")
    for obs_id, reason in failed:
        print(f"  failed {obs_id}: {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
