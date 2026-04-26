"""Tests for ``scripts/32_fetch_sentinel_previews.py``.

Exercises the script's mockable seams so the suite can fully cover
the fetch + manifest-update path without making any live HTTP call.
Credentials are never required.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "32_fetch_sentinel_previews.py"


@pytest.fixture
def fetch_module():
    """Load the script as a module (its filename starts with a digit
    so a normal ``import`` does not work)."""
    src = REPO_ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    spec = importlib.util.spec_from_file_location(
        "_fetch_sentinel_previews", SCRIPT_PATH,
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------
# Manifest-update unit tests
# ---------------------------------------------------------------------------


def _baseline_manifest() -> dict:
    return {
        "schema": "custody.demo.map_overlays.v1",
        "metadata": {"data_mode": "fixture"},
        "overlays": [
            {
                "overlay_id": "whitsun-aoi",
                "scenario_id": "whitsun",
                "observation_id": None,
                "source": "simulated",
                "image_kind": "footprint-only",
                "image_path": None,
                "asset_url": None,
                "missing_asset_reason": None,
            },
            {
                "overlay_id": "whitsun-sentinel-2-low-cloud-20231212",
                "scenario_id": "whitsun",
                "observation_id": "fixture-s2-l2a-whitsun-20231212-low-cloud",
                "source": "sentinel-2",
                "image_kind": "footprint-only",
                "image_path": None,
                "asset_url": None,
                "missing_asset_reason": (
                    "no committed Sentinel-2 RGB preview; footprint-only"
                ),
            },
            {
                "overlay_id": "whitsun-umbra-20231206",
                "scenario_id": "whitsun",
                "observation_id": "fixture-umbra-whitsun-20231206",
                "source": "umbra",
                "image_kind": "png",
                "image_path": "src/app/assets/overlays/whitsun.png",
                "asset_url": "/assets/overlays/whitsun.png",
                "missing_asset_reason": None,
            },
        ],
    }


def test_update_manifest_with_preview_promotes_sentinel_overlay(fetch_module):
    manifest = _baseline_manifest()
    ok = fetch_module.update_manifest_with_preview(
        manifest,
        observation_id="fixture-s2-l2a-whitsun-20231212-low-cloud",
        image_path="src/app/assets/evidence/sentinel/whitsun/foo/preview.png",
        asset_url="/assets/evidence/sentinel/whitsun/foo/preview.png",
    )
    assert ok is True
    sentinel = next(
        o for o in manifest["overlays"]
        if o["overlay_id"] == "whitsun-sentinel-2-low-cloud-20231212"
    )
    assert sentinel["image_kind"] == "sentinel_preview"
    assert sentinel["image_path"].startswith("src/app/assets/evidence/sentinel/")
    assert sentinel["asset_url"].startswith("/assets/evidence/sentinel/")
    assert sentinel["missing_asset_reason"] is None


def test_update_manifest_with_preview_never_touches_umbra(fetch_module):
    """Even if a hostile observation id collided, the script must
    refuse to mutate Umbra overlays."""
    manifest = _baseline_manifest()
    fetch_module.update_manifest_with_preview(
        manifest,
        observation_id="fixture-umbra-whitsun-20231206",
        image_path="bogus.png",
        asset_url="/bogus.png",
    )
    umbra = next(
        o for o in manifest["overlays"]
        if o["overlay_id"] == "whitsun-umbra-20231206"
    )
    assert umbra["image_kind"] == "png"
    assert umbra["asset_url"] == "/assets/overlays/whitsun.png"


def test_update_manifest_for_failure_preserves_footprint_only(fetch_module):
    manifest = _baseline_manifest()
    fetch_module.update_manifest_for_failure(
        manifest,
        observation_id="fixture-s2-l2a-whitsun-20231212-low-cloud",
        reason="sentinel hub preview unavailable: HTTP 401",
    )
    sentinel = next(
        o for o in manifest["overlays"]
        if o["overlay_id"] == "whitsun-sentinel-2-low-cloud-20231212"
    )
    assert sentinel["image_kind"] == "footprint-only"
    assert sentinel["image_path"] is None
    assert sentinel["asset_url"] is None
    assert "HTTP 401" in (sentinel["missing_asset_reason"] or "")


# ---------------------------------------------------------------------------
# Process API request shape
# ---------------------------------------------------------------------------


def test_build_process_request_has_required_keys(fetch_module):
    req = fetch_module._build_process_request(
        bbox=[114.55, 8.78, 114.75, 8.93],
        time_from="2023-07-15T00:00:00Z",
        time_to="2023-07-17T00:00:00Z",
        data_collection="sentinel-2-l2a",
        evalscript=fetch_module.S2_TRUE_COLOR_EVALSCRIPT,
        width=512, height=512,
    )
    assert req["input"]["bounds"]["bbox"] == [114.55, 8.78, 114.75, 8.93]
    assert req["input"]["data"][0]["type"] == "sentinel-2-l2a"
    assert req["output"]["responses"][0]["format"]["type"] == "image/png"
    assert "B04" in req["evalscript"]
    assert "B02" in req["evalscript"]


def test_evalscript_for_routes_correctly(fetch_module):
    s1 = fetch_module._evalscript_for({"source": "sentinel-1"})
    s2 = fetch_module._evalscript_for({"source": "sentinel-2"})
    assert "VV" in s1 and "B04" not in s1
    assert "B04" in s2 and "VV" not in s2
    # Unsupported source returns None.
    assert fetch_module._evalscript_for({"source": "umbra"}) is None


def test_data_collection_for_routes_correctly(fetch_module):
    assert fetch_module._data_collection_for({"source": "sentinel-1"}) == (
        "sentinel-1-grd"
    )
    assert fetch_module._data_collection_for({"source": "sentinel-2"}) == (
        "sentinel-2-l2a"
    )
    assert fetch_module._data_collection_for({"source": "umbra"}) is None


# ---------------------------------------------------------------------------
# AOI bbox + scenario classification
# ---------------------------------------------------------------------------


def test_aoi_bbox_round_trips_for_committed_aois(fetch_module):
    for path in fetch_module.AOI_FIXTURES.values():
        bbox = fetch_module._aoi_bbox_from_geojson(path)
        assert len(bbox) == 4
        w, s, e, n = bbox
        assert w < e
        assert s < n
        # All AOIs sit in the South China Sea quadrant.
        assert 110.0 < w < 120.0
        assert 0.0 < s < 15.0


def test_scenario_classifier(fetch_module):
    f = fetch_module._scenario_for
    assert f("fixture-s1-grd-whitsun-20231210") == "whitsun"
    assert f("fixture-s2-l2a-tennent-20230718-low-cloud") == "tennent"
    assert f("real-umbra-tennent-20230702") == "tennent"
    assert f("garbage") is None


# ---------------------------------------------------------------------------
# Mocked end-to-end fetch (no live HTTP)
# ---------------------------------------------------------------------------


def test_process_one_with_mock_fetcher_returns_png(fetch_module):
    fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32

    def fake_fetcher(**kwargs):
        # Confirm the script passes through the right knobs to the API.
        assert kwargs["data_collection"] == "sentinel-2-l2a"
        assert kwargs["width"] == 256 and kwargs["height"] == 256
        assert kwargs["bbox"][0] < kwargs["bbox"][2]
        assert "B04" in kwargs["evalscript"]
        return fake_png

    obs = {
        "observation_id": "fixture-s2-l2a-whitsun-20231212-low-cloud",
        "source": "sentinel-2",
        "timestamp": "2023-12-12T03:00:00+00:00",
    }
    ok, reason, png = fetch_module._process_one(
        obs,
        token="tok",
        aoi_bbox=[114.45, 9.75, 114.85, 10.25],
        width=256, height=256,
        fetcher=fake_fetcher,
    )
    assert ok is True
    assert reason == "ok"
    assert png == fake_png


def test_process_one_with_failing_fetcher_returns_failure(fetch_module):
    def fake_fetcher(**kwargs):
        raise RuntimeError("HTTP 401")

    obs = {
        "observation_id": "fixture-s1-grd-whitsun-20231210",
        "source": "sentinel-1",
        "timestamp": "2023-12-10T22:00:00+00:00",
    }
    ok, reason, png = fetch_module._process_one(
        obs,
        token="tok",
        aoi_bbox=[114.45, 9.75, 114.85, 10.25],
        width=256, height=256,
        fetcher=fake_fetcher,
    )
    assert ok is False
    assert "HTTP 401" in reason
    assert png is None


def test_process_one_rejects_unsupported_source(fetch_module):
    ok, reason, png = fetch_module._process_one(
        {"observation_id": "x", "source": "umbra",
         "timestamp": "2023-01-01T00:00:00Z"},
        token="tok", aoi_bbox=[0, 0, 1, 1], width=10, height=10,
    )
    assert ok is False
    assert "unsupported" in reason.lower()
    assert png is None


# ---------------------------------------------------------------------------
# Full-script main() with mocked OAuth + mocked fetcher
# ---------------------------------------------------------------------------


def test_main_with_missing_credentials_exits_clean(monkeypatch, fetch_module):
    monkeypatch.delenv("SENTINEL_HUB_CLIENT_ID", raising=False)
    monkeypatch.delenv("SENTINEL_HUB_CLIENT_SECRET", raising=False)
    rc = fetch_module.main([])
    assert rc == 0


def test_main_dry_run_with_mock_oauth_and_fetcher_does_not_write(
    monkeypatch, fetch_module, tmp_path,
):
    monkeypatch.setenv("SENTINEL_HUB_CLIENT_ID", "fake")
    monkeypatch.setenv("SENTINEL_HUB_CLIENT_SECRET", "fake")
    monkeypatch.setattr(
        fetch_module, "_get_oauth_token", lambda *a, **kw: "tok",
    )
    monkeypatch.setattr(
        fetch_module, "_request_preview",
        lambda **kwargs: b"\x89PNG\r\n\x1a\n",
    )
    # Mirror the manifest into a tmp file so the script does not
    # mutate the committed fixture.
    manifest_copy = tmp_path / "map_overlays.fixture.json"
    manifest_copy.write_text(
        fetch_module.MANIFEST_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    monkeypatch.setattr(fetch_module, "MANIFEST_PATH", manifest_copy)
    monkeypatch.setattr(fetch_module, "EVIDENCE_DIR", tmp_path / "ev")

    rc = fetch_module.main(["--dry-run"])
    assert rc == 0
    # Dry run must not write any PNGs ...
    assert not (tmp_path / "ev").exists()
    # ... and must leave the manifest unchanged.
    payload = json.loads(manifest_copy.read_text(encoding="utf-8"))
    sentinel = next(
        o for o in payload["overlays"]
        if o.get("observation_id") ==
        "fixture-s2-l2a-whitsun-20231212-low-cloud"
    )
    assert sentinel["image_kind"] == "footprint-only"


def test_main_writes_pngs_and_updates_manifest_with_mocks(
    monkeypatch, fetch_module, tmp_path,
):
    monkeypatch.setenv("SENTINEL_HUB_CLIENT_ID", "fake")
    monkeypatch.setenv("SENTINEL_HUB_CLIENT_SECRET", "fake")

    fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32

    def fake_fetch(**kwargs):
        return fake_png

    monkeypatch.setattr(fetch_module, "_get_oauth_token", lambda *a, **kw: "tok")
    monkeypatch.setattr(fetch_module, "_request_preview", fake_fetch)

    manifest_copy = tmp_path / "map_overlays.fixture.json"
    manifest_copy.write_text(
        fetch_module.MANIFEST_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    monkeypatch.setattr(fetch_module, "MANIFEST_PATH", manifest_copy)
    monkeypatch.setattr(
        fetch_module, "EVIDENCE_DIR", tmp_path / "evidence" / "sentinel",
    )

    rc = fetch_module.main([])
    assert rc == 0

    # Manifest now has Sentinel previews ...
    payload = json.loads(manifest_copy.read_text(encoding="utf-8"))
    sentinel = next(
        o for o in payload["overlays"]
        if o.get("observation_id") ==
        "fixture-s2-l2a-whitsun-20231212-low-cloud"
    )
    assert sentinel["image_kind"] == "sentinel_preview"
    assert sentinel["asset_url"].startswith(
        "/assets/evidence/sentinel/whitsun/"
    )
    assert sentinel["missing_asset_reason"] is None

    # ... and Umbra overlays are untouched.
    umbra = next(
        o for o in payload["overlays"]
        if o.get("source") == "umbra" and o.get("image_kind") == "png"
    )
    assert "/assets/overlays/" in (umbra.get("asset_url") or "")

    # PNGs were written under the redirected evidence dir.
    written = list(
        (tmp_path / "evidence" / "sentinel").rglob("preview.png")
    )
    assert written, "expected at least one preview.png to be written"
    for p in written:
        assert p.read_bytes() == fake_png


def test_main_failure_keeps_footprint_only_with_specific_reason(
    monkeypatch, fetch_module, tmp_path,
):
    monkeypatch.setenv("SENTINEL_HUB_CLIENT_ID", "fake")
    monkeypatch.setenv("SENTINEL_HUB_CLIENT_SECRET", "fake")

    def failing_fetch(**kwargs):
        raise RuntimeError("HTTP 429 throttled")

    monkeypatch.setattr(fetch_module, "_get_oauth_token", lambda *a, **kw: "tok")
    monkeypatch.setattr(fetch_module, "_request_preview", failing_fetch)

    manifest_copy = tmp_path / "map_overlays.fixture.json"
    manifest_copy.write_text(
        fetch_module.MANIFEST_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    monkeypatch.setattr(fetch_module, "MANIFEST_PATH", manifest_copy)
    monkeypatch.setattr(
        fetch_module, "EVIDENCE_DIR", tmp_path / "evidence" / "sentinel",
    )

    rc = fetch_module.main([])
    assert rc == 0  # graceful

    payload = json.loads(manifest_copy.read_text(encoding="utf-8"))
    sentinel = next(
        o for o in payload["overlays"]
        if o.get("observation_id") ==
        "fixture-s2-l2a-whitsun-20231212-low-cloud"
    )
    assert sentinel["image_kind"] == "footprint-only"
    assert sentinel["asset_url"] is None
    assert "HTTP 429" in (sentinel["missing_asset_reason"] or "")
