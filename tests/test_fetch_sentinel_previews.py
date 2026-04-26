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


def _noise_png(seed: int = 42, size: int = 96) -> bytes:
    """Build a noise-pattern PNG that passes content validation:
    > 5 KB and pixel-stddev far above the uniform threshold."""
    import io
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(seed=seed)
    arr = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _placeholder_png() -> bytes:
    """A tiny all-zero PNG (~85 B) that mimics the Sentinel Hub
    "no scene matched" placeholder.  Below the 5 KB size floor,
    so the fetcher's size guard rejects it before variance even
    runs."""
    import io
    import numpy as np
    from PIL import Image
    arr = np.zeros((4, 4), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


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
    """``--dry-run`` must not call live HTTP and must not mutate
    whatever state the manifest is currently in.  Snapshots the
    manifest before, runs the dry-run, and asserts byte-identical
    state afterwards regardless of which Sentinel overlays were
    promoted in earlier real fetches."""
    monkeypatch.setenv("SENTINEL_HUB_CLIENT_ID", "fake")
    monkeypatch.setenv("SENTINEL_HUB_CLIENT_SECRET", "fake")
    monkeypatch.setattr(
        fetch_module, "_get_oauth_token", lambda *a, **kw: "tok",
    )
    monkeypatch.setattr(
        fetch_module, "_request_preview",
        lambda **kwargs: b"\x89PNG\r\n\x1a\n",
    )
    manifest_copy = tmp_path / "map_overlays.fixture.json"
    original_text = fetch_module.MANIFEST_PATH.read_text(encoding="utf-8")
    manifest_copy.write_text(original_text, encoding="utf-8")
    monkeypatch.setattr(fetch_module, "MANIFEST_PATH", manifest_copy)
    monkeypatch.setattr(fetch_module, "EVIDENCE_DIR", tmp_path / "ev")

    rc = fetch_module.main(["--dry-run"])
    assert rc == 0
    # Dry run must not write any PNGs ...
    assert not (tmp_path / "ev").exists()
    # ... and must leave the manifest byte-identical to the snapshot.
    assert manifest_copy.read_text(encoding="utf-8") == original_text


def test_main_writes_pngs_and_updates_manifest_with_mocks(
    monkeypatch, fetch_module, tmp_path,
):
    """Each observation gets a unique valid noise PNG so all 7
    Whitsun observations get promoted (no validation rejection,
    no within-run duplicate detection)."""
    monkeypatch.setenv("SENTINEL_HUB_CLIENT_ID", "fake")
    monkeypatch.setenv("SENTINEL_HUB_CLIENT_SECRET", "fake")

    counter = {"i": 0}

    def fake_fetch(**kwargs):
        # Use a different seed per call so every preview is unique
        # and passes the within-run dedup check.
        counter["i"] += 1
        return _noise_png(seed=counter["i"])

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
    # Every written preview is the noise pattern the mock produced.
    for p in written:
        assert p.stat().st_size >= fetch_module.MIN_PREVIEW_SIZE_BYTES


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


# ---------------------------------------------------------------------------
# --scenario filter
# ---------------------------------------------------------------------------


def test_fixtures_for_scenario_helper(fetch_module):
    whit = fetch_module._fixtures_for_scenario("whitsun")
    tenn = fetch_module._fixtures_for_scenario("tennent")
    both = fetch_module._fixtures_for_scenario("all")
    assert whit == (fetch_module.SCENARIO_FIXTURES["whitsun"],)
    assert tenn == (fetch_module.SCENARIO_FIXTURES["tennent"],)
    assert set(both) == set(fetch_module.SCENARIO_FIXTURES.values())
    with pytest.raises(ValueError):
        fetch_module._fixtures_for_scenario("bogus")


def test_default_scenario_is_whitsun_in_dry_run(
    monkeypatch, fetch_module, capsys,
):
    """No --scenario argument → default ``whitsun`` (narrow scope)."""
    # No creds intentionally; dry-run must skip live HTTP entirely.
    monkeypatch.delenv("SENTINEL_HUB_CLIENT_ID", raising=False)
    monkeypatch.delenv("SENTINEL_HUB_CLIENT_SECRET", raising=False)
    rc = fetch_module.main(["--dry-run"])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "scenario=whitsun" in captured
    # Only Whitsun observation ids surface; no Tennent ids leak.
    assert "fixture-s1-grd-whitsun-" in captured
    assert "fixture-s2-l2a-whitsun-" in captured
    assert "tennent" not in captured.lower()


def test_dry_run_scenario_whitsun_lists_seven_observations(
    monkeypatch, fetch_module, capsys,
):
    monkeypatch.delenv("SENTINEL_HUB_CLIENT_ID", raising=False)
    monkeypatch.delenv("SENTINEL_HUB_CLIENT_SECRET", raising=False)
    rc = fetch_module.main(["--dry-run", "--scenario", "whitsun"])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "scenario=whitsun" in captured
    # 3 original + 4 in-between coverage records = 7.
    assert "7 preview(s) would be fetched" in captured
    assert "tennent" not in captured.lower()


def test_dry_run_scenario_tennent_lists_three_observations(
    monkeypatch, fetch_module, capsys,
):
    monkeypatch.delenv("SENTINEL_HUB_CLIENT_ID", raising=False)
    monkeypatch.delenv("SENTINEL_HUB_CLIENT_SECRET", raising=False)
    rc = fetch_module.main(["--dry-run", "--scenario", "tennent"])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "scenario=tennent" in captured
    assert "3 preview(s) would be fetched" in captured
    # No Whitsun observation ids should leak through the Tennent
    # scoping.
    assert "fixture-s1-grd-whitsun" not in captured
    assert "fixture-s2-l2a-whitsun" not in captured
    # Tennent ids do appear.
    assert "fixture-s1-grd-tennent" in captured


def test_dry_run_scenario_all_lists_ten_observations(
    monkeypatch, fetch_module, capsys,
):
    monkeypatch.delenv("SENTINEL_HUB_CLIENT_ID", raising=False)
    monkeypatch.delenv("SENTINEL_HUB_CLIENT_SECRET", raising=False)
    rc = fetch_module.main(["--dry-run", "--scenario", "all"])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "scenario=all" in captured
    # 7 Whitsun + 3 Tennent = 10.
    assert "10 preview(s) would be fetched" in captured


def test_scenario_unknown_value_rejected_by_argparse(fetch_module):
    """Argparse should reject any value outside the documented choices."""
    with pytest.raises(SystemExit):
        fetch_module.main(["--scenario", "bogus", "--dry-run"])


def test_dry_run_does_not_call_live_http(
    monkeypatch, fetch_module,
):
    """The new dry-run skips OAuth + Process API entirely so it can
    run without credentials and without burning live API quota."""
    monkeypatch.setenv("SENTINEL_HUB_CLIENT_ID", "fake")
    monkeypatch.setenv("SENTINEL_HUB_CLIENT_SECRET", "fake")

    oauth_calls: list = []
    process_calls: list = []

    def boom_oauth(*args, **kwargs):
        oauth_calls.append((args, kwargs))
        raise AssertionError("OAuth must not run during --dry-run")

    def boom_process(**kwargs):
        process_calls.append(kwargs)
        raise AssertionError("Process API must not run during --dry-run")

    monkeypatch.setattr(fetch_module, "_get_oauth_token", boom_oauth)
    monkeypatch.setattr(fetch_module, "_request_preview", boom_process)

    rc = fetch_module.main(["--dry-run", "--scenario", "all"])
    assert rc == 0
    assert oauth_calls == []
    assert process_calls == []


def test_live_main_with_scenario_whitsun_does_not_touch_tennent(
    monkeypatch, fetch_module, tmp_path,
):
    """Live mode (mocked fetcher) under --scenario whitsun must
    write only Whitsun PNGs and update only Whitsun overlay records."""
    monkeypatch.setenv("SENTINEL_HUB_CLIENT_ID", "fake")
    monkeypatch.setenv("SENTINEL_HUB_CLIENT_SECRET", "fake")

    counter = {"i": 0}

    def fake_fetch(**kwargs):
        counter["i"] += 1
        return _noise_png(seed=counter["i"])

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

    rc = fetch_module.main(["--scenario", "whitsun"])
    assert rc == 0

    # PNGs only under whitsun/, never under tennent/.
    written = list((tmp_path / "evidence" / "sentinel").rglob("preview.png"))
    assert written, "expected Whitsun PNGs to be written"
    for p in written:
        assert "whitsun" in str(p)
        assert "tennent" not in str(p)

    # Tennent overlays in the manifest are still footprint-only.
    payload = json.loads(manifest_copy.read_text(encoding="utf-8"))
    tennent = [
        o for o in payload["overlays"]
        if o.get("scenario_id") == "tennent"
        and o.get("source") in ("sentinel-1", "sentinel-2")
    ]
    assert tennent, "manifest should still contain Tennent Sentinel overlays"
    for o in tennent:
        assert o["image_kind"] == "footprint-only", o["overlay_id"]
        assert o["asset_url"] is None
        assert o["image_path"] is None


# ---------------------------------------------------------------------------
# Quality gates: size + variance + dedup
# ---------------------------------------------------------------------------


def test_validate_preview_png_rejects_empty_body(fetch_module):
    ok, reason = fetch_module._validate_preview_png(b"")
    assert ok is False
    assert "empty" in reason.lower()


def test_validate_preview_png_rejects_too_small(fetch_module):
    """The Sentinel Hub placeholder is ~334 B; any sub-5 KB body is
    treated as a placeholder regardless of its pixel content."""
    placeholder = _placeholder_png()
    assert len(placeholder) < fetch_module.MIN_PREVIEW_SIZE_BYTES
    ok, reason = fetch_module._validate_preview_png(placeholder)
    assert ok is False
    assert "too small" in reason.lower()


def test_validate_preview_png_rejects_uniform_image(fetch_module, monkeypatch):
    """A fully-zero PNG is uniform.  Patch ``MIN_PREVIEW_SIZE_BYTES``
    to 1 so the size guard doesn't pre-empt the variance check, then
    verify the variance branch fires."""
    import io
    import numpy as np
    from PIL import Image
    monkeypatch.setattr(fetch_module, "MIN_PREVIEW_SIZE_BYTES", 1)
    arr = np.zeros((128, 128, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    ok, reason = fetch_module._validate_preview_png(buf.getvalue())
    assert ok is False
    assert "uniform" in reason.lower()


def test_validate_preview_png_accepts_noisy_image(fetch_module):
    ok, reason = fetch_module._validate_preview_png(_noise_png())
    assert ok is True
    assert reason == "ok"


def test_main_does_not_promote_empty_placeholder(
    monkeypatch, fetch_module, tmp_path,
):
    """Sentinel Hub returns the 334 B all-zero placeholder for every
    request → no promotion, every overlay stays footprint-only with
    the new "empty placeholder" missing_asset_reason."""
    monkeypatch.setenv("SENTINEL_HUB_CLIENT_ID", "fake")
    monkeypatch.setenv("SENTINEL_HUB_CLIENT_SECRET", "fake")
    monkeypatch.setattr(fetch_module, "_get_oauth_token", lambda *a, **kw: "tok")
    placeholder = _placeholder_png()
    monkeypatch.setattr(
        fetch_module, "_request_preview", lambda **kwargs: placeholder,
    )

    manifest_copy = tmp_path / "map_overlays.fixture.json"
    manifest_copy.write_text(
        fetch_module.MANIFEST_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    monkeypatch.setattr(fetch_module, "MANIFEST_PATH", manifest_copy)
    monkeypatch.setattr(
        fetch_module, "EVIDENCE_DIR", tmp_path / "evidence" / "sentinel",
    )

    rc = fetch_module.main(["--scenario", "whitsun"])
    assert rc == 0
    # No PNGs written.
    written = list((tmp_path / "evidence" / "sentinel").rglob("preview.png"))
    assert written == []
    # All Whitsun Sentinel overlays remain footprint-only with the
    # new placeholder reason.
    payload = json.loads(manifest_copy.read_text(encoding="utf-8"))
    whitsun_sentinel = [
        o for o in payload["overlays"]
        if o.get("scenario_id") == "whitsun"
        and o.get("source") in ("sentinel-1", "sentinel-2")
    ]
    assert whitsun_sentinel
    for o in whitsun_sentinel:
        assert o["image_kind"] == "footprint-only", o["overlay_id"]
        assert o["asset_url"] is None
        assert "empty placeholder" in (o["missing_asset_reason"] or "")
        # Honesty markers preserved.
        assert o["weak_signal"] is True
        assert o["confirmation_layer"] is False


def test_main_promotes_first_duplicate_only(
    monkeypatch, fetch_module, tmp_path,
):
    """Sentinel Hub returns the SAME bytes for every request (mimics
    the Process API serving one scene to overlapping ±2-day windows).
    The first observation is promoted; later ones stay footprint-only
    with the duplicate reason."""
    monkeypatch.setenv("SENTINEL_HUB_CLIENT_ID", "fake")
    monkeypatch.setenv("SENTINEL_HUB_CLIENT_SECRET", "fake")
    monkeypatch.setattr(fetch_module, "_get_oauth_token", lambda *a, **kw: "tok")
    same_png = _noise_png(seed=7)
    monkeypatch.setattr(
        fetch_module, "_request_preview", lambda **kwargs: same_png,
    )

    manifest_copy = tmp_path / "map_overlays.fixture.json"
    manifest_copy.write_text(
        fetch_module.MANIFEST_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    monkeypatch.setattr(fetch_module, "MANIFEST_PATH", manifest_copy)
    monkeypatch.setattr(
        fetch_module, "EVIDENCE_DIR", tmp_path / "evidence" / "sentinel",
    )

    rc = fetch_module.main(["--scenario", "whitsun"])
    assert rc == 0

    # Exactly one PNG written.
    written = list((tmp_path / "evidence" / "sentinel").rglob("preview.png"))
    assert len(written) == 1, [p.name for p in written]

    payload = json.loads(manifest_copy.read_text(encoding="utf-8"))
    promoted = [
        o for o in payload["overlays"]
        if o.get("scenario_id") == "whitsun"
        and o.get("source") in ("sentinel-1", "sentinel-2")
        and o["image_kind"] == "sentinel_preview"
    ]
    duplicates = [
        o for o in payload["overlays"]
        if o.get("scenario_id") == "whitsun"
        and o.get("source") in ("sentinel-1", "sentinel-2")
        and o["image_kind"] == "footprint-only"
        and "Duplicate preview of" in (o["missing_asset_reason"] or "")
    ]
    # Exactly one promoted, the rest carry the duplicate reason.
    assert len(promoted) == 1, [o["overlay_id"] for o in promoted]
    assert len(duplicates) >= 6
    # Duplicate reason references the first promoted observation id.
    first_id = promoted[0]["observation_id"]
    for o in duplicates:
        assert first_id in o["missing_asset_reason"]
        assert o["weak_signal"] is True
        assert o["confirmation_layer"] is False


def test_main_mixed_state_some_promote_some_reject(
    monkeypatch, fetch_module, tmp_path,
):
    """Mock that returns valid noise for Sentinel-2 and the empty
    placeholder for Sentinel-1, mirroring the real-world result of
    the live fetch.  Verifies the manifest ends up in mixed state:
    promoted S2 + footprint-only S1, with Umbra untouched."""
    monkeypatch.setenv("SENTINEL_HUB_CLIENT_ID", "fake")
    monkeypatch.setenv("SENTINEL_HUB_CLIENT_SECRET", "fake")
    monkeypatch.setattr(fetch_module, "_get_oauth_token", lambda *a, **kw: "tok")

    placeholder = _placeholder_png()
    counter = {"i": 0}

    def hybrid_fetch(**kwargs):
        # The dataFilter routing tags S1 with "sentinel-1-grd" and
        # S2 with "sentinel-2-l2a"; use that to decide what to return.
        if kwargs.get("data_collection") == "sentinel-1-grd":
            return placeholder
        counter["i"] += 1
        return _noise_png(seed=counter["i"])

    monkeypatch.setattr(fetch_module, "_request_preview", hybrid_fetch)

    manifest_copy = tmp_path / "map_overlays.fixture.json"
    manifest_copy.write_text(
        fetch_module.MANIFEST_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    monkeypatch.setattr(fetch_module, "MANIFEST_PATH", manifest_copy)
    monkeypatch.setattr(
        fetch_module, "EVIDENCE_DIR", tmp_path / "evidence" / "sentinel",
    )

    rc = fetch_module.main(["--scenario", "whitsun"])
    assert rc == 0

    payload = json.loads(manifest_copy.read_text(encoding="utf-8"))

    # Sentinel-2 promoted ...
    s2 = [
        o for o in payload["overlays"]
        if o.get("scenario_id") == "whitsun" and o.get("source") == "sentinel-2"
    ]
    assert s2
    promoted_s2 = [o for o in s2 if o["image_kind"] == "sentinel_preview"]
    assert promoted_s2, "expected at least one Sentinel-2 promotion"
    for o in promoted_s2:
        assert o["asset_url"]
        assert o["weak_signal"] is True
        assert o["confirmation_layer"] is False

    # ... Sentinel-1 stays footprint-only.
    s1 = [
        o for o in payload["overlays"]
        if o.get("scenario_id") == "whitsun" and o.get("source") == "sentinel-1"
    ]
    assert s1
    for o in s1:
        assert o["image_kind"] == "footprint-only", o["overlay_id"]
        assert o["asset_url"] is None
        assert "empty placeholder" in (o["missing_asset_reason"] or "")
        assert o["weak_signal"] is True
        assert o["confirmation_layer"] is False

    # Umbra confirmation layer is unchanged in this run — it has its
    # own preview pipeline (scripts/30_*) and never touches the
    # Sentinel evidence directory.
    umbra = [
        o for o in payload["overlays"]
        if o.get("source") == "umbra" and o.get("image_kind") == "png"
    ]
    assert umbra
    for o in umbra:
        assert o.get("confirmation_layer") is True


def test_main_quality_gates_status_summary(
    monkeypatch, fetch_module, tmp_path, capsys,
):
    """Live run prints a final per-status summary line so the operator
    can see counts at a glance without re-reading individual lines."""
    monkeypatch.setenv("SENTINEL_HUB_CLIENT_ID", "fake")
    monkeypatch.setenv("SENTINEL_HUB_CLIENT_SECRET", "fake")
    monkeypatch.setattr(fetch_module, "_get_oauth_token", lambda *a, **kw: "tok")
    monkeypatch.setattr(
        fetch_module, "_request_preview",
        lambda **kwargs: _placeholder_png(),
    )

    manifest_copy = tmp_path / "map_overlays.fixture.json"
    manifest_copy.write_text(
        fetch_module.MANIFEST_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    monkeypatch.setattr(fetch_module, "MANIFEST_PATH", manifest_copy)
    monkeypatch.setattr(
        fetch_module, "EVIDENCE_DIR", tmp_path / "evidence" / "sentinel",
    )

    rc = fetch_module.main(["--scenario", "whitsun"])
    assert rc == 0
    captured = capsys.readouterr().out
    # Headline summary names every status bucket.
    for term in ("fetched", "promoted", "empty", "duplicate", "failed"):
        assert term in captured, f"summary missing {term!r}: {captured!r}"
