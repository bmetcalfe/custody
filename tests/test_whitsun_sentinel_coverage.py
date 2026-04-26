"""Tests for the Whitsun Sentinel in-between-coverage fixture.

These pin the metadata-only Sentinel coverage of the small Whitsun
interest zone across the demo window 2023-12-04 → 2023-12-15.  The
fixture is the source the dashboard reads to answer

    "What public observations covered the Whitsun interest zone
     between Umbra collects?"

Sentinel must remain a weak-signal cueing layer — the fixture is
metadata-only, no imagery is downloaded by the loader, and no record
must claim Sentinel as confirmation evidence.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
AOI_PATH = REPO_ROOT / "data" / "demo" / "whitsun_aoi.fixture.geojson"
OBS_PATH = REPO_ROOT / "data" / "demo" / "whitsun_sentinel_observations.fixture.json"
MAP_OVERLAYS_PATH = REPO_ROOT / "data" / "demo" / "map_overlays.fixture.json"


# ---------------------------------------------------------------------------
# AOI fixture: tight bbox, fixture/placeholder/not-operational labels
# ---------------------------------------------------------------------------


def test_whitsun_aoi_fixture_has_required_metadata_labels() -> None:
    blob = json.loads(AOI_PATH.read_text(encoding="utf-8"))
    md = blob["metadata"]
    # Required labels per dispatch.
    assert md.get("data_mode") == "fixture"
    description = (md.get("description") or "").lower()
    assert "placeholder" in description
    assert "operational" in description  # "not the real operational AOI"
    # The bbox must be the tight Whitsun interest zone.
    assert md.get("bbox_lon_west_deg") == pytest.approx(114.45)
    assert md.get("bbox_lon_east_deg") == pytest.approx(114.85)
    assert md.get("bbox_lat_south_deg") == pytest.approx(9.75)
    assert md.get("bbox_lat_north_deg") == pytest.approx(10.25)
    # Center matches the project-wide _WHITSUN_CENTER constant.
    assert md.get("center_lat_deg") == pytest.approx(9.98)
    assert md.get("center_lon_deg") == pytest.approx(114.63)
    # Caveats explicitly disclaim operational use.
    caveats = " ".join(md.get("caveats") or ()).lower()
    assert "demo" in caveats
    assert "operational" in caveats


# ---------------------------------------------------------------------------
# Sentinel observation fixture: in-between coverage
# ---------------------------------------------------------------------------


def _load_observations() -> list[dict]:
    blob = json.loads(OBS_PATH.read_text(encoding="utf-8"))
    return list(blob["observations"])


def _ts(iso: str) -> datetime:
    return datetime.fromisoformat(iso.replace("Z", "+00:00"))


def test_whitsun_sentinel_fixture_has_seven_observations() -> None:
    """3 original + 4 in-between coverage records = 7."""
    obs = _load_observations()
    assert len(obs) == 7


def test_whitsun_sentinel_metadata_count_matches_records() -> None:
    blob = json.loads(OBS_PATH.read_text(encoding="utf-8"))
    md = blob["metadata"]
    assert md["count"] == len(blob["observations"])


def test_whitsun_sentinel_covers_pre_between_and_post_umbra() -> None:
    """The fixture must show coverage across the demo window — not
    just the original two reveal-step records."""
    obs = _load_observations()
    pre_umbra = [o for o in obs if _ts(o["timestamp"]).date() < datetime(2023, 12, 6).date()]
    between = [
        o for o in obs
        if datetime(2023, 12, 6).date()
        < _ts(o["timestamp"]).date() < datetime(2023, 12, 13).date()
    ]
    post_followup = [
        o for o in obs
        if _ts(o["timestamp"]).date() > datetime(2023, 12, 13).date()
    ]
    assert pre_umbra, "expected at least one pre-Umbra Sentinel observation"
    assert between, "expected Sentinel observations between Umbra primary and follow-up"
    assert post_followup, "expected at least one post-follow-up Sentinel observation"


def test_whitsun_sentinel_includes_both_s1_and_s2() -> None:
    """The in-between coverage picture is incomplete without both
    SAR and optical."""
    obs = _load_observations()
    sources = {o["source"] for o in obs}
    assert "sentinel-1" in sources
    assert "sentinel-2" in sources
    s1 = [o for o in obs if o["source"] == "sentinel-1"]
    s2 = [o for o in obs if o["source"] == "sentinel-2"]
    assert len(s1) >= 2, "expected at least two Sentinel-1 acquisitions"
    assert len(s2) >= 3, "expected at least three Sentinel-2 acquisitions"


def test_whitsun_sentinel_records_carry_full_schema() -> None:
    """Every observation record carries the metadata fields the
    dispatch enumerates."""
    obs = _load_observations()
    required_keys = {
        "observation_id", "source", "collection",
        "timestamp", "datetime_start", "datetime_end",
        "sensor_type", "platform", "instrument",
        "resolution_m", "cloud_coverage", "polarization",
        "geometry", "bbox", "asset_links", "thumbnail_url",
        "confidence_weight", "usable_for_detection",
        "usable_for_context", "data_mode", "caveats",
        "raw_properties",
    }
    for o in obs:
        missing = required_keys - set(o.keys())
        assert not missing, (
            f"{o.get('observation_id')} missing keys: {sorted(missing)}"
        )


def test_whitsun_sentinel_records_use_aoi_bbox_not_broad_region() -> None:
    """All records must sit inside the small AOI bbox.  Any record
    whose bbox extends past 114.45/9.75 → 114.85/10.25 would suggest
    the broader Spratly region was ingested by mistake."""
    aoi = (114.45, 9.75, 114.85, 10.25)
    obs = _load_observations()
    for o in obs:
        bbox = o.get("bbox")
        assert bbox and len(bbox) == 4, o["observation_id"]
        w, s, e, n = bbox
        assert w >= aoi[0] - 1e-9, o["observation_id"]
        assert s >= aoi[1] - 1e-9, o["observation_id"]
        assert e <= aoi[2] + 1e-9, o["observation_id"]
        assert n <= aoi[3] + 1e-9, o["observation_id"]


def test_whitsun_sentinel_records_are_metadata_only() -> None:
    """No record must claim it bundled SAFE products or downloaded
    rasters; the loader is metadata-only."""
    obs = _load_observations()
    for o in obs:
        caveats = " ".join(o.get("caveats") or ()).lower()
        assert "metadata-only" in caveats, o["observation_id"]
        # No record must advertise itself as confirmation evidence.
        assert "confirmed detection" not in caveats, o["observation_id"]
        assert "definitive change" not in caveats, o["observation_id"]


def test_whitsun_sentinel_cloud_coverage_drives_usable_for_detection() -> None:
    """S2 records with cloud > 25% must have usable_for_detection=False
    so callers cannot accidentally treat them as detection-quality."""
    obs = _load_observations()
    for o in obs:
        if o["source"] != "sentinel-2":
            continue
        cc = o.get("cloud_coverage")
        if cc is not None and cc > 25.0:
            assert o["usable_for_detection"] is False, (
                f"{o['observation_id']} cloud {cc}% must not be usable for detection"
            )


# ---------------------------------------------------------------------------
# Map overlay manifest mirrors the observation fixture
# ---------------------------------------------------------------------------


def test_map_overlays_have_one_record_per_whitsun_sentinel_observation() -> None:
    obs = _load_observations()
    obs_ids = {o["observation_id"] for o in obs}

    payload = json.loads(MAP_OVERLAYS_PATH.read_text(encoding="utf-8"))
    overlays = payload["overlays"]
    overlay_obs_ids = {
        o.get("observation_id") for o in overlays
        if o.get("scenario_id") == "whitsun"
        and o.get("source") in ("sentinel-1", "sentinel-2")
    }
    missing = obs_ids - overlay_obs_ids
    assert not missing, (
        f"missing Sentinel overlay records for: {sorted(missing)}"
    )


def test_map_overlays_in_between_records_default_to_off() -> None:
    """In-between coverage records must default to off so the map
    isn't crowded; only the trace-narrative pair is on by default."""
    payload = json.loads(MAP_OVERLAYS_PATH.read_text(encoding="utf-8"))
    overlays = payload["overlays"]
    # The original two narrative records (2023-12-10 S1, 2023-12-12 S2)
    # remain default_visible=True; the four in-between records must be
    # default_visible=False.
    for o in overlays:
        if o.get("scenario_id") != "whitsun":
            continue
        if o.get("source") not in ("sentinel-1", "sentinel-2"):
            continue
        ct = o.get("collection_time")
        if ct in ("2023-12-04", "2023-12-07", "2023-12-09", "2023-12-14"):
            assert o["default_visible"] is False, ct


def test_no_sentinel_overlay_carries_confirmation_framing() -> None:
    """Cross-cutting honesty guard: every Sentinel overlay (including
    the new in-between records) must remain a weak-signal layer."""
    payload = json.loads(MAP_OVERLAYS_PATH.read_text(encoding="utf-8"))
    for o in payload["overlays"]:
        if o.get("source") not in ("sentinel-1", "sentinel-2"):
            continue
        assert o.get("weak_signal") is True, o["overlay_id"]
        assert o.get("confirmation_layer") is False, o["overlay_id"]
        blob = (
            (o.get("display_name") or "")
            + " "
            + " ".join(o.get("caveats") or ())
        ).lower()
        for forbidden in (
            "definitive change", "sentinel proves", "confirmed detection",
        ):
            assert forbidden not in blob, (
                f"{o['overlay_id']} carries forbidden phrase {forbidden!r}"
            )


# ---------------------------------------------------------------------------
# The new overlays plug into the existing dynamic overlay manager
# ---------------------------------------------------------------------------


def _import_evidence_callbacks():
    src_app = REPO_ROOT / "src" / "app"
    src = REPO_ROOT / "src"
    for p in (str(src), str(src_app)):
        if p not in sys.path:
            sys.path.insert(0, p)
    for k in list(sys.modules):
        if k.startswith("layout.") or k.startswith("callbacks."):
            del sys.modules[k]
    import callbacks.evidence_viewer as mod
    return mod


def test_whitsun_sentinel_cueing_lists_all_in_between_overlays_at_event_six() -> None:
    """At Whitsun event 06 (Sentinel-1 context observation arrives,
    after Candidate tracks at ev-04 and Sentinel-2 at ev-05), the
    cueing-context section must list every Sentinel overlay revealed
    so far — including the new in-between records."""
    cb = _import_evidence_callbacks()
    from custody.demo import available_overlays_for
    available = available_overlays_for(
        cb._OVERLAYS, scenario_id="whitsun", current_ordinal=6,
    )
    sentinel = [
        o for o in available
        if o.source in ("sentinel-1", "sentinel-2")
    ]
    obs_ids = {o.observation_id for o in sentinel}
    # All seven Whitsun Sentinel observations should be revealed by
    # event 06 (S2 reveals at ord 5, S1 at ord 6).
    expected = {
        "fixture-s1-grd-whitsun-20231210",
        "fixture-s2-l2a-whitsun-20231212-low-cloud",
        "fixture-s2-l2a-whitsun-20231215-cloudy",
        "fixture-s1-grd-whitsun-20231204",
        "fixture-s1-grd-whitsun-20231209",
        "fixture-s2-l2a-whitsun-20231207-mid-cloud",
        "fixture-s2-l2a-whitsun-20231214-low-cloud",
    }
    assert expected <= obs_ids, (
        f"missing in-between coverage at event 6: {expected - obs_ids}"
    )
