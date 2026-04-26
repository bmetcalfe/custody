"""Tests for :mod:`custody.ingest.sentinel` (Sentinel ingestion sprint)."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from custody.ingest.sentinel import (
    CONFIDENCE_WEIGHTS,
    DEFAULT_COLLECTIONS,
    ObservationArtifact,
    S2_LOW_CLOUD_PCT,
    SENTINEL_1_GRD_COLLECTION,
    SENTINEL_2_L2A_COLLECTION,
    best_low_cloud_sentinel_2,
    date_range,
    latest_sentinel_1,
    load_observation_cache,
    normalize_stac_item,
    write_observation_cache,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
STAC_FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "sentinel_stac"
DEMO_FIXTURE = (
    REPO_ROOT / "data" / "demo" / "whitsun_sentinel_observations.fixture.json"
)
TENNENT_AOI_FIXTURE = (
    REPO_ROOT / "data" / "demo" / "tennent_aoi.fixture.geojson"
)
TENNENT_DEMO_FIXTURE = (
    REPO_ROOT / "data" / "demo" / "tennent_sentinel_observations.fixture.json"
)


def _load_stac(filename: str) -> dict:
    return json.loads(
        (STAC_FIXTURE_DIR / filename).read_text(encoding="utf-8"),
    )


# ---------------------------------------------------------------------------
# Sentinel-1 normalization
# ---------------------------------------------------------------------------


def test_normalize_sentinel_1_basic() -> None:
    item = _load_stac("sentinel_1_grd_item.json")
    obs = normalize_stac_item(item)
    assert isinstance(obs, ObservationArtifact)
    assert obs.source == "sentinel-1"
    assert obs.sensor_type == "sar"
    assert obs.collection == SENTINEL_1_GRD_COLLECTION
    assert obs.platform == "sentinel-1a"
    assert obs.instrument == "c-sar"
    assert "VV" in obs.polarization
    assert "VH" in obs.polarization
    assert obs.orbit_direction == "ascending"
    assert obs.cloud_coverage is None
    assert obs.usable_for_detection is True
    assert obs.usable_for_context is True
    assert obs.confidence_weight == CONFIDENCE_WEIGHTS["sentinel-1"]
    assert obs.bbox == (114.45, 9.75, 114.85, 10.25)
    assert obs.thumbnail_url is not None


def test_normalize_sentinel_1_data_mode_default_real() -> None:
    obs = normalize_stac_item(_load_stac("sentinel_1_grd_item.json"))
    assert obs.data_mode == "real"


# ---------------------------------------------------------------------------
# Sentinel-2 normalization
# ---------------------------------------------------------------------------


def test_normalize_sentinel_2_low_cloud() -> None:
    item = _load_stac("sentinel_2_l2a_low_cloud_item.json")
    obs = normalize_stac_item(item)
    assert obs.source == "sentinel-2"
    assert obs.sensor_type == "optical"
    assert obs.collection == SENTINEL_2_L2A_COLLECTION
    assert obs.cloud_coverage == 8.0
    assert obs.usable_for_detection is True
    assert obs.usable_for_context is True
    assert obs.confidence_weight == CONFIDENCE_WEIGHTS["sentinel-2-low-cloud"]


def test_normalize_sentinel_2_high_cloud() -> None:
    item = _load_stac("sentinel_2_l2a_high_cloud_item.json")
    obs = normalize_stac_item(item)
    assert obs.source == "sentinel-2"
    assert obs.cloud_coverage == 72.0
    assert obs.usable_for_detection is False
    assert obs.usable_for_context is True
    assert obs.confidence_weight == CONFIDENCE_WEIGHTS["sentinel-2-cloudy"]
    cloudy_caveats = [c for c in obs.caveats if "cloud coverage" in c.lower()]
    assert cloudy_caveats, (
        "expected an explicit high-cloud caveat on Sentinel-2 cloudy records"
    )


def test_high_cloud_threshold_constant() -> None:
    assert S2_LOW_CLOUD_PCT == 25.0


# ---------------------------------------------------------------------------
# Confidence ordering
# ---------------------------------------------------------------------------


def test_confidence_ordering_umbra_above_sentinel_1_above_sentinel_2() -> None:
    assert (
        CONFIDENCE_WEIGHTS["umbra"]
        > CONFIDENCE_WEIGHTS["sentinel-1"]
        > CONFIDENCE_WEIGHTS["sentinel-2-low-cloud"]
        > CONFIDENCE_WEIGHTS["sentinel-2-cloudy"]
    )


def test_normalized_records_respect_confidence_ordering() -> None:
    s1 = normalize_stac_item(_load_stac("sentinel_1_grd_item.json"))
    s2_low = normalize_stac_item(_load_stac("sentinel_2_l2a_low_cloud_item.json"))
    s2_high = normalize_stac_item(_load_stac("sentinel_2_l2a_high_cloud_item.json"))
    assert s1.confidence_weight > s2_low.confidence_weight > s2_high.confidence_weight


# ---------------------------------------------------------------------------
# Cache round-trip
# ---------------------------------------------------------------------------


def test_cache_round_trip(tmp_path: Path) -> None:
    s1 = normalize_stac_item(_load_stac("sentinel_1_grd_item.json"))
    s2 = normalize_stac_item(_load_stac("sentinel_2_l2a_low_cloud_item.json"))
    out = tmp_path / "cache.json"
    write_observation_cache(
        (s1, s2), out, metadata={"data_mode": "real", "test": "round-trip"},
    )
    loaded = load_observation_cache(out)
    assert len(loaded) == 2
    by_id = {o.observation_id: o for o in loaded}
    assert by_id[s1.observation_id].source == "sentinel-1"
    assert by_id[s2.observation_id].source == "sentinel-2"
    assert by_id[s2.observation_id].cloud_coverage == 8.0
    # Top-level metadata is preserved.
    blob = json.loads(out.read_text(encoding="utf-8"))
    assert blob["metadata"]["test"] == "round-trip"
    assert blob["metadata"]["count"] == 2


def test_cache_load_demo_fixture() -> None:
    obs = load_observation_cache(DEMO_FIXTURE)
    assert len(obs) == 3
    assert all(o.data_mode == "fixture" for o in obs)
    sources = {o.source for o in obs}
    assert sources == {"sentinel-1", "sentinel-2"}


# ---------------------------------------------------------------------------
# Data mode validation
# ---------------------------------------------------------------------------


def test_invalid_data_mode_raises() -> None:
    item = _load_stac("sentinel_1_grd_item.json")
    with pytest.raises(ValueError):
        normalize_stac_item(item, data_mode="bogus")


def test_fixture_data_mode_adds_caveat() -> None:
    item = _load_stac("sentinel_1_grd_item.json")
    obs = normalize_stac_item(item, data_mode="fixture")
    joined = " ".join(obs.caveats).lower()
    assert "fixture record" in joined


# ---------------------------------------------------------------------------
# Default collections
# ---------------------------------------------------------------------------


def test_default_collections_listed() -> None:
    assert SENTINEL_1_GRD_COLLECTION in DEFAULT_COLLECTIONS
    assert SENTINEL_2_L2A_COLLECTION in DEFAULT_COLLECTIONS


# ---------------------------------------------------------------------------
# Demo helpers
# ---------------------------------------------------------------------------


def test_best_low_cloud_sentinel_2_picks_lowest_cloud() -> None:
    obs = load_observation_cache(DEMO_FIXTURE)
    best = best_low_cloud_sentinel_2(obs)
    assert best is not None
    assert best.source == "sentinel-2"
    assert best.cloud_coverage == 8.0


def test_latest_sentinel_1_returns_record() -> None:
    obs = load_observation_cache(DEMO_FIXTURE)
    latest = latest_sentinel_1(obs)
    assert latest is not None
    assert latest.source == "sentinel-1"


def test_date_range_returns_min_max_iso() -> None:
    obs = load_observation_cache(DEMO_FIXTURE)
    start, end = date_range(obs)
    assert start is not None and end is not None
    assert start <= end


# ---------------------------------------------------------------------------
# Guardrails: import boundary, no live HTTP at import
# ---------------------------------------------------------------------------


def test_no_forbidden_imports() -> None:
    import custody.ingest.sentinel as mod
    tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
    forbidden = (
        "custody.detection",
        "custody.fusion.tracker",
        "custody.taskrecommendation",
        "sentinelhub",
        "pystac_client",
        "pystac",
    )
    offending: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if any(alias.name.startswith(p) for p in forbidden):
                    offending.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod_name = node.module or ""
            if any(mod_name.startswith(p) for p in forbidden):
                offending.append(mod_name)
    assert not offending, f"forbidden imports: {offending}"


def test_module_does_not_make_network_calls_at_import() -> None:
    # Re-importing the module should not raise even with no network.
    import importlib
    import custody.ingest.sentinel as mod
    importlib.reload(mod)


# ---------------------------------------------------------------------------
# Tennent AOI extension
# ---------------------------------------------------------------------------


def test_tennent_aoi_fixture_loads() -> None:
    blob = json.loads(TENNENT_AOI_FIXTURE.read_text(encoding="utf-8"))
    assert blob["type"] == "FeatureCollection"
    assert blob["metadata"]["data_mode"] == "fixture"
    assert blob["metadata"]["scenario_role"] == (
        "fixed-site monitoring / construction context"
    )
    geom = blob["features"][0]["geometry"]
    assert geom["type"] == "Polygon"


def test_tennent_observation_fixture_loads() -> None:
    obs = load_observation_cache(TENNENT_DEMO_FIXTURE)
    assert len(obs) == 3
    assert all(o.data_mode == "fixture" for o in obs)
    sources = {o.source for o in obs}
    assert sources == {"sentinel-1", "sentinel-2"}


def test_tennent_fixture_has_expected_record_split() -> None:
    obs = load_observation_cache(TENNENT_DEMO_FIXTURE)
    s1 = [o for o in obs if o.source == "sentinel-1"]
    s2_low = [
        o for o in obs
        if o.source == "sentinel-2" and o.cloud_coverage is not None
        and o.cloud_coverage <= 25.0
    ]
    s2_cloudy = [
        o for o in obs
        if o.source == "sentinel-2" and o.cloud_coverage is not None
        and o.cloud_coverage > 25.0
    ]
    assert len(s1) == 1
    assert len(s2_low) == 1
    assert len(s2_cloudy) == 1


def test_normalization_identical_across_aois() -> None:
    """Tennent and Whitsun normalized records share the same field set."""
    whitsun = load_observation_cache(DEMO_FIXTURE)
    tennent = load_observation_cache(TENNENT_DEMO_FIXTURE)
    # Same dataclass, so __dataclass_fields__ will match by construction.
    w_fields = sorted(whitsun[0].__dataclass_fields__)
    t_fields = sorted(tennent[0].__dataclass_fields__)
    assert w_fields == t_fields
    # Confidence weight buckets are consistent across AOIs.
    w_s1 = next(o for o in whitsun if o.source == "sentinel-1")
    t_s1 = next(o for o in tennent if o.source == "sentinel-1")
    assert w_s1.confidence_weight == t_s1.confidence_weight


def test_tennent_observations_within_aoi_bbox() -> None:
    """Sanity: every Tennent record's bbox is inside the Tennent AOI bbox."""
    obs = load_observation_cache(TENNENT_DEMO_FIXTURE)
    aoi_blob = json.loads(TENNENT_AOI_FIXTURE.read_text(encoding="utf-8"))
    meta = aoi_blob["metadata"]
    aoi = (
        meta["bbox_lon_west_deg"], meta["bbox_lat_south_deg"],
        meta["bbox_lon_east_deg"], meta["bbox_lat_north_deg"],
    )
    for o in obs:
        assert o.bbox is not None
        assert o.bbox[0] >= aoi[0]
        assert o.bbox[1] >= aoi[1]
        assert o.bbox[2] <= aoi[2]
        assert o.bbox[3] <= aoi[3]


# ---------------------------------------------------------------------------
# CLI --aoi label dispatch (no live HTTP; offline mode only)
# ---------------------------------------------------------------------------


def _import_script():
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location(
        "script_29_ingest_sentinel_whitsun",
        REPO_ROOT / "scripts" / "29_ingest_sentinel_whitsun.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["script_29_ingest_sentinel_whitsun"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_cli_default_is_whitsun() -> None:
    import io
    mod = _import_script()
    buf = io.StringIO()
    rc = mod.main([], out=buf)
    assert rc == 0
    out = buf.getvalue()
    assert "whitsun_sentinel_observations.fixture.json" in out
    assert "fixture-s2-l2a-whitsun-20231212-low-cloud" in out


def test_cli_aoi_whitsun_label() -> None:
    import io
    mod = _import_script()
    buf = io.StringIO()
    rc = mod.main(["--aoi", "whitsun"], out=buf)
    assert rc == 0
    assert "fixture-s2-l2a-whitsun-20231212-low-cloud" in buf.getvalue()


def test_cli_aoi_tennent_label() -> None:
    import io
    mod = _import_script()
    buf = io.StringIO()
    rc = mod.main(["--aoi", "tennent"], out=buf)
    assert rc == 0
    out = buf.getvalue()
    assert "tennent_sentinel_observations.fixture.json" in out
    assert "fixture-s2-l2a-tennent-20230718-low-cloud" in out


def test_cli_aoi_path_form_still_works() -> None:
    """Backward compatibility: --aoi <path> still accepted."""
    import io
    mod = _import_script()
    buf = io.StringIO()
    rc = mod.main(
        [
            "--aoi", str(TENNENT_AOI_FIXTURE),
            "--fixture", str(TENNENT_DEMO_FIXTURE),
        ],
        out=buf,
    )
    assert rc == 0
    assert "fixture-s2-l2a-tennent-20230718-low-cloud" in buf.getvalue()


def test_cli_json_format_for_tennent() -> None:
    import io
    mod = _import_script()
    buf = io.StringIO()
    rc = mod.main(["--aoi", "tennent", "--format", "json"], out=buf)
    assert rc == 0
    parsed = json.loads(buf.getvalue())
    assert parsed["sentinel_1_count"] == 1
    assert parsed["sentinel_2_count"] == 2
    assert parsed["best_low_cloud_sentinel_2"] == (
        "fixture-s2-l2a-tennent-20230718-low-cloud"
    )
