"""Tests for :mod:`custody.fusion.scenes`.

Unit tests exercise Scene validation rules one-per-test so each failure
message is explicitly pinned.  Helper function tests cover filename parsing,
scene_id composition, and sidecar-path resolution.  Integration tests load
the committed Tennent/Whitsun parquets via the sidecar fallback chain; they
skip automatically if the raw parquet files aren't on disk.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from custody.fusion.observations import PositionObservation
from custody.fusion.scenes import (
    Scene,
    SceneLoaderError,
    _compute_scene_id,
    _parse_sensor_from_filename,
    _resolve_sidecar_path,
    load_scene_from_parquet,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PQ_TENNENT_0702 = REPO_ROOT / "data/processed/vlm_detections/position.parquet"
PQ_TENNENT_0723 = REPO_ROOT / "data/processed/vlm_detections/tennent_20230723_position.parquet"
PQ_WHITSUN_1206 = REPO_ROOT / "data/processed/vlm_detections/whitsun_20231206_position.parquet"


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _mk_obs(obs_id: str = "obs-1", acq: float = 1_688_306_455.344) -> PositionObservation:
    return PositionObservation(
        obs_id=obs_id, source_id="test", modality="SAR",
        acquisition_time=acq, ingestion_time=acq + 1.0,
        lat=8.855, lon=114.665, cov_pos=np.eye(2) * 400.0,
        raw_ref="test",
    )


def _ok_kwargs(**overrides) -> dict:
    """Build a valid Scene kwarg set; overrides are merged for failure tests."""
    kw = dict(
        scene_id="tennent_20230702_umbra-05",
        sensor="umbra-05",
        acquisition_time=1_688_306_455.344,
        pixel_size_m=0.342,
        center_lat=8.856, center_lon=114.665,
        footprint_latlon=(
            (8.83, 114.64), (8.83, 114.69),
            (8.88, 114.69), (8.88, 114.64),
        ),
        raw_scene_path=None,
        quality_flag="green",
        observations=(),
    )
    kw.update(overrides)
    return kw


# ---------------------------------------------------------------------------
# Scene __post_init__ validations (one per rule)
# ---------------------------------------------------------------------------


def test_scene_rejects_empty_scene_id():
    with pytest.raises(ValueError, match="scene_id must be a non-empty string"):
        Scene(**_ok_kwargs(scene_id=""))


def test_scene_rejects_malformed_scene_id():
    with pytest.raises(ValueError, match="must match.*pattern"):
        Scene(**_ok_kwargs(scene_id="tennent-20230702-umbra05"))  # hyphens, wrong separator


def test_scene_rejects_scene_id_without_date():
    with pytest.raises(ValueError, match="must match.*pattern"):
        Scene(**_ok_kwargs(scene_id="tennent_umbra-05"))


def test_scene_rejects_negative_acquisition_time():
    with pytest.raises(ValueError, match="acquisition_time.*outside valid range"):
        Scene(**_ok_kwargs(acquisition_time=-1.0))


def test_scene_rejects_future_acquisition_time():
    """Year 2200 epoch is past the sanity upper bound."""
    with pytest.raises(ValueError, match="acquisition_time.*outside valid range"):
        Scene(**_ok_kwargs(acquisition_time=7_258_118_400.0))


def test_scene_rejects_zero_pixel_size():
    with pytest.raises(ValueError, match="pixel_size_m.*outside"):
        Scene(**_ok_kwargs(pixel_size_m=0.0))


def test_scene_rejects_oversize_pixel_size():
    with pytest.raises(ValueError, match="pixel_size_m.*outside"):
        Scene(**_ok_kwargs(pixel_size_m=15.0))


def test_scene_rejects_out_of_range_lat():
    with pytest.raises(ValueError, match="center_lat.*outside"):
        Scene(**_ok_kwargs(center_lat=95.0))


def test_scene_rejects_out_of_range_lon():
    with pytest.raises(ValueError, match="center_lon.*outside"):
        Scene(**_ok_kwargs(center_lon=-200.0))


def test_scene_rejects_footprint_with_fewer_than_3_corners():
    with pytest.raises(ValueError, match="footprint_latlon must have >= 3 corners"):
        Scene(**_ok_kwargs(footprint_latlon=((8.83, 114.64), (8.88, 114.69))))


def test_scene_rejects_invalid_quality_flag():
    with pytest.raises(ValueError, match="quality_flag.*must be"):
        Scene(**_ok_kwargs(quality_flag="purple"))  # type: ignore[arg-type]


def test_scene_rejects_list_for_observations():
    """Observations must be a tuple — lists admit mutation after construction."""
    with pytest.raises(ValueError, match="observations must be a tuple"):
        Scene(**_ok_kwargs(observations=[_mk_obs()]))  # type: ignore[arg-type]


def test_scene_rejects_non_observation_in_observations():
    with pytest.raises(ValueError, match="expected PositionObservation"):
        Scene(**_ok_kwargs(observations=("not an obs",)))  # type: ignore[arg-type]


def test_scene_rejects_observation_with_mismatched_acquisition_time():
    """An obs acquired > 1s from the scene time is rejected."""
    scene_acq = 1_688_306_455.344
    bad_obs = _mk_obs(acq=scene_acq + 5.0)  # 5 s later — beyond ±1s tolerance
    with pytest.raises(ValueError, match="acquisition_time.*differs from scene"):
        Scene(**_ok_kwargs(acquisition_time=scene_acq, observations=(bad_obs,)))


def test_scene_accepts_observations_within_one_second_tolerance():
    scene_acq = 1_688_306_455.344
    close_obs = _mk_obs(acq=scene_acq + 0.5)
    s = Scene(**_ok_kwargs(acquisition_time=scene_acq, observations=(close_obs,)))
    assert len(s.observations) == 1


def test_scene_hash_and_eq_by_scene_id():
    """Two Scenes with identical scene_id are equal regardless of other fields."""
    a = Scene(**_ok_kwargs())
    b = Scene(**_ok_kwargs(sensor="umbra-06"))  # same scene_id, different sensor
    assert a == b
    assert hash(a) == hash(b)
    c = Scene(**_ok_kwargs(scene_id="tennent_20230723_umbra-05"))
    assert a != c


def test_scene_is_frozen():
    s = Scene(**_ok_kwargs())
    with pytest.raises(AttributeError):
        s.sensor = "umbra-99"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_parse_sensor_from_filename_matches_umbra():
    assert _parse_sensor_from_filename(
        "2023-07-02-14-00-55_UMBRA-05_GEC.tif"
    ) == "umbra-05"


def test_parse_sensor_from_filename_matches_sentinel():
    assert _parse_sensor_from_filename(
        "S1A_IW_SLC__1SDV_20240101_SENTINEL-1_processed.tif"
    ) == "sentinel-1"


def test_parse_sensor_from_filename_returns_none_on_no_match():
    assert _parse_sensor_from_filename("garbage_filename.tif") is None


def test_compute_scene_id_formats_correctly():
    # 1688306455.344 = 2023-07-02 14:00:55.344 UTC
    assert _compute_scene_id("tennent", 1_688_306_455.344, "UMBRA-05") == (
        "tennent_20230702_umbra-05"
    )


def test_compute_scene_id_lowercases_case_study_and_sensor():
    assert _compute_scene_id("TENNENT", 1_688_306_455.344, "Umbra-06") == (
        "tennent_20230702_umbra-06"
    )


# ---------------------------------------------------------------------------
# Sidecar path resolution
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not PQ_TENNENT_0702.exists(),
    reason=f"committed parquet missing: {PQ_TENNENT_0702}",
)
def test_resolve_sidecar_path_finds_legacy_tennent_0702():
    """The original Tennent 07-02 parquet has no date prefix; resolver handles the special case."""
    sidecar = _resolve_sidecar_path(PQ_TENNENT_0702)
    assert sidecar is not None
    assert sidecar.name == "tennent_20230702_vlm_summary.json"
    assert sidecar.exists()


@pytest.mark.skipif(
    not PQ_TENNENT_0723.exists(),
    reason=f"committed parquet missing: {PQ_TENNENT_0723}",
)
def test_resolve_sidecar_path_finds_dated_tennent():
    sidecar = _resolve_sidecar_path(PQ_TENNENT_0723)
    assert sidecar is not None
    assert sidecar.name == "tennent_20230723_vlm_summary.json"


def test_resolve_sidecar_path_returns_none_when_missing(tmp_path: Path):
    """Sidecar lookup in an ancestor that has no day0/scratch returns None."""
    fake_parquet = tmp_path / "fake_position.parquet"
    fake_parquet.touch()
    assert _resolve_sidecar_path(fake_parquet) is None


# ---------------------------------------------------------------------------
# Loader — integration against committed parquets
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not PQ_TENNENT_0702.exists(),
    reason="Tennent 07-02 parquet not on disk",
)
def test_load_tennent_0702_via_sidecar_fallback():
    s = load_scene_from_parquet(PQ_TENNENT_0702, case_study="tennent")
    assert s.scene_id == "tennent_20230702_umbra-05"
    assert s.sensor == "umbra-05"
    assert s.pixel_size_m == pytest.approx(0.342, abs=0.01)
    assert len(s.observations) == 42
    assert s.center_lat == pytest.approx(8.8557, abs=0.001)
    assert s.center_lon == pytest.approx(114.6651, abs=0.001)
    assert s.quality_flag == "green"
    assert len(s.footprint_latlon) >= 3
    # Sidecar metadata is merged into notes for provenance
    assert "_sidecar" in s.notes


@pytest.mark.skipif(
    not PQ_TENNENT_0723.exists(),
    reason="Tennent 07-23 parquet not on disk",
)
def test_load_tennent_0723_has_distinct_scene_id_and_correct_obs_count():
    s = load_scene_from_parquet(PQ_TENNENT_0723, case_study="tennent")
    assert s.scene_id == "tennent_20230723_umbra-05"
    assert len(s.observations) == 44


@pytest.mark.skipif(
    not PQ_WHITSUN_1206.exists(),
    reason="Whitsun 12-06 parquet not on disk",
)
def test_load_whitsun_1206_center_from_sidecar_not_obs_mean():
    """Whitsun sidecar has scene_center_latlon; loader should use it rather than
    falling back to observation-mean (which would be biased toward the upper-right flotilla).
    """
    s = load_scene_from_parquet(PQ_WHITSUN_1206, case_study="whitsun")
    assert s.scene_id == "whitsun_20231206_umbra-04"
    assert len(s.observations) == 115
    # Sidecar scene_center is ~(9.9692, 114.6319); observation mean would skew.
    assert s.center_lat == pytest.approx(9.9692, abs=0.01)
    assert s.center_lon == pytest.approx(114.6319, abs=0.01)
    # pixel_size_m derived from sidecar's approx_extent_km + full_scene_shape_px
    # (Whitsun summaries don't write pixel_size_m directly) — ≈ 0.316 m/px via
    # 5.57 km / 17602 px.  Allow 0.05 tolerance for rotated-raster approximation.
    assert s.pixel_size_m == pytest.approx(0.32, abs=0.05)


@pytest.mark.skipif(
    not PQ_TENNENT_0702.exists(),
    reason="Tennent 07-02 parquet not on disk",
)
def test_load_respects_kwargs_over_sidecar():
    """Explicit kwargs override sidecar values."""
    s = load_scene_from_parquet(
        PQ_TENNENT_0702, case_study="tennent",
        sensor="umbra-99",  # override
        pixel_size_m=0.5,   # override
    )
    assert s.sensor == "umbra-99"
    assert s.pixel_size_m == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Loader — error paths
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not PQ_TENNENT_0702.exists(),
    reason="Tennent 07-02 parquet not on disk",
)
def test_load_raises_when_no_sidecar_no_kwargs(tmp_path: Path):
    """Copy the parquet to a tmp dir (no sidecar reachable), no kwargs provided,
    and confirm SceneLoaderError cites the missing fields.
    """
    import shutil
    fake = tmp_path / "some_position.parquet"
    shutil.copy(PQ_TENNENT_0702, fake)
    with pytest.raises(SceneLoaderError, match="Could not resolve Scene.sensor"):
        load_scene_from_parquet(fake, case_study="tennent")


@pytest.mark.skipif(
    not PQ_TENNENT_0702.exists(),
    reason="Tennent 07-02 parquet not on disk",
)
def test_load_raises_with_partial_kwargs_naming_missing_field(tmp_path: Path):
    """Providing sensor but not pixel_size still raises for pixel_size."""
    import shutil
    fake = tmp_path / "some_position.parquet"
    shutil.copy(PQ_TENNENT_0702, fake)
    with pytest.raises(SceneLoaderError, match="pixel_size_m"):
        load_scene_from_parquet(fake, case_study="tennent", sensor="umbra-05")
