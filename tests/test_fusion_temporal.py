"""Tests for :mod:`custody.fusion.temporal`.

Unit tests exercise :class:`DirectSpatialMatcher` on hand-constructed small
cases covering the association corner cases (empty, one-to-many, out-of-gate).
Integration tests run the full :func:`temporal_persistence` against the
committed Tennent 07-02 / 07-23 parquets via the Scene loader; skip when the
committed parquets are absent.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pyproj import Geod

from custody.fusion.observations import PositionObservation
from custody.fusion.scenes import Scene, load_scene_from_parquet
from custody.fusion.temporal import (
    DirectSpatialMatcher,
    Match,
    SignatureMatcher,
    TemporalComparisonResult,
    _signature_vector,
    temporal_persistence,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PQ_0702 = REPO_ROOT / "data/processed/vlm_detections/position.parquet"
PQ_0723 = REPO_ROOT / "data/processed/vlm_detections/tennent_20230723_position.parquet"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_GEOD = Geod(ellps="WGS84")


def _offset_latlon(lat: float, lon: float, dist_m: float, bearing_deg: float = 0.0) -> tuple[float, float]:
    """Return (lat, lon) at ``dist_m`` meters from (lat, lon) along ``bearing_deg``."""
    lon2, lat2, _ = _GEOD.fwd(lon, lat, bearing_deg, dist_m)
    return float(lat2), float(lon2)


def _mk_obs(
    obs_id: str, lat: float, lon: float, acq: float = 1_688_306_455.0,
    bbox_px: tuple[int, int, int, int] | None = None,
    classification_conf: float | None = None,
) -> PositionObservation:
    return PositionObservation(
        obs_id=obs_id, source_id="test", modality="SAR",
        acquisition_time=acq, ingestion_time=acq + 1.0,
        lat=lat, lon=lon, cov_pos=np.eye(2) * 100.0,
        raw_ref="test",
        bbox_px=bbox_px, classification_conf=classification_conf,
    )


def _mk_scene(scene_id: str, acq: float, observations: tuple[PositionObservation, ...]) -> Scene:
    return Scene(
        scene_id=scene_id, sensor="umbra-05", acquisition_time=acq,
        pixel_size_m=0.34,
        center_lat=8.856, center_lon=114.665,
        footprint_latlon=((8.83, 114.64), (8.83, 114.69), (8.88, 114.69), (8.88, 114.64)),
        raw_scene_path=None, quality_flag="green",
        observations=observations,
    )


# ---------------------------------------------------------------------------
# DirectSpatialMatcher — association logic
# ---------------------------------------------------------------------------


def test_matcher_simple_two_pairs_within_gate():
    """2 A obs, 2 B obs, each A within 10 m of its paired B — matcher returns 2 matches."""
    base_lat, base_lon = 8.856, 114.665
    a1 = _mk_obs("a1", *_offset_latlon(base_lat, base_lon, 0, 0))
    a2 = _mk_obs("a2", *_offset_latlon(base_lat, base_lon, 200, 90))  # 200 m east of a1
    b1 = _mk_obs("b1", *_offset_latlon(base_lat, base_lon, 10, 0))    # 10 m from a1
    b2 = _mk_obs("b2", *_offset_latlon(base_lat, base_lon, 210, 90))  # ~10 m from a2
    matcher = DirectSpatialMatcher(gate_m=50.0)
    matches = matcher.match((a1, a2), (b1, b2))
    assert len(matches) == 2
    by_a = {m.obs_a_id: m for m in matches}
    assert by_a["a1"].obs_b_id == "b1"
    assert by_a["a2"].obs_b_id == "b2"
    assert all(m.distance_m < 15.0 for m in matches)


def test_matcher_extra_in_b_yields_one_unmatched_b():
    """3 B obs, 2 A obs — Hungarian picks the two best pairs; third B is unmatched."""
    base_lat, base_lon = 8.856, 114.665
    a1 = _mk_obs("a1", base_lat, base_lon)
    a2 = _mk_obs("a2", *_offset_latlon(base_lat, base_lon, 200, 90))
    b1 = _mk_obs("b1", *_offset_latlon(base_lat, base_lon, 5, 0))
    b2 = _mk_obs("b2", *_offset_latlon(base_lat, base_lon, 205, 90))
    b3 = _mk_obs("b3", *_offset_latlon(base_lat, base_lon, 1000, 0))  # far from both
    matcher = DirectSpatialMatcher(gate_m=50.0)
    matches = matcher.match((a1, a2), (b1, b2, b3))
    assert len(matches) == 2
    assert {m.obs_b_id for m in matches} == {"b1", "b2"}


def test_matcher_extra_in_a_yields_one_unmatched_a():
    base_lat, base_lon = 8.856, 114.665
    a1 = _mk_obs("a1", base_lat, base_lon)
    a2 = _mk_obs("a2", *_offset_latlon(base_lat, base_lon, 200, 90))
    a3 = _mk_obs("a3", *_offset_latlon(base_lat, base_lon, 1000, 0))  # far from any B
    b1 = _mk_obs("b1", *_offset_latlon(base_lat, base_lon, 5, 0))
    b2 = _mk_obs("b2", *_offset_latlon(base_lat, base_lon, 205, 90))
    matcher = DirectSpatialMatcher(gate_m=50.0)
    matches = matcher.match((a1, a2, a3), (b1, b2))
    assert len(matches) == 2
    assert {m.obs_a_id for m in matches} == {"a1", "a2"}


def test_matcher_out_of_gate_yields_no_matches():
    base_lat, base_lon = 8.856, 114.665
    a1 = _mk_obs("a1", base_lat, base_lon)
    b1 = _mk_obs("b1", *_offset_latlon(base_lat, base_lon, 100, 0))  # 100 m, outside 50 m
    matcher = DirectSpatialMatcher(gate_m=50.0)
    assert matcher.match((a1,), (b1,)) == []


def test_matcher_one_to_many_closer_wins():
    """Two A obs both within 50 m of one B — Hungarian picks the closer one."""
    base_lat, base_lon = 8.856, 114.665
    b1 = _mk_obs("b1", base_lat, base_lon)
    a_close = _mk_obs("a_close", *_offset_latlon(base_lat, base_lon, 10, 0))   # 10 m
    a_far = _mk_obs("a_far", *_offset_latlon(base_lat, base_lon, 30, 0))       # 30 m
    matcher = DirectSpatialMatcher(gate_m=50.0)
    matches = matcher.match((a_close, a_far), (b1,))
    # Only the nearer one claims b1; the other has no partner and isn't in matches.
    assert len(matches) == 1
    assert matches[0].obs_a_id == "a_close"
    assert matches[0].obs_b_id == "b1"


def test_matcher_empty_a_returns_empty():
    b1 = _mk_obs("b1", 8.856, 114.665)
    assert DirectSpatialMatcher(gate_m=50.0).match((), (b1,)) == []


def test_matcher_empty_b_returns_empty():
    a1 = _mk_obs("a1", 8.856, 114.665)
    assert DirectSpatialMatcher(gate_m=50.0).match((a1,), ()) == []


def test_matcher_both_empty_returns_empty():
    assert DirectSpatialMatcher(gate_m=50.0).match((), ()) == []


def test_matcher_distance_is_in_meters_not_degrees():
    """100 m offset produces a distance value of ~100, not the 0.001° equivalent."""
    base_lat, base_lon = 8.856, 114.665
    a = _mk_obs("a", base_lat, base_lon)
    b = _mk_obs("b", *_offset_latlon(base_lat, base_lon, 40, 0))
    matcher = DirectSpatialMatcher(gate_m=50.0)
    matches = matcher.match((a,), (b,))
    assert len(matches) == 1
    assert matches[0].distance_m == pytest.approx(40.0, abs=1.0)


def test_matcher_name_property():
    assert DirectSpatialMatcher(gate_m=50.0).name == "DirectSpatialMatcher(gate_m=50.0)"


# ---------------------------------------------------------------------------
# temporal_persistence — top-level orchestrator
# ---------------------------------------------------------------------------


def test_temporal_persistence_rejects_same_scene_id():
    s = _mk_scene("tennent_20230702_umbra-05", 1_688_306_455.0, ())
    # Making a second Scene with same scene_id
    s2 = _mk_scene("tennent_20230702_umbra-05", 1_688_306_455.0, ())
    with pytest.raises(ValueError, match="share scene_id"):
        temporal_persistence(s, s2, matcher=DirectSpatialMatcher(gate_m=50.0))


def test_temporal_persistence_rejects_out_of_order_scenes():
    """scene_a.acquisition_time must be < scene_b.acquisition_time."""
    a = _mk_scene("tennent_20230723_umbra-05", 1_690_207_370.0, ())  # 07-23
    b = _mk_scene("tennent_20230702_umbra-05", 1_688_306_455.0, ())  # 07-02 — earlier
    with pytest.raises(ValueError, match="chronological order"):
        temporal_persistence(a, b, matcher=DirectSpatialMatcher(gate_m=50.0))


def test_temporal_persistence_composes_matches_and_unmatched():
    """Manufactured scene pair: 2 persistent, 1 emerged, 1 disappeared."""
    base_lat, base_lon = 8.856, 114.665
    a1 = _mk_obs("a1", base_lat, base_lon, acq=1000.0)
    a2 = _mk_obs("a2", *_offset_latlon(base_lat, base_lon, 200, 90), acq=1000.0)
    a3 = _mk_obs("a3", *_offset_latlon(base_lat, base_lon, 5000, 0), acq=1000.0)
    b1 = _mk_obs("b1", *_offset_latlon(base_lat, base_lon, 5, 0), acq=2000.0)
    b2 = _mk_obs("b2", *_offset_latlon(base_lat, base_lon, 205, 90), acq=2000.0)
    b3 = _mk_obs("b3", *_offset_latlon(base_lat, base_lon, 7000, 0), acq=2000.0)
    sa = _mk_scene("tennent_20230702_umbra-05", 1000.0, (a1, a2, a3))
    sb = _mk_scene("tennent_20230723_umbra-05", 2000.0, (b1, b2, b3))
    r = temporal_persistence(sa, sb, matcher=DirectSpatialMatcher(gate_m=50.0))
    assert len(r.matches) == 2
    assert set(r.disappeared) == {"a3"}
    assert set(r.emerged) == {"b3"}
    assert r.scene_a_id == "tennent_20230702_umbra-05"
    assert r.scene_b_id == "tennent_20230723_umbra-05"
    assert "DirectSpatialMatcher" in r.matcher_name


def test_temporal_persistence_handles_empty_scenes():
    sa = _mk_scene("tennent_20230702_umbra-05", 1000.0, ())
    sb = _mk_scene("tennent_20230723_umbra-05", 2000.0, ())
    r = temporal_persistence(sa, sb, matcher=DirectSpatialMatcher(gate_m=50.0))
    assert r.matches == ()
    assert r.emerged == ()
    assert r.disappeared == ()


def test_temporal_persistence_all_unmatched_when_scenes_far_apart():
    """Two scenes with observations nowhere near each other — everything is emerged/disappeared."""
    a = _mk_obs("a", 8.856, 114.665, acq=1000.0)
    b = _mk_obs("b", 9.500, 115.000, acq=2000.0)  # far
    sa = _mk_scene("tennent_20230702_umbra-05", 1000.0, (a,))
    sb = _mk_scene("tennent_20230723_umbra-05", 2000.0, (b,))
    r = temporal_persistence(sa, sb, matcher=DirectSpatialMatcher(gate_m=50.0))
    assert r.matches == ()
    assert r.disappeared == ("a",)
    assert r.emerged == ("b",)


# ---------------------------------------------------------------------------
# Integration test — committed Tennent 07-02 / 07-23 parquets
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (PQ_0702.exists() and PQ_0723.exists()),
    reason="committed Tennent parquets not on disk",
)
def test_integration_tennent_0702_0723_reproduces_week3_stats():
    """End-to-end: load both committed Tennent scenes, run persistence, assert Week-3 stats."""
    sa = load_scene_from_parquet(PQ_0702, case_study="tennent")
    sb = load_scene_from_parquet(PQ_0723, case_study="tennent")
    r = temporal_persistence(sa, sb, matcher=DirectSpatialMatcher(gate_m=50.0))
    # Week-3 recorded output: 22 matches, 22 emerged, 20 disappeared.
    assert len(r.matches) == 22
    assert len(r.emerged) == 22
    assert len(r.disappeared) == 20
    # Distance distribution (recorded in tennent_temporal_comparison_summary.md):
    # min 6.61, mean 21.35, max 44.30 — tolerate sub-meter drift from AEQD-vs-Geod.
    ds = np.array(r.match_distances_m)
    assert ds.min() == pytest.approx(6.61, abs=0.1)
    assert float(ds.mean()) == pytest.approx(21.35, abs=0.5)
    assert ds.max() == pytest.approx(44.30, abs=0.5)
    # All matches within the gate
    assert (ds <= 50.0).all()


@pytest.mark.skipif(
    not (PQ_0702.exists() and PQ_0723.exists()),
    reason="committed Tennent parquets not on disk",
)
def test_integration_result_scene_ids_match_loaded_scenes():
    sa = load_scene_from_parquet(PQ_0702, case_study="tennent")
    sb = load_scene_from_parquet(PQ_0723, case_study="tennent")
    r = temporal_persistence(sa, sb, matcher=DirectSpatialMatcher(gate_m=50.0))
    assert r.scene_a_id == "tennent_20230702_umbra-05"
    assert r.scene_b_id == "tennent_20230723_umbra-05"


@pytest.mark.skipif(
    not (PQ_0702.exists() and PQ_0723.exists()),
    reason="committed Tennent parquets not on disk",
)
def test_integration_matches_reference_obs_ids_from_both_scenes():
    """Every persistent Match's obs_a_id comes from scene A, obs_b_id from scene B."""
    sa = load_scene_from_parquet(PQ_0702, case_study="tennent")
    sb = load_scene_from_parquet(PQ_0723, case_study="tennent")
    r = temporal_persistence(sa, sb, matcher=DirectSpatialMatcher(gate_m=50.0))
    a_ids = {o.obs_id for o in sa.observations}
    b_ids = {o.obs_id for o in sb.observations}
    for m in r.matches:
        assert m.obs_a_id in a_ids
        assert m.obs_b_id in b_ids
    # Disappeared are a subset of scene A, emerged a subset of scene B
    assert set(r.disappeared).issubset(a_ids)
    assert set(r.emerged).issubset(b_ids)


# ---------------------------------------------------------------------------
# Frozen-dataclass safety
# ---------------------------------------------------------------------------


def test_match_is_frozen():
    m = Match(obs_a_id="a", obs_b_id="b", distance_m=10.0)
    with pytest.raises(AttributeError):
        m.distance_m = 99.0  # type: ignore[misc]


def test_temporal_comparison_result_is_frozen():
    r = TemporalComparisonResult(
        scene_a_id="a", scene_b_id="b", matcher_name="test",
        matches=(), emerged=(), disappeared=(), match_distances_m=(),
    )
    with pytest.raises(AttributeError):
        r.matcher_name = "other"  # type: ignore[misc]


def test_direct_spatial_matcher_is_frozen():
    m = DirectSpatialMatcher(gate_m=50.0)
    with pytest.raises(AttributeError):
        m.gate_m = 100.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SignatureMatcher — V1 geometry-aware matcher (ADR-0019)
# ---------------------------------------------------------------------------


def test_signature_vector_wide_bbox_components():
    """100 × 50 px bbox → width 0.5, height 0.25, log2(2)/3 = 0.333, conf."""
    o = _mk_obs("x", 8.85, 114.66, bbox_px=(0, 0, 100, 50), classification_conf=0.85)
    v = _signature_vector(o)
    assert v is not None
    assert v[0] == pytest.approx(0.5)                # width / 200
    assert v[1] == pytest.approx(0.25)               # height / 200
    assert v[2] == pytest.approx(1.0 / 3.0)          # log2(100/50) / 3 = 1/3
    assert v[3] == pytest.approx(0.85)


def test_signature_vector_tall_bbox_has_negative_log_aspect():
    """50 × 100 px bbox → log2(0.5)/3 = -1/3 (symmetric around 0 for inverse aspect)."""
    o = _mk_obs("x", 8.85, 114.66, bbox_px=(0, 0, 50, 100), classification_conf=0.5)
    v = _signature_vector(o)
    assert v is not None
    assert v[2] == pytest.approx(-1.0 / 3.0)


def test_signature_vector_square_bbox_has_zero_log_aspect():
    o = _mk_obs("x", 8.85, 114.66, bbox_px=(0, 0, 60, 60), classification_conf=0.6)
    v = _signature_vector(o)
    assert v is not None
    assert v[2] == pytest.approx(0.0)


def test_signature_vector_returns_none_when_bbox_px_missing():
    o = _mk_obs("x", 8.85, 114.66, bbox_px=None, classification_conf=0.5)
    assert _signature_vector(o) is None


def test_signature_vector_returns_none_when_classification_conf_missing():
    o = _mk_obs("x", 8.85, 114.66, bbox_px=(0, 0, 50, 50), classification_conf=None)
    assert _signature_vector(o) is None


def test_signature_matcher_name_property():
    assert SignatureMatcher(gate_m=50.0, sig_gate=0.8).name == (
        "SignatureMatcher(gate_m=50.0, sig_gate=0.8)"
    )


def test_signature_matcher_is_frozen():
    m = SignatureMatcher()
    with pytest.raises(AttributeError):
        m.gate_m = 100.0  # type: ignore[misc]


def test_signature_matcher_permits_pair_with_similar_signatures():
    """Two obs within spatial gate with near-identical signatures → match."""
    base_lat, base_lon = 8.856, 114.665
    a = _mk_obs("a", base_lat, base_lon,
                bbox_px=(0, 0, 100, 50), classification_conf=0.85)
    b_lat, b_lon = _offset_latlon(base_lat, base_lon, 10, 0)
    b = _mk_obs("b", b_lat, b_lon,
                bbox_px=(0, 0, 102, 48), classification_conf=0.85)
    matcher = SignatureMatcher(gate_m=50.0, sig_gate=0.2)
    matches = matcher.match((a,), (b,))
    assert len(matches) == 1
    assert matches[0].obs_a_id == "a"
    assert matches[0].obs_b_id == "b"


def test_signature_matcher_rejects_pair_with_dissimilar_signatures():
    """Two obs within spatial gate but very different bbox sizes → not matched."""
    base_lat, base_lon = 8.856, 114.665
    a = _mk_obs("a", base_lat, base_lon,
                bbox_px=(0, 0, 20, 20), classification_conf=0.85)
    b_lat, b_lon = _offset_latlon(base_lat, base_lon, 10, 0)
    b = _mk_obs("b", b_lat, b_lon,
                bbox_px=(0, 0, 400, 400), classification_conf=0.35)  # 20x bigger + different conf
    matcher = SignatureMatcher(gate_m=50.0, sig_gate=0.3)  # tight sig gate
    matches = matcher.match((a,), (b,))
    assert matches == []


def test_signature_matcher_missing_bbox_bypasses_signature_gate():
    """Pair where one obs lacks bbox_px falls back to spatial-only eligibility."""
    base_lat, base_lon = 8.856, 114.665
    # Obs a has no bbox; obs b has one.  Signature distance is undefined for the
    # pair.  Matcher should treat as signature-eligible and fall through to
    # spatial gate.
    a = _mk_obs("a", base_lat, base_lon, bbox_px=None, classification_conf=0.85)
    b_lat, b_lon = _offset_latlon(base_lat, base_lon, 20, 0)
    b = _mk_obs("b", b_lat, b_lon,
                bbox_px=(0, 0, 400, 400), classification_conf=0.35)
    matcher = SignatureMatcher(gate_m=50.0, sig_gate=0.01)  # pathologically tight
    matches = matcher.match((a,), (b,))
    # Sig gate would reject if applied; missing-signature fallback allows
    # the pair via spatial gate alone.
    assert len(matches) == 1


def test_signature_matcher_missing_classification_conf_also_falls_back():
    """Same as above but with classification_conf=None instead of bbox_px=None."""
    base_lat, base_lon = 8.856, 114.665
    a = _mk_obs("a", base_lat, base_lon,
                bbox_px=(0, 0, 100, 50), classification_conf=None)
    b_lat, b_lon = _offset_latlon(base_lat, base_lon, 15, 0)
    b = _mk_obs("b", b_lat, b_lon,
                bbox_px=(0, 0, 300, 100), classification_conf=0.3)
    matcher = SignatureMatcher(gate_m=50.0, sig_gate=0.01)
    matches = matcher.match((a,), (b,))
    assert len(matches) == 1


def test_signature_matcher_one_to_many_signature_similarity_wins():
    """Two A obs both within spatial gate of one B; the similar-signature one wins.

    Spatial-only matcher would pick whichever is spatially closer; signature
    matcher filters out the dissimilar pair up-front so Hungarian has only one
    option — even if the similar-signature one is slightly farther away.
    """
    base_lat, base_lon = 8.856, 114.665
    # B is the reference
    b = _mk_obs("b", base_lat, base_lon,
                bbox_px=(0, 0, 100, 50), classification_conf=0.85)
    # a_close: 5 m away, VERY different bbox
    a_close_lat, a_close_lon = _offset_latlon(base_lat, base_lon, 5, 0)
    a_close = _mk_obs("a_close", a_close_lat, a_close_lon,
                      bbox_px=(0, 0, 500, 10), classification_conf=0.35)
    # a_similar: 20 m away, similar bbox
    a_sim_lat, a_sim_lon = _offset_latlon(base_lat, base_lon, 20, 90)
    a_similar = _mk_obs("a_similar", a_sim_lat, a_sim_lon,
                        bbox_px=(0, 0, 98, 52), classification_conf=0.85)
    matcher = SignatureMatcher(gate_m=50.0, sig_gate=0.3)  # tight sig gate
    matches = matcher.match((a_close, a_similar), (b,))
    assert len(matches) == 1
    assert matches[0].obs_a_id == "a_similar"


def test_signature_matcher_empty_inputs_return_empty():
    m = SignatureMatcher()
    assert m.match((), ()) == []
    a = _mk_obs("a", 8.85, 114.66, bbox_px=(0, 0, 50, 50), classification_conf=0.5)
    assert m.match((a,), ()) == []
    assert m.match((), (a,)) == []


def test_signature_matcher_emits_debug_log_on_fallback(caplog):
    """DEBUG log fires when signature fallback is exercised for any pair."""
    import logging
    base_lat, base_lon = 8.856, 114.665
    a = _mk_obs("a", base_lat, base_lon, bbox_px=None, classification_conf=None)
    b_lat, b_lon = _offset_latlon(base_lat, base_lon, 10, 0)
    b = _mk_obs("b", b_lat, b_lon,
                bbox_px=(0, 0, 50, 50), classification_conf=0.8)
    matcher = SignatureMatcher(gate_m=50.0, sig_gate=0.8)
    with caplog.at_level(logging.DEBUG, logger="custody.fusion.temporal"):
        matcher.match((a,), (b,))
    assert any(
        "SignatureMatcher" in r.message and "bypassed the signature gate" in r.message
        for r in caplog.records
    )


# ---------------------------------------------------------------------------
# SignatureMatcher same-geometry sanity integration (ADR-0019)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (PQ_0702.exists() and PQ_0723.exists()),
    reason="committed Tennent parquets not on disk",
)
def test_signature_matcher_on_same_geometry_pair_does_not_regress():
    """V1 sanity: SignatureMatcher on same-geometry pair should produce a
    persistent-count comparable to DirectSpatialMatcher's baseline of 22.

    A pure-regression tripwire — not a correctness assertion on the
    final V1 parameters (sig_gate may be tuned empirically before flip to
    accepted).  The test passes as long as SignatureMatcher recovers
    substantially all of DirectSpatialMatcher's matches on the easy case.
    """
    sa = load_scene_from_parquet(PQ_0702, case_study="tennent")
    sb = load_scene_from_parquet(PQ_0723, case_study="tennent")
    r = temporal_persistence(
        sa, sb, matcher=SignatureMatcher(gate_m=50.0, sig_gate=0.8),
    )
    # Baseline is 22 persistent.  Require >= 15 — generous tolerance lets us
    # catch catastrophic regression (e.g., sig gate so tight nothing passes)
    # without pinning the precise tuning.
    assert len(r.matches) >= 15, (
        f"SignatureMatcher recovered only {len(r.matches)} matches on the "
        "same-geometry Tennent 07-02↔07-23 pair; DirectSpatialMatcher baseline "
        "is 22.  V1 signature gating appears to be over-filtering same-geom matches."
    )
