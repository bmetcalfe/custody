"""
Focused tests for custody/anomalies.py.
Covers each of the three detection functions plus anomaly_score/breakdown.

Each detector now returns a DetectorResult.  Tests assert on:
  - .score  — the numeric contribution (behaviour-critical)
  - .evidence — the human-readable explanation string (content-critical)
  - .metadata — structured key/value pairs where populated
"""
from datetime import datetime, timedelta, UTC

import pytest

from custody.config import PROXIMITY_CRITICAL_KM, PROXIMITY_WARNING_KM, ZONES
from custody.models import DetectorResult, Vessel, HistoryEntry, Zone
from custody.anomalies import (
    in_sensitive_zone,
    loitering,
    route_deviation,
    anomaly_breakdown,
    anomaly_score,
)
from custody.behavior.detectors import vessel_proximity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EPOCH = datetime(2026, 1, 1, tzinfo=UTC)


def ts(hour: float) -> datetime:
    return _EPOCH + timedelta(hours=hour)


def entry(lat=0.0, lon=0.0, hour=0, speed_kmh=28.0, heading_deg=45.0) -> HistoryEntry:
    return HistoryEntry(lat, lon, ts(hour), speed_kmh, heading_deg)


def make_vessel(lat=0.0, lon=0.0, speed_kmh=30.0, heading_deg=45.0, history=None):
    return Vessel(
        id="TEST",
        lat=lat,
        lon=lon,
        speed_kmh=speed_kmh,
        heading_deg=heading_deg,
        last_seen=_EPOCH,
        history=history or [],
    )


_Z0 = ZONES[0]   # ZONE_ALPHA — single zone in current config
ZONE_LAT = (_Z0.min_lat + _Z0.max_lat) / 2
ZONE_LON = (_Z0.min_lon + _Z0.max_lon) / 2


# ---------------------------------------------------------------------------
# in_sensitive_zone  (no history access — unchanged)
# ---------------------------------------------------------------------------

class TestInSensitiveZone:
    def test_inside_returns_max_score(self):
        v = make_vessel(lat=ZONE_LAT, lon=ZONE_LON)
        assert in_sensitive_zone(v).score == 1.5

    def test_far_outside_returns_zero(self):
        v = make_vessel(lat=0.0, lon=0.0)
        assert in_sensitive_zone(v).score == 0.0

    def test_just_outside_boundary_returns_partial(self):
        v = make_vessel(lat=_Z0.min_lat - 0.01, lon=ZONE_LON)
        assert 0.0 < in_sensitive_zone(v).score < 1.5

    def test_at_halo_edge_returns_zero(self):
        v = make_vessel(lat=_Z0.min_lat - _Z0.halo, lon=ZONE_LON)
        assert in_sensitive_zone(v).score == pytest.approx(0.0, abs=1e-6)

    def test_proximity_score_increases_as_vessel_approaches(self):
        far  = make_vessel(lat=_Z0.min_lat - _Z0.halo * 0.9, lon=ZONE_LON)
        near = make_vessel(lat=_Z0.min_lat - _Z0.halo * 0.1, lon=ZONE_LON)
        assert in_sensitive_zone(near).score > in_sensitive_zone(far).score

    def test_on_zone_boundary_returns_max_score(self):
        v = make_vessel(lat=_Z0.min_lat, lon=ZONE_LON)
        assert in_sensitive_zone(v).score == 1.5

    # --- DetectorResult shape ---

    def test_returns_detector_result(self):
        v = make_vessel(lat=ZONE_LAT, lon=ZONE_LON)
        assert isinstance(in_sensitive_zone(v), DetectorResult)

    def test_name_is_sensitive_zone(self):
        for v in [make_vessel(lat=ZONE_LAT, lon=ZONE_LON), make_vessel(lat=0.0, lon=0.0)]:
            assert in_sensitive_zone(v).name == "sensitive_zone"

    def test_inside_evidence(self):
        v = make_vessel(lat=ZONE_LAT, lon=ZONE_LON)
        assert in_sensitive_zone(v).evidence == f"inside {_Z0.name}"

    def test_outside_evidence(self):
        v = make_vessel(lat=0.0, lon=0.0)
        assert "outside" in in_sensitive_zone(v).evidence

    def test_halo_evidence_contains_km_and_boundary(self):
        v = make_vessel(lat=_Z0.min_lat - 0.01, lon=ZONE_LON)
        evidence = in_sensitive_zone(v).evidence
        assert "km" in evidence
        assert "boundary" in evidence

    def test_halo_metadata_has_distance_key(self):
        v = make_vessel(lat=_Z0.min_lat - 0.01, lon=ZONE_LON)
        meta = in_sensitive_zone(v).metadata
        assert "distance_to_zone_km" in meta
        assert meta["distance_to_zone_km"] > 0.0

    def test_inside_metadata_has_zone_name(self):
        v = make_vessel(lat=ZONE_LAT, lon=ZONE_LON)
        assert in_sensitive_zone(v).metadata.get("zone_name") == _Z0.name

    def test_halo_metadata_has_zone_name(self):
        v = make_vessel(lat=_Z0.min_lat - 0.01, lon=ZONE_LON)
        assert in_sensitive_zone(v).metadata.get("zone_name") == _Z0.name

    def test_outside_metadata_has_no_zone_name(self):
        v = make_vessel(lat=0.0, lon=0.0)
        assert "zone_name" not in in_sensitive_zone(v).metadata


# ---------------------------------------------------------------------------
# loitering  (reads entry.speed_kmh from HistoryEntry)
# ---------------------------------------------------------------------------

class TestLoitering:
    def test_fast_vessel_returns_zero(self):
        v = make_vessel(speed_kmh=20.0)
        assert loitering(v).score == 0.0

    def test_speed_at_threshold_returns_zero(self):
        v = make_vessel(speed_kmh=5.0)
        assert loitering(v).score == 0.0

    def test_slow_no_history_returns_base_score(self):
        v = make_vessel(speed_kmh=2.0)
        assert loitering(v).score == 0.5

    def test_sustained_slow_increases_score(self):
        # 3 history entries all with speed_kmh=2 → 3 slow steps counted.
        history = [
            entry(hour=0, speed_kmh=2.0),
            entry(hour=1, speed_kmh=2.0),
            entry(hour=2, speed_kmh=2.0),
            entry(hour=3, speed_kmh=2.0),
        ]
        v = make_vessel(speed_kmh=2.0, history=history)
        assert loitering(v).score > 0.5

    def test_sustained_slow_capped_at_one(self):
        history = [entry(hour=i, speed_kmh=2.0) for i in range(10)]
        v = make_vessel(speed_kmh=2.0, history=history)
        assert loitering(v).score <= 1.0

    def test_fast_entry_breaks_chain(self):
        # Most recent entry has speed_kmh=30 → chain breaks immediately → base score.
        history = [
            entry(hour=0, speed_kmh=2.0),
            entry(hour=1, speed_kmh=2.0),
            entry(hour=2, speed_kmh=30.0),  # fast — first seen in reverse walk
        ]
        v = make_vessel(speed_kmh=2.0, history=history)
        assert loitering(v).score == 0.5

    def test_stored_speed_used_not_derived_from_displacement(self):
        # Large displacement but speed_kmh=2 stored → counts as slow.
        history = [
            entry(lat=0.0, hour=0, speed_kmh=2.0),
            entry(lat=9.0, hour=1, speed_kmh=2.0),  # huge displacement, stored speed is slow
        ]
        v = make_vessel(speed_kmh=2.0, history=history)
        assert loitering(v).score > 0.5

    # --- DetectorResult shape ---

    def test_returns_detector_result(self):
        assert isinstance(loitering(make_vessel(speed_kmh=2.0)), DetectorResult)

    def test_name_is_loitering(self):
        for v in [make_vessel(speed_kmh=2.0), make_vessel(speed_kmh=20.0)]:
            assert loitering(v).name == "loitering"

    def test_fast_evidence_mentions_speed(self):
        v = make_vessel(speed_kmh=20.0)
        assert "20.0" in loitering(v).evidence

    def test_slow_evidence_mentions_speed(self):
        v = make_vessel(speed_kmh=2.0)
        assert "2.0" in loitering(v).evidence

    def test_slow_with_steps_evidence_includes_count(self):
        history = [entry(hour=i, speed_kmh=2.0) for i in range(3)]
        v = make_vessel(speed_kmh=2.0, history=history)
        result = loitering(v)
        assert "3" in result.evidence
        assert "step" in result.evidence

    def test_slow_metadata_has_slow_steps_key(self):
        v = make_vessel(speed_kmh=2.0)
        assert "slow_steps" in loitering(v).metadata

    def test_slow_no_history_metadata_slow_steps_is_zero(self):
        v = make_vessel(speed_kmh=2.0)
        assert loitering(v).metadata["slow_steps"] == 0

    def test_slow_with_history_metadata_reflects_count(self):
        history = [entry(hour=i, speed_kmh=2.0) for i in range(3)]
        v = make_vessel(speed_kmh=2.0, history=history)
        assert loitering(v).metadata["slow_steps"] == 3

    def test_fast_has_no_metadata(self):
        v = make_vessel(speed_kmh=20.0)
        assert loitering(v).metadata == {}


# ---------------------------------------------------------------------------
# route_deviation  (reads vessel.history[0].heading_deg from HistoryEntry)
# ---------------------------------------------------------------------------

class TestRouteDeviation:
    def test_no_history_falls_back_to_default_baseline(self):
        v = make_vessel(heading_deg=45.0, history=[])
        assert route_deviation(v).score == 0.0

    def test_small_deviation_returns_zero(self):
        v = make_vessel(heading_deg=60.0, history=[])   # 15° from 45° fallback
        assert route_deviation(v).score == 0.0

    def test_moderate_deviation_returns_half(self):
        v = make_vessel(heading_deg=90.0, history=[])   # 45° from 45° fallback
        assert route_deviation(v).score == 0.5

    def test_large_deviation_returns_full(self):
        v = make_vessel(heading_deg=120.0, history=[])  # 75° from 45° fallback
        assert route_deviation(v).score == 1.0

    def test_wraparound_handled_correctly(self):
        # Stored baseline 355°; vessel at 350° → 5° diff → 0.0.
        history = [entry(heading_deg=355.0)]
        v = make_vessel(heading_deg=350.0, history=history)
        assert route_deviation(v).score == 0.0

    def test_baseline_read_from_first_history_entry(self):
        # Stored baseline 90°; vessel at 45° → 45° diff → 0.5.
        history = [entry(heading_deg=90.0)]
        v = make_vessel(heading_deg=45.0, history=history)
        assert route_deviation(v).score == 0.5

    def test_only_first_entry_used_as_baseline(self):
        # First entry=90°, second entry=10° (a big swing) — only first matters.
        history = [entry(heading_deg=90.0), entry(heading_deg=10.0)]
        v = make_vessel(heading_deg=45.0, history=history)
        assert route_deviation(v).score == 0.5   # still 45° from 90°, not from 10°

    def test_stored_heading_used_not_derived_from_displacement(self):
        # entry has identical lat/lon (zero displacement), but heading_deg=90 stored.
        history = [entry(lat=1.0, lon=1.0, heading_deg=90.0)]
        v = make_vessel(heading_deg=45.0, history=history)
        assert route_deviation(v).score == 0.5   # 45° diff from stored 90° baseline

    # --- DetectorResult shape ---

    def test_returns_detector_result(self):
        v = make_vessel(heading_deg=45.0)
        assert isinstance(route_deviation(v), DetectorResult)

    def test_name_is_route_deviation(self):
        for h in [45.0, 90.0, 180.0]:
            v = make_vessel(heading_deg=h)
            assert route_deviation(v).name == "route_deviation"

    def test_evidence_includes_deviation_degrees(self):
        v = make_vessel(heading_deg=90.0, history=[])   # 45° deviation
        assert "45.0" in route_deviation(v).evidence

    def test_evidence_includes_baseline_degrees(self):
        history = [entry(heading_deg=90.0)]
        v = make_vessel(heading_deg=45.0, history=history)
        # baseline is 90.0
        assert "90.0" in route_deviation(v).evidence

    def test_metadata_has_deviation_and_baseline_keys(self):
        v = make_vessel(heading_deg=90.0, history=[])
        meta = route_deviation(v).metadata
        assert "heading_deviation_deg" in meta
        assert "baseline_heading_deg" in meta

    def test_metadata_deviation_matches_expected_value(self):
        v = make_vessel(heading_deg=90.0, history=[])   # 45° from default baseline 45°
        assert route_deviation(v).metadata["heading_deviation_deg"] == pytest.approx(45.0)

    def test_metadata_baseline_reflects_history(self):
        history = [entry(heading_deg=90.0)]
        v = make_vessel(heading_deg=45.0, history=history)
        assert route_deviation(v).metadata["baseline_heading_deg"] == pytest.approx(90.0)


# ---------------------------------------------------------------------------
# anomaly_breakdown and anomaly_score
# ---------------------------------------------------------------------------

class TestAnomalyBreakdownAndScore:
    def test_breakdown_returns_all_keys(self):
        v = make_vessel()
        bd = anomaly_breakdown(v)
        assert set(bd.keys()) == {"sensitive_zone", "loitering", "route_deviation"}

    def test_breakdown_values_are_detector_results(self):
        v = make_vessel()
        for result in anomaly_breakdown(v).values():
            assert isinstance(result, DetectorResult)

    def test_clean_vessel_scores_zero(self):
        v = make_vessel(lat=0.0, lon=0.0, speed_kmh=28.0, heading_deg=45.0)
        assert anomaly_score(v) == 0.0

    def test_score_equals_sum_of_breakdown(self):
        v = make_vessel(lat=ZONE_LAT, lon=ZONE_LON, speed_kmh=2.0, heading_deg=120.0)
        bd = anomaly_breakdown(v)
        assert anomaly_score(v) == pytest.approx(sum(r.score for r in bd.values()))

    def test_all_signals_active_reaches_max(self):
        v = make_vessel(lat=ZONE_LAT, lon=ZONE_LON, speed_kmh=2.0, heading_deg=180.0)
        assert anomaly_score(v) == pytest.approx(3.0)

    def test_score_is_non_negative(self):
        for lat, lon, speed, heading in [
            (0.0, 0.0, 30.0, 45.0),
            (ZONE_LAT, ZONE_LON, 1.0, 200.0),
            (0.5, 0.5, 10.0, 10.0),
        ]:
            v = make_vessel(lat=lat, lon=lon, speed_kmh=speed, heading_deg=heading)
            assert anomaly_score(v) >= 0.0


# ---------------------------------------------------------------------------
# Multi-zone scoring — Step 13b
# ---------------------------------------------------------------------------

# A second zone far from ZONE_ALPHA for multi-zone tests.
_ZONE_B = Zone(name="ZONE_BETA", min_lat=5.0, max_lat=6.0, min_lon=5.0, max_lon=6.0, halo=0.1)


class TestInSensitiveZoneMultiZone:
    """Verify multi-zone iteration and max-score selection.

    Uses monkeypatch to replace custody.behavior.detectors.ZONES so that
    existing config and single-zone tests are unaffected.
    """

    def test_single_zone_matches_legacy_output(self, monkeypatch):
        """One-zone config must produce identical scores to the pre-multi-zone detector."""
        monkeypatch.setattr("custody.behavior.detectors.ZONES", [_Z0])
        v_in   = make_vessel(lat=ZONE_LAT, lon=ZONE_LON)
        v_near = make_vessel(lat=_Z0.min_lat - 0.01, lon=ZONE_LON)
        v_out  = make_vessel(lat=0.0, lon=0.0)
        assert in_sensitive_zone(v_in).score  == pytest.approx(1.5)
        assert 0.0 < in_sensitive_zone(v_near).score < 1.5
        assert in_sensitive_zone(v_out).score == pytest.approx(0.0)

    def test_vessel_inside_first_zone_only(self, monkeypatch):
        monkeypatch.setattr("custody.behavior.detectors.ZONES", [_Z0, _ZONE_B])
        v = make_vessel(lat=ZONE_LAT, lon=ZONE_LON)  # inside _Z0, far from _ZONE_B
        result = in_sensitive_zone(v)
        assert result.score == pytest.approx(1.5)
        assert result.metadata["zone_name"] == _Z0.name

    def test_vessel_inside_second_zone_only(self, monkeypatch):
        monkeypatch.setattr("custody.behavior.detectors.ZONES", [_Z0, _ZONE_B])
        v = make_vessel(lat=5.5, lon=5.5)  # inside _ZONE_B, far from _Z0
        result = in_sensitive_zone(v)
        assert result.score == pytest.approx(1.5)
        assert result.metadata["zone_name"] == _ZONE_B.name

    def test_max_score_selected_over_lower_score(self, monkeypatch):
        """Vessel in halo of _Z0 but inside _ZONE_B — _ZONE_B wins."""
        monkeypatch.setattr("custody.behavior.detectors.ZONES", [_Z0, _ZONE_B])
        v = make_vessel(lat=5.5, lon=5.5)
        assert in_sensitive_zone(v).score == pytest.approx(1.5)

    def test_vessel_near_one_zone_far_from_other(self, monkeypatch):
        monkeypatch.setattr("custody.behavior.detectors.ZONES", [_Z0, _ZONE_B])
        v = make_vessel(lat=_Z0.min_lat - 0.05, lon=ZONE_LON)  # in _Z0 halo, outside _ZONE_B
        result = in_sensitive_zone(v)
        assert 0.0 < result.score < 1.5
        assert result.metadata["zone_name"] == _Z0.name

    def test_vessel_outside_all_zones_returns_zero(self, monkeypatch):
        monkeypatch.setattr("custody.behavior.detectors.ZONES", [_Z0, _ZONE_B])
        v = make_vessel(lat=0.0, lon=0.0)
        assert in_sensitive_zone(v).score == pytest.approx(0.0)
        assert "zone_name" not in in_sensitive_zone(v).metadata

    def test_tie_resolved_by_list_order(self, monkeypatch):
        """Two zones with identical geometry — first in list wins."""
        z_copy = Zone(
            name="ZONE_COPY",
            min_lat=_Z0.min_lat, max_lat=_Z0.max_lat,
            min_lon=_Z0.min_lon, max_lon=_Z0.max_lon,
            halo=_Z0.halo,
        )
        monkeypatch.setattr("custody.behavior.detectors.ZONES", [_Z0, z_copy])
        v = make_vessel(lat=ZONE_LAT, lon=ZONE_LON)
        assert in_sensitive_zone(v).metadata["zone_name"] == _Z0.name

        monkeypatch.setattr("custody.behavior.detectors.ZONES", [z_copy, _Z0])
        assert in_sensitive_zone(v).metadata["zone_name"] == z_copy.name

    def test_empty_zones_returns_zero_score(self, monkeypatch):
        monkeypatch.setattr("custody.behavior.detectors.ZONES", [])
        v = make_vessel(lat=ZONE_LAT, lon=ZONE_LON)
        result = in_sensitive_zone(v)
        assert result.score == pytest.approx(0.0)
        assert result.name == "sensitive_zone"


# ---------------------------------------------------------------------------
# vessel_proximity — Step 14b
# ---------------------------------------------------------------------------

# Reference position for proximity tests (equatorial, well outside any zone).
_REF_LAT, _REF_LON = 0.0, 0.0


def _prox_rec(target_id: str, lat: float, lon: float) -> dict:
    return {"target_id": target_id, "lat": lat, "lon": lon}


def _lat_offset_for_km(km: float) -> float:
    """Approximate latitude offset in degrees for a given km (equatorial)."""
    return km / 111.0


class TestVesselProximity:
    # ── name contract ──────────────────────────────────────────────────────

    def test_name_is_vessel_proximity(self):
        result = vessel_proximity(_REF_LAT, _REF_LON, "V001", [])
        assert result.name == "vessel_proximity"

    def test_returns_detector_result(self):
        result = vessel_proximity(_REF_LAT, _REF_LON, "V001", [])
        assert isinstance(result, DetectorResult)

    # ── no context ─────────────────────────────────────────────────────────

    def test_no_others_returns_zero(self):
        result = vessel_proximity(_REF_LAT, _REF_LON, "V001", [])
        assert result.score == pytest.approx(0.0)

    def test_no_others_evidence(self):
        result = vessel_proximity(_REF_LAT, _REF_LON, "V001", [])
        assert "no other vessels" in result.evidence

    def test_no_others_metadata_empty(self):
        result = vessel_proximity(_REF_LAT, _REF_LON, "V001", [])
        assert result.metadata == {}

    # ── beyond warning range ───────────────────────────────────────────────

    def test_beyond_warning_range_returns_zero(self):
        offset = _lat_offset_for_km(PROXIMITY_WARNING_KM + 5.0)
        others = [_prox_rec("V002", _REF_LAT + offset, _REF_LON)]
        result = vessel_proximity(_REF_LAT, _REF_LON, "V001", others)
        assert result.score == pytest.approx(0.0)

    def test_beyond_warning_range_evidence(self):
        offset = _lat_offset_for_km(PROXIMITY_WARNING_KM + 5.0)
        others = [_prox_rec("V002", _REF_LAT + offset, _REF_LON)]
        result = vessel_proximity(_REF_LAT, _REF_LON, "V001", others)
        assert "no vessels within" in result.evidence

    def test_beyond_warning_range_metadata_empty(self):
        offset = _lat_offset_for_km(PROXIMITY_WARNING_KM + 5.0)
        others = [_prox_rec("V002", _REF_LAT + offset, _REF_LON)]
        result = vessel_proximity(_REF_LAT, _REF_LON, "V001", others)
        assert result.metadata == {}

    # ── warning range ──────────────────────────────────────────────────────

    def test_within_warning_range_returns_half(self):
        # Place vessel between critical and warning thresholds.
        mid_km = (PROXIMITY_CRITICAL_KM + PROXIMITY_WARNING_KM) / 2
        offset = _lat_offset_for_km(mid_km)
        others = [_prox_rec("V002", _REF_LAT + offset, _REF_LON)]
        result = vessel_proximity(_REF_LAT, _REF_LON, "V001", others)
        assert result.score == pytest.approx(0.5)

    def test_warning_range_evidence_contains_vessel_id(self):
        mid_km = (PROXIMITY_CRITICAL_KM + PROXIMITY_WARNING_KM) / 2
        offset = _lat_offset_for_km(mid_km)
        others = [_prox_rec("V002", _REF_LAT + offset, _REF_LON)]
        result = vessel_proximity(_REF_LAT, _REF_LON, "V001", others)
        assert "V002" in result.evidence
        assert "warning range" in result.evidence

    def test_warning_range_metadata(self):
        mid_km = (PROXIMITY_CRITICAL_KM + PROXIMITY_WARNING_KM) / 2
        offset = _lat_offset_for_km(mid_km)
        others = [_prox_rec("V002", _REF_LAT + offset, _REF_LON)]
        result = vessel_proximity(_REF_LAT, _REF_LON, "V001", others)
        assert result.metadata["nearest_vessel_id"] == "V002"
        assert result.metadata["distance_km"] > 0.0
        assert result.metadata["threshold_km"] == PROXIMITY_WARNING_KM

    # ── critical range ─────────────────────────────────────────────────────

    def test_within_critical_range_returns_one(self):
        offset = _lat_offset_for_km(PROXIMITY_CRITICAL_KM / 2)
        others = [_prox_rec("V002", _REF_LAT + offset, _REF_LON)]
        result = vessel_proximity(_REF_LAT, _REF_LON, "V001", others)
        assert result.score == pytest.approx(1.0)

    def test_critical_range_evidence_contains_vessel_id(self):
        offset = _lat_offset_for_km(PROXIMITY_CRITICAL_KM / 2)
        others = [_prox_rec("V002", _REF_LAT + offset, _REF_LON)]
        result = vessel_proximity(_REF_LAT, _REF_LON, "V001", others)
        assert "V002" in result.evidence
        assert "warning range" not in result.evidence

    def test_critical_range_metadata(self):
        offset = _lat_offset_for_km(PROXIMITY_CRITICAL_KM / 2)
        others = [_prox_rec("V002", _REF_LAT + offset, _REF_LON)]
        result = vessel_proximity(_REF_LAT, _REF_LON, "V001", others)
        assert result.metadata["nearest_vessel_id"] == "V002"
        assert result.metadata["distance_km"] < PROXIMITY_CRITICAL_KM
        assert result.metadata["threshold_km"] == PROXIMITY_CRITICAL_KM

    # ── boundary conditions ────────────────────────────────────────────────

    def test_just_below_critical_threshold_is_critical(self):
        offset = _lat_offset_for_km(PROXIMITY_CRITICAL_KM * 0.99)
        others = [_prox_rec("V002", _REF_LAT + offset, _REF_LON)]
        result = vessel_proximity(_REF_LAT, _REF_LON, "V001", others)
        assert result.score == pytest.approx(1.0)

    def test_just_below_warning_threshold_is_warning(self):
        offset = _lat_offset_for_km(PROXIMITY_WARNING_KM * 0.99)
        others = [_prox_rec("V002", _REF_LAT + offset, _REF_LON)]
        result = vessel_proximity(_REF_LAT, _REF_LON, "V001", others)
        assert result.score == pytest.approx(0.5)

    # ── anomaly_breakdown unchanged ────────────────────────────────────────

    def test_anomaly_breakdown_does_not_include_proximity(self):
        v = make_vessel(lat=_REF_LAT, lon=_REF_LON)
        bd = anomaly_breakdown(v)
        assert "vessel_proximity" not in bd
        assert set(bd.keys()) == {"sensitive_zone", "loitering", "route_deviation"}

    def test_anomaly_score_unchanged_by_proximity_module(self):
        v = make_vessel(lat=_REF_LAT, lon=_REF_LON)
        score = anomaly_score(v)
        # Score is purely based on the three existing detectors.
        assert score == pytest.approx(
            sum(r.score for r in anomaly_breakdown(v).values())
        )
