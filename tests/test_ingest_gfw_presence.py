"""
Tests for custody.ingest.gfw_presence — GFW v3 presence response parser (ADR-0011).

Fixture-based: every test uses either hand-built synthetic dicts or the
probe_4wings_presence_hourly.json fixture saved during Phase A
reconnaissance.  No live API calls.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from custody.fusion.observations import PositionObservation
from custody.ingest.gfw_presence import (
    DropReport,
    GFW_PRESENCE_POS_SIGMA_M,
    parse_gfw_presence_response,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "gfw_presence"
HOURLY_FIXTURE = FIXTURE_DIR / "probe_4wings_presence_hourly.json"


def _record(**overrides) -> dict:
    """Minimal well-formed GFW presence record; overrides replace any field."""
    rec = {
        "mmsi": "574128122",
        "date": "2023-07-02 17:00",
        "lat": 8.88,
        "lon": 114.68,
        "flag": "VNM",
        "geartype": "OTHER",
        "vesselType": "OTHER",
        "vesselId": "578b9a23b-b8a2-3c59-480d-6425098029c7",
        "shipName": "KIEM NGU - 414",
        "callsign": "KN414",
        "imo": "5454414",
        "entryTimestamp": "2023-07-02T00:00:00Z",
        "exitTimestamp": "2023-07-02T23:00:00Z",
        "hours": 1,
        "firstTransmissionDate": "2022-05-06T07:16:17Z",
        "lastTransmissionDate": "2025-11-20T09:57:37Z",
    }
    rec.update(overrides)
    return rec


def _wrap(records: list[dict]) -> dict:
    """Wrap a list of records in the GFW response envelope."""
    return {
        "total": 1,
        "limit": None,
        "offset": None,
        "nextOffset": None,
        "metadata": {},
        "entries": [{"public-global-presence:v4.0": records}],
    }


# ---------------------------------------------------------------------------
# 1-5. SCHEMA MAPPING
# ---------------------------------------------------------------------------


def test_valid_record_produces_position_observation():
    obs_list, drops = parse_gfw_presence_response(_wrap([_record()]))
    assert len(obs_list) == 1
    assert drops.total == 0
    obs = obs_list[0]
    assert isinstance(obs, PositionObservation)


def test_obs_id_format():
    obs_list, _ = parse_gfw_presence_response(_wrap([_record()]))
    obs = obs_list[0]
    assert obs.obs_id.startswith("gfw-presence-574128122-")
    # Epoch for 2023-07-02 17:00 UTC
    parts = obs.obs_id.rsplit("-", 1)
    epoch = int(parts[-1])
    assert epoch == int(obs.acquisition_time)


def test_identity_fields_preserved_in_notes():
    obs_list, _ = parse_gfw_presence_response(_wrap([_record()]))
    notes = obs_list[0].notes
    assert notes["flag"] == "VNM"
    assert notes["vesselType"] == "OTHER"
    assert notes["geartype"] == "OTHER"
    assert notes["shipName"] == "KIEM NGU - 414"
    assert notes["vesselId"] == "578b9a23b-b8a2-3c59-480d-6425098029c7"


def test_covariance_matches_gfw_presence_sigma():
    obs_list, _ = parse_gfw_presence_response(_wrap([_record()]))
    expected = (GFW_PRESENCE_POS_SIGMA_M ** 2) * np.eye(2)
    # GFW_PRESENCE_POS_SIGMA_M = 500 → diag(250000, 250000)
    np.testing.assert_allclose(obs_list[0].cov_pos, expected, atol=1e-9)
    np.testing.assert_allclose(obs_list[0].cov_pos, np.diag([250_000.0, 250_000.0]), atol=1e-9)


def test_acquisition_time_parsed_from_date_string():
    obs_list, _ = parse_gfw_presence_response(_wrap([_record(date="2023-07-02 17:00")]))
    # 2023-07-02 17:00 UTC → epoch 1688317200
    assert obs_list[0].acquisition_time == 1_688_317_200.0


def test_modality_is_ais():
    obs_list, _ = parse_gfw_presence_response(_wrap([_record()]))
    assert obs_list[0].modality == "AIS"


# ---------------------------------------------------------------------------
# 6-10. DROP POLICY
# ---------------------------------------------------------------------------


def test_lat_sentinel_dropped():
    obs_list, drops = parse_gfw_presence_response(_wrap([_record(lat=91.0)]))
    assert obs_list == []
    assert drops.invalid_position == 1


def test_lon_sentinel_dropped():
    obs_list, drops = parse_gfw_presence_response(_wrap([_record(lon=181.0)]))
    assert obs_list == []
    assert drops.invalid_position == 1


def test_out_of_range_lat_dropped():
    obs_list, drops = parse_gfw_presence_response(_wrap([_record(lat=95.0)]))
    assert obs_list == []
    assert drops.invalid_position == 1


def test_missing_mmsi_dropped():
    bad = _record()
    del bad["mmsi"]
    obs_list, drops = parse_gfw_presence_response(_wrap([bad]))
    assert obs_list == []
    assert drops.missing_mmsi == 1


def test_missing_date_dropped():
    bad = _record()
    del bad["date"]
    obs_list, drops = parse_gfw_presence_response(_wrap([bad]))
    assert obs_list == []
    assert drops.missing_timestamp == 1


def test_duplicate_mmsi_timestamp_second_dropped():
    a = _record(lat=8.88, lon=114.68)
    b = _record(lat=8.89, lon=114.69)   # different lat/lon, same mmsi+time
    obs_list, drops = parse_gfw_presence_response(_wrap([a, b]))
    assert len(obs_list) == 1
    assert drops.duplicate == 1


# ---------------------------------------------------------------------------
# 11-13. INTEGRATION + COV VALIDATION + NOTES
# ---------------------------------------------------------------------------


def test_hourly_fixture_roundtrip():
    assert HOURLY_FIXTURE.exists(), f"fixture missing: {HOURLY_FIXTURE}"
    with HOURLY_FIXTURE.open() as f:
        response = json.load(f)
    obs_list, drops = parse_gfw_presence_response(response)
    # Phase A probe recorded 59 hourly records; some may be duplicates on
    # (mmsi, timestamp) — allow for modest dedup.
    assert 40 <= len(obs_list) <= 59
    assert drops.total >= 0
    # Every emitted observation must validate as a real PositionObservation
    for obs in obs_list:
        assert isinstance(obs, PositionObservation)
        assert obs.modality == "AIS"
        assert obs.source_id == "gfw_presence"
        assert obs.cov_pos.shape == (2, 2)


def test_covariance_passes_observation_psd_validation():
    """PositionObservation.__post_init__ validates PSD — constructing the obs is the check."""
    obs_list, _ = parse_gfw_presence_response(_wrap([_record()]))
    obs = obs_list[0]
    assert obs.cov_pos.shape == (2, 2)
    eigs = np.linalg.eigvalsh(obs.cov_pos)
    assert eigs.min() > 0  # strictly positive-definite for non-zero σ


def test_notes_preserves_flag_geartype_shipname_together():
    obs_list, _ = parse_gfw_presence_response(_wrap([
        _record(flag="VNM", geartype="OTHER", shipName="KIEM NGU - 414"),
    ]))
    notes = obs_list[0].notes
    assert notes["flag"] == "VNM"
    assert notes["geartype"] == "OTHER"
    assert notes["shipName"] == "KIEM NGU - 414"


# ---------------------------------------------------------------------------
# 14-15. EMPTY / MALFORMED INPUT
# ---------------------------------------------------------------------------


def test_empty_response_yields_empty_list():
    obs_list, drops = parse_gfw_presence_response(_wrap([]))
    assert obs_list == []
    assert drops.total == 0


def test_response_without_entries_key_raises_value_error():
    with pytest.raises(ValueError, match="entries"):
        parse_gfw_presence_response({"total": 0})
