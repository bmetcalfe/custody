"""Tests for AnthropicBackend + shared parsing (ADR-0015).

Mocked-backend tests cover parsing variants, tokens/wall-time plumbing, and
PositionObservation round-trip of the detector_reasoning field.  One opt-in
real-API test is gated on ANTHROPIC_API_KEY; skips when the env var is unset.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from custody.detection.vlm_backends import AnthropicBackend, VLMDetection
from custody.detection.vlm_backends._parsing import parse_vlm_response
from custody.fusion.index import _position_from_row, index_observations
from custody.fusion.observations import PositionObservation


# ---------------------------------------------------------------------------
# Parser tests — cover the response shapes seen in the Week 2 VLM spike.
# ---------------------------------------------------------------------------


def test_parse_plain_json_list_with_bbox_array():
    raw = '[{"vessel_id": 1, "bbox": [10, 20, 60, 80], "confidence": "high", "reasoning": "bright compact return"}]'
    dets = parse_vlm_response(raw)
    assert len(dets) == 1
    assert dets[0].bbox == (10, 20, 60, 80)
    assert dets[0].confidence == "high"
    assert dets[0].reasoning == "bright compact return"


def test_parse_markdown_fenced_json():
    raw = (
        "Looking at this SAR image I see a vessel.\n"
        "```json\n"
        '[{"bbox": [155, 255, 620, 380], "confidence": "high"}]\n'
        "```\n"
        "Notes: bright feature."
    )
    dets = parse_vlm_response(raw)
    assert len(dets) == 1
    assert dets[0].bbox == (155, 255, 620, 380)


def test_parse_json_with_explanatory_preamble():
    raw = (
        "Here is my analysis of the SAR tile:\n\n"
        '{"vessels": [{"bounding_box": {"x1": 130, "y1": 240, "x2": 580, "y2": 420}, '
        '"confidence": "high", "reasoning": "metal superstructure return"}]}\n\n'
        "Additional commentary..."
    )
    dets = parse_vlm_response(raw)
    assert len(dets) == 1
    assert dets[0].bbox == (130, 240, 580, 420)
    assert dets[0].confidence == "high"
    assert "superstructure" in dets[0].reasoning


def test_parse_nested_detections_key_with_bounding_box_dict():
    raw = json.dumps(
        {
            "description": "SAR image analysis",
            "detections": [
                {
                    "id": 1,
                    "bounding_box": {"x1": 0, "y1": 0, "x2": 100, "y2": 50},
                    "confidence": "medium",
                },
                {
                    "id": 2,
                    "bounding_box": {"x1": 200, "y1": 200, "x2": 250, "y2": 240},
                    "confidence": "low",
                    "notes": "likely wake artifact",
                },
            ],
        }
    )
    dets = parse_vlm_response(raw)
    assert len(dets) == 2
    assert dets[1].reasoning == "likely wake artifact"


def test_parse_malformed_returns_empty_list_does_not_raise():
    # Unterminated JSON and mixed junk
    for junk in ("not json", "{broken", "```json\n{foo: bar}\n```", "", None, 42):
        assert parse_vlm_response(junk) == []  # type: ignore[arg-type]


def test_parse_case_insensitive_confidence_normalization():
    raw = '[{"bbox":[1,2,3,4],"confidence":"HIGH"},{"bbox":[5,6,7,8],"confidence":"Med"}]'
    dets = parse_vlm_response(raw)
    assert dets[0].confidence == "high"
    assert dets[1].confidence == "medium"


def test_parse_missing_confidence_defaults_to_medium():
    raw = '[{"bbox":[1,2,3,4]}]'
    dets = parse_vlm_response(raw)
    assert dets[0].confidence == "medium"


def test_parse_missing_bbox_skips_item():
    raw = '[{"confidence":"high","reasoning":"no box"}, {"bbox":[1,2,3,4]}]'
    dets = parse_vlm_response(raw)
    assert len(dets) == 1
    assert dets[0].bbox == (1, 2, 3, 4)


# ---------------------------------------------------------------------------
# Backend tests — mock the Anthropic client.
# ---------------------------------------------------------------------------


def _mock_anthropic_response(text: str, input_tokens: int = 100, output_tokens: int = 50):
    """Build a SimpleNamespace that looks like anthropic.types.Message."""
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def _make_backend_with_mock(mock_response):
    """Construct AnthropicBackend with a mocked client.messages.create()."""
    be = AnthropicBackend(model="claude-sonnet-4-6", api_key="fake-for-construction-only")
    be._client = MagicMock()
    be._client.messages.create.return_value = mock_response
    return be


def test_backend_detect_tile_parses_and_populates_metadata():
    raw = '```json\n[{"bbox":[10,20,60,80],"confidence":"high","reasoning":"vessel-like return"}]\n```'
    be = _make_backend_with_mock(_mock_anthropic_response(raw, 712, 699))
    tile = np.zeros((64, 64), dtype=np.uint8)
    resp = be.detect_tile(tile, prompt="describe")
    assert len(resp.detections) == 1
    assert resp.detections[0].bbox == (10, 20, 60, 80)
    assert resp.detections[0].raw_response == raw
    assert resp.tokens_used == {"input": 712, "output": 699}
    assert resp.wall_time_seconds is not None and resp.wall_time_seconds >= 0


def test_backend_detect_tile_passes_model_and_prompt_to_client():
    be = _make_backend_with_mock(_mock_anthropic_response("[]"))
    be.detect_tile(np.zeros((32, 32), dtype=np.uint8), prompt="my custom prompt")
    call_kwargs = be._client.messages.create.call_args.kwargs
    assert call_kwargs["model"] == "claude-sonnet-4-6"
    msg_content = call_kwargs["messages"][0]["content"]
    text_blocks = [c for c in msg_content if c.get("type") == "text"]
    assert any("my custom prompt" in c.get("text", "") for c in text_blocks)
    image_blocks = [c for c in msg_content if c.get("type") == "image"]
    assert len(image_blocks) == 1
    assert image_blocks[0]["source"]["media_type"] == "image/png"


def test_backend_detect_tile_malformed_response_empty_detections_no_raise():
    be = _make_backend_with_mock(_mock_anthropic_response("garbage not parseable"))
    resp = be.detect_tile(np.zeros((64, 64), dtype=np.uint8), prompt="x")
    assert resp.detections == []
    assert resp.tokens_used is not None


def test_backend_properties():
    be = AnthropicBackend(model="claude-opus-4-7", api_key="fake")
    assert be.model_name == "claude-opus-4-7"
    assert be.approx_cost_per_tile_usd == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# Parquet round-trip — detector_reasoning field
# ---------------------------------------------------------------------------


def _make_obs(obs_id: str, reasoning: str | None) -> PositionObservation:
    return PositionObservation(
        obs_id=obs_id,
        source_id="vlm-probe",
        modality="SAR",
        acquisition_time=1_688_306_455.0,
        ingestion_time=1_688_306_460.0,
        lat=8.8557,
        lon=114.6651,
        cov_pos=np.eye(2) * 100.0,
        raw_ref="probe://synthetic",
        detector_version="vlm_claude-sonnet-4-6",
        classification_conf=0.85,
        detector_reasoning=reasoning,
    )


def test_parquet_roundtrip_detector_reasoning_populated(tmp_path: Path):
    text = "Compact bright return on dark water; azimuth smearing consistent with metal hull."
    obs = _make_obs("vlm-populated-1", text)
    index_observations([obs], out_dir=tmp_path)
    import duckdb

    pq = tmp_path / "position.parquet"
    rows = duckdb.execute(
        f"SELECT * FROM read_parquet('{pq.as_posix()}') WHERE obs_id='vlm-populated-1'"
    ).fetchall()
    cols = [
        c[0]
        for c in duckdb.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{pq.as_posix()}')"
        ).fetchall()
    ]
    assert "detector_reasoning" in cols
    row = dict(zip(cols, rows[0]))
    loaded = _position_from_row(row)
    assert loaded.detector_reasoning == text


def test_parquet_roundtrip_detector_reasoning_none(tmp_path: Path):
    obs = _make_obs("vlm-legacy-1", None)
    index_observations([obs], out_dir=tmp_path)
    import duckdb

    pq = tmp_path / "position.parquet"
    rows = duckdb.execute(
        f"SELECT * FROM read_parquet('{pq.as_posix()}') WHERE obs_id='vlm-legacy-1'"
    ).fetchall()
    cols = [
        c[0]
        for c in duckdb.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{pq.as_posix()}')"
        ).fetchall()
    ]
    row = dict(zip(cols, rows[0]))
    loaded = _position_from_row(row)
    assert loaded.detector_reasoning is None


# ---------------------------------------------------------------------------
# Opt-in real-API smoke test
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set; skipping live-API smoke test",
)
def test_real_api_end_to_end_on_synthetic_tile():
    # 100x100 noise + a 20x20 bright square at center, just to have structure.
    rng = np.random.default_rng(0)
    tile = rng.integers(0, 80, size=(100, 100), dtype=np.uint8)
    tile[40:60, 40:60] = 240
    be = AnthropicBackend(model="claude-sonnet-4-6")
    resp = be.detect_tile(tile, prompt="What do you see in this image? Answer in one sentence.")
    # We're not asserting specific detection content — just that the plumbing works
    # end-to-end: tokens used, non-zero wall time, no exception.
    assert resp.tokens_used is not None
    assert resp.tokens_used["input"] > 0
    assert resp.tokens_used["output"] > 0
    assert resp.wall_time_seconds is not None and resp.wall_time_seconds > 0
