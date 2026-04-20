"""Tests for OpenAIBackend (ADR-0015).

Mirrors the Anthropic backend test structure: mocked-client tests exercise
metadata plumbing, request shape, malformed-response handling, and backend
properties.  One opt-in real-API test runs against a small synthetic tile
when OPENAI_API_KEY is set.  Parser tests live in the Anthropic test file —
both backends share ``_parsing.parse_vlm_response``.
"""
from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from custody.detection.vlm_backends import OpenAIBackend


# ---------------------------------------------------------------------------
# Mocked-backend tests
# ---------------------------------------------------------------------------


def _mock_openai_response(text: str, input_tokens: int = 100, output_tokens: int = 50):
    """Build a SimpleNamespace that looks like a Responses API result.

    The backend prefers ``response.output_text`` when present, so we supply it
    directly — this is the convenience attribute openai>=2.0 exposes on
    ``Response`` objects.
    """
    return SimpleNamespace(
        output_text=text,
        output=[],
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def _make_backend_with_mock(mock_response):
    """Construct OpenAIBackend with a mocked client.responses.create()."""
    be = OpenAIBackend(model="gpt-5.4-mini", api_key="fake-for-construction-only")
    be._client = MagicMock()
    be._client.responses.create.return_value = mock_response
    return be


def test_backend_detect_tile_parses_and_populates_metadata():
    raw = '```json\n[{"bbox":[10,20,60,80],"confidence":"high","reasoning":"vessel-like return"}]\n```'
    be = _make_backend_with_mock(_mock_openai_response(raw, 712, 699))
    tile = np.zeros((64, 64), dtype=np.uint8)
    resp = be.detect_tile(tile, prompt="describe")
    assert len(resp.detections) == 1
    assert resp.detections[0].bbox == (10, 20, 60, 80)
    assert resp.detections[0].confidence == "high"
    assert resp.detections[0].raw_response == raw
    assert resp.tokens_used == {"input": 712, "output": 699}
    assert resp.wall_time_seconds is not None and resp.wall_time_seconds >= 0


def test_backend_detect_tile_passes_model_and_prompt_to_client():
    be = _make_backend_with_mock(_mock_openai_response("[]"))
    be.detect_tile(np.zeros((32, 32), dtype=np.uint8), prompt="my custom prompt")
    call_kwargs = be._client.responses.create.call_args.kwargs
    assert call_kwargs["model"] == "gpt-5.4-mini"
    assert call_kwargs["max_output_tokens"] == 1024
    msg_content = call_kwargs["input"][0]["content"]
    text_blocks = [c for c in msg_content if c.get("type") == "input_text"]
    assert any("my custom prompt" in c.get("text", "") for c in text_blocks)
    image_blocks = [c for c in msg_content if c.get("type") == "input_image"]
    assert len(image_blocks) == 1
    assert image_blocks[0]["image_url"].startswith("data:image/png;base64,")


def test_backend_detect_tile_malformed_response_empty_detections_no_raise():
    be = _make_backend_with_mock(_mock_openai_response("garbage not parseable"))
    resp = be.detect_tile(np.zeros((64, 64), dtype=np.uint8), prompt="x")
    assert resp.detections == []
    assert resp.tokens_used is not None


def test_backend_extracts_text_from_output_blocks_when_output_text_missing():
    """Fallback path: no ``output_text`` attribute; join text from ``output`` items."""
    raw_segments = ["[{", '"bbox":[1,2,3,4],"confidence":"high"', "}]"]
    response = SimpleNamespace(
        output_text=None,
        output=[
            SimpleNamespace(
                content=[SimpleNamespace(text=seg) for seg in raw_segments]
            )
        ],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )
    be = _make_backend_with_mock(response)
    resp = be.detect_tile(np.zeros((32, 32), dtype=np.uint8), prompt="x")
    assert len(resp.detections) == 1
    assert resp.detections[0].bbox == (1, 2, 3, 4)


def test_backend_properties():
    be = OpenAIBackend(model="gpt-5.4", api_key="fake")
    assert be.model_name == "gpt-5.4"
    assert be.approx_cost_per_tile_usd == pytest.approx(0.003)


def test_backend_default_model_is_mini():
    be = OpenAIBackend(api_key="fake")
    assert be.model_name == "gpt-5.4-mini"


# ---------------------------------------------------------------------------
# Model-dependent cost lookup
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model,expected_cost",
    [
        ("gpt-5.4-mini", 0.0005),
        ("gpt-5.4-nano", 0.0003),
        ("gpt-5.4",      0.003),
        ("gpt-5.4-pro",  0.015),
    ],
)
def test_cost_lookup_returns_expected_value_for_known_models(model, expected_cost):
    be = OpenAIBackend(model=model, api_key="fake")
    assert be.approx_cost_per_tile_usd == pytest.approx(expected_cost)


@pytest.mark.parametrize(
    "effort,expected_cost",
    [
        ("minimal", 0.004),
        ("low",     0.010),
        ("medium",  0.020),
        ("high",    0.040),
        ("xhigh",   0.080),
    ],
)
def test_cost_lookup_accounts_for_reasoning_effort(effort, expected_cost):
    be = OpenAIBackend(model="gpt-5.4", api_key="fake", reasoning_effort=effort)
    assert be.approx_cost_per_tile_usd == pytest.approx(expected_cost)


def test_cost_lookup_unknown_model_falls_back_to_conservative_estimate(caplog):
    import logging
    be = OpenAIBackend(model="gpt-9.9-hyper-experimental", api_key="fake")
    with caplog.at_level(logging.WARNING, logger="custody.detection.vlm_backends.openai_backend"):
        cost = be.approx_cost_per_tile_usd
    assert cost == pytest.approx(0.04)
    # A warning should surface so users notice they're on an unrecognized model.
    assert any(
        "gpt-9.9-hyper-experimental" in rec.message and "fallback" in rec.message.lower()
        for rec in caplog.records
    ), f"expected fallback warning; got {[r.message for r in caplog.records]}"


def test_cost_lookup_unknown_reasoning_effort_falls_back():
    """Reasoning effort 'extreme' is not in the lookup; falls back to conservative value."""
    be = OpenAIBackend(model="gpt-5.4", api_key="fake", reasoning_effort="extreme")
    assert be.approx_cost_per_tile_usd == pytest.approx(0.04)


def test_reasoning_effort_is_forwarded_to_client():
    """When reasoning_effort is set, detect_tile must pass it to responses.create."""
    be = _make_backend_with_mock(_mock_openai_response("[]"))
    be._reasoning_effort = "high"
    be.detect_tile(np.zeros((32, 32), dtype=np.uint8), prompt="x")
    call_kwargs = be._client.responses.create.call_args.kwargs
    assert call_kwargs["reasoning"] == {"effort": "high"}


def test_reasoning_effort_absent_means_no_reasoning_kwarg():
    """When reasoning_effort is None (default), 'reasoning' must not appear in call kwargs."""
    be = _make_backend_with_mock(_mock_openai_response("[]"))
    be.detect_tile(np.zeros((32, 32), dtype=np.uint8), prompt="x")
    call_kwargs = be._client.responses.create.call_args.kwargs
    assert "reasoning" not in call_kwargs


def test_backend_max_tokens_configurable():
    be = _make_backend_with_mock(_mock_openai_response("[]"))
    be._max_tokens = 256
    be.detect_tile(np.zeros((16, 16), dtype=np.uint8), prompt="x")
    assert be._client.responses.create.call_args.kwargs["max_output_tokens"] == 256


def test_backend_supports_rgb_tile_encoding():
    """RGB tiles (HxWx3) go through the same encode path as grayscale."""
    be = _make_backend_with_mock(_mock_openai_response("[]"))
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    rgb[:, :, 0] = 128
    resp = be.detect_tile(rgb, prompt="x")
    assert resp.detections == []
    img_block = be._client.responses.create.call_args.kwargs["input"][0]["content"][0]
    assert img_block["type"] == "input_image"
    assert img_block["image_url"].startswith("data:image/png;base64,")


# ---------------------------------------------------------------------------
# Opt-in real-API smoke test
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set; skipping live-API smoke test",
)
def test_real_api_end_to_end_on_synthetic_tile():
    # 100x100 noise + a 20x20 bright square at center, just to have structure.
    rng = np.random.default_rng(0)
    tile = rng.integers(0, 80, size=(100, 100), dtype=np.uint8)
    tile[40:60, 40:60] = 240
    be = OpenAIBackend(model="gpt-5.4-mini")
    resp = be.detect_tile(tile, prompt="What do you see in this image? Answer in one sentence.")
    # Plumbing check only — no assertion on detection content.
    assert resp.tokens_used is not None
    assert resp.tokens_used["input"] > 0
    assert resp.tokens_used["output"] > 0
    assert resp.wall_time_seconds is not None and resp.wall_time_seconds > 0
