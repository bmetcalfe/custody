"""Tests for VLM backend abstract interface + stub backends (ADR-0015)."""
from __future__ import annotations

import numpy as np
import pytest

from custody.detection.vlm_backends import (
    MoondreamBackend,
    QwenBackend,
    VLMBackend,
    VLMDetection,
    VLMResponse,
)


def test_vlm_backend_cannot_be_instantiated_directly():
    """VLMBackend is abstract — raising TypeError without a concrete subclass."""
    with pytest.raises(TypeError):
        VLMBackend()  # type: ignore[abstract]


def test_vlm_detection_and_response_construct_with_reasonable_inputs():
    det = VLMDetection(
        bbox=(10, 20, 60, 80),
        confidence="high",
        reasoning="Compact bright return on dark water; azimuth smearing consistent with a vessel.",
        raw_response='{"vessels": [...]}',
    )
    assert det.bbox == (10, 20, 60, 80)
    assert det.confidence == "high"
    assert det.reasoning.startswith("Compact")
    assert det.raw_response is not None

    resp = VLMResponse(
        detections=[det],
        scene_description="Open water with a single bright vessel in the image center.",
        tokens_used={"input": 712, "output": 699},
        wall_time_seconds=12.4,
    )
    assert len(resp.detections) == 1
    assert resp.tokens_used["input"] == 712
    assert resp.wall_time_seconds == pytest.approx(12.4)

    # Default fields: empty detections list, Nones elsewhere
    bare = VLMResponse()
    assert bare.detections == []
    assert bare.scene_description is None
    assert bare.tokens_used is None
    assert bare.wall_time_seconds is None


def test_qwen_backend_raises_not_implemented_on_detect():
    be = QwenBackend()
    with pytest.raises(NotImplementedError, match="QwenBackend"):
        be.detect_tile(np.zeros((100, 100), dtype=np.uint8), prompt="describe")
    with pytest.raises(NotImplementedError):
        _ = be.model_name
    with pytest.raises(NotImplementedError):
        _ = be.approx_cost_per_tile_usd


def test_moondream_backend_raises_not_implemented_on_detect():
    be = MoondreamBackend()
    with pytest.raises(NotImplementedError, match="MoondreamBackend"):
        be.detect_tile(np.zeros((100, 100), dtype=np.uint8), prompt="describe")
    with pytest.raises(NotImplementedError):
        _ = be.model_name
    with pytest.raises(NotImplementedError):
        _ = be.approx_cost_per_tile_usd
