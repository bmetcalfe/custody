"""OpenAI (GPT) VLM backend (ADR-0015).

Uses the OpenAI Python SDK's Responses API, which is the current recommended
pattern for multimodal input in openai>=2.0.  Default model is ``gpt-5.4-mini``
for the Week 2 cross-model spike — low cost per tile, vision-capable.
"""
from __future__ import annotations

import base64
import logging
import time
from io import BytesIO
from typing import Optional

import numpy as np
from openai import OpenAI
from PIL import Image

from custody.detection.vlm_backends._parsing import parse_vlm_response
from custody.detection.vlm_backends.base import VLMBackend, VLMResponse


_log = logging.getLogger(__name__)


# Per-tile USD cost keyed by ``model_name`` (plus an optional
# ``+reasoning=<effort>`` suffix when reasoning is configured).  Values are
# based on current public pricing and the Task-2 cross-model spike measurements
# on real SAR tiles (2026-04-20):
#
#   gpt-5.4-mini          measured ~$0.00036/call → 0.0005 (upper side)
#   gpt-5.4 flagship      measured ~$0.00266/call → 0.003  (upper side)
#   gpt-5.4 + reasoning=h measured ~$0.03731/call → 0.04   (upper side)
#
# ``gpt-5.4-nano`` and ``gpt-5.4-pro`` are estimates — no live measurements yet.
# Update when we get measured data.
_MODEL_COST_PER_TILE_USD: dict[str, float] = {
    "gpt-5.4-mini":                 0.0005,
    "gpt-5.4-nano":                 0.0003,
    "gpt-5.4":                      0.003,
    "gpt-5.4-pro":                  0.015,
    "gpt-5.4+reasoning=minimal":    0.004,
    "gpt-5.4+reasoning=low":        0.010,
    "gpt-5.4+reasoning=medium":     0.020,
    "gpt-5.4+reasoning=high":       0.040,
    "gpt-5.4+reasoning=xhigh":      0.080,
}


# Conservative upper-bound fallback for unrecognized model strings.  Chosen so
# the cost gate errs on the side of blocking rather than allowing unexpected
# spend — the measured high end (gpt-5.4 with reasoning=high) is ~$0.037/call,
# so 0.04 covers it plus a margin.
_UNKNOWN_MODEL_FALLBACK_USD = 0.04


class OpenAIBackend(VLMBackend):
    def __init__(
        self,
        model: str = "gpt-5.4-mini",
        api_key: str | None = None,
        max_tokens: int = 1024,
        reasoning_effort: Optional[str] = None,
    ) -> None:
        """Configure the OpenAI Responses-API backend.

        ``reasoning_effort``, when set to ``"minimal"``/``"low"``/``"medium"``/
        ``"high"``/``"xhigh"``, is forwarded as ``reasoning={"effort": ...}``
        on each :meth:`detect_tile` call.  Reasoning variants cost ~10–100×
        more than the base model; :attr:`approx_cost_per_tile_usd` accounts for
        this via a compound lookup key.
        """
        self._model = model
        self._max_tokens = max_tokens
        self._reasoning_effort = reasoning_effort
        self._client = OpenAI(api_key=api_key) if api_key else OpenAI()

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def approx_cost_per_tile_usd(self) -> float:
        key = (
            f"{self._model}+reasoning={self._reasoning_effort}"
            if self._reasoning_effort
            else self._model
        )
        if key in _MODEL_COST_PER_TILE_USD:
            return _MODEL_COST_PER_TILE_USD[key]
        _log.warning(
            "OpenAIBackend: model %r not in cost lookup table; using conservative "
            "fallback $%.4f/tile.  Cost gate may over-block; update "
            "_MODEL_COST_PER_TILE_USD when measured pricing is available.",
            key, _UNKNOWN_MODEL_FALLBACK_USD,
        )
        return _UNKNOWN_MODEL_FALLBACK_USD

    def detect_tile(self, tile: np.ndarray, prompt: str) -> VLMResponse:
        img_b64 = _encode_tile_as_png_b64(tile)
        start = time.perf_counter()
        kwargs: dict = dict(
            model=self._model,
            max_output_tokens=self._max_tokens,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{img_b64}",
                        },
                        {"type": "input_text", "text": prompt},
                    ],
                }
            ],
        )
        if self._reasoning_effort:
            kwargs["reasoning"] = {"effort": self._reasoning_effort}
        response = self._client.responses.create(**kwargs)
        wall = time.perf_counter() - start
        raw_text = _extract_output_text(response)
        detections = parse_vlm_response(raw_text)
        for det in detections:
            det.raw_response = raw_text
        return VLMResponse(
            detections=detections,
            tokens_used={
                "input": int(response.usage.input_tokens),
                "output": int(response.usage.output_tokens),
            },
            wall_time_seconds=wall,
        )


def _extract_output_text(response) -> str:
    """Join all output_text blocks across message items in a Responses API result."""
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text:
        return text
    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        for block in getattr(item, "content", []) or []:
            block_text = getattr(block, "text", None)
            if isinstance(block_text, str):
                parts.append(block_text)
    return "".join(parts)


def _encode_tile_as_png_b64(tile: np.ndarray) -> str:
    arr = np.asarray(tile)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    if arr.ndim == 2:
        img = Image.fromarray(arr, mode="L")
    elif arr.ndim == 3 and arr.shape[2] == 3:
        img = Image.fromarray(arr, mode="RGB")
    else:
        raise ValueError(f"unsupported tile shape {arr.shape}; expected 2D or HxWx3")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")
