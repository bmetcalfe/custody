"""Anthropic (Claude) VLM backend (ADR-0015).

Uses the Anthropic Python SDK.  Default model is ``claude-sonnet-4-6``, chosen
from the Week 2 cross-model spike on Whitsun tile 1 which showed Sonnet 4.6
producing strong SAR-aware detection reasoning at ~$0.01/tile.
"""
from __future__ import annotations

import base64
import time
from io import BytesIO

import numpy as np
from anthropic import Anthropic
from PIL import Image

from custody.detection.vlm_backends._parsing import parse_vlm_response
from custody.detection.vlm_backends.base import VLMBackend, VLMResponse


class AnthropicBackend(VLMBackend):
    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: str | None = None,
        max_tokens: int = 1024,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._client = Anthropic(api_key=api_key) if api_key else Anthropic()

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def approx_cost_per_tile_usd(self) -> float:
        return 0.01

    def detect_tile(self, tile: np.ndarray, prompt: str) -> VLMResponse:
        img_b64 = _encode_tile_as_png_b64(tile)
        start = time.perf_counter()
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": img_b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        wall = time.perf_counter() - start
        raw_text = "".join(
            b.text for b in response.content if getattr(b, "type", None) == "text"
        )
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
