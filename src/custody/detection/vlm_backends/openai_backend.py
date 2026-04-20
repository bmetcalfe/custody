"""OpenAI (GPT) VLM backend (ADR-0015).

Uses the OpenAI Python SDK's Responses API, which is the current recommended
pattern for multimodal input in openai>=2.0.  Default model is ``gpt-5.4-mini``
for the Week 2 cross-model spike — low cost per tile, vision-capable.
"""
from __future__ import annotations

import base64
import time
from io import BytesIO

import numpy as np
from openai import OpenAI
from PIL import Image

from custody.detection.vlm_backends._parsing import parse_vlm_response
from custody.detection.vlm_backends.base import VLMBackend, VLMResponse


class OpenAIBackend(VLMBackend):
    def __init__(
        self,
        model: str = "gpt-5.4-mini",
        api_key: str | None = None,
        max_tokens: int = 1024,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._client = OpenAI(api_key=api_key) if api_key else OpenAI()

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def approx_cost_per_tile_usd(self) -> float:
        return 0.005

    def detect_tile(self, tile: np.ndarray, prompt: str) -> VLMResponse:
        img_b64 = _encode_tile_as_png_b64(tile)
        start = time.perf_counter()
        response = self._client.responses.create(
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
