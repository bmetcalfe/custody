"""VLM detection backends (ADR-0015).

Public surface:
- :class:`VLMBackend` — abstract backend interface
- :class:`VLMDetection` — raw per-target VLM output
- :class:`VLMResponse` — parsed VLM-call result
- :class:`AnthropicBackend`, :class:`OpenAIBackend` — concrete API backends
- :class:`QwenBackend`, :class:`MoondreamBackend` — stubs (NotImplementedError)
"""
from custody.detection.vlm_backends.anthropic_backend import AnthropicBackend
from custody.detection.vlm_backends.base import (
    VLMBackend,
    VLMDetection,
    VLMResponse,
)
from custody.detection.vlm_backends.moondream_backend import MoondreamBackend
from custody.detection.vlm_backends.openai_backend import OpenAIBackend
from custody.detection.vlm_backends.qwen_backend import QwenBackend

__all__ = [
    "AnthropicBackend",
    "MoondreamBackend",
    "OpenAIBackend",
    "QwenBackend",
    "VLMBackend",
    "VLMDetection",
    "VLMResponse",
]
