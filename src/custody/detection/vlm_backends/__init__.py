"""VLM detection backends (ADR-0015).

Public surface:
- :class:`VLMBackend` — abstract backend interface
- :class:`VLMDetection` — raw per-target VLM output
- :class:`VLMResponse` — parsed VLM-call result
- :class:`QwenBackend`, :class:`MoondreamBackend` — stubs (NotImplementedError)

Concrete API backends (AnthropicBackend, OpenAIBackend) land in subsequent
phases and will be re-exported here.
"""
from custody.detection.vlm_backends.anthropic_backend import AnthropicBackend
from custody.detection.vlm_backends.base import (
    VLMBackend,
    VLMDetection,
    VLMResponse,
)
from custody.detection.vlm_backends.moondream_backend import MoondreamBackend
from custody.detection.vlm_backends.qwen_backend import QwenBackend

__all__ = [
    "AnthropicBackend",
    "MoondreamBackend",
    "QwenBackend",
    "VLMBackend",
    "VLMDetection",
    "VLMResponse",
]
