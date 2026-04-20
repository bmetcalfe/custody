"""Stub: Moondream backend for local VLM inference.

Not implemented in the current milestone; placeholder to document the
pluggable-backend pattern per ADR-0015.  Concrete implementation deferred.
"""
from __future__ import annotations

import numpy as np

from .base import VLMBackend, VLMResponse


class MoondreamBackend(VLMBackend):
    def __init__(self, *args, **kwargs) -> None:
        self._args = args
        self._kwargs = kwargs

    def detect_tile(self, tile: np.ndarray, prompt: str) -> VLMResponse:
        raise NotImplementedError(
            "MoondreamBackend.detect_tile is not implemented yet. "
            "See ADR-0015 for the pluggable-backend design; local-model backends are deferred."
        )

    @property
    def model_name(self) -> str:
        raise NotImplementedError("MoondreamBackend is a stub")

    @property
    def approx_cost_per_tile_usd(self) -> float:
        raise NotImplementedError("MoondreamBackend is a stub")
