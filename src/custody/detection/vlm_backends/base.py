"""Abstract interface and data types for VLM detection backends.

Per ADR-0015, Custody's detection layer dispatches tile-level inference to a
VLM backend.  Concrete backends (Anthropic, OpenAI, future local models) implement
this interface; the rest of the pipeline stays backend-agnostic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class VLMDetection:
    """Raw detection output from a VLM backend, pre-fusion-layer conversion.

    bbox is in tile-local pixel coordinates with origin at the tile's top-left.
    """

    bbox: tuple[int, int, int, int]
    confidence: str
    reasoning: str
    raw_response: Optional[str] = None


@dataclass
class VLMResponse:
    """Full parsed response from a single VLM call."""

    detections: list[VLMDetection] = field(default_factory=list)
    scene_description: Optional[str] = None
    tokens_used: Optional[dict] = None
    wall_time_seconds: Optional[float] = None


class VLMBackend(ABC):
    """Pluggable backend for VLM-based SAR tile detection."""

    @abstractmethod
    def detect_tile(self, tile: np.ndarray, prompt: str) -> VLMResponse:
        """Send one tile to the VLM, return parsed detections + metadata."""
        raise NotImplementedError

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Stable identifier for logging and provenance (stored on observations)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def approx_cost_per_tile_usd(self) -> float:
        """Rough per-tile USD cost, for budgeting and scene-cost estimation."""
        raise NotImplementedError
