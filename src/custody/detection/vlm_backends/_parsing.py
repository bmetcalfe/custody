"""Tolerant parser for VLM JSON responses.

Shared across VLM backends (Anthropic, OpenAI, future).  VLM responses
vary in exact structure across models and prompt variants; the parser
tolerates:

- Plain JSON object or array
- JSON wrapped in ```json ... ``` or ``` ... ``` markdown fences
- JSON with explanatory preamble or trailing prose
- Various detection-list keys: ``vessels``, ``detections``, ``targets``, ``results``
- Bounding box as ``[x1, y1, x2, y2]`` list or ``{"x1": n, "y1": n, "x2": n, "y2": n}`` dict
- Confidence strings in mixed case

On any parse failure the parser returns an empty list — it never raises.
Callers get "no detections" rather than a crash on malformed model output.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

from custody.detection.vlm_backends.base import VLMDetection


_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)\s*```", re.DOTALL)
_DETECTION_LIST_KEYS = ("vessels", "detections", "targets", "results", "objects")
_BBOX_DICT_KEYS = ("bounding_box", "bbox_dict", "box")
_REASONING_KEYS = ("reasoning", "rationale", "explanation", "notes", "why")
_CONFIDENCE_ALIASES = {"high", "medium", "med", "low"}


def parse_vlm_response(raw_text: str) -> list[VLMDetection]:
    """Parse a VLM response string into a list of :class:`VLMDetection`."""
    if not isinstance(raw_text, str) or not raw_text.strip():
        return []
    # Try direct JSON first, then markdown-fenced, then preamble-stripped.
    data = _try_parse_json(raw_text.strip())
    if data is None:
        stripped = _strip_markdown_fences(raw_text)
        data = _try_parse_json(stripped)
    if data is None:
        data = _extract_first_json_block(raw_text)
    if data is None:
        return []
    items = _extract_detection_list(data)
    detections: list[VLMDetection] = []
    for item in items:
        det = _item_to_detection(item)
        if det is not None:
            detections.append(det)
    return detections


def _try_parse_json(text: str) -> Optional[Any]:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def _strip_markdown_fences(text: str) -> str:
    m = _FENCE_RE.search(text)
    return m.group(1).strip() if m else text.strip()


def _extract_first_json_block(text: str) -> Optional[Any]:
    """Scan for first balanced {...} or [...] and try to parse it."""
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        while start != -1:
            depth = 0
            in_str = False
            esc = False
            for i in range(start, len(text)):
                ch = text[i]
                if esc:
                    esc = False
                    continue
                if ch == "\\":
                    esc = True
                    continue
                if ch == '"':
                    in_str = not in_str
                    continue
                if in_str:
                    continue
                if ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : i + 1]
                        parsed = _try_parse_json(candidate)
                        if parsed is not None:
                            return parsed
                        break
            start = text.find(opener, start + 1)
    return None


_MAX_RECURSION_DEPTH = 10


def _extract_detection_list(data: Any) -> list[Any]:
    """Find the detection list, searching nested dict wrappers (BFS, shallowest wins).

    Some VLMs occasionally wrap detections inside a top-level key such as
    ``{"analysis": {"vessels": [...]}}``.  We search breadth-first so the
    *shallowest* matching key wins — this avoids accidentally matching a
    deeply-nested ``examples``/``notes`` list before reaching the real
    detections.  Recursion is bounded at ``_MAX_RECURSION_DEPTH`` levels to
    guard against pathological inputs.
    """
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return []

    frontier: list[dict] = [data]
    for _depth in range(_MAX_RECURSION_DEPTH + 1):
        # Check every node at this depth first — shallowest match wins.
        for node in frontier:
            for key in _DETECTION_LIST_KEYS:
                value = node.get(key)
                if isinstance(value, list):
                    return value
        # Expand to next depth.
        next_frontier: list[dict] = []
        for node in frontier:
            for v in node.values():
                if isinstance(v, dict):
                    next_frontier.append(v)
        if not next_frontier:
            break
        frontier = next_frontier
    return []


def _item_to_detection(item: Any) -> Optional[VLMDetection]:
    if not isinstance(item, dict):
        return None
    bbox = _extract_bbox(item)
    if bbox is None:
        return None
    try:
        bbox_tuple = tuple(int(round(float(v))) for v in bbox)
    except (TypeError, ValueError):
        return None
    if len(bbox_tuple) != 4:
        return None
    conf = str(item.get("confidence", "medium")).strip().lower()
    if conf == "med":
        conf = "medium"
    if conf not in _CONFIDENCE_ALIASES - {"med"}:
        conf = "medium"
    reasoning = ""
    for key in _REASONING_KEYS:
        v = item.get(key)
        if isinstance(v, str) and v.strip():
            reasoning = v.strip()
            break
    return VLMDetection(
        bbox=bbox_tuple,  # type: ignore[arg-type]
        confidence=conf,
        reasoning=reasoning,
    )


def _extract_bbox(item: dict) -> Optional[list]:
    v = item.get("bbox")
    if isinstance(v, list) and len(v) == 4:
        return v
    if isinstance(v, dict) and all(k in v for k in ("x1", "y1", "x2", "y2")):
        return [v["x1"], v["y1"], v["x2"], v["y2"]]
    for key in _BBOX_DICT_KEYS:
        v = item.get(key)
        if isinstance(v, dict) and all(k in v for k in ("x1", "y1", "x2", "y2")):
            return [v["x1"], v["y1"], v["x2"], v["y2"]]
        if isinstance(v, list) and len(v) == 4:
            return v
    return None
