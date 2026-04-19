"""Per-scene detector configuration loader.

CA-CFAR's alpha threshold is not scene-invariant.  Clutter distributions
differ between scene types (e.g., reclamation-dominated vs open-water vessel
flotillas), so a single global alpha can't serve both.  Rather than tuning
at call sites, per-scene overrides live in ``config/sar_detection_params.json``
and are resolved here.

Lookup order: ``scenes[scene_id]`` → ``default``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "sar_detection_params.json"


def load_sar_detection_params(config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or DEFAULT_CONFIG_PATH
    return json.loads(Path(path).read_text())


def get_alpha(source: str, scene_id: str, *, config_path: Path | None = None) -> float:
    params = load_sar_detection_params(config_path)
    if source not in params:
        raise KeyError(
            f"No SAR detection params configured for source '{source}'. "
            f"Known sources: {sorted(params.keys())}"
        )
    source_params = params[source]
    scenes = source_params.get("scenes", {})
    if scene_id in scenes:
        return float(scenes[scene_id]["alpha"])
    return float(source_params["default"]["alpha"])
