"""Tests for per-scene SAR detection config loader."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from custody.detection.config import get_alpha, load_sar_detection_params


def test_config_loads_without_error():
    params = load_sar_detection_params()
    assert "umbra" in params
    assert "default" in params["umbra"]
    assert "alpha" in params["umbra"]["default"]


def test_scene_specific_alpha_overrides_default(tmp_path: Path):
    cfg = {
        "umbra": {
            "default": {"alpha": 7.0},
            "scenes": {"2023-12-06-02-06-24_UMBRA-04": {"alpha": 4.0}},
        }
    }
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))
    assert get_alpha("umbra", "2023-12-06-02-06-24_UMBRA-04", config_path=p) == 4.0


def test_missing_scene_falls_back_to_default(tmp_path: Path):
    cfg = {"umbra": {"default": {"alpha": 7.0}, "scenes": {}}}
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))
    assert get_alpha("umbra", "some-unknown-scene", config_path=p) == 7.0


def test_missing_source_raises_keyerror(tmp_path: Path):
    cfg = {"umbra": {"default": {"alpha": 7.0}, "scenes": {}}}
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))
    with pytest.raises(KeyError, match="iceye"):
        get_alpha("iceye", "any-scene", config_path=p)


def test_production_config_has_tennent_default_and_whitsun_override():
    # Tennent scene uses the default alpha (no scene-specific entry)
    assert get_alpha("umbra", "2023-07-02-14-00-55_UMBRA-05") == 7.0
    # Whitsun scene 1 has its own override
    assert get_alpha("umbra", "2023-12-06-02-06-24_UMBRA-04") == 4.0
