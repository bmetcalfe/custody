"""Tests for :mod:`custody.hypotheses.evidence` (ADR-0021, Slice 2).

The adapters are a thin bridge only: they take an existing repo source
object, a scenario_id, caller-supplied supports/contradicts/confidence/
weight/reason, and return HypothesisEvidence.  Scenario-specific "which
hypothesis does this artifact support?" logic is explicitly out of scope
and belongs to Slice 3 scenario generators.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from custody.fusion.observations import (
    PositionObservation,
    PositionVelocityObservation,
)
from custody.fusion.scenes import Scene
from custody.fusion.temporal import Match
from custody.hypotheses.evidence import (
    from_mapping,
    from_match,
    from_observation,
    from_scene,
)
from custody.hypotheses.registry import (
    SCENARIO_TENNENT,
    SCENARIO_WHITSUN,
    TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
    TENNENT_NO_MEANINGFUL_ACTIVITY,
    WHITSUN_VESSEL_CLUSTER_ACTIVITY,
)
from custody.hypotheses.types import HypothesisEvidence
from custody.hypotheses.update import update_state


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


T_EPOCH = 1_688_306_455.344  # 2023-07-02T20:00:55.344+00:00
T_EPOCH_DT = datetime.fromtimestamp(T_EPOCH, tz=timezone.utc)


def _mk_pos_obs(
    obs_id: str = "obs-1",
    *,
    classification_conf: float | None = 0.72,
) -> PositionObservation:
    return PositionObservation(
        obs_id=obs_id,
        source_id="test-source",
        modality="SAR",
        acquisition_time=T_EPOCH,
        ingestion_time=T_EPOCH + 1.0,
        lat=8.856,
        lon=114.665,
        cov_pos=np.eye(2) * 400.0,
        raw_ref="test://raw",
        classification_conf=classification_conf,
    )


def _mk_pv_obs(obs_id: str = "obs-ais-1") -> PositionVelocityObservation:
    return PositionVelocityObservation(
        obs_id=obs_id,
        source_id="test-ais",
        modality="AIS",
        acquisition_time=T_EPOCH,
        ingestion_time=T_EPOCH + 1.0,
        lat=8.856,
        lon=114.665,
        v_n=0.5,
        v_e=-0.2,
        cov=np.eye(4) * 10.0,
        raw_ref="test://ais",
    )


def _mk_scene(
    quality_flag: str = "green",
    *,
    scene_id: str = "tennent_20230702_umbra-05",
) -> Scene:
    return Scene(
        scene_id=scene_id,
        sensor="umbra-05",
        acquisition_time=T_EPOCH,
        pixel_size_m=0.342,
        center_lat=8.856,
        center_lon=114.665,
        footprint_latlon=(
            (8.83, 114.64), (8.83, 114.69),
            (8.88, 114.69), (8.88, 114.64),
        ),
        raw_scene_path=None,
        quality_flag=quality_flag,  # type: ignore[arg-type]
    )


def _mk_match() -> Match:
    return Match(obs_a_id="obs-a-42", obs_b_id="obs-b-17", distance_m=12.5)


# ---------------------------------------------------------------------------
# from_observation
# ---------------------------------------------------------------------------


def test_from_observation_source_ref_contains_obs_id() -> None:
    obs = _mk_pos_obs(obs_id="obs-042")
    ev = from_observation(
        obs,
        scenario_id=SCENARIO_TENNENT,
        supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,),
        reason="bright scatterer",
    )
    assert isinstance(ev, HypothesisEvidence)
    assert ev.source_ref == "observation:obs-042"
    assert ev.source_kind == "observation"
    assert ev.evidence_id == "obs-042"


def test_from_observation_coerces_epoch_float_to_utc_datetime() -> None:
    obs = _mk_pos_obs()
    ev = from_observation(
        obs,
        scenario_id=SCENARIO_TENNENT,
        supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,),
        reason="r",
    )
    assert ev.timestamp == T_EPOCH_DT
    assert ev.timestamp.tzinfo is not None


def test_from_observation_uses_classification_conf_when_override_absent() -> None:
    obs = _mk_pos_obs(classification_conf=0.72)
    ev = from_observation(
        obs,
        scenario_id=SCENARIO_TENNENT,
        supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,),
        reason="r",
    )
    assert ev.confidence == pytest.approx(0.72)


def test_from_observation_falls_back_to_one_when_classification_conf_is_none() -> None:
    obs = _mk_pos_obs(classification_conf=None)
    ev = from_observation(
        obs,
        scenario_id=SCENARIO_TENNENT,
        supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,),
        reason="r",
    )
    assert ev.confidence == 1.0


def test_from_observation_position_velocity_defaults_confidence_to_one() -> None:
    obs = _mk_pv_obs()
    ev = from_observation(
        obs,
        scenario_id=SCENARIO_TENNENT,
        supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,),
        reason="r",
    )
    assert ev.confidence == 1.0
    assert ev.source_ref == "observation:obs-ais-1"


# ---------------------------------------------------------------------------
# from_scene
# ---------------------------------------------------------------------------


def test_from_scene_source_ref_and_timestamp() -> None:
    scene = _mk_scene()
    ev = from_scene(
        scene,
        scenario_id=SCENARIO_TENNENT,
        supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,),
        reason="scene shows persistent return",
    )
    assert ev.source_ref == "scene:tennent_20230702_umbra-05"
    assert ev.source_kind == "scene"
    assert ev.evidence_id == "tennent_20230702_umbra-05"
    assert ev.timestamp == T_EPOCH_DT


def test_from_scene_does_not_translate_quality_flag_to_confidence() -> None:
    """Adapter layer must not encode quality_flag → confidence; that's Slice 3."""
    for qf in ("green", "yellow", "red"):
        scene = _mk_scene(quality_flag=qf)
        ev = from_scene(
            scene,
            scenario_id=SCENARIO_TENNENT,
            supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,),
            reason="r",
        )
        assert ev.confidence == 1.0, (
            f"quality_flag={qf!r} must not affect confidence at the adapter layer"
        )


# ---------------------------------------------------------------------------
# from_match
# ---------------------------------------------------------------------------


def test_from_match_source_ref_contains_both_obs_ids_and_no_timestamp() -> None:
    m = _mk_match()
    ev = from_match(
        m,
        scenario_id=SCENARIO_TENNENT,
        supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,),
        reason="persistent across scenes",
    )
    assert ev.source_ref == "matcher_result:obs-a-42->obs-b-17"
    assert ev.source_kind == "matcher_result"
    assert ev.evidence_id == "match-obs-a-42-obs-b-17"
    assert ev.timestamp is None
    assert ev.confidence == 1.0


# ---------------------------------------------------------------------------
# from_mapping
# ---------------------------------------------------------------------------


def test_from_mapping_requires_source_kind() -> None:
    with pytest.raises(TypeError):
        from_mapping(
            {"id": "x"},
            scenario_id=SCENARIO_TENNENT,  # type: ignore[call-arg]
            supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,),
            reason="r",
        )


def test_from_mapping_uses_fallback_order_for_id_timestamp_confidence() -> None:
    m = {
        "event_id": "gfw-q-7",
        "acquisition_time": T_EPOCH,
        "score": 0.42,
        "other": "ignored",
    }
    ev = from_mapping(
        m,
        source_kind="gfw_presence",
        scenario_id=SCENARIO_TENNENT,
        supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,),
        reason="r",
    )
    assert ev.source_kind == "gfw_presence"
    assert ev.evidence_id == "gfw-q-7"
    assert ev.source_ref == "gfw_presence:gfw-q-7"
    assert ev.timestamp == T_EPOCH_DT
    assert ev.confidence == pytest.approx(0.42)


def test_from_mapping_coerces_float_epoch_acquisition_time() -> None:
    m = {"id": "vlm-1", "acquisition_time": T_EPOCH}
    ev = from_mapping(
        m,
        source_kind="vlm_detection",
        scenario_id=SCENARIO_TENNENT,
        supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,),
        reason="r",
    )
    assert ev.timestamp == T_EPOCH_DT


def test_from_mapping_without_known_keys_still_produces_evidence() -> None:
    """No id/timestamp/confidence keys present → stable evidence_id, None timestamp, confidence=1.0."""
    m1 = {"payload": "a"}
    m2 = {"payload": "a"}
    m3 = {"payload": "b"}
    ev1 = from_mapping(
        m1, source_kind="scratch", scenario_id=SCENARIO_TENNENT,
        supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,), reason="r",
    )
    ev2 = from_mapping(
        m2, source_kind="scratch", scenario_id=SCENARIO_TENNENT,
        supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,), reason="r",
    )
    ev3 = from_mapping(
        m3, source_kind="scratch", scenario_id=SCENARIO_TENNENT,
        supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,), reason="r",
    )
    assert ev1.timestamp is None
    assert ev1.confidence == 1.0
    assert ev1.evidence_id.startswith("scratch-")
    # Stable for identical mappings, different for different content.
    assert ev1.evidence_id == ev2.evidence_id
    assert ev1.evidence_id != ev3.evidence_id


# ---------------------------------------------------------------------------
# Scenario + hypothesis_id validation
# ---------------------------------------------------------------------------


def test_unknown_scenario_raises_value_error_for_all_adapters() -> None:
    obs = _mk_pos_obs()
    scene = _mk_scene()
    match = _mk_match()
    m = {"id": "x"}

    for call in (
        lambda: from_observation(
            obs, scenario_id="atlantis",
            supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,), reason="r",
        ),
        lambda: from_scene(
            scene, scenario_id="atlantis",
            supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,), reason="r",
        ),
        lambda: from_match(
            match, scenario_id="atlantis",
            supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,), reason="r",
        ),
        lambda: from_mapping(
            m, source_kind="custom", scenario_id="atlantis",
            supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,), reason="r",
        ),
    ):
        with pytest.raises(ValueError):
            call()


def test_invalid_supports_id_raises_and_lists_valid_ids() -> None:
    obs = _mk_pos_obs()
    with pytest.raises(ValueError) as excinfo:
        from_observation(
            obs,
            scenario_id=SCENARIO_TENNENT,
            supports=("totally_not_a_hypothesis",),
            reason="r",
        )
    msg = str(excinfo.value)
    assert "totally_not_a_hypothesis" in msg
    assert TENNENT_FIXED_RECLAMATION_OR_STRUCTURE in msg


def test_invalid_contradicts_id_raises_and_lists_valid_ids() -> None:
    obs = _mk_pos_obs()
    with pytest.raises(ValueError) as excinfo:
        from_observation(
            obs,
            scenario_id=SCENARIO_TENNENT,
            supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,),
            contradicts=("nope",),
            reason="r",
        )
    msg = str(excinfo.value)
    assert "nope" in msg
    assert TENNENT_NO_MEANINGFUL_ACTIVITY in msg


def test_cross_scenario_hypothesis_id_rejected() -> None:
    """A Whitsun hypothesis_id used on a Tennent evidence raises."""
    obs = _mk_pos_obs()
    with pytest.raises(ValueError):
        from_observation(
            obs,
            scenario_id=SCENARIO_TENNENT,
            supports=(WHITSUN_VESSEL_CLUSTER_ACTIVITY,),
            reason="r",
        )


# ---------------------------------------------------------------------------
# Passthrough / override behaviour
# ---------------------------------------------------------------------------


def test_source_object_is_not_mutated() -> None:
    obs = _mk_pos_obs()
    before = (obs.obs_id, obs.classification_conf, obs.acquisition_time)
    from_observation(
        obs,
        scenario_id=SCENARIO_TENNENT,
        supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,),
        contradicts=(TENNENT_NO_MEANINGFUL_ACTIVITY,),
        reason="r",
    )
    after = (obs.obs_id, obs.classification_conf, obs.acquisition_time)
    assert before == after


def test_supports_contradicts_reason_pass_through() -> None:
    obs = _mk_pos_obs()
    ev = from_observation(
        obs,
        scenario_id=SCENARIO_TENNENT,
        supports=[TENNENT_FIXED_RECLAMATION_OR_STRUCTURE],  # list accepted
        contradicts=[TENNENT_NO_MEANINGFUL_ACTIVITY],
        reason="exact-reason-text",
    )
    assert ev.supports == (TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,)
    assert ev.contradicts == (TENNENT_NO_MEANINGFUL_ACTIVITY,)
    assert ev.reason == "exact-reason-text"


def test_confidence_override_wins_over_object_derived() -> None:
    obs = _mk_pos_obs(classification_conf=0.72)
    ev = from_observation(
        obs,
        scenario_id=SCENARIO_TENNENT,
        supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,),
        confidence=0.33,
        reason="r",
    )
    assert ev.confidence == pytest.approx(0.33)


def test_weight_defaults_to_one_and_passes_through() -> None:
    obs = _mk_pos_obs()
    ev_default = from_observation(
        obs, scenario_id=SCENARIO_TENNENT,
        supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,), reason="r",
    )
    assert ev_default.weight == 1.0

    ev_override = from_observation(
        obs, scenario_id=SCENARIO_TENNENT,
        supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,),
        weight=2.5, reason="r",
    )
    assert ev_override.weight == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# Import-boundary sanity
# ---------------------------------------------------------------------------


def test_evidence_module_does_not_import_vlm_or_gfw_runtimes() -> None:
    """Heavy deps must not leak into the hypothesis adapter layer.

    AST-level scan of evidence.py — checks that no ``import`` or
    ``from ... import`` statement references ``custody.detection.*`` or
    ``custody.ingest.gfw_presence``.  A static text scan would false-positive
    on docstrings that explain why those imports are deliberately absent.
    """
    import ast
    import custody.hypotheses.evidence as mod

    src_path = Path(mod.__file__)
    tree = ast.parse(src_path.read_text(encoding="utf-8"))

    forbidden_prefixes = ("custody.detection", "custody.ingest.gfw_presence")
    offending: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if any(alias.name.startswith(p) for p in forbidden_prefixes):
                    offending.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod_name = node.module or ""
            if any(mod_name.startswith(p) for p in forbidden_prefixes):
                offending.append(mod_name)

    assert not offending, (
        f"{src_path.name} imports forbidden modules: {offending}; "
        "heavy deps belong outside the hypothesis layer"
    )


# ---------------------------------------------------------------------------
# Round-trip through update_state
# ---------------------------------------------------------------------------


def test_round_trip_through_update_state() -> None:
    obs = _mk_pos_obs()
    scene = _mk_scene()
    match = _mk_match()
    raw = {"id": "vlm-7", "acquisition_time": T_EPOCH, "score": 0.5}

    evs = (
        from_observation(
            obs, scenario_id=SCENARIO_TENNENT,
            supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,),
            reason="bright scatterer",
        ),
        from_scene(
            scene, scenario_id=SCENARIO_TENNENT,
            supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,),
            reason="persistent scene return",
        ),
        from_match(
            match, scenario_id=SCENARIO_TENNENT,
            supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,),
            reason="matched pair",
        ),
        from_mapping(
            raw, source_kind="vlm_detection", scenario_id=SCENARIO_TENNENT,
            supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,),
            reason="vlm annotation",
        ),
    )
    state = update_state(SCENARIO_TENNENT, evidence=evs)
    assert state.top_hypothesis == TENNENT_FIXED_RECLAMATION_OR_STRUCTURE
    assert len(state.explanation) == 4
