"""Tests for the Whitsun decision trace fixture.

Validates the canonical custody / reacquisition demo trace at
``data/demo/whitsun_decision_trace.fixture.json``.  No network access;
no dashboard code; no decision-layer runtime is exercised.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
TRACE_PATH = REPO_ROOT / "data" / "demo" / "whitsun_decision_trace.fixture.json"
SENTINEL_FIXTURE = (
    REPO_ROOT / "data" / "demo" / "whitsun_sentinel_observations.fixture.json"
)


@pytest.fixture(scope="module")
def trace() -> dict:
    return json.loads(TRACE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def sentinel_fixture() -> dict:
    return json.loads(SENTINEL_FIXTURE.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Loadability and schema marker
# ---------------------------------------------------------------------------


def test_trace_loads(trace) -> None:
    assert isinstance(trace, dict)
    assert trace["schema"] == "custody.demo.whitsun_decision_trace.v1"


def test_scenario_metadata_present(trace) -> None:
    meta = trace["scenario_metadata"]
    assert meta["scenario_id"] == "whitsun-custody-reacquisition-2023-12"
    assert meta["data_mode"] == "fixture"
    assert (
        meta["sentinel_observations_fixture_path"]
        == "data/demo/whitsun_sentinel_observations.fixture.json"
    )


# ---------------------------------------------------------------------------
# Event sequence (14 events, contiguous ordinals)
# ---------------------------------------------------------------------------


def test_event_count_and_ordering(trace) -> None:
    events = trace["events"]
    assert len(events) == 14
    ordinals = [e["ordinal"] for e in events]
    assert ordinals == list(range(1, 15))
    ids = [e["event_id"] for e in events]
    assert ids == [f"ev-{i:02d}" for i in range(1, 15)]


def test_each_event_has_required_fields(trace) -> None:
    required = {"event_id", "ordinal", "label", "timestamp", "kind", "summary", "data_mode", "refs"}
    for e in trace["events"]:
        assert required.issubset(e.keys()), f"missing fields on {e.get('event_id')}"


# ---------------------------------------------------------------------------
# Reference resolution: every events[*].refs.* ID exists in its section
# ---------------------------------------------------------------------------


def _all_ids(items, key):
    return {item[key] for item in items}


def _first_ids(items, key):
    return [item[key] for item in items]


def test_event_refs_resolve(trace, sentinel_fixture) -> None:
    obs_ids = _all_ids(trace["observations"], "observation_id")
    art_ids = _all_ids(trace["evidence_artifacts"], "evidence_artifact_id")
    det_ids = _all_ids(trace["detections"], "detection_id")
    track_ids = _all_ids(trace["candidate_tracks"], "track_id")
    custody_ids = _all_ids(trace["custody_state_snapshots"], "custody_state_id")
    option_ids = _all_ids(trace["candidate_tasking_options"], "tasking_option_id")
    score_ids = _all_ids(trace["score_breakdowns"], "score_breakdown_id")
    cf_ids = _all_ids(trace["counterfactuals"], "counterfactual_id")
    sentinel_obs_ids = {
        o["observation_id"] for o in sentinel_fixture["observations"]
    }

    selected_id = trace["selected_recommendation"]["selected_recommendation_id"]
    policy_id = trace["policy_rationale"]["policy_rationale_id"]
    human_id = trace["human_action"]["human_action_id"]
    outcome_id = trace["outcome"]["outcome_id"]
    followup_id = trace["followup_recommendation"]["followup_recommendation_id"]

    for ev in trace["events"]:
        refs = ev["refs"]
        # Single-id refs.
        if "observation_id" in refs:
            oid = refs["observation_id"]
            assert oid in obs_ids or oid in sentinel_obs_ids, (
                f"event {ev['event_id']} references unknown observation {oid}"
            )
        if "track_id" in refs:
            assert refs["track_id"] in track_ids, ev["event_id"]
        if "custody_state_id" in refs:
            assert refs["custody_state_id"] in custody_ids, ev["event_id"]
        if "selected_recommendation_id" in refs:
            assert refs["selected_recommendation_id"] == selected_id, ev["event_id"]
        if "policy_rationale_id" in refs:
            assert refs["policy_rationale_id"] == policy_id, ev["event_id"]
        if "human_action_id" in refs:
            assert refs["human_action_id"] == human_id, ev["event_id"]
        if "outcome_id" in refs:
            assert refs["outcome_id"] == outcome_id, ev["event_id"]
        if "followup_recommendation_id" in refs:
            assert refs["followup_recommendation_id"] == followup_id, ev["event_id"]
        # List refs.
        for tid in refs.get("track_ids", []):
            assert tid in track_ids, f"{ev['event_id']} -> {tid}"
        for did in refs.get("detection_ids", []):
            assert did in det_ids, f"{ev['event_id']} -> {did}"
        for aid in refs.get("evidence_artifact_ids", []):
            assert aid in art_ids, f"{ev['event_id']} -> {aid}"
        for oid in refs.get("tasking_option_ids", []):
            assert oid in option_ids, f"{ev['event_id']} -> {oid}"
        for sid in refs.get("score_breakdown_ids", []):
            assert sid in score_ids, f"{ev['event_id']} -> {sid}"
        for cid in refs.get("counterfactual_ids", []):
            assert cid in cf_ids, f"{ev['event_id']} -> {cid}"


# ---------------------------------------------------------------------------
# Sentinel observations are present and cross-referenced
# ---------------------------------------------------------------------------


def test_trace_includes_sentinel_observations(trace) -> None:
    sources = {o["source"] for o in trace["observations"]}
    assert "sentinel-1" in sources
    assert "sentinel-2" in sources


def test_sentinel_observation_ids_resolve_into_external_fixture(trace, sentinel_fixture) -> None:
    sentinel_ids_in_fixture = {
        o["observation_id"] for o in sentinel_fixture["observations"]
    }
    referenced = [
        o["observation_id"] for o in trace["observations"]
        if o["source"] in ("sentinel-1", "sentinel-2")
    ]
    assert referenced, "trace must reference at least one Sentinel observation"
    for oid in referenced:
        assert oid in sentinel_ids_in_fixture, (
            f"Sentinel observation {oid} not found in "
            f"data/demo/whitsun_sentinel_observations.fixture.json"
        )


# ---------------------------------------------------------------------------
# Operational hierarchy: Sentinel != Umbra
# ---------------------------------------------------------------------------


def test_sentinel_observations_not_treated_as_equivalent_to_umbra(trace) -> None:
    umbra = [o for o in trace["observations"] if o["source"] == "umbra-sar"]
    sentinel = [
        o for o in trace["observations"]
        if o["source"] in ("sentinel-1", "sentinel-2")
    ]
    assert umbra, "trace must include at least one Umbra observation"
    assert sentinel, "trace must include at least one Sentinel observation"
    max_sentinel = max(o["confidence_weight"] for o in sentinel)
    min_umbra = min(o["confidence_weight"] for o in umbra)
    assert max_sentinel < min_umbra, (
        "every Sentinel observation must have lower confidence_weight "
        "than every Umbra observation"
    )


# ---------------------------------------------------------------------------
# Tasking options
# ---------------------------------------------------------------------------


_EXPECTED_OPTION_LABELS = {"SAT-A", "SAT-B", "Wait", "Optical", "Expand Search"}


def test_tasking_options_match_expected_set(trace) -> None:
    labels = {o["label"] for o in trace["candidate_tasking_options"]}
    assert labels == _EXPECTED_OPTION_LABELS


def test_sat_b_is_selected(trace) -> None:
    rec = trace["selected_recommendation"]
    assert rec["tasking_option_id"] == "opt-sat-b"
    assert rec["rank"] == 1
    selected_option = next(
        o for o in trace["candidate_tasking_options"]
        if o["tasking_option_id"] == rec["tasking_option_id"]
    )
    assert selected_option["label"] == "SAT-B"


# ---------------------------------------------------------------------------
# Score breakdowns
# ---------------------------------------------------------------------------


def test_every_option_has_score_breakdown(trace) -> None:
    options = trace["candidate_tasking_options"]
    breakdowns = trace["score_breakdowns"]
    sb_by_option = {sb["tasking_option_id"]: sb for sb in breakdowns}
    for opt in options:
        oid = opt["tasking_option_id"]
        assert oid in sb_by_option, f"missing score breakdown for {oid}"
        sb = sb_by_option[oid]
        assert "components" in sb
        assert "total_score" in sb


def test_sat_b_score_is_highest(trace) -> None:
    breakdowns = trace["score_breakdowns"]
    by_option = {sb["tasking_option_id"]: sb for sb in breakdowns}
    sat_b_score = by_option["opt-sat-b"]["total_score"]
    other_scores = [
        sb["total_score"] for oid, sb in by_option.items()
        if oid != "opt-sat-b"
    ]
    assert all(sat_b_score > s for s in other_scores), (
        "SAT-B must be the highest-scoring option"
    )


# ---------------------------------------------------------------------------
# Required top-level sections
# ---------------------------------------------------------------------------


def test_policy_rationale_exists(trace) -> None:
    pol = trace["policy_rationale"]
    assert pol["policy_rationale_id"]
    assert pol["data_mode"] == "simulated"
    # No claim that this is a trained RL agent.
    rl_advisory = pol.get("rl_advisory", "").lower()
    assert "not used" in rl_advisory or "no trained" in rl_advisory or "not a trained" in rl_advisory or "trained rl is not" in rl_advisory


def test_human_action_exists(trace) -> None:
    ha = trace["human_action"]
    assert ha["action"] == "approve"
    assert ha["tasking_option_id"] == "opt-sat-b"


def test_outcome_exists(trace) -> None:
    out = trace["outcome"]
    assert out["track_id"] == "trk-002"
    assert out["result"] == "reacquired"
    assert out["custody_score_delta"] > 0


def test_counterfactuals_exist(trace) -> None:
    cfs = trace["counterfactuals"]
    assert len(cfs) >= 4
    cf_options = {cf["alternative_tasking_option_id"] for cf in cfs}
    # All unselected options should have a counterfactual.
    expected_unselected = {
        "opt-sat-a", "opt-wait", "opt-optical", "opt-expand-search",
    }
    assert expected_unselected.issubset(cf_options)


def test_followup_recommendation_exists(trace) -> None:
    fr = trace["followup_recommendation"]
    assert fr["followup_recommendation_id"]
    assert "next_candidate_collect_type" in fr


# ---------------------------------------------------------------------------
# Data mode discipline
# ---------------------------------------------------------------------------


def test_every_record_has_data_mode(trace) -> None:
    """All record-shaped sections must label data_mode for each entry."""
    record_collections = [
        "events",
        "observations",
        "evidence_artifacts",
        "detections",
        "candidate_tracks",
        "custody_state_snapshots",
        "candidate_tasking_options",
        "score_breakdowns",
        "counterfactuals",
    ]
    for key in record_collections:
        for entry in trace[key]:
            assert entry.get("data_mode") in ("fixture", "simulated"), (
                f"{key} entry missing data_mode: {entry}"
            )
    # Singletons.
    for key in (
        "scenario_metadata",
        "selected_recommendation",
        "policy_rationale",
        "human_action",
        "outcome",
        "followup_recommendation",
    ):
        entry = trace[key]
        assert entry.get("data_mode") in ("fixture", "simulated"), (
            f"{key} missing data_mode"
        )
