"""Tests for :mod:`custody.hypotheses.planner_queue` (ADR-0021 Slice 14).

Pin the seven-status enum, status-derivation rules, priority formula
+ clamping, sort order + tie-break, JSONL ledger loader (round-trips
Slice 13 records), formatters, and scope guardrails (no forbidden
imports, no live-tasking / sensor-command / collection-order language).
"""
from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from custody.hypotheses.collection_value import (
    CollectionRecommendation,
    rank_collection_candidates,
)
from custody.hypotheses.counterfactual import (
    CounterfactualReport,
    simulate_counterfactual_collects,
)
from custody.hypotheses.custody_health import (
    CustodyHealthStatus,
    HypothesisCustodyHealth,
    assess_custody_health,
)
from custody.hypotheses.mission_value import (
    MissionValueReport,
    attribute_mission_value,
)
from custody.hypotheses.optimizer import (
    PlanConstraint,
    PlanOptimizationReport,
    optimize_collection_plan,
)
from custody.hypotheses.planner_queue import (
    PlannerQueue,
    PlannerQueueItem,
    PlannerQueueStatus,
    build_planner_queue,
    create_queue_item,
    format_queue_json,
    format_queue_markdown,
    format_queue_text,
    load_review_ledger_jsonl,
    record_status_from_review,
    select_latest_review_for_scenario,
)
from custody.hypotheses.planner_review import (
    ReviewAction,
    append_review_record_jsonl,
    create_planner_review_record,
    create_review_subject_from_optimized_plan,
)
from custody.hypotheses.policy_eval import (
    PolicyEvaluationReport,
    evaluate_collection_policies,
)
from custody.hypotheses.registry import (
    SCENARIO_TENNENT,
    SCENARIO_WHITSUN,
    TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
    TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
    WHITSUN_AIS_DARK_OR_POORLY_OBSERVED_VESSELS,
    WHITSUN_VESSEL_CLUSTER_ACTIVITY,
)
from custody.hypotheses.types import HypothesisEvidence, HypothesisState
from custody.hypotheses.update import update_state


T_RECENT = datetime(2023, 8, 13, 12, 0, 0, tzinfo=timezone.utc)
T_REVIEW = datetime(2026, 4, 24, 12, 0, 0, tzinfo=timezone.utc)
T_GEN = datetime(2026, 4, 24, 13, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Pipeline fixtures
# ---------------------------------------------------------------------------


def _ev(
    evidence_id: str,
    scenario_id: str,
    supports: tuple[str, ...] = (),
    contradicts: tuple[str, ...] = (),
    *,
    confidence: float = 0.6,
) -> HypothesisEvidence:
    return HypothesisEvidence(
        evidence_id=evidence_id,
        source_ref=f"observation:{evidence_id}",
        source_kind="observation",
        scenario_id=scenario_id,
        timestamp=T_RECENT,
        supports=supports,
        contradicts=contradicts,
        confidence=confidence,
        weight=1.0,
        reason="synthetic",
    )


def _ambiguous_pipeline(
    scenario_id: str, hid_a: str, hid_b: str,
) -> tuple[
    HypothesisCustodyHealth,
    CollectionRecommendation,
    MissionValueReport,
    CounterfactualReport,
    PlanOptimizationReport,
    PolicyEvaluationReport,
]:
    state = update_state(
        scenario_id,
        evidence=(
            _ev("ev-a", scenario_id, supports=(hid_a,), confidence=1.0),
            _ev("ev-b", scenario_id, supports=(hid_b,), confidence=1.0),
        ),
    )
    health = assess_custody_health(state, as_of=T_RECENT)
    assert health.status is CustodyHealthStatus.AMBIGUOUS
    rec = rank_collection_candidates(state, health)
    mv = attribute_mission_value(rec, health)
    cf = simulate_counterfactual_collects(state, health, rec)
    opt = optimize_collection_plan(rec, mv, cf, scenario_id=scenario_id)
    policy = evaluate_collection_policies(
        rec, mv, cf, opt, scenario_id=scenario_id,
    )
    return health, rec, mv, cf, opt, policy


@pytest.fixture(scope="module")
def tennent_pipeline():
    return _ambiguous_pipeline(
        SCENARIO_TENNENT,
        TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
        TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
    )


@pytest.fixture(scope="module")
def whitsun_pipeline():
    return _ambiguous_pipeline(
        SCENARIO_WHITSUN,
        WHITSUN_AIS_DARK_OR_POORLY_OBSERVED_VESSELS,
        WHITSUN_VESSEL_CLUSTER_ACTIVITY,
    )


def _approve_review(tennent_pipeline, *, when: datetime = T_REVIEW):
    health, _, _, _, opt, _ = tennent_pipeline
    subject = create_review_subject_from_optimized_plan(opt)
    return create_planner_review_record(
        subject, action=ReviewAction.APPROVE, operator_reason="ok",
        reviewed_at=when,
    )


def _reject_review(tennent_pipeline, *, when: datetime = T_REVIEW):
    health, _, _, _, opt, _ = tennent_pipeline
    subject = create_review_subject_from_optimized_plan(opt)
    return create_planner_review_record(
        subject, action=ReviewAction.REJECT, operator_reason="no",
        reviewed_at=when,
    )


# ---------------------------------------------------------------------------
# Enum + frozen dataclasses
# ---------------------------------------------------------------------------


def test_status_enum_has_seven_members() -> None:
    assert {s.value for s in PlannerQueueStatus} == {
        "pending_review", "needs_evidence", "approved", "rejected",
        "deferred", "overridden", "resolved",
    }


def test_planner_queue_item_is_frozen(tennent_pipeline) -> None:
    health, rec, mv, cf, opt, _ = tennent_pipeline
    item = create_queue_item(
        scenario_id=SCENARIO_TENNENT, health=health, recommendation=rec,
        mission_value_report=mv, counterfactual_report=cf,
        optimization_report=opt,
    )
    with pytest.raises(FrozenInstanceError):
        item.priority_score = 0.0  # type: ignore[misc]


def test_planner_queue_is_frozen() -> None:
    queue = PlannerQueue(
        generated_at=T_GEN, items=(), summary="x", caveats=(),
    )
    with pytest.raises(FrozenInstanceError):
        queue.summary = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# record_status_from_review
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "action, expected",
    [
        (ReviewAction.APPROVE, PlannerQueueStatus.APPROVED),
        (ReviewAction.REJECT, PlannerQueueStatus.REJECTED),
        (ReviewAction.DEFER, PlannerQueueStatus.DEFERRED),
        (ReviewAction.OVERRIDE, PlannerQueueStatus.OVERRIDDEN),
    ],
)
def test_record_status_from_review_maps_actions(
    tennent_pipeline, action, expected,
) -> None:
    health, _, _, _, opt, _ = tennent_pipeline
    subject = create_review_subject_from_optimized_plan(opt)
    overrides = (
        ("optical_context",) if action is ReviewAction.OVERRIDE else ()
    )
    record = create_planner_review_record(
        subject, action=action, operator_reason="ok",
        reviewed_at=T_REVIEW, override_candidate_ids=overrides,
    )
    assert record_status_from_review(record) is expected


# ---------------------------------------------------------------------------
# Status derivation without a review
# ---------------------------------------------------------------------------


def test_no_review_ambiguous_yields_pending_review(tennent_pipeline) -> None:
    health, rec, mv, cf, opt, _ = tennent_pipeline
    item = create_queue_item(
        scenario_id=SCENARIO_TENNENT, health=health, recommendation=rec,
        mission_value_report=mv, counterfactual_report=cf,
        optimization_report=opt,
    )
    assert item.status is PlannerQueueStatus.PENDING_REVIEW


def test_no_review_lost_yields_needs_evidence(tennent_pipeline) -> None:
    """Construct a LOST state and feed empty evidence."""
    health_lost = assess_custody_health(
        update_state(SCENARIO_TENNENT, evidence=()),
    )
    assert health_lost.status is CustodyHealthStatus.LOST
    _, rec, mv, cf, opt, _ = tennent_pipeline
    item = create_queue_item(
        scenario_id=SCENARIO_TENNENT, health=health_lost, recommendation=rec,
        mission_value_report=mv, counterfactual_report=cf,
        optimization_report=opt,
    )
    assert item.status is PlannerQueueStatus.NEEDS_EVIDENCE


def test_no_review_stale_yields_needs_evidence(tennent_pipeline) -> None:
    """Force STALE by advancing as_of past stale_after_days."""
    state = update_state(
        SCENARIO_TENNENT,
        evidence=(_ev("e", SCENARIO_TENNENT,
                      supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,),
                      confidence=0.85),),
    )
    health_stale = assess_custody_health(
        state, as_of=T_RECENT + timedelta(days=60),
    )
    assert health_stale.status is CustodyHealthStatus.STALE
    _, rec, mv, cf, opt, _ = tennent_pipeline
    item = create_queue_item(
        scenario_id=SCENARIO_TENNENT, health=health_stale, recommendation=rec,
        mission_value_report=mv, counterfactual_report=cf,
        optimization_report=opt,
    )
    assert item.status is PlannerQueueStatus.NEEDS_EVIDENCE


def test_no_review_healthy_no_ambiguity_yields_resolved(tennent_pipeline) -> None:
    state = update_state(
        SCENARIO_TENNENT,
        evidence=(_ev("e", SCENARIO_TENNENT,
                      supports=(TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,),
                      confidence=0.85),),
    )
    health_h = assess_custody_health(state, as_of=T_RECENT)
    assert health_h.status is CustodyHealthStatus.HEALTHY
    assert not health_h.ambiguity_pairs
    _, rec, mv, cf, opt, _ = tennent_pipeline
    item = create_queue_item(
        scenario_id=SCENARIO_TENNENT, health=health_h, recommendation=rec,
        mission_value_report=mv, counterfactual_report=cf,
        optimization_report=opt,
    )
    assert item.status is PlannerQueueStatus.RESOLVED


# ---------------------------------------------------------------------------
# Priority formula + clamping
# ---------------------------------------------------------------------------


def test_priority_score_in_unit_interval(tennent_pipeline) -> None:
    health, rec, mv, cf, opt, _ = tennent_pipeline
    item = create_queue_item(
        scenario_id=SCENARIO_TENNENT, health=health, recommendation=rec,
        mission_value_report=mv, counterfactual_report=cf,
        optimization_report=opt,
    )
    assert 0.0 <= item.priority_score <= 1.0


def test_pending_review_beats_approved_when_other_fields_equal(
    tennent_pipeline,
) -> None:
    health, rec, mv, cf, opt, _ = tennent_pipeline
    pending = create_queue_item(
        scenario_id=SCENARIO_TENNENT, health=health, recommendation=rec,
        mission_value_report=mv, counterfactual_report=cf,
        optimization_report=opt,
    )
    approve = _approve_review(tennent_pipeline)
    approved = create_queue_item(
        scenario_id=SCENARIO_TENNENT, health=health, recommendation=rec,
        mission_value_report=mv, counterfactual_report=cf,
        optimization_report=opt, review_record=approve,
    )
    assert pending.priority_score > approved.priority_score


def test_pending_review_beats_rejected_when_other_fields_equal(
    tennent_pipeline,
) -> None:
    health, rec, mv, cf, opt, _ = tennent_pipeline
    pending = create_queue_item(
        scenario_id=SCENARIO_TENNENT, health=health, recommendation=rec,
        mission_value_report=mv, counterfactual_report=cf,
        optimization_report=opt,
    )
    reject = _reject_review(tennent_pipeline)
    rejected = create_queue_item(
        scenario_id=SCENARIO_TENNENT, health=health, recommendation=rec,
        mission_value_report=mv, counterfactual_report=cf,
        optimization_report=opt, review_record=reject,
    )
    assert pending.priority_score > rejected.priority_score


# ---------------------------------------------------------------------------
# build_planner_queue ranking + tie-break
# ---------------------------------------------------------------------------


def test_build_planner_queue_sorts_by_priority_desc(
    tennent_pipeline, whitsun_pipeline,
) -> None:
    t_health, t_rec, t_mv, t_cf, t_opt, _ = tennent_pipeline
    w_health, w_rec, w_mv, w_cf, w_opt, _ = whitsun_pipeline
    t_item = create_queue_item(
        scenario_id=SCENARIO_TENNENT, health=t_health, recommendation=t_rec,
        mission_value_report=t_mv, counterfactual_report=t_cf,
        optimization_report=t_opt,
    )
    w_item = create_queue_item(
        scenario_id=SCENARIO_WHITSUN, health=w_health, recommendation=w_rec,
        mission_value_report=w_mv, counterfactual_report=w_cf,
        optimization_report=w_opt,
    )
    queue = build_planner_queue([w_item, t_item], generated_at=T_GEN)
    priorities = [i.priority_score for i in queue.items]
    assert priorities == sorted(priorities, reverse=True)


def test_status_tiebreak_uses_documented_order() -> None:
    """When two items share the priority_score, status order decides:
    PENDING < NEEDS_EVIDENCE < DEFERRED < OVERRIDDEN < APPROVED < REJECTED < RESOLVED.
    """
    common = dict(
        priority_score=0.50,
        health_status=CustodyHealthStatus.AMBIGUOUS, health_score=0.35,
        primary_ambiguity=None, recommended_candidate_ids=(),
        planning_utility=0.0, mission_value_proxy=0.0,
        expected_ambiguity_resolution=0.0, expected_health_score_delta=0.0,
        latest_review_action=None, latest_review_id=None,
        reason="x", next_action="x", caveats=(),
    )
    a = PlannerQueueItem(item_id="a", scenario_id="a",
                         status=PlannerQueueStatus.RESOLVED, **common)
    b = PlannerQueueItem(item_id="b", scenario_id="b",
                         status=PlannerQueueStatus.PENDING_REVIEW, **common)
    queue = build_planner_queue([a, b], generated_at=T_GEN)
    # PENDING_REVIEW must come first by status order even though they share
    # priority and scenario tiebreak would give "a" alphabetically.
    assert queue.items[0].scenario_id == "b"


# ---------------------------------------------------------------------------
# item_id determinism
# ---------------------------------------------------------------------------


def test_item_id_is_deterministic(tennent_pipeline) -> None:
    health, rec, mv, cf, opt, _ = tennent_pipeline
    a = create_queue_item(
        scenario_id=SCENARIO_TENNENT, health=health, recommendation=rec,
        mission_value_report=mv, counterfactual_report=cf,
        optimization_report=opt,
    )
    b = create_queue_item(
        scenario_id=SCENARIO_TENNENT, health=health, recommendation=rec,
        mission_value_report=mv, counterfactual_report=cf,
        optimization_report=opt,
    )
    assert a.item_id == b.item_id


def test_item_id_changes_with_review(tennent_pipeline) -> None:
    health, rec, mv, cf, opt, _ = tennent_pipeline
    plain = create_queue_item(
        scenario_id=SCENARIO_TENNENT, health=health, recommendation=rec,
        mission_value_report=mv, counterfactual_report=cf,
        optimization_report=opt,
    )
    reviewed = create_queue_item(
        scenario_id=SCENARIO_TENNENT, health=health, recommendation=rec,
        mission_value_report=mv, counterfactual_report=cf,
        optimization_report=opt,
        review_record=_approve_review(tennent_pipeline),
    )
    assert plain.item_id != reviewed.item_id


# ---------------------------------------------------------------------------
# select_latest_review_for_scenario
# ---------------------------------------------------------------------------


def test_select_latest_review_returns_most_recent(tennent_pipeline) -> None:
    earlier = _approve_review(tennent_pipeline, when=T_REVIEW)
    later = _reject_review(
        tennent_pipeline, when=T_REVIEW + timedelta(hours=1),
    )
    chosen = select_latest_review_for_scenario(
        [earlier, later], SCENARIO_TENNENT,
    )
    assert chosen is later


def test_select_latest_review_returns_none_when_no_match(tennent_pipeline) -> None:
    record = _approve_review(tennent_pipeline)
    assert select_latest_review_for_scenario(
        [record], SCENARIO_WHITSUN,
    ) is None


# ---------------------------------------------------------------------------
# JSONL ledger loader
# ---------------------------------------------------------------------------


def test_load_review_ledger_round_trips_records(tennent_pipeline, tmp_path) -> None:
    record = _approve_review(tennent_pipeline)
    ledger = tmp_path / "reviews.jsonl"
    append_review_record_jsonl(ledger, record)
    loaded = load_review_ledger_jsonl(ledger)
    assert len(loaded) == 1
    rt = loaded[0]
    assert rt.review_id == record.review_id
    assert rt.action is record.action
    assert rt.scenario_id == record.scenario_id
    assert rt.subject.candidate_ids == record.subject.candidate_ids


def test_load_review_ledger_ignores_blank_lines(tennent_pipeline, tmp_path) -> None:
    record = _approve_review(tennent_pipeline)
    ledger = tmp_path / "reviews.jsonl"
    append_review_record_jsonl(ledger, record)
    # Append a blank and a record line.
    ledger.write_text(
        "\n" + ledger.read_text(encoding="utf-8") + "\n\n",
        encoding="utf-8",
    )
    loaded = load_review_ledger_jsonl(ledger)
    assert len(loaded) == 1


def test_load_review_ledger_raises_on_malformed_json(tmp_path) -> None:
    ledger = tmp_path / "broken.jsonl"
    ledger.write_text("{not json\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_review_ledger_jsonl(ledger)


def test_load_review_ledger_raises_when_file_missing(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_review_ledger_jsonl(tmp_path / "no.jsonl")


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def test_format_queue_text_includes_required_blocks(
    tennent_pipeline, whitsun_pipeline,
) -> None:
    t_health, t_rec, t_mv, t_cf, t_opt, _ = tennent_pipeline
    w_health, w_rec, w_mv, w_cf, w_opt, _ = whitsun_pipeline
    queue = build_planner_queue([
        create_queue_item(
            scenario_id=SCENARIO_TENNENT, health=t_health,
            recommendation=t_rec, mission_value_report=t_mv,
            counterfactual_report=t_cf, optimization_report=t_opt,
        ),
        create_queue_item(
            scenario_id=SCENARIO_WHITSUN, health=w_health,
            recommendation=w_rec, mission_value_report=w_mv,
            counterfactual_report=w_cf, optimization_report=w_opt,
        ),
    ], generated_at=T_GEN)
    text = format_queue_text(queue)
    assert "PLANNER WORK QUEUE" in text
    assert "Ranked work items" in text
    assert "Status:" in text
    assert "Priority:" in text
    assert "Queue summary" in text
    assert "Caveats" in text


def test_format_queue_json_is_parseable(tennent_pipeline) -> None:
    health, rec, mv, cf, opt, _ = tennent_pipeline
    queue = build_planner_queue([
        create_queue_item(
            scenario_id=SCENARIO_TENNENT, health=health, recommendation=rec,
            mission_value_report=mv, counterfactual_report=cf,
            optimization_report=opt,
        ),
    ], generated_at=T_GEN)
    parsed = json.loads(format_queue_json(queue))
    assert "generated_at" in parsed
    assert "items" in parsed
    assert isinstance(parsed["items"], list)
    assert parsed["items"][0]["scenario_id"] == SCENARIO_TENNENT


def test_format_queue_markdown_starts_with_h1(tennent_pipeline) -> None:
    health, rec, mv, cf, opt, _ = tennent_pipeline
    queue = build_planner_queue([
        create_queue_item(
            scenario_id=SCENARIO_TENNENT, health=health, recommendation=rec,
            mission_value_report=mv, counterfactual_report=cf,
            optimization_report=opt,
        ),
    ], generated_at=T_GEN)
    md = format_queue_markdown(queue)
    assert md.startswith("# Planner Work Queue\n")


def test_text_output_is_deterministic_with_fixed_generated_at(
    tennent_pipeline,
) -> None:
    health, rec, mv, cf, opt, _ = tennent_pipeline
    items = [create_queue_item(
        scenario_id=SCENARIO_TENNENT, health=health, recommendation=rec,
        mission_value_report=mv, counterfactual_report=cf,
        optimization_report=opt,
    )]
    a = format_queue_text(build_planner_queue(items, generated_at=T_GEN))
    b = format_queue_text(build_planner_queue(items, generated_at=T_GEN))
    assert a == b


# ---------------------------------------------------------------------------
# Language guardrails
# ---------------------------------------------------------------------------


_FORBIDDEN_PHRASES = (
    "tasking order",
    "live tasking",
    "sensor command",
    "production scheduler",
    "autonomous execution",
    "collection order",
    "revenue dollars",
)


def test_no_forbidden_language_outside_non_claim_caveat(tennent_pipeline) -> None:
    health, rec, mv, cf, opt, _ = tennent_pipeline
    queue = build_planner_queue([
        create_queue_item(
            scenario_id=SCENARIO_TENNENT, health=health, recommendation=rec,
            mission_value_report=mv, counterfactual_report=cf,
            optimization_report=opt,
        ),
    ], generated_at=T_GEN)
    blob = " ".join([
        format_queue_text(queue),
        format_queue_json(queue),
        format_queue_markdown(queue),
    ]).lower()
    safe = blob.replace(
        "no live tasking or sensor command is issued", "",
    )
    for needle in _FORBIDDEN_PHRASES:
        assert needle not in safe, (
            f"forbidden phrase {needle!r} appears outside the explicit "
            "non-claim caveat"
        )


def test_caveat_explicitly_mentions_no_live_tasking_and_sensor_command(
    tennent_pipeline,
) -> None:
    health, rec, mv, cf, opt, _ = tennent_pipeline
    queue = build_planner_queue([
        create_queue_item(
            scenario_id=SCENARIO_TENNENT, health=health, recommendation=rec,
            mission_value_report=mv, counterfactual_report=cf,
            optimization_report=opt,
        ),
    ], generated_at=T_GEN)
    text = format_queue_text(queue).lower()
    assert "decision support only" in text
    assert "no live tasking or sensor command is issued" in text


# ---------------------------------------------------------------------------
# Import-boundary scan
# ---------------------------------------------------------------------------


def test_planner_queue_module_has_no_forbidden_imports() -> None:
    import ast
    import custody.hypotheses.planner_queue as mod
    tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
    forbidden_prefixes = (
        "custody.detection",
        "custody.ingest.gfw_presence",
        "sentinelhub",
        "custody.fusion.tracker",
        "custody.taskrecommendation",
    )
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
        f"planner_queue.py imports forbidden modules: {offending}"
    )


def test_build_planner_queue_requires_tz_aware_generated_at() -> None:
    naive = datetime(2026, 4, 24, 12, 0, 0)
    with pytest.raises(ValueError):
        build_planner_queue([], generated_at=naive)
