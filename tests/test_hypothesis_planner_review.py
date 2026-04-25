"""Tests for :mod:`custody.hypotheses.planner_review` (ADR-0021 Slice 13).

Pin the four ReviewAction values, subject builders, validation rules,
deterministic hashing (review_id + packet_hash), JSON serialization,
JSONL append behaviour, and scope guardrails (no forbidden imports,
no live-tasking / sensor-command / collection-order language).
"""
from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
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
from custody.hypotheses.planner_review import (
    PlannerReviewRecord,
    ReviewAction,
    ReviewLedgerAppendResult,
    ReviewSubject,
    append_review_record_jsonl,
    create_planner_review_record,
    create_review_subject_from_optimized_plan,
    create_review_subject_from_policy_evaluation,
    format_review_markdown,
    format_review_text,
    record_to_dict,
    record_to_json,
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
from custody.hypotheses.types import HypothesisEvidence
from custody.hypotheses.update import update_state


T_RECENT = datetime(2023, 8, 13, 12, 0, 0, tzinfo=timezone.utc)
T_REVIEW = datetime(2026, 4, 24, 12, 0, 0, tzinfo=timezone.utc)


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
    scenario_id: str,
    hid_a: str,
    hid_b: str,
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


def _approve_subject(tennent_pipeline) -> ReviewSubject:
    _, _, _, _, opt, _ = tennent_pipeline
    return create_review_subject_from_optimized_plan(opt)


# ---------------------------------------------------------------------------
# Enum + frozen dataclasses
# ---------------------------------------------------------------------------


def test_review_action_has_exactly_four_members() -> None:
    assert {a.value for a in ReviewAction} == {
        "approve", "reject", "defer", "override",
    }


def test_review_subject_is_frozen() -> None:
    s = ReviewSubject(
        subject_type="optimized_plan", subject_id="recommended",
        scenario_id="x", candidate_ids=("a",), label="x", summary="x",
    )
    with pytest.raises(FrozenInstanceError):
        s.label = "y"  # type: ignore[misc]


def test_review_record_is_frozen(tennent_pipeline) -> None:
    subject = _approve_subject(tennent_pipeline)
    record = create_planner_review_record(
        subject, action=ReviewAction.APPROVE, operator_reason="ok",
        reviewed_at=T_REVIEW,
    )
    with pytest.raises(FrozenInstanceError):
        record.operator_reason = "y"  # type: ignore[misc]


def test_ledger_append_result_is_frozen() -> None:
    r = ReviewLedgerAppendResult(
        path="x", record_id="y", records_written=1, summary="z",
    )
    with pytest.raises(FrozenInstanceError):
        r.records_written = 2  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Subject builders
# ---------------------------------------------------------------------------


def test_subject_from_optimized_plan_extracts_candidates(tennent_pipeline) -> None:
    _, _, _, _, opt, _ = tennent_pipeline
    subject = create_review_subject_from_optimized_plan(opt)
    expected_ids = tuple(p.candidate_id for p in opt.recommended_plan.selected_items)
    assert subject.candidate_ids == expected_ids
    assert subject.scenario_id == SCENARIO_TENNENT
    assert subject.subject_type == "optimized_plan"
    assert subject.subject_id == "recommended"
    assert subject.label == "optimized_plan:recommended"


def test_subject_from_optimized_plan_with_explicit_strategy(tennent_pipeline) -> None:
    _, _, _, _, opt, _ = tennent_pipeline
    subject = create_review_subject_from_optimized_plan(
        opt, strategy="exhaustive",
    )
    expected_ids = tuple(p.candidate_id for p in opt.exhaustive_plan.selected_items)
    assert subject.candidate_ids == expected_ids
    assert subject.subject_id == "exhaustive"


def test_subject_from_policy_evaluation_uses_winner_by_default(tennent_pipeline) -> None:
    _, _, _, _, _, policy = tennent_pipeline
    subject = create_review_subject_from_policy_evaluation(policy)
    winner = next(
        e for e in policy.evaluations if e.policy_id == policy.winning_policy_id
    )
    assert subject.subject_id == winner.policy_id
    assert subject.candidate_ids == tuple(winner.selected_candidate_ids)
    assert subject.subject_type == "winning_policy"


def test_subject_from_policy_evaluation_with_explicit_policy_id(tennent_pipeline) -> None:
    _, _, _, _, _, policy = tennent_pipeline
    subject = create_review_subject_from_policy_evaluation(
        policy, policy_id="ais_context_first",
    )
    assert subject.subject_id == "ais_context_first"


def test_subject_from_policy_evaluation_unknown_id_raises(tennent_pipeline) -> None:
    _, _, _, _, _, policy = tennent_pipeline
    with pytest.raises(ValueError):
        create_review_subject_from_policy_evaluation(
            policy, policy_id="nonexistent_policy",
        )


def test_subject_from_optimized_plan_unknown_strategy_raises(tennent_pipeline) -> None:
    _, _, _, _, opt, _ = tennent_pipeline
    with pytest.raises(ValueError):
        create_review_subject_from_optimized_plan(opt, strategy="nope")


# ---------------------------------------------------------------------------
# Record creation — happy path
# ---------------------------------------------------------------------------


def test_approve_record_validates_and_serializes(tennent_pipeline) -> None:
    subject = _approve_subject(tennent_pipeline)
    record = create_planner_review_record(
        subject, action=ReviewAction.APPROVE,
        operator_reason="best ambiguity reduction under budget",
        reviewed_at=T_REVIEW, operator_id="ben",
    )
    assert record.action is ReviewAction.APPROVE
    assert record.scenario_id == SCENARIO_TENNENT
    d = record_to_dict(record)
    assert d["action"] == "approve"
    assert d["operator_id"] == "ben"
    assert d["reviewed_at"] == T_REVIEW.isoformat()


def test_reject_record_validates(tennent_pipeline) -> None:
    subject = _approve_subject(tennent_pipeline)
    record = create_planner_review_record(
        subject, action=ReviewAction.REJECT,
        operator_reason="need more evidence first",
        reviewed_at=T_REVIEW,
    )
    assert record.action is ReviewAction.REJECT


def test_defer_record_validates(tennent_pipeline) -> None:
    subject = _approve_subject(tennent_pipeline)
    record = create_planner_review_record(
        subject, action=ReviewAction.DEFER,
        operator_reason="wait for weather window",
        reviewed_at=T_REVIEW,
    )
    assert record.action is ReviewAction.DEFER


def test_override_record_with_override_candidate_validates(tennent_pipeline) -> None:
    subject = _approve_subject(tennent_pipeline)
    record = create_planner_review_record(
        subject, action=ReviewAction.OVERRIDE,
        operator_reason="ais coverage is decisive",
        reviewed_at=T_REVIEW,
        override_candidate_ids=("ais_coverage_query",),
    )
    assert record.action is ReviewAction.OVERRIDE
    assert record.override_candidate_ids == ("ais_coverage_query",)


def test_action_string_is_accepted_and_normalized(tennent_pipeline) -> None:
    subject = _approve_subject(tennent_pipeline)
    record = create_planner_review_record(
        subject, action="APPROVE",
        operator_reason="ok", reviewed_at=T_REVIEW,
    )
    assert record.action is ReviewAction.APPROVE


def test_reviewed_at_iso_string_becomes_tz_aware_utc(tennent_pipeline) -> None:
    subject = _approve_subject(tennent_pipeline)
    record = create_planner_review_record(
        subject, action=ReviewAction.APPROVE, operator_reason="ok",
        reviewed_at="2026-04-24T12:00:00+00:00",
    )
    assert record.reviewed_at.tzinfo is not None
    assert record.reviewed_at == T_REVIEW


def test_reviewed_at_with_offset_is_normalized_to_utc(tennent_pipeline) -> None:
    subject = _approve_subject(tennent_pipeline)
    record = create_planner_review_record(
        subject, action=ReviewAction.APPROVE, operator_reason="ok",
        reviewed_at="2026-04-24T07:00:00-05:00",
    )
    assert record.reviewed_at.utcoffset().total_seconds() == 0.0
    assert record.reviewed_at.hour == 12


# ---------------------------------------------------------------------------
# Validation failures
# ---------------------------------------------------------------------------


def test_override_without_candidate_ids_raises(tennent_pipeline) -> None:
    subject = _approve_subject(tennent_pipeline)
    with pytest.raises(ValueError):
        create_planner_review_record(
            subject, action=ReviewAction.OVERRIDE,
            operator_reason="bad", reviewed_at=T_REVIEW,
        )


@pytest.mark.parametrize(
    "action", [ReviewAction.APPROVE, ReviewAction.REJECT, ReviewAction.DEFER],
)
def test_non_override_with_overrides_raises(tennent_pipeline, action) -> None:
    subject = _approve_subject(tennent_pipeline)
    with pytest.raises(ValueError):
        create_planner_review_record(
            subject, action=action, operator_reason="bad",
            reviewed_at=T_REVIEW,
            override_candidate_ids=("optical_context",),
        )


@pytest.mark.parametrize("reason", ["", "   ", "\n\t  "])
def test_empty_or_whitespace_reason_raises(tennent_pipeline, reason: str) -> None:
    subject = _approve_subject(tennent_pipeline)
    with pytest.raises(ValueError):
        create_planner_review_record(
            subject, action=ReviewAction.APPROVE, operator_reason=reason,
            reviewed_at=T_REVIEW,
        )


def test_invalid_action_string_raises(tennent_pipeline) -> None:
    subject = _approve_subject(tennent_pipeline)
    with pytest.raises(ValueError):
        create_planner_review_record(
            subject, action="not_an_action",
            operator_reason="ok", reviewed_at=T_REVIEW,
        )


def test_naive_datetime_for_reviewed_at_raises(tennent_pipeline) -> None:
    subject = _approve_subject(tennent_pipeline)
    naive = datetime(2026, 4, 24, 12, 0, 0)
    with pytest.raises(ValueError):
        create_planner_review_record(
            subject, action=ReviewAction.APPROVE, operator_reason="ok",
            reviewed_at=naive,
        )


def test_naive_iso_string_for_reviewed_at_raises(tennent_pipeline) -> None:
    subject = _approve_subject(tennent_pipeline)
    with pytest.raises(ValueError):
        create_planner_review_record(
            subject, action=ReviewAction.APPROVE, operator_reason="ok",
            reviewed_at="2026-04-24T12:00:00",
        )


# ---------------------------------------------------------------------------
# Hashing determinism
# ---------------------------------------------------------------------------


def test_review_id_is_deterministic_for_identical_inputs(tennent_pipeline) -> None:
    subject = _approve_subject(tennent_pipeline)
    a = create_planner_review_record(
        subject, action=ReviewAction.APPROVE, operator_reason="ok",
        reviewed_at=T_REVIEW, operator_id="ben",
    )
    b = create_planner_review_record(
        subject, action=ReviewAction.APPROVE, operator_reason="ok",
        reviewed_at=T_REVIEW, operator_id="ben",
    )
    assert a.review_id == b.review_id


def test_review_id_changes_with_action(tennent_pipeline) -> None:
    subject = _approve_subject(tennent_pipeline)
    a = create_planner_review_record(
        subject, action=ReviewAction.APPROVE, operator_reason="ok",
        reviewed_at=T_REVIEW,
    )
    b = create_planner_review_record(
        subject, action=ReviewAction.REJECT, operator_reason="ok",
        reviewed_at=T_REVIEW,
    )
    assert a.review_id != b.review_id


def test_review_id_changes_with_reason(tennent_pipeline) -> None:
    subject = _approve_subject(tennent_pipeline)
    a = create_planner_review_record(
        subject, action=ReviewAction.APPROVE, operator_reason="reason a",
        reviewed_at=T_REVIEW,
    )
    b = create_planner_review_record(
        subject, action=ReviewAction.APPROVE, operator_reason="reason b",
        reviewed_at=T_REVIEW,
    )
    assert a.review_id != b.review_id


def test_review_id_changes_with_reviewed_at(tennent_pipeline) -> None:
    subject = _approve_subject(tennent_pipeline)
    a = create_planner_review_record(
        subject, action=ReviewAction.APPROVE, operator_reason="ok",
        reviewed_at=T_REVIEW,
    )
    later = T_REVIEW.replace(hour=13)
    b = create_planner_review_record(
        subject, action=ReviewAction.APPROVE, operator_reason="ok",
        reviewed_at=later,
    )
    assert a.review_id != b.review_id


def test_packet_hash_is_deterministic_for_identical_subject(tennent_pipeline) -> None:
    subject = _approve_subject(tennent_pipeline)
    a = create_planner_review_record(
        subject, action=ReviewAction.APPROVE, operator_reason="x",
        reviewed_at=T_REVIEW,
    )
    b = create_planner_review_record(
        subject, action=ReviewAction.REJECT, operator_reason="y",
        reviewed_at=T_REVIEW,
    )
    # Same subject -> same packet_hash regardless of action / reason.
    assert a.packet_hash == b.packet_hash


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_record_to_dict_contains_expected_keys(tennent_pipeline) -> None:
    subject = _approve_subject(tennent_pipeline)
    record = create_planner_review_record(
        subject, action=ReviewAction.APPROVE, operator_reason="ok",
        reviewed_at=T_REVIEW,
    )
    d = record_to_dict(record)
    expected = {
        "review_id", "scenario_id", "action", "subject", "operator_reason",
        "reviewed_at", "operator_id", "override_candidate_ids",
        "packet_hash", "caveats",
    }
    assert set(d.keys()) == expected


def test_record_to_json_is_parseable(tennent_pipeline) -> None:
    subject = _approve_subject(tennent_pipeline)
    record = create_planner_review_record(
        subject, action=ReviewAction.APPROVE, operator_reason="ok",
        reviewed_at=T_REVIEW,
    )
    parsed = json.loads(record_to_json(record))
    assert parsed["scenario_id"] == SCENARIO_TENNENT
    assert parsed["action"] == "approve"
    assert isinstance(parsed["subject"]["candidate_ids"], list)


def test_record_to_json_is_deterministic(tennent_pipeline) -> None:
    subject = _approve_subject(tennent_pipeline)
    record = create_planner_review_record(
        subject, action=ReviewAction.APPROVE, operator_reason="ok",
        reviewed_at=T_REVIEW,
    )
    assert record_to_json(record) == record_to_json(record)


# ---------------------------------------------------------------------------
# JSONL append
# ---------------------------------------------------------------------------


def test_append_creates_file_and_writes_one_line(tennent_pipeline, tmp_path) -> None:
    subject = _approve_subject(tennent_pipeline)
    record = create_planner_review_record(
        subject, action=ReviewAction.APPROVE, operator_reason="ok",
        reviewed_at=T_REVIEW,
    )
    ledger = tmp_path / "reviews.jsonl"
    result = append_review_record_jsonl(ledger, record)
    assert isinstance(result, ReviewLedgerAppendResult)
    assert result.records_written == 1
    assert ledger.exists()
    lines = ledger.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["review_id"] == record.review_id


def test_append_does_not_overwrite_existing(tennent_pipeline, tmp_path) -> None:
    subject = _approve_subject(tennent_pipeline)
    a = create_planner_review_record(
        subject, action=ReviewAction.APPROVE, operator_reason="reason a",
        reviewed_at=T_REVIEW,
    )
    b = create_planner_review_record(
        subject, action=ReviewAction.REJECT, operator_reason="reason b",
        reviewed_at=T_REVIEW,
    )
    ledger = tmp_path / "reviews.jsonl"
    append_review_record_jsonl(ledger, a)
    append_review_record_jsonl(ledger, b)
    lines = ledger.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["action"] == "approve"
    assert parsed[1]["action"] == "reject"


def test_append_raises_when_parent_missing(tennent_pipeline, tmp_path) -> None:
    subject = _approve_subject(tennent_pipeline)
    record = create_planner_review_record(
        subject, action=ReviewAction.APPROVE, operator_reason="ok",
        reviewed_at=T_REVIEW,
    )
    bad = tmp_path / "no" / "such" / "dir" / "reviews.jsonl"
    with pytest.raises(FileNotFoundError):
        append_review_record_jsonl(bad, record)


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def test_text_format_includes_required_blocks(tennent_pipeline) -> None:
    subject = _approve_subject(tennent_pipeline)
    record = create_planner_review_record(
        subject, action=ReviewAction.APPROVE, operator_reason="approve it",
        reviewed_at=T_REVIEW,
    )
    text = format_review_text(record)
    assert "HUMAN-IN-THE-LOOP REVIEW" in text
    assert "Operator action" in text
    assert "Human reason" in text
    assert "review_id" in text
    assert "approve it" in text


def test_markdown_format_includes_h1(tennent_pipeline) -> None:
    subject = _approve_subject(tennent_pipeline)
    record = create_planner_review_record(
        subject, action=ReviewAction.APPROVE, operator_reason="ok",
        reviewed_at=T_REVIEW,
    )
    md = format_review_markdown(record)
    assert md.startswith("# Human-in-the-Loop Review - ")
    assert "**APPROVE**" in md


def test_text_override_section_lists_candidates(tennent_pipeline) -> None:
    subject = _approve_subject(tennent_pipeline)
    record = create_planner_review_record(
        subject, action=ReviewAction.OVERRIDE,
        operator_reason="ais decisive",
        reviewed_at=T_REVIEW,
        override_candidate_ids=("ais_coverage_query",),
    )
    text = format_review_text(record)
    assert "Override candidates" in text
    assert "ais_coverage_query" in text


# ---------------------------------------------------------------------------
# Language guardrails
# ---------------------------------------------------------------------------


_FORBIDDEN_PHRASES = (
    "tasking order",
    "live tasking",
    "autonomous execution",
    "production scheduler",
    "sensor command",
    "collection order",
)


def test_no_forbidden_language_in_caveats_or_summary(tennent_pipeline) -> None:
    subject = _approve_subject(tennent_pipeline)
    record = create_planner_review_record(
        subject, action=ReviewAction.APPROVE, operator_reason="ok",
        reviewed_at=T_REVIEW,
    )
    blobs = list(record.caveats)
    blobs.extend([format_review_text(record), format_review_markdown(record)])
    text = " ".join(blobs).lower()
    # Note: the caveat itself contains "no live tasking or sensor command"
    # as an explicit non-claim.  We forbid the bigrams "live tasking",
    # "sensor command", etc., so the caveat phrasing must be careful;
    # the implementation uses the ASCII substrings only inside non-claim
    # contexts that the tests above pin.
    # Skip the caveat-non-claim phrasing for this check.
    safe_text = text.replace(
        "no live tasking or sensor command was issued", ""
    )
    for needle in _FORBIDDEN_PHRASES:
        assert needle not in safe_text, (
            f"forbidden phrase {needle!r} appears outside the non-claim caveat"
        )


def test_caveat_explicitly_disclaims_live_tasking_and_sensor_command(
    tennent_pipeline,
) -> None:
    subject = _approve_subject(tennent_pipeline)
    record = create_planner_review_record(
        subject, action=ReviewAction.APPROVE, operator_reason="ok",
        reviewed_at=T_REVIEW,
    )
    joined = " ".join(record.caveats).lower()
    assert "review record only" in joined
    assert "no live tasking or sensor command was issued" in joined


# ---------------------------------------------------------------------------
# Import-boundary scan
# ---------------------------------------------------------------------------


def test_planner_review_module_has_no_forbidden_imports() -> None:
    import ast
    import custody.hypotheses.planner_review as mod
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
        f"planner_review.py imports forbidden modules: {offending}"
    )
