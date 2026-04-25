"""Human-in-the-loop review ledger (ADR-0021 Slice 13).

Auditable controls for candidate collection recommendations.  Operators
can approve, reject, defer, or override an optimized plan or a winning
heuristic policy.  Each action produces a deterministic
:class:`PlannerReviewRecord` with stable ``review_id`` and
``packet_hash`` fingerprints suitable for an audit trail; records can
optionally be appended to a JSONL ledger for persistent history.

This module is a **review/audit layer only**.  It does not execute any
collect, issue tasking orders, or command sensors — every record carries
the caveat ``"review record only; no live tasking or sensor command was
issued"`` so downstream readers cannot misinterpret a record as an
execution receipt.

Scope guardrails
----------------

- Pure standard library (``hashlib``, ``json``, ``datetime``,
  ``pathlib``, ``enum``, ``dataclasses``).
- No imports from ``custody.detection.*``, ``custody.ingest.gfw_presence``,
  ``custody.fusion.tracker``, ``custody.taskrecommendation``, or
  Sentinel SDKs.
- The only wall-clock call is the optional ``reviewed_at=None`` fallback
  in :func:`create_planner_review_record`; tests pass an explicit
  ``reviewed_at`` to keep output deterministic.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from custody.hypotheses.optimizer import (
    OptimizedPlan,
    PlanOptimizationReport,
)
from custody.hypotheses.policy_eval import (
    PolicyEvaluation,
    PolicyEvaluationReport,
)


# ---------------------------------------------------------------------------
# Enum + dataclasses
# ---------------------------------------------------------------------------


class ReviewAction(Enum):
    APPROVE = "approve"
    REJECT = "reject"
    DEFER = "defer"
    OVERRIDE = "override"


@dataclass(frozen=True)
class ReviewSubject:
    """Identifies the artifact being reviewed (a plan or a policy)."""
    subject_type: str   # "optimized_plan" or "winning_policy"
    subject_id: str
    scenario_id: str
    candidate_ids: tuple[str, ...]
    label: str
    summary: str


@dataclass(frozen=True)
class PlannerReviewRecord:
    """One auditable review entry — produced by an operator action.

    ``review_id`` and ``packet_hash`` are deterministic hashes derived
    from the subject and operator inputs; identical inputs yield
    identical ids so downstream tools can deduplicate or reconcile.
    """
    review_id: str
    scenario_id: str
    action: ReviewAction
    subject: ReviewSubject
    operator_reason: str
    reviewed_at: datetime
    operator_id: str | None
    override_candidate_ids: tuple[str, ...]
    packet_hash: str
    caveats: tuple[str, ...]


@dataclass(frozen=True)
class ReviewLedgerAppendResult:
    """Result of appending a record to a JSONL ledger."""
    path: str
    record_id: str
    records_written: int
    summary: str


# ---------------------------------------------------------------------------
# Subject builders
# ---------------------------------------------------------------------------


_PLAN_BY_STRATEGY = ("recommended", "exhaustive", "greedy")


def _select_plan(
    report: PlanOptimizationReport, strategy: str,
) -> OptimizedPlan:
    if strategy == "recommended":
        return report.recommended_plan
    if strategy == "exhaustive":
        return report.exhaustive_plan
    if strategy == "greedy":
        return report.greedy_plan
    valid = ", ".join(_PLAN_BY_STRATEGY)
    raise ValueError(
        f"unknown strategy {strategy!r}; valid strategies: {valid}"
    )


def create_review_subject_from_optimized_plan(
    optimization_report: PlanOptimizationReport,
    *,
    strategy: str = "recommended",
) -> ReviewSubject:
    """Build a :class:`ReviewSubject` from an optimizer plan."""
    plan = _select_plan(optimization_report, strategy)
    candidate_ids = tuple(p.candidate_id for p in plan.selected_items)
    return ReviewSubject(
        subject_type="optimized_plan",
        subject_id=strategy,
        scenario_id=optimization_report.scenario_id,
        candidate_ids=candidate_ids,
        label=f"optimized_plan:{strategy}",
        summary=plan.summary,
    )


def create_review_subject_from_policy_evaluation(
    policy_report: PolicyEvaluationReport,
    *,
    policy_id: str | None = None,
) -> ReviewSubject:
    """Build a :class:`ReviewSubject` from a policy evaluation.

    ``policy_id=None`` selects the winning policy.  Unknown policy_ids
    raise :class:`ValueError`.
    """
    target_id = policy_id if policy_id is not None else policy_report.winning_policy_id
    match: PolicyEvaluation | None = None
    for e in policy_report.evaluations:
        if e.policy_id == target_id:
            match = e
            break
    if match is None:
        valid = ", ".join(e.policy_id for e in policy_report.evaluations)
        raise ValueError(
            f"unknown policy_id {target_id!r} in evaluation; valid ids: {valid}"
        )
    return ReviewSubject(
        subject_type="winning_policy",
        subject_id=match.policy_id,
        scenario_id=policy_report.scenario_id,
        candidate_ids=tuple(match.selected_candidate_ids),
        label=f"policy:{match.policy_id}",
        summary=match.reason,
    )


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------


_HASH_LEN = 16  # short, readable, and collision-resistant for our scale.


def _stable_hash(parts: tuple) -> str:
    payload = "\x1f".join(str(p) for p in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:_HASH_LEN]


def _packet_hash(subject: ReviewSubject) -> str:
    return _stable_hash((
        "subject",
        subject.subject_type,
        subject.subject_id,
        subject.scenario_id,
        ",".join(subject.candidate_ids),
        subject.label,
    ))


def _review_id(
    subject: ReviewSubject,
    action: ReviewAction,
    operator_reason: str,
    reviewed_at: datetime,
    operator_id: str | None,
    override_candidate_ids: tuple[str, ...],
) -> str:
    return _stable_hash((
        "review",
        subject.scenario_id,
        action.value,
        subject.subject_type,
        subject.subject_id,
        ",".join(subject.candidate_ids),
        ",".join(override_candidate_ids),
        operator_reason,
        reviewed_at.isoformat(),
        operator_id or "",
    ))


# ---------------------------------------------------------------------------
# Record creator
# ---------------------------------------------------------------------------


_BASE_CAVEATS: tuple[str, ...] = (
    "review record only; no live tasking or sensor command was issued",
    "audit trail entry; downstream execution and platform scheduling are "
    "out of scope for this module",
)


def _coerce_action(action: ReviewAction | str) -> ReviewAction:
    if isinstance(action, ReviewAction):
        return action
    if not isinstance(action, str):
        raise ValueError(
            f"action must be a ReviewAction or str, got {type(action).__name__}"
        )
    try:
        return ReviewAction(action.strip().lower())
    except ValueError:
        valid = ", ".join(a.value for a in ReviewAction)
        raise ValueError(
            f"unknown action {action!r}; valid actions: {valid}"
        ) from None


def _coerce_reviewed_at(reviewed_at: datetime | str | None) -> datetime:
    if reviewed_at is None:
        # Documented exception: this is the only wall-clock call in the
        # module; tests pass an explicit value to remain deterministic.
        return datetime.now(timezone.utc)
    if isinstance(reviewed_at, datetime):
        if reviewed_at.tzinfo is None:
            raise ValueError(
                "reviewed_at datetime must be timezone-aware"
            )
        return reviewed_at.astimezone(timezone.utc)
    if isinstance(reviewed_at, str):
        try:
            parsed = datetime.fromisoformat(reviewed_at)
        except ValueError as exc:
            raise ValueError(
                f"reviewed_at string must be ISO 8601: {reviewed_at!r}"
            ) from exc
        if parsed.tzinfo is None:
            raise ValueError(
                f"reviewed_at string must include timezone: {reviewed_at!r}"
            )
        return parsed.astimezone(timezone.utc)
    raise ValueError(
        f"reviewed_at must be datetime, ISO string, or None; got "
        f"{type(reviewed_at).__name__}"
    )


def create_planner_review_record(
    subject: ReviewSubject,
    *,
    action: ReviewAction | str,
    operator_reason: str,
    reviewed_at: datetime | str | None = None,
    operator_id: str | None = None,
    override_candidate_ids: tuple[str, ...] = (),
) -> PlannerReviewRecord:
    """Validate operator inputs and build a :class:`PlannerReviewRecord`.

    Validation rules:
      - ``operator_reason`` must be non-empty after :py:meth:`str.strip`.
      - ``action`` must be a valid :class:`ReviewAction` (str accepted
        and converted).
      - ``reviewed_at`` must yield a timezone-aware UTC datetime (None
        falls back to ``datetime.now(timezone.utc)``).
      - ``OVERRIDE`` requires at least one ``override_candidate_id``.
      - ``APPROVE`` / ``REJECT`` / ``DEFER`` must NOT carry override ids.
    """
    if not isinstance(subject, ReviewSubject):
        raise ValueError(
            f"subject must be a ReviewSubject, got {type(subject).__name__}"
        )

    if not operator_reason or not operator_reason.strip():
        raise ValueError("operator_reason must be a non-empty string")

    coerced_action = _coerce_action(action)
    coerced_when = _coerce_reviewed_at(reviewed_at)

    overrides = tuple(override_candidate_ids)
    if coerced_action is ReviewAction.OVERRIDE:
        if len(overrides) < 1:
            raise ValueError(
                "OVERRIDE action requires at least one override_candidate_id"
            )
    else:
        if overrides:
            raise ValueError(
                f"{coerced_action.value} action must not carry "
                f"override_candidate_ids"
            )

    pkt_hash = _packet_hash(subject)
    rid = _review_id(
        subject, coerced_action, operator_reason.strip(),
        coerced_when, operator_id, overrides,
    )
    return PlannerReviewRecord(
        review_id=rid,
        scenario_id=subject.scenario_id,
        action=coerced_action,
        subject=subject,
        operator_reason=operator_reason.strip(),
        reviewed_at=coerced_when,
        operator_id=operator_id,
        override_candidate_ids=overrides,
        packet_hash=pkt_hash,
        caveats=_BASE_CAVEATS,
    )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def record_to_dict(record: PlannerReviewRecord) -> dict:
    """Render a record as a deterministic JSON-friendly dict."""
    return {
        "review_id": record.review_id,
        "scenario_id": record.scenario_id,
        "action": record.action.value,
        "subject": {
            "subject_type": record.subject.subject_type,
            "subject_id": record.subject.subject_id,
            "scenario_id": record.subject.scenario_id,
            "candidate_ids": list(record.subject.candidate_ids),
            "label": record.subject.label,
            "summary": record.subject.summary,
        },
        "operator_reason": record.operator_reason,
        "reviewed_at": record.reviewed_at.isoformat(),
        "operator_id": record.operator_id,
        "override_candidate_ids": list(record.override_candidate_ids),
        "packet_hash": record.packet_hash,
        "caveats": list(record.caveats),
    }


def record_to_json(record: PlannerReviewRecord) -> str:
    """JSON encoding of :func:`record_to_dict` in a deterministic key order."""
    return json.dumps(record_to_dict(record), indent=2) + "\n"


# ---------------------------------------------------------------------------
# JSONL ledger
# ---------------------------------------------------------------------------


def append_review_record_jsonl(
    path: str | Path,
    record: PlannerReviewRecord,
) -> ReviewLedgerAppendResult:
    """Append one record to a JSONL ledger file.

    The file is created if missing; the parent directory must already
    exist.  Existing records are never overwritten.
    """
    p = Path(path)
    if not p.parent.exists():
        raise FileNotFoundError(
            f"ledger parent directory does not exist: {p.parent}"
        )
    line = json.dumps(record_to_dict(record)) + "\n"
    with p.open("a", encoding="utf-8") as fh:
        fh.write(line)
    return ReviewLedgerAppendResult(
        path=str(p),
        record_id=record.review_id,
        records_written=1,
        summary=(
            f"appended review record {record.review_id} "
            f"({record.action.value}) for scenario {record.scenario_id}"
        ),
    )


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def format_review_text(record: PlannerReviewRecord) -> str:
    """Plain-text review render."""
    parts: list[str] = []
    title = f"HUMAN-IN-THE-LOOP REVIEW - {record.scenario_id.upper()}"
    parts.append(title + "\n")
    parts.append("=" * len(title) + "\n")

    parts.append("\nReview subject\n")
    parts.append("--------------\n")
    parts.append(f"  {record.subject.subject_type}: {record.subject.subject_id}\n")
    if record.subject.candidate_ids:
        parts.append(
            f"  candidates: {', '.join(record.subject.candidate_ids)}\n"
        )
    else:
        parts.append("  candidates: (none)\n")

    parts.append("\nOperator action\n")
    parts.append("---------------\n")
    parts.append(f"  {record.action.value.upper()}\n")

    parts.append("\nHuman reason\n")
    parts.append("------------\n")
    parts.append(f"  {record.operator_reason}\n")

    if record.action is ReviewAction.OVERRIDE and record.override_candidate_ids:
        parts.append("\nOverride candidates\n")
        parts.append("-------------------\n")
        for cid in record.override_candidate_ids:
            parts.append(f"  - {cid}\n")

    parts.append("\nAudit fields\n")
    parts.append("------------\n")
    parts.append(f"  review_id: {record.review_id}\n")
    parts.append(f"  packet_hash: {record.packet_hash}\n")
    parts.append(f"  reviewed_at: {record.reviewed_at.isoformat()}\n")
    parts.append(
        f"  operator_id: {record.operator_id if record.operator_id else 'anonymous'}\n"
    )

    parts.append("\nCaveats\n")
    parts.append("-------\n")
    for c in record.caveats:
        parts.append(f"  - {c}\n")

    return "".join(parts)


def format_review_markdown(record: PlannerReviewRecord) -> str:
    """Markdown review render — ASCII only, scannable for vault / audit reports."""
    parts: list[str] = []
    parts.append(
        f"# Human-in-the-Loop Review - {record.scenario_id}\n\n"
    )

    parts.append("## Review subject\n\n")
    parts.append(f"- type: {record.subject.subject_type}\n")
    parts.append(f"- subject: {record.subject.subject_id}\n")
    candidates = (
        ", ".join(record.subject.candidate_ids)
        if record.subject.candidate_ids else "(none)"
    )
    parts.append(f"- candidates: {candidates}\n\n")

    parts.append("## Operator action\n\n")
    parts.append(f"**{record.action.value.upper()}**\n\n")

    parts.append("## Human reason\n\n")
    parts.append(f"{record.operator_reason}\n\n")

    if record.action is ReviewAction.OVERRIDE and record.override_candidate_ids:
        parts.append("## Override candidates\n\n")
        for cid in record.override_candidate_ids:
            parts.append(f"- {cid}\n")
        parts.append("\n")

    parts.append("## Audit fields\n\n")
    parts.append(f"- review_id: {record.review_id}\n")
    parts.append(f"- packet_hash: {record.packet_hash}\n")
    parts.append(f"- reviewed_at: {record.reviewed_at.isoformat()}\n")
    parts.append(
        f"- operator_id: {record.operator_id if record.operator_id else 'anonymous'}\n\n"
    )

    parts.append("## Caveats\n\n")
    for c in record.caveats:
        parts.append(f"- {c}\n")

    return "".join(parts)
