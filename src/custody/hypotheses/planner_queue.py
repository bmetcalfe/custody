"""Planner work queue — operational capstone (ADR-0021 Slice 14).

Composes the upstream Slice 4-13 outputs into a per-scenario
:class:`PlannerQueueItem` and ranks the items into a
:class:`PlannerQueue` that answers "which scenario needs attention
first, why, and what should the operator do next?"

This is a **decision-support queue only**.  No record in this module
issues a tasking order, commands a sensor, executes a collect, or
authorizes downstream action.  Every queue carries the explicit non-
claim caveat ``"queue is decision support only; no live tasking or
sensor command is issued"`` so downstream readers cannot misinterpret a
queue item as an execution authorization.

Scope guardrails
----------------

- Pure standard library (``hashlib``, ``json``, ``datetime``,
  ``pathlib``, ``enum``, ``dataclasses``).
- No imports from ``custody.detection.*``, ``custody.ingest.gfw_presence``,
  ``custody.fusion.tracker``, ``custody.taskrecommendation``, or
  Sentinel SDKs.
- The only wall-clock calls are the optional ``generated_at=None`` and
  ``as_of=None`` fallbacks; tests pass explicit values to keep output
  deterministic.

Status semantics
----------------

If a review record is present for the scenario:

  - APPROVE   → ``APPROVED``
  - REJECT    → ``REJECTED``
  - DEFER     → ``DEFERRED``
  - OVERRIDE  → ``OVERRIDDEN``

Otherwise:

  - LOST or STALE health             → ``NEEDS_EVIDENCE``
  - HEALTHY with no primary ambiguity → ``RESOLVED``
  - any other case                   → ``PENDING_REVIEW``
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from custody.hypotheses.collection_value import CollectionRecommendation
from custody.hypotheses.counterfactual import CounterfactualReport
from custody.hypotheses.custody_health import (
    CustodyHealthStatus,
    HypothesisCustodyHealth,
)
from custody.hypotheses.mission_value import MissionValueReport
from custody.hypotheses.optimizer import PlanOptimizationReport
from custody.hypotheses.planner_review import (
    PlannerReviewRecord,
    ReviewAction,
    ReviewSubject,
)
from custody.hypotheses.policy_eval import PolicyEvaluationReport


# ---------------------------------------------------------------------------
# Enum + dataclasses
# ---------------------------------------------------------------------------


class PlannerQueueStatus(Enum):
    PENDING_REVIEW = "pending_review"
    NEEDS_EVIDENCE = "needs_evidence"
    APPROVED = "approved"
    REJECTED = "rejected"
    DEFERRED = "deferred"
    OVERRIDDEN = "overridden"
    RESOLVED = "resolved"


@dataclass(frozen=True)
class PlannerQueueItem:
    """One scenario's queue entry — what's the state, why, what's next."""
    item_id: str
    scenario_id: str
    status: PlannerQueueStatus
    priority_score: float
    health_status: CustodyHealthStatus
    health_score: float
    primary_ambiguity: tuple[str, str] | None
    recommended_candidate_ids: tuple[str, ...]
    planning_utility: float
    mission_value_proxy: float
    expected_ambiguity_resolution: float
    expected_health_score_delta: float
    latest_review_action: str | None
    latest_review_id: str | None
    reason: str
    next_action: str
    caveats: tuple[str, ...]


@dataclass(frozen=True)
class PlannerQueue:
    """Ranked work queue across scenarios."""
    generated_at: datetime
    items: tuple[PlannerQueueItem, ...]
    summary: str
    caveats: tuple[str, ...]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


_HEALTH_URGENCY: dict[CustodyHealthStatus, float] = {
    CustodyHealthStatus.LOST: 0.95,
    CustodyHealthStatus.STALE: 0.85,
    CustodyHealthStatus.AMBIGUOUS: 0.80,
    CustodyHealthStatus.DEGRADED: 0.55,
    CustodyHealthStatus.HEALTHY: 0.20,
}

_REVIEW_MODIFIER: dict[PlannerQueueStatus, float] = {
    PlannerQueueStatus.PENDING_REVIEW: 0.10,
    PlannerQueueStatus.NEEDS_EVIDENCE: 0.08,
    PlannerQueueStatus.DEFERRED: -0.10,
    PlannerQueueStatus.OVERRIDDEN: -0.20,
    PlannerQueueStatus.APPROVED: -0.25,
    PlannerQueueStatus.REJECTED: -0.35,
    PlannerQueueStatus.RESOLVED: -0.40,
}

# Used as a tie-break when two items share priority_score.
_STATUS_ORDER: dict[PlannerQueueStatus, int] = {
    PlannerQueueStatus.PENDING_REVIEW: 0,
    PlannerQueueStatus.NEEDS_EVIDENCE: 1,
    PlannerQueueStatus.DEFERRED: 2,
    PlannerQueueStatus.OVERRIDDEN: 3,
    PlannerQueueStatus.APPROVED: 4,
    PlannerQueueStatus.REJECTED: 5,
    PlannerQueueStatus.RESOLVED: 6,
}

_BASE_CAVEATS: tuple[str, ...] = (
    "queue is decision support only; no live tasking or sensor command is issued",
    "planning utility is a deterministic proxy, not a financial model",
    "review status reflects operator audit records; downstream execution is "
    "out of scope for this queue",
)

_REVIEW_ACTION_TO_STATUS: dict[ReviewAction, PlannerQueueStatus] = {
    ReviewAction.APPROVE: PlannerQueueStatus.APPROVED,
    ReviewAction.REJECT: PlannerQueueStatus.REJECTED,
    ReviewAction.DEFER: PlannerQueueStatus.DEFERRED,
    ReviewAction.OVERRIDE: PlannerQueueStatus.OVERRIDDEN,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _round4(x: float) -> float:
    return round(float(x), 4)


def _stable_hash(parts: tuple) -> str:
    payload = "\x1f".join(str(p) for p in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Status mapping
# ---------------------------------------------------------------------------


def record_status_from_review(
    review_record: PlannerReviewRecord,
) -> PlannerQueueStatus:
    """Map a review record's action to the corresponding queue status."""
    return _REVIEW_ACTION_TO_STATUS[review_record.action]


def _derive_status(
    health: HypothesisCustodyHealth,
    review_record: PlannerReviewRecord | None,
) -> PlannerQueueStatus:
    if review_record is not None:
        return record_status_from_review(review_record)
    if health.status in (CustodyHealthStatus.LOST, CustodyHealthStatus.STALE):
        return PlannerQueueStatus.NEEDS_EVIDENCE
    if (
        health.status is CustodyHealthStatus.HEALTHY
        and not health.ambiguity_pairs
    ):
        return PlannerQueueStatus.RESOLVED
    return PlannerQueueStatus.PENDING_REVIEW


# ---------------------------------------------------------------------------
# JSONL ledger loader
# ---------------------------------------------------------------------------


def _record_from_dict(d: dict) -> PlannerReviewRecord:
    """Reverse of ``planner_review.record_to_dict``."""
    s = d["subject"]
    subject = ReviewSubject(
        subject_type=s["subject_type"],
        subject_id=s["subject_id"],
        scenario_id=s["scenario_id"],
        candidate_ids=tuple(s["candidate_ids"]),
        label=s["label"],
        summary=s["summary"],
    )
    action = ReviewAction(d["action"])
    reviewed_at = datetime.fromisoformat(d["reviewed_at"])
    return PlannerReviewRecord(
        review_id=d["review_id"],
        scenario_id=d["scenario_id"],
        action=action,
        subject=subject,
        operator_reason=d["operator_reason"],
        reviewed_at=reviewed_at,
        operator_id=d.get("operator_id"),
        override_candidate_ids=tuple(d.get("override_candidate_ids", [])),
        packet_hash=d["packet_hash"],
        caveats=tuple(d.get("caveats", [])),
    )


def load_review_ledger_jsonl(
    path: str | Path,
) -> tuple[PlannerReviewRecord, ...]:
    """Parse a Slice 13 JSONL ledger; do not write or mutate.

    Blank lines are ignored.  Malformed JSON or a record missing required
    fields raises :class:`ValueError` naming the offending line index.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"review ledger does not exist: {p}")
    out: list[PlannerReviewRecord] = []
    for i, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"malformed JSON on line {i} of {p}: {exc.msg}"
            ) from exc
        try:
            out.append(_record_from_dict(d))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"malformed review record on line {i} of {p}: {exc}"
            ) from exc
    return tuple(out)


def select_latest_review_for_scenario(
    records: tuple[PlannerReviewRecord, ...] | list,
    scenario_id: str,
) -> PlannerReviewRecord | None:
    """Return the most-recent record for ``scenario_id`` by ``reviewed_at``."""
    matches = [r for r in records if r.scenario_id == scenario_id]
    if not matches:
        return None
    return max(matches, key=lambda r: r.reviewed_at)


# ---------------------------------------------------------------------------
# Reason / next-action templates
# ---------------------------------------------------------------------------


def _ambiguity_phrase(pair: tuple[str, str] | None) -> str:
    if pair is None:
        return "no primary ambiguity reported"
    return f"primary ambiguity {pair[0]} vs {pair[1]}"


def _reason_and_next_action(
    status: PlannerQueueStatus,
    health: HypothesisCustodyHealth,
    review: PlannerReviewRecord | None,
) -> tuple[str, str]:
    pair = (
        sorted(health.ambiguity_pairs)[0]
        if health.ambiguity_pairs else None
    )
    if status is PlannerQueueStatus.PENDING_REVIEW:
        return (
            f"custody is {health.status.value} with "
            f"{_ambiguity_phrase(pair)}; awaiting operator review",
            "review recommended candidate collect plan",
        )
    if status is PlannerQueueStatus.NEEDS_EVIDENCE:
        return (
            f"custody is {health.status.value}; insufficient evidence to "
            f"rank targeted collects",
            "acquire broad confirmation evidence before targeted optimization",
        )
    if status is PlannerQueueStatus.RESOLVED:
        return (
            "custody is healthy with no primary ambiguity; no review needed",
            "continue monitoring; no targeted action required",
        )
    if review is None:  # defensive; the four review-derived statuses below
        return (status.value, "no action determined")
    if status is PlannerQueueStatus.APPROVED:
        return (
            f"custody plan approved by operator (review {review.review_id})",
            "no sensor command issued by Custody; external tasking system "
            "would execute if authorized",
        )
    if status is PlannerQueueStatus.DEFERRED:
        return (
            f"custody plan deferred by operator (review {review.review_id})",
            "revisit when new evidence or collection window is available",
        )
    if status is PlannerQueueStatus.REJECTED:
        return (
            f"custody plan rejected by operator (review {review.review_id})",
            "regenerate plan after upstream evidence is updated",
        )
    if status is PlannerQueueStatus.OVERRIDDEN:
        ids = ", ".join(review.override_candidate_ids) or "(none)"
        return (
            f"custody plan overridden by operator with {ids} "
            f"(review {review.review_id})",
            "monitor outcome of overridden candidate; re-evaluate next cycle",
        )
    return (status.value, "no action determined")  # pragma: no cover


# ---------------------------------------------------------------------------
# Item builder
# ---------------------------------------------------------------------------


def _select_top_mission_value(
    mission_value_report: MissionValueReport,
    candidate_ids: tuple[str, ...],
) -> float:
    """Mission value proxy = the highest mission-value total among the
    candidates already on the recommended plan, falling back to the
    report's top assessment."""
    by_id = {a.candidate_id: a.total_value for a in mission_value_report.ranked_assessments}
    if candidate_ids:
        values = [by_id[cid] for cid in candidate_ids if cid in by_id]
        if values:
            return max(values)
    if mission_value_report.ranked_assessments:
        return max(a.total_value for a in mission_value_report.ranked_assessments)
    return 0.0


def _select_expected_amb_res(
    counterfactual_report: CounterfactualReport,
    candidate_ids: tuple[str, ...],
) -> tuple[float, float]:
    """Pick the expected_ambiguity_resolution and expected_health_score_delta
    for the recommended candidates' best contributor."""
    by_id = {
        a.candidate_id: (
            a.expected_ambiguity_resolution,
            a.expected_health_score_delta,
        )
        for a in counterfactual_report.ranked_assessments
    }
    pool = [by_id[cid] for cid in candidate_ids if cid in by_id]
    if not pool and counterfactual_report.ranked_assessments:
        # Fall back to the top counterfactual assessment.
        top = counterfactual_report.ranked_assessments[0]
        return (
            top.expected_ambiguity_resolution,
            top.expected_health_score_delta,
        )
    if not pool:
        return 0.0, 0.0
    amb = max(p[0] for p in pool)
    delta = max(p[1] for p in pool)
    return amb, delta


def _priority_score(
    status: PlannerQueueStatus,
    health_status: CustodyHealthStatus,
    planning_utility: float,
    expected_ambiguity_resolution: float,
    mission_value_proxy: float,
) -> float:
    base = (
        0.40 * _HEALTH_URGENCY.get(health_status, 0.50)
        + 0.25 * _clamp01(planning_utility)
        + 0.20 * _clamp01(expected_ambiguity_resolution)
        + 0.15 * _clamp01(mission_value_proxy)
    )
    score = base + _REVIEW_MODIFIER.get(status, 0.0)
    return _round4(_clamp01(score))


def _item_id(
    scenario_id: str,
    status: PlannerQueueStatus,
    candidate_ids: tuple[str, ...],
    review_id: str | None,
) -> str:
    return _stable_hash((
        "queue_item",
        scenario_id,
        status.value,
        ",".join(candidate_ids),
        review_id or "",
    ))


def create_queue_item(
    *,
    scenario_id: str,
    health: HypothesisCustodyHealth,
    recommendation: CollectionRecommendation,
    mission_value_report: MissionValueReport,
    counterfactual_report: CounterfactualReport,
    optimization_report: PlanOptimizationReport,
    policy_report: PolicyEvaluationReport | None = None,
    review_record: PlannerReviewRecord | None = None,
    as_of: datetime | None = None,  # reserved for future use; not used here
) -> PlannerQueueItem:
    """Compose one :class:`PlannerQueueItem` from the upstream pipeline."""
    status = _derive_status(health, review_record)
    plan = optimization_report.recommended_plan
    candidate_ids = tuple(p.candidate_id for p in plan.selected_items)
    mv_proxy = _select_top_mission_value(mission_value_report, candidate_ids)
    amb_res, h_delta = _select_expected_amb_res(
        counterfactual_report, candidate_ids,
    )
    primary_pair = (
        sorted(health.ambiguity_pairs)[0]
        if health.ambiguity_pairs else None
    )
    planning_utility = plan.total_value
    priority = _priority_score(
        status,
        health.status,
        planning_utility,
        amb_res,
        mv_proxy,
    )
    reason, next_action = _reason_and_next_action(status, health, review_record)
    item = PlannerQueueItem(
        item_id=_item_id(
            scenario_id, status, candidate_ids,
            review_record.review_id if review_record is not None else None,
        ),
        scenario_id=scenario_id,
        status=status,
        priority_score=priority,
        health_status=health.status,
        health_score=_round4(health.score),
        primary_ambiguity=primary_pair,
        recommended_candidate_ids=candidate_ids,
        planning_utility=_round4(planning_utility),
        mission_value_proxy=_round4(mv_proxy),
        expected_ambiguity_resolution=_round4(amb_res),
        expected_health_score_delta=_round4(h_delta),
        latest_review_action=(
            review_record.action.value if review_record is not None else None
        ),
        latest_review_id=(
            review_record.review_id if review_record is not None else None
        ),
        reason=reason,
        next_action=next_action,
        caveats=_BASE_CAVEATS,
    )
    return item


# ---------------------------------------------------------------------------
# Queue builder
# ---------------------------------------------------------------------------


def _sort_key(item: PlannerQueueItem) -> tuple[float, int, str]:
    return (
        -item.priority_score,
        _STATUS_ORDER.get(item.status, 99),
        item.scenario_id,
    )


def build_planner_queue(
    items: list[PlannerQueueItem] | tuple[PlannerQueueItem, ...],
    *,
    generated_at: datetime | None = None,
) -> PlannerQueue:
    """Sort items and wrap them in a :class:`PlannerQueue`.

    ``generated_at=None`` falls back to ``datetime.now(timezone.utc)``;
    tests pass an explicit value for deterministic output.
    """
    if generated_at is None:
        generated_at = datetime.now(timezone.utc)
    elif generated_at.tzinfo is None:
        raise ValueError("generated_at must be timezone-aware")

    sorted_items = tuple(sorted(items, key=_sort_key))
    if sorted_items:
        top = sorted_items[0]
        summary = (
            f"{len(sorted_items)} item(s); top scenario {top.scenario_id!r} "
            f"(priority {top.priority_score:.2f}, status "
            f"{top.status.value})"
        )
    else:
        summary = "0 items in queue"
    return PlannerQueue(
        generated_at=generated_at,
        items=sorted_items,
        summary=summary,
        caveats=_BASE_CAVEATS,
    )


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _item_to_dict(item: PlannerQueueItem) -> dict:
    return {
        "item_id": item.item_id,
        "scenario_id": item.scenario_id,
        "status": item.status.value,
        "priority_score": item.priority_score,
        "health_status": item.health_status.value,
        "health_score": item.health_score,
        "primary_ambiguity": (
            list(item.primary_ambiguity)
            if item.primary_ambiguity is not None else None
        ),
        "recommended_candidate_ids": list(item.recommended_candidate_ids),
        "planning_utility": item.planning_utility,
        "mission_value_proxy": item.mission_value_proxy,
        "expected_ambiguity_resolution": item.expected_ambiguity_resolution,
        "expected_health_score_delta": item.expected_health_score_delta,
        "latest_review_action": item.latest_review_action,
        "latest_review_id": item.latest_review_id,
        "reason": item.reason,
        "next_action": item.next_action,
        "caveats": list(item.caveats),
    }


def format_queue_json(queue: PlannerQueue) -> str:
    """Deterministic JSON encoding of a queue."""
    return json.dumps({
        "generated_at": _iso(queue.generated_at),
        "summary": queue.summary,
        "items": [_item_to_dict(i) for i in queue.items],
        "caveats": list(queue.caveats),
    }, indent=2) + "\n"


def format_queue_text(queue: PlannerQueue) -> str:
    """Plain-text queue render."""
    parts: list[str] = []
    parts.append("PLANNER WORK QUEUE\n")
    parts.append("==================\n")
    parts.append(f"Generated at: {_iso(queue.generated_at)}\n")

    parts.append("\nRanked work items\n")
    parts.append("-----------------\n")
    if not queue.items:
        parts.append("  (no items)\n")
    for item in queue.items:
        parts.append(f"\n  {item.scenario_id}\n")
        parts.append(f"    Status: {item.status.value}\n")
        parts.append(f"    Priority: {item.priority_score:.2f}\n")
        parts.append(
            f"    Health: {item.health_status.value} "
            f"{item.health_score:.2f}\n"
        )
        if item.primary_ambiguity is not None:
            a, b = item.primary_ambiguity
            parts.append(f"    Primary ambiguity: {a} vs {b}\n")
        else:
            parts.append("    Primary ambiguity: (none)\n")
        if item.recommended_candidate_ids:
            parts.append(
                f"    Recommended candidates: "
                f"{', '.join(item.recommended_candidate_ids)}\n"
            )
        else:
            parts.append("    Recommended candidates: (none)\n")
        parts.append(f"    Planning utility: {item.planning_utility:.2f}\n")
        parts.append(
            f"    Mission value proxy: {item.mission_value_proxy:.2f}\n"
        )
        if item.latest_review_action is not None:
            parts.append(
                f"    Latest review: {item.latest_review_action} "
                f"({item.latest_review_id})\n"
            )
        parts.append(f"    Reason: {item.reason}\n")
        parts.append(f"    Next action: {item.next_action}\n")

    parts.append("\nQueue summary\n")
    parts.append("-------------\n")
    parts.append(f"  {queue.summary}\n")

    parts.append("\nCaveats\n")
    parts.append("-------\n")
    for c in queue.caveats:
        parts.append(f"  - {c}\n")

    return "".join(parts)


def format_queue_markdown(queue: PlannerQueue) -> str:
    """Markdown queue render — vault-friendly."""
    parts: list[str] = []
    parts.append("# Planner Work Queue\n\n")
    parts.append(f"_Generated at {_iso(queue.generated_at)}_\n\n")

    parts.append("## Ranked work items\n\n")
    if not queue.items:
        parts.append("(no items)\n\n")
    for item in queue.items:
        parts.append(
            f"### {item.scenario_id} — {item.status.value} "
            f"(priority {item.priority_score:.2f})\n\n"
        )
        parts.append(
            f"- Health: {item.health_status.value} ({item.health_score:.2f})\n"
        )
        if item.primary_ambiguity is not None:
            a, b = item.primary_ambiguity
            parts.append(f"- Primary ambiguity: {a} vs {b}\n")
        else:
            parts.append("- Primary ambiguity: (none)\n")
        if item.recommended_candidate_ids:
            parts.append(
                f"- Recommended candidates: "
                f"{', '.join(item.recommended_candidate_ids)}\n"
            )
        else:
            parts.append("- Recommended candidates: (none)\n")
        parts.append(f"- Planning utility: {item.planning_utility:.2f}\n")
        parts.append(
            f"- Mission value proxy: {item.mission_value_proxy:.2f}\n"
        )
        if item.latest_review_action is not None:
            parts.append(
                f"- Latest review: {item.latest_review_action} "
                f"({item.latest_review_id})\n"
            )
        parts.append(f"- Reason: {item.reason}\n")
        parts.append(f"- Next action: {item.next_action}\n\n")

    parts.append("## Queue summary\n\n")
    parts.append(f"{queue.summary}\n\n")

    parts.append("## Caveats\n\n")
    for c in queue.caveats:
        parts.append(f"- {c}\n")

    return "".join(parts)
