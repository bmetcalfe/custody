"""Collection-window scheduler-lite (ADR-0021 Slice 22).

Simulates placing availability-adjusted candidate collect TYPES into
provider-neutral collection windows under simple timing, capacity, and
conflict constraints.

This is schedule feasibility simulation only.  No execution
authorizations are issued, no sensor-control instructions are
generated, no platform-access decisions are claimed, and no orbital
scheduling is performed.

Design constraints
------------------

- Deterministic: no wall-clock, no randomness.
- Pure stdlib: no new runtime dependencies.
- No imports from ``custody.detection.*``, ``custody.ingest.gfw_presence``,
  matcher runtime, or external SDKs.
"""
from __future__ import annotations

import json as _json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations, product
from pathlib import Path

from custody.hypotheses.availability_optimizer import (
    AvailabilityAdjustedPlanItem,
    AvailabilityOptimizationReport,
)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CollectionWindow:
    window_id: str
    scenario_id: str
    candidate_id: str
    start_time: datetime
    end_time: datetime
    capacity_cost: float
    resource_type: str
    quality_score: float
    latency_hours: float
    notes: str | None = None


@dataclass(frozen=True)
class ScheduleConstraint:
    max_total_capacity: float = 1.0
    max_overlapping_collects: int = 1
    earliest_start: datetime | None = None
    latest_end: datetime | None = None
    allowed_resource_types: tuple[str, ...] = ()
    excluded_window_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScheduledCollect:
    candidate_id: str
    label: str
    window_id: str
    start_time: datetime
    end_time: datetime
    capacity_cost: float
    schedule_score: float
    reason: str
    caveats: tuple[str, ...]


@dataclass(frozen=True)
class SchedulePlan:
    scenario_id: str
    scheduled_collects: tuple[ScheduledCollect, ...]
    unscheduled_candidate_ids: tuple[str, ...]
    total_capacity_used: float
    total_schedule_score: float
    constraint: ScheduleConstraint
    summary: str
    caveats: tuple[str, ...]


@dataclass(frozen=True)
class ScheduleReport:
    scenario_id: str
    windows_considered: tuple[CollectionWindow, ...]
    plan: SchedulePlan
    summary: str
    caveats: tuple[str, ...]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


_BASE_CAVEATS: tuple[str, ...] = (
    "scheduler-lite is simulation only",
    "candidate collection windows are provider-neutral metadata",
    "no live tasking or sensor command is issued",
    "no platform access or orbital scheduling is claimed",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _round4(x: float) -> float:
    return round(float(x), 4)


def _parse_dt(raw: object, *, name: str) -> datetime:
    if isinstance(raw, datetime):
        dt = raw
    elif isinstance(raw, str):
        dt = datetime.fromisoformat(raw)
    else:
        raise ValueError(
            f"{name} must be ISO 8601 string or datetime; "
            f"got {type(raw).__name__}"
        )
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _make_aware(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def collection_window_from_mapping(
    mapping: Mapping[str, object],
) -> CollectionWindow:
    """Parse one :class:`CollectionWindow` from a dict."""
    window_id = mapping.get("window_id")
    if not isinstance(window_id, str) or not window_id:
        raise ValueError("window_id is required and must be a non-empty string")

    scenario_id = mapping.get("scenario_id")
    if not isinstance(scenario_id, str) or not scenario_id:
        raise ValueError(
            "scenario_id is required and must be a non-empty string"
        )

    candidate_id = mapping.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError(
            "candidate_id is required and must be a non-empty string"
        )

    start_time = _parse_dt(mapping.get("start_time"), name="start_time")
    end_time = _parse_dt(mapping.get("end_time"), name="end_time")
    if end_time <= start_time:
        raise ValueError(
            f"end_time ({end_time.isoformat()}) must be after "
            f"start_time ({start_time.isoformat()})"
        )

    cap_raw = mapping.get("capacity_cost")
    if cap_raw is None:
        raise ValueError("capacity_cost is required")
    capacity_cost = float(cap_raw)
    if capacity_cost <= 0:
        raise ValueError(
            f"capacity_cost must be > 0; got {capacity_cost}"
        )

    resource_type = mapping.get("resource_type")
    if not isinstance(resource_type, str) or not resource_type:
        raise ValueError(
            "resource_type is required and must be a non-empty string"
        )

    q_raw = mapping.get("quality_score")
    if q_raw is None:
        raise ValueError("quality_score is required")
    quality_score = float(q_raw)
    if not 0.0 <= quality_score <= 1.0:
        raise ValueError(
            f"quality_score must be in [0.0, 1.0]; got {quality_score}"
        )

    lat_raw = mapping.get("latency_hours")
    if lat_raw is None:
        raise ValueError("latency_hours is required")
    latency_hours = float(lat_raw)
    if latency_hours < 0:
        raise ValueError(
            f"latency_hours must be >= 0; got {latency_hours}"
        )

    notes = mapping.get("notes")
    if notes is not None and not isinstance(notes, str):
        raise ValueError("notes must be a string or null")

    return CollectionWindow(
        window_id=window_id,
        scenario_id=scenario_id,
        candidate_id=candidate_id,
        start_time=start_time,
        end_time=end_time,
        capacity_cost=capacity_cost,
        resource_type=resource_type,
        quality_score=quality_score,
        latency_hours=latency_hours,
        notes=notes,
    )


def load_collection_windows(path: str | Path) -> tuple[CollectionWindow, ...]:
    """Load collection windows from a JSON file (list of window dicts)."""
    with open(path, encoding="utf-8") as f:
        raw = _json.load(f)
    if not isinstance(raw, list):
        raise ValueError(
            f"collection-window file at {path} must be a JSON list"
        )
    windows = tuple(collection_window_from_mapping(r) for r in raw)
    return tuple(sorted(windows, key=lambda w: w.window_id))


# ---------------------------------------------------------------------------
# Schedule score
# ---------------------------------------------------------------------------


def _latency_factor(latency_hours: float) -> float:
    if latency_hours <= 6:
        return 1.0
    if latency_hours <= 24:
        return 0.8
    if latency_hours <= 72:
        return 0.6
    return 0.4


def _schedule_score(
    item: AvailabilityAdjustedPlanItem,
    window: CollectionWindow,
) -> float:
    raw = item.adjusted_utility * window.quality_score * _latency_factor(
        window.latency_hours,
    )
    return _round4(_clamp01(raw))


# ---------------------------------------------------------------------------
# Constraint validation
# ---------------------------------------------------------------------------


def _validate_constraint(c: ScheduleConstraint) -> ScheduleConstraint:
    if c.max_total_capacity <= 0:
        raise ValueError(
            f"max_total_capacity must be > 0; got {c.max_total_capacity}"
        )
    if c.max_overlapping_collects < 1:
        raise ValueError(
            f"max_overlapping_collects must be >= 1; got {c.max_overlapping_collects}"
        )
    es = _make_aware(c.earliest_start)
    le = _make_aware(c.latest_end)
    if es is not None and le is not None and le <= es:
        raise ValueError(
            "latest_end must be after earliest_start"
        )
    if es is c.earliest_start and le is c.latest_end:
        return c
    return ScheduleConstraint(
        max_total_capacity=c.max_total_capacity,
        max_overlapping_collects=c.max_overlapping_collects,
        earliest_start=es,
        latest_end=le,
        allowed_resource_types=c.allowed_resource_types,
        excluded_window_ids=c.excluded_window_ids,
    )


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------


def _eligible_windows_for(
    candidate: AvailabilityAdjustedPlanItem,
    windows: tuple[CollectionWindow, ...],
    constraint: ScheduleConstraint,
) -> tuple[CollectionWindow, ...]:
    excluded = set(constraint.excluded_window_ids)
    allowed = (
        set(constraint.allowed_resource_types)
        if constraint.allowed_resource_types else None
    )
    out: list[CollectionWindow] = []
    for w in windows:
        if w.candidate_id != candidate.candidate_id:
            continue
        if w.window_id in excluded:
            continue
        if (
            constraint.earliest_start is not None
            and w.start_time < constraint.earliest_start
        ):
            continue
        if (
            constraint.latest_end is not None
            and w.end_time > constraint.latest_end
        ):
            continue
        if allowed is not None and w.resource_type not in allowed:
            continue
        out.append(w)
    return tuple(out)


def _windows_overlap(a: CollectionWindow, b: CollectionWindow) -> bool:
    return a.start_time < b.end_time and b.start_time < a.end_time


def _max_overlap(windows: tuple[CollectionWindow, ...]) -> int:
    """Maximum number of windows overlapping at any single boundary point."""
    if not windows:
        return 0
    boundaries = sorted({w.start_time for w in windows} | {w.end_time for w in windows})
    best = 0
    for t in boundaries:
        # Count windows whose [start, end) covers t (using boundary-inclusive
        # at start, exclusive at end).  Use a small probe interior to t to
        # capture true overlap rather than just touching.
        count = sum(1 for w in windows if w.start_time <= t < w.end_time)
        if count > best:
            best = count
    return best


def _evaluate_assignment(
    assignment: tuple[tuple[AvailabilityAdjustedPlanItem, CollectionWindow], ...],
    constraint: ScheduleConstraint,
) -> tuple | None:
    """Score an assignment; return tie-break tuple or None if invalid."""
    if not assignment:
        return (0.0, 0.0, datetime.max.replace(tzinfo=timezone.utc), ())
    used_windows = [w for _, w in assignment]
    used_window_ids = {w.window_id for w in used_windows}
    if len(used_window_ids) != len(used_windows):
        return None  # window reuse
    used_candidates = {c.candidate_id for c, _ in assignment}
    if len(used_candidates) != len(assignment):
        return None  # candidate reuse
    total_cost = sum(w.capacity_cost for w in used_windows)
    if total_cost > constraint.max_total_capacity + 1e-9:
        return None
    if (
        _max_overlap(tuple(used_windows))
        > constraint.max_overlapping_collects
    ):
        return None
    total_score = sum(_schedule_score(c, w) for c, w in assignment)
    max_end = max(w.end_time for w in used_windows)
    sorted_ids = tuple(sorted(w.window_id for w in used_windows))
    # Sort key minimization: -total_score (higher better), total_cost
    # (lower better), max_end (earlier better), sorted_ids ascending.
    return (-total_score, total_cost, max_end, sorted_ids)


def _best_assignment(
    candidates: tuple[AvailabilityAdjustedPlanItem, ...],
    eligible: dict[str, tuple[CollectionWindow, ...]],
    constraint: ScheduleConstraint,
) -> tuple[tuple[AvailabilityAdjustedPlanItem, CollectionWindow], ...]:
    """Exhaustive search over all subsets and per-candidate window choices."""
    best_assign: tuple[
        tuple[AvailabilityAdjustedPlanItem, CollectionWindow], ...
    ] = ()
    best_key = _evaluate_assignment((), constraint)
    n = len(candidates)
    for size in range(0, n + 1):
        for subset in combinations(candidates, size):
            options_per_cand: list[tuple[CollectionWindow, ...]] = []
            ok = True
            for cand in subset:
                opts = eligible.get(cand.candidate_id, ())
                if not opts:
                    ok = False
                    break
                options_per_cand.append(opts)
            if not ok:
                continue
            for combo in product(*options_per_cand):
                assignment = tuple(
                    (cand, w) for cand, w in zip(subset, combo)
                )
                key = _evaluate_assignment(assignment, constraint)
                if key is None:
                    continue
                if best_key is None or key < best_key:
                    best_key = key
                    best_assign = assignment
    return best_assign


def _summarize_plan(
    scenario_id: str,
    scheduled: tuple[ScheduledCollect, ...],
    constraint: ScheduleConstraint,
    used_capacity: float,
) -> str:
    return (
        f"scheduled {len(scheduled)} candidate collect(s) for {scenario_id}, "
        f"capacity used {used_capacity:.2f} / "
        f"{constraint.max_total_capacity:.2f}"
    )


def schedule_collects(
    availability_report: AvailabilityOptimizationReport,
    windows: tuple[CollectionWindow, ...],
    *,
    scenario_id: str,
    constraint: ScheduleConstraint | None = None,
) -> ScheduleReport:
    """Simulate scheduling availability-adjusted candidates into windows."""
    resolved = _validate_constraint(
        constraint if constraint is not None else ScheduleConstraint(),
    )

    scenario_windows = tuple(
        w for w in windows if w.scenario_id == scenario_id
    )

    candidates = tuple(
        availability_report.recommended_plan.selected_items
    )

    eligible: dict[str, tuple[CollectionWindow, ...]] = {}
    for c in candidates:
        eligible[c.candidate_id] = _eligible_windows_for(
            c, scenario_windows, resolved,
        )

    assignment = _best_assignment(candidates, eligible, resolved)

    scheduled_ids = {c.candidate_id for c, _ in assignment}
    scheduled_collects: list[ScheduledCollect] = []
    for cand, w in assignment:
        score = _schedule_score(cand, w)
        scheduled_collects.append(ScheduledCollect(
            candidate_id=cand.candidate_id,
            label=cand.label,
            window_id=w.window_id,
            start_time=w.start_time,
            end_time=w.end_time,
            capacity_cost=w.capacity_cost,
            schedule_score=score,
            reason=(
                f"window {w.window_id} (resource {w.resource_type}, "
                f"quality {w.quality_score:.2f}, latency {w.latency_hours:.1f}h)"
            ),
            caveats=(),
        ))
    scheduled_collects.sort(key=lambda s: (s.start_time, s.candidate_id))

    unscheduled = tuple(
        sorted(
            c.candidate_id
            for c in candidates
            if c.candidate_id not in scheduled_ids
        )
    )

    total_capacity_used = _round4(
        sum(s.capacity_cost for s in scheduled_collects)
    )
    total_schedule_score = _round4(
        sum(s.schedule_score for s in scheduled_collects)
    )

    plan_summary = _summarize_plan(
        scenario_id, tuple(scheduled_collects), resolved, total_capacity_used,
    )
    plan = SchedulePlan(
        scenario_id=scenario_id,
        scheduled_collects=tuple(scheduled_collects),
        unscheduled_candidate_ids=unscheduled,
        total_capacity_used=total_capacity_used,
        total_schedule_score=total_schedule_score,
        constraint=resolved,
        summary=plan_summary,
        caveats=_BASE_CAVEATS,
    )

    report_summary = (
        f"scheduler-lite considered {len(scenario_windows)} window(s) for "
        f"{scenario_id}; {len(scheduled_collects)} scheduled, "
        f"{len(unscheduled)} unscheduled"
    )
    return ScheduleReport(
        scenario_id=scenario_id,
        windows_considered=scenario_windows,
        plan=plan,
        summary=report_summary,
        caveats=_BASE_CAVEATS,
    )


# ---------------------------------------------------------------------------
# Serializers
# ---------------------------------------------------------------------------


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt is not None else None


def _window_to_dict(w: CollectionWindow) -> dict:
    return {
        "window_id": w.window_id,
        "scenario_id": w.scenario_id,
        "candidate_id": w.candidate_id,
        "start_time": _iso(w.start_time),
        "end_time": _iso(w.end_time),
        "capacity_cost": w.capacity_cost,
        "resource_type": w.resource_type,
        "quality_score": w.quality_score,
        "latency_hours": w.latency_hours,
        "notes": w.notes,
    }


def _scheduled_to_dict(s: ScheduledCollect) -> dict:
    return {
        "candidate_id": s.candidate_id,
        "label": s.label,
        "window_id": s.window_id,
        "start_time": _iso(s.start_time),
        "end_time": _iso(s.end_time),
        "capacity_cost": s.capacity_cost,
        "schedule_score": s.schedule_score,
        "reason": s.reason,
        "caveats": list(s.caveats),
    }


def _constraint_to_dict(c: ScheduleConstraint) -> dict:
    return {
        "max_total_capacity": c.max_total_capacity,
        "max_overlapping_collects": c.max_overlapping_collects,
        "earliest_start": _iso(c.earliest_start),
        "latest_end": _iso(c.latest_end),
        "allowed_resource_types": list(c.allowed_resource_types),
        "excluded_window_ids": list(c.excluded_window_ids),
    }


def _plan_to_dict(plan: SchedulePlan) -> dict:
    return {
        "scenario_id": plan.scenario_id,
        "scheduled_collects": [
            _scheduled_to_dict(s) for s in plan.scheduled_collects
        ],
        "unscheduled_candidate_ids": list(plan.unscheduled_candidate_ids),
        "total_capacity_used": plan.total_capacity_used,
        "total_schedule_score": plan.total_schedule_score,
        "constraint": _constraint_to_dict(plan.constraint),
        "summary": plan.summary,
        "caveats": list(plan.caveats),
    }


def schedule_report_to_dict(report: ScheduleReport) -> dict:
    return {
        "scenario_id": report.scenario_id,
        "windows_considered": [
            _window_to_dict(w) for w in report.windows_considered
        ],
        "plan": _plan_to_dict(report.plan),
        "summary": report.summary,
        "caveats": list(report.caveats),
    }


def schedule_report_to_json(report: ScheduleReport) -> str:
    return _json.dumps(schedule_report_to_dict(report), indent=2)


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def _hr(title: str) -> str:
    underline = "-" * max(len(title), 3)
    return f"\n{title}\n{underline}\n"


def format_schedule_text(report: ScheduleReport) -> str:
    parts: list[str] = []
    title = f"COLLECTION-WINDOW SCHEDULER-LITE - {report.scenario_id.upper()}"
    parts.append(title + "\n")
    parts.append("=" * len(title) + "\n")

    plan = report.plan
    constraint = plan.constraint

    parts.append(_hr("Constraints"))
    parts.append(
        f"  max total capacity: {constraint.max_total_capacity:.2f}\n"
    )
    parts.append(
        f"  max overlapping collects: {constraint.max_overlapping_collects}\n"
    )
    if constraint.earliest_start is not None:
        parts.append(
            f"  earliest start: {constraint.earliest_start.isoformat()}\n"
        )
    if constraint.latest_end is not None:
        parts.append(
            f"  latest end: {constraint.latest_end.isoformat()}\n"
        )
    if constraint.allowed_resource_types:
        joined = ", ".join(constraint.allowed_resource_types)
        parts.append(f"  allowed resource types: {joined}\n")
    if constraint.excluded_window_ids:
        joined = ", ".join(constraint.excluded_window_ids)
        parts.append(f"  excluded window ids: {joined}\n")

    parts.append(_hr(f"Windows considered: {len(report.windows_considered)}"))
    for w in report.windows_considered:
        parts.append(
            f"  {w.window_id:32s}  candidate={w.candidate_id:24s}  "
            f"resource={w.resource_type:8s}  quality={w.quality_score:.2f}  "
            f"latency={w.latency_hours:5.1f}h\n"
        )

    parts.append(_hr("Scheduled collects"))
    if not plan.scheduled_collects:
        parts.append("  (none)\n")
    else:
        for s in plan.scheduled_collects:
            parts.append(
                f"  {s.candidate_id:24s}  window={s.window_id:32s}  "
                f"score={s.schedule_score:.2f}  "
                f"start={s.start_time.isoformat()}\n"
            )

    parts.append(_hr("Unscheduled candidates"))
    if not plan.unscheduled_candidate_ids:
        parts.append("  (none)\n")
    else:
        for cid in plan.unscheduled_candidate_ids:
            parts.append(f"  {cid}\n")

    parts.append(_hr("Schedule summary"))
    parts.append(f"  {plan.summary}\n")
    parts.append(
        f"  total schedule score: {plan.total_schedule_score:.2f}\n"
    )

    parts.append(_hr("Caveats"))
    for c in plan.caveats:
        parts.append(f"  - {c}\n")

    return "".join(parts)


def format_schedule_markdown(report: ScheduleReport) -> str:
    lines: list[str] = []
    name = report.scenario_id.capitalize()
    lines.append(f"# Collection-Window Scheduler-Lite - {name}")
    lines.append("")

    plan = report.plan
    constraint = plan.constraint

    lines.append("## Constraints")
    lines.append("")
    lines.append(
        f"- max total capacity: {constraint.max_total_capacity:.2f}"
    )
    lines.append(
        f"- max overlapping collects: {constraint.max_overlapping_collects}"
    )
    if constraint.earliest_start is not None:
        lines.append(
            f"- earliest start: {constraint.earliest_start.isoformat()}"
        )
    if constraint.latest_end is not None:
        lines.append(
            f"- latest end: {constraint.latest_end.isoformat()}"
        )
    if constraint.allowed_resource_types:
        joined = ", ".join(constraint.allowed_resource_types)
        lines.append(f"- allowed resource types: {joined}")
    if constraint.excluded_window_ids:
        joined = ", ".join(constraint.excluded_window_ids)
        lines.append(f"- excluded window ids: {joined}")
    lines.append("")

    lines.append(f"## Windows considered: {len(report.windows_considered)}")
    lines.append("")
    if report.windows_considered:
        lines.append(
            "| window_id | candidate | resource | quality | latency (h) |"
        )
        lines.append("|---|---|---|---|---|")
        for w in report.windows_considered:
            lines.append(
                f"| {w.window_id} | {w.candidate_id} | "
                f"{w.resource_type} | {w.quality_score:.2f} | "
                f"{w.latency_hours:.1f} |"
            )
    else:
        lines.append("(none)")
    lines.append("")

    lines.append("## Scheduled collects")
    lines.append("")
    if not plan.scheduled_collects:
        lines.append("(none)")
    else:
        lines.append("| candidate | window | score | start | end |")
        lines.append("|---|---|---|---|---|")
        for s in plan.scheduled_collects:
            lines.append(
                f"| {s.candidate_id} | {s.window_id} | "
                f"{s.schedule_score:.2f} | {s.start_time.isoformat()} | "
                f"{s.end_time.isoformat()} |"
            )
    lines.append("")

    lines.append("## Unscheduled candidates")
    lines.append("")
    if not plan.unscheduled_candidate_ids:
        lines.append("(none)")
    else:
        for cid in plan.unscheduled_candidate_ids:
            lines.append(f"- {cid}")
    lines.append("")

    lines.append("## Schedule summary")
    lines.append("")
    lines.append(f"- {plan.summary}")
    lines.append(
        f"- total schedule score: {plan.total_schedule_score:.2f}"
    )
    lines.append("")

    lines.append("## Caveats")
    lines.append("")
    for c in plan.caveats:
        lines.append(f"- {c}")
    lines.append("")
    return "\n".join(lines)
