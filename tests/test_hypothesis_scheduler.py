"""Tests for :mod:`custody.hypotheses.scheduler` (ADR-0021 Slice 22)."""
from __future__ import annotations

import ast
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from custody.hypotheses.availability_optimizer import (
    AvailabilityAdjustedPlanItem,
    AvailabilityOptimizationReport,
    AvailabilityOptimizedPlan,
    AvailabilityOptimizerConstraint,
)
from custody.hypotheses.scheduler import (
    CollectionWindow,
    ScheduleConstraint,
    ScheduledCollect,
    SchedulePlan,
    ScheduleReport,
    collection_window_from_mapping,
    format_schedule_markdown,
    format_schedule_text,
    load_collection_windows,
    schedule_collects,
    schedule_report_to_dict,
    schedule_report_to_json,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TENNENT_FIXTURE = (
    REPO_ROOT / "tests" / "fixtures" / "schedule"
    / "tennent_collection_windows.json"
)
WHITSUN_FIXTURE = (
    REPO_ROOT / "tests" / "fixtures" / "schedule"
    / "whitsun_collection_windows.json"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_window(**overrides) -> dict:
    base = {
        "window_id": "test-w1",
        "scenario_id": "tennent",
        "candidate_id": "optical_context",
        "start_time": "2023-08-14T14:00:00+00:00",
        "end_time": "2023-08-14T15:00:00+00:00",
        "capacity_cost": 0.3,
        "resource_type": "optical",
        "quality_score": 0.85,
        "latency_hours": 4.0,
        "notes": None,
    }
    base.update(overrides)
    return base


def _make_item(
    cid: str, label: str, adjusted_utility: float = 0.6,
) -> AvailabilityAdjustedPlanItem:
    return AvailabilityAdjustedPlanItem(
        candidate_id=cid,
        label=label,
        cost=0.5,
        base_planning_utility=adjusted_utility,
        feasibility_score=1.0,
        availability_status="feasible",
        adjusted_utility=adjusted_utility,
        reason="test",
        caveats=(),
    )


def _make_avail_report(
    items: tuple[AvailabilityAdjustedPlanItem, ...],
    *,
    scenario_id: str = "tennent",
) -> AvailabilityOptimizationReport:
    constraint = AvailabilityOptimizerConstraint(budget=10.0, max_collects=10)
    plan = AvailabilityOptimizedPlan(
        strategy="exhaustive",
        selected_items=items,
        total_cost=sum(i.cost for i in items),
        total_base_planning_utility=sum(i.base_planning_utility for i in items),
        total_adjusted_utility=sum(i.adjusted_utility for i in items),
        constraint=constraint,
        summary="x",
        caveats=(),
    )
    empty = AvailabilityOptimizedPlan(
        strategy="other", selected_items=(), total_cost=0.0,
        total_base_planning_utility=0.0, total_adjusted_utility=0.0,
        constraint=constraint, summary="x", caveats=(),
    )
    return AvailabilityOptimizationReport(
        scenario_id=scenario_id,
        base_plan_summary="b",
        availability_summary="a",
        adjusted_candidate_pool=(),
        exhaustive_plan=plan,
        greedy_plan=empty,
        recommended_plan=plan,
        comparison_summary="c",
    )


def _make_window(
    window_id: str, candidate_id: str,
    *,
    scenario_id: str = "tennent",
    start: datetime = datetime(2023, 8, 14, 14, 0, tzinfo=timezone.utc),
    duration_minutes: int = 60,
    capacity_cost: float = 0.3,
    resource_type: str = "optical",
    quality_score: float = 0.85,
    latency_hours: float = 4.0,
) -> CollectionWindow:
    return CollectionWindow(
        window_id=window_id,
        scenario_id=scenario_id,
        candidate_id=candidate_id,
        start_time=start,
        end_time=start + timedelta(minutes=duration_minutes),
        capacity_cost=capacity_cost,
        resource_type=resource_type,
        quality_score=quality_score,
        latency_hours=latency_hours,
    )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_collection_window_from_mapping_parses_valid_window() -> None:
    w = collection_window_from_mapping(_base_window())
    assert w.window_id == "test-w1"
    assert w.candidate_id == "optical_context"
    assert w.capacity_cost == 0.3


def test_iso_timestamps_become_timezone_aware_utc() -> None:
    w = collection_window_from_mapping(
        _base_window(start_time="2023-08-14T14:00:00"),
    )
    assert w.start_time.tzinfo is not None


def test_naive_timestamps_treated_as_utc() -> None:
    w = collection_window_from_mapping(
        _base_window(start_time="2023-08-14T14:00:00"),
    )
    assert w.start_time.utcoffset() == timedelta(0)


def test_end_before_start_raises() -> None:
    with pytest.raises(ValueError):
        collection_window_from_mapping(_base_window(
            start_time="2023-08-14T15:00:00+00:00",
            end_time="2023-08-14T14:00:00+00:00",
        ))


def test_capacity_cost_zero_raises() -> None:
    with pytest.raises(ValueError):
        collection_window_from_mapping(_base_window(capacity_cost=0))


def test_capacity_cost_negative_raises() -> None:
    with pytest.raises(ValueError):
        collection_window_from_mapping(_base_window(capacity_cost=-0.1))


def test_quality_score_above_1_raises() -> None:
    with pytest.raises(ValueError):
        collection_window_from_mapping(_base_window(quality_score=1.5))


def test_quality_score_below_0_raises() -> None:
    with pytest.raises(ValueError):
        collection_window_from_mapping(_base_window(quality_score=-0.1))


def test_latency_hours_negative_raises() -> None:
    with pytest.raises(ValueError):
        collection_window_from_mapping(_base_window(latency_hours=-1.0))


def test_empty_window_id_raises() -> None:
    with pytest.raises(ValueError):
        collection_window_from_mapping(_base_window(window_id=""))


def test_empty_candidate_id_raises() -> None:
    with pytest.raises(ValueError):
        collection_window_from_mapping(_base_window(candidate_id=""))


def test_load_tennent_windows() -> None:
    ws = load_collection_windows(TENNENT_FIXTURE)
    assert len(ws) == 7
    assert all(w.scenario_id == "tennent" for w in ws)


# ---------------------------------------------------------------------------
# Constraint validation
# ---------------------------------------------------------------------------


def test_max_total_capacity_zero_raises() -> None:
    avail = _make_avail_report((_make_item("optical_context", "x"),))
    with pytest.raises(ValueError):
        schedule_collects(
            avail, (_make_window("w1", "optical_context"),),
            scenario_id="tennent",
            constraint=ScheduleConstraint(max_total_capacity=0.0),
        )


def test_max_overlapping_collects_zero_raises() -> None:
    avail = _make_avail_report((_make_item("optical_context", "x"),))
    with pytest.raises(ValueError):
        schedule_collects(
            avail, (_make_window("w1", "optical_context"),),
            scenario_id="tennent",
            constraint=ScheduleConstraint(max_overlapping_collects=0),
        )


def test_latest_end_before_earliest_start_raises() -> None:
    avail = _make_avail_report(())
    with pytest.raises(ValueError):
        schedule_collects(
            avail, (), scenario_id="tennent",
            constraint=ScheduleConstraint(
                earliest_start=datetime(2023, 8, 14, 14, 0, tzinfo=timezone.utc),
                latest_end=datetime(2023, 8, 14, 13, 0, tzinfo=timezone.utc),
            ),
        )


# ---------------------------------------------------------------------------
# Scheduling logic
# ---------------------------------------------------------------------------


def test_max_total_capacity_excess_unscheduled() -> None:
    items = (
        _make_item("optical_context", "Optical", adjusted_utility=0.8),
        _make_item("repeat_sar", "Repeat SAR", adjusted_utility=0.7),
    )
    windows = (
        _make_window("w1", "optical_context", capacity_cost=0.6),
        _make_window(
            "w2", "repeat_sar",
            start=datetime(2023, 8, 14, 16, 0, tzinfo=timezone.utc),
            capacity_cost=0.6,
        ),
    )
    report = schedule_collects(
        _make_avail_report(items), windows, scenario_id="tennent",
        constraint=ScheduleConstraint(
            max_total_capacity=0.6, max_overlapping_collects=2,
        ),
    )
    assert len(report.plan.scheduled_collects) == 1
    assert len(report.plan.unscheduled_candidate_ids) == 1


def test_max_overlapping_collects_constraint_enforced() -> None:
    items = (
        _make_item("optical_context", "Optical", adjusted_utility=0.8),
        _make_item("repeat_sar", "Repeat SAR", adjusted_utility=0.7),
    )
    # Same window range -> overlap
    windows = (
        _make_window("w1", "optical_context"),
        _make_window("w2", "repeat_sar"),
    )
    report = schedule_collects(
        _make_avail_report(items), windows, scenario_id="tennent",
        constraint=ScheduleConstraint(
            max_total_capacity=10.0, max_overlapping_collects=1,
        ),
    )
    assert len(report.plan.scheduled_collects) == 1


def test_earliest_start_filters_windows() -> None:
    items = (_make_item("optical_context", "Optical"),)
    early = _make_window(
        "w-early", "optical_context",
        start=datetime(2023, 8, 14, 12, 0, tzinfo=timezone.utc),
    )
    late = _make_window(
        "w-late", "optical_context",
        start=datetime(2023, 8, 14, 18, 0, tzinfo=timezone.utc),
    )
    report = schedule_collects(
        _make_avail_report(items), (early, late), scenario_id="tennent",
        constraint=ScheduleConstraint(
            max_total_capacity=10.0,
            earliest_start=datetime(2023, 8, 14, 16, 0, tzinfo=timezone.utc),
        ),
    )
    assert report.plan.scheduled_collects[0].window_id == "w-late"


def test_latest_end_filters_windows() -> None:
    items = (_make_item("optical_context", "Optical"),)
    early = _make_window(
        "w-early", "optical_context",
        start=datetime(2023, 8, 14, 12, 0, tzinfo=timezone.utc),
    )
    late = _make_window(
        "w-late", "optical_context",
        start=datetime(2023, 8, 14, 18, 0, tzinfo=timezone.utc),
    )
    report = schedule_collects(
        _make_avail_report(items), (early, late), scenario_id="tennent",
        constraint=ScheduleConstraint(
            max_total_capacity=10.0,
            latest_end=datetime(2023, 8, 14, 14, 0, tzinfo=timezone.utc),
        ),
    )
    assert report.plan.scheduled_collects[0].window_id == "w-early"


def test_allowed_resource_types_filters() -> None:
    items = (_make_item("optical_context", "Optical"),)
    optical = _make_window("w-opt", "optical_context", resource_type="optical")
    sar = _make_window("w-sar", "optical_context", resource_type="sar")
    report = schedule_collects(
        _make_avail_report(items), (optical, sar), scenario_id="tennent",
        constraint=ScheduleConstraint(
            max_total_capacity=10.0, allowed_resource_types=("optical",),
        ),
    )
    assert report.plan.scheduled_collects[0].window_id == "w-opt"


def test_excluded_window_ids_filters() -> None:
    items = (_make_item("optical_context", "Optical"),)
    a = _make_window("w-a", "optical_context", quality_score=0.6)
    b = _make_window(
        "w-b", "optical_context", quality_score=0.9,
        start=datetime(2023, 8, 14, 18, 0, tzinfo=timezone.utc),
    )
    report = schedule_collects(
        _make_avail_report(items), (a, b), scenario_id="tennent",
        constraint=ScheduleConstraint(
            max_total_capacity=10.0, excluded_window_ids=("w-b",),
        ),
    )
    assert report.plan.scheduled_collects[0].window_id == "w-a"


def test_candidate_scheduled_at_most_once() -> None:
    items = (_make_item("optical_context", "Optical"),)
    a = _make_window("w-a", "optical_context", quality_score=0.6)
    b = _make_window(
        "w-b", "optical_context", quality_score=0.9,
        start=datetime(2023, 8, 14, 18, 0, tzinfo=timezone.utc),
    )
    report = schedule_collects(
        _make_avail_report(items), (a, b), scenario_id="tennent",
        constraint=ScheduleConstraint(max_total_capacity=10.0),
    )
    assert len(report.plan.scheduled_collects) == 1


def test_only_recommended_selected_items_scheduled() -> None:
    items = (_make_item("optical_context", "Optical"),)
    avail = _make_avail_report(items)
    other_window = _make_window("w-x", "repeat_sar")
    optical_window = _make_window("w-y", "optical_context")
    report = schedule_collects(
        avail, (other_window, optical_window), scenario_id="tennent",
        constraint=ScheduleConstraint(max_total_capacity=10.0),
    )
    scheduled_ids = {s.candidate_id for s in report.plan.scheduled_collects}
    assert scheduled_ids == {"optical_context"}


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------


def test_latency_factor_low_latency() -> None:
    items = (_make_item("optical_context", "Optical", adjusted_utility=1.0),)
    window = _make_window(
        "w1", "optical_context", quality_score=1.0, latency_hours=4.0,
    )
    report = schedule_collects(
        _make_avail_report(items), (window,), scenario_id="tennent",
        constraint=ScheduleConstraint(max_total_capacity=10.0),
    )
    s = report.plan.scheduled_collects[0]
    assert s.schedule_score == pytest.approx(1.0)


def test_latency_factor_medium_24h() -> None:
    items = (_make_item("optical_context", "Optical", adjusted_utility=1.0),)
    window = _make_window(
        "w1", "optical_context", quality_score=1.0, latency_hours=24.0,
    )
    report = schedule_collects(
        _make_avail_report(items), (window,), scenario_id="tennent",
        constraint=ScheduleConstraint(max_total_capacity=10.0),
    )
    assert report.plan.scheduled_collects[0].schedule_score == pytest.approx(0.8)


def test_latency_factor_high_72h() -> None:
    items = (_make_item("optical_context", "Optical", adjusted_utility=1.0),)
    window = _make_window(
        "w1", "optical_context", quality_score=1.0, latency_hours=72.0,
    )
    report = schedule_collects(
        _make_avail_report(items), (window,), scenario_id="tennent",
        constraint=ScheduleConstraint(max_total_capacity=10.0),
    )
    assert report.plan.scheduled_collects[0].schedule_score == pytest.approx(0.6)


def test_latency_factor_above_72() -> None:
    items = (_make_item("optical_context", "Optical", adjusted_utility=1.0),)
    window = _make_window(
        "w1", "optical_context", quality_score=1.0, latency_hours=96.0,
    )
    report = schedule_collects(
        _make_avail_report(items), (window,), scenario_id="tennent",
        constraint=ScheduleConstraint(max_total_capacity=10.0),
    )
    assert report.plan.scheduled_collects[0].schedule_score == pytest.approx(0.4)


def test_score_clamped_to_unit_interval() -> None:
    items = (_make_item("optical_context", "Optical", adjusted_utility=1.0),)
    window = _make_window(
        "w1", "optical_context", quality_score=1.0, latency_hours=2.0,
    )
    report = schedule_collects(
        _make_avail_report(items), (window,), scenario_id="tennent",
        constraint=ScheduleConstraint(max_total_capacity=10.0),
    )
    assert 0.0 <= report.plan.scheduled_collects[0].schedule_score <= 1.0


# ---------------------------------------------------------------------------
# Tie-breakers
# ---------------------------------------------------------------------------


def test_higher_total_score_wins() -> None:
    items = (_make_item("optical_context", "Optical", adjusted_utility=0.8),)
    cheap_low = _make_window(
        "w-low", "optical_context", quality_score=0.4, capacity_cost=0.1,
    )
    pricey_high = _make_window(
        "w-high", "optical_context", quality_score=0.95, capacity_cost=0.5,
        start=datetime(2023, 8, 14, 18, 0, tzinfo=timezone.utc),
    )
    report = schedule_collects(
        _make_avail_report(items), (cheap_low, pricey_high),
        scenario_id="tennent",
        constraint=ScheduleConstraint(max_total_capacity=1.0),
    )
    assert report.plan.scheduled_collects[0].window_id == "w-high"


def test_equal_score_lower_capacity_wins() -> None:
    items = (_make_item("optical_context", "Optical", adjusted_utility=1.0),)
    a = _make_window(
        "w-a", "optical_context",
        quality_score=0.8, capacity_cost=0.4, latency_hours=4.0,
    )
    b = _make_window(
        "w-b", "optical_context",
        quality_score=0.8, capacity_cost=0.2, latency_hours=4.0,
        start=datetime(2023, 8, 14, 18, 0, tzinfo=timezone.utc),
    )
    report = schedule_collects(
        _make_avail_report(items), (a, b), scenario_id="tennent",
        constraint=ScheduleConstraint(max_total_capacity=1.0),
    )
    assert report.plan.scheduled_collects[0].window_id == "w-b"


def test_equal_score_and_capacity_earlier_end_wins() -> None:
    items = (_make_item("optical_context", "Optical", adjusted_utility=1.0),)
    early = _make_window(
        "w-early", "optical_context",
        start=datetime(2023, 8, 14, 12, 0, tzinfo=timezone.utc),
        quality_score=0.8, capacity_cost=0.3,
    )
    late = _make_window(
        "w-late", "optical_context",
        start=datetime(2023, 8, 14, 18, 0, tzinfo=timezone.utc),
        quality_score=0.8, capacity_cost=0.3,
    )
    report = schedule_collects(
        _make_avail_report(items), (early, late), scenario_id="tennent",
        constraint=ScheduleConstraint(max_total_capacity=1.0),
    )
    assert report.plan.scheduled_collects[0].window_id == "w-early"


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_schedule_report_to_dict_json_safe() -> None:
    items = (_make_item("optical_context", "Optical"),)
    report = schedule_collects(
        _make_avail_report(items),
        (_make_window("w1", "optical_context"),),
        scenario_id="tennent",
        constraint=ScheduleConstraint(max_total_capacity=10.0),
    )
    d = schedule_report_to_dict(report)
    encoded = json.dumps(d)
    assert json.loads(encoded) == d


def test_schedule_report_to_json_parseable() -> None:
    items = (_make_item("optical_context", "Optical"),)
    report = schedule_collects(
        _make_avail_report(items),
        (_make_window("w1", "optical_context"),),
        scenario_id="tennent",
        constraint=ScheduleConstraint(max_total_capacity=10.0),
    )
    parsed = json.loads(schedule_report_to_json(report))
    assert parsed["scenario_id"] == "tennent"


def test_datetimes_serialized_as_iso() -> None:
    items = (_make_item("optical_context", "Optical"),)
    report = schedule_collects(
        _make_avail_report(items),
        (_make_window("w1", "optical_context"),),
        scenario_id="tennent",
        constraint=ScheduleConstraint(max_total_capacity=10.0),
    )
    d = schedule_report_to_dict(report)
    iso = d["plan"]["scheduled_collects"][0]["start_time"]
    datetime.fromisoformat(iso)  # round-trip succeeds


# ---------------------------------------------------------------------------
# Caveats
# ---------------------------------------------------------------------------


def test_caveats_include_simulation_only() -> None:
    items = (_make_item("optical_context", "Optical"),)
    report = schedule_collects(
        _make_avail_report(items),
        (_make_window("w1", "optical_context"),),
        scenario_id="tennent",
        constraint=ScheduleConstraint(max_total_capacity=10.0),
    )
    joined = " ".join(report.caveats).lower()
    assert "simulation only" in joined


def test_caveats_include_no_live_tasking() -> None:
    items = (_make_item("optical_context", "Optical"),)
    report = schedule_collects(
        _make_avail_report(items),
        (_make_window("w1", "optical_context"),),
        scenario_id="tennent",
        constraint=ScheduleConstraint(max_total_capacity=10.0),
    )
    joined = " ".join(report.caveats).lower()
    assert "no live tasking or sensor command is issued" in joined


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def test_text_format_contains_required_sections() -> None:
    items = (_make_item("optical_context", "Optical"),)
    report = schedule_collects(
        _make_avail_report(items),
        (_make_window("w1", "optical_context"),),
        scenario_id="tennent",
        constraint=ScheduleConstraint(max_total_capacity=10.0),
    )
    text = format_schedule_text(report)
    assert "COLLECTION-WINDOW SCHEDULER-LITE" in text
    assert "Constraints" in text
    assert "Scheduled collects" in text
    assert "Unscheduled candidates" in text
    assert "Caveats" in text


def test_markdown_format_contains_heading() -> None:
    items = (_make_item("optical_context", "Optical"),)
    report = schedule_collects(
        _make_avail_report(items),
        (_make_window("w1", "optical_context"),),
        scenario_id="tennent",
        constraint=ScheduleConstraint(max_total_capacity=10.0),
    )
    md = format_schedule_markdown(report)
    assert md.startswith("# Collection-Window Scheduler-Lite")


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------


def test_no_forbidden_imports() -> None:
    import custody.hypotheses.scheduler as mod
    tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
    forbidden = (
        "custody.detection",
        "custody.ingest.gfw_presence",
        "custody.ingest.sentinel",
        "custody.matcher",
        "custody.dashboard",
        "custody.fusion.tracker",
        "custody.taskrecommendation",
        "sentinelhub",
        "requests",
        "httpx",
        "urllib.request",
        "boto3",
        "google.cloud",
    )
    offending: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if any(alias.name.startswith(p) for p in forbidden):
                    offending.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod_name = node.module or ""
            if any(mod_name.startswith(p) for p in forbidden):
                offending.append(mod_name)
    assert not offending, f"forbidden imports: {offending}"


def test_source_file_no_forbidden_language() -> None:
    import custody.hypotheses.scheduler as mod
    src = Path(mod.__file__).read_text(encoding="utf-8").lower()
    forbidden = (
        "tasking order",
        "satellite command",
        "production scheduler",
        "autonomous constellation management",
        "real mps control",
        "sentinel integration",
        "collection order",
        "revenue dollars",
    )
    for needle in forbidden:
        assert needle not in src, f"forbidden token {needle!r} in source"
    # "live tasking" / "sensor command" / "platform access" appear only
    # inside explicit non-claim disclaimers; mask both before scanning.
    safe = src.replace(
        "no live tasking or sensor command is issued", "",
    ).replace(
        "no platform access or orbital scheduling is claimed", "",
    )
    for needle in ("live tasking", "sensor command", "platform access"):
        assert needle not in safe, (
            f"forbidden token {needle!r} appears outside explicit disclaimer"
        )
