"""Tests for :mod:`custody.hypotheses.baseline_comparison` (ADR-0021 Slice 24)."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from custody.hypotheses.baseline_comparison import (
    CrossScenarioComparisonReport,
    PlanningStrategyId,
    StrategyComparisonReport,
    StrategyEvaluation,
    compare_planning_strategies,
    compare_planning_strategies_across_scenarios,
    format_comparison_markdown,
    format_comparison_text,
    format_cross_scenario_markdown,
    format_cross_scenario_text,
    report_to_dict,
    report_to_json,
)


REPO_ROOT = Path(__file__).resolve().parents[1]

ART = {
    "tennent": REPO_ROOT / "tests" / "fixtures" / "artifacts" / "tennent_artifact_manifest.json",
    "whitsun": REPO_ROOT / "tests" / "fixtures" / "artifacts" / "whitsun_artifact_manifest.json",
}
CAT = {
    "tennent": REPO_ROOT / "tests" / "fixtures" / "availability" / "tennent_scene_availability.json",
    "whitsun": REPO_ROOT / "tests" / "fixtures" / "availability" / "whitsun_scene_availability.json",
}
WIN = {
    "tennent": REPO_ROOT / "tests" / "fixtures" / "schedule" / "tennent_collection_windows.json",
    "whitsun": REPO_ROOT / "tests" / "fixtures" / "schedule" / "whitsun_collection_windows.json",
}


def _compare_tennent(strategies=None) -> StrategyComparisonReport:
    return compare_planning_strategies(
        scenario_id="tennent",
        strategies=strategies,
        artifact_manifest_path=ART["tennent"],
        availability_catalog_path=CAT["tennent"],
        collection_windows_path=WIN["tennent"],
    )


# ---------------------------------------------------------------------------
# Strategy evaluation
# ---------------------------------------------------------------------------


def test_default_evaluates_all_six_strategies() -> None:
    r = _compare_tennent()
    ids = {e.strategy_id for e in r.evaluations}
    assert ids == {s.value for s in PlanningStrategyId}


def test_strategy_filter_evaluates_only_listed() -> None:
    r = _compare_tennent(strategies=("baseline_manual", "execution_feedback"))
    ids = [e.strategy_id for e in r.evaluations]
    assert sorted(ids) == ["baseline_manual", "execution_feedback"]


def test_invalid_strategy_raises() -> None:
    with pytest.raises(ValueError):
        compare_planning_strategies(
            scenario_id="tennent",
            strategies=("bogus",),
            artifact_manifest_path=ART["tennent"],
            availability_catalog_path=CAT["tennent"],
            collection_windows_path=WIN["tennent"],
        )


def test_baseline_manual_metrics() -> None:
    r = _compare_tennent()
    baseline = next(e for e in r.evaluations if e.strategy_id == "baseline_manual")
    assert baseline.ambiguity_resolved is False
    assert baseline.traceability_artifact_count == 1
    assert baseline.review_actions_required == 3
    assert baseline.health_score_delta == 0.0
    assert baseline.selected_candidate_ids == ()


def test_collection_value_only_selects_candidates() -> None:
    r = _compare_tennent()
    cv = next(e for e in r.evaluations if e.strategy_id == "collection_value_only")
    assert len(cv.selected_candidate_ids) >= 1
    assert "wait_or_monitor" not in cv.selected_candidate_ids


def test_mission_value_optimized_has_planning_utility() -> None:
    r = _compare_tennent()
    mv = next(e for e in r.evaluations if e.strategy_id == "mission_value_optimized")
    assert mv.planning_utility > 0
    assert mv.mission_value_proxy > 0


def test_availability_adjusted_includes_adjusted_utility() -> None:
    r = _compare_tennent()
    aa = next(e for e in r.evaluations if e.strategy_id == "availability_adjusted")
    assert aa.planning_utility >= 0


def test_scheduler_lite_has_schedule_count() -> None:
    r = _compare_tennent()
    sl = next(e for e in r.evaluations if e.strategy_id == "scheduler_lite")
    assert sl.schedule_feasible_count >= 0


def test_execution_feedback_has_post_health() -> None:
    r = _compare_tennent()
    ef = next(e for e in r.evaluations if e.strategy_id == "execution_feedback")
    assert ef.post_health_status is not None
    assert ef.post_health_score is not None


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def test_ranking_is_deterministic() -> None:
    a = _compare_tennent()
    b = _compare_tennent()
    assert (
        tuple(e.strategy_id for e in a.evaluations)
        == tuple(e.strategy_id for e in b.evaluations)
    )


def test_winning_strategy_is_rank_one() -> None:
    r = _compare_tennent()
    rank_one = next(e for e in r.evaluations if e.rank == 1)
    assert rank_one.strategy_id == r.winning_strategy_id


def test_ranks_are_contiguous() -> None:
    r = _compare_tennent()
    ranks = sorted(e.rank for e in r.evaluations)
    assert ranks == list(range(1, len(ranks) + 1))


def test_execution_feedback_typically_wins_under_favorable() -> None:
    r = _compare_tennent()
    assert r.winning_strategy_id == "execution_feedback"


# ---------------------------------------------------------------------------
# Cross-scenario
# ---------------------------------------------------------------------------


def test_cross_scenario_report_includes_both() -> None:
    r = compare_planning_strategies_across_scenarios(
        scenario_ids=("tennent", "whitsun"),
        artifact_manifest_paths=ART,
        availability_catalog_paths=CAT,
        collection_windows_paths=WIN,
    )
    assert r.scenario_ids == ("tennent", "whitsun")
    assert len(r.scenario_reports) == 2
    seen = {sr.scenario_id for sr in r.scenario_reports}
    assert seen == {"tennent", "whitsun"}


def test_cross_scenario_aggregate_scenario_id() -> None:
    r = compare_planning_strategies_across_scenarios(
        scenario_ids=("tennent", "whitsun"),
        artifact_manifest_paths=ART,
        availability_catalog_paths=CAT,
        collection_windows_paths=WIN,
    )
    assert all(e.scenario_id == "aggregate" for e in r.aggregate_evaluations)


def test_cross_scenario_winning_strategy_set() -> None:
    r = compare_planning_strategies_across_scenarios(
        scenario_ids=("tennent", "whitsun"),
        artifact_manifest_paths=ART,
        availability_catalog_paths=CAT,
        collection_windows_paths=WIN,
    )
    assert r.winning_strategy_id


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_report_to_dict_json_safe() -> None:
    r = _compare_tennent()
    d = report_to_dict(r)
    encoded = json.dumps(d)
    assert json.loads(encoded) == d


def test_report_to_json_parseable() -> None:
    r = _compare_tennent()
    parsed = json.loads(report_to_json(r))
    assert parsed["scenario_id"] == "tennent"


def test_cross_scenario_to_dict_json_safe() -> None:
    r = compare_planning_strategies_across_scenarios(
        scenario_ids=("tennent", "whitsun"),
        artifact_manifest_paths=ART,
        availability_catalog_paths=CAT,
        collection_windows_paths=WIN,
    )
    d = report_to_dict(r)
    encoded = json.dumps(d)
    assert json.loads(encoded) == d


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def test_text_format_contains_required_sections() -> None:
    r = _compare_tennent()
    text = format_comparison_text(r)
    assert "PLANNING STRATEGY COMPARISON" in text
    assert "Strategies evaluated" in text
    assert "Ranked strategy results" in text
    assert "Winning strategy" in text
    assert "Key deltas" in text
    assert "Caveats" in text


def test_markdown_format_contains_heading() -> None:
    r = _compare_tennent()
    md = format_comparison_markdown(r)
    assert md.startswith("# Planning Strategy Comparison")


def test_cross_text_includes_aggregate() -> None:
    r = compare_planning_strategies_across_scenarios(
        scenario_ids=("tennent", "whitsun"),
        artifact_manifest_paths=ART,
        availability_catalog_paths=CAT,
        collection_windows_paths=WIN,
    )
    text = format_cross_scenario_text(r)
    assert "CROSS-SCENARIO AGGREGATE" in text


def test_cross_markdown_includes_aggregate() -> None:
    r = compare_planning_strategies_across_scenarios(
        scenario_ids=("tennent", "whitsun"),
        artifact_manifest_paths=ART,
        availability_catalog_paths=CAT,
        collection_windows_paths=WIN,
    )
    md = format_cross_scenario_markdown(r)
    assert "Cross-Scenario Aggregate" in md


# ---------------------------------------------------------------------------
# Caveats
# ---------------------------------------------------------------------------


def test_caveats_include_deterministic_prototype() -> None:
    r = _compare_tennent()
    joined = " ".join(r.caveats).lower()
    assert "deterministic prototype" in joined


def test_caveats_include_proxy() -> None:
    r = _compare_tennent()
    joined = " ".join(r.caveats).lower()
    assert "proxy" in joined


def test_caveats_include_no_live_tasking() -> None:
    r = _compare_tennent()
    joined = " ".join(r.caveats).lower()
    assert "no live tasking or sensor command is issued" in joined


def test_caveats_include_not_revenue() -> None:
    r = _compare_tennent()
    joined = " ".join(r.caveats).lower()
    assert "not revenue" in joined


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------


def test_no_forbidden_imports() -> None:
    import custody.hypotheses.baseline_comparison as mod
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
    import custody.hypotheses.baseline_comparison as mod
    raw = Path(mod.__file__).read_text(encoding="utf-8").lower()
    src = " ".join(raw.split())
    forbidden = (
        "proven savings",
        "actual revenue",
        "real planner adoption",
        "operational kpi",
        "production cycle-time reduction",
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
    safe = src.replace(
        "no live tasking or sensor command is issued", "",
    )
    for needle in ("live tasking", "sensor command", "platform access"):
        assert needle not in safe, (
            f"forbidden token {needle!r} appears outside explicit disclaimer"
        )
