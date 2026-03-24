"""
Tests for whatif_panel — the pure data helpers for the What-If Analysis section.

Coverage:
  1. build_variants always includes Baseline with empty overrides
  2. build_variants includes Variant A
  3. build_variants includes Variant B only when provided
  4. build_variants returns 2 items (baseline + A) when no B
  5. build_variants returns 3 items (baseline + A + B) when B provided
  6. Custom labels are preserved
  7. build_whatif_results_df returns expected columns
  8. build_whatif_results_df returns one row per comparison result
  9. build_whatif_results_df handles empty input with correct columns
 10. build_whatif_results_df action counts are populated correctly
 11. build_whatif_results_df hold reason counts are populated correctly
 12. build_whatif_results_df avg metrics are rounded to 3dp
 13. App imports build_variants and build_whatif_results_df from whatif_panel
 14. App imports run_comparison from custody.whatif
 15. Smoke test: end-to-end with real simulation data
"""
import ast
import pathlib

import pytest

from whatif_panel import build_variants, build_whatif_results_df, _RESULTS_COLUMNS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_OVERRIDES_A = {"HOLD_LOOKAHEAD_THRESHOLD_SECONDS": 900.0, "HOLD_LOOKAHEAD_BOOST": 0.2}
_OVERRIDES_B = {"HOLD_LOOKAHEAD_THRESHOLD_SECONDS": 3600.0, "AIS_STALE_GAP_SECONDS": 0.0}

def _fake_summary(label, none=5, task=3, hold=2, no_sensor=1, preempted=0,
                  freshness=1, lookahead=1, avg_conf=0.75, avg_anom=0.4):
    return {
        "label": label,
        "record_count": none + task + hold + no_sensor + preempted,
        "action_counts": {
            "NONE": none, "TASK": task, "HOLD": hold,
            "NO_SENSOR": no_sensor, "PREEMPTED": preempted,
        },
        "hold_reason_counts": {"freshness": freshness, "lookahead": lookahead},
        "lookahead_hold_count": lookahead,
        "avg_custody_confidence": avg_conf,
        "avg_anomaly_score": avg_anom,
        "sensor_usage": {},
    }


# ---------------------------------------------------------------------------
# 1–6. build_variants
# ---------------------------------------------------------------------------

class TestBuildVariants:
    def test_baseline_always_first(self):
        variants = build_variants(_OVERRIDES_A)
        assert variants[0]["label"] == "Baseline"

    def test_baseline_has_empty_overrides(self):
        variants = build_variants(_OVERRIDES_A)
        assert variants[0]["overrides"] == {}

    def test_variant_a_is_second(self):
        variants = build_variants(_OVERRIDES_A)
        assert variants[1]["label"] == "Variant A"

    def test_variant_a_overrides_match_input(self):
        variants = build_variants(_OVERRIDES_A)
        assert variants[1]["overrides"] == _OVERRIDES_A

    def test_no_variant_b_gives_two_variants(self):
        variants = build_variants(_OVERRIDES_A)
        assert len(variants) == 2

    def test_variant_b_included_when_provided(self):
        variants = build_variants(_OVERRIDES_A, _OVERRIDES_B)
        assert len(variants) == 3

    def test_variant_b_is_third(self):
        variants = build_variants(_OVERRIDES_A, _OVERRIDES_B)
        assert variants[2]["label"] == "Variant B"

    def test_variant_b_overrides_match_input(self):
        variants = build_variants(_OVERRIDES_A, _OVERRIDES_B)
        assert variants[2]["overrides"] == _OVERRIDES_B

    def test_none_variant_b_gives_two_variants(self):
        variants = build_variants(_OVERRIDES_A, None)
        assert len(variants) == 2

    def test_custom_labels_preserved(self):
        variants = build_variants(
            _OVERRIDES_A, _OVERRIDES_B,
            variant_a_label="Low Threshold", variant_b_label="High Boost",
        )
        labels = [v["label"] for v in variants]
        assert "Low Threshold" in labels
        assert "High Boost" in labels

    def test_empty_overrides_are_valid(self):
        variants = build_variants({}, {})
        assert len(variants) == 3
        assert all(isinstance(v["overrides"], dict) for v in variants)


# ---------------------------------------------------------------------------
# 7–12. build_whatif_results_df
# ---------------------------------------------------------------------------

class TestBuildWhatifResultsDf:
    def test_returns_expected_columns(self):
        summaries = [_fake_summary("A"), _fake_summary("B")]
        df = build_whatif_results_df(summaries)
        assert list(df.columns) == _RESULTS_COLUMNS

    def test_empty_input_returns_correct_columns(self):
        df = build_whatif_results_df([])
        assert list(df.columns) == _RESULTS_COLUMNS
        assert len(df) == 0

    def test_row_count_matches_input(self):
        summaries = [_fake_summary("A"), _fake_summary("B"), _fake_summary("C")]
        df = build_whatif_results_df(summaries)
        assert len(df) == 3

    def test_label_column_populated(self):
        df = build_whatif_results_df([_fake_summary("Baseline"), _fake_summary("Variant A")])
        assert list(df["Label"]) == ["Baseline", "Variant A"]

    def test_action_counts_in_correct_columns(self):
        summary = _fake_summary("test", none=7, task=2, hold=3, no_sensor=1, preempted=0)
        df = build_whatif_results_df([summary])
        assert df.iloc[0]["NONE"] == 7
        assert df.iloc[0]["TASK"] == 2
        assert df.iloc[0]["HOLD"] == 3
        assert df.iloc[0]["NO_SENSOR"] == 1
        assert df.iloc[0]["PREEMPTED"] == 0

    def test_hold_reason_columns_populated(self):
        summary = _fake_summary("test", freshness=4, lookahead=2)
        df = build_whatif_results_df([summary])
        assert df.iloc[0]["HOLD: Freshness"] == 4
        assert df.iloc[0]["HOLD: Lookahead"] == 2

    def test_missing_hold_reasons_default_to_zero(self):
        summary = _fake_summary("test")
        summary["hold_reason_counts"] = {}  # no reasons recorded
        df = build_whatif_results_df([summary])
        assert df.iloc[0]["HOLD: Freshness"] == 0
        assert df.iloc[0]["HOLD: Lookahead"] == 0

    def test_avg_confidence_rounded_to_3dp(self):
        summary = _fake_summary("test", avg_conf=0.123456789)
        df = build_whatif_results_df([summary])
        assert df.iloc[0]["Avg Confidence"] == pytest.approx(0.123, abs=1e-3)

    def test_avg_anomaly_rounded_to_3dp(self):
        summary = _fake_summary("test", avg_anom=0.987654321)
        df = build_whatif_results_df([summary])
        assert df.iloc[0]["Avg Anomaly"] == pytest.approx(0.988, abs=1e-3)

    def test_records_column_is_sum_of_actions(self):
        summary = _fake_summary("test", none=3, task=2, hold=1, no_sensor=1, preempted=0)
        df = build_whatif_results_df([summary])
        assert df.iloc[0]["Records"] == 7


# ---------------------------------------------------------------------------
# Delta columns
# ---------------------------------------------------------------------------

class TestDeltaColumns:
    def test_delta_columns_exist_for_all_numeric_fields(self):
        df = build_whatif_results_df([_fake_summary("Baseline"), _fake_summary("Variant A")])
        for col in ["NONE", "TASK", "HOLD", "NO_SENSOR", "PREEMPTED",
                    "HOLD: Freshness", "HOLD: Lookahead", "Avg Confidence", "Avg Anomaly"]:
            assert f"{col} \u0394" in df.columns

    def test_baseline_count_deltas_are_zero(self):
        df = build_whatif_results_df([_fake_summary("Baseline"), _fake_summary("Variant A")])
        baseline = df[df["Label"] == "Baseline"].iloc[0]
        for col in ["NONE", "TASK", "HOLD", "NO_SENSOR", "PREEMPTED",
                    "HOLD: Freshness", "HOLD: Lookahead"]:
            assert baseline[f"{col} \u0394"] == 0

    def test_baseline_float_deltas_are_zero(self):
        df = build_whatif_results_df([_fake_summary("Baseline"), _fake_summary("Variant A")])
        baseline = df[df["Label"] == "Baseline"].iloc[0]
        assert baseline["Avg Confidence \u0394"] == pytest.approx(0.0)
        assert baseline["Avg Anomaly \u0394"] == pytest.approx(0.0)

    def test_nonbaseline_count_delta_is_signed_difference(self):
        base = _fake_summary("Baseline", task=3)
        variant = _fake_summary("Variant A", task=7)
        df = build_whatif_results_df([base, variant])
        row = df[df["Label"] == "Variant A"].iloc[0]
        assert row["TASK \u0394"] == 4

    def test_nonbaseline_negative_delta(self):
        base = _fake_summary("Baseline", hold=5)
        variant = _fake_summary("Variant A", hold=2)
        df = build_whatif_results_df([base, variant])
        row = df[df["Label"] == "Variant A"].iloc[0]
        assert row["HOLD \u0394"] == -3

    def test_float_delta_rounded_to_3dp(self):
        base = _fake_summary("Baseline", avg_conf=0.5)
        variant = _fake_summary("Variant A", avg_conf=0.5 + 0.123456789)
        df = build_whatif_results_df([base, variant])
        row = df[df["Label"] == "Variant A"].iloc[0]
        assert row["Avg Confidence \u0394"] == pytest.approx(0.123, abs=1e-3)

    def test_count_delta_is_integer_type(self):
        import numpy as np
        base = _fake_summary("Baseline", none=2)
        variant = _fake_summary("Variant A", none=5)
        df = build_whatif_results_df([base, variant])
        row = df[df["Label"] == "Variant A"].iloc[0]
        assert isinstance(row["NONE \u0394"], (int, np.integer))

    def test_columns_interleaved_metric_then_delta(self):
        cols = list(_RESULTS_COLUMNS)
        for col in ["NONE", "TASK", "HOLD"]:
            idx = cols.index(col)
            assert cols[idx + 1] == f"{col} \u0394"

    def test_three_variants_all_have_deltas(self):
        results = [
            _fake_summary("Baseline", task=2),
            _fake_summary("Variant A", task=4),
            _fake_summary("Variant B", task=6),
        ]
        df = build_whatif_results_df(results)
        assert df[df["Label"] == "Baseline"].iloc[0]["TASK \u0394"] == 0
        assert df[df["Label"] == "Variant A"].iloc[0]["TASK \u0394"] == 2
        assert df[df["Label"] == "Variant B"].iloc[0]["TASK \u0394"] == 4

    def test_empty_input_has_delta_columns(self):
        df = build_whatif_results_df([])
        assert "TASK \u0394" in df.columns
        assert "Avg Confidence \u0394" in df.columns

    def test_no_baseline_row_falls_back_to_first_row(self):
        results = [
            _fake_summary("Alpha", task=3),
            _fake_summary("Beta", task=5),
        ]
        df = build_whatif_results_df(results)
        # First row (Alpha) is the fallback baseline — its delta should be 0
        assert df.iloc[0]["TASK \u0394"] == 0
        assert df.iloc[1]["TASK \u0394"] == 2


# ---------------------------------------------------------------------------
# 13 & 14. App import checks
# ---------------------------------------------------------------------------

def _read_app():
    return (
        pathlib.Path(__file__).parent.parent / "src" / "app" / "streamlit_app.py"
    ).read_text(encoding="utf-8")


def test_app_imports_build_variants():
    src = _read_app()
    tree = ast.parse(src)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "whatif_panel":
            imports.extend(a.name for a in node.names)
    assert "build_variants" in imports


def test_app_imports_build_whatif_results_df():
    src = _read_app()
    tree = ast.parse(src)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "whatif_panel":
            imports.extend(a.name for a in node.names)
    assert "build_whatif_results_df" in imports


def test_app_imports_run_comparison():
    src = _read_app()
    tree = ast.parse(src)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "custody.whatif":
            imports.extend(a.name for a in node.names)
    assert "run_comparison" in imports


# ---------------------------------------------------------------------------
# 15. Smoke test: end-to-end with real simulation data
# ---------------------------------------------------------------------------

def test_end_to_end_with_simulation():
    """build_variants + run_comparison + build_whatif_results_df produce a valid table."""
    from custody.simulate import run_simulation
    from custody.whatif import run_comparison

    variants = build_variants(
        {"TASK_VALUE_THRESHOLD": 0.0},   # variant A: no HOLD gate
    )
    results = run_comparison(run_simulation, variants)
    df = build_whatif_results_df(results)

    assert list(df.columns) == _RESULTS_COLUMNS
    assert len(df) == 2  # baseline + variant A
    assert set(df["Label"]) == {"Baseline", "Variant A"}
    assert (df["Records"] > 0).all()
