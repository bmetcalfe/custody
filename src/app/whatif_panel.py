"""
Pure data-shaping helpers for the "What-If Analysis" dashboard section.

Contains no Streamlit calls.  Accepts plain Python values and returns
structures ready for display.

Public API
----------
build_variants(variant_a_overrides, variant_b_overrides) -> list[dict]
    Construct the variants list expected by run_comparison().

build_whatif_results_df(comparison_results)              -> pd.DataFrame
    Flatten run_comparison() output into a display-ready DataFrame with
    delta-vs-baseline columns for every numeric metric.
"""
import pandas as pd

# Numeric columns that receive a paired delta column.
# Count columns produce integer deltas; average columns produce float deltas.
_COUNT_COLS = [
    "NONE", "TASK", "HOLD", "NO_SENSOR", "PREEMPTED",
    "HOLD: Freshness", "HOLD: Lookahead",
]
_FLOAT_COLS = ["Avg Confidence", "Avg Anomaly"]
_DELTA_COLS = _COUNT_COLS + _FLOAT_COLS

# Full output column order: each metric immediately followed by its delta.
_RESULTS_COLUMNS = (
    ["Label", "Records"]
    + [col for c in _COUNT_COLS for col in (c, f"{c} \u0394")]
    + [col for c in _FLOAT_COLS for col in (c, f"{c} \u0394")]
)


def build_variants(
    variant_a_overrides: dict,
    variant_b_overrides: dict | None = None,
    variant_a_label: str = "Variant A",
    variant_b_label: str = "Variant B",
) -> list[dict]:
    """Construct the variants list expected by run_comparison().

    Always includes a Baseline with empty overrides as the first entry.
    Variant B is only included when variant_b_overrides is not None.

    Args:
        variant_a_overrides: Dict of custody.config overrides for Variant A.
        variant_b_overrides: Dict of custody.config overrides for Variant B,
                             or None to omit Variant B.
        variant_a_label:     Display name for Variant A.
        variant_b_label:     Display name for Variant B.

    Returns:
        List of variant dicts in the shape expected by run_comparison().
    """
    variants = [
        {"label": "Baseline", "overrides": {}},
        {"label": variant_a_label, "overrides": variant_a_overrides},
    ]
    if variant_b_overrides is not None:
        variants.append({"label": variant_b_label, "overrides": variant_b_overrides})
    return variants


def build_whatif_results_df(comparison_results: list[dict]) -> pd.DataFrame:
    """Flatten run_comparison() output into a display-ready DataFrame.

    Expands nested action_counts and hold_reason_counts dicts into individual
    metric columns.  For every numeric metric a corresponding delta column is
    appended (e.g. "TASK Δ") showing the signed difference from the Baseline
    row.  The Baseline row itself receives 0 / 0.0 deltas.

    Unknown action codes are ignored; missing codes default to 0.

    Args:
        comparison_results: List of summary dicts returned by run_comparison().

    Returns:
        DataFrame with columns matching _RESULTS_COLUMNS.
        Returns an empty DataFrame with the correct columns when the input is empty.
    """
    if not comparison_results:
        return pd.DataFrame(columns=_RESULTS_COLUMNS)

    # ── Build base rows ───────────────────────────────────────────────────────
    rows = []
    for r in comparison_results:
        ac = r.get("action_counts", {})
        hr = r.get("hold_reason_counts", {})
        rows.append({
            "Label":           r.get("label", ""),
            "Records":         r.get("record_count", 0),
            "NONE":            ac.get("NONE", 0),
            "TASK":            ac.get("TASK", 0),
            "HOLD":            ac.get("HOLD", 0),
            "NO_SENSOR":       ac.get("NO_SENSOR", 0),
            "PREEMPTED":       ac.get("PREEMPTED", 0),
            "HOLD: Freshness": hr.get("freshness", 0),
            "HOLD: Lookahead": hr.get("lookahead", 0),
            "Avg Confidence":  round(r.get("avg_custody_confidence", 0.0), 3),
            "Avg Anomaly":     round(r.get("avg_anomaly_score", 0.0), 3),
        })
    df = pd.DataFrame(rows)

    # ── Identify baseline row ─────────────────────────────────────────────────
    baseline_mask = df["Label"] == "Baseline"
    baseline = df[baseline_mask].iloc[0] if baseline_mask.any() else df.iloc[0]
    baseline_label = baseline["Label"]

    # ── Add delta columns ─────────────────────────────────────────────────────
    for col in _DELTA_COLS:
        delta_col = f"{col} \u0394"
        is_float = col in _FLOAT_COLS
        deltas = []
        for _, row in df.iterrows():
            if row["Label"] == baseline_label:
                deltas.append(0.0 if is_float else 0)
            else:
                diff = row[col] - baseline[col]
                deltas.append(round(diff, 3) if is_float else int(diff))
        df[delta_col] = deltas

    return df[_RESULTS_COLUMNS]
