"""
Scenario replay and what-if analysis for the custody package.

Run the same simulation or AIS replay under different config settings and
compare planner outcome metrics across runs.

Public API
----------
run_scenario(scenario_fn, config_overrides)  -> list[dict]
    Execute scenario_fn() with custody.config attributes patched to the
    supplied overrides.  Config is restored exactly after the call.

summarise(records, label)                    -> dict
    Compute planner-outcome metrics from a flat list of timeline records.
    Works with records from run_simulation() or from a flattened
    replay_ais_file() result.

run_comparison(scenario_fn, variants)        -> list[dict]
    Convenience wrapper: run scenario_fn() once per variant and return a
    list of summary dicts, one per variant.  Each dict includes the variant
    label and can be passed directly to pd.DataFrame() for side-by-side
    comparison.

Usage example
-------------
    from custody.simulate import run_simulation
    from custody.whatif import run_comparison

    results = run_comparison(
        run_simulation,
        [
            {"label": "baseline",         "overrides": {}},
            {"label": "no-lookahead",     "overrides": {"HOLD_LOOKAHEAD_THRESHOLD_SECONDS": 0.0}},
            {"label": "wider-lookahead",  "overrides": {"HOLD_LOOKAHEAD_THRESHOLD_SECONDS": 3600.0}},
        ],
    )
    for r in results:
        print(r["label"], r["action_counts"])
"""
from __future__ import annotations

import contextlib
from collections import Counter
from typing import Any, Callable
from unittest.mock import patch

import custody.config as _config
from custody.decision_trace import DecisionTrace


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _apply_overrides(overrides: dict[str, Any]):
    """Context manager: patch custody.config attributes, restore on exit."""
    with contextlib.ExitStack() as stack:
        for key, value in overrides.items():
            stack.enter_context(patch.object(_config, key, value))
        yield


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_scenario(
    scenario_fn: Callable[[], list[dict]],
    config_overrides: dict[str, Any] | None = None,
) -> list[dict]:
    """Run *scenario_fn* with custody.config patched to *config_overrides*.

    Config attributes are restored to their original values after the call,
    regardless of whether scenario_fn raises.

    Args:
        scenario_fn:      Zero-argument callable that returns a flat list of
                          timeline records (e.g. ``run_simulation``, or a
                          lambda that flattens ``replay_ais_file(...)``).
        config_overrides: Dict mapping custody.config attribute names to
                          temporary replacement values.  Pass ``None`` or
                          ``{}`` for an unmodified baseline run.

    Returns:
        The list of records returned by scenario_fn().
    """
    with _apply_overrides(config_overrides or {}):
        return scenario_fn()


def summarise(records: list[dict], label: str = "run") -> dict:
    """Compute planner-outcome metrics from a flat list of timeline records.

    Compatible with records produced by run_simulation() or by flattening
    replay_ais_file() output.  If a record lacks a field (e.g. AIS records
    have no sensor_access_count), it is silently skipped for that metric.

    Args:
        records: Flat list of timeline record dicts.
        label:   Identifier included in the returned dict (useful when
                 assembling comparison tables).

    Returns:
        Dict with keys:
            label                  – the supplied label string
            record_count           – total number of records
            action_counts          – Counter of action strings
            hold_reason_counts     – Counter of hold_reason values (excludes None)
            lookahead_hold_count   – number of HOLDs with hold_reason="lookahead"
            avg_custody_confidence – mean custody_confidence across all records
            avg_anomaly_score      – mean anomaly_score across all records
            sensor_usage           – Counter of sensor_id for TASK records
    """
    action_counts: Counter[str] = Counter()
    hold_reason_counts: Counter[str] = Counter()
    lookahead_hold_count = 0
    confidences: list[float] = []
    anomaly_scores: list[float] = []
    sensor_usage: Counter[str] = Counter()

    for record in records:
        action = record.get("action", "")
        action_counts[action] += 1

        trace = record.get("decision_trace")
        if isinstance(trace, DecisionTrace):
            reason = trace.arbitration.hold_reason
            if reason is not None:
                hold_reason_counts[reason] += 1
            if reason == "lookahead":
                lookahead_hold_count += 1

        conf = record.get("custody_confidence")
        if conf is not None:
            confidences.append(float(conf))

        score = record.get("anomaly_score")
        if score is not None:
            anomaly_scores.append(float(score))

        if action == "TASK":
            sid = record.get("sensor_id")
            if sid is not None:
                sensor_usage[sid] += 1

    return {
        "label": label,
        "record_count": len(records),
        "action_counts": dict(action_counts),
        "hold_reason_counts": dict(hold_reason_counts),
        "lookahead_hold_count": lookahead_hold_count,
        "avg_custody_confidence": (
            sum(confidences) / len(confidences) if confidences else 0.0
        ),
        "avg_anomaly_score": (
            sum(anomaly_scores) / len(anomaly_scores) if anomaly_scores else 0.0
        ),
        "sensor_usage": dict(sensor_usage),
    }


def run_comparison(
    scenario_fn: Callable[[], list[dict]],
    variants: list[dict[str, Any]],
) -> list[dict]:
    """Run *scenario_fn* once per variant and return a list of summary dicts.

    Each entry in *variants* is a dict with:
        "label"     – human-readable name for this run
        "overrides" – dict of custody.config overrides (empty {} for baseline)

    Args:
        scenario_fn: Zero-argument callable returning a flat list of records.
        variants:    List of variant dicts as described above.

    Returns:
        List of summary dicts (one per variant), each in the shape returned
        by summarise().  Suitable for pd.DataFrame(results) comparisons.
    """
    results = []
    for variant in variants:
        label = variant.get("label", "unnamed")
        overrides = variant.get("overrides", {})
        records = run_scenario(scenario_fn, overrides)
        results.append(summarise(records, label=label))
    return results
