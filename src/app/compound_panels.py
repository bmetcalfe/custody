"""
Pure data-shaping helpers for the compound-signal dashboard sections.

These functions contain no Streamlit calls.  They accept plain Python
objects (lists of CompoundSignal, dicts) and return pandas DataFrames or
dicts ready for display.  The Streamlit rendering calls stay in
streamlit_app.py.

Public API
----------
aggregate_compound_history  : list[CompoundSignal] -> dict
build_active_compounds_df   : list[CompoundSignal] -> pd.DataFrame
build_compound_history_df   : dict                 -> pd.DataFrame
"""
import pandas as pd

from custody.compounds import CompoundSignal


def aggregate_compound_history(compounds: list[CompoundSignal]) -> dict:
    """Aggregate a flat list of CompoundSignals into a per-code summary.

    Args:
        compounds: Ordered list of CompoundSignal objects (typically the
                   output of compounds_for_timeline, already filtered by
                   confidence threshold).

    Returns:
        Dict keyed by compound code.  Each value is a dict with keys:
            first  – earliest timestamp this code was seen
            last   – most recent timestamp this code was seen
            count  – total number of firings
            peak   – highest confidence value observed

        Keys appear in first-seen (ascending) order, matching the
        iteration order of compounds.  Empty input returns {}.
    """
    agg: dict = {}
    for c in compounds:
        if c.code not in agg:
            agg[c.code] = {
                "first": c.timestamp,
                "last": c.timestamp,
                "count": 1,
                "peak": c.confidence,
            }
        else:
            agg[c.code]["last"] = c.timestamp
            agg[c.code]["count"] += 1
            agg[c.code]["peak"] = max(agg[c.code]["peak"], c.confidence)
    return agg


def build_active_compounds_df(compounds: list[CompoundSignal]) -> pd.DataFrame:
    """Format a list of active CompoundSignals as a display DataFrame.

    Args:
        compounds: Compounds active at the selected playback step, already
                   filtered by confidence threshold and sorted.

    Returns:
        DataFrame with columns: Time, Code, Conf, Evidence, Components.
        Components values are formatted as ``key=value`` pairs joined by
        ``", "``; floats use two decimal places.
    """
    return pd.DataFrame([
        {
            "Time": str(c.timestamp).split("+")[0],
            "Code": c.code,
            "Conf": f"{c.confidence:.2f}",
            "Evidence": c.evidence,
            "Components": ", ".join(
                f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in c.components.items()
            ),
        }
        for c in compounds
    ])


def build_compound_history_df(agg: dict) -> pd.DataFrame:
    """Format a compound history aggregate dict as a display DataFrame.

    Args:
        agg: Output of aggregate_compound_history.

    Returns:
        DataFrame with columns: Code, First Seen, Last Seen, Count, Peak Conf.
        Rows are ordered by First Seen ascending.
    """
    return pd.DataFrame([
        {
            "Code": code,
            "First Seen": str(v["first"]).split("+")[0],
            "Last Seen": str(v["last"]).split("+")[0],
            "Count": v["count"],
            "Peak Conf": f"{v['peak']:.2f}",
        }
        for code, v in sorted(agg.items(), key=lambda kv: kv[1]["first"])
    ])
