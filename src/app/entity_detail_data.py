"""Pure-data helpers for entity detail display.

These functions extract and derive display-ready values from timeline
records without any UI framework dependency.  They are used by both the
current Streamlit entity_detail_panel.py and the future Dash entity
detail layout.

Imports from the custody engine are limited to data types (TrackState,
FusionAssessment, Decision) — no simulation or scoring logic is called.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from custody.decision import Decision
from custody.fusion import FusionAssessment
from custody.models import TrackState


def entity_id(record: dict) -> str:
    """Extract entity identifier from a record dict.

    Checks ``entity_id``, ``vessel_id``, and ``target_id`` keys in order,
    returning the first non-None value.  Falls back to ``"unknown"``.
    """
    return (
        record.get("entity_id")
        or record.get("vessel_id")
        or record.get("target_id")
        or "unknown"
    )


def derive_track_state(record: dict, prefix_window: list[dict]) -> TrackState:
    """Build a TrackState from the current record and its history prefix.

    Uses the record's ``uncertainty_km`` directly.  Scans the prefix
    backwards for the most recent TASK event with a non-empty
    ``collection_result`` to populate ``last_collection_time`` and
    ``last_collection_anomaly_score``.

    Args:
        record:        The current timeline record dict.
        prefix_window: All prior records for this entity, in order,
                       excluding the current record.

    Returns:
        A TrackState snapshot suitable for fusion/decision rebuilding.
    """
    uncertainty_km = float(record.get("uncertainty_km", 5.0))
    last_collection_time: Optional[datetime] = None
    last_collection_anomaly_score: float = 0.0

    for r in reversed(prefix_window):
        result = r.get("collection_result")
        if r.get("action") == "TASK" and result not in (None, "", "—", "NONE"):
            t = r.get("time")
            if isinstance(t, datetime):
                last_collection_time = t
                last_collection_anomaly_score = float(r.get("anomaly_score", 0.0))
            break

    return TrackState(
        uncertainty_km=uncertainty_km,
        last_collection_time=last_collection_time,
        last_collection_anomaly_score=last_collection_anomaly_score,
    )


def panel_summary(fa: FusionAssessment, decision: Decision) -> str:
    """One-sentence summary for the top of the reasoning panel.

    Args:
        fa:       FusionAssessment for the current timestep.
        decision: Decision for the current timestep.

    Returns:
        Markdown-formatted summary string.
    """
    fs = fa.fused_score
    unc = fa.uncertainty
    fs_word = "elevated" if fs >= 0.60 else ("moderate" if fs >= 0.35 else "low")
    unc_word = "unresolved" if unc >= 0.50 else ("moderate" if unc >= 0.30 else "low")
    action = decision.action.replace("_", " ")
    return (
        f"Fused significance is **{fs_word}** ({fs:.2f}), "
        f"uncertainty is **{unc_word}** ({unc:.2f}), "
        f"and the system recommends **{action}** as the highest-value action."
    )
