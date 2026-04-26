"""Demo / replay assets for the Custody dashboard.

Currently exposes the Whitsun decision-trace loader used by the Dash
replay view.  Demo modules read committed JSON fixtures and do not
contact any external service, run any model, or modify decision-layer
runtime behaviour.
"""

from custody.demo.decision_trace import (
    DecisionTrace,
    WHITSUN_DECISION_TRACE_PATH,
    load_whitsun_decision_trace,
)
from custody.demo.evidence import (
    EVIDENCE_MANIFEST_PATH,
    EvidenceDetection,
    EvidenceScene,
    evidence_scene_for_whitsun_event_ordinal,
    evidence_scenes_for_scenario,
    load_evidence_manifest,
)
from custody.demo.map_overlays import (
    MAP_OVERLAYS_PATH,
    OverlayArtifact,
    available_overlays_for,
    has_image_asset,
    load_map_overlays,
    overlay_by_id,
    overlays_for_scenario,
)

__all__ = [
    "DecisionTrace",
    "EVIDENCE_MANIFEST_PATH",
    "EvidenceDetection",
    "EvidenceScene",
    "MAP_OVERLAYS_PATH",
    "OverlayArtifact",
    "WHITSUN_DECISION_TRACE_PATH",
    "available_overlays_for",
    "evidence_scene_for_whitsun_event_ordinal",
    "evidence_scenes_for_scenario",
    "has_image_asset",
    "load_evidence_manifest",
    "load_map_overlays",
    "load_whitsun_decision_trace",
    "overlay_by_id",
    "overlays_for_scenario",
]
