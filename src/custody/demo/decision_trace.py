"""Whitsun decision-trace loader.

Reads ``data/demo/whitsun_decision_trace.fixture.json`` into a frozen
:class:`DecisionTrace` accessor and exposes lookup helpers used by the
Dash replay view.

The loader is read-only.  No live data fetch, no external service
call, and no decision-layer runtime mutation.
"""
from __future__ import annotations

import json as _json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
WHITSUN_DECISION_TRACE_PATH = (
    REPO_ROOT / "data" / "demo" / "whitsun_decision_trace.fixture.json"
)


@dataclass(frozen=True)
class DecisionTrace:
    """Structured accessor over the Whitsun decision-trace fixture."""

    raw: Mapping[str, Any]

    # ---- Top-level sections -----------------------------------------------

    @property
    def schema(self) -> str:
        return str(self.raw.get("schema", ""))

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self.raw.get("scenario_metadata") or {}

    @property
    def events(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.raw.get("events") or ())

    @property
    def observations(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.raw.get("observations") or ())

    @property
    def evidence_artifacts(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.raw.get("evidence_artifacts") or ())

    @property
    def detections(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.raw.get("detections") or ())

    @property
    def candidate_tracks(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.raw.get("candidate_tracks") or ())

    @property
    def custody_state_snapshots(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.raw.get("custody_state_snapshots") or ())

    @property
    def candidate_tasking_options(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.raw.get("candidate_tasking_options") or ())

    @property
    def score_breakdowns(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.raw.get("score_breakdowns") or ())

    @property
    def selected_recommendation(self) -> Mapping[str, Any]:
        return self.raw.get("selected_recommendation") or {}

    @property
    def policy_rationale(self) -> Mapping[str, Any]:
        return self.raw.get("policy_rationale") or {}

    @property
    def human_action(self) -> Mapping[str, Any]:
        return self.raw.get("human_action") or {}

    @property
    def outcome(self) -> Mapping[str, Any]:
        return self.raw.get("outcome") or {}

    @property
    def counterfactuals(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.raw.get("counterfactuals") or ())

    @property
    def followup_recommendation(self) -> Mapping[str, Any]:
        return self.raw.get("followup_recommendation") or {}

    # ---- Lookups ----------------------------------------------------------

    def get_event(self, event_id: str) -> Mapping[str, Any] | None:
        for ev in self.events:
            if ev.get("event_id") == event_id:
                return ev
        return None

    def get_observation(self, observation_id: str) -> Mapping[str, Any] | None:
        for o in self.observations:
            if o.get("observation_id") == observation_id:
                return o
        return None

    def get_evidence_artifact(
        self, evidence_artifact_id: str,
    ) -> Mapping[str, Any] | None:
        for a in self.evidence_artifacts:
            if a.get("evidence_artifact_id") == evidence_artifact_id:
                return a
        return None

    def get_detection(self, detection_id: str) -> Mapping[str, Any] | None:
        for d in self.detections:
            if d.get("detection_id") == detection_id:
                return d
        return None

    def get_track(self, track_id: str) -> Mapping[str, Any] | None:
        for t in self.candidate_tracks:
            if t.get("track_id") == track_id:
                return t
        return None

    def get_custody_state(
        self, custody_state_id: str,
    ) -> Mapping[str, Any] | None:
        for cs in self.custody_state_snapshots:
            if cs.get("custody_state_id") == custody_state_id:
                return cs
        return None

    def get_tasking_option(
        self, tasking_option_id: str,
    ) -> Mapping[str, Any] | None:
        for o in self.candidate_tasking_options:
            if o.get("tasking_option_id") == tasking_option_id:
                return o
        return None

    def get_score_breakdown(
        self, score_breakdown_id: str,
    ) -> Mapping[str, Any] | None:
        for sb in self.score_breakdowns:
            if sb.get("score_breakdown_id") == score_breakdown_id:
                return sb
        return None

    def get_score_breakdown_for_option(
        self, tasking_option_id: str,
    ) -> Mapping[str, Any] | None:
        for sb in self.score_breakdowns:
            if sb.get("tasking_option_id") == tasking_option_id:
                return sb
        return None

    def get_counterfactual(
        self, counterfactual_id: str,
    ) -> Mapping[str, Any] | None:
        for cf in self.counterfactuals:
            if cf.get("counterfactual_id") == counterfactual_id:
                return cf
        return None

    # ---- Convenience views ------------------------------------------------

    def event_ids(self) -> tuple[str, ...]:
        return tuple(str(e.get("event_id")) for e in self.events)

    def first_event_id(self) -> str | None:
        ids = self.event_ids()
        return ids[0] if ids else None

    def event_label_pairs(self) -> tuple[tuple[str, str], ...]:
        """``((event_id, label), ...)`` ordered by ordinal."""
        ordered = sorted(
            self.events, key=lambda e: int(e.get("ordinal", 0)),
        )
        return tuple(
            (str(e.get("event_id")), str(e.get("label", "")))
            for e in ordered
        )

    def options_table(self) -> tuple[Mapping[str, Any], ...]:
        """Per-option row with score breakdown and selection flag."""
        selected_id = self.selected_recommendation.get("tasking_option_id")
        rows: list[dict[str, Any]] = []
        for opt in self.candidate_tasking_options:
            sb = self.get_score_breakdown_for_option(
                str(opt.get("tasking_option_id")),
            )
            rows.append({
                "tasking_option_id": opt.get("tasking_option_id"),
                "label": opt.get("label"),
                "candidate_collect_type": opt.get("candidate_collect_type"),
                "expected_latency_hours": opt.get("expected_latency_hours"),
                "expected_cost_units": opt.get("expected_cost_units"),
                "components": (sb or {}).get("components", {}),
                "total_score": (sb or {}).get("total_score"),
                "is_selected": opt.get("tasking_option_id") == selected_id,
                "data_mode": opt.get("data_mode"),
            })
        return tuple(rows)

    # ---- Light validation -------------------------------------------------

    def validate(self) -> tuple[str, ...]:
        """Return a tuple of human-readable validation issues; empty if ok."""
        issues: list[str] = []
        obs_ids = {o.get("observation_id") for o in self.observations}
        track_ids = {t.get("track_id") for t in self.candidate_tracks}
        custody_ids = {
            c.get("custody_state_id") for c in self.custody_state_snapshots
        }
        opt_ids = {
            o.get("tasking_option_id")
            for o in self.candidate_tasking_options
        }
        sb_ids = {sb.get("score_breakdown_id") for sb in self.score_breakdowns}
        det_ids = {d.get("detection_id") for d in self.detections}
        art_ids = {
            a.get("evidence_artifact_id") for a in self.evidence_artifacts
        }
        cf_ids = {cf.get("counterfactual_id") for cf in self.counterfactuals}

        for ev in self.events:
            refs = ev.get("refs") or {}
            event_id = ev.get("event_id")
            if "observation_id" in refs and refs["observation_id"] not in obs_ids:
                issues.append(
                    f"{event_id}: unknown observation_id {refs['observation_id']}"
                )
            for tid in refs.get("track_ids", []):
                if tid not in track_ids:
                    issues.append(f"{event_id}: unknown track_id {tid}")
            for cid in refs.get("counterfactual_ids", []):
                if cid not in cf_ids:
                    issues.append(f"{event_id}: unknown counterfactual_id {cid}")
            for did in refs.get("detection_ids", []):
                if did not in det_ids:
                    issues.append(f"{event_id}: unknown detection_id {did}")
            for aid in refs.get("evidence_artifact_ids", []):
                if aid not in art_ids:
                    issues.append(
                        f"{event_id}: unknown evidence_artifact_id {aid}"
                    )
            for oid in refs.get("tasking_option_ids", []):
                if oid not in opt_ids:
                    issues.append(f"{event_id}: unknown tasking_option_id {oid}")
            for sid in refs.get("score_breakdown_ids", []):
                if sid not in sb_ids:
                    issues.append(f"{event_id}: unknown score_breakdown_id {sid}")
            cs = refs.get("custody_state_id")
            if cs is not None and cs not in custody_ids:
                issues.append(f"{event_id}: unknown custody_state_id {cs}")
        return tuple(issues)


def load_whitsun_decision_trace(
    path: str | Path | None = None,
) -> DecisionTrace:
    """Load and return the Whitsun decision trace as a :class:`DecisionTrace`."""
    p = Path(path) if path is not None else WHITSUN_DECISION_TRACE_PATH
    raw = _json.loads(p.read_text(encoding="utf-8"))
    return DecisionTrace(raw=raw)
