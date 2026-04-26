"""Callbacks for the Whitsun decision-trace replay panel.

A single ``selected_event_id`` drives every output panel.  Panels
**progressively reveal** based on the selected event's ordinal: a panel
shows only when an artifact has been revealed at or before the
currently selected step.  If nothing is yet available, panels render a
"not available at this step" placeholder.

Reveal ordinals (per the dispatch):

  01  scenario init only — no observations, no options, no scores,
      no policy, no human, no outcome, no counterfactuals
  02  + Umbra SAR observation
  03  + VLM evidence + detections
  04  + Sentinel-2 context observation
  05  + Sentinel-1 context observation
  06  + candidate tracks
  07  + custody-state snapshot (custody risk increases)
  08  + candidate tasking options table (options shown without score
      rationale yet)
  09  + score breakdowns populated in the options table
  10  + selected recommendation + policy rationale + score breakdown
      panel
  11  + human approval
  12  + follow-up collection (the followup observation row)
  13  + outcome
  14  + counterfactuals + follow-up recommendation

This module is read-only.  It does not mutate the trace, contact any
service, or change planner / RL runtime behaviour.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dash import Dash, Input, Output, html
import dash_bootstrap_components as dbc

from custody.demo import load_whitsun_decision_trace
from custody.demo.decision_trace import DecisionTrace

from layout.whitsun_replay import (
    WHITSUN_CONTEXT_MAP,
    WHITSUN_COUNTERFACTUALS,
    WHITSUN_EVENT_SUMMARY,
    WHITSUN_FOLLOWUP,
    WHITSUN_HEADER,
    WHITSUN_HUMAN_ACTION,
    WHITSUN_OBSERVATIONS_PANEL,
    WHITSUN_OPTIONS_TABLE,
    WHITSUN_OUTCOME,
    WHITSUN_POLICY_RATIONALE,
    WHITSUN_SCORE_BREAKDOWN,
    WHITSUN_SELECTED_EVENT_STORE,
    WHITSUN_TIMELINE_RADIO,
    data_mode_badge,
    source_badge,
)


_TEXT = "#e5e7eb"
_MUTED = "#9ca3af"
_ACCENT = "#5eead4"
_WARN = "#fbbf24"


# ---------------------------------------------------------------------------
# Per-panel reveal ordinals (panel becomes visible at this ordinal).
# ---------------------------------------------------------------------------


REVEAL_ORDINALS: Mapping[str, int] = {
    "observations": 2,
    "options_table": 8,
    "options_scores_visible": 9,
    "options_selected_visible": 10,
    "score_breakdown": 10,
    "policy_rationale": 10,
    "human_action": 11,
    "outcome": 13,
    "counterfactuals": 14,
    "followup_recommendation": 14,
}


# ---------------------------------------------------------------------------
# Module-level cache: trace + reveal index are computed once.
# ---------------------------------------------------------------------------


_TRACE: DecisionTrace = load_whitsun_decision_trace()


def _build_reveal_index(trace: DecisionTrace) -> dict[str, int]:
    """For every artifact id, return the minimum event ordinal that references it."""
    reveal: dict[str, int] = {}

    def _record(_id, _ord):
        if not _id:
            return
        prev = reveal.get(_id)
        if prev is None or _ord < prev:
            reveal[_id] = _ord

    for ev in trace.events:
        ord_ = int(ev.get("ordinal", 0))
        refs = ev.get("refs") or {}
        for key in (
            "observation_id", "track_id", "custody_state_id",
            "selected_recommendation_id", "policy_rationale_id",
            "human_action_id", "outcome_id", "followup_recommendation_id",
        ):
            _record(refs.get(key), ord_)
        for key in (
            "track_ids", "evidence_artifact_ids", "detection_ids",
            "tasking_option_ids", "score_breakdown_ids",
            "counterfactual_ids",
        ):
            for v in refs.get(key) or ():
                _record(v, ord_)
    return reveal


_REVEAL_INDEX: Mapping[str, int] = _build_reveal_index(_TRACE)


# ---------------------------------------------------------------------------
# Small render utilities
# ---------------------------------------------------------------------------


def _na(message: str = "not available at this step") -> html.Div:
    return html.Div(message, style={"color": _MUTED, "fontStyle": "italic"})


def _kv(label: str, value) -> html.Div:
    return html.Div(
        [
            html.Span(
                f"{label}: ",
                style={"color": _MUTED, "fontSize": "0.78rem"},
            ),
            html.Span(
                str(value), style={"color": _TEXT, "fontSize": "0.85rem"},
            ),
        ],
        style={"marginBottom": "2px"},
    )


def _caveats_block(caveats: list[str] | tuple[str, ...] | None) -> html.Div:
    if not caveats:
        return html.Div()
    return html.Ul(
        [html.Li(c, style={"fontSize": "0.72rem", "color": _MUTED})
         for c in caveats],
        style={"margin": "4px 0", "paddingLeft": "16px"},
    )


def _ord_for(event: Mapping[str, Any] | None) -> int:
    if not event:
        return 0
    try:
        return int(event.get("ordinal", 0))
    except (TypeError, ValueError):
        return 0


def _revealed(artifact_id: str | None, current_ord: int) -> bool:
    if not artifact_id:
        return False
    o = _REVEAL_INDEX.get(artifact_id)
    return o is not None and o <= current_ord


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def _render_header(event: Mapping[str, Any]) -> html.Div:
    if not event:
        return _na()
    return html.Div(
        [
            html.Div(
                [
                    html.Strong(
                        f"{int(event.get('ordinal', 0)):02d}. {event.get('label', '')}",
                        style={"color": _ACCENT, "fontSize": "0.95rem"},
                    ),
                    html.Span(
                        f"  ·  {event.get('timestamp', '')}",
                        style={"color": _MUTED, "fontSize": "0.78rem"},
                    ),
                ],
            ),
            html.Div(
                [
                    data_mode_badge(event.get("data_mode")),
                    html.Span(
                        f"kind: {event.get('kind', '(unknown)')}",
                        style={"color": _MUTED, "fontSize": "0.75rem"},
                    ),
                ],
                style={"marginTop": "4px"},
            ),
        ],
    )


def _render_event_summary(event: Mapping[str, Any]) -> html.Div:
    if not event:
        return _na()
    return html.Div(
        [
            html.Div(
                event.get("summary", ""),
                style={"color": _TEXT, "fontSize": "0.85rem"},
            ),
            html.Div(
                [
                    _kv("event_id", event.get("event_id")),
                    _kv("kind", event.get("kind")),
                    _kv("timestamp", event.get("timestamp")),
                ],
                style={"marginTop": "6px"},
            ),
        ],
    )


def _render_observations(
    trace: DecisionTrace, current_ord: int,
) -> html.Div:
    """Cumulative reveal of every observation / artifact / detection."""
    if current_ord < REVEAL_ORDINALS["observations"]:
        return _na()
    rows: list[html.Div] = []

    obs_seen: set[str] = set()
    art_seen: set[str] = set()
    det_seen: set[str] = set()

    # Walk events in ordinal order; collect refs whose reveal ordinal
    # is <= current_ord.  This preserves chronology in the panel.
    for ev in sorted(trace.events, key=lambda e: int(e.get("ordinal", 0))):
        ev_ord = int(ev.get("ordinal", 0))
        if ev_ord > current_ord:
            break
        refs = ev.get("refs") or {}
        oid = refs.get("observation_id")
        if oid and oid not in obs_seen:
            obs = trace.get_observation(oid) or {}
            if obs:
                obs_seen.add(oid)
                rows.append(
                    html.Div(
                        [
                            html.Div(
                                [
                                    source_badge(obs.get("source")),
                                    data_mode_badge(obs.get("data_mode")),
                                    html.Strong(
                                        obs.get("observation_id", oid),
                                        style={"color": _TEXT},
                                    ),
                                ],
                            ),
                            _kv(
                                "confidence_weight",
                                obs.get("confidence_weight", "n/a"),
                            ),
                            _kv(
                                "usable_for_detection",
                                obs.get("usable_for_detection"),
                            ),
                            _kv(
                                "usable_for_context",
                                obs.get("usable_for_context"),
                            ),
                            _kv("timestamp", obs.get("timestamp")),
                            _caveats_block(obs.get("caveats")),
                        ],
                        style={
                            "padding": "6px 0",
                            "borderBottom": "1px solid #2d2d2d",
                        },
                    )
                )
        for aid in refs.get("evidence_artifact_ids") or ():
            if aid in art_seen:
                continue
            art = trace.get_evidence_artifact(aid)
            if not art:
                continue
            art_seen.add(aid)
            rows.append(
                html.Div(
                    [
                        html.Div(
                            [
                                html.Strong(
                                    f"evidence: {art.get('evidence_artifact_id', aid)}",
                                    style={"color": _TEXT},
                                ),
                                data_mode_badge(art.get("data_mode")),
                            ],
                        ),
                        _kv("kind", art.get("kind")),
                        _kv("summary", art.get("summary")),
                        _kv("confidence", art.get("confidence")),
                        _caveats_block(art.get("caveats")),
                    ],
                    style={
                        "padding": "6px 0",
                        "borderBottom": "1px solid #2d2d2d",
                    },
                )
            )
        for did in refs.get("detection_ids") or ():
            if did in det_seen:
                continue
            det = trace.get_detection(did)
            if not det:
                continue
            det_seen.add(did)
            rows.append(
                html.Div(
                    [
                        html.Strong(
                            f"detection: {det.get('detection_id', did)}",
                            style={"color": _TEXT},
                        ),
                        _kv("score", det.get("score")),
                        _kv(
                            "lat / lon",
                            f"{det.get('lat')} / {det.get('lon')}",
                        ),
                        _kv(
                            "vessel_length_est_m",
                            det.get("vessel_length_est_m"),
                        ),
                    ],
                    style={
                        "padding": "6px 0",
                        "borderBottom": "1px solid #2d2d2d",
                    },
                )
            )

    if not rows:
        return _na()
    return html.Div(rows)


def _render_options_table(
    trace: DecisionTrace, current_ord: int,
) -> html.Div:
    if current_ord < REVEAL_ORDINALS["options_table"]:
        return _na()
    rows = trace.options_table()
    show_scores = current_ord >= REVEAL_ORDINALS["options_scores_visible"]
    show_selected = current_ord >= REVEAL_ORDINALS["options_selected_visible"]

    header_cells = [
        html.Th("label"), html.Th("collect type"),
        html.Th("latency h"), html.Th("cost"),
    ]
    if show_scores:
        header_cells.append(html.Th("score"))
    if show_selected:
        header_cells.append(html.Th("selected"))

    header = html.Tr(
        header_cells,
        style={"color": _MUTED, "fontSize": "0.75rem"},
    )
    body = []
    for row in rows:
        is_selected = bool(row.get("is_selected"))
        cells = [
            html.Td(row.get("label")),
            html.Td(row.get("candidate_collect_type")),
            html.Td(row.get("expected_latency_hours")),
            html.Td(row.get("expected_cost_units")),
        ]
        if show_scores:
            ts = row.get("total_score")
            cells.append(
                html.Td(f"{ts:.2f}" if ts is not None else "n/a"),
            )
        if show_selected:
            cells.append(
                html.Td(
                    dbc.Badge("SELECTED", color="success")
                    if is_selected else "",
                ),
            )
        body.append(
            html.Tr(
                cells,
                style={
                    "color": (
                        _ACCENT if (show_selected and is_selected) else _TEXT
                    ),
                    "fontSize": "0.78rem",
                },
            )
        )
    return html.Table(
        [html.Thead(header), html.Tbody(body)],
        style={"width": "100%", "borderCollapse": "collapse"},
    )


def _render_score_breakdown(
    trace: DecisionTrace, current_ord: int,
) -> html.Div:
    if current_ord < REVEAL_ORDINALS["score_breakdown"]:
        return _na()
    rec = trace.selected_recommendation
    if not rec:
        return _na("no selected recommendation")
    sb = trace.get_score_breakdown_for_option(
        str(rec.get("tasking_option_id"))
    )
    if not sb:
        return _na("no score breakdown for selected option")
    components = sb.get("components") or {}
    rows = [
        html.Tr(
            [
                html.Td(k),
                html.Td(
                    f"{v:+.2f}" if isinstance(v, (int, float)) else str(v),
                    style={"textAlign": "right"},
                ),
            ],
            style={
                "color": _TEXT, "fontSize": "0.78rem",
                "borderBottom": "1px solid #2d2d2d",
            },
        )
        for k, v in components.items()
    ]
    rows.append(
        html.Tr(
            [
                html.Td(html.Strong("total_score")),
                html.Td(
                    html.Strong(f"{sb.get('total_score'):.2f}"),
                    style={"textAlign": "right", "color": _ACCENT},
                ),
            ],
            style={"fontSize": "0.85rem"},
        )
    )
    return html.Div(
        [
            html.Div(
                f"selected option: {rec.get('tasking_option_id')}",
                style={"color": _MUTED, "fontSize": "0.75rem"},
            ),
            html.Table(
                [html.Tbody(rows)],
                style={"width": "100%", "borderCollapse": "collapse"},
            ),
        ],
    )


def _render_policy_rationale(
    trace: DecisionTrace, current_ord: int,
) -> html.Div:
    if current_ord < REVEAL_ORDINALS["policy_rationale"]:
        return _na()
    pol = trace.policy_rationale
    if not pol:
        return _na()
    return html.Div(
        [
            html.Div(
                [
                    dbc.Badge(
                        "HEURISTIC ADVISORY",
                        color="info", className="me-2",
                    ),
                    dbc.Badge(
                        "RL-READY SLOT (NO TRAINED RL)",
                        color="warning", className="me-2",
                    ),
                    data_mode_badge(pol.get("data_mode")),
                ],
            ),
            html.Div(
                pol.get("summary", ""),
                style={
                    "color": _TEXT, "fontSize": "0.82rem",
                    "marginTop": "6px",
                },
            ),
            html.Div(
                pol.get("rl_advisory", ""),
                style={
                    "color": _MUTED, "fontSize": "0.72rem",
                    "marginTop": "6px", "fontStyle": "italic",
                },
            ),
            _caveats_block(pol.get("caveats")),
        ],
    )


def _render_human_action(
    trace: DecisionTrace, current_ord: int,
) -> html.Div:
    if current_ord < REVEAL_ORDINALS["human_action"]:
        return _na()
    ha = trace.human_action
    if not ha:
        return _na()
    return html.Div(
        [
            html.Div(
                [
                    dbc.Badge(
                        (ha.get("action") or "n/a").upper(),
                        color=(
                            "success" if ha.get("action") == "approve"
                            else "secondary"
                        ),
                        className="me-2",
                    ),
                    data_mode_badge(ha.get("data_mode")),
                ],
            ),
            _kv("operator_id", ha.get("operator_id")),
            _kv("decided_at", ha.get("decided_at")),
            _kv("tasking_option_id", ha.get("tasking_option_id")),
            html.Div(
                ha.get("reason", ""),
                style={
                    "color": _TEXT, "fontSize": "0.8rem",
                    "marginTop": "6px",
                },
            ),
            _caveats_block(ha.get("caveats")),
        ],
    )


def _render_outcome(
    trace: DecisionTrace, current_ord: int,
) -> html.Div:
    if current_ord < REVEAL_ORDINALS["outcome"]:
        return _na()
    out = trace.outcome
    if not out:
        return _na()
    return html.Div(
        [
            html.Div(
                [
                    dbc.Badge(
                        (out.get("result") or "n/a").upper(),
                        color="success", className="me-2",
                    ),
                    data_mode_badge(out.get("data_mode")),
                ],
            ),
            _kv("track_id", out.get("track_id")),
            _kv("recorded_at", out.get("recorded_at")),
            _kv(
                "custody_score_delta",
                f"{out.get('custody_score_delta', 0):+.2f}",
            ),
            html.Div(
                out.get("summary", ""),
                style={
                    "color": _TEXT, "fontSize": "0.8rem",
                    "marginTop": "6px",
                },
            ),
            _caveats_block(out.get("caveats")),
        ],
    )


def _render_counterfactuals(
    trace: DecisionTrace, current_ord: int,
) -> html.Div:
    if current_ord < REVEAL_ORDINALS["counterfactuals"]:
        return _na()
    cfs = trace.counterfactuals
    if not cfs:
        return _na()
    rows = []
    for cf in cfs:
        rows.append(
            html.Tr(
                [
                    html.Td(cf.get("alternative_tasking_option_id")),
                    html.Td(cf.get("expected_result")),
                    html.Td(
                        f"{cf.get('expected_custody_score_delta', 0):+.2f}",
                        style={"textAlign": "right"},
                    ),
                    html.Td(
                        cf.get("summary", ""),
                        style={"fontSize": "0.74rem", "color": _MUTED},
                    ),
                ],
                style={
                    "color": _TEXT, "fontSize": "0.78rem",
                    "borderBottom": "1px solid #2d2d2d",
                },
            ),
        )
    header = html.Tr(
        [
            html.Th("alternative"), html.Th("expected"),
            html.Th("impact vs baseline (Δ score)"), html.Th("summary"),
        ],
        style={"color": _MUTED, "fontSize": "0.75rem"},
    )
    caption = html.Div(
        "Δ score is the expected custody-score delta for the alternative "
        "option, relative to the pre-decision baseline (cs-snap-002), "
        "not relative to the realised SAT-B outcome.",
        style={
            "color": _MUTED, "fontSize": "0.7rem",
            "fontStyle": "italic", "marginBottom": "4px",
        },
    )
    return html.Div(
        [
            caption,
            html.Table(
                [html.Thead(header), html.Tbody(rows)],
                style={"width": "100%", "borderCollapse": "collapse"},
            ),
        ],
    )


def _render_followup(
    trace: DecisionTrace, current_ord: int,
) -> html.Div:
    if current_ord < REVEAL_ORDINALS["followup_recommendation"]:
        return _na()
    fr = trace.followup_recommendation
    if not fr:
        return _na()
    thresholds = fr.get("next_candidate_review_thresholds") or {}
    return html.Div(
        [
            html.Div(
                [
                    data_mode_badge(fr.get("data_mode")),
                    html.Span(
                        fr.get("next_candidate_collect_type", ""),
                        style={"color": _ACCENT, "fontSize": "0.85rem"},
                    ),
                ],
            ),
            html.Div(
                fr.get("summary", ""),
                style={
                    "color": _TEXT, "fontSize": "0.82rem",
                    "marginTop": "4px",
                },
            ),
            _kv(
                "review thresholds",
                ", ".join(f"{k}={v}" for k, v in thresholds.items())
                or "n/a",
            ),
            _caveats_block(fr.get("caveats")),
        ],
    )


def _render_context_map(
    trace: DecisionTrace, current_ord: int,
) -> html.Div:
    """Lightweight context block: AOI plus revealed tracks / custody snapshot."""
    aoi = trace.metadata
    rows = [
        _kv(
            "AOI center (lat, lon)",
            f"{aoi.get('aoi_center_lat_deg')}, {aoi.get('aoi_center_lon_deg')}",
        ),
        _kv(
            "AOI fixture",
            aoi.get("aoi_fixture_path", "n/a"),
        ),
    ]

    # Show every track whose reveal ordinal <= current_ord.
    visible_tracks = [
        t for t in trace.candidate_tracks
        if _revealed(t.get("track_id"), current_ord)
    ]
    if visible_tracks:
        rows.append(
            html.Div(
                "tracks present:",
                style={
                    "color": _MUTED, "fontSize": "0.75rem",
                    "marginTop": "6px",
                },
            ),
        )
        for t in visible_tracks:
            rows.append(
                html.Div(
                    [
                        html.Strong(
                            t.get("track_id", ""),
                            style={"color": _TEXT},
                        ),
                        html.Span(
                            f" — {t.get('label', '')} "
                            f"(AIS: {t.get('ais_state', 'unknown')})",
                            style={"color": _MUTED, "fontSize": "0.75rem"},
                        ),
                    ],
                ),
            )

    # Show the latest custody snapshot revealed at or before this ordinal.
    visible_snapshots = [
        cs for cs in trace.custody_state_snapshots
        if _revealed(cs.get("custody_state_id"), current_ord)
    ]
    if visible_snapshots:
        latest = visible_snapshots[-1]
        rows.append(
            html.Div(
                [
                    html.Span(
                        f"custody {latest.get('status', '?').upper()} "
                        f"({latest.get('score'):.2f})",
                        style={
                            "color": _ACCENT, "fontSize": "0.85rem",
                            "marginTop": "6px", "display": "inline-block",
                        },
                    ),
                ],
            ),
        )
    return html.Div(rows)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register(app: Dash) -> None:
    """Register Whitsun replay callbacks on *app*."""

    @app.callback(
        Output(WHITSUN_SELECTED_EVENT_STORE, "data"),
        Input(WHITSUN_TIMELINE_RADIO, "value"),
    )
    def _sync_selected_event(event_id):
        return event_id

    @app.callback(
        Output(WHITSUN_HEADER, "children"),
        Output(WHITSUN_EVENT_SUMMARY, "children"),
        Output(WHITSUN_OBSERVATIONS_PANEL, "children"),
        Output(WHITSUN_CONTEXT_MAP, "children"),
        Output(WHITSUN_OPTIONS_TABLE, "children"),
        Output(WHITSUN_SCORE_BREAKDOWN, "children"),
        Output(WHITSUN_POLICY_RATIONALE, "children"),
        Output(WHITSUN_HUMAN_ACTION, "children"),
        Output(WHITSUN_OUTCOME, "children"),
        Output(WHITSUN_COUNTERFACTUALS, "children"),
        Output(WHITSUN_FOLLOWUP, "children"),
        Input(WHITSUN_SELECTED_EVENT_STORE, "data"),
    )
    def _refresh_all_panels(event_id):
        event = _TRACE.get_event(event_id) if event_id else {}
        ord_ = _ord_for(event)
        return (
            _render_header(event or {}),
            _render_event_summary(event or {}),
            _render_observations(_TRACE, ord_),
            _render_context_map(_TRACE, ord_),
            _render_options_table(_TRACE, ord_),
            _render_score_breakdown(_TRACE, ord_),
            _render_policy_rationale(_TRACE, ord_),
            _render_human_action(_TRACE, ord_),
            _render_outcome(_TRACE, ord_),
            _render_counterfactuals(_TRACE, ord_),
            _render_followup(_TRACE, ord_),
        )
