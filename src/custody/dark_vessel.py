"""
Dark-vessel detection and last-known-state management.

A vessel is 'dark' when its AIS transponder goes silent.  The simulation
models this via ``ais_dropout_hour`` on ``VesselSpec``; real deployments
would detect the gap from an AIS feed.

Dark vessels are distinguished from vessels that simply lack recent sensor
collection:

  dark        — transponder silent; position truth is no longer emitted.
                Last-known position/time/anomaly are frozen in TrackState.
                Uncertainty grows from the last-known state.

  uncollected — AIS still active; tasking gap; position truth is still
                visible but no ISR sensor has been pointed at the vessel.

Relevance doctrine
------------------
Not every dark vessel is operationally significant.  Dark condition matters
when the vessel is already being tracked (WATCHLIST / ACTIVE_CUSTODY) or
when an operator directive demands continuous custody.  Background vessels
with no directive are not relevant — transient AIS gaps are routine for
untracked traffic and must not pollute the portfolio.

Public API
----------
is_dark_relevant(attention_state, tracking_directive) -> bool
    True when dark status should trigger ACTIVE_CUSTODY escalation.

mark_dark(track, vessel, timestamp, anomaly_score) -> None
    Freeze last-known state at the first dark timestep.  Mutates TrackState.
    Callers must check ``track.is_dark`` before calling to avoid overwriting.

build_dark_reason(dark_since, current_time) -> str
    Human-readable explanation for the dark_vessel_reason record field.
"""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from custody.models import TrackState, Vessel

from custody.orchestration.attention import (
    WATCHLIST,
    ACTIVE_CUSTODY,
    DIRECTIVE_MAINTAIN_CUSTODY,
)


def is_dark_relevant(attention_state: str, tracking_directive: str) -> bool:
    """Return True if a dark-vessel event is operationally significant.

    Relevant when the vessel is already tracked (WATCHLIST / ACTIVE_CUSTODY)
    or when the operator has issued a MAINTAIN_CUSTODY directive.
    Background vessels with no directive remain irrelevant even when dark.
    """
    if tracking_directive == DIRECTIVE_MAINTAIN_CUSTODY:
        return True
    return attention_state in (WATCHLIST, ACTIVE_CUSTODY)


def mark_dark(
    track: "TrackState",
    vessel: "Vessel",
    timestamp: datetime,
    anomaly_score: float,
) -> None:
    """Freeze last-known state when a vessel's AIS signal is lost.

    Mutates ``track`` in-place.  Must only be called once; callers should
    guard with ``if not track.is_dark``.
    """
    track.is_dark          = True
    track.dark_since       = timestamp
    track.last_known_lat   = vessel.lat
    track.last_known_lon   = vessel.lon
    track.last_known_time  = timestamp
    track.last_known_anomaly = anomaly_score


def build_dark_reason(dark_since: datetime, current_time: datetime) -> str:
    """Return a human-readable explanation of how long the vessel has been dark.

    Args:
        dark_since:   Timestamp when AIS loss was first detected.
        current_time: Current simulation timestamp.

    Returns:
        Short string describing elapsed dark time.
    """
    elapsed_h = (current_time - dark_since).total_seconds() / 3600.0
    if elapsed_h < 0.5:
        return "AIS signal just lost; position reflects last known contact"
    return f"AIS dark {elapsed_h:.1f}h; position reflects last known contact"
