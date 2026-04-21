"""Temporal persistence analysis between Scene pairs (ADR-0018 downstream).

A :class:`Matcher` strategy pairs observations across two Scenes and the
:func:`temporal_persistence` function composes the matcher into a full
classification: matched pairs (persistent), unmatched B-only (emerged),
unmatched A-only (disappeared).

:class:`DirectSpatialMatcher` is the Week-3 approach: nearest-neighbor
matching under a fixed meter gate, projected to tangent-plane meters via
:mod:`custody.fusion.geo`.  Global minimum-total-distance pairing via
Hungarian assignment (``scipy.optimize.linear_sum_assignment``); pairs
outside the gate are dropped.

Geometry-aware matchers for heterogeneous scene pairs (across-orbit,
across-resolution) are future work per ADR-0018's downstream ADR candidates;
the :class:`Matcher` protocol is the extension point.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.optimize import linear_sum_assignment

from custody.fusion.geo import to_tangent_plane_array
from custody.fusion.observations import PositionObservation
from custody.fusion.scenes import Scene


# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Match:
    """One matched observation pair across two scenes."""

    obs_a_id: str
    obs_b_id: str
    distance_m: float


@dataclass(frozen=True)
class TemporalComparisonResult:
    """Outcome of a :func:`temporal_persistence` run.

    ``matches`` are the persistent pairs.  ``emerged`` and ``disappeared`` are
    tuples of obs_id values from scene B and scene A respectively that had no
    cross-scene match under the matcher's rules.
    """

    scene_a_id: str
    scene_b_id: str
    matcher_name: str
    matches: tuple[Match, ...]
    emerged: tuple[str, ...]
    disappeared: tuple[str, ...]
    match_distances_m: tuple[float, ...]


# ---------------------------------------------------------------------------
# Matcher protocol + direct spatial implementation
# ---------------------------------------------------------------------------


class Matcher(Protocol):
    """Pluggable matching strategy for cross-scene observation association."""

    def match(
        self,
        obs_a: tuple[PositionObservation, ...],
        obs_b: tuple[PositionObservation, ...],
    ) -> list[Match]:
        ...


_BIG_COST = 1.0e9


@dataclass(frozen=True)
class DirectSpatialMatcher:
    """Nearest-neighbor matching within a fixed meter gate (Week-3 Tennent approach).

    Distance computed in tangent-plane meters (AEQD anchored at the project
    AOI center per :mod:`custody.fusion.geo`); accurate to sub-millimeter over
    the ~2 km AOIs of interest.  Pair assignment via
    ``scipy.optimize.linear_sum_assignment`` with a ``_BIG_COST`` sentinel for
    entries beyond the gate, producing the global minimum-total-distance
    pairing subject to the gate.  Pairs whose cost lands on the sentinel are
    dropped post-hoc.

    One-to-many handling: when two obs in A are both within gate of one obs in
    B, the Hungarian solver picks the closer pair; the other A obs is
    unmatched (→ disappeared on the :func:`temporal_persistence` side).
    """

    gate_m: float

    @property
    def name(self) -> str:
        return f"DirectSpatialMatcher(gate_m={self.gate_m})"

    def match(
        self,
        obs_a: tuple[PositionObservation, ...],
        obs_b: tuple[PositionObservation, ...],
    ) -> list[Match]:
        if not obs_a or not obs_b:
            return []

        lats_a = np.array([o.lat for o in obs_a], dtype=float)
        lons_a = np.array([o.lon for o in obs_a], dtype=float)
        lats_b = np.array([o.lat for o in obs_b], dtype=float)
        lons_b = np.array([o.lon for o in obs_b], dtype=float)
        xs_a, ys_a = to_tangent_plane_array(lats_a, lons_a)
        xs_b, ys_b = to_tangent_plane_array(lats_b, lons_b)

        dx = xs_a[:, None] - xs_b[None, :]
        dy = ys_a[:, None] - ys_b[None, :]
        dist = np.hypot(dx, dy)
        cost = np.where(dist <= self.gate_m, dist, _BIG_COST)

        row_idx, col_idx = linear_sum_assignment(cost)
        out: list[Match] = []
        for i, j in zip(row_idx, col_idx):
            c = cost[i, j]
            if c < _BIG_COST:
                out.append(Match(
                    obs_a_id=obs_a[int(i)].obs_id,
                    obs_b_id=obs_b[int(j)].obs_id,
                    distance_m=float(c),
                ))
        return out


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def temporal_persistence(
    scene_a: Scene,
    scene_b: Scene,
    *,
    matcher: Matcher,
) -> TemporalComparisonResult:
    """Compare two Scenes and classify observations by cross-scene persistence.

    Preconditions:

    - ``scene_a.scene_id != scene_b.scene_id`` — a scene compared with itself
      has no meaningful persistence signal.
    - ``scene_a.acquisition_time < scene_b.acquisition_time`` — the result
      names "emerged" and "disappeared" assume chronological ordering.

    Returns a :class:`TemporalComparisonResult`.
    """
    if scene_a.scene_id == scene_b.scene_id:
        raise ValueError(
            f"scene_a and scene_b share scene_id {scene_a.scene_id!r}; "
            "temporal_persistence requires two distinct scenes"
        )
    if scene_a.acquisition_time >= scene_b.acquisition_time:
        raise ValueError(
            f"scene_a acquisition_time ({scene_a.acquisition_time}) must be "
            f"< scene_b acquisition_time ({scene_b.acquisition_time}); "
            "pass scenes in chronological order (older → newer)"
        )

    matches_list = matcher.match(scene_a.observations, scene_b.observations)
    matches = tuple(matches_list)

    matched_a_ids = {m.obs_a_id for m in matches}
    matched_b_ids = {m.obs_b_id for m in matches}
    disappeared = tuple(
        o.obs_id for o in scene_a.observations if o.obs_id not in matched_a_ids
    )
    emerged = tuple(
        o.obs_id for o in scene_b.observations if o.obs_id not in matched_b_ids
    )
    match_distances_m = tuple(m.distance_m for m in matches)
    matcher_name = getattr(matcher, "name", type(matcher).__name__)

    return TemporalComparisonResult(
        scene_a_id=scene_a.scene_id,
        scene_b_id=scene_b.scene_id,
        matcher_name=matcher_name,
        matches=matches,
        emerged=emerged,
        disappeared=disappeared,
        match_distances_m=match_distances_m,
    )
