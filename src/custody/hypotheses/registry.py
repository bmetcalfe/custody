"""Scenario-specific hypothesis registries for Tennent and Whitsun (ADR-0021).

Two case studies, two non-overlapping hypothesis sets (per ADR-0012 and
ADR-0013).  Priors are uniform within each scenario for Slice 1; no
defensible weighting exists yet, and deliberate weights can land with an
ADR if/when they do.

Callers should import the ``SCENARIO_*`` and ``TENNENT_*`` / ``WHITSUN_*``
hypothesis_id constants rather than stringly-typing the values.
"""
from __future__ import annotations

from custody.hypotheses.types import Hypothesis


# Scenario identifiers --------------------------------------------------------

SCENARIO_TENNENT = "tennent"
SCENARIO_WHITSUN = "whitsun"


# Tennent hypothesis IDs ------------------------------------------------------

TENNENT_FIXED_RECLAMATION_OR_STRUCTURE = "fixed_reclamation_or_structure"
TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY = "construction_or_reclamation_activity"
TENNENT_STATIONARY_SAR_SCATTER_OR_REEF_CLUTTER = "stationary_sar_scatter_or_reef_clutter"
TENNENT_TRANSIENT_VESSEL_ACTIVITY = "transient_vessel_activity"
TENNENT_NO_MEANINGFUL_ACTIVITY = "no_meaningful_activity"


# Whitsun hypothesis IDs ------------------------------------------------------

WHITSUN_VESSEL_CLUSTER_ACTIVITY = "vessel_cluster_activity"
WHITSUN_TRANSIENT_ANCHORAGE_OR_FISHING_PRESENCE = "transient_anchorage_or_fishing_presence"
WHITSUN_AIS_DARK_OR_POORLY_OBSERVED_VESSELS = "ais_dark_or_poorly_observed_vessels"
WHITSUN_DETECTOR_CLUTTER_FALSE_POSITIVES = "detector_clutter_false_positives"
WHITSUN_NO_PERSISTENT_ACTIVITY = "no_persistent_activity"


_TENNENT_SPECS: tuple[tuple[str, str, str], ...] = (
    (
        TENNENT_FIXED_RECLAMATION_OR_STRUCTURE,
        "Fixed reclamation or structure",
        "A persistent human-made feature (reclaimed land, platform, "
        "outpost) occupies the reef coordinates across scenes.",
    ),
    (
        TENNENT_CONSTRUCTION_OR_RECLAMATION_ACTIVITY,
        "Active construction or reclamation",
        "Scene-to-scene change indicates ongoing construction, dredging, "
        "or reclamation activity rather than a static structure.",
    ),
    (
        TENNENT_STATIONARY_SAR_SCATTER_OR_REEF_CLUTTER,
        "Stationary SAR scatter / reef clutter",
        "Persistent bright returns explained by natural reef scatter, "
        "exposed rock, or detector clutter rather than a built feature.",
    ),
    (
        TENNENT_TRANSIENT_VESSEL_ACTIVITY,
        "Transient vessel activity",
        "Returns are consistent with vessels present during some scenes "
        "but absent in others.",
    ),
    (
        TENNENT_NO_MEANINGFUL_ACTIVITY,
        "No meaningful activity",
        "Evidence does not support any notable activity at the reef "
        "during the observation window.",
    ),
)


_WHITSUN_SPECS: tuple[tuple[str, str, str], ...] = (
    (
        WHITSUN_VESSEL_CLUSTER_ACTIVITY,
        "Vessel cluster activity",
        "Multiple vessels are co-located at the reef across scenes in a "
        "pattern consistent with a deliberate flotilla presence.",
    ),
    (
        WHITSUN_TRANSIENT_ANCHORAGE_OR_FISHING_PRESENCE,
        "Transient anchorage or fishing presence",
        "Scattered vessel returns are consistent with ordinary anchorage "
        "or fishing presence rather than coordinated clustering.",
    ),
    (
        WHITSUN_AIS_DARK_OR_POORLY_OBSERVED_VESSELS,
        "AIS-dark or poorly observed vessels",
        "SAR returns exist where AIS coverage is known to be sparse, "
        "suggesting vessels not transmitting or not captured by GFW.",
    ),
    (
        WHITSUN_DETECTOR_CLUTTER_FALSE_POSITIVES,
        "Detector clutter / false positives",
        "Returns are consistent with SAR clutter or detector false "
        "positives rather than genuine vessels.",
    ),
    (
        WHITSUN_NO_PERSISTENT_ACTIVITY,
        "No persistent activity",
        "Evidence does not support a persistent vessel presence at the "
        "reef during the observation window.",
    ),
)


def _build(scenario_id: str, specs: tuple[tuple[str, str, str], ...]) -> tuple[Hypothesis, ...]:
    prior = 1.0 / len(specs)
    return tuple(
        Hypothesis(
            hypothesis_id=hid,
            label=label,
            description=desc,
            prior=prior,
            scenario_id=scenario_id,
        )
        for hid, label, desc in specs
    )


_REGISTRY: dict[str, tuple[Hypothesis, ...]] = {
    SCENARIO_TENNENT: _build(SCENARIO_TENNENT, _TENNENT_SPECS),
    SCENARIO_WHITSUN: _build(SCENARIO_WHITSUN, _WHITSUN_SPECS),
}


def _known_scenarios() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def get_hypotheses(scenario_id: str) -> tuple[Hypothesis, ...]:
    """Return the hypothesis set for ``scenario_id``.

    Raises:
        ValueError: if ``scenario_id`` is not registered.  The error lists
            the valid scenario identifiers.
    """
    try:
        return _REGISTRY[scenario_id]
    except KeyError:
        valid = ", ".join(_known_scenarios())
        raise ValueError(
            f"Unknown scenario_id {scenario_id!r}; valid scenarios: {valid}"
        ) from None


def get_hypothesis_ids(scenario_id: str) -> tuple[str, ...]:
    """Return just the hypothesis_id strings for ``scenario_id``."""
    return tuple(h.hypothesis_id for h in get_hypotheses(scenario_id))
