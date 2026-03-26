"""Engine-to-UI adapter layer.

This module is the single boundary between the custody simulation engine
and the UI framework (currently Streamlit, migrating to Dash).  UI code
should call functions here rather than importing from custody.* directly
for simulation execution and record slicing.

Pure helper modules in src/app/ (portfolio_overview, overview_filters,
overview_events, compound_panels, orbital_passes_panel, whatif_panel,
ground_track) remain importable from UI code — they are framework-agnostic
data transformers, not engine internals.

Design rules:
  - Record lists (list[dict]) are the primary abstraction.
  - DataFrame conversions exist as convenience for table rendering.
  - Records carry live objects (FusionAssessment, Decision, DecisionTrace);
    serialization is not this module's concern.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from custody.simulation import run_multi_target_simulation
from custody.simulation.scenarios import (
    MULTI_DAY_SCENARIO,
    PORTFOLIO_SCENARIO,
    TWO_VESSEL_SMOKE,
)

if TYPE_CHECKING:
    from custody.simulation.scenarios import ScenarioConfig

# ---------------------------------------------------------------------------
# Scenario catalog
# ---------------------------------------------------------------------------

_SCENARIOS: dict[str, "ScenarioConfig"] = {
    "multi_day_72h":   MULTI_DAY_SCENARIO,
    "portfolio_36h":   PORTFOLIO_SCENARIO,
    "two_vessel_smoke": TWO_VESSEL_SMOKE,
}


def scenario_names() -> list[str]:
    """Return available scenario names in display order."""
    return list(_SCENARIOS.keys())


def load_scenario(name: str = "multi_day_72h") -> list[dict]:
    """Run a named scenario and return the full record list.

    Args:
        name: Scenario name.  Must be one of :func:`scenario_names`.

    Returns:
        Flat list of record dicts, sorted by (time, target_id).
        Each record carries ``fusion_assessment``, ``mission_decision``,
        and ``decision_trace`` as live objects (not serialized).

    Raises:
        KeyError: If *name* is not a known scenario.
    """
    config = _SCENARIOS[name]
    return run_multi_target_simulation(config)


# ---------------------------------------------------------------------------
# Record-level accessors
# ---------------------------------------------------------------------------

def entity_ids(records: list[dict]) -> list[str]:
    """Return sorted unique entity IDs from a record list."""
    return sorted({r["target_id"] for r in records})


def timestep_count(records: list[dict]) -> int:
    """Return the number of distinct timesteps in the record list."""
    return len({r["time"] for r in records})


def _timestep_index(records: list[dict]) -> list:
    """Return a sorted list of unique timestep datetime values.

    Internal helper; results are cached per scenario in practice because
    the record list is immutable after load.
    """
    return sorted({r["time"] for r in records})


def records_at_timestep(records: list[dict], step: int) -> list[dict]:
    """Return all records at the given timestep index (0-based).

    Args:
        records: Full scenario record list.
        step:    Timestep index (0 = first timestep).

    Returns:
        List of record dicts for all entities at that timestep.
        Empty list if *step* is out of range.
    """
    times = _timestep_index(records)
    if step < 0 or step >= len(times):
        return []
    target_time = times[step]
    return [r for r in records if r["time"] == target_time]


def entity_timeline(records: list[dict], entity_id: str) -> list[dict]:
    """Return all records for one entity, ordered by time."""
    return [r for r in records if r["target_id"] == entity_id]


def entity_timeline_up_to(
    records: list[dict], entity_id: str, step: int,
) -> list[dict]:
    """Return entity records from timestep 0 through *step* (inclusive).

    Used for prefix windows in fusion/decision display and compound
    evaluation.
    """
    times = _timestep_index(records)
    if step < 0 or step >= len(times):
        return []
    cutoff = times[step]
    return [
        r for r in records
        if r["target_id"] == entity_id and r["time"] <= cutoff
    ]


# ---------------------------------------------------------------------------
# DataFrame convenience wrappers
# ---------------------------------------------------------------------------

def timestep_as_dataframe(records: list[dict], step: int) -> pd.DataFrame:
    """Convenience: :func:`records_at_timestep` → DataFrame.

    Provided for table rendering; the record list is the primary
    abstraction.
    """
    return pd.DataFrame(records_at_timestep(records, step))


def entity_timeline_as_dataframe(
    records: list[dict], entity_id: str,
) -> pd.DataFrame:
    """Convenience: :func:`entity_timeline` → DataFrame.

    Provided for table rendering; the record list is the primary
    abstraction.
    """
    return pd.DataFrame(entity_timeline(records, entity_id))
