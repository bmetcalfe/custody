"""
Smoke / regression entry-point for the two-vessel baseline scenario.

run_simulation() is an internal test fixture used by smoke tests, regression
suites, and fast unit-test baselines.  It is NOT a product-facing entry point;
the product runs PORTFOLIO_SCENARIO via run_multi_target_simulation().

For the full simulation loop, see custody/simulation/timeline.py.
For the scenario definition, see custody/simulation/scenarios.py (TWO_VESSEL_SMOKE).
"""
from custody.simulation import run_multi_target_simulation
from custody.simulation.scenarios import TWO_VESSEL_SMOKE


def run_simulation():
    """Run the two-vessel smoke scenario and return a flat list of records.

    Intended for smoke tests and regression baselines only.
    For the product scenario use run_multi_target_simulation(PORTFOLIO_SCENARIO).
    """
    return run_multi_target_simulation(TWO_VESSEL_SMOKE)


if __name__ == "__main__":
    records = run_simulation()
    for row in records[:12]:
        print(
            row["target_id"],
            row["time"].isoformat(),
            row["behavior_mode"],
            row["lat"],
            row["lon"],
            row["anomaly_score"],
            row["action"],
        )
