"""Tests for the multi-target simulation engine (Phase 1)."""
import pytest
from datetime import datetime, timezone

from custody.simulation import run_multi_target_simulation, DEFAULT_SCENARIO
from custody.simulation.generator import build_vessel_list, generate_background_vessels
from custody.simulation.scenarios import DEFAULT_SCENARIO, ScenarioConfig, VesselSpec, ProfilePhase
from custody.simulation.profiles import NORMAL_TRANSIT, LOITERING


class TestGenerator:
    def test_background_count(self):
        import random
        rng = random.Random(42)
        vessels = generate_background_vessels(27, (-3, 5, -3, 5), rng)
        assert len(vessels) == 27

    def test_vessel_ids_deterministic(self):
        import random
        rng1, rng2 = random.Random(42), random.Random(42)
        v1 = generate_background_vessels(10, (-3, 5, -3, 5), rng1)
        v2 = generate_background_vessels(10, (-3, 5, -3, 5), rng2)
        assert [v.vessel_id for v in v1] == [v.vessel_id for v in v2]

    def test_positions_in_region(self):
        import random
        rng = random.Random(99)
        vessels = generate_background_vessels(20, (0.0, 2.0, 0.0, 2.0), rng)
        for v in vessels:
            assert 0.0 <= v.start_lat <= 2.0
            assert 0.0 <= v.start_lon <= 2.0

    def test_build_vessel_list_includes_scripted(self):
        import random
        rng = random.Random(DEFAULT_SCENARIO.seed)
        specs = build_vessel_list(DEFAULT_SCENARIO, rng)
        ids = [s.vessel_id for s in specs]
        assert "SIGMA-1" in ids
        assert "SIGMA-2" in ids
        assert "SIGMA-3" in ids

    def test_build_vessel_list_total_count(self):
        import random
        rng = random.Random(DEFAULT_SCENARIO.seed)
        specs = build_vessel_list(DEFAULT_SCENARIO, rng)
        n_scripted = len(DEFAULT_SCENARIO.scripted_vessels)
        assert len(specs) == n_scripted + DEFAULT_SCENARIO.n_background


class TestTimeline:
    def setup_method(self):
        self.records = run_multi_target_simulation()

    def test_returns_list(self):
        assert isinstance(self.records, list)
        assert len(self.records) > 0

    def test_record_schema(self):
        """Output schema must match simulate.run_simulation() records."""
        required_keys = {
            "target_id", "time", "lat", "lon", "uncertainty_km",
            "custody_confidence", "anomaly_score", "speed_kmh", "heading_deg",
            "behavior_mode", "action", "sensor_id", "collection_result",
            "sensitive_zone", "loitering", "route_deviation",
            "behavior_state", "state_confidence", "sensor_access_count",
            "decision_trace",
        }
        for record in self.records[:5]:
            assert required_keys.issubset(set(record.keys()))

    def test_expected_vessel_count(self):
        ids = {r["target_id"] for r in self.records}
        n_scripted = len(DEFAULT_SCENARIO.scripted_vessels)
        assert len(ids) == n_scripted + DEFAULT_SCENARIO.n_background

    def test_deterministic_with_same_seed(self):
        records2 = run_multi_target_simulation()
        # First record of SIGMA-1 should be identical
        sigma1_a = next(r for r in self.records if r["target_id"] == "SIGMA-1")
        sigma1_b = next(r for r in records2    if r["target_id"] == "SIGMA-1")
        assert sigma1_a["lat"]          == sigma1_b["lat"]
        assert sigma1_a["anomaly_score"] == sigma1_b["anomaly_score"]

    def test_sigma1_anomaly_rises(self):
        """SIGMA-1 should show elevated anomaly in hours 14-20 (loitering in zone)."""
        sigma1 = sorted(
            [r for r in self.records if r["target_id"] == "SIGMA-1"],
            key=lambda r: r["time"],
        )
        early = [r["anomaly_score"] for r in sigma1[:6]]   # hours 0-5
        late  = [r["anomaly_score"] for r in sigma1[14:20]]  # hours 14-19
        assert max(late) > max(early), (
            f"SIGMA-1 should have higher anomaly in zone phase: "
            f"early_max={max(early):.2f}, late_max={max(late):.2f}"
        )

    def test_background_vessels_low_anomaly(self):
        """Background vessels should have generally low anomaly scores."""
        bg_scores = [
            r["anomaly_score"]
            for r in self.records
            if r["target_id"].startswith("BG-")
        ]
        avg = sum(bg_scores) / len(bg_scores)
        assert avg < 1.5, f"Background vessels have unexpectedly high avg anomaly: {avg:.2f}"

    def test_time_range(self):
        times = {r["time"] for r in self.records}
        start = DEFAULT_SCENARIO.start_time
        expected_end = start + __import__("datetime").timedelta(
            hours=DEFAULT_SCENARIO.duration_hours
        )
        assert min(times) == start
        assert max(times) == expected_end

    def test_scripted_vessels_anomalous_flag(self):
        import random
        rng = random.Random(DEFAULT_SCENARIO.seed)
        from custody.simulation.generator import build_vessel_list
        specs = build_vessel_list(DEFAULT_SCENARIO, rng)
        sigma_specs = [s for s in specs if s.vessel_id.startswith("SIGMA")]
        assert all(s.is_anomalous for s in sigma_specs)
        bg_specs = [s for s in specs if s.vessel_id.startswith("BG-")]
        assert all(not s.is_anomalous for s in bg_specs)
