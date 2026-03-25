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


class TestPortfolioScenario:
    """Tests for the 36-hour, 24-entity portfolio scenario."""

    def setup_method(self):
        from custody.simulation import run_multi_target_simulation, PORTFOLIO_SCENARIO
        self.scenario = PORTFOLIO_SCENARIO
        self.records = run_multi_target_simulation(PORTFOLIO_SCENARIO)

    def test_entity_count_in_expected_range(self):
        ids = {r["target_id"] for r in self.records}
        total = len(self.scenario.scripted_vessels) + self.scenario.n_background
        assert len(ids) == total

    def test_scripted_entities_present(self):
        ids = {r["target_id"] for r in self.records}
        assert "BRAVO-1" in ids    # zone loiterer
        assert "ECHO-1"  in ids    # rendezvous pair
        assert "ECHO-2"  in ids    # rendezvous pair
        assert "PORT-1"  in ids    # manual custody / dark vessel

    def test_deterministic_for_fixed_seed(self):
        from custody.simulation import run_multi_target_simulation, PORTFOLIO_SCENARIO
        records2 = run_multi_target_simulation(PORTFOLIO_SCENARIO)
        b1_a = next(r for r in self.records if r["target_id"] == "BRAVO-1")
        b1_b = next(r for r in records2    if r["target_id"] == "BRAVO-1")
        assert b1_a["lat"]           == b1_b["lat"]
        assert b1_a["anomaly_score"] == b1_b["anomaly_score"]

    def test_timeline_covers_full_duration(self):
        from datetime import timedelta
        times = {r["time"] for r in self.records}
        expected_end = self.scenario.start_time + timedelta(hours=self.scenario.duration_hours)
        assert min(times) == self.scenario.start_time
        assert max(times) == expected_end

    def test_bravo1_loiters(self):
        bravo1 = [r for r in self.records if r["target_id"] == "BRAVO-1"]
        loiter_records = [r for r in bravo1 if r["behavior_mode"] == "loiter"]
        assert len(loiter_records) >= 5, f"Expected BRAVO-1 to loiter for >=5 steps, got {len(loiter_records)}"

    def test_echo_pair_present_and_symmetric(self):
        """ECHO-1 and ECHO-2 should both appear and start on opposite sides of the rendezvous anchor."""
        ids = {r["target_id"] for r in self.records}
        assert "ECHO-1" in ids and "ECHO-2" in ids
        echo1_first = min((r for r in self.records if r["target_id"] == "ECHO-1"), key=lambda r: r["time"])
        echo2_first = min((r for r in self.records if r["target_id"] == "ECHO-2"), key=lambda r: r["time"])
        # ECHO-1 starts west of the 1.0 anchor; ECHO-2 starts east
        assert echo1_first["lon"] < 1.0
        assert echo2_first["lon"] > 1.0

    def test_multiple_action_outcomes(self):
        """Portfolio scenario should produce more than one distinct action type."""
        actions = {r["action"] for r in self.records}
        assert len(actions) >= 2

    def test_metadata_fields_present(self):
        """New metadata fields should be present on all records."""
        for r in self.records[:10]:
            assert "profile" in r, "Missing 'profile' field"
            assert "is_scripted" in r, "Missing 'is_scripted' field"
            assert "scenario_tags" in r, "Missing 'scenario_tags' field"

    def test_scripted_entities_marked_is_scripted(self):
        scripted_ids = {"BRAVO-1", "ECHO-1", "ECHO-2"}  # is_anomalous=True actors
        for r in self.records:
            if r["target_id"] in scripted_ids:
                assert r["is_scripted"], f"{r['target_id']} should be marked is_scripted"
            elif r["target_id"].startswith("BG-"):
                assert not r["is_scripted"], f"{r['target_id']} should not be marked is_scripted"

    def test_multiple_behavior_modes_present(self):
        """Portfolio scenario should exhibit transit, loiter, and at least one other mode."""
        modes = {r["behavior_mode"] for r in self.records}
        assert "transit" in modes
        assert "loiter" in modes
        assert len(modes) >= 3

    def test_scripted_entities_higher_anomaly_than_background(self):
        scripted_scores = [r["anomaly_score"] for r in self.records if r["target_id"].startswith("BRAVO-")]
        bg_scores       = [r["anomaly_score"] for r in self.records if r["target_id"].startswith("BG-")]
        scripted_max = max(scripted_scores)
        bg_avg = sum(bg_scores) / len(bg_scores)
        assert scripted_max > bg_avg, (
            f"Expected scripted max anomaly ({scripted_max:.2f}) > bg avg ({bg_avg:.2f})"
        )

    def test_patrol_archetype_vessels_exist(self):
        """At least one background vessel should exhibit loiter behavior (patrol archetype)."""
        bg_loiter = [
            r for r in self.records
            if r["target_id"].startswith("BG-") and r["behavior_mode"] == "loiter"
        ]
        assert len(bg_loiter) > 0, "Expected patrol-archetype background vessels to show loiter mode"

    def test_background_vessel_ids_sequential(self):
        bg_ids = sorted(r["target_id"] for r in self.records if r["target_id"].startswith("BG-"))
        unique_bg = sorted(set(bg_ids))
        assert unique_bg[0] == "BG-001"
        assert len(unique_bg) == self.scenario.n_background
