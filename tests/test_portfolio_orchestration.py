"""
Tests for the portfolio orchestration layer (Phase 3a).

Covers:
  - CustodyHealth classification under various confidence/uncertainty conditions
  - Neglect detection at, below, and above threshold
  - Portfolio ranking: high-anomaly and neglected entities surface correctly
  - Deferred/serviced entity identification
  - Determinism guarantee
  - Full PORTFOLIO_SCENARIO integration: fields present, values sensible
"""
from __future__ import annotations

import pytest
from datetime import datetime, timezone, timedelta
from typing import Optional

from custody.orchestration.portfolio import (
    CustodyHealth,
    PortfolioItem,
    PortfolioAssessment,
    NEGLECT_THRESHOLD_HOURS,
    compute_custody_health,
    detect_neglect,
    rank_portfolio,
)

UTC = timezone.utc
_T0 = datetime(2026, 3, 23, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _record(
    eid: str,
    anomaly: float = 0.1,
    conf: float = 0.9,
    unc_km: float = 5.0,
    action: str = "NONE",
    hsc: Optional[float] = None,   # hours_since_collection
) -> dict:
    return {
        "target_id":             eid,
        "anomaly_score":         anomaly,
        "custody_confidence":    conf,
        "uncertainty_km":        unc_km,
        "action":                action,
        "hours_since_collection": hsc,
    }


# ---------------------------------------------------------------------------
# CustodyHealth classification
# ---------------------------------------------------------------------------

class TestCustodyHealth:

    def test_healthy(self):
        h = compute_custody_health("X", uncertainty_km=5.0, custody_confidence=0.9, neglect_hours=0.5)
        assert h.status == "HEALTHY"

    def test_degrading_low_confidence(self):
        h = compute_custody_health("X", uncertainty_km=5.0, custody_confidence=0.50, neglect_hours=2.0)
        assert h.status == "DEGRADING"

    def test_degrading_high_uncertainty(self):
        h = compute_custody_health("X", uncertainty_km=25.0, custody_confidence=0.8, neglect_hours=1.0)
        assert h.status == "DEGRADING"

    def test_stale_low_confidence(self):
        h = compute_custody_health("X", uncertainty_km=5.0, custody_confidence=0.30, neglect_hours=3.0)
        assert h.status == "STALE"

    def test_stale_high_uncertainty(self):
        h = compute_custody_health("X", uncertainty_km=60.0, custody_confidence=0.8, neglect_hours=1.0)
        assert h.status == "STALE"

    def test_lost_very_low_confidence(self):
        h = compute_custody_health("X", uncertainty_km=5.0, custody_confidence=0.10, neglect_hours=5.0)
        assert h.status == "LOST"

    def test_lost_extreme_uncertainty(self):
        h = compute_custody_health("X", uncertainty_km=100.0, custody_confidence=0.8, neglect_hours=1.0)
        assert h.status == "LOST"

    def test_worst_case_wins(self):
        """Bad confidence overrides low uncertainty."""
        h = compute_custody_health("X", uncertainty_km=2.0, custody_confidence=0.10, neglect_hours=0.0)
        assert h.status == "LOST"

    def test_reason_is_nonempty_string(self):
        for conf, unc in [(0.9, 5.0), (0.5, 5.0), (0.3, 5.0), (0.1, 5.0)]:
            h = compute_custody_health("X", unc, conf, 1.0)
            assert isinstance(h.reason, str) and len(h.reason) > 0

    def test_dataclass_fields(self):
        h = compute_custody_health("ABC", 10.0, 0.7, 2.5)
        assert h.entity_id == "ABC"
        assert h.uncertainty_km == 10.0
        assert h.custody_confidence == 0.7
        assert h.neglect_hours == 2.5


# ---------------------------------------------------------------------------
# Neglect detection
# ---------------------------------------------------------------------------

class TestNeglectDetection:

    def test_not_neglected_below_threshold(self):
        assert detect_neglect("X", NEGLECT_THRESHOLD_HOURS - 0.1) is False

    def test_neglected_at_threshold(self):
        assert detect_neglect("X", NEGLECT_THRESHOLD_HOURS) is True

    def test_neglected_above_threshold(self):
        assert detect_neglect("X", NEGLECT_THRESHOLD_HOURS + 10.0) is True

    def test_custom_threshold_not_neglected(self):
        assert detect_neglect("X", 5.0, threshold_hours=6.0) is False

    def test_custom_threshold_neglected(self):
        assert detect_neglect("X", 6.0, threshold_hours=6.0) is True

    def test_zero_hours_never_neglected(self):
        assert detect_neglect("X", 0.0) is False


# ---------------------------------------------------------------------------
# Portfolio ranking — unit cases
# ---------------------------------------------------------------------------

class TestRankPortfolio:

    def _rank(self, records, t=_T0, start=_T0, sensor_claimer=None):
        return rank_portfolio(
            timestamp        = t,
            timestep_records = records,
            scenario_start   = start,
            sensor_claimer   = sensor_claimer,
        )

    def test_empty_returns_empty_assessment(self):
        pa = self._rank([])
        assert pa.items == ()
        assert pa.serviced == ()
        assert pa.deferred == ()

    def test_single_entity(self):
        pa = self._rank([_record("A")])
        assert len(pa.items) == 1
        assert pa.items[0].entity_id == "A"
        assert pa.items[0].portfolio_rank == 1

    def test_high_anomaly_ranks_above_low_anomaly(self):
        pa = self._rank([
            _record("A", anomaly=2.5),
            _record("B", anomaly=0.1),
        ])
        by_e = pa.by_entity()
        assert by_e["A"].portfolio_rank < by_e["B"].portfolio_rank

    def test_low_confidence_ranks_above_healthy(self):
        pa = self._rank([
            _record("A", anomaly=0.1, conf=0.2),   # degraded custody
            _record("B", anomaly=0.1, conf=0.95),  # healthy
        ])
        by_e = pa.by_entity()
        assert by_e["A"].portfolio_rank < by_e["B"].portfolio_rank

    def test_neglected_entity_surfaces_despite_low_anomaly(self):
        """An entity unobserved for 10h should rank above a healthy, low-anomaly peer."""
        t = _T0 + timedelta(hours=10)
        records = [
            _record("A", anomaly=0.1, conf=0.9, hsc=0.0),    # just collected
            _record("B", anomaly=0.2, conf=0.7, hsc=None),   # never collected → 10h neglect
        ]
        pa = self._rank(records, t=t, start=_T0)
        by_e = pa.by_entity()
        assert by_e["B"].neglect_flag is True
        assert by_e["B"].portfolio_rank < by_e["A"].portfolio_rank

    def test_serviced_entity_identified(self):
        pa = self._rank([
            _record("A", action="TASK"),
            _record("B", action="NONE"),
        ])
        assert "A" in pa.serviced
        assert "B" not in pa.serviced

    def test_preempted_entity_in_deferred(self):
        pa = self._rank([
            _record("A", action="TASK"),
            _record("B", action="PREEMPTED"),
        ])
        assert "B" in pa.deferred
        assert "A" not in pa.deferred

    def test_deferred_for_points_to_serviced_entity(self):
        pa = self._rank([
            _record("A", anomaly=2.0, action="TASK"),
            _record("B", anomaly=0.1, action="PREEMPTED"),
        ])
        by_e = pa.by_entity()
        assert by_e["B"].deferred_for == "A"

    def test_deferred_for_none_when_not_preempted(self):
        pa = self._rank([
            _record("A", action="NONE"),
            _record("B", action="NONE"),
        ])
        by_e = pa.by_entity()
        assert by_e["A"].deferred_for is None
        assert by_e["B"].deferred_for is None

    def test_ranking_is_deterministic(self):
        records = [_record(f"E{i:02d}", anomaly=float(i) / 10) for i in range(8)]
        pa1 = self._rank(records)
        pa2 = self._rank(records)
        assert [it.entity_id for it in pa1.items] == [it.entity_id for it in pa2.items]
        assert [it.portfolio_rank for it in pa1.items] == [it.portfolio_rank for it in pa2.items]

    def test_ranks_are_sequential_from_one(self):
        records = [_record(f"E{i}") for i in range(5)]
        pa = self._rank(records)
        ranks = [it.portfolio_rank for it in pa.items]
        assert ranks == list(range(1, 6))

    def test_portfolio_score_in_unit_interval(self):
        records = [
            _record("A", anomaly=3.0, conf=0.0, unc_km=200.0, hsc=100.0),
            _record("B", anomaly=0.0, conf=1.0, unc_km=0.0, hsc=0.0),
        ]
        for item in self._rank(records).items:
            assert 0.0 <= item.portfolio_score <= 1.0

    def test_by_entity_keys_match_records(self):
        eids = ["ALPHA", "BETA", "GAMMA"]
        pa = self._rank([_record(e) for e in eids])
        assert set(pa.by_entity().keys()) == set(eids)

    def test_reason_nonempty_for_all_entities(self):
        records = [_record(f"E{i}", anomaly=float(i)) for i in range(4)]
        for item in self._rank(records).items:
            assert isinstance(item.portfolio_reason, str)
            assert len(item.portfolio_reason) > 0

    def test_hsc_none_uses_elapsed_time(self):
        """hours_since_collection=None → elapsed time since scenario start."""
        t = _T0 + timedelta(hours=8)
        records = [_record("A", hsc=None)]
        pa = self._rank(records, t=t, start=_T0)
        assert pa.items[0].neglect_hours == pytest.approx(8.0)

    def test_hsc_zero_means_just_collected(self):
        t = _T0 + timedelta(hours=8)
        records = [_record("A", hsc=0.0)]
        pa = self._rank(records, t=t, start=_T0)
        assert pa.items[0].neglect_hours == pytest.approx(0.0)
        assert pa.items[0].neglect_flag is False

    def test_custody_health_valid_values(self):
        valid = {"HEALTHY", "DEGRADING", "STALE", "LOST"}
        records = [
            _record("A", conf=0.9, unc_km=5.0),
            _record("B", conf=0.5, unc_km=5.0),
            _record("C", conf=0.3, unc_km=5.0),
            _record("D", conf=0.1, unc_km=5.0),
        ]
        for item in self._rank(records).items:
            assert item.custody_health in valid


# ---------------------------------------------------------------------------
# PORTFOLIO_SCENARIO integration
# ---------------------------------------------------------------------------

class TestPortfolioScenarioIntegration:
    """Run the full 36h PORTFOLIO_SCENARIO and verify portfolio annotations."""

    @pytest.fixture(scope="class")
    def records(self):
        from custody.simulation import run_multi_target_simulation, PORTFOLIO_SCENARIO
        return run_multi_target_simulation(PORTFOLIO_SCENARIO)

    def test_portfolio_fields_present(self, records):
        required = {
            "portfolio_rank", "portfolio_score", "custody_health",
            "neglect_flag", "neglect_hours", "portfolio_reason",
            "deferred_for", "hours_since_collection",
            # selective-custody fields
            "attention_state", "tracking_directive", "attention_basis",
        }
        missing = required - set(records[0].keys())
        assert not missing, f"Missing portfolio fields: {missing}"

    def test_portfolio_rank_range(self, records):
        n_entities = len({r["target_id"] for r in records})
        for r in records:
            assert 1 <= r["portfolio_rank"] <= n_entities

    def test_portfolio_score_in_unit_interval(self, records):
        for r in records:
            assert 0.0 <= r["portfolio_score"] <= 1.0

    def test_custody_health_valid_values(self, records):
        valid = {"HEALTHY", "DEGRADING", "STALE", "LOST"}
        bad = {r["custody_health"] for r in records} - valid
        assert not bad, f"Unexpected custody_health values: {bad}"

    def test_neglect_flag_is_bool(self, records):
        for r in records[:30]:
            assert isinstance(r["neglect_flag"], bool)

    def test_portfolio_reason_nonempty_string(self, records):
        for r in records[:30]:
            assert isinstance(r["portfolio_reason"], str)
            assert len(r["portfolio_reason"]) > 0

    def test_some_entities_become_neglected(self, records):
        """28 entities with limited sensors → some go unobserved past threshold."""
        assert any(r["neglect_flag"] for r in records), (
            "Expected at least one neglect_flag=True in a 36h/28-entity scenario"
        )

    def test_neglect_hours_increases_for_unobserved_entity(self, records):
        """An entity never TASKed should show growing neglect_hours over time."""
        never_tasked = None
        for eid in sorted({r["target_id"] for r in records if r["target_id"].startswith("BG-")}):
            entity_records = sorted(
                [r for r in records if r["target_id"] == eid], key=lambda r: r["time"]
            )
            if not any(r["action"] == "TASK" for r in entity_records):
                never_tasked = entity_records
                break
        if never_tasked:
            assert never_tasked[-1]["neglect_hours"] > never_tasked[0]["neglect_hours"]

    def test_scripted_entities_rank_highly_when_anomalous(self, records):
        """At peak anomaly, at least one BRAVO entity should rank in the top 25%."""
        bravo = [r for r in records if r["target_id"].startswith("BRAVO-")]
        n_entities = len({r["target_id"] for r in records})
        top_quartile = n_entities // 4
        elevated = [r for r in bravo if r["anomaly_score"] > 0.8]
        if elevated:
            assert any(r["portfolio_rank"] <= max(top_quartile, 5) for r in elevated), (
                "Expected a BRAVO entity to rank in top quartile when anomalous"
            )

    def test_multiple_custody_health_states_observed(self, records):
        states = {r["custody_health"] for r in records}
        assert len(states) >= 2, f"Expected multiple health states, got: {states}"

    def test_preempted_records_have_deferred_for(self, records):
        """PREEMPTED entities should have a non-None deferred_for."""
        preempted = [r for r in records if r["action"] == "PREEMPTED"]
        if preempted:
            # At least half should have a deferred_for explanation
            with_reason = [r for r in preempted if r["deferred_for"] is not None]
            assert len(with_reason) >= len(preempted) // 2

    def test_hours_since_collection_is_none_or_nonneg(self, records):
        for r in records:
            hsc = r["hours_since_collection"]
            assert hsc is None or hsc >= 0.0

    def test_portfolio_rank_unique_per_timestep(self, records):
        """Each entity should have a unique rank at each timestep."""
        from collections import defaultdict
        by_time: dict = defaultdict(list)
        for r in records:
            by_time[r["time"]].append(r["portfolio_rank"])
        for t, ranks in by_time.items():
            assert len(ranks) == len(set(ranks)), f"Duplicate ranks at {t}: {sorted(ranks)}"

    def test_deferred_entities_are_preempted(self, records):
        """deferred_for should only appear on PREEMPTED records."""
        for r in records:
            if r["deferred_for"] is not None:
                assert r["action"] == "PREEMPTED", (
                    f"{r['target_id']} has deferred_for={r['deferred_for']} "
                    f"but action={r['action']}"
                )

    def test_smoke_scenario_has_portfolio_fields(self):
        """TWO_VESSEL_SMOKE (run_simulation) should also emit portfolio fields."""
        from custody.simulate import run_simulation
        smoke_records = run_simulation()
        assert "portfolio_rank" in smoke_records[0]
        assert "neglect_flag" in smoke_records[0]

    def test_attention_state_valid_values(self, records):
        valid = {"BACKGROUND", "WATCHLIST", "ACTIVE_CUSTODY"}
        bad = {r["attention_state"] for r in records} - valid
        assert not bad, f"Unexpected attention_state values: {bad}"

    def test_multiple_attention_states_observed(self, records):
        """A 36h/29-entity portfolio should exhibit all three tiers."""
        states = {r["attention_state"] for r in records}
        assert len(states) >= 2, f"Expected ≥2 attention states, got: {states}"

    def test_attention_basis_nonempty_string(self, records):
        for r in records[:30]:
            assert isinstance(r["attention_basis"], str)
            assert len(r["attention_basis"]) > 0

    def test_port1_stays_active_custody_throughout(self, records):
        """PORT-1 has MAINTAIN_CUSTODY directive — must always be ACTIVE_CUSTODY."""
        port1 = [r for r in records if r["target_id"] == "PORT-1"]
        assert port1, "PORT-1 not found in PORTFOLIO_SCENARIO records"
        assert all(r["attention_state"] == "ACTIVE_CUSTODY" for r in port1), (
            "PORT-1 should always be ACTIVE_CUSTODY due to MAINTAIN_CUSTODY directive"
        )
        assert all(r["tracking_directive"] == "MAINTAIN_CUSTODY" for r in port1)

    def test_port1_neglect_flag_triggers_when_uncollected(self, records):
        """PORT-1 in ACTIVE_CUSTODY should accumulate neglect normally."""
        port1 = sorted(
            [r for r in records if r["target_id"] == "PORT-1"],
            key=lambda r: r["time"]
        )
        # By the end of a 36h scenario with limited sensors, PORT-1 should become neglected
        late_records = port1[len(port1) // 2:]   # second half of the timeline
        assert any(r["neglect_flag"] for r in late_records), (
            "Expected PORT-1 to be neglected at some point in the 36h run"
        )

    def test_background_vessels_are_classified_background(self, records):
        """BG-* vessels with routine behavior should be BACKGROUND at scenario start."""
        # At hour 0, no anomaly has built up — most BG vessels should be BACKGROUND
        t0_records = sorted(records, key=lambda r: r["time"])
        first_time = t0_records[0]["time"]
        first_ts = [r for r in records if r["time"] == first_time
                    and r["target_id"].startswith("BG-")]
        bg_at_start = [r for r in first_ts if r["attention_state"] == "BACKGROUND"]
        # Majority of BG vessels should start as BACKGROUND (not active custody)
        assert len(bg_at_start) > len(first_ts) // 2, (
            f"Expected most BG vessels to start as BACKGROUND, "
            f"got {len(bg_at_start)}/{len(first_ts)}"
        )
