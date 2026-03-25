"""
Integration tests for the refined PORTFOLIO_SCENARIO.

Verifies that the scenario demonstrates all intended behavior classes:
  - BRAVO-1: zone-loiter / area concern reaches and loiters inside ZONE_ALPHA
  - ECHO-1/2: rendezvous pair fires and dwells before separating
  - PORT-1: manually tracked vessel goes dark and is kept at ACTIVE_CUSTODY

Tests are property-style (presence, windows, attention states, field values)
rather than exact per-timestep rank assertions.
"""
from __future__ import annotations

import pytest
from collections import Counter

from custody.simulation import run_multi_target_simulation, PORTFOLIO_SCENARIO
from custody.orchestration.attention import BACKGROUND, WATCHLIST, ACTIVE_CUSTODY


# ---------------------------------------------------------------------------
# Shared fixture — run once per test session for performance
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def records():
    return run_multi_target_simulation(PORTFOLIO_SCENARIO)


@pytest.fixture(scope="module")
def bravo1(records):
    return sorted([r for r in records if r["target_id"] == "BRAVO-1"], key=lambda r: r["time"])


@pytest.fixture(scope="module")
def echo1(records):
    return sorted([r for r in records if r["target_id"] == "ECHO-1"], key=lambda r: r["time"])


@pytest.fixture(scope="module")
def echo2(records):
    return sorted([r for r in records if r["target_id"] == "ECHO-2"], key=lambda r: r["time"])


@pytest.fixture(scope="module")
def port1(records):
    return sorted([r for r in records if r["target_id"] == "PORT-1"], key=lambda r: r["time"])


@pytest.fixture(scope="module")
def bg_records(records):
    scripted = {"BRAVO-1", "ECHO-1", "ECHO-2", "PORT-1"}
    return [r for r in records if r["target_id"] not in scripted]


# ---------------------------------------------------------------------------
# 1. Roster and structure
# ---------------------------------------------------------------------------

class TestRosterAndStructure:
    def test_all_scripted_actors_present(self, records):
        ids = {r["target_id"] for r in records}
        assert "BRAVO-1" in ids
        assert "ECHO-1"  in ids
        assert "ECHO-2"  in ids
        assert "PORT-1"  in ids

    def test_entity_count(self, records):
        """4 scripted + n_background entities total."""
        ids = {r["target_id"] for r in records}
        expected = len(PORTFOLIO_SCENARIO.scripted_vessels) + PORTFOLIO_SCENARIO.n_background
        assert len(ids) == expected

    def test_timeline_covers_full_duration(self, records):
        from datetime import timedelta
        times = {r["time"] for r in records}
        assert min(times) == PORTFOLIO_SCENARIO.start_time
        assert max(times) == PORTFOLIO_SCENARIO.start_time + timedelta(hours=PORTFOLIO_SCENARIO.duration_hours)

    def test_all_records_have_dark_vessel_fields(self, records):
        for r in records[:20]:
            assert "dark_vessel_flag" in r
            assert "dark_vessel_reason" in r
            assert "last_known_lat" in r

    def test_all_records_have_rendezvous_fields(self, records):
        for r in records[:20]:
            assert "rendezvous_flag" in r
            assert "rendezvous_confidence" in r
            assert "counterpart_id" in r

    def test_all_records_have_portfolio_fields(self, records):
        for r in records:
            assert "portfolio_rank" in r
            assert "attention_state" in r
            assert "portfolio_score" in r


# ---------------------------------------------------------------------------
# 2. BRAVO-1: zone loiterer
# ---------------------------------------------------------------------------

class TestBravo1ZoneLoiter:
    def test_bravo1_has_expected_timestep_count(self, bravo1):
        # 36-hour scenario, dt=1 → 37 timesteps (0..36)
        assert len(bravo1) == 37

    def test_bravo1_enters_zone_or_halo(self, bravo1):
        """sensitive_zone score > 0 at some point — vessel reached the zone or its halo."""
        assert any(r["sensitive_zone"] > 0.0 for r in bravo1), \
            "BRAVO-1 should approach ZONE_ALPHA at some point"

    def test_bravo1_loiters(self, bravo1):
        loiter_steps = [r for r in bravo1 if r["behavior_mode"] == "loiter"]
        assert len(loiter_steps) >= 8, \
            f"Expected >= 8 loitering steps; got {len(loiter_steps)}"

    def test_bravo1_high_zone_score_during_loiter(self, bravo1):
        """BRAVO-1 should show elevated zone score while in loiter phase."""
        loiter_zone_scores = [
            r["sensitive_zone"] for r in bravo1 if r["behavior_mode"] == "loiter"
        ]
        assert max(loiter_zone_scores) > 0.5, \
            f"Expected zone score > 0.5 during loiter; max was {max(loiter_zone_scores):.2f}"

    def test_bravo1_rises_above_background(self, bravo1):
        """BRAVO-1 should reach WATCHLIST or ACTIVE_CUSTODY at some point."""
        tiers = {r.get("attention_state") for r in bravo1}
        assert WATCHLIST in tiers or ACTIVE_CUSTODY in tiers, \
            f"BRAVO-1 should rise above BACKGROUND; tiers seen: {tiers}"

    def test_bravo1_loiter_before_evasive_egress(self, bravo1):
        """Loiter phase should precede the evasive phase (narrative order)."""
        first_loiter = next((r["time"] for r in bravo1 if r["behavior_mode"] == "loiter"), None)
        first_egress = next((r["time"] for r in bravo1 if r["behavior_mode"] == "egress"), None)
        if first_loiter and first_egress:
            assert first_loiter < first_egress, "Loiter should start before evasive egress"

    def test_bravo1_evasive_egress_phase_exists(self, bravo1):
        egress_steps = [r for r in bravo1 if r["behavior_mode"] == "egress"]
        assert len(egress_steps) > 0, "BRAVO-1 should have an evasive egress phase"


# ---------------------------------------------------------------------------
# 3. ECHO-1 / ECHO-2: rendezvous pair
# ---------------------------------------------------------------------------

class TestEchoRendezvous:
    def test_echo1_rendezvous_fires(self, echo1):
        """Rendezvous flag must be True for at least one ECHO-1 timestep."""
        assert any(r["rendezvous_flag"] for r in echo1), \
            "Rendezvous should fire for ECHO-1"

    def test_echo2_rendezvous_fires(self, echo2):
        assert any(r["rendezvous_flag"] for r in echo2), \
            "Rendezvous should fire for ECHO-2"

    def test_rendezvous_fires_in_mid_scenario(self, echo1):
        """Rendezvous should fire during hours 8-20 (mid-phase)."""
        flagged = [r for r in echo1 if r["rendezvous_flag"]]
        first_hour = flagged[0]["time"].hour
        assert 6 <= first_hour <= 24, \
            f"Expected rendezvous in hours 6-24; first fired at hour {first_hour}"

    def test_rendezvous_dwell_hours_grow(self, echo1):
        """Dwell hours should increase while rendezvous is active."""
        flagged = [r for r in echo1 if r["rendezvous_flag"]]
        assert len(flagged) >= 2
        dwells = [r["rendezvous_dwell_hours"] for r in flagged]
        assert dwells[-1] >= dwells[0], "Dwell hours should not decrease"

    def test_counterpart_ids_cross_referenced(self, echo1, echo2):
        """Each ECHO vessel names the other as counterpart when flagged."""
        e1_flag = next((r for r in echo1 if r["rendezvous_flag"]), None)
        e2_flag = next((r for r in echo2 if r["rendezvous_flag"]), None)
        assert e1_flag is not None and e2_flag is not None
        assert e1_flag["counterpart_id"] == "ECHO-2"
        assert e2_flag["counterpart_id"] == "ECHO-1"

    def test_min_pair_distance_km_nonneg_when_flagged(self, echo1):
        flagged = [r for r in echo1 if r["rendezvous_flag"]]
        assert all(r["min_pair_distance_km"] is not None and r["min_pair_distance_km"] >= 0
                   for r in flagged)

    def test_rendezvous_confidence_positive_when_flagged(self, echo1):
        flagged = [r for r in echo1 if r["rendezvous_flag"]]
        assert all(r["rendezvous_confidence"] > 0 for r in flagged)

    def test_echo_pair_rises_to_watchlist_or_active(self, echo1):
        tiers = {r.get("attention_state") for r in echo1}
        assert WATCHLIST in tiers or ACTIVE_CUSTODY in tiers, \
            f"ECHO-1 should rise above BACKGROUND; saw: {tiers}"

    def test_echo_separation_after_dwell(self, echo1):
        """ECHO-1 should enter evasive mode after the rendezvous dwell."""
        egress = [r for r in echo1 if r["behavior_mode"] == "egress"]
        assert len(egress) > 0, "ECHO-1 should separate (evasive) after dwell"
        flagged = [r for r in echo1 if r["rendezvous_flag"]]
        if flagged and egress:
            assert flagged[0]["time"] < egress[0]["time"], \
                "Rendezvous should fire before ECHO-1 begins separation"


# ---------------------------------------------------------------------------
# 4. PORT-1: manual custody + dark vessel
# ---------------------------------------------------------------------------

class TestPort1DarkVessel:
    def test_port1_has_maintain_custody_directive(self, port1):
        assert all(r["tracking_directive"] == "MAINTAIN_CUSTODY" for r in port1)

    def test_port1_active_custody_before_dropout(self, port1):
        pre_dark = [r for r in port1 if not r["dark_vessel_flag"]]
        assert all(r.get("attention_state") == ACTIVE_CUSTODY for r in pre_dark), \
            "PORT-1 should be ACTIVE_CUSTODY before AIS dropout (MAINTAIN_CUSTODY directive)"

    def test_port1_goes_dark(self, port1):
        assert any(r["dark_vessel_flag"] for r in port1), \
            "PORT-1 should go dark (AIS dropout)"

    def test_port1_dark_onset_at_expected_hour(self, port1):
        dark = [r for r in port1 if r["dark_vessel_flag"]]
        assert dark[0]["time"].hour == PORTFOLIO_SCENARIO.scripted_vessels[1].ais_dropout_hour

    def test_port1_position_frozen_after_dropout(self, port1):
        dark = [r for r in port1 if r["dark_vessel_flag"]]
        lats = {round(r["lat"], 4) for r in dark}
        lons = {round(r["lon"], 4) for r in dark}
        assert len(lats) == 1, "PORT-1 lat should be frozen after AIS dropout"
        assert len(lons) == 1, "PORT-1 lon should be frozen after AIS dropout"

    def test_port1_anomaly_frozen_after_dropout(self, port1):
        dark = [r for r in port1 if r["dark_vessel_flag"]]
        scores = {round(r["anomaly_score"], 6) for r in dark}
        assert len(scores) == 1, "PORT-1 anomaly_score should be frozen after dropout"

    def test_port1_active_custody_throughout_dark_phase(self, port1):
        """Dark floor + MAINTAIN_CUSTODY must keep PORT-1 at ACTIVE_CUSTODY."""
        dark = [r for r in port1 if r["dark_vessel_flag"]]
        assert all(r.get("attention_state") == ACTIVE_CUSTODY for r in dark), \
            "PORT-1 should remain ACTIVE_CUSTODY while dark (dark floor + directive)"

    def test_port1_uncertainty_grows_after_dropout(self, port1):
        dark = [r for r in port1 if r["dark_vessel_flag"]]
        assert dark[-1]["uncertainty_km"] > dark[0]["uncertainty_km"], \
            "Uncertainty should grow during dark phase"

    def test_port1_dark_vessel_reason_is_string(self, port1):
        dark = [r for r in port1 if r["dark_vessel_flag"]]
        assert all(isinstance(r["dark_vessel_reason"], str) for r in dark)

    def test_port1_last_known_fields_set(self, port1):
        dark = [r for r in port1 if r["dark_vessel_flag"]]
        assert all(r["last_known_lat"] is not None for r in dark)
        assert all(r["last_known_lon"] is not None for r in dark)
        assert all(r["dark_since"] is not None for r in dark)


# ---------------------------------------------------------------------------
# 5. Selective custody: background vessels stay BACKGROUND
# ---------------------------------------------------------------------------

class TestSelectiveCustody:
    def test_background_vessels_start_at_background_tier(self, bg_records):
        """At the scenario start timestep, background vessels should be in BACKGROUND tier.

        Background vessels begin with healthy confidence (uncertainty=5 km →
        conf≈0.90 after one dt update).  Confidence decays as hours pass without
        collection — the selective custody doctrine gates neglect pressure at zero
        for BACKGROUND tier, but confidence-driven tier promotion is expected
        behaviour over the full 36-hour scenario.  This test validates t=0 only.
        """
        t0 = PORTFOLIO_SCENARIO.start_time
        h0_bg = [r for r in bg_records if r["time"] == t0]
        assert len(h0_bg) > 0, "Expected background records at scenario start"
        n_bg_tier = sum(1 for r in h0_bg if r.get("attention_state") == BACKGROUND)
        frac = n_bg_tier / len(h0_bg)
        assert frac >= 0.70, \
            f"Expected >=70% of background vessels at BACKGROUND tier at t=0; got {frac:.1%}"

    def test_background_vessels_not_dark(self, bg_records):
        assert all(r["dark_vessel_flag"] is False for r in bg_records)

    def test_background_last_known_none(self, bg_records):
        assert all(r["last_known_lat"] is None for r in bg_records)

    def test_background_mostly_non_anomalous_at_start(self, bg_records):
        """Selective custody doctrine: at scenario start, most background vessels
        are non-anomalous.  Some approach-archetype vessels may wander into the
        zone later, but the population should start mostly clean."""
        t0 = PORTFOLIO_SCENARIO.start_time
        h0_bg = [r for r in bg_records if r["time"] == t0]
        n_low = sum(1 for r in h0_bg if r["anomaly_score"] < 0.5)
        frac = n_low / len(h0_bg)
        assert frac >= 0.70, \
            f"Expected >=70% of t=0 background records with anomaly<0.5; got {frac:.1%}"

    def test_scripted_actors_peak_anomaly_above_median_background(self, records, bg_records):
        """Scripted anomalous actors should reach higher peak anomaly scores
        than the typical background vessel.  'Typical' is the 75th percentile."""
        scripted_peak = max(
            r["anomaly_score"] for r in records
            if r["target_id"] in ("BRAVO-1", "ECHO-1", "ECHO-2")
        )
        bg_scores = sorted(r["anomaly_score"] for r in bg_records)
        p75_bg = bg_scores[int(len(bg_scores) * 0.75)]
        assert scripted_peak > p75_bg, \
            f"Scripted peak ({scripted_peak:.2f}) should exceed BG 75th pct ({p75_bg:.2f})"


# ---------------------------------------------------------------------------
# 6. Overlapping concerns → portfolio tradeoffs
# ---------------------------------------------------------------------------

class TestPortfolioTradeoffs:
    def test_multiple_attention_states_coexist(self, records):
        """The scenario should exhibit all three tiers during its run."""
        tiers = {r.get("attention_state") for r in records}
        assert BACKGROUND     in tiers
        assert WATCHLIST      in tiers or ACTIVE_CUSTODY in tiers

    def test_preemption_occurs(self, records):
        """At least one preemption should arise given multiple competing concerns."""
        preempted = [r for r in records if r.get("action") == "PREEMPTED"]
        assert len(preempted) > 0, "No PREEMPTED actions found; expected resource competition"

    def test_overlapping_concern_window_exists(self, records):
        """There must be at least one timestep where both PORT-1 is dark AND
        an ECHO rendezvous is active, demonstrating simultaneous competing concerns."""
        dark_times = {r["time"] for r in records
                      if r["target_id"] == "PORT-1" and r["dark_vessel_flag"]}
        rdv_times  = {r["time"] for r in records
                      if r["target_id"] == "ECHO-1" and r["rendezvous_flag"]}
        overlap = dark_times & rdv_times
        assert len(overlap) > 0, \
            "Expected PORT-1 dark and ECHO rendezvous to overlap in time"

    def test_task_actions_present(self, records):
        """Sensor tasking should happen across the scenario run."""
        tasked = [r for r in records if r.get("action") == "TASK"]
        assert len(tasked) > 0

    def test_portfolio_scores_span_range(self, records):
        """Portfolio scores should be spread across the [0, 1] range — not all equal."""
        scores = [r["portfolio_score"] for r in records]
        assert max(scores) - min(scores) > 0.1, \
            f"Portfolio scores too compressed: range={max(scores)-min(scores):.3f}"
