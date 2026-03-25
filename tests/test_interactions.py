"""
Tests for pairwise vessel interaction detection (custody/interactions.py).

Covers:
- detect_rendezvous_events: no event when too far apart
- no event when single-step proximity (below min_dwell threshold)
- event fires when dwell threshold is met (consecutive close steps)
- event does NOT fire when prior step breaks the consecutive streak
- dwell_hours scales with number of consecutive close steps
- min_separation_km reflects minimum over the dwell window
- confidence scales with dwell (saturates at 1.0)
- vessel_a < vessel_b always (lexicographic ordering)
- evidence string is non-empty and meaningful
- three vessels: only the close pair fires
- Portfolio/timeline integration: rendezvous fields present in records,
  both ECHO vessels flagged, counterpart IDs cross-reference correctly,
  flagged vessels receive non-zero compound_boost
"""
from __future__ import annotations

import pytest
from datetime import datetime, timezone

from custody.interactions import detect_rendezvous_events, RendezvousEvent

UTC = timezone.utc
T0 = datetime(2026, 3, 23, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pos(lat: float = 0.0, lon: float = 0.0) -> dict:
    """Minimal history record with lat/lon."""
    return {"lat": lat, "lon": lon}


def _close_pair(offset_lon: float = 0.001):
    """Return current_positions and 1-step histories both within 5 km."""
    positions = {"A": (0.0, 0.0), "B": (0.0, offset_lon)}
    histories = {
        "A": [_pos(0.0, 0.0)],
        "B": [_pos(0.0, offset_lon)],
    }
    return positions, histories


# ---------------------------------------------------------------------------
# TestDetectRendezvousEvents — unit tests
# ---------------------------------------------------------------------------

class TestDetectRendezvousEvents:

    def test_far_apart_no_event(self):
        positions = {"A": (0.0, 0.0), "B": (10.0, 0.0)}   # >1000 km
        histories = {"A": [], "B": []}
        events = detect_rendezvous_events(positions, histories, T0)
        assert events == []

    def test_single_step_below_min_dwell_no_event(self):
        """Current step is close but no prior history → total=1 < min_dwell=2."""
        positions = {"A": (0.0, 0.0), "B": (0.0, 0.001)}  # ~0.11 km
        histories = {"A": [], "B": []}
        events = detect_rendezvous_events(positions, histories, T0, min_dwell_steps=2)
        assert events == []

    def test_two_consecutive_close_steps_fires(self):
        """1 prior close step + current step = 2 ≥ min_dwell_steps=2."""
        positions, histories = _close_pair()
        events = detect_rendezvous_events(positions, histories, T0, min_dwell_steps=2)
        assert len(events) == 1

    def test_prior_step_not_close_breaks_streak(self):
        """Current close, but prior step has B far away → streak broken → no event."""
        positions = {"A": (0.0, 0.0), "B": (0.0, 0.001)}
        histories = {
            "A": [_pos(0.0, 0.0)],
            "B": [_pos(0.0, 5.0)],   # ~555 km away in prior step
        }
        events = detect_rendezvous_events(positions, histories, T0, min_dwell_steps=2)
        assert events == []

    def test_three_consecutive_steps_fires(self):
        """2 prior close steps + current = 3 ≥ min_dwell_steps=2."""
        positions = {"A": (0.0, 0.0), "B": (0.0, 0.001)}
        histories = {
            "A": [_pos(0.0, 0.0)] * 2,
            "B": [_pos(0.0, 0.001)] * 2,
        }
        events = detect_rendezvous_events(positions, histories, T0, min_dwell_steps=2)
        assert len(events) == 1

    def test_streak_broken_in_middle_counts_from_most_recent(self):
        """History has a gap: [far, close, close].  Only the 2 most recent
        consecutive steps count (plus current = 3), but the old far step breaks
        the lookback so only 2 consecutive steps + current are counted."""
        positions = {"A": (0.0, 0.0), "B": (0.0, 0.001)}
        histories = {
            "A": [_pos(0.0, 0.0), _pos(0.0, 0.0), _pos(0.0, 0.0)],
            # B: oldest record is far; two most recent are close
            "B": [_pos(0.0, 10.0), _pos(0.0, 0.001), _pos(0.0, 0.001)],
        }
        events = detect_rendezvous_events(positions, histories, T0, min_dwell_steps=2)
        # Walking backward: hist[-1] = (0.0, 0.001) close ✓, hist[-2] = (0.0, 0.001) close ✓,
        # hist[-3] = (0.0, 10.0) far → breaks streak.  dwell_prior=2, total=3 ≥ 2 → fires.
        assert len(events) == 1
        assert events[0].dwell_hours == pytest.approx(3.0)

    def test_dwell_hours_equals_steps_times_dt(self):
        """dwell_hours = total_dwell_steps * dt_hours."""
        positions = {"A": (0.0, 0.0), "B": (0.0, 0.001)}
        histories = {
            "A": [_pos(0.0, 0.0)] * 3,  # 3 prior steps
            "B": [_pos(0.0, 0.001)] * 3,
        }
        # total_dwell_steps = 4 (3 prior + 1 current), dt_hours = 1.0
        events = detect_rendezvous_events(positions, histories, T0, min_dwell_steps=2, dt_hours=1.0)
        assert len(events) == 1
        assert events[0].dwell_hours == pytest.approx(4.0)

    def test_dwell_hours_with_non_unit_dt(self):
        """dt_hours is honoured in dwell_hours calculation."""
        positions = {"A": (0.0, 0.0), "B": (0.0, 0.001)}
        histories = {"A": [_pos(0.0, 0.0)], "B": [_pos(0.0, 0.001)]}
        events = detect_rendezvous_events(positions, histories, T0, min_dwell_steps=2, dt_hours=2.0)
        # total=2 steps * 2.0 h/step = 4.0 h
        assert events[0].dwell_hours == pytest.approx(4.0)

    def test_min_separation_is_minimum_over_window(self):
        """min_separation_km = min of current + all prior close-step distances."""
        # Prior step: A and B ~0.11 km apart. Current step: exactly same point = 0 km.
        positions = {"A": (0.0, 0.0), "B": (0.0, 0.0)}   # current: same point
        histories = {
            "A": [_pos(0.0, 0.0)],
            "B": [_pos(0.0, 0.001)],   # prior: 0.111 km
        }
        events = detect_rendezvous_events(positions, histories, T0, min_dwell_steps=2)
        assert len(events) == 1
        assert events[0].min_separation_km == pytest.approx(0.0, abs=1e-6)

    def test_confidence_scales_with_dwell(self):
        """Longer dwell → higher confidence (up to 1.0)."""
        pos = {"A": (0.0, 0.0), "B": (0.0, 0.001)}
        hist_short = {"A": [_pos(0.0, 0.0)] * 1, "B": [_pos(0.0, 0.001)] * 1}
        hist_long  = {"A": [_pos(0.0, 0.0)] * 6, "B": [_pos(0.0, 0.001)] * 6}
        e_short = detect_rendezvous_events(pos, hist_short, T0, min_dwell_steps=2)
        e_long  = detect_rendezvous_events(pos, hist_long,  T0, min_dwell_steps=2)
        assert e_short[0].confidence < e_long[0].confidence

    def test_confidence_capped_at_one(self):
        pos = {"A": (0.0, 0.0), "B": (0.0, 0.001)}
        hist = {"A": [_pos(0.0, 0.0)] * 20, "B": [_pos(0.0, 0.001)] * 20}
        events = detect_rendezvous_events(pos, hist, T0, min_dwell_steps=2)
        assert events[0].confidence <= 1.0

    def test_confidence_is_nonnegative(self):
        positions, histories = _close_pair()
        events = detect_rendezvous_events(positions, histories, T0, min_dwell_steps=2)
        assert events[0].confidence >= 0.0

    def test_vessel_a_lt_vessel_b_lexicographic(self):
        """vessel_a < vessel_b always, regardless of insertion order."""
        positions = {"ZETA": (0.0, 0.0), "ALPHA": (0.0, 0.001)}
        histories = {"ZETA": [_pos(0.0, 0.0)], "ALPHA": [_pos(0.0, 0.001)]}
        events = detect_rendezvous_events(positions, histories, T0, min_dwell_steps=2)
        assert len(events) == 1
        assert events[0].vessel_a == "ALPHA"
        assert events[0].vessel_b == "ZETA"

    def test_event_covers_both_vessel_ids(self):
        """vessel_a and vessel_b together equal the input pair."""
        positions, histories = _close_pair()
        events = detect_rendezvous_events(positions, histories, T0)
        assert {events[0].vessel_a, events[0].vessel_b} == {"A", "B"}

    def test_evidence_is_nonempty_string(self):
        positions, histories = _close_pair()
        events = detect_rendezvous_events(positions, histories, T0)
        assert isinstance(events[0].evidence, str) and len(events[0].evidence) > 0

    def test_evidence_contains_vessel_ids(self):
        positions, histories = _close_pair()
        events = detect_rendezvous_events(positions, histories, T0)
        ev = events[0].evidence
        assert "A" in ev and "B" in ev

    def test_timestamp_written_to_event(self):
        positions, histories = _close_pair()
        events = detect_rendezvous_events(positions, histories, T0)
        assert events[0].timestamp == T0

    def test_three_vessels_only_close_pair_fires(self):
        positions = {
            "A": (0.0, 0.0),
            "B": (0.0, 0.001),   # ~0.11 km from A
            "C": (50.0, 0.0),    # far from both
        }
        histories = {
            "A": [_pos(0.0, 0.0)],
            "B": [_pos(0.0, 0.001)],
            "C": [_pos(50.0, 0.0)],
        }
        events = detect_rendezvous_events(positions, histories, T0, min_dwell_steps=2)
        assert len(events) == 1
        assert {events[0].vessel_a, events[0].vessel_b} == {"A", "B"}

    def test_two_independent_close_pairs_both_fire(self):
        """Two pairs each within threshold independently → two events."""
        positions = {
            "A1": (0.0, 0.0), "A2": (0.0, 0.001),    # pair 1: ~0.11 km
            "B1": (50.0, 0.0), "B2": (50.0, 0.001),  # pair 2: ~0.11 km
        }
        histories = {
            "A1": [_pos(0.0, 0.0)],   "A2": [_pos(0.0, 0.001)],
            "B1": [_pos(50.0, 0.0)], "B2": [_pos(50.0, 0.001)],
        }
        events = detect_rendezvous_events(positions, histories, T0, min_dwell_steps=2)
        assert len(events) == 2

    def test_custom_proximity_threshold_respected(self):
        """With a very tight threshold (0.05 km), a 0.11 km separation does not fire."""
        positions, histories = _close_pair(offset_lon=0.001)   # ~0.11 km
        events = detect_rendezvous_events(
            positions, histories, T0, proximity_km=0.05, min_dwell_steps=2
        )
        assert events == []

    def test_empty_positions_returns_empty(self):
        events = detect_rendezvous_events({}, {}, T0)
        assert events == []

    def test_single_vessel_returns_empty(self):
        events = detect_rendezvous_events({"A": (0.0, 0.0)}, {"A": []}, T0)
        assert events == []

    def test_imbalanced_histories_uses_shorter(self):
        """If A has 5 prior records and B has 1, look back only 1 step."""
        positions = {"A": (0.0, 0.0), "B": (0.0, 0.001)}
        histories = {
            "A": [_pos(0.0, 0.0)] * 5,
            "B": [_pos(0.0, 0.001)] * 1,  # only 1 prior record
        }
        events = detect_rendezvous_events(positions, histories, T0, min_dwell_steps=2)
        # min lookback = 1 → dwell_prior=1, total=2 ≥ 2 → fires
        assert len(events) == 1
        assert events[0].dwell_hours == pytest.approx(2.0)

    def test_min_dwell_steps_one_fires_immediately(self):
        """min_dwell_steps=1 fires on the first close step (no history required)."""
        positions = {"A": (0.0, 0.0), "B": (0.0, 0.001)}
        histories = {"A": [], "B": []}
        events = detect_rendezvous_events(positions, histories, T0, min_dwell_steps=1)
        assert len(events) == 1


# ---------------------------------------------------------------------------
# TestRendezvousScenarioIntegration — runs RENDEZVOUS_SMOKE end-to-end
# ---------------------------------------------------------------------------

class TestRendezvousScenarioIntegration:
    """Run RENDEZVOUS_SMOKE through the full simulation pipeline and verify
    that rendezvous fields are present, consistent, and semantically correct."""

    @pytest.fixture(scope="class")
    def records(self):
        from custody.simulation.timeline import run_multi_target_simulation
        from custody.simulation.scenarios import RENDEZVOUS_SMOKE
        return run_multi_target_simulation(RENDEZVOUS_SMOKE)

    def test_records_produced(self, records):
        assert len(records) > 0

    def test_rendezvous_fields_present_in_every_record(self, records):
        required = {
            "rendezvous_flag", "counterpart_id",
            "rendezvous_confidence", "rendezvous_dwell_hours",
            "min_pair_distance_km",
        }
        for rec in records:
            missing = required - rec.keys()
            assert not missing, f"Record missing fields: {missing}"

    def test_vessel_proximity_score_present(self, records):
        for rec in records:
            assert "vessel_proximity_score" in rec

    def test_rendezvous_flag_fires_for_both_echo_vessels(self, records):
        """At least one record per ECHO vessel has rendezvous_flag=True."""
        flagged = {r["target_id"] for r in records if r["rendezvous_flag"]}
        assert "ECHO-1" in flagged
        assert "ECHO-2" in flagged

    def test_counterpart_ids_are_cross_referenced(self, records):
        """ECHO-1's counterpart is ECHO-2 and vice versa when flagged."""
        for rec in records:
            if not rec["rendezvous_flag"]:
                continue
            if rec["target_id"] == "ECHO-1":
                assert rec["counterpart_id"] == "ECHO-2"
            elif rec["target_id"] == "ECHO-2":
                assert rec["counterpart_id"] == "ECHO-1"

    def test_rendezvous_confidence_in_unit_interval(self, records):
        for rec in records:
            assert 0.0 <= rec["rendezvous_confidence"] <= 1.0

    def test_dwell_hours_positive_when_flagged(self, records):
        for rec in records:
            if rec["rendezvous_flag"]:
                assert rec["rendezvous_dwell_hours"] > 0.0

    def test_min_pair_distance_km_nonnegative_when_flagged(self, records):
        for rec in records:
            if rec["rendezvous_flag"]:
                assert rec["min_pair_distance_km"] is not None
                assert rec["min_pair_distance_km"] >= 0.0

    def test_counterpart_none_when_not_flagged(self, records):
        for rec in records:
            if not rec["rendezvous_flag"]:
                assert rec["counterpart_id"] is None

    def test_non_flagged_records_have_zero_confidence(self, records):
        for rec in records:
            if not rec["rendezvous_flag"]:
                assert rec["rendezvous_confidence"] == 0.0
                assert rec["rendezvous_dwell_hours"] == 0.0

    def test_dwell_increases_monotonically_while_flagged(self, records):
        """Within a continuous flagged window, dwell_hours does not decrease."""
        for eid in ("ECHO-1", "ECHO-2"):
            vessel_records = sorted(
                (r for r in records if r["target_id"] == eid),
                key=lambda r: r["time"],
            )
            last_dwell = 0.0
            in_rendezvous = False
            for rec in vessel_records:
                if rec["rendezvous_flag"]:
                    if in_rendezvous:
                        assert rec["rendezvous_dwell_hours"] >= last_dwell, (
                            f"{eid}: dwell decreased from {last_dwell} to "
                            f"{rec['rendezvous_dwell_hours']}"
                        )
                    in_rendezvous = True
                    last_dwell = rec["rendezvous_dwell_hours"]
                else:
                    in_rendezvous = False
                    last_dwell = 0.0

    def test_flagged_vessels_have_nonzero_proximity_score(self, records):
        """When rendezvous_flag is True the vessel_proximity_score must be > 0."""
        for rec in records:
            if rec["rendezvous_flag"]:
                assert rec["vessel_proximity_score"] > 0.0, (
                    f"{rec['target_id']} flagged rendezvous but proximity_score=0"
                )

    def test_portfolio_fields_present(self, records):
        """Confirm that portfolio orchestration still runs and annotates records."""
        for rec in records:
            assert "portfolio_rank" in rec
            assert "attention_state" in rec
