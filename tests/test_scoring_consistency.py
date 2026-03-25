"""
Tests for scoring path consistency across simulation, portfolio, and UI.

Verifies that the unification of fusion/decision into the simulation loop
produces a single authoritative scoring path:

  1. Every simulation record carries a canonical FusionAssessment and Decision.
  2. The portfolio ranking consumes fused_score from the stored FusionAssessment
     rather than recomputing urgency from raw anomaly.
  3. The fusion/decision objects stored on the record match what
     build_fusion_assessment / build_decision would produce from the same
     record and history.
  4. No legacy scoring path silently diverges.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

import pytest

from custody.compounds import evaluate_compounds
from custody.decision import Decision, build_decision
from custody.fusion import FusionAssessment, build_fusion_assessment
from custody.models import TrackState
from custody.orchestration.portfolio import _portfolio_score

UTC = timezone.utc


# ---------------------------------------------------------------------------
# Fixtures — run each scenario once per test session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def portfolio_records():
    """Full 36h PORTFOLIO_SCENARIO records."""
    from custody.simulation import run_multi_target_simulation
    from custody.simulation.scenarios import PORTFOLIO_SCENARIO
    return run_multi_target_simulation(PORTFOLIO_SCENARIO)


@pytest.fixture(scope="module")
def smoke_records():
    """Two-vessel smoke scenario records."""
    from custody.simulate import run_simulation
    return run_simulation()


# ---------------------------------------------------------------------------
# 1. Every record carries canonical fusion/decision objects
# ---------------------------------------------------------------------------

class TestCanonicalObjectsPresent:

    def test_fusion_assessment_present_on_every_record(self, portfolio_records):
        for r in portfolio_records:
            fa = r.get("fusion_assessment")
            assert fa is not None, (
                f"Record for {r['target_id']} at {r['time']} missing fusion_assessment"
            )
            assert isinstance(fa, FusionAssessment)

    def test_mission_decision_present_on_every_record(self, portfolio_records):
        for r in portfolio_records:
            dec = r.get("mission_decision")
            assert dec is not None, (
                f"Record for {r['target_id']} at {r['time']} missing mission_decision"
            )
            assert isinstance(dec, Decision)

    def test_smoke_records_also_carry_objects(self, smoke_records):
        for r in smoke_records:
            assert isinstance(r.get("fusion_assessment"), FusionAssessment)
            assert isinstance(r.get("mission_decision"), Decision)

    def test_fusion_entity_id_matches_record(self, portfolio_records):
        for r in portfolio_records[:50]:
            assert r["fusion_assessment"].entity_id == r["target_id"]

    def test_decision_entity_id_matches_record(self, portfolio_records):
        for r in portfolio_records[:50]:
            assert r["mission_decision"].entity_id == r["target_id"]

    def test_fusion_timestamp_matches_record(self, portfolio_records):
        for r in portfolio_records[:50]:
            assert r["fusion_assessment"].timestamp == r["time"]

    def test_decision_timestamp_matches_record(self, portfolio_records):
        for r in portfolio_records[:50]:
            assert r["mission_decision"].timestamp == r["time"]


# ---------------------------------------------------------------------------
# 2. Stored fusion/decision is reproducible from the same inputs
# ---------------------------------------------------------------------------

class TestRoundTripConsistency:
    """Rebuild fusion/decision from the record and verify it matches stored."""

    @staticmethod
    def _derive_track(record, prefix):
        """Derive a TrackState the same way the entity detail panel does."""
        uncertainty_km = float(record.get("uncertainty_km", 5.0))
        last_collection_time = None
        last_collection_anomaly_score = 0.0
        for r in reversed(prefix):
            result = r.get("collection_result")
            if r.get("action") == "TASK" and result not in (None, "", "—", "NONE"):
                t = r.get("time")
                if isinstance(t, datetime):
                    last_collection_time = t
                    last_collection_anomaly_score = float(r.get("anomaly_score", 0.0))
                break
        return TrackState(
            uncertainty_km=uncertainty_km,
            last_collection_time=last_collection_time,
            last_collection_anomaly_score=last_collection_anomaly_score,
        )

    def test_fused_score_reproducible(self, portfolio_records):
        """Rebuilding fusion from the record produces the same fused_score."""
        by_entity: dict[str, list[dict]] = defaultdict(list)
        for r in portfolio_records:
            by_entity[r["target_id"]].append(r)

        mismatches = []
        # Sample: check every 4th entity, every 3rd record
        for eid in sorted(by_entity)[::4]:
            timeline = by_entity[eid]
            for i in range(0, len(timeline), 3):
                record = timeline[i]
                prefix = timeline[:i]
                track = self._derive_track(record, prefix)
                compounds = evaluate_compounds(record, window=prefix)
                rebuilt_fa = build_fusion_assessment(record, compounds, track)
                stored_fa = record["fusion_assessment"]
                if abs(rebuilt_fa.fused_score - stored_fa.fused_score) > 0.001:
                    mismatches.append(
                        f"{eid}@{record['time']}: stored={stored_fa.fused_score:.4f} "
                        f"rebuilt={rebuilt_fa.fused_score:.4f}"
                    )
        assert not mismatches, (
            f"fused_score diverged in {len(mismatches)} records:\n"
            + "\n".join(mismatches[:10])
        )

    def test_decision_action_reproducible(self, portfolio_records):
        """Rebuilding decision from the record produces the same action."""
        by_entity: dict[str, list[dict]] = defaultdict(list)
        for r in portfolio_records:
            by_entity[r["target_id"]].append(r)

        mismatches = []
        for eid in sorted(by_entity)[::4]:
            timeline = by_entity[eid]
            for i in range(0, len(timeline), 3):
                record = timeline[i]
                prefix = timeline[:i]
                track = self._derive_track(record, prefix)
                compounds = evaluate_compounds(record, window=prefix)
                fa = build_fusion_assessment(record, compounds, track)
                rebuilt_dec = build_decision(fa, record, track, compounds)
                stored_dec = record["mission_decision"]
                if rebuilt_dec.action != stored_dec.action:
                    mismatches.append(
                        f"{eid}@{record['time']}: stored={stored_dec.action} "
                        f"rebuilt={rebuilt_dec.action}"
                    )
        assert not mismatches, (
            f"decision action diverged in {len(mismatches)} records:\n"
            + "\n".join(mismatches[:10])
        )


# ---------------------------------------------------------------------------
# 3. Portfolio ranking uses fused_score, not raw anomaly
# ---------------------------------------------------------------------------

class TestPortfolioUsesFusedScore:

    def test_portfolio_score_uses_fused_when_provided(self):
        """_portfolio_score with fused_score should ignore raw anomaly for urgency."""
        # Same anomaly, different fused_score → different portfolio score
        score_high_fused = _portfolio_score(
            anomaly_score=0.5, custody_confidence=0.8,
            uncertainty_km=10.0, neglect_hours=2.0,
            attention_state="ACTIVE_CUSTODY", fused_score=0.9,
        )
        score_low_fused = _portfolio_score(
            anomaly_score=0.5, custody_confidence=0.8,
            uncertainty_km=10.0, neglect_hours=2.0,
            attention_state="ACTIVE_CUSTODY", fused_score=0.1,
        )
        assert score_high_fused > score_low_fused, (
            f"High fused_score ({score_high_fused}) should produce higher portfolio "
            f"score than low fused_score ({score_low_fused})"
        )

    def test_portfolio_score_ignores_anomaly_when_fused_provided(self):
        """When fused_score is set, raw anomaly should not affect the urgency term."""
        score_high_anomaly = _portfolio_score(
            anomaly_score=3.0, custody_confidence=0.8,
            uncertainty_km=10.0, neglect_hours=2.0,
            attention_state="ACTIVE_CUSTODY", fused_score=0.5,
        )
        score_low_anomaly = _portfolio_score(
            anomaly_score=0.0, custody_confidence=0.8,
            uncertainty_km=10.0, neglect_hours=2.0,
            attention_state="ACTIVE_CUSTODY", fused_score=0.5,
        )
        assert score_high_anomaly == score_low_anomaly, (
            "With the same fused_score, raw anomaly should not change portfolio score"
        )

    def test_portfolio_score_falls_back_without_fused(self):
        """Without fused_score, _portfolio_score uses raw anomaly (backward compat)."""
        score_a = _portfolio_score(
            anomaly_score=2.0, custody_confidence=0.8,
            uncertainty_km=10.0, neglect_hours=2.0,
            attention_state="ACTIVE_CUSTODY", fused_score=None,
        )
        score_b = _portfolio_score(
            anomaly_score=0.1, custody_confidence=0.8,
            uncertainty_km=10.0, neglect_hours=2.0,
            attention_state="ACTIVE_CUSTODY", fused_score=None,
        )
        assert score_a > score_b, (
            "Fallback path: higher anomaly should produce higher portfolio score"
        )

    def test_portfolio_retains_neglect_dynamics(self):
        """Neglect hours still raise portfolio score independent of fused_score."""
        score_neglected = _portfolio_score(
            anomaly_score=0.1, custody_confidence=0.8,
            uncertainty_km=10.0, neglect_hours=10.0,
            attention_state="ACTIVE_CUSTODY", fused_score=0.3,
        )
        score_fresh = _portfolio_score(
            anomaly_score=0.1, custody_confidence=0.8,
            uncertainty_km=10.0, neglect_hours=0.0,
            attention_state="ACTIVE_CUSTODY", fused_score=0.3,
        )
        assert score_neglected > score_fresh

    def test_portfolio_retains_uncertainty_dynamics(self):
        """High positional uncertainty still raises portfolio score."""
        score_uncertain = _portfolio_score(
            anomaly_score=0.1, custody_confidence=0.8,
            uncertainty_km=100.0, neglect_hours=2.0,
            attention_state="ACTIVE_CUSTODY", fused_score=0.3,
        )
        score_precise = _portfolio_score(
            anomaly_score=0.1, custody_confidence=0.8,
            uncertainty_km=1.0, neglect_hours=2.0,
            attention_state="ACTIVE_CUSTODY", fused_score=0.3,
        )
        assert score_uncertain > score_precise

    def test_portfolio_retains_tier_dynamics(self):
        """Attention tier baseline still differentiates entities."""
        score_active = _portfolio_score(
            anomaly_score=0.1, custody_confidence=0.8,
            uncertainty_km=10.0, neglect_hours=2.0,
            attention_state="ACTIVE_CUSTODY", fused_score=0.3,
        )
        score_bg = _portfolio_score(
            anomaly_score=0.1, custody_confidence=0.8,
            uncertainty_km=10.0, neglect_hours=2.0,
            attention_state="BACKGROUND", fused_score=0.3,
        )
        assert score_active > score_bg


# ---------------------------------------------------------------------------
# 4. Portfolio records use fused_score in integration
# ---------------------------------------------------------------------------

class TestPortfolioIntegrationWithFusedScore:

    def test_portfolio_score_in_range(self, portfolio_records):
        for r in portfolio_records:
            assert 0.0 <= r["portfolio_score"] <= 1.0

    def test_portfolio_ranking_reads_stored_fused_score(self, portfolio_records):
        """Prove the wiring: re-rank one timestep with and without the stored
        FusionAssessment and verify the portfolio scores differ — confirming
        that rank_portfolio actually reads the fusion_assessment field."""
        from custody.orchestration.portfolio import rank_portfolio
        from custody.simulation.scenarios import PORTFOLIO_SCENARIO

        by_time: dict[datetime, list[dict]] = defaultdict(list)
        for r in portfolio_records:
            by_time[r["time"]].append(r)

        # Pick a mid-scenario timestep with varied fused scores
        ts = sorted(by_time.keys())[18]
        ts_records = by_time[ts]

        # Rank with stored fusion_assessment (the production path)
        pa_with = rank_portfolio(
            timestamp=ts,
            timestep_records=ts_records,
            scenario_start=PORTFOLIO_SCENARIO.start_time,
        )

        # Strip fusion_assessment from copies and re-rank (fallback path)
        stripped = []
        for r in ts_records:
            copy = dict(r)
            copy["fusion_assessment"] = None
            stripped.append(copy)

        pa_without = rank_portfolio(
            timestamp=ts,
            timestep_records=stripped,
            scenario_start=PORTFOLIO_SCENARIO.start_time,
        )

        # At least one entity's portfolio_score must differ
        scores_with    = {it.entity_id: it.portfolio_score for it in pa_with.items}
        scores_without = {it.entity_id: it.portfolio_score for it in pa_without.items}
        diffs = [
            abs(scores_with[eid] - scores_without[eid])
            for eid in scores_with
        ]
        assert max(diffs) > 0.001, (
            "Portfolio scores identical with and without fusion_assessment — "
            "rank_portfolio may not be reading the stored fused_score"
        )


# ---------------------------------------------------------------------------
# 5. Fusion/decision field ranges and invariants
# ---------------------------------------------------------------------------

class TestFieldInvariants:

    def test_fused_score_in_unit_interval(self, portfolio_records):
        for r in portfolio_records:
            fs = r["fusion_assessment"].fused_score
            assert 0.0 <= fs <= 1.0, f"fused_score={fs} out of [0,1]"

    def test_uncertainty_in_unit_interval(self, portfolio_records):
        for r in portfolio_records:
            unc = r["fusion_assessment"].uncertainty
            assert 0.0 <= unc <= 1.0, f"uncertainty={unc} out of [0,1]"

    def test_decision_action_in_vocabulary(self, portfolio_records):
        valid = {"PASSIVE_MONITOR", "ELEVATE", "TASK_OPTICAL", "TASK_SAR", "ESCALATE"}
        for r in portfolio_records:
            assert r["mission_decision"].action in valid, (
                f"Unexpected action: {r['mission_decision'].action}"
            )

    def test_decision_priority_in_unit_interval(self, portfolio_records):
        for r in portfolio_records:
            p = r["mission_decision"].priority
            assert 0.0 <= p <= 1.0, f"priority={p} out of [0,1]"

    def test_decision_confidence_in_unit_interval(self, portfolio_records):
        for r in portfolio_records:
            c = r["mission_decision"].confidence
            assert 0.0 <= c <= 1.0, f"confidence={c} out of [0,1]"

    def test_decision_why_nonempty(self, portfolio_records):
        for r in portfolio_records[:100]:
            assert len(r["mission_decision"].why) >= 1
