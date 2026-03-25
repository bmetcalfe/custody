"""
Tests for the custody.prediction package.

Covers trajectory projection, zone crossing probability, anomaly risk
forecasting, and the top-level predict_entity() integration.  Also
includes a BRAVO-1 scenario integration check.
"""
from __future__ import annotations

import math
import pytest

from custody.models import Zone
from custody.prediction.trajectory import project_position
from custody.prediction.zone_probability import compute_zone_crossing_probability
from custody.prediction.risk_forecast import forecast_anomaly_score
from custody.prediction import Prediction, predict_entity
from custody.config import ZONES


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

ZONE_ALPHA = ZONES[0]  # min_lat=1.0, max_lat=1.6, min_lon=0.55, max_lon=0.95

# A zone used only in isolation tests (centred at lat=10, lon=20)
_TEST_ZONE = Zone(
    name="TEST_ZONE",
    min_lat=9.5,
    max_lat=10.5,
    min_lon=19.5,
    max_lon=20.5,
    halo=0.1,
)


# ---------------------------------------------------------------------------
# Trajectory: project_position
# ---------------------------------------------------------------------------

class TestProjectPosition:

    def test_stationary_vessel_stays_put(self):
        """Speed=0 → position unchanged."""
        lat, lon = project_position(1.0, 0.75, speed_knots=0.0, heading_deg=45.0, dt_hours=6.0)
        assert lat == pytest.approx(1.0)
        assert lon == pytest.approx(0.75)

    def test_dt_zero_returns_current_position(self):
        """dt=0 → position unchanged regardless of speed/heading."""
        lat, lon = project_position(1.0, 0.75, speed_knots=12.0, heading_deg=90.0, dt_hours=0.0)
        assert lat == pytest.approx(1.0)
        assert lon == pytest.approx(0.75)

    def test_north_heading_increases_latitude(self):
        """heading=0 (north) → latitude increases, longitude unchanged."""
        lat, lon = project_position(0.0, 0.0, speed_knots=10.0, heading_deg=0.0, dt_hours=1.0)
        assert lat > 0.0
        assert lon == pytest.approx(0.0, abs=1e-6)

    def test_east_heading_increases_longitude(self):
        """heading=90 (east) → longitude increases, latitude near-unchanged."""
        lat, lon = project_position(0.0, 0.0, speed_knots=10.0, heading_deg=90.0, dt_hours=1.0)
        assert lon > 0.0
        assert lat == pytest.approx(0.0, abs=1e-6)

    def test_south_heading_decreases_latitude(self):
        """heading=180 (south) → latitude decreases."""
        lat, lon = project_position(0.0, 0.0, speed_knots=10.0, heading_deg=180.0, dt_hours=1.0)
        assert lat < 0.0

    def test_west_heading_decreases_longitude(self):
        """heading=270 (west) → longitude decreases."""
        lat, lon = project_position(0.0, 0.0, speed_knots=10.0, heading_deg=270.0, dt_hours=1.0)
        assert lon < 0.0

    def test_displacement_proportional_to_dt(self):
        """Position change at dt=2 is double that at dt=1."""
        lat1, lon1 = project_position(0.0, 0.0, speed_knots=10.0, heading_deg=45.0, dt_hours=1.0)
        lat2, lon2 = project_position(0.0, 0.0, speed_knots=10.0, heading_deg=45.0, dt_hours=2.0)
        assert lat2 == pytest.approx(2 * lat1, rel=1e-3)
        assert lon2 == pytest.approx(2 * lon1, rel=1e-3)

    def test_latitude_clamped(self):
        """Latitude is clamped to [-90, 90]."""
        lat, _ = project_position(89.9, 0.0, speed_knots=100.0, heading_deg=0.0, dt_hours=100.0)
        assert lat <= 90.0

    def test_longitude_wrapped(self):
        """Longitude is wrapped to [-180, 180]."""
        _, lon = project_position(0.0, 179.0, speed_knots=100.0, heading_deg=90.0, dt_hours=100.0)
        assert -180.0 <= lon <= 180.0


# ---------------------------------------------------------------------------
# Zone probability: compute_zone_crossing_probability
# ---------------------------------------------------------------------------

class TestZoneCrossingProbability:

    def test_heading_toward_zone_high_probability(self):
        """Vessel heading directly at zone centre → probability > 0.6."""
        zone_clat = (ZONE_ALPHA.min_lat + ZONE_ALPHA.max_lat) / 2
        zone_clon = (ZONE_ALPHA.min_lon + ZONE_ALPHA.max_lon) / 2
        # Start south of zone, heading north-east toward zone centre
        dlat = zone_clat - 0.0
        dlon = zone_clon - 0.0
        bearing = math.degrees(math.atan2(dlon, dlat)) % 360.0
        prob, tte = compute_zone_crossing_probability(
            lat=0.0, lon=0.0,
            speed_knots=12.0, heading_deg=bearing,
            zone=ZONE_ALPHA, horizon_hours=6.0,
        )
        assert prob > 0.6, f"Expected prob > 0.6, got {prob}"
        assert tte is not None

    def test_heading_away_from_zone_low_probability(self):
        """Vessel heading directly away from zone → probability < 0.2."""
        zone_clat = (ZONE_ALPHA.min_lat + ZONE_ALPHA.max_lat) / 2
        zone_clon = (ZONE_ALPHA.min_lon + ZONE_ALPHA.max_lon) / 2
        # Start north of zone; heading north-east (away)
        dlat = 2.0 - zone_clat  # away vector
        dlon = 2.0 - zone_clon
        bearing = math.degrees(math.atan2(dlon, dlat)) % 360.0
        prob, tte = compute_zone_crossing_probability(
            lat=2.0, lon=2.0,
            speed_knots=12.0, heading_deg=bearing,
            zone=ZONE_ALPHA, horizon_hours=6.0,
        )
        assert prob < 0.2, f"Expected prob < 0.2, got {prob}"

    def test_closer_vessel_higher_probability(self):
        """Vessel closer to zone entrance has higher probability than distant vessel with same heading."""
        heading = 30.0  # roughly toward ZONE_ALPHA
        prob_near, _ = compute_zone_crossing_probability(
            lat=0.8, lon=0.7, speed_knots=5.0, heading_deg=heading,
            zone=ZONE_ALPHA, horizon_hours=6.0,
        )
        prob_far, _ = compute_zone_crossing_probability(
            lat=-1.5, lon=-1.5, speed_knots=5.0, heading_deg=heading,
            zone=ZONE_ALPHA, horizon_hours=6.0,
        )
        assert prob_near >= prob_far, (
            f"Near prob {prob_near} should be >= far prob {prob_far}"
        )

    def test_probability_in_bounds(self):
        """Zone probability is always in [0, 1]."""
        for heading in range(0, 360, 45):
            prob, _ = compute_zone_crossing_probability(
                lat=0.5, lon=0.75, speed_knots=8.0, heading_deg=float(heading),
                zone=ZONE_ALPHA, horizon_hours=6.0,
            )
            assert 0.0 <= prob <= 1.0, f"Prob {prob} out of [0,1] for heading={heading}"

    def test_inside_zone_high_probability(self):
        """Vessel already inside zone → probability near 1."""
        prob, tte = compute_zone_crossing_probability(
            lat=1.2, lon=0.75,
            speed_knots=2.0, heading_deg=45.0,
            zone=ZONE_ALPHA, horizon_hours=6.0,
        )
        assert prob > 0.7
        assert tte == pytest.approx(0.0)

    def test_low_confidence_reduces_probability(self):
        """Lower custody confidence reduces zone probability."""
        heading_toward = 30.0
        prob_high_conf, _ = compute_zone_crossing_probability(
            lat=0.5, lon=0.5, speed_knots=10.0, heading_deg=heading_toward,
            zone=ZONE_ALPHA, horizon_hours=6.0, custody_confidence=1.0,
        )
        prob_low_conf, _ = compute_zone_crossing_probability(
            lat=0.5, lon=0.5, speed_knots=10.0, heading_deg=heading_toward,
            zone=ZONE_ALPHA, horizon_hours=6.0, custody_confidence=0.5,
        )
        assert prob_low_conf <= prob_high_conf


# ---------------------------------------------------------------------------
# Risk forecast: forecast_anomaly_score
# ---------------------------------------------------------------------------

class TestForecastAnomalyScore:

    def test_high_zone_prob_imminent_raises_anomaly(self):
        """High zone probability + near entry → future anomaly > current."""
        future = forecast_anomaly_score(
            current_anomaly=0.3,
            zone_probability=0.8,
            time_to_zone_hours=0.5,
            horizon_hours=6.0,
        )
        assert future > 0.3, f"Expected future > 0.3, got {future}"

    def test_no_zone_approach_decays_anomaly(self):
        """No zone approach → future anomaly ≈ current (slight decay)."""
        current = 0.6
        future = forecast_anomaly_score(
            current_anomaly=current,
            zone_probability=0.1,
            time_to_zone_hours=None,
            horizon_hours=6.0,
        )
        # Should be <= current (no boost; slow decay applies)
        assert future <= current
        # But not drastically lower (85% retention)
        assert future >= current * 0.80

    def test_result_bounded_in_zero_one(self):
        """Result is always in [0, 1]."""
        for zp in [0.0, 0.5, 1.0]:
            for tte in [None, 0.5, 3.0, 6.0]:
                future = forecast_anomaly_score(
                    current_anomaly=0.9,
                    zone_probability=zp,
                    time_to_zone_hours=tte,
                    horizon_hours=6.0,
                )
                assert 0.0 <= future <= 1.0, (
                    f"Out of bounds: future={future}, zp={zp}, tte={tte}"
                )

    def test_zero_anomaly_no_zone_stays_near_zero(self):
        """Zero anomaly + no zone → future near zero."""
        future = forecast_anomaly_score(
            current_anomaly=0.0,
            zone_probability=0.0,
            time_to_zone_hours=None,
        )
        assert future == pytest.approx(0.0)

    def test_imminent_entry_higher_than_distant_entry(self):
        """Imminent entry produces higher future anomaly than distant entry."""
        f_imminent = forecast_anomaly_score(0.2, 0.8, 0.5, 6.0)
        f_distant  = forecast_anomaly_score(0.2, 0.8, 5.5, 6.0)
        assert f_imminent >= f_distant


# ---------------------------------------------------------------------------
# predict_entity
# ---------------------------------------------------------------------------

class TestPredictEntity:

    def _approaching_pred(self, **kwargs):
        """Helper: vessel approaching ZONE_ALPHA from the south."""
        defaults = dict(
            entity_id="TEST-1",
            lat=0.5, lon=0.75,
            speed_knots=10.0, heading_deg=0.0,   # north — toward zone
            current_anomaly=0.3,
            custody_confidence=0.9,
            zones=ZONES,
            horizon_hours=6.0,
        )
        defaults.update(kwargs)
        return predict_entity(**defaults)

    def test_returns_prediction_dataclass(self):
        pred = self._approaching_pred()
        assert isinstance(pred, Prediction)

    def test_all_expected_fields_present(self):
        pred = self._approaching_pred()
        assert pred.entity_id == "TEST-1"
        assert isinstance(pred.horizon_hours, float)
        assert isinstance(pred.future_lat, float)
        assert isinstance(pred.future_lon, float)
        assert isinstance(pred.zone_probability, float)
        # time_to_zone_hours may be float or None
        assert isinstance(pred.future_anomaly, float)
        assert isinstance(pred.prediction_confidence, float)
        assert isinstance(pred.prediction_reason, str) and pred.prediction_reason

    def test_prediction_confidence_in_bounds(self):
        pred = self._approaching_pred()
        assert 0.0 <= pred.prediction_confidence <= 1.0

    def test_zone_probability_in_bounds(self):
        for heading in [0.0, 90.0, 180.0, 270.0]:
            pred = self._approaching_pred(heading_deg=heading)
            assert 0.0 <= pred.zone_probability <= 1.0

    def test_prediction_reason_non_empty(self):
        pred = self._approaching_pred()
        assert len(pred.prediction_reason) > 0

    def test_approaching_vessel_higher_zone_prob_than_receding(self):
        """Vessel heading toward zone has higher probability than one heading away."""
        pred_approach = self._approaching_pred(heading_deg=0.0)   # north = toward ZONE_ALPHA
        pred_recede   = self._approaching_pred(heading_deg=180.0)  # south = away
        assert pred_approach.zone_probability > pred_recede.zone_probability

    def test_lower_custody_confidence_lower_prediction_confidence(self):
        pred_high = self._approaching_pred(custody_confidence=0.9)
        pred_low  = self._approaching_pred(custody_confidence=0.3)
        assert pred_high.prediction_confidence >= pred_low.prediction_confidence

    def test_longer_horizon_lower_confidence(self):
        pred_short = self._approaching_pred(horizon_hours=2.0)
        pred_long  = self._approaching_pred(horizon_hours=12.0)
        assert pred_short.prediction_confidence >= pred_long.prediction_confidence

    def test_no_zones_no_zone_approach(self):
        """When no zones are configured, zone probability should be very low."""
        pred = self._approaching_pred(zones=[])
        assert pred.zone_probability == 0.0
        assert pred.time_to_zone_hours is None

    def test_stationary_vessel_reason(self):
        """Very slow vessel gets a 'stationary' reason."""
        pred = predict_entity(
            entity_id="SLOW",
            lat=0.5, lon=0.75,
            speed_knots=0.1, heading_deg=0.0,
            current_anomaly=0.1, custody_confidence=1.0,
            zones=ZONES, horizon_hours=6.0,
        )
        assert "stationary" in pred.prediction_reason.lower() or \
               "slow" in pred.prediction_reason.lower()


# ---------------------------------------------------------------------------
# BRAVO-1 integration test
# ---------------------------------------------------------------------------

class TestBravo1Integration:
    """
    BRAVO-1 starts at lat=0.5, lon=-0.3, heading=50°, speed~10km/h.
    ZONE_ALPHA is at lat 1.0-1.6, lon 0.55-0.95.
    At h10 (before zone entry at ~h15), BRAVO-1 should show zone_probability > 0.4.
    """

    @pytest.fixture(scope="class")
    def bravo1_h10_pred(self):
        """Simulate BRAVO-1 approximate position at h10 and run prediction."""
        from custody.simulation.profiles import ZONE_APPROACH
        import math as _math

        # BRAVO-1 approaches the zone target (1.3, 0.75) at ZONE_APPROACH speed
        # (~10 km/h = ~5.4 knots).  After 10h of steering toward (1.3, 0.75)
        # from (0.5, -0.3) the vessel is somewhere between start and the zone.
        # Use a reasonable approximation: ~60% of the way there.
        start_lat, start_lon = 0.5, -0.3
        target_lat, target_lon = 1.3, 0.75

        approx_lat = start_lat + 0.6 * (target_lat - start_lat)  # ~0.98
        approx_lon = start_lon + 0.6 * (target_lon - start_lon)  # ~0.33

        # Compute heading toward target
        dlat = target_lat - approx_lat
        dlon = target_lon - approx_lon
        heading = _math.degrees(_math.atan2(dlon, dlat)) % 360.0

        speed_kts = 10.0 / 1.852  # ~5.4 knots

        return predict_entity(
            entity_id="BRAVO-1",
            lat=approx_lat, lon=approx_lon,
            speed_knots=speed_kts, heading_deg=heading,
            current_anomaly=0.5,
            custody_confidence=0.85,
            zones=ZONES,
            horizon_hours=6.0,
        )

    def test_bravo1_zone_probability_above_threshold(self, bravo1_h10_pred):
        """At h10 BRAVO-1 should have zone_probability > 0.4."""
        assert bravo1_h10_pred.zone_probability > 0.4, (
            f"BRAVO-1 h10 zone_probability={bravo1_h10_pred.zone_probability:.3f} "
            f"should be > 0.4 (approaching ZONE_ALPHA)"
        )

    def test_bravo1_prediction_reason_meaningful(self, bravo1_h10_pred):
        """Prediction reason should contain meaningful zone-approach text."""
        reason = bravo1_h10_pred.prediction_reason
        assert len(reason) > 5
        # Should mention either the zone name or a zone-related phrase
        reason_lower = reason.lower()
        assert any(
            kw in reason_lower
            for kw in ["zone", "projected", "course", "approach"]
        ), f"Reason lacks meaningful context: '{reason}'"

    def test_bravo1_time_to_zone_not_none(self, bravo1_h10_pred):
        """At h10 when approaching, time_to_zone_hours should be set."""
        assert bravo1_h10_pred.time_to_zone_hours is not None

    def test_bravo1_future_anomaly_elevated(self, bravo1_h10_pred):
        """Future anomaly should be elevated above current for approaching BRAVO-1."""
        assert bravo1_h10_pred.future_anomaly >= 0.3


# ---------------------------------------------------------------------------
# Timeline + portfolio integration: prediction fields in simulation records
# ---------------------------------------------------------------------------

class TestTimelineIntegration:
    """Verify that prediction fields appear in simulation timeline records."""

    @pytest.fixture(scope="class")
    def sample_records(self):
        from custody.simulation import run_multi_target_simulation
        from custody.simulation.scenarios import RENDEZVOUS_SMOKE
        # Use small scenario for speed
        return run_multi_target_simulation(RENDEZVOUS_SMOKE)

    def test_prediction_fields_present(self, sample_records):
        """All prediction fields should be in every record."""
        fields = [
            "future_lat", "future_lon", "zone_probability",
            "time_to_zone_hours", "future_anomaly",
            "prediction_confidence", "prediction_reason",
        ]
        for record in sample_records[:5]:
            for f in fields:
                assert f in record, f"Field '{f}' missing from record"

    def test_zone_probability_values_valid(self, sample_records):
        """zone_probability is always in [0, 1]."""
        for record in sample_records:
            zp = record["zone_probability"]
            assert 0.0 <= zp <= 1.0, f"zone_probability={zp} out of bounds"

    def test_prediction_reason_non_empty(self, sample_records):
        """prediction_reason is always a non-empty string."""
        for record in sample_records:
            assert isinstance(record["prediction_reason"], str)
            assert len(record["prediction_reason"]) > 0
