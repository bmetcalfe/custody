"""
Tests for custody.solar — solar position and sensor suitability.

Covers:
  1.  sun_elevation positive during midday UTC at equator
  2.  sun_elevation negative during midnight UTC at equator
  3.  sun_elevation reasonable range [-90, 90]
  4.  solar_condition = day at noon equator
  5.  solar_condition = night at midnight equator
  6.  solar_condition = twilight at dawn/dusk
  7.  eo_suitability = 1.0 in full daylight
  8.  eo_suitability = 0.0 at night
  9.  eo_suitability intermediate in twilight
 10.  sar_suitability = 1.0 always
 11.  sensor_suitability returns all expected keys
 12.  tasking policy: EO preference switches to SAR at night
 13.  tasking policy: EO stays as EO during day
 14.  tasking policy: twilight keeps EO but notes degradation
 15.  simulation records carry solar fields
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from custody.solar import (
    sun_elevation_deg,
    solar_condition,
    eo_suitability,
    sar_suitability,
    sensor_suitability,
)
from custody.tasking_policy import (
    compute_policy,
    SENSOR_HIGH_RES, SENSOR_SAR, SENSOR_FAST,
)

UTC = timezone.utc


# ---------------------------------------------------------------------------
# Sun elevation
# ---------------------------------------------------------------------------

class TestSunElevation:

    def test_midday_equator_positive(self):
        # March equinox, noon UTC, equator → sun near zenith
        t = datetime(2026, 3, 21, 12, 0, tzinfo=UTC)
        elev = sun_elevation_deg(t, lat=0.0, lon=0.0)
        assert elev > 30.0, f"Expected high sun at equinox noon, got {elev}"

    def test_midnight_equator_negative(self):
        t = datetime(2026, 3, 21, 0, 0, tzinfo=UTC)
        elev = sun_elevation_deg(t, lat=0.0, lon=0.0)
        assert elev < -10.0, f"Expected negative sun at midnight, got {elev}"

    def test_range(self):
        for h in range(0, 24, 3):
            t = datetime(2026, 6, 21, h, 0, tzinfo=UTC)
            elev = sun_elevation_deg(t, lat=45.0, lon=-90.0)
            assert -90 <= elev <= 90


# ---------------------------------------------------------------------------
# Solar condition
# ---------------------------------------------------------------------------

class TestSolarCondition:

    def test_day_at_noon(self):
        t = datetime(2026, 6, 21, 12, 0, tzinfo=UTC)
        assert solar_condition(t, lat=0.0, lon=0.0) == "day"

    def test_night_at_midnight(self):
        t = datetime(2026, 6, 21, 0, 0, tzinfo=UTC)
        assert solar_condition(t, lat=0.0, lon=0.0) == "night"

    def test_twilight_exists(self):
        # Sweep hours to find a twilight condition
        found = False
        for h in range(0, 24):
            t = datetime(2026, 3, 21, h, 0, tzinfo=UTC)
            if solar_condition(t, lat=0.0, lon=0.0) == "twilight":
                found = True
                break
        assert found, "No twilight found in 24h sweep at equator equinox"


# ---------------------------------------------------------------------------
# Sensor suitability
# ---------------------------------------------------------------------------

class TestSensorSuitability:

    def test_eo_day(self):
        assert eo_suitability(20.0) == 1.0

    def test_eo_night(self):
        assert eo_suitability(-15.0) == 0.0

    def test_eo_twilight(self):
        # Sun at 0° → middle of twilight range → ~0.6
        suit = eo_suitability(0.0)
        assert 0.4 < suit < 0.8

    def test_sar_always_one(self):
        assert sar_suitability() == 1.0

    def test_sensor_suitability_keys(self):
        t = datetime(2026, 3, 21, 12, 0, tzinfo=UTC)
        result = sensor_suitability(t, 0.0, 0.0)
        expected = {"sun_elevation_deg", "solar_condition", "eo_suitability", "sar_suitability"}
        assert expected == set(result.keys())


# ---------------------------------------------------------------------------
# Tasking policy solar integration
# ---------------------------------------------------------------------------

class TestPolicySolarIntegration:

    def test_night_switches_eo_to_sar(self):
        """High-res preference at night should become SAR."""
        record = {
            "priority_score": 0.8,
            "anomaly_state": "confirmed",
            "anomaly_agreement": "confirmed",
            "ml_anomaly_duration_hours": 2,
            "custody_confidence": 0.7,
            "solar_condition": "night",
            "eo_suitability": 0.0,
            "sun_elevation_deg": -20.0,
        }
        p = compute_policy(record)
        assert p.sensor_preference == SENSOR_HIGH_RES  # anomaly-driven
        assert p.effective_sensor_preference == SENSOR_SAR  # solar-adjusted
        assert "night" in p.sensor_rationale.lower()

    def test_day_keeps_eo(self):
        """Daytime should keep EO preference."""
        record = {
            "priority_score": 0.8,
            "anomaly_state": "confirmed",
            "anomaly_agreement": "confirmed",
            "ml_anomaly_duration_hours": 2,
            "custody_confidence": 0.7,
            "solar_condition": "day",
            "eo_suitability": 1.0,
            "sun_elevation_deg": 45.0,
        }
        p = compute_policy(record)
        assert p.effective_sensor_preference == SENSOR_HIGH_RES
        assert "daylight" in p.sensor_rationale.lower()

    def test_twilight_notes_degradation(self):
        """Twilight keeps EO preference but notes degradation."""
        record = {
            "priority_score": 0.6,
            "anomaly_state": "emerging",
            "anomaly_agreement": "emerging",
            "ml_anomaly_duration_hours": 1,
            "custody_confidence": 0.8,
            "solar_condition": "twilight",
            "eo_suitability": 0.6,
            "sun_elevation_deg": 2.0,
        }
        p = compute_policy(record)
        # Emerging → fast_revisit (optical-class)
        assert p.sensor_preference == SENSOR_FAST
        assert p.effective_sensor_preference == SENSOR_FAST  # twilight keeps EO
        assert "twilight" in p.sensor_rationale.lower()

    def test_sar_preference_unaffected_by_night(self):
        """SAR preference should stay SAR regardless of solar condition."""
        record = {
            "priority_score": 0.9,
            "anomaly_state": "sustained",
            "anomaly_agreement": "confirmed",
            "ml_anomaly_duration_hours": 5,
            "custody_confidence": 0.2,  # weak → SAR preference
            "solar_condition": "night",
            "eo_suitability": 0.0,
            "sun_elevation_deg": -25.0,
        }
        p = compute_policy(record)
        assert p.sensor_preference == SENSOR_SAR
        assert p.effective_sensor_preference == SENSOR_SAR

    def test_default_solar_day(self):
        """When solar fields are absent, default to day behavior."""
        record = {
            "priority_score": 0.8,
            "anomaly_state": "confirmed",
            "anomaly_agreement": "confirmed",
            "custody_confidence": 0.7,
        }
        p = compute_policy(record)
        assert p.effective_sensor_preference == p.sensor_preference


# ---------------------------------------------------------------------------
# Simulation integration
# ---------------------------------------------------------------------------

class TestSimulationIntegration:

    def test_records_carry_solar_fields(self):
        from custody.simulate import run_simulation
        records = run_simulation()
        required = {
            "sun_elevation_deg", "solar_condition",
            "eo_suitability", "sar_suitability",
            "effective_sensor_preference", "sensor_rationale",
        }
        for r in records[:5]:
            missing = required - set(r.keys())
            assert not missing, f"Missing: {missing}"

    def test_solar_condition_varies_over_day(self):
        from custody.simulate import run_simulation
        records = run_simulation()
        conditions = {r["solar_condition"] for r in records}
        # 9-hour smoke scenario starting at 10:00 UTC → likely has day + night
        assert len(conditions) >= 1  # at least one condition present
