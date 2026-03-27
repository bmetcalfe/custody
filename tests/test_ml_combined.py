"""
Tests for combined ML signal mode.

Covers:
  1. ml_combined_score: max of absolute and relative
  2. ml_combined_score: falls back to absolute when relative absent
  3. ml_combined_score: handles NaN relative gracefully
  4. absolute mode: unchanged behavior
  5. relative mode: uses relative score
  6. combined mode: detects global anomaly (high absolute, low relative)
  7. combined mode: detects vessel anomaly (low absolute, high relative)
  8. combined mode: detects when both high
  9. config toggle: switching modes changes _is_ml_high result
 10. enrich_record includes ml_anomaly_combined field
 11. persistence uses combined signal in combined mode
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

import numpy as np
import pytest

import custody.config as config
from custody.reasoning import (
    AnomalyAgreement,
    _is_ml_high,
    classify_agreement,
    compute_persistence,
    enrich_record,
    ml_combined_score,
)

UTC = timezone.utc
_T0 = datetime(2026, 3, 23, 10, 0, tzinfo=UTC)


def _rec(ml=0.0, rel=None, anom=0.0, hour=0):
    r = {
        "ml_anomaly_score": ml,
        "anomaly_score": anom,
        "custody_confidence": 0.8,
        "time": _T0 + timedelta(hours=hour),
    }
    if rel is not None:
        r["ml_anomaly_relative"] = rel
    return r


class _ConfigGuard:
    """Context manager to save/restore ML signal config."""
    def __init__(self, mode="absolute", use_rel=False):
        self.mode = mode
        self.use_rel = use_rel

    def __enter__(self):
        self._old_mode = config.ML_SIGNAL_MODE
        self._old_use_rel = config.USE_RELATIVE_ML_THRESHOLD
        config.ML_SIGNAL_MODE = self.mode
        config.USE_RELATIVE_ML_THRESHOLD = self.use_rel
        return self

    def __exit__(self, *args):
        config.ML_SIGNAL_MODE = self._old_mode
        config.USE_RELATIVE_ML_THRESHOLD = self._old_use_rel


# ---------------------------------------------------------------------------
# ml_combined_score
# ---------------------------------------------------------------------------

class TestCombinedScore:

    def test_max_of_both(self):
        assert ml_combined_score(_rec(ml=0.3, rel=0.9)) == 0.9
        assert ml_combined_score(_rec(ml=0.9, rel=0.3)) == 0.9

    def test_falls_back_to_absolute(self):
        assert ml_combined_score(_rec(ml=0.7)) == 0.7

    def test_handles_nan_relative(self):
        r = _rec(ml=0.5)
        r["ml_anomaly_relative"] = float("nan")
        assert ml_combined_score(r) == 0.5


# ---------------------------------------------------------------------------
# _is_ml_high by mode
# ---------------------------------------------------------------------------

class TestIsMlHigh:

    def test_absolute_mode(self):
        with _ConfigGuard("absolute"):
            assert _is_ml_high(_rec(ml=0.9)) is True
            assert _is_ml_high(_rec(ml=0.5)) is False

    def test_relative_mode(self):
        with _ConfigGuard("relative"):
            # High relative, low absolute → high
            assert _is_ml_high(_rec(ml=0.3, rel=0.97)) is True
            # Low relative → not high
            assert _is_ml_high(_rec(ml=0.3, rel=0.5)) is False

    def test_relative_fallback_when_absent(self):
        with _ConfigGuard("relative"):
            # No relative → falls back to absolute
            assert _is_ml_high(_rec(ml=0.9)) is True

    def test_combined_detects_global_anomaly(self):
        with _ConfigGuard("combined"):
            # High absolute, low relative → combined = max = 0.9 >= 0.8
            assert _is_ml_high(_rec(ml=0.9, rel=0.3)) is True

    def test_combined_detects_vessel_anomaly(self):
        with _ConfigGuard("combined"):
            # Low absolute, high relative → combined = max = 0.97 >= 0.8
            assert _is_ml_high(_rec(ml=0.3, rel=0.97)) is True

    def test_combined_both_low(self):
        with _ConfigGuard("combined"):
            assert _is_ml_high(_rec(ml=0.3, rel=0.5)) is False

    def test_combined_both_high(self):
        with _ConfigGuard("combined"):
            assert _is_ml_high(_rec(ml=0.9, rel=0.97)) is True


# ---------------------------------------------------------------------------
# Agreement with combined mode
# ---------------------------------------------------------------------------

class TestAgreementCombined:

    def test_global_anomaly_detected(self):
        """High absolute, low relative, low heuristic → EMERGING."""
        with _ConfigGuard("combined"):
            r = _rec(ml=0.9, rel=0.3, anom=0.2)
            assert classify_agreement(r) == AnomalyAgreement.EMERGING

    def test_vessel_anomaly_detected(self):
        """Low absolute, high relative, low heuristic → EMERGING."""
        with _ConfigGuard("combined"):
            r = _rec(ml=0.3, rel=0.97, anom=0.2)
            assert classify_agreement(r) == AnomalyAgreement.EMERGING

    def test_both_sources_confirm(self):
        """High combined + high heuristic → CONFIRMED."""
        with _ConfigGuard("combined"):
            r = _rec(ml=0.9, rel=0.97, anom=1.5)
            assert classify_agreement(r) == AnomalyAgreement.CONFIRMED

    def test_neither_high(self):
        with _ConfigGuard("combined"):
            r = _rec(ml=0.3, rel=0.5, anom=0.2)
            assert classify_agreement(r) == AnomalyAgreement.NORMAL


# ---------------------------------------------------------------------------
# Persistence with combined mode
# ---------------------------------------------------------------------------

class TestPersistenceCombined:

    def test_tracks_combined_streak(self):
        with _ConfigGuard("combined"):
            history = [_rec(ml=0.3, rel=0.97, hour=i) for i in range(3)]
            current = _rec(ml=0.3, rel=0.97, hour=3)
            p = compute_persistence(current, history)
            assert p["ml_anomaly_duration_hours"] == 4


# ---------------------------------------------------------------------------
# enrich_record
# ---------------------------------------------------------------------------

class TestEnrichCombined:

    def test_includes_combined_field(self):
        r = _rec(ml=0.5, rel=0.9)
        enriched = enrich_record(r, [])
        assert "ml_anomaly_combined" in enriched
        assert enriched["ml_anomaly_combined"] == 0.9

    def test_combined_without_relative(self):
        r = _rec(ml=0.7)
        enriched = enrich_record(r, [])
        assert enriched["ml_anomaly_combined"] == 0.7
