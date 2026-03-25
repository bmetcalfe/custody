"""
Prediction layer for Custody GEOINT.

Provides dead-reckoning trajectory projection, zone-crossing probability
estimation, and forward anomaly forecasting.

Public API
----------
Prediction       — frozen dataclass: per-entity advisory prediction.
predict_entity   — main entry point; composes trajectory + zone + risk modules.
"""
from custody.prediction.prediction import Prediction, predict_entity

__all__ = ["Prediction", "predict_entity"]
