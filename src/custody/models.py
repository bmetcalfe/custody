from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, NamedTuple, Optional


class BehaviorState(Enum):
    """Inferred operational state of a vessel."""
    TRANSIT = "transit"
    APPROACH = "approach"
    LOITER = "loiter"
    EGRESS = "egress"
    IDLE = "idle"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Zone:
    """A named geographic bounding box used for zone-based detection.

    Attributes:
        name:    Stable identifier (e.g. "ZONE_ALPHA").  Used in metadata,
                 alert context, and display labels.
        min_lat: Southern boundary (degrees).
        max_lat: Northern boundary (degrees).
        min_lon: Western boundary (degrees).
        max_lon: Eastern boundary (degrees).
        halo:    Distance in degrees outside the boundary that earns a
                 partial anomaly score.  ~0.1° ≈ 11 km at equatorial latitudes.
    """
    name: str
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float
    halo: float


@dataclass(frozen=True)
class DetectorResult:
    """Output of a single anomaly detector.

    Attributes:
        name:     Detector identifier (e.g. "sensitive_zone").
        score:    Numeric contribution to the composite anomaly score.
        evidence: Human-readable explanation of why this score was assigned.
        metadata: Optional structured data for future extension (reasoning
                  chain, sub-scores, thresholds used, etc.). Defaults to
                  empty dict; callers should treat it as read-only.
    """
    name: str
    score: float
    evidence: str
    metadata: dict[str, Any] = field(default_factory=dict)


class HistoryEntry(NamedTuple):
    lat: float
    lon: float
    timestamp: datetime
    speed_kmh: float
    heading_deg: float


@dataclass
class Vessel:
    id: str
    lat: float
    lon: float
    speed_kmh: float
    heading_deg: float
    last_seen: datetime
    history: list[HistoryEntry] = field(default_factory=list)


@dataclass
class TrackState:
    """Per-track custody metadata that evolves independently of kinematics.

    Owns uncertainty, collection timing, and the last-known anomaly score at
    collection time.  Policy decisions (e.g. "recent = < 2 hours") belong in
    the caller, not here.

    Dark-vessel fields (all None/False when AIS is active):
        is_dark            — True once AIS dropout has been detected.
        dark_since         — Timestamp of the first dark step.
        last_known_lat/lon — Position frozen at the first dark step.
        last_known_time    — Timestamp frozen at the first dark step.
        last_known_anomaly — Anomaly score frozen at the first dark step.
    """
    uncertainty_km: float = 5.0
    last_collection_time: Optional[datetime] = None
    last_collection_anomaly_score: float = 0.0

    # Dark-vessel state (AIS dropout)
    is_dark: bool = False
    dark_since: Optional[datetime] = None
    last_known_lat: Optional[float] = None
    last_known_lon: Optional[float] = None
    last_known_time: Optional[datetime] = None
    last_known_anomaly: float = 0.0

    def hours_since_collection(self, now: datetime) -> Optional[float]:
        """Return hours elapsed since the last successful collection, or None."""
        if self.last_collection_time is None:
            return None
        return (now - self.last_collection_time).total_seconds() / 3600

    def record_collection(
        self,
        now: datetime,
        anomaly_score: float,
        new_uncertainty: float,
    ) -> None:
        """Update track state after a successful collection event."""
        self.uncertainty_km = new_uncertainty
        self.last_collection_time = now
        self.last_collection_anomaly_score = anomaly_score

    @property
    def confidence(self) -> float:
        """Current custody confidence derived from uncertainty."""
        from custody.tracks import custody_confidence  # deferred to avoid circular import
        return custody_confidence(self.uncertainty_km)