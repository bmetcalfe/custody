"""SAR / EO detection modules for the v3 Custody pipeline.

- :mod:`custody.detection.sar_common` — GeoTIFF I/O and pixel↔lat/lon helpers,
  plus the bridge from pixel-space detections to :class:`PositionObservation`.
- :mod:`custody.detection.sar_cfar` — CA-CFAR point detection with optional
  structure-mask filtering, tuned for Umbra GEC imagery.
"""

from custody.detection.sar_cfar import (
    compute_cfar_threshold,
    compute_structure_mask,
    detect_points_cfar,
    detect_points_to_observations,
)
from custody.detection.sar_common import (
    detection_to_observation,
    detections_to_observations,
    latlon_to_pixel,
    pixel_to_latlon,
    read_geotiff,
)

__all__ = [
    "compute_cfar_threshold",
    "compute_structure_mask",
    "detect_points_cfar",
    "detect_points_to_observations",
    "detection_to_observation",
    "detections_to_observations",
    "latlon_to_pixel",
    "pixel_to_latlon",
    "read_geotiff",
]
