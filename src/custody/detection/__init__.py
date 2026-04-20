"""SAR / EO detection modules for the v3 Custody pipeline.

- :mod:`custody.detection.sar_common` — GeoTIFF I/O and pixel↔lat/lon helpers,
  plus the bridge from pixel-space detections to :class:`PositionObservation`.
- :mod:`custody.detection.sar_cfar` — CA-CFAR point detection with optional
  structure-mask filtering, tuned for Umbra GEC imagery.
"""

from custody.detection.sar_cfar import (
    compute_cfar_threshold,
    compute_os_cfar_threshold,
    compute_structure_mask,
    detect_points_cfar,
    detect_points_os_cfar,
    detect_points_os_cfar_to_observations,
    detect_points_to_observations,
)
from custody.detection.sar_common import (
    crop_to_aoi,
    detection_to_observation,
    detections_to_observations,
    latlon_to_pixel,
    pixel_to_latlon,
    read_geotiff,
)
from custody.detection.vlm_sar import (
    PROMPTS,
    detect_vessels_in_scene,
    detect_vessels_in_tile,
    vlm_detection_to_observation,
)

__all__ = [
    "PROMPTS",
    "compute_cfar_threshold",
    "compute_os_cfar_threshold",
    "compute_structure_mask",
    "crop_to_aoi",
    "detect_points_cfar",
    "detect_points_os_cfar",
    "detect_points_os_cfar_to_observations",
    "detect_points_to_observations",
    "detect_vessels_in_scene",
    "detect_vessels_in_tile",
    "detection_to_observation",
    "detections_to_observations",
    "latlon_to_pixel",
    "pixel_to_latlon",
    "read_geotiff",
    "vlm_detection_to_observation",
]
