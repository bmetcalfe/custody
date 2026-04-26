"""External-source ingestion and preprocessing.

Re-exports the v3 GFW presence parser (see
:mod:`custody.ingest.gfw_presence`) and the Sentinel STAC observation
ingestion layer (see :mod:`custody.ingest.sentinel`).  Legacy NOAA-AIS
preprocessing lives in :mod:`custody.ingest.noaa_preprocess`.
"""

from custody.ingest.gfw_presence import (
    DropReport,
    GFW_PRESENCE_POS_SIGMA_M,
    parse_gfw_presence_response,
)
from custody.ingest.sentinel import (
    CDSE_STAC_BASE_URL,
    CDSE_STAC_SEARCH_URL,
    CONFIDENCE_WEIGHTS,
    DEFAULT_COLLECTIONS,
    ObservationArtifact,
    SENTINEL_1_GRD_COLLECTION,
    SENTINEL_2_L2A_COLLECTION,
    best_low_cloud_sentinel_2,
    date_range,
    latest_sentinel_1,
    load_observation_cache,
    normalize_stac_item,
    search_sentinel_observations,
    write_observation_cache,
)

__all__ = [
    "CDSE_STAC_BASE_URL",
    "CDSE_STAC_SEARCH_URL",
    "CONFIDENCE_WEIGHTS",
    "DEFAULT_COLLECTIONS",
    "DropReport",
    "GFW_PRESENCE_POS_SIGMA_M",
    "ObservationArtifact",
    "SENTINEL_1_GRD_COLLECTION",
    "SENTINEL_2_L2A_COLLECTION",
    "best_low_cloud_sentinel_2",
    "date_range",
    "latest_sentinel_1",
    "load_observation_cache",
    "normalize_stac_item",
    "parse_gfw_presence_response",
    "search_sentinel_observations",
    "write_observation_cache",
]
