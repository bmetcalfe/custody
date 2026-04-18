"""AIS data ingestion and preprocessing, plus v3 external-source parsers.

The v3 GFW presence parser (see :mod:`custody.ingest.gfw_presence`) is
re-exported here for convenience.  Legacy NOAA-AIS preprocessing lives in
:mod:`custody.ingest.noaa_preprocess`.
"""

from custody.ingest.gfw_presence import (
    DropReport,
    GFW_PRESENCE_POS_SIGMA_M,
    parse_gfw_presence_response,
)

__all__ = [
    "DropReport",
    "GFW_PRESENCE_POS_SIGMA_M",
    "parse_gfw_presence_response",
]
