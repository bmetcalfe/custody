"""
Pure data-shaping helper for the "Next Orbital Passes" dashboard section.

Contains no Streamlit calls.  Accepts plain Python values and returns a list
of display-ready dicts that streamlit_app.py renders as a dataframe.

Public API
----------
build_orbital_passes_rows : (lat, lon, timestamp) -> list[dict]
    Returns one row per orbital satellite (SAT-A, SAT-B) describing the
    next pass window from the given observer position and time.
"""
from datetime import datetime

import custody.config as config
from custody.sensors import next_pass_window

# Orbital satellites exposed in the panel — schedule-based sensors (A1, B1, C1)
# are intentionally excluded because they have no orbital geometry.
_ORBITAL_SATELLITES = ["SAT-A", "SAT-B"]

# Sort key constants (lower = more actionable)
_SORT_IN_VIEW = 0
_SORT_UPCOMING = 1
_SORT_NO_PASS = 2


def build_orbital_passes_rows(
    lat: float,
    lon: float,
    timestamp: datetime,
    horizon_minutes: int = 180,
) -> list[dict]:
    """Return one row per orbital satellite describing the next pass window.

    Rows are sorted so the most actionable pass appears first:
      1. In View Now  (time_to_start_seconds == 0)
      2. Upcoming     (ascending time to start)
      3. No Pass In Horizon

    Args:
        lat:              Observer latitude in degrees (+N).
        lon:              Observer longitude in degrees (+E).
        timestamp:        Reference time (typically the current playback step).
        horizon_minutes:  How far ahead to search. Default 180 min (3 h).

    Returns:
        List of dicts — one entry per satellite in _ORBITAL_SATELLITES — with keys:

            Satellite           – satellite identifier ("SAT-A" or "SAT-B")
            Start               – datetime of pass start, or None if no pass found
            End                 – datetime of pass end, or None if no pass found
            Duration (min)      – pass duration in minutes (float), or None
            Time to Start (min) – minutes until pass start; 0.0 if already in view;
                                  None if no pass found
            Status              – "In View Now", "Upcoming", or "No Pass In Horizon"
            Within Threshold    – True when 0 < time_to_start_seconds
                                  <= HOLD_LOOKAHEAD_THRESHOLD_SECONDS (i.e. the pass
                                  is upcoming and close enough to trigger a lookahead
                                  HOLD in the planner); False otherwise
    """
    rows = []
    for sat_id in _ORBITAL_SATELLITES:
        pw = next_pass_window(sat_id, lat, lon, timestamp, horizon_minutes=horizon_minutes)
        if pw is None:
            rows.append({
                "Satellite": sat_id,
                "Start": None,
                "End": None,
                "Duration (min)": None,
                "Time to Start (min)": None,
                "Status": "No Pass In Horizon",
                "Within Threshold": False,
                "_sort_key": (_SORT_NO_PASS, float("inf")),
            })
        else:
            tts = pw.time_to_start_seconds
            tts_min = round(tts / 60.0, 1)
            within = 0 < tts <= config.HOLD_LOOKAHEAD_THRESHOLD_SECONDS
            if tts == 0:
                status = "In View Now"
                sort_key = (_SORT_IN_VIEW, 0.0)
            else:
                status = "Upcoming"
                sort_key = (_SORT_UPCOMING, tts)
            rows.append({
                "Satellite": sat_id,
                "Start": pw.start_time,
                "End": pw.end_time,
                "Duration (min)": round(pw.duration_seconds / 60.0, 1),
                "Time to Start (min)": tts_min,
                "Status": status,
                "Within Threshold": within,
                "_sort_key": sort_key,
            })

    rows.sort(key=lambda r: r["_sort_key"])
    for r in rows:
        del r["_sort_key"]
    return rows
