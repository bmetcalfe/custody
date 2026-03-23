import math
from datetime import timedelta

from custody.models import HistoryEntry


def update_position(vessel, hours):
    import math
    from datetime import timedelta

    # ✅ Save position, timestamp, speed, and heading BEFORE moving
    vessel.history.append(
        HistoryEntry(vessel.lat, vessel.lon, vessel.last_seen, vessel.speed_kmh, vessel.heading_deg)
    )

    heading_rad = math.radians(vessel.heading_deg)

    # Compass convention: 0° = north, 90° = east, clockwise.
    # North component = cos(heading), East component = sin(heading).
    vessel.lat += math.cos(heading_rad) * vessel.speed_kmh * hours * 0.01
    vessel.lon += math.sin(heading_rad) * vessel.speed_kmh * hours * 0.01

    vessel.last_seen += timedelta(hours=hours)

    return vessel


def update_uncertainty(current_uncertainty, hours):
    growth_rate = 3.0  # km per hour
    return current_uncertainty + growth_rate * hours


def custody_confidence(uncertainty):
    import math
    return math.exp(-uncertainty / 50)