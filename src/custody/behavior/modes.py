import random


def apply_behavior_mode(vessel, mode):
    """
    Update vessel speed/heading based on a simple behavior mode.
    This keeps the simulation believable without becoming complicated.
    """

    if mode == "transit":
        vessel.speed_kmh = 28
        vessel.heading_deg = 45

    elif mode == "approach":
        vessel.speed_kmh = 20
        vessel.heading_deg = 20  # NNE (compass): closes northward into the zone

    elif mode == "loiter":
        vessel.speed_kmh = 2
        vessel.heading_deg += random.choice([-20, -10, 0, 10, 20])

    elif mode == "egress":
        vessel.speed_kmh = 26
        vessel.heading_deg = 120

    elif mode == "idle":
        vessel.speed_kmh = 0

    # keep heading in a clean range
    vessel.heading_deg = vessel.heading_deg % 360

    return vessel


def get_behavior_mode(hour_index, schedule):
    """
    schedule is a dict mapping time ranges to mode:
    [
        ((0, 3), "transit"),
        ((4, 5), "approach"),
        ((6, 8), "loiter"),
        ((9, 12), "egress"),
    ]
    """
    for (start, end), mode in schedule:
        if start <= hour_index <= end:
            return mode
    return "transit"
