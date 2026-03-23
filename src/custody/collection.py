import random


def attempt_collection(opportunity, anomaly_breakdown=None):
    success_prob = opportunity.success_prob

    if anomaly_breakdown:
        # High-res sensor slightly better when characterizing loitering
        if (
            opportunity.sensor_type == "high_resolution"
            and anomaly_breakdown["loitering"].score > 0
        ):
            success_prob += 0.05

        # All-weather sensor more resilient in elevated context
        if (
            opportunity.sensor_type == "all_weather"
            and anomaly_breakdown["sensitive_zone"].score > 0
        ):
            success_prob += 0.03

    success_prob = min(success_prob, 0.95)
    return random.random() < success_prob


def apply_collection_effect(current_uncertainty, opportunity):
    if opportunity.sensor_type == "fast_revisit":
        # Quick reacquisition, moderate certainty improvement
        return max(6.0, current_uncertainty * 0.5)

    if opportunity.sensor_type == "high_resolution":
        # Best for collapsing uncertainty
        return max(4.0, current_uncertainty * 0.35)

    if opportunity.sensor_type == "all_weather":
        # Reliable but not as strong as high-res
        return max(5.0, current_uncertainty * 0.45)

    return current_uncertainty