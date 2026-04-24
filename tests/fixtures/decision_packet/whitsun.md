# Custody Decision Packet - Whitsun Reef

## Current belief

- **ais_dark_or_poorly_observed_vessels** - 1.000
- **transient_anchorage_or_fishing_presence** - 1.000
- **vessel_cluster_activity** - 1.000

## Custody health

**AMBIGUOUS** (0.35) - AMBIGUOUS: 3 hypotheses saturated at >= 0.95

Drivers:
- 3 hypotheses saturated at >= 0.95
- top-two margin 0.00 below ambiguity threshold 0.15

### Primary ambiguity

- ais_dark_or_poorly_observed_vessels vs transient_anchorage_or_fishing_presence
- ais_dark_or_poorly_observed_vessels vs vessel_cluster_activity
- transient_anchorage_or_fishing_presence vs vessel_cluster_activity

## Recommended candidate collects

1. **AIS coverage and message query** (0.75) - AIS coverage and message query distinguishes truly dark vessels from ordinary fishing traffic that simply was not captured
2. **Opportunistic optical context** (0.45) - daylight optical provides posture context but does not confirm AIS cooperativity
3. **Repeat SAR acquisition (same geometry)** (0.45) - persistence can help characterise the population but does not answer cooperativity

## Decision notes

- Additional VLM tuning has low expected value - current uncertainty is hypothesis-level, not detector-confidence-level.
- AIS absence alone is not proof of dark vessel activity; only meaningful when coverage is known.
- Sentinel-1/2 ingestion is roadmap, not part of this demo.
