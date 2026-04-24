# Custody Decision Packet - Tennent Reef

## Current belief

- **construction_or_reclamation_activity** - 1.000
- **fixed_reclamation_or_structure** - 1.000
- no_meaningful_activity - 0.000

## Custody health

**AMBIGUOUS** (0.35) - AMBIGUOUS: 2 hypotheses saturated at >= 0.95

Drivers:
- 2 hypotheses saturated at >= 0.95
- top-two margin 0.00 below ambiguity threshold 0.15

### Primary ambiguity

- construction_or_reclamation_activity vs fixed_reclamation_or_structure

## Recommended candidate collects

1. **Opportunistic optical context** (0.85) - daylight optical best separates visible construction / dredging signatures from a static structure
2. **Cross-geometry SAR acquisition** (0.75) - geometry diversity reveals whether the returns change with look angle, consistent with active worksite vs completed structure
3. **Repeat SAR acquisition (same geometry)** (0.60) - same-geometry repeat tracks whether the footprint is evolving scene-to-scene

## Decision notes

- Additional VLM tuning has low expected value - current uncertainty is hypothesis-level, not detector-confidence-level.
- Sentinel-1/2 ingestion is roadmap, not part of this demo.
