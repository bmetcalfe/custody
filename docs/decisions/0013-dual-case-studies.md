---
id: 0013
title: Dual case studies — Tennent Reef and Whitsun Reef both primary, neither supplementary
date: 2026-04-18
status: accepted
supersedes: partial — adjusts ADR-0012's "primary / supplementary" hierarchy
---

## Context

ADR-0012 established Tennent Reef as the primary demo feature (Vietnamese land reclamation, 5 scenes summer 2023) with Whitsun Reef as a supplementary case study (Chinese militia activity, 3 scenes winter 2023-24). The distinction was made on the basis of scene count and cadence support for the tip-and-cue narrative arc.

Week 2 Day 3 reconnaissance against the actual Whitsun Umbra imagery revealed that the 2023-12-06 scene contains 30+ distinct vessel-like returns — small compact bright targets with characteristic SAR azimuth-smearing artifacts, clustered in groups across open water. This is the canonical Chinese-maritime-militia-swarm visual signature, imaged approximately 2.75 years after the widely-reported March 2021 event that made Whitsun internationally notable.

Two observations from this:

1. The Whitsun scenes contain clean vessel-detection narratives that the Tennent scenes do not. Tennent scenes are dominated by a single large reclamation structure; vessel targets are harder to isolate. Whitsun scenes are open water with discrete point targets — exactly the scenario textbook SAR vessel detection is designed for.

2. Treating Whitsun as "supplementary" understates both its narrative value and its technical value. The demo gains significantly by showing the architecture handling two genuinely different scenario types: persistent structure monitoring (Tennent) and point-target vessel detection in a flotilla (Whitsun).

The "primary / supplementary" hierarchy established in ADR-0012 was a reasonable call given the information available at that time, but the reconnaissance since then indicates a different framing would better serve the project.

## Decision

The demo is reframed around **two co-equal case studies**, neither designated as primary:

**Case Study A — Tennent Reef:** Vietnamese land reclamation at Đá Tiên Nữ. 5 Umbra scenes, June-August 2023. AMTI-referenced ground truth for ongoing Vietnamese reclamation. Demonstrates: persistent AIS-dark structure monitoring, multi-scene change detection over 41 days, tip-and-cue orchestration with 48-hour revisit cadence.

**Case Study B — Whitsun Reef:** Chinese maritime militia activity at Đá Ba Đầu / Julian Felipe Reef. 3 Umbra scenes, December 2023 – March 2024. AMTI-referenced ground truth for the March 2021 swarm incident. Demonstrates: multi-vessel AIS-dark point-target detection, vessel flotilla tracking, cross-INT fusion where SAR detections exist without corresponding AIS broadcasts.

Both case studies appear in the public demo video and in README-level documentation. Neither is treated as supplementary. The positioning is that the architecture works on both scenario types, demonstrating flexibility rather than specialization.

The 90-second demo video cut covers both cases. Structure: approximately 45 seconds per case study, with Tennent first (establishes architecture, ends with the cued-revisit moment) and Whitsun second (demonstrates vessel-cluster detection and the AIS-dark militia scenario). Opening and closing framing describes the architecture's scenario-agnostic design.

## Consequences

- `docs/scenario.md` updated to present Tennent and Whitsun as co-equal case studies, each with its own section. The feature-identification table, Umbra inventory, and scene timing sections remain accurate but the narrative framing changes.
- `docs/positioning.md` "demo scenario" section rewritten to present both cases. The "what this is not" section's "not a political statement" bullet is strengthened — two cases involving two different claimants (Vietnam, China) make the dual-claimant neutrality explicit rather than implicit.
- `README.md` demo scenario section rewritten to present both cases in parallel.
- `CLAUDE.md` updated so CC's scenario quick-reference describes two cases, not one primary plus one supplementary.
- The 90-second demo video narrative structure changes from three acts about one target to a two-case structure. Act layout within each case is preserved — the Tennent tip-and-cue moment still works, the Whitsun flotilla-detection moment becomes the second case's hero shot.
- Stream B detection work produces observations for both scenes — Tennent scenes 1 and 2 today/tomorrow, Whitsun scene 1 today as a parallel track.
- No code changes. Architecture remains scenario-agnostic.
- ADR-0012 is partially superseded — the scenario identification (Tennent = Vietnamese reclamation, Whitsun = Chinese militia site) remains correct. What changes is the hierarchy between them.

## Framing language going forward

The project's public positioning uses **"case study"** rather than **"primary / supplementary,"** **"hero / supplementary,"** or **"main scenario / side scenario."** Both cases are real, both are demonstrated, both appear in the public video. The architecture is demonstrated as flexible across scenario types.

When describing the demo in external communication (LinkedIn post, interview conversations, documentation), the framing is: *"The Custody architecture handles both structure-monitoring and vessel-detection scenarios. Case A demonstrates persistent reclamation monitoring at a Vietnamese-controlled feature; Case B demonstrates vessel flotilla detection at a Chinese militia hub. The same pipeline — same EKF, same observation types, same tipcue layer — processes both without scenario-specific modification."*
