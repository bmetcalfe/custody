---
id: 0012
title: Scenario reframe — primary narrative is Vietnamese reclamation at Tennent Reef; Whitsun becomes supplementary
date: 2026-04-18
status: accepted
---

## Context

The original scenario specified in `docs/scenario.md` identified the primary Umbra-imaged feature as "Cuarteron-area target at 114.665°E / 8.856°N" and framed the demo narrative around Chinese maritime militia persistence following CSIS AMTI methodology.

Week 2 reconnaissance of the actual Umbra scene preview revealed the coordinates do not correspond to Cuarteron Reef. Cuarteron Reef is at approximately 8.86°N / 112.85°E — the target coordinates share latitude but are ~180 km east in longitude. The feature at 8.856°N / 114.665°E is **Tennent Reef** (Vietnamese: Đá Tiên Nữ; Filipino: Bahura ng Lopez-Jaena; Chinese: 无乜礁), a Vietnamese-controlled triangular atoll under active land reclamation since December 2021.

The five Umbra scenes over this feature (2023-07-02, 07-23, 08-07, 08-09, 08-13) image the eastern artificial island (Tiên Nữ B), then approximately 55 hectares and undergoing continuous expansion. AMTI's December 2022 and November 2023 reports explicitly track Tennent Reef as one of Vietnam's most significantly developed outposts, documenting the feature's growth by approximately 62 acres between 2022 and late 2023 — a period that includes our demo window.

The secondary feature at 9.969°N / 114.632°E overlaps the southwestern arm of **Whitsun Reef** (Vietnamese: Đá Ba Đầu; Filipino: Julian Felipe Reef; Chinese: 牛軛礁). Whitsun is the site of the canonical March 2021 Chinese maritime militia swarm event (~220 vessels), extensively documented by AMTI and widely reported internationally.

The original scenario framing was factually incorrect on three dimensions: geographic name (Tennent, not Cuarteron), claimant (Vietnam, not China), and activity type (land reclamation, not vessel loitering). This ADR corrects the scenario.

## Decision

The primary demo narrative is reframed from "AIS-dark Chinese militia trawler persistence" to **"AIS-dark reclamation activity monitoring at a contested Spratly feature."** The specific subject is active Vietnamese dredging and land reclamation at Tennent Reef's eastern island (Tiên Nữ B) during the summer 2023 window, against a CSIS AMTI ground-truth reference.

Whitsun Reef (3 scenes, December 2023 – March 2024) becomes a **supplementary case study** documented in the repository but not included in the 90-second demo cut. Whitsun's narrative — Chinese militia swarm at a contested reef, per AMTI's March 2021 coverage and subsequent reporting — is a distinct scenario and is presented as such, not as ancestry to Tennent.

Both narratives are retained. Neither is demoted or hidden. The demo video uses Tennent as the hero because the five-scene 41-day cadence supports the tip-and-cue narrative arc (Act 3's 48-hour cued revisit requires dense temporal coverage that only Tennent provides in our inventory). Whitsun's three scenes over three months support a longer-baseline change-detection case study rather than a tight-cadence cueing demonstration.

The AMTI methodology grounding is preserved but reframed: instead of citing AMTI's militia-focused reporting, the Tennent narrative cites AMTI's reclamation-focused reporting. Both are real and both are the same publication — AMTI tracks claimant activity across the Spratlys for all six claimants including Vietnam. AMTI has specifically documented Tennent Reef's expansion in their December 2022 and November 2023 reports.

## Consequences

- `docs/scenario.md` rewritten to identify Tennent Reef as the primary feature, document its Vietnamese control and active reclamation, and name Whitsun Reef's militia-swarm history as the supplementary case study context. The file's structure is preserved; only content identifying features, claimants, and narrative arcs changes.
- `docs/positioning.md` rewritten so the "what this is" and "demo scenario" sections name reclamation monitoring rather than militia detection. The SDA Custody Layer capability alignment is unchanged — the architectural story is agnostic to threat domain, which was the original claim and remains true.
- `README.md` updated correspondingly: migration banner's scenario description, the "Demo scenario" section, and the brief reference to AMTI methodology.
- `CLAUDE.md` updated to reflect the scenario's geographic and narrative reality. CC's pre-task checklist line about "AIS-dark vessel activity consistent with AMTI methodology" is reworded to "AIS-dark reclamation and construction activity consistent with AMTI reporting at this feature."
- **No code changes.** The fusion pipeline, EKF, observation types, spatial index, and tracker are all agnostic to scenario semantics. AIS data remains AIS data; SAR detections remain SAR detections; the architecture does not know or care whether a persistent bright return is a vessel or a dredger. ADR-0008's polymorphic observation types, ADR-0011's GFW presence ingestion, and all five Week 1 fusion modules continue working unchanged.
- Demo acts are re-described. Act 2's persistent target becomes "reclamation footprint growing over successive scenes" rather than "vessel loitering." Act 3's cued revisit demonstrates the same tip-and-cue architecture against a different target type. The covariance-aware planner and reasoning trace story is unchanged in substance.
- The voiceover script (not yet written) will be drafted against the corrected scenario. Week 9's voiceover work is unaffected in scope; only the subject-matter language changes.
- The Whitsun supplementary case study gets a dedicated section in `docs/scenario.md` and in the repository's eventual case-study documentation. Three-scene change detection over 3 months is a coherent case of its own and is presented as such.
- **This ADR documents a real Day 0 scoping error.** The original `scenario.md` was written based on coordinate pattern-matching without geographic verification. The correct coordinates were always in the Umbra metadata; they were never checked against a map during Day 0. This is a documented failure of the planning process and is captured here to prevent similar errors in future scenario work.

## Supersedes

- Partially supersedes the original `docs/scenario.md` (created 2026-04-18, committed 811c38e). That file is rewritten by this ADR's implementation, not retired — the structure and the Umbra scene inventory tables are preserved; only feature identification and narrative framing change.
- Partially supersedes the original `docs/positioning.md` demo-scenario section. The "what this is not" honest-scoping section in positioning.md is unchanged and if anything is strengthened by the reframe.
