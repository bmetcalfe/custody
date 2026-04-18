# Decision Log

Architectural and scoping decisions for Custody, recorded as ADRs (Architecture Decision Records).

**Convention:** MADR-lite. Each file has four sections — **Status**, **Context**, **Decision**, **Consequences** — plus a YAML frontmatter block with `id`, `title`, `status`, `date`, and (where relevant) `supersedes` / `superseded_by`.

**File naming:** `NNNN-kebab-case-title.md`, numbered sequentially starting at `0001`. Zero-padded to 4 digits.

**Immutability:** once merged, an ADR is never edited. Corrections, reversals, or refinements are recorded as a new ADR that references the prior one via `supersedes:` (new) and `superseded_by:` (old, the only allowed edit to a prior ADR).

**When to write one:** any decision that changes module boundaries, scope, vocabulary, or architectural claims. Bug fixes, refactors within a module, and test additions do not need ADRs. When in doubt, ask.

**Current ADRs:**
- [0001 — Rename v3 orchestration module to tipcue](0001-rename-v3-orchestration-to-tipcue.md)
- [0002 — Tipcue v1 consumes ACTIVE_CUSTODY only](0002-tipcue-v1-active-custody-only.md)
- [0003 — Rename fusion.py to belief_assessment.py to free the fusion/ namespace](0003-rename-fusion-to-belief-assessment.md)
- [0004 — Rename ObservationState to CollectionIntent; reserve Observation for raw detections](0004-rename-observationstate-to-collectionintent.md)
- [0005 — Replace scalar-radius track model with EKF; retain uncertainty_km and add position_sigma_km as shim properties](0005-ekf-tracks-with-scalar-radius-shim.md)
- [0006 — TrackState converted from @dataclass to a plain class](0006-trackstate-plain-class.md)
