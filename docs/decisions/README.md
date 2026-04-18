# Decision Log

Architectural and scoping decisions for Custody, recorded as ADRs (Architecture Decision Records).

**Convention:** MADR-lite. Each file has four sections — **Status**, **Context**, **Decision**, **Consequences** — plus a YAML frontmatter block with `id`, `title`, `status`, `date`, and (where relevant) `supersedes` / `superseded_by`.

**File naming:** `NNNN-kebab-case-title.md`, numbered sequentially starting at `0001`. Zero-padded to 4 digits.

**Immutability:** once merged, an ADR is never edited. Corrections, reversals, or refinements are recorded as a new ADR that references the prior one via `supersedes:` (new) and `superseded_by:` (old, the only allowed edit to a prior ADR).

**When to write one:** any decision that changes module boundaries, scope, vocabulary, or architectural claims. Bug fixes, refactors within a module, and test additions do not need ADRs. When in doubt, ask.

**Current ADRs:**
- [0001 — Rename v3 orchestration module to tipcue](0001-rename-v3-orchestration-to-tipcue.md)
- [0002 — Tipcue v1 consumes ACTIVE_CUSTODY only](0002-tipcue-v1-active-custody-only.md)
