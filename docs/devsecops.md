# Custody — DevSecOps Posture

*Companion to [docs/security.md](security.md). This file documents the local
development workflow, the CI expectations, dependency hygiene, and what
makes the prototype auditable.*

---

## Local development

### Prerequisites

- Python 3.12 (pinned via `.python-version`)
- [`uv`](https://docs.astral.sh/uv/) for environment and dependency
  management
- `git` (provenance manifests record `git rev-parse HEAD`)

### Bootstrap

```bash
uv sync
```

`uv sync` installs the runtime and dev dependencies declared in
`pyproject.toml` and pins them via `uv.lock`. The lock file is checked
into the repository so every contributor and CI runner resolves the same
dependency graph.

### Running the test suite

```bash
uv run pytest -q
```

The test suite is the source of truth for expected behaviour. Every slice
of [ADR-0021](decisions/0021-custody-as-uncertainty-to-tasking-engine.md)
ships with tests that land alongside (or before) the implementation.

### Running the CLIs

Each script is a self-contained CLI; none start a network listener:

```bash
uv run python scripts/13_decision_packet.py --scenario tennent
uv run python scripts/22_provenance_manifest.py --output-kind decision-packet \
    --scenario tennent --format json
```

---

## CI expectations

The repository runs a single GitHub Actions workflow,
[`.github/workflows/tests.yml`](../.github/workflows/tests.yml), on every
push to `main` and on every pull request. The workflow:

1. Checks out the repository.
2. Installs `uv`.
3. Runs `uv sync` to materialize the locked dependency graph.
4. Runs `uv run pytest -q`.

Any test failure blocks merge. There is no separate lint job today; a
formatter / linter pass is roadmap, not present.

---

## Dependency hygiene

- **No new runtime dependencies for hypothesis, API, or provenance
  layers.** Slices 1–18 of ADR-0021 are pure standard library plus
  numpy. New runtime dependencies require an ADR.
- **Pinning is via `uv.lock`.** Do not pin in `pyproject.toml` unless
  upstream resolution requires it (`tool.uv.override-dependencies`
  documents the one current override).
- **No optional / extras groups today.** Everything the prototype needs
  is in the default install.
- **Dev tooling is in `dev` group.** Currently only `pytest`. Adding new
  dev tools is fine; they do not affect the runtime closure.

---

## Auditability

The prototype is auditable because every artifact produced by a CLI or
service function carries enough metadata to reproduce it:

| Artifact                | Provenance Carried                                                    |
| ----------------------- | --------------------------------------------------------------------- |
| API response            | `provenance` block (run ID, command, scenarios, git commit, caveats)  |
| Decision packet (CLI)   | Deterministic `packet_hash`; rerunning yields identical bytes         |
| Review ledger entries   | Deterministic `review_id` + `packet_hash`                              |
| Planner queue items     | Deterministic `item_id`                                               |
| Provenance manifests    | `scripts/22_provenance_manifest.py` → text / JSON / Markdown          |

`stable_run_id` is keyed on command, args, scenarios, input references,
and git commit — wall-clock time is **not** part of the key, so the same
inputs always yield the same run ID. This is intentional: the run ID
tells you "what was produced," not "when it was produced."

`get_git_commit()` is best-effort. It returns the literal string
`"unknown"` if `git` is unavailable, the working tree is not a
repository, or the subprocess times out. Manifests built outside a git
checkout are still well-formed.

---

## Code-review guardrails

These are enforced by tests, not by reviewer vigilance:

- **Forbidden vocabulary.** Source files and CLI outputs are scanned for
  the bigrams that would imply capabilities the prototype does not have
  (see [`docs/security.md`](security.md) for the list). Adding any of
  these strings outside an explicit non-claim disclaimer breaks the
  build.
- **Forbidden imports.** AST-based import-boundary scans on the
  hypothesis layer, the API layer, and the provenance module reject
  imports of `custody.detection.*`, `custody.ingest.gfw_presence`,
  `custody.fusion.tracker`, `sentinelhub`, and matcher runtime modules.
- **Determinism.** Every CLI test runs twice and compares byte-for-byte;
  every API test asserts that re-running with the same `generated_at`
  yields the same `request_id`.

---

## What this is not

- Not a production DevSecOps reference. There is no deploy step, no
  container image, no environment promotion ladder.
- Not a supply-chain attestation system. The prototype trusts upstream
  packages as resolved by `uv`. Roadmap items in
  [`docs/security.md`](security.md) cover what would have to change.
- Not a compliance baseline. SDA Custody Layer alignment is documented in
  the implementation guide; this repo does not claim conformance with any
  specific accreditation regime.
