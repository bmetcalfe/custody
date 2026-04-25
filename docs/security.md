# Custody — Security Posture

*Honest scoping document. If any section embarrasses you to read, that section is wrong.*

Custody is a **local prototype**. It runs on a developer workstation against
synthetic Tennent / Whitsun fixtures. The decision API contract
([Slice 17](../src/custody/api/)) and provenance manifests
([Slice 18](../src/custody/provenance.py)) are reference designs, not deployed
services.

This document records what the prototype does and does not provide today, and
what would be required to harden it for any real-world deployment. None of the
controls listed under **Roadmap** are in scope for the current applied-research
project.

---

## Threat model in scope

The prototype defends against exactly two classes of mistake:

1. **Accidental scope creep.** Source-file scans, AST-based import-boundary
   tests, and explicit non-claim disclaimers prevent the codebase from
   silently growing into territory it has not earned (live execution paths,
   real-data lineage, financial estimates).
2. **Non-deterministic artifacts.** Provenance records, deterministic run
   IDs, fixed fixture timestamps, and JSON-serializable response shapes
   make every artifact reproducible and auditable after the fact.

Anything else — authenticated callers, encrypted transport, hardened
deployment, multi-tenant isolation, supply-chain attestation — is **out of
scope** for the current prototype.

---

## What the prototype provides today

- **Local-only execution.** The CLI scripts and the decision API service
  layer call deterministic functions in-process. No network listener is
  started; `custody.api.app.create_app()` is a stub that raises
  `RuntimeError` until a deployable HTTP layer is added.
- **Fixture-only inputs.** The core decision-loop CLIs run against the
  deterministic Tennent / Whitsun narrative fixtures. The artifact
  bridge, scene-availability bridge, and scheduler-lite consume small
  committed JSON fixtures under `tests/fixtures/artifacts/`,
  `tests/fixtures/availability/`, and `tests/fixtures/schedule/`. There
  is no live external data ingestion, no real-time fetch path, and no
  real-data lineage in any of these flows.
- **Deterministic outputs.** Wall-clock time is excluded from every stable
  identifier (`request_id`, `run_id`, `packet_hash`, `review_id`,
  `item_id`). Re-running the same command on the same commit yields
  byte-identical outputs.
- **Auditable provenance.** Every API response carries a `provenance`
  block (run ID, command, scenario IDs, input fixtures, git commit,
  default assumptions, default caveats). `scripts/22_provenance_manifest.py`
  emits the same record as a standalone manifest in text / JSON / Markdown.
- **Language guardrails.** Tests forbid the bigrams that would imply
  capabilities the prototype does not have ("tasking order", "live
  tasking", "sensor command", "production scheduler", and so on).
  Source-file scans catch the same bigrams in module source.
- **Import-boundary scans.** The hypothesis, API, and provenance layers
  may not import detection runtime, GFW ingest, fusion tracker, or
  Sentinel SDKs. Violations are pytest failures, not review notes.

---

## What the prototype does not provide

- **No authentication, no authorization.** The prototype API contract has
  no concept of caller identity, no API keys, no JWTs, no mTLS, no role
  separation.
- **No transport security.** Nothing is encrypted. Nothing is signed.
  Nothing is rate-limited.
- **No deployment.** The prototype does not run as a daemon, does not bind
  a port, does not have a containerization story, and is not orchestrated.
- **No real-data lineage.** Provenance manifests describe the synthetic
  fixture run; they are not a real-data lineage record.
- **No execution authorization.** Candidate collect types are prototype
  recommendations. The prototype does not issue execution authorizations,
  does not interface with platform schedulers, and does not communicate
  with any external collection system.
- **No financial estimate.** Planning utility, mission value, and
  portfolio score are deterministic proxies, not monetary estimates.

---

## Roadmap (deferred)

The following are not implemented and not in scope for the current
applied-research project. They are listed here so that a reader can see
honestly what would have to change for the prototype to leave the
laptop:

- Authenticated transport (mTLS, signed requests, scoped tokens)
- Caller identity, role separation, audit log of caller-attributed actions
- Encrypted-at-rest storage for any persisted artifacts
- Supply-chain attestation (SBOM, signed releases, reproducible builds)
- Hardened deployment topology, network egress controls, secrets management
- Real-data lineage (signed hashes of real ingest artifacts, not synthetic
  fixtures)
- Detection of prompt-injection / poisoned upstream evidence
- Multi-tenant isolation if more than one analyst ever shared the deployment

---

## Reporting issues

This is a private applied-research repository. Issues should be opened on
the GitHub repository or raised directly with the maintainer. There is no
disclosure timeline because there is no deployed service to disclose
against.
