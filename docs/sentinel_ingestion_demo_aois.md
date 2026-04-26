# Sentinel Ingestion — Demo AOIs

*Companion to [docs/sentinel_ingestion_whitsun.md](sentinel_ingestion_whitsun.md). Documents the AOI extension that lets the same Sentinel ingestion layer cover both demo scenarios.*

---

## Two demo AOIs, one ingestion layer

The Sentinel ingestion sprint shipped with one AOI (Whitsun). The bounded extension in this slice adds a second AOI (Tennent Reef) and generalizes the CLI so both scenarios can be queried, cached, and tested through the same code path.

| AOI       | Scenario role                                  | Default time window     | Center                 | Demo fixture                                                     |
| --------- | ---------------------------------------------- | ----------------------- | ---------------------- | ---------------------------------------------------------------- |
| `whitsun` | Maritime custody / reacquisition (primary)     | 2023-12-01 → 2023-12-15 | 9.98°N, 114.63°E       | `data/demo/whitsun_sentinel_observations.fixture.json`           |
| `tennent` | Fixed-site monitoring / construction context   | 2023-07-01 → 2023-08-20 | 8.8583°N, 114.6561°E   | `data/demo/tennent_sentinel_observations.fixture.json`           |

Both AOIs:

- Use the same normalized [`ObservationArtifact`](../src/custody/ingest/sentinel.py) model (frozen dataclass with `data_mode`, `confidence_weight`, `usable_for_detection`, `usable_for_context`, and explicit caveats).
- Source from the same CDSE STAC v1 endpoint when `--live` is passed.
- Default to offline operation against committed fixtures.
- Treat Sentinel observations as **public lower-confidence context, not high-confidence proof on their own** — Umbra remains the high-confidence tasked source for both scenarios.

## Tennent role: fixed-site monitoring / construction context

Whitsun and Tennent ask different questions of the Sentinel layer:

- **Whitsun** is a custody / reacquisition demo: Sentinel-1 and low-cloud Sentinel-2 act as intermediate context between Umbra SAR collects, helping show that the system does not go blind between premium SAR collects.
- **Tennent** is a fixed-site monitoring demo: Sentinel-1 ascending / descending passes provide repeat coverage over a static feature (the Vietnamese-controlled Tennent Reef outpost / construction footprint), and Sentinel-2 contributes optical context when cloud cover allows. The site does not move; the question is *what changed since last time*, not *where did the vessel go*.

Tennent's AMTI-published reference center (8°51'30" N, 114°39'22" E ≈ 8.8583°N, 114.6561°E) is captured in the `metadata.center_lat_deg` / `center_lon_deg` fields of the fixture geometry, alongside an explicit `replace-before-ops` caveat.

## How to run

The CLI (`scripts/29_ingest_sentinel_whitsun.py`) accepts an AOI label *or* a path. Backward compatibility with the original `--aoi <path>` form is preserved.

```bash
# Offline (default): Whitsun fixture
python scripts/29_ingest_sentinel_whitsun.py
python scripts/29_ingest_sentinel_whitsun.py --aoi whitsun

# Offline: Tennent fixture
python scripts/29_ingest_sentinel_whitsun.py --aoi tennent

# Live CDSE STAC v1 query (network required)
python scripts/29_ingest_sentinel_whitsun.py --live --aoi whitsun
python scripts/29_ingest_sentinel_whitsun.py --live --aoi tennent

# Backward-compatible path form (still works)
python scripts/29_ingest_sentinel_whitsun.py \
    --aoi data/demo/tennent_aoi.fixture.geojson \
    --fixture data/demo/tennent_sentinel_observations.fixture.json
```

When `--aoi` is one of the known labels, the CLI fills in the AOI path, the offline fixture path, the live cache path, and the default time window from the matching profile. Any `--start` / `--end` / `--fixture` / `--output` flag still wins over the profile default.

When `--live` is passed, the live cache is written to:

- Whitsun: `data/demo/whitsun_sentinel_observations.json` (gitignored)
- Tennent: `data/demo/tennent_sentinel_observations.json` (gitignored)

The committed `*.fixture.json` files are the demo source of truth.

## Tennent demo fixture contents

`data/demo/tennent_sentinel_observations.fixture.json` ships three deterministic records with `data_mode: "fixture"`:

| `observation_id`                                    | Source        | Date       | Cloud cover | Notes                              |
| --------------------------------------------------- | ------------- | ---------- | ----------- | ---------------------------------- |
| `fixture-s1-grd-tennent-20230715`                   | Sentinel-1    | 2023-07-15 | n/a         | VV + VH, ascending                 |
| `fixture-s2-l2a-tennent-20230718-low-cloud`         | Sentinel-2    | 2023-07-18 | 12 %         | low-cloud; usable_for_detection    |
| `fixture-s2-l2a-tennent-20230728-cloudy`            | Sentinel-2    | 2023-07-28 | 85 %         | cloudy; usable_for_context only    |

The 25 % cloud-cover threshold (`S2_LOW_CLOUD_PCT`) and confidence-weight buckets are shared with the Whitsun fixture; see [docs/sentinel_ingestion_whitsun.md](sentinel_ingestion_whitsun.md) for the full table and disclaimers.

## What is NOT changing

The bounded scope of this extension:

- No dashboard refactor.
- No Tennent decision-trace yet — the future trace can re-use these fixtures, but no decision-layer code is wired to Tennent Sentinel observations in this slice.
- No live Sentinel product downloads, no imagery processing, no detection inference.
- No planner / scheduler / RL changes.
- No new runtime dependencies.

## Tests

`tests/test_ingest_sentinel.py` covers:

- Tennent AOI fixture loads and is a valid GeoJSON `FeatureCollection`.
- Tennent observation fixture loads via `load_observation_cache` and contains exactly the expected S1 + S2-low + S2-cloudy split.
- Normalized records share the same field set across AOIs (no Tennent-only or Whitsun-only fields).
- All Tennent record bboxes lie inside the Tennent AOI bbox.
- CLI default still resolves to Whitsun for backward compatibility.
- CLI `--aoi tennent` label resolves to the Tennent fixture.
- CLI `--aoi <path>` legacy form still works.
- CLI `--format json` summary works for Tennent.

No test contacts the live CDSE STAC endpoint.
