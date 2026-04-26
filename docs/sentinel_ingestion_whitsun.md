# Sentinel Ingestion — Whitsun AOI

*Sentinel ingestion sprint. Companion to [docs/positioning.md](positioning.md), [docs/scenario.md](scenario.md), and the existing artifact / availability bridges.*

---

## What this step does

This sprint adds a thin **metadata-only** layer that asks one question:

> What Sentinel-1 / Sentinel-2 observations are available around the
> Whitsun AOI and time window, and how can they be represented as
> lower-confidence context between Umbra SAR collections?

It ships:

- A normalized `ObservationArtifact` dataclass (`src/custody/ingest/sentinel.py`)
  covering Umbra SAR, Sentinel-1, Sentinel-2, and simulated / fixture records.
- A small CDSE STAC v1 client (`requests`-based, no new dependency)
  that issues a single search POST and normalizes the result.
- Cache I/O (`write_observation_cache` / `load_observation_cache`) for
  round-tripping normalized observations through JSON.
- A demo placeholder AOI (`data/demo/whitsun_aoi.fixture.geojson`) and
  a committed deterministic observation fixture
  (`data/demo/whitsun_sentinel_observations.fixture.json`).
- A CLI driver (`scripts/29_ingest_sentinel_whitsun.py`) that runs
  offline-by-default against the committed fixture and live-on-demand
  against CDSE STAC v1.
- Tests against captured-shape STAC items under
  `tests/fixtures/sentinel_stac/`. No live HTTP is performed during
  the test suite.

## What this step does not do

- It does **not** download full Sentinel products or imagery.
- It does **not** run Sentinel vessel detection, model inference, or
  CFAR / VLM pipelines.
- It does **not** modify the Dash UI, the hypothesis layer, the
  scheduler, the planner queue, or any tasking module.
- It does **not** treat Sentinel observations as equivalent to
  Umbra-tasked SAR for vessel reacquisition. Umbra remains the
  high-confidence tasked source; Sentinel-1 and Sentinel-2 are
  intermediate public context.
- It does **not** issue any execution authorization, sensor command,
  or live tasking instruction.

## API used

- **Endpoint**: `https://stac.dataspace.copernicus.eu/v1/`
- **Search URL**: `https://stac.dataspace.copernicus.eu/v1/search`
- **Method**: `POST` with a JSON body containing `collections`,
  `intersects` (GeoJSON Geometry), `datetime` (ISO range), and `limit`.
- **Not used**: the deprecated RESTO / OpenSearch APIs, and the
  legacy `https://catalogue.dataspace.copernicus.eu/stac` endpoint.

## Collections used

Initial Whitsun query targets:

- `sentinel-1-grd` — public SAR ground-range-detected products
- `sentinel-2-l2a` — Level-2A optical (surface reflectance) products

Other collections may be passed via `--collections`. The normalizer
recognises Sentinel-1 / Sentinel-2 collection ids by prefix and falls
back to a `simulated` source classification for anything it doesn't
recognise.

## Credential assumptions

The CDSE STAC v1 search endpoint is currently anonymous for basic
metadata queries; no API key is required for the search call shipped
in this sprint. If CDSE later requires authenticated access, the
client should be wired through a `requests.Session` configured with
the appropriate token from a `.env` file (this repo already loads
`.env` via `python-dotenv` for the GFW path) — wiring that token is
out of scope for this sprint.

## Fixture vs real data

- `data/demo/whitsun_aoi.fixture.geojson` — **demo placeholder AOI**.
  Roughly 114.45–114.85°E by 9.75–10.25°N, centered on the existing
  `_WHITSUN_CENTER = (9.98, 114.63)` constant. Must be replaced with
  the real operational AOI before any non-demo use. The fixture is
  marked `data_mode: "fixture"` in its metadata block.
- `data/demo/whitsun_sentinel_observations.fixture.json` — **committed
  deterministic demo fixture** of three normalized observations: one
  Sentinel-1 GRD, one low-cloud Sentinel-2 L2A (`cloud_coverage: 8.0`),
  and one cloudy Sentinel-2 L2A (`cloud_coverage: 72.0`). Every record
  carries `data_mode: "fixture"` and explicit caveats. The future
  Whitsun decision-trace demo can run from this fixture even when
  CDSE STAC is unreachable.
- `data/demo/whitsun_sentinel_observations.json` — **live CDSE STAC
  output**. Written by the script when run with `--live`. This path
  is gitignored — only the `.fixture.json` form is committed.

The `ObservationArtifact.data_mode` field is `"real"`, `"fixture"`,
or `"simulated"`. Tests assert that normalized records preserve this
field and that fixture records carry an explicit `"fixture record;
not a live STAC query result"` caveat.

## Confidence weighting (demo heuristics)

These weights are **starter values**, not calibrated reliability
numbers. They are exposed as `CONFIDENCE_WEIGHTS` so they can be
swapped wholesale once a calibration pass exists.

| Source                                         | Weight | `usable_for_detection` | `usable_for_context` |
| ---------------------------------------------- | ------ | ---------------------- | -------------------- |
| Umbra tasked SAR                               | 1.00   | yes                    | yes                  |
| Sentinel-1 GRD public SAR                      | 0.45   | yes                    | yes                  |
| Sentinel-2 L2A, cloud cover ≤ 25%              | 0.30   | yes                    | yes                  |
| Sentinel-2 L2A, cloud cover > 25%              | 0.10   | no                     | yes                  |
| Simulated / synthesised                        | 0.50   | no                     | yes                  |

The 25% threshold (`S2_LOW_CLOUD_PCT`) is the demo cutoff between
"usable for detection" and "context-only". It is also a starter value.

Every normalized observation carries an explicit caveat block
including:

- `metadata-only ingestion; no imagery is downloaded`
- `confidence weights are demo heuristics, not calibrated sensor reliability values`
- `Sentinel observations are public lower-confidence context, not tasked Umbra collections`
- A bucket label (`confidence weight bucket: sentinel-1`,
  `sentinel-2-cloudy`, etc.)
- A high-cloud caveat when applicable

## How to run the ingestion

Offline (default — reads the committed fixture):

```bash
python scripts/29_ingest_sentinel_whitsun.py
```

Output (text):

```
Mode:                offline fixture (data/demo/whitsun_sentinel_observations.fixture.json)
AOI:                 data/demo/whitsun_aoi.fixture.geojson
Total observations:  3
  Sentinel-1:        1
  Sentinel-2:        2  (low-cloud 1, cloudy 1)
Date range:          2023-12-10T22:00:00+00:00  ->  2023-12-15T03:00:00+00:00
Best low-cloud S2:   fixture-s2-l2a-whitsun-20231212-low-cloud  cloud=8%  at 2023-12-12T03:00:00+00:00
Latest Sentinel-1:   fixture-s1-grd-whitsun-20231210  at 2023-12-10T22:00:00+00:00
```

Live (opt-in — issues a single CDSE STAC v1 search POST):

```bash
python scripts/29_ingest_sentinel_whitsun.py --live \
    --start 2023-12-01 --end 2023-12-15 --max-items 20
```

Options:

- `--aoi PATH` — alternate GeoJSON Polygon / Feature / FeatureCollection
- `--start ISO_DATE`, `--end ISO_DATE` — search window (defaults to
  the December 2023 Whitsun observation window)
- `--max-items INT` — paging limit
- `--collections LIST` — comma-separated collection ids
- `--output PATH` — where to write the live cache
- `--fixture PATH` — alternate offline fixture
- `--format {text,json}` — summary format

JSON summary form (`--format json`):

```json
{
  "mode": "offline fixture (...)",
  "aoi": "...whitsun_aoi.fixture.geojson",
  "output": null,
  "total": 3,
  "sentinel_1_count": 1,
  "sentinel_2_count": 2,
  "sentinel_2_low_cloud_count": 1,
  "sentinel_2_cloudy_count": 1,
  "date_range": ["2023-12-10T22:00:00+00:00", "2023-12-15T03:00:00+00:00"],
  "best_low_cloud_sentinel_2": "fixture-s2-l2a-whitsun-20231212-low-cloud",
  "latest_sentinel_1": "fixture-s1-grd-whitsun-20231210"
}
```

## How the output should feed the future decision trace / dashboard

Each `ObservationArtifact` is JSON-serialisable, deterministic, and
keyed on a stable `observation_id`. The shape is intentionally
compatible with downstream consumers:

- **Decision trace** — the future Whitsun decision-trace view can
  enumerate Sentinel observations between Umbra collects and render
  them as lower-confidence context, weighted by `confidence_weight`,
  with `usable_for_detection` controlling whether they're treated as
  detection candidates or context-only.
- **Hypothesis bridge (Slice 19 artifact bridge)** — Sentinel records
  can be promoted into `HypothesisEvidence` later via the existing
  artifact bridge. That promotion path is intentionally not wired in
  this sprint; it would require a small adapter that maps
  `ObservationArtifact` → `ArtifactRecord` and lives in a follow-up
  slice.
- **Dashboard** — the future Dash demo should be able to render the
  committed fixture without any network call. A live refresh button
  that runs the `--live` path is the natural next step, behind an
  explicit user action.

## What changes from existing modules

Nothing runtime. The hypothesis layer, scheduler, execution sim, and
strategy comparison harness are untouched. The only modifications are:

- New module `src/custody/ingest/sentinel.py`.
- New exports added to `src/custody/ingest/__init__.py`.
- New committed fixtures under `data/demo/` and
  `tests/fixtures/sentinel_stac/`.
- New CLI script `scripts/29_ingest_sentinel_whitsun.py`.
- New tests under `tests/test_ingest_sentinel.py`.
- One `.gitignore` entry for the live cache output path.
