# Data architecture — and why there is no database

## The decision

**Cascade Bridge ships no live database. That is deliberate.**

A reviewer is right to ask why. Here is the reasoning, and the schema we would
use the moment the answer changes.

## What the system actually moves

| Artifact | Size | Changes |
|---|---|---|
| Forecast payload (9 probabilities + metadata) | 5 KB | Once a month |
| County boundaries | 67 KB | Effectively never |
| Validation and baselines | 2 KB | On retrain |
| Trained models | ~4 MB | On retrain |
| Processed history (parquet) | tens of MB | On ingest |

The live surface a user touches is **73 KB, regenerated monthly**.

A database earns its place when you have concurrent writes, queries whose shape
you cannot predict, data too large to send whole, or per-user state. This system
has none of those. It is a batch pipeline that emits a small document.

## What we do instead

```
S3 (private)                    Cloudflare (public)
  models/          ─┐             dashboard/*.json   static, CDN-cached
  processed_v3/     ├─ monthly ─▶ *.geojson          static, CDN-cached
  seas5/           ─┘  batch      functions/api/*    stateless, no store
```

S3 is the system of record for models and history. The dashboard reads flat JSON
from a CDN. The API functions are stateless — they read the same public JSON and
hold nothing.

What this buys:

- **No cold start, no connection pool, no query latency.** A CDN edge read.
- **Nothing to be down during judging.** A database is a component that can fail
  while someone is clicking; a static file on a CDN is not.
- **Reproducible.** Every published number is a file in git. You can diff a
  forecast against last month's.
- **Free.** No idle instance.
- **Offline-capable.** The dashboard works from a local directory.

Adding Postgres here would be infrastructure theatre — visible complexity that
buys nothing and costs uptime.

## When this stops being right

Three triggers. Any one of them, and we build the schema below.

**1. Forecast verification (the real one).** Right now each monthly run
overwrites the last. To measure whether forecasts were *right* — the only
honest way to earn trust with a drought committee — every issued forecast must
be kept and later joined against what actually happened. That is a genuine
relational problem: forecast × outcome, over time, sliced by region and lead.

**2. Scale-out.** Eleven counties fits in a file. All 47, or IGAD's eight member
states at county resolution, with 12 lead months and 51 ensemble members, does
not.

**3. Per-user state.** Subscriptions, custom thresholds, delivery preferences,
who was alerted and when.

## The schema, for when it is needed

Postgres with PostGIS and TimescaleDB. Designed, not built.

```sql
-- Reference geography. Static.
CREATE TABLE region (
    region_id       TEXT PRIMARY KEY,          -- 'asal_north'
    name            TEXT NOT NULL,
    climate_zone    TEXT NOT NULL CHECK (climate_zone IN ('arid','semi-arid')),
    mean_annual_precip_mm INT,
    geom            GEOMETRY(MultiPolygon, 4326) NOT NULL
);

CREATE TABLE county (
    county_id       TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    region_id       TEXT REFERENCES region(region_id),   -- NULL = no coverage
    geom            GEOMETRY(MultiPolygon, 4326) NOT NULL
);
CREATE INDEX ON county USING GIST (geom);

-- One row per pipeline execution. Every forecast traces to the exact code,
-- models and inputs that produced it; without this, a number cannot be audited.
CREATE TABLE forecast_run (
    run_id          BIGSERIAL PRIMARY KEY,
    init_date       DATE NOT NULL,             -- when the forecast was issued
    generated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    source          TEXT NOT NULL,             -- 'ECMWF SEAS5 system 51'
    ensemble_members SMALLINT NOT NULL,
    ensemble_reduction TEXT NOT NULL,          -- 'mean' | 'quantiles'
    code_commit     TEXT NOT NULL,
    drought_model   TEXT NOT NULL,
    bridge_model    TEXT NOT NULL,
    threshold       NUMERIC(4,3) NOT NULL,
    status          TEXT NOT NULL CHECK (status IN ('experimental','provisional','operational')),
    UNIQUE (init_date, drought_model, bridge_model, code_commit)
);

-- The forecasts themselves. Append-only: a run is never edited, only superseded.
CREATE TABLE forecast (
    run_id          BIGINT REFERENCES forecast_run(run_id) ON DELETE CASCADE,
    region_id       TEXT REFERENCES region(region_id),
    valid_month     DATE NOT NULL,             -- first of the month
    lead_months     SMALLINT NOT NULL CHECK (lead_months BETWEEN 0 AND 12),
    drought_prob    NUMERIC(5,4) NOT NULL CHECK (drought_prob BETWEEN 0 AND 1),
    prob_p10        NUMERIC(5,4),              -- NULL until ensemble spread is carried
    prob_p90        NUMERIC(5,4),
    ensemble_agreement NUMERIC(5,4),
    PRIMARY KEY (run_id, region_id, valid_month),
    CHECK (prob_p10 IS NULL OR prob_p90 IS NULL OR prob_p10 <= prob_p90)
);
SELECT create_hypertable('forecast', 'valid_month');

-- What actually happened. The other half of verification.
CREATE TABLE observed_outcome (
    region_id       TEXT REFERENCES region(region_id),
    valid_month     DATE NOT NULL,
    drought         BOOLEAN NOT NULL,
    source          TEXT NOT NULL,             -- 'IPC' | 'NDMA' | ...
    recorded_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (region_id, valid_month, source)
);

-- Skill, computed not stored. This view is the reason the database exists.
CREATE VIEW forecast_skill AS
SELECT r.init_date, f.region_id, f.lead_months,
       count(*)                                             AS n,
       avg((f.drought_prob >= r.threshold)::int
           = o.drought::int)::NUMERIC(4,3)                  AS accuracy,
       avg(o.drought::int)::NUMERIC(4,3)                    AS base_rate,
       -- Brier score: mean squared error of the probability itself, which is
       -- the metric that survives a change of threshold.
       avg((f.drought_prob - o.drought::int) ^ 2)::NUMERIC(5,4) AS brier
FROM forecast f
JOIN forecast_run r     ON r.run_id = f.run_id
JOIN observed_outcome o ON o.region_id = f.region_id
                       AND o.valid_month = f.valid_month
GROUP BY r.init_date, f.region_id, f.lead_months;
```

Three design points worth stating:

- **`forecast` is append-only and keyed by `run_id`.** A re-run does not
  overwrite history; it adds a run. Otherwise you can never answer "what did we
  say in July?" — and that question is the whole point.
- **Skill is a view, not a table.** Derived numbers that are stored go stale
  silently. A view cannot disagree with its inputs.
- **Brier score alongside accuracy.** Accuracy depends on the threshold, which is
  a policy choice. Brier scores the probability itself, so it stays comparable
  when a county moves its threshold.

## The migration, when it comes

The static JSON is already the right shape — `docs/DASHBOARD_SCHEMA.md` defines
`run`, `regions[]` and `forecasts[]`, which map one-to-one onto `forecast_run`,
`region` and `forecast`. Adding a database is an insert step at the end of
`forecast_runner`, not a rewrite. The dashboard keeps reading JSON; the API
starts serving it from a query instead of a file.

Designing for that migration and not performing it yet is the actual engineering
decision here.
