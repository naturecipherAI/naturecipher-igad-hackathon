# Dashboard Data Schemas

Two files drive the dashboard. They are split because they change on different
cycles: `forecast.json` is rewritten every monthly run, `validation.json` changes
only when the model is retrained or revalidated. Keeping them together would let a
routine forecast run silently overwrite validation history.

Both are static JSON served from the same origin as `index.html`. No API, no build
step, no database.

```
dashboard/
  index.html        reads both files at load
  forecast.json     written by inference/forecast_runner.py
  validation.json   written by hand or by the training pipeline
  regions.geojson   optional, phase 2 map layer
```

---

## 1. forecast.json

### What changed from v1 and why

v1 was:

```json
{ "generated_at": "...", "threshold": 0.25,
  "regions": { "asal_north": [ { "month": 9, "year": 2026, "drought_prob": 0.217, ... } ] } }
```

Five problems, each fixed below.

| v1 problem | Consequence | Fix in v2 |
|---|---|---|
| No initialization date | A probability without an init date is uninterpretable. "October drought 0.262" means nothing until you know it was issued in July | `run.init_date` plus per-row `lead_months` |
| No uncertainty | SEAS5 ships 51 ensemble members. `seas5_adapter.py:84` collapses them with `.mean(dim="number")` and discards the spread, which is the single best uncertainty estimate available | `prob_p10` / `prob_p90` / `ensemble_agreement`, nullable |
| No provenance | A number on a dashboard cannot be traced back to the code and model that produced it | `run` block with commit, model and bridge versions, source GRIB keys |
| Regions as an object | Key order is not guaranteed, region metadata has nowhere to live, and consumers must cross-reference `regions.yaml` | Regions as an array carrying their own metadata |
| No status field | Nothing in the data distinguishes an experimental run from an operational one. The disclaimer lived only in prose | `run.status` enum, rendered as a badge |

### Schema

```jsonc
{
  "schema_version": "2.0",
  "generated_at": "2026-07-30T13:37:28.584606Z",   // ISO 8601 UTC, when the file was written

  "run": {
    "init_date": "2026-07-01",                     // forecast initialization, NOT the run date
    "source": "ECMWF SEAS5 seasonal-monthly-single-levels, originating_centre=ecmwf, system=51",
    "ensemble_members": 51,                        // members in the source GRIB
    "ensemble_reduction": "mean",                  // "mean" | "quantiles" | "full"
    "forecast_grib": "seas5/seas5_2026_07_forecast.grib",
    "climatology_grib": "seas5/seas5_1993-2016_07_hindcast_climatology.grib",
    "code_commit": "1c6f5aa",
    "drought_model": "block_3",
    "bridge_model": "cascade_v3",
    "status": "experimental"                       // "experimental" | "provisional" | "operational"
  },

  "config": {
    "threshold": 0.25,
    "threshold_rationale": "Recall-optimised. In anticipatory action a missed drought costs more than a false alarm."
  },

  "regions": [
    {
      "id": "asal_north",                          // must match inference/data_loader.py REGIONS
      "name": "ASAL North",
      "climate_zone": "arid",                      // "arid" | "semi-arid"
      "counties": ["Turkana", "Marsabit", "Samburu", "Isiolo"],
      "mean_annual_precipitation_mm": 350,
      "bbox": { "north": 4.62, "south": 0.45, "east": 41.91, "west": 34.95 },

      "forecasts": [
        {
          "valid_year": 2026,
          "valid_month": 10,
          "lead_months": 3,                        // valid_month - init_month
          "drought_prob": 0.262,                   // [0,1], classifier P(drought)
          "prob_p10": null,                        // ensemble 10th percentile, null until spread is wired
          "prob_p90": null,
          "ensemble_agreement": null,              // [0,1] fraction of members over threshold
          "drought_forecast": 1,                   // 0 | 1, derived: drought_prob >= threshold
          "signal": "drought",                     // "normal" | "drought", lowercase
          "margin": 0.012,                         // drought_prob - threshold, signed
          "drivers": null                          // optional [{feature, contribution}], SHAP if available
        }
      ]
    }
  ]
}
```

### Field rules

- `lead_months` is derived, never hand-set. July init, valid October, gives 3.
- `margin` is `drought_prob - threshold`. It is stored rather than computed in the
  client so the UI can rank signals by robustness. A signal with margin 0.012 is not
  the same finding as one with margin 0.31 and must not render identically.
- `prob_p10` / `prob_p90` / `ensemble_agreement` / `drivers` are `null` when not
  computed. **Never emit a placeholder number.** The dashboard renders `null` as an
  explicit "not computed" state.
- `signal` is redundant with `drought_forecast` and `threshold`, kept so downstream
  consumers (SMS gateway, ICPAC feed) do not each re-derive it and drift.
- Adding a region means appending to `regions[]`. No client change required.

---

## 2. validation.json

This file exists because of one rule: **a classifier metric without its base rate is
not interpretable.** 2021 to 2024 in northern Kenya spans the worst Horn of Africa
drought in forty years, so drought months are likely the majority class. An accuracy
of 0.667 against a 0.65 base rate is not a result. The schema therefore makes
`base_rate` and `baseline` required fields rather than optional ones, so a run cannot
report a metric while omitting what it should be compared against.

```jsonc
{
  "schema_version": "2.0",
  "updated_at": "2026-07-30T00:00:00Z",

  "backtest": {
    "region": "asal_north",
    "period": { "start": "2021-01", "end": "2024-12" },
    "n_samples": 48,                     // required. 48 monthly observations
    "base_rate": null,                   // required key. fraction of months labelled drought. null = not published
    "metrics": {
      "accuracy": 0.667,
      "f1": 0.692,
      "recall": 0.643,
      "precision": null
    },
    "confidence_interval": {
      "metric": "accuracy",
      "level": 0.95,
      "low": 0.534,                      // Wald interval at n=48, p=0.667
      "high": 0.800
    },
    "baseline": {
      "kind": "majority_class",          // "majority_class" | "era5_only" | "climatology"
      "computed": false,                 // false renders as an open gap in the UI, not a blank
      "metrics": null
    }
  },

  "retrospective": {
    "period": { "start": "2026-01", "end": "2026-03" },
    "kind": "hindcast",                  // "hindcast" | "operational". Never conflate the two
    "input_data": "ERA5 reanalysis, available after the fact",
    "claim": "Northeast Kenya drought emergency identified",
    "reference_event": "IPC declaration, March 2026",
    "caveat": "Retrospective. ERA5 for these months was used after the fact, not an ahead-of-time forecast."
  },

  "hindcasts": [],                       // mini-hindcast runs, one object per initialization

  "known_limitations": [
    "Forward forecasts experimental, mini-hindcast validation only",
    "Soil moisture uses persistence, SEAS5 provides no soil moisture layers",
    "Single July 2026 SEAS5 initialization",
    "Three ASAL regions only",
    "Ensemble spread discarded, forecasts carry no uncertainty band"
  ]
}
```

### Why `baseline.computed` is a boolean rather than just a null

A null reads as "no data yet" and disappears quietly. An explicit `false` lets the UI
render a labelled gap: *"ERA5-only baseline: not computed."* An honest system shows
what it has not yet tested. This is the single field that separates a forecast
dashboard from a marketing page.

---

## 3. The optional model service

The dashboard is static-first. Month selection and threshold are re-derived in the
browser from stored probabilities, so they need no server and stay correct offline.
**Changing the threshold changes the decision, never the forecast.** That distinction
is load-bearing: a threshold slider that looked like it re-ran the model would be
misleading.

Three things do need the model. The page probes `GET ./api/health` on load with a
1.5 second timeout. If it answers, these controls enable; if not, they render disabled
with the reason shown rather than hidden.

| Endpoint | Purpose | Notes |
|---|---|---|
| `GET /api/health` | Liveness probe | Must return 200 with S3 reachable, otherwise the dashboard advertises capability it cannot deliver |
| `POST /api/forecast` | Run a new initialization | Body `{init_date, regions[]}`. Returns a full `forecast.json` v2 payload |
| `POST /api/scenario` | Perturb and re-run | Body `{init_date, region, precip_scale, temp_delta}`. Returns the same shape with `run.status: "scenario"` |
| `POST /api/explain` | Feature attribution | Body `{region, valid_year, valid_month}`. Returns `drivers[]` matching the forecast row's `drivers` field |

Every response reuses the `forecast.json` v2 shape, so the client has one parser and a
live response is interchangeable with the committed file.

Not built yet. `inference/api.py` would wrap `forecast_runner.run_seas5_forecast` and
`run_retrospective`, which already return the right structure. Note that any hosted
instance holds AWS credentials, so it needs auth before it is public.

---

## 4. kenya-counties.geojson

The geometries referenced by `config/regions.yaml` (`geometry_file: asal_north.geojson`)
are not in the repository, so boundaries were sourced rather than approximated. Kenya's
ADM1 units are the 47 counties. `scripts/build_counties_geojson.py` downloads
geoBoundaries gbOpen KEN ADM1 (source RCMRD, public domain), simplifies with
Douglas-Peucker at 0.008 degrees, rounds to 4 decimals and tags each county with its
forecast region. Output is 67 KB from a 7.9 MB source.

```jsonc
{
  "type": "Feature",
  "properties": {
    "county": "Turkana",
    "region_id": "asal_north"            // joins to forecast.json regions[].id, null when uncovered
  },
  "geometry": { "type": "MultiPolygon", "coordinates": [] }
}
```

All 47 counties ship, not only the 11 covered. The other 36 render as uncovered, which
shows the coverage gap rather than cropping it out of frame. The client joins on
`region_id` and colours by that region's probability for the selected month. Counties
inherit a region value; the model does not resolve individual counties, and the map
says so.

geoBoundaries labels Tharaka-Nithi as `Tharaka`. That is handled by an alias in the
build script rather than by editing `regions.yaml` to match a source quirk.

---

## 5. grid.json, the gridded conditions field

A separate file, and deliberately not part of `forecast.json`, because it is a
different kind of claim at a different resolution. Mixing them would invite reading a
100 km conditions field as if it were a validated drought probability.

Written by `forecast_runner` on the same monthly run that writes `forecast.json`, via
`seas5_adapter.load_grid_field()` and `grid_to_payload()`.

```bash
python -m inference.forecast_runner                 # writes forecast.json and grid.json
python -m inference.forecast_runner --no-grid       # forecast only
python -m inference.forecast_runner --grid-output path/to/grid.json
```

Grid failures are non-fatal by design. The probabilities are computed before the grid
is attempted, so a missing GRIB or an absent `cfgrib` logs a warning and the run still
writes `forecast.json`. `--retrospective` reads historical parquet and never opens a
GRIB, so it emits no grid at all.

### What it is and is not

| | `forecast.json` | `grid.json` |
|---|---|---|
| Quantity | P(drought), classifier output | Precipitation and temperature anomalies |
| Resolution | Regional, 3 units | Native SEAS5 grid, roughly 1° |
| Baseline | n/a | That cell's own 1993-2016 hindcast climatology |
| Validated | Backtest exists, weakly | Not a prediction of an event, so no skill claim |
| Drives a decision | Yes, via the threshold | No. Context only |

Anomalies are computed per cell against **that same cell** across the hindcast, never
against a regional mean. Differencing a cell against a regional climatology converts a
permanent spatial gradient into an apparent anomaly, so a persistently dry cell would
read as drought every month no matter what the forecast said.

### Schema

```jsonc
{
  "schema_version": "2.0",
  "kind": "conditions_field",
  "resolution": {
    "lat_step_deg": 1.0,
    "lon_step_deg": 1.0,
    "note": "Native SEAS5 grid. Not downscaled. Do not render finer than this spacing."
  },
  "source_grib": "seas5/seas5_2026_07_forecast.grib",
  "baseline": "SEAS5 1993-2016 hindcast climatology, per cell",
  "variables": ["precip_mm", "precip_clim_mm", "precip_anomaly_mm",
                "precip_pct_of_normal", "temp_2m_c", "temp_anomaly_c"],
  "months": [
    {
      "valid_year": 2026,
      "valid_month": 10,
      "cells": [
        {
          "lat": 3.0, "lon": 38.0,
          "precip_mm": 87.61,
          "precip_clim_mm": 51.84,
          "precip_anomaly_mm": 35.77,
          "precip_pct_of_normal": 169.0,
          "temp_2m_c": 29.85,
          "temp_anomaly_c": 3.0
        }
      ]
    }
  ]
}
```

### Rules

- `resolution` is **required**, and the client must not render cells smaller than the
  declared step. The bridges target CHIRPS at 5 km and MODIS at 250 m, so it is
  tempting to draw at those sizes. Every value ultimately derives from a ~100 km
  driver, and a 250 m map built from 100 km inputs is a lie of resolution.
- A cell with no climatology counterpart emits `null`, never `0` and never an
  interpolated neighbour. Null renders as a gap. Zero would render as normal.
- Cells are a flat array with explicit `lat` and `lon` rather than a packed matrix, so
  a partial or ragged grid cannot be silently misaligned.
- No `drought_prob` field exists here, and none should be added without retrained
  per-cell bridges and pixel-level labels to validate against. See below.

### Why there is no per-pixel probability

Two blockers, neither of them engineering:

1. **Distribution shift.** The bridges and the `block_3` classifier were fitted on
   regionally averaged features. A single cell has far greater variance than the mean
   of forty cells, so per-cell inference is out of distribution and would be
   miscalibrated, most likely overconfident.
2. **No labels at that resolution.** IPC declarations, the ground truth, are issued per
   county and livelihood zone. There is no pixel-level drought label, so a pixel
   probability map could be rendered but never validated.

Reaching per-pixel probability means rebuilding `processed_v3/` without spatial
aggregation and retraining the bridges on cell-level data. That is Stage 2, not Stage 1.

---

## 6. Two mismatches this schema surfaced

Writing the schema forced a reconciliation between the docs and the config, and turned
up two disagreements that need a decision before submission.

1. **County membership.** `README.md` lists six counties for ASAL North (Turkana,
   Marsabit, Samburu, Isiolo, Laikipia, Baringo). `config/regions.yaml` lists four,
   omitting Laikipia and Baringo. The code reads the YAML, so the schema follows the
   YAML. The README is wrong, or the YAML is incomplete.

2. **Forecast months.** `docs/METHODS.md` states "Lead times: months 2-4 (August,
   September, October 2026)". Lead months 2 to 4 from a July initialization are
   September, October and November. The lead numbers are correct; the month names are
   wrong. `forecast.json` contains months 9, 10, 11, which agrees with the lead
   numbers. Fix the prose, not the data.
