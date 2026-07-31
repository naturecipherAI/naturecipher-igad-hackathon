# Runbook — regenerate the submission numbers

For whoever holds the S3 credentials. Everything here runs on your machine; no
credentials leave it. Total time about 25 minutes, most of it the SEAS5 download.

Four physics bugs were fixed on branch `feat/gridded-conditions-field`. **Every
published probability changes**, so the pipeline has to be re-run before
submission — the numbers currently in `dashboard/forecast.json` came from the
pre-fix code.

---

## What changed and why it matters

| Bug | Was | Now |
|---|---|---|
| `bias_correct` | Multiplied the forecast by its own anomaly ratio, squaring it. A forecast at 1.30x climatology came out at 1.69x. | Anomaly mapped onto the ERA5 climatology, as `METHODS.md` always described |
| PET | `mrort` (runoff) was mapped to `era5_pet_mm`. In the ASALs runoff is near zero, so SPEI computed precip minus roughly nothing | Hargreaves from `t2m`; runoff keeps its own column |
| SPI-3/6/12 | Three copies of one single-month z-score | Real 3/6/12-month accumulation windows |
| `vhi`, `vci_lag1/2` | Stripped from history, never rebuilt, silently zeroed by `fillna(0)` | Derived — VHI as the standard `0.5*VCI + 0.5*TCI` |

Direction of the change is not predictable in advance: the squared anomaly
exaggerated both wet and dry departures, so some probabilities will rise and
some will fall. **That is expected. Report whatever comes out.**

---

## 1. Set up (5 min)

```bash
git fetch origin
git checkout feat/gridded-conditions-field
git pull

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp credentials.env .env          # or write AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
```

Confirm S3 is reachable before anything else:

```bash
python -m inference.data_loader
```

Expect `S3 connection verified: s3://naturecipher-forecast`. If it fails, stop
here — nothing downstream will work.

---

## 2. Baselines — run this first (5 min)

This is the highest-value output. The judging panel is ICPAC's own early-warning
team; the first question about a 0.667 accuracy will be "against what baseline?"

```bash
python scripts/evaluate_baselines.py
```

Prints a table and writes `dashboard/validation.json`:

```
region           base   majF1  era5F1   casF1    lift
asal_north      0.xxx   0.xxx   0.xxx   0.xxx  +0.xxx
```

**Read the `lift` column carefully.** It is cascade F1 minus ERA5-only F1:

- **Positive** — the bridges earn their complexity. Lead with it.
- **Near zero or negative** — the bridges are not adding over their own inputs.
  Say so plainly and reframe the contribution as forecasting where satellites
  cannot observe, rather than as accuracy. **Do not hide this.** A panel of
  modellers will run the comparison themselves, and being the team that
  published it first is worth more than a number that does not survive scrutiny.

If the label column is not auto-detected, pass it: `--label <column>`.

---

## 3. Re-run the forecast (10 min, mostly download)

```bash
python -m inference.forecast_runner
```

Writes `dashboard/forecast.json` and `dashboard/grid.json`. The grid step is
non-fatal: if it prints `Grid field skipped`, the forecast is still valid and the
map layer simply has no conditions overlay.

Sanity checks before trusting the output:

- Probabilities are in `[0, 1]` and are **not** all identical
- `era5_spi3`, `era5_spi6`, `era5_spi12` differ from each other
- PET lands roughly 120–220 mm/month (not near zero, which was the runoff bug)

---

## 4. Verify against the stored reference (2 min)

```bash
python -m inference.smoke_test --tolerance 0.15
```

**This is expected to FAIL**, and that is the correct outcome. It compares
against `validation/forecast_2026.json`, produced by the pre-fix pipeline. A pass
would mean the fixes changed nothing.

Record the before/after deltas it prints — they are direct evidence of the fix
and belong in the submission. Once you accept the new numbers, upload the fresh
output to `validation/forecast_2026.json` so the smoke test is green for judges.

---

## 5. Send back

1. `dashboard/forecast.json`
2. `dashboard/grid.json` (if produced)
3. `dashboard/validation.json`
4. The baseline table from step 2, pasted as text
5. The smoke-test deltas from step 4

Or commit them directly:

```bash
git add dashboard/ && git commit -m "data: re-run pipeline after physics fixes" && git push
```

---

## If something breaks

| Symptom | Cause | Fix |
|---|---|---|
| `No parquet files found` | Wrong bucket or region | Check `AWS_REGION=us-east-1` and `S3_BUCKET` in `.env` |
| `ModuleNotFoundError: cfgrib` | eccodes binary missing | `conda install -c conda-forge eccodes` then reinstall `cfgrib` |
| `Grid field skipped` | GRIB parse or download failed | Non-fatal — `forecast.json` still written |
| Baselines: label not found | Column name differs | `python scripts/evaluate_baselines.py --label <name>` |
| PET still near zero | Stale checkout | Confirm `git log -1` shows the fix commit |


---

## Appendix — LLM layer (Groq)

The dashboard's chat and bulletin run as Cloudflare Pages Functions. Both fail
soft: without a key they return 503 and the rest of the dashboard is unaffected.

```bash
# from the repo root, so functions/ is picked up alongside dashboard/
npx wrangler pages secret put GROQ_API_KEY --project-name=naturecipher-drought
npx wrangler pages deploy dashboard --project-name=naturecipher-drought     --branch=main --commit-dirty=true
```

**Confirm the model id before demoing.** Groq retires hosted models on short
notice, and a stale id returns 404 at request time, not at deploy time:

```bash
curl -s -H "Authorization: Bearer $GROQ_API_KEY"   https://api.groq.com/openai/v1/models | python -m json.tool | grep '"id"'
```

The default is `llama-3.3-70b-versatile`. To use a different one, set the
`GROQ_MODEL` variable on the Pages project — no code change, no redeploy of the
functions themselves.

Verify after deploying:

```bash
curl -s -X POST https://naturecipher-drought.pages.dev/api/chat   -H 'content-type: application/json'   -d '{"messages":[{"role":"user","content":"Which region is flagged?"}]}'
```

A 503 means the secret is missing. A 502 naming a model id means `GROQ_MODEL`
needs updating.
