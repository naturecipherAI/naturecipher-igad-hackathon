# Cascade Bridge: Satellite-Independent Drought Early Warning

**IGAD Hackathon 2026 — NatureCipher AI**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## What it does

Cascade Bridge forecasts drought probability 1-3 months ahead for Kenya's Arid
and Semi-Arid Lands, using no satellite data at forecast time.

**The problem it solves:** satellites cannot photograph October in July. A
drought classifier built on CHIRPS rainfall and MODIS vegetation cannot forecast,
because those observations only exist for months that have already happened. This
is not a latency problem — it is that the future has not been observed.

Cascade Bridge makes those inputs exist. Chained XGBoost regressors synthesise the
CHIRPS, NDVI and LST indicators from ECMWF SEAS5 atmospheric forecasts, so the
full 51-feature classifier can run at a 3-month lead. The satellite record is not
discarded; it is distilled into the bridges at training time.

**Where it sits:** alongside ICPAC's Drought Watch and HUSIKA, not against them.
Those systems monitor current conditions well. This adds a forecast layer.

## Live Dashboard

**https://naturecipher-drought.pages.dev**

Issued from the July 2026 SEAS5 initialization, covering September-November 2026
across three ASAL county clusters. Interactive: scrub the valid month, drag the
decision threshold, click a region to filter. Changing the threshold changes the
decision, never the forecast.

Labelled experimental pending full hindcast validation.

## Architecture

```
SEAS5 Seasonal Forecast
  └── Bridge 1 → CHIRPS proxy
  └── Bridge 2 → NDVI proxy
  └── Bridge 3 → LST proxy
        └── 51 features → XGBoost → Drought probability
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for full detail.

## Validation

Run `python scripts/evaluate_baselines.py` to reproduce. It publishes the drought
base rate and two baselines alongside the cascade, because an accuracy figure
without them is not evidence of skill:

| | What it tests |
|---|---|
| **Base rate** | 2021-2024 spans the worst Horn of Africa drought in 40 years, so drought may be the majority class. Without this number, accuracy is unreadable |
| **Majority-class baseline** | The floor. Always predict the commoner label |
| **ERA5-only baseline** | The same classifier without the bridges. This tests the cascade's actual thesis — the bridges are deterministic functions of the ERA5 inputs, so they cannot add information that was not already there. If the lift is zero, the contribution is forecastability, not accuracy |

Current numbers are written to `dashboard/validation.json` and rendered in the
dashboard with a 95% confidence interval.

**The Jan-Mar 2026 result is a retrospective hindcast** on ERA5 reanalysis
available after the fact — not an ahead-of-time operational forecast. The
distinction matters and we keep it explicit.

## Quickstart

```bash
git clone https://github.com/naturecipherai/naturecipher-igad-hackathon
cd naturecipher-igad-hackathon
pip install -r requirements.txt
```

**To run the inference pipeline**, request AWS credentials:
📧 **kelvin@naturecipherai.com**

Once received:
```bash
cp credentials.env .env
python -m inference.smoke_test
```

## Tech Stack

- **ML:** XGBoost 2.0.3 (CPU inference)
- **Forecast data:** ECMWF SEAS5 via Copernicus CDS (`seasonal-monthly-single-levels`)
- **Historical data:** ERA5 reanalysis, CHIRPS, MODIS NDVI/LST
- **Runtime:** Python 3.11, xarray, cfgrib, pandas, scipy, boto3
- **Infrastructure:** AWS S3 (model + data store), AWS EC2 (monthly forecast run)
- **Dashboard:** Vanilla HTML/JS/CSS, static hosting

## Data & Model Access

Models and processed data are stored in a private S3 bucket.
Contact kelvin@naturecipherai.com to request read-only credentials.

All datasets are publicly available at their original sources —
see [docs/DATA_ACKNOWLEDGMENTS.md](docs/DATA_ACKNOWLEDGMENTS.md).

## Limitations

- Retrospective validation only; the Jan-Mar 2026 result is a hindcast
- Forward forecasts experimental, single July 2026 SEAS5 initialization
- Soil moisture uses a persistence assumption (SEAS5 provides no SM layers);
  justified by 4-8 week autocorrelation in semi-arid soils (Koster et al. 2004)
- Three ASAL regions, eleven counties
- Ensemble spread is discarded: SEAS5 ships 51 members, the pipeline averages them,
  so every probability is a point estimate with no uncertainty band
- The 0.25 decision threshold was tuned on the same window the metrics report on
- The conditions grid is at native SEAS5 resolution (~100 km) and is deliberately
  not downscaled; there is no per-pixel drought probability, because the bridges
  were fitted on regional means and no pixel-level ground truth exists

## Team

**NatureCipher AI** — Nairobi, Kenya
Kelvin — Founder & CEO
[naturecipherai.com](https://naturecipherai.com)

NVIDIA Inception Program member | 500 Global Pre-Acceleration

## License

Code: MIT License
See [LICENSE](LICENSE) for details.

## Acknowledgments

See [docs/DATA_ACKNOWLEDGMENTS.md](docs/DATA_ACKNOWLEDGMENTS.md) for full
dataset citations as required by hackathon rules.
