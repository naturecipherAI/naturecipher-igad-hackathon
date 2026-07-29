# Cascade Bridge: Satellite-Independent Drought Early Warning

**IGAD Hackathon 2026 — NatureCipher AI**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## What it does

Cascade Bridge is a satellite-independent drought early warning system for
Kenya's Arid and Semi-Arid Lands (ASALs). XGBoost cascade bridge regressors
synthesize CHIRPS precipitation, MODIS NDVI, and Land Surface Temperature
proxies from ERA5/SEAS5 atmospheric inputs alone, enabling drought probability
forecasts at 1-3 month leads without satellite data latency.

**The problem it solves:** Traditional drought early warning depends on satellite
data with days-to-weeks latency and multiple API dependencies. Cascade Bridge
eliminates those dependencies — one atmospheric data source drives the entire
51-feature prediction pipeline.

## Live Dashboard
🌍 **[View forecast →](https://placeholder-update-before-submission.com)**

Issued from July 2026 SEAS5 initialization. Covers August–October 2026
for three ASAL county clusters. Labeled experimental pending full hindcast validation.

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

| Test | Result |
|------|--------|
| Cascade backtest 2021-2024 (asal_north) | Accuracy 0.667, F1 0.692 |
| Retrospective Jan-Mar 2026 (hindcast) | Northeast emergency correctly identified |
| Forward Aug-Oct 2026 | Experimental — mini-hindcast in dashboard |

**Note:** The Jan-Mar 2026 result is a retrospective hindcast using ERA5 reanalysis
data available after the fact, not an ahead-of-time operational forecast.

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

- Retrospective validation only (Jan-Mar 2026 hindcast, not real-time forecast)
- Forward forecasts experimental — mini-hindcast validation only (3 seasons)
- Soil moisture uses persistence assumption (SEAS5 does not provide SM layers)
- Single July 2026 SEAS5 initialization
- Three ASAL regions only (agri-region models not included in this submission)

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
