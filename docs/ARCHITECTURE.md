# System Architecture

## Cascade Bridge: Drought Forecasts Beyond the Satellite Horizon

### The Problem
Drought early warning depends on satellite data (CHIRPS precipitation, MODIS NDVI, Land Surface Temperature) that can only observe months that have already happened. At 1-3 month forecast leads, these observations do not exist yet.

### The Solution
Cascade Bridge trains XGBoost regressors on 35 years of satellite history to synthesize what the satellites *would* see, given only an atmospheric weather forecast:

```
SEAS5 Seasonal Forecast (ERA5-format variables)
  │
  ├── Bridge 1: ERA5 → CHIRPS precipitation proxy
  ├── Bridge 2: ERA5 + CHIRPS proxy → MODIS NDVI proxy
  ├── Bridge 3a: ERA5 + NDVI proxy → LST Day proxy
  └── Bridge 3b: ERA5 + NDVI proxy → LST Night proxy
        │
        └── All 51 features assembled
              │
              └── XGBoost block_3 classifier
                    │
                    └── Drought probability per region per month
                          (threshold: 0.25, recall-optimized)
```

### Regions
Three ASAL (Arid and Semi-Arid Lands) county clusters:
- **ASAL North**: Turkana, Marsabit, Samburu, Isiolo
- **ASAL Northeast**: Wajir, Mandera, Garissa
- **ASAL Eastern**: Kitui, Makueni, Machakos, Tharaka-Nithi

### Data Flow
```
S3 (private)                    Judge machine
─────────────                   ─────────────
processed_v3/    ──────────→   data_loader.py
models/          ──────────→   cascade_bridge.py
                               forecast_runner.py
SEAS5 GRIB       ──────────→   seas5_adapter.py
(CDS download)                        │
                                       ▼
                               forecast.json → dashboard/index.html
```

### Model Details
- Framework: XGBoost 2.0.3 (CPU inference, no GPU required)
- Features: 51 (38 ERA5/CHIRPS + 13 MODIS proxies via bridges)
- Training period: 1990-2024
- Production model: enhanced block_3 (temporal block validation)
- Threshold: 0.25 (maximizes recall for drought detection)
