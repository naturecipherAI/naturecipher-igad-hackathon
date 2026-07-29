# Methods

## Forecast Data Source
ECMWF SEAS5 seasonal forecasts via Copernicus Climate Data Store (CDS).
Dataset: `seasonal-monthly-single-levels`, originating_centre=ecmwf, system=51.
Initialization: July 2026. Lead times: months 2-4 (August, September, October 2026).
Ensemble: 51 members.

## Variable Mapping

| ERA5 Feature | SEAS5 Variable | Transformation |
|---|---|---|
| era5_precip_mm | tprate (m/s) | × days × 86400 × 1000 |
| era5_temp_2m_c | t2m (K) | − 273.15 |
| era5_dewpoint_c | d2m (K) | − 273.15 |
| era5_vpd_hpa | t2m + d2m | August-Roche-Magnus |
| era5_et_mm | e (evaporation) | unit conversion |
| era5_runoff_mm | ro (runoff) | unit conversion |
| era5_pet_mm | t2m | Hargreaves method |
| era5_sm_layer1-4 | **NOT in SEAS5** | Persistence (see below) |

## Soil Moisture Persistence (Disclosed Assumption)
SEAS5 single-level monthly output does not include volumetric soil moisture layers.
We apply persistence: soil moisture layers 1-4 are held at their last observed
ERA5 values (most recent month in processed history). sm_anomaly is recomputed
against the ERA5 climatology baseline.

Justification: Soil moisture in semi-arid soils has autocorrelation timescales
of 4-8 weeks (Koster et al. 2004), making persistence a reasonable approximation
at 1-3 month leads. This assumption is disclosed and will be replaced with SEAS5
soil moisture in future versions when available.

## Bias Correction
Anomaly-mapping approach: SEAS5 anomaly vs SEAS5 hindcast climatology (1993-2016,
July initialization) is computed, then added to the ERA5 climatology from
processed history. This corrects for SEAS5 systematic biases while preserving
the forecast signal.

## Drought Threshold
Probability threshold: **0.25** (not default 0.5).
Chosen to maximize recall (drought detection rate) over precision.
In anticipatory action contexts, missed droughts are more costly than false alarms.

## Validation

### Cascade Bridge Backtest (2021-2024, asal_north)
- Accuracy: 0.667
- F1: 0.692
- Recall: 0.643

### Retrospective Validation (January-March 2026)
The cascade bridge pipeline was run on ERA5 reanalysis data for Jan-Mar 2026
(available via ~2 month lag by May 2026) as a stand-in for SEAS5 forecast fields.
Results correctly identified the northeast Kenya drought emergency declared by
IPC in March 2026.

**Important:** This is a retrospective hindcast, not an ahead-of-time operational
forecast. ERA5 data for those months was used after the fact. Forward forecasting
using SEAS5 is the subject of ongoing validation.

### Forward Forecast Status
August-October 2026 forecasts in this submission are **experimental**.
Mini-hindcast validation (3 July initializations: 2023, 2024, 2025) is
included in the dashboard. Full multi-year hindcast validation is in progress.
