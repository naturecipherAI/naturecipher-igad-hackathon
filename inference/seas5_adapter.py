"""
SEAS5 seasonal forecast adapter.

Maps ECMWF SEAS5 GRIB variables to ERA5_CORE_FEATURES for cascade bridge input.
Handles bias correction against 1993-2016 hindcast climatology.
Soil moisture uses persistence fill (documented assumption).

SEAS5 GRIB files stored on S3:
    s3://naturecipher-forecast/seas5/seas5_2026_07_forecast.grib
    s3://naturecipher-forecast/seas5/seas5_1993-2016_07_hindcast_climatology.grib

SEAS5 GRIB structure (from CDS seasonal-monthly-single-levels):
    Variables: t2m, d2m, tprate, erate, mrort
    Dims: (number=51 ensemble, [time=N init dates], step=3 lead months, lat, lon)
    valid_time: (step,) for forecast, (time, step) for climatology
"""

import logging
import os
import tempfile
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# SEAS5 variable name -> (ERA5 feature name, unit conversion)
# tprate = total precip rate (m/s -> mm/month, approx *86400*30*1000)
# erate  = evaporation rate (m/s -> mm/month, same, sign flip)
# mrort  = mean rate of runoff (m/s -> mm/month)
SECONDS_PER_MONTH = 86400.0 * 30.0

SEAS5_VAR_MAP = {
    "t2m":    ("era5_temp_2m_c",  lambda x: x - 273.15),
    "d2m":    ("era5_dewpoint_c", lambda x: x - 273.15),
    "tprate": ("era5_precip_mm",  lambda x: max(x * SECONDS_PER_MONTH * 1000, 0)),
    "erate":  ("era5_et_mm",      lambda x: abs(x) * SECONDS_PER_MONTH * 1000),
    "mrort":  ("era5_pet_mm",     lambda x: abs(x) * SECONDS_PER_MONTH * 1000),
}

# Season encoding matching training pipeline
MONTH_TO_SEASON = {
    1: 0, 2: 0,                  # JF
    3: 1, 4: 1, 5: 1,            # MAM
    6: 2, 7: 2, 8: 2, 9: 2,     # JJAS
    10: 3, 11: 3, 12: 3,         # OND
}


def _download_grib_from_s3(s3_key: str) -> str:
    """Download a GRIB file from S3 to a temp file."""
    from inference.data_loader import _s3_client, _bucket

    s3 = _s3_client()
    tmp = tempfile.NamedTemporaryFile(suffix=".grib", delete=False, prefix="seas5_")
    s3.download_fileobj(_bucket(), s3_key, tmp)
    tmp.close()
    logger.info(f"Downloaded s3://{_bucket()}/{s3_key} -> {tmp.name}")
    return tmp.name


def load_seas5_grib(grib_path: str) -> pd.DataFrame:
    """
    Read a SEAS5 GRIB file and extract spatial+ensemble-mean monthly values.

    Returns:
        DataFrame with columns: year, month, t2m, d2m, tprate, erate, mrort
        One row per valid month (lead time step).
        For climatology files (multiple init dates), returns one row per
        unique (year, month) after averaging across init dates.
    """
    import cfgrib

    datasets = cfgrib.open_datasets(grib_path)
    records = []

    for ds in datasets:
        # Spatial mean over the ASAL bounding box
        ds_mean = ds.mean(dim=["latitude", "longitude"])

        # Ensemble mean
        if "number" in ds_mean.dims:
            ds_mean = ds_mean.mean(dim="number")

        # Get valid_time array
        vt = ds_mean.coords["valid_time"].values

        # Handle time dimension (present in climatology, absent in single forecast)
        has_time = "time" in ds_mean.dims

        if has_time:
            n_times = ds_mean.sizes["time"]
            n_steps = ds_mean.sizes["step"]
            for ti in range(n_times):
                for si in range(n_steps):
                    sel = ds_mean.isel(time=ti, step=si)
                    valid = pd.Timestamp(vt[ti, si])
                    row = {"year": valid.year, "month": valid.month}
                    for var in ds_mean.data_vars:
                        row[str(var)] = float(sel[var].values)
                    records.append(row)
        else:
            n_steps = ds_mean.sizes["step"]
            for si in range(n_steps):
                sel = ds_mean.isel(step=si)
                valid = pd.Timestamp(vt[si])
                row = {"year": valid.year, "month": valid.month}
                for var in ds_mean.data_vars:
                    row[str(var)] = float(sel[var].values)
                records.append(row)

    df = pd.DataFrame(records)
    logger.info(f"Parsed GRIB: {len(df)} rows, months={sorted(df['month'].unique())}")
    return df


def build_climatology(climatology_grib_path: str) -> Dict[int, Dict[str, float]]:
    """
    Build monthly climatology means from hindcast GRIB (1993-2016).
    Returns: {month: {var_name: mean_value}}
    """
    df = load_seas5_grib(climatology_grib_path)
    clim = {}
    for month, group in df.groupby("month"):
        clim[int(month)] = {
            col: float(group[col].mean())
            for col in group.columns if col not in ("year", "month")
        }
    logger.info(f"Climatology built for months: {sorted(clim.keys())}")
    return clim


def bias_correct(forecast_df: pd.DataFrame, climatology: Dict[int, Dict[str, float]]) -> pd.DataFrame:
    """
    Simple multiplicative bias correction for rate variables.
    Temperature is kept as-is (SEAS5 t2m is already calibrated).
    """
    df = forecast_df.copy()
    for idx, row in df.iterrows():
        month = int(row["month"])
        if month not in climatology:
            continue
        clim = climatology[month]
        for var in ("tprate", "erate", "mrort"):
            if var in df.columns and var in clim and clim[var] != 0:
                ratio = row[var] / clim[var]
                # Clamp extreme ratios to avoid blow-up
                df.at[idx, var] = row[var] * min(max(ratio, 0.5), 2.0)
    return df


def seas5_to_era5_features(
    seas5_df: pd.DataFrame,
    historical_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Convert SEAS5 raw variables to ERA5_CORE_FEATURES format.

    Args:
        seas5_df: DataFrame from load_seas5_grib (raw SEAS5 values).
        historical_df: Optional historical DataFrame for anomalies/SPI.

    Returns:
        DataFrame with ERA5_CORE_FEATURES columns ready for cascade bridge.
    """
    from inference.cascade_bridge import ERA5_CORE_FEATURES

    rows = []
    for _, raw in seas5_df.iterrows():
        feat = {
            "year": int(raw["year"]),
            "month": int(raw["month"]),
            "season": MONTH_TO_SEASON.get(int(raw["month"]), 2),
        }

        # Apply unit conversions
        for seas5_var, (era5_name, convert) in SEAS5_VAR_MAP.items():
            if seas5_var in raw and not pd.isna(raw[seas5_var]):
                feat[era5_name] = convert(float(raw[seas5_var]))

        # VPD: derived from temperature and dewpoint
        t = feat.get("era5_temp_2m_c")
        td = feat.get("era5_dewpoint_c")
        if t is not None and td is not None:
            sat_vp = 6.1078 * np.exp(17.27 * t / (t + 237.3))
            act_vp = 6.1078 * np.exp(17.27 * td / (td + 237.3))
            feat["era5_vpd_hpa"] = max(sat_vp - act_vp, 0)

        # Evaporative stress
        et = feat.get("era5_et_mm", 0)
        pet = feat.get("era5_pet_mm", 0)
        feat["era5_evap_stress"] = et / pet if pet > 0 else 1.0

        rows.append(feat)

    df = pd.DataFrame(rows)

    # Soil moisture: SEAS5 does not include SM directly.
    # Persist from last historical observation (documented assumption).
    sm_cols = [
        "era5_sm_layer1", "era5_sm_layer2", "era5_sm_layer3", "era5_sm_layer4",
        "era5_sm_anomaly", "era5_sm_lag1", "era5_sm_lag2", "sm_trend",
    ]
    if historical_df is not None and len(historical_df) > 0:
        last_row = historical_df.sort_values(["year", "month"]).iloc[-1]
        for col in sm_cols:
            df[col] = last_row[col] if col in last_row.index else 0.0
    else:
        for col in sm_cols:
            df[col] = 0.0

    # Anomalies and SPI relative to historical climatology
    if historical_df is not None and len(historical_df) > 0:
        hist = historical_df.copy()
        for base_col, anom_col in [
            ("era5_temp_2m_c", "era5_temp_anomaly_c"),
            ("era5_precip_mm", "era5_precip_anomaly"),
        ]:
            if base_col in hist.columns and base_col in df.columns:
                month_means = hist.groupby("month")[base_col].mean()
                df[anom_col] = df.apply(
                    lambda r: r.get(base_col, 0) - month_means.get(int(r["month"]), 0),
                    axis=1,
                )
            else:
                df[anom_col] = 0.0

        # SPI: z-score approximation for forecast months
        if "era5_precip_mm" in hist.columns and "era5_precip_mm" in df.columns:
            month_stats = hist.groupby("month")["era5_precip_mm"].agg(["mean", "std"])
            for spi_col in ["era5_spi3", "era5_spi6", "era5_spi12"]:
                df[spi_col] = df.apply(
                    lambda r: (
                        (r["era5_precip_mm"] - month_stats.loc[int(r["month"]), "mean"])
                        / max(month_stats.loc[int(r["month"]), "std"], 0.01)
                        if int(r["month"]) in month_stats.index else 0
                    ),
                    axis=1,
                ).clip(-3.5, 3.5)
        else:
            for col in ["era5_spi3", "era5_spi6", "era5_spi12"]:
                df[col] = 0.0
    else:
        for col in ["era5_temp_anomaly_c", "era5_precip_anomaly",
                     "era5_spi3", "era5_spi6", "era5_spi12"]:
            df[col] = 0.0

    # Fill any remaining missing ERA5_CORE_FEATURES with 0
    for col in ERA5_CORE_FEATURES:
        if col not in df.columns:
            df[col] = 0.0

    return df


def load_and_convert(
    forecast_s3_key: str = "seas5/seas5_2026_07_forecast.grib",
    climatology_s3_key: str = "seas5/seas5_1993-2016_07_hindcast_climatology.grib",
    historical_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Full pipeline: download SEAS5 GRIBs from S3, bias-correct, convert to ERA5 features.

    Returns:
        DataFrame with ERA5_CORE_FEATURES ready for cascade bridge.
    """
    forecast_path = _download_grib_from_s3(forecast_s3_key)
    clim_path = _download_grib_from_s3(climatology_s3_key)

    try:
        forecast_raw = load_seas5_grib(forecast_path)
        climatology = build_climatology(clim_path)
        forecast_bc = bias_correct(forecast_raw, climatology)
        era5_df = seas5_to_era5_features(forecast_bc, historical_df)
        logger.info(f"SEAS5 adapter produced {len(era5_df)} forecast rows")
        return era5_df
    finally:
        os.unlink(forecast_path)
        os.unlink(clim_path)
