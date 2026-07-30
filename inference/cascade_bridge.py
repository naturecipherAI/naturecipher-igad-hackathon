"""
Cascade Bridge inference engine.

Synthesizes CHIRPS, NDVI, and LST proxies from ERA5/SEAS5 atmospheric inputs
using three chained XGBoost regressors:

    Bridge 1: ERA5 --> CHIRPS precipitation proxy
    Bridge 2: ERA5 + CHIRPS proxy --> NDVI proxy
    Bridge 3a: ERA5 + NDVI proxy --> LST Day proxy
    Bridge 3b: ERA5 + NDVI proxy --> LST Night proxy

All bridge models are loaded from S3 via data_loader.py at runtime.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats

from inference.data_loader import load_bridge_model, load_bridge_metadata

logger = logging.getLogger(__name__)

# ---- ERA5 features available from SEAS5/FourCastNet forecasts ----
ERA5_CORE_FEATURES = [
    "era5_precip_mm",
    "era5_temp_2m_c",
    "era5_dewpoint_c",
    "era5_vpd_hpa",
    "era5_sm_layer1",
    "era5_sm_layer2",
    "era5_sm_layer3",
    "era5_sm_layer4",
    "era5_sm_anomaly",
    "era5_pet_mm",
    "era5_et_mm",
    "era5_evap_stress",
    "era5_temp_anomaly_c",
    "era5_precip_anomaly",
    "era5_spi3",
    "era5_spi6",
    "era5_spi12",
    "era5_sm_lag1",
    "era5_sm_lag2",
    "sm_trend",
    "season",
    "month",
    "year",
]


# ---- Derived feature helpers ----

def _spi(series: pd.Series, window: int) -> pd.Series:
    """Standardized Precipitation Index via gamma-fitted CDF -> normal quantile."""
    rolling = series.rolling(window=window, min_periods=1).sum()
    spi = pd.Series(np.nan, index=series.index)
    non_zero = rolling[rolling > 0]
    if len(non_zero) < 10:
        return spi
    try:
        shape_p, _, scale_p = stats.gamma.fit(non_zero, floc=0)
        cdf = stats.gamma.cdf(rolling, shape_p, loc=0, scale=scale_p)
        cdf = np.clip(cdf, 0.0001, 0.9999)
        spi_vals = np.clip(stats.norm.ppf(cdf), -3.5, 3.5)
        spi[:] = np.where(np.isinf(spi_vals), np.nan, spi_vals)
    except Exception:
        pass
    return spi


def calculate_spei(precip_mm: pd.Series, pet_mm: pd.Series, window: int) -> pd.Series:
    """Standardized Precipitation-Evapotranspiration Index (log-logistic fit)."""
    wb = precip_mm - pet_mm
    rolling_wb = wb.rolling(window=window, min_periods=1).sum()
    spei = pd.Series(np.nan, index=precip_mm.index)
    try:
        shift = rolling_wb.min() * -1 + 1 if rolling_wb.min() < 0 else 0
        shifted = rolling_wb + shift
        c, loc, scale = stats.fisk.fit(shifted, floc=0)
        cdf = stats.fisk.cdf(shifted, c, loc=0, scale=scale)
        cdf = np.clip(cdf, 0.0001, 0.9999)
        spei_vals = np.clip(stats.norm.ppf(cdf), -3.5, 3.5)
        spei[:] = np.where(np.isinf(spei_vals), np.nan, spei_vals)
    except Exception:
        spei = _spi(rolling_wb, 1)
    return spei


def _climatology_anomaly(series: pd.Series, months: pd.Series) -> pd.Series:
    clim = series.groupby(months).transform("mean")
    return series - clim


def derive_chirps_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive all CHIRPS-dependent features from chirps_precip_mm."""
    p = df["chirps_precip_mm"].copy()
    m = df["month"]
    df["chirps_precip_anomaly"] = _climatology_anomaly(p, m)
    df["chirps_precip_3m_cumul"] = p.rolling(3, min_periods=1).sum()
    df["chirps_precip_6m_cumul"] = p.rolling(6, min_periods=1).sum()
    for w in [1, 3, 6, 12]:
        df[f"chirps_spi{w}"] = _spi(p, w)
    for lag in [1, 2, 3]:
        df[f"chirps_spi3_lag{lag}"] = df["chirps_spi3"].shift(lag)
    # Rolling slope of SPI-3 (trend)
    slopes = pd.Series(np.nan, index=df.index)
    for i in range(2, len(df)):
        y = df["chirps_spi3"].iloc[i - 2: i + 1].values
        if not np.isnan(y).any():
            slopes.iloc[i] = float(np.polyfit(np.arange(3), y, 1)[0])
    df["spi3_trend"] = slopes
    return df


def derive_ndvi_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive all NDVI-dependent features from ndvi_mean."""
    n = df["ndvi_mean"]
    m = df["month"]
    df["ndvi_anomaly"] = _climatology_anomaly(n, m)
    n_min = df.groupby("month")["ndvi_mean"].transform("min")
    n_max = df.groupby("month")["ndvi_mean"].transform("max")
    denom = (n_max - n_min).replace(0, np.nan)
    df["vci"] = ((n - n_min) / denom * 100).clip(0, 100)
    df["ndvi_min"] = n.rolling(3, min_periods=1).min()
    df["ndvi_max"] = n.rolling(3, min_periods=1).max()
    df["ndvi_std"] = n.rolling(3, min_periods=1).std()
    return df


def derive_lst_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive all LST-dependent features from lst_day_c / lst_night_c."""
    if "lst_night_c" in df.columns:
        df["lst_mean_c"] = (df["lst_day_c"] + df["lst_night_c"]) / 2
        ref = df["lst_mean_c"]
    else:
        ref = df["lst_day_c"]
    m = df["month"]
    df["lst_anomaly_c"] = _climatology_anomaly(ref, m)
    ref_name = ref.name if hasattr(ref, "name") else "lst_day_c"
    lst_max = df.groupby("month")[ref_name].transform("max")
    lst_min = df.groupby("month")[ref_name].transform("min")
    denom = (lst_max - lst_min).replace(0, np.nan)
    df["tci"] = ((lst_max - ref) / denom * 100).clip(0, 100)
    return df


# ---- Inference engine ----

class CascadeBridgeInference:
    """
    Run the cascade bridges on ERA5 forecast features to produce all 51
    drought model inputs.

    Models are downloaded from S3 on first use and cached for the session.
    """

    def __init__(self, region: str):
        self.region = region
        self._load_bridges()

    def _load_bridges(self):
        """Download bridge models from S3 and load into XGBoost."""
        logger.info(f"Loading cascade bridges for {self.region} from S3...")

        # Bridge 1: ERA5 -> CHIRPS
        path_b1 = load_bridge_model(self.region, "bridge1_chirps")
        self.b1 = xgb.XGBRegressor()
        self.b1.load_model(path_b1)
        os.unlink(path_b1)

        # Bridge 2: ERA5 + CHIRPS -> NDVI
        path_b2 = load_bridge_model(self.region, "bridge2_ndvi")
        self.b2 = xgb.XGBRegressor()
        self.b2.load_model(path_b2)
        os.unlink(path_b2)

        # Bridge 3a: ERA5 + NDVI -> LST day
        path_b3a = load_bridge_model(self.region, "bridge3a_lst_day")
        self.b3a = xgb.XGBRegressor()
        self.b3a.load_model(path_b3a)
        os.unlink(path_b3a)

        # Bridge 3b: ERA5 + NDVI -> LST night (optional)
        try:
            path_b3b = load_bridge_model(self.region, "bridge3b_lst_night")
            self.b3b = xgb.XGBRegressor()
            self.b3b.load_model(path_b3b)
            os.unlink(path_b3b)
        except Exception:
            self.b3b = None
            logger.info("  Bridge 3b (LST night) not available; will approximate from day LST")

        self.meta = load_bridge_metadata(self.region)
        logger.info(f"  All bridges loaded for {self.region}")

    def _get_inputs(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        available = [c for c in cols if c in df.columns]
        X = df[available].copy()
        X = X.fillna(X.median())
        return X

    def predict(self, era5_df: pd.DataFrame) -> pd.DataFrame:
        """
        Run cascade bridges on a DataFrame of ERA5 forecast features.

        Args:
            era5_df: DataFrame with columns matching ERA5_CORE_FEATURES,
                     sorted by (year, month). Can be multi-row (time series)
                     so rolling features (SPI, cumul, lags) compute correctly.

        Returns:
            DataFrame with all 51 features filled in.
        """
        df = era5_df.copy().sort_values(["year", "month"]).reset_index(drop=True)

        # Bridge 1: predict chirps_precip_mm
        X1 = self._get_inputs(df, self.meta["bridge1_inputs"])
        df["chirps_precip_mm"] = self.b1.predict(X1).clip(0)

        # Derive all CHIRPS-dependent features
        df = derive_chirps_features(df)

        # Cross-source SPI difference
        if "era5_spi12" in df.columns and "chirps_spi12" in df.columns:
            df["spi12_difference"] = df["chirps_spi12"] - df["era5_spi12"]

        # SPEI (needs PET from ERA5)
        if "era5_pet_mm" in df.columns:
            for w in [3, 6, 12]:
                df[f"spei{w}"] = calculate_spei(
                    df["chirps_precip_mm"], df["era5_pet_mm"], w
                )

        # Bridge 2: predict ndvi_mean
        X2 = self._get_inputs(df, self.meta["bridge2_inputs"])
        df["ndvi_mean"] = self.b2.predict(X2).clip(-1, 1)
        df = derive_ndvi_features(df)

        # Bridge 3: predict LST
        X3 = self._get_inputs(df, self.meta["bridge3_inputs"])
        df["lst_day_c"] = self.b3a.predict(X3)
        if self.b3b is not None:
            df["lst_night_c"] = self.b3b.predict(X3)
        else:
            df["lst_night_c"] = df["lst_day_c"] - 10.0
        df = derive_lst_features(df)

        return df
