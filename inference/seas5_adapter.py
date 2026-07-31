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

import atexit
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

FORECAST_GRIB_KEY = "seas5/seas5_2026_07_forecast.grib"
CLIMATOLOGY_GRIB_KEY = "seas5/seas5_1993-2016_07_hindcast_climatology.grib"

SEAS5_VAR_MAP = {
    "t2m":    ("era5_temp_2m_c",  lambda x: x - 273.15),
    "d2m":    ("era5_dewpoint_c", lambda x: x - 273.15),
    "tprate": ("era5_precip_mm",  lambda x: max(x * SECONDS_PER_MONTH * 1000, 0)),
    "erate":  ("era5_et_mm",      lambda x: abs(x) * SECONDS_PER_MONTH * 1000),
    "mrort":  ("era5_runoff_mm",  lambda x: abs(x) * SECONDS_PER_MONTH * 1000),
}

# Days per month, used by Hargreaves and by the rate-to-total conversions. The
# flat 30-day SECONDS_PER_MONTH above is retained only where a total is later
# rescaled by an anomaly ratio, which cancels the error.
DAYS_IN_MONTH = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
                 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}

# Mean monthly extraterrestrial radiation (MJ m-2 day-1) near the equator, which
# is where all three ASAL clusters sit. Hargreaves needs Ra; at these latitudes it
# varies little through the year, so a single equatorial profile is adequate and
# avoids carrying a latitude through the regional (spatially averaged) path.
RA_EQUATORIAL = {1: 35.2, 2: 36.4, 3: 36.7, 4: 35.5, 5: 33.5, 6: 32.3,
                 7: 32.7, 8: 34.2, 9: 35.8, 10: 36.0, 11: 35.1, 12: 34.6}


def _anchor_on_observed(
    df: pd.DataFrame,
    historical_df: pd.DataFrame,
    ratio_col: str,
    era5_col: str,
) -> pd.DataFrame:
    """
    Rescale one forecast column so its anomaly sits on the observed climatology.

    Months with no observed counterpart keep their converted SEAS5 value rather
    than becoming NaN, so a partial history degrades the correction instead of
    dropping the forecast.
    """
    if ratio_col not in df.columns or era5_col not in historical_df.columns:
        return df

    obs_clim = historical_df.groupby("month")[era5_col].mean()
    corrected = []
    for month, ratio, fallback in zip(df["month"], df[ratio_col], df[era5_col]):
        month = int(month)
        if month in obs_clim.index:
            corrected.append(float(obs_clim.loc[month]) * float(ratio))
        else:
            corrected.append(fallback)

    return df.assign(**{era5_col: corrected})


SPI_WINDOWS = {3: "era5_spi3", 6: "era5_spi6", 12: "era5_spi12"}
SPI_CLAMP = 3.5
MONTH_KEY = ["year", "month"]


def _precip_timeline(hist: pd.DataFrame, forecast: pd.DataFrame) -> pd.DataFrame:
    """
    Observed history followed by the forecast months, as one continuous series.

    An N-month accumulation on an early lead is mostly observed history, so the
    two have to be concatenated before the rolling sum rather than after.
    """
    cols = MONTH_KEY + ["era5_precip_mm"]
    joined = pd.concat([hist[cols], forecast[cols]], ignore_index=True)
    joined = joined.drop_duplicates(subset=MONTH_KEY, keep="last")
    return joined.sort_values(MONTH_KEY).reset_index(drop=True)


def _spi_over_window(timeline: pd.DataFrame, window: int, target: pd.DataFrame) -> List[float]:
    """
    Standardised precipitation index at one accumulation window.

    Standardised per calendar month, so a dry October is scored against other
    Octobers rather than against the annual mean.
    """
    accum = timeline["era5_precip_mm"].rolling(window, min_periods=1).sum()
    frame = timeline[MONTH_KEY].assign(accum=accum)
    stats = frame.groupby("month")["accum"].agg(["mean", "std"])

    scored = {}
    for year, month, total in zip(frame["year"], frame["month"], frame["accum"]):
        month = int(month)
        if month not in stats.index:
            continue
        spread = max(float(stats.loc[month, "std"]), 0.01)
        z = (float(total) - float(stats.loc[month, "mean"])) / spread
        scored[(int(year), month)] = float(np.clip(z, -SPI_CLAMP, SPI_CLAMP))

    return [
        scored.get((int(y), int(m)), 0.0)
        for y, m in zip(target["year"], target["month"])
    ]


def _hargreaves_pet_mm(temp_c: float, month: int) -> float:
    """
    Monthly potential evapotranspiration by the Hargreaves method.

    Uses the temperature-only form, since SEAS5 monthly single-level output gives
    no daily temperature range. The diurnal range is approximated at 12 C, typical
    of semi-arid East Africa.
    """
    days = DAYS_IN_MONTH.get(int(month), 30)
    ra = RA_EQUATORIAL.get(int(month), 34.5)
    daily = 0.0023 * (temp_c + 17.8) * (12.0 ** 0.5) * ra * 0.408
    return max(daily, 0.0) * days

# Season encoding matching training pipeline
MONTH_TO_SEASON = {
    1: 0, 2: 0,                  # JF
    3: 1, 4: 1, 5: 1,            # MAM
    6: 2, 7: 2, 8: 2, 9: 2,     # JJAS
    10: 3, 11: 3, 12: 3,         # OND
}


_GRIB_CACHE: Dict[str, str] = {}


def clear_grib_cache() -> None:
    """Delete every cached GRIB. Registered at exit; safe to call at any point."""
    for key, path in list(_GRIB_CACHE.items()):
        try:
            os.unlink(path)
        except OSError:
            pass
        _GRIB_CACHE.pop(key, None)


atexit.register(clear_grib_cache)


def _download_grib_from_s3(s3_key: str) -> str:
    """
    Download a GRIB from S3, or return the copy already on disk for this key.

    A single run resolves the same two GRIB keys repeatedly: load_and_convert is
    called once per region and load_grid_field once more, so an uncached run fetched
    the same two files eight times.

    The cache owns the file. Callers must not unlink the returned path, because the
    next caller gets the same one. Cleanup happens in clear_grib_cache at exit.
    """
    from inference.data_loader import _s3_client, _bucket

    cached = _GRIB_CACHE.get(s3_key)
    if cached and os.path.exists(cached):
        logger.info(f"Reusing cached GRIB for {s3_key} -> {cached}")
        return cached
    if cached:
        # Removed from underneath us; drop the stale entry and fetch again.
        _GRIB_CACHE.pop(s3_key, None)

    s3 = _s3_client()
    tmp = tempfile.NamedTemporaryFile(suffix=".grib", delete=False, prefix="seas5_")
    try:
        s3.download_fileobj(_bucket(), s3_key, tmp)
        tmp.close()
    except Exception:
        # A partial file must never be cached, or every later call reads a truncated GRIB.
        tmp.close()
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        raise

    _GRIB_CACHE[s3_key] = tmp.name
    logger.info(f"Downloaded s3://{_bucket()}/{s3_key} -> {tmp.name}")
    return tmp.name


CELL_KEY = ["year", "month", "lat", "lon"]


def _normalise_lon(lon: float) -> float:
    """ECMWF grids are often 0-360; the dashboard and GeoJSON expect -180-180."""
    return lon - 360.0 if lon > 180.0 else lon


def _cells(sel, valid: pd.Timestamp) -> pd.DataFrame:
    frame = sel.to_dataframe().reset_index()
    frame = frame.rename(columns={"latitude": "lat", "longitude": "lon"})
    keep = [c for c in frame.columns if c in SEAS5_VAR_MAP]
    frame = frame[["lat", "lon"] + keep].copy()
    frame["lon"] = frame["lon"].map(_normalise_lon)
    frame["year"] = valid.year
    frame["month"] = valid.month
    return frame[CELL_KEY + keep]


def load_seas5_grib_grid(grib_path: str) -> pd.DataFrame:
    """
    Read a SEAS5 GRIB and keep the native grid instead of collapsing it.

    Ensemble members are still averaged; only the spatial mean is dropped.

    Returns:
        DataFrame with one row per (year, month, lat, lon) and one column per
        SEAS5 variable present in the file.
    """
    import cfgrib

    frames = []
    for ds in cfgrib.open_datasets(grib_path):
        if "number" in ds.dims:
            ds = ds.mean(dim="number")

        vt = ds.coords["valid_time"].values
        has_time = "time" in ds.dims

        if has_time:
            for ti in range(ds.sizes["time"]):
                for si in range(ds.sizes["step"]):
                    frames.append(_cells(ds.isel(time=ti, step=si), pd.Timestamp(vt[ti, si])))
        else:
            for si in range(ds.sizes["step"]):
                frames.append(_cells(ds.isel(step=si), pd.Timestamp(vt[si])))

    grid = pd.concat(frames, ignore_index=True)

    # cfgrib splits a file into several datasets by grib attributes, so one cell can
    # arrive across multiple frames with a different variable filled in each. Collapse
    # on the cell key; first() skips NaN, so the variables merge rather than duplicate.
    grid = grid.groupby(CELL_KEY, as_index=False).first()

    logger.info(
        f"Parsed GRIB grid: {len(grid)} cell-months, "
        f"{grid[['lat', 'lon']].drop_duplicates().shape[0]} cells, "
        f"months={sorted(grid['month'].unique())}"
    )
    return grid


def load_seas5_grib(grib_path: str) -> pd.DataFrame:
    """
    Read a SEAS5 GRIB file and extract spatial+ensemble-mean monthly values.

    Derived from load_seas5_grib_grid so the gridded and regional paths cannot
    drift apart. Contract is unchanged: one row per valid (year, month).

    Returns:
        DataFrame with columns: year, month, t2m, d2m, tprate, erate, mrort
    """
    grid = load_seas5_grib_grid(grib_path)
    value_cols = [c for c in grid.columns if c not in CELL_KEY]
    df = grid.groupby(["year", "month"], as_index=False)[value_cols].mean()
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


RATE_VARS = ("tprate", "erate", "mrort")


ANOMALY_CLAMP = (0.5, 2.0)
ANOM_SUFFIX = "_anom_ratio"

# SEAS5 variable -> the ERA5 feature whose observed climatology anchors its anomaly.
ANOM_TARGET = {
    "tprate": "era5_precip_mm",
    "erate": "era5_et_mm",
}


def _anomaly_ratio(value: float, model_clim: float) -> float:
    """
    The forecast's departure from its own model climatology, as a ratio.

    This is the quantity SEAS5 actually predicts well. Its absolute magnitude is
    biased, its anomaly is not, which is why bias correction transplants the
    anomaly onto an observed climatology rather than rescaling the raw value.

    Clamped because a near-zero model climatology in an arid month makes the raw
    ratio explode; the clamp bounds a single dry cell's influence.
    """
    if model_clim == 0:
        return 1.0
    lo, hi = ANOMALY_CLAMP
    return min(max(value / model_clim, lo), hi)


def bias_correct(forecast_df: pd.DataFrame, climatology: Dict[int, Dict[str, float]]) -> pd.DataFrame:
    """
    Simple multiplicative bias correction for rate variables.
    Temperature is kept as-is (SEAS5 t2m is already calibrated).
    """
    df = forecast_df.copy()
    for var in RATE_VARS:
        df[var + ANOM_SUFFIX] = 1.0

    for idx, row in df.iterrows():
        month = int(row["month"])
        if month not in climatology:
            continue
        clim = climatology[month]
        for var in RATE_VARS:
            if var in df.columns and var in clim:
                df.at[idx, var + ANOM_SUFFIX] = _anomaly_ratio(row[var], clim[var])
    return df


def build_climatology_grid(climatology_grib_path: str) -> pd.DataFrame:
    """
    Per-cell monthly climatology from the hindcast GRIB, averaged over init years.

    Returns:
        DataFrame keyed on (month, lat, lon) with one column per SEAS5 variable.
    """
    grid = load_seas5_grib_grid(climatology_grib_path)
    value_cols = [c for c in grid.columns if c not in CELL_KEY]
    clim = grid.groupby(["month", "lat", "lon"], as_index=False)[value_cols].mean()
    logger.info(
        f"Grid climatology: {len(clim)} cell-months over months {sorted(clim['month'].unique())}"
    )
    return clim


def bias_correct_grid(forecast_grid: pd.DataFrame, climatology_grid: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same multiplicative factor as bias_correct, matched cell by cell.

    Cells with no climatology counterpart are left uncorrected rather than dropped,
    so a partial climatology cannot silently shrink the map.
    """
    clim = climatology_grid.rename(columns={c: c + "_clim" for c in climatology_grid.columns
                                            if c not in ("month", "lat", "lon")})
    df = forecast_grid.merge(clim, on=["month", "lat", "lon"], how="left")

    for var in RATE_VARS:
        col, clim_col = var, var + "_clim"
        if col in df.columns and clim_col in df.columns:
            factors = [
                1.0 if pd.isna(c) else _bias_factor(float(v), float(c))
                for v, c in zip(df[col], df[clim_col])
            ]
            df[col] = df[col] * pd.Series(factors, index=df.index)

    uncorrected = int(df[[v + "_clim" for v in RATE_VARS if v + "_clim" in df.columns]]
                      .isna().all(axis=1).sum()) if len(df) else 0
    if uncorrected:
        logger.warning(f"{uncorrected} grid cell-months had no climatology match, left uncorrected")

    return df.drop(columns=[c for c in df.columns if c.endswith("_clim")])


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

        # PET via Hargreaves, as documented in METHODS.md. The previous mapping took
        # mrort (mean runoff rate) as PET, which is a different physical quantity:
        # in the ASALs runoff is near zero while PET is high, so SPEI's water balance
        # (precip - PET) was computed against roughly nothing.
        if t is not None:
            feat["era5_pet_mm"] = _hargreaves_pet_mm(t, feat["month"])
        if "mrort" in raw and not pd.isna(raw["mrort"]):
            feat["era5_runoff_mm"] = abs(float(raw["mrort"])) * SECONDS_PER_MONTH * 1000

        for var in RATE_VARS:
            ratio_col = var + ANOM_SUFFIX
            if ratio_col in raw and not pd.isna(raw[ratio_col]):
                feat[ratio_col] = float(raw[ratio_col])

        # Evaporative stress
        et = feat.get("era5_et_mm", 0)
        pet = feat.get("era5_pet_mm", 0)
        feat["era5_evap_stress"] = et / pet if pet > 0 else 1.0

        rows.append(feat)

    df = pd.DataFrame(rows)

    # Anchor each anomaly on the observed (ERA5) climatology. The ratio is
    # dimensionless, so it transfers across the SEAS5 -> ERA5 unit change; this is
    # the step that makes the correction remove bias rather than repeat it.
    if historical_df is not None and len(historical_df) > 0:
        for seas5_var, era5_col in ANOM_TARGET.items():
            df = _anchor_on_observed(df, historical_df, seas5_var + ANOM_SUFFIX, era5_col)

    df = df.drop(columns=[c for c in df.columns if c.endswith(ANOM_SUFFIX)])

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

        # SPI over real accumulation windows. Previously SPI-3, SPI-6 and SPI-12
        # were the same single-month z-score, so three features that should carry
        # different timescales carried one value; a 12-month deficit was invisible.
        has_precip = "era5_precip_mm" in hist.columns and "era5_precip_mm" in df.columns
        if has_precip:
            timeline = _precip_timeline(hist, df)
            for window, spi_col in SPI_WINDOWS.items():
                df[spi_col] = _spi_over_window(timeline, window, df)
        else:
            for spi_col in SPI_WINDOWS.values():
                df[spi_col] = 0.0
    else:
        for col in ["era5_temp_anomaly_c", "era5_precip_anomaly",
                     "era5_spi3", "era5_spi6", "era5_spi12"]:
            df[col] = 0.0

    # Fill any remaining missing ERA5_CORE_FEATURES with 0
    for col in ERA5_CORE_FEATURES:
        if col not in df.columns:
            df[col] = 0.0

    return df


def _grid_spacing(values: List[float]) -> Optional[float]:
    uniq = sorted(set(round(float(v), 6) for v in values))
    if len(uniq) < 2:
        return None
    diffs = [round(b - a, 6) for a, b in zip(uniq, uniq[1:])]
    return float(min(diffs))


def grid_anomaly_field(
    forecast_grid: pd.DataFrame,
    climatology_grid: pd.DataFrame,
) -> pd.DataFrame:
    """
    Per-cell forecast conditions expressed against that cell's own hindcast climatology.

    Each cell is compared with itself across 1993-2016, never with the regional mean.
    Differencing a cell against a regional climatology would turn a permanent spatial
    gradient into an apparent anomaly, so a persistently dry cell would read as drought
    every month regardless of the forecast.

    This is a conditions field, not a drought probability. The bridges and the
    classifier were fitted on regionally averaged inputs, so running them per cell
    would be out-of-distribution inference with no pixel-level labels to validate it.

    Returns:
        DataFrame keyed on (year, month, lat, lon) with precipitation and temperature
        anomalies. Cells lacking a climatology counterpart carry null anomalies.
    """
    clim = climatology_grid.rename(columns={c: c + "_clim" for c in climatology_grid.columns
                                            if c not in ("month", "lat", "lon")})
    df = forecast_grid.merge(clim, on=["month", "lat", "lon"], how="left")

    out = df[CELL_KEY].copy()

    if "tprate" in df.columns:
        to_mm = SEAS5_VAR_MAP["tprate"][1]
        out["precip_mm"] = [to_mm(float(v)) if pd.notna(v) else np.nan for v in df["tprate"]]
        if "tprate_clim" in df.columns:
            out["precip_clim_mm"] = [
                to_mm(float(v)) if pd.notna(v) else np.nan for v in df["tprate_clim"]
            ]
            out["precip_anomaly_mm"] = out["precip_mm"] - out["precip_clim_mm"]
            out["precip_pct_of_normal"] = np.where(
                out["precip_clim_mm"] > 0,
                out["precip_mm"] / out["precip_clim_mm"] * 100.0,
                np.nan,
            )

    if "t2m" in df.columns:
        to_c = SEAS5_VAR_MAP["t2m"][1]
        out["temp_2m_c"] = [to_c(float(v)) if pd.notna(v) else np.nan for v in df["t2m"]]
        if "t2m_clim" in df.columns:
            clim_c = pd.Series(
                [to_c(float(v)) if pd.notna(v) else np.nan for v in df["t2m_clim"]],
                index=df.index,
            )
            out["temp_anomaly_c"] = out["temp_2m_c"] - clim_c

    return out.sort_values(CELL_KEY).reset_index(drop=True)


def load_grid_field(
    forecast_s3_key: str = FORECAST_GRIB_KEY,
    climatology_s3_key: str = CLIMATOLOGY_GRIB_KEY,
) -> pd.DataFrame:
    """
    Full gridded pipeline: download both GRIBs, bias-correct per cell, derive anomalies.
    """
    forecast_path = _download_grib_from_s3(forecast_s3_key)
    clim_path = _download_grib_from_s3(climatology_s3_key)
    forecast_grid = load_seas5_grib_grid(forecast_path)
    clim_grid = build_climatology_grid(clim_path)
    corrected = bias_correct_grid(forecast_grid, clim_grid)
    field = grid_anomaly_field(corrected, clim_grid)
    logger.info(f"Grid field: {len(field)} cell-months")
    return field


def grid_to_payload(field: pd.DataFrame, source_grib: str = "") -> Dict:
    """
    Serialise the gridded field into the dashboard's grid.json shape.

    Nulls are emitted as null, never as zero or as an interpolated value, so a cell
    with no climatology match renders as a gap rather than as normal conditions.
    """
    lat_step = _grid_spacing(field["lat"].tolist())
    lon_step = _grid_spacing(field["lon"].tolist())
    value_cols = [c for c in field.columns if c not in CELL_KEY]

    months = []
    for (year, month), group in field.groupby(["year", "month"], sort=True):
        cells = []
        for _, row in group.iterrows():
            cell = {"lat": round(float(row["lat"]), 4), "lon": round(float(row["lon"]), 4)}
            for col in value_cols:
                v = row[col]
                cell[col] = None if pd.isna(v) else round(float(v), 3)
            cells.append(cell)
        months.append({"valid_year": int(year), "valid_month": int(month), "cells": cells})

    return {
        "schema_version": "2.0",
        "kind": "conditions_field",
        "resolution": {
            "lat_step_deg": lat_step,
            "lon_step_deg": lon_step,
            "note": "Native SEAS5 grid. Not downscaled. Do not render finer than this spacing.",
        },
        "source_grib": source_grib,
        "baseline": "SEAS5 1993-2016 hindcast climatology, per cell",
        "variables": value_cols,
        "months": months,
    }


def load_and_convert(
    forecast_s3_key: str = FORECAST_GRIB_KEY,
    climatology_s3_key: str = CLIMATOLOGY_GRIB_KEY,
    historical_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Full pipeline: download SEAS5 GRIBs from S3, bias-correct, convert to ERA5 features.

    Returns:
        DataFrame with ERA5_CORE_FEATURES ready for cascade bridge.
    """
    forecast_path = _download_grib_from_s3(forecast_s3_key)
    clim_path = _download_grib_from_s3(climatology_s3_key)

    forecast_raw = load_seas5_grib(forecast_path)
    climatology = build_climatology(clim_path)
    forecast_bc = bias_correct(forecast_raw, climatology)
    era5_df = seas5_to_era5_features(forecast_bc, historical_df)
    logger.info(f"SEAS5 adapter produced {len(era5_df)} forecast rows")
    return era5_df
