"""
End-to-end forecast orchestrator.

Loads historical data from S3, runs cascade bridges on SEAS5 forecast input,
feeds the full 51-feature vector into the XGBoost drought model, and outputs
monthly drought probabilities per region.

Outputs, both consumed by dashboard/index.html:
    dashboard/forecast.json  regional drought probabilities, drives the decision
    dashboard/grid.json      gridded conditions field, map context only

Usage:
    # Full SEAS5 forecast (downloads GRIB from S3), writes both files
    python -m inference.forecast_runner

    # Skip the gridded field
    python -m inference.forecast_runner --no-grid

    # Retrospective mode (uses historical ERA5 data only, no GRIB needed)
    python -m inference.forecast_runner --retrospective --year 2026 --months 1 2 3
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from inference.data_loader import (
    REGIONS,
    load_historical_parquet,
    load_drought_model,
)
from inference.cascade_bridge import CascadeBridgeInference

logger = logging.getLogger(__name__)

SEASON_MAP = {"JF": 0, "MAM": 1, "JJAS": 2, "OND": 3}
THRESHOLD = 0.25  # Recall-optimized drought detection threshold

# Columns that come from observation (CHIRPS/MODIS) and must be bridge-predicted
OBSERVATION_COLS = [
    "chirps_precip_mm", "chirps_precip_anomaly", "chirps_precip_3m_cumul",
    "chirps_precip_6m_cumul", "chirps_spi1", "chirps_spi3", "chirps_spi6",
    "chirps_spi12", "chirps_spi3_lag1", "chirps_spi3_lag2", "chirps_spi3_lag3",
    "spi3_trend", "spi12_difference",
    "ndvi_mean", "ndvi_anomaly", "vci", "ndvi_min", "ndvi_max", "ndvi_std",
    "lst_day_c", "lst_night_c", "lst_anomaly_c", "tci", "vhi",
    "vci_lag1", "vci_lag2",
]


def run_region_forecast(
    region: str,
    forecast_era5_df: pd.DataFrame,
    historical_df: pd.DataFrame,
) -> List[Dict]:
    """
    Run the cascade bridge + drought model for a single region.

    Args:
        region: Region name (e.g. asal_north)
        forecast_era5_df: ERA5-format features for forecast months
        historical_df: Full historical time series (for bridge context)

    Returns:
        List of dicts with month, drought_prob, drought_forecast, signal
    """
    # Load drought model
    model_path = load_drought_model(region)
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    os.unlink(model_path)
    model_features = model.feature_names_in_

    # Prepare ERA5-only view of historical data (drop observation columns)
    hist_era5 = historical_df.drop(
        columns=[c for c in OBSERVATION_COLS if c in historical_df.columns],
        errors="ignore",
    )

    # Combine history + forecast for bridge context (rolling features need history)
    combined = pd.concat([hist_era5, forecast_era5_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["year", "month"], keep="last")
    combined = combined.sort_values(["year", "month"]).reset_index(drop=True)

    # Run cascade bridges
    bridge = CascadeBridgeInference(region)
    bridged = bridge.predict(combined)

    # Encode season
    if "season" in bridged.columns and not pd.api.types.is_integer_dtype(bridged["season"]):
        bridged["season"] = bridged["season"].map(SEASON_MAP).fillna(0).astype("int32")

    # Extract forecast rows
    forecast_months = forecast_era5_df[["year", "month"]].values.tolist()
    results = []
    for year, month in forecast_months:
        row = bridged[(bridged["year"] == year) & (bridged["month"] == month)]
        if row.empty:
            continue

        X = row.reindex(columns=model_features).fillna(0)
        if "season" in X.columns and not pd.api.types.is_integer_dtype(X["season"]):
            X["season"] = X["season"].map(SEASON_MAP).fillna(0).astype("int32")

        prob = float(model.predict_proba(X)[0, 1])
        pred = int(prob >= THRESHOLD)

        results.append({
            "month": int(month),
            "year": int(year),
            "drought_prob": round(prob, 3),
            "drought_forecast": pred,
            "signal": "DROUGHT" if pred else "normal",
        })

        bar = "#" * int(prob * 20)
        flag = " <-- DROUGHT" if pred else ""
        logger.info(f"  {int(year)}-{int(month):02d}  {bar:<20} {prob:.3f}{flag}")

    return results


def run_retrospective(year: int, months: List[int]) -> Dict:
    """
    Retrospective forecast using existing ERA5 data from S3.
    Useful for validation (no SEAS5 GRIB needed).
    """
    results = {}

    for region in REGIONS:
        print(f"\n{'='*55}")
        print(f"  {region.upper()} -- {year} retrospective forecast")
        print(f"{'='*55}")

        historical = load_historical_parquet(region)

        # Build forecast rows from historical data (ERA5-only view)
        forecast_rows = historical[
            (historical["year"] == year) & (historical["month"].isin(months))
        ]
        if forecast_rows.empty:
            print(f"  No {year} data for months {months} in {region}")
            continue

        era5_only = forecast_rows.drop(
            columns=[c for c in OBSERVATION_COLS if c in forecast_rows.columns],
            errors="ignore",
        )

        region_results = run_region_forecast(region, era5_only, historical)
        results[region] = region_results

        for r in region_results:
            flag = "DROUGHT" if r["drought_forecast"] else "normal "
            print(f"  {r['year']}-{r['month']:02d}  {flag}  (prob={r['drought_prob']:.3f})")

    return results


def run_seas5_forecast() -> Dict:
    """
    Full SEAS5 forecast: download GRIB from S3, convert, run bridges + model.
    """
    from inference.seas5_adapter import load_and_convert

    results = {}

    for region in REGIONS:
        print(f"\n{'='*55}")
        print(f"  {region.upper()} -- SEAS5 forecast")
        print(f"{'='*55}")

        historical = load_historical_parquet(region)

        # Convert SEAS5 to ERA5 features
        forecast_df = load_and_convert(historical_df=historical)

        region_results = run_region_forecast(region, forecast_df, historical)
        results[region] = region_results

        for r in region_results:
            flag = "DROUGHT" if r["drought_forecast"] else "normal "
            print(f"  {r['year']}-{r['month']:02d}  {flag}  (prob={r['drought_prob']:.3f})")

    return results


def save_forecast(results: Dict, output_path: str = "dashboard/forecast.json"):
    """Save forecast results as JSON for the dashboard."""
    output = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "threshold": THRESHOLD,
        "regions": results,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nForecast saved to {output_path}")


def run_grid_field() -> Optional[Dict]:
    """
    Build the gridded conditions field that backs the dashboard map layer.

    Errors are swallowed on purpose. The drought probabilities are the deliverable and
    are already computed by the time this runs, so a missing GRIB, an absent cfgrib or
    an unreadable grid must not discard a completed model run.

    Returns None when the field could not be built.
    """
    from inference.seas5_adapter import (
        FORECAST_GRIB_KEY,
        grid_to_payload,
        load_grid_field,
    )

    try:
        field = load_grid_field()
        payload = grid_to_payload(field, FORECAST_GRIB_KEY)
        cells = sum(len(m["cells"]) for m in payload["months"])
        res = payload["resolution"]
        print(f"\nGrid field: {cells} cell-months at "
              f"{res['lat_step_deg']} x {res['lon_step_deg']} degrees")
        return payload
    except Exception as e:
        logger.warning(f"Gridded field unavailable, continuing without it: {e}")
        print(f"\nGrid field skipped: {e}")
        return None


def save_grid(payload: Dict, output_path: str = "dashboard/grid.json"):
    """Save the gridded conditions field as JSON for the dashboard map layer."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Grid field saved to {output_path}")


def print_summary(results: Dict, year: int):
    """Print a summary table of forecast results."""
    print(f"\n{'='*55}")
    print(f"  SUMMARY -- {year} Drought Forecast")
    print(f"{'='*55}")

    header = f"  {'Month':<8}"
    for r in REGIONS:
        header += f"  {r.replace('asal_', ''):<16}"
    print(header)
    print(f"  {'-'*7}" + f"  {'-'*16}" * len(REGIONS))

    all_months = sorted(set(
        row["month"] for region_data in results.values() for row in region_data
    ))

    for month in all_months:
        yr = next(
            (row["year"] for rd in results.values() for row in rd if row["month"] == month),
            year,
        )
        line = f"  {yr}-{month:02d} "
        for region in REGIONS:
            region_data = results.get(region, [])
            row = next((r for r in region_data if r["month"] == month), None)
            if row:
                flag = "DROUGHT" if row["drought_forecast"] else "normal "
                line += f"  {flag} ({row['drought_prob']:.2f})  "
            else:
                line += f"  {'N/A':<16}"
        print(line)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="NatureCipher drought forecast runner")
    parser.add_argument("--retrospective", action="store_true",
                        help="Use historical ERA5 data instead of SEAS5 GRIB")
    parser.add_argument("--year", type=int, default=2026,
                        help="Forecast year (default: 2026)")
    parser.add_argument("--months", nargs="+", type=int, default=[1, 2, 3],
                        help="Months to forecast (default: 1 2 3)")
    parser.add_argument("--output", default="dashboard/forecast.json",
                        help="Output JSON path (default: dashboard/forecast.json)")
    parser.add_argument("--grid-output", default="dashboard/grid.json",
                        help="Gridded conditions field path (default: dashboard/grid.json)")
    parser.add_argument("--no-grid", action="store_true",
                        help="Skip the gridded conditions field")
    args = parser.parse_args()

    grid_payload = None
    if args.retrospective:
        print(f"Running retrospective forecast for {args.year}, months {args.months}")
        results = run_retrospective(args.year, args.months)
    else:
        print("Running SEAS5 seasonal forecast...")
        results = run_seas5_forecast()
        # Retrospective mode reads historical parquet and never touches a GRIB, so
        # there is no forecast grid to emit for it.
        if not args.no_grid:
            grid_payload = run_grid_field()

    if results:
        print_summary(results, args.year)
        save_forecast(results, args.output)

    if grid_payload:
        save_grid(grid_payload, args.grid_output)


if __name__ == "__main__":
    main()
