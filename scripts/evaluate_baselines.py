"""
Score the cascade against the baselines it has to beat.

An accuracy figure on its own is not evidence of skill. 2021-2024 in the northern
ASALs spans the worst Horn of Africa drought in forty years, so drought is likely
the majority class -- a model that answers "drought" every month would post a high
accuracy and perfect recall while knowing nothing. This script publishes the
number that makes the metric readable (the base rate) and the two baselines that
make it meaningful:

  majority class   always predict the commoner label. Beating this is the floor.
  ERA5-only        the same classifier on atmospheric features alone, no bridges.
                   This is the one that tests the cascade's actual thesis: if the
                   bridges add nothing over their own inputs, they are complexity
                   without payoff.

Writes dashboard/validation.json in the schema the dashboard already reads.

Usage:
    python scripts/evaluate_baselines.py                 # all regions
    python scripts/evaluate_baselines.py --region asal_north
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from inference.data_loader import REGIONS, load_historical_parquet
from inference.forecast_runner import OBSERVATION_COLS, SEASON_MAP, THRESHOLD

logger = logging.getLogger(__name__)

LABEL_CANDIDATES = ["drought", "drought_label", "is_drought", "label",
                    "drought_flag", "ipc_drought", "target"]
TEST_START_YEAR = 2021
NON_FEATURES = {"year", "month", "date", "region"}


def find_label(df: pd.DataFrame) -> str:
    """Locate the ground-truth column without hardcoding a name we haven't seen."""
    for name in LABEL_CANDIDATES:
        if name in df.columns:
            return name
    binary = [
        c for c in df.columns
        if df[c].dropna().isin([0, 1, True, False]).all() and df[c].nunique() == 2
    ]
    if len(binary) == 1:
        return binary[0]
    raise SystemExit(
        f"Could not identify the label column. Tried {LABEL_CANDIDATES}; "
        f"binary candidates were {binary}. Pass the right name via --label."
    )


def feature_columns(df: pd.DataFrame, label: str, era5_only: bool) -> list:
    cols = [
        c for c in df.columns
        if c not in NON_FEATURES and c != label
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    if era5_only:
        # Drop everything the bridges synthesise, leaving only what SEAS5 supplies.
        cols = [c for c in cols if c not in set(OBSERVATION_COLS)]
    return cols


def encode_season(df: pd.DataFrame) -> pd.DataFrame:
    if "season" in df.columns and not pd.api.types.is_numeric_dtype(df["season"]):
        df = df.copy()
        df["season"] = df["season"].map(SEASON_MAP).fillna(0).astype("int32")
    return df


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "accuracy": round(accuracy, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def wald_interval(p: float, n: int, z: float = 1.96) -> dict:
    """95% interval on a proportion, so a point estimate is never read as exact."""
    if n <= 0:
        return {"low": 0.0, "high": 0.0}
    half = z * ((p * (1 - p) / n) ** 0.5)
    return {"low": round(max(0.0, p - half), 3), "high": round(min(1.0, p + half), 3)}


def fit_and_score(train: pd.DataFrame, test: pd.DataFrame, feats: list,
                  label: str) -> dict:
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
        random_state=42,
    )
    model.fit(train[feats].fillna(0), train[label].astype(int))
    prob = model.predict_proba(test[feats].fillna(0))[:, 1]
    return metrics(test[label].astype(int).values, (prob >= THRESHOLD).astype(int))


def evaluate(region: str, label_override: str = None) -> dict:
    df = encode_season(load_historical_parquet(region))
    label = label_override or find_label(df)
    df = df.dropna(subset=[label])

    train = df[df["year"] < TEST_START_YEAR]
    test = df[df["year"] >= TEST_START_YEAR]
    if test.empty or train.empty:
        raise SystemExit(f"{region}: empty split at {TEST_START_YEAR}.")

    y_test = test[label].astype(int).values
    base_rate = float(y_test.mean())

    # The floor: always answer with whichever label is commoner in training.
    majority = int(train[label].astype(int).mode().iloc[0])
    majority_metrics = metrics(y_test, np.full_like(y_test, majority))

    cascade = fit_and_score(train, test, feature_columns(df, label, False), label)
    era5_only = fit_and_score(train, test, feature_columns(df, label, True), label)

    lift = round(cascade["f1"] - era5_only["f1"], 3)
    logger.info(f"{region}: cascade F1 {cascade['f1']} vs ERA5-only {era5_only['f1']} "
                f"(lift {lift:+})")

    return {
        "region": region,
        "period": {"start": f"{TEST_START_YEAR}-01", "end": f"{int(test['year'].max())}-12"},
        "n_samples": int(len(test)),
        "base_rate": round(base_rate, 3),
        "metrics": cascade,
        "confidence_interval": {
            "metric": "accuracy", "level": 0.95,
            **wald_interval(cascade["accuracy"], len(test)),
        },
        "baselines": {
            "majority_class": {"computed": True, "predicts": majority,
                               "metrics": majority_metrics},
            "era5_only": {"computed": True, "metrics": era5_only},
        },
        "cascade_lift_f1_over_era5_only": lift,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Cascade vs baselines")
    parser.add_argument("--region", choices=REGIONS, help="default: all regions")
    parser.add_argument("--label", help="ground-truth column, if auto-detect fails")
    parser.add_argument("--output", default="dashboard/validation.json")
    args = parser.parse_args()

    targets = [args.region] if args.region else list(REGIONS)
    results = [evaluate(r, args.label) for r in targets]
    headline = max(results, key=lambda r: r["n_samples"])

    payload = {
        "schema_version": "2.0",
        "updated_at": pd.Timestamp.utcnow().isoformat(),
        "backtest": headline,
        "per_region": results,
        "retrospective": {
            "period": {"start": "2026-01", "end": "2026-03"},
            "kind": "hindcast",
            "input_data": "ERA5 reanalysis, available after the fact",
            "claim": "Northeast Kenya drought emergency identified",
            "reference_event": "IPC declaration, March 2026",
            "caveat": ("Retrospective. ERA5 for these months was used after the "
                       "fact, not an ahead-of-time forecast."),
        },
        "known_limitations": [
            "Forward forecasts experimental, single July 2026 SEAS5 initialization",
            "Soil moisture uses persistence, SEAS5 provides no soil moisture layers",
            "Three ASAL regions only",
            "Ensemble spread discarded, forecasts carry no uncertainty band",
            "Decision threshold tuned on the same window the metrics are reported on",
        ],
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\n{'region':<16}{'base':>7}{'majF1':>8}{'era5F1':>8}{'casF1':>8}{'lift':>8}")
    for r in results:
        print(f"{r['region']:<16}{r['base_rate']:>7.3f}"
              f"{r['baselines']['majority_class']['metrics']['f1']:>8.3f}"
              f"{r['baselines']['era5_only']['metrics']['f1']:>8.3f}"
              f"{r['metrics']['f1']:>8.3f}{r['cascade_lift_f1_over_era5_only']:>+8.3f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
