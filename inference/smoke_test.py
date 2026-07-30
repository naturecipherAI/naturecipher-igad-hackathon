"""
Smoke test -- verifies the deployed pipeline reproduces reference validation results.

Runs a Jan-Mar 2026 retrospective forecast and compares drought probabilities
against the reference output stored on S3 (validation/forecast_2026.json).

Usage:
    python -m inference.smoke_test
    python -m inference.smoke_test --tolerance 0.15

Exit code 0 = PASS, 1 = FAIL
"""

import json
import logging
import sys

from inference.data_loader import REGIONS, load_validation_reference
from inference.forecast_runner import run_retrospective

logger = logging.getLogger(__name__)

DEFAULT_TOLERANCE = 0.10  # Allow 10% probability drift from reference


def smoke_test(tolerance: float = DEFAULT_TOLERANCE) -> bool:
    """
    Run the retrospective forecast and compare against reference.

    Returns True if all region/month drought probabilities are within
    tolerance of the reference values.
    """
    print("=" * 60)
    print("  NatureCipher Drought Pipeline -- Smoke Test")
    print("=" * 60)

    # Load reference
    print("\nLoading reference validation from S3...")
    reference = load_validation_reference()

    # Determine months from reference
    ref_months = sorted(set(
        row["month"]
        for region_data in reference.values()
        for row in region_data
    ))
    print(f"Reference months: {ref_months}")
    print(f"Reference regions: {list(reference.keys())}")

    # Run retrospective forecast
    print("\nRunning retrospective forecast...")
    results = run_retrospective(year=2026, months=ref_months)

    # Compare
    print(f"\n{'='*60}")
    print(f"  Comparison (tolerance = {tolerance:.0%})")
    print(f"{'='*60}")

    all_pass = True
    comparisons = []

    for region in REGIONS:
        ref_region = reference.get(region, [])
        pred_region = results.get(region, [])

        if not ref_region:
            print(f"\n  {region}: NO REFERENCE DATA")
            continue

        print(f"\n  {region}:")
        for ref_row in ref_region:
            month = ref_row["month"]
            ref_prob = ref_row["drought_prob"]
            ref_signal = ref_row["signal"]

            pred_row = next((r for r in pred_region if r["month"] == month), None)
            if pred_row is None:
                print(f"    {month:02d}: MISSING (ref={ref_prob:.3f} {ref_signal})")
                all_pass = False
                comparisons.append({
                    "region": region, "month": month,
                    "ref_prob": ref_prob, "pred_prob": None,
                    "status": "MISSING",
                })
                continue

            pred_prob = pred_row["drought_prob"]
            pred_signal = pred_row["signal"]
            diff = abs(pred_prob - ref_prob)
            passed = diff <= tolerance

            status = "PASS" if passed else "FAIL"
            if not passed:
                all_pass = False

            signal_match = "ok" if ref_signal == pred_signal else "SIGNAL MISMATCH"
            print(
                f"    {month:02d}: ref={ref_prob:.3f} pred={pred_prob:.3f} "
                f"diff={diff:.3f} [{status}] {signal_match}"
            )

            comparisons.append({
                "region": region, "month": month,
                "ref_prob": ref_prob, "pred_prob": pred_prob,
                "diff": round(diff, 4), "status": status,
                "signal_match": ref_signal == pred_signal,
            })

    # Summary
    n_total = len(comparisons)
    n_pass = sum(1 for c in comparisons if c["status"] == "PASS")
    n_fail = n_total - n_pass

    print(f"\n{'='*60}")
    if all_pass:
        print(f"  SMOKE TEST PASSED ({n_pass}/{n_total} checks within {tolerance:.0%})")
    else:
        print(f"  SMOKE TEST FAILED ({n_fail}/{n_total} checks exceeded {tolerance:.0%})")
    print(f"{'='*60}")

    return all_pass


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    import argparse
    parser = argparse.ArgumentParser(description="Pipeline smoke test")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                        help=f"Max allowed probability drift (default: {DEFAULT_TOLERANCE})")
    args = parser.parse_args()

    passed = smoke_test(tolerance=args.tolerance)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
