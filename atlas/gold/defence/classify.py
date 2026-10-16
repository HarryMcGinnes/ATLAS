
"""ATLAS Defence Gold — original POC classification."""

import argparse
from pathlib import Path

import pandas as pd

from poc_classifier import (
    run_regression_checks,
    apply_service_offering_mapping,
    retain_observed_accenture_wins,
    enforce_authoritative_contract_overrides,
    apply_reinvention,
    bool_series,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/silver/industry/defence/defence_enriched.parquet",
    )
    parser.add_argument(
        "--output",
        default="data/gold/defence/defence_classified.parquet",
    )
    args = parser.parse_args()

    # Validate original POC behaviour before processing data.
    run_regression_checks()

    df = pd.read_parquet(args.input)
    original_rows = len(df)
    original_value = pd.to_numeric(
        df["Value"], errors="coerce"
    ).fillna(0).sum()

    # Silver Core already provides supplier_group and is_accenture.
    required = {
        "CN ID",
        "Description",
        "Category",
        "Category Type",
        "Supplier Name",
        "supplier_group",
        "is_accenture",
        "Value",
        "defence_domain",
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing Silver fields: {sorted(missing)}")

    # Restore original POC classification decisions.
    df = apply_service_offering_mapping(df)
    df["is_addressable"] = bool_series(df["is_addressable"])

    # Preserve original Accenture and authoritative CN guarantees.
    df = retain_observed_accenture_wins(df)
    df = enforce_authoritative_contract_overrides(df)

    # Independent RP/RE enrichment.
    df = apply_reinvention(df)

    # Validate financial and population invariants.
    assert len(df) == original_rows, "Gold changed contract count"

    new_value = pd.to_numeric(
        df["Value"], errors="coerce"
    ).fillna(0).sum()

    assert abs(new_value - original_value) < 0.01, (
        "Gold changed total Defence value"
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)

    addressable = df[df["is_addressable"]]

    print("\nATLAS DEFENCE GOLD — POC CLASSIFIER")
    print("=" * 45)
    print(f"Contracts: {len(df):,}")
    print(f"Defence value: A${new_value / 1e9:,.3f}B")
    print(f"Addressable value: A${addressable['Value'].sum() / 1e9:,.3f}B")
    print(
        f"Accenture wins: A$"
        f"{addressable.loc[addressable['is_accenture'], 'Value'].sum() / 1e9:,.3f}B"
    )
    print(addressable["service_offering"].value_counts().to_string())
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
