
"""ATLAS Health Gold — original TAMBI Health classifier."""

import argparse
from pathlib import Path

import pandas as pd

from poc_classifier import (
    run_regression_checks,
    apply_classifier,
    export_audits,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="data/silver/industry/health/health_contracts.parquet",
    )
    parser.add_argument(
        "--output",
        default="data/gold/health/health_classified.parquet",
    )
    parser.add_argument(
        "--audit-dir",
        default="audits/gold/health",
    )

    args = parser.parse_args()

    # Check original TAMBI classification behaviour.
    run_regression_checks()

    df = pd.read_parquet(args.input)

    required = {
        "CN ID",
        "Description",
        "Category",
        "Category Type",
        "Supplier Name",
        "supplier_group",
        "is_accenture",
        "Value",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Missing Health Silver columns: {sorted(missing)}"
        )

    original_rows = len(df)
    original_value = pd.to_numeric(
        df["Value"], errors="coerce"
    ).fillna(0).sum()

    # Health Silver already handles agency scope and suppliers.
    # Reuse the original TAMBI addressability and SO logic.
    family_cols = {
        "cn_family_start_date",
        "cn_family_end_date",
        "cn_family_date_status",
    }

    print(
        "CN-family fields missing BEFORE classification:",
        sorted(family_cols - set(df.columns)),
    )

    df = apply_classifier(df)

    print(
        "CN-family fields missing AFTER classification:",
        sorted(family_cols - set(df.columns)),
    )
    # Validate population and financial totals.
    if len(df) != original_rows:
        raise RuntimeError("Health Gold changed contract count")

    new_value = pd.to_numeric(
        df["Value"], errors="coerce"
    ).fillna(0).sum()

    if abs(new_value - original_value) >= 0.01:
        raise RuntimeError("Health Gold changed procurement value")

    # Accenture wins must always remain addressable.
    acc = df["is_accenture"].fillna(False).astype(bool)
    addressable = df["is_addressable"].fillna(False).astype(bool)

    if (acc & ~addressable).any():
        raise RuntimeError(
            "Accenture addressability guarantee failed"
        )

    # Non-addressable contracts must not receive a Service Offering.
    if ((~addressable) & df["service_offering"].notna()).any():
        raise RuntimeError(
            "Non-addressable contracts received Service Offerings"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)

    audit_dir = Path(args.audit_dir)
    export_audits(df, audit_dir)

    addr = df[addressable]
    acc_rows = df[acc]

    print("\nATLAS HEALTH GOLD — TAMBI CLASSIFIER")
    print("=" * 50)
    print(f"Health contracts: {len(df):,}")
    print(f"Total procurement: A${new_value/1e9:,.3f}B")
    print(f"Addressable contracts: {len(addr):,}")
    print(f"Addressable TAM: A${addr['Value'].sum()/1e9:,.3f}B")
    print(f"Accenture contracts: {len(acc_rows):,}")
    print(f"Accenture wins: A${acc_rows['Value'].sum()/1e9:,.3f}B")

    print("\nADDRESSABLE SERVICE OFFERINGS")
    print(
        addr["service_offering"]
        .fillna("Unresolved")
        .value_counts()
        .to_string()
    )

    print(f"\nOutput: {output}")
    print(f"Audits: {audit_dir}")


if __name__ == "__main__":
    main()
