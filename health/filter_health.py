from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


# =============================================================================
# HEALTH MARKET SCOPE
# =============================================================================
#
# Historical Health portfolio agency names are retained in Agency Raw and rolled
# into stable reporting groups.
#
# This file ONLY:
#   - selects the Health market
#   - preserves all incoming Silver columns
#   - adds canonical Health agency grouping
#   - inherits shared supplier identity
#
# It DOES NOT:
#   - classify addressability
#   - classify Service Offering
#   - rebuild supplier identity
# =============================================================================

HEALTH_AGENCY_ALIASES = {
    "department of health":
        "Department of Health, Disability and Ageing",

    "department of health and aged care":
        "Department of Health, Disability and Ageing",

    "department of health disability and ageing":
        "Department of Health, Disability and Ageing",

    "department of health and aged care therapeutic goods administration":
        "Department of Health and Aged Care - Therapeutic Goods Administration",

    "australian digital health agency":
        "Australian Digital Health Agency",

    "australian aged care quality agency":
        "Australian Aged Care Quality Agency",

    "australian institute of health and welfare":
        "Australian Institute of Health and Welfare",

    "department of social services":
        "Department of Social Services",

    "independent health and aged care pricing authority":
        "Independent Health and Aged Care Pricing Authority",

    "national health funding body":
        "National Health Funding Body",

    "national health and medical research council":
        "National Health and Medical Research Council",

    "organ and tissue authority":
        "Organ and Tissue Authority",
}


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Filter the shared ATLAS Silver AusTender dataset "
            "to the Health market while preserving all source "
            "and shared-normalisation columns."
        )
    )

    parser.add_argument(
        "--input",
        default=(
            "data/austender/combined/"
            "austender_combined.parquet"
        ),
    )

    # Retained so the existing workflow/CLI remains compatible.
    parser.add_argument(
        "--agency-config",
        default=None,
        help=(
            "Compatibility option. Canonical longitudinal "
            "Health agency scope is defined in this script."
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "health/data/"
            "health_contracts_raw.parquet"
        ),
    )

    parser.add_argument(
        "--audit-dir",
        default=(
            "audits/austender/health"
        ),
    )

    return parser.parse_args()


# =============================================================================
# HELPERS
# =============================================================================

def clean_text(value: object) -> str:

    if value is None or pd.isna(value):
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value)
        .replace("\u00a0", " ")
        .strip(),
    )


def normalise_agency(
    value: object,
) -> str:

    return (
        clean_text(value)
        .lower()
    )


def choose_agency_column(
    df: pd.DataFrame,
) -> str:

    candidates = [
        "Agency",
        "01. Agency Name",
    ]

    for column in candidates:

        if column in df.columns:
            return column

    raise RuntimeError(
        "Shared Silver dataset does not contain an "
        "Agency field. Expected either 'Agency' or "
        "'01. Agency Name'."
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    args = parse_args()

    input_path = Path(
        args.input
    )

    output_path = Path(
        args.output
    )

    audit_dir = Path(
        args.audit_dir
    )

    if not input_path.exists():

        raise SystemExit(
            "Shared ATLAS Silver dataset not found: "
            f"{input_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Loading shared ATLAS Silver dataset: "
        f"{input_path}"
    )

    df = pd.read_parquet(
        input_path
    )

    if df.empty:

        raise RuntimeError(
            "Shared Silver dataset contains zero rows."
        )

    input_rows = len(
        df
    )

    if "Value" not in df.columns:

        raise RuntimeError(
            "Shared Silver dataset is missing Value."
        )

    df[
        "Value"
    ] = pd.to_numeric(
        df[
            "Value"
        ],
        errors="coerce",
    ).fillna(0)

    input_value = float(
        df[
            "Value"
        ].sum()
    )

    agency_column = (
        choose_agency_column(
            df
        )
    )

    print(
        f"Using agency field: "
        f"{agency_column}"
    )

    # =========================================================================
    # Preserve incoming agency
    # =========================================================================

    out = df.copy()

    if "Agency Raw" not in out.columns:

        out[
            "Agency Raw"
        ] = out[
            agency_column
        ]

    out[
        "_agency_normalised"
    ] = out[
        agency_column
    ].map(
        normalise_agency
    )

    # =========================================================================
    # Health scope
    # =========================================================================

    health_mask = out[
        "_agency_normalised"
    ].isin(
        HEALTH_AGENCY_ALIASES.keys()
    )

    health = out.loc[
        health_mask
    ].copy()

    if health.empty:

        raise RuntimeError(
            "Health filter produced zero rows."
        )

    # =========================================================================
    # Stable Health reporting agency
    # =========================================================================

    health[
        "agency_group"
    ] = health[
        "_agency_normalised"
    ].map(
        HEALTH_AGENCY_ALIASES
    )

    # Friendly Agency is the canonical reporting label.
    #
    # Original 01. Agency Name remains untouched.
    health[
        "Agency"
    ] = health[
        "agency_group"
    ]

    health[
        "is_health_scope"
    ] = True

    health[
        "atlas_market"
    ] = "Health"

    health.drop(
        columns=[
            "_agency_normalised",
        ],
        errors="ignore",
        inplace=True,
    )

    # =========================================================================
    # Shared supplier identity must already exist
    # =========================================================================

    required_shared_fields = [
        "supplier_group",
        "is_accenture",
    ]

    missing_shared = [
        column
        for column
        in required_shared_fields
        if column not in health.columns
    ]

    if missing_shared:

        raise RuntimeError(
            "Health filter expected supplier identity "
            "from the shared Silver layer. Missing: "
            + ", ".join(
                missing_shared
            )
        )

    blank_supplier = (
        health[
            "supplier_group"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
    )

    if blank_supplier.any():

        raise RuntimeError(
            "Health Silver contains "
            f"{int(blank_supplier.sum()):,} "
            "rows with blank supplier_group."
        )

    # =========================================================================
    # Health values
    # =========================================================================

    health[
        "Value"
    ] = pd.to_numeric(
        health[
            "Value"
        ],
        errors="coerce",
    ).fillna(0)

    health_value = float(
        health[
            "Value"
        ].sum()
    )

    # =========================================================================
    # Agency scope audit
    # =========================================================================

    agency_audit = (
        health
        .groupby(
            [
                "Agency Raw",
                "agency_group",
            ],
            dropna=False,
        )
        .agg(
            rows=(
                "Value",
                "size",
            ),
            value=(
                "Value",
                "sum",
            ),
        )
        .reset_index()
        .sort_values(
            "value",
            ascending=False,
        )
    )

    if "CN ID" in health.columns:

        contracts = (
            health
            .groupby(
                [
                    "Agency Raw",
                    "agency_group",
                ],
                dropna=False,
            )[
                "CN ID"
            ]
            .nunique()
            .rename(
                "contracts"
            )
            .reset_index()
        )

        agency_audit = (
            agency_audit
            .merge(
                contracts,
                on=[
                    "Agency Raw",
                    "agency_group",
                ],
                how="left",
            )
        )

    agency_audit.to_csv(
        audit_dir
        / "health_agency_scope.csv",
        index=False,
    )

    # =========================================================================
    # Supplier summary
    # =========================================================================

    supplier_audit = (
        health
        .groupby(
            "supplier_group",
            dropna=False,
        )
        .agg(
            rows=(
                "Value",
                "size",
            ),
            value=(
                "Value",
                "sum",
            ),
        )
        .reset_index()
        .sort_values(
            "value",
            ascending=False,
        )
    )

    if "CN ID" in health.columns:

        supplier_contracts = (
            health
            .groupby(
                "supplier_group",
                dropna=False,
            )[
                "CN ID"
            ]
            .nunique()
            .rename(
                "contracts"
            )
            .reset_index()
        )

        supplier_audit = (
            supplier_audit
            .merge(
                supplier_contracts,
                on="supplier_group",
                how="left",
            )
        )

    supplier_audit.to_csv(
        audit_dir
        / "health_supplier_summary.csv",
        index=False,
    )

    # =========================================================================
    # Source agency inventory
    #
    # Useful when AusTender introduces a new Health agency name.
    # =========================================================================

    source_agencies = (
        out
        .groupby(
            agency_column,
            dropna=False,
        )
        .agg(
            rows=(
                "Value",
                "size",
            ),
            value=(
                "Value",
                "sum",
            ),
        )
        .reset_index()
        .sort_values(
            "value",
            ascending=False,
        )
    )

    source_agencies[
        "currently_in_health_scope"
    ] = (
        source_agencies[
            agency_column
        ]
        .map(
            normalise_agency
        )
        .isin(
            HEALTH_AGENCY_ALIASES.keys()
        )
    )

    source_agencies.to_csv(
        audit_dir
        / "health_source_agency_inventory.csv",
        index=False,
    )

    # =========================================================================
    # Write wide Health Silver
    # =========================================================================

    health.to_parquet(
        output_path,
        index=False,
    )

    # =========================================================================
    # Summary
    # =========================================================================

    accenture_mask = (
        health[
            "is_accenture"
        ]
        .fillna(False)
        .astype(bool)
    )

    accenture_value = float(
        health.loc[
            accenture_mask,
            "Value",
        ].sum()
    )

    summary = {
        "input_rows": int(
            input_rows
        ),

        "input_value": (
            input_value
        ),

        "health_rows": int(
            len(health)
        ),

        "health_value": (
            health_value
        ),

        "health_supplier_groups": int(
            health[
                "supplier_group"
            ].nunique()
        ),

        "accenture_rows": int(
            accenture_mask.sum()
        ),

        "accenture_value": (
            accenture_value
        ),

        "columns_preserved": int(
            len(
                health.columns
            )
        ),

        "health_agencies": sorted(
            health[
                "agency_group"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        ),

        "output": str(
            output_path
        ),
    }

    (
        audit_dir
        / "health_filter_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    # =========================================================================
    # Console
    # =========================================================================

    print()

    print(
        "========================================"
    )

    print(
        "ATLAS HEALTH SILVER FILTER COMPLETE"
    )

    print(
        "========================================"
    )

    print(
        f"Input rows: "
        f"{input_rows:,}"
    )

    print(
        f"Health rows: "
        f"{len(health):,}"
    )

    print(
        f"Health value: "
        f"${health_value:,.2f}"
    )

    print(
        f"Health supplier groups: "
        f"{summary['health_supplier_groups']:,}"
    )

    print(
        f"Accenture rows: "
        f"{summary['accenture_rows']:,}"
    )

    print(
        f"Accenture value: "
        f"${accenture_value:,.2f}"
    )

    print(
        f"Columns retained: "
        f"{summary['columns_preserved']:,}"
    )

    print()

    print(
        "Health reporting agencies:"
    )

    for agency in summary[
        "health_agencies"
    ]:

        print(
            f"  - {agency}"
        )

    print()

    print(
        f"Wrote: "
        f"{output_path}"
    )

    print(
        "========================================"
    )


if __name__ == "__main__":
    main()
