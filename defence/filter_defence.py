from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


# =============================================================================
# DEFENCE MARKET SCOPE
# =============================================================================

DEFENCE_AGENCIES = {
    "department of defence": "Department of Defence",
    "australian signals directorate": "Australian Signals Directorate",
    "australian submarine agency": "Australian Submarine Agency",
}


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter the shared ATLAS Silver AusTender dataset "
            "to the Defence market while preserving all source "
            "and shared-normalisation columns."
        )
    )

    parser.add_argument(
        "--input",
        default="data/austender/combined/austender_combined.parquet",
    )

    parser.add_argument(
        "--output",
        default="defence/data/defence_contracts_raw.parquet",
    )

    parser.add_argument(
        "--audit-dir",
        default="audits/austender/defence",
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


def normalise_agency(value: object) -> str:
    return clean_text(value).lower()


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
        "Shared Silver dataset does not contain an Agency field. "
        "Expected either 'Agency' or '01. Agency Name'."
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
            f"Shared ATLAS Silver dataset not found: {input_path}"
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
        f"Loading shared ATLAS Silver dataset: {input_path}"
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

    input_value = (
        float(
            pd.to_numeric(
                df["Value"],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        )
        if "Value" in df.columns
        else None
    )

    agency_column = choose_agency_column(
        df
    )

    print(
        f"Using agency field: {agency_column}"
    )

    # -------------------------------------------------------------------------
    # Preserve the incoming agency exactly.
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Strict Defence scope only.
    # -------------------------------------------------------------------------

    defence_mask = out[
        "_agency_normalised"
    ].isin(
        DEFENCE_AGENCIES.keys()
    )

    defence = out.loc[
        defence_mask
    ].copy()

    if defence.empty:
        raise RuntimeError(
            "Defence filter produced zero rows."
        )

    # -------------------------------------------------------------------------
    # Add stable Defence reporting agency.
    #
    # Do not overwrite the original 01. Agency Name field.
    # -------------------------------------------------------------------------

    defence[
        "agency_group"
    ] = defence[
        "_agency_normalised"
    ].map(
        DEFENCE_AGENCIES
    )

    defence[
        "Agency"
    ] = defence[
        "agency_group"
    ]

    defence[
        "is_defence_scope"
    ] = True

    defence[
        "atlas_market"
    ] = "Defence"

    # Internal filter helper is no longer needed.
    defence.drop(
        columns=[
            "_agency_normalised",
        ],
        errors="ignore",
        inplace=True,
    )

    # -------------------------------------------------------------------------
    # Shared supplier identity must already exist.
    # We consume it; we do not rebuild it here.
    # -------------------------------------------------------------------------

    required_shared_fields = [
        "supplier_group",
        "is_accenture",
    ]

    missing_shared = [
        column
        for column in required_shared_fields
        if column not in defence.columns
    ]

    if missing_shared:
        raise RuntimeError(
            "Defence filter expected supplier identity from "
            "the shared Silver layer. Missing: "
            + ", ".join(
                missing_shared
            )
        )

    blank_supplier = (
        defence[
            "supplier_group"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
    )

    if blank_supplier.any():
        raise RuntimeError(
            "Defence Silver contains "
            f"{int(blank_supplier.sum()):,} rows "
            "with blank supplier_group."
        )

    # -------------------------------------------------------------------------
    # Value validation
    # -------------------------------------------------------------------------

    if "Value" not in defence.columns:
        raise RuntimeError(
            "Defence Silver dataset is missing Value."
        )

    defence[
        "Value"
    ] = pd.to_numeric(
        defence[
            "Value"
        ],
        errors="coerce",
    ).fillna(0)

    defence_value = float(
        defence[
            "Value"
        ].sum()
    )

    # -------------------------------------------------------------------------
    # Agency audit
    # -------------------------------------------------------------------------

    agency_audit = (
        defence
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

    if "CN ID" in defence.columns:
        contract_counts = (
            defence
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
                contract_counts,
                on=[
                    "Agency Raw",
                    "agency_group",
                ],
                how="left",
            )
        )

    agency_audit.to_csv(
        audit_dir
        / "defence_agency_scope.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Supplier audit for Defence slice
    # -------------------------------------------------------------------------

    supplier_audit = (
        defence
        .groupby(
            [
                "supplier_group",
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

    if "CN ID" in defence.columns:
        supplier_contracts = (
            defence
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
        / "defence_supplier_summary.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Excluded agency audit
    # -------------------------------------------------------------------------

    excluded = out.loc[
        ~defence_mask
    ].copy()

    if not excluded.empty:

        if "Value" in excluded.columns:
            excluded[
                "Value"
            ] = pd.to_numeric(
                excluded[
                    "Value"
                ],
                errors="coerce",
            ).fillna(0)

            excluded_audit = (
                excluded
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

        else:
            excluded_audit = (
                excluded
                .groupby(
                    agency_column,
                    dropna=False,
                )
                .size()
                .reset_index(
                    name="rows"
                )
            )

        excluded_audit.to_csv(
            audit_dir
            / "defence_excluded_agencies.csv",
            index=False,
        )

    # -------------------------------------------------------------------------
    # Write wide Defence Silver dataset.
    # -------------------------------------------------------------------------

    defence.to_parquet(
        output_path,
        index=False,
    )

    accenture_mask = (
        defence[
            "is_accenture"
        ]
        .fillna(False)
        .astype(bool)
    )

    accenture_value = float(
        defence.loc[
            accenture_mask,
            "Value",
        ].sum()
    )

    summary = {
        "input_rows": int(
            input_rows
        ),
        "input_value": input_value,
        "defence_rows": int(
            len(defence)
        ),
        "defence_value": defence_value,
        "defence_supplier_groups": int(
            defence[
                "supplier_group"
            ].nunique()
        ),
        "accenture_rows": int(
            accenture_mask.sum()
        ),
        "accenture_value": accenture_value,
        "columns_preserved": int(
            len(defence.columns)
        ),
        "output": str(
            output_path
        ),
    }

    (
        audit_dir
        / "defence_filter_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "========================================"
    )

    print(
        "ATLAS DEFENCE SILVER FILTER COMPLETE"
    )

    print(
        "========================================"
    )

    print(
        f"Input rows: "
        f"{input_rows:,}"
    )

    print(
        f"Defence rows: "
        f"{len(defence):,}"
    )

    print(
        f"Defence value: "
        f"${defence_value:,.2f}"
    )

    print(
        f"Defence supplier groups: "
        f"{summary['defence_supplier_groups']:,}"
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
        f"Wrote: {output_path}"
    )

    print(
        "========================================"
    )


if __name__ == "__main__":
    main()
