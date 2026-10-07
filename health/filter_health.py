from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter the shared ATLAS Silver AusTender dataset "
            "to the configured Health market while preserving "
            "all source and shared-normalisation columns."
        )
    )

    parser.add_argument(
        "--input",
        default=(
            "data/austender/combined/"
            "austender_combined.parquet"
        ),
    )

    parser.add_argument(
        "--agency-config",
        default=(
            "health/config/"
            "health_agencies.txt"
        ),
        help=(
            "One exact AusTender Agency name per line. "
            "Blank lines and # comments are ignored."
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
        "Shared Silver dataset does not contain "
        "an Agency field. Expected either "
        "'Agency' or '01. Agency Name'."
    )


def read_agency_config(
    path: Path,
) -> list[str]:

    if not path.exists():
        raise FileNotFoundError(
            f"Health agency config not found: {path}"
        )

    agencies: list[str] = []

    for raw_line in path.read_text(
        encoding="utf-8"
    ).splitlines():

        line = raw_line.strip()

        if not line:
            continue

        if line.startswith("#"):
            continue

        agencies.append(
            clean_text(line)
        )

    if not agencies:
        raise RuntimeError(
            "Health agency config contains no agencies."
        )

    return agencies


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    args = parse_args()

    input_path = Path(
        args.input
    )

    agency_config_path = Path(
        args.agency_config
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

    if "Value" not in df.columns:
        raise RuntimeError(
            "Shared Silver dataset is missing Value."
        )

    df["Value"] = pd.to_numeric(
        df["Value"],
        errors="coerce",
    ).fillna(0)

    input_rows = len(
        df
    )

    input_value = float(
        df["Value"].sum()
    )

    agency_column = choose_agency_column(
        df
    )

    print(
        f"Using agency field: {agency_column}"
    )

    # =========================================================================
    # Load configured Health agency scope
    # =========================================================================

    configured_agencies = read_agency_config(
        agency_config_path
    )

    configured_map = {
        normalise_agency(name): name
        for name in configured_agencies
    }

    print()
    print(
        f"Configured Health agencies: "
        f"{len(configured_agencies):,}"
    )

    for agency in configured_agencies:
        print(
            f"  - {agency}"
        )

    # =========================================================================
    # Preserve incoming agency exactly
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
        configured_map.keys()
    )

    health = out.loc[
        health_mask
    ].copy()

    if health.empty:
        raise RuntimeError(
            "Configured Health filter produced zero rows."
        )

    # =========================================================================
    # Stable reporting label
    #
    # We retain the exact configured name as agency_group.
    # Historical names remain distinct for now.
    # Machinery-of-government consolidation can happen later
    # in the Health enrichment layer.
    # =========================================================================

    health[
        "agency_group"
    ] = health[
        "_agency_normalised"
    ].map(
        configured_map
    )

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
    # Shared supplier identity
    # =========================================================================

    required_shared_fields = [
        "supplier_group",
        "is_accenture",
    ]

    missing_shared = [
        column
        for column in required_shared_fields
        if column not in health.columns
    ]

    if missing_shared:
        raise RuntimeError(
            "Health filter expected shared supplier "
            "identity fields. Missing: "
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
            f"{int(blank_supplier.sum()):,} rows "
            "with blank supplier_group."
        )

    # =========================================================================
    # Values
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
    # Audit: configured agency scope
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

        contract_counts = (
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
        / "health_agency_scope.csv",
        index=False,
    )

    # =========================================================================
    # Audit: configured agencies missing from current dataset
    # =========================================================================

    present_normalised = set(
        health[
            "agency_group"
        ]
        .dropna()
        .astype(str)
        .map(
            normalise_agency
        )
        .tolist()
    )

    missing_configured = [
        agency
        for agency in configured_agencies
        if normalise_agency(
            agency
        ) not in present_normalised
    ]

    pd.DataFrame(
        {
            "configured_agency": (
                missing_configured
            )
        }
    ).to_csv(
        audit_dir
        / "health_configured_agencies_missing.csv",
        index=False,
    )

    # =========================================================================
    # Audit: all source agency candidates
    # =========================================================================

    candidates = (
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

    candidates[
        "normalised_agency"
    ] = candidates[
        agency_column
    ].map(
        normalise_agency
    )

    candidates[
        "included_in_health_scope"
    ] = candidates[
        "normalised_agency"
    ].isin(
        configured_map.keys()
    )

    candidates.to_csv(
        audit_dir
        / "health_agency_candidates.csv",
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
    # Write Health Silver
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
        "input_value": input_value,

        "configured_health_agencies": int(
            len(configured_agencies)
        ),

        "configured_agencies_missing": (
            missing_configured
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
    # Console summary
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

    if missing_configured:

        print()
        print(
            "Configured agencies not present "
            "in current source:"
        )

        for agency in missing_configured:
            print(
                f"  - {agency}"
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
