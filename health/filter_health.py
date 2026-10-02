from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


AGENCY_COL = "Agency"
RAW_AGENCY_COL = "Agency Raw"
AGENCY_GROUP_COL = "agency_group"
VALUE_COL = "Value"
CN_ID_COL = "CN ID"


# ============================================================
# HEALTH AGENCY SCOPE
# ============================================================
#
# These aliases are intentionally longitudinal.
# Historical Department names are mapped into stable reporting
# labels so Health spend is not split by machinery-of-government
# changes.
#
# This stage ONLY scopes the Health population.
# It does NOT classify addressability or Service Offering.
# ============================================================

HEALTH_AGENCY_ALIASES = {
    "department of health": (
        "Department of Health, Disability and Ageing"
    ),
    "department of health and aged care": (
        "Department of Health, Disability and Ageing"
    ),
    "department of health disability and ageing": (
        "Department of Health, Disability and Ageing"
    ),
    "department of health and aged care therapeutic goods administration": (
        "Department of Health and Aged Care - "
        "Therapeutic Goods Administration"
    ),
    "australian digital health agency": (
        "Australian Digital Health Agency"
    ),
    "australian aged care quality agency": (
        "Australian Aged Care Quality Agency"
    ),
    "australian institute of health and welfare": (
        "Australian Institute of Health and Welfare"
    ),
    "department of social services": (
        "Department of Social Services"
    ),
    "independent health and aged care pricing authority": (
        "Independent Health and Aged Care Pricing Authority"
    ),
    "national health funding body": (
        "National Health Funding Body"
    ),
    "national health and medical research council": (
        "National Health and Medical Research Council"
    ),
    "organ and tissue authority": (
        "Organ and Tissue Authority"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter the shared ATLAS AusTender dataset "
            "to the Health market."
        )
    )

    parser.add_argument(
        "--input",
        default=(
            "data/austender/combined/"
            "austender_combined.parquet"
        ),
    )

    # Kept for backwards compatibility with the current workflow.
    # The canonical Health scope is defined in this script.
    parser.add_argument(
        "--agency-config",
        default=None,
        help=(
            "Compatibility option. The canonical Health "
            "agency aliases are defined in this script."
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
        default="audits/austender/health",
    )

    return parser.parse_args()


def clean(value: object) -> str:
    if value is None or pd.isna(value):
        return ""

    text = (
        str(value)
        .lower()
        .replace("&", " and ")
    )

    text = re.sub(
        r"[^a-z0-9+#./ -]+",
        " ",
        text,
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def canonical_agency_name(
    value: object,
) -> str | None:
    return HEALTH_AGENCY_ALIASES.get(
        clean(value)
    )


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
            f"Shared AusTender parquet not found: "
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
        f"Loading shared AusTender dataset: "
        f"{input_path}"
    )

    df = pd.read_parquet(
        input_path
    )

    if AGENCY_COL not in df.columns:
        raise RuntimeError(
            f"Shared dataset is missing "
            f"required column: {AGENCY_COL}"
        )

    raw_agency = df[
        AGENCY_COL
    ].copy()

    canonical = raw_agency.map(
        canonical_agency_name
    )

    mask = canonical.notna()

    health = df.loc[
        mask
    ].copy()

    if health.empty:
        raise RuntimeError(
            "Health agency scope returned zero rows."
        )

    health[
        RAW_AGENCY_COL
    ] = raw_agency.loc[
        health.index
    ]

    health[
        AGENCY_GROUP_COL
    ] = canonical.loc[
        health.index
    ].astype(str)

    health[
        AGENCY_COL
    ] = health[
        AGENCY_GROUP_COL
    ]

    health[
        "Health Portfolio Scope"
    ] = True

    if VALUE_COL in health.columns:
        health[
            VALUE_COL
        ] = pd.to_numeric(
            health[VALUE_COL],
            errors="coerce",
        ).fillna(0)

    health.to_parquet(
        output_path,
        index=False,
    )

    # ========================================================
    # AUDIT: canonical Health agencies
    # ========================================================

    if VALUE_COL in health.columns:
        agency_summary = (
            health
            .groupby(
                [
                    RAW_AGENCY_COL,
                    AGENCY_COL,
                ],
                dropna=False,
            )
            .agg(
                rows=(
                    CN_ID_COL,
                    "size",
                )
                if CN_ID_COL in health.columns
                else (
                    VALUE_COL,
                    "size",
                ),
                contracts=(
                    CN_ID_COL,
                    "nunique",
                )
                if CN_ID_COL in health.columns
                else (
                    VALUE_COL,
                    "size",
                ),
                value=(
                    VALUE_COL,
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
        agency_summary = (
            health
            .groupby(
                [
                    RAW_AGENCY_COL,
                    AGENCY_COL,
                ],
                dropna=False,
            )
            .size()
            .reset_index(
                name="rows"
            )
        )

    agency_summary.to_csv(
        audit_dir
        / "health_agency_scope_audit.csv",
        index=False,
    )

    total_value = (
        float(
            health[
                VALUE_COL
            ].sum()
        )
        if VALUE_COL in health.columns
        else None
    )

    summary = {
        "input_rows": int(
            len(df)
        ),
        "health_rows": int(
            len(health)
        ),
        "health_value": (
            total_value
        ),
        "canonical_agencies": sorted(
            health[
                AGENCY_COL
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

    print()
    print(
        "ATLAS Health raw filter complete."
    )

    print(
        f"Input rows:  "
        f"{len(df):,}"
    )

    print(
        f"Health rows: "
        f"{len(health):,}"
    )

    if total_value is not None:
        print(
            f"Health procurement: "
            f"A${total_value / 1e9:,.2f}B"
        )

    print(
        f"Wrote: {output_path}"
    )

    print(
        "Audit:"
    )

    print(
        audit_dir
        / "health_agency_scope_audit.csv"
    )


if __name__ == "__main__":
    main()
