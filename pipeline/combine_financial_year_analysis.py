from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# Financial Year Analysis source columns
# ---------------------------------------------------------------------

AGENCY = "01. Agency Name"
CONTRACT_TYPE = "02. Contract Type"
CN_ID = "03. Contract Notice ID"
EXECUTION_DATE = "06. Execution Date"
START_DATE = "07. Start Date"
END_DATE = "08. End Date"
AMENDMENT_REASON = "11. Amendment Reason"
DESCRIPTION = "14. Description"
CATEGORY_TYPE = "15. Category Type"
CATEGORY_CODE = "16. Category Code"
CATEGORY_TITLE = "17. Category Title"
SUPPLIER_NAME = "29. Supplier Name"
SUPPLIER_ABN = "30. Supplier ABN"
AGENCY_DIVISION = "35. Agency Division"
AGENCY_BRANCH = "36. Agency Branch"
VALUE = "48. Value"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine the ATLAS Financial Year Analysis historical baseline "
            "with monthly Financial Year Analysis CSV exports."
        )
    )

    parser.add_argument(
        "--baseline",
        default="data/austender/baseline/austender_baseline.parquet",
    )

    parser.add_argument(
        "--monthly-dir",
        default="data/austender/monthly",
    )

    parser.add_argument(
        "--output",
        default="data/austender/combined/austender_combined.parquet",
    )

    parser.add_argument(
        "--audit-dir",
        default="audits/austender",
    )

    return parser.parse_args()


def column_number(column_name: object) -> int | None:
    match = re.match(r"^\s*(\d{2})\.\s*", str(column_name))
    return int(match.group(1)) if match else None


def validate_financial_year_analysis_schema(
    df: pd.DataFrame,
    source_name: str,
) -> None:
    numbered_columns = {
        column_number(column)
        for column in df.columns
        if column_number(column) is not None
    }

    expected_numbers = set(range(1, 49))
    missing_numbers = sorted(expected_numbers - numbered_columns)

    if missing_numbers:
        raise RuntimeError(
            f"{source_name} is not the expected 48-column Financial Year "
            f"Analysis export. Missing numbered columns: {missing_numbers}"
        )

    required_columns = {
        AGENCY,
        CONTRACT_TYPE,
        CN_ID,
        EXECUTION_DATE,
        START_DATE,
        END_DATE,
        AMENDMENT_REASON,
        DESCRIPTION,
        CATEGORY_TYPE,
        CATEGORY_CODE,
        CATEGORY_TITLE,
        SUPPLIER_NAME,
        SUPPLIER_ABN,
        AGENCY_DIVISION,
        AGENCY_BRANCH,
        VALUE,
    }

    missing_required = sorted(required_columns - set(df.columns))

    if missing_required:
        raise RuntimeError(
            f"{source_name} is missing required Financial Year Analysis "
            f"columns: {', '.join(missing_required)}"
        )


def canonical_cn_root(value: object) -> str:
    if value is None or pd.isna(value):
        return ""

    text = str(value).strip().upper()

    if not text:
        return ""

    return re.sub(r"-A\d+$", "", text)


def amendment_number(value: object) -> int:
    if value is None or pd.isna(value):
        return 0

    text = str(value).strip().upper()
    match = re.search(r"-A(\d+)$", text)

    return int(match.group(1)) if match else 0


def parse_value(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.strip()
    )

    return pd.to_numeric(
        cleaned,
        errors="coerce",
    ).fillna(0.0)


def normalise_source(
    df: pd.DataFrame,
    source_kind: str,
    source_file: str,
) -> pd.DataFrame:
    validate_financial_year_analysis_schema(
        df,
        source_file,
    )

    out = df.copy()

    # -------------------------------------------------------------
    # Normalise all original FYA columns to consistent types.
    #
    # Baseline parquet may have inferred numeric types for fields
    # such as "09. Extension Options", while CSV exports load them
    # as strings. All descriptive/source fields are deliberately
    # stored as text so monthly CSVs and the baseline concatenate
    # consistently and can be written back to parquet safely.
    # -------------------------------------------------------------

    date_columns = {
        EXECUTION_DATE,
        START_DATE,
        END_DATE,
    }

    numeric_columns = {
        VALUE,
    }

    for column in out.columns:
        if column in date_columns or column in numeric_columns:
            continue

        out[column] = (
            out[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    # Dates
    for column in date_columns:
        out[column] = pd.to_datetime(
            out[column],
            errors="coerce",
            dayfirst=True,
        )

    # Contract value
    out[VALUE] = parse_value(
        out[VALUE]
    )

    # Provenance
    out["atlas_source_kind"] = source_kind
    out["atlas_source_file"] = source_file

    # CN reconciliation fields
    out["CN Root ID"] = out[CN_ID].map(
        canonical_cn_root
    )

    out["CN Amendment Number"] = out[CN_ID].map(
        amendment_number
    )

    # Compatibility aliases used elsewhere in ATLAS
    aliases = {
        AGENCY: "Agency",
        CN_ID: "CN ID",
        SUPPLIER_NAME: "Supplier Name",
        SUPPLIER_ABN: "Supplier ABN",
        DESCRIPTION: "Description",
        CATEGORY_TITLE: "Category",
        CATEGORY_CODE: "Category Code",
        CATEGORY_TYPE: "Category Type",
        AGENCY_DIVISION: "Agency Division",
        AGENCY_BRANCH: "Agency Branch",
        EXECUTION_DATE: "Publish Date",
        START_DATE: "Start Date",
        END_DATE: "End Date",
        VALUE: "Value",
    }

    for source_column, alias_column in aliases.items():
        out[alias_column] = out[source_column]

    # Historical spelling retained for older Defence code.
    out["Agency Divison"] = out["Agency Division"]

    return out


def read_baseline(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Historical baseline parquet not found: {path}"
        )

    df = pd.read_parquet(path)

    return normalise_source(
        df,
        source_kind="baseline",
        source_file=path.name,
    )


def read_monthly_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        dtype=str,
        low_memory=False,
        encoding="utf-8-sig",
    )

    return normalise_source(
        df,
        source_kind="monthly",
        source_file=path.name,
    )


def reconcile_current_contracts(
    df: pd.DataFrame,
    audit_dir: Path,
) -> pd.DataFrame:
    """
    Keep one current row per CN family.

    Winner order:
      1. Latest Execution Date
      2. Highest amendment suffix
      3. Monthly source beats baseline on a tie
      4. Latest ingest order as final tie-breaker
    """

    out = df.copy()

    out["_ingest_order"] = range(len(out))

    out["_source_priority"] = (
        out["atlas_source_kind"]
        .map(
            {
                "baseline": 10,
                "monthly": 20,
            }
        )
        .fillna(0)
        .astype(int)
    )

    has_cn = (
        out["CN Root ID"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    )

    duplicate_cn = has_cn & out.duplicated(
        "CN Root ID",
        keep=False,
    )

    if duplicate_cn.any():
        (
            out.loc[duplicate_cn]
            .sort_values(
                [
                    "CN Root ID",
                    EXECUTION_DATE,
                    "CN Amendment Number",
                    "_source_priority",
                    "_ingest_order",
                ],
                na_position="first",
            )
            .to_csv(
                audit_dir / "cn_all_versions.csv",
                index=False,
            )
        )

    keyed = out.loc[has_cn].sort_values(
        [
            "CN Root ID",
            EXECUTION_DATE,
            "CN Amendment Number",
            "_source_priority",
            "_ingest_order",
        ],
        ascending=[
            True,
            True,
            True,
            True,
            True,
        ],
        na_position="first",
    )

    winners = keyed.drop_duplicates(
        "CN Root ID",
        keep="last",
    )

    winner_indexes = set(winners.index)

    if duplicate_cn.any():
        superseded = out.loc[
            duplicate_cn
            & ~out.index.isin(winner_indexes)
        ].copy()

        superseded.to_csv(
            audit_dir / "cn_superseded_versions.csv",
            index=False,
        )

    unkeyed = out.loc[~has_cn]

    current = pd.concat(
        [
            winners,
            unkeyed,
        ],
        ignore_index=True,
        sort=False,
    )

    current.drop(
        columns=[
            "_ingest_order",
            "_source_priority",
        ],
        errors="ignore",
        inplace=True,
    )

    remaining_duplicates = (
        current.loc[
            current["CN Root ID"]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne(""),
            "CN Root ID",
        ]
        .duplicated()
        .sum()
    )

    if remaining_duplicates:
        raise RuntimeError(
            f"Reconciliation failed: "
            f"{remaining_duplicates:,} duplicate CN roots remain."
        )

    return current


def financial_year_label(
    value: object,
) -> str:
    if value is None or pd.isna(value):
        return ""

    timestamp = pd.Timestamp(value)

    start_year = (
        timestamp.year
        if timestamp.month >= 7
        else timestamp.year - 1
    )

    return f"{start_year}-{start_year + 1}"


def add_derived_fields(
    df: pd.DataFrame,
) -> pd.DataFrame:
    out = df.copy()

    out["Value (AUD)"] = out["Value"]

    out["Financial Year"] = out[
        "Publish Date"
    ].map(
        financial_year_label
    )

    duration_days = (
        out["End Date"]
        - out["Start Date"]
    ).dt.days + 1

    duration_years = (
        duration_days / 365.25
    ).clip(lower=1)

    out["Value Per Year"] = (
        out["Value"]
        / duration_years
    )

    invalid_duration = (
        out["Start Date"].isna()
        | out["End Date"].isna()
        | duration_days.le(0)
    )

    out.loc[
        invalid_duration,
        "Value Per Year",
    ] = out.loc[
        invalid_duration,
        "Value",
    ]

    return out


def main() -> None:
    args = parse_args()

    baseline_path = Path(
        args.baseline
    )

    monthly_dir = Path(
        args.monthly_dir
    )

    output_path = Path(
        args.output
    )

    audit_dir = Path(
        args.audit_dir
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
        f"Loading historical baseline: "
        f"{baseline_path}"
    )

    baseline = read_baseline(
        baseline_path
    )

    print(
        f"Baseline rows: "
        f"{len(baseline):,}"
    )

    frames = [
        baseline
    ]

    manifest_rows = [
        {
            "source_kind": "baseline",
            "source_file": baseline_path.name,
            "rows_loaded": int(
                len(baseline)
            ),
            "value_loaded": float(
                baseline["Value"].sum()
            ),
        }
    ]

    monthly_files = (
        sorted(
            monthly_dir.glob("*.csv")
        )
        if monthly_dir.exists()
        else []
    )

    print(
        f"Monthly Financial Year Analysis files found: "
        f"{len(monthly_files)}"
    )

    for path in monthly_files:
        print(
            f"Loading monthly export: "
            f"{path}"
        )

        monthly = read_monthly_csv(
            path
        )

        frames.append(
            monthly
        )

        manifest_rows.append(
            {
                "source_kind": "monthly",
                "source_file": path.name,
                "rows_loaded": int(
                    len(monthly)
                ),
                "value_loaded": float(
                    monthly["Value"].sum()
                ),
            }
        )

    combined_all = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    rows_before = len(
        combined_all
    )

    print(
        f"Rows before CN reconciliation: "
        f"{rows_before:,}"
    )

    current = reconcile_current_contracts(
        combined_all,
        audit_dir,
    )

    current = add_derived_fields(
        current
    )

    current.to_parquet(
        output_path,
        index=False,
    )

    pd.DataFrame(
        manifest_rows
    ).to_csv(
        audit_dir
        / "ingestion_manifest.csv",
        index=False,
    )

    summary = {
        "baseline_rows": int(
            len(baseline)
        ),
        "monthly_files": int(
            len(monthly_files)
        ),
        "rows_before_reconciliation": int(
            rows_before
        ),
        "rows_after_reconciliation": int(
            len(current)
        ),
        "superseded_rows": int(
            rows_before
            - len(current)
        ),
        "unique_cn_roots": int(
            current["CN Root ID"]
            .replace("", pd.NA)
            .nunique()
        ),
        "total_value": float(
            current["Value"].sum()
        ),
        "division_populated_rows": int(
            current["Agency Division"]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
            .sum()
        ),
        "branch_populated_rows": int(
            current["Agency Branch"]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
            .sum()
        ),
        "category_type_populated_rows": int(
            current["Category Type"]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
            .sum()
        ),
        "output": str(
            output_path
        ),
    }

    (
        audit_dir
        / "combine_summary.json"
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
        "ATLAS Financial Year Analysis "
        "combine complete."
    )

    print(
        f"Current rows: "
        f"{len(current):,}"
    )

    print(
        f"Superseded rows: "
        f"{rows_before - len(current):,}"
    )

    print(
        f"Current value: "
        f"${current['Value'].sum():,.2f}"
    )

    print(
        f"Division populated: "
        f"{summary['division_populated_rows']:,}"
    )

    print(
        f"Branch populated: "
        f"{summary['branch_populated_rows']:,}"
    )

    print(
        f"Category Type populated: "
        f"{summary['category_type_populated_rows']:,}"
    )

    print(
        f"Wrote: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()
