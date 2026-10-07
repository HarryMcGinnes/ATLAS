from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


# =============================================================================
# FYA SOURCE COLUMNS
# =============================================================================

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


# =============================================================================
# REVIEWED SHARED SUPPLIER FAMILIES
#
# IMPORTANT:
# - supplier grouping is LABEL-ONLY
# - never remove / duplicate / revalue rows
# - raw Supplier Name and Supplier ABN are never overwritten
# - ABN is retained for audit but is NOT used to blindly propagate families
# =============================================================================

CANONICAL_SUPPLIER_FAMILIES: list[tuple[str, list[str]]] = [
    ("Accenture", [
        r"\bACCENTURE\b",
    ]),
    ("Deloitte", [
        r"\bDELOITTE\b",
        r"\bDELOITTE TOUCHE TOHMATSU\b",
    ]),
    ("EY", [
        r"\bERNST\s*(?:AND|&)\s*YOUNG\b",
        r"^EY(?:\s|$)",
    ]),
    ("KPMG", [
        r"\bKPMG\b",
    ]),
    ("PwC", [
        r"\bPRICEWATERHOUSECOOPERS\b",
        r"\bPRICEWATERHOUSE\s+COOPERS\b",
        r"\bPWC\b",
    ]),
    ("Lockheed Martin", [
        r"\bLOCKHEED\s+MARTIN\b",
    ]),
    ("Leidos", [
        r"\bLEIDOS\b",
    ]),
    ("Fujitsu", [
        r"\bFUJITSU\b",
    ]),
    ("Thales", [
        r"\bTHALES\b",
    ]),
    ("IBM", [
        r"\bIBM\b",
        r"\bINTERNATIONAL\s+BUSINESS\s+MACHINES\b",
    ]),
    ("BAE Systems", [
        r"\bBAE\s+SYSTEMS\b",
    ]),
    ("Boeing", [
        r"\bBOEING\b",
    ]),
    ("Northrop Grumman", [
        r"\bNORTHROP\s+GRUMMAN\b",
    ]),
    ("Raytheon", [
        r"\bRAYTHEON\b",
    ]),
    ("Rheinmetall", [
        r"\bRHEINMETALL\b",
    ]),
    ("Saab", [
        r"\bSAAB\b",
    ]),
    ("Airbus", [
        r"\bAIRBUS\b",
        r"\bEADS\b",
    ]),
    ("DXC Technology", [
        r"\bDXC\b",
    ]),
    ("Microsoft", [
        r"\bMICROSOFT\b",
    ]),
    ("Amazon Web Services", [
        r"\bAMAZON\s+WEB\s+SERVICES\b",
        r"^AWS(?:\s|$)",
    ]),
    ("Oracle", [
        r"\bORACLE\b",
    ]),
    ("SAP", [
        r"^SAP(?:\s|$)",
        r"\bSAP AUSTRALIA\b",
    ]),
    ("Data#3", [
        r"\bDATA\s*#?\s*3\b",
    ]),
    ("Telstra", [
        r"\bTELSTRA\b",
    ]),
    ("Optus", [
        r"\bOPTUS\b",
    ]),
    ("Salesforce", [
        r"\bSALESFORCE\b",
    ]),
    ("ServiceNow", [
        r"\bSERVICENOW\b",
    ]),
    ("Capgemini", [
        r"\bCAPGEMINI\b",
    ]),
    ("Datacom", [
        r"\bDATACOM\b",
    ]),
    ("Aurecon", [
        r"\bAURECON\b",
        r"\bAUGILITY\b",
    ]),
    ("Jacobs", [
        r"\bJACOBS\b",
    ]),
    ("Downer", [
        r"\bDOWNER\b",
    ]),
    ("Ventia", [
        r"\bVENTIA\b",
    ]),
    ("Nova Systems", [
        r"\bNOVA\s+SYSTEMS\b",
    ]),
    ("QinetiQ", [
        r"\bQINETIQ\b",
    ]),
    ("KBR", [
        r"\bKBR\b",
        r"\bKELLOGG\s+BROWN\s+AND\s+ROOT\b",
    ]),
    ("Navantia", [
        r"\bNAVANTIA\b",
    ]),
    ("Synergy Group", [
        r"\bSYNERGY\s+GROUP\b",
    ]),
    ("SME Gateway", [
        r"\bSME\s+GATEWAY\b",
    ]),
]


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine ATLAS historical Financial Year Analysis baseline "
            "with monthly FYA extracts into the shared Silver dataset."
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
        default="audits/austender/combined",
    )

    return parser.parse_args()


# =============================================================================
# GENERIC HELPERS
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


def column_number(value: object) -> int | None:
    match = re.match(
        r"^\s*(\d{2})\.\s*",
        str(value),
    )

    return int(match.group(1)) if match else None


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


def canonical_cn_root(value: object) -> str:
    text = clean_text(value).upper()

    if not text:
        return ""

    return re.sub(
        r"-A\d+$",
        "",
        text,
    )


def amendment_number(value: object) -> int:
    text = clean_text(value).upper()

    match = re.search(
        r"-A(\d+)$",
        text,
    )

    return int(match.group(1)) if match else 0


def normalise_supplier(value: object) -> str:
    text = clean_text(value).upper()

    text = text.replace(
        "&",
        " AND ",
    )

    text = re.sub(
        r"[^A-Z0-9#]+",
        " ",
        text,
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def normalise_abn(value: object) -> str:
    digits = re.sub(
        r"\D",
        "",
        clean_text(value),
    )

    return digits if len(digits) == 11 else ""


def match_supplier_family(
    value: object,
) -> str:
    text = normalise_supplier(
        value
    )

    if not text:
        return ""

    for canonical, patterns in CANONICAL_SUPPLIER_FAMILIES:
        for pattern in patterns:
            if re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            ):
                return canonical

    return ""


# =============================================================================
# FYA SCHEMA VALIDATION
# =============================================================================

def validate_financial_year_analysis_schema(
    df: pd.DataFrame,
    source_name: str,
) -> None:

    numbered_columns = {
        column_number(column)
        for column in df.columns
        if column_number(column) is not None
    }

    missing_numbers = sorted(
        set(range(1, 49))
        - numbered_columns
    )

    if missing_numbers:
        raise RuntimeError(
            f"{source_name} is not the expected FYA dataset. "
            f"Missing numbered columns: {missing_numbers}"
        )

    required = {
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

    missing = sorted(
        required - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            f"{source_name} missing required FYA columns: "
            + ", ".join(missing)
        )


# =============================================================================
# NORMALISE EACH RAW SOURCE WITHOUT DROPPING COLUMNS
# =============================================================================

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

    date_columns = {
        EXECUTION_DATE,
        START_DATE,
        END_DATE,
    }

    for column in out.columns:
        if column in date_columns or column == VALUE:
            continue

        out[column] = (
            out[column]
            .fillna("")
            .astype(str)
            .map(clean_text)
        )

    for column in date_columns:
        out[column] = pd.to_datetime(
            out[column],
            errors="coerce",
            dayfirst=True,
        )

    out[VALUE] = parse_value(
        out[VALUE]
    )

    # -------------------------------------------------------------------------
    # Source provenance
    # -------------------------------------------------------------------------

    out["atlas_source_kind"] = source_kind
    out["atlas_source_file"] = source_file

    # -------------------------------------------------------------------------
    # Contract version fields.
    #
    # IMPORTANT:
    # These are INFORMATIONAL ONLY.
    # We do NOT collapse all amendments into one root row.
    # -------------------------------------------------------------------------

    out["CN Root ID"] = out[CN_ID].map(
        canonical_cn_root
    )

    out["CN Amendment Number"] = out[CN_ID].map(
        amendment_number
    )

    # -------------------------------------------------------------------------
    # Friendly aliases.
    #
    # Original 01.-48. source columns remain untouched.
    # -------------------------------------------------------------------------

    aliases = {
        AGENCY: "Agency",
        CONTRACT_TYPE: "Contract Type",
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

    for source, alias in aliases.items():
        out[alias] = out[source]

    # Keep legacy typo only as a compatibility alias.
    out["Agency Divison"] = out[
        "Agency Division"
    ]

    return out


def read_baseline(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Historical baseline not found: {path}"
        )

    return normalise_source(
        pd.read_parquet(path),
        "baseline",
        path.name,
    )


def read_monthly(path: Path) -> pd.DataFrame:
    return normalise_source(
        pd.read_csv(
            path,
            dtype=str,
            low_memory=False,
            encoding="utf-8-sig",
        ),
        "monthly",
        path.name,
    )


# =============================================================================
# EXACT CONTRACT-VERSION DEDUPLICATION
# =============================================================================

def deduplicate_exact_cn_versions(
    df: pd.DataFrame,
    audit_dir: Path,
) -> pd.DataFrame:
    """
    Preserve amendment history.

    Examples retained as separate records:
        CN123
        CN123-A1
        CN123-A2

    We only reconcile duplicate occurrences of THE SAME exact CN ID,
    which commonly happens when a monthly extract overlaps the baseline.

    Winner priority for an exact CN ID:
      1. latest Execution Date
      2. monthly beats baseline on a tie
      3. latest ingest position
    """

    out = df.copy()

    out["_ingest_order"] = range(
        len(out)
    )

    out["_source_priority"] = (
        out["atlas_source_kind"]
        .map({
            "baseline": 10,
            "monthly": 20,
        })
        .fillna(0)
        .astype(int)
    )

    exact_id = (
        out[CN_ID]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    out["_exact_cn_id"] = exact_id

    has_id = exact_id.ne("")

    duplicate_mask = (
        has_id
        & out.duplicated(
            "_exact_cn_id",
            keep=False,
        )
    )

    if duplicate_mask.any():
        (
            out.loc[duplicate_mask]
            .sort_values(
                [
                    "_exact_cn_id",
                    EXECUTION_DATE,
                    "_source_priority",
                    "_ingest_order",
                ],
                na_position="first",
            )
            .to_csv(
                audit_dir
                / "exact_cn_duplicate_versions.csv",
                index=False,
            )
        )

    keyed = (
        out.loc[has_id]
        .sort_values(
            [
                "_exact_cn_id",
                EXECUTION_DATE,
                "_source_priority",
                "_ingest_order",
            ],
            ascending=[
                True,
                True,
                True,
                True,
            ],
            na_position="first",
        )
    )

    winners = keyed.drop_duplicates(
        "_exact_cn_id",
        keep="last",
    )

    winner_indexes = set(
        winners.index
    )

    if duplicate_mask.any():
        superseded = out.loc[
            duplicate_mask
            & ~out.index.isin(
                winner_indexes
            )
        ].copy()

        superseded.to_csv(
            audit_dir
            / "exact_cn_duplicate_rows_removed.csv",
            index=False,
        )

    unkeyed = out.loc[
        ~has_id
    ]

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
            "_exact_cn_id",
        ],
        errors="ignore",
        inplace=True,
    )

    return current


# =============================================================================
# CONTRACT VERSION METADATA
# =============================================================================

def add_contract_version_metadata(
    df: pd.DataFrame,
) -> pd.DataFrame:

    out = df.copy()

    roots = (
        out["CN Root ID"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    valid = roots.ne("")

    out["cn_version_count"] = 1
    out["is_latest_cn_version"] = True

    if valid.any():

        counts = (
            out.loc[valid]
            .groupby("CN Root ID")
            ["CN ID"]
            .transform("size")
        )

        out.loc[
            valid,
            "cn_version_count",
        ] = counts.astype(int)

        max_amendment = (
            out.loc[valid]
            .groupby("CN Root ID")
            ["CN Amendment Number"]
            .transform("max")
        )

        out.loc[
            valid,
            "is_latest_cn_version",
        ] = (
            out.loc[
                valid,
                "CN Amendment Number",
            ]
            .eq(max_amendment)
        )

    return out


# =============================================================================
# SHARED SUPPLIER NORMALISATION
# =============================================================================

def add_supplier_identity(
    df: pd.DataFrame,
    audit_dir: Path,
) -> pd.DataFrame:

    before_rows = len(df)
    before_value = float(
        pd.to_numeric(
            df["Value"],
            errors="coerce",
        )
        .fillna(0)
        .sum()
    )

    out = df.copy()

    # Raw fields explicitly retained.
    out["Supplier Name Raw"] = out[
        "Supplier Name"
    ]

    out["Supplier ABN Raw"] = out[
        "Supplier ABN"
    ]

    out[
        "supplier_name_normalized"
    ] = out[
        "Supplier Name"
    ].map(
        normalise_supplier
    )

    out[
        "supplier_abn_normalized"
    ] = out[
        "Supplier ABN"
    ].map(
        normalise_abn
    )

    families = out[
        "Supplier Name"
    ].map(
        match_supplier_family
    )

    raw_name = (
        out["Supplier Name"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    out["supplier_group"] = (
        families.where(
            families.ne(""),
            raw_name,
        )
        .replace(
            "",
            "Unknown Supplier",
        )
    )

    out[
        "supplier_mapping_method"
    ] = "RAW_SUPPLIER_NAME"

    family_mask = families.ne("")

    out.loc[
        family_mask,
        "supplier_mapping_method",
    ] = "REVIEWED_FAMILY_ALIAS"

    out[
        "supplier_mapping_status"
    ] = "review"

    out.loc[
        family_mask,
        "supplier_mapping_status",
    ] = "mapped"

    # Stable ID for auditing.
    valid_abn = out[
        "supplier_abn_normalized"
    ].ne("")

    out["supplier_id"] = (
        "NAME:"
        + out[
            "supplier_name_normalized"
        ]
        .replace(
            "",
            "UNKNOWN",
        )
    )

    out.loc[
        valid_abn,
        "supplier_id",
    ] = (
        "ABN:"
        + out.loc[
            valid_abn,
            "supplier_abn_normalized",
        ]
    )

    out["is_accenture"] = out[
        "supplier_group"
    ].eq(
        "Accenture"
    )

    # -------------------------------------------------------------------------
    # Invariants: supplier grouping MUST NOT alter population/value.
    # -------------------------------------------------------------------------

    if len(out) != before_rows:
        raise RuntimeError(
            "Supplier grouping changed row count: "
            f"{before_rows:,} -> {len(out):,}"
        )

    after_value = float(
        pd.to_numeric(
            out["Value"],
            errors="coerce",
        )
        .fillna(0)
        .sum()
    )

    if abs(
        before_value
        - after_value
    ) > 0.01:
        raise RuntimeError(
            "Supplier grouping changed total value."
        )

    # Every raw source name containing Accenture must map to Accenture.
    raw_acc = (
        out[
            "supplier_name_normalized"
        ]
        .str.contains(
            r"\bACCENTURE\b",
            regex=True,
            na=False,
        )
    )

    bad_acc = (
        raw_acc
        & ~out[
            "supplier_group"
        ].eq(
            "Accenture"
        )
    )

    if bad_acc.any():
        sample = out.loc[
            bad_acc,
            [
                "CN ID",
                "Supplier Name",
                "Supplier ABN",
                "supplier_group",
            ],
        ].head(20)

        raise RuntimeError(
            "Accenture supplier reconciliation failed:\n"
            + sample.to_string(
                index=False
            )
        )

    # -------------------------------------------------------------------------
    # Supplier mapping audit
    # -------------------------------------------------------------------------

    mapping = (
        out
        .groupby(
            [
                "Supplier Name",
                "Supplier ABN",
                "supplier_name_normalized",
                "supplier_abn_normalized",
                "supplier_group",
                "supplier_mapping_method",
                "supplier_mapping_status",
                "supplier_id",
            ],
            dropna=False,
        )
        .agg(
            rows=(
                "CN ID",
                "size",
            ),
            contracts=(
                "CN ID",
                "nunique",
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

    mapping.to_csv(
        audit_dir
        / "supplier_identity_mapping.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Review queue: unmapped/unreviewed supplier names.
    # -------------------------------------------------------------------------

    (
        mapping.loc[
            mapping[
                "supplier_mapping_status"
            ].eq(
                "review"
            )
        ]
        .sort_values(
            "value",
            ascending=False,
        )
        .to_csv(
            audit_dir
            / "supplier_review_queue.csv",
            index=False,
        )
    )

    # -------------------------------------------------------------------------
    # ABN variant audit.
    #
    # ABN is evidence only; this does NOT change supplier_group.
    # -------------------------------------------------------------------------

    abn_rows = mapping.loc[
        mapping[
            "supplier_abn_normalized"
        ].ne("")
    ].copy()

    if not abn_rows.empty:

        abn_variants = (
            abn_rows
            .groupby(
                "supplier_abn_normalized",
                dropna=False,
            )
            .agg(
                supplier_names=(
                    "Supplier Name",
                    lambda values: " | ".join(
                        sorted(
                            {
                                clean_text(v)
                                for v in values
                                if clean_text(v)
                            }
                        )
                    ),
                ),
                supplier_groups=(
                    "supplier_group",
                    lambda values: " | ".join(
                        sorted(
                            {
                                clean_text(v)
                                for v in values
                                if clean_text(v)
                            }
                        )
                    ),
                ),
                raw_name_count=(
                    "Supplier Name",
                    "nunique",
                ),
                canonical_group_count=(
                    "supplier_group",
                    "nunique",
                ),
                value=(
                    "value",
                    "sum",
                ),
            )
            .reset_index()
            .sort_values(
                "value",
                ascending=False,
            )
        )

        abn_variants.to_csv(
            audit_dir
            / "supplier_abn_variants.csv",
            index=False,
        )

        (
            abn_variants.loc[
                abn_variants[
                    "canonical_group_count"
                ].gt(1)
            ]
            .to_csv(
                audit_dir
                / "supplier_abn_review_queue.csv",
                index=False,
            )
        )

    return out


# =============================================================================
# DERIVED SHARED FIELDS
# =============================================================================

def financial_year_label(
    value: object,
) -> str:

    if value is None or pd.isna(value):
        return ""

    timestamp = pd.Timestamp(
        value
    )

    start_year = (
        timestamp.year
        if timestamp.month >= 7
        else timestamp.year - 1
    )

    return (
        f"{start_year}-"
        f"{start_year + 1}"
    )


def add_derived_fields(
    df: pd.DataFrame,
) -> pd.DataFrame:

    out = df.copy()

    out["Value (AUD)"] = out[
        "Value"
    ]

    out[
        "Financial Year"
    ] = out[
        "Publish Date"
    ].map(
        financial_year_label
    )

    duration_days = (
        out[
            "End Date"
        ]
        - out[
            "Start Date"
        ]
    ).dt.days + 1

    duration_years = (
        duration_days
        / 365.25
    ).clip(
        lower=1
    )

    out[
        "Value Per Year"
    ] = (
        out[
            "Value"
        ]
        / duration_years
    )

    invalid_duration = (
        out[
            "Start Date"
        ].isna()
        | out[
            "End Date"
        ].isna()
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


# =============================================================================
# MAIN
# =============================================================================

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

    # -------------------------------------------------------------------------
    # Bronze inputs
    # -------------------------------------------------------------------------

    print(
        "Loading historical baseline: "
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

    manifest = [
        {
            "source_kind": "baseline",
            "source_file": baseline_path.name,
            "rows": int(
                len(baseline)
            ),
            "value": float(
                baseline[
                    "Value"
                ].sum()
            ),
        }
    ]

    monthly_files = (
        sorted(
            monthly_dir.glob(
                "*.csv"
            )
        )
        if monthly_dir.exists()
        else []
    )

    print(
        "Monthly FYA files found: "
        f"{len(monthly_files)}"
    )

    for path in monthly_files:

        print(
            f"Loading monthly extract: "
            f"{path}"
        )

        frame = read_monthly(
            path
        )

        frames.append(
            frame
        )

        manifest.append({
            "source_kind": "monthly",
            "source_file": path.name,
            "rows": int(
                len(frame)
            ),
            "value": float(
                frame[
                    "Value"
                ].sum()
            ),
        })

    combined_all = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    rows_before = len(
        combined_all
    )

    value_before = float(
        combined_all[
            "Value"
        ].sum()
    )

    print()
    print(
        f"Rows before exact-CN dedupe: "
        f"{rows_before:,}"
    )

    print(
        f"Value before exact-CN dedupe: "
        f"${value_before:,.2f}"
    )

    # -------------------------------------------------------------------------
    # Silver: dedupe only exact CN IDs
    # -------------------------------------------------------------------------

    combined = (
        deduplicate_exact_cn_versions(
            combined_all,
            audit_dir,
        )
    )

    print(
        f"Rows after exact-CN dedupe: "
        f"{len(combined):,}"
    )

    print(
        "Amendment CNs remain separate."
    )

    # -------------------------------------------------------------------------
    # Add amendment/version metadata
    # -------------------------------------------------------------------------

    combined = (
        add_contract_version_metadata(
            combined
        )
    )

    # -------------------------------------------------------------------------
    # Shared supplier identity
    # -------------------------------------------------------------------------

    print(
        "Applying shared supplier grouping..."
    )

    combined = add_supplier_identity(
        combined,
        audit_dir,
    )

    # -------------------------------------------------------------------------
    # Shared derived fields
    # -------------------------------------------------------------------------

    combined = add_derived_fields(
        combined
    )

    # -------------------------------------------------------------------------
    # Write Silver parquet
    # -------------------------------------------------------------------------

    combined.to_parquet(
        output_path,
        index=False,
    )

    pd.DataFrame(
        manifest
    ).to_csv(
        audit_dir
        / "ingestion_manifest.csv",
        index=False,
    )

    mapped = int(
        combined[
            "supplier_mapping_status"
        ]
        .eq(
            "mapped"
        )
        .sum()
    )

    review = int(
        combined[
            "supplier_mapping_status"
        ]
        .eq(
            "review"
        )
        .sum()
    )

    accenture = combined[
        combined[
            "is_accenture"
        ]
    ]

    summary = {
        "baseline_rows": int(
            len(baseline)
        ),
        "monthly_files": int(
            len(monthly_files)
        ),
        "rows_before_exact_cn_dedupe": int(
            rows_before
        ),
        "rows_after_exact_cn_dedupe": int(
            len(combined)
        ),
        "removed_exact_duplicate_cn_rows": int(
            rows_before
            - len(combined)
        ),
        "total_value": float(
            combined[
                "Value"
            ].sum()
        ),
        "supplier_groups": int(
            combined[
                "supplier_group"
            ].nunique()
        ),
        "mapped_supplier_rows": mapped,
        "supplier_review_rows": review,
        "accenture_rows": int(
            len(accenture)
        ),
        "accenture_value": float(
            accenture[
                "Value"
            ].sum()
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
        "========================================="
    )

    print(
        "ATLAS SHARED SILVER BUILD COMPLETE"
    )

    print(
        "========================================="
    )

    print(
        f"Rows: "
        f"{len(combined):,}"
    )

    print(
        f"Value: "
        f"${summary['total_value']:,.2f}"
    )

    print(
        f"Supplier groups: "
        f"{summary['supplier_groups']:,}"
    )

    print(
        f"Reviewed supplier rows: "
        f"{mapped:,}"
    )

    print(
        f"Supplier review rows: "
        f"{review:,}"
    )

    print(
        f"Accenture rows: "
        f"{summary['accenture_rows']:,}"
    )

    print(
        f"Accenture value: "
        f"${summary['accenture_value']:,.2f}"
    )

    print()
    print(
        f"Wrote: {output_path}"
    )

    print(
        "========================================="
    )


if __name__ == "__main__":
    main()
