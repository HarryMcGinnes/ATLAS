from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


# =============================================================================
# FINANCIAL YEAR ANALYSIS SOURCE COLUMNS
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
# SHARED SUPPLIER IDENTITY
# =============================================================================
#
# Supplier identity is a whole-of-ATLAS concept.
#
# Rules:
#   1. Preserve the original Supplier Name and Supplier ABN.
#   2. Normalise ABN and name into separate audit fields.
#   3. Where an ABN exists, use the ABN as the strongest identity signal.
#   4. Reviewed supplier-family rules provide the canonical display label.
#   5. If an ABN has no reviewed family, select one stable representative
#      source name for that ABN rather than splitting the supplier.
#   6. If no ABN exists, use reviewed name rules.
#   7. If nothing is recognised, KEEP the source supplier name unchanged.
#      Never drop an unknown supplier.
#   8. Conflicting reviewed supplier families on the same ABN are treated as
#      a hard pipeline error rather than silently merged.
#
# Defence and Health should consume supplier_group from this dataset rather
# than inventing separate supplier identities downstream.
# =============================================================================

SUPPLIER_FAMILIES: list[tuple[str, list[str]]] = [
    (
        "Accenture",
        [
            r"\bACCENTURE\b",
        ],
    ),
    (
        "Deloitte",
        [
            r"\bDELOITTE\b",
        ],
    ),
    (
        "EY",
        [
            r"\bERNST\s+(?:AND\s+)?YOUNG\b",
            r"^EY(?:\s|$)",
        ],
    ),
    (
        "KPMG",
        [
            r"\bKPMG\b",
        ],
    ),
    (
        "PwC",
        [
            r"\bPRICEWATERHOUSECOOPERS\b",
            r"\bPRICEWATERHOUSE\s+COOPERS\b",
            r"^PWC(?:\s|$)",
        ],
    ),
    (
        "IBM",
        [
            r"\bIBM\b",
            r"\bINTERNATIONAL\s+BUSINESS\s+MACHINES\b",
        ],
    ),
    (
        "Fujitsu",
        [
            r"\bFUJITSU\b",
        ],
    ),
    (
        "DXC Technology",
        [
            r"\bDXC\b",
            r"\bCSC AUSTRALIA\b",
        ],
    ),
    (
        "BAE Systems",
        [
            r"\bBAE\s+SYSTEMS\b",
        ],
    ),
    (
        "Boeing",
        [
            r"\bBOEING\b",
        ],
    ),
    (
        "Lockheed Martin",
        [
            r"\bLOCKHEED\s+MARTIN\b",
        ],
    ),
    (
        "Leidos",
        [
            r"\bLEIDOS\b",
        ],
    ),
    (
        "Thales",
        [
            r"\bTHALES\b",
        ],
    ),
    (
        "Raytheon",
        [
            r"\bRAYTHEON\b",
        ],
    ),
    (
        "Aurecon",
        [
            r"\bAURECON\b",
            r"\bAUGILITY\b",
        ],
    ),
    (
        "Downer",
        [
            r"\bDOWNER\b",
        ],
    ),
    (
        "Jacobs",
        [
            r"\bJACOBS\b",
        ],
    ),
    (
        "KBR",
        [
            r"\bKBR\b",
            r"\bKELLOGG\s+BROWN\s+AND\s+ROOT\b",
        ],
    ),
    (
        "QinetiQ",
        [
            r"\bQINETIQ\b",
        ],
    ),
    (
        "Navantia",
        [
            r"\bNAVANTIA\b",
        ],
    ),
    (
        "Nova Systems",
        [
            r"\bNOVA\s+SYSTEMS\b",
        ],
    ),
    (
        "Synergy Group",
        [
            r"\bSYNERGY\s+GROUP\b",
        ],
    ),
    (
        "Telstra",
        [
            r"\bTELSTRA\b",
        ],
    ),
    (
        "Optus",
        [
            r"\bOPTUS\b",
        ],
    ),
    (
        "Data#3",
        [
            r"\bDATA\s*#?\s*3\b",
        ],
    ),
    (
        "Microsoft",
        [
            r"\bMICROSOFT\b",
        ],
    ),
    (
        "Amazon Web Services",
        [
            r"\bAMAZON\s+WEB\s+SERVICES\b",
            r"^AWS(?:\s|$)",
        ],
    ),
    (
        "Oracle",
        [
            r"\bORACLE\b",
        ],
    ),
    (
        "SAP",
        [
            r"^SAP(?:\s|$)",
            r"\bSAP\s+AUSTRALIA\b",
        ],
    ),
    (
        "Capgemini",
        [
            r"\bCAPGEMINI\b",
        ],
    ),
    (
        "Datacom",
        [
            r"\bDATACOM\b",
        ],
    ),
    (
        "Salesforce",
        [
            r"\bSALESFORCE\b",
            r"\bSFDC\b",
        ],
    ),
    (
        "ServiceNow",
        [
            r"\bSERVICENOW\b",
        ],
    ),
    (
        "SME Gateway",
        [
            r"\bSME\s+GATEWAY\b",
        ],
    ),
]


# =============================================================================
# CLI
# =============================================================================

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


# =============================================================================
# SOURCE VALIDATION
# =============================================================================

def column_number(column_name: object) -> int | None:
    match = re.match(
        r"^\s*(\d{2})\.\s*",
        str(column_name),
    )

    return int(
        match.group(1)
    ) if match else None


def validate_financial_year_analysis_schema(
    df: pd.DataFrame,
    source_name: str,
) -> None:

    numbered_columns = {
        column_number(column)
        for column in df.columns
        if column_number(column) is not None
    }

    expected_numbers = set(
        range(1, 49)
    )

    missing_numbers = sorted(
        expected_numbers
        - numbered_columns
    )

    if missing_numbers:
        raise RuntimeError(
            f"{source_name} is not the expected "
            f"48-column Financial Year Analysis export. "
            f"Missing numbered columns: {missing_numbers}"
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

    missing_required = sorted(
        required_columns
        - set(df.columns)
    )

    if missing_required:
        raise RuntimeError(
            f"{source_name} is missing required "
            f"Financial Year Analysis columns: "
            f"{', '.join(missing_required)}"
        )


# =============================================================================
# CONTRACT ID HELPERS
# =============================================================================

def canonical_cn_root(
    value: object,
) -> str:

    if value is None or pd.isna(value):
        return ""

    text = str(
        value
    ).strip().upper()

    if not text:
        return ""

    return re.sub(
        r"-A\d+$",
        "",
        text,
    )


def amendment_number(
    value: object,
) -> int:

    if value is None or pd.isna(value):
        return 0

    text = str(
        value
    ).strip().upper()

    match = re.search(
        r"-A(\d+)$",
        text,
    )

    return int(
        match.group(1)
    ) if match else 0


# =============================================================================
# BASIC VALUE / TEXT HELPERS
# =============================================================================

def parse_value(
    series: pd.Series,
) -> pd.Series:

    cleaned = (
        series
        .astype(str)
        .str.replace(
            ",",
            "",
            regex=False,
        )
        .str.replace(
            "$",
            "",
            regex=False,
        )
        .str.strip()
    )

    return pd.to_numeric(
        cleaned,
        errors="coerce",
    ).fillna(0.0)


def clean_source_text(
    value: object,
) -> str:

    if value is None or pd.isna(value):
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value)
        .replace("\u00a0", " ")
        .strip(),
    )


# =============================================================================
# SUPPLIER NORMALISATION HELPERS
# =============================================================================

def normalise_supplier_name(
    value: object,
) -> str:

    raw = clean_source_text(
        value
    ).upper()

    raw = raw.replace(
        "&",
        " AND ",
    )

    raw = re.sub(
        r"[^A-Z0-9#]+",
        " ",
        raw,
    )

    return re.sub(
        r"\s+",
        " ",
        raw,
    ).strip()


def normalise_supplier_abn(
    value: object,
) -> str:

    if value is None or pd.isna(value):
        return ""

    digits = re.sub(
        r"\D",
        "",
        str(value),
    )

    digits = digits.lstrip("0")

    # Australian ABNs are normally 11 digits.
    # We keep non-empty source digits for audit purposes,
    # but only 11-digit values are treated as strong ABN IDs.
    return digits


def valid_abn(
    value: object,
) -> bool:

    text = normalise_supplier_abn(
        value
    )

    return bool(
        re.fullmatch(
            r"\d{11}",
            text,
        )
    )


def reviewed_supplier_family(
    value: object,
) -> str | None:

    normalised = normalise_supplier_name(
        value
    )

    if not normalised:
        return None

    for (
        canonical,
        patterns,
    ) in SUPPLIER_FAMILIES:

        if any(
            re.search(
                pattern,
                normalised,
                flags=re.IGNORECASE,
            )
            for pattern in patterns
        ):
            return canonical

    return None


# =============================================================================
# SOURCE NORMALISATION
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

    numeric_columns = {
        VALUE,
    }

    for column in out.columns:

        if (
            column in date_columns
            or column in numeric_columns
        ):
            continue

        out[column] = (
            out[column]
            .fillna("")
            .astype(str)
            .map(clean_source_text)
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

    # -------------------------------------------------------------
    # Provenance
    # -------------------------------------------------------------

    out[
        "atlas_source_kind"
    ] = source_kind

    out[
        "atlas_source_file"
    ] = source_file

    # -------------------------------------------------------------
    # CN reconciliation fields
    # -------------------------------------------------------------

    out[
        "CN Root ID"
    ] = out[
        CN_ID
    ].map(
        canonical_cn_root
    )

    out[
        "CN Amendment Number"
    ] = out[
        CN_ID
    ].map(
        amendment_number
    )

    # -------------------------------------------------------------
    # Compatibility aliases used elsewhere in ATLAS
    # -------------------------------------------------------------

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

    for (
        source_column,
        alias_column,
    ) in aliases.items():

        out[
            alias_column
        ] = out[
            source_column
        ]

    # Historical spelling retained for old Defence code.
    out[
        "Agency Divison"
    ] = out[
        "Agency Division"
    ]

    return out


# =============================================================================
# READ INPUTS
# =============================================================================

def read_baseline(
    path: Path,
) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(
            f"Historical baseline parquet "
            f"not found: {path}"
        )

    df = pd.read_parquet(
        path
    )

    return normalise_source(
        df,
        source_kind="baseline",
        source_file=path.name,
    )


def read_monthly_csv(
    path: Path,
) -> pd.DataFrame:

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


# =============================================================================
# CN / AMENDMENT RECONCILIATION
# =============================================================================

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

    out[
        "_ingest_order"
    ] = range(
        len(out)
    )

    out[
        "_source_priority"
    ] = (
        out[
            "atlas_source_kind"
        ]
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
        out[
            "CN Root ID"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    )

    duplicate_cn = (
        has_cn
        & out.duplicated(
            "CN Root ID",
            keep=False,
        )
    )

    if duplicate_cn.any():

        (
            out.loc[
                duplicate_cn
            ]
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
                audit_dir
                / "cn_all_versions.csv",
                index=False,
            )
        )

    keyed = (
        out.loc[
            has_cn
        ]
        .sort_values(
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
    )

    winners = (
        keyed
        .drop_duplicates(
            "CN Root ID",
            keep="last",
        )
    )

    winner_indexes = set(
        winners.index
    )

    if duplicate_cn.any():

        superseded = (
            out.loc[
                duplicate_cn
                & ~out.index.isin(
                    winner_indexes
                )
            ]
            .copy()
        )

        superseded.to_csv(
            audit_dir
            / "cn_superseded_versions.csv",
            index=False,
        )

    unkeyed = out.loc[
        ~has_cn
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
        ],
        errors="ignore",
        inplace=True,
    )

    remaining_duplicates = (
        current.loc[
            current[
                "CN Root ID"
            ]
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
            "Reconciliation failed: "
            f"{remaining_duplicates:,} "
            "duplicate CN roots remain."
        )

    return current


# =============================================================================
# SHARED SUPPLIER IDENTITY
# =============================================================================

def add_shared_supplier_identity(
    df: pd.DataFrame,
    audit_dir: Path,
) -> pd.DataFrame:

    out = df.copy()

    if "Supplier Name" not in out.columns:
        raise RuntimeError(
            "Combined dataset does not contain Supplier Name."
        )

    if "Supplier ABN" not in out.columns:
        raise RuntimeError(
            "Combined dataset does not contain Supplier ABN."
        )

    # -------------------------------------------------------------
    # Preserve source values explicitly
    # -------------------------------------------------------------

    out[
        "Supplier Name Raw"
    ] = out[
        "Supplier Name"
    ]

    out[
        "Supplier ABN Raw"
    ] = out[
        "Supplier ABN"
    ]

    # -------------------------------------------------------------
    # Normalised evidence fields
    # -------------------------------------------------------------

    out[
        "supplier_name_normalized"
    ] = out[
        "Supplier Name"
    ].map(
        normalise_supplier_name
    )

    out[
        "supplier_abn_normalized"
    ] = out[
        "Supplier ABN"
    ].map(
        normalise_supplier_abn
    )

    out[
        "_supplier_reviewed_name_group"
    ] = out[
        "Supplier Name"
    ].map(
        reviewed_supplier_family
    )

    # -------------------------------------------------------------
    # Find ABN → reviewed group relationships
    # -------------------------------------------------------------

    valid_abn_mask = (
        out[
            "supplier_abn_normalized"
        ]
        .astype(str)
        .str.fullmatch(
            r"\d{11}",
            na=False,
        )
    )

    abn_reviewed = (
        out.loc[
            valid_abn_mask
            & out[
                "_supplier_reviewed_name_group"
            ].notna(),
            [
                "supplier_abn_normalized",
                "_supplier_reviewed_name_group",
            ],
        ]
        .drop_duplicates()
    )

    conflict_counts = (
        abn_reviewed
        .groupby(
            "supplier_abn_normalized"
        )[
            "_supplier_reviewed_name_group"
        ]
        .nunique()
    )

    conflicting_abns = set(
        conflict_counts[
            conflict_counts > 1
        ].index
    )

    if conflicting_abns:

        conflict_rows = (
            out.loc[
                out[
                    "supplier_abn_normalized"
                ].isin(
                    conflicting_abns
                ),
                [
                    "Supplier Name",
                    "Supplier ABN",
                    "supplier_name_normalized",
                    "supplier_abn_normalized",
                    "_supplier_reviewed_name_group",
                    "CN ID",
                    "Value",
                ],
            ]
            .sort_values(
                [
                    "supplier_abn_normalized",
                    "Supplier Name",
                ]
            )
        )

        conflict_rows.to_csv(
            audit_dir
            / "supplier_abn_conflicts.csv",
            index=False,
        )

        raise RuntimeError(
            "Shared supplier identity found "
            f"{len(conflicting_abns):,} ABN(s) "
            "mapped to more than one reviewed "
            "canonical supplier. Review "
            "supplier_abn_conflicts.csv."
        )

    abn_to_reviewed_group = (
        abn_reviewed
        .drop_duplicates(
            "supplier_abn_normalized"
        )
        .set_index(
            "supplier_abn_normalized"
        )[
            "_supplier_reviewed_name_group"
        ]
        .to_dict()
    )

    # -------------------------------------------------------------
    # For unknown ABNs, determine one stable source-name label.
    #
    # Ranking:
    #   1. number of rows
    #   2. total contract value
    #   3. alphabetical raw name
    # -------------------------------------------------------------

    abn_name_population = (
        out.loc[
            valid_abn_mask
        ]
        .groupby(
            [
                "supplier_abn_normalized",
                "Supplier Name",
            ],
            dropna=False,
        )
        .agg(
            rows=(
                "Supplier Name",
                "size",
            ),
            total_value=(
                "Value",
                "sum",
            ),
        )
        .reset_index()
    )

    abn_name_population[
        "_supplier_name_sort"
    ] = (
        abn_name_population[
            "Supplier Name"
        ]
        .fillna("")
        .astype(str)
        .str.upper()
    )

    abn_representatives = (
        abn_name_population
        .sort_values(
            [
                "supplier_abn_normalized",
                "rows",
                "total_value",
                "_supplier_name_sort",
            ],
            ascending=[
                True,
                False,
                False,
                True,
            ],
        )
        .drop_duplicates(
            "supplier_abn_normalized",
            keep="first",
        )
    )

    abn_to_representative_name = (
        abn_representatives
        .set_index(
            "supplier_abn_normalized"
        )[
            "Supplier Name"
        ]
        .to_dict()
    )

    # -------------------------------------------------------------
    # Assign canonical supplier identity
    # -------------------------------------------------------------

    supplier_groups: list[str] = []
    mapping_methods: list[str] = []
    supplier_ids: list[str] = []
    mapping_statuses: list[str] = []

    for row in out.itertuples(
        index=False
    ):

        raw_name = clean_source_text(
            getattr(
                row,
                "Supplier_Name",
                "",
            )
            if hasattr(
                row,
                "Supplier_Name"
            )
            else ""
        )

        # itertuples sanitises names containing spaces, so use
        # dataframe values by position below instead.
        supplier_groups.append("")
        mapping_methods.append("")
        supplier_ids.append("")
        mapping_statuses.append("")

    # Replace placeholder lists using direct series iteration.
    supplier_groups = []
    mapping_methods = []
    supplier_ids = []
    mapping_statuses = []

    for (
        raw_name,
        normalized_name,
        normalized_abn,
        reviewed_name_group,
    ) in zip(
        out[
            "Supplier Name"
        ],
        out[
            "supplier_name_normalized"
        ],
        out[
            "supplier_abn_normalized"
        ],
        out[
            "_supplier_reviewed_name_group"
        ],
    ):

        raw_name_clean = (
            clean_source_text(
                raw_name
            )
            or "Unknown Supplier"
        )

        reviewed_group = (
            str(
                reviewed_name_group
            ).strip()
            if pd.notna(
                reviewed_name_group
            )
            else ""
        )

        abn = str(
            normalized_abn
            or ""
        ).strip()

        has_valid_abn = bool(
            re.fullmatch(
                r"\d{11}",
                abn,
            )
        )

        if has_valid_abn:

            reviewed_from_abn = (
                abn_to_reviewed_group.get(
                    abn
                )
            )

            if reviewed_from_abn:

                supplier_group = (
                    reviewed_from_abn
                )

                mapping_method = (
                    "ABN + reviewed supplier family"
                )

                mapping_status = "mapped"

            else:

                supplier_group = (
                    clean_source_text(
                        abn_to_representative_name.get(
                            abn,
                            raw_name_clean,
                        )
                    )
                    or raw_name_clean
                )

                mapping_method = (
                    "ABN representative source name"
                )

                mapping_status = "abn_grouped"

            supplier_id = (
                f"ABN:{abn}"
            )

        elif reviewed_group:

            supplier_group = (
                reviewed_group
            )

            mapping_method = (
                "Reviewed supplier name family"
            )

            mapping_status = "mapped"

            supplier_id = (
                "NAME:"
                + normalise_supplier_name(
                    reviewed_group
                )
            )

        else:

            supplier_group = (
                raw_name_clean
            )

            mapping_method = (
                "Raw supplier name preserved"
            )

            mapping_status = "review"

            supplier_id = (
                "NAME:"
                + (
                    normalized_name
                    or "UNKNOWN"
                )
            )

        supplier_groups.append(
            supplier_group
        )

        mapping_methods.append(
            mapping_method
        )

        supplier_ids.append(
            supplier_id
        )

        mapping_statuses.append(
            mapping_status
        )

    out[
        "supplier_group"
    ] = supplier_groups

    out[
        "supplier_mapping_method"
    ] = mapping_methods

    out[
        "supplier_mapping_status"
    ] = mapping_statuses

    out[
        "supplier_id"
    ] = supplier_ids

    out[
        "is_accenture"
    ] = out[
        "supplier_group"
    ].eq(
        "Accenture"
    )

    # -------------------------------------------------------------
    # Hard supplier invariants
    # -------------------------------------------------------------

    blank_supplier_group = (
        out[
            "supplier_group"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
    )

    if blank_supplier_group.any():

        raise RuntimeError(
            "Shared supplier identity failed: "
            f"{int(blank_supplier_group.sum()):,} "
            "rows have a blank supplier_group."
        )

    blank_supplier_id = (
        out[
            "supplier_id"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
    )

    if blank_supplier_id.any():

        raise RuntimeError(
            "Shared supplier identity failed: "
            f"{int(blank_supplier_id.sum()):,} "
            "rows have a blank supplier_id."
        )

    # -------------------------------------------------------------
    # AUDIT 1: raw supplier → canonical mapping
    # -------------------------------------------------------------

    supplier_mapping_audit = (
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
            total_value=(
                "Value",
                "sum",
            ),
        )
        .reset_index()
        .sort_values(
            "total_value",
            ascending=False,
        )
    )

    supplier_mapping_audit.to_csv(
        audit_dir
        / "supplier_identity_mapping.csv",
        index=False,
    )

    # -------------------------------------------------------------
    # AUDIT 2: canonical groups and all variants feeding them
    # -------------------------------------------------------------

    variant_audit = (
        supplier_mapping_audit
        .groupby(
            [
                "supplier_group",
                "supplier_id",
            ],
            dropna=False,
        )
        .agg(
            raw_names=(
                "Supplier Name",
                lambda s: " | ".join(
                    sorted(
                        {
                            str(v).strip()
                            for v in s
                            if str(v).strip()
                        }
                    )
                ),
            ),
            raw_abns=(
                "Supplier ABN",
                lambda s: " | ".join(
                    sorted(
                        {
                            str(v).strip()
                            for v in s
                            if str(v).strip()
                        }
                    )
                ),
            ),
            source_variants=(
                "Supplier Name",
                "nunique",
            ),
            rows=(
                "rows",
                "sum",
            ),
            contracts=(
                "contracts",
                "sum",
            ),
            total_value=(
                "total_value",
                "sum",
            ),
        )
        .reset_index()
        .sort_values(
            "total_value",
            ascending=False,
        )
    )

    variant_audit.to_csv(
        audit_dir
        / "supplier_group_variants.csv",
        index=False,
    )

    # -------------------------------------------------------------
    # AUDIT 3: suppliers not matched to reviewed families
    #
    # These are NOT dropped. They remain valid suppliers but are
    # surfaced for future review.
    # -------------------------------------------------------------

    review_audit = (
        supplier_mapping_audit.loc[
            supplier_mapping_audit[
                "supplier_mapping_status"
            ].eq(
                "review"
            )
        ]
        .copy()
        .sort_values(
            "total_value",
            ascending=False,
        )
    )

    review_audit.to_csv(
        audit_dir
        / "supplier_review_queue.csv",
        index=False,
    )

    # -------------------------------------------------------------
    # AUDIT 4: ABN groups with multiple source names
    # -------------------------------------------------------------

    multi_name_abn = (
        supplier_mapping_audit.loc[
            supplier_mapping_audit[
                "supplier_abn_normalized"
            ]
            .astype(str)
            .str.fullmatch(
                r"\d{11}",
                na=False,
            )
        ]
        .groupby(
            "supplier_abn_normalized"
        )
        .agg(
            raw_name_count=(
                "Supplier Name",
                "nunique",
            ),
            canonical_group_count=(
                "supplier_group",
                "nunique",
            ),
            raw_names=(
                "Supplier Name",
                lambda s: " | ".join(
                    sorted(
                        {
                            str(v).strip()
                            for v in s
                            if str(v).strip()
                        }
                    )
                ),
            ),
            canonical_groups=(
                "supplier_group",
                lambda s: " | ".join(
                    sorted(
                        {
                            str(v).strip()
                            for v in s
                            if str(v).strip()
                        }
                    )
                ),
            ),
            total_value=(
                "total_value",
                "sum",
            ),
        )
        .reset_index()
    )

    multi_name_abn = (
        multi_name_abn.loc[
            multi_name_abn[
                "raw_name_count"
            ] > 1
        ]
        .sort_values(
            "total_value",
            ascending=False,
        )
    )

    multi_name_abn.to_csv(
        audit_dir
        / "supplier_abn_name_variants.csv",
        index=False,
    )

    # Remove internal helper column only.
    out.drop(
        columns=[
            "_supplier_reviewed_name_group",
        ],
        errors="ignore",
        inplace=True,
    )

    return out


# =============================================================================
# DERIVED FIELDS
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

    out[
        "Value (AUD)"
    ] = out[
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

    # -----------------------------------------------------------------
    # Baseline
    # -----------------------------------------------------------------

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

    manifest_rows = [
        {
            "source_kind": "baseline",
            "source_file": baseline_path.name,
            "rows_loaded": int(
                len(baseline)
            ),
            "value_loaded": float(
                baseline[
                    "Value"
                ].sum()
            ),
        }
    ]

    # -----------------------------------------------------------------
    # Monthly extracts
    # -----------------------------------------------------------------

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
        "Monthly Financial Year Analysis files found: "
        f"{len(monthly_files)}"
    )

    for path in monthly_files:

        print(
            "Loading monthly export: "
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
                    monthly[
                        "Value"
                    ].sum()
                ),
            }
        )

    # -----------------------------------------------------------------
    # Concatenate
    # -----------------------------------------------------------------

    combined_all = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    rows_before = len(
        combined_all
    )

    print(
        "Rows before CN reconciliation: "
        f"{rows_before:,}"
    )

    # -----------------------------------------------------------------
    # CN reconciliation
    # -----------------------------------------------------------------

    current = reconcile_current_contracts(
        combined_all,
        audit_dir,
    )

    print(
        "Rows after CN reconciliation: "
        f"{len(current):,}"
    )

    # -----------------------------------------------------------------
    # Shared supplier identity
    # -----------------------------------------------------------------

    print(
        "Applying shared ATLAS supplier identity..."
    )

    current = add_shared_supplier_identity(
        current,
        audit_dir,
    )

    mapped_rows = int(
        current[
            "supplier_mapping_status"
        ].eq(
            "mapped"
        ).sum()
    )

    abn_grouped_rows = int(
        current[
            "supplier_mapping_status"
        ].eq(
            "abn_grouped"
        ).sum()
    )

    review_rows = int(
        current[
            "supplier_mapping_status"
        ].eq(
            "review"
        ).sum()
    )

    print(
        f"Supplier mapped rows:      "
        f"{mapped_rows:,}"
    )

    print(
        f"Supplier ABN-grouped rows: "
        f"{abn_grouped_rows:,}"
    )

    print(
        f"Supplier review rows:      "
        f"{review_rows:,}"
    )

    print(
        "Canonical supplier groups: "
        f"{current['supplier_group'].nunique():,}"
    )

    # -----------------------------------------------------------------
    # Derived ATLAS fields
    # -----------------------------------------------------------------

    current = add_derived_fields(
        current
    )

    # -----------------------------------------------------------------
    # Write canonical combined parquet
    # -----------------------------------------------------------------

    current.to_parquet(
        output_path,
        index=False,
    )

    # -----------------------------------------------------------------
    # Ingestion manifest
    # -----------------------------------------------------------------

    pd.DataFrame(
        manifest_rows
    ).to_csv(
        audit_dir
        / "ingestion_manifest.csv",
        index=False,
    )

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

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
            current[
                "CN Root ID"
            ]
            .replace(
                "",
                pd.NA,
            )
            .nunique()
        ),
        "total_value": float(
            current[
                "Value"
            ].sum()
        ),
        "canonical_supplier_groups": int(
            current[
                "supplier_group"
            ].nunique()
        ),
        "supplier_mapped_rows": (
            mapped_rows
        ),
        "supplier_abn_grouped_rows": (
            abn_grouped_rows
        ),
        "supplier_review_rows": (
            review_rows
        ),
        "accenture_rows": int(
            current[
                "is_accenture"
            ].sum()
        ),
        "accenture_value": float(
            current.loc[
                current[
                    "is_accenture"
                ],
                "Value",
            ].sum()
        ),
        "division_populated_rows": int(
            current[
                "Agency Division"
            ]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
            .sum()
        ),
        "branch_populated_rows": int(
            current[
                "Agency Branch"
            ]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
            .sum()
        ),
        "category_type_populated_rows": int(
            current[
                "Category Type"
            ]
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

    # -----------------------------------------------------------------
    # Console summary
    # -----------------------------------------------------------------

    print()
    print(
        "============================================"
    )

    print(
        "ATLAS Financial Year Analysis combine complete"
    )

    print(
        "============================================"
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
        f"Supplier groups: "
        f"{summary['canonical_supplier_groups']:,}"
    )

    print(
        f"Accenture rows: "
        f"{summary['accenture_rows']:,}"
    )

    print(
        "Accenture value: "
        f"${summary['accenture_value']:,.2f}"
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

    print()
    print(
        "Supplier audits:"
    )

    print(
        f"  {audit_dir / 'supplier_identity_mapping.csv'}"
    )

    print(
        f"  {audit_dir / 'supplier_group_variants.csv'}"
    )

    print(
        f"  {audit_dir / 'supplier_review_queue.csv'}"
    )

    print(
        f"  {audit_dir / 'supplier_abn_name_variants.csv'}"
    )

    print()
    print(
        f"Wrote: {output_path}"
    )


if __name__ == "__main__":
    main()
