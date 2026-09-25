from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


# =============================================================================
# ATLAS SHARED AUSTENDER INGESTION PIPELINE
# =============================================================================
#
# Purpose
# -------
# Combine native AusTender "Contract Notice Published" Excel exports into one
# shared ATLAS parquet that can feed Defence, Health and future market modules.
#
# The script:
#   1. reads every .xlsx file under the historical and monthly raw folders;
#   2. detects the AusTender table header automatically;
#   3. preserves all raw source columns;
#   4. standardises core data types;
#   5. derives shared fields such as Financial Year and Value Per Year;
#   6. adds a conservative canonical supplier_group while preserving raw supplier
#      name and ABN;
#   7. audits duplicate CN IDs and keeps the latest published version;
#   8. writes one combined parquet plus QA/audit outputs.
#
# IMPORTANT
# ---------
# This script does NOT apply Defence- or Health-specific market logic.
# It also does NOT invent fields that the native public export does not contain
# (for example Agency Division, Agency Branch, Category Type or Category Code).
# Downstream market builders must only rely on fields genuinely available in
# this shared source, or enrich them from a separate authoritative source.
# =============================================================================


EXPECTED_CORE_COLUMNS = [
    "Agency",
    "CN ID",
    "SON ID",
    "Supplier Name",
    "Supplier ABN",
    "Description",
    "Category",
    "Agency Ref. ID",
    "Publish Date",
    "Start Date",
    "End Date",
    "Value (AUD)",
]

DATE_COLUMNS = ["Publish Date", "Start Date", "End Date"]

# Reviewed supplier-family aliases. These are deliberately conservative.
# Unknown suppliers retain their cleaned source identity and are never silently
# fuzzy-merged.
SUPPLIER_FAMILY_PATTERNS: list[tuple[str, list[str]]] = [
    ("Accenture", [r"\bACCENTURE\b"]),
    ("Deloitte", [r"\bDELOITTE\b"]),
    ("EY", [
        r"\bERNST\s*(?:&|AND)\s*YOUNG\b",
        r"^EY(?:\s+(?:AUSTRALIA|OCEANIA|PTY|LIMITED|LTD|LLP))?\b",
    ]),
    ("KPMG", [r"\bKPMG\b"]),
    ("PwC", [
        r"\bPRICEWATERHOUSECOOPERS\b",
        r"\bPRICE\s+WATERHOUSE\s+COOPERS\b",
        r"\bPWC\b",
    ]),
    ("Lockheed Martin", [r"\bLOCKHEED\s+MARTIN\b"]),
    ("Leidos", [r"\bLEIDOS\b"]),
    ("Fujitsu", [r"\bFUJITSU\b"]),
    ("Thales", [r"\bTHALES\b"]),
    ("IBM", [r"\bIBM\b", r"\bINTERNATIONAL\s+BUSINESS\s+MACHINES\b"]),
    ("BAE Systems", [r"\bBAE\s+SYSTEMS\b"]),
    ("Boeing", [r"\bBOEING\b"]),
    ("Northrop Grumman", [r"\bNORTHROP\s+GRUMMAN\b"]),
    ("Raytheon", [r"\bRAYTHEON\b"]),
    ("Rheinmetall", [r"\bRHEINMETALL\b"]),
    ("Saab", [r"\bSAAB\b"]),
    ("Airbus", [r"\bAIRBUS\b", r"\bEADS\b.*\bCASA\b"]),
    ("DXC Technology", [r"\bDXC\s+TECHNOLOGY\b", r"^DXC\b"]),
    ("Capgemini", [r"\bCAPGEMINI\b"]),
    ("Atos / Eviden", [r"\bATOS\b", r"\bEVIDEN\b"]),
    ("Microsoft", [r"\bMICROSOFT\b"]),
    ("Amazon Web Services", [
        r"\bAMAZON\s+WEB\s+SERVICES\b",
        r"^AWS(?:\s|$)",
    ]),
    ("Oracle", [r"\bORACLE\b"]),
    ("SAP", [r"^SAP(?:\s|$)", r"\bSAP\s+AUSTRALIA\b"]),
    ("Unisys", [r"\bUNISYS\b"]),
    ("Kinetic IT", [r"\bKINETIC\s+IT\b"]),
    ("Data#3", [r"\bDATA\s*#?\s*3\b"]),
    ("Telstra", [r"\bTELSTRA\b"]),
    ("Optus", [r"\bOPTUS\b"]),
    ("ASC", [r"^ASC\s+(?:PTY\s+LTD|SHIPBUILDING\s+PTY\s+LTD)$"]),
    ("Lendlease", [r"\bLEND\s*LEASE\b", r"\bLENDLEASE\b"]),
    ("Jacobs", [r"\bJACOBS\b"]),
    ("Aurecon", [r"\bAURECON\b", r"\bAUGILITY\b"]),
    ("Downer", [r"\bDOWNER\b"]),
    ("Ventia", [r"\bVENTIA\b"]),
    ("Nova Systems", [r"\bNOVA\s+SYSTEMS\b"]),
    ("Qinetiq", [r"\bQINETIQ\b"]),
    ("CAE", [r"\bCAE\s+AUSTRALIA\b"]),
    ("Elbit Systems", [r"\bELBIT\s+SYSTEMS\b"]),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine native AusTender Excel exports into the shared ATLAS parquet."
    )
    parser.add_argument(
        "--historical-dir",
        default="data/austender/raw/historical",
        help="Folder containing annual historical AusTender .xlsx exports.",
    )
    parser.add_argument(
        "--monthly-dir",
        default="data/austender/raw/monthly",
        help="Folder containing monthly AusTender .xlsx exports.",
    )
    parser.add_argument(
        "--output",
        default="data/austender/combined/austender_combined.parquet",
        help="Combined parquet output path.",
    )
    parser.add_argument(
        "--audit-dir",
        default="audits/austender",
        help="Folder for ingestion, duplicate and supplier QA outputs.",
    )
    return parser.parse_args()


def normalise_abn(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    digits = re.sub(r"\D", "", str(value))
    # Excel can occasionally materialise an ABN-like number with .0.
    if digits.endswith("0") and str(value).strip().endswith(".0"):
        digits = digits[:-1]
    return digits.zfill(11) if 1 <= len(digits) <= 11 else digits


def normalise_supplier_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).upper().replace("&", " AND ")
    text = re.sub(r"[^A-Z0-9+#]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def cleaned_supplier_fallback(value: object) -> str:
    """Readable fallback identity for suppliers outside reviewed families."""
    raw = "" if value is None or pd.isna(value) else str(value).strip()
    if not raw:
        return "Unknown Supplier"

    cleaned = re.sub(r"\s+", " ", raw).strip()
    # Preserve source spelling/case where possible. This is a display identity,
    # not a fuzzy merge.
    return cleaned


def reviewed_supplier_family(value: object) -> str:
    text = normalise_supplier_text(value)
    if not text:
        return ""
    for canonical, patterns in SUPPLIER_FAMILY_PATTERNS:
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns):
            return canonical
    return ""


def financial_year_label(value: object) -> str:
    """Australian FY label based on a date, e.g. 2025-07-01 -> 2025-2026."""
    if value is None or pd.isna(value):
        return ""
    dt = pd.Timestamp(value)
    start_year = dt.year if dt.month >= 7 else dt.year - 1
    return f"{start_year}-{start_year + 1}"


def calculate_value_per_year(value: object, start: object, end: object) -> float:
    """Annualised contract value using actual contract duration.

    Uses a minimum duration of one year so short contracts are not artificially
    inflated above their full contract value.
    """
    try:
        val = float(value)
    except (TypeError, ValueError):
        return 0.0

    if pd.isna(start) or pd.isna(end):
        return val

    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    days = (end_dt - start_dt).days + 1
    if days <= 0:
        return val

    years = max(days / 365.25, 1.0)
    return val / years


def find_header_row(path: Path, max_rows: int = 40) -> int:
    """Find the zero-based Excel row containing the AusTender contract headers."""
    preview = pd.read_excel(path, header=None, nrows=max_rows, dtype=object)
    for idx, row in preview.iterrows():
        values = {
            str(v).strip()
            for v in row.tolist()
            if v is not None and not (isinstance(v, float) and pd.isna(v))
        }
        if {"Agency", "CN ID", "Supplier Name", "Publish Date"}.issubset(values):
            return int(idx)
    raise RuntimeError(f"Could not locate AusTender table header in {path}")


def read_criteria(path: Path) -> dict[str, str]:
    """Read useful metadata from the criteria block above the table."""
    preview = pd.read_excel(path, header=None, nrows=17, usecols="A:B", dtype=object)
    result: dict[str, str] = {}
    for _, row in preview.iterrows():
        key = "" if pd.isna(row.iloc[0]) else str(row.iloc[0]).strip()
        value = "" if len(row) < 2 or pd.isna(row.iloc[1]) else str(row.iloc[1]).strip()
        if key:
            result[key] = value
    return result


def read_austender_export(path: Path) -> pd.DataFrame:
    header_row = find_header_row(path)
    criteria = read_criteria(path)

    df = pd.read_excel(
        path,
        header=header_row,
        dtype={
            "CN ID": str,
            "SON ID": str,
            "Supplier Name": str,
            "Supplier ABN": str,
            "Agency Ref. ID": str,
        },
    )

    # Drop completely empty columns/rows that Excel sometimes carries.
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all").copy()

    missing = sorted(set(EXPECTED_CORE_COLUMNS) - set(df.columns))
    if missing:
        raise RuntimeError(
            f"{path.name} is missing expected AusTender columns: {', '.join(missing)}"
        )

    # Preserve native source values but clean obvious string whitespace.
    for col in ["Agency", "CN ID", "SON ID", "Supplier Name", "Supplier ABN",
                "Description", "Category", "Agency Ref. ID"]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    for col in DATE_COLUMNS:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df["Value (AUD)"] = pd.to_numeric(df["Value (AUD)"], errors="coerce").fillna(0.0)

    # Shared canonical aliases expected by downstream ATLAS processing.
    # The raw "Value (AUD)" column remains untouched.
    df["Value"] = df["Value (AUD)"]
    df["Financial Year"] = df["Publish Date"].map(financial_year_label)
    df["Value Per Year"] = [
        calculate_value_per_year(v, s, e)
        for v, s, e in zip(df["Value"], df["Start Date"], df["End Date"])
    ]

    df["supplier_abn_normalized"] = df["Supplier ABN"].map(normalise_abn)
    df["supplier_name_normalized"] = df["Supplier Name"].map(normalise_supplier_text)

    family = df["Supplier Name"].map(reviewed_supplier_family)
    df["supplier_group"] = [
        fam if fam else cleaned_supplier_fallback(raw)
        for fam, raw in zip(family, df["Supplier Name"])
    ]
    df["supplier_mapping_method"] = family.map(
        lambda x: "REVIEWED_FAMILY_ALIAS" if x else "RAW_SUPPLIER_NAME"
    )
    df["supplier_display"] = df["supplier_group"]
    df["supplier_id"] = df["supplier_group"].map(
        lambda x: re.sub(r"[^A-Z0-9]+", "_", str(x).upper()).strip("_") or "UNKNOWN"
    )
    df["is_accenture"] = df["supplier_group"].eq("Accenture")

    # Provenance: never lose where a row came from.
    df["source_file"] = path.name
    df["source_path"] = str(path.as_posix())
    df["source_date_range"] = criteria.get("Date Range", "")
    df["source_date_type"] = criteria.get("Date Type", "")
    df["source_export_count"] = criteria.get("Count", "")
    df["source_export_value_aud"] = criteria.get("Value (AUD)", "")

    return df


def discover_files(paths: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for folder in paths:
        if not folder.exists():
            continue
        files.extend(sorted(folder.rglob("*.xlsx")))
    return sorted(set(files), key=lambda p: str(p).lower())


def build_supplier_audits(df: pd.DataFrame, audit_dir: Path) -> None:
    supplier_cols = [
        "Supplier Name",
        "Supplier ABN",
        "supplier_name_normalized",
        "supplier_abn_normalized",
        "supplier_group",
        "supplier_mapping_method",
        "Value",
    ]

    mapping = (
        df[supplier_cols]
        .groupby(
            [
                "Supplier Name",
                "Supplier ABN",
                "supplier_name_normalized",
                "supplier_abn_normalized",
                "supplier_group",
                "supplier_mapping_method",
            ],
            dropna=False,
        )
        .agg(rows=("Value", "size"), total_value=("Value", "sum"))
        .reset_index()
        .sort_values("total_value", ascending=False)
    )
    mapping.to_csv(audit_dir / "supplier_mapping_audit.csv", index=False)

    # Same populated ABN presented under multiple raw supplier names.
    populated = df[df["supplier_abn_normalized"].astype(str).str.len().gt(0)].copy()
    if not populated.empty:
        abn_summary = (
            populated.groupby("supplier_abn_normalized", dropna=False)
            .agg(
                raw_name_count=("Supplier Name", "nunique"),
                canonical_group_count=("supplier_group", "nunique"),
                raw_names=("Supplier Name", lambda s: " | ".join(sorted(set(map(str, s))))),
                canonical_groups=("supplier_group", lambda s: " | ".join(sorted(set(map(str, s))))),
                rows=("CN ID", "size"),
                total_value=("Value", "sum"),
            )
            .reset_index()
        )
        abn_summary[abn_summary["raw_name_count"] > 1].sort_values(
            "total_value", ascending=False
        ).to_csv(audit_dir / "supplier_abn_name_variants.csv", index=False)

        abn_summary[abn_summary["canonical_group_count"] > 1].sort_values(
            "total_value", ascending=False
        ).to_csv(audit_dir / "supplier_abn_conflicts.csv", index=False)

    # Raw names that have not been mapped to a reviewed family. These are not
    # necessarily errors; this is the queue for future supplier curation.
    unknown = (
        df[df["supplier_mapping_method"].eq("RAW_SUPPLIER_NAME")]
        .groupby(["Supplier Name", "Supplier ABN", "supplier_group"], dropna=False)
        .agg(rows=("CN ID", "size"), total_value=("Value", "sum"))
        .reset_index()
        .sort_values("total_value", ascending=False)
    )
    unknown.to_csv(audit_dir / "supplier_unreviewed_queue.csv", index=False)


def resolve_duplicate_cn_ids(df: pd.DataFrame, audit_dir: Path) -> pd.DataFrame:
    """Resolve exact/revised duplicate CN IDs conservatively.

    If a CN ID appears more than once, all occurrences are written to an audit.
    Production keeps the row with the latest Publish Date; ties are broken by the
    later source file name and row order. This makes monthly refreshes deterministic
    while retaining evidence of every removed occurrence.
    """
    out = df.copy()
    out["_ingest_order"] = range(len(out))

    populated = out["CN ID"].fillna("").astype(str).str.strip().ne("")
    duplicate_mask = populated & out.duplicated("CN ID", keep=False)

    duplicates = out.loc[duplicate_mask].copy()
    duplicates.sort_values(
        ["CN ID", "Publish Date", "source_file", "_ingest_order"],
        inplace=True,
    )
    duplicates.to_csv(audit_dir / "duplicate_cn_occurrences.csv", index=False)

    if duplicate_mask.any():
        keep_order = (
            out.loc[populated]
            .sort_values(
                ["CN ID", "Publish Date", "source_file", "_ingest_order"],
                ascending=[True, True, True, True],
                na_position="first",
            )
            .drop_duplicates("CN ID", keep="last")
            .index
        )
        blank_cn = out.index[~populated]
        out = out.loc[list(keep_order) + list(blank_cn)].copy()

    out.sort_values(["Publish Date", "CN ID", "_ingest_order"], inplace=True, na_position="last")
    out.drop(columns=["_ingest_order"], inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


def main() -> None:
    args = parse_args()

    historical_dir = Path(args.historical_dir)
    monthly_dir = Path(args.monthly_dir)
    output_path = Path(args.output)
    audit_dir = Path(args.audit_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    files = discover_files([historical_dir, monthly_dir])
    if not files:
        raise SystemExit(
            "No .xlsx files found. Expected files under "
            f"{historical_dir} and/or {monthly_dir}."
        )

    print(f"Found {len(files):,} AusTender export file(s).")

    frames: list[pd.DataFrame] = []
    ingestion_rows: list[dict[str, object]] = []

    for idx, path in enumerate(files, start=1):
        print(f"[{idx}/{len(files)}] Reading {path}")
        frame = read_austender_export(path)
        frames.append(frame)

        ingestion_rows.append({
            "source_file": path.name,
            "source_path": str(path.as_posix()),
            "rows_loaded": int(len(frame)),
            "value_loaded": float(frame["Value"].sum()),
            "publish_date_min": frame["Publish Date"].min(),
            "publish_date_max": frame["Publish Date"].max(),
            "unique_cn_ids": int(frame["CN ID"].replace("", pd.NA).nunique()),
        })

    combined = pd.concat(frames, ignore_index=True, sort=False)

    rows_before = len(combined)
    value_before = float(combined["Value"].sum())

    build_supplier_audits(combined, audit_dir)
    combined = resolve_duplicate_cn_ids(combined, audit_dir)

    rows_after = len(combined)
    value_after = float(combined["Value"].sum())

    # Core QA checks.
    if combined["CN ID"].fillna("").astype(str).str.strip().ne("").any():
        remaining_dupes = combined.loc[
            combined["CN ID"].fillna("").astype(str).str.strip().ne(""),
            "CN ID",
        ].duplicated().sum()
        if remaining_dupes:
            raise RuntimeError(
                f"Duplicate CN ID resolution failed: {remaining_dupes:,} duplicate(s) remain."
            )

    combined.to_parquet(output_path, index=False)

    ingestion = pd.DataFrame(ingestion_rows)
    ingestion.to_csv(audit_dir / "ingestion_manifest.csv", index=False)

    summary = {
        "files_loaded": len(files),
        "historical_directory": str(historical_dir),
        "monthly_directory": str(monthly_dir),
        "rows_before_cn_deduplication": rows_before,
        "rows_after_cn_deduplication": rows_after,
        "duplicate_rows_removed": rows_before - rows_after,
        "value_before_cn_deduplication": value_before,
        "value_after_cn_deduplication": value_after,
        "publish_date_min": str(combined["Publish Date"].min()),
        "publish_date_max": str(combined["Publish Date"].max()),
        "financial_years": sorted(
            x for x in combined["Financial Year"].dropna().astype(str).unique().tolist() if x
        ),
        "unique_suppliers_raw": int(combined["Supplier Name"].nunique(dropna=True)),
        "unique_supplier_groups": int(combined["supplier_group"].nunique(dropna=True)),
        "output": str(output_path),
        "columns": list(combined.columns),
    }
    (audit_dir / "combine_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )

    print()
    print("ATLAS AusTender combine complete.")
    print(f"Files loaded: {len(files):,}")
    print(f"Rows before CN dedupe: {rows_before:,}")
    print(f"Rows after CN dedupe:  {rows_after:,}")
    print(f"Duplicate occurrences removed: {rows_before - rows_after:,}")
    print(f"Combined value after dedupe: ${value_after:,.2f}")
    print(f"Wrote: {output_path}")
    print(f"Audit directory: {audit_dir}")


if __name__ == "__main__":
    main()
