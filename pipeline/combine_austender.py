from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


EXPECTED_EXCEL_COLUMNS = [
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
    ("Amazon Web Services", [r"\bAMAZON\s+WEB\s+SERVICES\b", r"^AWS(?:\s|$)"]),
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


SOURCE_PRIORITY = {
    # Website exports are preferred when they overlap with the API because our
    # September reconciliation showed the website export can contain CNs that are
    # absent from the OCDS published-date feed.
    "historical_excel": 30,
    "monthly_excel": 50,
    "api_published": 40,
    # Modified stream represents a later state of an already known CN and wins
    # when its record timestamp is newer.
    "api_modified": 60,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build the canonical shared ATLAS AusTender dataset."
    )
    p.add_argument("--historical-dir", default="data/austender/raw/historical")
    p.add_argument("--monthly-dir", default="data/austender/raw/monthly")
    p.add_argument("--api-dir", default="data/austender/raw/api_monthly")
    p.add_argument(
        "--output",
        default="data/austender/combined/austender_combined.parquet",
    )
    p.add_argument("--audit-dir", default="audits/austender")
    return p.parse_args()


def normalise_abn(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    raw = str(value).strip()
    if raw.lower() in {"", "nan", "none", "abn exempt"}:
        return ""
    digits = re.sub(r"\D", "", raw)
    return digits if len(digits) == 11 else ""


def normalise_supplier_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).upper().replace("&", " AND ")
    text = re.sub(r"[^A-Z0-9+#]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def reviewed_supplier_family(value: object) -> str:
    text = normalise_supplier_text(value)
    for canonical, patterns in SUPPLIER_FAMILY_PATTERNS:
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns):
            return canonical
    return ""


def readable_supplier(value: object) -> str:
    raw = "" if value is None or pd.isna(value) else str(value).strip()
    return re.sub(r"\s+", " ", raw).strip() or "Unknown Supplier"


def financial_year_label(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    dt = pd.Timestamp(value)
    start_year = dt.year if dt.month >= 7 else dt.year - 1
    return f"{start_year}-{start_year + 1}"


def calculate_value_per_year(value: object, start: object, end: object) -> float:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return 0.0
    if pd.isna(start) or pd.isna(end):
        return val
    days = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
    if days <= 0:
        return val
    years = max(days / 365.25, 1.0)
    return val / years


def canonical_cn_root(value: object) -> str:
    """Treat explicit AusTender amendment suffixes as the same contract family."""
    text = "" if value is None or pd.isna(value) else str(value).strip().upper()
    if not text:
        return ""
    return re.sub(r"-A\d+$", "", text)


def amendment_number(value: object) -> int:
    text = "" if value is None or pd.isna(value) else str(value).strip().upper()
    m = re.search(r"-A(\d+)$", text)
    return int(m.group(1)) if m else 0


def find_header_row(path: Path, max_rows: int = 40) -> int:
    preview = pd.read_excel(path, header=None, nrows=max_rows, dtype=object)
    for idx, row in preview.iterrows():
        vals = {
            str(v).strip()
            for v in row.tolist()
            if v is not None and not (isinstance(v, float) and pd.isna(v))
        }
        if {"Agency", "CN ID", "Supplier Name", "Publish Date"}.issubset(vals):
            return int(idx)
    raise RuntimeError(f"Could not find AusTender table header in {path}")


def read_criteria(path: Path) -> dict[str, str]:
    preview = pd.read_excel(path, header=None, nrows=17, usecols="A:B", dtype=object)
    result: dict[str, str] = {}
    for _, row in preview.iterrows():
        key = "" if pd.isna(row.iloc[0]) else str(row.iloc[0]).strip()
        value = "" if pd.isna(row.iloc[1]) else str(row.iloc[1]).strip()
        if key:
            result[key] = value
    return result


def add_shared_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in DATE_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NaT
        out[col] = pd.to_datetime(out[col], errors="coerce", utc=True)

    if "Value (AUD)" not in out.columns:
        out["Value (AUD)"] = 0.0
    out["Value (AUD)"] = pd.to_numeric(out["Value (AUD)"], errors="coerce").fillna(0.0)

    out["Value"] = out["Value (AUD)"]
    out["Financial Year"] = out["Publish Date"].map(financial_year_label)
    out["Value Per Year"] = [
        calculate_value_per_year(v, s, e)
        for v, s, e in zip(out["Value"], out["Start Date"], out["End Date"])
    ]

    for col in [
        "Agency", "CN ID", "SON ID", "Supplier Name", "Supplier ABN",
        "Description", "Category", "Agency Ref. ID",
    ]:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].fillna("").astype(str).str.strip()

    if "Category Code" not in out.columns:
        out["Category Code"] = ""

    out["CN Root ID"] = out["CN ID"].map(canonical_cn_root)
    out["CN Amendment Number"] = out["CN ID"].map(amendment_number)

    out["supplier_abn_normalized"] = out["Supplier ABN"].map(normalise_abn)
    out["supplier_name_normalized"] = out["Supplier Name"].map(normalise_supplier_text)
    family = out["Supplier Name"].map(reviewed_supplier_family)
    out["supplier_group"] = [
        fam if fam else readable_supplier(raw)
        for fam, raw in zip(family, out["Supplier Name"])
    ]
    out["supplier_mapping_method"] = family.map(
        lambda x: "REVIEWED_FAMILY_ALIAS" if x else "RAW_SUPPLIER_NAME"
    )
    out["supplier_display"] = out["supplier_group"]
    out["supplier_id"] = out["supplier_group"].map(
        lambda x: re.sub(r"[^A-Z0-9]+", "_", str(x).upper()).strip("_") or "UNKNOWN"
    )
    out["is_accenture"] = out["supplier_group"].eq("Accenture")
    return out


def read_excel_export(path: Path, source_kind: str) -> pd.DataFrame:
    header_row = find_header_row(path)
    criteria = read_criteria(path)
    df = pd.read_excel(path, header=header_row, dtype=object)
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all").copy()

    missing = sorted(set(EXPECTED_EXCEL_COLUMNS) - set(df.columns))
    if missing:
        raise RuntimeError(f"{path.name} missing expected columns: {', '.join(missing)}")

    df["source_kind"] = source_kind
    df["source_file"] = path.name
    df["source_period"] = criteria.get("Date Range", "")
    df["source_record_timestamp"] = pd.to_datetime(
        df.get("Publish Date"), errors="coerce", utc=True
    )
    return add_shared_fields(df)


def read_api_parquet(path: Path, source_kind: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["source_kind"] = source_kind
    df["source_file"] = path.name
    df["source_period"] = re.search(r"(\d{4}-\d{2})", path.name).group(1) if re.search(r"(\d{4}-\d{2})", path.name) else ""
    if "api_record_timestamp" in df.columns:
        df["source_record_timestamp"] = pd.to_datetime(
            df["api_record_timestamp"], errors="coerce", utc=True
        )
    else:
        df["source_record_timestamp"] = pd.to_datetime(
            df.get("Publish Date"), errors="coerce", utc=True
        )
    return add_shared_fields(df)


def discover_xlsx(folder: Path) -> list[Path]:
    return sorted(folder.rglob("*.xlsx")) if folder.exists() else []


def discover_api_files(folder: Path) -> list[tuple[Path, str]]:
    if not folder.exists():
        return []
    files: list[tuple[Path, str]] = []
    for path in sorted(folder.glob("austender_*_published.parquet")):
        files.append((path, "api_published"))
    for path in sorted(folder.glob("austender_*_modified.parquet")):
        files.append((path, "api_modified"))
    return files


def reconcile_contracts(df: pd.DataFrame, audit_dir: Path) -> pd.DataFrame:
    """Return one current row per CN Root ID.

    Order of precedence:
      1. latest source record timestamp;
      2. highest amendment suffix (-A2 beats -A1);
      3. source priority (modified API > website monthly > API published > historical);
      4. later ingest order.

    Blank CN IDs are preserved because silently discarding source rows is worse than
    retaining an auditable unkeyed record.
    """
    out = df.copy()
    out["_ingest_order"] = range(len(out))
    out["_source_priority"] = out["source_kind"].map(SOURCE_PRIORITY).fillna(0).astype(int)
    out["_record_ts"] = pd.to_datetime(
        out["source_record_timestamp"], errors="coerce", utc=True
    )

    populated = out["CN Root ID"].fillna("").astype(str).str.strip().ne("")
    dupes = populated & out.duplicated("CN Root ID", keep=False)

    if dupes.any():
        out.loc[dupes].sort_values(
            ["CN Root ID", "_record_ts", "CN Amendment Number", "_source_priority", "_ingest_order"]
        ).to_csv(audit_dir / "cn_reconciliation_all_versions.csv", index=False)

    keyed = out.loc[populated].sort_values(
        ["CN Root ID", "_record_ts", "CN Amendment Number", "_source_priority", "_ingest_order"],
        ascending=[True, True, True, True, True],
        na_position="first",
    )
    winners = keyed.drop_duplicates("CN Root ID", keep="last")

    if dupes.any():
        winner_index = set(winners.index)
        superseded = out.loc[dupes & ~out.index.isin(winner_index)].copy()
        superseded.to_csv(audit_dir / "cn_reconciliation_superseded.csv", index=False)

    blanks = out.loc[~populated]
    result = pd.concat([winners, blanks], ignore_index=True)

    result.drop(
        columns=["_ingest_order", "_source_priority", "_record_ts"],
        errors="ignore",
        inplace=True,
    )
    result.sort_values(["Publish Date", "CN Root ID"], inplace=True, na_position="last")
    result.reset_index(drop=True, inplace=True)

    remaining = result.loc[
        result["CN Root ID"].fillna("").astype(str).str.strip().ne(""),
        "CN Root ID",
    ].duplicated().sum()
    if remaining:
        raise RuntimeError(f"CN reconciliation failed: {remaining} duplicate root IDs remain")

    return result


def export_supplier_audits(df: pd.DataFrame, audit_dir: Path) -> None:
    mapping = (
        df.groupby(
            [
                "Supplier Name",
                "Supplier ABN",
                "supplier_abn_normalized",
                "supplier_group",
                "supplier_mapping_method",
            ],
            dropna=False,
        )
        .agg(rows=("CN ID", "size"), total_value=("Value", "sum"))
        .reset_index()
        .sort_values("total_value", ascending=False)
    )
    mapping.to_csv(audit_dir / "supplier_mapping_audit.csv", index=False)

    populated = df[df["supplier_abn_normalized"].astype(str).str.len().gt(0)].copy()
    if not populated.empty:
        by_abn = (
            populated.groupby("supplier_abn_normalized", dropna=False)
            .agg(
                raw_name_count=("Supplier Name", "nunique"),
                supplier_group_count=("supplier_group", "nunique"),
                raw_names=("Supplier Name", lambda s: " | ".join(sorted(set(map(str, s))))),
                supplier_groups=("supplier_group", lambda s: " | ".join(sorted(set(map(str, s))))),
                rows=("CN ID", "size"),
                total_value=("Value", "sum"),
            )
            .reset_index()
        )
        by_abn[by_abn["raw_name_count"] > 1].sort_values(
            "total_value", ascending=False
        ).to_csv(audit_dir / "supplier_abn_name_variants.csv", index=False)
        by_abn[by_abn["supplier_group_count"] > 1].sort_values(
            "total_value", ascending=False
        ).to_csv(audit_dir / "supplier_abn_conflicts.csv", index=False)

    (
        df[df["supplier_mapping_method"].eq("RAW_SUPPLIER_NAME")]
        .groupby(["Supplier Name", "Supplier ABN", "supplier_group"], dropna=False)
        .agg(rows=("CN ID", "size"), total_value=("Value", "sum"))
        .reset_index()
        .sort_values("total_value", ascending=False)
        .to_csv(audit_dir / "supplier_unreviewed_queue.csv", index=False)
    )


def main() -> None:
    args = parse_args()
    historical_dir = Path(args.historical_dir)
    monthly_dir = Path(args.monthly_dir)
    api_dir = Path(args.api_dir)
    output_path = Path(args.output)
    audit_dir = Path(args.audit_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    manifest: list[dict[str, object]] = []

    sources: list[tuple[Path, str]] = []
    sources += [(p, "historical_excel") for p in discover_xlsx(historical_dir)]
    sources += [(p, "monthly_excel") for p in discover_xlsx(monthly_dir)]
    sources += discover_api_files(api_dir)

    if not sources:
        raise SystemExit("No AusTender source files found.")

    for i, (path, kind) in enumerate(sources, start=1):
        print(f"[{i}/{len(sources)}] {kind}: {path}")
        if path.suffix.lower() == ".xlsx":
            frame = read_excel_export(path, kind)
        else:
            frame = read_api_parquet(path, kind)

        frames.append(frame)
        manifest.append({
            "source_file": path.name,
            "source_kind": kind,
            "rows_loaded": int(len(frame)),
            "value_loaded": float(frame["Value"].sum()),
            "publish_date_min": frame["Publish Date"].min(),
            "publish_date_max": frame["Publish Date"].max(),
        })

    combined_all = pd.concat(frames, ignore_index=True, sort=False)
    rows_before = len(combined_all)
    value_before = float(combined_all["Value"].sum())

    combined = reconcile_contracts(combined_all, audit_dir)
    export_supplier_audits(combined, audit_dir)

    combined.to_parquet(output_path, index=False)
    pd.DataFrame(manifest).to_csv(audit_dir / "ingestion_manifest.csv", index=False)

    summary = {
        "sources_loaded": len(sources),
        "rows_before_reconciliation": rows_before,
        "rows_after_reconciliation": int(len(combined)),
        "rows_superseded": rows_before - int(len(combined)),
        "value_before_reconciliation": value_before,
        "value_after_reconciliation": float(combined["Value"].sum()),
        "unique_cn_roots": int(combined["CN Root ID"].replace("", pd.NA).nunique()),
        "publish_date_min": str(combined["Publish Date"].min()),
        "publish_date_max": str(combined["Publish Date"].max()),
        "unique_supplier_groups": int(combined["supplier_group"].nunique(dropna=True)),
        "output": str(output_path),
        "columns": list(combined.columns),
    }
    (audit_dir / "combine_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )

    print()
    print("ATLAS canonical AusTender dataset built.")
    print(f"Source rows:      {rows_before:,}")
    print(f"Current rows:     {len(combined):,}")
    print(f"Superseded rows:  {rows_before - len(combined):,}")
    print(f"Current value:    ${combined['Value'].sum():,.2f}")
    print(f"Wrote:            {output_path}")


if __name__ == "__main__":
    main()
