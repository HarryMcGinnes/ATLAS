from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate the canonical ATLAS AusTender dataset.")
    p.add_argument(
        "--input",
        default="data/austender/combined/austender_combined.parquet",
    )
    p.add_argument(
        "--output",
        default="audits/austender/validation_summary.json",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.input)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(path)

    required = {
        "Agency", "CN ID", "CN Root ID", "Supplier Name", "supplier_group",
        "Publish Date", "Start Date", "End Date", "Value", "Financial Year",
        "source_kind", "source_file",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError("Combined dataset missing required fields: " + ", ".join(missing))

    cn = df["CN Root ID"].fillna("").astype(str).str.strip()
    duplicate_roots = int(cn[cn.ne("")].duplicated().sum())
    if duplicate_roots:
        raise RuntimeError(f"{duplicate_roots:,} duplicate CN Root IDs remain after reconciliation.")

    values = pd.to_numeric(df["Value"], errors="coerce").fillna(0)
    if (values < 0).any():
        raise RuntimeError("Negative contract values detected.")

    if len(df) < 100_000:
        raise RuntimeError(f"Unexpectedly small combined dataset: {len(df):,} rows.")

    summary = {
        "rows": int(len(df)),
        "unique_cn_roots": int(cn.replace("", pd.NA).nunique()),
        "total_value": float(values.sum()),
        "publish_date_min": str(pd.to_datetime(df["Publish Date"], errors="coerce").min()),
        "publish_date_max": str(pd.to_datetime(df["Publish Date"], errors="coerce").max()),
        "supplier_groups": int(df["supplier_group"].nunique(dropna=True)),
        "source_kind_counts": df["source_kind"].value_counts(dropna=False).to_dict(),
        "validation": "PASS",
    }
    output.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("ATLAS combined dataset validation: PASS")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
