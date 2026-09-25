from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFENCE_AGENCIES = {
    "department of defence",
    "australian signals directorate",
    "australian submarine agency",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter the shared ATLAS AusTender dataset to the Defence market."
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
        default="audits/defence",
    )
    return parser.parse_args()


def normalise_agency(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).strip().lower().split())


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    audit_dir = Path(args.audit_dir)

    if not input_path.exists():
        raise SystemExit(f"Shared AusTender parquet not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    if "Agency" not in df.columns:
        raise RuntimeError("Shared AusTender dataset is missing required column: Agency")

    agency_key = df["Agency"].map(normalise_agency)
    mask = agency_key.isin(DEFENCE_AGENCIES)
    defence = df.loc[mask].copy()

    if defence.empty:
        raise RuntimeError("Defence filter returned zero rows.")

    # Shared supplier_group must already exist. Do not recalculate it here.
    if "supplier_group" not in defence.columns:
        raise RuntimeError(
            "Shared ATLAS dataset is missing supplier_group. "
            "Run pipeline/combine_austender.py first."
        )

    defence["is_defence_scope"] = True
    defence.to_parquet(output_path, index=False)

    agency_summary = (
        defence.groupby("Agency", dropna=False)
        .agg(
            rows=("CN ID", "size"),
            contracts=("CN ID", "nunique"),
            value=("Value", "sum"),
        )
        .reset_index()
        .sort_values("value", ascending=False)
    )
    agency_summary.to_csv(audit_dir / "defence_agency_summary.csv", index=False)

    summary = {
        "input_rows": int(len(df)),
        "defence_rows": int(len(defence)),
        "defence_contracts": int(defence["CN ID"].nunique()),
        "defence_value": float(pd.to_numeric(defence["Value"], errors="coerce").fillna(0).sum()),
        "agencies": sorted(defence["Agency"].dropna().astype(str).unique().tolist()),
        "output": str(output_path),
    }
    (audit_dir / "defence_filter_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )

    print("ATLAS Defence filter complete.")
    print(f"Input rows:   {len(df):,}")
    print(f"Defence rows: {len(defence):,}")
    print(f"Defence value: ${summary['defence_value']:,.2f}")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
