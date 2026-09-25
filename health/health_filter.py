from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter the shared ATLAS AusTender dataset to the Health market."
    )
    parser.add_argument(
        "--input",
        default="data/austender/combined/austender_combined.parquet",
    )
    parser.add_argument(
        "--agency-config",
        default="health/config/health_agencies.txt",
        help="One exact AusTender Agency name per line. Blank lines and # comments are ignored.",
    )
    parser.add_argument(
        "--output",
        default="health/data/health_contracts_raw.parquet",
    )
    parser.add_argument(
        "--audit-dir",
        default="audits/health",
    )
    return parser.parse_args()


def normalise_agency(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).strip().lower().split())


def load_agency_config(path: Path) -> tuple[list[str], set[str]]:
    if not path.exists():
        raise SystemExit(
            f"Health agency config not found: {path}\n"
            "Create the file with one exact AusTender Agency name per line."
        )

    raw_names = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        raw_names.append(line)

    if not raw_names:
        raise RuntimeError(f"Health agency config is empty: {path}")

    return raw_names, {normalise_agency(x) for x in raw_names}


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    config_path = Path(args.agency_config)
    output_path = Path(args.output)
    audit_dir = Path(args.audit_dir)

    if not input_path.exists():
        raise SystemExit(f"Shared AusTender parquet not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    configured_names, configured_keys = load_agency_config(config_path)

    df = pd.read_parquet(input_path)
    if "Agency" not in df.columns:
        raise RuntimeError("Shared AusTender dataset is missing required column: Agency")

    available = {
        normalise_agency(x): str(x).strip()
        for x in df["Agency"].dropna().astype(str).unique()
    }

    missing_configured = [
        name for name in configured_names
        if normalise_agency(name) not in available
    ]

    agency_key = df["Agency"].map(normalise_agency)
    mask = agency_key.isin(configured_keys)
    health = df.loc[mask].copy()

    if health.empty:
        raise RuntimeError(
            "Health filter returned zero rows. Review health/config/health_agencies.txt."
        )

    if "supplier_group" not in health.columns:
        raise RuntimeError(
            "Shared ATLAS dataset is missing supplier_group. "
            "Run pipeline/combine_austender.py first."
        )

    health["is_health_scope"] = True
    health.to_parquet(output_path, index=False)

    agency_summary = (
        health.groupby("Agency", dropna=False)
        .agg(
            rows=("CN ID", "size"),
            contracts=("CN ID", "nunique"),
            value=("Value", "sum"),
        )
        .reset_index()
        .sort_values("value", ascending=False)
    )
    agency_summary.to_csv(audit_dir / "health_agency_summary.csv", index=False)

    # Candidate audit: show every source agency containing common health terms,
    # even if not currently selected, so the scope can be reviewed deliberately.
    candidate_pattern = r"health|aged care|medical|therapeutic"
    candidates = (
        df[df["Agency"].fillna("").astype(str).str.contains(candidate_pattern, case=False, regex=True)]
        .groupby("Agency", dropna=False)
        .agg(rows=("CN ID", "size"), value=("Value", "sum"))
        .reset_index()
        .sort_values("value", ascending=False)
    )
    candidates["selected"] = candidates["Agency"].map(normalise_agency).isin(configured_keys)
    candidates.to_csv(audit_dir / "health_agency_candidates.csv", index=False)

    summary = {
        "input_rows": int(len(df)),
        "health_rows": int(len(health)),
        "health_contracts": int(health["CN ID"].nunique()),
        "health_value": float(pd.to_numeric(health["Value"], errors="coerce").fillna(0).sum()),
        "configured_agencies": configured_names,
        "matched_agencies": sorted(health["Agency"].dropna().astype(str).unique().tolist()),
        "configured_agencies_not_found": missing_configured,
        "output": str(output_path),
    }
    (audit_dir / "health_filter_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )

    print("ATLAS Health filter complete.")
    print(f"Input rows:  {len(df):,}")
    print(f"Health rows: {len(health):,}")
    print(f"Health value: ${summary['health_value']:,.2f}")
    if missing_configured:
        print("Configured agency names not found in current source:")
        for name in missing_configured:
            print(f"  - {name}")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
