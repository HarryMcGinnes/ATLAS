
"""ATLAS Health Gold - compare baseline and updated classifications."""

import argparse
from pathlib import Path

import pandas as pd


def load(path):
    df = pd.read_parquet(path)

    if "CN ID" not in df.columns:
        raise RuntimeError(f"Missing CN ID: {path}")

    if not df["CN ID"].is_unique:
        raise RuntimeError(f"Duplicate CN IDs: {path}")

    return df.set_index("CN ID")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", required=True)
    parser.add_argument("--new", required=True)
    parser.add_argument(
        "--output",
        default="audits/gold/health/health_qa_changes.csv",
    )
    args = parser.parse_args()

    old = load(args.old)
    new = load(args.new)

    common = old.index.intersection(new.index)

    print("\nHEALTH GOLD QA COMPARISON")
    print("=" * 60)
    print(f"Old contracts: {len(old):,}")
    print(f"New contracts: {len(new):,}")
    print(f"Shared CN IDs: {len(common):,}")
    print(f"Only in old: {len(old.index.difference(new.index)):,}")
    print(f"Only in new: {len(new.index.difference(old.index)):,}")

    old = old.loc[common]
    new = new.loc[common]

    old_addr = old["is_addressable"].fillna(False).astype(bool)
    new_addr = new["is_addressable"].fillna(False).astype(bool)

    old_so = old["service_offering"].fillna("Unresolved")
    new_so = new["service_offering"].fillna("Unresolved")

    old_conf = pd.to_numeric(
        old["addressability_confidence"], errors="coerce"
    )
    new_conf = pd.to_numeric(
        new["addressability_confidence"], errors="coerce"
    )

    addr_changed = old_addr.ne(new_addr)
    so_changed = old_so.ne(new_so)
    conf_changed = old_conf.ne(new_conf).fillna(False)

    changed = addr_changed | so_changed | conf_changed

    report = pd.DataFrame({
        "CN ID": common,
        "Agency": new["Agency"].to_numpy(),
        "Category": new["Category"].to_numpy(),
        "Description": new["Description"].to_numpy(),
        "Supplier Name": new["Supplier Name"].to_numpy(),
        "Value": new["Value"].to_numpy(),
        "is_accenture": new["is_accenture"].to_numpy(),
        "old_addressable": old_addr.to_numpy(),
        "new_addressable": new_addr.to_numpy(),
        "old_service_offering": old_so.to_numpy(),
        "new_service_offering": new_so.to_numpy(),
        "old_confidence": old_conf.to_numpy(),
        "new_confidence": new_conf.to_numpy(),
        "old_reason": old["addressability_reason"].to_numpy(),
        "new_reason": new["addressability_reason"].to_numpy(),
    })

    report = report.loc[changed.to_numpy()].copy()

    report["addressability_changed"] = addr_changed.loc[
        report["CN ID"]
    ].to_numpy()

    report["service_offering_changed"] = so_changed.loc[
        report["CN ID"]
    ].to_numpy()

    report["confidence_changed"] = conf_changed.loc[
        report["CN ID"]
    ].to_numpy()

    report["tam_value_impact"] = (
        report["Value"]
        * (
            report["new_addressable"].astype(int)
            - report["old_addressable"].astype(int)
        )
    )

    report = report.sort_values(
        "Value", ascending=False
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output, index=False)

    print(f"\nChanged contracts: {len(report):,}")
    print(f"Addressability changes: {addr_changed.sum():,}")
    print(f"Service Offering changes: {so_changed.sum():,}")
    print(f"Confidence changes: {conf_changed.sum():,}")
    print(
        f"Net TAM impact on shared CNs: "
        f"A${report['tam_value_impact'].sum()/1e6:,.2f}M"
    )

    acc_changes = report[
        report["is_accenture"].fillna(False).astype(bool)
    ]

    print(f"Accenture records changed: {len(acc_changes):,}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
