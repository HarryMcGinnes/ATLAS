
"""Audit original contracts and amendment values."""

import re
from pathlib import Path

import pandas as pd


INPUT = Path("data/silver/core/austender_contracts.parquet")
OUTPUT = Path("audits/silver/core/amendment_review.csv")

REVIEW_CNS = [
    "CN3352798",
    "CN3749665",
    "CN3834297",
    "CN1793161",
    "CN3244632",
    "CN3282367",
    "CN3314082",
]


def main():
    df = pd.read_parquet(INPUT)

    df["cn_id"] = df["CN ID"].astype("string").str.strip().str.upper()

    df["cn_family"] = df["cn_id"].str.replace(
        r"-A\d+$", "", regex=True
    )

    df["amendment_number"] = (
        df["cn_id"]
        .str.extract(r"-A(\d+)$", expand=False)
        .fillna("0")
        .astype(int)
    )

    df["contract_value"] = pd.to_numeric(
        df["Value"], errors="coerce"
    )

    review = df[df["cn_family"].isin(REVIEW_CNS)].copy()

    review = review.sort_values(
        ["cn_family", "amendment_number"]
    )

    columns = [
        "cn_id",
        "cn_family",
        "amendment_number",
        "Description",
        "Supplier Name",
        "contract_value",
        "Start Date",
        "End Date",
    ]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    review[columns].to_csv(OUTPUT, index=False)

    print("\nAMENDMENT VALUE AUDIT")
    print("=" * 65)

    for family, group in review.groupby("cn_family"):
        print(f"\n{family}")

        print(
            group[
                ["cn_id", "amendment_number", "contract_value"]
            ].to_string(index=False)
        )

        total = group["contract_value"].sum()
        latest = group.iloc[-1]["contract_value"]

        print(f"Sum of records: A${total:,.2f}")
        print(f"Latest amendment value: A${latest:,.2f}")

    print(f"\nAudit exported: {OUTPUT}")


if __name__ == "__main__":
    main()
