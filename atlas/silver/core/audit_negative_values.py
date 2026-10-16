
"""ATLAS read-only negative contract value audit."""

from pathlib import Path
import re
import pandas as pd


DATASETS = {
    "Bronze": Path("data/bronze/austender"),
    "Silver Core": Path(
        "data/silver/core/austender_contracts.parquet"
    ),
    "Defence Gold": Path(
        "data/gold/defence/defence_classified.parquet"
    ),
}

OUTPUT = Path("audits/silver/core/negative_values_audit.csv")


def find_column(df, exact_names, keywords=None):
    """Identify a column without changing the source dataset."""

    for name in exact_names:
        if name in df.columns:
            return name

    if keywords:
        matches = [
            col for col in df.columns
            if all(word in col.lower() for word in keywords)
        ]

        if len(matches) == 1:
            return matches[0]

    raise ValueError(
        f"Cannot uniquely identify column. "
        f"Expected: {exact_names}. "
        f"Available: {list(df.columns)}"
    )


def audit(name, df, source=""):
    value_col = find_column(
        df,
        [
            "Value",
            "37. Value",
            "Contract Value",
            "Contract Value (AUD)",
            "Contract Value (Including GST)",
        ],
        keywords=["value"],
    )

    cn_col = find_column(
        df,
        ["CN ID", "03. Contract Notice ID"],
    )

    values = pd.to_numeric(
        df[value_col]
        .astype("string")
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.strip(),
        errors="coerce",
    )

    ids = (
        df[cn_col]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    amendments = ids.str.contains(
        r"-A\d+$", regex=True
    )

    negative = values < 0

    return {
        "dataset": name,
        "source": source,
        "value_column": value_col,
        "rows": len(df),
        "negative_rows": int(negative.sum()),
        "negative_amendments": int(
            (negative & amendments).sum()
        ),
        "negative_total": float(values[negative].sum()),
        "negative_amendment_total": float(
            values[negative & amendments].sum()
        ),
        "missing_values": int(values.isna().sum()),
        "total_value": float(values.sum()),
    }


def main():
    results = []

    for name, path in DATASETS.items():
        if path.is_dir():
            files = sorted(path.rglob("*.parquet"))

            if not files:
                raise FileNotFoundError(
                    f"No Parquet files found in {path}"
                )

            for file in files:
                df = pd.read_parquet(file)
                results.append(
                    audit(name, df, str(file))
                )

        else:
            df = pd.read_parquet(path)
            results.append(
                audit(name, df, str(path))
            )

    report = pd.DataFrame(results)

    OUTPUT.parent.mkdir(
        parents=True, exist_ok=True
    )

    report.to_csv(OUTPUT, index=False)

    print("\nATLAS NEGATIVE VALUE AUDIT")
    print("=" * 90)

    print(report.to_string(index=False))

    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
