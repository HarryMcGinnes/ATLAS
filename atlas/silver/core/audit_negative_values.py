
"""ATLAS read-only audit of negative contract values."""

from pathlib import Path
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


def audit(name, df):
    values = pd.to_numeric(df["Value"], errors="coerce")
    ids = df["CN ID"].fillna("").astype(str).str.upper()

    amendments = ids.str.contains(r"-A\d+$", regex=True)
    negative = values < 0

    return {
        "dataset": name,
        "rows": len(df),
        "negative_rows": int(negative.sum()),
        "negative_amendments": int((negative & amendments).sum()),
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
            df = pd.concat(
                [pd.read_parquet(f) for f in files],
                ignore_index=True,
            )
        else:
            df = pd.read_parquet(path)

        results.append(audit(name, df))

    report = pd.DataFrame(results)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT, index=False)

    print("\nATLAS NEGATIVE VALUE AUDIT")
    print("=" * 70)
    print(report.to_string(index=False))

    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
