

from pathlib import Path
import argparse
import pandas as pd

from service_offering import classify_dataframe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/silver/industry/defence/defence_enriched.parquet"
    )
    parser.add_argument(
        "--output",
        default="data/gold/defence/defence_classified.parquet"
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    result = classify_dataframe(df)

    assert len(result) == len(df), "Row count changed"
    assert result["Value"].sum() == df["Value"].sum(), "Value changed"

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)

    print("\nDEFENCE SERVICE OFFERING RESULTS")
    print("=" * 40)
    print(result["service_offering"].fillna("Unclassified").value_counts())

    coverage = result["service_offering"].notna().mean() * 100
    print(f"\nClassification coverage: {coverage:.2f}%")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
