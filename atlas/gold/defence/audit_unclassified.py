
import pandas as pd

df = pd.read_parquet(
    "data/gold/defence/defence_classified.parquet"
)

unclassified = df[df["service_offering"].isna()].copy()

summary = (
    unclassified.groupby(
        ["Category Type", "Category"],
        dropna=False
    )
    .agg(
        contracts=("CN ID", "count"),
        total_value=("Value", "sum")
    )
    .reset_index()
    .sort_values("total_value", ascending=False)
)

print("\nTOP 30 UNCLASSIFIED CATEGORIES")
print(summary.head(30).to_string(index=False))

print(
    f"\nUnclassified value: "
    f"A${unclassified['Value'].sum()/1e9:,.2f}B"
)
