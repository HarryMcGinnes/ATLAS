
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


review_categories = [
    "Military services and national defence",
    "Professional engineering services",
    "Management support services",
    "Software",
]

review = unclassified[
    unclassified["Category"].isin(review_categories)
].copy()

columns = [
    "CN ID",
    "Category",
    "Description",
    "Supplier Name",
    "Value",
]

review = review.sort_values("Value", ascending=False)

print("\nTOP 40 CONTRACTS FOR CLASSIFICATION REVIEW")
print(review[columns].head(40).to_string(index=False))

review[columns].to_csv(
    "defence_classification_review.csv",
    index=False
)
