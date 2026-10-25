"""Audit Defence Service Offering classification results."""

from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "data/gold/defence/defence_classified.parquet"
)

OUTPUT_DIR = Path("audits/gold/defence")


def main():
    df = pd.read_parquet(INPUT_PATH)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------
    # BASIC CLASSIFICATION STATES
    # ------------------------------------------------------------

    no_match = (
        df["service_offering"].isna()
        & df["service_offering_score"].eq(0)
    )

    ambiguous = (
        df["service_offering"].isna()
        & df["service_offering_score"].gt(0)
    )

    weak_score = (
        df["service_offering"].notna()
        & df["service_offering_score"].lt(12)
    )

    close_score = (
        df["service_offering"].notna()
        & df["service_offering_score_margin"].lt(4)
    )

    df["classification_audit_status"] = "classified"

    df.loc[
        no_match,
        "classification_audit_status",
    ] = "no_match"

    df.loc[
        ambiguous,
        "classification_audit_status",
    ] = "ambiguous"

    df.loc[
        weak_score,
        "classification_audit_status",
    ] = "weak_score"

    df.loc[
        close_score,
        "classification_audit_status",
    ] = "close_score"

    # ------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------

    summary = (
        df.groupby(
            "classification_audit_status",
            dropna=False,
        )
        .agg(
            contracts=("CN ID", "count"),
            total_value=("Value", "sum"),
        )
        .reset_index()
        .sort_values(
            "total_value",
            ascending=False,
        )
    )

    summary["value_bn"] = (
        summary["total_value"] / 1e9
    )

    print()
    print("CLASSIFICATION AUDIT")
    print("=" * 60)
    print(
        summary[
            [
                "classification_audit_status",
                "contracts",
                "value_bn",
            ]
        ].to_string(index=False)
    )

    # ------------------------------------------------------------
    # UNCLASSIFIED CATEGORY SUMMARY
    # ------------------------------------------------------------

    unclassified = df[
        df["service_offering"].isna()
    ].copy()

    category_summary = (
        unclassified.groupby(
            ["Category Type", "Category"],
            dropna=False,
        )
        .agg(
            contracts=("CN ID", "count"),
            total_value=("Value", "sum"),
        )
        .reset_index()
        .sort_values(
            "total_value",
            ascending=False,
        )
    )

    print()
    print("TOP 30 UNCLASSIFIED CATEGORIES")
    print("=" * 60)
    print(
        category_summary
        .head(30)
        .to_string(index=False)
    )

    print(
        f"\nUnclassified value: "
        f"A${unclassified['Value'].sum() / 1e9:,.2f}B"
    )

    # ------------------------------------------------------------
    # REVIEW CANDIDATES
    # ------------------------------------------------------------

    review_terms = (
        r"software|digital|ict|information technology|"
        r"cloud|cyber|data|analytics|"
        r"systems? integration|"
        r"engineering|"
        r"consulting|advisory|transformation|"
        r"managed services?|"
        r"sustainment|support services?"
    )

    service_candidate = (
        unclassified["Description"]
        .fillna("")
        .str.contains(
            review_terms,
            case=False,
            regex=True,
        )
    )

    candidates = unclassified[
        service_candidate
    ].copy()

    candidates = candidates.sort_values(
        "Value",
        ascending=False,
    )

    review_columns = [
        "CN ID",
        "Category Type",
        "Category",
        "Description",
        "Supplier Name",
        "Value",
        "service_offering_score",
        "service_offering_runner_up",
        "service_offering_runner_up_score",
        "service_offering_score_margin",
        "service_offering_confidence",
        "service_offering_reason",
        "classification_audit_status",
    ]

    candidates[
        review_columns
    ].to_csv(
        OUTPUT_DIR
        / "classification_review_queue.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # AMBIGUOUS CLASSIFICATIONS
    # ------------------------------------------------------------

    df.loc[
        ambiguous,
        review_columns,
    ].sort_values(
        "Value",
        ascending=False,
    ).to_csv(
        OUTPUT_DIR
        / "classification_ambiguous.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # WEAK / CLOSE CLASSIFICATIONS
    # ------------------------------------------------------------

    classified_review = df[
        weak_score | close_score
    ].copy()

    classified_review[
        review_columns
    ].sort_values(
        "Value",
        ascending=False,
    ).to_csv(
        OUTPUT_DIR
        / "classification_low_confidence.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # TOP REVIEW CONTRACTS
    # ------------------------------------------------------------

    print()
    print("TOP 40 UNCLASSIFIED SERVICE CANDIDATES")
    print("=" * 60)

    print(
        candidates[
            [
                "CN ID",
                "Category",
                "Description",
                "Supplier Name",
                "Value",
            ]
        ]
        .head(40)
        .to_string(index=False)
    )

    print()
    print(f"Review candidates: {len(candidates):,}")
    print(
        f"Review value: "
        f"A${candidates['Value'].sum() / 1e9:,.2f}B"
    )

    print()
    print("Audit files written to:")
    print(OUTPUT_DIR)


    print("\nUNCLASSIFIED SCORE DISTRIBUTION")
    
    scores = pd.to_numeric(
        unclassified["service_offering_score"],
        errors="coerce"
    ).fillna(0)
    
    print(pd.cut(
        scores,
        bins=[-1, 0, 15, 25, 35, 50, 65, 100],
        labels=[
            "0",
            "1-15",
            "16-25",
            "26-35",
            "36-50",
            "51-65",
            "66-100"
        ]
    ).value_counts().sort_index())
    
    print("\nHIGH-SCORING UNCLASSIFIED CONTRACTS")
    
    high = unclassified.loc[
        scores >= 35,
        [
            "CN ID", "Category", "Description",
            "service_offering_score",
            "service_offering_reason"
        ]
    ].sort_values(
        "service_offering_score",
        ascending=False
    )
    
    print(high.head(20).to_string(index=False))



if __name__ == "__main__":
    main()
