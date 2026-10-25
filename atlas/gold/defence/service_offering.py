"""ATLAS Defence Service Offering classifier."""

from pathlib import Path
import sys

import pandas as pd


sys.path.insert(
    0,
    str(Path(__file__).resolve().parent),
)

from engine import load_config, classify_contract


def classify_dataframe(df):
    """Classify Defence contracts independently of addressability."""

    config = load_config()

    required = {
        "Description",
        "Category",
        "Category Type",
        "Supplier Name",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    results = [
        classify_contract(
            contract,
            config,
        )
        for contract in df.to_dict("records")
    ]

    invalid_results = [
        index
        for index, result in enumerate(results)
        if not isinstance(result, dict)
    ]

    if invalid_results:
        raise RuntimeError(
            "Classifier returned an invalid result for "
            f"{len(invalid_results):,} contracts. "
            f"First invalid position: {invalid_results[0]}"
        )

    classified = pd.DataFrame(
        results,
        index=df.index,
    )

    required_output = {
        "service_offering",
        "service_offering_score",
        "service_offering_grade",
        "service_offering_runner_up",
        "service_offering_runner_up_score",
        "service_offering_score_margin",
        "service_offering_confidence",
        "service_offering_reason",
        "classification_review_required",
    }

    missing_output = (
        required_output
        - set(classified.columns)
    )

    if missing_output:
        raise RuntimeError(
            "Classifier results are missing columns: "
            f"{sorted(missing_output)}"
        )

    output = pd.concat(
        [
            df.drop(
                columns=classified.columns,
                errors="ignore",
            ),
            classified,
        ],
        axis=1,
    )

    if len(output) != len(df):
        raise RuntimeError(
            "Service Offering classification changed row count."
        )

    return output
