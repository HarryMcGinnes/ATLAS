
"""ATLAS Defence Service Offering classifier."""

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine import load_config, classify_contract


def classify_dataframe(df):
    """Classify all Defence contracts independently of addressability."""

    config = load_config()

    required = {"Description", "Category", "Supplier Name"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    results = [
        classify_contract(row, config)
        for row in df.to_dict("records")
    ]

    classified = pd.DataFrame(results, index=df.index)

    return pd.concat(
        [
            df.drop(columns=classified.columns, errors="ignore"),
            classified,
        ],
        axis=1,
    )
