
from pathlib import Path
import argparse
import pandas as pd

DEFENCE_AGENCIES = {
    "department of defence": "Department of Defence",
    "australian signals directorate": "Australian Signals Directorate",
    "australian submarine agency": "Australian Submarine Agency",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/silver/core/austender_contracts.parquet",
    )
    parser.add_argument(
        "--output",
        default="data/silver/industry/defence/defence_contracts.parquet",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.input)

    agency_col = (
        "Agency" if "Agency" in df.columns
        else "01. Agency Name"
    )

    agency = df[agency_col].fillna("").astype(str).str.strip().str.lower()
    mask = agency.isin(DEFENCE_AGENCIES)

    defence = df.loc[mask].copy()

    if defence.empty:
        raise RuntimeError("Defence filter returned zero contracts")

    defence["Agency Raw"] = defence[agency_col]
    defence["agency_group"] = agency.loc[mask].map(DEFENCE_AGENCIES)
    defence["Agency"] = defence["agency_group"]
    defence["is_defence_scope"] = True
    defence["atlas_market"] = "Defence"

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    defence.to_parquet(output, index=False)

    print(f"Defence contracts: {len(defence):,}")
    print(f"Defence value: A${defence['Value'].sum()/1e9:,.3f}B")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()

