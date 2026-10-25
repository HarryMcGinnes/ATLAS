
from pathlib import Path
import argparse
import re

import pandas as pd


def key(value):
    if pd.isna(value):
        return ""
    return re.sub(r"[^A-Z0-9]+", "", str(value).upper())


def load_lookup(path):
    lookup = pd.read_csv(
        path, dtype=str, keep_default_na=False,
        encoding="utf-8-sig"
    )

    rename = {}
    for col in lookup.columns:
        compact = re.sub(r"[^a-z]", "", col.lower())
        if compact in {"agencydivision", "agencydivison", "division"}:
            rename[col] = "division"
        elif compact in {"agencybranch", "branch"}:
            rename[col] = "branch"
        elif compact == "domain":
            rename[col] = "domain"

    lookup = lookup.rename(columns=rename)

    required = {"division", "branch", "domain"}
    if not required.issubset(lookup.columns):
        raise ValueError("Domain lookup missing required columns")

    lookup["lookup_key"] = list(zip(
        lookup["division"].map(key),
        lookup["branch"].map(key)
    ))

    conflicts = lookup.groupby("lookup_key")["domain"].nunique()
    if (conflicts > 1).any():
        raise ValueError("Conflicting Defence domain mappings")

    return dict(zip(lookup["lookup_key"], lookup["domain"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/silver/industry/defence/defence_contracts.parquet"
    )
    parser.add_argument(
        "--lookup",
        default="defence/config/defence_domain_lookup.csv"
    )
    parser.add_argument(
        "--output",
        default="data/silver/industry/defence/defence_enriched.parquet"
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    mapping = load_lookup(args.lookup)

    division = df["35. Agency Division"].map(key)
    branch = df["36. Agency Branch"].map(key)

    lookup_keys = list(zip(division, branch))
    domains = [
        mapping.get(k, "Unmapped") for k in lookup_keys
    ]

    agency = df["Agency"].fillna("").str.lower().str.strip()

    df["defence_domain"] = domains

    df.loc[
        agency.eq("australian signals directorate"),
        "defence_domain"
    ] = "Cyber"

    df.loc[
        agency.eq("australian submarine agency"),
        "defence_domain"
    ] = "Maritime"

    df["domain_mapping_source"] = "Division + Branch lookup"

    df.loc[
        df["defence_domain"].eq("Unmapped"),
        "domain_mapping_source"
    ] = "No lookup match"

    df.loc[
        agency.isin([
            "australian signals directorate",
            "australian submarine agency"
        ]),
        "domain_mapping_source"
    ] = "Authoritative procuring agency"

    df["domain_confidence"] = (
        df["defence_domain"].ne("Unmapped").astype(int) * 100
    )
    df["domain_mapping_reason"] = df["domain_mapping_source"]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)

    print(f"Defence domain enrichment complete: {len(df):,} rows")
    print(df["defence_domain"].value_counts().to_string())
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
