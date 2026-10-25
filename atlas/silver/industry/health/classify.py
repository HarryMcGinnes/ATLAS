

from pathlib import Path
import argparse
import re

import pandas as pd


def normalise(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).replace("\u00a0", " ").strip()).lower()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/silver/core/austender_contracts.parquet",
    )
    parser.add_argument(
        "--agency-config",
        default="health/config/health_agencies.txt",
    )
    parser.add_argument(
        "--aliases",
        default="config/health/agency_aliases.csv",
    )
    parser.add_argument(
        "--output",
        default="data/silver/industry/health/health_contracts.parquet",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.input)

    agencies = [
        line.strip()
        for line in Path(args.agency_config).read_text(
            encoding="utf-8-sig"
        ).splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    aliases = pd.read_csv(
        args.aliases, dtype=str, keep_default_na=False
    )

    alias_map = dict(zip(
        aliases["source_agency"].map(normalise),
        aliases["canonical_agency"]
    ))

    # ATLAS scope is authoritative; aliases extend it
    # only to historical names of already included agencies.
    scope = {normalise(name) for name in agencies}
    canonical_scope = {
        alias_map.get(name, name)
        for name in scope
    }

    for source, canonical in alias_map.items():
        if canonical in canonical_scope:
            scope.add(source)

    agency_col = "01. Agency Name"
    raw_agency = df[agency_col].fillna("").astype(str)
    normalised = raw_agency.map(normalise)

    health = df.loc[normalised.isin(scope)].copy()

    if health.empty:
        raise RuntimeError("Health filter returned zero contracts")

    health["Agency Raw"] = raw_agency.loc[health.index]
    health["agency_group"] = normalised.loc[
        health.index
    ].map(alias_map).fillna(health["Agency Raw"])

    health["Agency"] = health["agency_group"]
    health["is_health_scope"] = True
    health["atlas_market"] = "Health"

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    health.to_parquet(output, index=False)

    print(f"Health contracts: {len(health):,}")
    print(f"Health value: A${health['Value'].sum()/1e9:,.3f}B")
    print(f"Accenture rows: {health['is_accenture'].sum():,}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
