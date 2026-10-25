
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


def load_legacy_combiner(path: Path):
    """Load existing reconciliation logic during migration."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing legacy combiner: {path}")

    spec = importlib.util.spec_from_file_location(
        "atlas_legacy_combiner", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bronze-dir",
        default="data/bronze/austender",
    )
    parser.add_argument(
        "--output",
        default="data/silver/core/austender_contracts.parquet",
    )
    parser.add_argument(
        "--audit-dir",
        default="audits/silver/core",
    )
    args = parser.parse_args()

    bronze = Path(args.bronze_dir)
    if not (bronze / "manifest.json").is_file():
        raise RuntimeError("Bronze manifest missing")

    # Temporary compatibility bridge to existing tested logic.
    legacy = load_legacy_combiner(
        Path("pipeline/combine_financial_year_analysis.py")
    )

    import json
    import pandas as pd

    manifest = json.loads(
        (bronze / "manifest.json").read_text(encoding="utf-8")
    )

    audit_dir = Path(args.audit_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for item in manifest:
        df = pd.read_parquet(item["output"])
        kind = (
            "baseline"
            if Path(item["source"]).suffix.lower() == ".parquet"
            else "monthly"
        )
        frames.append(
            legacy.normalise_source(
                df, kind, Path(item["source"]).name
            )
        )

    if not frames:
        raise RuntimeError("No Bronze inputs")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = legacy.deduplicate_exact_cn_versions(
        combined, audit_dir
    )
    combined = legacy.add_contract_version_metadata(combined)
    combined = legacy.add_supplier_identity(combined, audit_dir)
    combined = legacy.add_derived_fields(combined)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output, index=False)

    print(f"Silver Core complete: {len(combined):,} rows")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
