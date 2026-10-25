
from pathlib import Path
import argparse
import hashlib
import json

import pandas as pd


def ingest_file(path, output_dir):
    """Preserve source records and create a Bronze Parquet file."""
    path = Path(path)

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(
            path,
            dtype=str,
            encoding="utf-8-sig",
            low_memory=False,
        )
    elif path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file: {path}")

    if df.empty:
        raise ValueError(f"Empty source: {path}")

    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    output = output_dir / f"{path.stem}_{checksum[:12]}.parquet"

    df.to_parquet(output, index=False)

    return {
        "source": str(path),
        "checksum": checksum,
        "rows": len(df),
        "columns": len(df.columns),
        "output": str(output),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--monthly-dir", required=True)
    parser.add_argument("--output-dir", default="data/bronze/austender")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = [Path(args.baseline)]
    sources += sorted(
        Path(args.monthly_dir).glob(
            "financial_year_analysis_*.csv"
        )
    )

    manifest = []

    for source in sources:
        result = ingest_file(source, output_dir)
        manifest.append(result)
        print(f"Ingested: {source.name} ({result['rows']:,} rows)")

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("Bronze ingestion complete.")


if __name__ == "__main__":
    main()
