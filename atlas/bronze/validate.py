
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq


def main():
    bronze_dir = Path("data/bronze/austender")
    manifest_path = bronze_dir / "manifest.json"

    if not manifest_path.exists():
        sys.exit("FAIL: Bronze manifest missing")

    manifest = json.loads(manifest_path.read_text())

    if not manifest:
        sys.exit("FAIL: No Bronze sources found")

    for item in manifest:
        path = Path(item["output"])

        if not path.exists():
            sys.exit(f"FAIL: Missing output {path}")

        metadata = pq.read_metadata(path)

        if metadata.num_rows != item["rows"]:
            sys.exit(f"FAIL: Row mismatch in {path.name}")

        if metadata.num_columns != item["columns"]:
            sys.exit(f"FAIL: Column mismatch in {path.name}")

        print(f"PASS: {path.name} ({metadata.num_rows:,} rows)")

    print("Bronze validation PASSED")


if __name__ == "__main__":
    main()
