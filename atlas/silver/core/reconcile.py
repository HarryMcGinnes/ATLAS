
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", required=True)
    parser.add_argument("--new", required=True)
    args = parser.parse_args()

    old = pd.read_parquet(args.old)
    new = pd.read_parquet(args.new)

    print("\nATLAS SILVER CORE RECONCILIATION")
    print("=" * 50)

    for label, df in [("OLD", old), ("NEW", new)]:
        print(f"\n{label}")
        print(f"Rows: {len(df):,}")
        print(f"Columns: {len(df.columns)}")

        ids = df["CN ID"].astype("string").str.strip().str.upper()
        print(f"Unique CN IDs: {ids.nunique():,}")
        print(f"Duplicate CN IDs: {ids.duplicated().sum():,}")

        value = pd.to_numeric(df["Value"], errors="coerce")
        print(f"Total value: A${value.sum():,.2f}")

        accenture = df["supplier_group"].eq("Accenture")
        print(f"Accenture rows: {accenture.sum():,}")
        print(f"Accenture value: A${value[accenture].sum():,.2f}")

    old_ids = set(old["CN ID"].dropna().astype(str).str.strip().str.upper())
    new_ids = set(new["CN ID"].dropna().astype(str).str.strip().str.upper())

    print("\nCONTRACT DIFFERENCES")
    print(f"Only in old: {len(old_ids - new_ids):,}")
    print(f"Only in new: {len(new_ids - old_ids):,}")

    common = sorted(set(old.columns) & set(new.columns))
    print(f"\nShared columns: {len(common)}")
    print(f"Old-only columns: {sorted(set(old.columns) - set(new.columns))}")
    print(f"New-only columns: {sorted(set(new.columns) - set(old.columns))}")

    print("\nReconciliation report complete.")


if __name__ == "__main__":
    main()
