"""ATLAS Health: validate time windows and Accenture TAM reconciliation.

Usage:
  python validate_health_tam_fy.py --input health/data/processed/master_health_contracts.parquet
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.fillna(False).astype(str).str.strip().str.lower().isin(('true', '1', 'yes', 'y'))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='health/data/processed/master_health_contracts.parquet')
    args = parser.parse_args()
    path = Path(args.input)
    if not path.exists():
        parser.error(f'Input not found: {path}')
    df = pd.read_parquet(path) if path.suffix.lower() in ('.parquet', '.pq') else pd.read_csv(path, low_memory=False)
    required = {'Value', 'Financial Year', 'is_addressable'}
    missing = required - set(df.columns)
    if missing:
        parser.error('Missing columns: ' + ', '.join(sorted(missing)))
    if 'is_accenture' not in df.columns and 'supplier_group' not in df.columns:
        parser.error('Need is_accenture or supplier_group to identify Accenture wins')

    value = pd.to_numeric(df['Value'], errors='raise')
    if value.isna().any():
        parser.error('Null contract values detected')
    addr = as_bool(df['is_addressable'])
    acc = as_bool(df['is_accenture']) if 'is_accenture' in df.columns else df['supplier_group'].fillna('').astype(str).str.strip().str.casefold().eq('accenture')
    if 'supplier_group' in df.columns and 'is_accenture' in df.columns:
        group_acc = df['supplier_group'].fillna('').astype(str).str.strip().str.casefold().eq('accenture')
        if not acc.equals(group_acc):
            print(f'WARNING: {int((acc != group_acc).sum()):,} supplier identity disagreements')

    fy = pd.to_numeric(df['Financial Year'].astype('string').str.extract(r'((?:19|20)\d{2})', expand=False), errors='coerce')
    today = datetime.now(ZoneInfo('Australia/Sydney'))
    current_fy = today.year if today.month >= 7 else today.year - 1
    future_rows = fy.gt(current_fy).fillna(False)
    if future_rows.any():
        print(f'WARNING: {int(future_rows.sum()):,} records tagged with an FY after FY{current_fy}-{str((current_fy+1)%100).zfill(2)}. Review whether they are genuine forward-dated records.')
    print(f'Australian current FY: FY{current_fy}-{str((current_fy+1)%100).zfill(2)}')

    failed = False
    for label, n in [('Current FY', 1), ('Last 3 FYs', 3), ('Last 5 FYs', 5)]:
        scope = fy.ge(current_fy - n + 1) & fy.le(current_fy)
        all_acc = float(value[scope & acc].sum())
        tam_acc = float(value[scope & acc & addr].sum())
        tam = float(value[scope & addr].sum())
        missing_acc = int((scope & acc & ~addr).sum())
        diff = all_acc - tam_acc
        passed = abs(diff) <= 0.01
        failed |= not passed
        print(f'\n{label}: FY{current_fy-n+1}-{str((current_fy-n+2)%100).zfill(2)} to FY{current_fy}-{str((current_fy+1)%100).zfill(2)}')
        print(f'  TAM A${tam/1e6:,.3f}M | Accenture all A${all_acc/1e6:,.3f}M | Accenture TAM A${tam_acc/1e6:,.3f}M')
        print(f'  Reconciliation: {"PASS" if passed else "FAIL"} | Difference A${diff:,.2f} | Accenture rows non-addressable: {missing_acc}')
    print('\nAll history Accenture non-addressable records:', int((acc & ~addr).sum()))
    if failed:
        raise SystemExit('FAIL: Accenture TAM and all-market totals differ in at least one window.')
    print('PASS: All compared period totals reconcile; if the dashboard differs, investigate UI FY boundaries or filtering.')


if __name__ == '__main__':
    main()
