"""Read-only audit of ATLAS contract-family dates.

Usage (PowerShell):
 python audit_atlas_amendment_dates.py --master health/data/processed/master_health_contracts.parquet --baseline data/austender/baseline/austender_baseline.parquet

The baseline is optional; run without it to isolate dashboard suppression first.
No input files are modified and no dates or financial values are recalculated.
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path
import pandas as pd

ID_NAMES = ['CN ID', 'Contract Notice ID', 'CN_ID', 'cn_id']
START_NAMES = ['Start Date', 'Contract Start Date', 'Contract Period Start', 'Contract Start']
END_NAMES = ['End Date', 'Contract End Date', 'Contract Period End', 'Contract End', 'Expiry Date']
FAM_START = 'cn_family_start_date'
FAM_END = 'cn_family_end_date'
FAM_STATUS = 'cn_family_date_status'
CREDIBLE = {'valid', 'inferred_family_start', 'inferred_original_missing'}
CN_RE = re.compile(r'(?i)(-A\d+)$')

def column(columns, names):
    names_lower = {str(c).strip().lower(): c for c in columns}
    for n in names:
        if n.lower() in names_lower:
            return names_lower[n.lower()]
    return None

def read_selected(path, names):
    """Read only desired columns for parquet (baseline can be large)."""
    if not path.exists():
        raise SystemExit(f'Missing input: {path}')
    if path.suffix.lower() in ('.parquet', '.pq'):
        import pyarrow.parquet as pq
        available = pq.read_schema(path).names
        chosen = list(dict.fromkeys(c for c in names if c in available))
        return pd.read_parquet(path, columns=chosen)
    if path.suffix.lower() == '.csv':
        return pd.read_csv(path, usecols=lambda x: x in names, low_memory=False)
    raise SystemExit(f'Unsupported input extension: {path}')

def parse_dates(values):
    # 'mixed' accepts both ISO and AusTender DD/MMM/YYYY dates.
    try:
        return pd.to_datetime(values, errors='coerce', dayfirst=True, format='mixed')
    except (TypeError, ValueError):
        return pd.to_datetime(values, errors='coerce', dayfirst=True)

def clean_id(s):
    return s.fillna('').astype(str).str.strip().str.upper()

def family(s):
    return s.str.replace(CN_RE, '', regex=True)

def flag(value):
    return str(value).strip().lower() in ('true', '1', 'yes', 'y')

def show_counts(name, s, top=30):
    print('\n'+name)
    print(s.fillna('<missing>').astype(str).value_counts(dropna=False).head(top).to_string())

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--master', required=True, type=Path, help='Current Health dashboard master parquet/CSV')
    ap.add_argument('--baseline', type=Path, default=None, help='Optional historical baseline parquet/CSV')
    ap.add_argument('--output-dir', type=Path, default=Path('audits/amendment_dates'))
    args=ap.parse_args()

    df = pd.read_parquet(args.master) if args.master.suffix.lower() in ('.parquet', '.pq') else pd.read_csv(args.master, low_memory=False)
    idc=column(df.columns, ID_NAMES)
    sc=column(df.columns, START_NAMES)
    ec=column(df.columns, END_NAMES)
    absent=[c for c in (FAM_START,FAM_END,FAM_STATUS) if c not in df.columns]
    if not idc or absent:
        raise SystemExit('Required master columns missing: '+str({'CN ID': idc, 'family_fields': absent}))
    print(f'MASTER {args.master} | records {len(df):,} | fields {len(df.columns)}')
    print(f'Raw date fields: start={sc!r}, end={ec!r}')
    out=pd.DataFrame(index=df.index)
    out['cn_id']=clean_id(df[idc])
    out['family_id']=family(out['cn_id'])
    out['is_amendment']=out['cn_id'].ne(out['family_id'])
    out['date_status']=df[FAM_STATUS].fillna('review_missing_family').astype(str).str.strip()
    out['source_start']=parse_dates(df[sc]) if sc else pd.NaT
    out['source_end']=parse_dates(df[ec]) if ec else pd.NaT
    out['family_start']=parse_dates(df[FAM_START])
    out['family_end']=parse_dates(df[FAM_END])
    source_ok=out['source_start'].notna() & out['source_end'].notna() & out['source_end'].ge(out['source_start'])
    family_ok=out['family_start'].notna() & out['family_end'].notna() & out['family_end'].ge(out['family_start'])
    trusted=out['date_status'].isin(CREDIBLE) & family_ok
    out['dashboard_unknown_dates']=~trusted
    out['source_dates_complete_valid']=source_ok
    out['family_dates_complete_valid']=family_ok
    out['suppressed_despite_valid_family_dates']=~trusted & family_ok
    out['source_dates_valid_but_dashboard_unknown']=~trusted & source_ok
    out['raw_dates_reversed']=out['source_start'].notna() & out['source_end'].notna() & out['source_end'].lt(out['source_start'])
    out['family_dates_reversed']=out['family_start'].notna() & out['family_end'].notna() & out['family_end'].lt(out['family_start'])
    out['raw_start_missing']=out['source_start'].isna()
    out['raw_end_missing']=out['source_end'].isna()
    out['family_start_missing']=out['family_start'].isna()
    out['family_end_missing']=out['family_end'].isna()
    out['parent_in_master']=out['family_id'].isin(out['cn_id'])
    out['duplicate_cn_in_master']=out['cn_id'].duplicated(keep=False)
    out['parent_in_baseline']=pd.NA
    out['baseline_parent_start']=pd.NaT
    out['baseline_parent_end']=pd.NaT
    if 'is_addressable' in df.columns:
        out['is_addressable']=df['is_addressable'].map(flag)
    if 'Agency' in df.columns: out['agency']=df['Agency'].fillna('').astype(str)
    if 'Value' in df.columns: out['value']=pd.to_numeric(df['Value'],errors='coerce')

    if args.baseline:
        if args.baseline.suffix.lower() in ('.parquet', '.pq'):
            import pyarrow.parquet as pq
            names=pq.read_schema(args.baseline).names
        else:
            names=pd.read_csv(args.baseline,nrows=0).columns.tolist()
        bid=column(names, ID_NAMES)
        bs=column(names, START_NAMES)
        be=column(names, END_NAMES)
        if not bid:
            print('WARNING: Baseline has no recognized CN ID field; skipping parent match')
        else:
            b=read_selected(args.baseline, [bid]+[x for x in (bs,be) if x])
            b['_cn']=clean_id(b[bid])
            parent_ids=set(b['_cn'])
            out['parent_in_baseline']=out['family_id'].isin(parent_ids)
            parents=b.loc[b['_cn'].isin(set(out['family_id']))].copy()
            # Keep only actual parent notices, not amendment rows.
            parents=parents.loc[parents['_cn'].eq(family(parents['_cn']))]
            parents['baseline_parent_start']=parse_dates(parents[bs]) if bs else pd.NaT
            parents['baseline_parent_end']=parse_dates(parents[be]) if be else pd.NaT
            parents=parents.sort_values(['_cn','baseline_parent_start'],na_position='last').drop_duplicates('_cn', keep='first')
            mapping=parents.set_index('_cn')[['baseline_parent_start','baseline_parent_end']]
            out['baseline_parent_start']=out['family_id'].map(mapping['baseline_parent_start'])
            out['baseline_parent_end']=out['family_id'].map(mapping['baseline_parent_end'])
            print(f'BASELINE {args.baseline} | records {len(b):,} | distinct parent IDs {len(parent_ids):,}')

    show_counts('Family date statuses (all Health records)', out['date_status'])
    print('\nHEALTH METRICS')
    for label,col in [('Unknown dates (dashboard rule)','dashboard_unknown_dates'),('Unknown with valid family dates but rejected status','suppressed_despite_valid_family_dates'),('Unknown with valid raw dates','source_dates_valid_but_dashboard_unknown'),('Raw start missing','raw_start_missing'),('Raw end missing','raw_end_missing'),('Raw dates reversed','raw_dates_reversed'),('Family dates reversed','family_dates_reversed'),('Duplicate CN IDs','duplicate_cn_in_master')]:
        print(f'  {label}: {int(out[col].sum()):,}')
    unknown=out.loc[out['dashboard_unknown_dates']]
    print(f'\nUnknown date records: {len(unknown):,}; distinct CN families: {unknown.family_id.nunique():,}; amendments: {int(unknown.is_amendment.sum()):,}')
    show_counts('Unknown date statuses', unknown['date_status'])
    print('\nUnknown date breakdown (amendment / original):')
    print(pd.crosstab(unknown['date_status'], unknown['is_amendment']).to_string())
    if args.baseline:
        print('\nUnknown-date amendment parent found in baseline:')
        print(unknown.loc[unknown.is_amendment, 'parent_in_baseline'].value_counts(dropna=False).to_string())
    if 'is_addressable' in out:
        print('\nUnknown dates by addressability:')
        print(unknown['is_addressable'].value_counts(dropna=False).to_string())
    args.output_dir.mkdir(parents=True,exist_ok=True)
    out.to_csv(args.output_dir/'all_health_contract_date_audit.csv',index=False,date_format='%Y-%m-%d')
    unknown.to_csv(args.output_dir/'unknown_date_contracts.csv',index=False,date_format='%Y-%m-%d')
    print(f'\nAUDIT WRITTEN: {args.output_dir / "unknown_date_contracts.csv"}')
    print(f'AUDIT WRITTEN: {args.output_dir / "all_health_contract_date_audit.csv"}')
    print('No datasets or dashboard scripts changed.')

if __name__=='__main__':
    main()
