"""Explicit, auditable ATLAS Federal Health agency rename groupings.

Only legally/administratively verified rename or direct-successor aliases are merged.
Other agencies (including portfolio entities and TGA-labelled buyers) stay separate.
The original AusTender 'Agency' field is never overwritten.
"""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd

RENAMED_AGENCIES = {
    'department of health': 'Department of Health, Disability and Ageing',
    'department of health and aged care': 'Department of Health, Disability and Ageing',
    'department of health, disability and ageing': 'Department of Health, Disability and Ageing',
    'australian aged care quality agency': 'Aged Care Quality and Safety Commission',
    'aged care quality and safety commission': 'Aged Care Quality and Safety Commission',
}

def _key(value: object) -> str:
    return re.sub(r'\s+', ' ', str(value or '').replace('\u00a0', ' ')).strip().casefold()

def add_agency_groups(df: pd.DataFrame) -> pd.DataFrame:
    if 'Agency' not in df.columns:
        raise ValueError('Original AusTender Agency column is required')
    out = df.copy()
    original = out['Agency'].fillna('Unspecified agency').astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()
    original = original.replace('', 'Unspecified agency')
    # Retain exactly which original agency was reported in master; use a separate
    # reporting field rather than altering the canonical 'Agency' column.
    out['agency_group'] = original.map(lambda name: RENAMED_AGENCIES.get(_key(name), name))
    return out

def write_agency_audit(df: pd.DataFrame, destination: Path) -> Path:
    if not {'Agency', 'agency_group', 'CN ID', 'Value'}.issubset(df.columns):
        raise ValueError('Missing agency audit fields')
    audit = (df.groupby(['Agency', 'agency_group'], dropna=False)
              .agg(contract_notices=('CN ID', 'nunique'), net_contract_value_aud=('Value', 'sum'))
              .reset_index().rename(columns={'Agency':'original_agency'}))
    audit['renamed'] = audit['original_agency'].ne(audit['agency_group'])
    if abs(float(audit['net_contract_value_aud'].sum()) - float(df['Value'].sum())) > 0.01:
        raise ValueError('Agency grouping audit values do not reconcile')
    if int(audit['contract_notices'].sum()) != int(df['CN ID'].nunique()):
        raise ValueError('Agency grouping audit notice counts do not reconcile')
    destination.parent.mkdir(parents=True, exist_ok=True)
    audit.sort_values(['agency_group','original_agency']).to_csv(destination, index=False)
    return destination
