"""Shared ATLAS Health active-contract classification.

Mirrors the date validation/recovery rules in HealthContractExplorer.py.
Dates are as-at Australia/Sydney and historical signed CN values are NOT spend.
"""
from __future__ import annotations
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd

START_DATE_CANDIDATES = ['Start Date', 'Contract Start Date', 'Contract Period Start', 'Contract Start']
END_DATE_CANDIDATES = ['End Date', 'Contract End Date', 'Contract Period End', 'Contract End', 'Expiry Date']
CREDIBLE = {'valid', 'inferred_family_start', 'inferred_original_missing'}
REVIEW = {'review_original_missing', 'review_invalid_version_dates',
          'review_invalid_family_period', 'review_missing_start', 'review_missing_end'}

def _first(df, names):
    mapping = {str(c).lower():c for c in df.columns}
    return next((mapping[n.lower()] for n in names if n.lower() in mapping), None)

def _parse(series):
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors='coerce').dt.tz_localize(None) if getattr(series.dt, 'tz', None) is not None else pd.to_datetime(series, errors='coerce')
    s=series.astype('string').str.strip()
    iso=s.str.match(r'^\d{4}-\d{1,2}-\d{1,2}(?:[ T].*)?$',na=False)
    result=pd.Series(pd.NaT,index=series.index,dtype='datetime64[ns]')
    result.loc[iso]=pd.to_datetime(s.loc[iso],errors='coerce',format='ISO8601').dt.tz_localize(None) if iso.any() else result.loc[iso]
    if (~iso).any():
        result.loc[~iso]=pd.to_datetime(s.loc[~iso],errors='coerce',format='mixed',dayfirst=True)
    return result

def add_contract_status(df: pd.DataFrame, as_at=None):
    """Return df with _contract_status, _contract_end and _expiry_12m.

    No fallback to an unverified individual date: UNKNOWN is retained.
    """
    missing={'cn_family_start_date','cn_family_end_date','cn_family_date_status'}-set(df.columns)
    if missing:
        raise ValueError('Active view needs validated Contract Explorer family fields: '+', '.join(sorted(missing)))
    if 'CN ID' not in df.columns:
        raise ValueError('Active view needs CN ID')
    out=df.copy()
    start_col=_first(out,START_DATE_CANDIDATES)
    end_col=_first(out,END_DATE_CANDIDATES)
    raw_start=_parse(out[start_col]) if start_col else pd.Series(pd.NaT,index=out.index,dtype='datetime64[ns]')
    raw_end=_parse(out[end_col]) if end_col else pd.Series(pd.NaT,index=out.index,dtype='datetime64[ns]')
    family_start=_parse(out['cn_family_start_date'])
    family_end=_parse(out['cn_family_end_date'])
    status=out['cn_family_date_status'].astype('string').fillna('review_missing_family')
    ids=out['CN ID'].astype('string').str.strip().str.upper()
    root=(out['CN Root ID'].astype('string').str.strip().str.upper()
          if 'CN Root ID' in out else ids.str.replace(r'-A\d+$','',regex=True))
    valid_raw=raw_start.notna()&raw_end.notna()&raw_start.le(raw_end)
    observed_start=raw_start.where(valid_raw).groupby(root).transform('min')
    observed_end=raw_end.where(valid_raw).groupby(root).transform('max')
    parent_present=ids.eq(root).groupby(root).transform('any')
    recovered=(status.isin(REVIEW)&observed_start.notna()&observed_end.notna()
               &observed_start.le(observed_end))
    family_start=family_start.where(~recovered,observed_start)
    family_end=family_end.where(~recovered,observed_end)
    status=status.mask(recovered,'inferred_family_start')
    status=status.mask(recovered&~parent_present.fillna(False),'inferred_original_missing')
    credible=(status.isin(CREDIBLE)&family_start.notna()&family_end.notna()
              &family_end.ge(family_start))
    start=family_start.where(credible)
    end=family_end.where(credible)
    today=pd.Timestamp(as_at if as_at is not None else datetime.now(ZoneInfo('Australia/Sydney')).date()).normalize()
    active=credible&start.le(today)&end.ge(today)
    upcoming=credible&start.gt(today)
    completed=credible&end.lt(today)
    out['_contract_status']='Unknown Dates'
    out.loc[active,'_contract_status']='Active contracts'
    out.loc[upcoming,'_contract_status']='Upcoming contracts'
    out.loc[completed,'_contract_status']='Completed contracts'
    out['_contract_end']=end
    out['_expiry_12m']=active&end.le(today+pd.DateOffset(years=1))
    return out
