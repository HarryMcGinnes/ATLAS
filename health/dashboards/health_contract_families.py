"""Contract families for the ATLAS Health dashboards.

A contract family is an original AusTender contract notice plus its amendments
(CN1234567, CN1234567-A1, CN1234567-A2, ...). An amendment is part of the original
contract. It is never counted as a new or different contract.

Rules used throughout the Health dashboards:

* Contract COUNTS, row lists and "contracts" labels use the family.
* Dollar values stay notice-level. AusTender amendment values in the source are
  signed changes, so the notices of a family sum to its current contract value.
  Totals therefore do not change when families are grouped.
* Family attributes (agency, supplier, Service Offering, dates, classification)
  come from the original notice, or from the earliest notice if the original is
  not in the dataset.
"""
from __future__ import annotations

import re

import pandas as pd

FAMILY_COL = "contract_family_id"
AMENDMENT_COL = "is_amendment"

_AMENDMENT_SUFFIX = r"-A\d+$"


def family_id(cn_id: object) -> str:
    """Family root for one contract notice ID, e.g. CN4085822-A2 -> CN4085822."""
    text = str(cn_id or "").strip().upper()
    return re.sub(_AMENDMENT_SUFFIX, "", text)


def add_family_fields(df: pd.DataFrame, cn_col: str = "CN ID") -> pd.DataFrame:
    """Return a copy with contract_family_id and is_amendment added."""
    out = df.copy()
    ids = out[cn_col].astype("string").str.strip().str.upper()
    out[FAMILY_COL] = ids.str.replace(_AMENDMENT_SUFFIX, "", regex=True).astype(str)
    out[AMENDMENT_COL] = ids.str.contains(_AMENDMENT_SUFFIX, regex=True, na=False).astype(bool)
    return out


def count_families(frame: pd.DataFrame, cn_col: str = "CN ID") -> int:
    """Number of distinct contracts (families) in a frame."""
    if frame is None or len(frame) == 0:
        return 0
    if FAMILY_COL in frame.columns:
        return int(frame[FAMILY_COL].nunique())
    return int(frame[cn_col].astype(str).map(family_id).nunique())


def apply_family_fy(df: pd.DataFrame, fy_col: str = "Financial Year", cn_col: str = "CN ID") -> pd.DataFrame:
    """Set every notice's financial year to its family's year.

    The original notice decides. If the original has no financial year, the family
    has none, even when its amendments do. Only a family with no original in the
    data falls back to its earliest notice. The notice's own year is kept in
    'Notice Financial Year' for audit.
    """
    out = add_family_fields(df, cn_col)
    if "Notice Financial Year" not in out.columns:
        out["Notice Financial Year"] = out[fy_col].fillna("").astype(str)
    version = (
        out[cn_col].astype("string").str.extract(r"-A(\d+)$", expand=False).fillna("0").astype(int).to_numpy()
    )
    ordered = out.assign(_version=version)[[FAMILY_COL, "_version", fy_col]].sort_values(
        "_version", kind="stable"
    )
    first = ordered.groupby(FAMILY_COL, sort=False).head(1).set_index(FAMILY_COL)[fy_col]
    out[fy_col] = out[FAMILY_COL].map(first).fillna("").astype(str)
    return out
