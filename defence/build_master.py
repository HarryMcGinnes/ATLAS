from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

# =============================================================================
# ATLAS DEFENCE GOLD ENRICHMENT - SILVER INPUT
# =============================================================================
# Design principle:
#   * Consume the wide Defence Silver parquet produced by filter_defence.py.
#   * Preserve every incoming Silver/source column.
#   * Inherit shared supplier identity; never regroup suppliers here.
#   * Add Defence-specific organisation, domain, TAM and RP/RE enrichment only.
#   * Addressability is decided in this order:
#       1) Category Type
#       2) Category
#       3) Description
#       4) Winning supplier as weak supporting context only
#
# The dashboards continue to receive:
#   master_output/master_defence_contracts.parquet
# with their expected derived fields.
# =============================================================================

# -------------------------------
# Raw source headers - DO NOT RENAME
# -------------------------------
CN_ID = "CN ID"
AGENCY = "Agency"
CONTRACT_TYPE = "Contract Type"
FINANCIAL_YEAR = "Financial Year"
DESCRIPTION = "Description"
CATEGORY_TYPE = "Category Type"
CATEGORY_CODE = "Category Code"
CATEGORY = "Category"
SUPPLIER_NAME = "Supplier Name"
SUPPLIER_ABN = "Supplier ABN"
RAW_DIVISION = "Agency Division"
RAW_BRANCH = "Agency Branch"
VALUE = "Value"
VALUE_PER_YEAR = "Value Per Year"
START_DATE = "Start Date"
END_DATE = "End Date"
SOURCE_SUPPLIER_DISPLAY = "supplier_display"
SOURCE_DIV_CLEAN = "division_group"
SOURCE_BRANCH_CLEAN = "branch_group"

# Derived compatibility fields used by existing dashboards.
CANON_DIVISION = "defence_division_group"
CANON_BRANCH = "defence_branch_group"

FINAL_SERVICE_OFFERINGS = [
    "Strategy, Transformation & Advisory",
    "SI & Engineering",
    "Data, AI & Automation",
    "Cloud Infrastructure & Cyber",
    "Managed Services & Operations",
]

DEFENCE_AGENCIES = {
    "department of defence",
    "australian signals directorate",
    "australian submarine agency",
}

MANUAL_NON_ADDRESSABLE_CN_IDS = {
    "CN3840723", "CN4050683", "CN359557", "CN2953962", "CN1384831",
    "CN3486107", "CN3658983", "CN3667078", "CN3296931", "CN1926632",
    "CN4179707",

}

AUTHORITATIVE_SERVICE_OFFERING_OVERRIDES = {
    "CN4066214": "Data, AI & Automation",
}

# -----------------------------------------------------------------------------
# Text helpers
# -----------------------------------------------------------------------------
def clean(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9+#./ -]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def upper_cn(value: object) -> str:
    return str(value or "").strip().upper()


def bool_series(series: pd.Series) -> pd.Series:
