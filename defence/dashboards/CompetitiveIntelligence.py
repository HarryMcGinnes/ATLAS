"""
Build a Competitive Positioning dashboard inside Accenture's Defence TAM.

Supplier-vs-supplier baseline:
- Accenture is always purple across all charts; comparison suppliers are orange.
- If comparing two non-Accenture suppliers, colours are blue and orange for contrast.

Purpose
-------
This dashboard answers:
  "How does Supplier 1 compare with Supplier 2 across Accenture-addressable Defence service offerings?"

It uses the same market definition as the Accenture TAM dashboard:
  1. Load AusTender data or an existing classified CSV
  2. Filter to Defence-related records by default
  3. Restrict market analysis to contracts classified as Accenture-addressable
  4. Compare Supplier 1 vs Supplier 2 across service offering, market share, contracts and Defence domains
  5. Recalculate Service Offering leaderboards for All FY, Last 5 FY, Last 3 FY and Current FY

Outputs
-------
- CompareDashboard.html
- supplier_summary.csv
- supplier_capability.csv
- supplier_capability_yearly.csv
- top_contracts.csv
- defence_scope_audit_by_agency.csv

Usage
-----
  python CompareDashboard.py --input "master_classification_output/master_defence_contracts.parquet"
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
import plotly.io as pio


VALUE_COL = "Value"
ANNUALISED_VALUE_COL = "Value Per Year"
SUPPLIER_COL = "Supplier Name"
SUPPLIER_ABN_COL = "Supplier ABN"
DESCRIPTION_COL = "Description"
CATEGORY_COL = "Category"
CATEGORY_TYPE_COL = "Category Type"
FIN_YEAR_COL = "Financial Year"
CN_ID_COL = "CN ID"

ACTIVE_VALUE_COL = "Current Active Contract Value"
REMAINING_VALUE_COL = "Estimated Remaining Contract Value"
CURRENT_RUN_RATE_COL = "Estimated Annual Run Rate"
DISPLAY_START_DATE_COL = "_contract_start_date_display"
DISPLAY_END_DATE_COL = "_contract_end_date_display"
EXTENSION_COUNT_COL = "_extension_count"
AGENCY_COL = "Agency"
AGENCY_DIVISION_COL = "Agency Division"
AGENCY_BRANCH_COL = "Agency Branch"

DEFAULT_FOCUS_SUPPLIERS = [
    "Accenture",
    "KPMG",
    "Deloitte",
    "EY",
    "PwC",
    "Fujitsu",
    "Leidos",
]

REQUIRED_CLASSIFICATION_COLS = [
    "capability",
    "is_addressable",
    "supplier_group",
    "is_accenture",
]

DEFENCE_AGENCY_TERMS = [
    "department of defence",
    "australian signals directorate",
    "australian geospatial",
    "defence intelligence",
    "defence housing australia",
]

DEFENCE_ORG_CODES = {
    "ADHQ", "AIRF", "ARMY", "ASA", "ASD", "ASSEC", "CASG", "CDG", "CFO", "CIOG",
    "DDG", "DFG", "DIG", "DIO", "DPG", "DSTG", "GWEO", "JCG", "JOC", "NAVY",
    "NPS", "NSSG", "SEG", "SP&I", "SPI", "VCDF",
}

DEFENCE_TEXT_TERMS = [
    "air force", "army", "navy", "naval", "maritime", "casg", "ciog", "ddg",
    "joint capabilities", "aerospace systems", "land systems", "maritime systems",
    "defence digital", "defence intelligence", "defence science", "guided weapons",
    "capability acquisition", "security and estate", "defence people", "defence finance",
]



MASTER_REQUIRED_COLUMNS = {
    "capability",
    "is_addressable",
    "supplier_group",
    "is_accenture",
    "defence_domain",
    "ReinventionPartner",
    "ReinventionEngine",
    "Value",
}

SEVEN_DOMAINS = {
    "Maritime",
    "Land",
    "Air",
    "Joint",
    "Cyber",
    "Space",
    "Capability Enabler",
    "Unmapped",
}

STRICT_DEFENCE_AGENCIES = {
    "department of defence",
    "australian signals directorate",
    "australian submarine agency",
}


def _normalise_dashboard_agency(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).replace("\u00a0", " ").strip().lower())


def _parse_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])




def use_master_supplier_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Use the canonical supplier identity written by build_master.py.

    Dashboard code must never rebuild, alias-match or fuzzy-match suppliers.
    Every supplier filter, grouping, ranking and KPI uses supplier_group exactly
    as stored in the canonical master parquet. Raw Supplier Name is display/audit
    evidence only.
    """
    required = {"supplier_group", "is_accenture"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(
            "Canonical master dataset is missing supplier identity columns: "
            + ", ".join(missing)
            + ". Rebuild the master dataset with build_master.py."
        )

    out = df.copy()
    out["supplier_group"] = (
        out["supplier_group"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .replace("", "Unknown")
    )
    out["is_accenture"] = _parse_bool_series(out["is_accenture"])

    inconsistent = out["is_accenture"].ne(out["supplier_group"].eq("Accenture"))
    if inconsistent.any():
        sample = out.loc[inconsistent, ["supplier_group", "is_accenture"]].head(10)
        raise SystemExit(
            "Canonical supplier identity is inconsistent: is_accenture does not "
            "match supplier_group == 'Accenture'. Rebuild build_master.py. Sample:\n"
            + sample.to_string(index=False)
        )
    return out


def load_master_dataset(path: Path) -> pd.DataFrame:
    """Load the canonical master Defence parquet without reclassification.

    The dashboard layer is intentionally read-only with respect to capability,
    addressability, supplier grouping and Defence-domain mapping. Those fields
    are created only by build_master.py.
    """
    if not path.exists():
        raise SystemExit(f"Master Defence parquet not found: {path}")
    if path.suffix.lower() not in {".parquet", ".pq"}:
        raise SystemExit(
            "Dashboard input must be the canonical master Defence parquet, not "
            "a raw parquet or classified CSV. Expected master_defence_contracts.parquet."
        )

    df = pd.read_parquet(path)
    duplicate_columns = df.columns[df.columns.duplicated()].tolist()
    if duplicate_columns:
        raise SystemExit(
            "Master parquet contains duplicate column names: "
            + ", ".join(sorted(set(duplicate_columns)))
        )

    missing = sorted(MASTER_REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise SystemExit(
            "Input is not a valid canonical master Defence dataset. Missing columns: "
            + ", ".join(missing)
            + ". Run build_master.py first."
        )

    if AGENCY_COL not in df.columns:
        raise SystemExit(
            f"Canonical master dataset is missing the mandatory procurer field: {AGENCY_COL}"
        )
    agency_norm = df[AGENCY_COL].map(_normalise_dashboard_agency)
    invalid_agencies = sorted(
        df.loc[~agency_norm.isin(STRICT_DEFENCE_AGENCIES), AGENCY_COL]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    if invalid_agencies:
        raise SystemExit(
            "Canonical master dataset contains non-Defence procurers and must be rebuilt. "
            "Allowed agencies are Department of Defence, Australian Signals Directorate, "
            "and Australian Submarine Agency. Unexpected agencies include: "
            + "; ".join(invalid_agencies[:20])
        )

    out = df.copy()
    out["is_addressable"] = _parse_bool_series(out["is_addressable"])
    out["is_accenture"] = _parse_bool_series(out["is_accenture"])
    if "is_market_relevant" in out.columns:
        out["is_market_relevant"] = _parse_bool_series(out["is_market_relevant"])
    if "is_defence_scope" in out.columns:
        out["is_defence_scope"] = _parse_bool_series(out["is_defence_scope"])
        out = out[out["is_defence_scope"]].copy()

    out["capability"] = out["capability"].fillna("Unclassified").astype(str)
    out["ReinventionPartner"] = out["ReinventionPartner"].fillna("Unclassified").replace("", "Unclassified").astype(str)
    out["ReinventionEngine"] = out["ReinventionEngine"].fillna("Unclassified").replace("", "Unclassified").astype(str)
    out["supplier_group"] = out["supplier_group"].fillna("Unknown").replace("", "Unknown").astype(str)
    out["defence_domain"] = out["defence_domain"].fillna("Unmapped").replace("", "Unmapped").astype(str)

    invalid_domains = sorted(set(out["defence_domain"]) - SEVEN_DOMAINS)
    if invalid_domains:
        raise SystemExit(
            "Master parquet contains unsupported Defence domains: "
            + ", ".join(invalid_domains)
        )

    out["is_competitor_market"] = out["is_addressable"] & ~out["is_accenture"]
    out["is_white_space"] = out["is_competitor_market"]

    print(f"Using canonical master Defence dataset without reclassification: {path}")
    print(f"Master Defence rows loaded: {len(out):,}")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="master_output/master_defence_contracts.parquet")
    parser.add_argument("--output-dir", default="CompareDashboard_Output")
    parser.add_argument(
        "--value-mode",
        choices=["total", "annualised"],
        default="total",
        help="Backward-compatible option. For this current-position dashboard, prefer --current-value-mode.",
    )
    parser.add_argument(
        "--as-at",
        help="Date used to decide which contracts are current/active. Default: today's date. Format: YYYY-MM-DD.",
    )
    parser.add_argument(
        "--current-value-mode",
        choices=["active", "remaining", "annualised"],
        default="annualised",
        help=(
            "active = full value of contracts active as at --as-at; "
            "remaining = pro-rated estimated value remaining from --as-at to contract end; "
            "annualised = annual run-rate using Value Per Year or a derived annualised value. Default is annualised because it best represents current market expenditure."
        ),
    )
    parser.add_argument("--top-n-suppliers", type=int, default=30)
    parser.add_argument("--top-n-contracts", type=int, default=500)
    parser.add_argument("--fy-start", help="Optional first financial year to include, e.g. 2021-2022")
    parser.add_argument("--fy-end", help="Optional final financial year to include, e.g. 2025-2026")
    parser.add_argument(
        "--include-all-agencies",
        action="store_true",
        help="Compatibility option only. The canonical master input is already Defence-only.",
    )
    return parser.parse_args()


def money(value: float) -> str:
    if pd.isna(value):
        return "$0"
    value = float(value)
    sign = "-" if value < 0 else ""
    value = abs(value)
    if value >= 1_000_000_000:
        return f"{sign}${value / 1_000_000_000:,.2f}B"
    if value >= 100_000_000:
        return f"{sign}${value / 1_000_000:,.0f}M"
    if value >= 10_000_000:
        return f"{sign}${value / 1_000_000:,.1f}M"
    if value >= 1_000_000:
        return f"{sign}${value / 1_000_000:,.2f}M"
    if value >= 1_000:
        return f"{sign}${value / 1_000:,.0f}K"
    return f"{sign}${value:,.0f}"


def pct_value(value: float) -> str:
    if pd.isna(value):
        return "0.0%"
    return f"{float(value):,.1f}%"


def _norm_code(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper().replace(" ", "")


def _contains_defence_text(value: object) -> bool:
    text = str(value or "").lower()
    return any(term in text for term in DEFENCE_TEXT_TERMS)


# Canonical supplier identities. ABN is authoritative when available; cleaned
# supplier-name aliases are the fallback. Multiple legal entities may therefore
# roll into one executive supplier group.
SUPPLIER_GROUP_BY_ABN = {
    "49096776895": "Accenture",
    "74490121060": "Deloitte",
    "51194660183": "KPMG",
    "75288172749": "EY",
    "20607773295": "PwC",
    "79612590155": "Leidos",
    "18008476944": "DXC",
    "79000024733": "IBM",
    "80003074468": "Oracle",
    "29008423005": "BAE",
    "51006870846": "BAE",
    "66008642751": "Thales",
    "19001011427": "Fujitsu",
    "31010545267": "Fujitsu",
    "30008425509": "Lockheed",
    "35063709295": "Raytheon",
    "64006678119": "Boeing",
    "12079749287": "Jacobs",
    "54005139873": "Aurecon",
    "33051775556": "Telstra",
    "64086174781": "Telstra",
    "53000983700": "Downer",
    "34129384032": "Downer",
    "39613308008": "Nova Defence",
    "33163511304": "Nova Defence",
    "91007660317": "Kellogg Brown & Root",
    "90155020303": "Navantia",
    "68125805647": "Qinetiq",
    "64008605034": "ASC",
    "31143526229": "Elbit Systems",
    "97072941943": "Kinetic IT",
    "88122798207": "Cubic Defence",
    "99059951183": "CEA Technologies",
    "65119369827": "Synergy Group",
    "51106981560": "SME Gateway",
    "46003855561": "Dell",
    "57195873179": "University of New South Wales",
    "56609386156": "Servegate Australia",
    "95820883147": "Projects Assured",
    "52119884945": "Whizdom",
}


def _normalise_supplier_abn(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = re.sub(r"[^0-9]", "", str(value))
    return text.lstrip("0") or ""


def _normalise_supplier_name(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9+#]+", " ", text)
    text = re.sub(
        r"\b(?:pty|proprietary|limited|ltd|incorporated|inc|holdings|group|"
        r"australia|australian|asia pacific|apac|services|service)\b",
        " ",
        text,
    )
    return re.sub(r"\s+", " ", text).strip()


SUPPLIER_NAME_RULES = [
    (r"\baccenture\b", "Accenture"),
    (r"\bdeloitte\b", "Deloitte"),
    (r"\bkpmg\b", "KPMG"),
    (r"\b(?:pricewaterhousecoopers|price waterhouse coopers|pwc)\b", "PwC"),
    (r"\b(?:ernst and young|ernst young)\b", "EY"),
    (r"\bleidos\b", "Leidos"),
    (r"\bdxc\b", "DXC"),
    (r"\b(?:international business machines|ibm)\b", "IBM"),
    (r"\bmicrosoft\b", "Microsoft"),
    (r"\b(?:amazon web services|amazon|aws)\b", "Amazon / AWS"),
    (r"\boracle\b", "Oracle"),
    (r"\bsap\b", "SAP"),
    (r"\bdata\s*#?\s*3\b", "Data#3"),
    (r"\bdatacom\b", "Datacom"),
    (r"\bbae systems?\b", "BAE"),
    (r"\bthales\b", "Thales"),
    (r"\bfujitsu\b", "Fujitsu"),
    (r"\bsaab\b", "Saab"),
    (r"\bnec\b", "NEC"),
    (r"\blockheed martin\b", "Lockheed"),
    (r"\bnorthrop grumman\b", "Northrop Grumman"),
    (r"\braytheon\b", "Raytheon"),
    (r"\bcapgemini\b", "Capgemini"),
    (r"\b(?:cgi|cgi federal)\b", "CGI"),
    (r"\bbooz allen\b", "Booz Allen"),
    (r"\bjacobs\b", "Jacobs"),
    (r"\baurecon\b", "Aurecon"),
    (r"\bcardno\b", "Cardno"),
    (r"\babt associates\b", "Abt Associates"),
    (r"\bkellogg brown and root\b|\bkbr\b", "Kellogg Brown & Root"),
    (r"\bboeing\b", "Boeing"),
    (r"\btelstra\b", "Telstra"),
    (r"\bdowner\b", "Downer"),
    (r"\bnova defence\b", "Nova Defence"),
    (r"\bnavantia\b", "Navantia"),
    (r"\bqinetiq\b", "Qinetiq"),
    (r"\basc\b", "ASC"),
    (r"\belbit\b", "Elbit Systems"),
    (r"\bkinetic it\b", "Kinetic IT"),
    (r"\bcubic defence\b", "Cubic Defence"),
    (r"\bcea technologies\b", "CEA Technologies"),
    (r"\bsynergy group\b", "Synergy Group"),
    (r"\bdell\b", "Dell"),
]


def supplier_group(value: object, abn: object = "") -> str:
    """Return one canonical supplier group using ABN first, then name aliases."""
    abn_key = _normalise_supplier_abn(abn)
    if abn_key in SUPPLIER_GROUP_BY_ABN:
        return SUPPLIER_GROUP_BY_ABN[abn_key]

    raw = str(value or "").strip()
    normalised = _normalise_supplier_name(value)
    for pattern, label in SUPPLIER_NAME_RULES:
        if re.search(pattern, normalised):
            return label
    return raw.title() if raw else "Unknown"


def canonicalise_supplier_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rebuild supplier_group consistently for every dashboard calculation."""
    out = df.copy()
    raw_names = (
        out[SUPPLIER_COL]
        if SUPPLIER_COL in out.columns
        else out.get("supplier_group", pd.Series("Unknown", index=out.index))
    )
    abns = (
        out[SUPPLIER_ABN_COL]
        if SUPPLIER_ABN_COL in out.columns
        else pd.Series("", index=out.index)
    )
    out["supplier_group"] = [
        supplier_group(name, abn)
        for name, abn in zip(raw_names, abns)
    ]
    out["is_accenture"] = out["supplier_group"].eq("Accenture")
    return out


def filter_defence_scope(df: pd.DataFrame, include_all_agencies: bool = False) -> pd.DataFrame:
    """Return the canonical Defence dataset unchanged.

    Defence scope is established only in build_master.py by
    exact Agency allow-list before domain or capability classification.
    """
    if include_all_agencies:
        print("Warning: --include-all-agencies is ignored for canonical master inputs.")
    return df.copy()


def export_defence_scope_audit(original_df: pd.DataFrame, filtered_df: pd.DataFrame, value_col: str, output_dir: Path) -> None:
    if AGENCY_COL not in original_df.columns:
        return
    original = original_df.copy()
    original["_scope"] = "Excluded"
    original.loc[filtered_df.index.intersection(original.index), "_scope"] = "Included Defence scope"
    (
        original.groupby(["_scope", AGENCY_COL], dropna=False)
        .agg(value=(value_col, "sum"), contracts=(value_col, "size"))
        .reset_index()
        .sort_values(["_scope", "value"], ascending=[True, False])
        .to_csv(output_dir / "defence_scope_audit_by_agency.csv", index=False)
    )


def load_raw_data(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except ImportError as exc:
        raise SystemExit(
            "Unable to read Parquet because no Parquet engine is installed. "
            "Install pyarrow or fastparquet, then rerun: pip install pyarrow"
        ) from exc
    except FileNotFoundError as exc:
        raise SystemExit(f"Input file not found: {path}") from exc


def load_or_classify(args: argparse.Namespace) -> pd.DataFrame:
    return load_master_dataset(Path(args.input))


def get_value_col(df: pd.DataFrame, value_mode: str) -> str:
    if value_mode == "annualised":
        if ANNUALISED_VALUE_COL not in df.columns:
            raise SystemExit(f"--value-mode annualised requested but column is missing: {ANNUALISED_VALUE_COL}")
        return ANNUALISED_VALUE_COL
    return VALUE_COL


def clean_value_columns(df: pd.DataFrame, value_cols: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for col in value_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


START_DATE_CANDIDATES = [
    "Contract Start Date",
    "Start Date",
    "Contract Period Start",
    "Contract Start",
    "StartDate",
]

END_DATE_CANDIDATES = [
    "Contract End Date",
    "End Date",
    "Contract Period End",
    "Contract End",
    "Expiry Date",
    "EndDate",
]

EXTENSION_COUNT_CANDIDATES = [
    "Extension Count",
    "Extensions",
    "Number of Extensions",
    "No. of Extensions",
    "Contract Extensions",
    "Extension Number",
    "Amendment Count",
    "Amendments",
    "Number of Amendments",
    "Variation Count",
    "Variations",
    "Revision Count",
]

AMENDMENT_NUMBER_CANDIDATES = [
    "Amendment Number",
    "Amendment No",
    "Amendment",
    "Variation Number",
    "Variation No",
    "Revision Number",
]


def first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    exact = {str(c).lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in exact:
            return exact[candidate.lower()]
    # Fuzzy fallback: AusTender extracts sometimes vary slightly.
    for c in df.columns:
        lower = str(c).lower()
        if "start" in lower and "date" in lower:
            return c
    return None


def first_existing_end_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    exact = {str(c).lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in exact:
            return exact[candidate.lower()]
    for c in df.columns:
        lower = str(c).lower()
        if "end" in lower and "date" in lower:
            return c
        if "expiry" in lower and "date" in lower:
            return c
    return None


def first_existing_extension_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    exact = {str(c).lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in exact:
            return exact[candidate.lower()]
    for c in df.columns:
        lower = str(c).lower()
        if ("extension" in lower or "amendment" in lower or "variation" in lower or "revision" in lower) and ("count" in lower or "number" in lower or "no" in lower):
            return c
    return None


def add_contract_date_and_extension_display_columns(
    out: pd.DataFrame,
    start: pd.Series,
    end: pd.Series,
) -> pd.DataFrame:
    """Add stable browser/table fields for dates and extension counts.

    End date comes from the same end-date column used to decide whether a contract
    is currently active. Extension count is taken from an explicit extension/
    amendment/variation field where present. If not present, it is inferred as
    duplicate rows per CN ID minus one, which is the best fallback available from
    AusTender-style extracts.
    """
    out = out.copy()
    out[DISPLAY_START_DATE_COL] = start.dt.strftime("%Y-%m-%d").where(start.notna(), "")
    out[DISPLAY_END_DATE_COL] = end.dt.strftime("%Y-%m-%d").where(end.notna(), "")

    explicit_col = first_existing_extension_col(out, EXTENSION_COUNT_CANDIDATES)
    amendment_col = first_existing_extension_col(out, AMENDMENT_NUMBER_CANDIDATES)

    if explicit_col is not None:
        ext = pd.to_numeric(out[explicit_col], errors="coerce").fillna(0)
    elif amendment_col is not None:
        ext = pd.to_numeric(out[amendment_col], errors="coerce").fillna(0)
    elif CN_ID_COL in out.columns:
        ext = out.groupby(CN_ID_COL)[CN_ID_COL].transform("size").sub(1).clip(lower=0)
    else:
        ext = pd.Series(0, index=out.index)

    out[EXTENSION_COUNT_COL] = pd.to_numeric(ext, errors="coerce").fillna(0).clip(lower=0).round().astype(int)
    return out


def parse_as_at(value: str | None) -> pd.Timestamp:
    if value:
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            raise SystemExit(f"Could not parse --as-at date: {value}. Use YYYY-MM-DD.")
        return pd.Timestamp(parsed).normalize()
    return pd.Timestamp.today().normalize()


def add_current_contract_value_columns(df: pd.DataFrame, base_value_col: str, as_at: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, object]]:
    """Keep only contracts active as at a date and add current-value measures.

    Active means: start date is blank or <= as_at, and end date is blank or >= as_at.

    Annual run-rate is derived from each contract's own start/end dates first:
        annual run-rate = total contract value / max(contract duration in years, 1)

    The `max(..., 1)` guard is deliberate. It prevents a short contract from showing
    a value/year greater than its total contract value, which was making the table
    look wrong for contracts such as a $700K contract showing a $900K/year run-rate.

    Value Per Year is used only as a fallback when dates are missing/invalid, and it
    is also capped at the total contract value for display and TAM calculations.
    """
    out = df.copy()
    start_col = first_existing_col(out, START_DATE_CANDIDATES)
    end_col = first_existing_end_col(out, END_DATE_CANDIDATES)
    if end_col is None:
        raise SystemExit(
            "Cannot build a current-position view because no contract end-date column was found. "
            "Expected something like 'Contract End Date' or 'End Date'."
        )

    start = pd.to_datetime(out[start_col], errors="coerce", dayfirst=True) if start_col else pd.NaT
    end = pd.to_datetime(out[end_col], errors="coerce", dayfirst=True)

    if start_col:
        active_start = start.isna() | (start <= as_at)
    else:
        active_start = pd.Series(True, index=out.index)
    active_end = end.isna() | (end >= as_at)
    active = active_start & active_end
    out = out.loc[active].copy()

    start = pd.to_datetime(out[start_col], errors="coerce", dayfirst=True) if start_col else pd.Series(pd.NaT, index=out.index)
    end = pd.to_datetime(out[end_col], errors="coerce", dayfirst=True)
    base = pd.to_numeric(out[base_value_col], errors="coerce").fillna(0).clip(lower=0)

    # Full value of the currently-active contract.
    out[ACTIVE_VALUE_COL] = base

    total_days = (end - start).dt.days
    remaining_days = (end - as_at).dt.days
    remaining_ratio = (remaining_days / total_days).clip(lower=0, upper=1)
    remaining_ratio = remaining_ratio.where(total_days > 0, 1).fillna(1)
    out[REMAINING_VALUE_COL] = base * remaining_ratio

    # Primary annual run-rate: derive from the contract's own duration.
    # Use at least one year so annual run-rate never exceeds total contract value.
    duration_years = (total_days / 365.25).where(total_days > 0)
    duration_years = duration_years.clip(lower=1)
    derived_run_rate = base / duration_years

    # Fallback only when dates are missing/invalid. Cap fallback at total value too.
    if ANNUALISED_VALUE_COL in out.columns:
        fallback_run_rate = pd.to_numeric(out[ANNUALISED_VALUE_COL], errors="coerce")
        fallback_run_rate = fallback_run_rate.where(fallback_run_rate > 0)
        fallback_run_rate = fallback_run_rate.clip(lower=0)
        fallback_run_rate = fallback_run_rate.where(fallback_run_rate <= base, base)
    else:
        fallback_run_rate = pd.Series(pd.NA, index=out.index, dtype="float64")

    run_rate = derived_run_rate.where(derived_run_rate.notna(), fallback_run_rate)
    run_rate = run_rate.fillna(base).clip(lower=0)
    # Final safety cap: no annualised display/TAM value can exceed total contract value.
    out[CURRENT_RUN_RATE_COL] = run_rate.where(run_rate <= base, base)

    out = add_contract_date_and_extension_display_columns(out, start, end)

    metadata = {
        "as_at": as_at.strftime("%Y-%m-%d"),
        "start_col": start_col or "not available",
        "end_col": end_col,
        "active_rows": int(len(out)),
    }
    return out, metadata


def current_value_col_from_mode(mode: str) -> str:
    if mode == "remaining":
        return REMAINING_VALUE_COL
    if mode == "annualised":
        return CURRENT_RUN_RATE_COL
    return ACTIVE_VALUE_COL


def financial_year_start_year(value: object) -> int | None:
    if pd.isna(value):
        return None
    match = re.search(r"(20\d{2}|19\d{2})", str(value).strip())
    if not match:
        return None
    return int(match.group(1))


def format_fy_range_label(fy_start: str | None, fy_end: str | None) -> str:
    if fy_start and fy_end:
        return f"{fy_start} to {fy_end}"
    if fy_start:
        return f"{fy_start} onwards"
    if fy_end:
        return f"up to {fy_end}"
    return "whole dataset period"


def filter_financial_year_range(df: pd.DataFrame, fy_start: str | None, fy_end: str | None) -> pd.DataFrame:
    if not fy_start and not fy_end:
        return df
    if FIN_YEAR_COL not in df.columns:
        raise SystemExit(f"Cannot filter by financial year because column is missing: {FIN_YEAR_COL}")

    start_year = financial_year_start_year(fy_start) if fy_start else None
    end_year = financial_year_start_year(fy_end) if fy_end else None
    if fy_start and start_year is None:
        raise SystemExit(f"Could not parse --fy-start: {fy_start}")
    if fy_end and end_year is None:
        raise SystemExit(f"Could not parse --fy-end: {fy_end}")
    if start_year is not None and end_year is not None and start_year > end_year:
        raise SystemExit("--fy-start must be earlier than or equal to --fy-end")

    fy_year = df[FIN_YEAR_COL].map(financial_year_start_year)
    mask = pd.Series(True, index=df.index)
    if start_year is not None:
        mask &= fy_year >= start_year
    if end_year is not None:
        mask &= fy_year <= end_year
    filtered = df.loc[mask].copy()
    if filtered.empty:
        raise SystemExit("Financial year filter returned no rows.")
    return filtered


def safe_capability(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "capability" not in out.columns:
        out["capability"] = "Unclassified"
    out["capability"] = out["capability"].fillna("Unclassified").replace("", "Unclassified")
    return out


def apply_capability_overrides(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror the canonical build_master high-confidence rules defensively.

    The master parquet remains authoritative. These checks protect the dashboard
    when an older master file is supplied.
    """
    out = df.copy()
    text = _combined_domain_text(out, [DESCRIPTION_COL, CATEGORY_COL, CATEGORY_TYPE_COL])

    def assign(mask: pd.Series, capability: str, detailed: str, reason: str, addressable: bool = True) -> None:
        out.loc[mask, "capability"] = capability
        if "executive_capability" in out.columns:
            out.loc[mask, "executive_capability"] = capability
        if "detailed_capability" in out.columns:
            out.loc[mask, "detailed_capability"] = detailed
        if "classification_reason" in out.columns:
            out.loc[mask, "classification_reason"] = reason
        if "is_addressable" in out.columns:
            out.loc[mask, "is_addressable"] = addressable
        if "addressability" in out.columns:
            out.loc[mask, "addressability"] = "Addressable" if addressable else "Not Addressable"

    crane_mask = text.str.contains(
        r"\bcrane(?:s)?\s+(?:service|services|hire|maintenance|repair|repairs|inspection)\b",
        regex=True, na=False,
    )
    assign(crane_mask, "Non-addressable", "Crane / Physical Lifting Services", "Physical crane service is outside Accenture TAM.", False)

    aircraft_term = text.str.contains(r"\b(?:aircraft|airframe|aerospace)\b", regex=True, na=False)
    industrial = text.str.contains(
        r"\b(?:production|manufacture|manufactured|manufacturer|manufacturers|manufacturing|assembly|fabrication|factory build|component production)\b",
        regex=True, na=False,
    )
    digital_exception = text.str.contains(
        r"\b(?:research and development|research & development|r&d|capability development|systems? engineering|software|digital|ict|cyber|cloud|data|analytics|modelling|modeling|simulation|integration|architecture)\b",
        regex=True, na=False,
    )
    aircraft_industrial = aircraft_term & industrial & ~digital_exception
    assign(aircraft_industrial, "Non-addressable", "Aircraft Production / Manufacturing", "Physical aircraft production/manufacturing is outside Accenture TAM.", False)

    lifecycle = text.str.contains(
        r"\b(?:life\s*cycle|lifecycle|whole[-\s]?of[-\s]?life)\s+(?:cost|costing|cost analysis|cost model(?:ling|ing)?)\b",
        regex=True, na=False,
    )
    assign(lifecycle, "Strategy, Transformation & Advisory", "Life Cycle Cost Analysis", "Lifecycle cost analysis is advisory work.")

    network = text.str.contains(r"\bnetwork(?:ing)?\b", regex=True, na=False)
    network_ops = network & text.str.contains(
        r"\b(?:support|maintenance|maintain|managed|monitoring|operations?|service desk|help desk|sustainment|administration)\b",
        regex=True, na=False,
    )
    assign(network, "Cloud Infrastructure & Cyber", "Network Infrastructure / Engineering", "Network infrastructure maps to Cloud Infrastructure & Cyber.")
    assign(network_ops, "Managed Services & Operations", "Network Support / Operations", "Ongoing network support/maintenance maps to Managed Services & Operations.")

    training_equipment = text.str.contains(
        r"\b(?:system|systems)\s+training\s+(?:equipment|device|devices|system|systems|simulator|simulators)\b",
        regex=True, na=False,
    )
    assign(training_equipment, "SI & Engineering", "Training System Equipment / Integration", "Training-system equipment maps to SI & Engineering.")

    recurring = text.str.contains(
        r"\b(?:renewal|renewals|licence|licences|licensed|licensing|license|licenses|subscription|subscriptions)\b",
        regex=True, na=False,
    )
    assign(recurring, "Managed Services & Operations", "Licence / Subscription / Renewal", "Renewal, licence/license or subscription maps to Managed Services & Operations.")

    return out


def collapse_duplicate_named_column(
    df: pd.DataFrame,
    column_name: str,
    default_value: object | None = None,
) -> pd.DataFrame:
    """Collapse duplicate columns with the same name into one Series.

    Values are coalesced from left to right, taking the first non-null and
    non-blank value on each row. This protects grouping and selection logic
    from merges, renames, or external mapping functions that return duplicate
    column labels.
    """
    positions = [i for i, name in enumerate(df.columns) if name == column_name]
    if len(positions) <= 1:
        return df

    candidates = df.iloc[:, positions].copy()
    candidates = candidates.replace(
        {
            "": pd.NA,
            "nan": pd.NA,
            "NaN": pd.NA,
            "none": pd.NA,
            "None": pd.NA,
            "null": pd.NA,
        }
    )
    collapsed = candidates.bfill(axis=1).iloc[:, 0]
    if default_value is not None:
        collapsed = collapsed.fillna(default_value)

    keep_mask = df.columns != column_name
    out = df.loc[:, keep_mask].copy()
    out[column_name] = collapsed
    return out


def remove_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate column labels while preserving useful domain values."""
    out = df.copy()

    if list(out.columns).count("defence_domain") > 1:
        out = collapse_duplicate_named_column(out, "defence_domain", default_value="Joint")

    if out.columns.duplicated().any():
        duplicate_names = sorted(
            set(out.columns[out.columns.duplicated(keep=False)].astype(str))
        )
        print(
            "Warning: duplicate columns detected and de-duplicated:",
            ", ".join(duplicate_names),
        )
        out = out.loc[:, ~out.columns.duplicated(keep="first")].copy()

    return out


def _combined_domain_text(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Combine available mapping fields into normalised lower-case text."""
    combined = pd.Series("", index=df.index, dtype="string")
    for column in columns:
        if column in df.columns:
            combined = combined.str.cat(
                df[column].fillna("").astype(str),
                sep=" ",
                na_rep="",
            )
    return (
        combined
        .str.lower()
        .str.replace(r"[^a-z0-9]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def apply_service_domain_overrides(df: pd.DataFrame) -> pd.DataFrame:
    """Apply strong service ownership rules to Defence domains.

    Division and branch ownership are stronger evidence than generic contract
    descriptions. For example, Army Aviation remains Land because the procuring
    organisation is Army, despite the word "aviation".
    """
    out = df.copy()

    organisation_text = _combined_domain_text(
        out,
        [
            AGENCY_DIVISION_COL,
            AGENCY_BRANCH_COL,
            "div_clean",
            "branch_clean",
            AGENCY_COL,
        ],
    )

    description_text = _combined_domain_text(
        out,
        [DESCRIPTION_COL, CATEGORY_COL, CATEGORY_TYPE_COL],
    )

    maritime_org = organisation_text.str.contains(
        r"\b(?:navy|naval|maritime|fleet|submarine|chief of navy|navy headquarters|"
        r"maritime systems|nssg|naval shipbuilding|asa)\b",
        regex=True,
        na=False,
    )
    land_org = organisation_text.str.contains(
        r"\b(?:army|land systems|land capability|army headquarters|chief of army|"
        r"forces command|special operations command|infantry|armoured|armored|"
        r"artillery|army aviation)\b",
        regex=True,
        na=False,
    )
    air_org = organisation_text.str.contains(
        r"\b(?:air force|airforce|raaf|air force headquarters|air force executive|"
        r"chief of air force|air combat|air mobility|air warfare|aerospace systems)\b",
        regex=True,
        na=False,
    )

    # Organisational ownership wins. Apply Air first, then Land so Army Aviation
    # cannot be pulled into Air by the word "aviation".
    out.loc[air_org, "defence_domain"] = "Air"
    out.loc[maritime_org, "defence_domain"] = "Maritime"
    out.loc[land_org, "defence_domain"] = "Land"

    unresolved = out["defence_domain"].isin(["Joint", "", "Unknown", "Other"])

    maritime_desc = description_text.str.contains(
        r"\b(?:naval|maritime|warship|shipbuilding|submarine|surface fleet|frigate)\b",
        regex=True,
        na=False,
    )
    land_desc = description_text.str.contains(
        r"\b(?:soldier|infantry|armoured|armored|artillery|land combat|land vehicle)\b",
        regex=True,
        na=False,
    )
    air_desc = description_text.str.contains(
        r"\b(?:raaf|air force|aircraft|air combat|air mobility|airlift|aerospace)\b",
        regex=True,
        na=False,
    )

    # Description is used only when organisational ownership is not specific.
    out.loc[unresolved & maritime_desc, "defence_domain"] = "Maritime"
    out.loc[unresolved & land_desc, "defence_domain"] = "Land"
    out.loc[unresolved & air_desc, "defence_domain"] = "Air"

    return out


def ensure_defence_domain_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and preserve the canonical seven-domain mapping."""
    out = remove_duplicate_columns(df)
    if "defence_domain" not in out.columns:
        raise SystemExit("Canonical master dataset is missing defence_domain.")
    out["defence_domain"] = out["defence_domain"].fillna("Unmapped").replace("", "Unmapped").astype(str)
    invalid = sorted(set(out["defence_domain"]) - SEVEN_DOMAINS)
    if invalid:
        raise SystemExit("Unsupported Defence domains in master dataset: " + ", ".join(invalid))
    counts = out["defence_domain"].value_counts(dropna=False)
    print("Canonical Defence domain counts:")
    for domain in ["Maritime", "Land", "Air", "Joint", "Cyber", "Space", "Capability Enabler", "Unmapped"]:
        print(f"  {domain}: {int(counts.get(domain, 0)):,}")
    return out


DOMAIN_COLOURS = {
    "Maritime": "#1F77B4",
    "Land": "#D62728",
    "Air": "#17BECF",
    "Space": "#111827",
    "Cyber": "#7F3C8D",
    "Joint": "#9467BD",
    "Capability Enabler": "#8C8C8C",
    "Unmapped": "#D0D5DD",
}

def classify_defence_domain(row: pd.Series) -> str:
    """Map a contract to one of four executive Defence domains.

    The output is deliberately limited to Maritime, Land, Air or Joint.
    Anything not clearly service-specific is treated as Joint.
    """
    parts = []
    for col in [AGENCY_BRANCH_COL, AGENCY_DIVISION_COL, AGENCY_COL, DESCRIPTION_COL, CATEGORY_COL, CATEGORY_TYPE_COL]:
        if col in row.index and pd.notna(row.get(col)):
            parts.append(str(row.get(col)))
    text = " ".join(parts).lower()

    if any(term in text for term in ["navy", "naval", "maritime", "ship", "submarine", "fleet"]):
        return "Maritime"
    if any(term in text for term in ["army", "land ", " land", "soldier", "infantry", "armoured", "armored"]):
        return "Land"
    if any(term in text for term in ["air force", "raaf", "aerospace", "aviation", "aircraft", "air combat", "air lift", "airlift"]):
        return "Air"
    return "Joint"

def prepare_competitive_model(
    df: pd.DataFrame,
    value_col: str,
    top_n_suppliers: int,
) -> dict[str, object]:
    data = remove_duplicate_columns(df)
    data = use_master_supplier_columns(data)
    data = safe_capability(data)

    # The canonical master parquet is the single source of truth for capability
    # and addressability. Do not reclassify or reintroduce excluded contracts in
    # the dashboard layer.
    data = data.loc[_parse_bool_series(data["is_addressable"])].copy()

    if data.empty:
        raise SystemExit("No addressable Defence contracts found after filters.")

    if value_col not in data.columns:
        raise SystemExit(f"Selected value column is missing: {value_col}")

    data[value_col] = pd.to_numeric(data[value_col], errors="coerce").fillna(0)
    if VALUE_COL in data.columns:
        data[VALUE_COL] = pd.to_numeric(data[VALUE_COL], errors="coerce").fillna(0)
    else:
        data[VALUE_COL] = data[value_col]
    data["supplier_group"] = data["supplier_group"].fillna("Unknown").replace("", "Unknown")
    if FIN_YEAR_COL not in data.columns:
        data[FIN_YEAR_COL] = "Unknown"
    data[FIN_YEAR_COL] = data[FIN_YEAR_COL].astype(str)
    data["_fy_start_year"] = data[FIN_YEAR_COL].map(financial_year_start_year)

    total_tam = float(data[value_col].sum())
    print(f"Compare dashboard canonical TAM: ${total_tam/1e9:,.2f}B")
    print("TAM definition: canonical master rows where is_addressable == True")
    capability_total = data.groupby("capability", dropna=False)[value_col].sum().rename("capability_market").reset_index()
    supplier_total = data.groupby("supplier_group", dropna=False)[value_col].sum().rename("supplier_addressable_value").reset_index()
    supplier_total["tam_share_pct"] = supplier_total["supplier_addressable_value"] / total_tam * 100 if total_tam else 0
    supplier_total["rank_in_tam"] = supplier_total["supplier_addressable_value"].rank(method="min", ascending=False).astype(int)

    # Supplier dropdown: keep the main competitors at the top, then append the
    # top-N suppliers by addressable value. Focus suppliers that are also in the
    # top-N are de-duplicated, so the list remains compact and predictable.
    top_suppliers = (
        supplier_total
        .sort_values("supplier_addressable_value", ascending=False)
        .head(top_n_suppliers)["supplier_group"]
        .tolist()
    )
    available_suppliers = set(supplier_total["supplier_group"].astype(str))
    focus_suppliers = [s for s in DEFAULT_FOCUS_SUPPLIERS if s in available_suppliers]
    supplier_options = focus_suppliers + [s for s in top_suppliers if s not in focus_suppliers]
    if "Accenture" in supplier_total["supplier_group"].values and "Accenture" not in supplier_options:
        supplier_options.insert(0, "Accenture")
    if not supplier_options:
        supplier_options = top_suppliers

    supplier_cap = (
        data.groupby(["supplier_group", "capability"], dropna=False)
        .agg(
            supplier_capability_value=(value_col, "sum"),
            contracts=(CN_ID_COL, "nunique") if CN_ID_COL in data.columns else (value_col, "size"),
        )
        .reset_index()
        .merge(capability_total, on="capability", how="left")
    )
    supplier_cap["capability_share_pct"] = supplier_cap["supplier_capability_value"] / supplier_cap["capability_market"].replace(0, pd.NA) * 100
    supplier_cap["capability_rank"] = supplier_cap.groupby("capability")["supplier_capability_value"].rank(method="min", ascending=False).fillna(0).astype(int)

    # Fill zero rows for selected suppliers and every capability so charts are consistent.
    capabilities = capability_total.sort_values("capability_market", ascending=False)["capability"].tolist()
    grid = pd.MultiIndex.from_product([supplier_options, capabilities], names=["supplier_group", "capability"]).to_frame(index=False)
    supplier_cap_full = grid.merge(supplier_cap, on=["supplier_group", "capability"], how="left").merge(
        capability_total, on="capability", how="left", suffixes=("", "_total")
    )
    if "capability_market_total" in supplier_cap_full.columns:
        supplier_cap_full["capability_market"] = supplier_cap_full["capability_market"].fillna(supplier_cap_full["capability_market_total"])
        supplier_cap_full = supplier_cap_full.drop(columns=["capability_market_total"])
    supplier_cap_full["supplier_capability_value"] = supplier_cap_full["supplier_capability_value"].fillna(0)
    supplier_cap_full["contracts"] = supplier_cap_full["contracts"].fillna(0).astype(int)
    supplier_cap_full["capability_share_pct"] = supplier_cap_full["supplier_capability_value"] / supplier_cap_full["capability_market"].replace(0, pd.NA) * 100

    ranks = supplier_cap[["supplier_group", "capability", "capability_rank"]]
    supplier_cap_full = supplier_cap_full.drop(columns=["capability_rank"], errors="ignore").merge(ranks, on=["supplier_group", "capability"], how="left")
    supplier_cap_full["capability_rank"] = supplier_cap_full["capability_rank"].fillna(0).astype(int)

    # Supplier KPI summary.
    rows = []
    for supplier in supplier_options:
        sdata = data[data["supplier_group"].eq(supplier)]
        total_awards = float(sdata[value_col].sum())
        share = total_awards / total_tam * 100 if total_tam else 0
        rank = int(supplier_total.loc[supplier_total["supplier_group"].eq(supplier), "rank_in_tam"].iloc[0]) if not supplier_total.loc[supplier_total["supplier_group"].eq(supplier)].empty else 0
        cap_values = sdata.groupby("capability")[value_col].sum().sort_values(ascending=False)
        largest_cap = cap_values.index[0] if not cap_values.empty else "n/a"
        largest_cap_value = float(cap_values.iloc[0]) if not cap_values.empty else 0
        cap_count = int((cap_values > 0).sum())
        concentration = largest_cap_value / total_awards * 100 if total_awards else 0
        contracts = int(sdata[CN_ID_COL].nunique()) if CN_ID_COL in sdata.columns else int(len(sdata))

        fastest_cap = "n/a"
        fastest_growth = None
        if not sdata.empty and sdata["_fy_start_year"].notna().any():
            yearly = (
                sdata.dropna(subset=["_fy_start_year"])
                .groupby(["capability", "_fy_start_year"], dropna=False)[value_col]
                .sum()
                .reset_index()
            )
            candidates = []
            for cap, g in yearly.groupby("capability"):
                g = g.sort_values("_fy_start_year")
                nonzero = g[g[value_col] > 0]
                if len(nonzero) >= 2:
                    first = float(nonzero[value_col].iloc[0])
                    last = float(nonzero[value_col].iloc[-1])
                    years = max(1, int(nonzero["_fy_start_year"].iloc[-1] - nonzero["_fy_start_year"].iloc[0]))
                    if first > 0 and last > 0:
                        cagr = ((last / first) ** (1 / years) - 1) * 100
                        candidates.append((cap, cagr))
            if candidates:
                fastest_cap, fastest_growth = max(candidates, key=lambda x: x[1])

        rows.append({
            "supplier_group": supplier,
            "total_awards": total_awards,
            "addressable_awards": total_awards,
            "tam_share_pct": share,
            "rank_in_tam": rank,
            "largest_capability": largest_cap,
            "largest_capability_value": largest_cap_value,
            "capability_count": cap_count,
            "concentration_pct": concentration,
            "contracts": contracts,
            "fastest_growing_capability": fastest_cap,
            "fastest_growth_pct": fastest_growth,
        })
    supplier_summary = pd.DataFrame(rows).sort_values("addressable_awards", ascending=False)

    yearly = (
        data
        .groupby(["supplier_group", "capability", FIN_YEAR_COL], dropna=False)
        .agg(value=(value_col, "sum"), contracts=(CN_ID_COL, "nunique") if CN_ID_COL in data.columns else (value_col, "size"))
        .reset_index()
    )
    cap_year_total = (
        data.groupby(["capability", FIN_YEAR_COL], dropna=False)[value_col]
        .sum()
        .rename("capability_year_market")
        .reset_index()
    )
    yearly = yearly.merge(cap_year_total, on=["capability", FIN_YEAR_COL], how="left")
    yearly["share_pct"] = yearly["value"] / yearly["capability_year_market"].replace(0, pd.NA) * 100
    yearly["_fy_start_year"] = yearly[FIN_YEAR_COL].map(financial_year_start_year)
    yearly = yearly.sort_values(["supplier_group", "capability", "_fy_start_year", FIN_YEAR_COL])

    # Keep both the current dashboard value column and the original total contract value.
    # When value_col is the annual run-rate, the table still needs VALUE_COL so
    # "Total contract value" does not render as $0 in the browser.
    top_contract_cols = [
        "supplier_group", SUPPLIER_COL, SUPPLIER_ABN_COL, CN_ID_COL, DESCRIPTION_COL, "capability", "ReinventionPartner", "ReinventionEngine", "detailed_capability",
        "addressability", FIN_YEAR_COL, AGENCY_COL, AGENCY_DIVISION_COL, AGENCY_BRANCH_COL,
        "defence_domain", "domain_confidence", "domain_mapping_reason",
        DISPLAY_START_DATE_COL, DISPLAY_END_DATE_COL, EXTENSION_COUNT_COL,
        value_col, VALUE_COL, ACTIVE_VALUE_COL, REMAINING_VALUE_COL, CURRENT_RUN_RATE_COL,
    ]
    top_contract_cols = list(dict.fromkeys([c for c in top_contract_cols if c in data.columns]))
    top_contracts = data[data["supplier_group"].isin(supplier_options)].sort_values(value_col, ascending=False)[top_contract_cols]

    if not isinstance(data["defence_domain"], pd.Series):
        raise RuntimeError(
            "defence_domain is still not one-dimensional after cleanup. "
            "Check add_defence_domain_columns() for duplicate output labels."
        )

    domain_summary = (
        data.loc[data["supplier_group"].isin(supplier_options)]
        .groupby(["supplier_group", "defence_domain"], dropna=False)
        .agg(
            domain_value=(value_col, "sum"),
            total_contract_value=(VALUE_COL, "sum"),
            contracts=(CN_ID_COL, "nunique") if CN_ID_COL in data.columns else (value_col, "size"),
        )
        .reset_index()
        .sort_values(["supplier_group", "domain_value"], ascending=[True, False])
    )

    domain_yearly = (
        data.loc[data["supplier_group"].isin(supplier_options)]
        .groupby(["supplier_group", "defence_domain", FIN_YEAR_COL], dropna=False)
        .agg(
            domain_value=(value_col, "sum"),
            total_contract_value=(VALUE_COL, "sum"),
            contracts=(CN_ID_COL, "nunique") if CN_ID_COL in data.columns else (value_col, "size"),
        )
        .reset_index()
    )
    domain_yearly["_fy_start_year"] = domain_yearly[FIN_YEAR_COL].map(financial_year_start_year)

    return {
        "data": data,
        "total_tam": total_tam,
        "capability_total": capability_total,
        "supplier_total": supplier_total,
        "supplier_summary": supplier_summary,
        "supplier_capability": supplier_cap_full,
        "supplier_capability_yearly": yearly,
        "top_contracts": top_contracts,
        "domain_summary": domain_summary,
        "domain_yearly": domain_yearly,
        "supplier_options": supplier_options,
        "capabilities": capabilities,
    }



SUPPLIER_COLOUR_MAP = {
    "Accenture": "#A100FF",
    "Deloitte": "#86BC25",
    "KPMG": "#00338D",
    "EY": "#FFE600",
    "PwC": "#E0301E",
    "Leidos": "#850F88",
    "Boeing": "#1A409F",
    "Lockheed": "#003087",
    "IBM": "#1F70C1",
    "Oracle": "#C74634",
    "Fujitsu": "#D6001C",
    "Aurecon": "#2E7D32",
    "DXC": "#5B2C83",
    "Microsoft": "#737373",
    "Amazon / AWS": "#FF9900",
    "SAP": "#0FAAFF",
    "Data#3": "#34495E",
    "BAE": "#2F5597",
    "Thales": "#005EB8",
    "Raytheon": "#B22222",
    "Northrop Grumman": "#3A5F8A",
    "Saab": "#005596",
    "NEC": "#C8102E",
    "Datacom": "#00AEEF",
}


def supplier_colour(value: object) -> str:
    """Return the designated colour for a canonical supplier_group label."""
    supplier = str(value or "").strip()
    if supplier in SUPPLIER_COLOUR_MAP:
        return SUPPLIER_COLOUR_MAP[supplier]

    text = supplier.lower()
    if text == "ey":
        return "#FFE600"

    fallback_rules = [
        ("accenture", "#A100FF"),
        ("deloitte", "#86BC25"),
        ("kpmg", "#00338D"),
        ("ernst", "#FFE600"),
        ("pricewaterhouse", "#E0301E"),
        ("pwc", "#E0301E"),
        ("leidos", "#850F88"),
        ("boeing", "#1A409F"),
        ("lockheed", "#003087"),
        ("fujitsu", "#D6001C"),
        ("raytheon", "#B22222"),
        ("northrop", "#3A5F8A"),
        ("amazon", "#FF9900"),
        ("aws", "#FF9900"),
        ("data#3", "#34495E"),
        ("data 3", "#34495E"),
    ]
    for token, colour in fallback_rules:
        if token in text:
            return colour
    return "#5B6B7A"


def records_for_json(df: pd.DataFrame, columns: list[str]) -> list[dict[str, object]]:
    """Convert selected existing columns to JSON-safe record dictionaries."""
    existing = [column for column in columns if column in df.columns]
    frame = df.loc[:, existing].copy()
    frame = frame.astype(object).where(pd.notna(frame), None)
    return frame.to_dict(orient="records")


def build_html(
    model: dict[str, object],
    value_col: str,
    timeline_value_col: str,
    output_html: Path,
    value_mode: str,
    period_label: str,
    as_at_date: str,
) -> None:
    supplier_summary: pd.DataFrame = model["supplier_summary"]  # type: ignore[assignment]
    supplier_capability: pd.DataFrame = model["supplier_capability"]  # type: ignore[assignment]
    yearly: pd.DataFrame = model["supplier_capability_yearly"]  # type: ignore[assignment]
    top_contracts: pd.DataFrame = model["top_contracts"]  # type: ignore[assignment]
    timeline_contracts: pd.DataFrame = model["timeline_contracts"]  # type: ignore[assignment]
    domain_summary: pd.DataFrame = model["domain_summary"]  # type: ignore[assignment]
    domain_yearly: pd.DataFrame = model["domain_yearly"]  # type: ignore[assignment]
    supplier_options: list[str] = model["supplier_options"]  # type: ignore[assignment]
    capabilities: list[str] = model["capabilities"]  # type: ignore[assignment]
    total_tam: float = float(model["total_tam"])

    if not supplier_options:
        raise SystemExit("No suppliers available for dashboard.")

    default_supplier = "Accenture" if "Accenture" in supplier_options else supplier_options[0]
    default_compare = next((s for s in supplier_options if s != default_supplier), default_supplier)

    summary_payload = records_for_json(supplier_summary, [
        "supplier_group", "total_awards", "addressable_awards", "tam_share_pct", "rank_in_tam",
        "largest_capability", "largest_capability_value", "capability_count", "concentration_pct",
        "contracts", "fastest_growing_capability", "fastest_growth_pct",
    ])
    capability_payload = records_for_json(supplier_capability, [
        "supplier_group", "capability", "supplier_capability_value", "capability_market",
        "capability_share_pct", "capability_rank", "contracts",
    ])
    yearly_payload = records_for_json(yearly, [
        "supplier_group", "capability", FIN_YEAR_COL, "value", "contracts", "capability_year_market", "share_pct", "_fy_start_year",
    ])
    # Send both the displayed current annual value and the original total value
    # to the browser. Without VALUE_COL here, the table's Total contract value
    # column falls back to zero.
    contract_cols = [c for c in [
        "supplier_group",
        CN_ID_COL,
        DESCRIPTION_COL,
        "capability",
        "detailed_capability",
        "addressability",
        "defence_domain",
        "domain_confidence",
        "domain_mapping_reason",
        FIN_YEAR_COL,
        AGENCY_COL,
        AGENCY_DIVISION_COL,
        AGENCY_BRANCH_COL,
        DISPLAY_START_DATE_COL,
        DISPLAY_END_DATE_COL,
        EXTENSION_COUNT_COL,
        timeline_value_col,
        VALUE_COL,
        ACTIVE_VALUE_COL,
        REMAINING_VALUE_COL,
        CURRENT_RUN_RATE_COL,
    ] if c in timeline_contracts.columns]
    contract_cols = list(dict.fromkeys(contract_cols))

    # The timeline payload is the only current-contract-only dataset.
    # Keep a balanced number of active records per supplier.
    timeline_source = timeline_contracts.copy()
    if DISPLAY_END_DATE_COL in timeline_source.columns:
        timeline_source["_timeline_end_sort"] = pd.to_datetime(
            timeline_source[DISPLAY_END_DATE_COL], errors="coerce"
        )
        timeline_source = timeline_source.sort_values(
            ["supplier_group", "_timeline_end_sort", timeline_value_col],
            ascending=[True, True, False],
        )
    timeline_source = (
        timeline_source.groupby("supplier_group", group_keys=False, dropna=False)
        .head(750)
        .drop(columns=["_timeline_end_sort"], errors="ignore")
    )
    contracts_payload = records_for_json(timeline_source, contract_cols)
    domain_payload = records_for_json(domain_summary, [
        "supplier_group", "defence_domain", "domain_value", "total_contract_value", "contracts"
    ])
    domain_yearly_payload = records_for_json(domain_yearly, [
        "supplier_group", "defence_domain", FIN_YEAR_COL, "domain_value",
        "total_contract_value", "contracts", "_fy_start_year"
    ])

    options_html = "\n".join(f'<option value="{html.escape(s)}">{html.escape(s)}</option>' for s in supplier_options)
    capability_options_html = "\n".join(f'<option value="{html.escape(c)}">{html.escape(c)}</option>' for c in capabilities)
    basis_label = (
        "Total contract value"
        if value_mode == "total"
        else "Annualised contract value / run-rate"
    )
    supplier_colour_payload = {supplier: supplier_colour(supplier) for supplier in supplier_options}

    html_doc = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Competitive Positioning inside Accenture Defence TAM</title>
  <script src="https://cdn.plot.ly/plotly-3.5.0.min.js"></script>
  <style>
    body {{ font-family: Arial, Helvetica, sans-serif; margin: 0; background: #f6f8fb; color: #1f2937; }}
    .wrap {{ max-width:1480px; margin:0 auto; padding:28px; }}
    h1 {{ margin:0 0 8px 0; font-size:30px; }}
    .subtitle {{ margin:0 0 20px 0; color:#526070; font-size:15px; line-height:1.45; }}
    .executive-summary {{
  background:#ffffff;
  border:1px solid #e5e7eb;
  border-left:5px solid #A100FF;
  border-radius:14px;
  padding:18px 20px;
  margin:0 0 18px 0;
  box-shadow:0 1px 2px rgba(0,0,0,0.04);
}}
.executive-summary h2 {{ margin:0 0 9px 0; font-size:20px; color:#111827; }}
.executive-summary p {{ margin:0 0 10px 0; color:#475467; line-height:1.55; }}
.executive-summary p:last-child {{ margin-bottom:0; }}
    .insight-box {{
      background:#ffffff;
      border:1px solid #e5e7eb;
      border-left:5px solid #A100FF;
      border-radius:14px;
      padding:16px 18px;
      margin:18px 0;
      box-shadow:0 1px 2px rgba(0,0,0,0.04);
    }}
    .insight-box h2 {{ margin:0 0 7px 0; font-size:19px; color:#111827; }}
    .insight-box p {{ margin:0; color:#475467; font-size:13px; line-height:1.55; }}
    .insight-box .question-line {{ margin-top:8px; color:#344054; font-weight:600; }}
    .insight-box strong[data-tooltip] {{ color:#111827; border-bottom:1px dotted #667085; cursor:help; outline:none; }}

    .executive-summary strong[data-tooltip] {{
      color: #111827;
      border-bottom: 1px dotted #667085;
      cursor: help;
      outline: none;
    }}
    .exec-tooltip {{
      position: fixed;
      left: 0;
      top: 0;
      max-width: min(360px, calc(100vw - 24px));
      padding: 10px 12px;
      border-radius: 9px;
      background: #101828;
      color: #ffffff;
      font-size: 12px;
      font-weight: 400;
      line-height: 1.4;
      text-align: left;
      white-space: normal;
      box-shadow: 0 8px 24px rgba(16,24,40,.22);
      opacity: 0;
      visibility: hidden;
      pointer-events: none;
      z-index: 2147483647;
      transition: opacity .12s ease;
    }}
    .exec-tooltip.is-visible {{
      opacity: 1;
      visibility: visible;
    }}
    .method {{ background: #ffffff; border: 1px solid #e5e7eb; border-radius: 14px; padding: 15px 17px; margin-bottom: 18px; line-height: 1.45; }}
    .selectors {{
      display: grid;
      grid-template-columns: 1fr 1fr 0.72fr;
      gap: 10px;
      margin: 12px 0;
    }}
    .floating-selectors {{
      position: fixed;
      top: 6px;
      left: 50%;
      width: min(1480px, calc(100vw - 56px));
      transform: translate(-50%, -140%);
      display: grid;
      grid-template-columns: 1fr 1fr 0.72fr;
      gap: 10px;
      padding: 5px 8px;
      box-sizing: border-box;
      background: rgba(246, 248, 251, 0.98);
      border: 1px solid rgba(208, 213, 221, 0.95);
      border-radius: 12px;
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.16);
      backdrop-filter: blur(8px);
      -webkit-backdrop-filter: blur(8px);
      opacity: 0;
      visibility: hidden;
      pointer-events: none;
      z-index: 99999;
      transition:
        transform 0.22s ease,
        opacity 0.18s ease,
        visibility 0.18s ease;
    }}
    .floating-selectors.is-visible {{
      transform: translate(-50%, 0);
      opacity: 1;
      visibility: visible;
      pointer-events: auto;
    }}
    .floating-selectors .selector-card {{
      box-shadow: none;
      padding: 6px 10px;
      border-radius: 10px;
    }}
    .selector-card {{ background: #ffffff; border: 1px solid #e5e7eb; border-radius: 12px; padding: 10px 12px; }}
    .growth-selector {{ max-width: 460px; margin: 10px 0 12px 0; }}
    .growth-section-header {{ padding: 12px 14px 0 14px; }}
    .growth-section-header h3 {{ margin: 0 0 4px 0; font-size: 18px; }}
    .growth-section-header p {{ margin: 0 0 10px 0; color: #526070; font-size: 13px; }}
    .selector-card label {{ display:block; font-size: 10px; text-transform: uppercase; letter-spacing: .04em; color: #667085; margin-bottom: 3px; }}
    select {{ width: 100%; height: 34px; padding: 0 10px; border: 1px solid #d0d5dd; border-radius: 8px; background: #fff; font-size: 13px; }}
    .floating-selectors .selector-card {{
      display: grid;
      grid-template-columns: auto minmax(0, 1fr);
      align-items: center;
      column-gap: 8px;
    }}
    .floating-selectors .selector-card label {{
      display: block;
      margin: 0;
      white-space: nowrap;
      font-size: 10px;
    }}
    .floating-selectors select {{ height: 32px; font-size: 13px; }}
    .cards {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin: 18px 0 22px 0; }}
    .supplier-kpi-block {{ background: #ffffff; border: 1px solid #e5e7eb; border-radius: 14px; padding: 14px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); min-width: 0; }}
    .supplier-kpi-block.selected {{ border-top: 4px solid #A100FF; }}
    .supplier-kpi-block.compare {{ border-top: 4px solid #5B6B7A; }}
    .supplier-kpi-title {{ font-size: 13px; text-transform: uppercase; letter-spacing: .04em; color: #344054; margin-bottom: 10px; }}
    .supplier-kpi-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }}
    .card {{ background: #ffffff; border: 1px solid #e5e7eb; border-radius: 14px; padding: 16px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); min-width: 0; overflow: hidden; }}
    .card-title {{ font-size: 12px; text-transform: uppercase; letter-spacing: .04em; color: #667085; margin-bottom: 8px; }}
    .card-value {{ font-size: clamp(20px, 1.6vw, 25px); font-weight: 700; color: #111827; margin-bottom: 5px; white-space: normal; overflow-wrap: anywhere; line-height: 1.12; }}
    .card-value.kpi-long {{ font-size: clamp(18px, 1.35vw, 23px); }}
    .card-subtitle {{ font-size: 12px; color: #667085; line-height: 1.35; overflow-wrap: anywhere; }}
    .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin-bottom: 18px; }}
    .panel {{ background: #ffffff; border: 1px solid #e5e7eb; border-radius: 14px; padding: 10px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }}
    .panel-wide {{ grid-column: 1 / -1; }}
    .share-panel {{ padding-bottom: 30px; overflow: visible; }}
    .chart-heading {{ padding: 18px 18px 0 18px; }}
    .chart-title {{ margin: 0; font-size: 21px; font-weight: 700; color: #111827; line-height: 1.25; }}
    .chart-subtitle {{ margin: 5px 0 0 0; font-size: 13px; color: #526070; line-height: 1.35; }}
    .table-panel {{ padding: 14px; }}
    .timeline-panel {{ padding: 14px; }}
    .timeline-header {{
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 18px;
      margin-bottom: 6px;
    }}
    .timeline-header h3 {{ margin: 0 0 4px 0; font-size: 18px; }}
    .timeline-header p {{ margin: 0; color: #526070; font-size: 13px; }}
    .timeline-filter {{ min-width: 220px; }}
    .timeline-filter label {{
      display: block;
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: .04em;
      color: #667085;
      margin-bottom: 3px;
    }}
    .timeline-legend {{
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      gap: 10px 18px;
      align-items: center;
      padding: 8px 2px 10px 2px;
      color: #526070;
      font-size: 12px;
    }}
    .timeline-legend-left,
    .timeline-legend-right {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
    }}
    .timeline-domain-item {{
      display: inline-flex;
      align-items: center;
      gap: 5px;
      white-space: nowrap;
    }}
    .timeline-domain-swatch {{
      width: 18px;
      height: 6px;
      border-radius: 999px;
      display: inline-block;
    }}
    .timeline-capability-key {{
      display: inline-flex;
      align-items: center;
      gap: 5px;
      white-space: nowrap;
    }}
    .timeline-capability-swatch {{
      width: 14px;
      height: 10px;
      border: 1px solid #A100FF;
      background: rgba(161, 0, 255, .10);
      display: inline-block;
    }}
    .today-key {{ color: #111827; font-size: 20px; font-weight: 700; line-height: 1; }}
    .timeline-scroll {{
      overflow: visible;
      border: 1px solid #e5e7eb;
      border-radius: 10px;
      background: #ffffff;
    }}
    .timeline-audit {{
      margin-top: 8px;
      color: #667085;
      font-size: 12px;
      line-height: 1.4;
    }}
    .table-panel h3 {{ margin: 0 0 4px 0; font-size: 18px; }}
    .table-panel p {{ margin: 0 0 12px 0; color: #526070; font-size: 13px; }}
    .table-scroll {{ max-height: 520px; overflow: auto; border: 1px solid #e5e7eb; border-radius: 10px; }}
    .domain-tiles {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; padding: 0 14px 14px 14px; }}
    .domain-tile {{ border: 1px solid #e5e7eb; border-radius: 12px; padding: 12px; background: #ffffff; min-width: 0; }}
    .domain-tile-title {{ font-size: 12px; text-transform: uppercase; letter-spacing: .04em; color: #667085; margin-bottom: 6px; }}
    .domain-tile-value {{ font-size: 20px; font-weight: 700; color: #111827; margin-bottom: 4px; }}
    .domain-tile-subtitle {{ font-size: 12px; color: #667085; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
    th {{ position: sticky; top: 0; background: #f2f4f7; text-align: left; padding: 8px; border-bottom: 1px solid #e5e7eb; }}
    td {{ padding: 7px 8px; border-bottom: 1px solid #eef2f7; vertical-align: top; }}
    tr:hover {{ background: #fafafa; }}
    .share-range-controls {{ display: flex; flex-wrap: wrap; gap: 8px; padding: 12px 18px 0 18px; }}
    .range-pill {{ display: inline-flex; align-items: center; gap: 6px; border: 1px solid #d0d5dd; border-radius: 999px; padding: 7px 12px; background: #ffffff; font-size: 12px; cursor: pointer; white-space: nowrap; }}
    .range-pill:has(input:checked) {{ background: #eef2ff; border-color: #636efa; color: #243b9f; font-weight: 600; }}
    .card.kpi-win {{ background: #ecfdf3; border-color: #6ce9a6; }}
    .card.kpi-loss {{ background: #fef3f2; border-color: #fda29b; }}
    .card.kpi-tie {{ background: #f8fafc; border-color: #d0d5dd; }}
    .card.kpi-win .card-value,
    .card.kpi-loss .card-value,
    .card.kpi-tie .card-value {{ color: inherit; }}
    .leaderboard-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; padding: 16px 18px 22px 18px; }}
    .leaderboard-card {{ border: 1px solid #e5e7eb; border-radius: 14px; padding: 14px; background: #ffffff; min-width: 0; }}
    .leaderboard-card h4 {{ margin: 0 0 4px 0; font-size: 16px; color: #111827; line-height: 1.25; }}
    .leaderboard-market {{ font-size: 12px; color: #667085; margin-bottom: 12px; }}
    .leader-row {{ display: grid; grid-template-columns: 26px minmax(130px, 1fr) minmax(110px, 2.2fr) 54px; gap: 8px; align-items: center; padding: 6px 0; border-top: 1px solid #f1f5f9; }}
    .leader-row:first-of-type {{ border-top: none; }}
    .leader-rank {{ font-size: 12px; font-weight: 700; color: #667085; text-align: center; }}
    .leader-name {{ font-size: 12px; color: #111827; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    .leader-bar-track {{ height: 12px; background: #eef2f7; border-radius: 999px; overflow: hidden; }}
    .leader-bar-fill {{ height: 100%; border-radius: 999px; min-width: 2px; }}
    .leader-share {{ font-size: 12px; font-weight: 700; text-align: right; color: #111827; }}
    .leader-row.focus-selected {{ background: #faf5ff; margin: 0 -8px; padding-left: 8px; padding-right: 8px; border-radius: 8px; }}
    .leader-row.focus-compare {{ outline: 1px solid rgba(134,188,37,.45); margin: 0 -8px; padding-left: 8px; padding-right: 8px; border-radius: 8px; }}
    .leader-gap {{
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 20px;
      padding: 2px 0;
      color: #98A2B3;
      font-size: 18px;
      line-height: 1;
      letter-spacing: 3px;
      user-select: none;
    }}
    .leaderboard-note {{ padding: 0 18px 12px 18px; font-size: 12px; color: #667085; }}
    .note {{ color: #526070; font-size: 13px; line-height: 1.45; margin-top: 14px; }}
    .question-list {{
      margin: 10px 0 0 0;
      padding: 0;
      list-style: none;
      color: #344054;
      font-size: 13px;
      font-weight: 600;
      line-height: 1.45;
    }}
    .question-list li {{
      position: relative;
      margin: 5px 0 0 0;
      padding-left: calc(18px + (var(--level, 0) * 18px));
    }}
    .question-list li::before {{
      content: '';
      position: absolute;
      left: calc(2px + (var(--level, 0) * 18px));
      top: 0.62em;
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: #A100FF;
      transform: translateY(-50%);
    }}
    @media (max-width: 720px) {{ .supplier-kpi-grid {{ grid-template-columns: 1fr; }} }}
    @media (max-width: 1000px) {{
      .cards, .grid2, .selectors, .domain-tiles, .leaderboard-grid {{
        grid-template-columns: 1fr;
      }}
      .floating-selectors {{
        width: calc(100vw - 24px);
        grid-template-columns: 1fr;
        top: 6px;
        max-height: calc(100vh - 12px);
        overflow-y: auto;
      }}
    }}
  </style>
</head>
<body>
<div class="wrap">
  <h1>Competitive Positioning inside Accenture Defence TAM</h1>
  <div style="font-size:12px;color:#667085;margin-bottom:8px;">Colour version: supplier-group designated colours</div>

  <section class="executive-summary" aria-labelledby="executiveSummaryTitle">
      <h2 id="executiveSummaryTitle">Executive purpose</h2>
      <p>
        This dashboard provides a direct comparison of two suppliers across <strong tabindex="0" data-tooltip="Accenture's Total Addressable Market (TAM) is the portion of Australian Defence procurement that aligns with Accenture's core service offerings. It represents the market where Accenture can realistically compete.">Accenture's addressable Defence market</strong>. It brings together market position, Service Offering strength, current contracts and Defence-domain presence so users can quickly understand where each supplier is established, where the competitive gap is widest and where future growth may be available.
      </p>
      <ul class="question-list" aria-label="Questions this dashboard helps answer">
        <li style="--level:0">Who is better positioned overall?</li>
        <li style="--level:0">Where does each supplier lead by Service Offering?</li>
        <li style="--level:0">Which current contracts underpin that position?</li>
        <li style="--level:0">Which Defence domains are most important to the selected supplier?</li>
      </ul>
  </section>

  <div class="selectors" id="originalSupplierSelectors">
    <div class="selector-card">
      <label for="supplierSelect">Selected supplier</label>
      <select id="supplierSelect">{options_html}</select>
    </div>
    <div class="selector-card">
      <label for="compareSelect">Compare against</label>
      <select id="compareSelect">{options_html}</select>
    </div>
    <div class="selector-card">
      <label for="periodSelect">Financial-year period</label>
      <select id="periodSelect"></select>
    </div>
  </div>

  <div class="floating-selectors" id="floatingSupplierSelectors" aria-hidden="true">
    <div class="selector-card">
      <label for="floatingSupplierSelect">Selected supplier</label>
      <select id="floatingSupplierSelect">{options_html}</select>
    </div>
    <div class="selector-card">
      <label for="floatingCompareSelect">Compare against</label>
      <select id="floatingCompareSelect">{options_html}</select>
    </div>
    <div class="selector-card">
      <label for="floatingPeriodSelect">Financial-year period</label>
      <select id="floatingPeriodSelect"></select>
    </div>
  </div>

  <section class="insight-box">
    <h2>Start with the headline comparison</h2>
    <p>The KPI cards provide the quickest read of relative scale, market share, strongest Service Offering and breadth of presence. They help establish whether one supplier is leading overall, or whether the advantage is concentrated in only one part of the market.</p>
    <ul class="question-list" aria-label="Questions this section helps answer">
      <li style="--level:0">Who has the larger position?</li>
      <li style="--level:0">Who is more diversified?</li>
      <li style="--level:0">Is either supplier overly dependent on a single Service Offering?</li>
    </ul>
  </section>

  <div class="cards">
    <div class="supplier-kpi-block selected">
      <div class="supplier-kpi-title" id="selectedKpiTitle">Selected supplier</div>
      <div class="supplier-kpi-grid">
        <div class="card" id="selectedCardValue"><div class="card-title" id="selectedKpiValueTitle">Period value</div><div class="card-value" id="selectedKpiAddressable">-</div><div class="card-subtitle" id="selectedKpiValueSubtitle">Award value inside the selected addressable TAM period</div></div>
        <div class="card" id="selectedCardShare"><div class="card-title">Addressable TAM share</div><div class="card-value" id="selectedKpiShare">-</div><div class="card-subtitle" id="selectedKpiRank">-</div></div>
        <div class="card" id="selectedCardLargest"><div class="card-title">Largest Service Offering</div><div class="card-value kpi-long" id="selectedKpiLargest">-</div><div class="card-subtitle" id="selectedKpiLargestValue">-</div></div>
        <div class="card" id="selectedCardPresence"><div class="card-title">Service Offering presence</div><div class="card-value kpi-long" id="selectedKpiBreadth">-</div><div class="card-subtitle" id="selectedKpiConcentration">-</div></div>
      </div>
    </div>
    <div class="supplier-kpi-block compare">
      <div class="supplier-kpi-title" id="compareKpiTitle">Comparison supplier</div>
      <div class="supplier-kpi-grid">
        <div class="card" id="compareCardValue"><div class="card-title" id="compareKpiValueTitle">Period value</div><div class="card-value" id="compareKpiAddressable">-</div><div class="card-subtitle" id="compareKpiValueSubtitle">Award value inside the selected addressable TAM period</div></div>
        <div class="card" id="compareCardShare"><div class="card-title">Addressable TAM share</div><div class="card-value" id="compareKpiShare">-</div><div class="card-subtitle" id="compareKpiRank">-</div></div>
        <div class="card" id="compareCardLargest"><div class="card-title">Largest Service Offering</div><div class="card-value kpi-long" id="compareKpiLargest">-</div><div class="card-subtitle" id="compareKpiLargestValue">-</div></div>
        <div class="card" id="compareCardPresence"><div class="card-title">Service Offering presence</div><div class="card-value kpi-long" id="compareKpiBreadth">-</div><div class="card-subtitle" id="compareKpiConcentration">-</div></div>
      </div>
    </div>
  </div>

  <div class="grid2">
    <section class="insight-box panel-wide">
      <h2>Compare where each supplier is strongest</h2>
      <p>The Service Offering comparison shows the scale of each supplier's position across the market. It reveals where one supplier clearly leads, where the two are closely matched and where a supplier has little or no presence.</p>
      <ul class="question-list" aria-label="Questions this chart helps answer">
        <li style="--level:0">Which Service Offerings define each supplier's competitive position?</li>
        <li style="--level:0">Where is Accenture already credible?</li>
        <li style="--level:0">Where is the largest gap to close?</li>
      </ul>
    </section>
    <div class="panel panel-wide">
      <div class="chart-heading">
        <h3 class="chart-title" id="profileChartTitle">Supplier vs Supplier - Value by Service Offering</h3>
        <p class="chart-subtitle">Comparison of Supplier 1 vs Supplier 2 across Accenture-addressable Defence service offerings.</p>
      </div>
      <div id="profileChart" style="height:700px;"></div>
    </div>
    <section class="insight-box panel-wide">
      <h2>See the wider competitive field</h2>
      <p>The supplier leaderboards place the selected pair within the broader market for every Service Offering. They show whether a supplier is a category leader, a credible challenger or a smaller participant, while also identifying the organisations that currently set the benchmark.</p>
      <ul class="question-list" aria-label="Questions this chart helps answer">
        <li style="--level:0">Who leads each Service Offering?</li>
        <li style="--level:0">How far behind is the selected supplier?</li>
        <li style="--level:0">Which competitors matter most in each segment?</li>
      </ul>
    </section>
    <div class="panel panel-wide share-panel">
      <div class="chart-heading">
        <h3 class="chart-title">Service Offering Supplier Leaderboards</h3>
        <p class="chart-subtitle">Each Service Offering is ranked independently, so large and small markets remain readable. The top five suppliers are shown, with Supplier 1 and Supplier 2 always included for comparison.</p>
      </div>
      <div id="shareRangeControls" class="share-range-controls"></div>
      <div class="leaderboard-note">Rankings, market shares and Service Offering totals recalculate for the selected financial-year period.</div>
      <div id="capabilityLeaderboards" class="leaderboard-grid"></div>
    </div>
    <section class="insight-box panel-wide">
      <h2>Understand the contracts behind the position</h2>
      <p>The current-contract timeline translates market position into the live work that sustains it. It highlights the scale, duration and concentration of active contracts, helping users distinguish between a broad portfolio and a position driven by only a small number of major awards.</p>
      <ul class="question-list" aria-label="Questions this chart helps answer">
        <li style="--level:0">Which current contracts anchor the first selected supplier's position?</li>
        <li style="--level:0">When do its major contracts end?</li>
        <li style="--level:0">Where could upcoming expiries create opportunity?</li>
      </ul>
    </section>
    <div class="panel panel-wide timeline-panel">
      <div class="timeline-header">
          <div>

              <h2 id="timelineSectionTitle"
                  style="margin:0 0 6px 0;
                        font-size:30px;
                        font-weight:700;
                        color:#111827;">
                  First Selected Supplier - Current Contracts
              </h2>

              <h3 id="timelineTitle"
                  style="margin:0 0 6px 0;
                        font-size:20px;
                        font-weight:600;
                        color:#344054;">
                  Accenture current contract timeline
              </h3>

              <p id="timelineSubtitle">
                  The first supplier selector controls the contracts shown...
              </p>

          </div>
      </div>
      <div class="timeline-legend">
        <div class="timeline-legend-left">
          <span><b>●</b> Start date</span>
          <span><b>◆</b> End date</span>
          <span class="today-key">│</span><span>As-at date</span>
          <span class="timeline-capability-key">
            <span class="timeline-capability-swatch"></span>
            Service Offering section
          </span>
        </div>
        <div id="timelineDomainLegend" class="timeline-legend-right"></div>
      </div>
      <div id="contractsTimelineScroll" class="timeline-scroll">
        <div id="contractsTimeline"></div>
      </div>
      <div id="timelineAudit" class="timeline-audit"></div>
    </div>
    <section class="insight-box panel-wide">
      <h2>Read the supplier's Defence footprint</h2>
      <p>The domain view shows where the selected supplier's position is concentrated across Defence. It helps reveal whether the supplier has a balanced footprint or depends heavily on one customer environment, and where its strongest relationships appear to sit.</p>
      <ul class="question-list" aria-label="Questions this chart helps answer">
        <li style="--level:0">Which Defence domains matter most to the selected supplier?</li>
        <li style="--level:0">Is its position broad or concentrated?</li>
        <li style="--level:0">Where are there underrepresented areas for growth?</li>
      </ul>
    </section>
    <div class="panel panel-wide">
      <div class="table-panel">
        <h3 id="domainSectionTitle">Defence domain breakdown</h3>
        <p id="domainSectionSubtitle">Contract value for the selected financial-year period, grouped into Maritime, Land, Air, Joint, Cyber, Space and Capability Enabler.</p>
      </div>
      <div id="domainChart" style="height:500px;"></div>
      <div id="domainTiles" class="domain-tiles"></div>
    </div>
  </div>

  <p class="note">
    Note: this is not whole-of-Defence supplier share. It is supplier positioning inside Defence contracts that the classifier marks as relevant to Accenture's service-offering portfolio.
  </p>
</div>

<script>
const SUPPLIER_SUMMARY = {json.dumps(summary_payload)};
const SUPPLIER_CAPABILITY = {json.dumps(capability_payload)};
const YEARLY = {json.dumps(yearly_payload)};
const CONTRACTS = {json.dumps(contracts_payload)};
const DOMAIN_SUMMARY = {json.dumps(domain_payload)};
const DOMAIN_YEARLY = {json.dumps(domain_yearly_payload)};
const DOMAIN_COLOURS = {json.dumps(DOMAIN_COLOURS)};
const VALUE_COL = {json.dumps(value_col)};
const TIMELINE_VALUE_COL = {json.dumps(timeline_value_col)};
const TOTAL_CAPABILITIES = {len(capabilities)};
const DEFAULT_SUPPLIER = {json.dumps(default_supplier)};
const DEFAULT_COMPARE = {json.dumps(default_compare)};
const AS_AT_DATE = {json.dumps(as_at_date)};

function money(v) {{
  v = Number(v || 0);
  const sign = v < 0 ? '-' : '';
  v = Math.abs(v);
  if (v >= 1e9) return sign + '$' + (v/1e9).toFixed(2) + 'B';
  if (v >= 1e8) return sign + '$' + (v/1e6).toFixed(0) + 'M';
  if (v >= 1e7) return sign + '$' + (v/1e6).toFixed(1) + 'M';
  if (v >= 1e6) return sign + '$' + (v/1e6).toFixed(2) + 'M';
  if (v >= 1e3) return sign + '$' + (v/1e3).toFixed(0) + 'K';
  return sign + '$' + v.toFixed(0);
}}
function pct(v) {{ return (Number(v || 0)).toFixed(1) + '%'; }}
function shortText(v, n) {{
  v = String(v || '');
  return v.length > n ? v.slice(0, n - 1) + '…' : v;
}}
const SUPPLIER_COLOURS = {json.dumps(supplier_colour_payload)};
const UNKNOWN_SUPPLIER_GREY = '#5B6B7A';

function companyColour(name) {{
  const supplierName = String(name || '').trim();
  const exact = SUPPLIER_COLOURS[supplierName];
  if (exact) return exact;

  const lower = supplierName.toLowerCase();
  const fallbackColours = [
    [['accenture'], '#A100FF'],
    [['deloitte'], '#86BC25'],
    [['kpmg'], '#00338D'],
    [['ey', 'ernst'], '#FFE600'],
    [['pwc', 'pricewaterhouse'], '#E0301E'],
    [['leidos'], '#850F88'],
    [['boeing'], '#1A409F'],
    [['lockheed'], '#003087'],
    [['ibm'], '#1F70C1'],
    [['oracle'], '#C74634'],
    [['fujitsu'], '#D6001C'],
    [['aurecon'], '#2E7D32'],
    [['dxc'], '#5B2C83'],
    [['microsoft'], '#737373'],
    [['amazon', 'aws'], '#FF9900'],
    [['sap'], '#0FAAFF'],
    [['data#3', 'data 3'], '#34495E'],
    [['bae'], '#2F5597'],
    [['thales'], '#005EB8'],
    [['raytheon'], '#B22222'],
    [['northrop'], '#3A5F8A'],
    [['saab'], '#005596'],
    [['nec'], '#C8102E'],
    [['datacom'], '#00AEEF']
  ];

  for (const [tokens, colour] of fallbackColours) {{
    if (tokens.some(token => lower === token || lower.includes(token))) {{
      return colour;
    }}
  }}
  return UNKNOWN_SUPPLIER_GREY;
}}

function coloursForPair(supplier, compare) {{
  return {{
    supplier: companyColour(supplier),
    compare: companyColour(compare)
  }};
}}

function supplierTextColour(name) {{
  const colour = companyColour(name).toUpperCase();

  // Light brand colours remain as accents, while text switches to a
  // dark accessible colour so EY yellow and Deloitte green stay readable.
  const lightColours = new Set([
    '#FFE600', // EY yellow
    '#86BC25'  // Deloitte green
  ]);

  return lightColours.has(colour) ? '#344054' : colour;
}}

function updateSupplierColours(supplier, compare) {{
  const selectedAccent = companyColour(supplier);
  const compareAccent = companyColour(compare);

  const selectedBlock = document.querySelector('.supplier-kpi-block.selected');
  const compareBlock = document.querySelector('.supplier-kpi-block.compare');

  if (selectedBlock) selectedBlock.style.borderTopColor = selectedAccent;
  if (compareBlock) compareBlock.style.borderTopColor = compareAccent;

  const selectedTitle = document.getElementById('selectedKpiTitle');
  const compareTitle = document.getElementById('compareKpiTitle');

  if (selectedTitle) {{
    selectedTitle.style.color = '#344054';
    selectedTitle.style.borderLeft = '5px solid ' + selectedAccent;
    selectedTitle.style.paddingLeft = '9px';
  }}
  if (compareTitle) {{
    compareTitle.style.color = '#344054';
    compareTitle.style.borderLeft = '5px solid ' + compareAccent;
    compareTitle.style.paddingLeft = '9px';
  }}

  [
    'selectedKpiAddressable',
    'selectedKpiShare',
    'selectedKpiLargest',
    'selectedKpiBreadth',
    'compareKpiAddressable',
    'compareKpiShare',
    'compareKpiLargest',
    'compareKpiBreadth'
  ].forEach(id => {{
    const el = document.getElementById(id);
    if (el) el.style.color = '#111827';
  }});
}}
function availableFinancialYears() {{
  return Array.from(new Map(
    YEARLY
      .filter(r => r._fy_start_year !== null && r._fy_start_year !== undefined)
      .map(r => [String(r['Financial Year']), Number(r._fy_start_year)])
  ).entries()).sort((a, b) => a[1] - b[1]).map(x => x[0]);
}}

function buildGlobalPeriodOptions() {{
  const years = availableFinancialYears();

  // Show the full coverage directly in the All FY option, e.g. 2015-2026.
  const firstFY = years.length ? String(years[0]) : '';
  const lastFY = years.length ? String(years[years.length - 1]) : '';
  const firstStart = (firstFY.match(/(?:19|20)\d{{2}}/) || [''])[0];
  const lastMatches = lastFY.match(/(?:19|20)\d{{2}}/g) || [];
  const lastEnd = lastMatches.length ? lastMatches[lastMatches.length - 1] : '';
  const allYearsLabel = firstStart && lastEnd
    ? 'All FY (' + firstStart + '-' + lastEnd + ')'
    : 'All FY';

  const ranges = [{{key:'all', label:allYearsLabel, years:years}}];
  if (years.length >= 5) ranges.push({{key:'last5', label:'Last 5 FY (' + years.slice(-5)[0] + ' to ' + years.slice(-5).slice(-1)[0] + ')', years:years.slice(-5)}});
  if (years.length >= 3) ranges.push({{key:'last3', label:'Last 3 FY (' + years.slice(-3)[0] + ' to ' + years.slice(-3).slice(-1)[0] + ')', years:years.slice(-3)}});
  window.GLOBAL_PERIOD_RANGES = ranges;
  const html = ranges.map((r,i) => '<option value="' + i + '">' + r.label + '</option>').join('');
  const original = document.getElementById('periodSelect');
  const floating = document.getElementById('floatingPeriodSelect');
  if (original) original.innerHTML = html;
  if (floating) floating.innerHTML = html;
}}

function selectedGlobalPeriod() {{
  const el = document.getElementById('periodSelect');
  const index = el ? Number(el.value || 0) : 0;
  return (window.GLOBAL_PERIOD_RANGES || [])[index] || {{key:'all', label:'All FY', years:[]}};
}}

function periodYearSet() {{
  return new Set(selectedGlobalPeriod().years || []);
}}

function periodYearlyRows() {{
  const range = selectedGlobalPeriod();
  const years = new Set(range.years || []);
  if (range.key === 'all' || !years.size) return YEARLY.slice();
  return YEARLY.filter(r => years.has(String(r['Financial Year'] || '')));
}}

function rowsForSupplier(supplier) {{
  const rows = periodYearlyRows();
  const capSupplier = new Map();
  const capYearMarket = new Map();
  rows.forEach(r => {{
    const cap = String(r.capability || 'Unclassified');
    const fy = String(r['Financial Year'] || '');
    const marketKey = cap + '||' + fy;
    if (!capYearMarket.has(marketKey)) capYearMarket.set(marketKey, Number(r.capability_year_market || 0));
    if (r.supplier_group !== supplier) return;
    if (!capSupplier.has(cap)) capSupplier.set(cap, {{supplier_group:supplier, capability:cap, supplier_capability_value:0, contracts:0}});
    const row = capSupplier.get(cap);
    row.supplier_capability_value += Number(r.value || 0);
    row.contracts += Number(r.contracts || 0);
  }});
  const marketByCap = new Map();
  capYearMarket.forEach((value, key) => {{
    const cap = key.split('||')[0];
    marketByCap.set(cap, (marketByCap.get(cap) || 0) + Number(value || 0));
  }});
  return Array.from(capSupplier.values()).map(r => {{
    const market = Number(marketByCap.get(r.capability) || 0);
    return {{...r, capability_market:market, capability_share_pct:market ? r.supplier_capability_value / market * 100 : 0, capability_rank:0}};
  }});
}}

function yearlyFor(supplier, capability) {{
  return periodYearlyRows().filter(r => r.supplier_group === supplier && r.capability === capability)
    .sort((a,b) => (a._fy_start_year || 0) - (b._fy_start_year || 0) || String(a['{FIN_YEAR_COL}']).localeCompare(String(b['{FIN_YEAR_COL}'])));
}}

function summaryFor(supplier) {{
  const rows = periodYearlyRows();
  const supplierTotals = new Map();
  const supplierCaps = new Map();
  const capYearMarket = new Map();
  rows.forEach(r => {{
    const name = String(r.supplier_group || 'Unknown');
    const cap = String(r.capability || 'Unclassified');
    const fy = String(r['Financial Year'] || '');
    supplierTotals.set(name, (supplierTotals.get(name) || 0) + Number(r.value || 0));
    const key = name + '||' + cap;
    supplierCaps.set(key, (supplierCaps.get(key) || 0) + Number(r.value || 0));
    const marketKey = cap + '||' + fy;
    if (!capYearMarket.has(marketKey)) capYearMarket.set(marketKey, Number(r.capability_year_market || 0));
  }});
  const totalTam = Array.from(capYearMarket.values()).reduce((a,b) => a + Number(b || 0), 0);
  const awards = Number(supplierTotals.get(supplier) || 0);
  const ranked = Array.from(supplierTotals.entries()).sort((a,b) => b[1] - a[1]);
  const rank = ranked.findIndex(x => x[0] === supplier) + 1;
  const caps = Array.from(supplierCaps.entries())
    .filter(([key, value]) => key.startsWith(supplier + '||') && Number(value || 0) > 0)
    .map(([key, value]) => [key.split('||')[1], Number(value || 0)])
    .sort((a,b) => b[1] - a[1]);
  return {{
    supplier_group:supplier,
    addressable_awards:awards,
    tam_share_pct:totalTam ? awards / totalTam * 100 : 0,
    rank_in_tam:rank > 0 ? rank : 0,
    largest_capability:caps.length ? caps[0][0] : 'n/a',
    largest_capability_value:caps.length ? caps[0][1] : 0,
    capability_count:caps.length,
    concentration_pct:awards && caps.length ? caps[0][1] / awards * 100 : 0,
    contracts:0
  }};
}}

function setSupplierKpi(prefix, supplier, roleLabel) {{
  const s = summaryFor(supplier);
  const periodLabel = selectedGlobalPeriod().label || 'Selected period';
  document.getElementById(prefix + 'KpiTitle').textContent = roleLabel + ': ' + supplier;
  document.getElementById(prefix + 'KpiValueTitle').textContent = periodLabel + ' value';
  document.getElementById(prefix + 'KpiValueSubtitle').textContent = 'Award value inside ' + periodLabel + ' addressable TAM';
  document.getElementById(prefix + 'KpiAddressable').textContent = money(s.addressable_awards);
  document.getElementById(prefix + 'KpiShare').textContent = pct(s.tam_share_pct);
  document.getElementById(prefix + 'KpiRank').textContent = s.rank_in_tam ? 'Rank #' + s.rank_in_tam + ' inside ' + periodLabel + ' addressable TAM' : 'No awards in ' + periodLabel;
  document.getElementById(prefix + 'KpiLargest').textContent = s.largest_capability || 'n/a';
  document.getElementById(prefix + 'KpiLargestValue').textContent = money(s.largest_capability_value) + ' in ' + periodLabel;
  const capabilityCount = Number(s.capability_count || 0);
  document.getElementById(prefix + 'KpiBreadth').textContent = capabilityCount + (capabilityCount === 1 ? ' Service Offering' : ' Service Offerings');
  document.getElementById(prefix + 'KpiConcentration').textContent = supplier + ' has presence in ' + capabilityCount + ' of ' + TOTAL_CAPABILITIES + ' capability segments';
}}

function setComparisonClass(cardId, ownValue, otherValue) {{
  const el = document.getElementById(cardId);
  if (!el) return;
  el.classList.remove('kpi-win', 'kpi-loss', 'kpi-tie');
  const a = Number(ownValue || 0);
  const b = Number(otherValue || 0);
  const tolerance = Math.max(1e-9, Math.max(Math.abs(a), Math.abs(b)) * 0.0001);
  el.classList.add(Math.abs(a - b) <= tolerance ? 'kpi-tie' : (a > b ? 'kpi-win' : 'kpi-loss'));
}}

function updateKpis(supplier, compare) {{
  setSupplierKpi('selected', supplier, 'Selected supplier');
  setSupplierKpi('compare', compare, 'Comparison supplier');
  const a = summaryFor(supplier);
  const b = summaryFor(compare);
  setComparisonClass('selectedCardValue', a.addressable_awards, b.addressable_awards);
  setComparisonClass('compareCardValue', b.addressable_awards, a.addressable_awards);
  setComparisonClass('selectedCardShare', a.tam_share_pct, b.tam_share_pct);
  setComparisonClass('compareCardShare', b.tam_share_pct, a.tam_share_pct);
  setComparisonClass('selectedCardLargest', a.largest_capability_value, b.largest_capability_value);
  setComparisonClass('compareCardLargest', b.largest_capability_value, a.largest_capability_value);
  setComparisonClass('selectedCardPresence', a.capability_count, b.capability_count);
  setComparisonClass('compareCardPresence', b.capability_count, a.capability_count);
}}

function buildShareRangeControls() {{
  const el = document.getElementById('shareRangeControls');
  if (el) el.style.display = 'none';
}}

function selectedShareRange() {{
  return selectedGlobalPeriod();
}}

function updateProfileChart(supplier, compare) {{
  document.getElementById('profileChartTitle').innerHTML = '<b>' + supplier + '</b> vs <b>' + compare + '</b> - Value by Service Offering';
  const colours = coloursForPair(supplier, compare);
  const selectedMap = new Map(rowsForSupplier(supplier).map(r => [r.capability, r]));
  const compareMap = new Map(rowsForSupplier(compare).map(r => [r.capability, r]));

  // Build the category list from BOTH suppliers. Previously the chart used only
  // Supplier 1's non-zero Service Offerings, which hid categories where Supplier
  // 1 had no value but Supplier 2 did (for example PwC vs Deloitte in Managed
  // Services & Operations).
  const allCapabilities = Array.from(new Set([
    ...selectedMap.keys(),
    ...compareMap.keys()
  ])).filter(capability => {{
    const selectedValue = Number((selectedMap.get(capability) || {{}}).supplier_capability_value || 0);
    const compareValue = Number((compareMap.get(capability) || {{}}).supplier_capability_value || 0);
    return selectedValue > 0 || compareValue > 0;
  }});

  const caps = allCapabilities
    .sort((a, b) => {{
      const aCombined = Number((selectedMap.get(a) || {{}}).supplier_capability_value || 0) +
        Number((compareMap.get(a) || {{}}).supplier_capability_value || 0);
      const bCombined = Number((selectedMap.get(b) || {{}}).supplier_capability_value || 0) +
        Number((compareMap.get(b) || {{}}).supplier_capability_value || 0);
      return bCombined - aCombined;
    }})
    .slice(0, 12)
    .reverse();

  const selectedRows = caps.map(c => selectedMap.get(c) || {{capability:c}});
  const compareRows = caps.map(c => compareMap.get(c) || {{capability:c}});
  const selectedValues = selectedRows.map(r => Number(r.supplier_capability_value || 0) / 1e9);
  const compareValues = compareRows.map(r => Number(r.supplier_capability_value || 0) / 1e9);
  const selectedText = selectedRows.map(r => Number(r.supplier_capability_value || 0) > 0 ? money(r.supplier_capability_value) : '');
  const compareText = compareRows.map(r => Number(r.supplier_capability_value || 0) > 0 ? money(r.supplier_capability_value) : '');
  const custom = selectedRows.map(r => [money(r.supplier_capability_value || 0), pct(r.capability_share_pct || 0), r.capability_rank || '-', money(r.capability_market || 0)]);
  const compareCustom = compareRows.map(r => [money(r.supplier_capability_value || 0), pct(r.capability_share_pct || 0), r.capability_rank || '-', money(r.capability_market || 0)]);

  Plotly.react('profileChart', [
    {{type:'bar', orientation:'h', y:caps, x:selectedValues, name:supplier, marker:{{color: colours.supplier, opacity:0.92}}, text:selectedText, texttemplate:'%{{text}}', textposition:'outside', cliponaxis:false, customdata: custom, hovertemplate:'<b>%{{y}}</b><br>' + supplier + ': %{{customdata[0]}}<br>Share of Service Offering: %{{customdata[1]}}<br>Service Offering rank: #%{{customdata[2]}}<br>Total Service Offering: %{{customdata[3]}}<extra></extra>'}},
    {{type:'bar', orientation:'h', y:caps, x:compareValues, name:compare, marker:{{color: colours.compare, opacity:0.92}}, text:compareText, texttemplate:'%{{text}}', textposition:'outside', cliponaxis:false, customdata: compareCustom, hovertemplate:'<b>%{{y}}</b><br>' + compare + ': %{{customdata[0]}}<br>Share of Service Offering: %{{customdata[1]}}<br>Service Offering rank: #%{{customdata[2]}}<br>Total Service Offering: %{{customdata[3]}}<extra></extra>'}}
  ], {{
    title: '',
    template:'plotly_white', colorway:[colours.supplier, colours.compare], height:700, barmode:'group', margin:{{t:35,l:240,r:130,b:90}},
    xaxis:{{
      title:{{
        text:'Value ($B)',
        font:{{size:15}}
      }},
      automargin:true,
      showgrid:true,
      zeroline:true
    }}, yaxis:{{title:'Service Offering', automargin:true}}, legend:{{orientation:'h', y:-0.18}}
  }}, {{responsive:true, displayModeBar:false, staticPlot:true, scrollZoom:false, doubleClick:false}});
}}


function shouldLabelSupplier(name) {{
  const lower = String(name || '').toLowerCase();
  return lower.includes('raytheon') ||
    lower.includes('leidos') ||
    lower.includes('data#3') ||
    lower.includes('amazon') || lower.includes('aws') ||
    lower.includes('fujitsu') ||
    lower.includes('lockheed') ||
    lower.includes('accenture') ||
    lower.includes('deloitte') ||
    lower.includes('sap') ||
    lower.includes('bae') ||
    lower === 'ey' || lower.includes('ernst') ||
    lower.includes('kpmg');
}}

function selectedPeriodLandscapeData() {{
  const range = selectedShareRange();
  const yearSet = new Set(range.years || []);
  const useAllYears = !yearSet.size || range.key === 'all';
  const capSupplier = new Map();
  const capYearMarket = new Map();

  YEARLY.forEach(r => {{
    const fy = String(r['Financial Year'] || '');
    if (!useAllYears && !yearSet.has(fy)) return;
    const cap = String(r.capability || 'Unclassified');
    const supplierName = String(r.supplier_group || 'Unknown');
    const key = cap + '||' + supplierName;
    const value = Number(r.value || 0);
    if (!capSupplier.has(key)) {{
      capSupplier.set(key, {{supplier_group: supplierName, capability: cap, value: 0, contracts: 0}});
    }}
    const row = capSupplier.get(key);
    row.value += value;
    row.contracts += Number(r.contracts || 0);

    // capability_year_market is repeated once per supplier. Keep one value per capability/FY.
    const marketKey = cap + '||' + fy;
    if (!capYearMarket.has(marketKey)) capYearMarket.set(marketKey, Number(r.capability_year_market || 0));
  }});

  const marketByCap = new Map();
  capYearMarket.forEach((market, key) => {{
    const cap = key.split('||')[0];
    marketByCap.set(cap, (marketByCap.get(cap) || 0) + Number(market || 0));
  }});

  const rows = [];
  capSupplier.forEach(row => {{
    const market = Number(marketByCap.get(row.capability) || 0);
    rows.push({{
      supplier_group: row.supplier_group,
      capability: row.capability,
      value: row.value,
      contracts: row.contracts,
      capability_market: market,
      share_pct: market ? row.value / market * 100 : 0
    }});
  }});

  const capabilities = Array.from(marketByCap.entries())
    .filter(x => Number(x[1] || 0) > 0)
    .sort((a,b) => Number(b[1] || 0) - Number(a[1] || 0))
    .map(x => x[0]);

  return {{range, rows, capabilities, marketByCap}};
}}

function escapeHtml(value) {{
  return String(value == null ? '' : value)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#039;');
}}

function updateCapabilityLeaderboards(supplier, compare) {{
  const data = selectedPeriodLandscapeData();
  const container = document.getElementById('capabilityLeaderboards');
  if (!container) return;

  const cards = data.capabilities.map(cap => {{
    const allRows = data.rows
      .filter(r => r.capability === cap && Number(r.value || 0) > 0)
      .sort((a,b) => Number(b.share_pct || 0) - Number(a.share_pct || 0));

    let visible = allRows.slice(0, 5);
    [supplier, compare].forEach(name => {{
      const row = allRows.find(r => r.supplier_group === name);
      if (row && !visible.some(v => v.supplier_group === name)) visible.push(row);
    }});
    visible = visible.sort((a,b) => Number(b.share_pct || 0) - Number(a.share_pct || 0));
    const maxShare = Math.max(1, ...visible.map(r => Number(r.share_pct || 0)));

    let previousDisplayedRank = 0;
    const rowsHtml = visible.map(r => {{
      const overallRank = allRows.findIndex(x => x.supplier_group === r.supplier_group) + 1;
      const selectedClass = r.supplier_group === supplier ? ' focus-selected' : '';
      const compareClass = r.supplier_group === compare ? ' focus-compare' : '';
      const width = Math.max(1.5, Number(r.share_pct || 0) / maxShare * 100);

      const needsGap = previousDisplayedRank > 0 && overallRank > previousDisplayedRank + 1;
      previousDisplayedRank = overallRank;

      const gapHtml = needsGap
        ? '<div class="leader-gap" aria-label="Additional suppliers omitted">...</div>'
        : '';

      return gapHtml +
        '<div class="leader-row' + selectedClass + compareClass + '" title="' +
        escapeHtml(r.supplier_group + ': ' + pct(r.share_pct) + ', ' + money(r.value) + ', ' + r.contracts + ' contracts') + '">' +
        '<div class="leader-rank">#' + overallRank + '</div>' +
        '<div class="leader-name">' + escapeHtml(r.supplier_group) + '</div>' +
        '<div class="leader-bar-track"><div class="leader-bar-fill" style="width:' + width.toFixed(1) + '%;background:' + companyColour(r.supplier_group) + '"></div></div>' +
        '<div class="leader-share">' + pct(r.share_pct) + '</div>' +
        '</div>';
    }}).join('');

    return '<section class="leaderboard-card">' +
      '<h4>' + escapeHtml(cap) + '</h4>' +
      '<div class="leaderboard-market">' + money(data.marketByCap.get(cap) || 0) + ' addressable market · ' + data.range.label + '</div>' +
      rowsHtml + '</section>';
  }}).join('');

  container.innerHTML = cards || '<div class="leaderboard-card">No supplier data is available for this period.</div>';
}}

function updateRadarChart(supplier, compare) {{
  const colours = coloursForPair(supplier, compare);
  const caps = Array.from(new Set(SUPPLIER_CAPABILITY.map(r => r.capability)));
  const sRows = rowsForSupplier(supplier).sort((a,b)=>Number(b.supplier_capability_value||0)-Number(a.supplier_capability_value||0));
  const selectedCaps = sRows.slice(0, 7).map(r => r.capability);
  const radarCaps = selectedCaps.length ? selectedCaps : caps.slice(0, 7);
  const sMap = new Map(rowsForSupplier(supplier).map(r => [r.capability, r]));
  const cMap = new Map(rowsForSupplier(compare).map(r => [r.capability, r]));
  const theta = radarCaps.concat([radarCaps[0]]);
  const sVals = radarCaps.map(c => Number((sMap.get(c)||{{}}).capability_share_pct || 0));
  const cVals = radarCaps.map(c => Number((cMap.get(c)||{{}}).capability_share_pct || 0));
  Plotly.react('radarChart', [
    {{type:'scatterpolar', r:sVals.concat([sVals[0]||0]), theta:theta, fill:'toself', name:supplier, line:{{color: colours.supplier}}, marker:{{color: colours.supplier, opacity:0.92}}, hovertemplate:'%{{theta}}<br>' + supplier + ' share: %{{r:.1f}}%<extra></extra>'}},
    {{type:'scatterpolar', r:cVals.concat([cVals[0]||0]), theta:theta, fill:'toself', name:compare, line:{{color: colours.compare}}, marker:{{color: colours.compare, opacity:0.92}}, hovertemplate:'%{{theta}}<br>' + compare + ' share: %{{r:.1f}}%<extra></extra>'}}
  ], {{
    title:'Competitive fingerprint by Service Offering share', template:'plotly_white', colorway:[colours.supplier, colours.compare], height:540,
    margin:{{t:70,l:50,r:50,b:50}}, polar:{{radialaxis:{{visible:true, ticksuffix:'%', rangemode:'tozero'}}}}, legend:{{orientation:'h', y:-0.12}}
  }}, {{responsive:true}});
}}

function updateDomainChart(supplier) {{
  const domainTitle = document.getElementById('domainSectionTitle');
  const domainSubtitle = document.getElementById('domainSectionSubtitle');
  if (domainTitle) {{
    domainTitle.textContent = supplier + ' Defence domain breakdown';
  }}
  if (domainSubtitle) {{
    domainSubtitle.textContent =
      supplier + "'s " + selectedGlobalPeriod().label + " contract value grouped into Maritime, Land, Air, Joint, Cyber, Space and Capability Enabler.";
  }}

  const domainOrder = ['Maritime', 'Land', 'Air', 'Cyber', 'Space', 'Joint', 'Capability Enabler'];
  const range = selectedGlobalPeriod();
  const yearSet = new Set(range.years || []);
  const domainRows = DOMAIN_YEARLY.filter(r =>
    r.supplier_group === supplier &&
    (range.key === 'all' || !yearSet.size || yearSet.has(String(r['Financial Year'] || '')))
  );
  const rowsByDomain = new Map();
  domainRows.forEach(r => {{
    const domain = r.defence_domain || 'Joint';
    if (!rowsByDomain.has(domain)) rowsByDomain.set(domain, {{defence_domain:domain, domain_value:0, total_contract_value:0, contracts:0}});
    const row = rowsByDomain.get(domain);
    row.domain_value += Number(r.domain_value || 0);
    row.total_contract_value += Number(r.total_contract_value || 0);
    row.contracts += Number(r.contracts || 0);
  }});

  const rows = domainOrder.map(domain => {{
    const r = rowsByDomain.get(domain) || {{}};
    return {{
      supplier_group: supplier,
      defence_domain: domain,
      domain_value: Number(r.domain_value || 0),
      total_contract_value: Number(r.total_contract_value || 0),
      contracts: Number(r.contracts || 0)
    }};
  }}).filter(r => r.domain_value > 0 || r.contracts > 0);
  const positiveRows = rows.filter(r => r.domain_value > 0);

  if (!positiveRows.length) {{
    Plotly.react('domainChart', [], {{
      title: supplier + ' value by Defence domain',
      template: 'plotly_white', height: 500,
      annotations: [{{text: 'No addressable contracts found for the selected supplier in this period.', x: 0.5, y: 0.5, xref: 'paper', yref: 'paper', showarrow: false}}],
      xaxis: {{visible: false}}, yaxis: {{visible: false}}
    }}, {{responsive:true}});
    document.getElementById('domainTiles').innerHTML = '';
    return;
  }}

  const displayRows = rows.filter(r => r.domain_value > 0);
  const labels = displayRows.map(r => r.defence_domain);
  const values = displayRows.map(r => r.domain_value);
  const colours = labels.map(d => DOMAIN_COLOURS[d] || '#7F7F7F');
  const total = values.reduce((a,b) => a + b, 0);
  const custom = displayRows.map(r => [money(r.domain_value || 0), money(r.total_contract_value || 0), r.contracts || 0, pct(total ? Number(r.domain_value || 0) / total * 100 : 0)]);

  Plotly.react('domainChart', [{{
    type: 'pie',
    labels: labels,
    values: values,
    hole: 0.52,
    sort: false,
    marker: {{colors: colours}},
    textinfo: 'label+percent',
    textposition: 'auto',
    customdata: custom,
    hovertemplate:
      '<b>%{{label}}</b><br>' +
      selectedGlobalPeriod().label + ' value: %{{customdata[0]}}<br>' +
      'Share of selected supplier: %{{customdata[3]}}<br>' +
      'Total contract value: %{{customdata[1]}}<br>' +
      'Contracts: %{{customdata[2]}}' +
      '<extra></extra>'
  }}], {{
    title: supplier + ' value by Defence domain',
    template: 'plotly_white',
    height: 500,
    margin: {{t: 80, l: 40, r: 220, b: 40}},
    legend: {{orientation: 'v', x: 1.02, y: 0.95}},
    annotations: [{{
      text: money(total) + '<br><span style="font-size:12px;color:#526070">' + selectedGlobalPeriod().label + ' value</span>',
      x: 0.5, y: 0.5, showarrow: false, font: {{size: 22, color: '#111827'}}
    }}]
  }}, {{responsive:true}});

  document.getElementById('domainTiles').innerHTML = displayRows.map(r => {{
    const share = total ? (r.domain_value / total * 100) : 0;
    return '<div class="domain-tile">' +
      '<div class="domain-tile-title">' + r.defence_domain + '</div>' +
      '<div class="domain-tile-value">' + money(r.domain_value) + '</div>' +
      '<div class="domain-tile-subtitle">' + pct(share) + ' / ' + r.contracts + ' contracts</div>' +
      '</div>';
  }}).join('');
}}

function updateGrowthChart(supplier, compare, capability) {{
  const colours = coloursForPair(supplier, compare);
  const a = yearlyFor(supplier, capability);
  const b = yearlyFor(compare, capability);
  const years = Array.from(new Set(a.map(r=>r['{FIN_YEAR_COL}']).concat(b.map(r=>r['{FIN_YEAR_COL}'])))).sort();
  const aMap = new Map(a.map(r=>[r['{FIN_YEAR_COL}'], r]));
  const bMap = new Map(b.map(r=>[r['{FIN_YEAR_COL}'], r]));
  const aVals = years.map(y => Number((aMap.get(y)||{{}}).value || 0) / 1e9);
  const bVals = years.map(y => Number((bMap.get(y)||{{}}).value || 0) / 1e9);
  const aShare = years.map(y => Number((aMap.get(y)||{{}}).share_pct || 0));
  const bShare = years.map(y => Number((bMap.get(y)||{{}}).share_pct || 0));
  Plotly.react('growthChart', [
    {{type:'bar', x:years, y:aVals, name:supplier + ' awards', marker:{{color: colours.supplier, opacity:0.92}}, customdata:aShare, hovertemplate:'%{{x}}<br>' + supplier + ': $%{{y:.2f}}B<br>Share: %{{customdata:.1f}}%<extra></extra>'}},
    {{type:'bar', x:years, y:bVals, name:compare + ' awards', marker:{{color: colours.compare, opacity:0.92}}, customdata:bShare, hovertemplate:'%{{x}}<br>' + compare + ': $%{{y:.2f}}B<br>Share: %{{customdata:.1f}}%<extra></extra>'}}
  ], {{
    title: capability + ' awards over time: ' + supplier + ' vs ' + compare,
    template:'plotly_white', colorway:[colours.supplier, colours.compare], height:560, barmode:'group', margin:{{t:80,l:70,r:40,b:90}},
    xaxis:{{title:'Financial year', tickangle:-35}}, yaxis:{{title:'$B'}}, legend:{{orientation:'h', y:-0.18}}
  }}, {{responsive:true}});
}}

function parseDateSafe(value) {{
  if (!value) return null;
  const d = new Date(String(value) + (String(value).length === 10 ? 'T00:00:00' : ''));
  return Number.isNaN(d.getTime()) ? null : d;
}}

function dateLabel(value) {{
  const d = parseDateSafe(value);
  if (!d) return '';
  return d.toLocaleDateString('en-AU', {{day:'2-digit', month:'short', year:'numeric'}});
}}

function daysBetween(a, b) {{
  const da = parseDateSafe(a);
  const db = parseDateSafe(b);
  if (!da || !db) return null;
  return Math.round((db.getTime() - da.getTime()) / 86400000);
}}

function timelineAgency(r) {{
  return String(
    r['Agency Division'] ||
    r['Agency Branch'] ||
    r['Agency'] ||
    'Unknown agency'
  );
}}

function timelineBranch(r) {{
  return String(
    r['Agency Branch'] ||
    r['Agency Division'] ||
    r['Agency'] ||
    'Unknown branch'
  );
}}

function isUnclassifiedContract(r) {{
  const capability = String(r.capability || '').trim().toLowerCase();
  const addressability = String(r.addressability || '').trim().toLowerCase();
  return capability === '' ||
    capability === 'unclassified' ||
    addressability === 'unclassified' ||
    addressability.includes('review');
}}

function updateContractsTimeline(supplier) {{
  const timelineTitle = document.getElementById('timelineTitle');
  const timelineSubtitle = document.getElementById('timelineSubtitle');
  const audit = document.getElementById('timelineAudit');

  const capabilityOrder = new Map([
    ['Strategy, Transformation & Advisory', 0],
    ['SI & Engineering', 1],
    ['Data, AI & Automation', 2],
    ['Cloud Infrastructure & Cyber', 3],
    ['Managed Services & Operations', 4],
    ['Unclassified', 5]
  ]);

  const capabilityColours = {{
    'Strategy, Transformation & Advisory': '#A100FF',
    'SI & Engineering': '#2563EB',
    'Data, AI & Automation': '#16A34A',
    'Cloud Infrastructure & Cyber': '#F59E0B',
    'Managed Services & Operations': '#667085',
    'Unclassified': '#98A2B3'
  }};

  const domainOrder = new Map([
    ['Maritime', 0],
    ['Land', 1],
    ['Air', 2],
    ['Cyber', 3],
    ['Space', 4],
    ['Joint', 5],
    ['Capability Enabler', 6],
    ['Unmapped', 7]
  ]);

  function capabilityName(r) {{
    const value = String(r.capability || '').trim();
    return value || 'Unclassified';
  }}

  function hexToRgba(hex, alpha) {{
    const clean = String(hex || '#98A2B3').replace('#', '');
    const full = clean.length === 3
      ? clean.split('').map(ch => ch + ch).join('')
      : clean;
    const number = parseInt(full, 16);
    const red = (number >> 16) & 255;
    const green = (number >> 8) & 255;
    const blue = number & 255;
    return 'rgba(' + red + ',' + green + ',' + blue + ',' + alpha + ')';
  }}

  const selectedRange = selectedGlobalPeriod();
  const selectedYears = new Set(selectedRange.years || []);
  let supplierRows = CONTRACTS.filter(r =>
    r.supplier_group === supplier &&
    (selectedRange.key === 'all' || !selectedYears.size || selectedYears.has(String(r['Financial Year'] || '')))
  );
  const currentRows = supplierRows.filter(r => {{
    const start = parseDateSafe(r['_contract_start_date_display']);
    const end = parseDateSafe(r['_contract_end_date_display']);
    const asAt = parseDateSafe(AS_AT_DATE);
    return start && end && asAt && start <= asAt && end >= asAt;
  }});

  const invalidDateCount = supplierRows.filter(r =>
    !parseDateSafe(r['_contract_start_date_display']) ||
    !parseDateSafe(r['_contract_end_date_display'])
  ).length;

  let rows = currentRows.slice().sort((a, b) => {{
    const capabilityA = capabilityName(a);
    const capabilityB = capabilityName(b);
    const ca = capabilityOrder.has(capabilityA) ? capabilityOrder.get(capabilityA) : 98;
    const cb = capabilityOrder.has(capabilityB) ? capabilityOrder.get(capabilityB) : 98;
    if (ca !== cb) return ca - cb;

    const capabilityCompare = capabilityA.localeCompare(capabilityB);
    if (capabilityCompare !== 0) return capabilityCompare;

    const da = domainOrder.has(a.defence_domain) ? domainOrder.get(a.defence_domain) : 99;
    const db = domainOrder.has(b.defence_domain) ? domainOrder.get(b.defence_domain) : 99;
    if (da !== db) return da - db;

    const branchCompare = timelineBranch(a).localeCompare(timelineBranch(b));
    if (branchCompare !== 0) return branchCompare;

    return String(a['_contract_end_date_display'] || '').localeCompare(
      String(b['_contract_end_date_display'] || '')
    );
  }});

  const sectionTitle = document.getElementById('timelineSectionTitle');

  if (sectionTitle) {{
    sectionTitle.textContent = 'First Selected Supplier - Current Contracts';
  }}
  if (timelineTitle) {{
    timelineTitle.textContent = supplier + ' current contract timeline';
  }}
  if (timelineSubtitle) {{
    timelineSubtitle.textContent =
      'The first supplier selector controls this view. Showing ' + supplier +
      ' current ongoing contracts as at ' + dateLabel(AS_AT_DATE) +
      ', grouped by Service Offering → Defence domain → branch → project description.';
  }}

  if (!rows.length) {{
    Plotly.react('contractsTimeline', [], {{
      template: 'plotly_white',
      height: 420,
      margin: {{t: 30, l: 40, r: 30, b: 50}},
      annotations: [{{
        text: 'No current contracts with valid start and end dates were found for ' + supplier + '.',
        x: 0.5,
        y: 0.5,
        xref: 'paper',
        yref: 'paper',
        showarrow: false,
        font: {{size: 14, color: '#526070'}}
      }}],
      xaxis: {{visible: false}},
      yaxis: {{visible: false}}
    }}, {{responsive: true, displayModeBar: false}});

    audit.textContent =
      supplierRows.length + ' supplier records in the timeline payload; ' +
      currentRows.length + ' current contracts with valid dates; ' +
      invalidDateCount + ' records excluded because start or end date is missing/invalid.';
    return;
  }}

  const timelineDomainLegend = document.getElementById('timelineDomainLegend');
  if (timelineDomainLegend) {{
    const activeDomains = Array.from(
      new Set(rows.map(r => String(r.defence_domain || 'Unmapped')))
    ).sort((a, b) => {{
      const oa = domainOrder.has(a) ? domainOrder.get(a) : 99;
      const ob = domainOrder.has(b) ? domainOrder.get(b) : 99;
      return oa - ob;
    }});
    timelineDomainLegend.innerHTML =
      '<span style="font-weight:600;color:#344054">Domain colour:</span>' +
      activeDomains.map(domain => {{
        const colour = DOMAIN_COLOURS[domain] || '#7F7F7F';
        return '<span class="timeline-domain-item">' +
          '<span class="timeline-domain-swatch" style="background:' + colour + '"></span>' +
          domain +
          '</span>';
      }}).join('');
  }}

  const capabilityGroups = new Map();
  rows.forEach(r => {{
    const capability = capabilityName(r);
    if (!capabilityGroups.has(capability)) capabilityGroups.set(capability, []);
    capabilityGroups.get(capability).push(r);
  }});

  const orderedCapabilities = Array.from(capabilityGroups.keys()).sort((a, b) => {{
    const oa = capabilityOrder.has(a) ? capabilityOrder.get(a) : 98;
    const ob = capabilityOrder.has(b) ? capabilityOrder.get(b) : 98;
    if (oa !== ob) return oa - ob;
    return a.localeCompare(b);
  }});

  // Assign compact spacing for small capability groups and a little more room
  // only where there are enough contracts to need it.
  const rowItems = [];
  const capabilitySections = [];
  let cursor = rows.length + orderedCapabilities.length * 1.05;

  orderedCapabilities.forEach(capability => {{
    const groupRows = capabilityGroups.get(capability) || [];
    const compactGroup = groupRows.length <= 2;
    const headerY = cursor;
    cursor -= compactGroup ? 0.56 : 0.72;

    const groupItems = [];
    groupRows.forEach(r => {{
      groupItems.push({{row: r, y: cursor}});
      rowItems.push({{row: r, y: cursor}});
      cursor -= compactGroup ? 0.78 : 0.92;
    }});

    const currentValue = groupRows.reduce(
      (sum, r) => sum + Number(r[TIMELINE_VALUE_COL] || 0),
      0
    );
    const remaining = groupRows
      .map(r => daysBetween(AS_AT_DATE, r['_contract_end_date_display']))
      .filter(v => v !== null && isFinite(v));
    const averageRemaining = remaining.length
      ? Math.round(remaining.reduce((sum, v) => sum + v, 0) / remaining.length)
      : 0;

    capabilitySections.push({{
      capability,
      colour: capabilityColours[capability] || '#475467',
      headerY,
      firstY: groupItems.length ? groupItems[0].y : headerY,
      lastY: groupItems.length ? groupItems[groupItems.length - 1].y : headerY,
      contractCount: groupRows.length,
      currentValue,
      averageRemaining
    }});

    cursor -= compactGroup ? 0.38 : 0.52;
  }});

  const compactTimeline = rows.length <= 6;

  const tickVals = rowItems.map(item => item.y);
  const tickText = rowItems.map(item => {{
    const r = item.row;
    const branch = timelineBranch(r) || 'Unknown branch';
    const description = shortText(
      r['Description'] || 'No project description',
      compactTimeline ? 46 : 58
    );
    return '<b>' + branch + '</b><br>' +
      '<span style="color:#667085">' + description + '</span>';
  }});

  const traces = [];
  const shapes = [];
  const annotations = [];
  const shownDomains = new Set();

  // Service Offering section backgrounds, outlines and executive summary headers.
  capabilitySections.forEach(section => {{
    const compactSection = section.contractCount <= 2;
    const yTop = section.headerY + (compactSection ? 0.24 : 0.30);
    const yBottom = section.lastY - (compactSection ? 0.32 : 0.40);

    shapes.push({{
      type: 'rect',
      xref: 'paper',
      x0: 0,
      x1: 1,
      yref: 'y',
      y0: yBottom,
      y1: yTop,
      fillcolor: hexToRgba(section.colour, 0.045),
      line: {{color: hexToRgba(section.colour, 0.55), width: 1.4}},
      layer: 'below'
    }});

    shapes.push({{
      type: 'rect',
      xref: 'paper',
      x0: 0,
      x1: 1,
      yref: 'y',
      y0: section.headerY - (compactSection ? 0.20 : 0.24),
      y1: section.headerY + (compactSection ? 0.20 : 0.24),
      fillcolor: hexToRgba(section.colour, 0.14),
      line: {{color: hexToRgba(section.colour, 0.72), width: 1}},
      layer: 'below'
    }});

    const contractWord = section.contractCount === 1 ? 'contract' : 'contracts';
    annotations.push({{
      x: 0.008,
      y: section.headerY,
      xref: 'paper',
      yref: 'y',
      xanchor: 'left',
      yanchor: 'middle',
      showarrow: false,
      align: 'left',
      text:
        '<b>' + section.capability + '</b>' +
        ' &nbsp; <span style="color:#475467">' +
        money(section.currentValue) + ' current value' +
        ' &nbsp;|&nbsp; ' + section.contractCount + ' ' + contractWord +
        ' &nbsp;|&nbsp; ' + section.averageRemaining + ' avg days remaining' +
        '</span>',
      font: {{size: compactSection ? 11 : 12, color: section.colour}}
    }});
  }});

  // Branch separators inside each capability/domain grouping.
  let previousKey = null;
  rowItems.forEach(item => {{
    const r = item.row;
    const key = capabilityName(r) + '||' +
      String(r.defence_domain || 'Unmapped') + '||' +
      timelineBranch(r);

    if (previousKey !== null && key !== previousKey) {{
      shapes.push({{
        type: 'line',
        xref: 'paper',
        x0: 0,
        x1: 1,
        yref: 'y',
        y0: item.y + 0.5,
        y1: item.y + 0.5,
        line: {{color: '#D0D5DD', width: 1, dash: 'dot'}},
        layer: 'below'
      }});
    }}
    previousKey = key;
  }});

  // Contract traces retain domain colours.
  rowItems.forEach(item => {{
    const r = item.row;
    const domain = String(r.defence_domain || 'Unmapped');
    const colour = DOMAIN_COLOURS[domain] || '#7F7F7F';
    const start = r['_contract_start_date_display'];
    const end = r['_contract_end_date_display'];
    const totalDays = daysBetween(start, end);
    const remainingDays = daysBetween(AS_AT_DATE, end);

    const custom = [
      r['CN ID'] || '',
      r['Description'] || '',
      capabilityName(r),
      r.detailed_capability || '',
      domain,
      r['Agency'] || '',
      r['Agency Division'] || '',
      r['Agency Branch'] || '',
      dateLabel(start),
      dateLabel(end),
      remainingDays == null ? '' : remainingDays,
      money(r['Value'] || 0),
      money(r[TIMELINE_VALUE_COL] || 0),
      Number(r['_extension_count'] || 0),
      r['Financial Year'] || '',
      totalDays == null ? '' : totalDays
    ];

    const hoverTemplate =
      '<b>%{{customdata[0]}}</b><br>' +
      '%{{customdata[1]}}<br><br>' +
      '<b>Service Offering:</b> %{{customdata[2]}}<br>' +
      '<b>Detailed Service Offering:</b> %{{customdata[3]}}<br>' +
      '<b>Domain:</b> %{{customdata[4]}}<br>' +
      '<b>Agency:</b> %{{customdata[5]}}<br>' +
      '<b>Division:</b> %{{customdata[6]}}<br>' +
      '<b>Branch:</b> %{{customdata[7]}}<br>' +
      '<b>Start:</b> %{{customdata[8]}}<br>' +
      '<b>End:</b> %{{customdata[9]}}<br>' +
      '<b>Days remaining:</b> %{{customdata[10]}}<br>' +
      '<b>Total contract value:</b> %{{customdata[11]}}<br>' +
      '<b>Current timeline value:</b> %{{customdata[12]}}<br>' +
      '<b>Amendments/extensions:</b> %{{customdata[13]}}<br>' +
      '<b>FY signed:</b> %{{customdata[14]}}<br>' +
      '<b>Duration:</b> %{{customdata[15]}} days' +
      '<extra></extra>';

    traces.push({{
      type: 'scatter',
      mode: 'lines',
      x: [start, end],
      y: [item.y, item.y],
      name: domain,
      legendgroup: domain,
      showlegend: !shownDomains.has(domain),
      line: {{color: colour, width: compactTimeline ? 6 : 8}},
      customdata: [custom, custom],
      hovertemplate: hoverTemplate
    }});
    shownDomains.add(domain);

    // Wider transparent hover target across the full duration.
    traces.push({{
      type: 'scatter',
      mode: 'lines',
      x: [start, end],
      y: [item.y, item.y],
      name: domain + ' hover area',
      legendgroup: domain,
      showlegend: false,
      line: {{color: 'rgba(0,0,0,0)', width: compactTimeline ? 20 : 25}},
      customdata: [custom, custom],
      hovertemplate: hoverTemplate
    }});

    traces.push({{
      type: 'scatter',
      mode: 'markers',
      x: [start],
      y: [item.y],
      name: domain + ' start',
      legendgroup: domain,
      showlegend: false,
      marker: {{
        size: 8,
        color: '#FFFFFF',
        line: {{color: colour, width: 2}},
        symbol: 'circle'
      }},
      customdata: [custom],
      hovertemplate: hoverTemplate
    }});

    traces.push({{
      type: 'scatter',
      mode: 'markers',
      x: [end],
      y: [item.y],
      name: domain + ' end',
      legendgroup: domain,
      showlegend: false,
      marker: {{size: 9, color: colour, symbol: 'diamond'}},
      customdata: [custom],
      hovertemplate: hoverTemplate
    }});
  }});

  const yMin = Math.min(...tickVals) - 0.65;
  const yMax = Math.max(...capabilitySections.map(section => section.headerY)) + 0.6;

  shapes.push({{
    type: 'line',
    x0: AS_AT_DATE,
    x1: AS_AT_DATE,
    y0: yMin,
    y1: yMax,
    xref: 'x',
    yref: 'y',
    line: {{color: '#111827', width: 2, dash: 'dash'}}
  }});

  const longestLabelLength = tickText.reduce(
    (maxLength, label) => Math.max(maxLength, String(label || '').length),
    0
  );
  const chartHeight = compactTimeline
    ? Math.max(320, rows.length * 46 + orderedCapabilities.length * 42 + 82)
    : Math.max(450, rows.length * 48 + orderedCapabilities.length * 50 + 105);

  const leftMargin = compactTimeline
    ? Math.min(470, Math.max(330, longestLabelLength * 5.5 + 55))
    : Math.min(540, Math.max(410, longestLabelLength * 5.8 + 65));

  Plotly.react('contractsTimeline', traces, {{
    template: 'plotly_white',
    height: chartHeight,
    margin: {{t: compactTimeline ? 24 : 30, l: leftMargin, r: 30, b: 60}},
    hovermode: 'closest',
    xaxis: {{
      title: 'Contract start date → contract end date',
      type: 'date',
      showgrid: true,
      gridcolor: '#EEF2F7',
      fixedrange: true
    }},
    yaxis: {{
      tickmode: 'array',
      tickvals: tickVals,
      ticktext: tickText,
      range: [yMin, yMax],
      fixedrange: true,
      automargin: true,
      ticklabelstandoff: 18,
      tickfont: {{size: compactTimeline ? 11 : 12}},
      showgrid: false,
      zeroline: false
    }},
    showlegend: false,
    shapes: shapes,
    annotations: annotations
  }}, {{
    responsive: true,
    displaylogo: false,
    displayModeBar: false,
    scrollZoom: false,
    doubleClick: false
  }});

  audit.textContent =
    rows.length + ' current contracts displayed across ' +
    orderedCapabilities.length + ' Service Offerings; ' +
    invalidDateCount + ' supplier records excluded because start or end date is missing/invalid.';
}}

function renderAll() {{
  const supplier = document.getElementById('supplierSelect').value;
  const compare = document.getElementById('compareSelect').value;

  updateSupplierColours(supplier, compare);
  updateKpis(supplier, compare);
  updateProfileChart(supplier, compare);
  updateDomainChart(supplier);
  updateCapabilityLeaderboards(supplier, compare);
  updateContractsTimeline(supplier);
}}

document.getElementById('supplierSelect').value = DEFAULT_SUPPLIER;
document.getElementById('compareSelect').value = DEFAULT_COMPARE;

const originalSelectors = document.getElementById('originalSupplierSelectors');
const floatingSelectors = document.getElementById('floatingSupplierSelectors');
const supplierSelect = document.getElementById('supplierSelect');
const compareSelect = document.getElementById('compareSelect');
const floatingSupplierSelect = document.getElementById('floatingSupplierSelect');
const floatingCompareSelect = document.getElementById('floatingCompareSelect');
const periodSelect = document.getElementById('periodSelect');
const floatingPeriodSelect = document.getElementById('floatingPeriodSelect');

function syncFloatingSelectorsFromOriginal() {{
  if (floatingSupplierSelect && supplierSelect) {{
    floatingSupplierSelect.value = supplierSelect.value;
  }}
  if (floatingCompareSelect && compareSelect) {{
    floatingCompareSelect.value = compareSelect.value;
  }}
  if (floatingPeriodSelect && periodSelect) {{
    floatingPeriodSelect.value = periodSelect.value;
  }}
}}

function setFloatingSelectorsVisible(visible) {{
  if (!floatingSelectors) return;
  floatingSelectors.classList.toggle('is-visible', visible);
  floatingSelectors.setAttribute('aria-hidden', visible ? 'false' : 'true');
}}

function updateFloatingSelectorVisibility() {{
  if (!originalSelectors) return;
  const rect = originalSelectors.getBoundingClientRect();

  // Show only after the original selector panel has completely passed above
  // the viewport. Hide it again when the user scrolls back to the original.
  setFloatingSelectorsVisible(rect.bottom <= 10);
}}

if (supplierSelect) {{
  supplierSelect.addEventListener('change', () => {{
    syncFloatingSelectorsFromOriginal();
    renderAll();
  }});
}}

if (compareSelect) {{
  compareSelect.addEventListener('change', () => {{
    syncFloatingSelectorsFromOriginal();
    renderAll();
  }});
}}

if (floatingSupplierSelect) {{
  floatingSupplierSelect.addEventListener('change', () => {{
    if (supplierSelect) supplierSelect.value = floatingSupplierSelect.value;
    renderAll();
  }});
}}

if (floatingCompareSelect) {{
  floatingCompareSelect.addEventListener('change', () => {{
    if (compareSelect) compareSelect.value = floatingCompareSelect.value;
    renderAll();
  }});
}}


if (periodSelect) {{
  periodSelect.addEventListener('change', () => {{
    syncFloatingSelectorsFromOriginal();
    renderAll();
  }});
}}

if (floatingPeriodSelect) {{
  floatingPeriodSelect.addEventListener('change', () => {{
    if (periodSelect) periodSelect.value = floatingPeriodSelect.value;
    renderAll();
  }});
}}
let floatingScrollFrame = null;
window.addEventListener('scroll', () => {{
  if (floatingScrollFrame !== null) return;
  floatingScrollFrame = window.requestAnimationFrame(() => {{
    floatingScrollFrame = null;
    updateFloatingSelectorVisibility();
  }});
}}, {{ passive: true }});

window.addEventListener('resize', updateFloatingSelectorVisibility);

buildGlobalPeriodOptions();
buildShareRangeControls();
syncFloatingSelectorsFromOriginal();
updateFloatingSelectorVisibility();
renderAll();
</script>

<div id="execTooltip" class="exec-tooltip" role="tooltip" aria-hidden="true"></div>
<script>
(function() {{
  const tooltip = document.getElementById('execTooltip');
  if (!tooltip) return;

  let activeTarget = null;

  function placeTooltip(target) {{
    if (!activeTarget || activeTarget !== target) return;

    const rect = target.getBoundingClientRect();
    const tipRect = tooltip.getBoundingClientRect();
    const gap = 10;
    const edge = 12;

    let left = rect.left + (rect.width / 2) - (tipRect.width / 2);
    left = Math.max(edge, Math.min(left, window.innerWidth - tipRect.width - edge));

    let top = rect.top - tipRect.height - gap;
    if (top < edge) {{
      top = rect.bottom + gap;
    }}
    top = Math.max(edge, Math.min(top, window.innerHeight - tipRect.height - edge));

    tooltip.style.left = Math.round(left) + 'px';
    tooltip.style.top = Math.round(top) + 'px';
  }}

  function showTooltip(target) {{
    const text = target.getAttribute('data-tooltip');
    if (!text) return;

    activeTarget = target;
    tooltip.textContent = text;
    tooltip.classList.add('is-visible');
    tooltip.setAttribute('aria-hidden', 'false');

    requestAnimationFrame(() => placeTooltip(target));
  }}

  function hideTooltip(target) {{
    if (target && activeTarget !== target) return;
    activeTarget = null;
    tooltip.classList.remove('is-visible');
    tooltip.setAttribute('aria-hidden', 'true');
  }}

  document.querySelectorAll('.executive-summary strong[data-tooltip], .insight-box strong[data-tooltip]').forEach(target => {{
    target.addEventListener('mouseenter', () => showTooltip(target));
    target.addEventListener('mouseleave', () => hideTooltip(target));
    target.addEventListener('focus', () => showTooltip(target));
    target.addEventListener('blur', () => hideTooltip(target));
  }});

  window.addEventListener('scroll', () => {{
    if (activeTarget) placeTooltip(activeTarget);
  }}, {{ passive: true }});

  window.addEventListener('resize', () => {{
    if (activeTarget) placeTooltip(activeTarget);
  }});
}})();
</script>

</body>
</html>
"""
    output_html.write_text(html_doc, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_all = load_or_classify(args)
    value_col = get_value_col(df_all, args.value_mode)
    df_all = clean_value_columns(df_all, [VALUE_COL, ANNUALISED_VALUE_COL])
    df_all = filter_financial_year_range(df_all, args.fy_start, args.fy_end)

    # Full selected-period Defence data drives KPIs, market totals, rankings,
    # capability charts, leaderboards and domain analysis.
    df = filter_defence_scope(
        df_all,
        include_all_agencies=args.include_all_agencies,
    )
    model = prepare_competitive_model(df, value_col, args.top_n_suppliers)

    # Only the timeline is restricted to contracts active as at the selected date.
    # Use the model's already-addressable, domain-mapped dataset so timeline
    # classification is consistent with the rest of the dashboard.
    as_at = parse_as_at(args.as_at)
    timeline_value_col = current_value_col_from_mode(args.current_value_mode)
    timeline_contracts, current_metadata = add_current_contract_value_columns(
        model["data"],  # type: ignore[arg-type]
        base_value_col=VALUE_COL,
        as_at=as_at,
    )
    model["timeline_contracts"] = timeline_contracts
    model["current_metadata"] = current_metadata

    export_defence_scope_audit(df_all, df, value_col, output_dir)

    supplier_summary: pd.DataFrame = model["supplier_summary"]  # type: ignore[assignment]
    supplier_capability: pd.DataFrame = model["supplier_capability"]  # type: ignore[assignment]
    yearly: pd.DataFrame = model["supplier_capability_yearly"]  # type: ignore[assignment]
    top_contracts: pd.DataFrame = model["top_contracts"]  # type: ignore[assignment]
    domain_summary: pd.DataFrame = model["domain_summary"]  # type: ignore[assignment]
    domain_yearly: pd.DataFrame = model["domain_yearly"]  # type: ignore[assignment]

    supplier_summary.to_csv(output_dir / "supplier_summary.csv", index=False)
    supplier_capability.to_csv(output_dir / "supplier_capability.csv", index=False)
    yearly.to_csv(output_dir / "supplier_capability_yearly.csv", index=False)
    top_contracts.head(args.top_n_contracts).to_csv(output_dir / "top_contracts.csv", index=False)
    timeline_contracts.to_csv(output_dir / "current_timeline_contracts.csv", index=False)
    domain_summary.to_csv(output_dir / "supplier_domain_summary.csv", index=False)
    domain_yearly.to_csv(output_dir / "supplier_domain_yearly.csv", index=False)

    dashboard_path = output_dir / "CompareDashboard.html"
    build_html(
        model,
        value_col,
        timeline_value_col,
        dashboard_path,
        args.value_mode,
        format_fy_range_label(args.fy_start, args.fy_end),
        as_at.strftime("%Y-%m-%d"),
    )

    print("Competitive positioning dashboard complete")
    print(f"Dashboard value mode: {args.value_mode} / column used: {value_col}")
    print(f"Selected period: {format_fy_range_label(args.fy_start, args.fy_end)}")
    print(f"Timeline as-at date: {as_at.strftime('%Y-%m-%d')}")
    print(f"Timeline value mode: {args.current_value_mode} / column used: {timeline_value_col}")
    print(f"Rows after Defence filter for full dashboard: {len(df):,}")
    print(f"Current addressable rows in timeline: {len(timeline_contracts):,}")
    print(f"Addressable TAM in scope: {money(float(model['total_tam']))}")
    print(f"Suppliers in comparison dropdowns: {len(model['supplier_options'])}")
    print(f"Wrote: {dashboard_path}")
    print(f"Wrote: {output_dir / 'supplier_summary.csv'}")
    print(f"Wrote: {output_dir / 'supplier_capability.csv'}")
    print(f"Wrote: {output_dir / 'supplier_capability_yearly.csv'}")
    print(f"Wrote: {output_dir / 'top_contracts.csv'}")
    print(f"Wrote: {output_dir / 'current_timeline_contracts.csv'}")
    print(f"Wrote: {output_dir / 'supplier_domain_summary.csv'}")
    print(f"Wrote: {output_dir / 'supplier_domain_yearly.csv'}")


if __name__ == "__main__":
    main()