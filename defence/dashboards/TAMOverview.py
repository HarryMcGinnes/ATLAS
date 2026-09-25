from __future__ import annotations

import argparse
import hashlib
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots



VALUE_COL = "Value"
ANNUALISED_VALUE_COL = "Value Per Year"
SUPPLIER_COL = "Supplier Name"
SUPPLIER_ABN_COL = "Supplier ABN"
DESCRIPTION_COL = "Description"
CATEGORY_COL = "Category"
CATEGORY_TYPE_COL = "Category Type"
FIN_YEAR_COL = "Financial Year"
CN_ID_COL = "CN ID"
AGENCY_COL = "Agency"
AGENCY_BRANCH_COL = "Agency Branch"

COMPETITOR_FOCUS = [
    "Accenture",
    "Deloitte",
    "KPMG",
    "EY",
    "Leidos",
    "Lockheed",
    "Data#3",
    "Aurecon",
]

COMPETITOR_EXPANDED = COMPETITOR_FOCUS + [
    "PwC",
    "DXC",
    "IBM",
    "Microsoft",
    "Amazon / AWS",
    "Oracle",
    "SAP",
    "BAE",
    "Thales",
    "Fujitsu",
    "Aurecon",
]



# Dashboard 1 fixed rival universe supplied by the business. ABN is the
# authoritative match where available; supplier-name aliases are the fallback.
RIVAL_ABN_TO_NAME = {
    "79612590155": "Leidos",
    "29008423005": "BAE",
    "39613308008": "Nova Defence",
    "91007660317": "Kellogg Brown & Root",
    "51194660183": "KPMG",
    "30008425509": "Lockheed",
    "79000024733": "IBM",
    "64006678119": "Boeing",
    "19001011427": "Fujitsu",
    "12079749287": "Jacobs",
    "49096776895": "Accenture",
    "33051775556": "Telstra",
    "31010545267": "Fujitsu",
    "51006870846": "BAE",
    "74490121060": "Deloitte",
    "53000983700": "Downer",
    "18008476944": "DXC",
    "51106981560": "SME Gateway",
    "66008642751": "Thales",
    "35063709295": "Raytheon",
    "34129384032": "Downer",
    "75288172749": "EY",
    "46003855561": "Dell",
    "20607773295": "PwC",
    "90155020303": "Navantia",
    "65119369827": "Synergy Group",
    "68125805647": "Qinetiq",
    "64008605034": "ASC",
    "57195873179": "University of New South Wales",
    "31143526229": "Elbit Systems",
    "56609386156": "Servegate Australia",
    "33163511304": "Nova Defence",
    "97072941943": "Kinetic IT",
    "95820883147": "Projects Assured",
    "64086174781": "Telstra",
    "88122798207": "Cubic Defence",
    "99059951183": "CEA Technologies",
    "80003074468": "Oracle",
    "54005139873": "Aurecon",
    "52119884945": "Whizdom",
}

RIVAL_DISPLAY_ORDER = list(dict.fromkeys(RIVAL_ABN_TO_NAME.values()))

RIVAL_NAME_ALIASES = {
    "accenture": "Accenture",
    "accenture australia": "Accenture",
    "bae": "BAE",
    "bae systems": "BAE",
    "boeing": "Boeing",
    "boeing australia": "Boeing",
    "deloitte": "Deloitte",
    "dell": "Dell",
    "dell australia": "Dell",
    "downer": "Downer",
    "downer edi": "Downer",
    "dxc": "DXC",
    "dxc technology australia": "DXC",
    "ernst & young": "EY",
    "ey": "EY",
    "fujitsu": "Fujitsu",
    "fujitsu australia": "Fujitsu",
    "ibm": "IBM",
    "ibm australia": "IBM",
    "jacobs": "Jacobs",
    "jacobs australia": "Jacobs",
    "kbr": "Kellogg Brown & Root",
    "kellogg brown & root": "Kellogg Brown & Root",
    "kpmg": "KPMG",
    "kpmg australia": "KPMG",
    "leidos": "Leidos",
    "leidos australia": "Leidos",
    "lockheed": "Lockheed",
    "lockheed martin": "Lockheed",
    "oracle": "Oracle",
    "oracle australia": "Oracle",
    "pwc": "PwC",
    "pwc australia": "PwC",
    "pricewaterhousecoopers": "PwC",
    "raytheon": "Raytheon",
    "telstra": "Telstra",
    "thales": "Thales",
    "thales adi": "Thales",
    "aurecon": "Aurecon",
    "aurecon australasia": "Aurecon",
}

RIVAL_COLOURS = {
    "Accenture": "#A100FF",
    "Deloitte": "#86BC25",
    "KPMG": "#00338D",
    "EY": "#FFE600",
    "PwC": "#E0301E",
    "Leidos": "#667785",
    "IBM": "#1F70C1",
    "Oracle": "#C74634",
    "Fujitsu": "#D6001C",
    "Aurecon": "#2E7D32",
}

REQUIRED_CLASSIFICATION_COLS = [
    "capability",
    "service_line",
    "is_addressable",
    "confidence",
    "matched_terms",
    "supplier_group",
    "is_accenture",
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
    parser.add_argument("--output-dir", default="DefenceTAMDashboard_output")
    parser.add_argument(
        "--value-mode",
        choices=["total", "annualised"],
        default="total",
        help=(
            "total = full contract ceiling/value over the contract life. "
            "annualised = Value Per Year, useful for yearly run-rate comparisons."
        ),
    )
    parser.add_argument("--top-n-suppliers", type=int, default=12)
    parser.add_argument("--top-n-capabilities", type=int, default=18)
    parser.add_argument(
        "--include-all-agencies",
        action="store_true",
        help="Compatibility option only. The canonical master input is already Defence-only.",
    )
    parser.add_argument(
        "--fy-start",
        help=(
            "Optional first financial year to include, e.g. 2021-2022. "
            "Use the same format as the Financial Year column."
        ),
    )
    parser.add_argument(
        "--fy-end",
        help=(
            "Optional final financial year to include, e.g. 2025-2026. "
            "Use the same format as the Financial Year column."
        ),
    )
    parser.add_argument(
        "--refresh-audits",
        action="store_true",
        help=(
            "Force regeneration of supplier, classification and market-relevance "
            "audit files. By default, valid cached audits are reused."
        ),
    )
    return parser.parse_args()


def money(value: float) -> str:
    if pd.isna(value):
        return "$0"
    abs_value = abs(float(value))
    if abs_value >= 1_000_000_000:
        return f"${value / 1_000_000_000:,.2f}B"
    if abs_value >= 1_000_000:
        return f"${value / 1_000_000:,.1f}M"
    if abs_value >= 1_000:
        return f"${value / 1_000:,.1f}K"
    return f"${value:,.0f}"




def short_money(value: float) -> str:
    """Human-friendly currency formatting for chart labels.

    Examples:
      18,000,000,000 -> $18.00B
      230,000,000    -> $230M
      23,400,000     -> $23.4M
      2,340,000      -> $2.34M
      950,000        -> $950K
    """
    if pd.isna(value):
        return "$0"

    value = float(value)
    abs_value = abs(value)

    if abs_value >= 1_000_000_000:
        return f"${value / 1_000_000_000:,.2f}B"
    if abs_value >= 100_000_000:
        return f"${value / 1_000_000:,.0f}M"
    if abs_value >= 10_000_000:
        return f"${value / 1_000_000:,.1f}M"
    if abs_value >= 1_000_000:
        return f"${value / 1_000_000:,.2f}M"
    if abs_value >= 1_000:
        return f"${value / 1_000:,.0f}K"
    return f"${value:,.0f}"


def pct(value: float) -> str:
    if pd.isna(value):
        return "0.0%"
    return f"{value * 100:,.1f}%"


def to_billions(series: pd.Series) -> pd.Series:
    return series / 1_000_000_000


# Canonical supplier identities used by every Dashboard 1 aggregation.
# ABN is authoritative where available. Cleaned supplier-name rules are the
# fallback so legal-entity and spelling variants roll into one supplier.
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
    return re.sub(r"[^0-9]", "", str(value)).lstrip("0")


def _normalise_supplier_name(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9+#]+", " ", text)
    text = re.sub(
        r"\b(?:pty|proprietary|limited|ltd|incorporated|inc|holdings|"
        r"australia|australian|asia pacific|apac)\b",
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
    (r"\bsaab\b", "SAAB"),
    (r"\bnec\b", "NEC"),
    (r"\blockheed martin\b|\blockheed\b", "Lockheed"),
    (r"\bnorthrop grumman\b|\bnorthrop\b", "Northrop Grumman"),
    (r"\braytheon\b", "Raytheon"),
    (r"\bcapgemini\b", "Capgemini"),
    (r"\b(?:cgi|cgi federal)\b", "CGI"),
    (r"\bbooz allen\b", "Booz Allen"),
    (r"\bjacobs\b", "Jacobs"),
    (r"\baurecon\b|\baugility\b", "Aurecon"),
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
    """Apply one supplier identity to every Dashboard 1 calculation."""
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
]


def _norm_code(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper().replace(" ", "")


def _contains_defence_text(value: object) -> bool:
    text = str(value or "").lower()
    return any(term in text for term in DEFENCE_TEXT_TERMS)


def filter_defence_scope(df: pd.DataFrame, include_all_agencies: bool = False) -> pd.DataFrame:
    """Return the canonical Defence dataset unchanged.

    Defence scope is established only in build_master.py by
    exact Agency allow-list before domain or capability classification.
    """
    if include_all_agencies:
        print("Warning: --include-all-agencies is ignored for canonical master inputs.")
    return df.copy()


def export_defence_scope_audit(original_df: pd.DataFrame, filtered_df: pd.DataFrame, value_col: str, output_dir: Path) -> None:
    """Write a small audit showing which agencies were included/excluded."""
    if AGENCY_COL not in original_df.columns:
        return

    original = original_df.copy()
    original["_scope"] = "Excluded"
    original.loc[filtered_df.index.intersection(original.index), "_scope"] = "Included Defence scope"

    audit = (
        original.groupby(["_scope", AGENCY_COL], dropna=False)
        .agg(value=(value_col, "sum"), contracts=(value_col, "size"))
        .reset_index()
        .sort_values(["_scope", "value"], ascending=[True, False])
    )
    audit.to_csv(output_dir / "defence_scope_audit_by_agency.csv", index=False)


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



def _financial_year_start_year(value: object) -> int | None:
    """Extract the first calendar year from a Financial Year value.

    Examples:
      "2024-2025" -> 2024
      "2024/25" -> 2024
      "FY2024-25" -> 2024
    """
    if pd.isna(value):
        return None
    text = str(value).strip()
    import re
    match = re.search(r"(20\d{2}|19\d{2})", text)
    if not match:
        return None
    return int(match.group(1))


def _format_fy_range_label(fy_start: str | None, fy_end: str | None) -> str:
    if fy_start and fy_end:
        return f"{fy_start} to {fy_end}"
    if fy_start:
        return f"{fy_start} onwards"
    if fy_end:
        return f"up to {fy_end}"
    return "whole dataset period"


def filter_financial_year_range(df: pd.DataFrame, fy_start: str | None, fy_end: str | None) -> pd.DataFrame:
    """Filter the dataset to an inclusive Financial Year range."""
    if not fy_start and not fy_end:
        return df
    if FIN_YEAR_COL not in df.columns:
        raise SystemExit(f"Cannot filter by financial year because column is missing: {FIN_YEAR_COL}")

    start_year = _financial_year_start_year(fy_start) if fy_start else None
    end_year = _financial_year_start_year(fy_end) if fy_end else None
    if fy_start and start_year is None:
        raise SystemExit(f"Could not parse --fy-start: {fy_start}")
    if fy_end and end_year is None:
        raise SystemExit(f"Could not parse --fy-end: {fy_end}")
    if start_year is not None and end_year is not None and start_year > end_year:
        raise SystemExit("--fy-start must be earlier than or equal to --fy-end")

    fy_year = df[FIN_YEAR_COL].map(_financial_year_start_year)
    mask = pd.Series(True, index=df.index)
    if start_year is not None:
        mask &= fy_year >= start_year
    if end_year is not None:
        mask &= fy_year <= end_year

    filtered = df[mask].copy()
    if filtered.empty:
        available = sorted(df[FIN_YEAR_COL].dropna().astype(str).unique().tolist())
        raise SystemExit(
            "Financial year filter returned no rows. "
            f"Requested: {_format_fy_range_label(fy_start, fy_end)}. "
            f"Available values include: {available[:5]} ... {available[-5:]}"
        )
    return filtered

def year_count(df: pd.DataFrame) -> int:
    if FIN_YEAR_COL not in df.columns:
        return 1
    return max(1, int(df[FIN_YEAR_COL].dropna().astype(str).nunique()))


def summarise(df: pd.DataFrame, value_col: str) -> dict[str, float | int | str]:
    total = df[value_col].sum()
    addressable = df.loc[df["is_addressable"], value_col].sum()
    not_addressable = total - addressable
    accenture_total = df.loc[df["is_accenture"], value_col].sum()
    accenture_addressable = df.loc[df["is_addressable"] & df["is_accenture"], value_col].sum()
    competitor_market = addressable - accenture_addressable
    years = year_count(df)

    return {
        "value_col": value_col,
        "years": years,
        "total": total,
        "addressable": addressable,
        "not_addressable": not_addressable,
        "accenture_total": accenture_total,
        "accenture_addressable": accenture_addressable,
        "competitor_market": competitor_market,
        "addressable_pct_total": addressable / total if total else 0,
        "accenture_share_of_addressable": accenture_addressable / addressable if addressable else 0,
        "competitor_market_share": competitor_market / addressable if addressable else 0,
        "avg_total_per_year": total / years,
        "avg_addressable_per_year": addressable / years,
        "avg_accenture_addressable_per_year": accenture_addressable / years,
        "avg_competitor_market_per_year": competitor_market / years,
    }


def add_billion_col(df: pd.DataFrame, value_col: str, out_col: str = "value_b") -> pd.DataFrame:
    out = df.copy()
    out[out_col] = out[value_col] / 1_000_000_000
    return out




def _donut_financial_year_options(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """FY slicers for the overview donuts: All years, Last 5 FY, Last 3 FY."""
    if FIN_YEAR_COL not in df.columns:
        return [("All years", df)]

    tmp = df.copy()
    tmp["_fy_start_year"] = tmp[FIN_YEAR_COL].map(_financial_year_start_year)
    fy = (
        tmp[[FIN_YEAR_COL, "_fy_start_year"]]
        .dropna()
        .drop_duplicates()
        .sort_values("_fy_start_year")
    )
    fy_labels = fy[FIN_YEAR_COL].astype(str).tolist()
    if not fy_labels:
        return [("All years", df)]

    options: list[tuple[str, pd.DataFrame]] = [("All years", df)]

    if len(fy_labels) >= 5:
        last5 = fy_labels[-5:]
        options.append((
            f"Last 5 FY ({last5[0]} to {last5[-1]})",
            tmp[tmp[FIN_YEAR_COL].astype(str).isin(last5)].drop(columns=["_fy_start_year"]),
        ))

    if len(fy_labels) >= 3:
        last3 = fy_labels[-3:]
        options.append((
            f"Last 3 FY ({last3[0]} to {last3[-1]})",
            tmp[tmp[FIN_YEAR_COL].astype(str).isin(last3)].drop(columns=["_fy_start_year"]),
        ))

    return options


def chart_donut_overview_interactive(df: pd.DataFrame, value_col: str) -> str:
    """Render the TAM progression as one wide Plotly chart.

    Using one canvas guarantees identical donut diameters, aligned centres,
    unclipped centre text and precisely positioned arrows. The highlighted
    subset slice in the first two donuts is centred at 3 o'clock so it points
    directly toward the next stage.
    """
    capability_colours = {
        "Strategy, Transformation & Advisory": "#4C1D95",
        "SI & Engineering": "#6D28D9",
        "Cloud Infrastructure & Cyber": "#8B5CF6",
        "Managed Services & Operations": "#A78BFA",
        "Data, AI & Automation": "#C4B5FD",
        "Unclassified": "#DDD6FE",
    }

    options = []
    seen = set()
    for label, subset in _donut_financial_year_options(df):
        if label in seen:
            continue
        seen.add(label)
        s = summarise(subset, value_col)

        total = float(s["total"] or 0)
        addressable = float(s["addressable"] or 0)
        not_addressable = float(s["not_addressable"] or 0)
        competitor_market = float(s["competitor_market"] or 0)
        accenture_wins = float(s["accenture_addressable"] or 0)

        accenture_rows = subset[
            subset["is_addressable"].fillna(False).astype(bool)
            & subset["is_accenture"].fillna(False).astype(bool)
        ].copy()
        accenture_rows["capability"] = (
            accenture_rows["capability"]
            .fillna("Unclassified")
            .replace("", "Unclassified")
            .astype(str)
        )
        capability_values = (
            accenture_rows.groupby("capability", dropna=False)[value_col]
            .sum()
            .sort_values(ascending=False)
        )
        capability_rows = []
        for capability, value in capability_values.items():
            value = float(value or 0)
            if value <= 0:
                continue
            capability_rows.append({
                "capability": str(capability),
                "value": value,
                "value_label": money(value),
                "share_pct": (value / accenture_wins * 100) if accenture_wins else 0,
                "colour": capability_colours.get(str(capability), "#7F56D9"),
            })

        options.append({
            "label": label,
            "total": total,
            "addressable": addressable,
            "not_addressable": not_addressable,
            "competitor_market": competitor_market,
            "accenture_wins": accenture_wins,
            "addressable_pct_total": (addressable / total * 100) if total else 0,
            "not_addressable_pct_total": (not_addressable / total * 100) if total else 0,
            "competitor_share_of_addressable": (competitor_market / addressable * 100) if addressable else 0,
            "accenture_share_of_addressable": (accenture_wins / addressable * 100) if addressable else 0,
            "addressable_label": money(addressable),
            "not_addressable_label": money(not_addressable),
            "competitor_label": money(competitor_market),
            "accenture_label": money(accenture_wins),
            "capabilities": capability_rows,
        })

    chart_data = json.dumps(options)
    controls = []
    for i, opt in enumerate(options):
        checked = " checked" if i == 0 else ""
        controls.append(
            f'<label class="range-pill"><input type="radio" name="donutRange" value="{i}"{checked}> {opt["label"]}</label>'
        )
    controls_html = "\n".join(controls)

    return f"""
<div class="donut-overview wide-donut-overview">
  <div class="capability-range-header">
    <h3>Total Addressable Market Overview</h3>
  </div>
  <div class="range-controls">{controls_html}</div>

  <div class="wide-donut-panel">
    <div class="wide-donut-headings" aria-hidden="true">
      <div><b>1. Total Defence Procurement Market</b><span>Out of total Defence procurement</span></div>
      <div><b>2. Within Addressable Market (TAM)</b><span>Breakdown of addressable TAM</span></div>
      <div><b>3. Breakdown of Accenture Wins</b><span>Distribution across service offerings</span></div>
    </div>
    <div class="wide-chart-stage">
      <div id="wideDonutFlowChart" class="wide-donut-chart"></div>
      <div id="transitionOne" class="donut-transition donut-transition-one" aria-hidden="true"></div>
      <div id="transitionTwo" class="donut-transition donut-transition-two" aria-hidden="true"></div>
    </div>
    <div class="wide-donut-legends">
      <div id="wideLegendOne" class="wide-stage-legend"></div>
      <div id="wideLegendTwo" class="wide-stage-legend"></div>
      <div id="wideLegendThree" class="wide-stage-legend wide-stage-legend-capabilities"></div>
    </div>
  </div>

  <div class="donut-note"><b>Note:</b> Figures may not sum precisely to 100% due to rounding. The first donut shows the full Defence procurement market and highlights the addressable portion; the second treats that addressable portion as the full market and highlights Accenture wins; the third breaks Accenture wins down by service offering.</div>
</div>

<script>
(function() {{
  const datasets = {chart_data};
  const chartId = 'wideDonutFlowChart';

  // Equal donut diameters with intentionally wider inter-stage corridors.
  // The corridors carry the subset label and arrow without touching either donut.
  const DOMAINS = [
    {{x: [0.040, 0.290], y: [0.255, 0.885]}},
    {{x: [0.375, 0.625], y: [0.255, 0.885]}},
    {{x: [0.710, 0.960], y: [0.255, 0.885]}}
  ];
  const CENTRES = [0.165, 0.500, 0.835];

  function pctLabel(v) {{
    return Number(v || 0).toFixed(1) + '%';
  }}

  function moneyLabel(v) {{
    const value = Number(v || 0);
    const absValue = Math.abs(value);
    if (absValue >= 1e9) return '$' + (value / 1e9).toFixed(2) + 'B';
    if (absValue >= 1e6) return '$' + (value / 1e6).toFixed(1) + 'M';
    if (absValue >= 1e3) return '$' + (value / 1e3).toFixed(1) + 'K';
    return '$' + value.toFixed(0);
  }}

  function focusRotation(focus, total) {{
    const angle = total > 0 ? (focus / total) * 360 : 0;
    return 90 - (angle / 2);
  }}

  function makeDonut(rows, domain, rotation, pullFirst) {{
    return {{
      type: 'pie',
      hole: 0.64,
      sort: false,
      direction: 'clockwise',
      rotation: rotation,
      labels: rows.map(r => r.name),
      values: rows.map(r => r.value),
      domain: domain,
      textinfo: 'none',
      hovertemplate: '<b>%{{label}}</b><br>Value: %{{value:$,.0f}}<br>Share: %{{percent}}<extra></extra>',
      marker: {{
        colors: rows.map(r => r.colour),
        line: {{color: '#ffffff', width: 3}}
      }},
      pull: rows.map((r, i) => pullFirst && i === 0 ? 0.045 : 0),
      showlegend: false
    }};
  }}

  function centreAnnotations(x, value, label) {{
    return [
      {{
        x: x, y: 0.575, xref: 'paper', yref: 'paper',
        text: '<b>' + value + '</b>', showarrow: false,
        xanchor: 'center', yanchor: 'middle', align: 'center',
        font: {{size: 27, color: '#0B2545', family: 'Arial Black'}}
      }},
      {{
        x: x, y: 0.465, xref: 'paper', yref: 'paper',
        text: label, showarrow: false,
        xanchor: 'center', yanchor: 'middle', align: 'center',
        font: {{size: 11, color: '#526070'}}
      }}
    ];
  }}

  function legendItem(name, share, value, colour) {{
    return '<div class="wide-legend-item">' +
      '<span class="wide-legend-swatch" style="background:' + colour + '"></span>' +
      '<span><b>' + name + '</b><small>' + pctLabel(share) + ' · ' + value + '</small></span>' +
      '</div>';
  }}

  function render(index) {{
    const d = datasets[index] || datasets[0];
    const capabilities = (d.capabilities || []).filter(r => Number(r.value || 0) > 0);

    const stageOneRows = [
      {{name: 'Addressable (TAM)', value: d.addressable, colour: '#4C1D95'}},
      {{name: 'Not addressable', value: d.not_addressable, colour: '#D8CCF3'}}
    ];
    const stageTwoRows = [
      {{name: 'Accenture wins', value: d.accenture_wins, colour: '#9F7AEA'}},
      {{name: 'Competitors', value: d.competitor_market, colour: '#5B21B6'}}
    ];
    const stageThreeRows = capabilities.map(r => ({{
      name: r.capability, value: r.value, colour: r.colour
    }}));

    const traces = [
      makeDonut(stageOneRows, DOMAINS[0], focusRotation(d.addressable, d.total), true),
      makeDonut(stageTwoRows, DOMAINS[1], focusRotation(d.accenture_wins, d.addressable), true),
      makeDonut(stageThreeRows, DOMAINS[2], 0, false)
    ];

    const annotations = [
      ...centreAnnotations(CENTRES[0], moneyLabel(d.total), 'Total Defence Procurement'),
      ...centreAnnotations(CENTRES[1], d.addressable_label, 'Addressable Market (TAM)'),
      ...centreAnnotations(CENTRES[2], d.accenture_label, 'Accenture Wins')
    ];

    const layout = {{
      template: 'plotly_white',
      height: 515,
      margin: {{t: 8, l: 8, r: 8, b: 8}},
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      showlegend: false,
      annotations: annotations
    }};

    function transitionHtml(percent, label, value) {{
      return '<div class="transition-copy">' +
        '<b>' + pctLabel(percent) + '</b>' +
        '<span>' + label + '</span>' +
        '<strong>' + value + '</strong>' +
      '</div>' +
      '<div class="transition-arrow"><span class="transition-line"></span><span class="transition-head"></span></div>';
    }}

    const transitionOne = document.getElementById('transitionOne');
    const transitionTwo = document.getElementById('transitionTwo');
    if (transitionOne) transitionOne.innerHTML = transitionHtml(d.addressable_pct_total, 'Addressable to Accenture', d.addressable_label);
    if (transitionTwo) transitionTwo.innerHTML = transitionHtml(d.accenture_share_of_addressable, 'Accenture wins', d.accenture_label);

    Plotly.react(chartId, traces, layout, {{
      responsive: true,
      displayModeBar: false,
      staticPlot: true,
      scrollZoom: false,
      doubleClick: false
    }});

    const legendOne = document.getElementById('wideLegendOne');
    const legendTwo = document.getElementById('wideLegendTwo');
    const legendThree = document.getElementById('wideLegendThree');

    if (legendOne) legendOne.innerHTML =
      legendItem('Addressable (TAM)', d.addressable_pct_total, d.addressable_label, '#4C1D95') +
      legendItem('Not addressable', d.not_addressable_pct_total, d.not_addressable_label, '#D8CCF3');

    if (legendTwo) legendTwo.innerHTML =
      legendItem('Accenture wins', d.accenture_share_of_addressable, d.accenture_label, '#9F7AEA') +
      legendItem('Competitors', d.competitor_share_of_addressable, d.competitor_label, '#5B21B6');

    if (legendThree) legendThree.innerHTML = capabilities.map(r =>
      legendItem(r.capability, r.share_pct, r.value_label, r.colour)
    ).join('');
  }}

  document.querySelectorAll('input[name="donutRange"]').forEach(input => {{
    input.addEventListener('change', function() {{ render(Number(this.value)); }});
  }});

  render(0);
}})();
</script>
"""

def chart_market_funnel(summary: dict[str, float | int | str]) -> str:
    """Donut showing the addressable market split with non-clipping annotations."""
    comp_share = summary["competitor_market_share"]
    acc_share = summary["accenture_share_of_addressable"]

    labels = ["Competitor-owned addressable", "Accenture wins"]
    values = [summary["competitor_market"], summary["accenture_addressable"]]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.62,
        sort=False,
        direction="clockwise",
        marker=dict(colors=["#EF553B", "#A100FF"], line=dict(color="#ffffff", width=2)),
        textinfo="none",
        hovertemplate="%{label}<br>%{value:$,.0f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text="How is the addressable market split?",
            x=0.02,
            xanchor="left",
            font=dict(size=18),
        ),
        template="plotly_white",
        height=500,
        margin=dict(t=95, l=80, r=245, b=80),
        legend=dict(
            orientation="v",
            y=0.62,
            x=1.02,
            xanchor="left",
            font=dict(size=12),
            traceorder="normal",
        ),
        annotations=[
            dict(
                text=f"{money(summary['addressable'])}<br><span style='font-size:12px;color:#526070'>addressable TAM</span>",
                x=0.50,
                y=0.50,
                showarrow=False,
                font=dict(size=24, color="#12263f"),
            ),
            dict(
                text=f"Accenture wins<br>{pct(acc_share)}<br><span style='font-size:10px'>{money(summary['accenture_addressable'])}</span>",
                x=0.76,
                y=0.94,
                xref="paper",
                yref="paper",
                showarrow=True,
                ax=42,
                ay=-28,
                arrowwidth=1,
                arrowcolor="#A100FF",
                font=dict(size=11, color="#A100FF"),
                align="center",
            ),
            dict(
                text=f"Competitor-owned<br>addressable<br>{pct(comp_share)}<br><span style='font-size:10px'>{money(summary['competitor_market'])}</span>",
                x=0.18,
                y=0.14,
                xref="paper",
                yref="paper",
                showarrow=True,
                ax=-45,
                ay=30,
                arrowwidth=1,
                arrowcolor="#EF553B",
                font=dict(size=11, color="#EF553B"),
                align="center",
            ),
        ],
    )
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=False,
        config={
            "staticPlot": True,
            "displayModeBar": False,
            "scrollZoom": False,
            "doubleClick": False,
            "responsive": True,
        },
    )

def chart_annual_trend(df: pd.DataFrame, value_col: str) -> str:
    if FIN_YEAR_COL not in df.columns:
        return "<p>No Financial Year column found.</p>"

    total = df.groupby(FIN_YEAR_COL, as_index=False)[value_col].sum()
    total["series"] = "Total procurement"
    addressable = df[df["is_addressable"]].groupby(FIN_YEAR_COL, as_index=False)[value_col].sum()
    addressable["series"] = "Addressable to Accenture"
    accenture = df[df["is_addressable"] & df["is_accenture"]].groupby(FIN_YEAR_COL, as_index=False)[value_col].sum()
    accenture["series"] = "Accenture wins in addressable market"

    annual = pd.concat([total, addressable, accenture], ignore_index=True)
    annual["value_b"] = annual[value_col] / 1_000_000_000
    annual = annual.sort_values(FIN_YEAR_COL)

    fig = px.line(
        annual,
        x=FIN_YEAR_COL,
        y="value_b",
        color="series",
        markers=True,
        title="Annual trend - same value basis as the headline numbers",
        labels={"value_b": "$B", FIN_YEAR_COL: "Financial year", "series": "Series"},
    )
    fig.update_traces(hovertemplate="%{fullData.name}<br>%{x}<br>$%{y:,.2f}B<extra></extra>")
    fig.update_layout(template="plotly_white", height=460, margin=dict(t=70, l=60, r=30, b=50))
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=False,
        config={
            "staticPlot": True,
            "displayModeBar": False,
            "scrollZoom": False,
            "doubleClick": False,
            "responsive": True,
        },
    )



def _annual_market_dataset_for_chart(df: pd.DataFrame, value_col: str) -> dict:
    """Prepare annual addressable and Accenture-win rows for the FY chart."""
    if FIN_YEAR_COL not in df.columns:
        return {"rows": []}

    addressable = df[df["is_addressable"]].groupby(FIN_YEAR_COL)[value_col].sum()
    accenture = df[df["is_addressable"] & df["is_accenture"]].groupby(FIN_YEAR_COL)[value_col].sum()

    frame = pd.DataFrame({
        FIN_YEAR_COL: addressable.index,
        "addressable": addressable.values,
        "accenture": accenture.reindex(addressable.index).fillna(0).values,
    }).copy()
    frame["_fy_start_year"] = frame[FIN_YEAR_COL].map(_financial_year_start_year)
    frame = frame.sort_values(["_fy_start_year", FIN_YEAR_COL]).reset_index(drop=True)
    frame["competitor_market"] = (frame["addressable"] - frame["accenture"]).clip(lower=0)
    frame["addressable_b"] = frame["addressable"] / 1_000_000_000
    frame["accenture_b"] = frame["accenture"] / 1_000_000_000
    frame["competitor_market_b"] = frame["competitor_market"] / 1_000_000_000
    frame["accenture_share_pct"] = frame["accenture"] / frame["addressable"].replace(0, pd.NA) * 100

    rows = []
    for _, row in frame.iterrows():
        share = 0.0 if pd.isna(row["accenture_share_pct"]) else float(row["accenture_share_pct"])
        rows.append({
            "fy": str(row[FIN_YEAR_COL]),
            "addressable_b": float(row["addressable_b"]),
            "competitor_market_b": float(row["competitor_market_b"]),
            "accenture_b": float(row["accenture_b"]),
            "addressable_label": short_money(row["addressable"]),
            "competitor_label": short_money(row["competitor_market"]),
            "accenture_label": short_money(row["accenture"]),
            "accenture_share_pct": share,
            "accenture_share_label": f"{share:,.1f}%",
        })
    return {"rows": rows}


def _annual_financial_year_options(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """Only expose the three stakeholder-requested FY filters for the annual chart."""
    if FIN_YEAR_COL not in df.columns:
        return [("All years", df)]

    tmp = df.copy()
    tmp["_fy_start_year"] = tmp[FIN_YEAR_COL].map(_financial_year_start_year)
    fy = (
        tmp[[FIN_YEAR_COL, "_fy_start_year"]]
        .dropna()
        .drop_duplicates()
        .sort_values("_fy_start_year")
    )
    fy_labels = fy[FIN_YEAR_COL].astype(str).tolist()
    if not fy_labels:
        return [("All years", df)]

    options: list[tuple[str, pd.DataFrame]] = [("All years", df)]
    if len(fy_labels) >= 5:
        last5 = fy_labels[-5:]
        options.append((
            f"Last 5 FY ({last5[0]} to {last5[-1]})",
            tmp[tmp[FIN_YEAR_COL].astype(str).isin(last5)].drop(columns=["_fy_start_year"]),
        ))
    if len(fy_labels) >= 3:
        last3 = fy_labels[-3:]
        options.append((
            f"Last 3 FY ({last3[0]} to {last3[-1]})",
            tmp[tmp[FIN_YEAR_COL].astype(str).isin(last3)].drop(columns=["_fy_start_year"]),
        ))
    return options


def _linear_trend(values: list[float | None]) -> dict[str, object]:
    """Return fitted values and slope for a simple least-squares linear trend."""
    valid_pairs = [
        (index, float(value))
        for index, value in enumerate(values)
        if value is not None and pd.notna(value)
    ]

    if len(valid_pairs) < 2:
        return {
            "trend_values": [None for _ in values],
            "slope": None,
            "direction": "flat",
            "start": None,
            "end": None,
        }

    x = [pair[0] for pair in valid_pairs]
    y = [pair[1] for pair in valid_pairs]

    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    denominator = sum((xi - x_mean) ** 2 for xi in x)

    slope = (
        sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y)) / denominator
        if denominator
        else 0.0
    )
    intercept = y_mean - slope * x_mean

    fitted_all = [
        None if value is None or pd.isna(value) else float(intercept + slope * index)
        for index, value in enumerate(values)
    ]
    fitted_valid = [value for value in fitted_all if value is not None]

    if abs(slope) < 0.01:
        direction = "flat"
    elif slope > 0:
        direction = "up"
    else:
        direction = "down"

    return {
        "trend_values": fitted_all,
        "slope": float(slope),
        "direction": direction,
        "start": float(fitted_valid[0]) if fitted_valid else None,
        "end": float(fitted_valid[-1]) if fitted_valid else None,
    }


def _market_and_share_trend_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    """Calculate linear trends for annual TAM and Accenture share."""
    market = _linear_trend([
        float(r.get("addressable_b") or 0)
        if r.get("addressable_b") is not None
        else None
        for r in rows
    ])
    share = _linear_trend([
        float(r.get("accenture_share_pct") or 0)
        if r.get("accenture_share_pct") is not None
        else None
        for r in rows
    ])

    return {
        "market_trend_values": market["trend_values"],
        "market_slope_b_per_year": market["slope"],
        "market_direction": market["direction"],
        "market_start_b": market["start"],
        "market_end_b": market["end"],
        "share_trend_values": share["trend_values"],
        "share_slope_pp_per_year": share["slope"],
        "share_direction": share["direction"],
        "share_start_pct": share["start"],
        "share_end_pct": share["end"],
    }


def chart_market_share_by_year(df: pd.DataFrame, value_col: str) -> str:
    """Interactive FY chart with a linear trend in Accenture market share.

    - grey = competitor-owned addressable market
    - purple = Accenture wins
    - dashed line = least-squares trend in Accenture share of annual TAM
    - no CAGR KPI is shown because Defence procurement values are lumpy
    """
    if FIN_YEAR_COL not in df.columns:
        return "<p>No Financial Year column found.</p>"

    options = []
    seen = set()
    for label, subset in _annual_financial_year_options(df):
        if label in seen:
            continue
        seen.add(label)
        payload = _annual_market_dataset_for_chart(subset, value_col)
        payload["label"] = label
        payload["subtitle"] = "Selected period: " + label
        payload["trend_summary"] = _market_and_share_trend_summary(payload["rows"])
        options.append(payload)

    chart_data = json.dumps(options)
    controls = []
    for i, opt in enumerate(options):
        checked = " checked" if i == 0 else ""
        controls.append(f'<label class="range-pill"><input type="radio" name="annualRange" value="{i}"{checked}> {opt["label"]}</label>')
    controls_html = "\n".join(controls)

    return f"""
<div class="annual-range-wrap">
  <div class="capability-range-header">
    <div>
      <h3>Accenture's % share of the TAM by Year</h3>
      <p>Grey columns show competitor-owned addressable market. Purple stacked segments show Accenture wins.</p>
    </div>
  </div>
  <div class="range-controls">{controls_html}</div>
  <div id="annualRangeChart" style="height:640px; width:100%;"></div>
</div>
<script>
(function() {{
  const datasets = {chart_data};
  const chartId = 'annualRangeChart';

  function render(index) {{
    const d = datasets[index] || datasets[0];
    const rows = d.rows || [];
    const fy = rows.map(r => r.fy);
    const maxY = Math.max(0.1, ...rows.map(r => Number(r.addressable_b || 0)));
    const trend = d.trend_summary || {{}};
    const marketTrendValues = (trend.market_trend_values || []).map(v =>
      v === null || v === undefined ? null : Number(v)
    );
    const shareTrendValues = (trend.share_trend_values || []).map(v =>
      v === null || v === undefined ? null : Number(v)
    );
    const marketSlope = trend.market_slope_b_per_year === null || trend.market_slope_b_per_year === undefined
      ? null
      : Number(trend.market_slope_b_per_year);
    const shareSlope = trend.share_slope_pp_per_year === null || trend.share_slope_pp_per_year === undefined
      ? null
      : Number(trend.share_slope_pp_per_year);

    const competitorTrace = {{
      type: 'bar',
      x: fy,
      y: rows.map(r => r.competitor_market_b),
      name: 'Competitor-owned addressable market',
      marker: {{color: '#D7DEE9'}},
      text: rows.map(r => r.addressable_label),
      textposition: 'outside',
      cliponaxis: false,
      customdata: rows.map(r => [r.competitor_label, r.addressable_label, r.accenture_label, r.accenture_share_label]),
      hovertemplate:
        '<b>%{{x}}</b><br>' +
        'Total addressable market: %{{customdata[1]}}<br>' +
        'Competitor-owned addressable: %{{customdata[0]}}<br>' +
        'Accenture wins: %{{customdata[2]}}<br>' +
        'Accenture share: %{{customdata[3]}}' +
        '<extra></extra>'
    }};

    const accentureTrace = {{
      type: 'bar',
      x: fy,
      y: rows.map(r => r.accenture_b),
      name: 'Accenture wins',
      marker: {{color: '#8A00FF'}},
      text: rows.map(r => r.accenture_b > 0 ? r.accenture_share_label : ''),
      textposition: 'outside',
      textfont: {{color: '#8A00FF', size: 13, family: 'Arial Black'}},
      cliponaxis: false,
      customdata: rows.map(r => [r.accenture_label, r.accenture_share_label, r.addressable_label]),
      hovertemplate:
        '<b>%{{x}}</b><br>' +
        'Accenture wins: %{{customdata[0]}}<br>' +
        'Share of annual addressable market: %{{customdata[1]}}<br>' +
        'Total addressable market: %{{customdata[2]}}' +
        '<extra></extra>'
    }};

    const layout = {{
      template: 'plotly_white',
      height: 640,
      margin: {{t: 32, l: 82, r: 55, b: 110}},
      barmode: 'stack',
      bargap: 0.34,
      xaxis: {{title: 'Financial year', tickangle: -35, automargin: true}},
      yaxis: {{
        title: {{text: 'B$', font: {{size: 15}}}},
        range: [0, maxY * 1.28],
        showgrid: true,
        zeroline: true,
        tickformat: '.1f',
        automargin: true
      }},
      yaxis2: {{
        title: {{text: 'Accenture share (%)', font: {{size: 14, color: '#8A00FF'}}}},
        overlaying: 'y',
        side: 'right',
        rangemode: 'tozero',
        ticksuffix: '%',
        showgrid: false,
        zeroline: false,
        tickfont: {{color: '#8A00FF'}},
        automargin: true
      }},
      legend: {{orientation: 'h', y: -0.18, x: 0, xanchor: 'left'}},
      hovermode: 'closest',
      uniformtext: {{mode: 'hide', minsize: 9}},
      annotations: []
    }};

    Plotly.react(chartId, [competitorTrace, accentureTrace], layout, {{responsive: true, displayModeBar: false, staticPlot: false, scrollZoom: false, doubleClick: false}});
  }}

  document.querySelectorAll('input[name="annualRange"]').forEach(input => {{
    input.addEventListener('change', function() {{ render(Number(this.value)); }});
  }});
  render(0);
}})();
</script>
"""

def addressable_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return only Accenture-addressable contracts."""
    return df[df["is_addressable"]].copy()


def _supplier_share_frame(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    data = addressable_df(df)
    supplier = (
        data.groupby("supplier_group", as_index=False)[value_col]
        .sum()
        .rename(columns={value_col: "supplier_value"})
        .sort_values("supplier_value", ascending=False)
    )
    total = supplier["supplier_value"].sum()
    supplier["value_b"] = supplier["supplier_value"] / 1_000_000_000
    supplier["share_pct"] = supplier["supplier_value"] / total * 100 if total else 0
    return supplier


def chart_competitor_share(df: pd.DataFrame, value_col: str) -> str:
    """Focused competitor share chart for Accenture-relevant suppliers only."""
    supplier = _supplier_share_frame(df, value_col)
    focus = supplier[supplier["supplier_group"].isin(COMPETITOR_EXPANDED)].copy()

    missing = [name for name in COMPETITOR_FOCUS if name not in set(focus["supplier_group"])]
    if missing:
        focus = pd.concat([
            focus,
            pd.DataFrame({"supplier_group": missing, "supplier_value": 0, "value_b": 0, "share_pct": 0}),
        ], ignore_index=True)

    focus = focus.sort_values("supplier_value", ascending=True)
    fig = go.Figure(go.Bar(
        y=focus["supplier_group"],
        x=focus["value_b"],
        orientation="h",
        text=focus["share_pct"].map(lambda x: f"{x:.1f}%"),
        textposition="outside",
        cliponaxis=False,
        customdata=focus[["supplier_value", "share_pct"]],
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Addressable contract value: $%{x:,.2f}B<br>"
            "Share of total addressable market: %{customdata[1]:,.1f}%"
            "<extra></extra>"
        ),
    ))
    fig.update_layout(
        title="Competitor share of Accenture-addressable market",
        template="plotly_white",
        height=max(520, 32 * len(focus) + 160),
        margin=dict(t=80, l=180, r=90, b=60),
        xaxis=dict(title="$B, addressable contracts only"),
        yaxis=dict(title="Supplier"),
        annotations=[dict(
            text="Filtered to contracts classified as relevant to Accenture service offerings. Percentages are share of the full addressable market, not share of this competitor subset.",
            x=0, y=1.12, xref="paper", yref="paper", showarrow=False, xanchor="left",
            font=dict(size=12, color="#526070"),
        )],
    )
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=False,
        config={
            "staticPlot": True,
            "displayModeBar": False,
            "scrollZoom": False,
            "doubleClick": False,
            "responsive": True,
        },
    )


def chart_competitor_capability_matrix(df: pd.DataFrame, value_col: str, top_n_capabilities: int) -> str:
    """Heatmap of supplier share by capability inside the addressable market."""
    data = addressable_df(df)
    data["capability"] = data["capability"].fillna("Unclassified").replace("", "Unclassified")
    top_caps = (
        data.groupby("capability")[value_col]
        .sum()
        .sort_values(ascending=False)
        .head(top_n_capabilities)
        .index.tolist()
    )
    data_focus = data[data["capability"].isin(top_caps) & data["supplier_group"].isin(COMPETITOR_FOCUS)].copy()
    if data_focus.empty:
        return "<p>No competitor Service Offering data for the selected period.</p>"

    supplier_cap = data_focus.groupby(["supplier_group", "capability"])[value_col].sum().rename("supplier_value")
    cap_total = data.groupby("capability")[value_col].sum().rename("capability_total")
    matrix = supplier_cap.reset_index().merge(cap_total.reset_index(), on="capability", how="left")
    matrix["share_pct"] = matrix["supplier_value"] / matrix["capability_total"].replace(0, pd.NA) * 100

    supplier_order = [sup for sup in COMPETITOR_FOCUS if sup in set(matrix["supplier_group"])]
    cap_order = top_caps
    pivot = matrix.pivot(index="supplier_group", columns="capability", values="share_pct").reindex(index=supplier_order, columns=cap_order).fillna(0)
    value_pivot = matrix.pivot(index="supplier_group", columns="capability", values="supplier_value").reindex(index=supplier_order, columns=cap_order).fillna(0) / 1_000_000_000
    text = [[f"{v:.1f}%" if v > 0 else "" for v in row] for row in pivot.values]

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale="Purples",
        text=text,
        texttemplate="%{text}",
        customdata=value_pivot.values,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Service Offering: %{x}<br>"
            "Supplier value: $%{customdata:,.2f}B<br>"
            "Share of Service Offering: %{z:,.1f}%"
            "<extra></extra>"
        ),
    ))
    fig.update_layout(
        title="Competitor share by Service Offering",
        template="plotly_white",
        height=max(520, 42 * len(supplier_order) + 260),
        margin=dict(t=80, l=140, r=40, b=170),
        xaxis=dict(title="Service Offering", tickangle=-35),
        yaxis=dict(title="Supplier"),
    )
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=False,
        config={
            "staticPlot": True,
            "displayModeBar": False,
            "scrollZoom": False,
            "doubleClick": False,
            "responsive": True,
        },
    )


def competitor_contracts_table(df: pd.DataFrame, value_col: str, top_n: int = 60) -> str:
    """HTML table of top contracts that explain competitor value."""
    data = addressable_df(df)
    data = data[data["supplier_group"].isin(COMPETITOR_FOCUS)].copy()
    if data.empty:
        return "<p>No addressable competitor contracts found for the selected period.</p>"
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce").fillna(0)
    data = data.sort_values(value_col, ascending=False).head(top_n).copy()

    cols = []
    for col in ["supplier_group", CN_ID_COL, DESCRIPTION_COL, "capability", FIN_YEAR_COL, AGENCY_COL, AGENCY_BRANCH_COL, value_col]:
        if col in data.columns:
            cols.append(col)
    table = data[cols].copy()
    rename = {
        "supplier_group": "Supplier",
        CN_ID_COL: "CN ID",
        DESCRIPTION_COL: "Description",
        "capability": "Service Offering",
        FIN_YEAR_COL: "FY",
        AGENCY_COL: "Agency",
        AGENCY_BRANCH_COL: "Branch",
        value_col: "Value",
    }
    table = table.rename(columns=rename)
    if "Value" in table.columns:
        table["Value"] = table["Value"].map(money)
    if "Description" in table.columns:
        table["Description"] = table["Description"].astype(str).str.slice(0, 150)

    header = "".join(f"<th>{c}</th>" for c in table.columns)
    rows = []
    for _, row in table.iterrows():
        cells = "".join(f"<td>{str(row[c])}</td>" for c in table.columns)
        rows.append(f"<tr>{cells}</tr>")
    return f"""
    <div class="table-panel">
      <h3>Top addressable contracts behind competitor value</h3>
      <p>Largest addressable contracts for Accenture and selected competitors. This shows which contract awards are driving the supplier share charts.</p>
      <div class="table-scroll">
        <table class="contracts-table">
          <thead><tr>{header}</tr></thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
      </div>
    </div>
    """



def _supplier_identity_audit(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Return contract/value evidence for every raw-name + ABN -> canonical mapping."""
    data = df.copy()
    if SUPPLIER_COL not in data.columns:
        data[SUPPLIER_COL] = data.get("supplier_group", "Unknown")
    if SUPPLIER_ABN_COL not in data.columns:
        data[SUPPLIER_ABN_COL] = ""

    data[value_col] = pd.to_numeric(data[value_col], errors="coerce").fillna(0)
    data["supplier_abn_normalised"] = data[SUPPLIER_ABN_COL].map(_normalise_supplier_abn)
    data["supplier_name_normalised"] = data[SUPPLIER_COL].map(_normalise_supplier_name)
    data["supplier_group_rebuilt"] = [
        supplier_group(name, abn)
        for name, abn in zip(data[SUPPLIER_COL], data[SUPPLIER_ABN_COL])
    ]

    return (
        data.groupby(
            [
                "supplier_group_rebuilt",
                SUPPLIER_COL,
                SUPPLIER_ABN_COL,
                "supplier_abn_normalised",
                "supplier_name_normalised",
            ],
            dropna=False,
        )
        .agg(
            rows=(value_col, "size"),
            contracts=(CN_ID_COL, "nunique") if CN_ID_COL in data.columns else (value_col, "size"),
            addressable_value=(value_col, "sum"),
        )
        .reset_index()
        .sort_values(["supplier_group_rebuilt", "addressable_value"], ascending=[True, False])
    )


def _supplier_abn_conflict_audit(identity: pd.DataFrame) -> pd.DataFrame:
    """Find ABNs that resolve to more than one canonical supplier group."""
    populated = identity[identity["supplier_abn_normalised"].astype(str).ne("")].copy()
    if populated.empty:
        return pd.DataFrame(columns=["supplier_abn_normalised", "canonical_groups", "raw_names", "addressable_value"])

    conflicts = (
        populated.groupby("supplier_abn_normalised", dropna=False)
        .agg(
            canonical_group_count=("supplier_group_rebuilt", "nunique"),
            canonical_groups=("supplier_group_rebuilt", lambda s: " | ".join(sorted(set(map(str, s))))),
            raw_names=(SUPPLIER_COL, lambda s: " | ".join(sorted(set(map(str, s))))),
            addressable_value=("addressable_value", "sum"),
        )
        .reset_index()
    )
    return conflicts[conflicts["canonical_group_count"] > 1].sort_values("addressable_value", ascending=False)


def _supplier_variant_summary(identity: pd.DataFrame) -> pd.DataFrame:
    """Show all legal-name and ABN variants rolled into each canonical supplier."""
    return (
        identity.groupby("supplier_group_rebuilt", dropna=False)
        .agg(
            raw_name_count=(SUPPLIER_COL, "nunique"),
            abn_count=("supplier_abn_normalised", lambda s: len({x for x in map(str, s) if x})),
            raw_names=(SUPPLIER_COL, lambda s: " | ".join(sorted(set(map(str, s))))),
            abns=("supplier_abn_normalised", lambda s: " | ".join(sorted({x for x in map(str, s) if x}))),
            rows=("rows", "sum"),
            contracts=("contracts", "sum"),
            addressable_value=("addressable_value", "sum"),
        )
        .reset_index()
        .sort_values("addressable_value", ascending=False)
    )


def _potential_supplier_merge_audit(identity: pd.DataFrame) -> pd.DataFrame:
    """Flag similar unresolved supplier names without an all-to-all comparison.

    Candidates are blocked by first character and shared meaningful tokens before
    SequenceMatcher is used. This preserves the practical QA purpose while avoiding
    millions of unnecessary string comparisons.
    """
    totals = (
        identity.groupby("supplier_group_rebuilt", dropna=False)["addressable_value"]
        .sum()
        .sort_values(ascending=False)
    )

    records: list[dict[str, object]] = []
    for name, value in totals.items():
        name = str(name)
        if name in {"", "Unknown"}:
            continue
        normalised = _normalise_supplier_name(name)
        if not normalised:
            continue
        tokens = {
            token for token in normalised.split()
            if len(token) >= 3 and token not in {"the", "and", "group", "services", "service"}
        }
        records.append({
            "name": name,
            "normalised": normalised,
            "tokens": tokens,
            "value": float(value),
            "first": normalised[:1],
        })

    # Build small candidate blocks. Shared tokens catch variants such as
    # "Deloitte Touche" vs "Deloitte"; first-character blocking catches typos.
    token_index: dict[str, list[int]] = {}
    first_index: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        first_index.setdefault(record["first"], []).append(index)
        for token in record["tokens"]:
            token_index.setdefault(token, []).append(index)

    candidate_pairs: set[tuple[int, int]] = set()
    for indices in first_index.values():
        # Avoid pathological blocks while retaining the highest-value names.
        ranked = sorted(indices, key=lambda i: records[i]["value"], reverse=True)[:250]
        for pos, left in enumerate(ranked):
            for right in ranked[pos + 1:]:
                candidate_pairs.add((min(left, right), max(left, right)))

    for indices in token_index.values():
        ranked = sorted(indices, key=lambda i: records[i]["value"], reverse=True)[:250]
        for pos, left in enumerate(ranked):
            for right in ranked[pos + 1:]:
                candidate_pairs.add((min(left, right), max(left, right)))

    rows: list[dict[str, object]] = []
    for left_index, right_index in candidate_pairs:
        left = records[left_index]
        right = records[right_index]
        ratio = SequenceMatcher(None, left["normalised"], right["normalised"]).ratio()
        token_overlap = bool(left["tokens"] & right["tokens"])
        if ratio >= 0.88 or (ratio >= 0.76 and token_overlap):
            rows.append({
                "supplier_1": left["name"],
                "supplier_2": right["name"],
                "similarity": round(ratio, 3),
                "supplier_1_value": left["value"],
                "supplier_2_value": right["value"],
                "review_reason": "Similar canonical names; review before adding an alias or ABN mapping.",
            })

    if not rows:
        return pd.DataFrame(columns=[
            "supplier_1", "supplier_2", "similarity", "supplier_1_value",
            "supplier_2_value", "review_reason"
        ])

    return (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["supplier_1", "supplier_2"])
        .sort_values(
            ["similarity", "supplier_1_value", "supplier_2_value"],
            ascending=[False, False, False],
        )
    )


def validate_supplier_consolidation(df: pd.DataFrame, value_col: str, output_dir: Path) -> None:
    """Audit the canonical master supplier groups without rebuilding them."""
    data = use_master_supplier_columns(addressable_df(df)).copy()
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce").fillna(0)

    total = float(data[value_col].sum())
    share = _supplier_share_frame(data, value_col)
    grouped_total = float(share["supplier_value"].sum())
    tolerance = max(1.0, abs(total) * 1e-10)
    if abs(total - grouped_total) > tolerance:
        raise SystemExit(
            "Canonical supplier totals do not reconcile to addressable TAM "
            f"({grouped_total} vs {total})."
        )

    share.to_csv(output_dir / "canonical_supplier_share_audit.csv", index=False)

    raw_col = SUPPLIER_COL if SUPPLIER_COL in data.columns else None
    abn_col = SUPPLIER_ABN_COL if SUPPLIER_ABN_COL in data.columns else None
    group_cols = ["supplier_group"]
    if raw_col:
        group_cols.append(raw_col)
    if abn_col:
        group_cols.append(abn_col)
    identity = (
        data.groupby(group_cols, dropna=False)
        .agg(
            rows=(value_col, "size"),
            contracts=(CN_ID_COL, "nunique") if CN_ID_COL in data.columns else (value_col, "size"),
            addressable_value=(value_col, "sum"),
        )
        .reset_index()
        .sort_values(["supplier_group", "addressable_value"], ascending=[True, False])
    )
    identity.to_csv(output_dir / "supplier_identity_mapping_audit.csv", index=False)

    variants = (
        identity.groupby("supplier_group", dropna=False)
        .agg(
            raw_name_count=(raw_col, "nunique") if raw_col else ("rows", "size"),
            abn_count=(abn_col, "nunique") if abn_col else ("rows", "size"),
            rows=("rows", "sum"),
            contracts=("contracts", "sum"),
            addressable_value=("addressable_value", "sum"),
        )
        .reset_index()
        .sort_values("addressable_value", ascending=False)
    )
    variants.to_csv(output_dir / "supplier_canonical_variant_summary.csv", index=False)
    print(
        "Canonical master supplier groups validated: "
        f"{len(share):,} suppliers reconcile to {short_money(grouped_total)}."
    )

def export_competitor_outputs(df: pd.DataFrame, value_col: str, output_dir: Path) -> None:
    """Write CSVs that let the user inspect contract-level evidence."""
    data = addressable_df(df).copy()
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce").fillna(0)

    contract_cols = [c for c in ["supplier_group", SUPPLIER_COL, SUPPLIER_ABN_COL, CN_ID_COL, DESCRIPTION_COL, "capability", "ReinventionPartner", "ReinventionEngine", "service_line", FIN_YEAR_COL, AGENCY_COL, AGENCY_BRANCH_COL, value_col] if c in data.columns]
    data[data["supplier_group"].isin(COMPETITOR_FOCUS)].sort_values(value_col, ascending=False)[contract_cols].to_csv(
        output_dir / "addressable_competitor_contracts.csv", index=False
    )

    supplier_share = _supplier_share_frame(df, value_col)
    supplier_share.to_csv(output_dir / "addressable_supplier_market_share.csv", index=False)

    data["capability"] = data["capability"].fillna("Unclassified").replace("", "Unclassified")
    cap_total = data.groupby("capability")[value_col].sum().rename("capability_total")
    sc = data[data["supplier_group"].isin(COMPETITOR_FOCUS)].groupby(["supplier_group", "capability"])[value_col].sum().rename("supplier_value").reset_index()
    sc = sc.merge(cap_total.reset_index(), on="capability", how="left")
    sc["share_pct"] = sc["supplier_value"] / sc["capability_total"].replace(0, pd.NA) * 100
    sc.to_csv(output_dir / "competitor_capability_share.csv", index=False)


def export_classification_audit_outputs(df: pd.DataFrame, value_col: str, output_dir: Path) -> None:
    """Write capability audit CSVs to diagnose granular classification quality.

    These files are deliberately contract-evidence first. They let you see which
    high-value contracts are driving each capability and whether broad generic
    terms such as review/assessment are over-influencing the roll-up.
    """
    data = df.copy()
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce").fillna(0)
    data["capability"] = data.get("capability", "Unclassified")
    if "detailed_capability" not in data.columns:
        data["detailed_capability"] = "Unclassified"
    if "matched_terms" not in data.columns:
        data["matched_terms"] = ""
    if "generic_terms" not in data.columns:
        data["generic_terms"] = ""
    if "specific_terms" not in data.columns:
        data["specific_terms"] = ""
    if "confidence" not in data.columns:
        data["confidence"] = 0

    addr = data[data["is_addressable"]].copy()

    capability_summary = (
        addr.groupby("capability", dropna=False)
        .agg(
            segment_value=(value_col, "sum"),
            contracts=(CN_ID_COL, "nunique") if CN_ID_COL in addr.columns else (value_col, "size"),
            suppliers=("supplier_group", "nunique"),
            avg_confidence=("confidence", "mean"),
            accenture_wins=(value_col, lambda s: s[addr.loc[s.index, "is_accenture"]].sum()),
        )
        .reset_index()
        .sort_values("segment_value", ascending=False)
    )
    capability_summary["accenture_share_pct"] = capability_summary["accenture_wins"] / capability_summary["segment_value"].replace(0, pd.NA) * 100
    capability_summary.to_csv(output_dir / "classification_capability_summary.csv", index=False)

    detailed_summary = (
        addr.groupby(["capability", "detailed_capability"], dropna=False)
        .agg(
            segment_value=(value_col, "sum"),
            contracts=(CN_ID_COL, "nunique") if CN_ID_COL in addr.columns else (value_col, "size"),
            avg_confidence=("confidence", "mean"),
        )
        .reset_index()
        .sort_values("segment_value", ascending=False)
    )
    detailed_summary.to_csv(output_dir / "classification_detailed_capability_summary.csv", index=False)

    evidence_cols = [c for c in [CN_ID_COL, FIN_YEAR_COL, AGENCY_COL, AGENCY_BRANCH_COL, SUPPLIER_COL, "supplier_group", DESCRIPTION_COL, CATEGORY_COL, CATEGORY_TYPE_COL, value_col, "addressability", "is_market_relevant", "market_relevance", "market_exclusion_reason", "capability", "detailed_capability", "confidence", "matched_terms", "specific_terms", "generic_terms"] if c in data.columns]
    addr.sort_values(value_col, ascending=False)[evidence_cols].head(1000).to_csv(
        output_dir / "classification_top_addressable_contracts_audit.csv", index=False
    )

    generic_strategy = addr[
        addr["capability"].eq("Strategy, Transformation & Advisory")
        & addr["generic_terms"].astype(str).ne("")
        & addr["specific_terms"].astype(str).eq("")
    ].copy()
    generic_strategy.sort_values(value_col, ascending=False)[evidence_cols].head(1000).to_csv(
        output_dir / "classification_generic_strategy_audit.csv", index=False
    )

    review_queue = data[
        (data["is_addressable"] & (pd.to_numeric(data["confidence"], errors="coerce").fillna(0) < 65))
        | data["capability"].eq("Unclassified")
    ].copy()
    review_queue.sort_values(value_col, ascending=False)[evidence_cols].head(1000).to_csv(
        output_dir / "classification_review_queue.csv", index=False
    )



def export_market_relevance_audit(df: pd.DataFrame, value_col: str, output_dir: Path) -> None:
    """Write an audit of contracts removed by the supplier/context market relevance layer."""
    if "is_market_relevant" not in df.columns:
        return
    data = df.copy()
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce").fillna(0)
    cols = [c for c in [
        CN_ID_COL, FIN_YEAR_COL, AGENCY_COL, AGENCY_BRANCH_COL, SUPPLIER_COL, "supplier_group",
        DESCRIPTION_COL, CATEGORY_COL, CATEGORY_TYPE_COL, value_col, "capability", "detailed_capability",
        "market_relevance", "market_exclusion_reason", "matched_terms"
    ] if c in data.columns]
    data[~data["is_market_relevant"].astype(bool)].sort_values(value_col, ascending=False)[cols].to_csv(
        output_dir / "market_relevance_exclusions_audit.csv", index=False
    )


def _normalise_abn(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = re.sub(r"[^0-9]", "", str(value))
    return text.zfill(11) if text else ""


def _rival_label_for_row(row: pd.Series) -> str | None:
    for abn_col in ("Supplier ABN", "ABN", "supplier_abn"):
        if abn_col in row.index:
            abn = _normalise_abn(row.get(abn_col))
            if abn in RIVAL_ABN_TO_NAME:
                return RIVAL_ABN_TO_NAME[abn]

    raw_name = str(row.get(SUPPLIER_COL, "") or "").strip().lower()
    group_name = str(row.get("supplier_group", "") or "").strip().lower()

    canonical_group = supplier_group(
        row.get(SUPPLIER_COL, row.get("supplier_group", "")),
        row.get(SUPPLIER_ABN_COL, ""),
    )
    if canonical_group in RIVAL_DISPLAY_ORDER:
        return canonical_group

    for candidate in (raw_name, group_name):
        candidate_norm = _normalise_supplier_name(candidate)
        if candidate in RIVAL_NAME_ALIASES:
            return RIVAL_NAME_ALIASES[candidate]
        if candidate_norm in RIVAL_NAME_ALIASES:
            return RIVAL_NAME_ALIASES[candidate_norm]

    # Exact fallback for the less common business-supplied rival names.
    for label in RIVAL_DISPLAY_ORDER:
        if label.lower() in raw_name or raw_name in label.lower():
            return label
    return None


def _rival_share_frame(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    data = df[df["is_addressable"]].copy()
    data["_rival_label"] = data.apply(_rival_label_for_row, axis=1)
    data = data[data["_rival_label"].notna()].copy()
    if data.empty:
        return pd.DataFrame(columns=["supplier", value_col])
    return (
        data.groupby("_rival_label", as_index=False)[value_col]
        .sum()
        .rename(columns={"_rival_label": "supplier"})
    )

def chart_supplier_share(df: pd.DataFrame, value_col: str, top_n: int) -> str:
    """Interactive supplier-share chart with FY slicers and market views.

    Views:
    - Rivals: Accenture plus consulting / tech / integration peers
    - Top 15 suppliers by addressable TAM share
    - Suppliers ranked 16-30 by addressable TAM share

    UI update: the narrative summary is rendered as KPI cards so it stays
    readable on smaller screens.
    """
    if FIN_YEAR_COL not in df.columns:
        options = [("All years", df)]
    else:
        options = _annual_financial_year_options(df)

    rival_suppliers = RIVAL_DISPLAY_ORDER


    colour_map = {
        "Accenture": "#A100FF",
        "Deloitte": "#86BC25",
        "KPMG": "#00338D",
        "EY": "#FFE600",
        "PwC": "#E0301E",
        "Leidos": "#850F88",
        "Data#3": "#34495E",
        "DXC": "#5B2C83",
        "IBM": "#1F70C1",
        "Microsoft": "#737373",
        "Amazon / AWS": "#FF9900",
        "Oracle": "#C74634",
        "SAP": "#0FAAFF",
        "Fujitsu": "#D6001C",
        "Aurecon": "#2E7D32",
        "Lockheed":"#003087",
        "Boeing":"#1A409F",
    }
    colour_map.update(RIVAL_COLOURS)

    datasets = []
    seen = set()
    for label, subset in options:
        if label in seen:
            continue
        seen.add(label)
        addressable_subset = subset[subset["is_addressable"]].copy()
        if CN_ID_COL in addressable_subset.columns:
            supplier = (
                addressable_subset
                .groupby("supplier_group", as_index=False)
                .agg(
                    supplier_value=(value_col, "sum"),
                    contracts=(CN_ID_COL, "nunique"),
                )
                .sort_values("supplier_value", ascending=False)
                .reset_index(drop=True)
            )
        else:
            supplier = (
                addressable_subset
                .groupby("supplier_group", as_index=False)
                .agg(
                    supplier_value=(value_col, "sum"),
                    contracts=(value_col, "size"),
                )
                .sort_values("supplier_value", ascending=False)
                .reset_index(drop=True)
            )
        supplier = supplier.rename(columns={"supplier_value": value_col})
        total = float(supplier[value_col].sum()) if not supplier.empty else 0.0
        if total <= 0:
            rows_all = []
        else:
            supplier["rank"] = supplier.index + 1
            supplier["share"] = supplier[value_col] / total * 100
            supplier["value_b"] = supplier[value_col] / 1_000_000_000
            rows_all = []
            for _, row in supplier.iterrows():
                name = str(row["supplier_group"])
                rows_all.append({
                    "supplier": name,
                    "value_b": float(row["value_b"]),
                    "value_label": short_money(row[value_col]),
                    "contracts": int(row.get("contracts", 0) or 0),
                    "share": float(row["share"]),
                    "share_label": f"{float(row['share']):,.1f}%",
                    "rank": int(row["rank"]),
                    "colour": colour_map.get(name, "#5F6F7F"),
                })

        # The rival view is a filtered slice of the same canonical ranked
        # supplier table used by Top 15 and Rank 16-30. Do not re-aggregate
        # suppliers through a second alias-matching path, because that can
        # produce different values for the same company between views.
        rival_set = set(rival_suppliers)
        rivals = [
            dict(row)
            for row in rows_all
            if row["supplier"] in rival_set
        ]

        if not any(r["supplier"] == "Accenture" for r in rivals):
            rivals.append({
                "supplier": "Accenture",
                "value_b": 0.0,
                "value_label": "$0",
                "contracts": 0,
                "share": 0.0,
                "share_label": "0.0%",
                "rank": 0,
                "overall_rank": 0,
                "page_rank": 0,
                "colour": "#A100FF",
            })

        rivals = sorted(rivals, key=lambda r: r["share"], reverse=True)

        # Keep two separate ranks:
        # - overall_rank: rank across the complete addressable supplier market
        # - page_rank: rank within the currently displayed page/view
        def add_page_rank(rows):
            return [
                {
                    **row,
                    "overall_rank": int(row.get("rank", 0) or 0),
                    "page_rank": index + 1,
                }
                for index, row in enumerate(rows)
            ]

        datasets.append({
            "label": label,
            "total_label": short_money(total),
            # Keep the complete ranked supplier list so the Accenture KPI card
            # can remain visible even when the selected chart view is Top 15
            # or Suppliers ranked 16-30 and Accenture is not in that displayed slice.
            "all_suppliers": [
                {
                    **row,
                    "overall_rank": int(row.get("rank", 0) or 0),
                    "page_rank": int(row.get("rank", 0) or 0),
                }
                for row in rows_all
            ],
            "views": {
                "top15": add_page_rank(rows_all[:15]),
                "rank16to30": add_page_rank(rows_all[15:30]),
                "rivals": add_page_rank(rivals),
            }
        })

    chart_data = json.dumps(datasets)
    controls = []
    for i, opt in enumerate(datasets):
        checked = " checked" if i == 0 else ""
        controls.append(f'<label class="range-pill"><input type="radio" name="supplierShareRange" value="{i}"{checked}> {opt["label"]}</label>')
    controls_html = "\n".join(controls)

    return f"""
<div class="supplier-share-wrap">
  <div class="capability-range-header supplier-share-header">
    <div>
      <h3>Total % share of Accenture-addressable market by Supplier</h3>
      <p>Use the FY slicer and view selector to compare direct rivals or inspect the broader supplier leaderboard.</p>
    </div>
    <div class="supplier-view-control">
      <label for="supplierShareView">Supplier view</label>
      <select id="supplierShareView">
        <option value="top15">Top 15 suppliers</option>
        <option value="rank16to30">Suppliers ranked 16-30</option>
        <option value="rivals">Rivals / consulting & tech peers</option>
      </select>
    </div>
  </div>
  <div class="range-controls">{controls_html}</div>
  <div id="supplierShareInsight" class="supplier-insight-strip"></div>
  <div id="supplierShareChart" style="width:100%; min-height:720px;"></div>
</div>
<script>
(function() {{
  const datasets = {chart_data};
  const chartId = 'supplierShareChart';
  const viewLabels = {{
    rivals: 'Rivals / consulting & tech peers',
    top15: 'Top 15 suppliers',
    rank16to30: 'Suppliers ranked 16-30'
  }};

  function getSelectedRangeIndex() {{
    const checked = document.querySelector('input[name="supplierShareRange"]:checked');
    return checked ? Number(checked.value) : 0;
  }}

  function getSelectedView() {{
    const el = document.getElementById('supplierShareView');
    return el ? el.value : 'rivals';
  }}

  function metricCard(title, value, detail, className) {{
    return '<div class="supplier-mini-card ' + (className || '') + '">' +
      '<span>' + title + '</span>' +
      '<b>' + value + '</b>' +
      '<em>' + detail + '</em>' +
      '</div>';
  }}

  function render() {{
    const d = datasets[getSelectedRangeIndex()] || datasets[0];
    const view = getSelectedView();
    const rows = ((d.views || {{}})[view] || []).slice();
    const sorted = rows.slice().sort((a, b) => Number(a.share || 0) - Number(b.share || 0));
    const maxX = Math.max(0.2, ...rows.map(r => Number(r.share || 0)));
    // Accenture is the anchor metric for this view, so always source it
    // from the full ranked list for the selected period rather than only
    // from the currently displayed chart slice.
    const acc = (d.all_suppliers || []).find(r => r.supplier === 'Accenture') || {{
      supplier: 'Accenture',
      value_b: 0,
      value_label: '$0',
      contracts: 0,
      share: 0,
      share_label: '0.0%',
      rank: 0,
      colour: '#A100FF'
    }};
    const leader = rows.slice().sort((a, b) => Number(b.share || 0) - Number(a.share || 0))[0];

    const insight = document.getElementById('supplierShareInsight');
    let insightHtml = '';
    if (leader) {{
      insightHtml += metricCard('Leader', leader.supplier, leader.share_label + ' of addressable TAM', 'leader');
    }}
    const rankText = acc.overall_rank && acc.overall_rank > 0 ? 'Overall rank #' + acc.overall_rank : 'No current rank';
    insightHtml += metricCard('Accenture', acc.share_label, acc.value_label + ' / ' + rankText, 'accenture');
    insightHtml += metricCard('Selected period', d.label, 'Total addressable: ' + d.total_label, 'period');
    insightHtml += metricCard('Showing', viewLabels[view], rows.length + ' suppliers in view', 'view');
    insight.innerHTML = insightHtml;

    const trace = {{
      type: 'bar',
      orientation: 'h',
      y: sorted.map(r => r.supplier),
      x: sorted.map(r => r.share),
      marker: {{color: sorted.map(r => r.colour || '#5F6F7F')}},
      text: sorted.map(r => r.share_label),
      textposition: 'outside',
      cliponaxis: false,
      customdata: sorted.map(r => [
        r.value_label,
        r.overall_rank,
        r.share_label,
        r.contracts,
        r.page_rank
      ]),
      hovertemplate:
        '<b>%{{y}}</b><br>' +
        'Value: %{{customdata[0]}}<br>' +
        'Contracts: %{{customdata[3]:,.0f}}<br>' +
        'Share of addressable TAM: %{{customdata[2]}}<br>' +
        'Overall rank: #%{{customdata[1]}}' +
        '<extra></extra>'
    }};

    const chartHeight = Math.max(720, 42 * Math.max(sorted.length, 8) + 220);
    const chartElement = document.getElementById(chartId);
    if (chartElement) chartElement.style.height = chartHeight + 'px';

    const layout = {{
      template: 'plotly_white',
      height: chartHeight,
      margin: {{t: 24, l: 250, r: 105, b: 65}},
      xaxis: {{
        title: 'Share of Accenture-addressable TAM (%)',
        range: [0, maxX * 1.16],
        ticksuffix: '%',
        showgrid: true,
        zeroline: true
      }},
      yaxis: {{title: '', automargin: true}},
      showlegend: false,
      annotations: []
    }};

    Plotly.react(chartId, [trace], layout, {{responsive: true, displayModeBar: false, staticPlot: false, scrollZoom: false, doubleClick: false}});
  }}

  document.querySelectorAll('input[name="supplierShareRange"]').forEach(input => {{
    input.addEventListener('change', render);
  }});
  document.getElementById('supplierShareView').addEventListener('change', render);
  render();
}})();
</script>
"""

def chart_capability(df: pd.DataFrame, value_col: str, top_n: int) -> str:
    """Addressable segment value with Accenture wins overlaid in purple."""
    payload = _capability_dataset_for_chart(df, value_col, top_n)
    cap = pd.DataFrame(payload["rows"])
    if cap.empty:
        return "<p>No addressable Service Offering data for the selected period.</p>"

    max_x = max(float(cap["segment_value_b"].max()), 0.1)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=cap["capability"], x=cap["segment_value_b"], name="Addressable segment value",
        orientation="h", marker=dict(color="#D9DEE8"), text=cap["segment_label"],
        textposition="outside", cliponaxis=False,
        customdata=cap[["segment_label", "wins_label", "accenture_share_pct"]],
        hovertemplate=("<b>%{y}</b><br>Addressable segment value: %{customdata[0]}<br>"
                       "Accenture wins: %{customdata[1]}<br>Accenture historical share: %{customdata[2]:,.1f}%<extra></extra>"),
    ))
    fig.add_trace(go.Bar(
        y=cap["capability"], x=cap["accenture_wins_b"], name="Accenture wins",
        orientation="h", marker=dict(color="#A100FF"), text=cap["wins_label"],
        textposition="outside", cliponaxis=False,
        customdata=cap[["wins_label", "accenture_share_pct", "segment_label"]],
        hovertemplate=("<b>%{y}</b><br>Accenture wins: %{customdata[0]}<br>"
                       "Share of segment: %{customdata[1]:,.1f}%<br>Addressable segment value: %{customdata[2]}<extra></extra>"),
    ))
    fig.update_layout(
        title="Addressable Segment Value with Accenture Wins by Service Offering",
        template="plotly_white", height=max(560, 34 * len(cap) + 190),
        margin=dict(t=90, l=230, r=90, b=80),
        xaxis=dict(title="$B", range=[0, max_x * 1.18], showgrid=True),
        yaxis=dict(title="Service Offering"), barmode="overlay", bargap=0.38,
        legend=dict(orientation="h", y=-0.14),
        annotations=[dict(
            text="Grey bar = total Accenture-addressable segment. Purple overlay = Accenture wins inside that segment.",
            x=0, y=1.10, xref="paper", yref="paper", showarrow=False, xanchor="left",
            font=dict(size=12, color="#526070"),
        )],
    )
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=False,
        config={
            "staticPlot": True,
            "displayModeBar": False,
            "scrollZoom": False,
            "doubleClick": False,
            "responsive": True,
        },
    )

def _capability_dataset_for_chart(df: pd.DataFrame, value_col: str, top_n: int) -> dict:
    """Prepare rows for the interactive capability chart."""
    data = df.copy()
    data["capability"] = data["capability"].fillna("Unclassified").replace("", "Unclassified")

    segment = (
        data[data["is_addressable"]]
        .groupby("capability", as_index=False)[value_col]
        .sum()
        .rename(columns={value_col: "segment_value"})
    )
    accenture = (
        data[data["is_addressable"] & data["is_accenture"]]
        .groupby("capability", as_index=False)[value_col]
        .sum()
        .rename(columns={value_col: "accenture_wins"})
    )

    cap = segment.merge(accenture, on="capability", how="left")
    cap["accenture_wins"] = cap["accenture_wins"].fillna(0)
    cap = cap[cap["segment_value"] > 0].copy()
    cap["accenture_share_pct"] = cap["accenture_wins"] / cap["segment_value"].replace(0, pd.NA) * 100
    cap = cap.sort_values("segment_value", ascending=False).head(top_n).sort_values("segment_value")
    cap["segment_value_b"] = cap["segment_value"] / 1_000_000_000
    cap["accenture_wins_b"] = cap["accenture_wins"] / 1_000_000_000

    rows = []
    for _, row in cap.iterrows():
        share = 0.0 if pd.isna(row["accenture_share_pct"]) else float(row["accenture_share_pct"])
        rows.append({
            "capability": str(row["capability"]),
            "segment_value_b": float(row["segment_value_b"]),
            "accenture_wins_b": float(row["accenture_wins_b"]),
            "accenture_share_pct": share,
            "segment_label": short_money(row["segment_value"]),
            "wins_label": short_money(row["accenture_wins"]),
            "share_label": f"{share:,.1f}%",
        })
    return {"rows": rows}

def _financial_year_options(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """Create finite range options for in-page radio buttons."""
    if FIN_YEAR_COL not in df.columns:
        return [("All years", df)]

    tmp = df.copy()
    tmp["_fy_start_year"] = tmp[FIN_YEAR_COL].map(_financial_year_start_year)
    fy = (
        tmp[[FIN_YEAR_COL, "_fy_start_year"]]
        .dropna()
        .drop_duplicates()
        .sort_values("_fy_start_year")
    )
    fy_labels = fy[FIN_YEAR_COL].astype(str).tolist()
    if not fy_labels:
        return [("All years", df)]

    options: list[tuple[str, pd.DataFrame]] = [("All years", df)]

    if len(fy_labels) >= 5:
        last5 = fy_labels[-5:]
        options.append((f"Last 5 FY ({last5[0]} to {last5[-1]})", tmp[tmp[FIN_YEAR_COL].astype(str).isin(last5)].drop(columns=["_fy_start_year"])))
    if len(fy_labels) >= 3:
        last3 = fy_labels[-3:]
        options.append((f"Last 3 FY ({last3[0]} to {last3[-1]})", tmp[tmp[FIN_YEAR_COL].astype(str).isin(last3)].drop(columns=["_fy_start_year"])))

    latest = fy_labels[-1]
    options.append((f"Latest FY ({latest})", tmp[tmp[FIN_YEAR_COL].astype(str).eq(latest)].drop(columns=["_fy_start_year"])))

    for label in reversed(fy_labels):
        options.append((label, tmp[tmp[FIN_YEAR_COL].astype(str).eq(label)].drop(columns=["_fy_start_year"])))

    return options


def chart_capability_interactive(df: pd.DataFrame, value_col: str, top_n: int) -> str:
    """Interactive horizontal capability chart with three executive FY slicers.

    - Grey bar = total Accenture-addressable capability segment.
    - Purple overlay = Accenture wins inside that segment.
    - Slicers are restricted to All years, Last 5 FY and Last 3 FY.
    - The Plotly-internal title and explanatory annotations are removed.
    """
    options = []
    seen = set()
    for label, subset in _annual_financial_year_options(df):
        if label in seen:
            continue
        seen.add(label)
        payload = _capability_dataset_for_chart(subset, value_col, top_n)
        payload["label"] = label
        options.append(payload)

    chart_data = json.dumps(options)
    controls = []
    for i, opt in enumerate(options):
        checked = " checked" if i == 0 else ""
        controls.append(
            f'<label class="range-pill"><input type="radio" '
            f'name="capabilityRange" value="{i}"{checked}> {opt["label"]}</label>'
        )
    controls_html = "\n".join(controls)

    return f"""
<div class="capability-range-wrap">
  <div class="capability-range-header">
    <div>
      <h3>Accenture Wins by Service Offering</h3>
      <p>Grey bars show total addressable segment value. Purple overlays show the value Accenture has won inside that segment.</p>
    </div>
  </div>
  <div class="range-controls">{controls_html}</div>
  <div id="capabilityRangeChart" style="height:760px; width:100%;"></div>
</div>
<script>
(function() {{
  const datasets = {chart_data};
  const chartId = 'capabilityRangeChart';

  function buildAnnotations(rows, maxX) {{
    const annotations = [];

    rows.forEach(row => {{
      const segmentX = Number(row.segment_value_b || 0);
      const winsX = Number(row.accenture_wins_b || 0);
      const hasWins = winsX > 0;

      const isTinySegment = segmentX < maxX * 0.06;
      const labelsClose = hasWins && Math.abs(segmentX - winsX) < maxX * 0.045;
      const separateLabels = isTinySegment || labelsClose;

      annotations.push({{
        text: row.segment_label,
        x: segmentX + (maxX * 0.014),
        y: row.capability,
        xref: 'x',
        yref: 'y',
        showarrow: false,
        xanchor: 'left',
        yanchor: 'middle',
        yshift: separateLabels ? -14 : 0,
        font: {{size: 12, color: '#111827'}}
      }});

      if (hasWins) {{
        const winsLabelX = Math.max(winsX + (maxX * 0.014), maxX * 0.025);
        annotations.push({{
          text: '<b>' + row.wins_label + '</b>',
          x: winsLabelX,
          y: row.capability,
          xref: 'x',
          yref: 'y',
          showarrow: false,
          xanchor: 'left',
          yanchor: 'middle',
          yshift: separateLabels ? 16 : 13,
          font: {{size: 12, color: '#A100FF'}}
        }});
      }}
    }});

    return annotations;
  }}

  function render(index) {{
    const d = datasets[index] || datasets[0];
    const rows = d.rows || [];
    const maxX = Math.max(0.1, ...rows.map(r => Number(r.segment_value_b || 0)));
    const y = rows.map(r => r.capability);
    const chartHeight = Math.max(620, 54 * rows.length + 220);
    const chartElement = document.getElementById(chartId);
    if (chartElement) chartElement.style.height = chartHeight + 'px';

    const segmentTrace = {{
      type: 'bar',
      orientation: 'h',
      y: y,
      x: rows.map(r => r.segment_value_b),
      name: 'Addressable segment value',
      marker: {{color: '#D9DEE8'}},
      width: 0.62,
      text: [],
      cliponaxis: false,
      customdata: rows.map(r => [r.segment_label, r.wins_label, r.accenture_share_pct]),
      hovertemplate: '<b>%{{y}}</b><br>Addressable segment value: %{{customdata[0]}}<br>Accenture wins: %{{customdata[1]}}<br>Accenture historical share: %{{customdata[2]:,.1f}}%<extra></extra>'
    }};

    const accentureTrace = {{
      type: 'bar',
      orientation: 'h',
      y: y,
      x: rows.map(r => r.accenture_wins_b),
      name: 'Accenture wins',
      marker: {{color: '#A100FF'}},
      width: 0.62,
      text: [],
      cliponaxis: false,
      customdata: rows.map(r => [r.wins_label, r.accenture_share_pct, r.segment_label]),
      hovertemplate: '<b>%{{y}}</b><br>Accenture wins: %{{customdata[0]}}<br>Share of segment: %{{customdata[1]:,.1f}}%<br>Addressable segment value: %{{customdata[2]}}<extra></extra>'
    }};

    const layout = {{
      template: 'plotly_white',
      height: chartHeight,
      margin: {{t: 28, l: 270, r: 150, b: 100}},
      xaxis: {{
        title: {{text: 'B$', font: {{size: 15}}}},
        range: [0, maxX * 1.32],
        showgrid: true,
        zeroline: true,
        automargin: true
      }},
      yaxis: {{
        title: '',
        automargin: true
      }},
      barmode: 'overlay',
      bargap: 0.48,
      legend: {{
        orientation: 'h',
        y: -0.16,
        x: 1,
        xanchor: 'right',
        yanchor: 'top'
      }},
      annotations: buildAnnotations(rows, maxX)
    }};

    Plotly.react(chartId, [segmentTrace, accentureTrace], layout, {{responsive: true, displayModeBar: false, staticPlot: true, scrollZoom: false, doubleClick: false}});
  }}

  document.querySelectorAll('input[name="capabilityRange"]').forEach(input => {{
    input.addEventListener('change', function() {{ render(Number(this.value)); }});
  }});
  render(0);
}})();
</script>
"""

def chart_capability_penetration_by_year(df: pd.DataFrame, value_col: str, top_n: int) -> str:
    """Show time scale for addressability and Accenture penetration by capability."""
    if FIN_YEAR_COL not in df.columns:
        return "<p>No Financial Year column found.</p>"

    data = df.copy()
    data["capability"] = data["capability"].fillna("Unclassified").replace("", "Unclassified")

    top_caps = (
        data[data["is_addressable"]]
        .groupby("capability")[value_col]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
    )
    data = data[data["capability"].isin(top_caps)].copy()

    keys = ["capability", FIN_YEAR_COL]
    total = data.groupby(keys)[value_col].sum().rename("total_market")
    addressable = data[data["is_addressable"]].groupby(keys)[value_col].sum().rename("addressable_market")
    accenture = data[data["is_addressable"] & data["is_accenture"]].groupby(keys)[value_col].sum().rename("accenture_wins")

    metrics = pd.concat([total, addressable, accenture], axis=1).fillna(0).reset_index()
    metrics["addressable_pct"] = metrics["addressable_market"] / metrics["total_market"].replace(0, pd.NA) * 100
    metrics["accenture_share_pct"] = metrics["accenture_wins"] / metrics["addressable_market"].replace(0, pd.NA) * 100

    cap_order = (
        metrics.groupby("capability")["addressable_market"]
        .sum()
        .sort_values(ascending=True)
        .index.tolist()
    )
    fy_order = sorted(metrics[FIN_YEAR_COL].dropna().astype(str).unique().tolist())

    addr_matrix = metrics.pivot(index="capability", columns=FIN_YEAR_COL, values="addressable_pct").reindex(index=cap_order, columns=fy_order)
    share_matrix = metrics.pivot(index="capability", columns=FIN_YEAR_COL, values="accenture_share_pct").reindex(index=cap_order, columns=fy_order)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("% of Service Offering addressable", "Accenture share of addressable market"),
        horizontal_spacing=0.18,
    )
    fig.add_trace(go.Heatmap(
        z=addr_matrix.values,
        x=addr_matrix.columns,
        y=addr_matrix.index,
        coloraxis="coloraxis",
        text=[["" if pd.isna(v) else f"{v:.0f}%" for v in row] for row in addr_matrix.values],
        texttemplate="%{text}",
        hovertemplate="%{y}<br>%{x}<br>Addressable: %{z:.1f}%<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Heatmap(
        z=share_matrix.values,
        x=share_matrix.columns,
        y=share_matrix.index,
        coloraxis="coloraxis2",
        text=[["" if pd.isna(v) else f"{v:.1f}%" for v in row] for row in share_matrix.values],
        texttemplate="%{text}",
        hovertemplate="%{y}<br>%{x}<br>Accenture share: %{z:.1f}%<extra></extra>",
    ), row=1, col=2)

    fig.update_layout(
        title="Service Offering addressability and Accenture penetration over time",
        template="plotly_white",
        height=max(520, 28 * len(cap_order) + 220),
        margin=dict(t=90, l=190, r=50, b=80),
        coloraxis=dict(colorscale="Blues", colorbar=dict(title="Addressable %", x=0.44)),
        coloraxis2=dict(colorscale="Purples", colorbar=dict(title="Accenture share %", x=1.02)),
    )
    fig.update_xaxes(title_text="Financial year", tickangle=-35)
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=False,
        config={
            "staticPlot": True,
            "displayModeBar": False,
            "scrollZoom": False,
            "doubleClick": False,
            "responsive": True,
        },
    )



def chart_addressable_mix(summary: dict[str, float | int | str]) -> str:
    """Donut showing addressable market with labels placed to avoid clipping."""
    not_addr_pct = summary["not_addressable"] / summary["total"] if summary["total"] else 0
    comp_pct = summary["competitor_market"] / summary["total"] if summary["total"] else 0
    acc_pct = summary["accenture_addressable"] / summary["total"] if summary["total"] else 0

    labels = ["Not addressable", "Competitor-owned addressable", "Accenture wins"]
    values = [summary["not_addressable"], summary["competitor_market"], summary["accenture_addressable"]]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.62,
        sort=False,
        direction="clockwise",
        marker=dict(colors=["#636EFA", "#EF553B", "#A100FF"], line=dict(color="#ffffff", width=2)),
        textinfo="none",
        hovertemplate="%{label}<br>%{value:$,.0f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text="What % of total procurement is addressable?",
            x=0.02,
            xanchor="left",
            font=dict(size=18),
        ),
        template="plotly_white",
        height=500,
        margin=dict(t=95, l=80, r=245, b=80),
        legend=dict(
            orientation="v",
            y=0.62,
            x=1.02,
            xanchor="left",
            font=dict(size=12),
            traceorder="normal",
        ),
        annotations=[
            dict(
                text=f"{pct(summary['addressable_pct_total'])}<br><span style='font-size:12px;color:#526070'>addressable</span><br><span style='font-size:13px'>{money(summary['addressable'])}</span>",
                x=0.50,
                y=0.50,
                showarrow=False,
                font=dict(size=28, color="#12263f"),
            ),
            dict(
                text=f"Accenture wins<br>{pct(acc_pct)}<br><span style='font-size:10px'>{money(summary['accenture_addressable'])}</span>",
                x=0.73,
                y=0.93,
                xref="paper",
                yref="paper",
                showarrow=True,
                ax=42,
                ay=-28,
                arrowwidth=1,
                arrowcolor="#A100FF",
                font=dict(size=11, color="#A100FF"),
                align="center",
            ),
            dict(
                text=f"Competitor-owned<br>addressable<br>{pct(comp_pct)}",
                x=0.12,
                y=0.78,
                xref="paper",
                yref="paper",
                showarrow=True,
                ax=-35,
                ay=-8,
                arrowwidth=1,
                arrowcolor="#EF553B",
                font=dict(size=11, color="#EF553B"),
                align="center",
            ),
            dict(
                text=f"Not addressable<br>{pct(not_addr_pct)}",
                x=0.72,
                y=0.18,
                xref="paper",
                yref="paper",
                showarrow=True,
                ax=35,
                ay=25,
                arrowwidth=1,
                arrowcolor="#636EFA",
                font=dict(size=11, color="#243B9F"),
                align="center",
            ),
        ],
    )
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=False,
        config={
            "staticPlot": True,
            "displayModeBar": False,
            "scrollZoom": False,
            "doubleClick": False,
            "responsive": True,
        },
    )

def build_html(df: pd.DataFrame, summary: dict[str, float | int | str], value_col: str, output_html: Path, top_n_suppliers: int, top_n_capabilities: int, value_mode: str, fy_start: str | None = None, fy_end: str | None = None) -> None:
    basis_label = "Total contract value" if value_mode == "total" else "Annualised contract value / run-rate"
    period_label = _format_fy_range_label(fy_start, fy_end)

    cards = [
        ("Total procurement", money(summary["total"]), f"{basis_label}, {period_label}"),
        ("Addressable to Accenture", money(summary["addressable"]), f"{pct(summary['addressable_pct_total'])} of total procurement"),
        ("Accenture wins", money(summary["accenture_addressable"]), f"{pct(summary['accenture_share_of_addressable'])} of addressable market"),
        ("Competitor-owned market", money(summary["competitor_market"]), "Addressable market currently awarded to other suppliers"),
        ("Avg annual addressable", money(summary["avg_addressable_per_year"]), f"Average across {summary['years']} financial years"),
        ("Avg annual Accenture wins", money(summary["avg_accenture_addressable_per_year"]), f"Average across {summary['years']} financial years"),
    ]

    cards_html = "\n".join(
        f"""
        <div class="card">
            <div class="card-title">{title}</div>
            <div class="card-value">{value}</div>
            <div class="card-subtitle">{subtitle}</div>
        </div>
        """
        for title, value, subtitle in cards
    )

    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Accenture Defence Addressable Market Dashboard</title>
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
    .cards {{ display: grid; grid-template-columns: repeat(3, minmax(240px, 1fr)); gap: 14px; margin: 18px 0 22px 0; }}
    .card {{ background: #ffffff; border: 1px solid #e5e7eb; border-radius: 14px; padding: 17px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); min-width: 0; overflow: hidden; }}
    .card-title {{ font-size: clamp(11px, 0.9vw, 13px); text-transform: uppercase; letter-spacing: .04em; color: #667085; margin-bottom: 8px; white-space: normal; line-height: 1.25; }}
    .card-value {{ font-size: clamp(22px, 2.2vw, 30px); font-weight: 700; color: #111827; margin-bottom: 5px; line-height: 1.05; white-space: normal; overflow-wrap: anywhere; }}
    .card-subtitle {{ font-size: clamp(11px, 0.9vw, 13px); color: #667085; line-height: 1.3; white-space: normal; overflow-wrap: anywhere; }}
    .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin-bottom: 18px; }}
    .panel {{ background: #ffffff; border: 1px solid #e5e7eb; border-radius: 14px; padding: 18px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); overflow: visible; }}
    .panel-wide {{ grid-column: 1 / -1; }}
    .range-controls {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0 8px 0; }}
    .range-pill {{ display: inline-flex; align-items: center; gap: 6px; border: 1px solid #d0d5dd; border-radius: 999px; padding: 7px 12px; background: #ffffff; font-size: 12px; cursor: pointer; white-space: nowrap; }}
    .range-pill:has(input:checked) {{ background: #eef2ff; border-color: #636efa; color: #243b9f; font-weight: 600; }}
    .capability-range-header h3 {{ margin: 8px 0 4px 0; font-size: 18px; }}
    .capability-range-header p {{ margin: 0 0 10px 0; color: #526070; font-size: 13px; }}
    .table-panel h3 {{ margin: 12px 0 4px 0; font-size: 18px; }}
    .table-panel p {{ margin: 0 0 12px 0; color: #526070; font-size: 13px; }}
    .table-scroll {{ max-height: 620px; overflow: auto; border: 1px solid #e5e7eb; border-radius: 10px; }}
    .contracts-table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
    .contracts-table th {{ position: sticky; top: 0; background: #f2f4f7; text-align: left; padding: 8px; border-bottom: 1px solid #e5e7eb; }}
    .contracts-table td {{ padding: 7px 8px; border-bottom: 1px solid #eef2f7; vertical-align: top; }}
    .contracts-table tr:hover {{ background: #fafafa; }}
    .note {{ color: #526070; font-size: 13px; line-height: 1.45; margin-top: 14px; }}
    .donut-overview {{ width: 100%; }}
    .wide-donut-panel {{
      position: relative;
      border: 1px solid #e5e7eb;
      border-radius: 14px;
      background: #ffffff;
      padding: 18px 14px 24px 14px;
      overflow: hidden;
    }}
    .wide-donut-headings {{
      position: relative;
      z-index: 1;
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 56px;
      padding: 0 18px;
      text-align: center;
    }}
    .wide-donut-headings div {{
      min-width: 0;
      display: flex;
      flex-direction: column;
      align-items: center;
    }}
    .wide-donut-headings b {{
      font-size: clamp(14px, 1.2vw, 18px);
      line-height: 1.3;
      color: #111827;
    }}
    .wide-donut-headings span {{
      margin-top: 8px;
      font-size: 12.5px;
      line-height: 1.35;
      color: #526070;
    }}
    .wide-chart-stage {{
      position: relative;
      width: 100%;
      height: 515px;
      overflow: visible;
    }}
    .wide-donut-chart {{
      position: relative;
      z-index: 1;
      width: 100%;
      height: 515px;
      margin-top: 2px;
    }}
    .donut-transition {{
      position: absolute;
      top: 35.5%;
      width: 112px;
      transform: translate(-50%, -50%);
      z-index: 8;
      pointer-events: none;
      text-align: center;
      color: #5B21B6;
    }}
    .donut-transition-one {{ left: 33.35%; }}
    .donut-transition-two {{ left: 66.65%; }}
    .transition-copy {{
      display: flex;
      flex-direction: column;
      align-items: center;
      line-height: 1.1;
      margin-bottom: 10px;
      white-space: nowrap;
    }}
    .transition-copy b {{ font-size: 18px; font-weight: 800; margin-bottom: 4px; }}
    .transition-copy span {{ font-size: 11.5px; margin-bottom: 4px; }}
    .transition-copy strong {{ font-size: 16px; font-weight: 800; }}
    .transition-arrow {{ position: relative; width: 100%; height: 18px; }}
    .transition-line {{
      position: absolute;
      left: 0;
      right: 12px;
      top: 8px;
      height: 3px;
      border-radius: 999px;
      background: #5B21B6;
    }}
    .transition-head {{
      position: absolute;
      right: 0;
      top: 1px;
      width: 0;
      height: 0;
      border-top: 8px solid transparent;
      border-bottom: 8px solid transparent;
      border-left: 13px solid #5B21B6;
    }}
    .wide-donut-legends {{
      position: relative;
      z-index: 1;
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 56px;
      padding: 0 18px 18px 18px;
      align-items: start;
    }}
    .wide-stage-legend {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px 18px;
      min-width: 0;
    }}
    .wide-stage-legend-capabilities {{
      grid-template-columns: repeat(2, minmax(0, 1fr));
      row-gap: 14px;
      padding-right: 4px;
    }}
    .wide-legend-item {{
      display: grid;
      grid-template-columns: 15px minmax(0, 1fr);
      gap: 9px;
      align-items: start;
      min-width: 0;
      color: #344054;
    }}
    .wide-legend-swatch {{
      width: 15px;
      height: 15px;
      border-radius: 3px;
      margin-top: 2px;
    }}
    .wide-legend-item b {{
      display: block;
      font-size: 12.5px;
      line-height: 1.25;
      overflow-wrap: anywhere;
    }}
    .wide-legend-item small {{
      display: block;
      margin-top: 3px;
      font-size: 11.5px;
      line-height: 1.25;
      color: #667085;
    }}
    @media (max-width: 1050px) {{
      .wide-donut-headings,
      .wide-donut-legends {{
        gap: 16px;
        padding-left: 8px;
        padding-right: 8px;
      }}
      .wide-donut-chart {{ height: 470px; }}
      .wide-stage-legend,
      .wide-stage-legend-capabilities {{ grid-template-columns: 1fr; }}
    }}
    @media (max-width: 760px) {{
      .wide-donut-panel {{ overflow-x: auto; }}
      .wide-donut-headings,
      .wide-donut-legends,
      .wide-chart-stage {{ min-width: 980px; }}
    }}
    .donut-grid {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 34px minmax(0, 1fr) 34px minmax(0, 1fr);
      gap: 0;
      margin-top: 14px;
      align-items: stretch;
    }}
    .donut-flow-arrow {{
      display: flex;
      align-items: center;
      justify-content: center;
      position: relative;
      min-width: 34px;
      pointer-events: none;
      z-index: 5;
    }}
    .donut-flow-line {{
      position: absolute;
      top: 45.5%;
      left: -5px;
      right: 1px;
      height: 3px;
      border-radius: 999px;
      background: #5B21B6;
    }}
    .donut-flow-head {{
      position: absolute;
      top: calc(45.5% - 7px);
      right: -2px;
      width: 0;
      height: 0;
      border-top: 8px solid transparent;
      border-bottom: 8px solid transparent;
      border-left: 12px solid #5B21B6;
    }}
    .donut-card {{
      border: 1px solid #e5e7eb;
      border-radius: 14px;
      background: #ffffff;
      padding: 14px 12px 8px 12px;
      min-width: 0;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }}
    .donut-card-title {{
      min-height: 48px;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 0 12px 4px 12px;
      text-align: center;
      font-size: clamp(14px, 1.25vw, 18px);
      font-weight: 700;
      line-height: 1.3;
      color: #111827;
      overflow-wrap: anywhere;
    }}
    .donut-card-subtitle {{
      min-height: 34px;
      display: flex;
      align-items: flex-start;
      justify-content: center;
      padding: 0 18px 2px 18px;
      text-align: center;
      font-size: 12.5px;
      line-height: 1.35;
      color: #526070;
    }}
    .executive-donut-chart {{
      position: absolute;
      inset: 0;
      height: 420px;
      width: 100%;
    }}
    .donut-stage {{
      position: relative;
      height: 420px;
      width: 100%;
      flex: 0 0 420px;
      overflow: visible;
    }}
    .donut-stage-clean .executive-donut-chart {{
      position: absolute;
      inset: 0;
    }}
    .donut-callout {{
      position: absolute;
      z-index: 6;
      width: 116px;
      pointer-events: none;
      line-height: 1.12;
    }}
    .donut-callout b {{
      display: block;
      font-size: 22px;
      line-height: 1;
      margin-bottom: 5px;
      font-weight: 800;
    }}
    .donut-callout span {{
      display: block;
      font-size: 12px;
      line-height: 1.15;
      margin-bottom: 5px;
    }}
    .donut-callout strong {{
      display: block;
      font-size: 16px;
      line-height: 1.05;
      font-weight: 800;
    }}
    .donut-callout-left {{
      left: 4px;
      bottom: 24px;
      text-align: left;
      color: #8B6FD6;
    }}
    .donut-callout-right {{
      right: 0;
      top: 50%;
      transform: translateY(-50%);
      text-align: left;
      color: #4C1D95;
    }}
    .capability-donut-legend {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px 18px;
      padding: 10px 18px 18px 18px;
      margin-top: -4px;
      min-height: 132px;
      align-content: start;
    }}
    .capability-legend-item {{
      display: grid;
      grid-template-columns: 16px minmax(0, 1fr);
      column-gap: 10px;
      align-items: start;
      font-size: 12px;
      line-height: 1.35;
      color: #344054;
      min-width: 0;
    }}
    .capability-legend-swatch {{
      width: 16px;
      height: 16px;
      border-radius: 3px;
      margin-top: 2px;
    }}
    .capability-legend-name {{
      font-weight: 700;
      font-size: 12.5px;
      overflow-wrap: anywhere;
    }}
    .capability-legend-metrics {{ font-size: 11.5px; color: #667085; }}
    .donut-note {{ margin-top: 14px; border: 1px solid #e5e7eb; border-radius: 12px; background: #ffffff; padding: 12px 14px; color: #526070; font-size: 13px; }}
    .supplier-share-header {{ display: flex; justify-content: space-between; gap: 16px; align-items: flex-start; flex-wrap: wrap; }}
    .supplier-view-control {{ display: flex; flex-direction: column; gap: 5px; min-width: 240px; }}
    .supplier-view-control label {{ font-size: 11px; text-transform: uppercase; letter-spacing: .04em; color: #667085; }}
    .supplier-view-control select {{ border: 1px solid #d0d5dd; border-radius: 10px; padding: 8px 10px; font-size: 13px; background: #ffffff; }}
    .supplier-insight-strip {{ display: grid; grid-template-columns: repeat(4, minmax(180px, 1fr)); gap: 12px; margin: 12px 0 8px 0; }}
    .supplier-mini-card {{ border: 1px solid #e5e7eb; border-radius: 12px; background: #ffffff; padding: 12px 14px; box-shadow: 0 1px 2px rgba(0,0,0,0.03); min-width: 0; }}
    .supplier-mini-card span {{ display: block; font-size: 11px; text-transform: uppercase; letter-spacing: .04em; color: #667085; margin-bottom: 5px; }}
    .supplier-mini-card b {{ display: block; font-size: clamp(16px, 1.6vw, 22px); line-height: 1.05; color: #111827; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .supplier-mini-card em {{ display: block; margin-top: 5px; font-size: 12px; line-height: 1.25; color: #526070; font-style: normal; }}
    .supplier-mini-card.accenture {{ border-color: #d9b8ff; background: #fbf7ff; }}
    .supplier-mini-card.accenture b {{ color: #A100FF; }}
    .supplier-mini-card.leader {{ border-color: #b7ddb9; background: #f3fbf3; }}
    .supplier-mini-card.leader b {{ color: #2E7D32; }}
    @media (max-width: 1150px) {{
      .donut-grid {{ grid-template-columns: 1fr; gap: 14px; }}
      .donut-flow-arrow {{ display: none; }}
      .donut-card-title {{ min-height: 44px; font-size: 16px; }}
      .donut-stage {{ height: 400px; flex-basis: 400px; }}
      .executive-donut-chart {{ height: 400px; }}
    }}
    @media (max-width: 760px) {{
      .cards, .grid2, .donut-grid {{ grid-template-columns: 1fr; }}
      .donut-callout {{ display: none; }}
      .donut-stage {{ height: 380px; flex-basis: 380px; }}
      .executive-donut-chart {{ height: 380px; }}
      .capability-donut-legend {{ grid-template-columns: 1fr; }}
      .donut-card-title {{ min-height: 0; font-size: 15px; padding: 2px 8px 8px 8px; }}
    }}
  </style>
</head>
<body>
<div class="wrap">
  <h1>Accenture Defence Addressable Market Dashboard</h1>
  <p class="subtitle">
  </p>

    <section class="executive-summary" aria-labelledby="executiveSummaryTitle">
        <h2 id="executiveSummaryTitle">Executive Summary</h2>

        <p>
            This dashboard provides an executive view of
            <strong tabindex="0"
                    data-tooltip="The subset of Defence procurement classified as relevant to Accenture's consulting, technology, engineering, data, cloud, cyber and managed-services capabilities.">
            Accenture's addressable Defence market
            </strong>.
            It shows the size of the opportunity, Accenture's historical position,
            the service offerings where demand is concentrated, and the suppliers
            currently winning the work.
        </p>

        <p>
            The
            <strong tabindex="0"
                    data-tooltip="Headline measures covering total Defence procurement, addressable market value, Accenture wins, competitor-owned market and average annual values.">
            headline KPI cards
            </strong>
            summarise market scale and Accenture's share. The
            <strong tabindex="0"
                    data-tooltip="Three donut charts showing what proportion of Defence procurement is addressable, how the addressable market is split between Accenture and competitors, and how Accenture's wins are distributed by service offering.">
            addressable-market overview
            </strong>
            explains the relationship between total procurement, Accenture-addressable
            work and competitor-owned opportunity. The
            <strong tabindex="0"
                    data-tooltip="A service-offering chart comparing total addressable value with the value won by Accenture in each service offering.">
            service-offering view
            </strong>
            identifies the largest areas of demand and where Accenture has the strongest
            or weakest market penetration.
        </p>
    </section>

  <div class="cards">
    {cards_html}
  </div>

  <div class="grid2">
    <div class="panel panel-wide">
    {chart_donut_overview_interactive(df, value_col)}
</div>

<div class="panel panel-wide">
    {chart_capability_interactive(df, value_col, top_n_capabilities)}
</div>


<div class="panel panel-wide">
    {chart_market_share_by_year(df, value_col)}
</div>

<div class="panel panel-wide">
    {chart_supplier_share(df, value_col, top_n_suppliers)}
</div>
  </div>

  <p class="note">
    Important: using total contract value means multi-year contracts are counted at their full ceiling/value in the year they appear.
    Use <code>--value-mode annualised</code> when you want a clearer yearly run-rate comparison using the Value Per Year column.
  </p>
</div>

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

  document.querySelectorAll('.executive-summary strong[data-tooltip]').forEach(target => {{
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
    output_html.write_text(html, encoding="utf-8")



AUDIT_OUTPUT_NAMES = [
    "dashboard_input_snapshot.csv",
    "canonical_supplier_share_audit.csv",
    "supplier_identity_mapping_audit.csv",
    "supplier_canonical_variant_summary.csv",
    "classification_capability_summary.csv",
    "classification_detailed_capability_summary.csv",
    "classification_top_addressable_contracts_audit.csv",
    "classification_generic_strategy_audit.csv",
    "classification_review_queue.csv",
    "market_relevance_exclusions_audit.csv",
]


def _audit_cache_key(args: argparse.Namespace, input_path: Path) -> dict[str, object]:
    stat = input_path.stat()
    script_path = Path(__file__).resolve()
    script_stat = script_path.stat()
    return {
        "input_path": str(input_path.resolve()),
        "input_size": stat.st_size,
        "input_mtime_ns": stat.st_mtime_ns,
        "script_size": script_stat.st_size,
        "script_mtime_ns": script_stat.st_mtime_ns,
        "value_mode": args.value_mode,
        "fy_start": args.fy_start,
        "fy_end": args.fy_end,
    }


def _audit_cache_is_valid(
    output_dir: Path,
    cache_key: dict[str, object],
) -> bool:
    manifest_path = output_dir / "dashboard1_audit_cache.json"
    if not manifest_path.exists():
        return False

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    if manifest.get("cache_key") != cache_key:
        return False

    required_outputs = manifest.get("outputs", [])
    return bool(required_outputs) and all((output_dir / name).exists() for name in required_outputs)


def _write_audit_cache_manifest(
    output_dir: Path,
    cache_key: dict[str, object],
) -> None:
    existing_outputs = [
        name for name in AUDIT_OUTPUT_NAMES
        if (output_dir / name).exists()
    ]
    payload = {
        "cache_key": cache_key,
        "outputs": existing_outputs,
    }
    (output_dir / "dashboard1_audit_cache.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )



def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_all = load_or_classify(args)
    df_all = use_master_supplier_columns(df_all)
    value_col = get_value_col(df_all, args.value_mode)
    df_all = clean_value_columns(df_all, [VALUE_COL, ANNUALISED_VALUE_COL])
    df = filter_defence_scope(df_all, include_all_agencies=args.include_all_agencies)
    df = filter_financial_year_range(df, args.fy_start, args.fy_end)
    summary = summarise(df, value_col)

    classified_path = output_dir / "dashboard_input_snapshot.csv"
    summary_path = output_dir / "market_summary.csv"
    dashboard_path = output_dir / "AccentureDefenceTAMDashboard.html"

    export_defence_scope_audit(df_all, df, value_col, output_dir)
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    export_competitor_outputs(df, value_col, output_dir)

    input_path = Path(args.input)
    cache_key = _audit_cache_key(args, input_path)
    use_cached_audits = (
        not args.refresh_audits
        and _audit_cache_is_valid(output_dir, cache_key)
    )

    if use_cached_audits:
        print("Reusing cached Dashboard 1 audit outputs.")
    else:
        print("Refreshing Dashboard 1 audit outputs.")
        # Snapshot and QA outputs remain available, but are regenerated only
        # when the source, script or relevant runtime options have changed.
        df.to_csv(classified_path, index=False)
        validate_supplier_consolidation(df, value_col, output_dir)
        export_classification_audit_outputs(df, value_col, output_dir)
        export_market_relevance_audit(df, value_col, output_dir)
        _write_audit_cache_manifest(output_dir, cache_key)

    build_html(
        df,
        summary,
        value_col,
        dashboard_path,
        args.top_n_suppliers,
        args.top_n_capabilities,
        args.value_mode,
        args.fy_start,
        args.fy_end,
    )

    print("Dashboard complete")
    print(f"Value mode: {args.value_mode} / column used: {value_col}")
    print(f"Value basis: {'Total contract value' if args.value_mode == 'total' else 'Annualised contract value / run-rate'}")
    print(f"Selected period: {_format_fy_range_label(args.fy_start, args.fy_end)}")
    print(f"Financial years detected: {summary['years']}")
    print(f"Rows after Defence filter: {len(df):,}")
    print(f"Defence filter active: {not args.include_all_agencies}")
    print(f"Total procurement: {money(summary['total'])}")
    print(f"Addressable market: {money(summary['addressable'])} ({pct(summary['addressable_pct_total'])} of total)")
    print(f"Accenture addressable wins: {money(summary['accenture_addressable'])} ({pct(summary['accenture_share_of_addressable'])} of addressable)")
    print(f"Competitor-owned market: {money(summary['competitor_market'])} ({pct(summary['competitor_market_share'])} of addressable)")
    print(f"Avg annual addressable market: {money(summary['avg_addressable_per_year'])}")
    print(f"Avg annual Accenture addressable wins: {money(summary['avg_accenture_addressable_per_year'])}")
    print(f"Wrote: {dashboard_path}")
    print(f"Wrote: {classified_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {output_dir / 'addressable_competitor_contracts.csv'}")
    print(f"Wrote: {output_dir / 'addressable_supplier_market_share.csv'}")
    print(f"Wrote: {output_dir / 'canonical_supplier_share_audit.csv'}")
    print(f"Wrote: {output_dir / 'supplier_identity_mapping_audit.csv'}")
    print(f"Wrote: {output_dir / 'supplier_canonical_variant_summary.csv'}")
    print(f"Wrote: {output_dir / 'competitor_capability_share.csv'}")
    print(f"Wrote: {output_dir / 'defence_scope_audit_by_agency.csv'}")
    print(f"Wrote: {output_dir / 'classification_capability_summary.csv'}")
    print(f"Wrote: {output_dir / 'classification_detailed_capability_summary.csv'}")
    print(f"Wrote: {output_dir / 'classification_top_addressable_contracts_audit.csv'}")
    print(f"Wrote: {output_dir / 'classification_generic_strategy_audit.csv'}")
    print(f"Wrote: {output_dir / 'classification_review_queue.csv'}")
    print(f"Wrote: {output_dir / 'market_relevance_exclusions_audit.csv'}")


if __name__ == "__main__":
    main()
