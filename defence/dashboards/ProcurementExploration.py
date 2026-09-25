from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd


# =========================
# Canonical source columns
# =========================
AGENCY_COL = "Agency"
DIVISION_COL = "Agency Division"
BRANCH_COL = "Agency Branch"
FY_COL = "Financial Year"
VALUE_COL = "Value"
ANNUALISED_VALUE_COL = "Value Per Year"
CAPABILITY_COL = "capability"
RP_COL = "ReinventionPartner"
RE_COL = "ReinventionEngine"
DOMAIN_COL = "defence_domain"
SUPPLIER_GROUP_COL = "supplier_group"
RAW_SUPPLIER_COL = "Supplier Name"
CN_ID_COL = "CN ID"
DESCRIPTION_COL = "Description"
PROCUREMENT_METHOD_COL = "Procurement Method"
CONSULTANCY_COL = "Consultancy"
CATEGORY_COL = "Category"
AMENDMENTS_COL = "Amendments"
EXTENSIONS_CANDIDATES = ["#Extensions", "Extensions", "Extension Count", "Number of Extensions"]

START_DATE_CANDIDATES = [
    "Start Date",
    "Contract Start Date",
    "Contract Period Start",
    "Contract Start",
]
END_DATE_CANDIDATES = [
    "End Date",
    "Contract End Date",
    "Contract Period End",
    "Contract End",
    "Expiry Date",
]

REQUIRED_COLUMNS = {
    AGENCY_COL,
    DIVISION_COL,
    BRANCH_COL,
    VALUE_COL,
    CAPABILITY_COL,
    "is_addressable",
    SUPPLIER_GROUP_COL,
    RP_COL,
    RE_COL,
    DOMAIN_COL,
}


# =========================
# Division consolidation
# =========================
DIVISION_ALIASES = {
    "ADFHQ": "ADFHQ",
    "AUSTRALIAN DEFENCE HEAD QUARTERS": "ADFHQ",
    "ADHQ": "ADFHQ",

    "AIR FORCE": "AIRF",
    "AIR FORCE CAPABILITY DIVISION": "AIRF",
    "AIR FORCE EXECUTIVE SUB GROUP": "AIRF",
    "AIRFORCE": "AIRF",
    "AIRF": "AIRF",

    "ARMY": "ARMY",
    "ARMY HEADQUARTERS": "ARMY",

    "AUSTRALIAN SIGNALS DIRECTORATE": "ASD",
    "ASD": "ASD",

    "AUSTRALIAN SUBMARINE AGENCY": "ASA",
    "ASA": "ASA",

    "CASG": "CASG",
    "CAPABILITY ACQUISITION AND SUSTAINMENT GROUP": "CASG",

    "CHIEF FINANCE OFFICER GROUP": "CFOG",
    "CFO": "CFOG",
    "CFOG": "CFOG",
    "DEFENCE FINANCE GROUP": "DFG",
    "DFG": "DFG",

    "CIOG": "CIOG",
    "CHIEF INFORMATION OFFICER GROUP": "CIOG",
    "CIOG CTO": "CIOG CTO",

    "DEFENCE INTELLIGENCE GROUP": "DIG",
    "DIG": "DIG",

    "DEFENCE PEOPLE GROUP": "DPG",
    "DPG": "DPG",

    "DEFENCE SUPPORT AND REFORM GROUP": "DSRG",
    "DSRG": "DSRG",

    "DEFENCE SCIENCE AND TECHNOLOGY GROUP": "DSTG",
    "DST GROUP": "DSTG",
    "DSTG": "DSTG",
    "DSTO": "DSTG",

    "ESTATE AND INFRASTRUCTURE GROUP": "E&IG",
    "ESTATE INFRASTRUCTURE GROUP": "E&IG",
    "E IG": "E&IG",
    "E I": "E&IG",

    "GUIDED WEAPONS AND EXPLOSIVE ORDNANCE": "GWEO",
    "GWEO": "GWEO",

    "INTELLIGENCE AND SECURITY": "I&S",
    "INTELLIGENCE SECURITY": "I&S",
    "I S": "I&S",

    "JOINT CAPABILITIES GROUP": "JCG",
    "JOINT CAPABILITY GROUP": "JCG",
    "JCG": "JCG",

    "JOINT OPERATIONS COMMAND": "JOC",
    "JOC": "JOC",

    "NAVY": "NAVY",
    "NAVY STRATEGIC COMMAND": "NAVY",

    "NUCLEAR POWERED SUBMARINE PROGRAM": "NPS",
    "NPS": "NPS",
    "NAVAL SHIPBUILDING AND SUSTAINMENT GROUP": "NSSG",
    "NSSG": "NSSG",

    "OFFICE OF THE SECRETARY AND CDF": "OSCDF",
    "OSCDF": "OSCDF",
    "OSCF": "OSCDF",

    "SECURITY AND ESTATE GROUP": "SEG",
    "SEG": "SEG",

    "STRATEGIC POLICY AND INDUSTRY": "SP&I",
    "STRATEGIC POLICY AND INTELLIGENCE": "SP&I",
    "SP I": "SP&I",
    "SPI": "SP&I",
    "STRATEGIC POLICY": "SP&I",

    "VICE CHIEF OF THE DEFENCE FORCE": "VCDF",
    "VCDF": "VCDF",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Procurement Exploration dashboard."
    )
    parser.add_argument(
        "--input",
        default="master_output/master_defence_contracts.parquet",
        help="Canonical master Defence parquet.",
    )
    parser.add_argument(
        "--output-dir",
        default="ProcurementExplorationDashboard_output",
    )
    parser.add_argument(
        "--value-mode",
        choices=["total", "annualised"],
        default="total",
        help="Value used in KPIs and tables.",
    )
    return parser.parse_args()


def parse_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )


def clean_text(series: pd.Series, fallback: str) -> pd.Series:
    return (
        series.fillna(fallback)
        .astype(str)
        .str.replace(r"_x000D_", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .replace("", fallback)
    )


def normalise_key(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).replace("_x000D_", " ").upper()
    text = text.replace("&", " AND ")
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    text = re.sub(r"\bAND\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {str(c).lower(): c for c in df.columns}
    for candidate in candidates:
        found = lower_map.get(candidate.lower())
        if found is not None:
            return found
    return None


def resolve_ddg(raw_division: object, raw_branch: object) -> str:
    """Resolve DDG using the confirmed Defence organisation hierarchy.

    The abbreviation DDG is authoritative for Defence Digital Group. Branch
    labels such as ``DDG - ICTDD`` provide supporting validation and must not
    cause the division to be labelled unresolved or Defence Delivery Group.
    A fully written ``Defence Delivery Group`` value is retained only when that
    exact organisation name is present in the source division.
    """
    division_text = normalise_key(raw_division)
    branch_text = normalise_key(raw_branch)

    if "DEFENCE DELIVERY GROUP" in division_text:
        return "Defence Delivery Group"

    if (
        division_text == "DDG"
        or "DEFENCE DIGITAL GROUP" in division_text
        or branch_text == "DDG"
        or branch_text.startswith("DDG ")
        or "DEFENCE DIGITAL GROUP" in branch_text
    ):
        return "DDG - Defence Digital Group"

    # Historical DDG records are confirmed as Defence Digital Group. There is
    # deliberately no DDG - Unresolved fallback.
    return "DDG - Defence Digital Group"


def consolidated_division(raw_division: object, raw_branch: object) -> str:
    key = normalise_key(raw_division)

    # Historical DDG CTO was absorbed into Defence Digital Group. Roll those
    # contracts into the current DDG division while retaining the source label
    # separately on each contract for audit/display.
    if key in {"DDG CTO", "DEFENCE DIGITAL GROUP CTO", "CHIEF TECHNOLOGY OFFICER DDG"}:
        return "DDG - Defence Digital Group"

    if key == "DDG" or "DEFENCE DIGITAL GROUP" in key or "DEFENCE DELIVERY GROUP" in key:
        return resolve_ddg(raw_division, raw_branch)

    if key in DIVISION_ALIASES:
        return DIVISION_ALIASES[key]

    cleaned = re.sub(r"\s+", " ", str(raw_division or "")).strip()
    return cleaned or "Unspecified division"


def consolidated_branch(raw_branch: object, consolidated_div: str) -> str:
    raw = "" if raw_branch is None or pd.isna(raw_branch) else str(raw_branch)
    text = raw.replace("_x000D_", " ")
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return "Unspecified branch"

    # Remove repeated division prefixes where present.
    div_variants = [
        consolidated_div,
        consolidated_div.replace("&", "AND"),
        consolidated_div.replace(" - ", " "),
    ]
    for variant in div_variants:
        text = re.sub(
            rf"^\s*{re.escape(variant)}\s*[-:|/]\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

    # Standardise common suffixes and punctuation.
    text = re.sub(r"\bDIV\b", "Division", text, flags=re.IGNORECASE)
    text = re.sub(r"\bBR\b", "Branch", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*[-–—]\s*", " - ", text)
    text = re.sub(r"\s+", " ", text).strip(" -")

    # Title case all-caps text while preserving acronyms.
    if text.isupper() and len(text) > 4:
        words = []
        for token in text.split():
            words.append(token if len(token) <= 5 and token.isalpha() else token.title())
        text = " ".join(words)

    return text or "Unspecified branch"


def financial_year_start(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    match = re.search(r"(19\d{2}|20\d{2})", str(value))
    return int(match.group(1)) if match else None


def load_data(path: Path, value_mode: str) -> tuple[pd.DataFrame, str]:
    if not path.exists():
        raise SystemExit(f"Master Defence parquet not found: {path}")

    df = pd.read_parquet(path)
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise SystemExit(
            "Input is missing required columns: " + ", ".join(missing)
        )

    value_col = ANNUALISED_VALUE_COL if value_mode == "annualised" else VALUE_COL
    if value_col not in df.columns:
        raise SystemExit(f"Requested value column not found: {value_col}")

    out = df.copy()
    out["is_addressable"] = parse_bool(out["is_addressable"])

    if "is_market_relevant" in out.columns:
        out["is_market_relevant"] = parse_bool(out["is_market_relevant"])
    else:
        out["is_market_relevant"] = True

    # Match Dashboard 1 exactly: TAM is defined by is_addressable only.
    # is_market_relevant is retained as an audit field but is not used as an
    # additional dashboard filter.
    out = out[out["is_addressable"]].copy()
    if out.empty:
        raise SystemExit("No addressable contracts were found.")

    out[AGENCY_COL] = clean_text(out[AGENCY_COL], "Unknown agency")
    out[DIVISION_COL] = clean_text(out[DIVISION_COL], "Unspecified division")
    out[BRANCH_COL] = clean_text(out[BRANCH_COL], "Unspecified branch")
    out[CAPABILITY_COL] = clean_text(out[CAPABILITY_COL], "Unclassified")
    out[RP_COL] = clean_text(out[RP_COL], "Unclassified")
    out[RE_COL] = clean_text(out[RE_COL], "Unclassified")
    out[DOMAIN_COL] = clean_text(out[DOMAIN_COL], "Unmapped")

    # Preserve the source division before any dashboard roll-up. This keeps
    # historical identifiers such as DDG CTO visible on contract records even
    # though they are reported under the current DDG organisation.
    out["_division_original"] = out[DIVISION_COL].astype(str)

    # Canonical supplier identity comes only from build_master.py.
    # Raw Supplier Name is retained for contract-level audit/display only.
    out["_supplier"] = clean_text(out[SUPPLIER_GROUP_COL], "Unknown supplier")

    out["_division"] = [
        consolidated_division(div, branch)
        for div, branch in zip(out[DIVISION_COL], out[BRANCH_COL])
    ]
    out["_branch"] = [
        consolidated_branch(branch, div)
        for branch, div in zip(out[BRANCH_COL], out["_division"])
    ]

    out["_value"] = pd.to_numeric(out[value_col], errors="coerce").fillna(0).clip(lower=0)
    out["_total_value"] = pd.to_numeric(out[VALUE_COL], errors="coerce").fillna(0).clip(lower=0)

    out[FY_COL] = clean_text(out[FY_COL], "Unknown FY") if FY_COL in out.columns else "Unknown FY"
    out["_fy_start"] = out[FY_COL].map(financial_year_start)

    start_col = first_existing(out, START_DATE_CANDIDATES)
    end_col = first_existing(out, END_DATE_CANDIDATES)

    out["_start"] = (
        pd.to_datetime(out[start_col], errors="coerce", dayfirst=True)
        if start_col
        else pd.Series(pd.NaT, index=out.index)
    )
    out["_end"] = (
        pd.to_datetime(out[end_col], errors="coerce", dayfirst=True)
        if end_col
        else pd.Series(pd.NaT, index=out.index)
    )
    duration_days = (out["_end"] - out["_start"]).dt.days
    out["_duration_days"] = duration_days.where(duration_days >= 0)

    return out, value_col


def fmt_date(value: pd.Timestamp | pd.NaT) -> str:
    if pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%d %b %Y")


def fmt_duration(days: object) -> str:
    if days is None or pd.isna(days):
        return ""
    days = float(days)
    years = days / 365.25
    if years >= 2:
        return f"{years:.1f} yrs"
    if years >= 1:
        return f"{years:.2f} yrs"
    return f"{int(round(days))} days"


def safe_str(value: object, fallback: str = "") -> str:
    if value is None or pd.isna(value):
        return fallback
    return str(value).strip()


def build_payload(df: pd.DataFrame) -> dict[str, object]:
    extensions_col = first_existing(df, EXTENSIONS_CANDIDATES)

    records: list[dict[str, object]] = []
    for _, row in df.iterrows():
        record = {
            "agency": safe_str(row.get(AGENCY_COL), "Unknown agency"),
            "division": safe_str(row.get("_division"), "Unspecified division"),
            "division_original": safe_str(row.get("_division_original")),
            "legacy_org_identifier": (
                safe_str(row.get("_division_original"))
                if safe_str(row.get("_division_original"))
                and safe_str(row.get("_division_original")) != safe_str(row.get("_division"))
                else ""
            ),
            "branch": safe_str(row.get("_branch"), "Unspecified branch"),
            "domain": safe_str(row.get(DOMAIN_COL), "Unmapped"),
            "procuring_entity": (
                (
                    safe_str(row.get("_division"))
                    if safe_str(row.get("_division")).lower()
                    not in {"", "unspecified division", "unknown division"}
                    else "(blank)"
                )
                + " > "
                + (
                    safe_str(row.get("_branch"))
                    if safe_str(row.get("_branch")).lower()
                    not in {"", "unspecified branch", "unknown branch"}
                    else "(blank)"
                )
                + (
                    " [Historical division: " + safe_str(row.get("_division_original")) + "]"
                    if safe_str(row.get("_division_original"))
                    and safe_str(row.get("_division_original")) != safe_str(row.get("_division"))
                    else ""
                )
            ),
            "financial_year": safe_str(row.get(FY_COL), "Unknown FY"),
            "fy_start": int(row["_fy_start"]) if pd.notna(row.get("_fy_start")) else None,
            "contract_id": safe_str(row.get(CN_ID_COL)),
            "description": safe_str(row.get(DESCRIPTION_COL)),
            "supplier": safe_str(row.get("_supplier"), "Unknown supplier"),
            "capability": safe_str(row.get(CAPABILITY_COL), "Unclassified"),
            "reinvention_partner": safe_str(row.get(RP_COL), "Unclassified"),
            "reinvention_engine": safe_str(row.get(RE_COL), "Unclassified"),
            "value": float(row.get("_value", 0) or 0),
            "total_value": float(row.get("_total_value", 0) or 0),
            "start_date": fmt_date(row.get("_start")),
            "end_date": fmt_date(row.get("_end")),
            "start_date_iso": (
                pd.Timestamp(row.get("_start")).strftime("%Y-%m-%d")
                if pd.notna(row.get("_start"))
                else ""
            ),
            "end_date_iso": (
                pd.Timestamp(row.get("_end")).strftime("%Y-%m-%d")
                if pd.notna(row.get("_end"))
                else ""
            ),
            "duration_days": (
                float(row["_duration_days"])
                if pd.notna(row.get("_duration_days"))
                else None
            ),
            "duration_label": fmt_duration(row.get("_duration_days")),
            "procurement_method": safe_str(row.get(PROCUREMENT_METHOD_COL)),
            "category": safe_str(row.get(CATEGORY_COL)),
            "extensions": safe_str(row.get(extensions_col)) if extensions_col else "",
        }
        records.append(record)

    agencies = sorted(df[AGENCY_COL].dropna().astype(str).unique().tolist())
    domains = sorted(df[DOMAIN_COL].dropna().astype(str).unique().tolist())
    hierarchy = (
        df[[AGENCY_COL, "_division", "_branch"]]
        .drop_duplicates()
        .sort_values([AGENCY_COL, "_division", "_branch"])
    )

    return {
        "records": records,
        "agencies": agencies,
        "domains": domains,
        "hierarchy": [
            {
                "agency": str(r[AGENCY_COL]),
                "division": str(r["_division"]),
                "branch": str(r["_branch"]),
            }
            for _, r in hierarchy.iterrows()
        ],
    }


def build_html(payload: dict[str, object], output_path: Path, value_mode: str) -> None:
    agencies = payload["agencies"]
    all_agencies_label = "All Defence agencies"
    default_agency = all_agencies_label

    agency_options = "\n".join(
        [
            f'<option value="{html_escape(all_agencies_label)}">{html_escape(all_agencies_label)}</option>'
        ]
        + [
            f'<option value="{html_escape(a)}">{html_escape(a)}</option>'
            for a in agencies
        ]
    )

    value_basis = (
        "Total contract value"
        if value_mode == "total"
        else "Annualised contract value"
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Procurement Exploration</title>
<style>
:root {{
  --purple:#A100FF;
  --ink:#101828;
  --muted:#667085;
  --line:#E4E7EC;
  --bg:#F6F8FB;
  --card:#FFFFFF;
  --soft:#F9FAFB;
  --green:#067647;
  --amber:#B54708;
}}
* {{ box-sizing:border-box; }}
body {{
  margin:0;
  font-family:Arial,Helvetica,sans-serif;
  background:var(--bg);
  color:var(--ink);
}}
.wrap {{ max-width:1720px; margin:0 auto; padding:28px; }}
h1 {{ margin:0 0 8px 0; font-size:30px; }}
.subtitle {{ margin:0 0 20px 0; color:#526070; font-size:15px; line-height:1.45; }}
.scope {{
  color:#667085;
  font-size:12px;
  margin:0 0 8px 0;
}}
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
.executive-summary .source {{
  margin-top:12px;
  color:#667085;
  font-size:12px;
}}
.executive-summary strong[data-tooltip] {{
  color:#111827;
  border-bottom:1px dotted #667085;
  cursor:help;
  outline:none;
}}
.exec-tooltip {{
  position:fixed;
  left:0;
  top:0;
  max-width:min(360px, calc(100vw - 24px));
  padding:10px 12px;
  border-radius:9px;
  background:#101828;
  color:#ffffff;
  font-size:12px;
  font-weight:400;
  line-height:1.4;
  text-align:left;
  white-space:normal;
  box-shadow:0 8px 24px rgba(16,24,40,.22);
  opacity:0;
  visibility:hidden;
  pointer-events:none;
  z-index:2147483647;
  transition:opacity .12s ease;
}}
.exec-tooltip.is-visible {{
  opacity:1;
  visibility:visible;
}}
.selectors {{
  display:grid;
  grid-template-columns:repeat(4,minmax(210px,1fr));
  gap:10px;
  margin:12px 0;
}}
.selectors .field {{
  background:#ffffff;
  border:1px solid #e5e7eb;
  border-radius:12px;
  padding:10px 12px;
}}
.field label {{
  display:block;
  font-size:10px;
  text-transform:uppercase;
  letter-spacing:.04em;
  color:#667085;
  margin-bottom:3px;
}}
select,input {{
  width:100%;
  height:34px;
  border:1px solid #d0d5dd;
  border-radius:8px;
  background:#fff;
  padding:0 10px;
  font-size:13px;
}}
.kpis {{
  display:grid;
  grid-template-columns:repeat(6,minmax(0,1fr));
  gap:11px;
  margin:18px 0 22px 0;
}}
.kpi {{
  background:#fff;
  border:1px solid var(--line);
  border-radius:14px;
  padding:14px 15px;
  min-width:0;
  box-shadow:0 1px 2px rgba(16,24,40,.04);
}}
.kpi:first-child {{ border-top:4px solid var(--purple); }}
.kpi-label {{
  color:var(--muted);
  font-size:11px;
  text-transform:uppercase;
  font-weight:700;
  letter-spacing:.045em;
  margin-bottom:7px;
}}
.kpi-value {{
  font-size:clamp(19px,1.7vw,27px);
  font-weight:750;
  line-height:1.08;
}}
.kpi-sub {{
  color:var(--muted);
  font-size:12px;
  margin-top:5px;
  line-height:1.35;
}}

.insight-grid {{
  display:grid;
  grid-template-columns:repeat(3,minmax(0,1fr));
  gap:14px;
  margin:0 0 20px 0;
}}
.leader-card {{
  background:#fff;
  border:1px solid var(--line);
  border-radius:14px;
  padding:16px 17px;
  box-shadow:0 1px 2px rgba(16,24,40,.04);
}}
.leader-card h3 {{ margin:0 0 4px 0; font-size:17px; }}
.leader-card p {{ margin:0 0 12px 0; color:var(--muted); font-size:12px; line-height:1.4; }}
.leader-row {{
  display:grid;
  grid-template-columns:28px minmax(0,1fr) auto;
  gap:9px;
  align-items:center;
  padding:9px 0;
  border-top:1px solid #F0F2F5;
}}
.leader-row:first-child {{ border-top:0; }}
.leader-rank {{
  width:24px; height:24px; border-radius:50%;
  display:flex; align-items:center; justify-content:center;
  background:#F4EBFF; color:#7F00CC; font-size:11px; font-weight:800;
}}
.leader-name {{ min-width:0; font-size:13px; font-weight:700; color:#1D2939; }}
.leader-detail {{ margin-top:2px; color:var(--muted); font-size:11px; font-weight:400; }}
.leader-value {{ text-align:right; white-space:nowrap; font-size:13px; font-weight:800; }}
.leader-positive {{ color:#067647; }}
.leader-negative {{ color:#B42318; }}
.leader-empty {{ padding:18px 0; color:var(--muted); font-size:12px; }}

.toolbar {{
  margin:0 0 18px 0;
  background:#fff;
  border:1px solid var(--line);
  border-radius:14px;
  padding:14px;
  display:grid;
  grid-template-columns:2fr 1fr 1fr 1fr 1.1fr 1fr auto auto;
  gap:10px;
  align-items:end;
}}
.check-wrap {{
  min-height:42px;
  display:flex;
  align-items:center;
  gap:8px;
  padding:0 6px;
  color:#344054;
  font-size:13px;
}}
.check-wrap input {{
  width:auto;
  min-height:auto;
}}
.clear-btn {{
  min-height:42px;
  border:1px solid #D0D5DD;
  border-radius:9px;
  background:#fff;
  color:#344054;
  padding:0 16px;
  font-size:13px;
  font-weight:700;
  cursor:pointer;
  white-space:nowrap;
}}
.clear-btn:hover {{
  border-color:var(--purple);
  color:#7F00CC;
  background:#FCF5FF;
}}
.summary-line {{
  margin:14px 2px 10px;
  color:#475467;
  font-size:13px;
}}
.year-section {{
  margin-top:12px;
  background:#fff;
  border:1px solid var(--line);
  border-radius:14px;
  overflow:hidden;
}}
.year-toggle {{
  width:100%;
  border:0;
  background:#fff;
  padding:15px 17px;
  display:flex;
  justify-content:space-between;
  align-items:center;
  cursor:pointer;
  font-size:15px;
  font-weight:700;
  color:var(--ink);
}}
.year-toggle:hover {{ background:#FCFCFD; }}
.year-meta {{
  color:var(--muted);
  font-size:12px;
  font-weight:500;
}}
.chev {{
  display:inline-block;
  transition:transform .18s ease;
  margin-right:8px;
}}
.year-section.open .chev {{ transform:rotate(90deg); }}
.year-body {{
  display:none;
  border-top:1px solid var(--line);
}}
.year-section.open .year-body {{ display:block; }}
.table-wrap {{
  overflow:auto;
  max-height:620px;
}}
table {{
  width:100%;
  border-collapse:collapse;
  min-width:1480px;
  font-size:12px;
}}
thead th {{
  position:sticky;
  top:0;
  z-index:2;
  background:#F9FAFB;
  color:#475467;
  text-align:left;
  padding:10px 9px;
  border-bottom:1px solid var(--line);
  white-space:nowrap;
  cursor:pointer;
}}
tbody td {{
  padding:10px 9px;
  border-bottom:1px solid #F0F2F5;
  vertical-align:top;
}}
tbody tr:hover {{ background:#FCFCFD; }}
.money {{ text-align:right; white-space:nowrap; font-variant-numeric:tabular-nums; }}
.nowrap {{ white-space:nowrap; }}
.desc {{ width:420px; min-width:340px; max-width:480px; line-height:1.4; white-space:normal; overflow-wrap:anywhere; word-break:normal; }}
.supplier-accent {{
  color:#6B00B6;
  font-weight:700;
}}
.badge {{
  display:inline-block;
  border-radius:999px;
  padding:3px 7px;
  font-size:11px;
  font-weight:700;
  white-space:nowrap;
}}
.badge-large {{ background:#ECFDF3; color:var(--green); }}
.badge-long {{ background:#FFFAEB; color:var(--amber); }}
.empty {{
  background:#fff;
  border:1px dashed #D0D5DD;
  border-radius:14px;
  padding:36px;
  text-align:center;
  color:var(--muted);
  margin-top:16px;
}}
.method {{
  margin-top:16px;
  color:var(--muted);
  font-size:12px;
  line-height:1.5;
}}
@media(max-width:1100px) {{
  .selectors {{ grid-template-columns:1fr; }}
  .kpis {{ grid-template-columns:repeat(3,1fr); }}
  .toolbar {{ grid-template-columns:1fr 1fr; }}
  .insight-grid {{ grid-template-columns:1fr; }}
}}
@media(max-width:650px) {{
  .wrap {{ padding:15px; }}
  .kpis {{ grid-template-columns:1fr 1fr; }}
  .toolbar {{ grid-template-columns:1fr; }}
}}
</style>
</head>
<body>
<div class="wrap">
  <h1>Procurement Exploration</h1>
  

  <section class="executive-summary">
    <h2>Executive summary</h2>

    <p>
      This dashboard provides an interactive view of Accenture's addressable Defence procurement market, allowing users to explore procurement activity from
      <strong tabindex="0"
              data-tooltip="The default view combines Department of Defence, Australian Signals Directorate and Australian Submarine Agency procurement across the historical dataset.">
        All Defence agencies
      </strong>
      down to individual divisions, branches, suppliers and contracts.
    </p>

    <p>
      The dashboard supports opportunity identification, account planning and procurement analysis by revealing where Defence organisations are investing, which suppliers are winning work, and how demand is distributed across Accenture's service offerings. Users can progressively filter the data to investigate individual contracts or broader procurement trends.
    </p>
  </section>

  <section class="insight-grid">
    <div class="leader-card">
      <h3>Top 5 Defence spenders (in our TAM)</h3>
      <p id="spendSubtitle">Consolidated divisions ranked by total addressable contract value over the latest five financial years. These cards are not affected by page filters.</p>
      <div id="topSpenders"></div>
    </div>
    <div class="leader-card">
      <h3>Top 5 fastest-growing spenders (in our TAM)</h3>
      <p id="growthSubtitle">Average annual change over the latest five financial years.</p>
      <div id="topGrowth"></div>
    </div>
    <div class="leader-card">
      <h3>Top 5 fastest-declining spenders (in our TAM)</h3>
      <p id="declineSubtitle">Average annual decline over the latest five financial years.</p>
      <div id="topDecline"></div>
    </div>
  </section>

  <section class="selectors">
    <div class="field">
      <label for="agencySelect">Agency</label>
      <select id="agencySelect">{agency_options}</select>
    </div>
    <div class="field">
      <label for="divisionSelect">Consolidated division</label>
      <select id="divisionSelect"></select>
    </div>
    <div class="field">
      <label for="branchSelect">Consolidated branch</label>
      <select id="branchSelect"></select>
    </div>
    <div class="field">
      <label for="domainSelect">Defence domain</label>
      <select id="domainSelect"></select>
    </div>
  </section>

  <section class="kpis">
    <div class="kpi"><div class="kpi-label">Selected TAM</div><div class="kpi-value" id="kpiTam">-</div><div class="kpi-sub">{value_basis}</div></div>
    <div class="kpi"><div class="kpi-label">Contracts</div><div class="kpi-value" id="kpiContracts">-</div><div class="kpi-sub">Distinct contracts</div></div>
    <div class="kpi"><div class="kpi-label">Average value</div><div class="kpi-value" id="kpiAvgValue">-</div><div class="kpi-sub" id="kpiMedianValue">-</div></div>
    <div class="kpi"><div class="kpi-label">Average length</div><div class="kpi-value" id="kpiAvgLength">-</div><div class="kpi-sub" id="kpiMedianLength">-</div></div>
    <div class="kpi"><div class="kpi-label">Suppliers</div><div class="kpi-value" id="kpiSuppliers">-</div><div class="kpi-sub">Distinct supplier groups</div></div>
    <div class="kpi"><div class="kpi-label">Largest contract</div><div class="kpi-value" id="kpiLargest">-</div><div class="kpi-sub" id="kpiLargestSub">-</div></div>
  </section>

  <section class="toolbar">
    <div class="field">
      <label for="searchInput">Search description, supplier, contract ID or category</label>
      <input id="searchInput" type="text" placeholder="Search contracts...">
    </div>
    <div class="field">
      <label for="capabilityFilter">Service Offering (ranked by spend)</label>
      <select id="capabilityFilter"><option value="All Service Offerings">All Service Offerings</option></select>
    </div>
    <div class="field">
      <label for="rpFilter">Reinvention Partner (ranked by spend)</label>
      <select id="rpFilter"><option value="All Reinvention Partners">All Reinvention Partners</option></select>
    </div>
    <div class="field">
      <label for="reFilter">Reinvention Engine (ranked by spend)</label>
      <select id="reFilter"><option value="All Reinvention Engines">All Reinvention Engines</option></select>
    </div>
    <div class="field">
      <label for="supplierFilter">Supplier (ranked by total winnings)</label>
      <select id="supplierFilter"><option value="All suppliers">All suppliers</option></select>
    </div>
    <div class="field">
      <label for="minValueInput">Minimum value</label>
      <input id="minValueInput" type="number" min="0" step="100000" placeholder="e.g. 1000000">
    </div>
    <label class="check-wrap"><input id="datedOnly" type="checkbox"> Dated contracts only</label>
    <button id="clearFiltersBtn" class="clear-btn" type="button">Clear all filters</button>
  </section>

  <div class="summary-line" id="summaryLine"></div>
  <div id="yearContainer"></div>

  <p class="method">
    The Procuring Entity column is displayed as <b>Division &gt; Branch</b>; missing organisation levels are shown as <b>(blank)</b>. Contracts active as at the page-open date appear only in the top <b>Ongoing contracts</b> bracket. They are excluded from the financial-year brackets so no contract is listed twice. All remaining contracts are grouped by their recorded financial year. The default agency scope includes Department of Defence, Australian Signals Directorate and Australian Submarine Agency. Division and branch aliases are consolidated before display.
    All filters cascade across the dashboard. Agency, division and branch filters now work bidirectionally: selecting a division reduces the branch list, while selecting a branch reduces the division list to only divisions that contain that branch. Service Offerings and suppliers are re-ranked by total spend in the current scope whenever agency, division, branch, search, minimum value, dated-contract, capability or supplier filters change. The abbreviation <b>DDG</b> is treated as <b>Defence Digital Group</b>; branch values such as <b>DDG - ICTDD</b> are used as supporting validation, so historical DDG records are no longer labelled unresolved.
  </p>
</div>

<script>
const DATA = {json.dumps(payload, ensure_ascii=False)};
const ALL_AGENCIES = "All Defence agencies";
const ALL_DOMAINS = "All domains";
const DEFAULT_AGENCY = {json.dumps(default_agency)};
const VALUE_BASIS = {json.dumps(value_basis)};
const AS_AT_DATE = new Date().toISOString().slice(0,10);
const YEAR_ROWS = new Map();

function esc(value) {{
  return String(value ?? '')
    .replaceAll('&','&amp;')
    .replaceAll('<','&lt;')
    .replaceAll('>','&gt;')
    .replaceAll('"','&quot;')
    .replaceAll("'",'&#039;');
}}
function money(v) {{
  v = Number(v || 0);
  const sign = v < 0 ? '-' : '';
  const abs = Math.abs(v);
  if (abs >= 1e9) return sign + '$' + (abs/1e9).toFixed(2) + 'B';
  if (abs >= 1e8) return sign + '$' + (abs/1e6).toFixed(0) + 'M';
  if (abs >= 1e7) return sign + '$' + (abs/1e6).toFixed(1) + 'M';
  if (abs >= 1e6) return sign + '$' + (abs/1e6).toFixed(2) + 'M';
  if (abs >= 1e3) return sign + '$' + (abs/1e3).toFixed(0) + 'K';
  return sign + '$' + abs.toFixed(0);
}}
function number(v) {{ return Number(v || 0).toLocaleString('en-AU'); }}
function duration(days) {{
  if (days === null || days === undefined || !isFinite(Number(days))) return 'n/a';
  const d = Number(days);
  const y = d / 365.25;
  if (y >= 2) return y.toFixed(1) + ' yrs';
  if (y >= 1) return y.toFixed(2) + ' yrs';
  return Math.round(d) + ' days';
}}
function median(values) {{
  const a = values.filter(v => Number.isFinite(v)).sort((x,y) => x-y);
  if (!a.length) return null;
  const m = Math.floor(a.length/2);
  return a.length % 2 ? a[m] : (a[m-1]+a[m])/2;
}}
function uniqueSorted(values) {{
  return [...new Set(values)].sort((a,b) => String(a).localeCompare(String(b)));
}}

function setOptions(select, values, allLabel, labelMap = null, preserveValue = true) {{
  const previous = preserveValue ? select.value : allLabel;
  const options = [allLabel, ...values];
  select.innerHTML = options
    .map(v => {{
      const label = labelMap && labelMap.has(v) ? labelMap.get(v) : v;
      return '<option value="' + esc(v) + '">' + esc(label) + '</option>';
    }})
    .join('');
  select.value = options.includes(previous) ? previous : allLabel;
}}

function divisionsForAgency(agency, branch = 'All branches') {{
  return uniqueSorted(
    DATA.hierarchy
      .filter(r =>
        (agency === ALL_AGENCIES || r.agency === agency) &&
        (branch === 'All branches' || r.branch === branch)
      )
      .map(r => r.division)
  );
}}

function branchesFor(agency, division = 'All divisions') {{
  return uniqueSorted(
    DATA.hierarchy
      .filter(r =>
        (agency === ALL_AGENCIES || r.agency === agency) &&
        (division === 'All divisions' || r.division === division)
      )
      .map(r => r.branch)
  );
}}

function organisationRows() {{
  const agency = document.getElementById('agencySelect').value;
  const division = document.getElementById('divisionSelect').value;
  const branch = document.getElementById('branchSelect').value;
  const domain = document.getElementById('domainSelect').value;

  return DATA.records.filter(r =>
    (agency === ALL_AGENCIES || r.agency === agency) &&
    (division === 'All divisions' || r.division === division) &&
    (branch === 'All branches' || r.branch === branch) &&
    (domain === ALL_DOMAINS || r.domain === domain)
  );
}}

function rowsForFilterOptions(excludeFilter) {{
  const search = document.getElementById('searchInput').value.trim().toLowerCase();
  const capability = document.getElementById('capabilityFilter').value;
  const supplier = document.getElementById('supplierFilter').value;
  const rp = document.getElementById('rpFilter').value;
  const re = document.getElementById('reFilter').value;
  const minValue = Number(document.getElementById('minValueInput').value || 0);
  const datedOnly = document.getElementById('datedOnly').checked;

  return organisationRows().filter(r => {{
    const hay = [
      r.description,
      r.supplier,
      r.contract_id,
      r.category,
      r.procurement_method,
      r.capability,
      r.reinvention_partner,
      r.reinvention_engine,
      r.procuring_entity
    ].join(' ').toLowerCase();

    return (!search || hay.includes(search)) &&
      (excludeFilter === 'capability' || capability === 'All Service Offerings' || r.capability === capability) &&
      (excludeFilter === 'supplier' || supplier === 'All suppliers' || r.supplier === supplier) &&
      (excludeFilter === 'rp' || rp === 'All Reinvention Partners' || r.reinvention_partner === rp) &&
      (excludeFilter === 're' || re === 'All Reinvention Engines' || r.reinvention_engine === re) &&
      Number(r.value || 0) >= minValue &&
      (!datedOnly || (r.start_date && r.end_date));
  }});
}}

function refreshCapabilityFilter() {{
  const select = document.getElementById('capabilityFilter');
  const rows = rowsForFilterOptions('capability');
  const totals = new Map();

  rows.forEach(r => {{
    totals.set(r.capability, (totals.get(r.capability) || 0) + Number(r.value || 0));
  }});

  const capabilities = [...totals.keys()].sort((a,b) =>
    (totals.get(b) - totals.get(a)) || String(a).localeCompare(String(b))
  );

  const labels = new Map(capabilities.map(c => [c, c + ' — ' + money(totals.get(c))]));
  setOptions(select, capabilities, 'All Service Offerings', labels, true);
}}

function refreshRankedFilter(selectId, field, allLabel, excludeFilter) {{
  const select = document.getElementById(selectId);
  const rows = rowsForFilterOptions(excludeFilter);
  const totals = new Map();
  rows.forEach(r => totals.set(r[field], (totals.get(r[field]) || 0) + Number(r.value || 0)));
  const values = [...totals.keys()].sort((a,b) => (totals.get(b)-totals.get(a)) || String(a).localeCompare(String(b)));
  const labels = new Map(values.map(v => [v, v + ' — ' + money(totals.get(v))]));
  setOptions(select, values, allLabel, labels, true);
}}

function refreshRpFilter() {{ refreshRankedFilter('rpFilter', 'reinvention_partner', 'All Reinvention Partners', 'rp'); }}
function refreshReFilter() {{ refreshRankedFilter('reFilter', 'reinvention_engine', 'All Reinvention Engines', 're'); }}

function refreshSupplierFilter() {{
  const select = document.getElementById('supplierFilter');
  const rows = rowsForFilterOptions('supplier');
  const totals = new Map();

  rows.forEach(r => {{
    totals.set(r.supplier, (totals.get(r.supplier) || 0) + Number(r.value || 0));
  }});

  const suppliers = [...totals.keys()].sort((a,b) =>
    (totals.get(b) - totals.get(a)) || String(a).localeCompare(String(b))
  );

  const labels = new Map(suppliers.map(s => [s, s + ' — ' + money(totals.get(s))]));
  setOptions(select, suppliers, 'All suppliers', labels, true);
}}

function refreshDependentFilters(changedFilter = null) {{
  if (changedFilter !== 'capability') refreshCapabilityFilter();
  if (changedFilter !== 'supplier') refreshSupplierFilter();
  if (changedFilter !== 'rp') refreshRpFilter();
  if (changedFilter !== 're') refreshReFilter();

  // Re-run the opposite filter after any invalid selection has been reset.
  if (changedFilter === 'capability') refreshSupplierFilter();
  if (changedFilter === 'supplier') refreshCapabilityFilter();
  if (changedFilter === 'rp' || changedFilter === 're') {{ refreshCapabilityFilter(); refreshSupplierFilter(); }}
}}

function populateDivisions(preserveDivision = false) {{
  const agency = document.getElementById('agencySelect').value;
  const branch = document.getElementById('branchSelect').value || 'All branches';
  const divisionSelect = document.getElementById('divisionSelect');
  const previousDivision = preserveDivision ? divisionSelect.value : 'All divisions';

  setOptions(
    divisionSelect,
    divisionsForAgency(agency, branch),
    'All divisions',
    null,
    false
  );

  if (
    preserveDivision &&
    [...divisionSelect.options].some(o => o.value === previousDivision)
  ) {{
    divisionSelect.value = previousDivision;
  }}
}}

function populateBranches(preserveBranch = false) {{
  const agency = document.getElementById('agencySelect').value;
  const division = document.getElementById('divisionSelect').value || 'All divisions';
  const branchSelect = document.getElementById('branchSelect');
  const previousBranch = preserveBranch ? branchSelect.value : 'All branches';

  setOptions(
    branchSelect,
    branchesFor(agency, division),
    'All branches',
    null,
    false
  );

  if (
    preserveBranch &&
    [...branchSelect.options].some(o => o.value === previousBranch)
  ) {{
    branchSelect.value = previousBranch;
  }}
}}

function populateDomains(preserveDomain = true) {{
  const select = document.getElementById('domainSelect');
  const previous = preserveDomain ? select.value : ALL_DOMAINS;
  const agency = document.getElementById('agencySelect').value;
  const division = document.getElementById('divisionSelect').value || 'All divisions';
  const branch = document.getElementById('branchSelect').value || 'All branches';
  const values = uniqueSorted(DATA.records.filter(r =>
    (agency === ALL_AGENCIES || r.agency === agency) &&
    (division === 'All divisions' || r.division === division) &&
    (branch === 'All branches' || r.branch === branch)
  ).map(r => r.domain));
  setOptions(select, values, ALL_DOMAINS, null, false);
  if (preserveDomain && [...select.options].some(o => o.value === previous)) select.value = previous;
}}

function refreshOrganisationFilters(changedFilter) {{
  if (changedFilter === 'agency') {{
    document.getElementById('branchSelect').innerHTML =
      '<option value="All branches">All branches</option>';
    populateDivisions(false);
    populateBranches(false);
    populateDomains(false);
  }} else if (changedFilter === 'division') {{
    populateBranches(false);
    populateDomains(true);
  }} else if (changedFilter === 'branch') {{
    populateDivisions(true);
    populateDomains(true);
  }} else if (changedFilter === 'domain') {{
    // Domain directly constrains the selected contract population.
  }}

  refreshDependentFilters();
  render();
}}

function baseRows() {{
  return organisationRows();
}}

function filteredRows() {{
  const search = document.getElementById('searchInput').value.trim().toLowerCase();
  const capability = document.getElementById('capabilityFilter').value;
  const supplier = document.getElementById('supplierFilter').value;
  const rp = document.getElementById('rpFilter').value;
  const re = document.getElementById('reFilter').value;
  const minValue = Number(document.getElementById('minValueInput').value || 0);
  const datedOnly = document.getElementById('datedOnly').checked;

  return baseRows().filter(r => {{
    const hay = [r.description,r.supplier,r.contract_id,r.category,r.procurement_method]
      .join(' ')
      .toLowerCase();
    return (!search || hay.includes(search)) &&
      (capability === 'All Service Offerings' || r.capability === capability) &&
      (supplier === 'All suppliers' || r.supplier === supplier) &&
      (rp === 'All Reinvention Partners' || r.reinvention_partner === rp) &&
      (re === 'All Reinvention Engines' || r.reinvention_engine === re) &&
      Number(r.value || 0) >= minValue &&
      (!datedOnly || (r.start_date && r.end_date));
  }});
}}
function entityName(r) {{
  return r.division || r.agency || 'Unknown procurer';
}}
function renderLeaderRows(containerId, rows, valueFormatter, detailFormatter, valueClass = '') {{
  const el = document.getElementById(containerId);
  if (!el) return;
  if (!rows.length) {{
    el.innerHTML = '<div class="leader-empty">No comparable records in the selected scope.</div>';
    return;
  }}
  el.innerHTML = rows.map((r, i) =>
    '<div class="leader-row">' +
      '<div class="leader-rank">' + (i + 1) + '</div>' +
      '<div class="leader-name">' + esc(r.name) +
        '<div class="leader-detail">' + esc(detailFormatter(r)) + '</div>' +
      '</div>' +
      '<div class="leader-value ' + valueClass + '">' + esc(valueFormatter(r)) + '</div>' +
    '</div>'
  ).join('');
}}
function financialYearBounds(startYear) {{
  // Australian financial year: 1 July to 30 June.
  const start = new Date(Date.UTC(Number(startYear), 6, 1));
  const end = new Date(Date.UTC(Number(startYear) + 1, 6, 1));
  return {{start, end}};
}}

function contractSpendInFinancialYear(record, fyStartYear) {{
  // Spread the total contract value evenly across its dated contract period,
  // then allocate the appropriate daily share into each financial year.
  const totalValue = Number(record.total_value || record.value || 0);
  if (!(totalValue > 0) || !record.start_date_iso || !record.end_date_iso) return 0;

  const contractStart = new Date(record.start_date_iso + 'T00:00:00Z');
  // Add one day so the recorded end date is included in the contract period.
  const contractEndExclusive = new Date(record.end_date_iso + 'T00:00:00Z');
  contractEndExclusive.setUTCDate(contractEndExclusive.getUTCDate() + 1);
  if (!Number.isFinite(contractStart.getTime()) || !Number.isFinite(contractEndExclusive.getTime())) return 0;

  const totalDays = (contractEndExclusive - contractStart) / 86400000;
  if (!(totalDays > 0)) return 0;

  const fy = financialYearBounds(fyStartYear);
  const overlapStart = contractStart > fy.start ? contractStart : fy.start;
  const overlapEnd = contractEndExclusive < fy.end ? contractEndExclusive : fy.end;
  const overlapDays = Math.max(0, (overlapEnd - overlapStart) / 86400000);
  return totalValue * (overlapDays / totalDays);
}}

function updateLeaderboards() {{
  // Executive insight cards use the full canonical TAM dataset and ignore page
  // filters. Contract ceilings are spread evenly over their dated duration so
  // the cards represent estimated annual expenditure rather than award value.
  const allRows = DATA.records || [];
  const allFys = [...new Set(
    allRows.map(r => Number(r.fy_start)).filter(Number.isFinite)
  )].sort((a,b) => a-b);
  const selectedFys = allFys.slice(-5);

  // A division is considered currently active when it has a new award in either
  // of the latest two available FYs. Historical DDG CTO records are already
  // rolled into DDG - Defence Digital Group while retaining their identifier.
  const activeFyWindow = selectedFys.slice(-2);
  const activeDivisions = new Set(
    allRows
      .filter(r => activeFyWindow.includes(Number(r.fy_start)))
      .map(r => entityName(r))
      .filter(Boolean)
  );
  const rows = allRows.filter(r => activeDivisions.has(entityName(r)));

  const spendSubtitle = document.getElementById('spendSubtitle');
  const growthSubtitle = document.getElementById('growthSubtitle');
  const declineSubtitle = document.getElementById('declineSubtitle');

  if (!selectedFys.length) {{
    if (spendSubtitle) spendSubtitle.textContent = 'No financial-year data is available.';
    if (growthSubtitle) growthSubtitle.textContent = 'No financial-year data is available.';
    if (declineSubtitle) declineSubtitle.textContent = 'No financial-year data is available.';
    renderLeaderRows('topSpenders', [], r => '', r => '');
    renderLeaderRows('topGrowth', [], r => '', r => '');
    renderLeaderRows('topDecline', [], r => '', r => '');
    return;
  }}

  const earliest = selectedFys[0];
  const latest = selectedFys[selectedFys.length - 1];
  const elapsedYears = latest - earliest;
  const periodLabel = 'FY' + earliest + ' to FY' + latest;

  if (spendSubtitle) {{
    spendSubtitle.textContent = periodLabel + '; active divisions only. Total contract values are spread evenly across each contract period and allocated into the financial years in which the contract was active. These cards are not affected by page filters.';
  }}

  const annualSpendByEntity = new Map();
  rows.forEach(r => {{
    const name = entityName(r);
    if (!annualSpendByEntity.has(name)) annualSpendByEntity.set(name, new Map());
    const entityYears = annualSpendByEntity.get(name);
    selectedFys.forEach(fy => {{
      const allocated = contractSpendInFinancialYear(r, fy);
      entityYears.set(fy, (entityYears.get(fy) || 0) + allocated);
    }});
  }});

  const spendRows = [...annualSpendByEntity.entries()].map(([name, years]) => {{
    const value = selectedFys.reduce((sum, fy) => sum + Number(years.get(fy) || 0), 0);
    return {{name, value, years}};
  }});
  const topSpend = spendRows
    .slice()
    .sort((a,b) => b.value - a.value)
    .slice(0,5);
  const totalTam = spendRows.reduce((sum, r) => sum + r.value, 0);
  renderLeaderRows(
    'topSpenders',
    topSpend,
    r => money(r.value),
    r => totalTam > 0 ? ((r.value / totalTam) * 100).toFixed(1) + '% of five-year spend' : '0.0% of five-year spend'
  );

  if (selectedFys.length < 2 || elapsedYears <= 0) {{
    if (growthSubtitle) growthSubtitle.textContent = 'At least two financial years are required for comparison.';
    if (declineSubtitle) declineSubtitle.textContent = 'At least two financial years are required for comparison.';
    renderLeaderRows('topGrowth', [], r => '', r => '');
    renderLeaderRows('topDecline', [], r => '', r => '');
    return;
  }}

  const changeText = periodLabel + ' (' + elapsedYears + ' annual intervals); active divisions only. Growth and decline compare estimated annualised contract spend in the first and last FY.';
  if (growthSubtitle) growthSubtitle.textContent = changeText;
  if (declineSubtitle) declineSubtitle.textContent = changeText;

  const annualChange = spendRows.map(r => {{
    const earliestSpend = Number(r.years.get(earliest) || 0);
    const latestSpend = Number(r.years.get(latest) || 0);
    return {{
      name: r.name,
      earliest: earliestSpend,
      latest: latestSpend,
      totalChange: latestSpend - earliestSpend,
      annualChange: (latestSpend - earliestSpend) / elapsedYears
    }};
  }});

  const growth = annualChange
    .filter(r => r.annualChange > 0)
    .sort((a,b) => b.annualChange - a.annualChange)
    .slice(0,5);
  const decline = annualChange
    .filter(r => r.annualChange < 0)
    .sort((a,b) => a.annualChange - b.annualChange)
    .slice(0,5);

  renderLeaderRows(
    'topGrowth',
    growth,
    r => '+' + money(r.annualChange) + '/yr',
    r => money(r.earliest) + ' annualised in FY' + earliest + ' to ' + money(r.latest) + ' in FY' + latest + ' (' + money(r.totalChange) + ' change)',
    'leader-positive'
  );

  renderLeaderRows(
    'topDecline',
    decline,
    r => money(r.annualChange) + '/yr',
    r => money(r.earliest) + ' annualised in FY' + earliest + ' to ' + money(r.latest) + ' in FY' + latest + ' (' + money(r.totalChange) + ' change)',
    'leader-negative'
  );
}}

function updateKpis(rows) {{
  const values = rows.map(r => Number(r.value || 0));
  const durations = rows.map(r => Number(r.duration_days)).filter(Number.isFinite);
  const total = values.reduce((a,b) => a+b,0);
  const largest = rows.slice().sort((a,b) => Number(b.value||0)-Number(a.value||0))[0];

  document.getElementById('kpiTam').textContent = money(total);
  document.getElementById('kpiContracts').textContent = number(rows.length);
  document.getElementById('kpiAvgValue').textContent = rows.length ? money(total/rows.length) : 'n/a';
  document.getElementById('kpiMedianValue').textContent = rows.length ? 'Median: ' + money(median(values)) : 'Median: n/a';
  document.getElementById('kpiAvgLength').textContent = durations.length ? duration(durations.reduce((a,b)=>a+b,0)/durations.length) : 'n/a';
  document.getElementById('kpiMedianLength').textContent = durations.length ? 'Median: ' + duration(median(durations)) : 'Median: n/a';
  document.getElementById('kpiSuppliers').textContent = number(new Set(rows.map(r => r.supplier)).size);
  document.getElementById('kpiLargest').textContent = largest ? money(largest.value) : 'n/a';
  document.getElementById('kpiLargestSub').textContent = largest ? largest.supplier : 'No matching contract';
}}
function rowHtml(r) {{
  const largeBadge = Number(r.value || 0) >= 20000000 ? '<span class="badge badge-large">Large</span>' : '';
  const longBadge = Number(r.duration_days || 0) >= 1826 ? '<span class="badge badge-long">5+ years</span>' : '';
  const supplierClass = String(r.supplier || '').toLowerCase().includes('accenture') ? 'supplier-accent' : '';
  return `<tr>
    <td class="nowrap">${{esc(r.contract_id)}}</td>
    <td class="desc">${{esc(r.description)}}</td>
    <td>${{esc(r.procuring_entity)}}</td>
    <td class="${{supplierClass}}">${{esc(r.supplier)}}</td>
    <td>${{esc(r.capability)}}</td>
    <td>${{esc(r.reinvention_partner)}}</td>
    <td>${{esc(r.reinvention_engine)}}</td>
    <td class="money">${{money(r.value)}} ${{largeBadge}}</td>
    <td class="nowrap">${{esc(r.start_date)}}</td>
    <td class="nowrap">${{esc(r.end_date)}}</td>
    <td class="nowrap">${{esc(r.duration_label)}} ${{longBadge}}</td>
    <td>${{esc(r.category)}}</td>
    <td class="nowrap">${{esc(r.extensions)}}</td>
  </tr>`;
}}

function tableHtml(rows) {{
  return `<div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>Contract ID</th>
          <th>Description</th>
          <th>Procuring Entity</th>
          <th>Supplier</th>
          <th>Service Offering</th>
          <th>Reinvention Partner</th>
          <th>Reinvention Engine</th>
          <th class="money">Contract value</th>
          <th>Start date</th>
          <th>End date</th>
          <th>Contract length</th>
          <th>Category</th>
          <th>Extensions</th>
        </tr>
      </thead>
      <tbody>${{rows.map(rowHtml).join('')}}</tbody>
    </table>
  </div>`;
}}

function contractBracket(id, title, rows, metaLabel = 'contracts') {{
  const spend = rows.reduce((s,r) => s + Number(r.value || 0), 0);
  return `<section class="year-section" id="${{id}}">
    <button class="year-toggle" type="button" onclick="toggleYear('${{id}}')">
      <span><span class="chev">▶</span>${{esc(title)}}</span>
      <span class="year-meta">${{number(rows.length)}} ${{metaLabel}} · ${{money(spend)}}</span>
    </button>
    <div class="year-body" data-loaded="false"></div>
  </section>`;
}}

function toggleYear(id) {{
  const section = document.getElementById(id);
  if (!section) return;

  const body = section.querySelector('.year-body');
  const isOpening = !section.classList.contains('open');

  if (isOpening && body.dataset.loaded !== 'true') {{
    const rows = YEAR_ROWS.get(id) || [];
    body.innerHTML = tableHtml(rows);
    body.dataset.loaded = 'true';
  }}

  section.classList.toggle('open');
}}

function clearAllFilters() {{
  document.getElementById('searchInput').value = '';
  document.getElementById('minValueInput').value = '';
  document.getElementById('datedOnly').checked = false;

  document.getElementById('agencySelect').value = DEFAULT_AGENCY;

  const divisionSelect = document.getElementById('divisionSelect');
  setOptions(
    divisionSelect,
    divisionsForAgency(DEFAULT_AGENCY),
    'All divisions',
    null,
    false
  );

  const branchSelect = document.getElementById('branchSelect');
  setOptions(
    branchSelect,
    branchesFor(DEFAULT_AGENCY, 'All divisions'),
    'All branches',
    null,
    false
  );

  const domainSelect = document.getElementById('domainSelect');
  setOptions(domainSelect, uniqueSorted(DATA.records.map(r => r.domain)), ALL_DOMAINS, null, false);

  document.getElementById('capabilityFilter').innerHTML =
    '<option value="All Service Offerings">All Service Offerings</option>';
  document.getElementById('supplierFilter').innerHTML =
    '<option value="All suppliers">All suppliers</option>';
  document.getElementById('rpFilter').innerHTML = '<option value="All Reinvention Partners">All Reinvention Partners</option>';
  document.getElementById('reFilter').innerHTML = '<option value="All Reinvention Engines">All Reinvention Engines</option>';

  refreshDependentFilters();
  render();
}}

function render() {{
  const rows = filteredRows();
  updateKpis(rows);

  document.getElementById('summaryLine').textContent =
    number(rows.length) + ' matching contracts · ' +
    money(rows.reduce((s,r) => s + Number(r.value || 0), 0)) +
    ' · ' + VALUE_BASIS;

  const ongoingRows = [];
  const historicalRows = [];

  rows.forEach(r => {{
    const isOngoing = Boolean(
      r.start_date_iso &&
      r.end_date_iso &&
      r.start_date_iso <= AS_AT_DATE &&
      r.end_date_iso >= AS_AT_DATE
    );

    if (isOngoing) {{
      ongoingRows.push(r);
    }} else {{
      historicalRows.push(r);
    }}
  }});

  const groups = new Map();
  historicalRows.forEach(r => {{
    const fy = r.financial_year || 'Unknown FY';
    if (!groups.has(fy)) groups.set(fy, []);
    groups.get(fy).push(r);
  }});

  const years = [...groups.keys()].sort((a,b) => {{
    const ra = groups.get(a)[0];
    const rb = groups.get(b)[0];
    const fa = ra.fy_start === null ? -Infinity : Number(ra.fy_start);
    const fb = rb.fy_start === null ? -Infinity : Number(rb.fy_start);
    return fb - fa;
  }});

  const container = document.getElementById('yearContainer');
  if (!rows.length) {{
    container.innerHTML = '<div class="empty">No contracts match the selected filters.</div>';
    return;
  }}

  YEAR_ROWS.clear();
  const sections = [];

  if (ongoingRows.length) {{
    const ongoingId = 'ongoing_contracts';
    const sortedOngoing = ongoingRows
      .slice()
      .sort((a,b) => Number(b.value||0)-Number(a.value||0));
    YEAR_ROWS.set(ongoingId, sortedOngoing);
    sections.push(contractBracket(
      ongoingId,
      'Ongoing contracts',
      sortedOngoing,
      'active contracts'
    ));
  }}

  years.forEach(year => {{
    const yearRows = groups.get(year)
      .slice()
      .sort((a,b) => Number(b.value||0)-Number(a.value||0));
    const id = 'year_' + String(year).replace(/[^A-Za-z0-9]/g,'_');
    YEAR_ROWS.set(id, yearRows);
    sections.push(contractBracket(id, year, yearRows));
  }});

  container.innerHTML = sections.join('');
}}

document.getElementById('agencySelect').value = DEFAULT_AGENCY;
document.getElementById('agencySelect').addEventListener(
  'change',
  () => refreshOrganisationFilters('agency')
);
document.getElementById('divisionSelect').addEventListener(
  'change',
  () => refreshOrganisationFilters('division')
);
document.getElementById('branchSelect').addEventListener(
  'change',
  () => refreshOrganisationFilters('branch')
);
document.getElementById('domainSelect').addEventListener(
  'change',
  () => refreshOrganisationFilters('domain')
);
document.getElementById('searchInput').addEventListener('input', () => {{
  refreshDependentFilters();
  render();
}});
document.getElementById('capabilityFilter').addEventListener('change', () => {{
  refreshDependentFilters('capability');
  render();
}});
document.getElementById('supplierFilter').addEventListener('change', () => {{
  refreshDependentFilters('supplier');
  render();
}});
document.getElementById('rpFilter').addEventListener('change', () => {{ refreshDependentFilters('rp'); render(); }});
document.getElementById('reFilter').addEventListener('change', () => {{ refreshDependentFilters('re'); render(); }});
document.getElementById('minValueInput').addEventListener('input', () => {{
  refreshDependentFilters();
  render();
}});
document.getElementById('datedOnly').addEventListener('change', () => {{
  refreshDependentFilters();
  render();
}});
document.getElementById('clearFiltersBtn').addEventListener('click', clearAllFilters);

updateLeaderboards();
refreshOrganisationFilters('agency');
</script>

<div id="execTooltip" class="exec-tooltip" role="tooltip" aria-hidden="true"></div>
<script>
(function() {{
  const tooltip = document.getElementById('execTooltip');
  const targets = document.querySelectorAll('.executive-summary strong[data-tooltip]');
  if (!tooltip || !targets.length) return;
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
    if (top < edge) top = rect.bottom + gap;
    top = Math.max(edge, Math.min(top, window.innerHeight - tipRect.height - edge));
    tooltip.style.left = Math.round(left) + 'px';
    tooltip.style.top = Math.round(top) + 'px';
  }}

  function showTooltip(target) {{
    const message = target.getAttribute('data-tooltip');
    if (!message) return;
    activeTarget = target;
    tooltip.textContent = message;
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

  targets.forEach(target => {{
    target.addEventListener('mouseenter', () => showTooltip(target));
    target.addEventListener('mouseleave', () => hideTooltip(target));
    target.addEventListener('focus', () => showTooltip(target));
    target.addEventListener('blur', () => hideTooltip(target));
  }});

  window.addEventListener('scroll', () => {{
    if (activeTarget) placeTooltip(activeTarget);
  }}, {{passive:true}});
  window.addEventListener('resize', () => {{
    if (activeTarget) placeTooltip(activeTarget);
  }});
}})();
</script>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")


def html_escape(value: object) -> str:
    text = str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#039;")
    )


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df, value_col = load_data(input_path, args.value_mode)
    payload = build_payload(df)

    html_path = output_dir / "ProcurementExplorationDashboard.html"
    build_html(payload, html_path, args.value_mode)

    division_audit = (
        df.groupby([AGENCY_COL, DIVISION_COL, "_division"], dropna=False)
        .agg(
            contracts=(CN_ID_COL, "nunique") if CN_ID_COL in df.columns else ("_value", "size"),
            value=("_value", "sum"),
        )
        .reset_index()
        .sort_values("value", ascending=False)
    )
    branch_audit = (
        df.groupby([AGENCY_COL, "_division", BRANCH_COL, "_branch"], dropna=False)
        .agg(
            contracts=(CN_ID_COL, "nunique") if CN_ID_COL in df.columns else ("_value", "size"),
            value=("_value", "sum"),
        )
        .reset_index()
        .sort_values("value", ascending=False)
    )

    division_audit.to_csv(output_dir / "division_consolidation_audit.csv", index=False)
    branch_audit.to_csv(output_dir / "branch_consolidation_audit.csv", index=False)

    print("Procurement Exploration dashboard complete")
    print(f"Input: {input_path}")
    print(f"Value mode: {args.value_mode} / column used: {value_col}")
    print(f"TAM-only rows: {len(df):,}")
    print(f"Selected TAM across all Defence agencies: ${df['_value'].sum()/1e9:,.2f}B")
    print("TAM definition aligned with Dashboard 1: is_addressable == True")
    print(f"Wrote: {html_path}")
    print(f"Wrote: {output_dir / 'division_consolidation_audit.csv'}")
    print(f"Wrote: {output_dir / 'branch_consolidation_audit.csv'}")


if __name__ == "__main__":
    main()