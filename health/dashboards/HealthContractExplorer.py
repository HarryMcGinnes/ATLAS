from __future__ import annotations

import argparse
import json
import re
import os
from datetime import datetime
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from urllib.parse import urlparse
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo
from pathlib import Path

import pandas as pd

AGENCY_COL = "Agency"
DIVISION_CANDIDATES = ["Agency Division", "Agency Divison", "Division"]
FY_COL = "Financial Year"
VALUE_COL = "Value"
ANNUALISED_VALUE_COL = "Value Per Year"
CAPABILITY_COL = "capability"
SUPPLIER_GROUP_COL = "supplier_group"
RAW_SUPPLIER_COL = "Supplier Name"
CN_ID_COL = "CN ID"
DESCRIPTION_COL = "Description"
PROCUREMENT_METHOD_COL = "Procurement Method"
CATEGORY_COL = "Category"
EXTENSIONS_CANDIDATES = ["#Extensions", "Extensions", "Extension Count", "Number of Extensions"]
START_DATE_CANDIDATES = ["Start Date", "Contract Start Date", "Contract Period Start", "Contract Start"]
END_DATE_CANDIDATES = ["End Date", "Contract End Date", "Contract Period End", "Contract End", "Expiry Date"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the Health TAM Exploratory Analysis dashboard.")
    p.add_argument(
        "--input",
        default="health/data/processed/master_health_contracts.parquet",
        help="Canonical Health master dataset (parquet or CSV).",
    )
    p.add_argument("--output-dir", default="docs/health")
    p.add_argument(
        "--value-mode",
        choices=["total", "annualised"],
        default="total",
        help="Value used in KPIs and tables.",
    )
    p.add_argument("--atm-rss-file", default="", help="Optional saved AusTender RSS XML for offline builds and tests.")
    p.add_argument("--atm-timeout", type=int, default=12, help="Timeout for official AusTender ATM RSS fetch.")
    return p.parse_args()


def clean_text(series: pd.Series, fallback: str) -> pd.Series:
    return (
        series.fillna(fallback)
        .astype(str)
        .str.replace(r"_x000D_", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .replace("", fallback)
    )


def first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {str(c).lower(): c for c in df.columns}
    for candidate in candidates:
        found = lower_map.get(candidate.lower())
        if found is not None:
            return found
    return None


def financial_year_start(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    m = re.search(r"(19\d{2}|20\d{2})", str(value))
    return int(m.group(1)) if m else None


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
    text = str(value).strip()
    return text or fallback


def parse_bool_series(series: pd.Series) -> pd.Series:
    """Parse canonical boolean fields safely from parquet/CSV representations."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )


def parse_atlas_date_series(series: pd.Series) -> pd.Series:
    """Parse datetime columns without swapping the month/day in ISO dates."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")
    text = series.astype("string").str.strip()
    iso = text.str.match(r"^\d{4}-\d{1,2}-\d{1,2}(?:[ T].*)?$", na=False)
    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    result.loc[iso] = pd.to_datetime(text.loc[iso], errors="coerce", format="ISO8601")
    result.loc[~iso] = pd.to_datetime(
        text.loc[~iso], errors="coerce", format="mixed", dayfirst=True
    )
    return result


def load_data(path: Path, value_mode: str) -> tuple[pd.DataFrame, str]:
    if not path.exists():
        raise SystemExit(f"Health procurement dataset not found: {path}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, low_memory=False)
    elif path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        raise SystemExit(f"Unsupported input type: {path.suffix}. Use CSV or parquet.")

    if AGENCY_COL not in df.columns or VALUE_COL not in df.columns:
        raise SystemExit(f"Health input must contain '{AGENCY_COL}' and '{VALUE_COL}'.")

    if "is_addressable" not in df.columns:
        raise SystemExit(
            "Health TAM Exploratory Analysis requires the canonical 'is_addressable' field. "
            "Rebuild master_output/health_portfolio_contracts.parquet with build_health_filter_v2.py first."
        )

    # Preserve the complete Health market.
    # The dashboard market switch controls addressability.
    df = df.copy()
    df["is_addressable"] = parse_bool_series(
        df["is_addressable"]
    )

    if df.empty:
        raise SystemExit("No Health contracts found in the input dataset.")

    if SUPPLIER_GROUP_COL not in df.columns:
        if "supplier_display" in df.columns:
            df[SUPPLIER_GROUP_COL] = df["supplier_display"]
        elif RAW_SUPPLIER_COL in df.columns:
            df[SUPPLIER_GROUP_COL] = df[RAW_SUPPLIER_COL]
        else:
            df[SUPPLIER_GROUP_COL] = "Unknown supplier"

    if "service_offering" in df.columns:
        df[CAPABILITY_COL] = df["service_offering"]
    elif "Service Offering" in df.columns:
        df[CAPABILITY_COL] = df["Service Offering"]
    elif CAPABILITY_COL not in df.columns:
        if "Service Offering" in df.columns:
            df[CAPABILITY_COL] = df["Service Offering"]
        else:
            df[CAPABILITY_COL] = "Unclassified"

    value_col = ANNUALISED_VALUE_COL if value_mode == "annualised" else VALUE_COL
    if value_col not in df.columns:
        raise SystemExit(f"Requested value column not found: {value_col}")

    out = df.copy()
    out[AGENCY_COL] = clean_text(out[AGENCY_COL], "Unknown agency")
    division_col = first_existing(out, DIVISION_CANDIDATES)
    if division_col:
        out["_division"] = clean_text(out[division_col], "Unspecified division")
    else:
        out["_division"] = "Unspecified division"
    out[SUPPLIER_GROUP_COL] = clean_text(out[SUPPLIER_GROUP_COL], "Unknown supplier")
    out[CAPABILITY_COL] = clean_text(out[CAPABILITY_COL], "Unclassified")
    out[FY_COL] = clean_text(out[FY_COL], "Unknown FY") if FY_COL in out.columns else "Unknown FY"

    out["_value"] = pd.to_numeric(out[value_col], errors="coerce").fillna(0)
    out["_total_value"] = pd.to_numeric(out[VALUE_COL], errors="coerce").fillna(0)
    out["_fy_start"] = out[FY_COL].map(financial_year_start)
    out["_labour_hire"] = labour_hire_mask(out)


    start_col = first_existing(out, START_DATE_CANDIDATES)
    end_col = first_existing(out, END_DATE_CANDIDATES)

    out["_start"] = (
        parse_atlas_date_series(out[start_col])
        if start_col
        else pd.Series(pd.NaT, index=out.index)
    )

    out["_end"] = (
        parse_atlas_date_series(out[end_col])
        if end_col
        else pd.Series(pd.NaT, index=out.index)
    )

    # Use validated original-to-latest amendment family dates.
    family_cols = {
        "cn_family_start_date",
        "cn_family_end_date",
        "cn_family_date_status",
    }
    
    missing_family_cols = family_cols - set(out.columns)

    if missing_family_cols:
        raise RuntimeError(
            "Health Contract Explorer missing CN-family fields: "
            + ", ".join(sorted(missing_family_cols))
        )
        

    family_start = parse_atlas_date_series(out["cn_family_start_date"])
    family_end = parse_atlas_date_series(out["cn_family_end_date"])

    family_status = (
        out["cn_family_date_status"]
        .astype("string")
        .fillna("review_missing_family")
    )

    # ATLAS HEALTH FAMILY-DATE DISPLAY V2
    # The refreshed Silver/Gold date statuses are canonical. Older master
    # artifacts may still report review_original_missing: safely recompute
    # observed amendment dates using *only valid individual date pairs*.
    credible_statuses = {
        "valid", "inferred_family_start", "inferred_original_missing",
    }
    raw_start = out["_start"].copy()
    raw_end = out["_end"].copy()
    family_root = (
        out["CN Root ID"].astype("string").str.strip().str.upper()
        if "CN Root ID" in out.columns
        else out[CN_ID_COL].astype("string").str.strip().str.upper()
              .str.replace(r"-A\d+$", "", regex=True)
    )
    ids = out[CN_ID_COL].astype("string").str.strip().str.upper()
    raw_valid = raw_start.notna() & raw_end.notna() & raw_start.le(raw_end)
    observed_start = raw_start.where(raw_valid).groupby(family_root).transform("min")
    observed_end = raw_end.where(raw_valid).groupby(family_root).transform("max")
    parent_present = ids.eq(family_root).groupby(family_root).transform("any")
    needs_recovery = family_status.isin({
        "review_original_missing", "review_invalid_version_dates",
        "review_invalid_family_period", "review_missing_start",
        "review_missing_end",
    })
    recovered = (
        needs_recovery & observed_start.notna() & observed_end.notna()
        & observed_start.le(observed_end)
    )
    family_start = family_start.where(~recovered, observed_start)
    family_end = family_end.where(~recovered, observed_end)
    family_status = family_status.mask(recovered, "inferred_family_start")
    family_status = family_status.mask(
        recovered & ~parent_present.fillna(False), "inferred_original_missing"
    )
    credible = (
        family_status.isin(credible_statuses)
        & family_start.notna() & family_end.notna()
        & family_end.ge(family_start)
    )
    out["_start"] = family_start.where(credible)
    out["_end"] = family_end.where(credible)
    out["_date_status"] = family_status

    duration = (out["_end"] - out["_start"]).dt.days
    out["_duration_days"] = duration.where(duration >= 0)


    if out.empty:
        raise SystemExit("No Health contracts were found.")
    return out, value_col



# Lightweight, evidence-based labour-hire overlay. Independent of TAM/SO classification.
# Prefer explicit labour-hire wording; generic ICT and consulting are NOT sufficient.
def labour_hire_mask(df: pd.DataFrame) -> pd.Series:
    import re
    category = df['Category'].fillna('').astype(str) if 'Category' in df else pd.Series('', index=df.index)
    description = df['Description'].fillna('').astype(str) if 'Description' in df else pd.Series('', index=df.index)
    category_signal = category.str.contains(
        r'temporary\s+personnel\s+services?|labou?r\s+hire|staff\s+augmentation|'
        r'contractor\s+personnel|personnel\s+placement', case=False, regex=True, na=False
    )
    description_signal = description.str.contains(
        r'labou?r[\s-]+hire|staff[\s-]+augmentation|workforce[\s-]+augmentation|'
        r'contractor[\s-]+resourc(?:e|es|ing)|contract[\s-]+personnel|'
        r'temporary[\s-]+personnel|(?:provision|supply|engagement)\s+of\s+'
        r'(?:ict\s+|it\s+|specialist\s+)?(?:contractors?|temporary\s+staff|personnel)',
        case=False, regex=True, na=False
    )
    return (category_signal | description_signal).astype(bool)

def build_payload(df: pd.DataFrame) -> dict[str, object]:
    extensions_col = first_existing(df, EXTENSIONS_CANDIDATES)
    records: list[dict[str, object]] = []

    for _, row in df.iterrows():
        records.append({
            "is_addressable": bool(row.get("is_addressable", False)),
            "agency": safe_str(row.get(AGENCY_COL), "Unknown agency"),
            "division": safe_str(row.get("_division"), "Unspecified division"),
            "financial_year": safe_str(row.get(FY_COL), "Unknown FY"),
            "fy_start": int(row["_fy_start"]) if pd.notna(row.get("_fy_start")) else None,
            "contract_id": safe_str(row.get(CN_ID_COL)),
            "description": safe_str(row.get(DESCRIPTION_COL)),
            "supplier": safe_str(row.get(SUPPLIER_GROUP_COL), "Unknown supplier"),
            "capability": safe_str(row.get(CAPABILITY_COL), "Unclassified"),
            "is_addressable": bool(row.get("is_addressable", False)),
            "value": float(row.get("_value", 0) or 0),
            "total_value": float(row.get("_total_value", 0) or 0),
            "start_date": fmt_date(row.get("_start")) + (" (inferred)" if safe_str(row.get("_date_status")).startswith("inferred_") else ""),
            "end_date": fmt_date(row.get("_end")),
            "start_date_iso": pd.Timestamp(row.get("_start")).strftime("%Y-%m-%d") if pd.notna(row.get("_start")) else "",
            "end_date_iso": pd.Timestamp(row.get("_end")).strftime("%Y-%m-%d") if pd.notna(row.get("_end")) else "",
            "duration_days": float(row["_duration_days"]) if pd.notna(row.get("_duration_days")) else None,
            "duration_label": fmt_duration(row.get("_duration_days")),
            "procurement_method": safe_str(row.get(PROCUREMENT_METHOD_COL)),
            "category": safe_str(row.get(CATEGORY_COL)),
            "is_labour_hire": bool(row.get("_labour_hire", False)),
            "extensions": safe_str(row.get(extensions_col)) if extensions_col else "",
        })

    agencies = sorted(df[AGENCY_COL].dropna().astype(str).unique().tolist())
    return {"records": records, "agencies": agencies}


def html_escape(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#039;")
    )


# ATLAS open-tender opportunity signals are heuristics, NOT procurement forecasts.
ATM_RSS_URL = "https://www.tenders.gov.au/public_data/rss/rss.xml"
ATM_SIGNALS = {
    "SI & Engineering": ("system integration", "software development", "digital transformation", "application development", "platform modernisation", "systems integration", "implementation", "software engineering", "enterprise architecture"),
    "Data, AI & Automation": ("data analytics", "data platform", "artificial intelligence", "machine learning", "data governance", "analytics", "automation", "business intelligence", "data warehouse"),
    "Cloud Infrastructure & Cyber": ("cyber security", "cybersecurity", "cloud", "security operations", "identity management", "zero trust", "infrastructure migration", "penetration testing"),
    "Managed Services & Operations": ("managed services", "service desk", "application support", "ict support", "service management", "operational support", "help desk"),
    "Strategy, Transformation & Advisory": ("strategy", "transformation advisory", "consulting", "business case", "program assurance", "operating model", "change management", "strategic review"),
}
ATM_HEALTH_HINTS = (
    "health", "aged care", "medicare", "hospital", "clinical", "medical", "pharmaceutical", "disability", "ndis",
    "human services", "social services", "mental health", "therapeutic goods", "healthcare", "health funding",
    "national health", "health and aged care", "health, disability", "digital health",
)


# Exactly the agencies in ATLAS Health Silver scope. Mention in an RSS title/summary
# is only contextual evidence, NOT verification that an agency issued a tender.
ATLAS_HEALTH_AGENCIES = (
    "Department of Health, Disability and Ageing",
    "Department of Health and Aged Care - Therapeutic Goods Administration",
    "Department of Health and Aged Care",
    "Department of Health",
    "Australian Digital Health Agency",
    "Australian Institute of Health and Welfare",
    "Aged Care Quality and Safety Commission",
    "Australian Aged Care Quality Agency",
    "Independent Health and Aged Care Pricing Authority",
    "National Health Funding Body",
    "National Health and Medical Research Council",
    "National Mental Health Commission",
    "Department of Social Services",
)
# Short-name matches are included only where substantially distinctive; generic
# terms like 'Health' or 'Social Services' cannot establish the buyer.
ATLAS_AGENCY_TERMS = {
    "Australian Digital Health Agency": ("Australian Digital Health Agency",),
    "Australian Institute of Health and Welfare": ("Australian Institute of Health and Welfare",),
    "Aged Care Quality and Safety Commission": ("Aged Care Quality and Safety Commission",),
    "Australian Aged Care Quality Agency": ("Australian Aged Care Quality Agency",),
    "Independent Health and Aged Care Pricing Authority": ("Independent Health and Aged Care Pricing Authority",),
    "National Health Funding Body": ("National Health Funding Body",),
    "National Health and Medical Research Council": ("National Health and Medical Research Council",),
    "National Mental Health Commission": ("National Mental Health Commission",),
    "Department of Social Services": ("Department of Social Services",),
    "Department of Health, Disability and Ageing": ("Department of Health, Disability and Ageing",),
    "Department of Health and Aged Care - Therapeutic Goods Administration":
        ("Department of Health and Aged Care - Therapeutic Goods Administration",),
    "Department of Health and Aged Care": ("Department of Health and Aged Care",),
    "Department of Health": ("Department of Health",),
}
# Goods-only supply is outside ATLAS's service-led TAM. These are review flags,
# not a replacement for the canonical awarded-CN classification rules.
ATM_GOODS_HINTS = (
    "spare parts", "replacement parts", "consumables", "medical supplies",
    "equipment supply", "supply of equipment", "supply and delivery",
    "accessories", "medical devices", "pharmaceutical products", "medicines supply",
    "office furniture", "vehicle purchase", "hardware purchase",
)


def agency_mentions(text: str) -> list[str]:
    mentions = []
    for agency, terms in ATLAS_AGENCY_TERMS.items():
        if any(re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text, flags=re.I)
               for term in terms):
            mentions.append(agency)
    # Avoid counting shorter historical names when the notice names the longer entity.
    return [agency for agency in mentions if not any(
        agency != other and agency.casefold() in other.casefold() for other in mentions
    )]


def atm_addressability(text: str, offering: str) -> tuple[str, str]:
    low = text.casefold()
    goods = [term for term in ATM_GOODS_HINTS if term in low]
    if goods and offering == "Needs review":
        return "Likely non-addressable", "Goods/equipment supply indicator: " + goods[0]
    if goods:
        return "Needs review", "Goods/equipment language overlaps service keywords; manual review required"
    if offering == "Needs review":
        return "Needs review", "No specific ATLAS service offering evidence in available summary"
    return "Potentially addressable", "Service keyword evidence only; review full tender scope"


def _rss_txt(node: ET.Element, name: str) -> str:
    found = node.find(name)
    return "" if found is None else " ".join((found.text or "").split())


def load_atm_feed(rss_file: str = "", timeout: int = 12) -> tuple[list[dict[str, str]], str, str]:
    """Fetch public ATM RSS; a feed entry is NOT proof an ATM is still open."""
    if rss_file:
        raw = Path(rss_file).read_bytes()
        source = f"Local RSS snapshot: {Path(rss_file).name}"
    else:
        try:
            request = Request(ATM_RSS_URL, headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
                "Accept": "application/rss+xml,application/xml,text/xml,*/*",
                "Accept-Language": "en-AU,en;q=0.9",
            })
            with urlopen(request, timeout=max(1, timeout)) as response:
                raw = response.read(5_000_001)
            if len(raw) > 5_000_000:
                raise ValueError("AusTender RSS exceeded the safe 5 MB limit")
            source = "Official AusTender ATM RSS"
        except HTTPError as exc:
            message = (f"AusTender RSS request returned HTTP {exc.code}. "
                       "The feed may reject requests from GitHub Actions. This is a retrieval failure, not zero open tenders; "
                       "check the official live ATM list.")
            return [], "Unavailable", message
        except Exception as exc:
            return [], "Unavailable", (f"Unable to retrieve AusTender ATM RSS ({type(exc).__name__}). "
                                      "Current tender availability is unknown; check the official live ATM list.")
    try:
        root = ET.fromstring(raw)
        nodes = root.findall(".//item")
        if not nodes:
            # Atom-style feed fallback, including namespaced entries.
            nodes = [n for n in root.iter() if n.tag.rsplit('}', 1)[-1] == "entry"]
        results = []
        seen = set()
        for item in nodes:
            title = _rss_txt(item, "title")
            desc = _rss_txt(item, "description") or _rss_txt(item, "summary")
            link = _rss_txt(item, "link")
            if not link:
                e = item.find("{http://www.w3.org/2005/Atom}link")
                link = (e.attrib.get("href") or "") if e is not None else ""
            parsed = urlparse(link)
            if parsed.scheme != "https" or parsed.hostname not in {"www.tenders.gov.au", "tenders.gov.au"}:
                continue
            ident = _rss_txt(item, "guid") or link
            if not title or ident in seen:
                continue
            seen.add(ident)
            published = _rss_txt(item, "pubDate") or _rss_txt(item, "published")
            results.append({"title": title, "description": re.sub(r"<[^>]+>", " ", desc)[:800], "url": link, "ref": ident[:180], "published": published})
        ts = datetime.now(ZoneInfo("Australia/Sydney")).strftime("%d %b %Y %H:%M %Z")
        return results, ts, f"{source} · {len(results)} records retrieved. RSS may omit some open ATMs and retain closed notices; verify all notices and closing dates on AusTender."
    except ET.ParseError:
        return [], "Unavailable", "AusTender ATM RSS could not be parsed; no tenders displayed."


def classify_atm(text: str) -> tuple[str, str]:
    low = text.casefold()
    matches = {offering: sum(1 for keyword in words if keyword in low) for offering, words in ATM_SIGNALS.items()}
    best = sorted(matches.items(), key=lambda p: (-p[1], p[0]))[0]
    if best[1] == 0:
        return "Needs review", "No specific ATLAS offering keywords found"
    return best[0], f"Matched {best[1]} offering keyword(s) in published text"


def build_atm_insights(df: pd.DataFrame, notices: list[dict[str, str]]) -> list[dict[str, object]]:
    # Accenture presence is defined by historic awarded CNs, never inferred from an ATM.
    if "is_accenture" in df.columns:
        acc = parse_bool_series(df["is_accenture"])
    else:
        acc = df[SUPPLIER_GROUP_COL].astype(str).str.contains(r"\baccenture\b", case=False, regex=True, na=False)
    eligible = df["is_addressable"]
    market = df.loc[eligible].groupby(CAPABILITY_COL)["_total_value"].sum().to_dict()
    won = df.loc[eligible & acc].groupby(CAPABILITY_COL)["_total_value"].sum().to_dict()
    output = []
    for notice in notices:
        desc = notice["title"] + " " + notice["description"]
        offering, explanation = classify_atm(desc)
        health_hint = any(term in desc.casefold() for term in ATM_HEALTH_HINTS)
        agencies = agency_mentions(desc)
        addressability, address_reason = atm_addressability(desc, offering)
        # RSS does not establish who the issuing agency is.
        total = float(market.get(offering, 0))
        existing = float(won.get(offering, 0))
        if offering == "Needs review":
            signal = "Manual qualification"
            rationale = "Insufficient description to connect this ATM to a defined ATLAS Service Offering."
        elif existing <= 0:
            signal = "Potential whitespace"
            rationale = "No historical Accenture-awarded value in this Health Service Offering is identifiable in the current ATLAS master."
        elif total > 0 and existing / total >= 0.05:
            signal = "Established Health presence"
            rationale = "Accenture has recorded Health wins in this Service Offering (at least 5% of its historical addressable value)."
        else:
            signal = "Expansion opportunity"
            rationale = "Accenture has some historical Health wins in this Service Offering, but below 5% of its historical addressable value."
        output.append({**notice, "offering": offering, "match_evidence": explanation,
                       "health_relevance": "Health keyword match" if health_hint else "Health relevance unverified",
                       "health_candidate": health_hint, "agency_mentions": agencies,
                       "scope_candidate": bool(agencies) and addressability == "Potentially addressable",
                       "addressability": addressability, "addressability_reason": address_reason,
                       "signal": signal, "rationale": rationale,
                       "health_market_value": round(total, 2), "accenture_wins_value": round(existing, 2)})
    return output


def atm_section(df: pd.DataFrame, rss_file: str, timeout: int) -> tuple[str, str, str]:
    notices, ts, status = load_atm_feed(rss_file, timeout)
    insights = build_atm_insights(df, notices)
    return json.dumps(insights, ensure_ascii=True).replace("</", "<\\/"), ts, status

def build_html(payload: dict[str, object], output_path: Path, value_mode: str, freshness: str = "", atm_json: str = "[]", atm_timestamp: str = "Unavailable", atm_status: str = "") -> None:
    all_agencies = "All Health agencies"
    agency_options = "\n".join(
        [f'<option value="{html_escape(all_agencies)}">{html_escape(all_agencies)}</option>']
        + [f'<option value="{html_escape(a)}">{html_escape(a)}</option>' for a in payload["agencies"]]
    )
    value_basis = "Total contract value" if value_mode == "total" else "Annualised contract value"

    html = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Health TAM Exploratory Analysis</title>
<style>
:root{{--purple:#A100FF;--ink:#101828;--muted:#667085;--line:#E4E7EC;--bg:#F6F8FB;--card:#FFFFFF;--green:#067647;--amber:#B54708;}}
*{{box-sizing:border-box}} body{{margin:0;font-family:Arial,Helvetica,sans-serif;background:var(--bg);color:var(--ink)}}
.wrap{{max-width:1720px;margin:0 auto;padding:28px}} h1{{margin:0 0 8px;font-size:30px}} .subtitle{{margin:0 0 20px;color:#526070;font-size:15px;line-height:1.45}}
.executive-summary{{background:#fff;border:1px solid #e5e7eb;border-left:5px solid var(--purple);border-radius:14px;padding:18px 20px;margin:0 0 18px;box-shadow:0 1px 2px rgba(0,0,0,.04)}}
.executive-summary h2{{margin:0 0 9px;font-size:20px}} .executive-summary p{{margin:0;color:#475467;line-height:1.55}}
.insight-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:0 0 20px}}
.leader-card{{background:#fff;border:1px solid var(--line);border-radius:14px;padding:16px 17px;box-shadow:0 1px 2px rgba(16,24,40,.04)}}
.leader-card h3{{margin:0 0 4px;font-size:17px}} .leader-card p{{margin:0 0 12px;color:var(--muted);font-size:12px;line-height:1.4}}
.leader-row{{display:grid;grid-template-columns:28px minmax(0,1fr) auto;gap:9px;align-items:center;padding:9px 0;border-top:1px solid #F0F2F5}}
.leader-row:first-child{{border-top:0}} .leader-rank{{width:24px;height:24px;border-radius:50%;display:flex;align-items:center;justify-content:center;background:#F4EBFF;color:#7F00CC;font-size:11px;font-weight:800}}
.leader-name{{min-width:0;font-size:13px;font-weight:700}} .leader-detail{{margin-top:2px;color:var(--muted);font-size:11px;font-weight:400}} .leader-value{{text-align:right;white-space:nowrap;font-size:13px;font-weight:800}} .leader-positive{{color:#067647}} .leader-negative{{color:#B42318}}
.field{{background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:10px 12px}} .field label{{display:block;font-size:10px;text-transform:uppercase;letter-spacing:.04em;color:#667085;margin-bottom:3px}}
select,input{{width:100%;height:34px;border:1px solid #d0d5dd;border-radius:8px;background:#fff;padding:0 10px;font-size:13px}}
.kpis{{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:11px;margin:10px 0 18px}} .kpi{{background:#fff;border:1px solid var(--line);border-radius:14px;padding:14px 15px;min-width:0;box-shadow:0 1px 2px rgba(16,24,40,.04)}} .kpi:first-child{{border-top:4px solid var(--purple)}}
.kpi-label{{color:var(--muted);font-size:11px;text-transform:uppercase;font-weight:700;letter-spacing:.045em;margin-bottom:7px}} .kpi-value{{font-size:clamp(19px,1.7vw,27px);font-weight:750;line-height:1.08}} .kpi-sub{{color:var(--muted);font-size:12px;margin-top:5px;line-height:1.35}}
.toolbar{{margin:0 0 18px;background:#fff;border:1px solid var(--line);border-radius:14px;padding:14px;display:grid;grid-template-columns:1.35fr 1.25fr 1.05fr 1.05fr .72fr auto auto;gap:10px;align-items:end}}
.toolbar{{grid-template-columns:1.35fr 1.15fr 1fr 1fr .85fr .75fr auto auto}}
.check-wrap{{min-height:42px;display:flex;align-items:center;gap:8px;padding:0 6px;color:#344054;font-size:13px}} .check-wrap input{{width:auto}}
.clear-btn{{min-height:42px;border:1px solid #D0D5DD;border-radius:9px;background:#fff;color:#344054;padding:0 16px;font-size:13px;font-weight:700;cursor:pointer;white-space:nowrap}}
.summary-line{{margin:14px 2px 10px;color:#475467;font-size:13px}} .year-section{{margin-top:12px;background:#fff;border:1px solid var(--line);border-radius:14px;overflow:hidden}}
.year-toggle{{width:100%;border:0;background:#fff;padding:15px 17px;display:flex;justify-content:space-between;align-items:center;cursor:pointer;font-size:15px;font-weight:700;color:var(--ink)}} .year-meta{{color:var(--muted);font-size:12px;font-weight:500}}
.chev{{display:inline-block;transition:transform .18s ease;margin-right:8px}} .year-section.open .chev{{transform:rotate(90deg)}} .year-body{{display:none;border-top:1px solid var(--line)}} .year-section.open .year-body{{display:block}}
.table-wrap{{overflow:auto;max-height:620px}} table{{width:100%;border-collapse:collapse;min-width:1250px;font-size:12px}} thead th{{position:sticky;top:0;z-index:2;background:#F9FAFB;color:#475467;text-align:left;padding:10px 9px;border-bottom:1px solid var(--line);white-space:nowrap}} tbody td{{padding:10px 9px;border-bottom:1px solid #F0F2F5;vertical-align:top}} tbody tr:hover{{background:#FCFCFD}}

.pagination{{
  display:flex;
  justify-content:space-between;
  align-items:center;
  flex-wrap:wrap;
  gap:12px;
  padding:12px 16px;
  border-top:1px solid #E4E7EC;
  font-size:12px;
  color:#667085;
}}
.pagination > div{{
  display:flex;
  align-items:center;
  gap:10px;
}}
.pagination button{{
  border:1px solid #D0D5DD;
  border-radius:7px;
  background:white;
  padding:7px 12px;
  font-size:12px;
  cursor:pointer;
}}
.pagination button:disabled{{
  opacity:.4;
  cursor:not-allowed;
}}

.money{{text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}} .nowrap{{white-space:nowrap}} .desc{{width:420px;min-width:340px;max-width:480px;line-height:1.4;white-space:normal;overflow-wrap:anywhere}} .supplier-accent{{color:#6B00B6;font-weight:700}}
.badge{{display:inline-block;border-radius:999px;padding:3px 7px;font-size:11px;font-weight:700;white-space:nowrap}} .badge-large{{background:#ECFDF3;color:var(--green)}} .badge-long{{background:#FFFAEB;color:var(--amber)}} .empty{{background:#fff;border:1px dashed #D0D5DD;border-radius:14px;padding:36px;text-align:center;color:var(--muted);margin-top:16px}} .method{{margin-top:16px;color:var(--muted);font-size:12px;line-height:1.5}}
@media(max-width:1200px){{.kpis{{grid-template-columns:repeat(3,1fr)}}.toolbar{{grid-template-columns:repeat(2,minmax(0,1fr))}}.insight-grid{{grid-template-columns:1fr}}}} @media(max-width:650px){{.wrap{{padding:15px}}.kpis{{grid-template-columns:1fr 1fr}}.toolbar{{grid-template-columns:1fr}}}}

.atlas-market{{display:flex;align-items:center;justify-content:space-between;gap:18px;margin:20px 0 26px;padding:24px 28px;border:1px solid #e5e7eb;background:#fff;border-radius:18px;flex-wrap:wrap}}
.atlas-market-copy h2{{font-size:22px;margin:0 0 6px;color:#101828}}
.atlas-market-copy p{{margin:0;color:#667085;font-size:14px}}
.atlas-market-switch{{display:flex;gap:5px;background:#f1f2f6;padding:6px;border-radius:14px}}
.atlas-market-switch button{{border:0;background:transparent;color:#344054;padding:13px 22px;border-radius:10px;font-size:14px;font-weight:700;cursor:pointer;white-space:nowrap}}
.atlas-market-switch button[aria-pressed="true"]{{background:#A100FF;color:#fff;box-shadow:0 2px 7px rgba(70,0,130,.12)}}
.atlas-market-switch button:focus-visible{{outline:3px solid #805ad5;outline-offset:2px}}
#marketMode{{display:none}}
@media(max-width:650px){{.atlas-market{{padding:17px}}.atlas-market-switch{{width:100%}}.atlas-market-switch button{{flex:1;padding:12px 8px;font-size:12px}}}}

/* Distinct open-opportunities intelligence; isolated from historical CN filters. */
.atm-area{{margin:30px 0 12px;background:#fff;border:1px solid var(--line);border-radius:16px;padding:22px}}
.atm-area h2{{font-size:23px;margin:0 0 8px}} .atm-sub{{color:#526070;font-size:13px;line-height:1.6}}
.atm-controls{{display:flex;flex-wrap:wrap;gap:10px;margin:17px 0}} .atm-controls select,.atm-controls input{{width:auto;min-width:190px;flex:1}}
.atm-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}} .atm-card{{background:#fcfcfe;border:1px solid var(--line);border-radius:12px;padding:16px;min-width:0}}
.atm-card h3{{font-size:16px;line-height:1.35;margin:8px 0}} .atm-tags{{display:flex;gap:6px;flex-wrap:wrap}} .atm-tag{{font-size:11px;padding:5px 8px;border-radius:8px;background:#f1e9ff;color:#6132a1;font-weight:700}}
.atm-card p{{font-size:12px;line-height:1.5;color:#475467}} .atm-card a{{color:#6c22b0;font-weight:700;text-decoration:underline}} .atm-info{{color:#667085;font-size:11px}}
@media(max-width:850px){{.atm-grid{{grid-template-columns:1fr}}}}

/* OPTION A: two complete, independently filtered workspaces. */
.explorer-heading{{display:flex;justify-content:space-between;align-items:center;gap:20px;margin:10px 0 22px}}
.explorer-eyebrow{{font-weight:800;font-size:11px;letter-spacing:.12em;color:#7811b2;margin-bottom:9px}}
.explorer-heading h1{{letter-spacing:-.035em}}.explorer-heading .subtitle{{margin:0}}.explorer-build-note{{white-space:nowrap;border:1px solid #e4daed;border-radius:99px;padding:8px 12px;font-size:11px;color:#694b78;background:#faf6ff}}
.explorer-tabs{{display:flex;gap:8px;border-bottom:1px solid #e4dfea}}
.explorer-tab{{border:1px solid #e4dfea;border-bottom:0;border-radius:12px 12px 0 0;background:#eeebf2;color:#5a5263;font-weight:750;padding:15px 26px;cursor:pointer;font-size:14px}}
.explorer-tab.active,.explorer-tab[aria-selected="true"]{{background:#fff;color:#6b159a;border-top:3px solid var(--purple);padding-top:13px}}
.explorer-tab:focus-visible{{outline:3px solid #9f65cf;outline-offset:2px}}
.explorer-view{{padding-top:16px}}.explorer-view[hidden]{{display:none!important}}
.explorer-section-heading{{display:flex;gap:18px;align-items:flex-start;justify-content:space-between;margin-bottom:18px}}
.explorer-section-heading h2{{font-size:23px;margin:0 0 6px}}.explorer-section-heading p{{margin:0;max-width:900px}}
.explorer-subtle{{font-size:11px;color:#786f82;white-space:nowrap;padding-top:6px}}
#viewTenders .atm-area{{border:0;background:transparent;padding:0;margin:0}}
#viewTenders .atm-controls{{position:sticky;top:0;z-index:14;display:flex;flex-wrap:wrap;gap:10px;padding:14px;background:#fff;border:1px solid #e4dfea;border-radius:13px;box-shadow:0 7px 22px rgba(31,11,50,.08);margin:16px 0 20px}}
#viewTenders .atm-controls input,#viewTenders .atm-controls select{{height:42px;min-width:170px;flex:1}}
#viewTenders .atm-controls input{{flex:1.4;min-width:240px}}
#viewTenders .atm-controls .clear-btn{{height:42px}}
.atm-kpis{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:13px;margin:0 0 25px}}
.atm-stat{{background:#fff;border:1px solid var(--line);border-radius:13px;padding:18px;min-width:0}}
.atm-stat span,.atm-stat small{{display:block;font-size:11px;color:#667085}}.atm-stat span{{font-weight:700}}.atm-stat strong{{font-size:27px;display:block;margin:7px 0 3px;color:#2e1540}}
.explorer-list-heading{{display:flex;align-items:baseline;justify-content:space-between;gap:12px;margin:0 0 12px}}.explorer-list-heading h3{{margin:0;font-size:19px}}
#viewTenders .atm-grid{{display:block;background:white;border:1px solid var(--line);border-radius:13px;overflow:hidden}}
#viewTenders .atm-card{{border:0;border-radius:0;border-bottom:1px solid #eee9f2;background:white;padding:0}}
#viewTenders .atm-card:last-child{{border-bottom:0}}
.atm-card-summary{{list-style:none;padding:17px 19px;cursor:pointer;display:grid;grid-template-columns:minmax(0,2fr) minmax(0,1.1fr) minmax(0,1.1fr) auto;gap:12px;align-items:center}}
.atm-card-summary::-webkit-details-marker{{display:none}}.atm-card-summary:hover{{background:#faf8fd}}
.atm-card-title{{display:block;font-size:14px;font-weight:750;color:#24162e;margin-bottom:4px;line-height:1.35}}
.atm-card-meta{{display:block;font-size:11px;color:#667085;line-height:1.4}}
.atm-card-summary .atm-tag{{display:inline-block;margin:0}}.atm-expand{{color:#7529a6;font-weight:700;font-size:12px;white-space:nowrap}}
.atm-card-details{{padding:5px 20px 19px;background:#fcfaff;border-top:1px solid #eee9f2}}
.atm-card-details p{{max-width:1100px}}.atm-info{{line-height:1.5}}
.explorer-method{{background:#fff;border:1px solid #e8e3ed;border-radius:12px;padding:15px 18px;margin:18px 0;color:#667085;font-size:12px}}
.explorer-method summary{{cursor:pointer;font-weight:750;color:#6c328d;font-size:13px}}.explorer-method p{{line-height:1.65}}
#viewHistorical .toolbar{{position:sticky;top:0;z-index:14;box-shadow:0 7px 22px rgba(31,11,50,.09)}}
#viewHistorical .atlas-market{{margin:0 0 19px;padding:18px 21px}}
#viewHistorical .atlas-market-copy h2{{font-size:17px}}
#viewHistorical .insight-grid{{margin-top:20px}}
@media(max-width:1100px){{.atm-kpis{{grid-template-columns:repeat(2,minmax(0,1fr))}}.atm-card-summary{{grid-template-columns:minmax(0,1.5fr) minmax(0,1fr) auto}}.atm-card-summary .atm-position{{display:none}}}}
@media(max-width:700px){{.explorer-heading{{display:block}}.explorer-build-note,.explorer-subtle{{display:none}}.explorer-tabs{{gap:5px}}.explorer-tab{{flex:1;padding:13px 8px;font-size:12px}}.explorer-section-heading{{display:block}}.atm-kpis{{grid-template-columns:repeat(2,minmax(0,1fr))}}.atm-card-summary{{grid-template-columns:1fr auto}}.atm-card-summary .atm-position,.atm-card-summary .atm-address{{display:none}}#viewTenders .atm-controls{{position:relative}}#viewHistorical .toolbar{{position:relative}}}}
</style>
</head>
<body>
<div class="wrap">
<div class="explorer-heading"><div><div class="explorer-eyebrow">FEDERAL HEALTH INTELLIGENCE / EXPLORER</div><h1>Health Contract Explorer</h1><p class="subtitle">Find current market opportunities or investigate historical contract awards, without losing your filters.</p></div><span class="explorer-build-note">ATLAS · AusTender intelligence</span></div>
<div class="explorer-tabs" role="tablist" aria-label="Explorer view">
  <button type="button" id="tabTenders" class="explorer-tab active" role="tab" aria-controls="viewTenders" aria-selected="true" data-explorer-view="tenders">Open Tenders</button>
  <button type="button" id="tabHistorical" class="explorer-tab" role="tab" aria-controls="viewHistorical" aria-selected="false" data-explorer-view="historical">Historical Contracts</button>
</div>
<div id="viewTenders" class="explorer-view" role="tabpanel" aria-labelledby="tabTenders">
<section class="atm-area" id="open-tenders">
<div class="explorer-section-heading"><div><h2>Open Tenders / Approaches to Market</h2><p class="atm-sub">Opportunity signals based on ATLAS service-offering evidence and historical Health awards. Agency mentions are <strong>not verified buyers</strong>.</p></div><span class="explorer-subtle">Current feed snapshot · status unverified</span></div>
<p class="atm-info">ATM feed retrieved: {html_escape(atm_timestamp)} · {html_escape(atm_status)} <a href="https://www.tenders.gov.au/?event=public.ATM.list" target="_blank" rel="noopener noreferrer">Check official live ATM list</a></p>
<div class="atm-controls" aria-label="Opportunity filters">
 <input type="search" id="atmSearch" placeholder="Search tenders, agencies or service offerings..." aria-label="Search open tenders">
 <select id="atmScope" aria-label="ATM scope"><option value="scope">Health Agency + Addressable</option><option value="agency">Health Agency</option><option value="health">Health Mentioned</option><option value="all">All Feed Notices</option></select>
 <select id="atmOffering" aria-label="ATM Service Offering"><option value="all">All Service Offerings</option></select>
 <select id="atmSignal" aria-label="Opportunity signal"><option value="all">All opportunity signals</option></select>
 <button id="atmReset" class="clear-btn" type="button">Clear filters</button>
</div>
<div class="atm-kpis" aria-label="Opportunity snapshot">
 <div class="atm-stat"><span>Feed notices</span><strong id="atmKpiFeed">—</strong><small>Not confirmed open</small></div>
 <div class="atm-stat"><span>ATLAS agency mentions</span><strong id="atmKpiAgency">—</strong><small>Buyer unverified</small></div>
 <div class="atm-stat"><span>Service candidates</span><strong id="atmKpiCandidate">—</strong><small>Preliminary assessment</small></div>
 <div class="atm-stat"><span>Showing</span><strong id="atmKpiVisible">—</strong><small>Filtered notices</small></div>
</div>
<div class="explorer-list-heading"><h3>Opportunity watchlist</h3><span id="atmCount" class="atm-info"></span></div><div id="atmCards" class="atm-grid"></div>
<details class="explorer-method"><summary>How opportunity categorisation works · methodology &amp; limitations</summary><p class="atm-sub">Methodology: The configured ATLAS agency names must be explicitly mentioned in RSS title/summary for the first two filters. Mentions do not prove buyer identity; missing mentions do not prove an ATM is outside scope. Addressability is preliminary: specific service keywords are "Potentially addressable", goods-only indicators are "Likely non-addressable", other cases require manual review. Service Offering inferred from published title/summary keyword evidence. Historical Accenture presence is calculated from addressable Health CNs in this build, using signed values. It is <strong>not</strong> evidence of incumbency with the ATM buyer. “Potential whitespace” means no historical Accenture value is identified for that offering; “Established Health presence” means ≥5% of historical offering value; otherwise a nonzero presence is an “Expansion opportunity”. Agency, tender closing date, budget, and current open status must be verified at the linked AusTender notice. Source: Department of Finance AusTender ATM RSS.</p></details>
</section>
</div>
<div id="viewHistorical" class="explorer-view" role="tabpanel" aria-labelledby="tabHistorical" hidden>
<div class="explorer-section-heading historical-heading"><div><h2>Historical Contracts</h2><p class="atm-sub">Explore awarded contract notices by agency, supplier, Service Offering and financial year. All original contract-level functionality is retained.</p></div><span class="explorer-subtle">Signed contract values</span></div>
<section class="atlas-market" aria-label="Market selection">
  <div class="atlas-market-copy"><h2>Explore the Federal Health Market</h2><p id="atlasMarketDescription">Showing contracts addressable to Accenture</p></div>
  <div class="atlas-market-switch" role="group" aria-label="Choose market scope">
    <button type="button" data-market="tam" aria-pressed="true">Accenture TAM</button>
    <button type="button" data-market="all" aria-pressed="false">All Federal Health Market</button>
  </div>
  <select id="marketMode" aria-label="Market scope"><option value="tam">Accenture TAM</option><option value="all">All Federal Health Market</option></select>
</section>


<section class="insight-grid">
<div class="leader-card"><h3>Largest active contracts</h3><p id="activeSubtitle">Highest-value contracts active as at today within the selected agency.</p><div id="topActiveContracts"></div></div>
<div class="leader-card"><h3>Major contracts ending in the next 12 months</h3><p id="endingSubtitle">Highest-value addressable active contracts approaching their recorded end date.</p><div id="topEndingSoon"></div></div>
<div class="leader-card"><h3>Recently completed major contracts</h3><p id="completedSubtitle">Highest-value contracts that ended within the past 12 months.</p><div id="topRecentlyCompleted"></div></div>
</section>

<section class="kpis">
<div class="kpi"><div class="kpi-label">Selected market</div><div class="kpi-value" id="kpiTotal">-</div><div class="kpi-sub">{value_basis}</div></div>
<div class="kpi"><div class="kpi-label">Contracts</div><div class="kpi-value" id="kpiContracts">-</div><div class="kpi-sub">Matching contracts</div></div>
<div class="kpi"><div class="kpi-label">Average value</div><div class="kpi-value" id="kpiAvgValue">-</div><div class="kpi-sub" id="kpiMedianValue">-</div></div>
<div class="kpi"><div class="kpi-label">Average length</div><div class="kpi-value" id="kpiAvgLength">-</div><div class="kpi-sub" id="kpiMedianLength">-</div></div>
<div class="kpi"><div class="kpi-label">Suppliers</div><div class="kpi-value" id="kpiSuppliers">-</div><div class="kpi-sub">Distinct supplier groups</div></div>
<div class="kpi"><div class="kpi-label">Largest contract</div><div class="kpi-value" id="kpiLargest">-</div><div class="kpi-sub" id="kpiLargestSub">-</div></div>
</section>

<section class="toolbar">
<div class="field"><label for="searchInput">Search description, supplier, contract ID or category</label><input id="searchInput" type="text" placeholder="Search contracts..."></div>
<div class="field"><label for="agencySelect">Agency</label><select id="agencySelect">{agency_options}</select></div>
<div class="field"><label for="capabilityFilter">Service Offering</label><select id="capabilityFilter"><option value="All Service Offerings">All Service Offerings</option></select></div>
<div class="field"><label for="supplierFilter">Supplier</label><select id="supplierFilter"><option value="All suppliers">All suppliers</option></select></div>
<div class="field"><label for="labourHireFilter">Labour Hire</label><select id="labourHireFilter"><option value="all">All contracts</option><option value="yes">Labour hire only</option><option value="no">Exclude labour hire</option></select></div>
<div class="field"><label for="minValueInput">Minimum value</label><input id="minValueInput" type="number" min="0" step="100000" placeholder="e.g. 1000000"></div>
<label class="check-wrap"><input id="datedOnly" type="checkbox"> Dated contracts only</label>
<button id="clearFiltersBtn" class="clear-btn" type="button">Clear all filters</button>
</section>




<div class="summary-line" id="summaryLine"></div>
<p class="method">Dates marked (inferred) use the earliest credible recorded start and latest credible recorded end across available CN family members. This is an observed range, not confirmation of the current legal contract term.</p>
<div id="yearContainer"></div>



<details class="explorer-method"><summary>Historical contract methodology &amp; value definitions</summary><p class="method">The dashboard supports both Accenture TAM and All Federal Health Market. Ongoing contracts are shown once in the top bracket; remaining contracts are grouped by their recorded financial year. Supplier and Service Offering filters are ranked by addressable spend within the current agency scope. Negative amendments are retained as signed values.</p></details>
</div>
</div>

<script>

const ATM_NOTICES = {atm_json};
const ATM_FEED_AVAILABLE = {json.dumps(atm_timestamp != "Unavailable")};
(function initATM() {{
  const search = document.getElementById('atmSearch');
  const scope = document.getElementById('atmScope');
  const offer = document.getElementById('atmOffering');
  const signal = document.getElementById('atmSignal');
  const cards = document.getElementById('atmCards');
  const count = document.getElementById('atmCount');
  const esc = x => String(x ?? '').replace(/[&<>"']/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}})[ch]);
  for (const [sel, key] of [[offer,'offering'],[signal,'signal']]) {{
    for (const v of [...new Set(ATM_NOTICES.map(n => n[key]))].sort()) {{
      const option = document.createElement('option'); option.value=v; option.textContent=v;sel.append(option);
    }}
  }}
  function renderATM() {{
    const q = search.value.trim().toLowerCase();
    const rows = ATM_NOTICES.filter(n => (scope.value === 'all' || (scope.value === 'health' && n.health_candidate) || (scope.value === 'agency' && n.agency_mentions.length > 0) || (scope.value === 'scope' && n.scope_candidate)) && (offer.value==='all'||offer.value===n.offering) && (signal.value==='all'||signal.value===n.signal) && (!q||(n.title+' '+n.description+' '+n.ref).toLowerCase().includes(q)));
    document.getElementById('atmKpiFeed').textContent = ATM_FEED_AVAILABLE ? ATM_NOTICES.length.toLocaleString() : '—';
    document.getElementById('atmKpiAgency').textContent = ATM_FEED_AVAILABLE ? ATM_NOTICES.filter(n => n.agency_mentions.length > 0).length.toLocaleString() : '—';
    document.getElementById('atmKpiCandidate').textContent = ATM_FEED_AVAILABLE ? ATM_NOTICES.filter(n => n.scope_candidate).length.toLocaleString() : '—';
    document.getElementById('atmKpiVisible').textContent = ATM_FEED_AVAILABLE ? rows.length.toLocaleString() : '—';
    count.textContent = ATM_FEED_AVAILABLE
      ? `${{rows.length}} matching notices of ${{ATM_NOTICES.length}} feed records. Agency identity and open status require verification on AusTender.`
      : 'ATM feed unavailable at build time — current opportunities could not be determined.';
    cards.replaceChildren();
    if (!ATM_FEED_AVAILABLE) {{cards.innerHTML='<p class="atm-sub"><strong>Live feed not available.</strong> This is not confirmation that no tenders are open. Check the official AusTender link above.</p>';return;}}
    if (!rows.length) {{cards.innerHTML='<p class="atm-sub">No matching notices for this scope. This does not mean there are no relevant current tenders: the RSS may omit buyer-agency names. Select “Health Mentioned” or “All Feed Notices” to review unverified notices.</p>';return;}}
    for (const n of rows) {{
      const article = document.createElement('details'); article.className='atm-card';
      article.innerHTML = `<summary class="atm-card-summary">
        <span><span class="atm-card-title">${{esc(n.title)}}</span><span class="atm-card-meta">${{esc(n.ref)}} · ${{esc(n.agency_mentions.join(', ')||'Buyer not identified')}} · Unverified buyer</span></span>
        <span class="atm-address"><span class="atm-tag">${{esc(n.offering)}}</span><span class="atm-card-meta">${{esc(n.addressability)}}</span></span>
        <span class="atm-position"><span class="atm-tag">${{esc(n.signal)}}</span></span>
        <span class="atm-expand">Details ▾</span>
      </summary>
      <div class="atm-card-details"><p>${{esc(n.description)}}</p>
      <p class="atm-info"><strong>ATLAS agency mentioned:</strong> ${{esc(n.agency_mentions.join(', ')||'None identified')}} · <strong>Buyer not verified</strong></p>
      <p class="atm-info"><strong>Addressability:</strong> ${{esc(n.addressability_reason)}}</p>
      <p><strong>Why it matters:</strong> ${{esc(n.rationale)}}</p><p class="atm-info">${{esc(n.match_evidence)}} · Published: ${{esc(n.published||'Not provided')}}</p>
      <a href="${{esc(n.url)}}" target="_blank" rel="noopener noreferrer">Verify notice and closing date on AusTender</a></div>`;
      cards.append(article);
    }}
  }}
  for (const el of [search,scope,offer,signal]) el.addEventListener(el===search?'input':'change',renderATM);
  document.getElementById('atmReset').addEventListener('click', () => {{
    search.value=''; scope.value='scope'; offer.value='all'; signal.value='all'; renderATM();
  }});
  renderATM();
}})();

const DATA = {json.dumps(payload, ensure_ascii=False)};
const ALL_AGENCIES = "All Health agencies";
const DEFAULT_AGENCY = ALL_AGENCIES;
const VALUE_BASIS = {json.dumps(value_basis)};
const AS_AT_DATE = new Date().toISOString().slice(0,10);
const YEAR_ROWS = new Map();
function fyFromDate(iso) {{
  if (!iso) return null;
  const d = new Date(iso + 'T00:00:00Z');
  if (isNaN(d.getTime())) return null;
  return d.getUTCFullYear() - (d.getUTCMonth() < 6 ? 1 : 0);
}}

function fyLabel(year) {{
  return year === null
    ? 'Unknown FY'
    : 'FY' + year + '–' + String((year + 1) % 100).padStart(2, '0');
}}

function contractStatus(r) {{
  if (!r.start_date_iso || !r.end_date_iso ||
      r.start_date_iso > r.end_date_iso) return 'Unknown Dates';
  if (r.start_date_iso > AS_AT_DATE) return 'Upcoming contracts';
  if (r.end_date_iso < AS_AT_DATE) return 'Completed contracts';
  return 'Active contracts';
}}


function esc(v){{return String(v??'').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;').replaceAll("'",'&#039;')}}
function money(v){{v=Number(v||0);const s=v<0?'-':'';const a=Math.abs(v);if(a>=1e9)return s+'$'+(a/1e9).toFixed(2)+'B';if(a>=1e8)return s+'$'+(a/1e6).toFixed(0)+'M';if(a>=1e7)return s+'$'+(a/1e6).toFixed(1)+'M';if(a>=1e6)return s+'$'+(a/1e6).toFixed(2)+'M';if(a>=1e3)return s+'$'+(a/1e3).toFixed(0)+'K';return s+'$'+a.toFixed(0)}}
function number(v){{return Number(v||0).toLocaleString('en-AU')}}
function duration(days){{if(days===null||days===undefined||!isFinite(Number(days)))return'n/a';const d=Number(days),y=d/365.25;if(y>=2)return y.toFixed(1)+' yrs';if(y>=1)return y.toFixed(2)+' yrs';return Math.round(d)+' days'}}
function median(values){{const a=values.filter(v=>Number.isFinite(v)).sort((x,y)=>x-y);if(!a.length)return null;const m=Math.floor(a.length/2);return a.length%2?a[m]:(a[m-1]+a[m])/2}}

function marketRows(){{return DATA.records.filter(r=>document.getElementById("marketMode").value==="all"||r.is_addressable)}}
function agencyRows(){{const agency=document.getElementById('agencySelect').value;return marketRows().filter(r=>agency===ALL_AGENCIES||r.agency===agency)}}
function scopeRows(){{return agencyRows()}}
function rowsForOptions(exclude){{const search=document.getElementById('searchInput').value.trim().toLowerCase();const cap=document.getElementById('capabilityFilter').value;const sup=document.getElementById('supplierFilter').value;const min=Number(document.getElementById('minValueInput').value||0);const dated=document.getElementById('datedOnly').checked;const labour=document.getElementById('labourHireFilter').value;return scopeRows().filter(r=>{{const hay=[r.description,r.supplier,r.contract_id,r.category,r.procurement_method,r.capability,r.agency].join(' ').toLowerCase();return(!search||hay.includes(search))&&(exclude==='capability'||cap==='All Service Offerings'||r.capability===cap)&&(exclude==='supplier'||sup==='All suppliers'||r.supplier===sup)&&(labour==='all'||r.is_labour_hire===(labour==='yes'))&&Number(r.value||0)>=min&&(!dated||(r.start_date&&r.end_date))}})}}
function setRanked(selectId,field,allLabel,exclude){{const select=document.getElementById(selectId);const prev=select.value||allLabel;const totals=new Map();rowsForOptions(exclude).forEach(r=>totals.set(r[field],(totals.get(r[field])||0)+Number(r.value||0)));const vals=[...totals.keys()].sort((a,b)=>(totals.get(b)-totals.get(a))||String(a).localeCompare(String(b)));select.innerHTML=[allLabel,...vals].map(v=>'<option value="'+esc(v)+'">'+esc(v===allLabel?v:v+' — '+money(totals.get(v)))+'</option>').join('');select.value=[...select.options].some(o=>o.value===prev)?prev:allLabel}}
function refreshFilters(changed=null){{if(changed!=='capability')setRanked('capabilityFilter','capability','All Service Offerings','capability');if(changed!=='supplier')setRanked('supplierFilter','supplier','All suppliers','supplier')}}
function filteredRows(){{const search=document.getElementById('searchInput').value.trim().toLowerCase();const cap=document.getElementById('capabilityFilter').value;const sup=document.getElementById('supplierFilter').value;const min=Number(document.getElementById('minValueInput').value||0);const dated=document.getElementById('datedOnly').checked;const labour=document.getElementById('labourHireFilter').value;return scopeRows().filter(r=>{{const hay=[r.description,r.supplier,r.contract_id,r.category,r.procurement_method].join(' ').toLowerCase();return(!search||hay.includes(search))&&(cap==='All Service Offerings'||r.capability===cap)&&(sup==='All suppliers'||r.supplier===sup)&&(labour==='all'||r.is_labour_hire===(labour==='yes'))&&Number(r.value||0)>=min&&(!dated||(r.start_date&&r.end_date))}})}}

function renderContractRows(id,rows,dateLabel){{const el=document.getElementById(id);if(!rows.length){{el.innerHTML='<div class="leader-detail">No matching contracts.</div>';return}}el.innerHTML=rows.map((r,i)=>{{const title=(r.description||r.contract_id||'Contract');const supplier=r.supplier||'Unknown supplier';const agency=r.agency||'Unknown agency';const date=r.end_date||'n/a';return '<div class="leader-row"><div class="leader-rank">'+(i+1)+'</div><div class="leader-name">'+esc(supplier)+'<div class="leader-detail">'+esc(title)+'</div><div class="leader-detail">'+esc(agency)+' · '+dateLabel+' '+esc(date)+'</div></div><div class="leader-value">'+esc(money(r.value))+'</div></div>'}}).join('')}}
function updateContractInsights(){{const agency=document.getElementById('agencySelect').value;const rows=marketRows().filter(r=>agency===ALL_AGENCIES||r.agency===agency);const today=new Date(AS_AT_DATE+'T00:00:00Z');const in12=new Date(today);in12.setUTCFullYear(in12.getUTCFullYear()+1);const ago12=new Date(today);ago12.setUTCFullYear(ago12.getUTCFullYear()-1);const parse=d=>d?new Date(d+'T00:00:00Z'):null;const active=rows.filter(r=>{{const s=parse(r.start_date_iso),e=parse(r.end_date_iso);return s&&e&&s<=today&&e>=today}}).sort((a,b)=>Number(b.value||0)-Number(a.value||0)).slice(0,5);const ending=rows.filter(r=>{{const s=parse(r.start_date_iso),e=parse(r.end_date_iso);return s&&e&&s<=today&&e>=today&&e<=in12}}).sort((a,b)=>Number(b.value||0)-Number(a.value||0)).slice(0,5);const completed=rows.filter(r=>{{const e=parse(r.end_date_iso);return e&&e<today&&e>=ago12}}).sort((a,b)=>Number(b.value||0)-Number(a.value||0)).slice(0,5);const scopeLabel=agency===ALL_AGENCIES?'all Health agencies':agency;document.getElementById('activeSubtitle').textContent='Highest-value contracts active as at today across '+scopeLabel+'.';document.getElementById('endingSubtitle').textContent='Highest-value addressable active contracts ending within the next 12 months across '+scopeLabel+'.';document.getElementById('completedSubtitle').textContent='Highest-value contracts completed within the past 12 months across '+scopeLabel+'.';renderContractRows('topActiveContracts',active,'Ends');renderContractRows('topEndingSoon',ending,'Ends');renderContractRows('topRecentlyCompleted',completed,'Ended')}}


function updateKpis(rows) {{
  const vals = rows.map(r => Number(r.value || 0));
  const dur = rows
    .filter(r => r.duration_days !== null && r.duration_days !== undefined)
    .map(r => Number(r.duration_days))
    .filter(Number.isFinite);

  const total = vals.reduce((a,b) => a+b, 0);
  const largest = rows.slice().sort(
    (a,b) => Number(b.value || 0) - Number(a.value || 0)
  )[0];

  const setText = (id, value) => {{
    const el = document.getElementById(id);
    if (el) el.textContent = value;
    else console.warn('Missing KPI element:', id);
  }};

  setText('kpiTotal', money(total));
  setText('kpiContracts', number(rows.length));  setText('kpiAvgValue',
    rows.length ? money(total / rows.length) : 'n/a');
  setText('kpiMedianValue',
    rows.length ? 'Median: ' + money(median(vals)) : 'Median: n/a');
  setText('kpiAvgLength',
    dur.length ? duration(dur.reduce((a,b) => a+b, 0) / dur.length) : 'n/a');
  setText('kpiMedianLength',
    dur.length ? 'Median: ' + duration(median(dur)) : 'Median: n/a');
  setText('kpiSuppliers',
    number(new Set(rows.map(r => r.supplier)).size));
  setText('kpiLargest',
    largest ? money(largest.value) : 'n/a');
  setText('kpiLargestSub',
    largest ? largest.supplier : 'No matching contract');
}}


function rowHtml(r){{const large=Number(r.value||0)>=20000000?'<span class="badge badge-large">Large</span>':'',long=Number(r.duration_days||0)>=1826?'<span class="badge badge-long">5+ years</span>':'',sc=String(r.supplier||'').toLowerCase().includes('accenture')?'supplier-accent':'';return`<tr><td class="nowrap">${{esc(r.contract_id)}}</td><td class="desc">${{esc(r.description)}}</td><td>${{esc(r.agency)}}</td><td>${{esc(r.division)}}</td><td class="${{sc}}">${{esc(r.supplier)}}</td><td>${{esc(r.capability)}}</td><td class="money">${{money(r.value)}} ${{large}}</td><td class="nowrap">${{esc(r.start_date)}}</td><td class="nowrap">${{esc(r.end_date)}}</td><td>${{esc(r.duration_label)}} ${{long}}</td><td>${{esc(r.category)}}</td><td class="nowrap">${{esc(r.extensions)}}</td></tr>`}}

const PAGE_SIZE = 100;
const PAGE_STATE = new Map();

function tableHtml(rows, page = 0) {{
  const totalPages = Math.max(1, Math.ceil(rows.length / PAGE_SIZE));
  const currentPage = Math.max(0, Math.min(page, totalPages - 1));
  const start = currentPage * PAGE_SIZE;
  const visibleRows = rows.slice(start, start + PAGE_SIZE);

  return `
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Contract ID</th>
            <th>Description</th>
            <th>Agency</th>
            <th>Division</th>
            <th>Supplier</th>
            <th>Service Offering</th>
            <th class="money">Contract value</th>
            <th>Start date</th>
            <th>End date</th>
            <th>Contract length</th>
            <th>Category</th>
            <th>Extensions</th>
          </tr>
        </thead>
        <tbody>${{visibleRows.map(rowHtml).join('')}}</tbody>
      </table>
    </div>
    <div class="pagination">
      <span>
        Showing ${{start + 1}}–${{Math.min(start + PAGE_SIZE, rows.length)}}
        of ${{number(rows.length)}} contracts
      </span>
      <div>
        <button type="button" data-page-action="prev"
          ${{currentPage === 0 ? 'disabled' : ''}}>Previous</button>
        <span>Page ${{currentPage + 1}} of ${{totalPages}}</span>
        <button type="button" data-page-action="next"
          ${{currentPage >= totalPages - 1 ? 'disabled' : ''}}>Next</button>
      </div>
    </div>`;
}}
function bracket(id,title,rows,label='contracts'){{const spend=rows.reduce((s,r)=>s+Number(r.value||0),0);return`<section class="year-section" id="${{id}}"><button class="year-toggle" type="button" onclick="toggleYear('${{id}}')"><span><span class="chev">▶</span>${{esc(title)}}</span><span class="year-meta">${{number(rows.length)}} ${{label}} · ${{money(spend)}}</span></button><div class="year-body" data-loaded="false"></div></section>`}}

function renderYearPage(id) {{
  const section = document.getElementById(id);
  const body = section.querySelector('.year-body');
  const rows = YEAR_ROWS.get(id) || [];
  const page = PAGE_STATE.get(id) || 0;

  body.innerHTML = tableHtml(rows, page);
  body.dataset.loaded = 'true';

  body.querySelectorAll('[data-page-action]').forEach(button => {{
    button.addEventListener('click', () => {{
      const direction = button.dataset.pageAction === 'next' ? 1 : -1;
      const totalPages = Math.max(1, Math.ceil(rows.length / PAGE_SIZE));
      const nextPage = Math.max(
        0,
        Math.min((PAGE_STATE.get(id) || 0) + direction, totalPages - 1)
      );

      PAGE_STATE.set(id, nextPage);
      renderYearPage(id);
    }});
  }});
}}

function toggleYear(id) {{
  const section = document.getElementById(id);
  const opening = !section.classList.contains('open');

  if (opening && section.querySelector('.year-body').dataset.loaded !== 'true') {{
    PAGE_STATE.set(id, 0);
    renderYearPage(id);
  }}

  section.classList.toggle('open');
}}


function render() {{
  const rows = filteredRows();

  updateContractInsights();
  updateKpis(rows);

  document.getElementById('summaryLine').textContent =
    number(rows.length) + ' matching contracts · ' +
    money(rows.reduce((s,r) => s + Number(r.value || 0), 0)) +
    ' · ' + VALUE_BASIS;

  const container = document.getElementById('yearContainer');
  YEAR_ROWS.clear();
  PAGE_STATE.clear();

  if (!rows.length) {{
    container.innerHTML =
      '<div class="empty">No contracts match the selected filters.</div>';
    return;
  }}

  const groups = new Map();

  rows.forEach(r => {{
    const status = contractStatus(r);
    let group = status;

    if (status === 'Completed contracts') {{
      const fy = fyFromDate(r.start_date_iso);
      group = fy === null
        ? 'Unknown Dates'
        : fyLabel(fy);
    }}

    if (!groups.has(group)) groups.set(group, []);
    groups.get(group).push(r);
  }});

  const priority = key => {{
    if (key === 'Active contracts') return 0;
    if (key === 'Upcoming contracts') return 1;
    if (key === 'Unknown Dates') return 3;
    return 2;
  }};

  const keys = [...groups.keys()].sort((a,b) => {{
    const difference = priority(a) - priority(b);
    return difference || (priority(a) === 2
      ? b.localeCompare(a)
      : a.localeCompare(b));
  }});

  const sections = [];

  keys.forEach((key, index) => {{
    const groupRows = groups.get(key)
      .slice()
      .sort((a,b) => Number(b.value || 0) - Number(a.value || 0));

    const id = 'contract_group_' + index;
    YEAR_ROWS.set(id, groupRows);
    sections.push(bracket(id, key, groupRows));
  }});

  container.innerHTML = sections.join('');
}}

  

function clearAllFilters() {{
  document.getElementById('agencySelect').value = DEFAULT_AGENCY;
  document.getElementById('searchInput').value = '';
  document.getElementById('minValueInput').value = '';
  document.getElementById('datedOnly').checked = false;
   document.getElementById('labourHireFilter').value = 'all';

  document.getElementById('capabilityFilter').innerHTML =
    '<option value="All Service Offerings">All Service Offerings</option>';

  document.getElementById('supplierFilter').innerHTML =
    '<option value="All suppliers">All suppliers</option>';

  refreshFilters();
  render();
}}
document.getElementById('marketMode').addEventListener('change',()=>{{refreshFilters();render()}});

(function() {{
  const select = document.getElementById('marketMode');
  const buttons = document.querySelectorAll('.atlas-market-switch button');
  const description = document.getElementById('atlasMarketDescription');
  function syncMarketSwitch() {{
    buttons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.market === select.value)));
    description.textContent = select.value === 'tam'
      ? 'Showing contracts addressable to Accenture'
      : 'Showing all procurement contracts in the selected market';
  }}
  buttons.forEach(button => button.addEventListener('click', () => {{
    if (select.value === button.dataset.market) return;
    select.value = button.dataset.market;
    syncMarketSwitch();
    select.dispatchEvent(new Event('change', {{bubbles: true}}));
  }}));
  select.addEventListener('change', syncMarketSwitch);
  syncMarketSwitch();
}})();
document.getElementById('agencySelect').addEventListener('change',()=>{{refreshFilters();render()}});document.getElementById('searchInput').addEventListener('input',()=>{{refreshFilters();render()}});document.getElementById('capabilityFilter').addEventListener('change',()=>{{refreshFilters('capability');render()}});document.getElementById('supplierFilter').addEventListener('change',()=>{{refreshFilters('supplier');render()}});document.getElementById('minValueInput').addEventListener('input',()=>{{refreshFilters();render()}});document.getElementById('datedOnly').addEventListener('change',()=>{{refreshFilters();render()}});document.getElementById('labourHireFilter').addEventListener('change',()=>{{refreshFilters();render()}});document.getElementById('clearFiltersBtn').addEventListener('click',clearAllFilters);

// Accept only recognised query parameters; do not alter general Explorer defaults.
const inbound=new URLSearchParams(window.location.search);
if(inbound.get('view')==='historical'||inbound.get('labour_hire')==='yes'){{
  const agency=inbound.get('agency');
  if(agency&&[...document.getElementById('agencySelect').options].some(o=>o.value===agency))document.getElementById('agencySelect').value=agency;
  const market=inbound.get('market');
  if(market==='tam'||market==='all'){{document.getElementById('marketMode').value=market;document.getElementById('marketMode').dispatchEvent(new Event('change',{{bubbles:true}}));}}
  if(inbound.get('labour_hire')==='yes')document.getElementById('labourHireFilter').value='yes';
}}
refreshFilters();
render();
(function initExplorerTabs() {{
  const tabs = [...document.querySelectorAll('[data-explorer-view]')];
  const tender = document.getElementById('viewTenders');
  const historical = document.getElementById('viewHistorical');
  function activate(view, updateHash=true) {{
    const showTenders = view !== 'historical';
    tender.hidden = !showTenders;
    historical.hidden = showTenders;
    tabs.forEach(tab => {{
      const active = tab.dataset.explorerView === (showTenders ? 'tenders' : 'historical');
      tab.classList.toggle('active', active);
      tab.setAttribute('aria-selected', String(active));
      tab.tabIndex = active ? 0 : -1;
    }});
    if (updateHash) history.replaceState(null, '', showTenders ? '#open-tenders' : '#historical-contracts');
  }}
  tabs.forEach(tab => tab.addEventListener('click', () => activate(tab.dataset.explorerView)));
  window.addEventListener('hashchange', () => activate(window.location.hash === '#historical-contracts' ? 'historical' : 'tenders', false));
  activate(window.location.hash === '#historical-contracts' || new URLSearchParams(window.location.search).get('view') === 'historical' ? 'historical' : 'tenders', false);
}})();
</script>
</body></html>'''

    output_path.write_text(atlas_navigation(html, "health", page="explorer", freshness=freshness), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df, value_col = load_data(input_path, args.value_mode)
    payload = build_payload(df)
    html_path = output_dir / "HealthContractExplorer.html"
    atm_json, atm_timestamp, atm_status = atm_section(df, args.atm_rss_file, args.atm_timeout)
    build_html(payload, html_path, args.value_mode, freshness=atlas_freshness(df), atm_json=atm_json, atm_timestamp=atm_timestamp, atm_status=atm_status)
    print("ATM intelligence:", atm_status)

    agency_audit = (
        df.groupby([AGENCY_COL, "_division"], dropna=False)
        .agg(contracts=(CN_ID_COL, "nunique") if CN_ID_COL in df.columns else ("_value", "size"), value=("_value", "sum"))
        .reset_index()
        .sort_values("value", ascending=False)
    )
    agency_audit.to_csv(output_dir / "agency_division_procurement_audit.csv", index=False)

    print("Health TAM Exploratory Analysis dashboard complete")
    print(f"Input: {input_path}")
    print(f"Value mode: {args.value_mode} / column used: {value_col}")
    print(f"Addressable rows: {len(df):,}")
    print(f"Addressable Health TAM: A${df['_value'].sum()/1e9:,.2f}B")
    print(f"Wrote: {html_path}")
    print(f"Wrote: {output_dir / 'agency_division_procurement_audit.csv'}")




def atlas_freshness(df: pd.DataFrame) -> str:
    """Expose provenance without mistaking HTML build time for source refresh time."""
    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo
    from html import escape
    import os

    built = datetime.now(ZoneInfo("Australia/Sydney")).strftime("%d %b %Y %H:%M %Z")
    pieces = ["Dashboard last refreshed: " + built]
    reported = os.environ.get("ATLAS_DATA_REFRESHED_AT", "").strip()
    if not reported:
        for column in ("atlas_refresh_timestamp", "data_refresh_timestamp", "last_refreshed", "refresh_timestamp"):
            if column in df.columns:
                values = df[column].dropna().astype(str).str.strip()
                values = values[values.ne("")]
                if not values.empty:
                    reported = str(values.iloc[-1]); break
    if reported:
        stamp = pd.to_datetime(reported, errors="coerce", utc=True)
        if pd.notna(stamp):
            pieces.insert(0, "Data refreshed: " + stamp.strftime("%d %b %Y %H:%M UTC"))
        else:
            pieces.insert(0, "Data refreshed: " + reported)
    for column in ("05. Publish Date", "Publish Date"):
        if column in df.columns:
            source = df[column]
            if pd.api.types.is_datetime64_any_dtype(source):
                values = pd.to_datetime(source, errors="coerce")
            else:
                text = source.astype("string").str.strip()
                iso = text.str.match(r"^\d{4}-\d{1,2}-\d{1,2}(?:[ T].*)?$", na=False)
                values = pd.Series(pd.NaT, index=source.index, dtype="datetime64[ns]")
                values.loc[iso] = pd.to_datetime(text.loc[iso], errors="coerce", format="ISO8601")
                values.loc[~iso] = pd.to_datetime(text.loc[~iso], errors="coerce", format="mixed", dayfirst=True)
            if values.notna().any():
                pieces.append("Latest source publication: " + values.max().strftime("%d %b %Y"))
                break
    return " · ".join(escape(p) for p in pieces)


def atlas_navigation(html_text: str, industry: str, page: str = "overview", freshness: str = "") -> str:
    """Consistent Health navigation; no analytics or filters are modified."""
    # All Health HTML files are published side by side in docs/health/.
    # Relative links also work when GitHub Pages is hosted under /ATLAS/
    # or a different base path (such as a custom domain).
    site = "../"
    links = [
        ("Health home", "index.html", "home"),
        ("TAM overview", "AccentureHealthTAMOverview.html", "overview"),
        ("Competitive intelligence", "HealthCompetitiveIntelligence.html", "compare"),
        ("Contract explorer", "HealthContractExplorer.html", "explorer"),
        ("Agency intelligence", "HealthAgencyIntelligence.html", "agency"),
        ("Methodology", "Methodology.html", "methodology"),
    ]
    nav_links = "".join(
        f'<a href="{url}"' + (' aria-current="page"' if ident == page else '') + f'>{label}</a>'
        for label, url, ident in links
    )
    nav = (
        '<div class="atlas-health-top"><nav class="atlas-health-nav" aria-label="ATLAS Health dashboard navigation">'
        f'<a class="atlas-health-brand" href="{site}">ATLAS</a>'
        f'<div class="atlas-health-links">{nav_links}</div>'
        '</nav>'
        + (f'<div class="atlas-health-meta" aria-label="Data provenance">{freshness}</div>' if freshness else '')
        + '</div>'
    )
    css = """<style>
    .atlas-health-top{background:#11101c;color:#f8f7fc;font-family:Arial,Helvetica,sans-serif}
    .atlas-health-nav{display:flex;gap:18px;align-items:center;justify-content:space-between;max-width:1500px;margin:auto;padding:14px 22px}
    .atlas-health-brand{font-size:22px;font-weight:800;letter-spacing:.055em;color:white!important;text-decoration:none;white-space:nowrap}
    .atlas-health-links{display:flex;align-items:center;gap:6px;flex-wrap:wrap}
    .atlas-health-links a{font-size:13px;font-weight:700;color:#e8e3ef;text-decoration:none;padding:9px 12px;border-radius:9px;white-space:nowrap}
    .atlas-health-links a:hover,.atlas-health-links a[aria-current=page]{background:#6730ad;color:#fff}
    .atlas-health-links a:focus-visible{outline:2px solid #e9c9ff;outline-offset:2px}
    .atlas-health-meta{max-width:1500px;margin:auto;padding:0 22px 12px;color:#cbc4d8;font-size:11px;line-height:1.4}
    @media(max-width:850px){.atlas-health-nav{align-items:flex-start;flex-direction:column;gap:10px}.atlas-health-links a{padding:7px 9px}.atlas-health-meta{padding-bottom:12px}}
    </style>"""
    html_text = html_text.replace('</head>', css + '</head>', 1)
    html_text = html_text.replace('<body>', '<body>' + nav, 1)
    return html_text


if __name__ == "__main__":
    main()

