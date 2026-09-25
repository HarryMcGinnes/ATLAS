from __future__ import annotations

import argparse
import json
import re
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
        default="master_output/health_portfolio_contracts.parquet",
        help="Health portfolio procurement dataset (parquet or CSV).",
    )
    p.add_argument("--output-dir", default="HealthTAMExploratoryAnalysis_output")
    p.add_argument(
        "--value-mode",
        choices=["total", "annualised"],
        default="total",
        help="Value used in KPIs and tables.",
    )
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

    # TAM-only dashboard: non-addressable Health procurement never enters the payload,
    # KPI cards, lifecycle panels, filters or contract tables.
    addressable_mask = parse_bool_series(df["is_addressable"])
    df = df.loc[addressable_mask].copy()

    # Defensive TAM guardrail. The master classifier remains the source of truth, but
    # the TAM explorer must never surface obvious clinical/product procurement if an
    # older parquet was generated before the latest Health exclusions. Accenture rows
    # are retained because observed Accenture wins are addressable by business rule.
    desc = df.get(DESCRIPTION_COL, pd.Series("", index=df.index)).fillna("").astype(str).str.lower()
    cat = df.get(CATEGORY_COL, pd.Series("", index=df.index)).fillna("").astype(str).str.lower()
    supplier = df.get(SUPPLIER_GROUP_COL, df.get(RAW_SUPPLIER_COL, pd.Series("", index=df.index))).fillna("").astype(str)
    is_accenture = supplier.str.contains(r"\baccenture\b", case=False, regex=True, na=False)
    obvious_non_tam = (
        desc.str.contains(r"\b(vaccine|vaccines|medicine|medicines|pharmaceutical|pharmaceuticals|medical products?|medical equipment|medical devices?|blood products?|plasma)\b", regex=True, na=False)
        | cat.str.contains(r"\b(medical equipment|medical devices?|patient care and treatment products|disease prevention and control|pharmaceutical|vaccines?)\b", regex=True, na=False)
        | desc.str.fullmatch(r"\s*funding pool\s*", na=False)
    )
    df = df.loc[~obvious_non_tam | is_accenture].copy()
    if df.empty:
        raise SystemExit("No addressable Health contracts were found in the input dataset.")

    if SUPPLIER_GROUP_COL not in df.columns:
        if "supplier_display" in df.columns:
            df[SUPPLIER_GROUP_COL] = df["supplier_display"]
        elif RAW_SUPPLIER_COL in df.columns:
            df[SUPPLIER_GROUP_COL] = df[RAW_SUPPLIER_COL]
        else:
            df[SUPPLIER_GROUP_COL] = "Unknown supplier"

    if CAPABILITY_COL not in df.columns:
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

    out["_value"] = pd.to_numeric(out[value_col], errors="coerce").fillna(0).clip(lower=0)
    out["_total_value"] = pd.to_numeric(out[VALUE_COL], errors="coerce").fillna(0).clip(lower=0)
    out["_fy_start"] = out[FY_COL].map(financial_year_start)

    start_col = first_existing(out, START_DATE_CANDIDATES)
    end_col = first_existing(out, END_DATE_CANDIDATES)
    out["_start"] = pd.to_datetime(out[start_col], errors="coerce", dayfirst=True) if start_col else pd.Series(pd.NaT, index=out.index)
    out["_end"] = pd.to_datetime(out[end_col], errors="coerce", dayfirst=True) if end_col else pd.Series(pd.NaT, index=out.index)
    duration = (out["_end"] - out["_start"]).dt.days
    out["_duration_days"] = duration.where(duration >= 0)

    if out.empty:
        raise SystemExit("No Health contracts were found.")
    return out, value_col


def build_payload(df: pd.DataFrame) -> dict[str, object]:
    extensions_col = first_existing(df, EXTENSIONS_CANDIDATES)
    records: list[dict[str, object]] = []

    for _, row in df.iterrows():
        records.append({
            "agency": safe_str(row.get(AGENCY_COL), "Unknown agency"),
            "division": safe_str(row.get("_division"), "Unspecified division"),
            "financial_year": safe_str(row.get(FY_COL), "Unknown FY"),
            "fy_start": int(row["_fy_start"]) if pd.notna(row.get("_fy_start")) else None,
            "contract_id": safe_str(row.get(CN_ID_COL)),
            "description": safe_str(row.get(DESCRIPTION_COL)),
            "supplier": safe_str(row.get(SUPPLIER_GROUP_COL), "Unknown supplier"),
            "capability": safe_str(row.get(CAPABILITY_COL), "Unclassified"),
            "value": float(row.get("_value", 0) or 0),
            "total_value": float(row.get("_total_value", 0) or 0),
            "start_date": fmt_date(row.get("_start")),
            "end_date": fmt_date(row.get("_end")),
            "start_date_iso": pd.Timestamp(row.get("_start")).strftime("%Y-%m-%d") if pd.notna(row.get("_start")) else "",
            "end_date_iso": pd.Timestamp(row.get("_end")).strftime("%Y-%m-%d") if pd.notna(row.get("_end")) else "",
            "duration_days": float(row["_duration_days"]) if pd.notna(row.get("_duration_days")) else None,
            "duration_label": fmt_duration(row.get("_duration_days")),
            "procurement_method": safe_str(row.get(PROCUREMENT_METHOD_COL)),
            "category": safe_str(row.get(CATEGORY_COL)),
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


def build_html(payload: dict[str, object], output_path: Path, value_mode: str) -> None:
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
.check-wrap{{min-height:42px;display:flex;align-items:center;gap:8px;padding:0 6px;color:#344054;font-size:13px}} .check-wrap input{{width:auto}}
.clear-btn{{min-height:42px;border:1px solid #D0D5DD;border-radius:9px;background:#fff;color:#344054;padding:0 16px;font-size:13px;font-weight:700;cursor:pointer;white-space:nowrap}}
.summary-line{{margin:14px 2px 10px;color:#475467;font-size:13px}} .year-section{{margin-top:12px;background:#fff;border:1px solid var(--line);border-radius:14px;overflow:hidden}}
.year-toggle{{width:100%;border:0;background:#fff;padding:15px 17px;display:flex;justify-content:space-between;align-items:center;cursor:pointer;font-size:15px;font-weight:700;color:var(--ink)}} .year-meta{{color:var(--muted);font-size:12px;font-weight:500}}
.chev{{display:inline-block;transition:transform .18s ease;margin-right:8px}} .year-section.open .chev{{transform:rotate(90deg)}} .year-body{{display:none;border-top:1px solid var(--line)}} .year-section.open .year-body{{display:block}}
.table-wrap{{overflow:auto;max-height:620px}} table{{width:100%;border-collapse:collapse;min-width:1250px;font-size:12px}} thead th{{position:sticky;top:0;z-index:2;background:#F9FAFB;color:#475467;text-align:left;padding:10px 9px;border-bottom:1px solid var(--line);white-space:nowrap}} tbody td{{padding:10px 9px;border-bottom:1px solid #F0F2F5;vertical-align:top}} tbody tr:hover{{background:#FCFCFD}}
.money{{text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}} .nowrap{{white-space:nowrap}} .desc{{width:420px;min-width:340px;max-width:480px;line-height:1.4;white-space:normal;overflow-wrap:anywhere}} .supplier-accent{{color:#6B00B6;font-weight:700}}
.badge{{display:inline-block;border-radius:999px;padding:3px 7px;font-size:11px;font-weight:700;white-space:nowrap}} .badge-large{{background:#ECFDF3;color:var(--green)}} .badge-long{{background:#FFFAEB;color:var(--amber)}} .empty{{background:#fff;border:1px dashed #D0D5DD;border-radius:14px;padding:36px;text-align:center;color:var(--muted);margin-top:16px}} .method{{margin-top:16px;color:var(--muted);font-size:12px;line-height:1.5}}
@media(max-width:1200px){{.kpis{{grid-template-columns:repeat(3,1fr)}}.toolbar{{grid-template-columns:repeat(2,minmax(0,1fr))}}.insight-grid{{grid-template-columns:1fr}}}} @media(max-width:650px){{.wrap{{padding:15px}}.kpis{{grid-template-columns:1fr 1fr}}.toolbar{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<div class="wrap">
<h1>Health TAM Exploratory Analysis</h1>
<p class="subtitle">Explore Accenture-addressable Health procurement by agency, supplier, service offering and contract.</p>
<section class="executive-summary"><h2>Executive summary</h2><p>This dashboard provides an interactive view of the Accenture-addressable Health market. Select an agency to understand where TAM is concentrated, which suppliers are winning addressable work, and which individual contracts underpin that spend.</p></section>

<section class="insight-grid">
<div class="leader-card"><h3>Largest active contracts</h3><p id="activeSubtitle">Highest-value addressable contracts active as at today within the selected agency.</p><div id="topActiveContracts"></div></div>
<div class="leader-card"><h3>Major contracts ending in the next 12 months</h3><p id="endingSubtitle">Highest-value addressable active contracts approaching their recorded end date.</p><div id="topEndingSoon"></div></div>
<div class="leader-card"><h3>Recently completed major contracts</h3><p id="completedSubtitle">Highest-value addressable contracts that ended within the past 12 months.</p><div id="topRecentlyCompleted"></div></div>
</section>

<section class="kpis">
<div class="kpi"><div class="kpi-label">Selected TAM</div><div class="kpi-value" id="kpiTotal">-</div><div class="kpi-sub">{value_basis}</div></div>
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
<div class="field"><label for="minValueInput">Minimum value</label><input id="minValueInput" type="number" min="0" step="100000" placeholder="e.g. 1000000"></div>
<label class="check-wrap"><input id="datedOnly" type="checkbox"> Dated contracts only</label>
<button id="clearFiltersBtn" class="clear-btn" type="button">Clear all filters</button>
</section>

<div class="summary-line" id="summaryLine"></div><div id="yearContainer"></div>
<p class="method">The dashboard is scoped to contracts classified as Accenture-addressable within the Health portfolio. Ongoing contracts are shown once in the top bracket; remaining contracts are grouped by their recorded financial year. Supplier and Service Offering filters are ranked by addressable spend within the current agency scope.</p>
</div>

<script>
const DATA = {json.dumps(payload, ensure_ascii=False)};
const ALL_AGENCIES = "All Health agencies";
const DEFAULT_AGENCY = ALL_AGENCIES;
const VALUE_BASIS = {json.dumps(value_basis)};
const AS_AT_DATE = new Date().toISOString().slice(0,10);
const YEAR_ROWS = new Map();

function esc(v){{return String(v??'').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;').replaceAll("'",'&#039;')}}
function money(v){{v=Number(v||0);const s=v<0?'-':'';const a=Math.abs(v);if(a>=1e9)return s+'$'+(a/1e9).toFixed(2)+'B';if(a>=1e8)return s+'$'+(a/1e6).toFixed(0)+'M';if(a>=1e7)return s+'$'+(a/1e6).toFixed(1)+'M';if(a>=1e6)return s+'$'+(a/1e6).toFixed(2)+'M';if(a>=1e3)return s+'$'+(a/1e3).toFixed(0)+'K';return s+'$'+a.toFixed(0)}}
function number(v){{return Number(v||0).toLocaleString('en-AU')}}
function duration(days){{if(days===null||days===undefined||!isFinite(Number(days)))return'n/a';const d=Number(days),y=d/365.25;if(y>=2)return y.toFixed(1)+' yrs';if(y>=1)return y.toFixed(2)+' yrs';return Math.round(d)+' days'}}
function median(values){{const a=values.filter(v=>Number.isFinite(v)).sort((x,y)=>x-y);if(!a.length)return null;const m=Math.floor(a.length/2);return a.length%2?a[m]:(a[m-1]+a[m])/2}}

function agencyRows(){{const agency=document.getElementById('agencySelect').value;return DATA.records.filter(r=>agency===ALL_AGENCIES||r.agency===agency)}}
function scopeRows(){{return agencyRows()}}
function rowsForOptions(exclude){{const search=document.getElementById('searchInput').value.trim().toLowerCase();const cap=document.getElementById('capabilityFilter').value;const sup=document.getElementById('supplierFilter').value;const min=Number(document.getElementById('minValueInput').value||0);const dated=document.getElementById('datedOnly').checked;return scopeRows().filter(r=>{{const hay=[r.description,r.supplier,r.contract_id,r.category,r.procurement_method,r.capability,r.agency].join(' ').toLowerCase();return(!search||hay.includes(search))&&(exclude==='capability'||cap==='All Service Offerings'||r.capability===cap)&&(exclude==='supplier'||sup==='All suppliers'||r.supplier===sup)&&Number(r.value||0)>=min&&(!dated||(r.start_date&&r.end_date))}})}}
function setRanked(selectId,field,allLabel,exclude){{const select=document.getElementById(selectId);const prev=select.value||allLabel;const totals=new Map();rowsForOptions(exclude).forEach(r=>totals.set(r[field],(totals.get(r[field])||0)+Number(r.value||0)));const vals=[...totals.keys()].sort((a,b)=>(totals.get(b)-totals.get(a))||String(a).localeCompare(String(b)));select.innerHTML=[allLabel,...vals].map(v=>'<option value="'+esc(v)+'">'+esc(v===allLabel?v:v+' — '+money(totals.get(v)))+'</option>').join('');select.value=[...select.options].some(o=>o.value===prev)?prev:allLabel}}
function refreshFilters(changed=null){{if(changed!=='capability')setRanked('capabilityFilter','capability','All Service Offerings','capability');if(changed!=='supplier')setRanked('supplierFilter','supplier','All suppliers','supplier')}}
function filteredRows(){{const search=document.getElementById('searchInput').value.trim().toLowerCase();const cap=document.getElementById('capabilityFilter').value;const sup=document.getElementById('supplierFilter').value;const min=Number(document.getElementById('minValueInput').value||0);const dated=document.getElementById('datedOnly').checked;return scopeRows().filter(r=>{{const hay=[r.description,r.supplier,r.contract_id,r.category,r.procurement_method].join(' ').toLowerCase();return(!search||hay.includes(search))&&(cap==='All Service Offerings'||r.capability===cap)&&(sup==='All suppliers'||r.supplier===sup)&&Number(r.value||0)>=min&&(!dated||(r.start_date&&r.end_date))}})}}

function renderContractRows(id,rows,dateLabel){{const el=document.getElementById(id);if(!rows.length){{el.innerHTML='<div class="leader-detail">No matching contracts.</div>';return}}el.innerHTML=rows.map((r,i)=>{{const title=(r.description||r.contract_id||'Contract');const supplier=r.supplier||'Unknown supplier';const agency=r.agency||'Unknown agency';const date=r.end_date||'n/a';return '<div class="leader-row"><div class="leader-rank">'+(i+1)+'</div><div class="leader-name">'+esc(supplier)+'<div class="leader-detail">'+esc(title)+'</div><div class="leader-detail">'+esc(agency)+' · '+dateLabel+' '+esc(date)+'</div></div><div class="leader-value">'+esc(money(r.value))+'</div></div>'}}).join('')}}
function updateContractInsights(){{const agency=document.getElementById('agencySelect').value;const rows=(DATA.records||[]).filter(r=>agency===ALL_AGENCIES||r.agency===agency);const today=new Date(AS_AT_DATE+'T00:00:00Z');const in12=new Date(today);in12.setUTCFullYear(in12.getUTCFullYear()+1);const ago12=new Date(today);ago12.setUTCFullYear(ago12.getUTCFullYear()-1);const parse=d=>d?new Date(d+'T00:00:00Z'):null;const active=rows.filter(r=>{{const s=parse(r.start_date_iso),e=parse(r.end_date_iso);return s&&e&&s<=today&&e>=today}}).sort((a,b)=>Number(b.value||0)-Number(a.value||0)).slice(0,5);const ending=rows.filter(r=>{{const s=parse(r.start_date_iso),e=parse(r.end_date_iso);return s&&e&&s<=today&&e>=today&&e<=in12}}).sort((a,b)=>Number(b.value||0)-Number(a.value||0)).slice(0,5);const completed=rows.filter(r=>{{const e=parse(r.end_date_iso);return e&&e<today&&e>=ago12}}).sort((a,b)=>Number(b.value||0)-Number(a.value||0)).slice(0,5);const scopeLabel=agency===ALL_AGENCIES?'all Health agencies':agency;document.getElementById('activeSubtitle').textContent='Highest-value addressable contracts active as at today across '+scopeLabel+'.';document.getElementById('endingSubtitle').textContent='Highest-value addressable active contracts ending within the next 12 months across '+scopeLabel+'.';document.getElementById('completedSubtitle').textContent='Highest-value addressable contracts completed within the past 12 months across '+scopeLabel+'.';renderContractRows('topActiveContracts',active,'Ends');renderContractRows('topEndingSoon',ending,'Ends');renderContractRows('topRecentlyCompleted',completed,'Ended')}}
function updateKpis(rows){{const vals=rows.map(r=>Number(r.value||0)),dur=rows.map(r=>Number(r.duration_days)).filter(Number.isFinite),total=vals.reduce((a,b)=>a+b,0),largest=rows.slice().sort((a,b)=>Number(b.value||0)-Number(a.value||0))[0];document.getElementById('kpiTotal').textContent=money(total);document.getElementById('kpiContracts').textContent=number(rows.length);document.getElementById('kpiAvgValue').textContent=rows.length?money(total/rows.length):'n/a';document.getElementById('kpiMedianValue').textContent=rows.length?'Median: '+money(median(vals)):'Median: n/a';document.getElementById('kpiAvgLength').textContent=dur.length?duration(dur.reduce((a,b)=>a+b,0)/dur.length):'n/a';document.getElementById('kpiMedianLength').textContent=dur.length?'Median: '+duration(median(dur)):'Median: n/a';document.getElementById('kpiSuppliers').textContent=number(new Set(rows.map(r=>r.supplier)).size);document.getElementById('kpiLargest').textContent=largest?money(largest.value):'n/a';document.getElementById('kpiLargestSub').textContent=largest?largest.supplier:'No matching contract'}}
function rowHtml(r){{const large=Number(r.value||0)>=20000000?'<span class="badge badge-large">Large</span>':'',long=Number(r.duration_days||0)>=1826?'<span class="badge badge-long">5+ years</span>':'',sc=String(r.supplier||'').toLowerCase().includes('accenture')?'supplier-accent':'';return`<tr><td class="nowrap">${{esc(r.contract_id)}}</td><td class="desc">${{esc(r.description)}}</td><td>${{esc(r.agency)}}</td><td>${{esc(r.division)}}</td><td class="${{sc}}">${{esc(r.supplier)}}</td><td>${{esc(r.capability)}}</td><td class="money">${{money(r.value)}} ${{large}}</td><td class="nowrap">${{esc(r.start_date)}}</td><td class="nowrap">${{esc(r.end_date)}}</td><td class="nowrap">${{esc(r.duration_label)}} ${{long}}</td><td>${{esc(r.category)}}</td><td class="nowrap">${{esc(r.extensions)}}</td></tr>`}}
function tableHtml(rows){{return`<div class="table-wrap"><table><thead><tr><th>Contract ID</th><th>Description</th><th>Agency</th><th>Division</th><th>Supplier</th><th>Service Offering</th><th class="money">Contract value</th><th>Start date</th><th>End date</th><th>Contract length</th><th>Category</th><th>Extensions</th></tr></thead><tbody>${{rows.map(rowHtml).join('')}}</tbody></table></div>`}}
function bracket(id,title,rows,label='contracts'){{const spend=rows.reduce((s,r)=>s+Number(r.value||0),0);return`<section class="year-section" id="${{id}}"><button class="year-toggle" type="button" onclick="toggleYear('${{id}}')"><span><span class="chev">▶</span>${{esc(title)}}</span><span class="year-meta">${{number(rows.length)}} ${{label}} · ${{money(spend)}}</span></button><div class="year-body" data-loaded="false"></div></section>`}}
function toggleYear(id){{const s=document.getElementById(id),b=s.querySelector('.year-body'),open=!s.classList.contains('open');if(open&&b.dataset.loaded!=='true'){{b.innerHTML=tableHtml(YEAR_ROWS.get(id)||[]);b.dataset.loaded='true'}}s.classList.toggle('open')}}
function render(){{const rows=filteredRows();updateContractInsights();updateKpis(rows);document.getElementById('summaryLine').textContent=number(rows.length)+' matching contracts · '+money(rows.reduce((s,r)=>s+Number(r.value||0),0))+' · '+VALUE_BASIS;const ongoing=[],historical=[];rows.forEach(r=>{{const active=Boolean(r.start_date_iso&&r.end_date_iso&&r.start_date_iso<=AS_AT_DATE&&r.end_date_iso>=AS_AT_DATE);(active?ongoing:historical).push(r)}});const groups=new Map();historical.forEach(r=>{{const fy=r.financial_year||'Unknown FY';if(!groups.has(fy))groups.set(fy,[]);groups.get(fy).push(r)}});const years=[...groups.keys()].sort((a,b)=>Number(groups.get(b)[0].fy_start??-Infinity)-Number(groups.get(a)[0].fy_start??-Infinity));const c=document.getElementById('yearContainer');if(!rows.length){{c.innerHTML='<div class="empty">No contracts match the selected filters.</div>';return}}YEAR_ROWS.clear();const sections=[];if(ongoing.length){{const rr=ongoing.slice().sort((a,b)=>Number(b.value||0)-Number(a.value||0));YEAR_ROWS.set('ongoing_contracts',rr);sections.push(bracket('ongoing_contracts','Ongoing contracts',rr,'active contracts'))}}years.forEach(y=>{{const rr=groups.get(y).slice().sort((a,b)=>Number(b.value||0)-Number(a.value||0));const id='year_'+String(y).replace(/[^A-Za-z0-9]/g,'_');YEAR_ROWS.set(id,rr);sections.push(bracket(id,y,rr))}});c.innerHTML=sections.join('')}}
function clearAllFilters(){{document.getElementById('agencySelect').value=DEFAULT_AGENCY;document.getElementById('searchInput').value='';document.getElementById('minValueInput').value='';document.getElementById('datedOnly').checked=false;document.getElementById('capabilityFilter').innerHTML='<option value="All Service Offerings">All Service Offerings</option>';document.getElementById('supplierFilter').innerHTML='<option value="All suppliers">All suppliers</option>';refreshFilters();render()}}

document.getElementById('agencySelect').addEventListener('change',()=>{{refreshFilters();render()}});document.getElementById('searchInput').addEventListener('input',()=>{{refreshFilters();render()}});document.getElementById('capabilityFilter').addEventListener('change',()=>{{refreshFilters('capability');render()}});document.getElementById('supplierFilter').addEventListener('change',()=>{{refreshFilters('supplier');render()}});document.getElementById('minValueInput').addEventListener('input',()=>{{refreshFilters();render()}});document.getElementById('datedOnly').addEventListener('change',()=>{{refreshFilters();render()}});document.getElementById('clearFiltersBtn').addEventListener('click',clearAllFilters);
refreshFilters();render();
</script>
</body></html>'''

    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df, value_col = load_data(input_path, args.value_mode)
    payload = build_payload(df)
    html_path = output_dir / "HealthTAMExploratoryAnalysis.html"
    build_html(payload, html_path, args.value_mode)

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


if __name__ == "__main__":
    main()
