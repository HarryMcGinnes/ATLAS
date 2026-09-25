from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

import pandas as pd


VALUE_COL = "Value"
ANNUALISED_VALUE_COL = "Value Per Year"
SUPPLIER_COL = "Supplier Name"
SUPPLIER_GROUP_COL = "supplier_group"
DESCRIPTION_COL = "Description"
CATEGORY_COL = "Category"
FIN_YEAR_COL = "Financial Year"
CN_ID_COL = "CN ID"
AGENCY_COL = "Agency"
CAPABILITY_COL = "capability"
SERVICE_OFFERING_COL = "Service Offering"
ADDRESSABLE_COL = "is_addressable"

START_DATE_CANDIDATES = ["Start Date", "Contract Start Date", "Contract Period Start", "Contract Start", "StartDate"]
END_DATE_CANDIDATES = ["End Date", "Contract End Date", "Contract Period End", "Contract End", "Expiry Date", "EndDate"]

SERVICE_OFFERING_ORDER = [
    "Strategy, Transformation & Advisory",
    "SI & Engineering",
    "Data, AI & Automation",
    "Cloud Infrastructure & Cyber",
    "Managed Services & Operations",
    "Uncategorised",
]

SUPPLIER_COLOUR_MAP = {
    "Accenture": "#A100FF",
    "Deloitte": "#86BC25",
    "KPMG": "#00338D",
    "EY": "#FFE600",
    "PwC": "#E0301E",
    "IBM": "#1F70C1",
    "Fujitsu": "#D6001C",
    "DXC": "#5B2C83",
    "DXC Technology": "#5B2C83",
    "Microsoft": "#737373",
    "Amazon / AWS": "#FF9900",
    "Amazon Web Services": "#FF9900",
    "Oracle": "#C74634",
    "SAP": "#0FAAFF",
    "Telstra": "#008193",
    "Datacom": "#00AEEF",
    "Data#3": "#34495E",
    "Capgemini": "#0070AD",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Health TAM supplier-vs-supplier competitive intelligence dashboard.")
    p.add_argument("--input", default="master_output/health_portfolio_contracts.parquet")
    p.add_argument("--output-dir", default="HealthCompareDashboard_Output")
    p.add_argument("--value-mode", choices=["total", "annualised"], default="total")
    p.add_argument("--as-at", help="Date used for active-contract view, YYYY-MM-DD. Defaults to today.")
    p.add_argument("--top-n-suppliers", type=int, default=30)
    p.add_argument("--top-n-contracts", type=int, default=500)
    return p.parse_args()


def _parse_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.fillna(False).astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def _clean_text(series: pd.Series, fallback: str) -> pd.Series:
    return (
        series.fillna(fallback)
        .astype(str)
        .str.replace(r"_x000D_", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .replace("", fallback)
    )


def first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    exact = {str(c).lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in exact:
            return exact[candidate.lower()]
    return None


def financial_year_start(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    m = re.search(r"(19\d{2}|20\d{2})", str(value))
    return int(m.group(1)) if m else None


def money(value: float) -> str:
    value = float(value or 0)
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


def supplier_colour(name: object) -> str:
    name = str(name or "").strip()
    if name in SUPPLIER_COLOUR_MAP:
        return SUPPLIER_COLOUR_MAP[name]
    lower = name.lower()
    rules = [
        ("accenture", "#A100FF"), ("deloitte", "#86BC25"), ("kpmg", "#00338D"),
        ("ernst", "#FFE600"), ("pwc", "#E0301E"), ("pricewaterhouse", "#E0301E"),
        ("ibm", "#1F70C1"), ("fujitsu", "#D6001C"), ("dxc", "#5B2C83"),
        ("microsoft", "#737373"), ("amazon", "#FF9900"), ("aws", "#FF9900"),
        ("oracle", "#C74634"), ("sap", "#0FAAFF"), ("telstra", "#008193"),
        ("datacom", "#00AEEF"), ("capgemini", "#0070AD"),
    ]
    for token, colour in rules:
        if token in lower:
            return colour
    return "#5B6B7A"


def load_data(path: Path, value_mode: str) -> tuple[pd.DataFrame, str]:
    if not path.exists():
        raise SystemExit(f"Health master dataset not found: {path}")
    if path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path, low_memory=False)
    else:
        raise SystemExit("Input must be parquet or CSV.")

    required = {VALUE_COL, AGENCY_COL, SUPPLIER_GROUP_COL, ADDRESSABLE_COL}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(
            "Health comparison dashboard requires the canonical Health master dataset. Missing: "
            + ", ".join(missing)
            + ". Rebuild it with build_health_filter_v2.py."
        )

    out = df.loc[_parse_bool_series(df[ADDRESSABLE_COL])].copy()
    if out.empty:
        raise SystemExit("No addressable Health contracts found.")

    # Service Offering is authoritative from the Health filter. Addressable rows that
    # remain unresolved are shown as a visible 'Uncategorised' bucket so TAM reconciles.
    if SERVICE_OFFERING_COL in out.columns:
        out[CAPABILITY_COL] = out[SERVICE_OFFERING_COL]
    elif CAPABILITY_COL not in out.columns:
        raise SystemExit("Health master is missing Service Offering/capability classification.")
    out[CAPABILITY_COL] = out[CAPABILITY_COL].fillna("Uncategorised").astype(str).str.strip().replace("", "Uncategorised")

    out[SUPPLIER_GROUP_COL] = _clean_text(out[SUPPLIER_GROUP_COL], "Unknown supplier")
    out[AGENCY_COL] = _clean_text(out[AGENCY_COL], "Unknown agency")
    if FIN_YEAR_COL not in out.columns:
        out[FIN_YEAR_COL] = "Unknown FY"
    out[FIN_YEAR_COL] = _clean_text(out[FIN_YEAR_COL], "Unknown FY")
    out["_fy_start"] = out[FIN_YEAR_COL].map(financial_year_start)

    value_col = ANNUALISED_VALUE_COL if value_mode == "annualised" else VALUE_COL
    if value_col not in out.columns:
        raise SystemExit(f"Requested value column not found: {value_col}")
    out["_value"] = pd.to_numeric(out[value_col], errors="coerce").fillna(0).clip(lower=0)
    out["_total_value"] = pd.to_numeric(out[VALUE_COL], errors="coerce").fillna(0).clip(lower=0)

    start_col = first_existing(out, START_DATE_CANDIDATES)
    end_col = first_existing(out, END_DATE_CANDIDATES)
    out["_start"] = pd.to_datetime(out[start_col], errors="coerce", dayfirst=True) if start_col else pd.Series(pd.NaT, index=out.index)
    out["_end"] = pd.to_datetime(out[end_col], errors="coerce", dayfirst=True) if end_col else pd.Series(pd.NaT, index=out.index)

    print(f"Using canonical Health TAM dataset: {path}")
    print(f"Addressable Health rows loaded: {len(out):,}")
    print(f"Health TAM in selected value basis: {money(out['_value'].sum())}")
    return out, value_col


def build_model(df: pd.DataFrame, top_n_suppliers: int, as_at: pd.Timestamp) -> dict[str, object]:
    total_tam = float(df["_value"].sum())

    supplier_total = (
        df.groupby(SUPPLIER_GROUP_COL, dropna=False)["_value"].sum()
        .rename("supplier_value").reset_index().sort_values("supplier_value", ascending=False)
    )
    supplier_total["tam_share_pct"] = supplier_total["supplier_value"] / total_tam * 100 if total_tam else 0
    supplier_total["rank_in_tam"] = supplier_total["supplier_value"].rank(method="min", ascending=False).astype(int)

    top_suppliers = supplier_total.head(max(1, top_n_suppliers))[SUPPLIER_GROUP_COL].astype(str).tolist()
    if "Accenture" in set(supplier_total[SUPPLIER_GROUP_COL].astype(str)):
        supplier_options = ["Accenture"] + [s for s in top_suppliers if s != "Accenture"]
    else:
        supplier_options = top_suppliers

    cap_total = (
        df.groupby(CAPABILITY_COL, dropna=False)["_value"].sum()
        .rename("capability_market").reset_index()
    )
    cap_order = cap_total.sort_values("capability_market", ascending=False)[CAPABILITY_COL].astype(str).tolist()

    supplier_cap = (
        df.groupby([SUPPLIER_GROUP_COL, CAPABILITY_COL], dropna=False)
        .agg(
            supplier_capability_value=("_value", "sum"),
            contracts=(CN_ID_COL, "nunique") if CN_ID_COL in df.columns else ("_value", "size"),
        )
        .reset_index().merge(cap_total, on=CAPABILITY_COL, how="left")
    )
    supplier_cap["capability_share_pct"] = supplier_cap["supplier_capability_value"] / supplier_cap["capability_market"].replace(0, pd.NA) * 100
    supplier_cap["capability_rank"] = supplier_cap.groupby(CAPABILITY_COL)["supplier_capability_value"].rank(method="min", ascending=False).astype(int)

    yearly = (
        df.groupby([SUPPLIER_GROUP_COL, CAPABILITY_COL, FIN_YEAR_COL], dropna=False)
        .agg(value=("_value", "sum"), contracts=(CN_ID_COL, "nunique") if CN_ID_COL in df.columns else ("_value", "size"))
        .reset_index()
    )
    cap_year_market = (
        df.groupby([CAPABILITY_COL, FIN_YEAR_COL], dropna=False)["_value"].sum()
        .rename("capability_year_market").reset_index()
    )
    yearly = yearly.merge(cap_year_market, on=[CAPABILITY_COL, FIN_YEAR_COL], how="left")
    yearly["share_pct"] = yearly["value"] / yearly["capability_year_market"].replace(0, pd.NA) * 100
    yearly["_fy_start"] = yearly[FIN_YEAR_COL].map(financial_year_start)

    agency_yearly = (
        df.loc[df[SUPPLIER_GROUP_COL].isin(supplier_options)]
        .groupby([SUPPLIER_GROUP_COL, AGENCY_COL, FIN_YEAR_COL], dropna=False)
        .agg(value=("_value", "sum"), contracts=(CN_ID_COL, "nunique") if CN_ID_COL in df.columns else ("_value", "size"))
        .reset_index()
    )
    agency_yearly["_fy_start"] = agency_yearly[FIN_YEAR_COL].map(financial_year_start)

    # Active contracts for the first selected supplier's timeline. Unknown end dates
    # are not treated as active because the timeline needs a bounded interval.
    active = df[
        df["_start"].notna() & df["_end"].notna() &
        (df["_start"] <= as_at) & (df["_end"] >= as_at)
    ].copy()
    active = active[active[SUPPLIER_GROUP_COL].isin(supplier_options)].copy()
    active["_start_display"] = active["_start"].dt.strftime("%Y-%m-%d")
    active["_end_display"] = active["_end"].dt.strftime("%Y-%m-%d")
    active = active.sort_values([SUPPLIER_GROUP_COL, "_value"], ascending=[True, False])

    contract_cols = [c for c in [
        SUPPLIER_GROUP_COL, CN_ID_COL, DESCRIPTION_COL, CAPABILITY_COL, AGENCY_COL,
        FIN_YEAR_COL, "_start_display", "_end_display", "_value", "_total_value", CATEGORY_COL,
    ] if c in active.columns]
    active_contracts = active[contract_cols].copy()

    return {
        "total_tam": total_tam,
        "supplier_total": supplier_total,
        "supplier_capability": supplier_cap,
        "yearly": yearly,
        "agency_yearly": agency_yearly,
        "active_contracts": active_contracts,
        "supplier_options": supplier_options,
        "capabilities": cap_order,
    }


def records(df: pd.DataFrame, cols: list[str]) -> list[dict[str, object]]:
    existing = [c for c in cols if c in df.columns]
    frame = df[existing].copy().astype(object).where(pd.notna(df[existing]), None)
    return frame.to_dict(orient="records")


def build_html(model: dict[str, object], output_path: Path, as_at: pd.Timestamp, value_mode: str) -> None:
    supplier_total: pd.DataFrame = model["supplier_total"]  # type: ignore[assignment]
    supplier_cap: pd.DataFrame = model["supplier_capability"]  # type: ignore[assignment]
    yearly: pd.DataFrame = model["yearly"]  # type: ignore[assignment]
    agency_yearly: pd.DataFrame = model["agency_yearly"]  # type: ignore[assignment]
    active_contracts: pd.DataFrame = model["active_contracts"]  # type: ignore[assignment]
    supplier_options: list[str] = model["supplier_options"]  # type: ignore[assignment]
    capabilities: list[str] = model["capabilities"]  # type: ignore[assignment]

    if not supplier_options:
        raise SystemExit("No suppliers available for comparison.")

    default_supplier = "Accenture" if "Accenture" in supplier_options else supplier_options[0]
    default_compare = next((s for s in supplier_options if s != default_supplier), default_supplier)
    options_html = "\n".join(f'<option value="{html.escape(s)}">{html.escape(s)}</option>' for s in supplier_options)

    supplier_payload = records(supplier_total, [SUPPLIER_GROUP_COL, "supplier_value", "tam_share_pct", "rank_in_tam"])
    cap_payload = records(supplier_cap, [SUPPLIER_GROUP_COL, CAPABILITY_COL, "supplier_capability_value", "capability_market", "capability_share_pct", "capability_rank", "contracts"])
    yearly_payload = records(yearly, [SUPPLIER_GROUP_COL, CAPABILITY_COL, FIN_YEAR_COL, "value", "contracts", "capability_year_market", "share_pct", "_fy_start"])
    agency_payload = records(agency_yearly, [SUPPLIER_GROUP_COL, AGENCY_COL, FIN_YEAR_COL, "value", "contracts", "_fy_start"])
    contract_payload = records(active_contracts, [SUPPLIER_GROUP_COL, CN_ID_COL, DESCRIPTION_COL, CAPABILITY_COL, AGENCY_COL, FIN_YEAR_COL, "_start_display", "_end_display", "_value", "_total_value", CATEGORY_COL])
    colour_payload = {s: supplier_colour(s) for s in supplier_options}

    html_doc = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Competitive Intelligence inside Accenture Health TAM</title>
<script src="https://cdn.plot.ly/plotly-3.5.0.min.js"></script>
<style>
body{{font-family:Arial,Helvetica,sans-serif;margin:0;background:#f6f8fb;color:#1f2937}}
.wrap{{max-width:1480px;margin:0 auto;padding:28px}}
h1{{margin:0 0 8px;font-size:30px}}
.subtitle{{color:#526070;font-size:14px;margin-bottom:18px}}
.executive,.insight,.panel,.supplier-kpi-block{{background:#fff;border:1px solid #e5e7eb;border-radius:14px;box-shadow:0 1px 2px rgba(0,0,0,.04)}}
.executive,.insight{{border-left:5px solid #A100FF;padding:16px 18px;margin:16px 0}}
.executive h2,.insight h2{{margin:0 0 7px;font-size:19px}}
.executive p,.insight p{{margin:0;color:#475467;font-size:13px;line-height:1.5}}
.questions{{margin:8px 0 0;padding-left:20px;color:#344054;font-size:13px;font-weight:600;line-height:1.5}}
.selectors{{display:grid;grid-template-columns:1fr 1fr .72fr;gap:10px;margin:14px 0}}
.selector-card{{background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:10px 12px}}
.selector-card label{{display:block;font-size:10px;text-transform:uppercase;letter-spacing:.04em;color:#667085;margin-bottom:4px}}
select{{width:100%;height:34px;border:1px solid #d0d5dd;border-radius:8px;background:#fff;padding:0 10px;font-size:13px}}
.cards{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:18px 0}}
.supplier-kpi-block{{padding:14px;border-top:4px solid #A100FF}}
.supplier-kpi-title{{font-size:13px;text-transform:uppercase;letter-spacing:.04em;color:#344054;margin-bottom:10px;padding-left:9px;border-left:5px solid #A100FF}}
.supplier-kpi-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}}
.card{{border:1px solid #e5e7eb;border-radius:12px;padding:14px;min-width:0}}
.card-title{{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#667085;margin-bottom:7px}}
.card-value{{font-size:24px;font-weight:700;color:#111827;line-height:1.15;overflow-wrap:anywhere}}
.card-subtitle{{font-size:12px;color:#667085;margin-top:5px;line-height:1.35}}
.card.kpi-win{{background:#ecfdf3;border-color:#6ce9a6}}
.card.kpi-loss{{background:#fef3f2;border-color:#fda29b}}
.card.kpi-tie{{background:#f8fafc;border-color:#d0d5dd}}
.grid{{display:grid;grid-template-columns:1fr;gap:18px}}
.panel{{padding:12px}}
.chart-heading{{padding:10px 10px 0}}
.chart-title{{margin:0;font-size:21px}}
.chart-subtitle{{margin:5px 0 0;color:#526070;font-size:13px}}
.leaderboard-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;padding:16px 10px 10px}}
.leaderboard-card{{border:1px solid #e5e7eb;border-radius:12px;padding:14px}}
.leaderboard-card h4{{margin:0 0 4px;font-size:16px}}
.leaderboard-market{{font-size:12px;color:#667085;margin-bottom:10px}}
.leader-row{{display:grid;grid-template-columns:28px minmax(120px,1fr) minmax(110px,2fr) 58px;gap:8px;align-items:center;padding:6px 0;border-top:1px solid #f1f5f9;font-size:12px}}
.leader-row.focus-selected{{background:#faf5ff;margin:0 -7px;padding-left:7px;padding-right:7px;border-radius:8px}}
.leader-row.focus-compare{{outline:1px solid #d0d5dd;margin:0 -7px;padding-left:7px;padding-right:7px;border-radius:8px}}
.leader-bar-track{{height:12px;background:#eef2f7;border-radius:999px;overflow:hidden}}
.leader-bar-fill{{height:100%;border-radius:999px;min-width:2px}}
.leader-share{{font-weight:700;text-align:right}}
.leaderboard-card{{overflow:visible}}
.leader-row{{position:relative;cursor:default}}
.leader-row.leaderboard-hover:hover{{background:#f8fafc}}
.leader-row.focus-selected.leaderboard-hover:hover{{background:#f3e8ff}}
.leader-tooltip{{
  display:none;position:absolute;z-index:50;left:34%;top:calc(100% + 7px);
  min-width:230px;max-width:310px;padding:11px 13px;background:#111827;color:#fff;
  border-radius:9px;box-shadow:0 8px 24px rgba(15,23,42,.22);font-size:12px;
  line-height:1.55;text-align:left;pointer-events:none
}}
.leader-tooltip::before{{
  content:'';position:absolute;top:-6px;left:22px;width:12px;height:12px;
  background:#111827;transform:rotate(45deg)
}}
.leader-tooltip-rule{{height:1px;background:#475467;margin:7px 0}}
.leader-row.leaderboard-hover:hover .leader-tooltip,
.leader-row.leaderboard-hover:focus .leader-tooltip{{display:block}}
.leader-gap-row{{
  display:grid;
  grid-template-columns:28px minmax(120px,1fr) minmax(110px,2fr) 58px;
  gap:8px;align-items:center;
  min-height:24px;
  margin:2px -7px;
  padding:0 7px;
  border-radius:7px;
  background:#faf5ff;
  color:#98A2B3;
  font-weight:700;
  text-align:center;
}}
.leader-gap-row .leader-gap-dots{{grid-column:1 / -1;letter-spacing:.22em}}
.note{{color:#667085;font-size:12px;margin-top:14px;line-height:1.4}}
@media(max-width:1000px){{.selectors,.cards,.leaderboard-grid{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<div class="wrap">
<h1>Competitive Intelligence inside Accenture Health TAM</h1>
<div class="subtitle">Supplier-vs-supplier positioning across the addressable Health market.</div>

<section class="executive">
<h2>Executive purpose</h2>
<p>This dashboard compares two suppliers inside Accenture's addressable Health market. It brings together market position, Service Offering strength, current contracts and agency presence so users can understand where each supplier is established and where competitive gaps or contract opportunities may exist.</p>
<ul class="questions">
<li>How large is each supplier's position inside Health TAM?</li>
<li>Where does each supplier lead by Service Offering?</li>
<li>Which current contracts underpin that position?</li>
<li>Which Health agencies are most important to the selected supplier?</li>
</ul>
</section>

<div class="selectors">
<div class="selector-card"><label>Selected supplier</label><select id="supplierSelect">{options_html}</select></div>
<div class="selector-card"><label>Compare against</label><select id="compareSelect">{options_html}</select></div>
<div class="selector-card"><label>Financial-year period</label><select id="periodSelect"></select></div>
</div>

<section class="insight"><h2>Start with the headline comparison</h2><p>The KPI cards show relative scale, TAM share, strongest Service Offering and breadth across the addressable Health market.</p></section>
<div class="cards">
<div class="supplier-kpi-block" id="selectedBlock"><div class="supplier-kpi-title" id="selectedTitle"></div><div class="supplier-kpi-grid">
<div class="card" id="selectedCardValue"><div class="card-title">Period value</div><div class="card-value" id="selectedValue"></div><div class="card-subtitle" id="selectedValueSub"></div></div>
<div class="card" id="selectedCardShare"><div class="card-title">Addressable TAM share</div><div class="card-value" id="selectedShare"></div><div class="card-subtitle" id="selectedRank"></div></div>
<div class="card" id="selectedCardLargest"><div class="card-title">Largest Service Offering</div><div class="card-value" id="selectedLargest"></div><div class="card-subtitle" id="selectedLargestSub"></div></div>
<div class="card" id="selectedCardBreadth"><div class="card-title">Service Offering presence</div><div class="card-value" id="selectedBreadth"></div><div class="card-subtitle" id="selectedBreadthSub"></div></div>
</div></div>
<div class="supplier-kpi-block" id="compareBlock"><div class="supplier-kpi-title" id="compareTitle"></div><div class="supplier-kpi-grid">
<div class="card" id="compareCardValue"><div class="card-title">Period value</div><div class="card-value" id="compareValue"></div><div class="card-subtitle" id="compareValueSub"></div></div>
<div class="card" id="compareCardShare"><div class="card-title">Addressable TAM share</div><div class="card-value" id="compareShare"></div><div class="card-subtitle" id="compareRank"></div></div>
<div class="card" id="compareCardLargest"><div class="card-title">Largest Service Offering</div><div class="card-value" id="compareLargest"></div><div class="card-subtitle" id="compareLargestSub"></div></div>
<div class="card" id="compareCardBreadth"><div class="card-title">Service Offering presence</div><div class="card-value" id="compareBreadth"></div><div class="card-subtitle" id="compareBreadthSub"></div></div>
</div></div>
</div>

<div class="grid">
<section class="insight"><h2>Compare where each supplier is strongest</h2><p>The Service Offering comparison shows the scale of each supplier's position across Health TAM, including unresolved addressable work as <b>Uncategorised</b>.</p></section>
<div class="panel"><div class="chart-heading"><h3 class="chart-title" id="profileTitle"></h3><p class="chart-subtitle">Award value by Service Offering inside the selected Health TAM period.</p></div><div id="profileChart" style="height:650px"></div></div>

<section class="insight"><h2>See the wider competitive field</h2><p>Each Service Offering leaderboard ranks suppliers independently within the selected period. The top five are shown, with both selected suppliers included even when they sit outside the top five.</p></section>
<div class="panel"><div class="chart-heading"><h3 class="chart-title">Service Offering Supplier Leaderboards</h3><p class="chart-subtitle">Market share is calculated against the addressable value of each Service Offering.</p></div><div id="leaderboards" class="leaderboard-grid"></div></div>

<section class="insight"><h2>Understand the current contracts behind the position</h2><p>The current-contract timeline is controlled by the first supplier selector and shows active addressable Health contracts with valid start and end dates.</p></section>
<div class="panel"><div class="chart-heading"><h3 class="chart-title" id="timelineTitle"></h3><p class="chart-subtitle" id="timelineSubtitle"></p></div><div id="timelineChart"></div></div>

<section class="insight"><h2>Read the supplier's Health footprint</h2><p>The agency view replaces the Defence-domain logic. It shows where the selected supplier's addressable Health position is concentrated across the Health agencies in scope.</p></section>
<div class="panel"><div class="chart-heading"><h3 class="chart-title" id="agencyTitle"></h3><p class="chart-subtitle">Selected-period award value by Health agency.</p></div><div id="agencyChart" style="height:500px"></div></div>
</div>

<p class="note">Note: this is not whole-of-Health procurement share. It is supplier positioning inside Health contracts classified as addressable to Accenture. Service Offering and addressability are read from the canonical Health master dataset and are not reclassified in this dashboard.</p>
</div>
<script>
const SUPPLIER_TOTAL={json.dumps(supplier_payload)};
const SUPPLIER_CAP={json.dumps(cap_payload)};
const YEARLY={json.dumps(yearly_payload)};
const AGENCY_YEARLY={json.dumps(agency_payload)};
const CONTRACTS={json.dumps(contract_payload)};
const SUPPLIER_COLOURS={json.dumps(colour_payload)};
const CAPABILITIES={json.dumps(capabilities)};
const DEFAULT_SUPPLIER={json.dumps(default_supplier)};
const DEFAULT_COMPARE={json.dumps(default_compare)};
const AS_AT_DATE={json.dumps(as_at.strftime('%Y-%m-%d'))};
const VALUE_MODE={json.dumps(value_mode)};

function money(v){{v=Number(v||0);if(v>=1e9)return '$'+(v/1e9).toFixed(2)+'B';if(v>=1e8)return '$'+(v/1e6).toFixed(0)+'M';if(v>=1e7)return '$'+(v/1e6).toFixed(1)+'M';if(v>=1e6)return '$'+(v/1e6).toFixed(2)+'M';if(v>=1e3)return '$'+(v/1e3).toFixed(0)+'K';return '$'+v.toFixed(0)}}
function pct(v){{return Number(v||0).toFixed(1)+'%'}}
function esc(v){{return String(v==null?'':v).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;')}}
function colour(name){{return SUPPLIER_COLOURS[name]||'#5B6B7A'}}
function parseDate(v){{if(!v)return null;const d=new Date(v+'T00:00:00');return Number.isNaN(d.getTime())?null:d}}
function dateLabel(v){{const d=parseDate(v);return d?d.toLocaleDateString('en-AU',{{day:'2-digit',month:'short',year:'numeric'}}):''}}

function availableYears(){{return Array.from(new Map(YEARLY.filter(r=>r._fy_start!=null).map(r=>[String(r['Financial Year']),Number(r._fy_start)])).entries()).sort((a,b)=>a[1]-b[1]).map(x=>x[0])}}
function buildPeriods(){{const y=availableYears();const first=y.length?String(y[0]).match(/(?:19|20)\\d{{2}}/):null;const last=y.length?String(y[y.length-1]).match(/(?:19|20)\\d{{2}}/g):null;const label=first&&last?'All FY ('+first[0]+'-'+last[last.length-1]+')':'All FY';const p=[{{key:'all',label:label,years:y}}];if(y.length>=5)p.push({{key:'last5',label:'Last 5 FY',years:y.slice(-5)}});if(y.length>=3)p.push({{key:'last3',label:'Last 3 FY',years:y.slice(-3)}});window.PERIODS=p;document.getElementById('periodSelect').innerHTML=p.map((r,i)=>'<option value="'+i+'">'+r.label+'</option>').join('')}}
function period(){{return window.PERIODS[Number(document.getElementById('periodSelect').value||0)]||window.PERIODS[0]}}
function periodRows(){{const p=period();const ys=new Set(p.years);return p.key==='all'?YEARLY:YEARLY.filter(r=>ys.has(String(r['Financial Year']||'')))}}
function periodAgencyRows(){{const p=period();const ys=new Set(p.years);return p.key==='all'?AGENCY_YEARLY:AGENCY_YEARLY.filter(r=>ys.has(String(r['Financial Year']||'')))}}

function summaryFor(supplier){{
 const rows=periodRows();const supplierTotals=new Map();const supplierCaps=new Map();const marketKeys=new Map();
 rows.forEach(r=>{{const s=String(r.supplier_group||'Unknown');const cap=String(r.capability||'Uncategorised');const fy=String(r['Financial Year']||'');supplierTotals.set(s,(supplierTotals.get(s)||0)+Number(r.value||0));const k=s+'||'+cap;supplierCaps.set(k,(supplierCaps.get(k)||0)+Number(r.value||0));const mk=cap+'||'+fy;if(!marketKeys.has(mk))marketKeys.set(mk,Number(r.capability_year_market||0))}});
 const totalTam=Array.from(marketKeys.values()).reduce((a,b)=>a+b,0);const awards=Number(supplierTotals.get(supplier)||0);const ranked=Array.from(supplierTotals.entries()).sort((a,b)=>b[1]-a[1]);const rank=ranked.findIndex(x=>x[0]===supplier)+1;
 const caps=Array.from(supplierCaps.entries()).filter(([k,v])=>k.startsWith(supplier+'||')&&Number(v)>0).map(([k,v])=>[k.split('||')[1],Number(v)]).sort((a,b)=>b[1]-a[1]);
 return {{supplier,awards,tam_share_pct:totalTam?awards/totalTam*100:0,rank:rank>0?rank:0,largest:caps.length?caps[0][0]:'n/a',largestValue:caps.length?caps[0][1]:0,breadth:caps.length,totalTam}};
}}
function rowsForSupplier(supplier){{
 const rows=periodRows();const vals=new Map();const markets=new Map();
 rows.forEach(r=>{{const cap=String(r.capability||'Uncategorised');const fy=String(r['Financial Year']||'');const mk=cap+'||'+fy;if(!markets.has(mk))markets.set(mk,Number(r.capability_year_market||0));if(r.supplier_group===supplier)vals.set(cap,(vals.get(cap)||0)+Number(r.value||0))}});
 const marketByCap=new Map();markets.forEach((v,k)=>{{const cap=k.split('||')[0];marketByCap.set(cap,(marketByCap.get(cap)||0)+v)}});
 return Array.from(new Set([...marketByCap.keys(),...vals.keys()])).map(cap=>{{const v=Number(vals.get(cap)||0),m=Number(marketByCap.get(cap)||0);return{{capability:cap,value:v,market:m,share:m?v/m*100:0}}}})
}}
function setKpis(prefix,supplier){{const s=summaryFor(supplier),p=period();document.getElementById(prefix+'Title').textContent=(prefix==='selected'?'Selected supplier: ':'Comparison supplier: ')+supplier;document.getElementById(prefix+'Value').textContent=money(s.awards);document.getElementById(prefix+'ValueSub').textContent='Award value inside '+p.label+' Health TAM';document.getElementById(prefix+'Share').textContent=pct(s.tam_share_pct);document.getElementById(prefix+'Rank').textContent=s.rank?'Rank #'+s.rank+' inside '+p.label+' TAM':'No awards in selected period';document.getElementById(prefix+'Largest').textContent=s.largest;document.getElementById(prefix+'LargestSub').textContent=money(s.largestValue)+' in '+p.label;document.getElementById(prefix+'Breadth').textContent=s.breadth+' Service Offering'+(s.breadth===1?'':'s');document.getElementById(prefix+'BreadthSub').textContent='Presence in '+s.breadth+' of '+CAPABILITIES.length+' addressable categories'}}

function setComparisonClass(cardId,ownValue,otherValue){{
 const el=document.getElementById(cardId);if(!el)return;
 el.classList.remove('kpi-win','kpi-loss','kpi-tie');
 const a=Number(ownValue||0),b=Number(otherValue||0);
 const tolerance=Math.max(1e-9,Math.max(Math.abs(a),Math.abs(b))*0.0001);
 el.classList.add(Math.abs(a-b)<=tolerance?'kpi-tie':(a>b?'kpi-win':'kpi-loss'));
}}

function updateKpiComparison(supplier,compare){{
 const a=summaryFor(supplier),b=summaryFor(compare);
 setComparisonClass('selectedCardValue',a.awards,b.awards);
 setComparisonClass('compareCardValue',b.awards,a.awards);
 setComparisonClass('selectedCardShare',a.tam_share_pct,b.tam_share_pct);
 setComparisonClass('compareCardShare',b.tam_share_pct,a.tam_share_pct);
 setComparisonClass('selectedCardLargest',a.largestValue,b.largestValue);
 setComparisonClass('compareCardLargest',b.largestValue,a.largestValue);
 setComparisonClass('selectedCardBreadth',a.breadth,b.breadth);
 setComparisonClass('compareCardBreadth',b.breadth,a.breadth);
}}

function updateProfile(supplier,compare){{
 document.getElementById('profileTitle').innerHTML='<b>'+esc(supplier)+'</b> vs <b>'+esc(compare)+'</b> - Value by Service Offering';
 const a=new Map(rowsForSupplier(supplier).map(r=>[r.capability,r]));const b=new Map(rowsForSupplier(compare).map(r=>[r.capability,r]));
 const caps=Array.from(new Set([...a.keys(),...b.keys()])).filter(c=>Number((a.get(c)||{{}}).value||0)>0||Number((b.get(c)||{{}}).value||0)>0).sort((x,y)=>(Number((a.get(y)||{{}}).value||0)+Number((b.get(y)||{{}}).value||0))-(Number((a.get(x)||{{}}).value||0)+Number((b.get(x)||{{}}).value||0))).reverse();
 const av=caps.map(c=>Number((a.get(c)||{{}}).value||0)/1e6),bv=caps.map(c=>Number((b.get(c)||{{}}).value||0)/1e6);
 Plotly.react('profileChart',[{{type:'bar',orientation:'h',y:caps,x:av,name:supplier,marker:{{color:colour(supplier)}},text:caps.map(c=>money((a.get(c)||{{}}).value||0)),textposition:'outside',cliponaxis:false,customdata:caps.map(c=>pct((a.get(c)||{{}}).share||0)),hovertemplate:'<b>%{{y}}</b><br>'+esc(supplier)+': %{{text}}<br>Share of Service Offering: %{{customdata}}<extra></extra>'}},{{type:'bar',orientation:'h',y:caps,x:bv,name:compare,marker:{{color:colour(compare)}},text:caps.map(c=>money((b.get(c)||{{}}).value||0)),textposition:'outside',cliponaxis:false,customdata:caps.map(c=>pct((b.get(c)||{{}}).share||0)),hovertemplate:'<b>%{{y}}</b><br>'+esc(compare)+': %{{text}}<br>Share of Service Offering: %{{customdata}}<extra></extra>'}}],{{template:'plotly_white',barmode:'group',height:650,margin:{{t:30,l:260,r:150,b:75}},xaxis:{{title:'Award value ($M)',fixedrange:true}},yaxis:{{fixedrange:true,automargin:true}},legend:{{orientation:'h',y:-.14}}}},{{responsive:true,displayModeBar:false,scrollZoom:false}})
}}

function landscape(){{
 const rows=periodRows(), capSup=new Map(), marketKeys=new Map();rows.forEach(r=>{{const cap=String(r.capability||'Uncategorised'),s=String(r.supplier_group||'Unknown'),fy=String(r['Financial Year']||'');const k=cap+'||'+s;capSup.set(k,(capSup.get(k)||0)+Number(r.value||0));const mk=cap+'||'+fy;if(!marketKeys.has(mk))marketKeys.set(mk,Number(r.capability_year_market||0))}});const marketByCap=new Map();marketKeys.forEach((v,k)=>{{const cap=k.split('||')[0];marketByCap.set(cap,(marketByCap.get(cap)||0)+v)}});const out=[];capSup.forEach((v,k)=>{{const [cap,s]=k.split('||');const m=Number(marketByCap.get(cap)||0);out.push({{capability:cap,supplier_group:s,value:v,share:m?v/m*100:0}})}});return{{rows:out,marketByCap}}}}
function updateLeaderboards(supplier,compare){{
  const d=landscape();
  const p=period();
  const ys=new Set(p.years);

  // Contract counts for the exact same supplier + Service Offering + FY period.
  const contractCounts=new Map();
  YEARLY.filter(r=>p.key==='all'||ys.has(String(r['Financial Year']||''))).forEach(r=>{{
    const cap=String(r.capability||'Uncategorised');
    const s=String(r.supplier_group||'Unknown');
    const k=cap+'||'+s;
    contractCounts.set(k,(contractCounts.get(k)||0)+Number(r.contracts||0));
  }});

  const caps=Array.from(d.marketByCap.entries())
    .filter(x=>x[1]>0)
    .sort((a,b)=>b[1]-a[1])
    .map(x=>x[0]);

  document.getElementById('leaderboards').innerHTML=caps.map(cap=>{{
    const all=d.rows
      .filter(r=>r.capability===cap&&r.value>0)
      .sort((a,b)=>b.share-a.share);

    // Always show top 5, plus the two selected suppliers when they sit outside it.
    const top=all.slice(0,5);
    const extras=[];
    [supplier,compare].forEach(s=>{{
      const r=all.find(x=>x.supplier_group===s);
      if(r && !top.some(x=>x.supplier_group===s) && !extras.some(x=>x.supplier_group===s)) {{
        extras.push(r);
      }}
    }});
    extras.sort((a,b)=>all.findIndex(x=>x.supplier_group===a.supplier_group)-all.findIndex(x=>x.supplier_group===b.supplier_group));

    const visibleRows=[];
    top.forEach(r=>visibleRows.push({{type:'supplier',row:r}}));

    // Insert "..." whenever there is a rank jump between the top five and an
    // off-top-five selected/comparison supplier, and again between extras if needed.
    let previousRank=top.length;
    extras.forEach(r=>{{
      const rank=all.findIndex(x=>x.supplier_group===r.supplier_group)+1;
      if(rank>previousRank+1) visibleRows.push({{type:'gap'}});
      visibleRows.push({{type:'supplier',row:r}});
      previousRank=rank;
    }});

    const visibleSupplierRows=visibleRows.filter(x=>x.type==='supplier').map(x=>x.row);
    const max=Math.max(1,...visibleSupplierRows.map(r=>r.share));

    const body=visibleRows.map(item=>{{
      if(item.type==='gap') {{
        return '<div class="leader-gap-row" aria-label="Additional suppliers omitted"><div class="leader-gap-dots">...</div></div>';
      }}

      const r=item.row;
      const rank=all.findIndex(x=>x.supplier_group===r.supplier_group)+1;
      const cls=r.supplier_group===supplier?' focus-selected':(r.supplier_group===compare?' focus-compare':'');
      const nContracts=Number(contractCounts.get(cap+'||'+r.supplier_group)||0);
      const tooltip=
        '<b>'+esc(r.supplier_group)+'</b><br>'+
        esc(cap)+'<br>'+
        '<span style="color:#98A2B3">'+esc(p.label)+'</span>'+
        '<div class="leader-tooltip-rule"></div>'+
        '<b>Award value:</b> '+money(r.value)+'<br>'+
        '<b>Contracts:</b> '+nContracts.toLocaleString()+'<br>'+
        '<b>Market share:</b> '+pct(r.share)+'<br>'+
        '<b>Rank:</b> #'+rank;

      return '<div class="leader-row'+cls+' leaderboard-hover" tabindex="0">'+
        '<div>#'+rank+'</div>'+
        '<div>'+esc(r.supplier_group)+'</div>'+
        '<div class="leader-bar-track"><div class="leader-bar-fill" style="width:'+Math.max(1.5,r.share/max*100).toFixed(1)+'%;background:'+colour(r.supplier_group)+'"></div></div>'+
        '<div class="leader-share">'+pct(r.share)+'</div>'+
        '<div class="leader-tooltip">'+tooltip+'</div>'+
        '</div>';
    }}).join('');

    return '<section class="leaderboard-card"><h4>'+esc(cap)+'</h4>'+
      '<div class="leaderboard-market">'+money(d.marketByCap.get(cap)||0)+' addressable market · '+esc(p.label)+'</div>'+
      body+
      '</section>';
  }}).join('');
}}

function updateAgency(supplier){{const p=period(),ys=new Set(p.years);const rows=periodAgencyRows().filter(r=>r.supplier_group===supplier);const by=new Map();rows.forEach(r=>by.set(r.Agency,(by.get(r.Agency)||0)+Number(r.value||0)));const sorted=Array.from(by.entries()).sort((a,b)=>b[1]-a[1]);document.getElementById('agencyTitle').textContent=supplier+' Health agency footprint';if(!sorted.length){{Plotly.react('agencyChart',[],{{template:'plotly_white',annotations:[{{text:'No agency data for this period',showarrow:false,x:.5,y:.5,xref:'paper',yref:'paper'}}]}},{{responsive:true}});return}}const labels=sorted.map(x=>x[0]),values=sorted.map(x=>x[1]);Plotly.react('agencyChart',[{{type:'pie',labels,values,hole:.52,sort:false,textinfo:'label+percent',hovertemplate:'<b>%{{label}}</b><br>%{{value:$,.0f}}<br>%{{percent}} of '+esc(supplier)+'<extra></extra>'}}],{{template:'plotly_white',height:500,margin:{{t:35,l:30,r:220,b:35}},legend:{{orientation:'v',x:1.02,y:.95}},annotations:[{{text:money(values.reduce((a,b)=>a+b,0))+'<br><span style="font-size:12px;color:#667085">'+esc(p.label)+'</span>',x:.5,y:.5,showarrow:false,font:{{size:20}}}}]}},{{responsive:true,displayModeBar:false}})}}

function updateTimeline(supplier){{
 const p=period(),ys=new Set(p.years);
 let rows=CONTRACTS.filter(r=>r.supplier_group===supplier&&(p.key==='all'||ys.has(String(r['Financial Year']||''))));
 document.getElementById('timelineTitle').textContent=supplier+' current addressable contracts';
 document.getElementById('timelineSubtitle').textContent='Active as at '+dateLabel(AS_AT_DATE)+' · grouped by Service Offering · ranked by total contract value within each section · first supplier selector controls this view.';
 if(!rows.length){{Plotly.react('timelineChart',[],{{template:'plotly_white',height:380,annotations:[{{text:'No active addressable contracts with valid dates for '+esc(supplier)+'.',showarrow:false,x:.5,y:.5,xref:'paper',yref:'paper'}}],xaxis:{{visible:false}},yaxis:{{visible:false}}}},{{responsive:true,displayModeBar:false}});return}}

 const asAt=parseDate(AS_AT_DATE);
 const capabilityColour={{
   'Strategy, Transformation & Advisory':'#A100FF',
   'SI & Engineering':'#2563EB',
   'Data, AI & Automation':'#16A34A',
   'Cloud Infrastructure & Cyber':'#F59E0B',
   'Managed Services & Operations':'#667085',
   'Uncategorised':'#98A2B3'
 }};
 function rgba(hex,a){{
   const h=String(hex||'#98A2B3').replace('#','');
   const full=h.length===3?h.split('').map(x=>x+x).join(''):h;
   const n=parseInt(full,16);
   return 'rgba('+((n>>16)&255)+','+((n>>8)&255)+','+(n&255)+','+a+')';
 }}
 function capName(r){{const v=String(r.capability||'').trim();return v||'Uncategorised'}}
 function dayCount(r){{
   const end=parseDate(r._end_display);
   return asAt&&end?Math.max(0,Math.round((end-asAt)/86400000)):null;
 }}
 function compact(v,n){{
   const s=String(v||'').replace(/\s+/g,' ').trim();
   return s.length>n?s.slice(0,n-1)+'…':s;
 }}

 const groups=new Map();
 rows.forEach(r=>{{
   const cap=capName(r);
   if(!groups.has(cap))groups.set(cap,[]);
   groups.get(cap).push(r);
 }});
 const orderedCaps=Array.from(groups.entries()).sort((a,b)=>{{
   const av=a[1].reduce((s,r)=>s+Number(r._total_value||0),0);
   const bv=b[1].reduce((s,r)=>s+Number(r._total_value||0),0);
   return bv-av;
 }}).map(x=>x[0]);
 orderedCaps.forEach(cap=>groups.get(cap).sort((a,b)=>Number(b._total_value||0)-Number(a._total_value||0)));

 const maxContracts=120;
 const selectedByCap=new Map();
 let remaining=maxContracts;
 orderedCaps.forEach(cap=>{{
   const all=groups.get(cap)||[];
   const take=all.slice(0,Math.max(0,remaining));
   selectedByCap.set(cap,take);
   remaining-=take.length;
 }});
 const activeCaps=orderedCaps.filter(cap=>(selectedByCap.get(cap)||[]).length);

 // Mirror the Defence timeline density: compact contract rows, a distinct
 // Service Offering header, then a small gap before the next section.
 const rowItems=[];
 const sections=[];
 let cursor=0;
 activeCaps.forEach(cap=>{{
   const group=selectedByCap.get(cap)||[];
   const sectionValue=group.reduce((s,r)=>s+Number(r._total_value||0),0);
   const remainingDays=group.map(dayCount).filter(v=>v!==null&&Number.isFinite(v));
   const avgDays=remainingDays.length?Math.round(remainingDays.reduce((a,b)=>a+b,0)/remainingDays.length):0;

   // Small sections can stay compact. Sections with 3+ contracts need more
   // vertical breathing room because every row carries three lines of text.
   const denseGroup=group.length>1;
   const headerGap=denseGroup?0.74:0.66;
   const rowStep=denseGroup?1.14:0.88;
   const sectionGap=denseGroup?0.56:0.42;

   const headerY=cursor;
   cursor+=headerGap;
   const firstY=cursor;
   group.forEach((r,idx)=>{{
     rowItems.push({{row:r,y:cursor,cap,rank:idx+1}});
     cursor+=rowStep;
   }});
   const lastY=group.length ? cursor-rowStep : firstY;
   sections.push({{cap,headerY,firstY,lastY,count:group.length,value:sectionValue,avgDays,colour:capabilityColour[cap]||'#475467',rowStep,denseGroup}});
   cursor+=sectionGap;
 }});

 const tickVals=rowItems.map(x=>x.y);
 const tickText=rowItems.map(item=>{{
   const r=item.row;
   const days=dayCount(r);
   const agency=compact(r.Agency||'Unknown agency',48);
   const desc=compact(r.Description||'No description',58);
   return '<b>#'+item.rank+' · '+money(r._total_value||0)+' · '+esc(agency)+'</b><br>'+ 
     '<span style="color:#475467">'+esc(desc)+'</span><br>'+ 
     '<span style="color:#667085">Ends '+dateLabel(r._end_display)+(days===null?'':' · '+days.toLocaleString()+' days left')+'</span>';
 }});

 const traces=[];
 rowItems.forEach(item=>{{
   const r=item.row,c=colour(supplier),days=dayCount(r);
   const custom=[[r['CN ID']||'',r.Description||'',item.cap,r.Agency||'',money(r._total_value||0),dateLabel(r._start_display),dateLabel(r._end_display),days===null?'':days.toLocaleString()],[r['CN ID']||'',r.Description||'',item.cap,r.Agency||'',money(r._total_value||0),dateLabel(r._start_display),dateLabel(r._end_display),days===null?'':days.toLocaleString()]];
   const hover='<b>%{{customdata[0]}}</b><br>%{{customdata[1]}}<br><br>Service Offering: %{{customdata[2]}}<br>Agency: %{{customdata[3]}}<br>Total contract value: %{{customdata[4]}}<br>Start: %{{customdata[5]}}<br>End: %{{customdata[6]}}<br>Days remaining: %{{customdata[7]}}<extra></extra>';
   traces.push({{type:'scatter',mode:'lines',x:[r._start_display,r._end_display],y:[item.y,item.y],line:{{color:c,width:7}},showlegend:false,customdata:custom,hovertemplate:hover}});
   traces.push({{type:'scatter',mode:'markers',x:[r._start_display],y:[item.y],marker:{{size:7,color:'#FFFFFF',line:{{color:c,width:2}},symbol:'circle'}},showlegend:false,customdata:[custom[0]],hovertemplate:hover}});
   traces.push({{type:'scatter',mode:'markers',x:[r._end_display],y:[item.y],marker:{{size:9,color:c,symbol:'diamond'}},showlegend:false,customdata:[custom[0]],hovertemplate:hover}});
 }});

 const shapes=[];
 const annotations=[];
 sections.forEach(s=>{{
   const yTop=s.headerY+0.26;
   const yBottom=s.lastY+0.36;
   shapes.push({{type:'rect',xref:'paper',x0:0,x1:1,yref:'y',y0:yTop,y1:yBottom,fillcolor:rgba(s.colour,0.045),line:{{color:rgba(s.colour,0.48),width:1.2}},layer:'below'}});
   shapes.push({{type:'rect',xref:'paper',x0:0,x1:1,yref:'y',y0:s.headerY-0.24,y1:s.headerY+0.24,fillcolor:rgba(s.colour,0.14),line:{{color:rgba(s.colour,0.62),width:1}},layer:'below'}});
   annotations.push({{xref:'paper',x:0.008,yref:'y',y:s.headerY,xanchor:'left',yanchor:'middle',showarrow:false,align:'left',text:'<b>'+esc(s.cap)+'</b> &nbsp; <span style="color:#475467">'+money(s.value)+' active value &nbsp;|&nbsp; '+s.count+' '+(s.count===1?'contract':'contracts')+' &nbsp;|&nbsp; '+s.avgDays+' avg days remaining</span>',font:{{size:12,color:s.colour}}}});
 }});

 const yMin=-0.48;
 const yMax=Math.max(cursor-0.04,1);
 shapes.push({{type:'line',x0:AS_AT_DATE,x1:AS_AT_DATE,y0:yMin,y1:yMax,xref:'x',yref:'y',line:{{color:'#111827',width:2.4,dash:'dash'}}}});
 annotations.push({{x:AS_AT_DATE,y:1.055,xref:'x',yref:'paper',showarrow:false,text:'<b>Today · '+dateLabel(AS_AT_DATE)+'</b>',font:{{size:12,color:'#111827'}},bgcolor:'#ffffff',bordercolor:'#111827',borderwidth:1,borderpad:4}});

 // Keep the timeline readable at normal browser zoom
 const denseContractCount=sections.filter(s=>s.denseGroup).reduce((n,s)=>n+s.count,0);
 const h=Math.max(500,rowItems.length*52+denseContractCount*14+sections.length*46+120);
 const longest=Math.max(0,...tickText.map(v=>String(v).replace(/<[^>]*>/g,'').length));
 const leftMargin=Math.min(560,Math.max(430,longest*4.4+70));

 Plotly.react('timelineChart',traces,{{
   template:'plotly_white',height:h,margin:{{t:86,l:leftMargin,r:38,b:64}},
   xaxis:{{type:'date',title:'Contract start date → contract end date',fixedrange:true,showgrid:true,gridcolor:'#E9EDF3',tickformat:'%Y',dtick:'M12',ticks:'outside',showline:true,linecolor:'#D0D5DD'}},
   // Duplicate the year ruler at the top so the time scale remains visible when
   // reading the upper Service Offering sections.
   xaxis2:{{type:'date',overlaying:'x',matches:'x',side:'top',fixedrange:true,showgrid:false,zeroline:false,tickformat:'%Y',dtick:'M12',ticks:'outside',showline:true,linecolor:'#D0D5DD',title:''}},
   yaxis:{{tickmode:'array',tickvals:tickVals,ticktext:tickText,range:[yMax,yMin],fixedrange:true,automargin:false,tickfont:{{size:11}},ticklabelstandoff:14,showgrid:false,zeroline:false}},
   shapes:shapes,annotations:annotations,hoverlabel:{{align:'left'}},showlegend:false
 }},{{responsive:true,displayModeBar:false,scrollZoom:false}})
}}

function render(){{const s=document.getElementById('supplierSelect').value,c=document.getElementById('compareSelect').value;document.getElementById('selectedBlock').style.borderTopColor=colour(s);document.getElementById('compareBlock').style.borderTopColor=colour(c);document.getElementById('selectedTitle').style.borderLeftColor=colour(s);document.getElementById('compareTitle').style.borderLeftColor=colour(c);setKpis('selected',s);setKpis('compare',c);updateKpiComparison(s,c);updateProfile(s,c);updateLeaderboards(s,c);updateTimeline(s);updateAgency(s)}}
buildPeriods();document.getElementById('supplierSelect').value=DEFAULT_SUPPLIER;document.getElementById('compareSelect').value=DEFAULT_COMPARE;['supplierSelect','compareSelect','periodSelect'].forEach(id=>document.getElementById(id).addEventListener('change',render));render();
</script>
</body>
</html>"""
    output_path.write_text(html_doc, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = Path(args.input)
    df, value_col = load_data(path, args.value_mode)
    as_at = pd.to_datetime(args.as_at, errors="coerce") if args.as_at else pd.Timestamp.today().normalize()
    if pd.isna(as_at):
        raise SystemExit("Could not parse --as-at. Use YYYY-MM-DD.")
    as_at = pd.Timestamp(as_at).normalize()

    model = build_model(df, args.top_n_suppliers, as_at)
    dashboard_path = output_dir / "HealthCompareDashboard.html"
    build_html(model, dashboard_path, as_at, args.value_mode)

    model["supplier_total"].to_csv(output_dir / "supplier_summary.csv", index=False)  # type: ignore[index]
    model["supplier_capability"].to_csv(output_dir / "supplier_service_offering.csv", index=False)  # type: ignore[index]
    model["yearly"].to_csv(output_dir / "supplier_service_offering_yearly.csv", index=False)  # type: ignore[index]
    model["agency_yearly"].to_csv(output_dir / "supplier_agency_yearly.csv", index=False)  # type: ignore[index]
    model["active_contracts"].head(args.top_n_contracts).to_csv(output_dir / "current_contracts.csv", index=False)  # type: ignore[index]

    print("Health competitive intelligence dashboard complete")
    print(f"Input: {path}")
    print(f"Value mode: {args.value_mode} / column used: {value_col}")
    print(f"As-at date: {as_at.strftime('%Y-%m-%d')}")
    print(f"Addressable Health TAM: {money(float(model['total_tam']))}")
    print(f"Suppliers in dropdown: {len(model['supplier_options'])}")
    print(f"Wrote: {dashboard_path}")


if __name__ == "__main__":
    main()
