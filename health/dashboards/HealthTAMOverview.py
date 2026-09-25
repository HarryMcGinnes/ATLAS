from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

VALUE_COL = "Value"
ANNUALISED_VALUE_COL = "Value Per Year"
FY_COL = "Financial Year"
SUPPLIER_COL = "Supplier Name"
SUPPLIER_ABN_COL = "Supplier ABN"
DESCRIPTION_COL = "Description"
CATEGORY_COL = "Category"
CATEGORY_TYPE_COL = "Category Type"
CN_ID_COL = "CN ID"
AGENCY_COL = "Agency"

ACCENTURE_ABNS = {"49096776895"}

SERVICE_OFFERINGS = [
    "Strategy, Transformation & Advisory",
    "SI & Engineering",
    "Data, AI & Automation",
    "Cloud Infrastructure & Cyber",
    "Managed Services & Operations",
]

SO_COLOURS = {
    "Strategy, Transformation & Advisory": "#4C1D95",
    "SI & Engineering": "#6D28D9",
    "Data, AI & Automation": "#C4B5FD",
    "Cloud Infrastructure & Cyber": "#8B5CF6",
    "Managed Services & Operations": "#A78BFA",
    "Unclassified": "#DDD6FE",
}

# First-pass Health TAM classifier. The intent is conservative and auditable:
# remove clearly physical/clinical/commodity procurement first, then recognise
# consulting, digital, data, cloud/cyber and managed-service evidence.
NON_ADDRESSABLE_CATEGORY_TERMS = [
    "blood product", "plasma", "immunoglobulin", "vaccine", "pharmaceutical",
    "medicine", "medication", "drug", "medical equipment", "medical device",
    "laboratory equipment", "laboratory supplies", "diagnostic", "surgical",
    "prosthetic", "implant", "ambulance", "vehicle", "motor vehicle",
    "furniture", "office supplies", "stationery", "electricity", "gas supply",
    "food", "catering", "cleaning", "property lease", "rental of property",
    "construction", "building construction", "facilities maintenance",
]

ADDRESSABLE_TERMS = [
    "consult", "advisory", "strategy", "transformation", "business design",
    "operating model", "change management", "program management", "programme management",
    "project management", "project administration", "assurance", "review", "evaluation",
    "digital", "computer services", "information technology", "ict", "software",
    "application", "system integration", "systems integration", "engineering",
    "development services", "solution design", "architecture", "platform",
    "data", "analytics", "artificial intelligence", " ai ", "automation",
    "business intelligence", "reporting", "cloud", "saas", "iaas", "paas",
    "cyber", "security operations", "identity and access", "network",
    "managed service", "operational services", "operations service", "support services",
    "software maintenance", "system support", "systems support", "service desk",
    "infrastructure services", "infrastructure operation", "hosting", "subscription",
    "license", "licence", "renewal", "training", "workforce transformation",
]

SO_RULES = {
    "Managed Services & Operations": [
        "managed service", "operational services", "operations service", "support services",
        "software maintenance", "maintenance and support", "system support", "systems support",
        "service desk", "infrastructure operation", "infrastructure services", "hosting",
        "subscription", "renewal", "license", "licence", "system administrator",
        "administration services",
    ],
    "Cloud Infrastructure & Cyber": [
        "cloud", "saas", "iaas", "paas", "cyber", "security operations", "security operation",
        "identity and access", "identity management", "network security", "infrastructure security",
        "zero trust", "soc ", "security centre", "security center",
    ],
    "Data, AI & Automation": [
        "data analytics", "analytics", "artificial intelligence", "machine learning",
        "automation", "business intelligence", "data platform", "data solution",
        "data collection", "data reporting", "reporting mechanism", "data warehouse",
    ],
    "SI & Engineering": [
        "system integration", "systems integration", "software engineering", "hardware engineering",
        "software development", "development services", "application development", "system development",
        "solution design", "technical design", "platform development", "implementation",
        "computer services", "digital implementation", "digital delivery",
    ],
    "Strategy, Transformation & Advisory": [
        "strategy", "strategic", "advisory", "consult", "transformation", "business design",
        "operating model", "change management", "program management", "programme management",
        "project management", "project administration", "assurance", "review", "evaluation",
        "business requirement", "requirements development", "portfolio management", "training",
        "management support",
    ],
}

SUPPLIER_RULES = [
    (r"\baccenture\b", "Accenture"),
    (r"\bdeloitte\b", "Deloitte"),
    (r"\bkpmg\b", "KPMG"),
    (r"\b(?:ernst\s*(?:&|and)?\s*young|ey)\b", "EY"),
    (r"\b(?:pricewaterhousecoopers|pwc)\b", "PwC"),
    (r"\bibm\b|international business machines", "IBM"),
    (r"\bfujitsu\b", "Fujitsu"),
    (r"\bdxc\b", "DXC"),
    (r"\bcapgemini\b", "Capgemini"),
    (r"\bdatacom\b", "Datacom"),
    (r"\bdata\s*#?\s*3\b", "Data#3"),
    (r"\bmicrosoft\b", "Microsoft"),
    (r"amazon web services|\baws\b", "Amazon / AWS"),
    (r"\boracle\b", "Oracle"),
    (r"\bsap\b", "SAP"),
    (r"\btelstra\b", "Telstra"),
]


def norm(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).lower().replace("\u00a0", " ")).strip()


def norm_abn(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\D", "", str(value)).lstrip("0")


def supplier_group(name: object, abn: object) -> str:
    a = norm_abn(abn)
    if a in ACCENTURE_ABNS:
        return "Accenture"
    raw = str(name or "").strip()
    text = norm(raw)
    for pattern, label in SUPPLIER_RULES:
        if re.search(pattern, text):
            return label
    return raw if raw else "Unknown"


def classify_addressability(row: pd.Series) -> tuple[bool, str, str]:
    category = norm(row.get(CATEGORY_COL, ""))
    description = norm(row.get(DESCRIPTION_COL, ""))
    ctype = norm(row.get(CATEGORY_TYPE_COL, ""))
    text = f" {category} {description} "

    # Strong physical/clinical procurement exclusions take precedence.
    for term in NON_ADDRESSABLE_CATEGORY_TERMS:
        if term in category:
            return False, "Non-addressable", f"Excluded physical/clinical category: {term}"

    # Goods containing equipment/components/accessories remain conservative.
    if ctype == "goods" and any(t in category for t in ["equipment", "component", "accessories", "supplies"]):
        return False, "Non-addressable", "Goods/equipment rule"

    matched = [term.strip() for term in ADDRESSABLE_TERMS if term in text]
    if not matched:
        return False, "Non-addressable", "No recognised consulting/technology/services evidence"

    return True, "Addressable", "Matched: " + ", ".join(dict.fromkeys(matched[:6]))


def classify_service_offering(row: pd.Series) -> str:
    if not bool(row.get("is_addressable", False)):
        return "Unclassified"
    text = " " + norm(row.get(CATEGORY_COL, "")) + " " + norm(row.get(DESCRIPTION_COL, "")) + " "

    # Specific content rules first, then strategy/advisory as the general fallback.
    for so in [
        "Managed Services & Operations",
        "Cloud Infrastructure & Cyber",
        "Data, AI & Automation",
        "SI & Engineering",
        "Strategy, Transformation & Advisory",
    ]:
        if any(term in text for term in SO_RULES[so]):
            return so
    return "Strategy, Transformation & Advisory"


def load_health(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Health dataset not found: {path}")
    if path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path, low_memory=False)
    else:
        raise SystemExit("Input must be CSV or Parquet")
    required = {VALUE_COL, FY_COL, SUPPLIER_COL, DESCRIPTION_COL, CATEGORY_COL}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit("Missing required columns: " + ", ".join(missing))
    return df


def _as_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )


def classify(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare the dashboard from the canonical Health master dataset.

    Addressability and Service Offering are now produced upstream by
    build_health_filter_v2.py.  The dashboard must consume those fields rather
    than reclassify contracts independently.
    """
    out = df.copy()
    out[VALUE_COL] = pd.to_numeric(out[VALUE_COL], errors="coerce").fillna(0)
    if ANNUALISED_VALUE_COL in out.columns:
        out[ANNUALISED_VALUE_COL] = pd.to_numeric(out[ANNUALISED_VALUE_COL], errors="coerce").fillna(0)

    # Dashboard-only supplier grouping. This does not affect addressability/SO.
    abns = out[SUPPLIER_ABN_COL] if SUPPLIER_ABN_COL in out.columns else pd.Series("", index=out.index)
    out["supplier_group"] = [supplier_group(n, a) for n, a in zip(out[SUPPLIER_COL], abns)]
    out["is_accenture"] = out["supplier_group"].eq("Accenture")

    if "is_addressable" not in out.columns:
        raise SystemExit(
            "Health master dataset does not contain 'is_addressable'. "
            "Run build_health_filter_v2.py first."
        )
    out["is_addressable"] = _as_bool_series(out["is_addressable"])

    # The Health build writes both the canonical snake_case field and a
    # dashboard-friendly alias. Prefer the canonical field if present.
    if "service_offering" in out.columns:
        out["Service Offering"] = out["service_offering"]
    elif "Service Offering" not in out.columns:
        raise SystemExit(
            "Health master dataset does not contain a Service Offering field. "
            "Run build_health_filter_v2.py first."
        )

    # Enforce the Health classifier invariant in the dashboard too:
    # non-addressable contracts must not display a Service Offering.
    out.loc[~out["is_addressable"], "Service Offering"] = pd.NA
    return out


def fy_start(v: object) -> int:
    m = re.search(r"(19|20)\d{2}", str(v))
    return int(m.group(0)) if m else -1


def fy_options(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    fys = sorted(df[FY_COL].dropna().astype(str).unique(), key=fy_start)
    options = [("All years", df)]
    if len(fys) >= 5:
        subset = fys[-5:]
        options.append((f"Last 5 FY ({subset[0]} to {subset[-1]})", df[df[FY_COL].astype(str).isin(subset)]))
    if len(fys) >= 3:
        subset = fys[-3:]
        options.append((f"Last 3 FY ({subset[0]} to {subset[-1]})", df[df[FY_COL].astype(str).isin(subset)]))
    return options


def money(v: float) -> str:
    v = float(v or 0)
    a = abs(v)
    if a >= 1e9:
        return f"${v/1e9:,.2f}B"
    if a >= 1e6:
        return f"${v/1e6:,.1f}M"
    if a >= 1e3:
        return f"${v/1e3:,.1f}K"
    return f"${v:,.0f}"


def summary(df: pd.DataFrame, value_col: str) -> dict:
    total = float(df[value_col].sum())
    addr = float(df.loc[df["is_addressable"], value_col].sum())
    acc = float(df.loc[df["is_addressable"] & df["is_accenture"], value_col].sum())
    years = max(1, df[FY_COL].astype(str).nunique())
    return {
        "total": total,
        "addressable": addr,
        "not_addressable": total - addr,
        "accenture": acc,
        "competitor": max(0.0, addr - acc),
        "addressable_pct": (addr / total * 100) if total else 0,
        "accenture_share": (acc / addr * 100) if addr else 0,
        "years": years,
        "avg_addressable": addr / years,
        "avg_accenture": acc / years,
    }


def build_payloads(df: pd.DataFrame, value_col: str) -> dict:
    donut_sets, service_sets, supplier_sets = [], [], []
    for label, sub in fy_options(df):
        s = summary(sub, value_col)
        wins = sub[sub["is_addressable"] & sub["is_accenture"]]
        win_so = wins.groupby("Service Offering")[value_col].sum().sort_values(ascending=False)
        donut_sets.append({
            "label": label, **s,
            "total_label": money(s["total"]), "addressable_label": money(s["addressable"]),
            "not_addressable_label": money(s["not_addressable"]), "accenture_label": money(s["accenture"]),
            "competitor_label": money(s["competitor"]),
            "offerings": [{"name": k, "value": float(v), "label": money(v), "colour": SO_COLOURS.get(k, "#DDD6FE")} for k, v in win_so.items() if v > 0],
        })

        addr = sub[sub["is_addressable"]].copy()
        wins = wins.copy()

        # Keep the five defined Service Offerings, but do not let addressable
        # contracts disappear from the chart just because their SO is unresolved.
        # Any blank, null or unexpected SO is grouped into a final
        # "Uncategorised" bucket so the service-offering bars reconcile back to
        # the full addressable TAM.
        def _chart_so(series: pd.Series) -> pd.Series:
            cleaned = series.fillna("").astype(str).str.strip()
            return cleaned.where(cleaned.isin(SERVICE_OFFERINGS), "Uncategorised")

        addr["_chart_service_offering"] = _chart_so(addr["Service Offering"])
        wins["_chart_service_offering"] = _chart_so(wins["Service Offering"])

        so_total = addr.groupby("_chart_service_offering")[value_col].sum()
        so_acc = wins.groupby("_chart_service_offering")[value_col].sum()
        chart_offerings = SERVICE_OFFERINGS + ["Uncategorised"]
        service_rows = [{
            "name": so,
            "total": float(so_total.get(so, 0)),
            "accenture": float(so_acc.get(so, 0)),
            "total_b": float(so_total.get(so, 0))/1e9,
            "accenture_b": float(so_acc.get(so, 0))/1e9,
            "total_label": money(so_total.get(so, 0)),
            "accenture_label": money(so_acc.get(so, 0)),
        } for so in chart_offerings if so_total.get(so, 0) > 0]
        # First horizontal-bar category renders at the bottom, so ascending values
        # display as largest-to-smallest from top to bottom.
        service_rows.sort(key=lambda r: (r["accenture"], r["total"]))
        service_sets.append({"label": label, "rows": service_rows})

        sup = (addr.groupby("supplier_group").agg(value=(value_col,"sum"), contracts=(CN_ID_COL,"nunique") if CN_ID_COL in addr.columns else (value_col,"size")).reset_index().sort_values("value",ascending=False).head(15))
        tam = float(addr[value_col].sum())
        supplier_sets.append({
            "label": label,
            "rows": [{"name": r.supplier_group, "value": float(r.value), "value_label": money(r.value), "share": (float(r.value)/tam*100 if tam else 0), "contracts": int(r.contracts)} for r in sup.itertuples()]
        })
    return {"donut": donut_sets, "service": service_sets, "supplier": supplier_sets}


def controls(name: str, sets: list[dict]) -> str:
    return "".join(f'<label class="pill"><input type="radio" name="{name}" value="{i}" {"checked" if i==0 else ""}> {d["label"]}</label>' for i,d in enumerate(sets))


def build_html(df: pd.DataFrame, value_col: str, output: Path) -> None:
    payload = build_payloads(df, value_col)
    s = summary(df, value_col)
    data = json.dumps(payload)
    html = f'''<!doctype html>
<html><head><meta charset="utf-8"><title>Accenture Health TAM Overview</title>
<script src="https://cdn.plot.ly/plotly-3.5.0.min.js"></script>
<style>
body{{font-family:Arial,Helvetica,sans-serif;margin:0;background:#f6f8fb;color:#1f2937}}.wrap{{max-width:1480px;margin:auto;padding:28px}}h1{{margin:0 0 8px;font-size:30px}}
.subtitle{{color:#526070;margin:0 0 18px}}.summary{{background:#fff;border:1px solid #e5e7eb;border-left:5px solid #A100FF;border-radius:14px;padding:18px 20px;margin-bottom:18px;box-shadow:0 1px 2px rgba(0,0,0,.04)}}
.summary h2{{margin:0 0 8px;font-size:20px}}.summary p{{margin:5px 0;color:#475467;line-height:1.5}}.warning{{background:#fff7ed;border:1px solid #fed7aa;border-radius:12px;padding:12px 14px;color:#9a3412;margin:12px 0}}
.cards{{display:grid;grid-template-columns:repeat(3,minmax(220px,1fr));gap:14px;margin:18px 0 22px}}.card,.panel{{background:#fff;border:1px solid #e5e7eb;border-radius:14px;box-shadow:0 1px 2px rgba(0,0,0,.04)}}.card{{padding:17px}}.ct{{font-size:12px;text-transform:uppercase;color:#667085;letter-spacing:.04em}}.cv{{font-size:29px;font-weight:700;color:#111827;margin:8px 0 5px}}.cs{{font-size:13px;color:#667085}}
.panel{{padding:18px;margin-bottom:18px}}.panel h3{{margin:4px 0;font-size:18px}}.panel p{{margin:4px 0 10px;color:#526070;font-size:13px}}.controls{{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0}}.pill{{display:inline-flex;align-items:center;gap:6px;border:1px solid #d0d5dd;border-radius:999px;padding:7px 12px;background:#fff;font-size:12px;cursor:pointer}}.pill:has(input:checked){{background:#eef2ff;border-color:#636efa;color:#243b9f;font-weight:600}}
.flow-head{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:0;text-align:center;margin:14px 0 0}}.flow-head b{{font-size:17px;line-height:1.25}}.flow-head span{{display:block;color:#667085;font-size:12px;margin-top:5px}}.donut-stage{{position:relative;height:430px;min-width:980px}}#donutChart{{height:430px;width:100%;margin:0 auto}}.flow-transition{{position:absolute;top:45%;width:118px;transform:translate(-50%,-50%);z-index:5;text-align:center;color:#5B21B6;pointer-events:none}}.flow-transition.one{{left:33.5%}}.flow-transition.two{{left:66.5%}}.flow-transition .pct{{font-size:18px;font-weight:800;line-height:1}}.flow-transition .lbl{{font-size:11px;line-height:1.2;margin-top:4px}}.flow-transition .val{{font-size:15px;font-weight:700;margin-top:4px}}.flow-arrow{{height:18px;position:relative;margin-top:8px}}.flow-arrow:before{{content:'';position:absolute;left:0;right:12px;top:8px;height:3px;border-radius:999px;background:#5B21B6}}.flow-arrow:after{{content:'';position:absolute;right:0;top:1px;border-top:8px solid transparent;border-bottom:8px solid transparent;border-left:13px solid #5B21B6}}.donut-legends{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px;padding:0 24px 10px;min-width:980px}}.donut-legend{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px 16px;align-items:start;padding:12px 14px 14px;border:1px solid #E2E8F0;border-radius:10px;background:#FAFBFD;min-height:70px}}.donut-legend-title{{grid-column:1/-1;font-size:12px;font-weight:800;color:#475467;text-transform:uppercase;letter-spacing:.04em;padding-bottom:6px;border-bottom:1px solid #E9EDF3;margin-bottom:2px}}.legend-item{{display:grid;grid-template-columns:14px minmax(0,1fr);gap:8px;align-items:start;font-size:12px;color:#344054}}.legend-swatch{{width:14px;height:14px;border-radius:3px;margin-top:1px}}.legend-item b{{display:block;font-size:12px;line-height:1.2}}.legend-item small{{display:block;color:#667085;margin-top:2px;line-height:1.2}}.note{{font-size:12px;color:#667085;margin-top:8px}}@media(max-width:800px){{.cards{{grid-template-columns:1fr}}.flow-scroll{{overflow:auto}}}}
</style></head><body><div class="wrap">
<h1>Accenture Health Addressable Market Dashboard</h1><p class="subtitle">Initial Health Portfolio TAM view using the filtered AusTender Health dataset.</p>
<section class="summary"><h2>Executive Summary</h2><p>This dashboard provides a first-pass view of Health portfolio procurement, the portion provisionally addressable to Accenture, Accenture's historical wins, service-offering demand and leading suppliers.</p><div class="warning"><b>Classifier status:</b> this is a first-pass Health TAM classifier. It deliberately excludes obvious clinical products, medical goods and commodities, then identifies consulting, digital, data, cloud, cyber and managed-services evidence. Review the exported audit before using the TAM as a final executive figure.</div></section>
<div class="cards">
<div class="card"><div class="ct">Total Health procurement</div><div class="cv">{money(s['total'])}</div><div class="cs">Full Health Portfolio dataset</div></div>
<div class="card"><div class="ct">Provisional addressable TAM</div><div class="cv">{money(s['addressable'])}</div><div class="cs">{s['addressable_pct']:.1f}% of Health procurement</div></div>
<div class="card"><div class="ct">Accenture wins in TAM</div><div class="cv">{money(s['accenture'])}</div><div class="cs">{s['accenture_share']:.1f}% of provisional TAM</div></div>
<div class="card"><div class="ct">Competitor-owned TAM</div><div class="cv">{money(s['competitor'])}</div><div class="cs">Addressable value awarded to other suppliers</div></div>
<div class="card"><div class="ct">Avg annual addressable</div><div class="cv">{money(s['avg_addressable'])}</div><div class="cs">Across {s['years']} financial years</div></div>
<div class="card"><div class="ct">Avg annual Accenture wins</div><div class="cv">{money(s['avg_accenture'])}</div><div class="cs">Across {s['years']} financial years</div></div>
</div>
<div class="panel"><h3>Total Addressable Market Overview</h3><div class="controls">{controls('donutRange',payload['donut'])}</div><div class="flow-scroll"><div class="flow-head"><div><b>1. Total Health Procurement Market</b><span>Out of Health portfolio procurement</span></div><div><b>2. Within Addressable Market (TAM)</b><span>Accenture vs competitor-owned TAM</span></div><div><b>3. Accenture Wins</b><span>Accenture wins within the addressable Health market</span></div></div><div class="donut-stage"><div id="donutChart"></div><div id="flowTransitionOne" class="flow-transition one"></div><div id="flowTransitionTwo" class="flow-transition two"></div></div><div class="donut-legends"><div id="donutLegend1" class="donut-legend"></div><div id="donutLegend2" class="donut-legend"></div><div id="donutLegend3" class="donut-legend"></div></div></div></div>
<div class="panel"><h3>Accenture Wins by Service Offering</h3><p>Each Service Offering is shown as two separate bars: Accenture wins in purple and the remaining addressable market in grey.</p><div class="controls">{controls('serviceRange',payload['service'])}</div><div id="serviceChart"></div></div>
<div class="panel"><h3>Top Suppliers by Share of Accenture-addressable Health TAM</h3><div class="controls">{controls('supplierRange',payload['supplier'])}</div><div id="supplierChart"></div></div>
<p class="note">Source: filtered AusTender Health portfolio dataset. Total contract value is used unless --value-mode annualised is selected.</p>
</div><script>
const DATA={data};
function selected(name){{const e=document.querySelector('input[name="'+name+'"]:checked');return e?Number(e.value):0;}}
function fmtPct(v){{return Number(v||0).toFixed(1)+'%';}}
function money(v){{v=Number(v||0);let a=Math.abs(v);if(a>=1e9)return '$'+(v/1e9).toFixed(2)+'B';if(a>=1e6)return '$'+(v/1e6).toFixed(1)+'M';if(a>=1e3)return '$'+(v/1e3).toFixed(1)+'K';return '$'+v.toFixed(0);}}
function legendItem(title,pct,value,colour){{return '<div class="legend-item"><span class="legend-swatch" style="background:'+colour+'"></span><div><b>'+title+'</b><small>'+fmtPct(pct)+' · '+value+'</small></div></div>';}}
function legendTitle(text){{return '<div class="donut-legend-title">'+text+'</div>';}}
function transitionHtml(pct,label,value){{return '<div class="pct">'+fmtPct(pct)+'</div><div class="lbl">'+label+'</div><div class="val">'+value+'</div><div class="flow-arrow"></div>';}}
function renderDonut(){{const d=DATA.donut[selected('donutRange')];const tr=[
{{type:'pie',hole:.64,labels:['Addressable (TAM)','Not addressable'],values:[d.addressable,d.not_addressable],domain:{{x:[.035,.298],y:[.14,.90]}},sort:false,direction:'clockwise',rotation:90-((d.addressable/Math.max(d.total,1))*360/2),marker:{{colors:['#4C1D95','#D8CCF3'],line:{{color:'#fff',width:3}}}},textinfo:'none',showlegend:false,hovertemplate:'<b>%{{label}}</b><br>%{{value:$,.0f}}<br>%{{percent}}<extra></extra>'}},
{{type:'pie',hole:.64,labels:['Accenture wins','Competitors'],values:[d.accenture,d.competitor],domain:{{x:[.368,.632],y:[.14,.90]}},sort:false,direction:'clockwise',rotation:90-((d.accenture/Math.max(d.addressable,1))*360/2),marker:{{colors:['#9F7AEA','#4C1D95'],line:{{color:'#fff',width:3}}}},textinfo:'none',showlegend:false,hovertemplate:'<b>%{{label}}</b><br>%{{value:$,.0f}}<br>%{{percent}}<extra></extra>'}},
{{type:'pie',hole:.64,labels:['Accenture wins'],values:[d.accenture],domain:{{x:[.702,.965],y:[.14,.90]}},sort:false,marker:{{colors:['#9F7AEA'],line:{{color:'#fff',width:3}}}},textinfo:'none',showlegend:false,hovertemplate:'<b>Accenture wins</b><br>%{{value:$,.0f}}<extra></extra>'}}
];const ann=[
{{x:.1665,y:.52,xref:'paper',yref:'paper',text:'<b>'+d.total_label+'</b><br><span style="font-size:11px;color:#526070">Total Health Procurement</span>',showarrow:false,xanchor:'center',yanchor:'middle',align:'center',font:{{size:24,color:'#344054'}}}},
{{x:.5000,y:.52,xref:'paper',yref:'paper',text:'<b>'+d.addressable_label+'</b><br><span style="font-size:11px;color:#526070">Addressable TAM</span>',showarrow:false,xanchor:'center',yanchor:'middle',align:'center',font:{{size:24,color:'#344054'}}}},
{{x:.8335,y:.52,xref:'paper',yref:'paper',text:'<b>'+d.accenture_label+'</b><br><span style="font-size:11px;color:#526070">Accenture Wins</span>',showarrow:false,xanchor:'center',yanchor:'middle',align:'center',font:{{size:24,color:'#344054'}}}}
];
Plotly.react('donutChart',tr,{{height:430,margin:{{t:4,l:4,r:4,b:4}},showlegend:false,paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',annotations:ann}},{{responsive:true,displayModeBar:false,staticPlot:true,scrollZoom:false,doubleClick:false}});
document.getElementById('flowTransitionOne').innerHTML=transitionHtml(d.addressable_pct,'Addressable to Accenture',d.addressable_label);document.getElementById('flowTransitionTwo').innerHTML=transitionHtml(d.accenture_share,'Accenture wins',d.accenture_label);
document.getElementById('donutLegend1').innerHTML=legendTitle('1 · Total Health Procurement')+legendItem('Not addressable',100-d.addressable_pct,d.not_addressable_label,'#D8CCF3')+legendItem('Addressable (TAM)',d.addressable_pct,d.addressable_label,'#4C1D95');
document.getElementById('donutLegend2').innerHTML=legendTitle('2 · Addressable Market (TAM)')+legendItem('Competitors',100-d.accenture_share,d.competitor_label,'#4C1D95')+legendItem('Accenture wins',d.accenture_share,d.accenture_label,'#9F7AEA');
document.getElementById('donutLegend3').innerHTML=legendTitle('3 · Accenture Wins')+legendItem('Accenture wins',100,d.accenture_label,'#9F7AEA');}}
function renderService(){{
  const d=DATA.service[selected('serviceRange')];
  // Keep the executive order: highest Accenture wins at the top.
  // Plotly renders the final horizontal category at the top, so sort ascending.
  const r=(d.rows||[]).slice().sort((a,b)=>(a.accenture-b.accenture)||(a.total-b.total));
  const accenture=r.map(x=>Number(x.accenture_b||0));
  const others=r.map(x=>Math.max(0,Number(x.total_b||0)-Number(x.accenture_b||0)));
  const mx=Math.max(.1,...r.map((x,i)=>Math.max(accenture[i],others[i])));

  function shortMoney(v){{
    v=Number(v||0)*1e9;
    const a=Math.abs(v);
    if(a>=1e9)return '$'+(v/1e9).toFixed(2)+'B';
    if(a>=1e6)return '$'+(v/1e6).toFixed(1)+'M';
    if(a>=1e3)return '$'+(v/1e3).toFixed(1)+'K';
    return '$'+v.toFixed(0);
  }}

  const accTrace={{
    type:'bar',orientation:'h',
    y:r.map(x=>x.name),x:accenture,
    name:'Accenture wins',
    marker:{{color:'#A100FF'}},
    offsetgroup:'accenture',
    text:r.map((x,i)=>{{
      const share=x.total>0?(x.accenture/x.total*100):0;
      return shortMoney(accenture[i])+'  ·  '+share.toFixed(1)+'% share';
    }}),
    textposition:'outside',cliponaxis:false,
    textfont:{{size:12,color:'#6B00B6'}},
    customdata:r.map((x,i)=>[x.total_label,sharePct(x),shortMoney(others[i])]),
    hovertemplate:'<b>%{{y}}</b><br>Accenture wins: %{{text}}<br>Total addressable segment: %{{customdata[0]}}<br>Other suppliers: %{{customdata[2]}}<extra></extra>'
  }};

  const othersTrace={{
    type:'bar',orientation:'h',
    y:r.map(x=>x.name),x:others,
    name:'Other addressable market',
    marker:{{color:'#D9DEE8'}},
    offsetgroup:'others',
    text:others.map(v=>shortMoney(v)),
    textposition:'outside',cliponaxis:false,
    textfont:{{size:12,color:'#344054'}},
    customdata:r.map(x=>x.total_label),
    hovertemplate:'<b>%{{y}}</b><br>Other addressable market: %{{text}}<br>Total addressable segment: %{{customdata}}<extra></extra>'
  }};

  function sharePct(x){{return x.total>0?(x.accenture/x.total*100):0;}}

  Plotly.react('serviceChart',[accTrace,othersTrace],{{
    height:Math.max(560,r.length*105+170),
    barmode:'group',bargap:.30,bargroupgap:.08,
    margin:{{l:300,r:155,t:28,b:92}},
    xaxis:{{
      title:'Contract value (A$ billions)',
      range:[0,mx*1.30],
      showgrid:true,gridcolor:'#EEF1F5',zeroline:true,zerolinecolor:'#98A2B3',
      tickfont:{{size:12}},titlefont:{{size:12,color:'#475467'}}
    }},
    yaxis:{{
      title:'',automargin:true,tickfont:{{size:13,color:'#1F2937'}},
      categoryorder:'array',categoryarray:r.map(x=>x.name)
    }},
    legend:{{orientation:'h',x:0,y:-0.14,xanchor:'left',yanchor:'top',font:{{size:12}}}},
    paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)'
  }},{{responsive:true,displayModeBar:false,staticPlot:true,scrollZoom:false,doubleClick:false}});
}}
function renderSupplier(){{const d=DATA.supplier[selected('supplierRange')],r=(d.rows||[]).slice().sort((a,b)=>a.share-b.share),mx=Math.max(.1,...r.map(x=>x.share));Plotly.react('supplierChart',[{{type:'bar',orientation:'h',y:r.map(x=>x.name),x:r.map(x=>x.share),marker:{{color:r.map(x=>x.name==='Accenture'?'#A100FF':'#667785')}},text:r.map(x=>fmtPct(x.share)),textposition:'outside',cliponaxis:false,customdata:r.map(x=>[x.value_label,x.contracts]),hovertemplate:'<b>%{{y}}</b><br>Share: %{{x:.1f}}%<br>Value: %{{customdata[0]}}<br>Contracts: %{{customdata[1]}}<extra></extra>'}}],{{height:760,margin:{{l:260,r:100,t:20,b:70}},xaxis:{{title:'Share of provisional TAM (%)',range:[0,mx*1.18],ticksuffix:'%'}}}},{{responsive:true,displayModeBar:false}});}}
[['donutRange',renderDonut],['serviceRange',renderService],['supplierRange',renderSupplier]].forEach(([n,f])=>document.querySelectorAll('input[name="'+n+'"]').forEach(x=>x.addEventListener('change',f)));renderDonut();renderService();renderSupplier();
</script></body></html>'''
    output.write_text(html, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="master_output/health_portfolio_contracts.parquet")
    p.add_argument("--output-dir", default="HealthTAMOverview_output")
    p.add_argument("--value-mode", choices=["total", "annualised"], default="total")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = load_health(Path(args.input))
    df = classify(df)
    value_col = VALUE_COL if args.value_mode == "total" else ANNUALISED_VALUE_COL
    if value_col not in df.columns:
        raise SystemExit(f"Missing value column: {value_col}")

    audit_cols = [c for c in [CN_ID_COL,AGENCY_COL,FY_COL,DESCRIPTION_COL,CATEGORY_TYPE_COL,CATEGORY_COL,SUPPLIER_COL,SUPPLIER_ABN_COL,"supplier_group",VALUE_COL,"is_addressable","addressability","addressability_reason","Service Offering"] if c in df.columns]
    df[audit_cols].sort_values(VALUE_COL,ascending=False).to_csv(outdir/"health_tam_classification_audit.csv",index=False)
    try:
        df.to_parquet(outdir/"health_tam_classified.parquet", index=False)
    except ImportError:
        df.to_csv(outdir/"health_tam_classified.csv", index=False)
        print("Parquet engine not installed; wrote classified CSV instead.")
    s = summary(df,value_col)
    pd.DataFrame([s]).to_csv(outdir/"health_market_summary.csv",index=False)
    dashboard = outdir/"AccentureHealthTAMOverview.html"
    build_html(df,value_col,dashboard)

    print("="*72)
    print("HEALTH TAM OVERVIEW - FIRST PASS")
    print("="*72)
    print(f"Rows loaded:              {len(df):,}")
    print(f"Total Health procurement: {money(s['total'])}")
    print(f"Provisional TAM:          {money(s['addressable'])} ({s['addressable_pct']:.1f}%)")
    print(f"Accenture TAM wins:       {money(s['accenture'])} ({s['accenture_share']:.1f}% share)")
    print(f"Dashboard:                {dashboard}")
    print(f"Classification audit:     {outdir/'health_tam_classification_audit.csv'}")
    print("="*72)


if __name__ == "__main__":
    main()
