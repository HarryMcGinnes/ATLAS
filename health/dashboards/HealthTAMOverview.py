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

    # Canonical supplier grouping is authoritative across all Health dashboards.
    if "supplier_group" not in out.columns:
        raise SystemExit("Missing canonical supplier_group; rebuild the Health master dataset.")
    out["supplier_group"] = out["supplier_group"].fillna("Unknown supplier").astype(str)
    if "is_accenture" in out.columns:
        out["is_accenture"] = _as_bool_series(out["is_accenture"])
    else:
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

def validate_health_tam_reconciliation(
    df: pd.DataFrame,
    value_col: str,
) -> None:
    """Validate Health KPI totals against All Years chart data."""

    full = summary(df, value_col)
    options = fy_options(df)

    all_years = next(
        (subset for label, subset in options if label == "All years"),
        None,
    )

    if all_years is None:
        raise RuntimeError("Health TAM: All years option missing")

    actual = summary(all_years, value_col)

    for metric in ["total", "addressable", "accenture"]:
        expected = float(full[metric])
        observed = float(actual[metric])

        if abs(expected - observed) > 0.01:
            raise RuntimeError(
                f"Health TAM reconciliation failed: {metric}. "
                f"Expected A${expected:,.2f}, "
                f"got A${observed:,.2f}"
            )

    print("\nHEALTH TAM DASHBOARD RECONCILIATION: PASS")
    print(f"Total procurement: A${full['total']/1e9:,.3f}B")
    print(f"Addressable TAM: A${full['addressable']/1e9:,.3f}B")
    print(f"Accenture wins: A${full['accenture']/1e9:,.3f}B")



def build_payloads(df: pd.DataFrame, value_col: str, market_mode: str = "tam") -> dict:
    """Build one complete payload from the already-selected market population.

    ``df`` is addressable-only for the Accenture TAM page and contains every
    Federal Health contract for the All Market page.  Every supplier and
    Service Offering chart therefore uses the same population as its page.
    The TAM decomposition remains visible as a clearly-labelled subset within
    the All Market view.
    """
    if market_mode not in {"tam", "all"}:
        raise ValueError(f"Unsupported market mode: {market_mode}")

    donut_sets, service_sets, supplier_sets = [], [], []
    for label, sub in fy_options(df):
        selected = sub.copy()
        s = summary(selected, value_col)

        addressable_accenture = selected[
            selected["is_addressable"] & selected["is_accenture"]
        ].copy()
        win_so = (
            addressable_accenture.groupby("Service Offering", dropna=False)[value_col]
            .sum()
            .sort_values(ascending=False)
        )
        donut_sets.append({
            "label": label,
            **s,
            "total_label": money(s["total"]),
            "addressable_label": money(s["addressable"]),
            "not_addressable_label": money(s["not_addressable"]),
            "accenture_label": money(s["accenture"]),
            "competitor_label": money(s["competitor"]),
            "offerings": [
                {
                    "name": str(k) if pd.notna(k) and str(k).strip() else "Uncategorised",
                    "value": float(v),
                    "label": money(v),
                    "colour": SO_COLOURS.get(str(k), "#DDD6FE"),
                }
                for k, v in win_so.items()
                if float(v) > 0
            ],
        })

        accenture_selected = selected[selected["is_accenture"]].copy()

        def _chart_so(frame: pd.DataFrame) -> pd.Series:
            cleaned = frame["Service Offering"].fillna("").astype(str).str.strip()
            bucketed = cleaned.where(cleaned.isin(SERVICE_OFFERINGS), "Uncategorised")
            if market_mode == "all":
                bucketed = bucketed.where(
                    frame["is_addressable"],
                    "Non-addressable / Unclassified",
                )
            return bucketed

        selected["_chart_service_offering"] = _chart_so(selected)
        accenture_selected["_chart_service_offering"] = _chart_so(accenture_selected)

        so_total = selected.groupby("_chart_service_offering")[value_col].sum()
        so_acc = accenture_selected.groupby("_chart_service_offering")[value_col].sum()
        chart_offerings = SERVICE_OFFERINGS + ["Uncategorised"]
        if market_mode == "all":
            chart_offerings.append("Non-addressable / Unclassified")

        service_rows = [{
            "name": so,
            "total": float(so_total.get(so, 0)),
            "accenture": float(so_acc.get(so, 0)),
            "total_b": float(so_total.get(so, 0)) / 1e9,
            "accenture_b": float(so_acc.get(so, 0)) / 1e9,
            "total_label": money(so_total.get(so, 0)),
            "accenture_label": money(so_acc.get(so, 0)),
        } for so in chart_offerings if float(so_total.get(so, 0)) > 0]

        # Plotly renders the final horizontal category at the top.
        service_rows.sort(key=lambda r: (r["accenture"], r["total"]))
        service_sets.append({"label": label, "rows": service_rows})

        supplier_summary = (
            selected.groupby("supplier_group", dropna=False)
            .agg(
                value=(value_col, "sum"),
                contracts=(CN_ID_COL, "nunique")
                if CN_ID_COL in selected.columns
                else (value_col, "size"),
            )
            .reset_index()
            .sort_values("value", ascending=False)
            .head(15)
        )
        selected_total = float(selected[value_col].sum())
        supplier_sets.append({
            "label": label,
            "rows": [
                {
                    "name": str(r.supplier_group),
                    "value": float(r.value),
                    "value_label": money(r.value),
                    "share": float(r.value) / selected_total * 100 if selected_total else 0,
                    "contracts": int(r.contracts),
                }
                for r in supplier_summary.itertuples()
            ],
        })

    return {"donut": donut_sets, "service": service_sets, "supplier": supplier_sets}

def controls(name: str, sets: list[dict]) -> str:
    return "".join(f'<label class="pill"><input type="radio" name="{name}" value="{i}" {"checked" if i==0 else ""}> {d["label"]}</label>' for i,d in enumerate(sets))



def build_html(
    df: pd.DataFrame,
    value_col: str,
    output: Path,
    market_mode: str = "tam",
) -> None:
    if market_mode not in {"tam", "all"}:
        raise ValueError(f"Unsupported market mode: {market_mode}")

    payload = build_payloads(df, value_col, market_mode=market_mode)
    s = summary(df, value_col)
    data = json.dumps(payload, ensure_ascii=False)

    is_tam = market_mode == "tam"
    market_label = "Accenture TAM" if is_tam else "All Federal Health Market"
    page_title = (
        "Accenture Health Addressable Market Dashboard"
        if is_tam
        else "Federal Health Procurement Market Dashboard"
    )
    page_subtitle = (
        "Showing only contracts classified as addressable to Accenture."
        if is_tam
        else "Showing every procurement contract in the Federal Health industry scope."
    )
    first_stage_title = "Accenture TAM" if is_tam else "Total Health Procurement Market"
    first_stage_subtitle = (
        "Selected addressable Health contracts"
        if is_tam
        else "All Federal Health procurement in scope"
    )
    service_title = (
        "Accenture Wins by Service Offering"
        if is_tam
        else "Selected Market by Service Offering"
    )
    service_copy = (
        "Each Service Offering is shown as Accenture wins in purple and the remaining addressable market in grey."
        if is_tam
        else "Each Service Offering is shown as Accenture awards in purple and the remaining selected market in grey. Non-addressable contracts are retained in an explicit bucket."
    )
    supplier_title = (
        "Top Suppliers by Share of Accenture-addressable Health TAM"
        if is_tam
        else "Top Suppliers by Share of All Federal Health Market"
    )
    selected_market_card_note = (
        "Contracts where is_addressable is True"
        if is_tam
        else "All Federal Health contracts in scope"
    )
    supplier_axis_title = (
        "Share of addressable TAM (%)"
        if is_tam
        else "Share of All Federal Health Market (%)"
    )

    tam_active = " active" if is_tam else ""
    all_active = " active" if not is_tam else ""
    tam_current = ' aria-current="page"' if is_tam else ""
    all_current = ' aria-current="page"' if not is_tam else ""

    market_controls = (
        '<section class="atlas-market-controls" aria-label="Market selection">'
        '<div><h2>Explore the Federal Health Market</h2>'
        '<p>Choose the procurement population for every KPI and visual on this page.</p></div>'
        '<div class="atlas-market-actions">'
        f'<a class="atlas-market-choice{tam_active}" href="AccentureHealthTAMOverview.html"{tam_current}>Accenture TAM</a>'
        f'<a class="atlas-market-choice{all_active}" href="AccentureHealthTAMOverview_AllMarket.html"{all_current}>All Federal Health Market</a>'
        '<button type="button" id="atlasClearFilters">Clear All Filters</button>'
        '</div></section>'
    )

    html = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>__DOCUMENT_TITLE__</title>
<script src="https://cdn.plot.ly/plotly-3.5.0.min.js"></script>
<style>
body{font-family:Arial,Helvetica,sans-serif;margin:0;background:#f6f8fb;color:#1f2937}.wrap{max-width:1480px;margin:auto;padding:28px}h1{margin:0 0 8px;font-size:30px}
.subtitle{color:#526070;margin:0 0 18px}.summary{background:#fff;border:1px solid #e5e7eb;border-left:5px solid #A100FF;border-radius:14px;padding:18px 20px;margin-bottom:18px;box-shadow:0 1px 2px rgba(0,0,0,.04)}
.summary h2{margin:0 0 8px;font-size:20px}.summary p{margin:5px 0;color:#475467;line-height:1.5}.warning{background:#fff7ed;border:1px solid #fed7aa;border-radius:12px;padding:12px 14px;color:#9a3412;margin:12px 0}
.atlas-market-controls{display:flex;justify-content:space-between;align-items:center;gap:18px;flex-wrap:wrap;padding:20px 24px;margin:18px 0 22px;background:white;border:1px solid #e5e7eb;border-radius:16px}.atlas-market-controls h2{margin:0 0 5px;font-size:20px}.atlas-market-controls p{margin:0;color:#667085;font-size:13px}.atlas-market-actions{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.atlas-market-choice,#atlasClearFilters{font:700 13px Arial,sans-serif;padding:12px 15px;border:1px solid #d0d5dd;border-radius:9px;background:#fff;color:#344054;text-decoration:none;cursor:pointer}.atlas-market-choice.active{background:#A100FF;color:white;border-color:#A100FF}#atlasClearFilters:hover{border-color:#A100FF;color:#7F00CC;background:#FCF5FF}
.cards{display:grid;grid-template-columns:repeat(3,minmax(220px,1fr));gap:14px;margin:18px 0 22px}.card,.panel{background:#fff;border:1px solid #e5e7eb;border-radius:14px;box-shadow:0 1px 2px rgba(0,0,0,.04)}.card{padding:17px}.ct{font-size:12px;text-transform:uppercase;color:#667085;letter-spacing:.04em}.cv{font-size:29px;font-weight:700;color:#111827;margin:8px 0 5px}.cs{font-size:13px;color:#667085}
.panel{padding:18px;margin-bottom:18px}.panel h3{margin:4px 0;font-size:18px}.panel p{margin:4px 0 10px;color:#526070;font-size:13px}.controls{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0}.pill{display:inline-flex;align-items:center;gap:6px;border:1px solid #d0d5dd;border-radius:999px;padding:7px 12px;background:#fff;font-size:12px;cursor:pointer}.pill:has(input:checked){background:#eef2ff;border-color:#636efa;color:#243b9f;font-weight:600}
.flow-head{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:0;text-align:center;margin:14px 0 0}.flow-head b{font-size:17px;line-height:1.25}.flow-head span{display:block;color:#667085;font-size:12px;margin-top:5px}.donut-stage{position:relative;height:430px;min-width:980px}#donutChart{height:430px;width:100%;margin:0 auto}.flow-transition{position:absolute;top:45%;width:118px;transform:translate(-50%,-50%);z-index:5;text-align:center;color:#5B21B6;pointer-events:none}.flow-transition.one{left:33.5%}.flow-transition.two{left:66.5%}.flow-transition .pct{font-size:18px;font-weight:800;line-height:1}.flow-transition .lbl{font-size:11px;line-height:1.2;margin-top:4px}.flow-transition .val{font-size:15px;font-weight:700;margin-top:4px}.flow-arrow{height:18px;position:relative;margin-top:8px}.flow-arrow:before{content:'';position:absolute;left:0;right:12px;top:8px;height:3px;border-radius:999px;background:#5B21B6}.flow-arrow:after{content:'';position:absolute;right:0;top:1px;border-top:8px solid transparent;border-bottom:8px solid transparent;border-left:13px solid #5B21B6}.donut-legends{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px;padding:0 24px 10px;min-width:980px}.donut-legend{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px 16px;align-items:start;padding:12px 14px 14px;border:1px solid #E2E8F0;border-radius:10px;background:#FAFBFD;min-height:70px}.donut-legend-title{grid-column:1/-1;font-size:12px;font-weight:800;color:#475467;text-transform:uppercase;letter-spacing:.04em;padding-bottom:6px;border-bottom:1px solid #E9EDF3;margin-bottom:2px}.legend-item{display:grid;grid-template-columns:14px minmax(0,1fr);gap:8px;align-items:start;font-size:12px;color:#344054}.legend-swatch{width:14px;height:14px;border-radius:3px;margin-top:1px}.legend-item b{display:block;font-size:12px;line-height:1.2}.legend-item small{display:block;color:#667085;margin-top:2px;line-height:1.2}.note{font-size:12px;color:#667085;margin-top:8px}@media(max-width:800px){.cards{grid-template-columns:1fr}.flow-scroll{overflow:auto}.atlas-market-actions{width:100%}.atlas-market-choice,#atlasClearFilters{flex:1;text-align:center}}
</style></head><body><div class="wrap">
<h1>__PAGE_TITLE__</h1><p class="subtitle">__PAGE_SUBTITLE__</p>
<section class="summary"><h2>Executive Summary</h2><p>Explore contracted value across the Federal Health portfolio, the addressable share, Accenture wins and the agencies, Service Offerings and suppliers underpinning the market.</p><div class="warning"><b>Classifier status:</b> classification decisions are read from the canonical Health master. ATLAS does not reclassify contracts in the dashboard. Review classification audit and methodology before treating TAM as a definitive addressable market estimate.</div></section>
__MARKET_CONTROLS__
<div class="cards">
<div class="card"><div class="ct">Selected market</div><div class="cv">__SELECTED_TOTAL__</div><div class="cs">__SELECTED_NOTE__</div></div>
<div class="card"><div class="ct">Addressable TAM</div><div class="cv">__ADDRESSABLE__</div><div class="cs">__ADDRESSABLE_PCT__% of selected market</div></div>
<div class="card"><div class="ct">Accenture wins in TAM</div><div class="cv">__ACCENTURE__</div><div class="cs">__ACCENTURE_SHARE__% of addressable TAM</div></div>
<div class="card"><div class="ct">Competitor-owned TAM</div><div class="cv">__COMPETITOR__</div><div class="cs">Addressable value awarded to other suppliers</div></div>
<div class="card"><div class="ct">Avg annual addressable</div><div class="cv">__AVG_ADDRESSABLE__</div><div class="cs">Across __YEARS__ financial years</div></div>
<div class="card"><div class="ct">Avg annual Accenture wins</div><div class="cv">__AVG_ACCENTURE__</div><div class="cs">Across __YEARS__ financial years</div></div>
</div>
<div class="panel"><h3>Total Addressable Market Overview</h3><div class="controls">__DONUT_CONTROLS__</div><div class="flow-scroll"><div class="flow-head"><div><b>1. __FIRST_STAGE_TITLE__</b><span>__FIRST_STAGE_SUBTITLE__</span></div><div><b>2. Within Addressable Market (TAM)</b><span>Accenture vs competitor-owned TAM</span></div><div><b>3. Accenture Wins</b><span>Accenture wins within the addressable Health market</span></div></div><div class="donut-stage"><div id="donutChart"></div><div id="flowTransitionOne" class="flow-transition one"></div><div id="flowTransitionTwo" class="flow-transition two"></div></div><div class="donut-legends"><div id="donutLegend1" class="donut-legend"></div><div id="donutLegend2" class="donut-legend"></div><div id="donutLegend3" class="donut-legend"></div></div></div></div>
<div class="panel"><h3>__SERVICE_TITLE__</h3><p>__SERVICE_COPY__</p><div class="controls">__SERVICE_CONTROLS__</div><div id="serviceChart"></div></div>
<div class="panel"><h3>__SUPPLIER_TITLE__</h3><div class="controls">__SUPPLIER_CONTROLS__</div><div id="supplierChart"></div></div>
<p class="note">Source: filtered AusTender Health portfolio dataset. Every visual uses the selected __MARKET_LABEL__ population unless a narrower TAM subset is explicitly labelled. Total contract value is used unless --value-mode annualised is selected.</p>
</div><script>
const DATA=__DATA__;
const MARKET_MODE=__MARKET_MODE__;
const MARKET_LABEL=__MARKET_LABEL_JSON__;
const FIRST_STAGE_TITLE=__FIRST_STAGE_TITLE_JSON__;
const SUPPLIER_AXIS_TITLE=__SUPPLIER_AXIS_TITLE__;
const FILTER_STORAGE_KEY='atlas-health-tam-overview-filters-v1';
const FILTER_NAMES=['donutRange','serviceRange','supplierRange'];
function selected(name){const e=document.querySelector('input[name="'+name+'"]:checked');return e?Number(e.value):0;}
function fmtPct(v){return Number(v||0).toFixed(1)+'%';}
function money(v){v=Number(v||0);let a=Math.abs(v);if(a>=1e9)return '$'+(v/1e9).toFixed(2)+'B';if(a>=1e6)return '$'+(v/1e6).toFixed(1)+'M';if(a>=1e3)return '$'+(v/1e3).toFixed(1)+'K';return '$'+v.toFixed(0);}
function legendItem(title,pct,value,colour){return '<div class="legend-item"><span class="legend-swatch" style="background:'+colour+'"></span><div><b>'+title+'</b><small>'+fmtPct(pct)+' · '+value+'</small></div></div>';}
function legendTitle(text){return '<div class="donut-legend-title">'+text+'</div>';}
function transitionHtml(pct,label,value){return '<div class="pct">'+fmtPct(pct)+'</div><div class="lbl">'+label+'</div><div class="val">'+value+'</div><div class="flow-arrow"></div>';}
function saveFilters(){
  const state={};
  FILTER_NAMES.forEach(name=>{const e=document.querySelector('input[name="'+name+'"]:checked');state[name]=e?e.value:'0';});
  try{sessionStorage.setItem(FILTER_STORAGE_KEY,JSON.stringify(state));}catch(_error){}
}
function restoreFilters(){
  let state={};
  try{state=JSON.parse(sessionStorage.getItem(FILTER_STORAGE_KEY)||'{}')||{};}catch(_error){state={};}
  FILTER_NAMES.forEach(name=>{
    const desired=String(state[name]??'0');
    const option=document.querySelector('input[name="'+name+'"][value="'+desired+'"]');
    const first=document.querySelector('input[name="'+name+'"]');
    if(option)option.checked=true;else if(first)first.checked=true;
  });
}
function renderDonut(){const d=DATA.donut[selected('donutRange')];const tr=[
{type:'pie',hole:.64,labels:['Addressable (TAM)','Not addressable'],values:[d.addressable,d.not_addressable],domain:{x:[.035,.298],y:[.14,.90]},sort:false,direction:'clockwise',rotation:90-((d.addressable/Math.max(d.total,1))*360/2),marker:{colors:['#4C1D95','#D8CCF3'],line:{color:'#fff',width:3}},textinfo:'none',showlegend:false,hovertemplate:'<b>%{label}</b><br>%{value:$,.0f}<br>%{percent}<extra></extra>'},
{type:'pie',hole:.64,labels:['Accenture wins','Competitors'],values:[d.accenture,d.competitor],domain:{x:[.368,.632],y:[.14,.90]},sort:false,direction:'clockwise',rotation:90-((d.accenture/Math.max(d.addressable,1))*360/2),marker:{colors:['#9F7AEA','#4C1D95'],line:{color:'#fff',width:3}},textinfo:'none',showlegend:false,hovertemplate:'<b>%{label}</b><br>%{value:$,.0f}<br>%{percent}<extra></extra>'},
{type:'pie',hole:.64,labels:['Accenture wins'],values:[d.accenture],domain:{x:[.702,.965],y:[.14,.90]},sort:false,marker:{colors:['#9F7AEA'],line:{color:'#fff',width:3}},textinfo:'none',showlegend:false,hovertemplate:'<b>Accenture wins</b><br>%{value:$,.0f}<extra></extra>'}
];const ann=[
{x:.1665,y:.52,xref:'paper',yref:'paper',text:'<b>'+d.total_label+'</b><br><span style="font-size:11px;color:#526070">'+FIRST_STAGE_TITLE+'</span>',showarrow:false,xanchor:'center',yanchor:'middle',align:'center',font:{size:24,color:'#344054'}},
{x:.5000,y:.52,xref:'paper',yref:'paper',text:'<b>'+d.addressable_label+'</b><br><span style="font-size:11px;color:#526070">Addressable TAM</span>',showarrow:false,xanchor:'center',yanchor:'middle',align:'center',font:{size:24,color:'#344054'}},
{x:.8335,y:.52,xref:'paper',yref:'paper',text:'<b>'+d.accenture_label+'</b><br><span style="font-size:11px;color:#526070">Accenture Wins</span>',showarrow:false,xanchor:'center',yanchor:'middle',align:'center',font:{size:24,color:'#344054'}}
];
Plotly.react('donutChart',tr,{height:430,margin:{t:4,l:4,r:4,b:4},showlegend:false,paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',annotations:ann},{responsive:true,displayModeBar:false,staticPlot:true,scrollZoom:false,doubleClick:false});
document.getElementById('flowTransitionOne').innerHTML=transitionHtml(d.addressable_pct,'Addressable to Accenture',d.addressable_label);document.getElementById('flowTransitionTwo').innerHTML=transitionHtml(d.accenture_share,'Accenture wins',d.accenture_label);
document.getElementById('donutLegend1').innerHTML=legendTitle('1 · '+FIRST_STAGE_TITLE)+legendItem('Not addressable',100-d.addressable_pct,d.not_addressable_label,'#D8CCF3')+legendItem('Addressable (TAM)',d.addressable_pct,d.addressable_label,'#4C1D95');
document.getElementById('donutLegend2').innerHTML=legendTitle('2 · Addressable Market (TAM)')+legendItem('Competitors',100-d.accenture_share,d.competitor_label,'#4C1D95')+legendItem('Accenture wins',d.accenture_share,d.accenture_label,'#9F7AEA');
document.getElementById('donutLegend3').innerHTML=legendTitle('3 · Accenture Wins')+legendItem('Accenture wins',100,d.accenture_label,'#9F7AEA');}
function renderService(){
  const d=DATA.service[selected('serviceRange')];
  const r=(d.rows||[]).slice().sort((a,b)=>(a.accenture-b.accenture)||(a.total-b.total));
  const accenture=r.map(x=>Number(x.accenture_b||0));
  const others=r.map(x=>Math.max(0,Number(x.total_b||0)-Number(x.accenture_b||0)));
  const mx=Math.max(.1,...r.map((x,i)=>Math.max(accenture[i],others[i])));
  function shortMoney(v){v=Number(v||0)*1e9;const a=Math.abs(v);if(a>=1e9)return '$'+(v/1e9).toFixed(2)+'B';if(a>=1e6)return '$'+(v/1e6).toFixed(1)+'M';if(a>=1e3)return '$'+(v/1e3).toFixed(1)+'K';return '$'+v.toFixed(0);}
  function sharePct(x){return x.total>0?(x.accenture/x.total*100):0;}
  const otherName=MARKET_MODE==='tam'?'Other addressable market':'Other selected market';
  const segmentLabel=MARKET_MODE==='tam'?'Total addressable segment':'Total selected-market segment';
  const accTrace={type:'bar',orientation:'h',y:r.map(x=>x.name),x:accenture,name:'Accenture awards',marker:{color:'#A100FF'},offsetgroup:'accenture',text:r.map((x,i)=>{const share=x.total>0?(x.accenture/x.total*100):0;return shortMoney(accenture[i])+'  ·  '+share.toFixed(1)+'% share';}),textposition:'outside',cliponaxis:false,textfont:{size:12,color:'#6B00B6'},customdata:r.map((x,i)=>[x.total_label,sharePct(x),shortMoney(others[i])]),hovertemplate:'<b>%{y}</b><br>Accenture awards: %{text}<br>'+segmentLabel+': %{customdata[0]}<br>'+otherName+': %{customdata[2]}<extra></extra>'};
  const othersTrace={type:'bar',orientation:'h',y:r.map(x=>x.name),x:others,name:otherName,marker:{color:'#D9DEE8'},offsetgroup:'others',text:others.map(v=>shortMoney(v)),textposition:'outside',cliponaxis:false,textfont:{size:12,color:'#344054'},customdata:r.map(x=>x.total_label),hovertemplate:'<b>%{y}</b><br>'+otherName+': %{text}<br>'+segmentLabel+': %{customdata}<extra></extra>'};
  Plotly.react('serviceChart',[accTrace,othersTrace],{height:Math.max(560,r.length*105+170),barmode:'group',bargap:.30,bargroupgap:.08,margin:{l:300,r:155,t:28,b:92},xaxis:{title:'Contract value (A$ billions)',range:[0,mx*1.30],showgrid:true,gridcolor:'#EEF1F5',zeroline:true,zerolinecolor:'#98A2B3',tickfont:{size:12},titlefont:{size:12,color:'#475467'}},yaxis:{title:'',automargin:true,tickfont:{size:13,color:'#1F2937'},categoryorder:'array',categoryarray:r.map(x=>x.name)},legend:{orientation:'h',x:0,y:-0.14,xanchor:'left',yanchor:'top',font:{size:12}},paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)'},{responsive:true,displayModeBar:false,staticPlot:true,scrollZoom:false,doubleClick:false});
}
function renderSupplier(){const d=DATA.supplier[selected('supplierRange')],r=(d.rows||[]).slice().sort((a,b)=>a.share-b.share),mx=Math.max(.1,...r.map(x=>x.share));Plotly.react('supplierChart',[{type:'bar',orientation:'h',y:r.map(x=>x.name),x:r.map(x=>x.share),marker:{color:r.map(x=>x.name==='Accenture'?'#A100FF':'#667785')},text:r.map(x=>fmtPct(x.share)),textposition:'outside',cliponaxis:false,customdata:r.map(x=>[x.value_label,x.contracts]),hovertemplate:'<b>%{y}</b><br>Share: %{x:.1f}%<br>Value: %{customdata[0]}<br>Contracts: %{customdata[1]}<extra></extra>'}],{height:760,margin:{l:260,r:100,t:20,b:70},xaxis:{title:SUPPLIER_AXIS_TITLE,range:[0,mx*1.18],ticksuffix:'%'}},{responsive:true,displayModeBar:false});}
function renderAll(){renderDonut();renderService();renderSupplier();}
restoreFilters();
[['donutRange',renderDonut],['serviceRange',renderService],['supplierRange',renderSupplier]].forEach(([n,f])=>document.querySelectorAll('input[name="'+n+'"]').forEach(x=>x.addEventListener('change',()=>{saveFilters();f();})));
document.querySelectorAll('.atlas-market-choice').forEach(link=>link.addEventListener('click',saveFilters));
document.getElementById('atlasClearFilters').addEventListener('click',()=>{FILTER_NAMES.forEach(name=>{const first=document.querySelector('input[name="'+name+'"]');if(first)first.checked=true;});saveFilters();renderAll();});
renderAll();
</script></body></html>"""

    replacements = {
        "__DOCUMENT_TITLE__": page_title,
        "__PAGE_TITLE__": page_title,
        "__PAGE_SUBTITLE__": page_subtitle,
        "__MARKET_CONTROLS__": market_controls,
        "__SELECTED_TOTAL__": money(s["total"]),
        "__SELECTED_NOTE__": selected_market_card_note,
        "__ADDRESSABLE__": money(s["addressable"]),
        "__ADDRESSABLE_PCT__": f"{s['addressable_pct']:.1f}",
        "__ACCENTURE__": money(s["accenture"]),
        "__ACCENTURE_SHARE__": f"{s['accenture_share']:.1f}",
        "__COMPETITOR__": money(s["competitor"]),
        "__AVG_ADDRESSABLE__": money(s["avg_addressable"]),
        "__AVG_ACCENTURE__": money(s["avg_accenture"]),
        "__YEARS__": str(s["years"]),
        "__FIRST_STAGE_TITLE__": first_stage_title,
        "__FIRST_STAGE_SUBTITLE__": first_stage_subtitle,
        "__SERVICE_TITLE__": service_title,
        "__SERVICE_COPY__": service_copy,
        "__SUPPLIER_TITLE__": supplier_title,
        "__MARKET_LABEL__": market_label,
        "__DONUT_CONTROLS__": controls("donutRange", payload["donut"]),
        "__SERVICE_CONTROLS__": controls("serviceRange", payload["service"]),
        "__SUPPLIER_CONTROLS__": controls("supplierRange", payload["supplier"]),
        "__DATA__": data,
        "__MARKET_MODE__": json.dumps(market_mode),
        "__MARKET_LABEL_JSON__": json.dumps(market_label),
        "__FIRST_STAGE_TITLE_JSON__": json.dumps(first_stage_title),
        "__SUPPLIER_AXIS_TITLE__": json.dumps(supplier_axis_title),
    }
    for token, value in replacements.items():
        html = html.replace(token, str(value))

    final_html = atlas_overview_extras(
        atlas_navigation(html, "health", page="overview", freshness=atlas_freshness(df)),
        df,
        value_col,
        market_mode,
    )
    output.write_text(atlas_option_a_style(final_html, "overview"), encoding="utf-8")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="master_output/health_portfolio_contracts.parquet")
    p.add_argument("--output-dir", default="HealthTAMOverview_output")
    p.add_argument("--value-mode", choices=["total", "annualised"], default="total")
    p.add_argument("--health-home", default="docs/health/index.html", help="Health landing HTML output, relative to repo root")
    p.add_argument("--skip-health-home", action="store_true", help="Do not rebuild the Health landing page")
    return p.parse_args()



def main() -> None:
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = classify(load_health(Path(args.input)))
    value_col = VALUE_COL if args.value_mode == "total" else ANNUALISED_VALUE_COL
    if value_col not in df.columns:
        raise SystemExit(f"Missing value column: {value_col}")

    validate_health_tam_reconciliation(df, value_col)

    audit_cols = [
        c for c in [
            CN_ID_COL,
            AGENCY_COL,
            FY_COL,
            DESCRIPTION_COL,
            CATEGORY_TYPE_COL,
            CATEGORY_COL,
            SUPPLIER_COL,
            SUPPLIER_ABN_COL,
            "supplier_group",
            VALUE_COL,
            "is_addressable",
            "addressability",
            "addressability_reason",
            "Service Offering",
        ]
        if c in df.columns
    ]
    df[audit_cols].sort_values(VALUE_COL, ascending=False).to_csv(
        outdir / "health_tam_classification_audit.csv",
        index=False,
    )
    try:
        df.to_parquet(outdir / "health_tam_classified.parquet", index=False)
    except ImportError:
        df.to_csv(outdir / "health_tam_classified.csv", index=False)
        print("Parquet engine not installed; wrote classified CSV instead.")

    all_summary = summary(df, value_col)
    tam_df = df.loc[df["is_addressable"]].copy()
    if tam_df.empty:
        raise SystemExit("Cannot generate Accenture TAM view: no addressable Health contracts.")
    tam_summary = summary(tam_df, value_col)

    pd.DataFrame([all_summary]).to_csv(
        outdir / "health_market_summary.csv",
        index=False,
    )
    pd.DataFrame([tam_summary]).to_csv(
        outdir / "health_market_summary_tam.csv",
        index=False,
    )
    pd.DataFrame([all_summary]).to_csv(
        outdir / "health_market_summary_all_market.csv",
        index=False,
    )

    tam_dashboard = outdir / "AccentureHealthTAMOverview.html"
    all_dashboard = outdir / "AccentureHealthTAMOverview_AllMarket.html"
    build_html(tam_df, value_col, tam_dashboard, market_mode="tam")
    build_html(df, value_col, all_dashboard, market_mode="all")
    if not args.skip_health_home:
        build_health_home(df, VALUE_COL, Path(args.health_home))

    print("=" * 72)
    print("HEALTH TAM OVERVIEW - CANONICAL DATA")
    print("=" * 72)
    print(f"Rows loaded:              {len(df):,}")
    print(f"Total Health procurement: {money(all_summary['total'])}")
    print(
        f"Provisional TAM:          {money(all_summary['addressable'])} "
        f"({all_summary['addressable_pct']:.1f}%)"
    )
    print(
        f"Accenture TAM wins:       {money(all_summary['accenture'])} "
        f"({all_summary['accenture_share']:.1f}% share)"
    )
    print(f"TAM dashboard:            {tam_dashboard}")
    print(f"All Market dashboard:     {all_dashboard}")
    print(f"Classification audit:     {outdir / 'health_tam_classification_audit.csv'}")
    print("=" * 72)



def atlas_freshness(df: pd.DataFrame) -> str:
    """Expose provenance without mistaking HTML build time for source refresh time."""
    from datetime import datetime, timezone
    from html import escape
    import os

    from zoneinfo import ZoneInfo

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
        ("Methodology", "index.html#atlas-methodology", "methodology"),
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



def atlas_quality_panel(df: pd.DataFrame, value_col: str, market_mode: str) -> str:
    """Show coverage / review statistics for the *selected* market population."""
    from html import escape
    count = len(df)
    unique_cn = int(df['CN ID'].nunique()) if 'CN ID' in df.columns else count
    duplicate_count = count - unique_cn
    status = (df['cn_family_date_status'].fillna('').astype(str).str.strip()
              if 'cn_family_date_status' in df.columns else None)
    if status is not None:
        review = int(status.str.startswith('review_').sum())
        inferred = int(status.str.startswith('inferred_').sum())
        status_text = f'{count-review:,} of {count:,} records not flagged for date review'
        inferred_text = f'{inferred:,} inferred date records'
    else:
        review = None
        status_text = 'Contract-family quality fields unavailable'
        inferred_text = 'No family-date status field in input'
    so = df.get('Service Offering')
    if so is None:
        unclassified_text = 'Not available'
    else:
        unknown = (so.isna() | so.astype(str).str.strip().str.lower().isin(['', 'unclassified', 'uncategorised', 'unresolved', 'none'])) & df['is_addressable']
        total_unclassified = float(pd.to_numeric(df.loc[unknown, value_col], errors='coerce').fillna(0).sum())
        unclassified_text = money(total_unclassified)
    label = 'Accenture TAM' if market_mode == 'tam' else 'All Federal Health Market'
    return f"""<section class="atlas-health-quality" aria-label="Data quality and coverage">
      <div class="atlas-quality-title"><h2>Data coverage & quality</h2><p>Selected scope: {escape(label)} · Review indicators, not a guarantee of perfect source records.</p></div>
      <div class="atlas-quality-metrics">
        <div><strong>{count:,}</strong><span>Contract notifications</span></div>
        <div><strong>{duplicate_count:,}</strong><span>Duplicate CN IDs</span></div>
        <div><strong>{review if review is not None else 'n/a'}</strong><span>Date review flags</span></div>
        <div><strong>{unclassified_text}</strong><span>Unclassified contracted value</span></div>
      </div><p class="atlas-quality-note">{escape(status_text)} · {escape(inferred_text)}. Inferred amendment periods are observed date ranges, not verified legal contract terms.</p>
    </section>"""


def atlas_methodology_panel() -> str:
    """Methodology explanation, distinguishing verified aggregation from scoring assumptions."""
    return r"""<section id="atlas-methodology" class="atlas-health-methodology" aria-labelledby="atlasMethodHeading">
      <div class="atlas-method-heading"><div><p class="atlas-eyebrow">HOW ATLAS DEFINES THE MARKET</p><h2 id="atlasMethodHeading">Classification methodology</h2>
      <p>Follow a contract from AusTender through industry scope, addressability and Service Offering allocation.</p></div></div>
      <div class="atlas-method-steps" role="list">
        <div role="listitem"><span class="atlas-step-num">01</span><b>Federal Health scope</b><small>Keep contract notifications from the configured Health agency set and aliases.</small></div>
        <div role="listitem"><span class="atlas-step-num">02</span><b>Addressability</b><small>Apply upstream exclusion and positive-evidence rules to assign the canonical <code>is_addressable</code> flag.</small></div>
        <div role="listitem"><span class="atlas-step-num">03</span><b>Service Offering</b><small>Assign addressable work to a service area using the upstream classifier; retain unresolved cases as unclassified.</small></div>
        <div role="listitem"><span class="atlas-step-num">04</span><b>Publish & audit</b><small>Sum recorded contract values, retain each CN, and expose review fields where available.</small></div>
      </div>
      <div class="atlas-formula-grid">
        <div><h3>Addressable TAM calculation</h3><code>TAM = Σ Value(CN), where is_addressable = True</code>
        <p>Uses the canonical addressability flag from the Health master, not a reclassification inside this dashboard.</p></div>
        <div><h3>Service Offering allocation</h3><code>Offering value = Σ Value(CN), where is_addressable = True and service_offering = selected offering</code>
        <p>Each contract retains its assigned Service Offering; unresolved classifications are shown separately.</p></div>
      </div>
      <details><summary>What evidence informs classification?</summary><p>Classification examines the procurement category and type, contract description and supplier context. For example, physical clinical goods are typically excluded; evidence of advisory, digital, data, integration, cloud, cyber and managed services can support addressability. Exact precedence and exceptions are defined upstream, not in these dashboard files.</p></details>
      <details><summary>How are weights and confidence calculated?</summary><p>ATLAS consumes the result of the current Health classifier. Its complete scoring function and numeric weights must be verified from <code>atlas/gold/health</code> and the Health master-building scripts before publishing exact percentages or thresholds. No unverified scoring formula is presented here.</p></details>
      <details><summary>How are amendments handled?</summary><p>Amendments remain separate Contract Notice records. When the original CN is missing, the displayed inferred family period uses the earliest credible start and latest credible end among available family members. This range is not necessarily the legally current contract period.</p></details>
      <p class="atlas-method-caveat">Source: Australian Government AusTender contract notices. Measures represent the sum of recorded contract values, not annual government expenditure or a live pipeline forecast. Public procurement reporting has coverage and amendment limitations.</p>
    </section>"""


def atlas_agency_breakdown(html_text: str, df: pd.DataFrame, value_col: str, market_mode: str = 'tam') -> str:
    """Add an FY-linked agency chart using exactly the market population displayed."""
    if 'Agency' not in df.columns:
        return html_text
    per_period = []
    for label, subset in fy_options(df):
        summary_frame = subset.copy()
        summary_frame['Agency'] = summary_frame['Agency'].fillna('Unspecified agency').astype(str).str.strip().replace('', 'Unspecified agency')
        summary_frame[value_col] = pd.to_numeric(summary_frame[value_col], errors='coerce').fillna(0)
        grouped = summary_frame.groupby('Agency', dropna=False)[value_col].agg(['sum', 'size'])
        per_period.append({
            'period': label,
            'rows': [{'agency': str(a), 'value': float(row['sum']), 'contracts': int(row['size'])}
                     for a, row in grouped.iterrows()]
        })
    label = 'Accenture TAM' if market_mode == 'tam' else 'All Federal Health Market'
    from html import escape
    panel = f"""<section class="atlas-health-agencies" aria-labelledby="atlasAgencyHeading">
      <div class="atlas-agency-heading"><div><p class="atlas-eyebrow">FEDERAL HEALTH SPENDING</p><h2 id="atlasAgencyHeading">{escape(label)} by Agency</h2>
      <p>Share of recorded contract value by agency. Follows the TAM overview financial-year selection above.</p></div>
      <div class="atlas-agency-total" id="atlasAgencyTotal">-</div></div>
      <div id="atlasAgencyChart" style="width:100%;min-height:400px"></div>
      <p class="atlas-chart-note">Shares use the selected market's total value as the denominator. Values are AusTender contract awards, not annual expenditure.</p>
    </section>"""
    script = r"""<script>(function(){
      const periods = __DATA__;
      function renderAgency(){
        const control = document.querySelector('input[name="donutRange"]:checked');
        const i = control ? Number(control.value) : 0;
        const p = periods[i] || periods[0];
        if(!p){return;}
        const list = p.rows.map(r=>({agency:String(r.agency),value:Number(r.value||0),contracts:Number(r.contracts||0)}))
          .filter(r=>r.value!==0).sort((a,b)=>b.value-a.value);
        const total = list.reduce((s,r)=>s+r.value,0);
        const el = document.getElementById('atlasAgencyTotal');
        if(el){el.textContent = 'Total: A$' + (Math.abs(total)>=1e9 ? (total/1e9).toFixed(2)+'B' : (total/1e6).toFixed(1)+'M');}
        if(!window.Plotly){return;}
        const ascending = list.slice().reverse();
        const height = Math.max(390,ascending.length*42+100);
        Plotly.react('atlasAgencyChart',[{type:'bar',orientation:'h',x:ascending.map(r=>r.value/1e9),y:ascending.map(r=>r.agency),
          marker:{color:'#7951ae'},text:ascending.map(r=>total?((r.value/total)*100).toFixed(1)+'%':'n/a'),textposition:'outside',cliponaxis:false,
          customdata:ascending.map(r=>[r.value,r.contracts]),
          hovertemplate:'<b>%{y}</b><br>Contract value: A$%{customdata[0]:,.0f}<br>CNs: %{customdata[1]:,}<br>Market share: %{text}<extra></extra>'}],
          {template:'plotly_white',height,margin:{t:22,b:54,l:330,r:100},xaxis:{title:'Contract value (A$ billions)',zeroline:true},
           yaxis:{automargin:true},showlegend:false},{responsive:true,displayModeBar:false});
      }
      document.querySelectorAll('input[name="donutRange"]').forEach(el=>el.addEventListener('change',renderAgency));
      renderAgency();
    })();</script>""".replace('__DATA__', json.dumps(per_period, ensure_ascii=False, allow_nan=False))
    position = '<div class="panel"><h3>'
    if position in html_text:
        html_text = html_text.replace(position, panel + position, 1)
    else:
        html_text = html_text.replace('</body>', panel + '</body>', 1)
    return html_text.replace('</body>', script + '</body>', 1)


def atlas_overview_extras(html_text: str, df: pd.DataFrame, value_col: str, market_mode: str) -> str:
    css = """<style>
    .atlas-health-quality,.atlas-health-agencies,.atlas-health-methodology{max-width:1480px;margin:22px auto;background:white;border:1px solid #e6e8f0;border-radius:16px;padding:24px;box-shadow:0 1px 3px #11182708}
    .atlas-health-quality h2,.atlas-health-agencies h2,.atlas-health-methodology h2{margin:0 0 6px;font-size:22px;color:#171326}
    .atlas-quality-title p,.atlas-agency-heading p,.atlas-method-heading p{margin:0 0 10px;color:#596477;font-size:13px;line-height:1.5}
    .atlas-quality-metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-top:15px}
    .atlas-quality-metrics div{background:#f7f5fb;border:1px solid #eae6f1;border-radius:11px;padding:16px;display:flex;flex-direction:column;gap:5px}
    .atlas-quality-metrics strong{font-size:23px;font-variant-numeric:tabular-nums}.atlas-quality-metrics span{font-size:12px;color:#667085}
    .atlas-quality-note,.atlas-chart-note,.atlas-method-caveat{color:#667085;font-size:12px;line-height:1.55}
    .atlas-agency-heading{display:flex;justify-content:space-between;align-items:center;gap:20px;flex-wrap:wrap}
    .atlas-agency-total{font-size:17px;font-weight:800;color:#5e258c}
    .atlas-eyebrow{font-size:11px!important;color:#7b42ab!important;font-weight:800;letter-spacing:.095em;margin-bottom:6px!important}
    .atlas-method-steps{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:20px 0}
    .atlas-method-steps div{display:flex;flex-direction:column;gap:9px;border:1px solid #e5dfed;border-radius:12px;padding:16px;background:#faf9fc}
    .atlas-step-num{font-size:13px;font-weight:800;color:#7c3aed}.atlas-method-steps b{font-size:14px}.atlas-method-steps small{font-size:12px;line-height:1.5;color:#596477}
    .atlas-formula-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:15px;margin:15px 0}
    .atlas-formula-grid>div{background:#f6f3fa;padding:16px;border-radius:12px}.atlas-formula-grid h3{font-size:15px;margin:0 0 12px}
    .atlas-formula-grid code{display:block;white-space:normal;overflow-wrap:anywhere;font-size:12px;color:#5c2b83;line-height:1.5}.atlas-formula-grid p{color:#667085;font-size:12px;line-height:1.5}
    .atlas-health-methodology details{border-top:1px solid #ebe7f0;padding:14px 4px}.atlas-health-methodology details summary{cursor:pointer;color:#3b2d4c;font-weight:700;font-size:13px}
    .atlas-health-methodology details p{font-size:13px;line-height:1.65;color:#596477;margin:10px 0 0}
    @media(max-width:800px){.atlas-quality-metrics,.atlas-method-steps{grid-template-columns:repeat(2,minmax(0,1fr))}.atlas-formula-grid{grid-template-columns:1fr}.atlas-health-quality,.atlas-health-agencies,.atlas-health-methodology{padding:15px}}
    @media(max-width:490px){.atlas-quality-metrics,.atlas-method-steps{grid-template-columns:1fr}}
    </style>"""
    html_text = html_text.replace('</head>', css+'</head>', 1)
    quality = atlas_quality_panel(df, value_col, market_mode)
    # Put data-quality indicators before the KPI cards so they are visible immediately.
    html_text = html_text.replace('<div class="cards">', quality+'<div class="cards">', 1)
    html_text = atlas_agency_breakdown(html_text, df, value_col, market_mode)
    methodology = atlas_methodology_panel()
    html_text = html_text.replace('</body>', methodology+'</body>', 1)
    return html_text


def build_health_home(df: pd.DataFrame, value_col: str, output_path: Path) -> None:
    """Build the Health homepage from canonical master fields (no reclassification).

    The existing docs/health/index.html is replaced only when this function is
    explicitly called by the Health TAM Overview build. A Git commit retains
    the old page for straightforward rollback.
    """
    from html import escape

    def market_payload(frame: pd.DataFrame) -> dict:
        frame = frame.copy()
        amount = pd.to_numeric(frame[value_col], errors="coerce").fillna(0)
        total = float(amount.sum())
        agencies = frame.get("Agency", pd.Series("Unspecified", index=frame.index))
        agency_frame = pd.DataFrame({
            "agency": agencies.fillna("Unspecified agency").astype(str).str.strip().replace("", "Unspecified agency"),
            "value": amount,
        })
        grouped = agency_frame.groupby("agency", dropna=False)["value"].agg(["sum", "size"]).sort_values("sum", ascending=False)
        agency_data = [
            {"agency": str(name), "value": float(row["sum"]), "count": int(row["size"])}
            for name, row in grouped.iterrows()
        ]
        # Never publish agency figures that do not reconcile to the selected market.
        if abs(sum(row["value"] for row in agency_data) - total) > max(0.01, abs(total) * 1e-12):
            raise RuntimeError("Health homepage agency totals do not reconcile to selected market")
        accenture = frame["is_accenture"] if "is_accenture" in frame.columns else frame["supplier_group"].eq("Accenture")
        acc_value = float(amount[accenture].sum())
        status = frame["cn_family_date_status"].fillna("").astype(str) if "cn_family_date_status" in frame else None
        review = int(status.str.startswith("review_").sum()) if status is not None else None
        inferred = int(status.str.startswith("inferred_").sum()) if status is not None else None
        cls = frame.get("Service Offering")
        unknown_count = (int(((cls.isna() | cls.astype(str).str.strip().str.lower().isin(["", "unclassified", "uncategorised", "unresolved", "none"])) & frame["is_addressable"]).sum()) if cls is not None else None)
        return {
            "total": total,
            "count": len(frame),
            "agency_count": int(len(grouped)),
            "accenture_value": acc_value,
            "agencies": agency_data,
            "review": review,
            "inferred": inferred,
            "unclassified": unknown_count,
        }

    tam = df.loc[df["is_addressable"]].copy()
    if tam.empty:
        raise RuntimeError("Cannot build Health landing page: no addressable contracts")
    payload = {"tam": market_payload(tam), "all": market_payload(df)}
    source_meta = atlas_freshness(df)
    # Avoid unsafe HTML/script injection from raw agency names inside JSON.
    safe_data = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    page = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
    <title>ATLAS | Federal Health Intelligence</title>
    <style>
    :root{--ink:#171427;--muted:#606779;--purple:#6d28d9;--border:#e7e4ef;--bg:#f7f6f9}
    *{box-sizing:border-box} body{font-family:Arial,Helvetica,sans-serif;margin:0;background:var(--bg);color:var(--ink)}
    a{color:inherit;text-decoration:none} .top{background:#12101c;color:white}.nav{max-width:1180px;margin:auto;padding:16px 24px;display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap}
    .brand{font-size:24px;font-weight:900;letter-spacing:.06em}.nav-links{display:flex;align-items:center;gap:6px;flex-wrap:wrap}
    .nav-links a{font-size:13px;color:#f0edf7;padding:9px 12px;border-radius:8px;font-weight:700}.nav-links a:hover,.nav-links a[aria-current=page]{background:#63369c}
    .meta{max-width:1180px;margin:auto;padding:0 24px 14px;font-size:11px;line-height:1.5;color:#d0c8df}
    main{max-width:1180px;margin:auto;padding:34px 24px 70px} .hero{margin:0 0 23px}.eyebrow{font-size:11px;color:#7535ac;font-weight:900;letter-spacing:.13em;margin:0 0 8px}
    h1{font-size:clamp(29px,4.5vw,43px);margin:0 0 12px;letter-spacing:-.03em} h2{font-size:23px;margin:0 0 9px} h3{font-size:15px;margin:0 0 7px}
    p{line-height:1.55}.lead{max-width:840px;font-size:16px;color:#50596a;margin:0 0 21px}
    .panel{background:white;border:1px solid var(--border);border-radius:17px;padding:24px;margin:19px 0;box-shadow:0 1px 3px #12101c08}
    .switch{display:inline-flex;gap:5px;background:#eeedf3;border-radius:12px;padding:5px;flex-wrap:wrap}.switch button{background:none;border:0;border-radius:9px;color:#465061;font-size:13px;font-weight:800;padding:11px 17px;cursor:pointer}
    .switch button[aria-pressed=true]{background:var(--purple);color:white}.switch button:focus-visible{outline:3px solid #a783e7;outline-offset:2px}
    .kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-top:18px}
    .kpi{padding:16px 18px;border-radius:12px;border:1px solid #eeeaf3;background:#faf9fc}.kpi small{display:block;color:#687184;font-size:12px;line-height:1.4}.kpi strong{display:block;font-size:24px;margin:8px 0 3px;font-variant-numeric:tabular-nums}
    .section-head{display:flex;justify-content:space-between;align-items:flex-start;gap:20px;flex-wrap:wrap}.section-head p{margin:0;color:var(--muted);font-size:13px}
    .rows{margin-top:19px;display:flex;flex-direction:column;gap:13px}.agency-row{display:grid;grid-template-columns:minmax(160px,1.35fr) minmax(100px,2fr) 145px;gap:16px;align-items:center}
    .agency-name{font-size:12px;font-weight:800;line-height:1.35}.agency-bar{height:16px;background:#f0eaf5;border-radius:5px;overflow:hidden}.agency-bar>div{height:100%;background:#8756c2;border-radius:5px}
    .agency-num{text-align:right;font-size:13px;font-weight:800;white-space:nowrap}.agency-num span{display:block;color:var(--muted);font-size:11px;font-weight:400;margin-top:3px}.foot{font-size:12px;color:#6b7280;line-height:1.6}
    .cards{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.tile{background:#fff;border:1px solid #e8e4ee;border-radius:14px;padding:19px;display:block;transition:transform .15s,border-color .15s}
    .tile:hover{transform:translateY(-2px);border-color:#b38bd6}.tile p{font-size:13px;color:#677085;margin:0 0 15px}.tile strong{font-size:13px;color:#7035a8}
    .steps{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-top:20px}.step{border:1px solid #e9e3f0;background:#faf9fc;border-radius:12px;padding:17px}
    .step .num{font-size:12px;font-weight:900;color:var(--purple);display:block;margin-bottom:10px}.step p{font-size:12px;color:#637085;margin:9px 0 0}
    .formulas{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:20px 0}.formula{background:#f4f1f8;border-radius:12px;padding:16px}
    .formula code{font-size:12px;color:#562388;line-height:1.65;display:block;overflow-wrap:anywhere;white-space:normal}.formula p{font-size:12px;color:#687184;margin:9px 0 0}
    details{border-top:1px solid #eae5ef;padding:14px 0}summary{cursor:pointer;font-size:13px;font-weight:800}details p{font-size:13px;color:#637085;margin-bottom:0}
    .qbar{display:flex;flex-wrap:wrap;gap:12px;margin-top:15px}.q{background:#f6f3fb;border-radius:10px;padding:14px 18px;min-width:125px;flex:1}.q b{font-size:20px;display:block;margin-bottom:4px}.q span{color:#677085;font-size:11px}
    @media(max-width:870px){.kpis,.steps{grid-template-columns:repeat(2,minmax(0,1fr))}.cards{grid-template-columns:1fr}.agency-row{grid-template-columns:minmax(130px,1.3fr) minmax(70px,1fr) 110px;gap:8px}.agency-name{font-size:11px}.agency-num{font-size:11px}}
    @media(max-width:520px){main{padding:24px 13px}.panel{padding:17px}.kpis,.steps,.formulas{grid-template-columns:1fr}.agency-row{grid-template-columns:1fr 72px}.agency-bar{grid-column:1/-1;grid-row:2}.agency-num{grid-column:2;grid-row:1}}
    </style></head><body>
    <header class="top"><nav class="nav" aria-label="Health navigation"><a class="brand" href="../">ATLAS</a><div class="nav-links"><a href="../">ATLAS home</a><a aria-current="page" href="index.html">Federal Health</a><a href="AccentureHealthTAMOverview.html">TAM overview</a><a href="HealthCompetitiveIntelligence.html">Competitive intelligence</a><a href="HealthContractExplorer.html">Contract explorer</a><a href="HealthAgencyIntelligence.html">Agency intelligence</a></div></nav><div class="meta">__FRESHNESS__</div></header>
    <main><section class="hero"><p class="eyebrow">FEDERAL HEALTH INTELLIGENCE</p><h1>Where is the Health market moving?</h1>
    <p class="lead">Explore Federal Health procurement through Accenture's addressable market, competitor positioning, agency concentration and individual contract evidence.</p></section>
    <section class="panel" aria-label="Market snapshot"><div class="section-head"><div><h2>Health market snapshot</h2><p>Historical AusTender contract values, not annual budget expenditure.</p></div>
    <div class="switch" role="group" aria-label="Market scope"><button type="button" data-scope="tam" aria-pressed="true">Accenture TAM</button><button type="button" data-scope="all" aria-pressed="false">All Federal Health Market</button></div></div>
    <div class="kpis"><div class="kpi"><small>Selected market value</small><strong id="kValue">–</strong><small>Aggregate recorded contract value</small></div>
    <div class="kpi"><small>Contract notices</small><strong id="kContracts">–</strong><small>Distinct CN records</small></div>
    <div class="kpi"><small>Accenture awards</small><strong id="kAccenture">–</strong><small>Value in selected market</small></div>
    <div class="kpi"><small>Federal Health agencies</small><strong id="kAgencies">–</strong><small>Agencies with contracts</small></div></div></section>
    <section class="panel" id="atlas-agency"><div class="section-head"><div><p class="eyebrow">MARKET CONCENTRATION</p><h2 id="agencyTitle">TAM by Agency</h2>
    <p>Share of the selected market's recorded contract value. Select the market above to compare.</p></div></div>
    <div class="rows" id="agencyRows"></div><p class="foot">Source: AusTender. Each contract notice contributes its recorded value, including amendments. Values may be negative where amendments reduce reported contract value.</p></section>
    <section aria-label="Explore dashboards"><h2>Explore the market</h2><div class="cards">
    <a class="tile" href="AccentureHealthTAMOverview.html"><h3>TAM overview</h3><p>Understand addressable spend, Service Offering demand, leading suppliers and agency share.</p><strong>Open overview</strong></a>
    <a class="tile" href="HealthCompetitiveIntelligence.html"><h3>Competitive intelligence</h3><p>Compare supplier market positions, contract portfolios and agency concentration.</p><strong>Compare suppliers</strong></a>
    <a class="tile" href="HealthContractExplorer.html"><h3>Contract explorer</h3><p>Investigate the specific contracts, amendments and suppliers underpinning the numbers.</p><strong>Explore contracts</strong></a>
    <a class="tile" href="HealthAgencyIntelligence.html"><h3>Agency intelligence</h3><p>Explore each agency by supplier, Service Offering and contracted value.</p><strong>Explore agencies</strong></a>
    </div></section>
    <section class="panel" id="atlas-methodology"><p class="eyebrow">TRANSPARENT CLASSIFICATION</p><h2>How ATLAS defines the Health market</h2>
    <p class="foot">Classification is performed upstream in the Health pipeline, not inside the dashboards. This view explains the decision stages and verified aggregation formulas.</p>
    <div class="steps"><div class="step"><span class="num">01</span><h3>Agency scope</h3><p>Keep the configured Federal Health agencies and their historical aliases.</p></div>
    <div class="step"><span class="num">02</span><h3>Addressability</h3><p>Assess Category, Contract Type and Description against exclusion rules and service evidence, using supplier context where supported.</p></div>
    <div class="step"><span class="num">03</span><h3>Service Offering</h3><p>Allocate addressable records to Strategy, SI, Managed Services, Data & AI, Cloud & Cyber or retain as Unclassified.</p></div>
    <div class="step"><span class="num">04</span><h3>Audit & report</h3><p>Preserve contract notices, values and evidence. Aggregate only the canonical classified records.</p></div></div>
    <div class="formulas"><div class="formula"><h3>Addressable TAM</h3><code>Σ Value(CN), where is_addressable = True</code><p>Value is the amount reported for each contract notice.</p></div>
    <div class="formula"><h3>Service Offering market</h3><code>Σ Value(CN), where is_addressable = True and service_offering = X</code><p>One canonical Service Offering decision per included record.</p></div></div>
    <details><summary>What do category, description and supplier signals mean?</summary><p>Physical clinical and commodity purchases can be excluded by category/type rules. Positive signals can include consulting, transformation, integration, software, digital, cloud, cyber, data and managed-service delivery. The authoritative ordering and exceptions are defined in the upstream classifier.</p></details>
    <details><summary>What numerical weights and scoring thresholds are used?</summary><p>Exact weights and thresholds are not included in the dashboard code available for this build. They must be verified directly against the current Health classifier before we publish an exact scoring formula.</p></details>
    <details><summary>Why do some amendments have inferred dates?</summary><p>ATLAS may calculate the earliest credible start and latest credible end from available CN family records when the original is absent. This represents an observed range, not independent verification of the legally current contract period.</p></details>
    </section>
    <section class="panel"><h2>Data confidence</h2><p class="foot">Counts below refer to the selected market and are derived from the current canonical dataset.</p>
    <div class="qbar"><div class="q"><b id="qReview">–</b><span>Contract-family date review flags</span></div><div class="q"><b id="qInferred">–</b><span>Inferred family dates</span></div><div class="q"><b id="qUnclassified">–</b><span>Unclassified Service Offerings</span></div></div></section>
    <p class="foot">ATLAS is a historical procurement intelligence tool. The addressable market is a rules-based estimate, not a forecast or a statement of procurement eligibility. The source data and classifier should be audited for high-value decisions.</p></main>
    <script>
    const MARKETS=__DATA__;
    let scope='tam';
    const nf=new Intl.NumberFormat('en-AU');
    function money(n){n=Number(n||0);const x=Math.abs(n);return (n<0?'−':'')+'A$'+(x>=1e9?(x/1e9).toFixed(2)+'B':x>=1e6?(x/1e6).toFixed(1)+'M':nf.format(Math.round(x)))}
    function text(id,s){const e=document.getElementById(id);if(e)e.textContent=s}
    function addRow(target,r,max,total){
      const row=document.createElement('div');row.className='agency-row';
      const label=document.createElement('div');label.className='agency-name';label.textContent=r.agency;
      const bar=document.createElement('div');bar.className='agency-bar';const fill=document.createElement('div');
      fill.style.width=(max>0?Math.max(0,(Math.max(0,r.value)/max)*100):0).toFixed(2)+'%';bar.appendChild(fill);
      const number=document.createElement('div');number.className='agency-num';number.textContent=money(r.value);
      const share=document.createElement('span');share.textContent=(total?r.value/total*100:0).toFixed(1)+'% · '+nf.format(r.count)+' CNs';number.appendChild(share);
      row.append(label,bar,number);target.appendChild(row);
    }
    function render(){
      const m=MARKETS[scope];const tam=scope==='tam';
      document.querySelectorAll('[data-scope]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.scope===scope)));
      text('kValue',money(m.total));text('kContracts',nf.format(m.count));text('kAccenture',money(m.accenture_value));text('kAgencies',nf.format(m.agency_count));
      text('agencyTitle',(tam?'Accenture TAM':'All Federal Health Market')+' by Agency');
      const target=document.getElementById('agencyRows');target.replaceChildren();
      const max=Math.max(0,...m.agencies.map(r=>r.value));m.agencies.forEach(r=>addRow(target,r,max,m.total));
      text('qReview',m.review===null?'Not available':nf.format(m.review));text('qInferred',m.inferred===null?'Not available':nf.format(m.inferred));text('qUnclassified',m.unclassified===null?'Not available':nf.format(m.unclassified));
    }
    document.querySelectorAll('[data-scope]').forEach(b=>b.addEventListener('click',()=>{scope=b.dataset.scope;render()}));
    render();
    </script></body></html>'''
    page = page.replace("__DATA__", safe_data).replace("__FRESHNESS__", source_meta)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(atlas_option_a_style(page, "home"), encoding="utf-8")
    print(f"Wrote Health executive landing: {output_path}")


def atlas_option_a_style(html_text: str, page: str) -> str:
    """Presentation-only Option A shell. Existing analytics and event handlers remain untouched."""
    css = '<style id="atlas-option-a-design-system">\n:root{--atlas-bg:#f6f8fb;--atlas-ink:#101828;--atlas-line:#e4dfea;--atlas-purple:#a100ff;--atlas-muted:#667085}\nhtml{scroll-behavior:smooth}body{background:var(--atlas-bg);color:var(--atlas-ink)}\n.wrap{max-width:1500px;margin-left:auto;margin-right:auto;padding-top:28px}\n.atlas-option-a-heading{display:flex;align-items:flex-start;justify-content:space-between;gap:20px;margin:8px 0 20px}\n.atlas-option-a-eyebrow{font-weight:800;font-size:11px;letter-spacing:.12em;color:#7811b2;margin:0 0 9px}\n.atlas-option-a-heading h1{font-size:clamp(27px,3vw,34px);letter-spacing:-.035em;margin:0 0 8px}\n.atlas-option-a-heading .subtitle,.atlas-option-a-heading .sub{margin:0;color:#526070;line-height:1.55}\n.atlas-option-a-chip{white-space:nowrap;border:1px solid #e4daed;border-radius:99px;padding:8px 12px;font-size:11px;color:#694b78;background:#faf6ff}\n.atlas-option-a-tabs{display:flex;gap:8px;align-items:center;border-bottom:1px solid #e4dfea;margin:0 0 20px;flex-wrap:wrap}\n.atlas-option-a-tabs a{display:inline-block;border:1px solid #e4dfea;border-bottom:0;border-radius:12px 12px 0 0;background:#eeebf2;color:#5a5263;text-decoration:none;font-weight:750;padding:14px 22px;font-size:13px}\n.atlas-option-a-tabs a.active{background:white;color:#6b159a;border-top:3px solid #a100ff;padding-top:12px}\n.atlas-option-a-tabs a:hover{background:#fbf5ff;color:#6b159a}\n.atlas-option-a-tabs a:focus-visible{outline:3px solid #9f65cf;outline-offset:2px}\n.atlas-health-top{background:#11101c;color:#f8f7fc}\n.atlas-health-top .atlas-health-nav{max-width:1500px;padding:14px 24px}\n.atlas-health-links a[aria-current=page]{background:#6730ad;color:#fff}\n.atlas-market,.atlas-market-controls,.panel,.summary,.executive,.insight,.supplier-kpi-block,.kpi,.kpis .kpi,.card,.tile,.selector-card,.field,.toolbar,.atlas-health-agencies,.atlas-health-quality{border-radius:14px}\n.atlas-market,.atlas-market-controls,.panel,.summary,.executive,.insight,.supplier-kpi-block{box-shadow:0 1px 3px rgba(16,24,40,.045)}\n.atlas-market,.atlas-market-controls{border:1px solid var(--atlas-line);background:#fff}\n.atlas-market-choice[aria-pressed=true],.atlas-market-switch button[aria-pressed=true],.switch button[aria-pressed=true],button.active{background:#a100ff;color:#fff;border-color:#a100ff}\n.atlas-option-a-sticky{position:sticky;top:0;z-index:30;box-shadow:0 7px 22px rgba(31,11,50,.09)!important;border:1px solid #e4dfea!important;background:#fff!important}\n.atlas-option-a-sticky select,.atlas-option-a-sticky input{max-width:100%}\n.atlas-option-a-sticky:focus-within{z-index:31}\n.atlas-option-a-jumps{display:flex;flex-wrap:wrap;gap:9px;margin:0 0 14px}\n.atlas-option-a-jumps a{border:1px solid #e3dce9;background:#fff;color:#65318e;text-decoration:none;font-size:12px;font-weight:700;border-radius:999px;padding:8px 12px}\n.atlas-option-a-jumps a:hover{background:#f4ebff}\nmain,.wrap{scroll-margin-top:90px}section[id],.panel[id]{scroll-margin-top:110px}\n@media(max-width:760px){.wrap{padding:18px 14px}.atlas-option-a-heading{display:block}.atlas-option-a-chip{display:none}.atlas-option-a-tabs{gap:4px}.atlas-option-a-tabs a{flex:1;text-align:center;padding:12px 8px;font-size:11px}.atlas-option-a-sticky{position:relative;top:auto}.atlas-option-a-jumps{gap:5px}}\n</style>'
    html_text = html_text.replace('</head>', css + '</head>', 1)
    headings = {
        'overview': ('FEDERAL HEALTH INTELLIGENCE / TAM OVERVIEW', 'ATLAS · market intelligence'),
        'compare': ('FEDERAL HEALTH INTELLIGENCE / COMPETITIVE INTELLIGENCE', 'ATLAS · supplier positioning'),
        'agency': ('FEDERAL HEALTH INTELLIGENCE / AGENCY INTELLIGENCE', 'ATLAS · agency analytics'),
        'home': ('FEDERAL HEALTH INTELLIGENCE / HOME', 'ATLAS · AusTender intelligence'),
    }
    eyebrow, chip = headings[page]
    if page == 'home':
        # Preserve homepage's special hero copy and chart controls.
        html_text = html_text.replace('class="hero"', 'class="hero atlas-option-a-heading"', 1)
        html_text = html_text.replace('<p class="eyebrow">FEDERAL HEALTH INTELLIGENCE</p>',
            '<p class="eyebrow">' + eyebrow + '</p>', 1)
        html_text = html_text.replace('class="switch"', 'class="switch atlas-option-a-sticky"', 1)
        return html_text
    import re
    heading_re = re.compile(r'(<h1[^>]*>.*?</h1>\s*<(?:p|div)[^>]*class="(?:subtitle|sub)"[^>]*>.*?</(?:p|div)>)', re.S)
    # Competitive subtitle uses id before class; support either attribute order.
    if page == 'compare':
        heading_re = re.compile(r'(<h1[^>]*>.*?</h1>\s*<div\s+id="healthScopeDescription"\s+class="subtitle"[^>]*>.*?</div>)', re.S)
    match = heading_re.search(html_text)
    if match:
        block = match.group(1)
        header = ('<div class="atlas-option-a-heading"><div><div class="atlas-option-a-eyebrow">'
                  + eyebrow + '</div>' + block + '</div><span class="atlas-option-a-chip">'
                  + chip + '</span></div>')
        html_text = html_text[:match.start()] + header + html_text[match.end():]
    # Tabs are navigation anchors within the current page, not competing data views.
    # Keep the Explorer's separate view tabs unchanged.
    tabs = {
        'overview': '<a class="active" href="#atlas-overview-main">TAM analysis</a><a href="HealthCompetitiveIntelligence.html">Competitive intelligence</a><a href="HealthAgencyIntelligence.html">Agency intelligence</a><a href="HealthContractExplorer.html">Contract explorer</a>',
        'compare': '<a class="active" href="#atlas-compare-main">Supplier comparison</a><a href="HealthAgencyIntelligence.html">Agency intelligence</a><a href="HealthContractExplorer.html">Contract explorer</a>',
        'agency': '<a class="active" href="#atlas-agency-main">Agency analysis</a><a href="HealthCompetitiveIntelligence.html">Competitive intelligence</a><a href="HealthContractExplorer.html">Contract explorer</a>',
    }
    tabbar = '<nav class="atlas-option-a-tabs" aria-label="Health intelligence views">' + tabs[page] + '</nav>'
    where = html_text.find('</div>', html_text.find('class="atlas-option-a-heading"'))
    # locate the outer closing div by anchoring after the chip
    marker = '</span></div>'
    idx = html_text.find(marker, html_text.find('class="atlas-option-a-heading"'))
    if idx >= 0:
        idx += len(marker)
        html_text = html_text[:idx] + tabbar + html_text[idx:]
    if page == 'overview':
        html_text = html_text.replace('<section class="atlas-market-controls"', '<section id="atlas-overview-main" class="atlas-market-controls atlas-option-a-sticky"', 1)
        # If the TAM/All Market selector is the second-level market controls, make its placement obvious.
        if 'id="atlas-overview-main"' not in html_text:
            html_text = html_text.replace('<section class="summary"', '<section id="atlas-overview-main" class="summary"', 1)
        # retain existing chart range selection behavior; style-only.
    elif page == 'compare':
        html_text = html_text.replace('<section class="atlas-market"', '<section id="atlas-compare-main" class="atlas-market atlas-option-a-sticky"', 1)
        # Supplier dropdowns follow the market selector; keep both easy to locate.
        html_text = html_text.replace('class="selectors"', 'class="selectors atlas-option-a-sticky"', 1)
    elif page == 'agency':
        html_text = html_text.replace('<div class="panel toolbar">', '<div id="atlas-agency-main" class="panel toolbar atlas-option-a-sticky">', 1)
    return html_text

if __name__ == "__main__":
    main()

