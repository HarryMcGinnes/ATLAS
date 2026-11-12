from __future__ import annotations

import argparse
import html
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

OFFERINGS = [
    "Strategy, Transformation & Advisory", "SI & Engineering",
    "Managed Services & Operations", "Data, AI & Automation",
    "Cloud Infrastructure & Cyber",
]


def boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.fillna(False).astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])


def load(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Health master not found: {path}")
    df = pd.read_parquet(path) if path.suffix.lower() in (".parquet", ".pq") else pd.read_csv(path, low_memory=False)
    required = {"CN ID", "Agency", "Value", "supplier_group", "is_addressable"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError("Missing canonical fields: " + ", ".join(sorted(missing)))
    if df["CN ID"].isna().any() or df["CN ID"].duplicated().any():
        raise ValueError("Missing or duplicate CN IDs in Health master")
    df = df.copy()
    df["Value"] = pd.to_numeric(df["Value"], errors="raise")
    if df["Value"].isna().any():
        raise ValueError("Null contract value prevents reconciliation")
    df["is_addressable"] = boolean(df["is_addressable"])
    for col in ("Agency", "supplier_group"):
        df[col] = df[col].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
    source = "service_offering" if "service_offering" in df else "Service Offering"
    df["_offering"] = df[source].fillna("Unresolved").astype(str).str.strip() if source in df else "Unresolved"
    df["_offering"] = df["_offering"].replace({"": "Unresolved"})
    df.loc[~df["is_addressable"], "_offering"] = "Non-addressable / Unclassified"
    if "Financial Year" not in df.columns:
        raise ValueError("Financial Year missing from Health master: FY filtering requires the canonical field")
    if "Financial Year" in df.columns:
        df["_fy"] = df["Financial Year"].fillna("Unknown FY").astype(str)
        df["_fy_start"] = pd.to_numeric(df["_fy"].str.extract(r"((?:19|20)\d{2})")[0], errors="coerce")
    else:
        df["_fy_start"] = float("nan")
    return df


def grouped(df: pd.DataFrame, cols: list[str]) -> list[dict]:
    g = df.groupby(cols, dropna=False).agg(value=("Value", "sum"), contracts=("CN ID", "nunique")).reset_index()
    return [{**{key: str(r[key]) for key in cols}, "value": float(r["value"]), "contracts": int(r["contracts"])} for r in g.to_dict("records")]


def build_data(df: pd.DataFrame) -> dict:
    fy_values = sorted(set(df["_fy_start"].dropna().astype(int)))
    scopes = {}
    for scope, frame in (("tam", df[df["is_addressable"]]), ("all", df)):
        scopes[scope] = {
            "total": float(frame["Value"].sum()),
            "contracts": int(frame["CN ID"].nunique()),
            "by_supplier": grouped(frame, ["supplier_group", "Agency", "_offering", "_fy_start"]),
        }
        actual = sum(x["value"] for x in scopes[scope]["by_supplier"])
        if abs(actual - scopes[scope]["total"]) > 0.01:
            raise ValueError(f"Supplier breakdown does not reconcile for {scope}: {actual}")
    now = datetime.now(ZoneInfo('Australia/Sydney'))
    current_fy = now.year if now.month >= 7 else now.year - 1
    return {"scopes": scopes, "agencies": sorted(df["Agency"].unique().tolist()),
            "fys": fy_values, "current_fy": current_fy}


PAGE = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ATLAS | Health Competitive Intelligence</title>
<script src="https://cdn.plot.ly/plotly-3.5.0.min.js"></script>
<style>
:root{--purple:#a100ff;--ink:#101828;--muted:#667085;--line:#e4dfea}
*{box-sizing:border-box}body{margin:0;background:#f6f8fb;color:var(--ink);font:14px Arial,Helvetica,sans-serif}
a{text-decoration:none}header{background:#11101c;color:white}nav{max-width:1500px;margin:auto;display:flex;align-items:center;gap:18px;padding:14px 22px;flex-wrap:wrap}
nav a{color:#e8e3ef;font-weight:700;font-size:13px;padding:9px 11px;border-radius:9px}nav a:hover,nav a[aria-current="page"]{background:#6730ad;color:white}nav a.brand{font-size:23px;letter-spacing:.05em;margin-right:10px;color:white;padding-left:0}
main{max-width:1500px;margin:auto;padding:28px 22px 60px}.eyebrow{font-size:11px;letter-spacing:.12em;font-weight:bold;color:#7811b2}h1{font-size:32px;margin:9px 0}h2{font-size:19px;margin:0 0 8px}.sub,.note{color:var(--muted);line-height:1.5}.sub{margin:0 0 20px}.note{font-size:12px}
.panel,.kpi{background:white;border:1px solid var(--line);border-radius:14px;box-shadow:0 1px 3px #1018280a}.panel{padding:20px;margin:16px 0}.controls{display:flex;gap:12px;flex-wrap:wrap;align-items:end}.field{flex:1;min-width:200px}.field label{display:block;font-size:12px;color:var(--muted);margin-bottom:7px;font-weight:700}
select,button{font:inherit;border:1px solid #cbc6d6;border-radius:9px;padding:11px 12px;background:white;cursor:pointer}select{width:100%;min-width:0}.scope{display:flex;gap:7px;flex-wrap:wrap}.scope button.active{background:var(--purple);color:white;border-color:var(--purple)}button:focus-visible,select:focus-visible{outline:3px solid #c99aff;outline-offset:2px}
.kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-top:16px}.kpi{padding:18px;min-width:0}.kpi span{font-size:12px;color:var(--muted)}.kpi strong{display:block;font-size:25px;margin:9px 0}.two{display:grid;grid-template-columns:1fr 1fr;gap:16px}.chart{min-height:350px}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:11px;border-bottom:1px solid #eee9f2;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#f9f7fc}tbody tr:hover{background:#faf8fd}.table-scroll{max-height:560px;overflow:auto}.table-scroll thead{position:sticky;top:0}
@media(max-width:850px){.two,.kpis{grid-template-columns:1fr 1fr}}@media(max-width:560px){.two,.kpis{grid-template-columns:1fr}.scope button{flex:1}}
</style></head><body><header><nav aria-label="ATLAS Health navigation"><a class="brand" href="../">ATLAS</a><a href="index.html">Health home</a><a href="AccentureHealthTAMOverview.html">TAM overview</a><a href="HealthCompetitiveIntelligence.html" aria-current="page">Competitive intelligence</a><a href="HealthContractExplorer.html">Contract explorer</a><a href="HealthAgencyIntelligence.html">Agency intelligence</a><a href="Methodology.html">Methodology</a></nav></header><main>
<div class="eyebrow">FEDERAL HEALTH INTELLIGENCE / COMPETITIVE INTELLIGENCE</div><h1>Health Competitive Intelligence</h1><p class="sub">Compare supplier positions, agency relationships and Service Offering mix · Built __STAMP__ (Australia/Sydney)</p>
<section class="panel controls"><div class="field"><label for="agency">Agency</label><select id="agency"></select></div><div class="field"><label for="period">Financial year</label><select id="period"></select></div><div class="field"><label for="supplier">Focus supplier</label><select id="supplier"></select></div><div class="scope"><button id="tam" class="active" aria-pressed="true">Accenture TAM</button><button id="all" aria-pressed="false">All Federal Health Market</button><button id="clear">Clear filters</button></div></section>
<div class="kpis"><div class="kpi"><span>Market contract value</span><strong id="marketValue">—</strong></div><div class="kpi"><span>Contract notices</span><strong id="contractCount">—</strong></div><div class="kpi"><span>Suppliers</span><strong id="supplierCount">—</strong></div><div class="kpi"><span>Focus supplier market share</span><strong id="focusShare">—</strong></div></div>
<div class="two"><section class="panel"><h2>Supplier leaderboard</h2><p class="note">Net recorded contract value, including negative amendments.</p><div id="leaderChart" class="chart"></div></section><section class="panel"><h2>Focus supplier by Service Offering</h2><p class="note">Selected supplier within the current agency and market scope.</p><div id="offeringChart" class="chart"></div></section></div>
<div class="two"><section class="panel"><h2>Competitive trend by financial year</h2><div id="trendChart" class="chart"></div></section><section class="panel"><h2>Focus supplier by agency</h2><div id="agencyChart" class="chart"></div></section></div>
<section class="panel"><h2>Supplier comparison</h2><p class="note">Ranked by net historical contract value for the selected period. Click a supplier to focus the charts above.</p><div class="table-scroll" id="supplierTable"></div></section>
<p class="note">The default Last 5 FYs ends with the current Australian financial year, including the partial current FY. Source: canonical ATLAS Health master. Addressability and supplier_group are inherited, not reclassified. Financial Year is a recorded procurement classification, not annual expenditure or a pro-rata allocation of multiyear contracts. Historical contract notices and amendments are included as provided; use signed values with care.</p>
</main><script>
const DATA=__DATA__;const $=id=>document.getElementById(id);let mode='tam';const nf=new Intl.NumberFormat('en-AU');
function money(v){v=Number(v)||0;let a=Math.abs(v),s=v<0?'−':'';return s+'A$'+(a>=1e9?(a/1e9).toFixed(2)+'B':a>=1e6?(a/1e6).toFixed(1)+'M':nf.format(Math.round(a)))}
function pct(v){return Number.isFinite(v)?v.toFixed(1)+'%':'n/a'}function safe(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function periodOptions(){let f=DATA.fys,now=DATA.current_fy;let label=y=>'FY'+y+'–'+String((y+1)%100).padStart(2,'0');return [{v:'last5',l:'Last 5 FYs (default)'},{v:'last3',l:'Last 3 FYs'},{v:'current',l:'Current '+label(now)},{v:'all',l:'All historical FYs'},...f.slice().reverse().map(y=>({v:String(y),l:label(y)})),{v:'unknown',l:'Unknown FY'}]}
function inPeriod(r){let p=$('period').value,fy=Number(r._fy_start),current=DATA.current_fy;if(p==='all')return true;if(p==='unknown')return !Number.isInteger(fy)||fy<1900;if(!Number.isInteger(fy)||fy<1900)return false;if(p==='last5')return fy>=current-4&&fy<=current;if(p==='last3')return fy>=current-2&&fy<=current;if(p==='current')return fy===current;return fy===Number(p)}
function scoped(){let agency=$('agency').value;return DATA.scopes[mode].by_supplier.filter(r=>inPeriod(r)&&(agency==='__ALL__'||r.Agency===agency))}
function rank(rows,key){let m=new Map();rows.forEach(r=>m.set(r[key],(m.get(r[key])||0)+r.value));return [...m].map(([name,value])=>({name,value})).sort((a,b)=>b.value-a.value)}
function bar(id,rows){let d=rows.filter(r=>r.value!==0).slice(0,12).reverse();if(!window.Plotly)return;Plotly.react(id,[{type:'bar',orientation:'h',x:d.map(r=>r.value/1e6),y:d.map(r=>r.name),marker:{color:'#8750b7'},customdata:d.map(r=>r.value),hovertemplate:'<b>%{y}</b><br>A$%{customdata:,.0f}<extra></extra>'}],{margin:{l:210,r:25,t:12,b:55},height:Math.max(350,d.length*34+90),xaxis:{title:'Net value (A$ millions)',zeroline:true},yaxis:{automargin:true},paper_bgcolor:'transparent',plot_bgcolor:'transparent'},{responsive:true,displayModeBar:false})}
function refreshSuppliers(rows){let old=$('supplier').value,rs=rank(rows,'supplier_group');$('supplier').innerHTML=rs.map(r=>'<option value="'+safe(r.name)+'">'+safe(r.name)+'</option>').join('');if(rs.some(r=>r.name===old))$('supplier').value=old;else if(rs.some(r=>r.name==='Accenture'))$('supplier').value='Accenture';else if(rs.length)$('supplier').value=rs[0].name}
function render(rebuild=true){let rs=scoped();if(rebuild)refreshSuppliers(rs);let suppliers=rank(rs,'supplier_group'),val=rs.reduce((s,r)=>s+r.value,0),focus=$('supplier').value,focusRows=rs.filter(r=>r.supplier_group===focus),focusValue=focusRows.reduce((s,r)=>s+r.value,0);$('marketValue').textContent=money(val);$('supplierCount').textContent=nf.format(suppliers.length);$('focusShare').textContent=val>0?pct(focusValue/val*100):'n/a';
// CN counts aggregated by supplier/agency/offering/FY may overlap across cells if a CN has multiple classifications.
// Only the overall distinct count is exact for an unfiltered global scope; the displayed aggregate is a notice-cell count.
$('contractCount').textContent=nf.format(rs.reduce((s,r)=>s+r.contracts,0));bar('leaderChart',suppliers);bar('offeringChart',rank(focusRows,'_offering'));bar('agencyChart',rank(focusRows,'Agency'));
let fyMap=new Map();rs.forEach(r=>{let y=Number(r._fy_start);if(!Number.isFinite(y))return;if(!fyMap.has(y))fyMap.set(y,{fy:y,total:0,focus:0});let x=fyMap.get(y);x.total+=r.value;if(r.supplier_group===focus)x.focus+=r.value});let trend=[...fyMap.values()].sort((a,b)=>a.fy-b.fy);if(window.Plotly)Plotly.react('trendChart',[{x:trend.map(x=>x.fy),y:trend.map(x=>x.total/1e6),type:'scatter',mode:'lines+markers',name:'Selected market'},{x:trend.map(x=>x.fy),y:trend.map(x=>x.focus/1e6),type:'scatter',mode:'lines+markers',name:focus}],{margin:{l:65,r:20,t:16,b:55},xaxis:{title:'FY start',dtick:1},yaxis:{title:'Value (A$ millions)'},paper_bgcolor:'transparent',plot_bgcolor:'transparent',legend:{orientation:'h',y:1.18}},{responsive:true,displayModeBar:false});
$('supplierTable').innerHTML='<table><thead><tr><th>Supplier</th><th>Net value</th><th>Share</th></tr></thead><tbody>'+suppliers.slice(0,40).map(s=>'<tr><td><button data-supplier="'+safe(s.name)+'">'+safe(s.name)+'</button></td><td>'+money(s.value)+'</td><td>'+(val>0?pct(s.value/val*100):'n/a')+'</td></tr>').join('')+'</tbody></table>';
$('supplierTable').querySelectorAll('[data-supplier]').forEach(b=>b.onclick=()=>{$('supplier').value=b.dataset.supplier;render(false)});
}
$('agency').innerHTML='<option value="__ALL__">All Federal Health agencies</option>'+DATA.agencies.map(x=>'<option value="'+safe(x)+'">'+safe(x)+'</option>').join('');$('period').innerHTML=periodOptions().map(x=>'<option value="'+x.v+'">'+x.l+'</option>').join('');$('period').value='last5';
function setMode(m){mode=m;['tam','all'].forEach(x=>{$(x).classList.toggle('active',m===x);$(x).setAttribute('aria-pressed',String(m===x))});render()}
$('tam').onclick=()=>setMode('tam');$('all').onclick=()=>setMode('all');$('agency').onchange=()=>render();$('period').onchange=()=>render();$('supplier').onchange=()=>render(false);$('clear').onclick=()=>{$('agency').value='__ALL__';$('period').value='last5';setMode('tam')};render();
</script></body></html>'''


def main():
    parser = argparse.ArgumentParser(description="Build ATLAS Federal Health Competitive Intelligence")
    parser.add_argument("--input", default="health/data/processed/master_health_contracts.parquet")
    parser.add_argument("--output-dir", default="docs/health")
    parser.add_argument("--value-mode", choices=["total", "annualised"], default="total",
                        help="Use total contract value or annualised contract value.")
    args = parser.parse_args()
    df = load(Path(args.input))
    if args.value_mode == "annualised":
        if "Value Per Year" not in df.columns:
            parser.error("--value-mode annualised requires the Value Per Year column")
        annual = pd.to_numeric(df["Value Per Year"], errors="raise")
        if annual.isna().any():
            parser.error("Value Per Year contains null values")
        df["Value"] = annual
    data = build_data(df)
    stamp = html.escape(datetime.now(ZoneInfo("Australia/Sydney")).strftime("%d %b %Y %H:%M %Z"))
    page = PAGE.replace("__STAMP__", stamp).replace("__DATA__", json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c"))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "HealthCompetitiveIntelligence.html"
    path.write_text(page, encoding="utf-8")
    print(f"Health Competitive Intelligence: {path}")
    print(f"Source: {len(df):,} Health records; all-market value A${df['Value'].sum()/1e9:.3f}B")
    print("Reconciliation: PASS")


if __name__ == "__main__":
    main()
