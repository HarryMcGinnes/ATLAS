"""ATLAS Federal Health — competitive positioning & Service Offering leaderboards.

The existing canonical master is used unchanged. All/Australian FY and active
contract filters are independent of TAM/All Market. Agency names are grouped
for *display only* using explicitly approved historical rename aliases.
"""
from __future__ import annotations
import argparse
import html
import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd
from health_active_contracts import add_contract_status
from health_agency_groups import add_agency_groups, write_agency_audit

OFFERINGS = [
    'Strategy, Transformation & Advisory',
    'SI & Engineering',
    'Managed Services & Operations',
    'Data, AI & Automation',
    'Cloud Infrastructure & Cyber',
]

def to_bool(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.fillna(False).astype(str).str.strip().str.lower().isin(['true','1','yes','y'])

def load(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f'Canonical Health master not found: {path}')
    df = pd.read_parquet(path) if path.suffix.lower() in ('.parquet','.pq') else pd.read_csv(path, low_memory=False)
    required = {'CN ID','Agency','Value','supplier_group','is_addressable','is_accenture','Financial Year'}
    missing = sorted(required-set(df.columns))
    if missing:
        raise ValueError('Missing Health master fields: '+', '.join(missing))
    if df['CN ID'].isna().any() or df['CN ID'].duplicated().any():
        raise ValueError('Null or duplicate CN IDs prevent reliable Health competition metrics')
    df = df.copy()
    df['Value'] = pd.to_numeric(df['Value'], errors='raise')
    if df['Value'].isna().any() or not df['Value'].map(math.isfinite).all():
        raise ValueError('Null or nonfinite Health contract values')
    df['is_addressable'] = to_bool(df['is_addressable'])
    df['is_accenture'] = to_bool(df['is_accenture'])
    if (df['is_accenture'] & ~df['is_addressable']).any():
        raise ValueError('Some Accenture contracts are not addressable: investigate classification')
    df['Agency'] = df['Agency'].fillna('Unspecified agency').astype(str).str.strip().replace('', 'Unspecified agency')
    df['supplier_group'] = df['supplier_group'].fillna('Unknown supplier').astype(str).str.strip().replace('', 'Unknown supplier')
    col = 'service_offering' if 'service_offering' in df else 'Service Offering'
    if col not in df:
        raise ValueError('Missing canonical Service Offering')
    df['_offering'] = df[col].fillna('Unresolved').astype(str).str.strip().replace({'':'Unresolved','Uncategorised':'Unresolved'})
    df.loc[~df['is_addressable'],'_offering'] = 'Non-addressable / Unclassified'
    fy = df['Financial Year'].astype('string').str.extract(r'((?:19|20)\d{2})', expand=False)
    df['_fy_key'] = fy.fillna('unknown').astype(str)
    df = add_agency_groups(df)
    df = add_contract_status(df)
    return df

def grouped(frame: pd.DataFrame, columns: list[str]) -> list[dict]:
    g=(frame.groupby(columns, dropna=False, sort=False)
       .agg(value=('Value','sum'),contracts=('CN ID','nunique')).reset_index())
    return [{**{col: str(row[col]) for col in columns},'value':float(row['value']),
             'contracts':int(row['contracts'])} for row in g.to_dict('records')]

def build_data(df: pd.DataFrame) -> dict:
    out={}
    for mode, frame in [('all',df),('tam',df.loc[df['is_addressable']])]:
        cells=grouped(frame,['supplier_group','agency_group','_offering','_fy_key','_contract_status'])
        value=float(frame['Value'].sum())
        if abs(sum(r['value'] for r in cells)-value)>0.01:
            raise ValueError(f'Competitive Intelligence {mode} cells do not reconcile')
        wins=float(frame.loc[frame['is_accenture'],'Value'].sum())
        out[mode]={'by_supplier':cells,'total':value,'wins':wins,'contracts':len(frame)}
    if abs(out['all']['wins']-out['tam']['wins'])>0.01:
        raise ValueError('Accenture wins differ between All Market and TAM')
    now=datetime.now(ZoneInfo('Australia/Sydney'))
    years=sorted({int(y) for y in df['_fy_key'] if y.isdigit()})
    return {'scopes':out, 'agencies':sorted(df['agency_group'].unique().tolist()),
            'fys':years,'current_fy':now.year if now.month>=7 else now.year-1,
            'as_at':now.date().isoformat(), 'offerings':OFFERINGS}

PAGE = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ATLAS | Health Competitive Intelligence</title><script src="https://cdn.plot.ly/plotly-3.5.0.min.js"></script>
<style>
:root{--purple:#a100ff;--navy:#0041a9;--ink:#101828;--muted:#667085;--line:#e4dfea;--bg:#f6f8fb}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--ink);font:14px Arial,Helvetica,sans-serif}
a{text-decoration:none}header{background:#11101c;color:white}nav{max-width:1500px;margin:auto;display:flex;align-items:center;gap:15px;padding:14px 22px;flex-wrap:wrap}
nav a{color:#e8e3ef;font-weight:700;font-size:13px;padding:9px 11px;border-radius:9px}nav a:hover,nav a[aria-current=page]{background:#6730ad;color:#fff}nav a.brand{font-size:23px;letter-spacing:.05em;margin-right:6px;color:white;padding-left:0}
main{max-width:1500px;margin:auto;padding:27px 22px 65px}h1{font-size:33px;margin:7px 0}h2{font-size:19px;margin:0 0 8px}h3{font-size:17px;margin:0 0 8px}.eyebrow{font-size:11px;letter-spacing:.12em;font-weight:800;color:#7811b2}.sub{color:#526070;margin:0 0 19px;line-height:1.55}.note{color:var(--muted);font-size:12px;line-height:1.5}
.panel,.supplier-block{background:white;border:1px solid var(--line);border-radius:14px;box-shadow:0 1px 3px #1018280a}.panel{margin:16px 0;padding:21px}.controls{position:sticky;top:0;z-index:30;display:flex;gap:12px;flex-wrap:wrap;align-items:end;box-shadow:0 6px 20px #27163616;background:white}.field{flex:1;min-width:190px}.field label{display:block;font-size:11px;color:#667085;font-weight:800;letter-spacing:.04em;margin-bottom:7px;text-transform:uppercase}select,button{font:inherit;border:1px solid #cfc9da;border-radius:9px;padding:10px 12px;background:white;cursor:pointer}select{width:100%;min-width:0}button:hover{background:#f9f1ff}button:focus-visible,select:focus-visible{outline:3px solid #c99aff;outline-offset:2px}
.switches{display:flex;flex-wrap:wrap;gap:6px;align-items:center}.switches button.active{background:#a100ff;color:#fff;border-color:#a100ff}.status-pair{display:flex;gap:5px;flex-wrap:nowrap}.status-pair button{white-space:nowrap}.status-pair button.active{background:#a100ff;color:white;border-color:#a100ff}.scope-label{font-size:11px;font-weight:700;color:#667085;margin:0 0 5px}.filter-group{min-width:200px}.filter-group.market{min-width:280px}.clear{border-color:#b8a6d2;color:#692f92;font-weight:700}
.insight-box{border:1px solid #e4dfea;background:white;border-left:5px solid #a100ff;border-radius:13px;padding:16px 20px;margin:16px 0}.insight-box p{margin:7px 0 10px;line-height:1.55;color:#526070}.questions{list-style:none;padding:0;margin:5px 0}.questions li{margin:8px 0}.questions li:before{content:'•';color:#a100ff;font-weight:900;margin-right:8px}
.cards{display:grid;grid-template-columns:1fr 1fr;gap:16px}.supplier-block{padding:14px;border-top:4px solid #a100ff}.supplier-block.compare{border-color:#0244a5}.supplier-block-title{text-transform:uppercase;color:#372a54;letter-spacing:.05em;font-size:13px;font-weight:800;margin-bottom:12px}.supplier-block.compare .supplier-block-title{color:#0244a5}.supplier-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}.metric{border:1px solid #e7deed;border-radius:12px;padding:14px;min-width:0;background:#f9f7fc}.metric .label{color:#667085;font-size:11px;text-transform:uppercase;letter-spacing:.03em}.metric strong{display:block;margin:9px 0 6px;font-size:clamp(18px,2vw,28px);line-height:1.18;overflow-wrap:anywhere}.metric small{color:#667085;line-height:1.45}.supplier-block.compare .metric{background:#f4f9fd;border-color:#d5e4f6}.metric.larger{border-color:#93e9ad;background:#ebfbf1}.metric.smaller{border-color:#ffa49d;background:#fff4f3}
.two{display:grid;grid-template-columns:1fr 1fr;gap:16px}.two .panel{margin:16px 0}.chart{min-height:360px}.leaderboard-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin-top:16px}.leaderboard-card{border:1px solid #e3e7ed;border-radius:13px;padding:15px;min-width:0}.leaderboard-card h3{font-size:16px;margin:0 0 4px}.lb-sub{font-size:11px;color:#64748b;margin-bottom:13px}.leader-row{display:grid;grid-template-columns:34px minmax(110px,1fr) minmax(80px,1.3fr) 56px;gap:10px;align-items:center;min-height:29px;border-radius:7px;padding:3px 7px;font-size:12px}.leader-row.focus{background:#f9f0ff}.leader-row.compare{outline:1px solid #a9c9eb}.lb-rank{font-weight:800;color:#475569}.lb-name{overflow:hidden;white-space:nowrap;text-overflow:ellipsis}.lb-track{background:#edf0f5;border-radius:10px;height:12px;overflow:hidden}.lb-track i{display:block;height:100%;border-radius:10px;background:#607284}.leader-row.focus .lb-track i{background:#a100ff}.leader-row.compare .lb-track i{background:#0041a9}.lb-share{text-align:right;font-weight:800;white-space:nowrap}.lb-separator{text-align:center;color:#94a3b8;letter-spacing:.25em;padding:2px}.no-data{font-size:12px;color:#667085}.summary-stripe{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:15px}.summary-stripe>div{background:#faf8fd;border:1px solid #ede6f3;border-radius:10px;padding:13px}.summary-stripe span{display:block;color:#667085;font-size:11px}.summary-stripe b{display:block;font-size:22px;margin-top:8px}
table{width:100%;border-collapse:collapse;font-size:12px}th,td{padding:11px;border-bottom:1px solid #eee9f2;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#f9f7fc}tbody tr:hover{background:#faf8fd}.table-scroll{max-height:540px;overflow:auto}.table-scroll thead{position:sticky;top:0;z-index:1}td button{border:none;background:transparent;color:#6b229a;text-align:left;padding:0;font-weight:700}
@media(max-width:1000px){.cards,.two,.leaderboard-grid{grid-template-columns:1fr}.controls{position:static}.supplier-grid{grid-template-columns:repeat(2,1fr)}}@media(max-width:550px){main{padding:20px 13px}.supplier-grid,.summary-stripe{grid-template-columns:1fr}.leader-row{grid-template-columns:28px minmax(65px,1fr) minmax(60px,1fr) 48px;gap:5px}}
</style></head><body><header><nav aria-label="ATLAS Health navigation"><a class="brand" href="../">ATLAS</a><a href="index.html">Health home</a><a href="AccentureHealthTAMOverview.html">TAM overview</a><a href="HealthCompetitiveIntelligence.html" aria-current="page">Competitive intelligence</a><a href="HealthContractExplorer.html">Contract explorer</a><a href="HealthAgencyIntelligence.html">Agency intelligence</a><a href="Methodology.html">Methodology</a></nav></header>
<main><div class="eyebrow">FEDERAL HEALTH INTELLIGENCE / COMPETITIVE INTELLIGENCE</div><h1>Health Competitive Intelligence</h1><p class="sub">Compare supplier positions, strengths and agency relationships across Federal Health. Built __STAMP__ (Australia/Sydney).</p>
<section class="panel controls" aria-label="Dashboard filters">
<div class="field"><label for="agency">Agency</label><select id="agency"></select></div>
<div class="field"><label for="period">Financial year</label><select id="period"></select></div>
<div class="field"><label for="selected">Selected supplier</label><select id="selected"></select></div>
<div class="field"><label for="compare">Compare against</label><select id="compare"></select></div>
<div class="filter-group"><div class="scope-label">Contract status</div><div class="status-pair"><button type="button" id="statusAll" class="active" aria-pressed="true">All contracts</button><button type="button" id="statusActive" aria-pressed="false">Active contracts</button></div></div>
<div class="filter-group market"><div class="scope-label">Market scope</div><div class="switches"><button type="button" id="tam" aria-pressed="false">Accenture TAM</button><button type="button" id="all" class="active" aria-pressed="true">All Federal Health Market</button><button type="button" id="clear" class="clear">Clear filters</button></div></div>
</section>
<section class="insight-box"><h2>Start with the headline comparison</h2><p>The KPI cards provide a quick read of relative scale, market share, strongest Service Offering and breadth of presence. Use the switches to compare active contracts with the historical market.</p><ul class="questions"><li>Who has the larger supplier position?</li><li>Who is more diversified across Service Offerings?</li><li>Is either supplier particularly concentrated in one offering?</li></ul></section>
<div class="cards"><section class="supplier-block selected"><div class="supplier-block-title" id="selectedTitle">Selected supplier</div><div class="supplier-grid"><div class="metric" id="selectedValueCard"><span class="label">Selected period value</span><strong id="selectedValue">–</strong><small id="selectedValueNote"></small></div><div class="metric" id="selectedShareCard"><span class="label">Market share</span><strong id="selectedShare">–</strong><small id="selectedRank"></small></div><div class="metric"><span class="label">Largest Service Offering</span><strong id="selectedLargest">–</strong><small id="selectedLargestValue"></small></div><div class="metric"><span class="label">Service Offering presence</span><strong id="selectedBreadth">–</strong><small id="selectedConcentration"></small></div></div></section>
<section class="supplier-block compare"><div class="supplier-block-title" id="compareTitle">Comparison supplier</div><div class="supplier-grid"><div class="metric" id="compareValueCard"><span class="label">Selected period value</span><strong id="compareValue">–</strong><small id="compareValueNote"></small></div><div class="metric" id="compareShareCard"><span class="label">Market share</span><strong id="compareShare">–</strong><small id="compareRank"></small></div><div class="metric"><span class="label">Largest Service Offering</span><strong id="compareLargest">–</strong><small id="compareLargestValue"></small></div><div class="metric"><span class="label">Service Offering presence</span><strong id="compareBreadth">–</strong><small id="compareConcentration"></small></div></div></section></div>
<section class="insight-box"><h2>Compare where each supplier is strongest</h2><p>The Service Offering comparison reveals where a supplier leads and where the gap is widest. Every value reflects the selected agency, FY, contract-status and market filters.</p><ul class="questions"><li>Which Service Offerings define the suppliers' competitive positions?</li><li>Where is Accenture already credible?</li><li>Where are the largest gaps to close?</li></ul></section>
<section class="panel"><h2>Supplier vs supplier · Service Offering value</h2><div id="profileChart" class="chart"></div></section>
<section class="insight-box"><h2>See the wider competitive field</h2><p>Each Service Offering is ranked independently. The top five suppliers are shown, with both selected comparison suppliers included even if they rank outside the top five.</p><ul class="questions"><li>Who leads each Service Offering?</li><li>How far behind is the selected supplier?</li><li>Which competitors matter most in each segment?</li></ul></section>
<section class="panel"><h2>Service Offering Supplier Leaderboards</h2><p class="note">Shares and ranks are calculated separately for each of the five addressable Service Offerings, using the current filters. All Market includes non-addressable value in overall KPIs; that value has no Service Offering leaderboard.</p><div id="leaderboards" class="leaderboard-grid"></div></section>
<div class="two"><section class="panel"><h2>Competitive trend by financial year</h2><p class="note">Signed award value attributed to notice FY, not annual spending.</p><div id="trendChart" class="chart"></div></section><section class="panel"><h2>Selected supplier by agency</h2><p class="note">Names consolidated only for confirmed historical renames.</p><div id="agencyChart" class="chart"></div></section></div>
<section class="panel"><h2>Overall supplier comparison</h2><p class="note">Click a supplier to focus the comparison. Market shares use all suppliers in the selected scope.</p><div id="supplierTable" class="table-scroll"></div></section>
<p class="note">Source: canonical ATLAS Federal Health master. The original Agency field, supplier groups, addressability, and Service Offering classification remain unchanged. The reporting-only agency_group consolidates confirmed name changes. Active Contracts uses validated CN-family dates as at __AS_AT__. Negative amendments are retained. Values are contract notices, not current cash expenditure or remaining obligation.</p>
</main><script>
const DATA=__DATA__;
const $=id=>document.getElementById(id);const nf=new Intl.NumberFormat('en-AU');
let marketMode='all', statusMode='all';
const money=v=>{v=Number(v)||0;let a=Math.abs(v);return (v<0?'−':'')+'A$'+(a>=1e9?(a/1e9).toFixed(2)+'B':a>=1e6?(a/1e6).toFixed(1)+'M':nf.format(Math.round(a)))};
const percent=v=>Number.isFinite(v)?v.toFixed(1)+'%':'n/a';
const escapeHtml=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const sum=rs=>rs.reduce((a,r)=>a+r.value,0);
function rank(rs,key){const m=new Map();rs.forEach(r=>m.set(r[key],(m.get(r[key])||0)+r.value));return [...m].map(([name,value])=>({name,value})).sort((a,b)=>b.value-a.value||a.name.localeCompare(b.name))}
function labelFY(y){return 'FY'+y+'–'+String((y+1)%100).padStart(2,'0')}
function periodOptions(){return [{v:'all',l:'All historical FYs (default)'},{v:'last5',l:'Last 5 FYs'},{v:'last3',l:'Last 3 FYs'},{v:'current',l:'Current '+labelFY(DATA.current_fy)},...DATA.fys.slice().reverse().map(y=>({v:String(y),l:labelFY(y)})),{v:'unknown',l:'Unknown FY'}]}
function inPeriod(r){const p=$('period').value,key=String(r._fy_key),fy=Number(key),now=DATA.current_fy;if(p==='all')return true;if(p==='unknown')return key==='unknown';if(!/^\d{4}$/.test(key))return false;if(p==='last5')return fy>=now-4&&fy<=now;if(p==='last3')return fy>=now-2&&fy<=now;if(p==='current')return fy===now;return key===p}
function scoped(){const agency=$('agency').value;return DATA.scopes[marketMode].by_supplier.filter(r=>inPeriod(r)&&(agency==='__ALL__'||r.agency_group===agency)&&(statusMode!=='active'||r._contract_status==='Active contracts'))}
function updateSelectors(rs){const ranked=rank(rs,'supplier_group'),allNames=ranked.map(r=>r.name);let s=$('selected').value,c=$('compare').value;
  $('selected').innerHTML=allNames.map(x=>'<option value="'+escapeHtml(x)+'">'+escapeHtml(x)+'</option>').join('');
  $('compare').innerHTML=allNames.map(x=>'<option value="'+escapeHtml(x)+'">'+escapeHtml(x)+'</option>').join('');
  if(!allNames.includes(s))s=allNames.find(n=>/^accenture(?:\b|\s|$)/i.test(n))||allNames[0]||'';
  if(!allNames.includes(c)||c===s)c=allNames.find(n=>n==='KPMG'&&n!==s)||allNames.find(n=>n!==s)||s;
  $('selected').value=s;$('compare').value=c;
}
function supplierSummary(rs,who,total,allRanks){const part=rs.filter(r=>r.supplier_group===who),value=sum(part),byOff=rank(part.filter(r=>DATA.offerings.includes(r._offering)),'_offering'),largest=byOff[0];return {value,share:total>0?100*value/total:NaN,rank:allRanks.findIndex(x=>x.name===who)+1,byOff,largest,breadth:byOff.filter(x=>x.value>0).length}}
function supplierCards(prefix,who,summary){$(''+prefix+'Title').textContent=(prefix==='selected'?'Selected supplier: ':'Comparison supplier: ')+(who||'No supplier');$(prefix+'Value').textContent=money(summary.value);$(prefix+'ValueNote').textContent='Signed recorded value for '+$('period').selectedOptions[0]?.textContent;$(prefix+'Share').textContent=percent(summary.share);$(prefix+'Rank').textContent=summary.rank?'Rank #'+summary.rank+' in the selected market':'Not ranked';$(prefix+'Largest').textContent=summary.largest?.name||'No classified offering';$(prefix+'LargestValue').textContent=summary.largest?money(summary.largest.value)+' in this Service Offering':'–';$(prefix+'Breadth').textContent=summary.breadth+' / '+DATA.offerings.length+' Service Offerings';$(prefix+'Concentration').textContent=summary.breadth?'Count of offerings with positive signed value':'No classified service offering value'}
function colorComparison(a,b){[['selectedValueCard',a.value,b.value],['compareValueCard',b.value,a.value],['selectedShareCard',a.share,b.share],['compareShareCard',b.share,a.share]].forEach(([id,x,y])=>{$(id).classList.toggle('larger',x>y);$(id).classList.toggle('smaller',x<y)})}
function plotBar(id,ranked){if(!window.Plotly)return;const rs=ranked.filter(x=>x.value!==0).slice(0,12).reverse();Plotly.react(id,[{type:'bar',orientation:'h',x:rs.map(x=>x.value/1e6),y:rs.map(x=>x.name),marker:{color:'#8252b6'},customdata:rs.map(x=>x.value),hovertemplate:'%{y}<br>A$%{customdata:,.0f}<extra></extra>'}],{height:Math.max(350,rs.length*30+80),margin:{l:215,r:20,t:14,b:55},xaxis:{title:'Signed contract value (A$ millions)',zeroline:true},yaxis:{automargin:true},paper_bgcolor:'transparent',plot_bgcolor:'transparent'},{responsive:true,displayModeBar:false})}
function plotProfile(a,b,first,second){if(!window.Plotly)return;const x=DATA.offerings,dict=rs=>new Map(rs.byOff.map(v=>[v.name,v.value]));const am=dict(a),bm=dict(b);Plotly.react('profileChart',[{type:'bar',name:first,x,y:x.map(k=>(am.get(k)||0)/1e6),marker:{color:'#a100ff'},customdata:x.map(k=>am.get(k)||0),hovertemplate:'<b>'+escapeHtml(first)+'</b> · %{x}<br>A$%{customdata:,.0f}<extra></extra>'},{type:'bar',name:second,x,y:x.map(k=>(bm.get(k)||0)/1e6),marker:{color:'#0041a9'},customdata:x.map(k=>bm.get(k)||0),hovertemplate:'<b>'+escapeHtml(second)+'</b> · %{x}<br>A$%{customdata:,.0f}<extra></extra>'}],{barmode:'group',height:430,margin:{l:70,r:20,t:20,b:115},yaxis:{title:'Recorded value (A$ millions)',zeroline:true},xaxis:{tickangle:-20,automargin:true},paper_bgcolor:'transparent',plot_bgcolor:'transparent',legend:{orientation:'h',y:1.12}},{responsive:true,displayModeBar:false})}
function showLeaderboards(rs,first,second){const cards=DATA.offerings.map(off=>{const relevant=rs.filter(r=>r._offering===off),market=sum(relevant),sup=rank(relevant,'supplier_group'),head=sup.slice(0,5),wanted=[first,second];wanted.forEach(w=>{const found=sup.find(x=>x.name===w);if(found&&!head.some(x=>x.name===w))head.push(found)});head.sort((a,b)=>sup.findIndex(x=>x.name===a.name)-sup.findIndex(x=>x.name===b.name));const best=Math.max(...sup.map(x=>x.value),0);let last=0;const lines=head.map(x=>{const n=sup.findIndex(p=>p.name===x.name)+1;const spacer=n>last+1?'<div class="lb-separator">···</div>':'';last=n;const chosen=x.name===first?'focus':x.name===second?'compare':'';return spacer+'<div class="leader-row '+chosen+'" title="'+escapeHtml(x.name)+' · '+money(x.value)+'"><span class="lb-rank">#'+n+'</span><span class="lb-name">'+escapeHtml(x.name)+'</span><span class="lb-track"><i style="width:'+Math.max(0,Math.min(100,best>0?100*x.value/best:0))+'%"></i></span><span class="lb-share">'+(market>0?percent(100*x.value/market):'n/a')+'</span></div>'}).join('');return '<div class="leaderboard-card"><h3>'+escapeHtml(off)+'</h3><div class="lb-sub">'+money(market)+' addressable offering value · '+escapeHtml($('period').selectedOptions[0]?.textContent||'')+'</div>'+(lines||'<p class="no-data">No matching classified contracts.</p>')+'</div>'});$('leaderboards').innerHTML=cards.join('')}
function trend(rs,first,second){if(!window.Plotly)return;const yearMap=new Map();rs.forEach(r=>{if(!/^\d{4}$/.test(r._fy_key))return;const y=Number(r._fy_key);if(!yearMap.has(y))yearMap.set(y,{year:y,market:0,a:0,b:0});const p=yearMap.get(y);p.market+=r.value;if(r.supplier_group===first)p.a+=r.value;if(r.supplier_group===second)p.b+=r.value});const years=[...yearMap.values()].sort((a,b)=>a.year-b.year);Plotly.react('trendChart',[{type:'scatter',mode:'lines+markers',name:first,x:years.map(x=>labelFY(x.year)),y:years.map(x=>x.a/1e6),line:{color:'#a100ff'}},{type:'scatter',mode:'lines+markers',name:second,x:years.map(x=>labelFY(x.year)),y:years.map(x=>x.b/1e6),line:{color:'#0041a9'}}],{height:390,margin:{l:65,r:20,t:20,b:70},xaxis:{title:'Notice financial year'},yaxis:{title:'Signed value (A$ millions)'},paper_bgcolor:'transparent',plot_bgcolor:'transparent',legend:{orientation:'h',y:1.12}},{responsive:true,displayModeBar:false})}
function supplierTable(rs,first,second,total){const arr=rank(rs,'supplier_group');$('supplierTable').innerHTML='<table><thead><tr><th>Supplier</th><th>Recorded value</th><th>Market share</th><th>Rank</th></tr></thead><tbody>'+arr.slice(0,60).map((x,i)=>'<tr><td><button type="button" data-supplier="'+escapeHtml(x.name)+'">'+escapeHtml(x.name)+'</button>'+(x.name===first?' · Selected':x.name===second?' · Comparison':'')+'</td><td>'+money(x.value)+'</td><td>'+(total>0?percent(100*x.value/total):'n/a')+'</td><td>#'+(i+1)+'</td></tr>').join('')+'</tbody></table>';$('supplierTable').querySelectorAll('button[data-supplier]').forEach(b=>b.onclick=()=>{$('selected').value=b.dataset.supplier;render(false)})}
function render(refresh=true){const rs=scoped();if(refresh)updateSelectors(rs);const first=$('selected').value,second=$('compare').value,total=sum(rs),allRanks=rank(rs,'supplier_group'),a=supplierSummary(rs,first,total,allRanks),b=supplierSummary(rs,second,total,allRanks);supplierCards('selected',first,a);supplierCards('compare',second,b);colorComparison(a,b);plotProfile(a,b,first,second);showLeaderboards(rs,first,second);trend(rs,first,second);plotBar('agencyChart',rank(rs.filter(x=>x.supplier_group===first),'agency_group'));supplierTable(rs,first,second,total)}
function setMode(m){marketMode=m;[['tam','tam'],['all','all']].forEach(([id,value])=>{$(id).classList.toggle('active',m===value);$(id).setAttribute('aria-pressed',String(m===value))});render()}
function setStatus(status){statusMode=status;[['statusAll','all'],['statusActive','active']].forEach(([id,value])=>{$(id).classList.toggle('active',status===value);$(id).setAttribute('aria-pressed',String(status===value))});render()}
$('agency').innerHTML='<option value="__ALL__">All Federal Health agencies</option>'+DATA.agencies.map(x=>'<option value="'+escapeHtml(x)+'">'+escapeHtml(x)+'</option>').join('');
$('period').innerHTML=periodOptions().map(x=>'<option value="'+x.v+'">'+escapeHtml(x.l)+'</option>').join('');$('period').value='all';
$('agency').onchange=()=>render();$('period').onchange=()=>render();$('selected').onchange=()=>render(false);$('compare').onchange=()=>render(false);$('tam').onclick=()=>setMode('tam');$('all').onclick=()=>setMode('all');$('statusAll').onclick=()=>setStatus('all');$('statusActive').onclick=()=>setStatus('active');
$('clear').onclick=()=>{$('agency').value='__ALL__';$('period').value='all';$('selected').value='';$('compare').value='';setStatus('all');setMode('all')};render();
</script></body></html>'''

def main():
    parser=argparse.ArgumentParser(description='ATLAS Federal Health Competitive Intelligence')
    parser.add_argument('--input',default='health/data/processed/master_health_contracts.parquet')
    parser.add_argument('--output-dir',default='docs/health')
    parser.add_argument('--value-mode',choices=['total','annualised'],default='total')
    args=parser.parse_args()
    df=load(Path(args.input))
    if args.value_mode=='annualised':
        if 'Value Per Year' not in df.columns:
            parser.error('Annualised mode requires Value Per Year')
        df['Value']=pd.to_numeric(df['Value Per Year'],errors='raise')
        if df['Value'].isna().any():
            parser.error('Value Per Year contains null values')
    data=build_data(df)
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    audit=write_agency_audit(df,out/'HealthAgencyGroupingAudit.csv')
    stamp=datetime.now(ZoneInfo('Australia/Sydney')).strftime('%d %b %Y %H:%M %Z')
    html_page=PAGE.replace('__DATA__',json.dumps(data,ensure_ascii=False,allow_nan=False).replace('<','\\u003c')).replace('__STAMP__',html.escape(stamp)).replace('__AS_AT__',html.escape(data['as_at']))
    path=out/'HealthCompetitiveIntelligence.html';path.write_text(html_page,encoding='utf-8')
    print(f'Health Competitive Intelligence: {path}')
    print(f'Health grouping audit: {audit}')
    print(f'TAM: A${data["scopes"]["tam"]["total"]/1e9:.3f}B; All: A${data["scopes"]["all"]["total"]/1e9:.3f}B')
    print(f'Accenture all-vs-TAM reconciled: A${data["scopes"]["tam"]["wins"]/1e9:.3f}B')

if __name__=='__main__':
    main()
