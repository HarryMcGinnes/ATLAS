from __future__ import annotations
import argparse
import html
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd
from health_active_contracts import add_contract_status
from health_agency_groups import add_agency_groups, write_agency_audit

REQUIRED = ['CN ID', 'Agency', 'Value', 'supplier_group', 'is_addressable']
OFFERINGS = ['Strategy, Transformation & Advisory', 'SI & Engineering', 'Managed Services & Operations', 'Data, AI & Automation', 'Cloud Infrastructure & Cyber']

def bools(s):
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    return s.fillna(False).astype(str).str.strip().str.lower().isin(['true','1','yes','y'])

def read_master(path):
    df = pd.read_parquet(path) if path.suffix.lower() in ('.parquet','.pq') else pd.read_csv(path, low_memory=False)
    missing = sorted(set(REQUIRED)-set(df.columns))
    if missing: raise ValueError('Missing canonical Health master columns: '+', '.join(missing))
    if df['CN ID'].isna().any() or df['CN ID'].duplicated().any():
        raise ValueError('Missing or duplicate CN IDs in Health master')
    df = df.copy()
    df['Value'] = pd.to_numeric(df['Value'], errors='raise')
    if df['Value'].isna().any(): raise ValueError('Null contract values: cannot reconcile')
    df['is_addressable'] = bools(df['is_addressable'])
    df['Agency'] = df['Agency'].fillna('Unspecified agency').astype(str).str.strip().replace('', 'Unspecified agency')
    df['supplier_group'] = df['supplier_group'].fillna('Unknown supplier').astype(str).str.strip().replace('', 'Unknown supplier')
    so = 'service_offering' if 'service_offering' in df else 'Service Offering'
    if so not in df: raise ValueError('Missing canonical Service Offering')
    df['_offering'] = df[so].fillna('Unresolved').astype(str).str.strip().replace({'':'Unresolved','Uncategorised':'Unresolved'})
    df.loc[~df['is_addressable'], '_offering'] = 'Non-addressable / Unclassified'
    # Financial Year is the recorded AusTender notice FY, NOT a pro-rata spending year.
    if 'Financial Year' not in df.columns:
        raise ValueError('Financial Year missing from Health master: FY filtering requires the canonical field')
    starts = df['Financial Year'].astype('string').str.extract(r'((?:19|20)\d{2})', expand=False)
    df['_fy_start'] = pd.to_numeric(starts, errors='coerce').astype('Int64')
    df['_fy_key'] = df['_fy_start'].astype('string').fillna('unknown').astype(str)
    df = add_agency_groups(df)
    df['_labour_hire'] = labour_hire_mask(df)
    df = add_contract_status(df)
    return df


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

def rows(frame, group_cols):
    grouped = frame.groupby(group_cols, dropna=False, sort=False).agg(value=('Value','sum'),contracts=('CN ID','nunique')).reset_index()
    return [{**{c:str(r[c]) for c in group_cols},'value':float(r['value']),'contracts':int(r['contracts'])} for r in grouped.to_dict('records')]

def payload(df):
    result = {}
    for mode, frame in [('tam',df[df.is_addressable]),('all',df)]:
        total = float(frame['Value'].sum())
        agencies = rows(frame,['agency_group','_fy_key','_contract_status'])
        suppliers = rows(frame,['agency_group','supplier_group','_fy_key','_contract_status'])
        offerings = rows(frame,['agency_group','_offering','_fy_key','_contract_status'])
        matrix = rows(frame,['agency_group','supplier_group','_offering','_fy_key','_contract_status'])
        if abs(sum(r['value'] for r in agencies)-total)>0.01: raise ValueError('Agency totals do not reconcile')
        if abs(sum(r['value'] for r in suppliers)-total)>0.01: raise ValueError('Supplier totals do not reconcile')
        if abs(sum(r['value'] for r in offerings)-total)>0.01: raise ValueError('Offering totals do not reconcile')
        if abs(sum(r['value'] for r in matrix)-total)>0.01: raise ValueError('Matrix totals do not reconcile')
        labour = frame.loc[frame['_labour_hire']]
        labour_agencies = rows(labour, ['agency_group', '_fy_key', '_contract_status'])
        labour_suppliers = rows(labour, ['agency_group', 'supplier_group', '_fy_key', '_contract_status'])
        if abs(sum(x['value'] for x in labour_agencies) - float(labour['Value'].sum())) > 0.01:
            raise ValueError('Labour hire agency totals do not reconcile')
        # Independently auditable relationship and expiry aggregates.
        active = frame.loc[frame['_contract_status'].eq('Active contracts')]
        ending = active.loc[active['_expiry_12m']].copy()
        ending['_end_iso'] = ending['_contract_end'].dt.strftime('%Y-%m-%d')
        relationships = rows(active, ['agency_group','supplier_group','_fy_key'])
        expiring = rows(ending, ['agency_group','supplier_group','_fy_key'])
        # Minimum credible end date for each agency/supplier/FY relationship.
        end_group = active.groupby(['agency_group','supplier_group','_fy_key'], dropna=False)['_contract_end'].min().reset_index()
        next_end = {(str(r['agency_group']),str(r['supplier_group']),str(r['_fy_key'])):
                    r['_contract_end'].strftime('%Y-%m-%d') for r in end_group.to_dict('records')}
        for r in relationships:
            r['next_expiry'] = next_end.get((r['agency_group'],r['supplier_group'],r['_fy_key']), '')
        accenture_mask = bools(frame['is_accenture'])
        accenture_agencies = rows(frame.loc[accenture_mask], ['agency_group','_fy_key','_contract_status'])
        result[mode] = dict(accenture_agencies=accenture_agencies,relationships=relationships, expiring=expiring, total=total,contracts=len(frame),agencies=agencies,suppliers=suppliers,offerings=offerings,matrix=matrix,
                            labour_agencies=labour_agencies, labour_suppliers=labour_suppliers)
    if 'is_accenture' not in df:
        raise ValueError('Missing is_accenture: cannot reconcile the donut wins with canonical master')
    if (bools(df['is_accenture']) & ~df['is_addressable']).any():
        raise ValueError('Accenture contract(s) excluded from TAM: resolve source classification first')
    result['fys'] = sorted(int(v) for v in df['_fy_start'].dropna().unique())
    now = datetime.now(ZoneInfo('Australia/Sydney'))
    result['current_fy'] = now.year if now.month >= 7 else now.year - 1
    result['as_at'] = now.date().isoformat()
    return result

HTML = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ATLAS | Health Agency Intelligence</title>
<script src="https://cdn.plot.ly/plotly-3.5.0.min.js"></script><style>
:root{--purple:#a100ff;--ink:#1d1830;--muted:#667085;--border:#e5e1eb}*{box-sizing:border-box}body{margin:0;font:14px Arial,Helvetica,sans-serif;background:#f7f6fa;color:var(--ink)}a{color:inherit;text-decoration:none}header{background:#11101c;color:white;padding:16px max(20px,calc((100vw - 1260px)/2))}nav{display:flex;align-items:center;gap:20px;flex-wrap:wrap}nav a{font-weight:700;font-size:13px}nav a:not(.brand){padding:9px 12px;border-radius:9px;color:#e8e3ef}nav a[aria-current="page"]{background:#6730ad;color:#fff}nav a:hover{background:#6730ad}nav .brand{font-size:24px;letter-spacing:.06em;margin-right:18px}main{max-width:1500px;margin:auto;padding:28px 20px 70px}h1{font-size:32px;margin:0 0 8px}h2{font-size:19px;margin:0 0 10px}p{line-height:1.5}.sub{color:var(--muted);margin:0 0 20px}.panel{background:#fff;border:1px solid var(--border);border-radius:14px;padding:20px;margin:16px 0}.toolbar{display:flex;gap:12px;align-items:end;flex-wrap:wrap}.field{flex:1;min-width:220px}.field label{display:block;font-size:12px;color:var(--muted);font-weight:bold;margin-bottom:7px}select,button{font:inherit;padding:11px 12px;border:1px solid #cbc6d6;border-radius:9px;background:#fff;cursor:pointer}select{width:100%}button.active{background:var(--purple);color:white;border-color:var(--purple)}.modes{display:flex;gap:7px;flex-wrap:wrap}.kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.kpi{background:#fff;border:1px solid var(--border);padding:18px;border-radius:12px}.kpi span{display:block;color:var(--muted);font-size:12px}.kpi strong{display:block;font-size:25px;margin:10px 0}.chart{min-height:350px}.two{display:grid;grid-template-columns:1fr 1fr;gap:16px}.note{color:var(--muted);font-size:12px}.matrix{overflow:auto;max-height:650px}table{border-collapse:collapse;width:100%;font-size:12px}th,td{padding:11px 12px;border-bottom:1px solid #edeaf1;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left;position:sticky;left:0;background:white}thead th{position:sticky;top:0;background:#f2eef8;z-index:2}thead th:first-child{z-index:3;background:#f2eef8}.bar{height:8px;background:#eee8f5;border-radius:9px;overflow:hidden;margin-top:6px}.bar>i{display:block;height:100%;background:var(--purple)}@media(max-width:850px){.two,.kpis{grid-template-columns:1fr 1fr}}@media(max-width:560px){.two,.kpis{grid-template-columns:1fr}}
.labour-head{display:flex;justify-content:space-between;align-items:start;gap:16px;flex-wrap:wrap}.labour-head h2{margin-bottom:6px}.labour-head p{margin:0;max-width:700px}.labour-link{border:1px solid #a100ff;background:#fbf5ff;color:#7511ab;border-radius:9px;padding:11px 14px;font-size:12px;font-weight:700;white-space:nowrap}.labour-link:hover{background:#f1e5ff}.labour-layout{display:grid;grid-template-columns:1.1fr 1fr;gap:20px;margin-top:18px}.labour-kpis{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.labour-kpis>div{background:#f8f6fb;padding:14px;border:1px solid #eee8f4;border-radius:10px}.labour-kpis span{display:block;color:#667085;font-size:11px}.labour-kpis strong{display:block;margin-top:9px;font-size:22px}.labour-suppliers h3{font-size:13px;margin:0 0 8px}.labour-supplier{display:flex;justify-content:space-between;gap:12px;padding:6px 0;border-bottom:1px solid #f0ecf5;font-size:12px}.labour-supplier b{white-space:nowrap}@media(max-width:900px){.labour-layout{grid-template-columns:1fr}}@media(max-width:560px){.labour-kpis{grid-template-columns:1fr}}
.executive-insights{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.executive-insights>div{border:1px solid #eee8f4;background:#faf8fc;padding:14px;border-radius:10px}.executive-insights b{display:block;margin:6px 0;font-size:19px}.executive-insights span{font-size:12px;color:var(--muted)}@media(max-width:760px){.executive-insights{grid-template-columns:1fr}}
.atlas-status-toggle{display:flex;gap:5px;align-items:center;flex-wrap:nowrap}.atlas-status-toggle button{white-space:nowrap;padding:11px 12px}.atlas-status-toggle button.active{background:var(--purple);border-color:var(--purple);color:#fff}.atlas-status-toggle button:focus-visible{outline:3px solid #c99aff;outline-offset:2px}@media(max-width:560px){.atlas-status-field{width:100%}.atlas-status-toggle button{flex:1}}

/* Executive three-stage donut view, responsive without changing Plotly bars. */
#agencyDonuts{padding:24px;margin-top:14px}.donut-row{display:grid;grid-template-columns:minmax(0,1fr) 90px minmax(0,1fr) 90px minmax(0,1fr);gap:8px;align-items:center;margin-top:18px}.donut-stage{text-align:center;min-width:0}.donut-stage h3{margin:0;font-size:16px;min-height:24px}.donut-stage p{margin:5px 0 14px}.donut-circle{width:min(100%,310px);aspect-ratio:1;border-radius:50%;margin:0 auto;display:grid;place-items:center;background:conic-gradient(#4e20a0 0 100%);position:relative}.donut-inner{width:66%;aspect-ratio:1;border-radius:50%;background:white;display:flex;justify-content:center;align-items:center;flex-direction:column;padding:10px;z-index:1}.donut-inner b{font-size:clamp(18px,2.1vw,30px);letter-spacing:-.035em;white-space:nowrap}.donut-inner span{font-size:11px;color:#667085;margin-top:8px;line-height:1.3}.donut-arrow{display:flex;flex-direction:column;align-items:center;gap:4px;color:#561ec3;white-space:nowrap;font-size:12px}.donut-arrow b{font-size:19px}.donut-arrow span{font-size:11px}.donut-arrow .arrow-shaft{font-size:40px;line-height:1;font-weight:800;max-width:100%}.donut-legend{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-evenly;gap:8px 14px;background:#faf9fc;border:1px solid #e7e2ed;border-radius:10px;padding:13px 8px;margin-top:22px;font-size:11px;min-height:52px}.donut-key{display:flex;align-items:center;gap:6px}.donut-key i{width:12px;height:12px;border-radius:3px;flex:none;display:inline-block}@media(max-width:1050px){.donut-row{grid-template-columns:1fr;gap:12px}.donut-circle{max-width:280px}.donut-stage h3{min-height:0}.donut-arrow{flex-direction:row;justify-content:center}.donut-arrow .arrow-shaft{font-size:25px}.donut-legend{margin-top:12px}}
</style></head><body><header><nav><a class="brand" href="../">ATLAS</a><a href="index.html">Health home</a><a href="AccentureHealthTAMOverview.html">TAM overview</a><a href="HealthCompetitiveIntelligence.html">Competitive intelligence</a><a href="HealthContractExplorer.html">Contract explorer</a><a href="HealthAgencyIntelligence.html" aria-current="page">Agency intelligence</a><a href="Methodology.html">Methodology</a></nav></header><main>
<h1>Federal Health Agency Intelligence</h1><p class="sub">What agencies buy, who wins the work, and where suppliers compete. Built __STAMP__ (Australia/Sydney).</p>
<div class="panel toolbar"><div class="field"><label for="agency">Agency</label><select id="agency"></select></div><div class="field" style="flex:0 1 225px;min-width:190px"><label for="fyFilter">Financial year</label><select id="fyFilter" aria-label="Financial year"></select></div><div class="field atlas-status-field" style="flex:0 1 auto;min-width:240px"><label>Contract status</label><div class="atlas-status-toggle" role="group" aria-label="Contract status"><button type="button" id="statusAll" class="active" aria-pressed="true">All contracts</button><button type="button" id="statusActive" aria-pressed="false">Active contracts</button></div></div><div class="modes"><button id="tam" aria-pressed="false">Accenture TAM</button><button id="all" class="active" aria-pressed="true">All Federal Health Market</button><button id="clear">Clear filters</button></div></div>
<section class="panel" id="agencyDonuts"><h2>Total Addressable Market Overview</h2><p class="note">The three-stage picture for the selected agency, period and contract status. Contracted values are signed award values, not annual expenditure.</p><div class="donut-row"><div class="donut-stage"><h3 id="ring1Title">Total Health procurement market</h3><p id="ring1Subtitle" class="note">Addressable vs non-addressable</p><div class="donut-circle" id="ring1"><div class="donut-inner"><b id="ring1Value">–</b><span id="ring1Label">Market</span></div></div><div class="donut-legend" id="ring1Legend"></div></div><div class="donut-arrow"><b id="ring1Rate">–</b><span>Addressable</span><b id="ring1Transfer">–</b><span class="arrow-shaft">→</span></div><div class="donut-stage"><h3>Within addressable TAM</h3><p class="note">Accenture versus competitors</p><div class="donut-circle" id="ring2"><div class="donut-inner"><b id="ring2Value">–</b><span>Addressable TAM</span></div></div><div class="donut-legend" id="ring2Legend"></div></div><div class="donut-arrow"><b id="ring2Rate">–</b><span>Accenture wins</span><b id="ring2Transfer">–</b><span class="arrow-shaft">→</span></div><div class="donut-stage"><h3>Accenture wins</h3><p class="note">Inside the addressable Health market</p><div class="donut-circle ring-accenture" id="ring3"><div class="donut-inner"><b id="ring3Value">–</b><span>Accenture wins</span></div></div><div class="donut-legend" id="ring3Legend"></div></div></div><p class="note" id="donutNote">Values reconcile to the selection shown below.</p></section>
<div class="kpis"><div class="kpi"><span>Agency contract value</span><strong id="value">–</strong></div><div class="kpi"><span>Contract notices</span><strong id="contracts">–</strong></div><div class="kpi"><span>Suppliers</span><strong id="suppliers">–</strong></div><div class="kpi"><span>Largest supplier share</span><strong id="topshare">–</strong></div></div>
<section class="panel"><h2>Executive insights</h2><p class="note">Automatically generated from the selected market, agency, financial year and contract status. Contract value is signed award value, not annual spending.</p><div id="executiveInsights" class="executive-insights" aria-live="polite"></div></section>
<div class="two"><section class="panel"><h2>What this agency buys</h2><div id="offerChart" class="chart"></div></section><section class="panel"><h2>Who wins the contracts</h2><div id="supplierChart" class="chart"></div></section></div>
<section class="panel" id="labourHireInsights"><div class="labour-head"><div><h2>Labour Hire Insights</h2><p class="note">Explicit labour hire / temporary personnel indicators in AusTender Category or Description. Indicative contracted value, not headcount or actual expenditure.</p></div><a id="labourExplore" class="labour-link" href="HealthContractExplorer.html?view=historical&amp;labour_hire=yes#historical-contracts" target="_blank" rel="noopener noreferrer">View labour hire contracts ↗</a></div><div class="labour-layout"><div class="labour-kpis"><div><span>Contract value</span><strong id="labourValue">–</strong></div><div><span>Contract notices</span><strong id="labourCount">–</strong></div><div><span>Share of selected market</span><strong id="labourShare">–</strong></div></div><div class="labour-suppliers"><h3>Top labour hire suppliers</h3><div id="labourSuppliers"></div></div></div><p class="note">Uses the current agency, financial year and TAM selection. The link opens Historical Contracts in a new tab with those filters and Labour Hire = Yes.</p></section>
<section class="panel"><h2>Current agency–supplier relationships</h2><p class="note">Verified active contracts as at <span id="relationshipAsAt"></span>. Uses Contract Explorer's validated CN-family dates. The table shows active relationships even when All Contracts is selected; FY, agency and market selections still apply. Recorded values are not remaining contract value.</p><div id="relationships" class="matrix"></div></section>
<section class="panel"><h2>Supplier × Service Offering breakdown</h2><p class="note">Each cell shows signed contract value. The count of contract notices is available on hover. Select a supplier row to focus on its offering mix.</p><div id="matrix" class="matrix"></div></section>
<p class="note">Financial-year filters reflect the recorded AusTender FY on each notice, not annual cash expenditure; multiyear contract values are not spread across years. Historical contract values and negative amendments are retained. Source: canonical ATLAS Health master; supplier grouping and addressability are inherited unchanged. No independent reclassification is performed. Contract notices and amendments remain separate records.</p>
</main><script>
const DATA=__DATA__;let mode='all';let contractStatus='all';const $=id=>document.getElementById(id);const nf=new Intl.NumberFormat('en-AU');function money(x){let a=Math.abs(Number(x)||0);return (x<0?'−':'')+'A$'+(a>=1e9?(a/1e9).toFixed(2)+'B':a>=1e6?(a/1e6).toFixed(1)+'M':nf.format(Math.round(a)))}function pct(v){return Number.isFinite(v)?v.toFixed(1)+'%':'n/a'}function total(a){return a.reduce((s,x)=>s+x.value,0)}function group(data,k){let m=new Map();data.forEach(r=>{let n=r[k];if(!m.has(n))m.set(n,{name:n,value:0,contracts:0});let x=m.get(n);x.value+=r.value;x.contracts+=r.contracts});return [...m.values()].sort((a,b)=>b.value-a.value)}function setText(id,v){$(id).textContent=v}function chart(id,data){if(!window.Plotly)return;let d=data.filter(x=>x.value!==0).slice(0,15).reverse();Plotly.react(id,[{type:'bar',orientation:'h',x:d.map(x=>x.value/1e6),y:d.map(x=>x.name),marker:{color:'#8252b6'},customdata:d.map(x=>[x.contracts,x.value]),hovertemplate:'<b>%{y}</b><br>Value: A$%{customdata[1]:,.0f}<br>CNs: %{customdata[0]}<extra></extra>'}],{margin:{l:220,r:30,t:15,b:55},height:Math.max(370,d.length*35+90),xaxis:{title:'Net contract value (A$ millions)',zeroline:true},yaxis:{automargin:true},paper_bgcolor:'transparent',plot_bgcolor:'transparent'} ,{responsive:true,displayModeBar:false})}function matchesFY(r){let choice=$('fyFilter').value;let key=String(r._fy_key);if(choice==='all')return true;if(choice==='unknown')return key==='unknown';if(!/^\d{4}$/.test(key))return false;let y=Number(key),now=DATA.current_fy;if(choice==='last5')return y>=now-4&&y<=now;if(choice==='last3')return y>=now-2&&y<=now;if(choice==='current')return y===now;return key===choice}
function fyOptions(){let year=DATA.current_fy;let label=y=>'FY'+y+'–'+String((y+1)%100).padStart(2,'0');return [{v:'all',l:'All historical FYs (default)'},{v:'last5',l:'Last 5 FYs'},{v:'last3',l:'Last 3 FYs'},{v:'current',l:'Current '+label(year)},...DATA.fys.slice().reverse().map(y=>({v:String(y),l:label(y)})),{v:'unknown',l:'Unknown FY'}]}
function renderDonuts(filter){
  const market=total(filter(DATA.all.agencies));
  const tam=total(filter(DATA.tam.agencies));
  const wins=total(filter(DATA.tam.accenture_agencies));
  const selectedBase=mode==='all'?market:tam;
  const tamShare=selectedBase>0?Math.max(0,Math.min(100,100*tam/selectedBase)):0;
  const winShare=tam>0?Math.max(0,Math.min(100,100*wins/tam)):0;
  const ring=(id,share,fore,back)=>{$(id).style.background='conic-gradient('+fore+' 0 '+share+'%, '+back+' '+share+'% 100%)'};
  ring('ring1',tamShare,'#4e20a0','#d9cef2');ring('ring2',winShare,'#9f75e9','#4e20a0');ring('ring3',wins>0?100:0,'#9f75e9','#eee9f4');
  setText('ring1Title',mode==='tam'?'Accenture TAM':'Total Health procurement market');setText('ring1Subtitle',mode==='tam'?'Selected addressable Health contracts':'All Federal Health procurement in scope');
  setText('ring1Value',money(selectedBase));setText('ring1Label',mode==='tam'?'Addressable TAM':'Total Health market');setText('ring2Value',money(tam));setText('ring3Value',money(wins));
  setText('ring1Rate',selectedBase>0?pct(100*tam/selectedBase):'n/a');setText('ring1Transfer',money(tam));setText('ring2Rate',tam>0?pct(100*wins/tam):'n/a');setText('ring2Transfer',money(wins));
  const legend=(color,name,value)=>'<span class="donut-key"><i style="background:'+color+'"></i>'+esc(name)+' · '+money(value)+'</span>';
  $('ring1Legend').innerHTML=legend('#d9cef2','Not addressable',mode==='tam'?0:market-tam)+legend('#4e20a0','Addressable',tam);
  $('ring2Legend').innerHTML=legend('#4e20a0','Competitors',tam-wins)+legend('#9f75e9','Accenture',wins);
  $('ring3Legend').innerHTML=legend('#9f75e9','Accenture wins',wins);
  setText('donutNote',(mode==='all'?'All Market':'TAM')+' · '+(contractStatus==='active'?'Active Contracts':'All Contracts')+' · '+$('fyFilter').selectedOptions[0]?.textContent+'. If signed values create an invalid ratio, donut segments are visually capped but amounts remain signed.');
}
function render(){let d=DATA[mode],agency=$('agency').value;let filter=a=>a.filter(r=>(agency==='__ALL__'||r.agency_group===agency)&&matchesFY(r)&&(contractStatus!=='active'||r._contract_status==='Active contracts'));let aa=filter(d.agencies),ss=filter(d.suppliers),oo=filter(d.offerings),mm=filter(d.matrix);let v=total(aa),c=aa.reduce((s,r)=>s+r.contracts,0);let sup=group(ss,'supplier_group'),off=group(oo,'_offering');renderDonuts(filter);setText('value',money(v));setText('contracts',nf.format(c));setText('suppliers',nf.format(sup.length));setText('topshare',v>0&&sup.length?pct(sup[0].value/v*100):'n/a');chart('offerChart',off);chart('supplierChart',sup);let labourAg=filter(d.labour_agencies),labourSup=filter(d.labour_suppliers);let lv=total(labourAg),lc=labourAg.reduce((s,r)=>s+r.contracts,0),ls=group(labourSup,'supplier_group');setText('labourValue',money(lv));setText('labourCount',nf.format(lc));setText('labourShare',v>0?pct(lv/v*100):'n/a');$('labourSuppliers').innerHTML=ls.length?ls.slice(0,5).map(x=>'<div class="labour-supplier"><span>'+esc(x.name)+'</span><b>'+money(x.value)+'</b></div>').join(''):'<span class="note">No matching labour hire contracts.</span>';let relationshipRows=d.relationships.filter(r=>(agency==='__ALL__'||r.agency_group===agency)&&matchesFY(r));
let byRel=new Map();relationshipRows.forEach(r=>{let key=r.agency_group+'\u0001'+r.supplier_group;let old=byRel.get(key);if(!old){old={name:r.supplier_group,agency:r.agency_group,value:0,contracts:0,next_expiry:r.next_expiry};byRel.set(key,old)}old.value+=r.value;old.contracts+=r.contracts;if(r.next_expiry&&(!old.next_expiry||r.next_expiry<old.next_expiry))old.next_expiry=r.next_expiry});
let ranked=[...byRel.values()].sort((a,b)=>b.value-a.value).slice(0,35);$('relationshipAsAt').textContent=DATA.as_at;
$('relationships').innerHTML=ranked.length?'<table><thead><tr><th>Supplier</th><th>Agency</th><th>Active notices</th><th>Recorded value</th><th>Next expiry</th></tr></thead><tbody>'+ranked.map(r=>'<tr><td>'+esc(r.name)+'</td><td>'+esc(r.agency)+'</td><td>'+nf.format(r.contracts)+'</td><td>'+money(r.value)+'</td><td>'+esc(r.next_expiry||'Unknown')+'</td></tr>').join('')+'</tbody></table>':'<p class="note">No verified active contract relationships in this selection.</p>';
let expiring=d.expiring.filter(r=>(agency==='__ALL__'||r.agency_group===agency)&&matchesFY(r));let evalue=total(expiring);let biggest=sup[0];let accentureWins=total(filter(d.accenture_agencies));
$('executiveInsights').innerHTML='<div><span>Leading supplier in selection</span><b>'+esc(biggest?biggest.name:'No records')+'</b><span>'+(biggest&&v>0?pct(biggest.value/v*100)+' of selected market':'No comparable share')+'</span></div>'+'<div><span>Accenture position</span><b>'+(accentureWins?money(accentureWins):'No recorded wins')+'</b><span>'+(v>0?pct(accentureWins/v*100)+' of selected market':'In selected scope')+'</span></div>'+'<div><span>Active contracts expiring within 12 months</span><b>'+money(evalue)+'</b><span>Recorded value; subject to selected agency and FY</span></div>';
let query=new URLSearchParams({view:'historical',labour_hire:'yes',market:mode});if(agency!=='__ALL__'){query.set('agency',agency);query.set('agency_group',agency)}query.set('fy',$('fyFilter').value);if(contractStatus==='active')query.set('status','active');$('labourExplore').href='HealthContractExplorer.html?'+query.toString()+'#historical-contracts';let offerings=[...new Set(mm.map(x=>x._offering))].sort(), suppliers=group(ss,'supplier_group').slice(0,25);let matrixMap=new Map();mm.forEach(x=>{let k=x.supplier_group+'\u0001'+x._offering;let old=matrixMap.get(k);if(old){old.value+=x.value;old.contracts+=x.contracts}else matrixMap.set(k,{...x})});let h='<table><thead><tr><th>Supplier</th>'+offerings.map(x=>'<th>'+esc(x)+'</th>').join('')+'<th>Total</th></tr></thead><tbody>';suppliers.forEach(s=>{h+='<tr><td>'+esc(s.name)+'</td>';offerings.forEach(o=>{let r=matrixMap.get(s.name+'\u0001'+o);h+='<td title="'+(r?r.contracts:0)+' CNs">'+(r?money(r.value):'–')+'</td>'});h+='<td><b>'+money(s.value)+'</b></td></tr>'});h+='</tbody></table>';$('matrix').innerHTML=h}function esc(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}$('fyFilter').innerHTML=fyOptions().map(x=>'<option value="'+x.v+'">'+x.l+'</option>').join('');$('fyFilter').value='all';let agencies=[...new Set(DATA.all.agencies.map(x=>x.agency_group))].sort();$('agency').innerHTML='<option value="__ALL__">All Federal Health agencies</option>'+agencies.map(x=>'<option value="'+esc(x)+'">'+esc(x)+'</option>').join('');function setMode(m){mode=m;['tam','all'].forEach(x=>{$(x).classList.toggle('active',x===m);$(x).setAttribute('aria-pressed',String(x===m))});render()}$('agency').addEventListener('change',render);$('fyFilter').addEventListener('change',render);function setContractStatus(s){contractStatus=s;[['statusAll','all'],['statusActive','active']].forEach(([id,value])=>{$(id).classList.toggle('active',s===value);$(id).setAttribute('aria-pressed',String(s===value))});render()}$('statusAll').onclick=()=>setContractStatus('all');$('statusActive').onclick=()=>setContractStatus('active');$('tam').onclick=()=>setMode('tam');$('all').onclick=()=>setMode('all');$('clear').onclick=()=>{$('agency').value='__ALL__';$('fyFilter').value='all';contractStatus='all';$('statusAll').classList.add('active');$('statusActive').classList.remove('active');$('statusAll').setAttribute('aria-pressed','true');$('statusActive').setAttribute('aria-pressed','false');setMode('all')};render();
</script></body></html>'''

def main():
    p=argparse.ArgumentParser(description='ATLAS Federal Health Agency Intelligence')
    p.add_argument('--input',default='health/data/processed/master_health_contracts.parquet')
    p.add_argument('--output-dir',default='docs/health')
    args=p.parse_args()
    df=read_master(Path(args.input))
    data=payload(df)
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    audit=write_agency_audit(df, out/'HealthAgencyGroupingAudit.csv')
    print(f'Health agency grouping audit: {audit}')
    stamp=datetime.now(ZoneInfo('Australia/Sydney')).strftime('%d %b %Y %H:%M %Z')
    rendered=HTML.replace('__STAMP__',html.escape(stamp)).replace('__DATA__',json.dumps(data,ensure_ascii=False,allow_nan=False).replace('<','\\u003c'))
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    path=out/'HealthAgencyIntelligence.html';path.write_text(atlas_option_a_style(rendered, 'agency'),encoding='utf-8')
    print(f'Health Agency Intelligence: {path}')
    for mode in ("tam", "all"):
        d = data[mode]
        print(f'{mode}: {d["contracts"]:,} CNs, A${d["total"]/1e9:.3f}B')
    print('Reconciliation: PASS (agency, supplier, offering, matrix)')

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
    # Primary Health navigation is already in the header. Do not inject a
    # second navigation row; preserve internal tabs in Contract Explorer.
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

if __name__=='__main__':main()
