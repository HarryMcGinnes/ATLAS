from __future__ import annotations
import argparse
import html
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd

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
    return df

def rows(frame, group_cols):
    grouped = frame.groupby(group_cols, dropna=False, sort=False).agg(value=('Value','sum'),contracts=('CN ID','nunique')).reset_index()
    return [{**{c:str(r[c]) for c in group_cols},'value':float(r['value']),'contracts':int(r['contracts'])} for r in grouped.to_dict('records')]

def payload(df):
    result = {}
    for mode, frame in [('tam',df[df.is_addressable]),('all',df)]:
        total = float(frame['Value'].sum())
        agencies = rows(frame,['Agency'])
        suppliers = rows(frame,['Agency','supplier_group'])
        offerings = rows(frame,['Agency','_offering'])
        matrix = rows(frame,['Agency','supplier_group','_offering'])
        if abs(sum(r['value'] for r in agencies)-total)>0.01: raise ValueError('Agency totals do not reconcile')
        if abs(sum(r['value'] for r in suppliers)-total)>0.01: raise ValueError('Supplier totals do not reconcile')
        if abs(sum(r['value'] for r in offerings)-total)>0.01: raise ValueError('Offering totals do not reconcile')
        if abs(sum(r['value'] for r in matrix)-total)>0.01: raise ValueError('Matrix totals do not reconcile')
        result[mode] = dict(total=total,contracts=len(frame),agencies=agencies,suppliers=suppliers,offerings=offerings,matrix=matrix)
    return result

HTML = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ATLAS | Health Agency Intelligence</title>
<script src="https://cdn.plot.ly/plotly-3.5.0.min.js"></script><style>
:root{--purple:#6d28d9;--ink:#1d1830;--muted:#667085;--border:#e5e1eb}*{box-sizing:border-box}body{margin:0;font:14px Arial,Helvetica,sans-serif;background:#f7f6fa;color:var(--ink)}a{color:inherit;text-decoration:none}header{background:#14101e;color:white;padding:16px max(20px,calc((100vw - 1260px)/2))}nav{display:flex;align-items:center;gap:20px;flex-wrap:wrap}nav a{font-weight:700;font-size:13px}nav .brand{font-size:24px;letter-spacing:.06em;margin-right:18px}main{max-width:1300px;margin:auto;padding:28px 20px 70px}h1{font-size:32px;margin:0 0 8px}h2{font-size:19px;margin:0 0 10px}p{line-height:1.5}.sub{color:var(--muted);margin:0 0 20px}.panel{background:#fff;border:1px solid var(--border);border-radius:14px;padding:20px;margin:16px 0}.toolbar{display:flex;gap:12px;align-items:end;flex-wrap:wrap}.field{flex:1;min-width:220px}.field label{display:block;font-size:12px;color:var(--muted);font-weight:bold;margin-bottom:7px}select,button{font:inherit;padding:11px 12px;border:1px solid #cbc6d6;border-radius:9px;background:#fff;cursor:pointer}select{width:100%}button.active{background:var(--purple);color:white;border-color:var(--purple)}.modes{display:flex;gap:7px;flex-wrap:wrap}.kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.kpi{background:#fff;border:1px solid var(--border);padding:18px;border-radius:12px}.kpi span{display:block;color:var(--muted);font-size:12px}.kpi strong{display:block;font-size:25px;margin:10px 0}.chart{min-height:350px}.two{display:grid;grid-template-columns:1fr 1fr;gap:16px}.note{color:var(--muted);font-size:12px}.matrix{overflow:auto;max-height:650px}table{border-collapse:collapse;width:100%;font-size:12px}th,td{padding:11px 12px;border-bottom:1px solid #edeaf1;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left;position:sticky;left:0;background:white}thead th{position:sticky;top:0;background:#f2eef8;z-index:2}thead th:first-child{z-index:3;background:#f2eef8}.bar{height:8px;background:#eee8f5;border-radius:9px;overflow:hidden;margin-top:6px}.bar>i{display:block;height:100%;background:var(--purple)}@media(max-width:850px){.two,.kpis{grid-template-columns:1fr 1fr}}@media(max-width:560px){.two,.kpis{grid-template-columns:1fr}}
</style></head><body><header><nav><a class="brand" href="../">ATLAS</a><a href="index.html">Health home</a><a href="AccentureHealthTAMOverview.html">TAM overview</a><a href="HealthCompetitiveIntelligence.html">Competitive intelligence</a><a href="HealthContractExplorer.html">Contract explorer</a><a href="HealthAgencyIntelligence.html" aria-current="page">Agency intelligence</a></nav></header><main>
<h1>Federal Health Agency Intelligence</h1><p class="sub">What agencies buy, who wins the work, and where suppliers compete. Built __STAMP__ (Australia/Sydney).</p>
<div class="panel toolbar"><div class="field"><label for="agency">Agency</label><select id="agency"></select></div><div class="modes"><button id="tam" class="active" aria-pressed="true">Accenture TAM</button><button id="all" aria-pressed="false">All Federal Health Market</button><button id="clear">Clear filters</button></div></div>
<div class="kpis"><div class="kpi"><span>Agency contract value</span><strong id="value">–</strong></div><div class="kpi"><span>Contract notices</span><strong id="contracts">–</strong></div><div class="kpi"><span>Suppliers</span><strong id="suppliers">–</strong></div><div class="kpi"><span>Largest supplier share</span><strong id="topshare">–</strong></div></div>
<div class="two"><section class="panel"><h2>What this agency buys</h2><div id="offerChart" class="chart"></div></section><section class="panel"><h2>Who wins the contracts</h2><div id="supplierChart" class="chart"></div></section></div>
<section class="panel"><h2>Agency intelligence summary</h2><p id="insight"></p><p class="note">Shares are based on signed recorded contract values. Negative amendments are retained; where net totals are non-positive, percentage interpretations are suppressed. These are historical awards, not annual expenditure.</p></section>
<section class="panel"><h2>Supplier × Service Offering breakdown</h2><p class="note">Each cell shows signed contract value. The count of contract notices is available on hover. Select a supplier row to focus on its offering mix.</p><div id="matrix" class="matrix"></div></section>
<p class="note">Source: canonical ATLAS Health master; supplier grouping and addressability are inherited unchanged. No independent reclassification is performed. Contract notices and amendments remain separate records.</p>
</main><script>
const DATA=__DATA__;let mode='tam';const $=id=>document.getElementById(id);const nf=new Intl.NumberFormat('en-AU');function money(x){let a=Math.abs(Number(x)||0);return (x<0?'−':'')+'A$'+(a>=1e9?(a/1e9).toFixed(2)+'B':a>=1e6?(a/1e6).toFixed(1)+'M':nf.format(Math.round(a)))}function pct(v){return Number.isFinite(v)?v.toFixed(1)+'%':'n/a'}function total(a){return a.reduce((s,x)=>s+x.value,0)}function group(data,k){let m=new Map();data.forEach(r=>{let n=r[k];if(!m.has(n))m.set(n,{name:n,value:0,contracts:0});let x=m.get(n);x.value+=r.value;x.contracts+=r.contracts});return [...m.values()].sort((a,b)=>b.value-a.value)}function setText(id,v){$(id).textContent=v}function chart(id,data){if(!window.Plotly)return;let d=data.filter(x=>x.value!==0).slice(0,15).reverse();Plotly.react(id,[{type:'bar',orientation:'h',x:d.map(x=>x.value/1e6),y:d.map(x=>x.name),marker:{color:'#8252b6'},customdata:d.map(x=>[x.contracts,x.value]),hovertemplate:'<b>%{y}</b><br>Value: A$%{customdata[1]:,.0f}<br>CNs: %{customdata[0]}<extra></extra>'}],{margin:{l:220,r:30,t:15,b:55},height:Math.max(370,d.length*35+90),xaxis:{title:'Net contract value (A$ millions)',zeroline:true},yaxis:{automargin:true},paper_bgcolor:'transparent',plot_bgcolor:'transparent'} ,{responsive:true,displayModeBar:false})}function render(){let d=DATA[mode],agency=$('agency').value;let filter=a=>agency==='__ALL__'?a:a.filter(r=>r.Agency===agency);let aa=filter(d.agencies),ss=filter(d.suppliers),oo=filter(d.offerings),mm=filter(d.matrix);let v=total(aa),c=aa.reduce((s,r)=>s+r.contracts,0);let sup=group(ss,'supplier_group'),off=group(oo,'_offering');setText('value',money(v));setText('contracts',nf.format(c));setText('suppliers',nf.format(sup.length));setText('topshare',v>0&&sup.length?pct(sup[0].value/v*100):'n/a');chart('offerChart',off);chart('supplierChart',sup);let who=sup[0],what=off[0];let name=agency==='__ALL__'?'Federal Health agencies':agency;setText('insight',!who?'No contracts in this selection.':name+' has '+money(v)+' in net recorded contract value across '+nf.format(c)+' contract notices. '+(what?'The largest Service Offering by net value is '+what.name+' ('+money(what.value)+'). ':'')+'The largest supplier is '+who.name+' ('+money(who.value)+(v>0?', '+pct(who.value/v*100)+' of the selected market':'')+'). These are descriptive historical observations, not an assessment of future opportunities.');let offerings=[...new Set(mm.map(x=>x._offering))].sort(), suppliers=group(ss,'supplier_group').slice(0,25);let matrixMap=new Map(mm.map(x=>[x.supplier_group+'\u0001'+x._offering,x]));let h='<table><thead><tr><th>Supplier</th>'+offerings.map(x=>'<th>'+esc(x)+'</th>').join('')+'<th>Total</th></tr></thead><tbody>';suppliers.forEach(s=>{h+='<tr><td>'+esc(s.name)+'</td>';offerings.forEach(o=>{let r=matrixMap.get(s.name+'\u0001'+o);h+='<td title="'+(r?r.contracts:0)+' CNs">'+(r?money(r.value):'–')+'</td>'});h+='<td><b>'+money(s.value)+'</b></td></tr>'});h+='</tbody></table>';$('matrix').innerHTML=h}function esc(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}let agencies=[...new Set(DATA.all.agencies.map(x=>x.Agency))].sort();$('agency').innerHTML='<option value="__ALL__">All Federal Health agencies</option>'+agencies.map(x=>'<option value="'+esc(x)+'">'+esc(x)+'</option>').join('');function setMode(m){mode=m;['tam','all'].forEach(x=>{$(x).classList.toggle('active',x===m);$(x).setAttribute('aria-pressed',String(x===m))});render()}$('agency').addEventListener('change',render);$('tam').onclick=()=>setMode('tam');$('all').onclick=()=>setMode('all');$('clear').onclick=()=>{$('agency').value='__ALL__';render()};render();
</script></body></html>'''

def main():
    p=argparse.ArgumentParser(description='ATLAS Federal Health Agency Intelligence')
    p.add_argument('--input',default='health/data/processed/master_health_contracts.parquet')
    p.add_argument('--output-dir',default='docs/health')
    args=p.parse_args()
    df=read_master(Path(args.input))
    data=payload(df)
    stamp=datetime.now(ZoneInfo('Australia/Sydney')).strftime('%d %b %Y %H:%M %Z')
    rendered=HTML.replace('__STAMP__',html.escape(stamp)).replace('__DATA__',json.dumps(data,ensure_ascii=False,allow_nan=False).replace('<','\\u003c'))
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    path=out/'HealthAgencyIntelligence.html';path.write_text(rendered,encoding='utf-8')
    print(f'Health Agency Intelligence: {path}')
    for mode,d in data.items(): print(f'{mode}: {d["contracts"]:,} CNs, A${d["total"]/1e9:.3f}B')
    print('Reconciliation: PASS (agency, supplier, offering, matrix)')
if __name__=='__main__':main()
