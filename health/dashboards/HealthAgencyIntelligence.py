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
    df['_labour_hire'] = labour_hire_mask(df)
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
        agencies = rows(frame,['Agency'])
        suppliers = rows(frame,['Agency','supplier_group'])
        offerings = rows(frame,['Agency','_offering'])
        matrix = rows(frame,['Agency','supplier_group','_offering'])
        if abs(sum(r['value'] for r in agencies)-total)>0.01: raise ValueError('Agency totals do not reconcile')
        if abs(sum(r['value'] for r in suppliers)-total)>0.01: raise ValueError('Supplier totals do not reconcile')
        if abs(sum(r['value'] for r in offerings)-total)>0.01: raise ValueError('Offering totals do not reconcile')
        if abs(sum(r['value'] for r in matrix)-total)>0.01: raise ValueError('Matrix totals do not reconcile')
        labour = frame.loc[frame['_labour_hire']]
        labour_agencies = rows(labour, ['Agency'])
        labour_suppliers = rows(labour, ['Agency', 'supplier_group'])
        if abs(sum(x['value'] for x in labour_agencies) - float(labour['Value'].sum())) > 0.01:
            raise ValueError('Labour hire agency totals do not reconcile')
        result[mode] = dict(total=total,contracts=len(frame),agencies=agencies,suppliers=suppliers,offerings=offerings,matrix=matrix,
                            labour_agencies=labour_agencies, labour_suppliers=labour_suppliers)
    return result

HTML = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ATLAS | Health Agency Intelligence</title>
<script src="https://cdn.plot.ly/plotly-3.5.0.min.js"></script><style>
:root{--purple:#a100ff;--ink:#1d1830;--muted:#667085;--border:#e5e1eb}*{box-sizing:border-box}body{margin:0;font:14px Arial,Helvetica,sans-serif;background:#f7f6fa;color:var(--ink)}a{color:inherit;text-decoration:none}header{background:#11101c;color:white;padding:16px max(20px,calc((100vw - 1260px)/2))}nav{display:flex;align-items:center;gap:20px;flex-wrap:wrap}nav a{font-weight:700;font-size:13px}nav a:not(.brand){padding:9px 12px;border-radius:9px;color:#e8e3ef}nav a[aria-current="page"]{background:#6730ad;color:#fff}nav a:hover{background:#6730ad}nav .brand{font-size:24px;letter-spacing:.06em;margin-right:18px}main{max-width:1500px;margin:auto;padding:28px 20px 70px}h1{font-size:32px;margin:0 0 8px}h2{font-size:19px;margin:0 0 10px}p{line-height:1.5}.sub{color:var(--muted);margin:0 0 20px}.panel{background:#fff;border:1px solid var(--border);border-radius:14px;padding:20px;margin:16px 0}.toolbar{display:flex;gap:12px;align-items:end;flex-wrap:wrap}.field{flex:1;min-width:220px}.field label{display:block;font-size:12px;color:var(--muted);font-weight:bold;margin-bottom:7px}select,button{font:inherit;padding:11px 12px;border:1px solid #cbc6d6;border-radius:9px;background:#fff;cursor:pointer}select{width:100%}button.active{background:var(--purple);color:white;border-color:var(--purple)}.modes{display:flex;gap:7px;flex-wrap:wrap}.kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.kpi{background:#fff;border:1px solid var(--border);padding:18px;border-radius:12px}.kpi span{display:block;color:var(--muted);font-size:12px}.kpi strong{display:block;font-size:25px;margin:10px 0}.chart{min-height:350px}.two{display:grid;grid-template-columns:1fr 1fr;gap:16px}.note{color:var(--muted);font-size:12px}.matrix{overflow:auto;max-height:650px}table{border-collapse:collapse;width:100%;font-size:12px}th,td{padding:11px 12px;border-bottom:1px solid #edeaf1;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left;position:sticky;left:0;background:white}thead th{position:sticky;top:0;background:#f2eef8;z-index:2}thead th:first-child{z-index:3;background:#f2eef8}.bar{height:8px;background:#eee8f5;border-radius:9px;overflow:hidden;margin-top:6px}.bar>i{display:block;height:100%;background:var(--purple)}@media(max-width:850px){.two,.kpis{grid-template-columns:1fr 1fr}}@media(max-width:560px){.two,.kpis{grid-template-columns:1fr}}
.labour-head{display:flex;justify-content:space-between;align-items:start;gap:16px;flex-wrap:wrap}.labour-head h2{margin-bottom:6px}.labour-head p{margin:0;max-width:700px}.labour-link{border:1px solid #a100ff;background:#fbf5ff;color:#7511ab;border-radius:9px;padding:11px 14px;font-size:12px;font-weight:700;white-space:nowrap}.labour-link:hover{background:#f1e5ff}.labour-layout{display:grid;grid-template-columns:1.1fr 1fr;gap:20px;margin-top:18px}.labour-kpis{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.labour-kpis>div{background:#f8f6fb;padding:14px;border:1px solid #eee8f4;border-radius:10px}.labour-kpis span{display:block;color:#667085;font-size:11px}.labour-kpis strong{display:block;margin-top:9px;font-size:22px}.labour-suppliers h3{font-size:13px;margin:0 0 8px}.labour-supplier{display:flex;justify-content:space-between;gap:12px;padding:6px 0;border-bottom:1px solid #f0ecf5;font-size:12px}.labour-supplier b{white-space:nowrap}@media(max-width:900px){.labour-layout{grid-template-columns:1fr}}@media(max-width:560px){.labour-kpis{grid-template-columns:1fr}}
</style></head><body><header><nav><a class="brand" href="../">ATLAS</a><a href="index.html">Health home</a><a href="AccentureHealthTAMOverview.html">TAM overview</a><a href="HealthCompetitiveIntelligence.html">Competitive intelligence</a><a href="HealthContractExplorer.html">Contract explorer</a><a href="HealthAgencyIntelligence.html" aria-current="page">Agency intelligence</a><a href="Methodology.html">Methodology</a></nav></header><main>
<h1>Federal Health Agency Intelligence</h1><p class="sub">What agencies buy, who wins the work, and where suppliers compete. Built __STAMP__ (Australia/Sydney).</p>
<div class="panel toolbar"><div class="field"><label for="agency">Agency</label><select id="agency"></select></div><div class="modes"><button id="tam" class="active" aria-pressed="true">Accenture TAM</button><button id="all" aria-pressed="false">All Federal Health Market</button><button id="clear">Clear filters</button></div></div>
<div class="kpis"><div class="kpi"><span>Agency contract value</span><strong id="value">–</strong></div><div class="kpi"><span>Contract notices</span><strong id="contracts">–</strong></div><div class="kpi"><span>Suppliers</span><strong id="suppliers">–</strong></div><div class="kpi"><span>Largest supplier share</span><strong id="topshare">–</strong></div></div>
<div class="two"><section class="panel"><h2>What this agency buys</h2><div id="offerChart" class="chart"></div></section><section class="panel"><h2>Who wins the contracts</h2><div id="supplierChart" class="chart"></div></section></div>
<section class="panel"><h2>Agency intelligence summary</h2><p id="insight"></p><p class="note">Shares are based on signed recorded contract values. Negative amendments are retained; where net totals are non-positive, percentage interpretations are suppressed. These are historical awards, not annual expenditure.</p></section>
<section class="panel" id="labourHireInsights"><div class="labour-head"><div><h2>Labour Hire Insights</h2><p class="note">Explicit labour hire / temporary personnel indicators in AusTender Category or Description. Indicative contracted value, not headcount or actual expenditure.</p></div><a id="labourExplore" class="labour-link" href="HealthContractExplorer.html?view=historical&amp;labour_hire=yes#historical-contracts" target="_blank" rel="noopener noreferrer">View labour hire contracts ↗</a></div><div class="labour-layout"><div class="labour-kpis"><div><span>Contract value</span><strong id="labourValue">–</strong></div><div><span>Contract notices</span><strong id="labourCount">–</strong></div><div><span>Share of selected market</span><strong id="labourShare">–</strong></div></div><div class="labour-suppliers"><h3>Top labour hire suppliers</h3><div id="labourSuppliers"></div></div></div><p class="note">Uses the current agency and TAM selection. The link opens Historical Contracts in a new tab with those filters and Labour Hire = Yes.</p></section>
<section class="panel"><h2>Supplier × Service Offering breakdown</h2><p class="note">Each cell shows signed contract value. The count of contract notices is available on hover. Select a supplier row to focus on its offering mix.</p><div id="matrix" class="matrix"></div></section>
<p class="note">Source: canonical ATLAS Health master; supplier grouping and addressability are inherited unchanged. No independent reclassification is performed. Contract notices and amendments remain separate records.</p>
</main><script>
const DATA=__DATA__;let mode='tam';const $=id=>document.getElementById(id);const nf=new Intl.NumberFormat('en-AU');function money(x){let a=Math.abs(Number(x)||0);return (x<0?'−':'')+'A$'+(a>=1e9?(a/1e9).toFixed(2)+'B':a>=1e6?(a/1e6).toFixed(1)+'M':nf.format(Math.round(a)))}function pct(v){return Number.isFinite(v)?v.toFixed(1)+'%':'n/a'}function total(a){return a.reduce((s,x)=>s+x.value,0)}function group(data,k){let m=new Map();data.forEach(r=>{let n=r[k];if(!m.has(n))m.set(n,{name:n,value:0,contracts:0});let x=m.get(n);x.value+=r.value;x.contracts+=r.contracts});return [...m.values()].sort((a,b)=>b.value-a.value)}function setText(id,v){$(id).textContent=v}function chart(id,data){if(!window.Plotly)return;let d=data.filter(x=>x.value!==0).slice(0,15).reverse();Plotly.react(id,[{type:'bar',orientation:'h',x:d.map(x=>x.value/1e6),y:d.map(x=>x.name),marker:{color:'#8252b6'},customdata:d.map(x=>[x.contracts,x.value]),hovertemplate:'<b>%{y}</b><br>Value: A$%{customdata[1]:,.0f}<br>CNs: %{customdata[0]}<extra></extra>'}],{margin:{l:220,r:30,t:15,b:55},height:Math.max(370,d.length*35+90),xaxis:{title:'Net contract value (A$ millions)',zeroline:true},yaxis:{automargin:true},paper_bgcolor:'transparent',plot_bgcolor:'transparent'} ,{responsive:true,displayModeBar:false})}function render(){let d=DATA[mode],agency=$('agency').value;let filter=a=>agency==='__ALL__'?a:a.filter(r=>r.Agency===agency);let aa=filter(d.agencies),ss=filter(d.suppliers),oo=filter(d.offerings),mm=filter(d.matrix);let v=total(aa),c=aa.reduce((s,r)=>s+r.contracts,0);let sup=group(ss,'supplier_group'),off=group(oo,'_offering');setText('value',money(v));setText('contracts',nf.format(c));setText('suppliers',nf.format(sup.length));setText('topshare',v>0&&sup.length?pct(sup[0].value/v*100):'n/a');chart('offerChart',off);chart('supplierChart',sup);let who=sup[0],what=off[0];let name=agency==='__ALL__'?'Federal Health agencies':agency;setText('insight',!who?'No contracts in this selection.':name+' has '+money(v)+' in net recorded contract value across '+nf.format(c)+' contract notices. '+(what?'The largest Service Offering by net value is '+what.name+' ('+money(what.value)+'). ':'')+'The largest supplier is '+who.name+' ('+money(who.value)+(v>0?', '+pct(who.value/v*100)+' of the selected market':'')+'). These are descriptive historical observations, not an assessment of future opportunities.');let labourAg=filter(d.labour_agencies),labourSup=filter(d.labour_suppliers);let lv=total(labourAg),lc=labourAg.reduce((s,r)=>s+r.contracts,0),ls=group(labourSup,'supplier_group');setText('labourValue',money(lv));setText('labourCount',nf.format(lc));setText('labourShare',v>0?pct(lv/v*100):'n/a');$('labourSuppliers').innerHTML=ls.length?ls.slice(0,5).map(x=>'<div class="labour-supplier"><span>'+esc(x.name)+'</span><b>'+money(x.value)+'</b></div>').join(''):'<span class="note">No matching labour hire contracts.</span>';let query=new URLSearchParams({view:'historical',labour_hire:'yes',market:mode});if(agency!=='__ALL__')query.set('agency',agency);$('labourExplore').href='HealthContractExplorer.html?'+query.toString()+'#historical-contracts';let offerings=[...new Set(mm.map(x=>x._offering))].sort(), suppliers=group(ss,'supplier_group').slice(0,25);let matrixMap=new Map(mm.map(x=>[x.supplier_group+'\u0001'+x._offering,x]));let h='<table><thead><tr><th>Supplier</th>'+offerings.map(x=>'<th>'+esc(x)+'</th>').join('')+'<th>Total</th></tr></thead><tbody>';suppliers.forEach(s=>{h+='<tr><td>'+esc(s.name)+'</td>';offerings.forEach(o=>{let r=matrixMap.get(s.name+'\u0001'+o);h+='<td title="'+(r?r.contracts:0)+' CNs">'+(r?money(r.value):'–')+'</td>'});h+='<td><b>'+money(s.value)+'</b></td></tr>'});h+='</tbody></table>';$('matrix').innerHTML=h}function esc(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}let agencies=[...new Set(DATA.all.agencies.map(x=>x.Agency))].sort();$('agency').innerHTML='<option value="__ALL__">All Federal Health agencies</option>'+agencies.map(x=>'<option value="'+esc(x)+'">'+esc(x)+'</option>').join('');function setMode(m){mode=m;['tam','all'].forEach(x=>{$(x).classList.toggle('active',x===m);$(x).setAttribute('aria-pressed',String(x===m))});render()}$('agency').addEventListener('change',render);$('tam').onclick=()=>setMode('tam');$('all').onclick=()=>setMode('all');$('clear').onclick=()=>{$('agency').value='__ALL__';setMode('tam')};render();
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
    path=out/'HealthAgencyIntelligence.html';path.write_text(atlas_option_a_style(rendered, 'agency'),encoding='utf-8')
    print(f'Health Agency Intelligence: {path}')
    for mode,d in data.items(): print(f'{mode}: {d["contracts"]:,} CNs, A${d["total"]/1e9:.3f}B')
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
