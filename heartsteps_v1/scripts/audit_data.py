"""Read all four public HeartSteps tables; write only isolated audit artifacts.

No statistic, FQI, calibration, or experimental core is invoked here.
"""
from __future__ import annotations
import hashlib
import json
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'heartsteps_v1'
RAW = ROOT / 'HeartStepsV1-main'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def fetch(url, dest):
    if not dest.exists():
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open('xb') as f:
            f.write(data)
    return {'url': url, 'local_path': str(dest.relative_to(ROOT)), 'sha256': sha(dest)}


def load_suggestions():
    d = pd.read_csv(RAW / 'data_files/suggestions.csv', low_memory=False)
    d['source_row'] = np.arange(len(d)) + 2
    for c in ['sugg.select.utime', 'sugg.decision.utime', 'sugg.context.utime']:
        d[c] = pd.to_datetime(d[c], format='mixed', utc=True)
    d = d.sort_values(['user.index', 'decision.index']).reset_index(drop=True)
    d['local_date'] = (d['sugg.select.utime'] + pd.to_timedelta(d['sugg.gmtoff'], unit='s')).dt.floor('D')
    firstdate = d.groupby('user.index')['local_date'].transform('min')
    d['study_day_from_first_record_date'] = (d.local_date-firstdate).dt.days
    d['elapsed_hours'] = (d['sugg.select.utime'] - d.groupby('user.index')['sugg.select.utime'].transform('min')).dt.total_seconds()/3600
    d['travel_flag'] = d['decision.index.nogap'].isna()
    d['context_age_min'] = (d['sugg.decision.utime'] - d['sugg.context.utime']).dt.total_seconds()/60
    d['scheduled_to_decision_min'] = (d['sugg.decision.utime'] - d['sugg.select.utime']).dt.total_seconds()/60
    return d


def audit():
    for name in ['data_raw', 'data_processed', 'scripts', 'results', 'diagnostics', 'logs']:
        (OUT/name).mkdir(parents=True, exist_ok=True)
    protected = sorted((ROOT/'src').rglob('*.py')) + sorted((ROOT/'configs').glob('*'))
    base = OUT/'diagnostics/protected_source_hashes_before.json'
    hashes = {str(p.relative_to(ROOT)).replace('\\','/'): sha(p) for p in protected if p.is_file()}
    if base.exists():
        assert json.loads(base.read_text(encoding='utf-8')) == hashes, 'PREEXISTING_SOURCE_CHANGED'
    else:
        write_json(base, hashes)
    sources = []
    for label,url in {
        'suggestions_documentation.md': 'https://raw.githubusercontent.com/wiki/klasnja/HeartStepsV1/Documentation-for-suggestions.csv.md',
        'steps_documentation.md': 'https://raw.githubusercontent.com/wiki/klasnja/HeartStepsV1/Documentation-for-jawbone.csv.md',
        'users_documentation.md': 'https://raw.githubusercontent.com/wiki/klasnja/HeartStepsV1/Documentation-for-users.csv.md',
        'workspace.analysis.R': 'https://raw.githubusercontent.com/StatisticalReinforcementLearningLab/HeartstepsV1Code/master/workspace.analysis.R',
        'Main_Effects_Analysis.R': 'https://raw.githubusercontent.com/StatisticalReinforcementLearningLab/HeartstepsV1Code/master/Main%20Effects%20Analysis.R',
        'official_data_tree.json': 'https://api.github.com/repos/klasnja/HeartStepsV1/git/trees/main?recursive=1',
    }.items():
        sources.append(fetch(url, OUT/'data_raw/reference'/label))
    official = json.loads((OUT/'data_raw/reference/official_data_tree.json').read_text())
    blob = {x['path']: x.get('sha') for x in official['tree']}
    tables = {}; files=[]; missing=[]
    for name in ['suggestions','users','jbsteps','gfsteps']:
        path=RAW/'data_files'/f'{name}.csv'; content=path.read_bytes()
        gitblob=hashlib.sha1(f'blob {len(content)}\0'.encode()+content).hexdigest()
        f=pd.read_csv(path, low_memory=False); tables[name]=f
        files.append({'file':name+'.csv','path':str(path),'rows':len(f),'columns':len(f.columns),'participants':f['user.index'].nunique(),'sha256':sha(path),'git_blob_sha1':gitblob,'official_blob_sha1':blob.get('data_files/'+name+'.csv'),'official_exact_match':gitblob==blob.get('data_files/'+name+'.csv')})
        for col in f:
            missing.append({'table':name,'column':col,'rows':len(f),'missing':int(f[col].isna().sum()),'missing_fraction':float(f[col].isna().mean()),'dtype':str(f[col].dtype),'unique_nonmissing':f[col].nunique(dropna=True)})
    pd.DataFrame(files).to_csv(OUT/'diagnostics/data_file_inventory.csv',index=False)
    pd.DataFrame(missing).to_csv(OUT/'diagnostics/all_columns_missingness.csv',index=False)
    write_json(OUT/'data_raw/source_registry.json',{'data_repository':'https://github.com/klasnja/HeartStepsV1','tree_sha':official['sha'],'files':files,'references':sources})
    d=load_suggestions(); users=tables['users']; av=d[d.avail.eq(True)]
    freq=[]
    for scope,g in [('all',d),('available',av)]:
        for col in ['avail','send','sugg.select.slot','sugg.decision.slot','recognized.activity','dec.location.category','dec.weather.condition','is.prefetch']:
            for v,n in g[col].value_counts(dropna=False).items():
                freq.append({'scope':scope,'variable':col,'value':'NA' if pd.isna(v) else str(v),'count':int(n),'denominator':len(g),'fraction':n/len(g)})
    pd.DataFrame(freq).to_csv(OUT/'diagnostics/category_counts.csv',index=False)
    rows=[]; edges=[]
    for pid,g in d.groupby('user.index'):
        a=g[g.avail.eq(True)]
        rows.append({'participant':pid,'decision_points':len(g),'available':len(a),'A0_available':int(a.send.eq(False).sum()),'A1_available':int(a.send.eq(True).sum()),'A1_fraction_available':float(a.send.eq(True).mean()),'study_day_min':g.study_day_from_first_record_date.min(),'study_day_max':g.study_day_from_first_record_date.max(),'decision_index_min':g['decision.index'].min(),'decision_index_max':g['decision.index'].max(),'first_scheduled_UTC':str(g['sugg.select.utime'].min()),'last_scheduled_UTC':str(g['sugg.select.utime'].max()),'travel_rows':int(g.travel_flag.sum()),'disconnect_rows':int(g.connect.eq(False).sum()),'jb_pre_missing':int(g.jbsteps30pre.isna().sum()),'jb_post_missing':int(g.jbsteps30.isna().sum()),'available_jb_pre_missing':int(a.jbsteps30pre.isna().sum()),'available_jb_post_missing':int(a.jbsteps30.isna().sum())})
        for i in range(len(g)-1):
            x,y=g.iloc[i],g.iloc[i+1]
            gap=(y['sugg.select.utime']-x['sugg.select.utime']).total_seconds()/3600
            edges.append({'participant':pid,'from_index':int(x['decision.index']),'to_index':int(y['decision.index']),'index_delta':int(y['decision.index']-x['decision.index']),'from_scheduled_UTC':str(x['sugg.select.utime']),'to_scheduled_UTC':str(y['sugg.select.utime']),'gap_hours':gap,'travel_from':bool(x.travel_flag),'travel_to':bool(y.travel_flag),'connect_from':bool(x.connect),'connect_to':bool(y.connect),'from_available':bool(x.avail),'to_available':bool(y.avail),'nonpositive_gap':gap<=0,'gap_over_24h':gap>24})
    participants=pd.DataFrame(rows); participants.to_csv(OUT/'diagnostics/participant_audit.csv',index=False)
    ed=pd.DataFrame(edges); ed.to_csv(OUT/'diagnostics/all_scheduled_edges.csv',index=False)
    stepqc=[]
    for name in ['jbsteps','gfsteps']:
        f=tables[name].copy(); f['parsed_time']=pd.to_datetime(f['steps.utime'],format='mixed',utc=True)
        for pid,g in f.groupby('user.index'):
            stepqc.append({'table':name,'participant':pid,'records':len(g),'first_time':str(g.parsed_time.min()),'last_time':str(g.parsed_time.max()),'duplicate_timestamp_rows':int(g.duplicated('parsed_time').sum()),'missing_steps':int(g.steps.isna().sum()),'negative_steps':int(g.steps.lt(0).sum()),'zero_steps':int(g.steps.eq(0).sum()),'decision_index_missing':int(g['decision.index'].isna().sum())})
    pd.DataFrame(stepqc).to_csv(OUT/'diagnostics/step_table_qc.csv',index=False)
    candidatecols=['jbsteps30pre','jbsteps30','gfsteps30pre','gfsteps30','dec.location.category','recognized.activity','dec.temperature','dec.weather.condition','sugg.context.utime']
    miss=[]
    for scope,g in [('all',d),('available',av)]:
        for c in candidatecols:
            miss.append({'scope':scope,'column':c,'count':len(g),'missing':int(g[c].isna().sum()),'fraction':float(g[c].isna().mean())})
    pd.DataFrame(miss).to_csv(OUT/'diagnostics/state_outcome_missingness.csv',index=False)
    bad=d[d.avail.eq(False)&d.send.eq(True)]
    bad.to_csv(OUT/'diagnostics/unavailable_sent_anomalies.csv',index=False)
    users[['user.index','first.notif.utime','last.notif.utime','travel.start','travel.end','dropout.date','totaldays']].to_csv(OUT/'diagnostics/user_followup_travel.csv',index=False)
    d[['source_row','user.index','decision.index','decision.index.nogap','sugg.select.utime','sugg.decision.utime','sugg.context.utime','context_age_min','scheduled_to_decision_min','sugg.select.slot','sugg.decision.slot','study_day_from_first_record_date','elapsed_hours','travel_flag','avail','send','is.randomized','is.prefetch','jbsteps30pre','jbsteps30']].to_csv(OUT/'diagnostics/all_decision_timing.csv',index=False)
    zero_checks={c:bool(np.array_equal(d[c].fillna(0).to_numpy(),d[c+'.zero'].to_numpy())) for c in ['jbsteps30','jbsteps30pre']}
    summary={'participants':int(d['user.index'].nunique()),'decision_points':len(d),'available':len(av),'available_A1':int(av.send.eq(True).sum()),'available_A0':int(av.send.eq(False).sum()),'available_A1_fraction':float(av.send.eq(True).mean()),'unavailable_sent':len(bad),'duplicate_decision_keys':int(d.duplicated(['user.index','decision.index']).sum()),'index_nonconsecutive_edges':int(ed.index_delta.ne(1).sum()),'nonpositive_scheduled_edges':int(ed.nonpositive_gap.sum()),'over24h_scheduled_edges':int(ed.gap_over_24h.sum()),'travel_rows':int(d.travel_flag.sum()),'disconnected_rows':int(d.connect.eq(False).sum()),'available_send_randomization_mismatch':int((av.send!=av['is.randomized']).sum()),'context_after_decision_rows':int(d.context_age_min.lt(0).sum()),'slot_mismatch_rows':int((d['sugg.select.slot'].ne(d['sugg.decision.slot'])&d['sugg.decision.slot'].notna()).sum()),'zero_column_exact_fillna0':zero_checks,'all_official_files_exact_match':all(f['official_exact_match'] for f in files)}
    write_json(OUT/'diagnostics/data_audit_summary.json',summary)
    lines=['# HeartSteps V1 descriptive data audit','', 'All four CSVs were fully loaded; no sampling of participants or rows.', '', '```json',json.dumps(summary,indent=2), '```', '', '## Missingness', '', pd.DataFrame(miss).to_markdown(index=False),'','## Per-participant counts','',''+participants.to_markdown(index=False),'','## Interpretation','', '- Available decisions alone define randomized binary treatment. Unavailable decisions are not assigned treatment=0 in the analysis.', '- The original R preprocessing uses log(jbsteps30.zero + 0.5), not log(steps) or log1p(steps). The adapter will retain original missingness and use log(jbsteps30 + 0.5) only when measured.', '- The raw decision.index includes travel; decision.index.nogap compresses it. Both are retained in diagnostics. Non-walking minute records can be absent by design: step-record gaps do not by themselves establish device failure.', '- Public wiki contains apparent typos: gmtoff is seconds (e.g. -14400), not minutes; jbsteps30pre is 30 minutes in the R code, despite one wiki description saying 10.', '- Context records are captured at the decision or prefetch time. Upload/availability time of wearable measurements is not documented, so pre-action measurement does not prove real-time server availability.', '', '## Source references','']
    lines.extend('- '+s['url']+' (local SHA-256 in data_raw/source_registry.json)' for s in sources)
    (OUT/'diagnostics/data_audit_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    audit()
