"""200 predictive panels, then strictly conditional 30-draw pilot. Never B199."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import os
from pathlib import Path
import time
import traceback
from types import SimpleNamespace
import numpy as np
import pandas as pd
from src.rs_cusum_phase2rd.protocol import stable_seed
from .common import BASE,PARENT,ROOT,CFG,CANDIDATES,sha,write_json,protected_check,rebuild,screen
from .models import schedule_data
from .generator import simulate,component_seeds
from .diagnostics import describe,gates

FIT=None
TEMPLATE=None

def log(message):
    line=time.strftime('%Y-%m-%d %H:%M:%S')+' '+message
    print(line,flush=True)
    with (BASE/'logs/run.log').open('a',encoding='utf-8') as f:f.write(line+'\n')

def init_worker():
    global FIT,TEMPLATE
    FIT=json.loads((BASE/'diagnostics/fitted_models.json').read_text());TEMPLATE=schedule_data()

def freeze():
    target=BASE/'specification/frozen_registry.json'
    if target.exists():raise RuntimeError('FREEZE_ALREADY_EXISTS')
    assert json.loads((BASE/'diagnostics/engineering_tests.json').read_text())['status']=='PASS'
    protected=protected_check()
    parent=json.loads((PARENT/'seed_registry.json').read_text())
    validation=[{'draw_id':i,**{k:stable_seed(CFG['validation_master'],'M4-HS-v0','generator-validation',k,i) for k in ['draw_seed','design_seed','reward_seed']}} for i in range(200)]
    seeds={'validation':validation,'pilot':parent['pilot_draws']}
    vals=[s[k] for items in seeds.values() for s in items for k in ['draw_seed','design_seed','reward_seed']]
    components=[v for items in seeds.values() for s in items for v in component_seeds(s).values()]
    assert len(vals)==len(set(vals)) and len(components)==len(set(components)) and not set(vals)&set(components)
    assert len(seeds['pilot'])==30
    write_json(BASE/'specification/seed_registry.json',seeds)
    paths=[BASE/'M4_HS_V0_SPEC.md',BASE/'specification/config.json',BASE/'specification/seed_registry.json',BASE/'diagnostics/fitted_models.json',BASE/'diagnostics/engineering_tests.json']+sorted((BASE/'src').glob('*.py'))
    write_json(target,{'timestamp':pd.Timestamp.now(tz='UTC').isoformat(),'before_first_simulation':True,'files':{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in paths},'protected':protected,'parent_pilot_registry_hash':sha(PARENT/'seed_registry.json'),'validation_planned':200,'pilot_conditionally_planned':30,'formal_allowed':False})
    log('FROZEN before first trajectory simulation; validation=200; pilot conditional=30; B199 disabled')

def verify_freeze():
    frozen=json.loads((BASE/'specification/frozen_registry.json').read_text())
    assert all(sha(ROOT/p)==h for p,h in frozen['files'].items()),'FROZEN_CONTENT_CHANGED'

def atomic_csv(d,path):
    tmp=Path(str(path)+'.tmp');d.to_csv(tmp,index=False,compression='gzip' if str(path).endswith('.gz') else None);os.replace(tmp,path)

def calculate_pilot(panel,sc,directory):
    from src.rs_cusum.patient_balance import action_feature
    from src.rs_cusum.statistic import boundary_factor
    from src.rs_cusum_phase2r.core import fit_candidate
    from src.rs_cusum_phase2rb.engine import PreparedAnalysis
    from src.rs_cusum_phase2rc.c0_engine import observed_arrays
    from src.rs_cusum_realdata_bridge_v0.runner import layers
    a=json.loads((PARENT/'protocol.json').read_text())['analysis']
    fits={};diags=[]
    for u in CANDIDATES:
        fit=fit_candidate(panel,u,0,60480,a['gamma'],a['ridge_alpha'],a['fqi_max_iterations'],a['fqi_tolerance'],'patient_balanced','patient');fits[u]=fit
        for side in ['left','right']:
            obj=getattr(fit,side);m=obj.model
            diags.append({'candidate':u,'side':side,'converged':m.converged,'iterations':m.iterations,'coefficient_delta':m.coefficient_delta,'n':len(obj.records),'patients':len(obj.patient_ids),**m.diagnostics})
            if not m.converged:raise RuntimeError('FQI_NOT_CONVERGED')
    write_json(directory/'fqi.json',diags)
    support={s.candidate:s for s in sc.candidates};bounds={u:boundary_factor(support[u],0,60480,len(panel.patient_ids),a['boundary_scaling']) for u in CANDIDATES}
    with np.load(PARENT/'data_processed/evaluation_grid.npz') as z:es=z['states'];ea=z['actions']
    prep=PreparedAnalysis(SimpleNamespace(analysis_start=0,analysis_end=60480),panel,CANDIDATES,support,fits,action_feature(es,ea),30240,panel.patient_ids,bounds)
    raw,norm,joint=observed_arrays(prep)
    if not all(np.isfinite(x).all() for x in [raw,norm,joint]):raise FloatingPointError('NONFINITE_STATISTIC')
    values=layers(raw,norm,joint,CANDIDATES.index(30240))
    target=directory/'process.npz';tmp=Path(str(target)+'.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,raw=raw,normalized=norm,joint=joint,D0_D4=values,candidates=np.asarray(CANDIDATES))
    os.replace(tmp,target)
    return values.tolist()

def task(stage,seed):
    start=time.perf_counter();directory=BASE/'results'/stage/f"draw_{seed['draw_id']:03d}"
    directory.mkdir(parents=True,exist_ok=False)
    record={'stage':stage,'seed':seed,'components':component_seeds(seed),'status':'INVALID','failure_category':None,'generation_status':'NOT_STARTED','fqi_status':'NOT_RUN'}
    failure_layer='GENERATION'
    try:
        d,components=simulate(FIT,TEMPLATE,seed);record['generation_status']='PASS'
        atomic_csv(d,directory/'scheduled.csv.gz')
        failure_layer='PANEL_REBUILD';panel,edges,_=rebuild(d)
        atomic_csv(edges,directory/'transition_audit.csv.gz')
        record['generation_scheduled_patients']=len(d['user.index'].unique())
        failure_layer='DIAGNOSTICS';metrics,patients,support,strata,missing=describe(d,panel,edges)
        record.update(metrics=metrics,patients=patients,support=support,strata=strata,missing_lengths=missing)
        if stage=='pilot':
            failure_layer='SUPPORT_REJECTION';sc,_=screen(panel)
            if tuple(sc.admissible_candidates)!=CANDIDATES:raise RuntimeError('FROZEN_CANDIDATE_NOT_ADMISSIBLE')
            failure_layer='FQI_OR_STATISTIC';record['fqi_status']='STARTED'
            record['D0_D4']=calculate_pilot(panel,sc,directory);record['fqi_status']='PASS'
        record['status']='VALID'
    except Exception as exc:
        record.update(failure_category=failure_layer,error=str(exc),traceback=traceback.format_exc())
    record['runtime_seconds']=time.perf_counter()-start
    record['files']={p.name:sha(p) for p in directory.iterdir() if p.is_file() and not p.name.endswith('.tmp')}
    write_json(directory/'checkpoint.json',record)
    return stage,seed['draw_id'],record['status'],record['runtime_seconds']

def verify_checkpoint(path,seed,stage):
    record=json.loads(path.read_text());assert record['seed']==seed and record['stage']==stage
    assert record['components']==component_seeds(seed)
    for name,h in record['files'].items():assert sha(path.parent/name)==h
    assert not list(path.parent.glob('*.tmp'))
    if record['status']=='VALID':
        assert record['generation_scheduled_patients']==37
        assert all(np.isfinite(v) for v in record['metrics'].values())
        if stage=='pilot':assert np.isfinite(record['D0_D4']).all() and record['fqi_status']=='PASS'
    return record

def run_stage(stage,workers):
    verify_freeze();seeds=json.loads((BASE/'specification/seed_registry.json').read_text())[stage]
    root=BASE/'results'/stage;root.mkdir(exist_ok=True)
    permitted={f"draw_{x['draw_id']:03d}" for x in seeds}
    assert {p.name for p in root.iterdir()}<=permitted,'EXTRA_DRAW_DIRECTORY'
    pending=[]
    for seed in seeds:
        path=root/f"draw_{seed['draw_id']:03d}"/'checkpoint.json'
        if path.exists():verify_checkpoint(path,seed,stage)
        elif path.parent.exists():raise RuntimeError('INCOMPLETE_OUTPUT_STOP_NO_SILENT_REPLACEMENT:'+str(path.parent))
        else:pending.append(seed)
    log(f'{stage} planned={len(seeds)} complete={len(seeds)-len(pending)} pending={len(pending)}; no completed seeds repeated')
    with ProcessPoolExecutor(max_workers=workers,initializer=init_worker) as executor:
        tasks={executor.submit(task,stage,seed):seed for seed in pending}
        completed=len(seeds)-len(pending)
        for future in as_completed(tasks):
            st,i,status,seconds=future.result();completed+=1
            log(f'{st} {completed}/{len(seeds)} draw_id={i} {status} runtime={seconds:.2f}s')
            write_json(BASE/'results/run_status.json',{'stage':stage,'completed':completed,'planned':len(seeds),'last_draw':i,'updated_utc':pd.Timestamp.now(tz='UTC').isoformat(),'pid':os.getpid()})
    return [verify_checkpoint(root/f"draw_{s['draw_id']:03d}"/'checkpoint.json',s,stage) for s in seeds]

def validation_summary(rows):
    from heartsteps_v1.scripts.run_observed import load_panel
    from heartsteps_v1.scripts.adapter import config
    panel,s=load_panel(config());s['valid']=True
    d=schedule_data();obs,opat,osup,ostr,omiss=describe(d,panel,s)
    write_json(BASE/'diagnostics/observed_predictive_reference.json',{'metrics':obs,'patients':opat,'support':osup,'strata':ostr,'missing_lengths':omiss})
    valid=[r for r in rows if r['status']=='VALID']
    table=pd.DataFrame([dict(draw_id=r['seed']['draw_id'],**r['metrics']) for r in valid]);table.to_csv(BASE/'diagnostics/validation_panel_metrics.csv',index=False)
    detail={}
    for key in ['patients','support','strata']:
        detail[key]=pd.DataFrame([dict(draw_id=r['seed']['draw_id'],**x) for r in valid for x in r[key]])
        detail[key].to_csv(BASE/f'diagnostics/validation_{key}.csv',index=False)
    fit=json.loads((BASE/'diagnostics/fitted_models.json').read_text())
    checks,coverage=gates(obs,table,pd.DataFrame(opat),detail['patients'],detail['support'],fit['all_positive_families_adequate'])
    checks.to_csv(BASE/'diagnostics/generator_gate_checks.csv',index=False);coverage.to_csv(BASE/'diagnostics/patient_predictive_coverage.csv',index=False)
    pred=[]
    for name,val in obs.items():
        if name in table:
            q=table[name].astype(float).quantile([.05,.5,.95]);pred.append({'metric':name,'observed':val,'sim_q05':q.iloc[0],'sim_median':q.iloc[1],'sim_q95':q.iloc[2]})
    pd.DataFrame(pred).to_csv(BASE/'diagnostics/predictive_comparison.csv',index=False)
    passed=len(rows)==len(valid)==200 and bool(checks['pass'].all())
    result={'gate':'GENERATOR_VALIDATION_PASS' if passed else 'GENERATOR_VALIDATION_FAIL','attempted':len(rows),'valid_generated':len(valid),'invalid_generated':len(rows)-len(valid),'failed_checks':checks.loc[~checks['pass']].to_dict('records'),'pilot_allowed':passed,'statistics_computed':False,'p_values_computed':False}
    write_json(BASE/'results/generator_validation_status.json',result)
    log(result['gate']+' failed_checks='+','.join(checks.loc[~checks['pass'],'check']))
    return passed

def pilot_summary(rows):
    valid=[r for r in rows if r['status']=='VALID'];n=len(rows)
    summary={'attempted':n,'valid':len(valid),'invalid':n-len(valid),'support_failures':sum(r['failure_category']=='SUPPORT_REJECTION' for r in rows),'fqi_statistic_failures':sum(r['failure_category']=='FQI_OR_STATISTIC' for r in rows),'total_draw_seconds':sum(r['runtime_seconds'] for r in rows),'formal_B199_allowed':False,'p_values_computed':False}
    summary['gate']='M4_HS_PILOT_FEASIBLE' if len(valid)>=29 and summary['fqi_statistic_failures']==0 and all(sum(any(s['candidate']==u and s['admissible'] for s in r.get('support',[])) for r in rows)>=29 for u in CANDIDATES) else 'M4_HS_PILOT_NOT_FEASIBLE'
    write_json(BASE/'results/pilot_status.json',summary)
    if valid:pd.DataFrame([dict(draw_id=r['seed']['draw_id'],runtime_seconds=r['runtime_seconds'],**dict(zip(['D0','D1','D2','D3','D4'],r['D0_D4']))) for r in valid]).to_csv(BASE/'results/pilot_D0_D4.csv',index=False)
    return summary

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','run']);parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    if args.stage=='freeze':freeze();return
    # Exclusive stage lock prevents two active runners. A stale lock needs explicit inspection.
    lock=BASE/'results/runner.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    try:
        rows=run_stage('validation',args.workers);passed=validation_summary(rows)
        if passed:pilot_summary(run_stage('pilot',args.workers))
        else:write_json(BASE/'results/pilot_status.json',{'gate':'NOT_RUN_GENERATOR_VALIDATION_FAIL','attempted':0,'valid':0,'invalid':0,'frozen_pilot_seeds_consumed':0,'formal_B199_allowed':False,'p_values_computed':False})
        verify_freeze();write_json(BASE/'diagnostics/protected_after.json',protected_check())
        write_json(BASE/'results/COMPUTATION_COMPLETE.json',{'validation_complete':True,'conditional_pilot_resolved':True,'no_next_stage':True})
        log('COMPUTATION_COMPLETE; no B199; reports pending')
    finally:
        lock.unlink()

if __name__=='__main__':
    try:main()
    except Exception:
        write_json(BASE/'results/TECHNICAL_FAILURE.json',{'traceback':traceback.format_exc(),'utc':pd.Timestamp.now(tz='UTC').isoformat()})
        raise
