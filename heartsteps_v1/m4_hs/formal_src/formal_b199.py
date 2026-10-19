"""Independent B199 orchestration; imports frozen M4-HS v0 without editing it."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import csv
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
import pandas as pd
from src.rs_cusum_phase2rd.protocol import stable_seed
from heartsteps_v1.m4_hs.src.common import BASE,PARENT,ROOT,CANDIDATES,sha,write_json,protected_check,rebuild,screen
from heartsteps_v1.m4_hs.src.models import schedule_data
from heartsteps_v1.m4_hs.src.generator import simulate,component_seeds
from heartsteps_v1.m4_hs.src.diagnostics import describe
from heartsteps_v1.m4_hs.src.runner import verify_freeze,atomic_csv,calculate_pilot

OUT=BASE/'results/formal_b199'
PROTOCOL=BASE/'M4_HS_FORMAL_B199_PROTOCOL.md'
FIT=None
TEMPLATE=None

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def log(msg):
    line=time.strftime('%Y-%m-%d %H:%M:%S')+' '+msg
    print(line,flush=True)
    with (OUT/'formal.log').open('a',encoding='utf-8') as f:f.write(line+'\n')

def old_protection():
    verify_freeze();summary=protected_check()
    registry=read(BASE/'hash_registry_final.json')['files']
    for p,h in registry.items():assert sha(BASE/p)==h,('PRIOR_M4_HS_CHANGED',p)
    for x in read(PARENT/'data_raw/source_registry.json')['files']:assert sha(x['path'])==x['sha256']
    return {**summary,'m4_hs_prior_files_checked':len(registry),'prior_m4_hs_changed':[]}

def seeds():
    with (OUT/'formal_seed_registry.csv').open(encoding='utf-8',newline='') as f:return [{k:int(v) for k,v in r.items()} for r in csv.DictReader(f)]

def all_seed_values(rows):
    return [int(r[k]) for r in rows for k in ['draw_seed','design_seed','reward_seed']]+[int(v) for r in rows for v in component_seeds(r).values()]

def seed_check(new):
    old=read(BASE/'specification/seed_registry.json')
    prior=all_seed_values(old['validation']+old['pilot'])+[int(read(PARENT/'seed_registry.json')['evaluation_grid'])]
    values=all_seed_values(new)
    assert len(new)==199 and [r['draw_id'] for r in new]==list(range(199))
    assert len(values)==len(set(values)) and not set(values)&set(prior)
    return {'registered':199,'duplicate_seed':0,'prior_seed_overlap':0,'unique_triplet_and_component_seeds':len(values)}

def freeze():
    assert not OUT.exists(),'FORMAL_DIRECTORY_ALREADY_EXISTS: inspect; never overwrite'
    protected=old_protection()
    assert read(BASE/'results/pilot_status.json')['gate']=='M4_HS_PILOT_FEASIBLE'
    assert read(BASE/'results/generator_validation_status.json')['gate']=='GENERATOR_VALIDATION_PASS'
    rows=[{'draw_id':j,**{k:stable_seed(2026091602,'HeartSteps','M4-HS-v0','formal-B199',k,j) for k in ['draw_seed','design_seed','reward_seed']}} for j in range(199)]
    seed_audit=seed_check(rows);OUT.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(OUT/'formal_seed_registry.csv',index=False)
    write_json(OUT/'formal_component_seed_registry.json',[{'draw_id':r['draw_id'],**component_seeds(r)} for r in rows])
    write_json(OUT/'formal_seed_registry_hash.json',{'master':2026091602,'triplets_sha256':sha(OUT/'formal_seed_registry.csv'),'components_sha256':sha(OUT/'formal_component_seed_registry.json'),**seed_audit,'frozen_before_first_draw':True})
    files=[PROTOCOL,Path(__file__),OUT/'formal_seed_registry.csv',OUT/'formal_component_seed_registry.json',OUT/'formal_seed_registry_hash.json',BASE/'hash_registry_final.json',BASE/'manifest.json',BASE/'specification/frozen_registry.json']
    files += [Path(x['path']) for x in read(PARENT/'data_raw/source_registry.json')['files']]
    files += [PARENT/'protocol.json',PARENT/'data_processed/evaluation_grid.npz',PARENT/'data_processed/transitions.csv',PARENT/'results/observed_process.npz',PARENT/'results/observed_global_statistics.csv',PARENT/'results/candidate_process.csv',PARENT/'results/observed_status.json']
    write_json(OUT/'formal_frozen_hashes.json',{'files':{str(p.resolve()):sha(p) for p in files},'prior_protection':protected,'utc':pd.Timestamp.now(tz='UTC').isoformat(),'numpy':np.__version__,'pandas':pd.__version__,'gamma':0.9825931938526898,'candidates':list(CANDIDATES),'B':199,'alpha':.05,'p_function':'src.rs_cusum_phase2r.core.finite_p','critical':'numpy.quantile(valid, .95, axis=0), default linear','reject':'p<=.05','valid_draw_convention':'finite whole draws; V>0 and all critical values finite positive; no replacement'})
    log('FROZEN: 199 new triplets; prior pilot/validation excluded; no observed values loaded')

def verify():
    for p,h in read(OUT/'formal_frozen_hashes.json')['files'].items():assert sha(p)==h,('FORMAL_FROZEN_FILE_CHANGED',p)
    verify_freeze();seed_check(seeds())

def init():
    global FIT,TEMPLATE
    FIT=read(BASE/'diagnostics/fitted_models.json');TEMPLATE=schedule_data()

def one(seed):
    folder=OUT/'draws'/f"draw_{seed['draw_id']:03d}";folder.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();stage='GENERATIVE_MODEL_FAILURE'
    result={'seed':seed,'component_seeds':component_seeds(seed),'valid':False,'failure_category':None,'fqi_status':'NOT_RUN','stage':'formal_b199'}
    try:
        d,_=simulate(FIT,TEMPLATE,seed);atomic_csv(d,folder/'scheduled.csv.gz')
        assert set(d['user.index'])==set(range(1,38))
        stage='PANEL_REBUILD_FAILURE';panel,edges,_=rebuild(d);atomic_csv(edges,folder/'transition_audit.csv.gz')
        stage='DIAGNOSTICS_FAILURE';metrics,patients,support,strata,missing=describe(d,panel,edges)
        result.update(metrics=metrics,patients=patients,support=support,strata=strata,missing_lengths=missing)
        pd.DataFrame(support).to_csv(folder/'support.csv',index=False)
        stage='SUPPORT_REJECTION';sc,_=screen(panel)
        if tuple(sc.admissible_candidates)!=CANDIDATES:raise RuntimeError('FROZEN_CANDIDATE_NOT_ADMISSIBLE')
        stage='FQI_NUMERICAL_FAILURE';result['fqi_status']='STARTED'
        values=calculate_pilot(panel,sc,folder)
        if not np.isfinite(values).all():stage='NONFINITE_STATISTIC';raise FloatingPointError('NONFINITE_STATISTIC')
        result.update(D0_D4=values,fqi_status='PASS',valid=True)
    except Exception as exc:
        if 'NONFINITE_STATISTIC' in str(exc):stage='NONFINITE_STATISTIC'
        result.update(failure_category=stage,error=str(exc),traceback=traceback.format_exc())
    result['runtime_seconds']=time.perf_counter()-started
    result['files']={p.name:sha(p) for p in folder.iterdir() if p.is_file() and not p.name.endswith('.tmp')}
    write_json(folder/'checkpoint.json',result)
    return seed['draw_id'],result['valid'],result['runtime_seconds']

def checkpoint(seed):
    folder=OUT/'draws'/f"draw_{seed['draw_id']:03d}";r=read(folder/'checkpoint.json')
    assert r['seed']==seed and r['component_seeds']==component_seeds(seed) and r['stage']=='formal_b199'
    for p,h in r['files'].items():assert sha(folder/p)==h
    assert not list(folder.glob('*.tmp'))
    if r['valid']:assert np.isfinite(r['D0_D4']).all() and r['fqi_status']=='PASS'
    else:assert r['failure_category'] and r['error']
    return r

def compute(workers):
    verify();registered=seeds();pending=[];allowed={f"draw_{r['draw_id']:03d}" for r in registered}
    (OUT/'draws').mkdir(exist_ok=True)
    assert {x.name for x in (OUT/'draws').iterdir()}<=allowed
    for r in registered:
        folder=OUT/'draws'/f"draw_{r['draw_id']:03d}"
        if (folder/'checkpoint.json').exists():checkpoint(r)
        elif folder.exists():raise RuntimeError('PARTIAL_DRAW_REQUIRES_EXPLICIT_TECHNICAL_REVIEW:'+str(folder))
        else:pending.append(r)
    log(f'FORMAL_START completed={199-len(pending)} pending={len(pending)} workers={workers}; completed seeds never repeated')
    with ProcessPoolExecutor(max_workers=workers,initializer=init) as pool:
        futures=[pool.submit(one,r) for r in pending];completed=199-len(pending)
        for f in as_completed(futures):
            j,valid,seconds=f.result();completed+=1
            log(f'FORMAL {completed}/199 draw={j} valid={valid} runtime={seconds:.2f}s')
            write_json(OUT/'run_status.json',{'stage':'COMPUTATION','completed':completed,'planned':199,'last_draw':j,'pid':os.getpid(),'updated_utc':pd.Timestamp.now(tz='UTC').isoformat()})
    verify();write_json(OUT/'COMPUTATION_COMPLETE.json',{'completed':199,'observed_numeric_values_loaded':False})

def audit_one(seed):
    """Independent readback; reconstruct support only, never refit FQI."""
    from src.rs_cusum.types import SupportPanel,SupportTransition
    from src.rs_cusum.support import load_support_rule,screen_candidates
    from src.rs_cusum_realdata_bridge_v0.runner import layers
    r=checkpoint(seed);folder=OUT/'draws'/f"draw_{seed['draw_id']:03d}"
    if not r['valid']:return {'draw_id':seed['draw_id'],'valid':False,'recorded_failure':r['failure_category'],'audit':'PASS_RECORDED_INVALID'}
    d=pd.read_csv(folder/'scheduled.csv.gz',low_memory=False)
    for key in ['sugg.select.utime','sugg.decision.utime','sugg.context.utime']:d[key]=pd.to_datetime(d[key],utc=True,format='mixed')
    assert len(d)==7741 and d.source_row.is_unique and set(d['user.index'])==set(range(1,38))
    assert np.isfinite(d[['latent_pre','latent_post','latent_location','latent_action']]).all().all()
    assert set(d.latent_action)<=set([0,1]) and set(d.latent_location)<=set([0,1])
    for prefix,col,latent in [('pre','jbsteps30pre','latent_pre'),('post','jbsteps30','latent_post')]:
        mask=d[prefix+'_observed'];assert np.array_equal(mask,d[col].notna())
        assert np.array_equal(d.loc[mask,col],d.loc[mask,latent])
        assert (d[latent]>=0).all() and np.array_equal(d[latent],np.floor(d[latent]))
    assert np.array_equal(d.location_observed,d['dec.location.category'].notna())
    good=d.technical_eligible
    assert d.loc[~good,'send'].isna().all() and (d.loc[~d.avail,'latent_action']==0).all()
    assert np.array_equal(d.loc[good,'send'].astype(int),d.loc[good,'latent_action'])
    assert (d.loc[good,'sugg.decision.utime']-d.loc[good,'sugg.select.utime']).dt.total_seconds().eq(90).all()
    assert (d.loc[good,'sugg.context.utime']==d.loc[good,'sugg.decision.utime']).all()
    template=TEMPLATE.set_index('source_row').loc[d.source_row]
    assert np.array_equal(d['user.index'],template['user.index']) and np.array_equal(d['sugg.select.slot'],template['sugg.select.slot'])
    assert (d['sugg.select.utime'].array==template['sugg.select.utime'].array).all()
    edges=pd.read_csv(folder/'transition_audit.csv.gz',low_memory=False);edges=edges[edges.valid].reset_index(drop=True)
    dd=d.set_index('source_row');cur=dd.loc[edges.from_source_row].reset_index();nxt=dd.loc[edges.to_source_row].reset_index()
    assert cur.avail.all() and nxt.avail.all() and cur.technical_eligible.all() and nxt.technical_eligible.all()
    assert np.array_equal(cur['user.index'],nxt['user.index']) and np.array_equal(edges.participant,cur['user.index'])
    assert np.array_equal(edges.action,cur.latent_action)
    assert np.allclose(edges.reward,np.log(cur.jbsteps30+.5),rtol=1e-13,atol=1e-13)
    names=['log_jbsteps30pre_plus_0p5','slot_sine','slot_cosine','home_or_work']
    for prefix,g in [('state_',cur),('next_state_',nxt)]:
        expected=np.column_stack([np.log(g.jbsteps30pre+.5),np.sin(2*np.pi*(g['sugg.select.slot']-1)/5),np.cos(2*np.pi*(g['sugg.select.slot']-1)/5),g.latent_location])
        actual=edges[[prefix+n for n in names]].to_numpy();assert np.isfinite(actual).all() and np.allclose(actual,expected,rtol=1e-13,atol=1e-13)
    gap=(nxt['sugg.decision.utime']-cur['sugg.decision.utime']).dt.total_seconds()/60
    assert np.allclose(edges.gap_minutes,gap) and gap.between(30,1440).all()
    origin=TEMPLATE.groupby('user.index')['sugg.select.utime'].min()
    elapsed=(cur['sugg.decision.utime'].array-pd.DatetimeIndex(cur['user.index'].map(origin))).total_seconds()/60
    assert np.allclose(edges.elapsed_minutes,elapsed)
    assert not pd.DataFrame({'pid':edges.participant,'t':np.floor(elapsed)}).duplicated().any()
    immediate={}
    badprefix=np.r_[0,np.cumsum((~d.technical_eligible).to_numpy())]
    pos=dict(zip(d.source_row,d.index))
    for _,g in d.groupby('user.index'):
        avail=g.loc[g.avail,'source_row'].to_list();immediate.update(zip(avail[:-1],avail[1:]))
    for fr,to in zip(edges.from_source_row,edges.to_source_row):
        assert immediate[fr]==to
        assert badprefix[pos[to]+1]-badprefix[pos[fr]]==0
    recs=tuple(SupportTransition(str(int(pid)),int(np.floor(t)),int(a)) for pid,t,a in zip(edges.participant,elapsed,edges.action))
    protocol=read(PARENT/'protocol.json');rule=load_support_rule(ROOT/protocol['analysis']['support_config'],'primary')
    sc=screen_candidates(SupportPanel(recs,4,protocol['analysis']['clock']),CANDIDATES,0,60480,rule)
    assert tuple(sc.admissible_candidates)==CANDIDATES
    recorded={s['candidate']:s for s in r['support']}
    for c in sc.candidates:
        assert c.admissible==recorded[c.candidate]['admissible']
        for side in [c.left,c.right]:
            s=recorded[c.candidate];prefix=side.side
            assert side.transition_count==s[prefix+'_n'] and side.action_counts[0]==s[prefix+'_A0'] and side.action_counts[1]==s[prefix+'_A1'] and len(side.active_patients)==s[prefix+'_patients']
            assert side.maximum_action_patient_share[1]==s[prefix+'_dominance_A1']
    fqi=read(folder/'fqi.json');assert len(fqi)==14 and all(x['converged'] for x in fqi)
    assert {(x['candidate'],x['side']) for x in fqi}=={(u,s) for u in CANDIDATES for s in ['left','right']}
    with np.load(folder/'process.npz',allow_pickle=False) as z:
        assert np.array_equal(z['candidates'],CANDIDATES)
        assert all(z[k].shape==(7,64) and np.isfinite(z[k]).all() for k in ['raw','normalized','joint'])
        calculated=layers(z['raw'],z['normalized'],z['joint'],CANDIDATES.index(30240))
        assert np.array_equal(calculated,z['D0_D4']) and np.array_equal(calculated,r['D0_D4'])
    return {'draw_id':seed['draw_id'],'valid':True,'audit':'PASS','support_recomputed':True,'D0_D4_recomputed_from_saved_process':True,'n_transitions':len(edges),'fqi_refits_read':14}

def audit(workers):
    assert (OUT/'COMPUTATION_COMPLETE.json').exists();verify();registered=seeds()
    dirs=list((OUT/'draws').iterdir());assert len(dirs)==199 and {p.name for p in dirs}=={f'draw_{i:03d}' for i in range(199)}
    assert not list(OUT.rglob('*.tmp'))
    with ProcessPoolExecutor(max_workers=workers,initializer=init) as pool:
        results=list(pool.map(audit_one,registered))
    protection=old_protection();pd.DataFrame(results).to_csv(OUT/'independent_readback.csv',index=False)
    valid=sum(r['valid'] for r in results)
    a=read(PARENT/'protocol.json')['analysis'];assert a['gamma']==0.9825931938526898 and CANDIDATES==(12096,18144,24192,30240,36288,42336,48384)
    record={'status':'PASS','registered':199,'completed':199,'valid':valid,'invalid':199-valid,'missing':0,'duplicate':0,'extra':0,'unregistered':0,'prior_seed_reuse':0,'temporary_outputs':[],'all_successful_processes_finite':True,'support_recomputed':True,'layer_reduction_recomputed':True,'all_successful_fqi_readable':True,'observed_values_not_loaded_before_audit':True,'protected':protection,'gamma':a['gamma'],'candidates':list(CANDIDATES),'passed_utc':pd.Timestamp.now(tz='UTC').isoformat()}
    write_json(BASE/'diagnostics/formal_b199_integrity_audit.json',record);log(f'INDEPENDENT_AUDIT_PASS valid={valid}/199; observed unblinding now permitted')

def table_md(d):
    def fmt(x):
        if isinstance(x,float):return f'{x:.7g}'
        return str(x)
    return '| '+' | '.join(d.columns)+' |\n| '+' | '.join(['---']*len(d.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(x) for x in row)+' |' for row in d.itertuples(index=False,name=None))+'\n'

def summarize():
    from src.rs_cusum_phase2r.core import finite_p
    assert read(BASE/'diagnostics/formal_b199_integrity_audit.json')['status']=='PASS';verify()
    rows=[checkpoint(s) for s in seeds()];valid=[r for r in rows if r['valid']]
    ref=pd.DataFrame([dict(draw_id=r['seed']['draw_id'],**dict(zip(['D0','D1','D2','D3','D4'],r['D0_D4']))) for r in valid],columns=['draw_id','D0','D1','D2','D3','D4'])
    ref.to_csv(OUT/'D0_D4_reference_distribution.csv',index=False)
    draws=[];fits=[];support=[]
    for r in rows:
        j=r['seed']['draw_id'];draws.append({'draw_id':j,'valid':r['valid'],'failure_category':r['failure_category'],'error':r.get('error',''),'runtime_seconds':r['runtime_seconds'],'transitions':r.get('metrics',{}).get('transitions'),'action_rate':r.get('metrics',{}).get('action_rate'),'patients':r.get('metrics',{}).get('patients')})
        support.extend([dict(draw_id=j,**s) for s in r.get('support',[])])
        p=OUT/'draws'/f'draw_{j:03d}'/'fqi.json'
        if p.exists():fits.extend([dict(draw_id=j,**f) for f in read(p)])
    pd.DataFrame(draws).to_csv(OUT/'formal_draw_summary.csv',index=False);pd.DataFrame(fits).to_csv(OUT/'formal_fqi_diagnostics.csv',index=False)
    s=pd.DataFrame(support);s.to_csv(OUT/'formal_support_all.csv',index=False)
    survival=[]
    for u in CANDIDATES:
        g=s[s.candidate==u] if len(s) else pd.DataFrame()
        n=int(g.admissible.sum()) if len(g) else 0
        survival.append({'candidate':u,'day':u/1440,'registered':199,'evaluated':len(g),'support_pass':n,'support_fail':int((~g.admissible).sum()) if len(g) else 0,'not_evaluated':199-len(g),'survival_over_registered':n/199})
    pd.DataFrame(survival).to_csv(OUT/'formal_candidate_survival.csv',index=False)
    # First numerical observed read in this stage, strictly after independent audit.
    with np.load(PARENT/'results/observed_process.npz',allow_pickle=False) as z:observed=z['observed_D0_D4'].copy();tau=float(z['tau_hat_D4'])
    results=[];v=len(valid);values=ref[['D0','D1','D2','D3','D4']].to_numpy(float)
    crit=np.quantile(values,.95,axis=0) if v else np.full(5,np.nan)
    available=bool(v and np.isfinite(crit).all() and (crit>0).all())
    for j,layer in enumerate(['D0','D1','D2','D3','D4']):
        x=values[:,j];q=np.quantile(x,[.5,.9,.95,.975,.99]) if v else np.full(5,np.nan)
        p=finite_p(float(observed[j]),x) if available else np.nan
        results.append({'layer':layer,'primary':layer=='D4','valid_draws':v,'mean':float(x.mean()) if v else np.nan,'SD':float(x.std(ddof=1)) if v>1 else np.nan,'median':q[0],'q90':q[1],'q95':q[2],'q975':q[3],'q99':q[4],'maximum':float(x.max()) if v else np.nan,'observed':float(observed[j]),'count_ge_observed':int((x>=observed[j]).sum()) if v else None,'p_value':p,'critical_value_005':crit[j],'reject_alpha005':bool(p<=.05) if available else None,'observed_exceeds_interpolated_critical':bool(observed[j]>crit[j]) if available else None,'calibration_available':available})
    cal=pd.DataFrame(results);cal.to_csv(OUT/'formal_calibration_results.csv',index=False)
    primary=results[4];gate='M4_HS_FORMAL_CALIBRATION_COMPLETE' if available else 'M4_HS_FORMAL_CALIBRATION_UNAVAILABLE'
    status={'gate':gate,'registered':199,'completed':199,'valid':v,'invalid':199-v,'primary':primary,'tau_hat_elapsed_days':tau/1440,'fqi_fits':len(fits),'fqi_all_converged':all(f['converged'] for f in fits),'failure_counts':pd.Series([r['failure_category'] for r in rows if not r['valid']],dtype=object).value_counts().to_dict(),'observed_loaded_utc':pd.Timestamp.now(tz='UTC').isoformat(),'next_stage':'STOP'}
    write_json(OUT/'formal_status.json',status)
    plot(cal,values,tau,available)
    interpretation=('在冻结的 M4-HS v0 fitted-stationary-null calibration 下拒绝 stationarity null，提供与 Q-function stationarity 不一致的证据。' if primary['reject_alpha005'] else '在当前 M4-HS v0 calibration 下未拒绝 stationarity null；不等于证明 Q-function stationary。') if available else '有效 reference 不足或临界值无效，calibration unavailable，不能作显著性判断。'
    report=f'''# M4-HS v0 formal B=199 results

Status: {gate}. Registered/completed: 199/199. Valid: {v}. Invalid: {199-v}. The old 30 pilot draws and 200 generator-validation panels are excluded. No seed was replaced or rerun. Primary layer: D4; D0-D3 are diagnostic, not additional primary tests.

## Calibration results

{table_md(cal)}

The frozen convention is `(1 + count >= observed)/(valid draws + 1)`, implemented by existing `finite_p`. The critical value is NumPy's default linear empirical 95th quantile. Reject uses p<=0.05. All ties count toward the upper tail. q99 is mathematically defined but, with 199 draws, its extreme tail has roughly two observations and substantial Monte Carlo uncertainty. With all 199 valid, attainable p-values have increments of 0.005. Quantile and p-value boundary differences, if present, must not be hidden.

## Primary conclusion and unchanged localization

{interpretation}

Observed descriptive argmax remains {tau/1440:g} elapsed study days. This is the location of the maximum frozen observed D4 process, not a confirmed exact change time. M4-HS changes only the reference distribution. The bootstrap's own argmax is not used to redefine the observed location. This result does not establish a causal change in intervention effect.

## Engineering and integrity

Independent readback passed all registered checkpoints before observed numerical unblinding. Triplet/component seeds match the pre-run registry, with no duplicates, extras, unregistered draws, missing IDs or reuse from validation/pilot. Successful process shapes are (7,64), finite, and their D0-D4 match the original layer reduction. Candidate support was recomputed from saved transitions using the unchanged primary rule. FQI diagnostics read: {len(fits)}; all recorded fits converged: {status['fqi_all_converged']}. Failure categories: {status['failure_counts']}. Full diagnostics are in formal_draw_summary.csv, formal_candidate_survival.csv, formal_support_all.csv and formal_fqi_diagnostics.csv.

All prior HeartSteps, M4-HS v0, observed, grid, model, core and raw-data hashes were preserved. Generator-family selection, patient effects, coefficients, burn-in, gamma, support and candidate definitions were not changed. No observed fit was repeated. Runtime per draw and total/median can be read from formal_draw_summary.csv; summed parallel draw runtime is not stage wall time.

## Limitations and Type-I validation

The accepted v0 discrepancies remain: home/work 40.85% versus observed 45.05%, median segments 496 versus 464, missing-episode P90 7 versus 5. Technical causes are combined; conditional masks and complete-pair nuisance fitting may miss informative missingness. Patient effects are conditioned-on shrunken estimates, not full nuisance uncertainty. Scheduled administrative horizons are fixed. Gamma 0.9825931938526898 discounts available opportunities, not fixed wall-clock durations. A stationary augmented scheduled generator does not prove the four-state pooled embedded Q is exactly Markov/stationary.

This B=199 application produces a fitted-null conditional reference for one dataset. It is not a repeated-null Type-I validation study. An additional independently prespecified Type-I/misspecification study would be needed to claim that the HeartSteps-specific adapter has nominal 5% error control; old Locked-M4 calibration and the 200 predictive panels do not establish that claim. No such extra study is executed or automatically authorized here. Do not tune v0 in response to this p-value.

## Stop

No additional generator, sensitivity, power, alternative, simulation study, Ohio analysis or paper writing is started. All outputs belong to the independent formal_b199 stage. The prior v0 manifests are untouched; this stage has its own manifest.
'''
    (BASE/'M4_HS_FORMAL_B199_RESULTS.md').write_text(report,encoding='utf-8')
    txt=f'''M4-HS v0 正式 B=199 已完成：{v}/199 有效，{199-v} 个无效。
Support/FQI等失败分类：{status['failure_counts']}。
Observed D4 = {primary['observed']:.12g}
D4 95% critical value = {primary['critical_value_005']:.12g}
Bootstrap D4 >= observed：{primary['count_ge_observed']}/{v}
Formal p-value = {primary['p_value']:.12g}
Alpha=0.05 reject = {primary['reject_alpha005']}
Observed descriptive tau_hat仍为 {tau/1440:g} elapsed study days，不是confirmed change point。
{interpretation}
不能解释为12.6天发生精确、因果性的treatment-effect改变，也不能证明真实Type-I error恰为5%。
已知局限保持：home/work偏低、segments偏多、缺失episode尾部偏长，patient effects/缺失模型和opportunity-time gamma存在假设。
若要声称HeartSteps适配器具有名义5% Type-I控制，需要另外预注册的验证；本轮没有执行。
完成后停止。没有重跑/混入pilot、换seed、调整generator或更改observed结果。
'''
    (BASE/'M4_HS_FORMAL_B199_SUMMARY.txt').write_text(txt,encoding='utf-8')
    log(gate+'; final report generated; no further experiments')

def plot(cal,values,tau,available):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest=OUT/'figures';dest.mkdir(exist_ok=True)
    if available:
        d4=values[:,4];obs=float(cal.iloc[4].observed)
        fig,ax=plt.subplots(figsize=(7,4));ax.hist(d4,bins=20,color='#527d9e',edgecolor='white');ax.axvline(obs,color='#ae3838',label=f'Observed D4 = {obs:.2f}');ax.set(xlabel='D4',ylabel='Formal fitted-null draws',title='M4-HS v0: 199 registered draws');ax.legend();fig.tight_layout();fig.savefig(dest/'D4_reference_distribution.png',dpi=180);plt.close(fig)
        fig,axes=plt.subplots(1,5,figsize=(13,3.6))
        for j,ax in enumerate(axes):
            ax.bar(['Observed','95% critical'],[cal.iloc[j].observed,cal.iloc[j].critical_value_005],color=['#ae3838','#527d9e']);ax.set_title(f'D{j}'+(' (primary)' if j==4 else ''));ax.tick_params(axis='x',rotation=45)
        fig.suptitle('Frozen observed statistics versus fitted-null reference thresholds');fig.tight_layout();fig.savefig(dest/'D0_D4_observed_vs_critical.png',dpi=180);plt.close(fig)
        fig,ax=plt.subplots(figsize=(7,4));ax.step(np.sort(d4),np.arange(1,len(d4)+1)/len(d4),where='post',color='#527d9e');ax.axvline(obs,color='#ae3838',label='Observed D4');ax.set(xlabel='D4',ylabel='Empirical CDF',title='M4-HS v0 fitted-null reference');ax.legend();fig.tight_layout();fig.savefig(dest/'D4_empirical_CDF.png',dpi=180);plt.close(fig)
    curve=pd.read_csv(PARENT/'results/candidate_process.csv')
    fig,ax=plt.subplots(figsize=(7,4));ax.plot(curve.candidate_elapsed_day,curve.D4_candidate_max_joint,'o-',color='#527d9e');ax.axvline(tau/1440,ls='--',color='#ae3838',label=f'Descriptive observed argmax: {tau/1440:g} days');ax.set(xlabel='Elapsed study days',ylabel='Observed D4 candidate process',title='Unchanged observed HeartSteps process');ax.legend();fig.tight_layout();fig.savefig(dest/'observed_D4_candidate_curve.png',dpi=180);plt.close(fig)

def seal():
    verify();protection=old_protection()
    selected=list(OUT.rglob('*'))+[PROTOCOL,BASE/'M4_HS_FORMAL_B199_RESULTS.md',BASE/'M4_HS_FORMAL_B199_SUMMARY.txt',BASE/'diagnostics/formal_b199_integrity_audit.json',Path(__file__)]
    files={str(p.relative_to(BASE)).replace('\\','/'):sha(p) for p in selected if p.is_file() and p.name!='runner.lock'}
    write_json(BASE/'formal_b199_manifest.json',{'stage':'M4-HS-v0-formal-B199','files':files,'status':read(OUT/'formal_status.json'),'protection':protection,'previous_manifests_modified':False,'next_stage':'STOP'})

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','run','seal']);parser.add_argument('--workers',type=int,default=8);args=parser.parse_args()
    if args.stage=='freeze':freeze();return
    if args.stage=='seal':seal();return
    lock=OUT/'runner.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    try:
        if (OUT/'formal_status.json').exists():raise RuntimeError('FORMAL_STAGE_ALREADY_SUMMARIZED: do not repeat')
        compute(args.workers);audit(args.workers);summarize();seal()
    finally:lock.unlink()

if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write_json(OUT/'TECHNICAL_FAILURE.json',{'traceback':traceback.format_exc(),'utc':pd.Timestamp.now(tz='UTC').isoformat()})
        raise
