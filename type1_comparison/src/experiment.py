"""Frozen paired stationary-null comparison; no changes to existing methods."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.rs_cusum.patient_balance import action_feature
from src.rs_cusum.statistic import boundary_factor
from src.rs_cusum_phase2r.core import finite_p, fit_candidate
from src.rs_cusum_phase2rb.engine import PreparedAnalysis, run_influence_arm
from src.rs_cusum_phase2rc.c0_engine import observed_arrays
from src.rs_cusum_phase2rd.protocol import stable_seed
from src.rs_cusum_realdata_bridge_v0.runner import layers
from heartsteps_v1.m4_hs.src.common import BASE, CANDIDATES, PARENT, ROOT, rebuild, screen, sha
from heartsteps_v1.m4_hs.src.generator import component_seeds, simulate
from heartsteps_v1.m4_hs.src.models import (PIDS, SPECS, design, fit_data, logistic,
                                             ridge, schedule_data)
from heartsteps_v1.m4_hs.src.runner import calculate_pilot

OUT = ROOT / 'type1_comparison'
PROTOCOL = OUT / 'ORIGINAL_VS_M4_TYPE1_PROTOCOL.md'
MASTER = 2026091701
R, B, ALPHA = 200, 199, .05
LAYERS = ('D0', 'D1', 'D2', 'D3', 'D4')
INNER_FIT = None
INNER_TEMPLATE = None


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(str(path) + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def atomic_csv(frame, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(str(path) + '.tmp')
    frame.to_csv(temp, index=False, compression='gzip' if str(path).endswith('.gz') else None)
    os.replace(temp, path)


def atomic_pickle(value, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(str(path) + '.tmp')
    temp.write_bytes(pickle.dumps(value, protocol=5))
    os.replace(temp, path)


def atomic_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(str(path) + '.tmp')
    with temp.open('wb') as f: np.savez_compressed(f, **arrays)
    os.replace(temp, path)


def own_files(folder):
    return {p.name: digest(p) for p in folder.iterdir() if p.is_file() and p.name != 'checkpoint.json' and not p.name.endswith('.tmp')}


def seal(folder, record):
    record = dict(record, files=own_files(folder))
    atomic_json(folder/'checkpoint.json', record)


def verify_seal(folder, expect=None):
    cp = folder/'checkpoint.json'
    if not cp.exists(): raise RuntimeError('INCOMPLETE_UNSEALED_OUTPUT:' + str(folder))
    rec = read(cp)
    if expect is not None and rec.get('seed') != expect: raise RuntimeError('CHECKPOINT_SEED_MISMATCH:' + str(folder))
    if list(folder.glob('*.tmp')): raise RuntimeError('TEMP_OUTPUT:' + str(folder))
    if set(rec['files']) != {p.name for p in folder.iterdir() if p.is_file() and p.name != 'checkpoint.json'}:
        raise RuntimeError('CHECKPOINT_FILE_SET_MISMATCH:' + str(folder))
    for name, h in rec['files'].items():
        if digest(folder/name) != h: raise RuntimeError('CHECKPOINT_HASH_MISMATCH:' + str(folder/name))
    return rec


def old_seed_values():
    past = read(BASE/'specification/seed_registry.json')
    with (BASE/'results/formal_b199/formal_seed_registry.csv').open(newline='', encoding='utf-8') as f:
        past_rows = list(csv.DictReader(f))
    values = []
    for row in past['validation'] + past['pilot'] + past_rows:
        seed = {k:int(row[k]) for k in ('draw_seed','design_seed','reward_seed')}
        values.extend(seed.values()); values.extend(component_seeds(seed).values())
    values.append(int(read(PARENT/'seed_registry.json')['evaluation_grid']))
    return values


def outer_seeds():
    return [dict(outer_id=i,
                 draw_seed=stable_seed(MASTER,'type1','outer',i,'draw_seed'),
                 design_seed=stable_seed(MASTER,'type1','outer',i,'design_seed'),
                 reward_seed=stable_seed(MASTER,'type1','outer',i,'reward_seed'),
                 original_seed=stable_seed(MASTER,'type1','original',i)) for i in range(R)]


def inner_seed(outer_id, draw_id):
    return dict(outer_id=outer_id, draw_id=draw_id,
                draw_seed=stable_seed(MASTER,'type1','inner',outer_id,draw_id,'draw_seed'),
                design_seed=stable_seed(MASTER,'type1','inner',outer_id,draw_id,'design_seed'),
                reward_seed=stable_seed(MASTER,'type1','inner',outer_id,draw_id,'reward_seed'))


def registered():
    with (OUT/'protocol/outer_seed_registry.csv').open(newline='', encoding='utf-8') as f:
        outer = [{k:int(v) for k,v in r.items()} for r in csv.DictReader(f)]
    with (OUT/'protocol/inner_seed_registry.csv').open(newline='', encoding='utf-8') as f:
        inner = [{k:int(v) for k,v in r.items()} for r in csv.DictReader(f)]
    assert outer == outer_seeds()
    assert inner == [inner_seed(i,b) for i in range(R) for b in range(B)]
    assert digest(OUT/'protocol/outer_seed_registry.csv') == read(OUT/'protocol/outer_seed_registry_hash.json')['sha256']
    assert digest(OUT/'protocol/inner_seed_registry.csv') == read(OUT/'protocol/inner_seed_registry_hash.json')['sha256']
    return outer, inner


def protect_paths():
    paths = [PROTOCOL, OUT/'ORIGINAL_VS_M4_IMPLEMENTATION_AUDIT.md',
             OUT/'src/__init__.py', OUT/'src/experiment.py', OUT/'src/finalize.py',
             PARENT/'protocol.json', PARENT/'data_processed/evaluation_grid.npz',
             BASE/'diagnostics/fitted_models.json', BASE/'specification/config.json',
             BASE/'specification/frozen_registry.json',
             BASE/'formal_b199_manifest.json', BASE/'results/formal_b199/formal_status.json',
             BASE/'results/formal_b199/formal_frozen_hashes.json',
             BASE/'formal_src/formal_b199.py',
             ROOT/'src/rs_cusum_phase2rb/engine.py', ROOT/'src/rs_cusum_phase2rc/c0_engine.py',
             ROOT/'src/rs_cusum_phase2r/core.py', ROOT/'src/rs_cusum/support.py',
             ROOT/'src/rs_cusum_phase2rd/protocol.py',
             ROOT/'src/rs_cusum_realdata_bridge_v0/runner.py',
             ROOT/'src/rs_cusum/patient_balance.py', ROOT/'src/rs_cusum/statistic.py',
             ROOT/'configs/rs_cusum_sensitivity.yaml',
             ROOT/'heartsteps_v1/scripts/adapter.py']
    paths += list((BASE/'src').glob('*.py'))
    paths += [BASE/p for p in read(BASE/'formal_b199_manifest.json')['files']]
    return sorted(set(paths), key=str)


def verify_freeze():
    fr = read(OUT/'protocol/frozen_hash_registry.json')
    for path,h in fr['files'].items():
        if digest(path) != h: raise RuntimeError('FROZEN_CONTENT_CHANGED:' + path)
    registered()


def freeze():
    reg = OUT/'protocol/frozen_hash_registry.json'
    if reg.exists(): raise RuntimeError('ALREADY_FROZEN')
    if not PROTOCOL.exists() or not (OUT/'ORIGINAL_VS_M4_IMPLEMENTATION_AUDIT.md').exists():
        raise RuntimeError('AUDIT_OR_PROTOCOL_MISSING')
    from heartsteps_v1.m4_hs.formal_src.formal_b199 import old_protection
    protected = old_protection()
    outer=outer_seeds(); inner=[inner_seed(i,b) for i in range(R) for b in range(B)]
    prior=set(old_seed_values()); values=[]
    for row in outer:
        seed={k:row[k] for k in ('draw_seed','design_seed','reward_seed')}
        values += list(seed.values()) + list(component_seeds(seed).values()) + [row['original_seed']]
    for row in inner:
        values += [row[k] for k in ('draw_seed','design_seed','reward_seed')]
        values += list(component_seeds(row).values())
    if len(values)!=len(set(values)) or prior.intersection(values): raise RuntimeError('SEED_DUPLICATE_OR_OVERLAP')
    atomic_csv(pd.DataFrame(outer),OUT/'protocol/outer_seed_registry.csv')
    atomic_csv(pd.DataFrame(inner),OUT/'protocol/inner_seed_registry.csv')
    atomic_json(OUT/'protocol/outer_seed_registry_hash.json',dict(sha256=digest(OUT/'protocol/outer_seed_registry.csv'),master=MASTER,registered=R,prior_overlap=0,duplicate=0))
    atomic_json(OUT/'protocol/inner_seed_registry_hash.json',dict(sha256=digest(OUT/'protocol/inner_seed_registry.csv'),master=MASTER,registered=R*B,prior_overlap=0,duplicate=0))
    atomic_json(reg,dict(files={str(p.resolve()):digest(p) for p in protect_paths()},
                         prior_protection=protected, planned_outer=R, planned_original_draws=R*B,
                         planned_inner_draws=R*B, master=MASTER, frozen_before_first_outer=True,
                         paired_ci_seed=stable_seed(MASTER,'type1','final','paired-ci'),
                         seed_registry_hashes={p.name:digest(p) for p in (OUT/'protocol/outer_seed_registry.csv',OUT/'protocol/inner_seed_registry.csv')}))
    print('FROZEN: 200 outer seeds, 200 Original seeds, 39,800 inner seed triplets; no prior overlap')


def load_schedule(path):
    d=pd.read_csv(path,low_memory=False)
    for c in ('sugg.select.utime','sugg.decision.utime','sugg.context.utime'):
        d[c]=pd.to_datetime(d[c],utc=True,format='mixed')
    return d


def fit_inner_null(d):
    # Latent truth fields are intentionally inaccessible to the nuisance estimator.
    visible=d.drop(columns=[c for c in d if c.startswith('latent_') or c=='technical_eligible'],errors='ignore')
    tables,targets,initial,_=fit_data(visible)
    truth_families=read(BASE/'diagnostics/fitted_models.json')['models']
    models={}; diagnostics={}
    for name,rows in tables.items():
        x=design(name,rows); y=np.asarray(targets[name],dtype=float)
        binary=(y==0).astype(float) if name in ('pre','reward') else y
        coef,diag=logistic(x,binary)
        m=dict(predictors=['intercept']+['slot'+str(k) for k in range(2,6)]+
               ([] if name in ('reward','loc_observed') else ['log_gap'])+SPECS[name]+
               ['patient_'+str(p) for p in PIDS],coef=coef.tolist(),fit=diag,base_dimension=x.shape[1]-37)
        diagnostics[name]=dict(binary=diag,rows=len(rows))
        if name in ('pre','reward'):
            family=truth_families[name]['positive']['family']
            if family!='empirical_log': raise RuntimeError('UNSUPPORTED_FROZEN_POSITIVE_FAMILY:'+family)
            pos=y>0
            if not np.any(pos): raise RuntimeError('NO_POSITIVE_COUNT:'+name)
            b,resid=ridge(x[pos],np.log(y[pos]))
            if not np.isfinite(b).all() or not np.isfinite(resid).all(): raise RuntimeError('NONFINITE_POSITIVE_FIT:'+name)
            m['positive']=dict(family=family,coef=b.tolist(),residual=resid.tolist(),nb_shape=None,adequate=True,n=int(pos.sum()))
            diagnostics[name]['positive']=dict(family=family,n=int(pos.sum()),residual_sd=float(np.std(resid)))
        models[name]=m
    fit=dict(models=models,initial_pools=initial,patients=list(PIDS),scheduled_n=len(d),
             specification='M4-HS-v0',inner_fitted_on_outer_only=True,
             all_positive_families_adequate=True)
    return fit,diagnostics


def prepare_outer(panel,sc,folder):
    a=read(PARENT/'protocol.json')['analysis']; fits={}; diags=[]
    for u in CANDIDATES:
        f=fit_candidate(panel,u,0,60480,a['gamma'],a['ridge_alpha'],a['fqi_max_iterations'],
                        a['fqi_tolerance'],'patient_balanced','patient')
        fits[u]=f
        for side in ('left','right'):
            obj=getattr(f,side); m=obj.model
            diags.append(dict(candidate=u,side=side,converged=bool(m.converged),
                              iterations=int(m.iterations),coefficient_delta=float(m.coefficient_delta),
                              n=len(obj.records),patients=len(obj.patient_ids)))
            if not m.converged: raise RuntimeError('OUTER_FQI_NOT_CONVERGED')
    support={s.candidate:s for s in sc.candidates}
    bounds={u:boundary_factor(support[u],0,60480,len(panel.patient_ids),a['boundary_scaling']) for u in CANDIDATES}
    with np.load(PARENT/'data_processed/evaluation_grid.npz',allow_pickle=False) as z:
        design_matrix=action_feature(z['states'],z['actions'])
    prep=PreparedAnalysis(SimpleNamespace(analysis_start=0,analysis_end=60480),panel,CANDIDATES,
                          support,fits,design_matrix,30240,panel.patient_ids,bounds)
    raw,norm,joint=observed_arrays(prep)
    if any(x.shape!=(7,64) or not np.isfinite(x).all() for x in (raw,norm,joint)):
        raise FloatingPointError('OUTER_NONFINITE_OR_WRONG_SHAPE')
    values=layers(raw,norm,joint,CANDIDATES.index(30240))
    atomic_json(folder/'fqi.json',diags)
    atomic_npz(folder/'process.npz',raw=raw,normalized=norm,joint=joint,D0_D4=values,
               candidates=np.asarray(CANDIDATES))
    atomic_pickle(prep,folder/'prepared.pkl')
    return prep,values


def inner_init(fit_path,schedule_path):
    global INNER_FIT,INNER_TEMPLATE
    INNER_FIT=read(fit_path); INNER_TEMPLATE=load_schedule(schedule_path)


def inner_task(seed,draw_root):
    j=seed['draw_id']; folder=Path(draw_root)/f'draw_{j:03d}'
    folder.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter(); category='INNER_GENERATION'
    record=dict(seed=seed,component_seeds=component_seeds(seed),valid=False,
                failure_category=None,fqi_status='NOT_RUN')
    try:
        d,_=simulate(INNER_FIT,INNER_TEMPLATE,seed)
        atomic_csv(d,folder/'scheduled.csv.gz')
        category='INNER_PANEL'; panel,edges,_=rebuild(d)
        atomic_csv(edges,folder/'transition_audit.csv.gz')
        category='INNER_SUPPORT'; sc,rows=screen(panel)
        atomic_csv(pd.DataFrame(rows),folder/'support.csv')
        record.update(n_transitions=len(panel.records),n_patients=len(panel.patient_ids),
                      support=rows,action1=int(sum(r.action for r in panel.records)))
        if tuple(sc.admissible_candidates)!=CANDIDATES:
            raise RuntimeError('FROZEN_CANDIDATE_NOT_ADMISSIBLE')
        category='INNER_FQI'; record['fqi_status']='STARTED'
        values=calculate_pilot(panel,sc,folder)
        if not np.isfinite(values).all():
            category='INNER_NONFINITE'; raise FloatingPointError('NONFINITE_LAYER')
        record.update(valid=True,fqi_status='PASS',D0_D4=values)
    except Exception as exc:
        if 'NONFINITE' in str(exc): category='INNER_NONFINITE'
        record.update(failure_category=category,error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
    record['runtime_seconds']=time.perf_counter()-started
    seal(folder,record)
    return j,record['valid']


def inner_run(outer_id,folder,schedule_path,workers):
    reg=[inner_seed(outer_id,b) for b in range(B)]
    draw_root=folder/'draws';draw_root.mkdir(exist_ok=True)
    expected={f'draw_{b:03d}' for b in range(B)}
    if {p.name for p in draw_root.iterdir()}-expected: raise RuntimeError('EXTRA_INNER_DRAW')
    pending=[]
    for seed in reg:
        path=draw_root/f"draw_{seed['draw_id']:03d}"
        if path.exists():
            cp=verify_seal(path,seed)
            if cp['component_seeds']!=component_seeds(seed): raise RuntimeError('INNER_COMPONENT_SEED_MISMATCH')
        else: pending.append(seed)
    if pending:
        with ProcessPoolExecutor(max_workers=workers,initializer=inner_init,
                                 initargs=(str(folder/'fitted_null.json'),str(schedule_path))) as pool:
            futures={pool.submit(inner_task,s,str(draw_root)):s for s in pending}
            for f in as_completed(futures): f.result()
    records=[verify_seal(draw_root/f"draw_{s['draw_id']:03d}",s) for s in reg]
    if len(records)!=B: raise RuntimeError('INNER_COUNT_MISMATCH')
    valid=[r for r in records if r['valid']]
    vals=np.asarray([r['D0_D4'] for r in valid],float) if valid else np.empty((0,5))
    if vals.size and (vals.shape!=(len(valid),5) or not np.isfinite(vals).all()):
        raise RuntimeError('INVALID_INNER_LAYER_MATRIX')
    return records,vals


def outer_run(seed,truth_fit,template,workers):
    i=seed['outer_id'];folder=OUT/'outer'/f'r{i:03d}';folder.mkdir(parents=True,exist_ok=True)
    final=folder/'checkpoint.json'
    if final.exists(): return verify_seal(folder,seed)
    start=time.perf_counter()
    truth=folder/'truth';observed=folder/'observed';orig=folder/'original';m4=folder/'m4'
    for sub in (truth,observed,orig,m4): sub.mkdir(exist_ok=True)
    truth_seed={k:seed[k] for k in ('draw_seed','design_seed','reward_seed')}
    if (truth/'checkpoint.json').exists():
        verify_seal(truth,truth_seed);d=load_schedule(truth/'pseudo_observed_panel.csv.gz')
    elif any(truth.iterdir()): raise RuntimeError('INCOMPLETE_TRUTH:'+str(truth))
    else:
        try:
            d,comps=simulate(truth_fit,template,truth_seed)
            atomic_csv(d,truth/'pseudo_observed_panel.csv.gz')
            seal(truth,dict(seed=truth_seed,components=comps,status='VALID',scheduled_rows=len(d)))
        except Exception as exc:
            if any(truth.iterdir()): raise
            seal(truth,dict(seed=truth_seed,status='INVALID',category='OUTER_GENERATION',error=str(exc),traceback=traceback.format_exc()))
            seal(folder,dict(seed=seed,status='OUTER_ANALYSIS_UNAVAILABLE',reason='OUTER_GENERATION',runtime_seconds=time.perf_counter()-start))
            return verify_seal(folder,seed)
    if (observed/'checkpoint.json').exists():
        ob=verify_seal(observed)
        if ob['status']!='VALID':
            seal(folder,dict(seed=seed,status='OUTER_ANALYSIS_UNAVAILABLE',reason=ob['reason'],runtime_seconds=time.perf_counter()-start))
            return verify_seal(folder,seed)
        prep=pickle.loads((observed/'prepared.pkl').read_bytes());obs=np.asarray(ob['D0_D4'],float)
    elif any(observed.iterdir()): raise RuntimeError('INCOMPLETE_OBSERVED:'+str(observed))
    else:
        try:
            panel,edges,_=rebuild(d);atomic_csv(edges,observed/'transition_audit.csv.gz')
            sc,rows=screen(panel);atomic_csv(pd.DataFrame(rows),observed/'support.csv')
            if tuple(sc.admissible_candidates)!=CANDIDATES:
                seal(observed,dict(status='UNAVAILABLE',reason='OUTER_SUPPORT',
                                   n_transitions=len(panel.records),admissible=list(sc.admissible_candidates)))
                seal(folder,dict(seed=seed,status='OUTER_ANALYSIS_UNAVAILABLE',reason='OUTER_SUPPORT',runtime_seconds=time.perf_counter()-start))
                return verify_seal(folder,seed)
            prep,obs=prepare_outer(panel,sc,observed)
            seal(observed,dict(status='VALID',D0_D4=obs.tolist(),n_transitions=len(panel.records),
                               n_patients=len(panel.patient_ids),admissible=list(sc.admissible_candidates)))
        except Exception as exc:
            # If a stage left any unsealed files, stop for explicit technical review.
            if (observed/'checkpoint.json').exists(): raise
            atomic_json(observed/'failure.json',dict(category='OUTER_PANEL_OR_FQI',error=str(exc),traceback=traceback.format_exc()))
            seal(observed,dict(status='UNAVAILABLE',reason='OUTER_PANEL_OR_FQI'))
            seal(folder,dict(seed=seed,status='OUTER_ANALYSIS_UNAVAILABLE',reason='OUTER_PANEL_OR_FQI',runtime_seconds=time.perf_counter()-start))
            return verify_seal(folder,seed)
    if (orig/'checkpoint.json').exists(): original=verify_seal(orig)
    elif any(orig.iterdir()): raise RuntimeError('INCOMPLETE_ORIGINAL:'+str(orig))
    else:
        try:
            t=time.perf_counter();ref=run_influence_arm(prep,'M0',B,seed['original_seed'],'gaussian')
            for k,key in enumerate(LAYERS):
                if not np.isclose(ref.observed[key],obs[k],rtol=1e-10,atol=1e-10):
                    raise RuntimeError('ORIGINAL_OBSERVED_STATISTIC_MISMATCH:'+key)
            draw=np.column_stack([np.asarray(ref.bootstrap[k],float) for k in LAYERS])
            if draw.shape!=(B,5) or not np.isfinite(draw).all(): raise RuntimeError('INVALID_ORIGINAL_DRAWS')
            crit=np.quantile(draw,.95,axis=0)
            if not np.isfinite(crit).all() or not (crit>0).all(): raise RuntimeError('INVALID_ORIGINAL_CRITICAL')
            p=[finite_p(obs[j],draw[:,j]) for j in range(5)]
            atomic_csv(pd.DataFrame(draw,columns=LAYERS),orig/'calibration_distribution.csv')
            seal(orig,dict(status='VALID',seed=seed['original_seed'],observed=obs.tolist(),
                           p=p,critical=crit.tolist(),reject=[x<=ALPHA for x in p],
                           runtime_seconds=time.perf_counter()-t,diagnostics=ref.diagnostics))
            original=verify_seal(orig)
        except Exception as exc:
            atomic_json(orig/'failure.json',dict(category='ORIGINAL_CALIBRATION',error=str(exc),traceback=traceback.format_exc()))
            seal(orig,dict(status='UNAVAILABLE',seed=seed['original_seed'],reason='ORIGINAL_CALIBRATION'))
            original=verify_seal(orig)
    if (m4/'checkpoint.json').exists(): m4result=verify_seal(m4)
    else:
        if not (m4/'fitted_null.json').exists():
            if any(m4.iterdir()): raise RuntimeError('INCOMPLETE_INNER_FIT:'+str(m4))
            try:
                t=time.perf_counter();fit,diagnostics=fit_inner_null(d)
                atomic_json(m4/'fitted_null.json',fit)
                atomic_json(m4/'fit_diagnostics.json',dict(models=diagnostics,runtime_seconds=time.perf_counter()-t,
                                                           source='outer pseudo-observed scheduled fields only',
                                                           original_truth_coefficients_used=False))
            except Exception as exc:
                atomic_json(m4/'failure.json',dict(category='INNER_NULL_FIT',error=str(exc),traceback=traceback.format_exc()))
                seal(m4,dict(status='UNAVAILABLE',reason='INNER_NULL_FIT',attempted_draws=0,valid_draws=0))
                m4result=verify_seal(m4)
        if not (m4/'checkpoint.json').exists():
            t=time.perf_counter(); records,vals=inner_run(i,m4,truth/'pseudo_observed_panel.csv.gz',workers)
            if len(vals): atomic_csv(pd.DataFrame(vals,columns=LAYERS),m4/'calibration_distribution.csv')
            crit=np.quantile(vals,.95,axis=0) if len(vals) else np.full(5,np.nan)
            available=bool(len(vals) and np.isfinite(crit).all() and (crit>0).all())
            p=[finite_p(obs[j],vals[:,j]) for j in range(5)] if available else None
            seal(m4,dict(status='VALID' if available else 'UNAVAILABLE',reason=None if available else 'NO_VALID_CRITICAL',
                         attempted_draws=B,valid_draws=len(vals),invalid_draws=B-len(vals),
                         p=p,critical=crit.tolist() if available else None,
                         reject=[x<=ALPHA for x in p] if available else None,
                         runtime_seconds=time.perf_counter()-t,
                         failure_categories={k:sum(r.get('failure_category')==k for r in records)
                                             for k in sorted({r.get('failure_category') for r in records if not r['valid']})}))
            m4result=verify_seal(m4)
    outcome='PAIRED_VALID' if original['status']=='VALID' and m4result['status']=='VALID' else 'METHOD_UNAVAILABLE'
    seal(folder,dict(seed=seed,status=outcome,observed_D0_D4=obs.tolist(),
                     observed_checkpoint_sha=digest(observed/'checkpoint.json'),
                     original_checkpoint_sha=digest(orig/'checkpoint.json'),
                     m4_checkpoint_sha=digest(m4/'checkpoint.json'),
                     runtime_seconds=time.perf_counter()-start))
    return verify_seal(folder,seed)


def run(workers):
    verify_freeze()
    if workers<1 or workers>8: raise ValueError('workers must be 1..8 to prevent oversubscription')
    if (OUT/'results/COMPUTATION_COMPLETE.json').exists(): raise RuntimeError('COMPUTATION_ALREADY_COMPLETE')
    lock=OUT/'results/runner.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('x',encoding='utf-8') as f: f.write(str(os.getpid()))
    try:
        seeds,_=registered();truth_fit=read(BASE/'diagnostics/fitted_models.json');template=schedule_data()
        started=time.perf_counter();completed=0; unavailable=0
        for seed in seeds:
            record=outer_run(seed,truth_fit,template,workers)
            completed+=1;unavailable+=record['status']!='PAIRED_VALID'
            if completed%25==0:
                elapsed=time.perf_counter()-started;estimate=elapsed/completed*(R-completed)
                atomic_json(OUT/'results/run_status.json',dict(completed=completed,planned=R,
                            unavailable_or_failed=unavailable,elapsed_seconds=elapsed,
                            estimated_remaining_seconds=estimate,pid=os.getpid()))
                print(f'ENGINEERING {completed}/{R}; unavailable_or_failed={unavailable}; elapsed={elapsed:.1f}s; remaining_estimate={estimate:.1f}s',flush=True)
        atomic_json(OUT/'results/COMPUTATION_COMPLETE.json',dict(completed=R,planned=R,
                    no_interim_inference=True,finished_utc=pd.Timestamp.now(tz='UTC').isoformat()))
    finally:
        lock.unlink()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=('freeze','run','finalize'))
    ap.add_argument('--workers',type=int,default=8);args=ap.parse_args()
    if args.stage=='freeze': freeze()
    elif args.stage=='run': run(args.workers)
    else:
        from .finalize import finalize
        finalize()


if __name__=='__main__': main()
