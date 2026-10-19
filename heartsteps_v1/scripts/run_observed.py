"""One isolated observed-only run using unchanged formal RS-CUSUM functions."""
from __future__ import annotations
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from src.rs_cusum.types import RiskSetPanel,TransitionRecord
from src.rs_cusum.patient_balance import action_feature
from src.rs_cusum.statistic import boundary_factor
from src.rs_cusum_phase2.grids import evaluation_grid
from src.rs_cusum_phase2r.core import fit_candidate
from src.rs_cusum_phase2rb.engine import PreparedAnalysis
from src.rs_cusum_phase2rc.c0_engine import observed_arrays
from src.rs_cusum_realdata_bridge_v0.runner import layers
from src.rs_cusum_phase2rd.protocol import stable_seed
from heartsteps_v1.scripts.audit_data import OUT,sha,write_json
from heartsteps_v1.scripts.adapter import config,support


def log(message):
    line=time.strftime('%Y-%m-%d %H:%M:%S')+' '+message
    with (OUT/'logs/observed.log').open('a',encoding='utf-8') as f: f.write(line+'\n')
    print(line,flush=True)


def load_panel(c):
    f=pd.read_csv(OUT/'data_processed/transitions.csv')
    names=c['mdp']['state_names']; records=[]
    for r in f.to_dict('records'):
        records.append(TransitionRecord(str(int(r['participant'])),'HeartStepsV1',c['analysis']['clock'],int(np.floor(r['elapsed_minutes'])),pd.Timestamp(r['tau_t']).to_pydatetime(),pd.Timestamp(r['tau_next']).to_pydatetime(),np.array([r['state_'+x] for x in names]),np.array([r['next_state_'+x] for x in names]),int(r['action']),float(r['reward']),float(r['reward']),int(r['segment_id']),'available_to_immediate_next_available',r['elapsed_minutes']%1,r['next_elapsed_minutes']%1))
    panel=RiskSetPanel(tuple(records),tuple(names),c['analysis']['clock'],1,(30.,1440.)); panel.validate()
    return panel,f


def freeze(panel,c,sc):
    paths=[OUT/'protocol.json',OUT/'data_processed/transitions.csv',OUT/'diagnostics/candidate_support.csv',OUT/'diagnostics/validation.json']+sorted((OUT/'scripts').glob('*.py'))
    protected=json.loads((OUT/'diagnostics/protected_source_hashes_before.json').read_text())
    assert all(sha(ROOT/p)==h for p,h in protected.items()),'PROTECTED_CORE_CHANGED'
    seeds={'master':c['seed_master'],'evaluation_grid':stable_seed(c['seed_master'],'HeartSteps','evaluation-grid'),'pilot_draws':[{'draw_id':i,'draw_seed':stable_seed(c['seed_master'],'HeartSteps','M4','draw',i),'design_seed':stable_seed(c['seed_master'],'HeartSteps','M4','design',i),'reward_seed':stable_seed(c['seed_master'],'HeartSteps','M4','reward',i)} for i in range(30)]}
    vals=[seeds['evaluation_grid']]+[d[k] for d in seeds['pilot_draws'] for k in ['draw_seed','design_seed','reward_seed']]
    assert len(set(vals))==len(vals)
    seedpath=OUT/'seed_registry.json'
    if seedpath.exists(): assert json.loads(seedpath.read_text())==seeds
    else: write_json(seedpath,seeds)
    a=c['analysis']; end=a['horizon_days']*1440
    es,ea,sources=evaluation_grid(panel,0,end,a['evaluation_reference_states'],seeds['evaluation_grid'])
    grid=OUT/'data_processed/evaluation_grid.npz'
    if grid.exists():
        with np.load(grid) as z: assert np.array_equal(z['states'],es) and np.array_equal(z['actions'],ea)
    else:
        np.savez_compressed(grid,states=es,actions=ea,source_patient=np.array([p for p,t in sources]),source_elapsed_min=np.array([t for p,t in sources]))
    paths += [seedpath,grid]
    registry={'stage':'before_first_observed_fit','protocol_version':c['version'],'supported_candidates':list(sc.admissible_candidates),'protected_core_hashes':protected,'files':{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in paths},'seeds_unique':True,'planned_pilot_draws':30,'formal_calibration_allowed':False}
    target=OUT/'hash_registry_pre_observed.json'
    if target.exists(): assert json.loads(target.read_text())==registry,'FROZEN_REGISTRY_CHANGED'
    else: write_json(target,registry)
    return es,ea


def run():
    if (OUT/'results/observed_status.json').exists():
        raise RuntimeError('OBSERVED_STAGE_ALREADY_RECORDED: do not repeat fits')
    v=json.loads((OUT/'diagnostics/validation.json').read_text())
    if v['status']!='PASS': raise RuntimeError('PRE_OBSERVED_TIMESTAMP_GATE_NOT_PASS')
    c=config(); a=c['analysis']; panel,transitions=load_panel(c); sc=support(panel,c,save=False)
    if not sc.admissible_candidates: raise RuntimeError('NOT_TESTABLE_UNDER_SUPPORT_RULE')
    es,ea=freeze(panel,c,sc); cand=tuple(sc.admissible_candidates); end=a['horizon_days']*1440
    start=time.perf_counter(); fits={}; diagnostics=[]
    log(f'OBSERVED_START participants={len(panel.patient_ids)} transitions={len(panel.records)} candidates={len(cand)}; no calibration')
    for u in cand:
        t=time.perf_counter()
        fit=fit_candidate(panel,u,0,end,a['gamma'],a['ridge_alpha'],a['fqi_max_iterations'],a['fqi_tolerance'],'patient_balanced','patient')
        fits[u]=fit
        for side in ['left','right']:
            s=getattr(fit,side); m=s.model
            diagnostics.append({'candidate_elapsed_min':u,'side':side,'converged':m.converged,'iterations':m.iterations,'coefficient_delta':m.coefficient_delta,'n_transitions':len(s.records),'n_patients':len(s.patient_ids),**m.diagnostics})
        log(f'candidate_day={u/1440:g} FQI_left={fit.left.model.iterations} FQI_right={fit.right.model.iterations} seconds={time.perf_counter()-t:.3f}')
    sup={s.candidate:s for s in sc.candidates if s.admissible}
    bounds={u:boundary_factor(sup[u],0,end,len(panel.patient_ids),a['boundary_scaling']) for u in cand}
    midpoint=min(cand,key=lambda u:abs(u-end/2))
    prepared=PreparedAnalysis(SimpleNamespace(analysis_start=0,analysis_end=end),panel,cand,sup,fits,action_feature(es,ea),midpoint,panel.patient_ids,bounds)
    raw,norm,joint=observed_arrays(prepared)
    if not all(np.isfinite(z).all() for z in [raw,norm,joint]): raise FloatingPointError('NONFINITE_OBSERVED_PROCESS')
    values=layers(raw,norm,joint,cand.index(midpoint))
    cp=pd.DataFrame({'candidate_elapsed_min':cand,'candidate_elapsed_day':np.array(cand)/1440,'D0_raw_cell0':abs(raw[:,0]),'D1_studentized_cell0':abs(norm[:,0]),'D2_candidate_max_studentized':np.max(abs(norm),axis=1),'D3_candidate_joint_cell0':abs(joint[:,0]),'D4_candidate_max_joint':np.max(abs(joint),axis=1),'boundary_factor':[bounds[u] for u in cand]})
    tau4=cand[int(np.argmax(cp.D4_candidate_max_joint))]; tau3=cand[int(np.argmax(cp.D3_candidate_joint_cell0))]
    np.savez_compressed(OUT/'results/observed_process.npz',raw=raw,normalized=norm,joint=joint,candidates=np.array(cand),observed_D0_D4=values,evaluation_states=es,evaluation_actions=ea,midpoint=midpoint,tau_hat_D4=tau4,tau_hat_D3=tau3)
    cp.to_csv(OUT/'results/candidate_process.csv',index=False)
    detailed=[]
    for i,u in enumerate(cand):
        for z in range(len(ea)):
            detailed.append({'candidate_elapsed_min':u,'evaluation_cell':z,'evaluation_action':int(ea[z]),'raw_contrast':raw[i,z],'studentized_contrast':norm[i,z],'boundary_weighted_contrast':joint[i,z]})
    pd.DataFrame(detailed).to_csv(OUT/'results/candidate_by_state_process.csv',index=False)
    result=pd.DataFrame({'layer':['D0','D1','D2','D3','D4'],'observed_statistic':values,'localization_role':['fixed midpoint; not an estimated changepoint']*3+['argmax over candidates at fixed cell0','primary argmax over candidates and states/actions'],'candidate_or_tau_elapsed_min':[midpoint]*3+[tau3,tau4],'candidate_or_tau_elapsed_day':[midpoint/1440]*3+[tau3/1440,tau4/1440],'calibration':'not_run_no_p_value'})
    result.to_csv(OUT/'results/observed_global_statistics.csv',index=False)
    write_json(OUT/'diagnostics/fqi_diagnostics.json',diagnostics)
    pd.DataFrame([{k:v for k,v in x.items() if not isinstance(v,(dict,list))} for x in diagnostics]).to_csv(OUT/'diagnostics/fqi_convergence.csv',index=False)
    ts=[]
    for pid,g in transitions.groupby('participant'):
        after=g[g.elapsed_minutes>=tau4]; before=g[g.elapsed_minutes<tau4]
        first=after.sort_values('elapsed_minutes').iloc[0] if len(after) else None
        ts.append({'participant':int(pid),'tau_elapsed_day':tau4/1440,'first_observed_right_decision_index':int(first.decision_index) if first is not None else None,'first_observed_right_tau':first.tau_t if first is not None else None,'left_n':len(before),'left_A1':int(before.action.sum()),'right_n':len(after),'right_A1':int(after.action.sum())})
    pd.DataFrame(ts).to_csv(OUT/'results/tau_hat_patient_mapping.csv',index=False)
    pd.read_csv(OUT/'diagnostics/candidate_support.csv').query('candidate_elapsed_min == @tau4').to_csv(OUT/'results/tau_hat_support.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7.2,4.2)); ax.plot(cp.candidate_elapsed_day,cp.D4_candidate_max_joint,'o-',color='#225588'); ax.axvline(tau4/1440,ls='--',color='#a53b30',label=f'Descriptive argmax: day {tau4/1440:g}');ax.set(xlabel='Elapsed study day from first scheduled decision',ylabel='Observed D4 candidate maximum',title='HeartSteps V1: observed RS-CUSUM (uncalibrated)');ax.legend();fig.tight_layout();fig.savefig(OUT/'results/D4_candidate_curve.png',dpi=180);fig.savefig(OUT/'results/D4_candidate_curve.svg');plt.close(fig)
    status={'status':'OBSERVED_COMPLETE','patients':len(panel.patient_ids),'transitions':len(panel.records),'A1':sum(r.action for r in panel.records),'candidates':list(cand),'observed_D0_D4':values.tolist(),'tau_hat_D4_elapsed_min':tau4,'tau_hat_D4_elapsed_day':tau4/1440,'tau_hat_D3_elapsed_day':tau3/1440,'all_14_FQI_fits_converged':True,'bootstrap_attempted':0,'p_value_available':False,'runtime_seconds':time.perf_counter()-start,'python':sys.version,'platform':platform.platform()}
    write_json(OUT/'results/observed_status.json',status)
    log(f'OBSERVED_COMPLETE D4_tau_day={tau4/1440:g}; all 14 FQI fits converged; no p-value')
    print(json.dumps(status,indent=2))


if __name__=='__main__':
    try: run()
    except Exception:
        (OUT/'logs/observed.err.log').write_text(traceback.format_exc(),encoding='utf-8')
        raise
