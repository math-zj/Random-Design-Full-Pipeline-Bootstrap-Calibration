"""HeartSteps-only row/clock adapter. Mathematical estimators are imported."""
from __future__ import annotations
import json
import math
import sys
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.rs_cusum.types import RiskSetPanel, TransitionRecord
from src.rs_cusum.support import load_support_rule, screen_candidates
from src.rs_cusum.patient_balance import action_feature
from src.rs_cusum_phase2.grids import candidate_grid
from heartsteps_v1.scripts.audit_data import load_suggestions, write_json, OUT


def config():
    return json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))


def state(row):
    slot = float(row['sugg.select.slot'])
    return np.asarray([np.log(float(row['jbsteps30pre'])+.5),
                       np.sin(2*np.pi*(slot-1)/5), np.cos(2*np.pi*(slot-1)/5),
                       float(str(row['dec.location.category']).strip().lower() in {'home','work'})])


def row_issues(row, require_state=False):
    reasons=[]
    if not bool(row['connect']): reasons.append('DISCONNECTED')
    if pd.isna(row['decision.index.nogap']): reasons.append('TRAVEL_DAY')
    if any(pd.isna(row[c]) for c in ['sugg.select.utime','sugg.decision.utime','sugg.context.utime']): reasons.append('MISSING_TIMESTAMP')
    if pd.isna(row['send']): reasons.append('UNKNOWN_SEND')
    if row['avail'] == False and row['send'] == True: reasons.append('UNAVAILABLE_SENT')
    if row['avail'] == True and (pd.isna(row['is.randomized']) or row['send'] != row['is.randomized']): reasons.append('SEND_RANDOMIZATION_MISMATCH')
    if pd.notna(row['sugg.decision.slot']) and row['sugg.decision.slot'] != row['sugg.select.slot']: reasons.append('SLOT_TIMEZONE_MISMATCH')
    if pd.notna(row['context_age_min']) and row['context_age_min'] < 0: reasons.append('FUTURE_CONTEXT')
    if require_state:
        if pd.isna(row['jbsteps30pre']) or row['jbsteps30pre'] < 0: reasons.append('MISSING_OR_NEGATIVE_PRE_STEPS')
        if pd.isna(row['dec.location.category']): reasons.append('MISSING_LOCATION')
        if row['sugg.select.slot'] not in [1,2,3,4,5]: reasons.append('INVALID_SLOT')
    return reasons


def build(save=True):
    c=config(); d=load_suggestions(); end=int(c['analysis']['horizon_days']*1440)
    records=[]; edges=[]; scheme=[]
    for pid,g0 in d.groupby('user.index'):
        g=g0.reset_index(drop=True); origin=g['sugg.select.utime'].min()
        available=np.flatnonzero(g.avail.eq(True))
        segment=-1; previous_next=None
        for i,j in zip(available[:-1],available[1:]):
            x,y=g.iloc[i],g.iloc[j]; between=g.iloc[i+1:j]
            reasons=['current:'+r for r in row_issues(x,True)]+['next:'+r for r in row_issues(y,True)]
            if pd.isna(x.jbsteps30) or x.jbsteps30<0: reasons.append('MISSING_OR_NEGATIVE_REWARD')
            for _,b in between.iterrows():
                reasons.extend('intervening:'+r for r in row_issues(b))
                if b['send'] == True: reasons.append('INTERVENING_ACTION')
            ti=x['sugg.decision.utime']; tn=y['sugg.decision.utime']
            gap=(tn-ti).total_seconds()/60 if pd.notna(ti) and pd.notna(tn) else np.nan
            elapsed=(ti-origin).total_seconds()/60 if pd.notna(ti) else np.nan
            next_elapsed=(tn-origin).total_seconds()/60 if pd.notna(tn) else np.nan
            if not np.isfinite(gap) or not (c['mdp']['edge_gap_min_minutes']<=gap<=c['mdp']['edge_gap_max_minutes']): reasons.append('EDGE_DURATION_OUTSIDE_30MIN_24H')
            if not np.isfinite(elapsed) or not np.isfinite(next_elapsed) or elapsed<0 or next_elapsed>end: reasons.append('OUTSIDE_FIXED_42DAY_WINDOW')
            span=g.iloc[i:j+1]['sugg.decision.utime']
            if span.isna().any() or (span.diff().dropna().dt.total_seconds()<=0).any(): reasons.append('NONMONOTONE_OR_UNKNOWN_INTERIOR_TIME')
            reasons=sorted(set(reasons)); valid=not reasons
            row={'participant':int(pid),'from_source_row':int(x.source_row),'to_source_row':int(y.source_row),'decision_index':int(x['decision.index']),'next_decision_index':int(y['decision.index']),'skipped_unavailable':int(j-i-1),'tau_t':str(ti),'tau_next':str(tn),'gap_minutes':gap,'elapsed_minutes':elapsed,'next_elapsed_minutes':next_elapsed,'study_day_elapsed':elapsed/1440,'scheduled_calendar_study_day':x.study_day_from_first_record_date,'slot':int(x['sugg.select.slot']),'context_timestamp':str(x['sugg.context.utime']),'context_age_min':x.context_age_min,'pre_window_begin':str(ti-pd.Timedelta(minutes=30)),'pre_window_end':str(ti),'pre_steps':x.jbsteps30pre,'location':x['dec.location.category'],'action':int(bool(x['send'])),'reward_window_begin':str(ti),'reward_window_end':str(ti+pd.Timedelta(minutes=30)),'post_steps':x.jbsteps30,'reward':float(np.log(x.jbsteps30+.5)) if pd.notna(x.jbsteps30) and x.jbsteps30>=0 else np.nan,'next_pre_steps':y.jbsteps30pre,'valid':valid,'reasons':'|'.join(reasons),'segment_id':None}
            if valid:
                if previous_next != int(x['decision.index']): segment+=1
                row['segment_id']=segment; previous_next=int(y['decision.index'])
                st=state(x); nx=state(y)
                for k,name in enumerate(c['mdp']['state_names']): row['state_'+name]=st[k]; row['next_state_'+name]=nx[k]
                records.append(TransitionRecord(str(pid),'HeartStepsV1',c['analysis']['clock'],int(math.floor(elapsed)),ti.to_pydatetime(),tn.to_pydatetime(),st,nx,int(x['send']),row['reward'],row['reward'],segment,'available_to_immediate_next_available',elapsed%1,next_elapsed%1))
            else:
                previous_next=None
            edges.append(row)
        # Data-level comparison only. Never run a second observed method.
        for i in available:
            if i+1>=len(g): continue
            x,y=g.iloc[i],g.iloc[i+1]
            scheme.append({'participant':int(pid),'current_index':int(x['decision.index']),'next_scheduled_available':bool(y.avail),'next_scheduled_state_usable':not row_issues(y,True),'requires_max_over_unavailable_actions':not bool(y.avail),'gap_minutes':(y['sugg.decision.utime']-x['sugg.decision.utime']).total_seconds()/60})
    panel=RiskSetPanel(tuple(records),tuple(c['mdp']['state_names']),c['analysis']['clock'],1,(30.,1440.)); panel.validate()
    edges=pd.DataFrame(edges)
    if save:
        edges.to_csv(OUT/'diagnostics/transition_audit_all.csv',index=False)
        pd.DataFrame(scheme).to_csv(OUT/'diagnostics/next_scheduled_comparison.csv',index=False)
        rejected=edges[~edges.valid]; counter=Counter(r for reason in rejected.reasons for r in reason.split('|'))
        pd.DataFrame([{'reason':k,'edges':v} for k,v in sorted(counter.items())]).to_csv(OUT/'diagnostics/transition_rejections.csv',index=False)
        selected=edges[edges.valid].copy(); selected.to_csv(OUT/'data_processed/transitions.csv',index=False)
        selected.groupby('participant').agg(transitions=('action','size'),A1=('action','sum'),A1_fraction=('action','mean'),segments=('segment_id','nunique'),start_elapsed_minutes=('elapsed_minutes','min'),end_elapsed_minutes=('next_elapsed_minutes','max'),gap_minutes_median=('gap_minutes','median'),gap_minutes_max=('gap_minutes','max')).to_csv(OUT/'diagnostics/analysis_participants.csv')
        segments=[]
        arrdir=OUT/'data_processed/segments'; arrdir.mkdir(exist_ok=True)
        for pid in panel.patient_ids:
            pr=[r for r in records if r.patient_id==pid]
            for sid in sorted({r.segment_id for r in pr}):
                rr=sorted([r for r in pr if r.segment_id==sid],key=lambda r:r.elapsed_index)
                assert all(np.array_equal(a.next_state,b.state) and a.tau_next==b.tau_t for a,b in zip(rr[:-1],rr[1:]))
                states=np.vstack([r.state for r in rr]+[rr[-1].next_state]); actions=np.array([r.action for r in rr]); rewards=np.array([r.reward_binary for r in rr])
                np.savez_compressed(arrdir/f'user_{pid}_segment_{sid:03d}.npz',States=states,Actions=actions,Rewards=rewards,patient_id=pid,segment_id=sid,tau_utc=np.array([r.tau_t.isoformat() for r in rr]+[rr[-1].tau_next.isoformat()]),elapsed_index=np.array([r.elapsed_index for r in rr]))
                segments.append({'participant':pid,'segment_id':sid,'T':len(rr),'A1':int(actions.sum()),'first_elapsed_index':rr[0].elapsed_index,'last_elapsed_index':rr[-1].elapsed_index,'start_utc':rr[0].tau_t.isoformat(),'end_utc':rr[-1].tau_next.isoformat()})
        pd.DataFrame(segments).to_csv(OUT/'diagnostics/segments.csv',index=False)
        write_json(OUT/'diagnostics/adapter_summary.json',{'participants':len(panel.patient_ids),'transitions':len(records),'A1':sum(r.action for r in records),'A1_fraction':sum(r.action for r in records)/len(records),'segments':len(segments),'considered_available_pairs':len(edges),'rejected_pairs':len(rejected),'next_scheduled_unavailable':sum(r['requires_max_over_unavailable_actions'] for r in scheme),'original_patient_is_cluster':True,'next_available_skipped_count_distribution':{str(k):int(v) for k,v in selected.skipped_unavailable.value_counts().items()},'cross_gap_transitions':0})
    return panel,edges,c


def support(panel,c,save=True):
    a=c['analysis']; end=a['horizon_days']*1440
    candidates=candidate_grid(0,end,a['candidate_fractions'])
    rule=load_support_rule(ROOT/a['support_config'],a['support_rule'])
    screen=screen_candidates(panel.support_view(),candidates,0,end,rule)
    rows=[]; designs=[]
    for v in screen.candidates:
        row={'candidate_elapsed_min':v.candidate,'candidate_elapsed_day':v.candidate/1440,'admissible':v.admissible,'rejection_reasons':'|'.join(v.failure_reasons),'required_each_action_each_side':v.required_count_each_side_action,'patients_both_sides':len(v.patients_both_sides)}
        for side in [v.left,v.right]:
            name=side.side
            row.update({name+'_transitions':side.transition_count,name+'_A0':side.action_counts[0],name+'_A1':side.action_counts[1],name+'_patients':len(side.active_patients),name+'_A0_patients':len(side.unique_action_patients[0]),name+'_A1_patients':len(side.unique_action_patients[1]),name+'_max_A1_patient_share':side.maximum_action_patient_share[1],name+'_max_total_patient_share':side.maximum_total_patient_share})
            rr=[r for r in panel.records if (0<=r.elapsed_index<v.candidate if name=='left' else v.candidate<=r.elapsed_index<end)]
            if rr:
                ids=[r.patient_id for r in rr]; counts=Counter(ids); w=np.array([1/(len(counts)*counts[k]) for k in ids]); s=np.vstack([r.state for r in rr]); ac=np.array([r.action for r in rr]); x=action_feature(s,ac); gram=x.T@(w[:,None]*x)
                dd={'candidate_elapsed_min':v.candidate,'side':name,'columns':x.shape[1],'full_rank':int(np.linalg.matrix_rank(x)),'raw_gram_condition':float(np.linalg.cond(gram)),'ridge_condition':float(np.linalg.cond(gram+a['ridge_alpha']*np.eye(x.shape[1])))}
                for k in [0,1]:
                    block=np.column_stack([np.ones((ac==k).sum()),s[ac==k]])
                    dd[f'action{k}_block_rank']=int(np.linalg.matrix_rank(block)) if len(block) else 0
                designs.append(dd)
        rows.append(row)
    if save:
        pd.DataFrame(rows).to_csv(OUT/'diagnostics/candidate_support.csv',index=False)
        pd.DataFrame(designs).to_csv(OUT/'diagnostics/candidate_design_diagnostics.csv',index=False)
        write_json(OUT/'diagnostics/support_summary.json',{'total_candidates':len(candidates),'supported_candidates':len(screen.admissible_candidates),'rejected_candidates':len(candidates)-len(screen.admissible_candidates),'candidate_indices':list(candidates),'admissible_indices':list(screen.admissible_candidates),'rank_used_to_change_support':False,'primary_thresholds_unmodified':True})
    return screen


if __name__=='__main__':
    panel,edges,c=build(); sc=support(panel,c)
    print((OUT/'diagnostics/adapter_summary.json').read_text())
    print((OUT/'diagnostics/support_summary.json').read_text())
