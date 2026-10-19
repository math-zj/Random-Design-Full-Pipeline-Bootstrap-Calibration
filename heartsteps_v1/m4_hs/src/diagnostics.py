"""Predictive QC only. No inference or CUSUM calculation."""
from __future__ import annotations
import numpy as np
import pandas as pd
from heartsteps_v1.scripts.adapter import row_issues
from .common import CFG,screen

QUANTILES=(.25,.5,.75,.9,.95,.99)

def episodes(mask):
    vals=[];n=0
    for x in mask:
        if x:n+=1
        elif n:vals.append(n);n=0
    if n:vals.append(n)
    return vals

def correlation(a,b):
    a=np.asarray(a);b=np.asarray(b);valid=np.isfinite(a)&np.isfinite(b)
    return float(np.corrcoef(a[valid],b[valid])[0,1]) if valid.sum()>2 else np.nan

def describe(d,panel,edges):
    s=edges.loc[edges.valid].copy();metrics={};patients=[];strata=[]
    metrics.update(transitions=len(s),segments=s.groupby('participant').segment_id.nunique().sum(),scheduled=len(d),available_rate=d.avail.eq(True).mean(),action_rate=s.action.mean(),home_rate=(s.location.isin(['home','work'])).mean(),interval_q50=s.gap_minutes.median(),interval_q95=s.gap_minutes.quantile(.95),patients=len(panel.patient_ids))
    for name,col in [('reward','post_steps'),('pre','pre_steps')]:
        x=s[col].to_numpy();metrics[name+'_zero']=float(np.mean(x==0))
        for q,val in zip(QUANTILES,np.quantile(np.log(x+.5),QUANTILES)):metrics[f'{name}_q{int(100*q)}']=val
    missing=[];pairs_pre=[];pairs_loc=[]
    for pid,g in d.groupby('user.index'):
        r=s[s.participant==pid]
        eligible=np.array([not row_issues(x,True) and pd.notna(x.jbsteps30) for _,x in g.iterrows()])
        ep=episodes(~eligible);missing.extend(ep)
        lp=np.log(g.jbsteps30pre.to_numpy()+.5)
        lo=np.array([np.nan if pd.isna(x) else float(x in ['home','work']) for x in g['dec.location.category']])
        pairs_pre.extend(zip(lp[:-1],lp[1:]));pairs_loc.extend(zip(lo[:-1],lo[1:]))
        patients.append({'patient':int(pid),'transitions':len(r),'A0':int((r.action==0).sum()),'A1':int(r.action.sum()),'segments':r.segment_id.nunique(),'available_rate':g.avail.eq(True).mean(),'home_rate':r.location.isin(['home','work']).mean(),'reward_zero':(r.post_steps==0).mean(),'reward_median':r.reward.median(),'pre_zero':(r.pre_steps==0).mean(),'pre_median':np.log(r.pre_steps+.5).median(),'missing_episodes':len(ep),'missing_points':int((~eligible).sum()),'missing_reentries':int(np.sum((~eligible[:-1])&eligible[1:]))})
    pairs_pre=np.asarray(pairs_pre);pairs_loc=np.asarray(pairs_loc);valid=np.isfinite(pairs_loc).all(axis=1)
    metrics['pre_autocorr']=correlation(pairs_pre[:,0],pairs_pre[:,1]);metrics['location_stay']=float(np.mean(pairs_loc[valid,0]==pairs_loc[valid,1]))
    metrics['missing_length_q50']=float(np.median(missing)) if missing else 0.
    metrics['missing_length_q90']=float(np.quantile(missing,.9)) if missing else 0.
    metrics['missing_episodes']=len(missing)
    metrics['reentries']=sum(max(0,int(p['segments'])-1) for p in patients)
    for variable in ['action','slot']:
        for key,g in s.groupby(variable):
            row={'stratum':variable,'value':int(key),'n':len(g),'reward_zero':(g.post_steps==0).mean(),'pre_zero':(g.pre_steps==0).mean(),'home_rate':g.location.isin(['home','work']).mean()}
            for name,col in [('reward','post_steps'),('pre','pre_steps')]:
                for q,val in zip(QUANTILES,np.quantile(np.log(g[col]+.5),QUANTILES)):row[f'{name}_q{int(q*100)}']=val
            strata.append(row)
    sc,support=screen(panel)
    metrics['all_candidates_supported']=len(sc.admissible_candidates)==7
    return metrics,patients,support,strata,missing

def gates(observed,table,observed_patients,sim_patients,support,fit_adequate):
    rows=[]
    def add(name,obs,sim,tolerance,mode='absolute'):
        err=abs(sim-obs)/(abs(obs) if obs!=0 else np.nan) if mode=='relative' else abs(sim-obs)
        rows.append({'check':name,'observed':obs,'simulated':sim,'error':err,'tolerance':tolerance,'comparison':mode,'pass':bool(np.isfinite(err) and err<=tolerance)})
    med=table.median(numeric_only=True)
    for name,tol in [('transitions',.2),('segments',.3),('reentries',.3),('interval_q50',.25),('interval_q95',.25)]:add(name,observed[name],med[name],tol,'relative')
    for name,tol in [('available_rate',.08),('action_rate',.05),('home_rate',.1),('reward_zero',.1),('pre_zero',.1),('pre_autocorr',.2),('location_stay',.15)]:add(name,observed[name],med[name],tol)
    for name in ['pre','reward']:
        for q in QUANTILES:
            key=f'{name}_q{int(q*100)}';add(key,observed[key],med[key],1. if q==.99 else .75)
    add('missing_length_q50',observed['missing_length_q50'],med['missing_length_q50'],1.)
    add('missing_length_q90',observed['missing_length_q90'],med['missing_length_q90'],max(2.,.5*observed['missing_length_q90']))
    cover=[]
    for _,r in observed_patients.iterrows():
        g=sim_patients[sim_patients.patient==r.patient]
        for col in ['transitions','A1','segments','home_rate','reward_zero','reward_median','pre_zero','pre_median']:
            lo,hi=g[col].quantile([.05,.95]);cover.append({'patient':int(r.patient),'metric':col,'observed':r[col],'q05':lo,'q95':hi,'covered':lo<=r[col]<=hi})
    cover=pd.DataFrame(cover)
    def minimum(name,val,threshold):rows.append({'check':name,'observed':None,'simulated':val,'error':None,'tolerance':threshold,'comparison':'>=','pass':bool(np.isfinite(val) and val>=threshold)})
    for col in ['transitions','A1']:minimum('patient_coverage_'+col,cover[cover.metric==col].covered.mean(),.8)
    for u,g in support.groupby('candidate'):minimum('candidate_'+str(u),g.admissible.mean(),.95)
    minimum('joint_support',table.all_candidates_supported.mean(),.9)
    minimum('generated_complete',len(table),200)
    minimum('positive_families_adequate',int(fit_adequate),1)
    return pd.DataFrame(rows),cover
