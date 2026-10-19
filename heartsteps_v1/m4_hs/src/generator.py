"""Full scheduled-panel regeneration, separate from the frozen analysis."""
from __future__ import annotations
import copy
import numpy as np
import pandas as pd
from src.rs_cusum_phase2rd.protocol import stable_seed
from .models import PIDS,bernoulli,count,logstep
from .common import CFG

def compile_models(fit):
    models=copy.deepcopy(fit['models'])
    for m in models.values():
        m['coef']=np.asarray(m['coef'])
        if 'positive' in m:
            m['positive']['coef']=np.asarray(m['positive']['coef'])
            m['positive']['residual']=np.asarray(m['positive']['residual'])
    return models

def component_seeds(seed):
    return {k:stable_seed(seed['design_seed'] if k!='reward' else seed['reward_seed'],'M4-HS-v0',k) for k in ['initialization','location','pre','technical','availability','action','mask','reward']}

def simulate(fit,template,seed):
    models=compile_models(fit);components=component_seeds(seed)
    rng={k:np.random.default_rng(v) for k,v in components.items()};out=[]
    for pid,g0 in template.groupby('user.index'):
        pid=int(pid);g=g0.reset_index(drop=True);pool=fit['initial_pools'][str(pid)] if str(pid) in fit['initial_pools'] else fit['initial_pools'][pid]
        prev=dict(pool[int(rng['initialization'].integers(len(pool)))])
        histpre=logstep(prev['pre']);histloc=prev['loc'];preunknown=0.;locunknown=0.
        times=g['sugg.select.utime'];gaps=times.diff().dt.total_seconds().to_numpy()/3600
        slotgap={}
        for k in range(1,6):
            vals=gaps[(g['sugg.select.slot'].to_numpy()==k)&np.isfinite(gaps)&(gaps>0)]
            # Only design gaps, not outcomes/support, determine burn-in periodic gaps.
            if not len(vals):raise RuntimeError('NO_POSITIVE_TEMPLATE_GAP_FOR_SLOT')
            slotgap[k]=float(np.median(vals))
        firstslot=int(g.iloc[0]['sugg.select.slot'])
        for idx in range(-CFG['burn_in'],len(g)):
            if idx<0:
                slot=(firstslot-1+idx)%5+1;gap=slotgap[slot];raw=None
            else:
                raw=g.iloc[idx];slot=int(raw['sugg.select.slot']);gap=slotgap[slot] if idx==0 else float(gaps[idx])
            z={'pid':pid,'slot':slot,'gap':np.log1p(max(0,gap)),'prev_pre':logstep(prev['pre']),'prev_post':logstep(prev['post']),'prev_a':prev['a'],'prev_loc':prev['loc'],'prev_c':prev['c'],'prev_v':prev['v'],'prev_opre':prev['opre'],'prev_opost':prev['opost'],'prev_oloc':prev['oloc'],'hist_pre':histpre,'hist_loc':histloc,'hist_pre_unknown':preunknown,'hist_loc_unknown':locunknown}
            loc=bernoulli(models,'location',z,rng['location']);z['loc']=loc
            pre=count(models,'pre',z,rng['pre']);z['pre']=logstep(pre)
            c=bernoulli(models,'technical',z,rng['technical']);z['c']=c
            v=bernoulli(models,'available',z,rng['availability']);z['v']=v
            a=float(rng['action'].random()<CFG['action_probability']) if v else 0.;z['a']=a
            post=count(models,'reward',z,rng['reward'])
            opre=bernoulli(models,'pre_observed',z,rng['mask']);z['opre']=opre
            opost=bernoulli(models,'post_observed',z,rng['mask'])
            oloc=bernoulli(models,'loc_observed',z,rng['mask'])
            if idx>=0:
                scheduled=raw['sugg.select.utime'];decision=scheduled+pd.Timedelta(seconds=CFG['canonical_delay_seconds'])
                row={'user.index':pid,'source_row':int(raw.source_row),'decision.index':int(raw['decision.index']),'decision.index.nogap':int(raw['decision.index']),'sugg.select.utime':scheduled,'sugg.decision.utime':decision if c else pd.NaT,'sugg.context.utime':decision if c else pd.NaT,'sugg.select.slot':slot,'sugg.decision.slot':slot,'context_age_min':0. if c else np.nan,'study_day_from_first_record_date':raw.study_day_from_first_record_date,'avail':bool(v),'connect':bool(c),'send':bool(a) if c else np.nan,'is.randomized':bool(a) if c and v else np.nan,'jbsteps30pre':pre if opre else np.nan,'jbsteps30':post if opost else np.nan,'dec.location.category':('home' if loc else 'other') if oloc else np.nan,'latent_pre':pre,'latent_post':post,'latent_location':loc,'latent_action':int(a),'technical_eligible':bool(c),'pre_observed':bool(opre),'post_observed':bool(opost),'location_observed':bool(oloc)}
                out.append(row)
            if opre:histpre=logstep(pre);preunknown=0.
            if oloc:histloc=loc;locunknown=0.
            prev={'pre':pre,'post':post,'loc':loc,'a':a,'c':c,'v':v,'opre':opre,'opost':opost,'oloc':oloc}
    d=pd.DataFrame(out)
    for c in ['sugg.select.utime','sugg.decision.utime','sugg.context.utime']:d[c]=pd.to_datetime(d[c],utc=True)
    assert tuple(sorted(d['user.index'].unique()))==PIDS
    assert np.isfinite(d[['latent_pre','latent_post','latent_location','latent_action']].to_numpy()).all()
    return d,components
