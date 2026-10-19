"""Pre-specified stationary nuisance models; no CUSUM imports or tuning."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit,gammaln,digamma
from scipy.stats import nbinom
from heartsteps_v1.scripts.audit_data import load_suggestions
from heartsteps_v1.scripts.adapter import row_issues
from .common import CFG,BASE,write_json

PIDS=tuple(range(1,38))
SPECS={
 'location':['prev_loc','prev_pre'],
 'pre':['prev_pre','prev_post','prev_a','loc'],
 'technical':['prev_c','prev_v','hist_pre','hist_loc','hist_pre_unknown','hist_loc_unknown'],
 'available':['prev_v','c','hist_pre','hist_loc','hist_pre_unknown','hist_loc_unknown'],
 'reward':['pre','loc','a','v'],
 'pre_observed':['prev_opre','prev_opost','c','hist_pre','hist_loc','hist_pre_unknown','hist_loc_unknown'],
 'post_observed':['prev_opre','prev_opost','opre','c','hist_pre','hist_loc','hist_pre_unknown','hist_loc_unknown'],
 'loc_observed':['c','prev_oloc']
}

def logstep(y): return float(np.log(y+.5))/4

def fixed_vector(name,row):
    return [1.]+[float(row['slot']==k) for k in range(2,6)]+([] if name in ('reward','loc_observed') else [float(row.get('gap',0))])+[float(row[k]) for k in SPECS[name]]

def design(name,rows):
    base=np.array([fixed_vector(name,r) for r in rows],float)
    ids=np.zeros((len(rows),37))
    ids[np.arange(len(rows)),[int(r['pid'])-1 for r in rows]]=1
    return np.column_stack([base,ids])

def penalty(p):
    out=np.ones(p)*CFG['slope_penalty'];out[0]=0;out[-37:]=CFG['patient_penalty'];return out

def logistic(x,y):
    pen=penalty(x.shape[1])
    def fun(b):
        z=x@b
        return np.sum(np.logaddexp(0,z)-y*z)+.5*np.sum(pen*b*b),x.T@(expit(z)-y)+pen*b
    result=minimize(fun,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':2000,'ftol':1e-11,'gtol':1e-6})
    if not result.success or not np.isfinite(result.x).all(): raise RuntimeError('LOGISTIC_FIT_FAILURE:'+str(result.message))
    return result.x,{'converged':bool(result.success),'iterations':result.nit,'objective':result.fun,'n':len(y),'event_rate':np.mean(y),'probability_quantiles':np.quantile(expit(x@result.x),[0,.05,.5,.95,1]).tolist()}

def ridge(x,y):
    b=np.linalg.solve(x.T@x+np.diag(penalty(x.shape[1])),x.T@y)
    residual=y-x@b; residual-=residual.mean()
    return b,residual

def fit_nb(x,y):
    """Shifted NB: z=y-1. No support-dependent choices or parameter clipping."""
    z=y-1.; b0,_=ridge(x,np.log(y)); init=np.r_[b0,-1.]
    pen=penalty(x.shape[1])
    def fun(theta):
        eta=x@theta[:-1]; k=theta[-1]
        if np.any(abs(eta)>500) or abs(k)>100: return np.inf,np.zeros_like(theta)
        mu=np.exp(eta); r=np.exp(k); logden=np.log(r+mu)
        ll=gammaln(z+r)-gammaln(r)-gammaln(z+1)+r*(k-logden)+z*(eta-logden)
        gb=x.T@((z+r)*mu/(r+mu)-z)+pen*theta[:-1]
        gr=-np.sum((digamma(z+r)-digamma(r)+k-logden+1-(r+z)/(r+mu))*r)
        return -np.sum(ll)+.5*np.sum(pen*theta[:-1]**2),np.r_[gb,gr]
    res=minimize(fun,init,jac=True,method='L-BFGS-B',options={'maxiter':2000,'ftol':1e-10})
    return res.x[:-1],float(np.exp(res.x[-1])),{'converged':bool(res.success),'message':str(res.message),'iterations':res.nit,'objective':float(res.fun)}

def positive_fit(x,y,hold):
    train=~hold
    if min(train.sum(),hold.sum())<20: raise RuntimeError('INSUFFICIENT_POSITIVE_FITTING_HOLDOUT')
    b,resid=ridge(x[train],np.log(y[train]));err=np.log(y[hold])-x[hold]@b
    qlo,qhi=np.quantile(resid,[.05,.95])
    logdiag={'converged':True,'median_log_error':float(np.median(err)),'mean_absolute_log_error':float(np.mean(abs(err))),'coverage90':float(np.mean((err>=qlo)&(err<=qhi))),'training_n':int(train.sum()),'holdout_n':int(hold.sum())}
    nb_b,nb_r,nbdiag=fit_nb(x[train],y[train])
    mu=np.exp(x[hold]@nb_b);prob=nb_r/(nb_r+mu)
    med=1+nbinom.ppf(.5,nb_r,prob);lo=1+nbinom.ppf(.05,nb_r,prob);hi=1+nbinom.ppf(.95,nb_r,prob)
    nbdiag.update(median_log_error=float(np.median(np.log(y[hold])-np.log(med))),mean_absolute_log_error=float(np.mean(abs(np.log(y[hold])-np.log(med)))),coverage90=float(np.mean((y[hold]>=lo)&(y[hold]<=hi))),mean_log_score=float(np.mean(nbinom.logpmf(y[hold]-1,nb_r,prob))))
    def adequate(d):
        return d['converged'] and np.isfinite(d['median_log_error']) and abs(d['median_log_error'])<=CFG['positive_median_log_error_max'] and CFG['positive_coverage_min']<=d['coverage90']<=CFG['positive_coverage_max']
    family='empirical_log' if adequate(logdiag) or not adequate(nbdiag) else 'shifted_nb'
    ok=adequate(logdiag) if family=='empirical_log' else adequate(nbdiag)
    if family=='empirical_log': coef,resid=ridge(x,np.log(y)); shape=None
    else:
        coef,shape,refit=fit_nb(x,y);resid=np.array([])
        if not refit['converged']: raise RuntimeError('NB_FULL_REFIT_FAILURE')
    return {'family':family,'coef':coef.tolist(),'residual':resid.tolist(),'nb_shape':shape,'adequate':ok,'comparison':{'empirical_log':logdiag,'shifted_nb':nbdiag},'n':len(y)}

def schedule_data():
    d=load_suggestions()
    origin=d.groupby('user.index')['sugg.select.utime'].transform('min')
    minutes=(d['sugg.select.utime']-origin).dt.total_seconds()/60
    return d.loc[minutes.between(0,60480)].copy().reset_index(drop=True)

def fit_data(d):
    tables={k:[] for k in SPECS};targets={k:[] for k in SPECS};initial={};templates={}
    for pid,g in d.groupby('user.index'):
        gg=g.reset_index(drop=True);initial[int(pid)]=[];templates[int(pid)]=gg
        lastpre=0.;lastloc=0.;preunknown=1.;locunknown=1.;prev=None
        for j,r in gg.iterrows():
            pre=float(r.jbsteps30pre);post=float(r.jbsteps30);loc=np.nan if pd.isna(r['dec.location.category']) else float(str(r['dec.location.category']).lower() in {'home','work'})
            c=float(not row_issues(r,False));v=float(bool(r.avail));a=float(bool(r.send)) if pd.notna(r.send) else np.nan
            opre=float(np.isfinite(pre) and pre>=0);opost=float(np.isfinite(post) and post>=0);oloc=float(np.isfinite(loc))
            if prev is not None:
                gap=(r['sugg.select.utime']-prev['time']).total_seconds()/3600
                z={'pid':int(pid),'ordinal':j,'slot':int(r['sugg.select.slot']),'gap':np.log1p(max(0,gap)),'prev_pre':prev['pre'],'prev_post':prev['post'],'prev_a':prev['a'],'prev_loc':prev['loc'],'prev_c':prev['c'],'prev_v':prev['v'],'prev_opre':prev['opre'],'prev_opost':prev['opost'],'prev_oloc':prev['oloc'],'hist_pre':lastpre,'hist_loc':lastloc,'hist_pre_unknown':preunknown,'hist_loc_unknown':locunknown,'c':c,'v':v,'a':a,'pre':logstep(pre) if opre else np.nan,'loc':loc,'opre':opre}
                outcomes={'location':loc,'pre':pre if opre else np.nan,'technical':c,'available':v,'reward':post if opost else np.nan,'pre_observed':opre,'post_observed':opost,'loc_observed':oloc}
                for name in SPECS:
                    target=outcomes[name]
                    valid=np.isfinite(target) and np.isfinite(fixed_vector(name,z)).all()
                    if name=='reward': valid=valid and pd.notna(r.send) and (not v or (pd.notna(r['is.randomized']) and r.send==r['is.randomized'])) and (v or a==0)
                    if valid: tables[name].append(z.copy());targets[name].append(target)
            if opre and opost and oloc and np.isfinite(a): initial[int(pid)].append({'pre':pre,'post':post,'loc':loc,'a':a,'c':c,'v':v,'opre':opre,'opost':opost,'oloc':oloc})
            if opre:lastpre=logstep(pre);preunknown=0.
            if oloc:lastloc=loc;locunknown=0.
            prev={'time':r['sugg.select.utime'],'pre':logstep(pre) if opre else np.nan,'post':logstep(post) if opost else np.nan,'a':a,'loc':loc,'c':c,'v':v,'opre':opre,'opost':opost,'oloc':oloc}
    assert all(initial[p] for p in PIDS)
    return tables,targets,initial,templates

def fit_all():
    d=schedule_data();tables,targets,initial,templates=fit_data(d);models={}
    for name,rows in tables.items():
        x=design(name,rows);y=np.array(targets[name]);binary=(y==0).astype(float) if name in ('pre','reward') else y
        coef,diag=logistic(x,binary)
        m={'predictors':['intercept']+['slot'+str(k) for k in range(2,6)]+([] if name in ('reward','loc_observed') else ['log_gap'])+SPECS[name]+['patient_'+str(p) for p in PIDS],'coef':coef.tolist(),'fit':diag,'base_dimension':x.shape[1]-37}
        if name in ('pre','reward'):
            pos=y>0;hold=np.array([r['ordinal']%5==0 for r in rows])[pos]
            m['positive']=positive_fit(x[pos],y[pos],hold)
        models[name]=m
    fit={'models':models,'initial_pools':initial,'patients':list(PIDS),'scheduled_n':len(d),'specification':'M4-HS-v0','all_positive_families_adequate':all(models[k]['positive']['adequate'] for k in ('pre','reward'))}
    write_json(BASE/'diagnostics/fitted_models.json',fit)
    co=[]
    for name,m in models.items():
        for term,val in zip(m['predictors'],m['coef']):co.append({'model':name,'component':'zero' if name in ('pre','reward') else 'binary','term':term,'coefficient':val})
        if 'positive' in m:
            for term,val in zip(m['predictors'],m['positive']['coef']):co.append({'model':name,'component':'positive_'+m['positive']['family'],'term':term,'coefficient':val})
    pd.DataFrame(co).to_csv(BASE/'diagnostics/model_coefficients.csv',index=False)
    return fit,d

def predictor(m,row,coef=None):
    name=row['_model'];b=np.asarray(m['coef'] if coef is None else coef)
    return float(np.dot(fixed_vector(name,row),b[:m['base_dimension']])+b[m['base_dimension']+int(row['pid'])-1])

def bernoulli(models,name,row,rng):
    z=dict(row,_model=name);prob=float(expit(predictor(models[name],z)))
    if not np.isfinite(prob):raise FloatingPointError('NONFINITE_PROBABILITY:'+name)
    return float(rng.random()<prob)

def count(models,name,row,rng):
    z=dict(row,_model=name);m=models[name]
    if bernoulli(models,name,row,rng):return 0.
    pos=m['positive'];eta=predictor(m,z,pos['coef'])
    if pos['family']=='empirical_log':
        value=np.exp(eta+float(rng.choice(pos['residual'])))
        if not np.isfinite(value) or value<=0 or value>2**53:raise FloatingPointError('UNREPRESENTABLE_POSITIVE_COUNT:'+name)
        return float(math.ceil(value))
    mu=np.exp(eta);r=pos['nb_shape']
    if not np.isfinite(mu):raise FloatingPointError('NONFINITE_NB_MEAN:'+name)
    return float(1+rng.negative_binomial(r,r/(r+mu)))
