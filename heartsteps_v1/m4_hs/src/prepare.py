"""Engineering tests and fit diagnostics, before any registered simulation."""
import json
import time
import numpy as np
import pandas as pd
from heartsteps_v1.scripts.audit_data import load_suggestions
from heartsteps_v1.scripts import adapter
from .common import BASE,PARENT,protected_check,rebuild,write_json
from .models import fit_all,SPECS

def tests():
    protection=protected_check();d=load_suggestions();t=time.perf_counter()
    native,en,c=rebuild(d,False);fast,ef,_=rebuild(d,True)
    pd.testing.assert_frame_equal(en,ef,check_dtype=False,check_exact=True)
    assert len(native.records)==len(fast.records)==4766
    for a,b in zip(native.records,fast.records):
        assert a.patient_id==b.patient_id and a.elapsed_index==b.elapsed_index and a.segment_id==b.segment_id and a.action==b.action and a.reward_binary==b.reward_binary and a.tau_t==b.tau_t and a.tau_next==b.tau_next
        assert np.array_equal(a.state,b.state) and np.array_equal(a.next_state,b.next_state)
    saved=pd.read_csv(PARENT/'data_processed/transitions.csv')
    assert np.array_equal(saved.action.to_numpy(),ef.loc[ef.valid].action.to_numpy())
    assert np.allclose(saved.reward,ef.loc[ef.valid].reward,rtol=1e-14,atol=1e-14)
    # Explicit adversity: no selected simulation seeds are used.
    small=d[d['user.index']==1].head(12).copy().reset_index(drop=True)
    origin=pd.Timestamp('2020-01-01',tz='UTC')
    for i in range(len(small)):
        ti=origin+pd.Timedelta(hours=2*i)
        for col in ['sugg.select.utime','sugg.decision.utime','sugg.context.utime']:small.loc[i,col]=ti
        small.loc[i,['avail','connect','send','is.randomized']]=[True,True,bool(i%2),bool(i%2)]
        small.loc[i,['decision.index','decision.index.nogap','sugg.select.slot','sugg.decision.slot']]=[i+1,i+1,i%5+1,i%5+1]
        small.loc[i,['context_age_min','jbsteps30pre','jbsteps30']]=[0,i*10,i*20]
        small.loc[i,'dec.location.category']='home'
    cases={}
    modifications={
      'missing_pre':(3,'jbsteps30pre',np.nan),
      'disconnect':(3,'connect',False),
      'travel':(3,'decision.index.nogap',np.nan),
      'future_context':(3,'context_age_min',-1),
      'unavailable_sent':(3,'avail',False),
      'missing_reward':(3,'jbsteps30',np.nan),
      'long_gap':(5,'sugg.decision.utime',origin+pd.Timedelta(days=4))}
    for name,(i,col,value) in modifications.items():
        data=small.copy();data.loc[i,col]=value
        n,x,_=rebuild(data,False);f,y,_=rebuild(data,True)
        pd.testing.assert_frame_equal(x,y,check_dtype=False,check_exact=True)
        assert (~y.valid).any() and set(f.patient_ids)=={'1'}
        assert y[y.valid].segment_id.nunique()>=2
        cases[name]={'valid_edges':int(y.valid.sum()),'segments':int(y[y.valid].segment_id.nunique()),'rejections':y.loc[~y.valid,'reasons'].tolist()}
    assert adapter.build.__globals__['load_suggestions'] is load_suggestions
    assert not any('day' in x or 'tau' in x or 'candidate' in x for v in SPECS.values() for x in v)
    z=np.array([0.,1.,15.,1000.]);r=np.log(z+.5)
    assert r[0]<0 and not np.array_equal(r,r.astype(int)) and np.allclose(np.exp(r)-.5,z)
    record={'status':'PASS','full_original_native_facade_equivalent':True,'observed_records':4766,'frozen_panel_reward_action_equivalent':True,'original_loader_unchanged':True,'adversarial_cases':cases,'protected':protection,'runtime_seconds':time.perf_counter()-t}
    write_json(BASE/'diagnostics/engineering_tests.json',record)
    print('ENGINEERING_TESTS_PASS',flush=True)

def main():
    for path in ['specification','src','diagnostics','results','logs']:(BASE/path).mkdir(parents=True,exist_ok=True)
    if (BASE/'diagnostics/engineering_tests.json').exists():raise RuntimeError('TESTS_ALREADY_RECORDED')
    tests()
    fit,d=fit_all()
    print(json.dumps({'fit_complete':True,'scheduled':len(d),'positive_adequacy':fit['all_positive_families_adequate'],'positive_comparison':{k:fit['models'][k]['positive']['comparison'] for k in ['pre','reward']}},indent=2),flush=True)

if __name__=='__main__':main()
