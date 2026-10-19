"""Independent readback, type-I summaries, plots and manifest after 200/200."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from .experiment import (ALPHA,B,CANDIDATES,LAYERS,OUT,R,atomic_csv,atomic_json,
                         digest,inner_seed,read,registered,verify_freeze,verify_seal)
from heartsteps_v1.m4_hs.src.generator import component_seeds
from heartsteps_v1.m4_hs.formal_src.formal_b199 import old_protection


def wilson(k,n):
    if n==0:return (None,None)
    z=1.959963984540054;p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return (float(center-half),float(center+half))


def finite_check(values,name):
    x=np.asarray(values,float)
    if not np.isfinite(x).all(): raise RuntimeError('NONFINITE_'+name)
    return x


def inspect_outer(seed):
    i=seed['outer_id'];folder=OUT/'outer'/f'r{i:03d}'
    outer=verify_seal(folder,seed)
    truth_seed={k:seed[k] for k in ('draw_seed','design_seed','reward_seed')}
    truth=verify_seal(folder/'truth',truth_seed)
    if truth['status']!='VALID':
        if outer['status']!='OUTER_ANALYSIS_UNAVAILABLE': raise RuntimeError('TRUTH_FAILURE_STATUS_MISMATCH')
        return dict(outer_id=i,status=outer['status'],reason=outer.get('reason'),observed_D4=None,
                    p_original=None,reject_original=None,p_m4=None,reject_m4=None,
                    original_runtime=None,m4_runtime=None,inner_attempted=0,inner_valid=0,inner_invalid=0)
    observed=verify_seal(folder/'observed')
    if observed['status']!='VALID':
        if outer['status']!='OUTER_ANALYSIS_UNAVAILABLE': raise RuntimeError('OUTER_UNAVAILABLE_STATUS_MISMATCH')
        return dict(outer_id=i,status=outer['status'],reason=outer.get('reason'),observed_D4=None,
                    p_original=None,reject_original=None,p_m4=None,reject_m4=None,
                    original_runtime=None,m4_runtime=None,inner_attempted=0,inner_valid=0,inner_invalid=0)
    if digest(folder/'observed/checkpoint.json')!=outer['observed_checkpoint_sha']:
        raise RuntimeError('OBSERVED_CHECKPOINT_MISMATCH')
    values=finite_check(observed['D0_D4'],'OBSERVED');
    if values.shape!=(5,) or tuple(observed['admissible'])!=CANDIDATES: raise RuntimeError('OBSERVED_SHAPE_OR_SUPPORT')
    with np.load(folder/'observed/process.npz',allow_pickle=False) as f:
        if any(f[k].shape!=(7,64) for k in ('raw','normalized','joint')): raise RuntimeError('OBSERVED_PROCESS_SHAPE')
        if not np.array_equal(f['D0_D4'],values): raise RuntimeError('OBSERVED_LAYER_READBACK')
        finite_check(f['joint'],'OBSERVED_JOINT')
    original=verify_seal(folder/'original')
    m4=verify_seal(folder/'m4')
    if digest(folder/'original/checkpoint.json')!=outer['original_checkpoint_sha'] or digest(folder/'m4/checkpoint.json')!=outer['m4_checkpoint_sha']:
        raise RuntimeError('CALIBRATION_CHECKPOINT_MISMATCH')
    op=orej=ort=None
    if original['status']=='VALID':
        od=pd.read_csv(folder/'original/calibration_distribution.csv')[list(LAYERS)].to_numpy(float)
        if od.shape!=(B,5): raise RuntimeError('ORIGINAL_DRAW_COUNT')
        finite_check(od,'ORIGINAL_REFERENCE')
        calc=np.array([(1+np.count_nonzero(od[:,j]>=values[j]))/(B+1) for j in range(5)])
        if not np.array_equal(calc,np.asarray(original['p'])): raise RuntimeError('ORIGINAL_P_READBACK')
        op=float(calc[4]);orej=bool(op<=ALPHA);ort=float(original['runtime_seconds'])
    mp=mrej=mrt=None
    records=[]
    if (folder/'m4/draws').exists():
        draw_root=folder/'m4/draws';expected={f'draw_{b:03d}' for b in range(B)}
        if {p.name for p in draw_root.iterdir()}!=expected: raise RuntimeError('INNER_DRAW_ID_COVERAGE')
        for b in range(B):
            s=inner_seed(i,b);rec=verify_seal(draw_root/f'draw_{b:03d}',s)
            if rec['component_seeds']!=component_seeds(s): raise RuntimeError('COMPONENT_SEED_MISMATCH')
            if rec['valid']:
                finite_check(rec['D0_D4'],'INNER_LAYERS')
                with np.load(draw_root/f'draw_{b:03d}'/'process.npz',allow_pickle=False) as f:
                    if f['joint'].shape!=(7,64) or not np.array_equal(f['D0_D4'],rec['D0_D4']):
                        raise RuntimeError('INNER_PROCESS_READBACK')
                    finite_check(f['joint'],'INNER_PROCESS')
                fd=read(draw_root/f'draw_{b:03d}'/'fqi.json')
                if len(fd)!=14 or not all(x['converged'] for x in fd): raise RuntimeError('INNER_FQI_READBACK')
                if len(rec['support'])!=7 or not all(x['admissible'] for x in rec['support']):
                    raise RuntimeError('INNER_SUPPORT_READBACK')
            else:
                if not rec.get('failure_category') or not rec.get('error'):
                    raise RuntimeError('INNER_INVALID_UNCLASSIFIED')
            records.append(rec)
    if m4['status']=='VALID':
        if len(records)!=B: raise RuntimeError('M4_VALID_WITHOUT_FULL_ATTEMPTS')
        md=pd.read_csv(folder/'m4/calibration_distribution.csv')[list(LAYERS)].to_numpy(float)
        expected=np.asarray([x['D0_D4'] for x in records if x['valid']],float)
        if not np.array_equal(md,expected): raise RuntimeError('M4_DISTRIBUTION_READBACK')
        finite_check(md,'M4_REFERENCE')
        v=len(md)
        calc=np.array([(1+np.count_nonzero(md[:,j]>=values[j]))/(v+1) for j in range(5)])
        if not np.array_equal(calc,np.asarray(m4['p'])): raise RuntimeError('M4_P_READBACK')
        mp=float(calc[4]);mrej=bool(mp<=ALPHA);mrt=float(m4['runtime_seconds'])
    if len(records)!=m4['attempted_draws'] or sum(x['valid'] for x in records)!=m4['valid_draws']:
        raise RuntimeError('M4_DRAW_ACCOUNTING')
    status='PAIRED_VALID' if original['status']=='VALID' and m4['status']=='VALID' else 'METHOD_UNAVAILABLE'
    if outer['status']!=status: raise RuntimeError('FINAL_OUTER_STATUS_MISMATCH')
    return dict(outer_id=i,status=status,reason=None if status=='PAIRED_VALID' else m4.get('reason') or original.get('reason'),
                observed_D4=float(values[4]),p_original=op,reject_original=orej,
                p_m4=mp,reject_m4=mrej,original_runtime=ort,m4_runtime=mrt,
                inner_attempted=len(records),inner_valid=sum(x['valid'] for x in records),
                inner_invalid=sum(not x['valid'] for x in records),
                inner_failure_categories=dict(Counter(x['failure_category'] for x in records if not x['valid'])))


def plots(frame,summary,paired):
    figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
    names=['Original','M4-HS'];rows=[summary[x] for x in names]
    fig,ax=plt.subplots(figsize=(6,4));x=np.arange(2)
    vals=[r['estimate'] for r in rows]
    lo=[v-r['wilson95'][0] for v,r in zip(vals,rows)]
    hi=[r['wilson95'][1]-v for v,r in zip(vals,rows)]
    ax.errorbar(x,vals,yerr=np.array([lo,hi]),fmt='o',capsize=4)
    ax.axhline(ALPHA,color='crimson',linestyle='--',label='nominal 0.05')
    ax.set_xticks(x,names);ax.set_ylabel('D4 rejection rate');ax.set_ylim(bottom=0)
    ax.set_title('R=200 stationary-null outer datasets');ax.legend();fig.tight_layout()
    fig.savefig(figdir/'type1_comparison.png',dpi=160);plt.close(fig)
    table=np.array([[paired['n00'],paired['n01']],[paired['n10'],paired['n11']]])
    fig,ax=plt.subplots(figsize=(5,4));im=ax.imshow(table,cmap='Blues')
    for (i,j),v in np.ndenumerate(table):ax.text(j,i,str(v),ha='center',va='center')
    ax.set_xticks([0,1],['M4 no','M4 yes']);ax.set_yticks([0,1],['Original no','Original yes'])
    ax.set_title('Paired D4 rejection; R=200 null outers');fig.colorbar(im,ax=ax);fig.tight_layout()
    fig.savefig(figdir/'paired_rejections.png',dpi=160);plt.close(fig)
    pair=frame[frame.status=='PAIRED_VALID']
    for column,name in [('p_original','original'),('p_m4','m4')]:
        fig,ax=plt.subplots(figsize=(6,4));ax.hist(frame[column].dropna().astype(float),bins=np.linspace(0,1,21),edgecolor='white')
        ax.set_xlabel('D4 p-value');ax.set_ylabel('outer count');ax.set_title(f'{name}: R=200 stationary-null outers')
        fig.tight_layout();fig.savefig(figdir/f'{name}_pvalue_histogram.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(5,5));ax.scatter(pair.p_original.astype(float),pair.p_m4.astype(float),s=20,alpha=.65)
    ax.plot([0,1],[0,1],color='gray',linestyle='--');ax.set_xlim(0,1);ax.set_ylim(0,1)
    ax.set_xlabel('Original D4 p');ax.set_ylabel('M4-HS D4 p');ax.set_title('Same outer datasets; R=200 null outers')
    fig.tight_layout();fig.savefig(figdir/'paired_pvalue_scatter.png',dpi=160);plt.close(fig)
    if pair.shape[0]:
        fig,ax=plt.subplots(figsize=(5,3));ax.errorbar([0],[paired['difference']],
            yerr=[[paired['difference']-paired['difference_ci95'][0]],
                  [paired['difference_ci95'][1]-paired['difference']]],fmt='o',capsize=4)
        ax.axhline(0,color='gray',linestyle='--');ax.set_xticks([0],['Original - M4'])
        ax.set_ylabel('Paired D4 rejection-rate difference')
        ax.set_title('R=200 stationary-null outer datasets');fig.tight_layout()
        fig.savefig(figdir/'paired_difference.png',dpi=160);plt.close(fig)


def finalize():
    if not (OUT/'results/COMPUTATION_COMPLETE.json').exists(): raise RuntimeError('COMPUTATION_NOT_COMPLETE')
    if (OUT/'results/runner.lock').exists(): raise RuntimeError('RUNNER_STILL_ACTIVE_OR_STALE_LOCK')
    if (OUT/'manifest.json').exists(): raise RuntimeError('ALREADY_FINALIZED')
    verify_freeze();protected=old_protection();outer_seeds,inner_seeds=registered()
    expected={f'r{i:03d}' for i in range(R)}
    actual={p.name for p in (OUT/'outer').iterdir()}
    if actual!=expected: raise RuntimeError('OUTER_ID_COVERAGE_MISMATCH')
    if list(OUT.rglob('*.tmp')): raise RuntimeError('TEMPORARY_OUTPUTS_PRESENT')
    rows=[inspect_outer(seed) for seed in outer_seeds]
    if [r['outer_id'] for r in rows]!=list(range(R)): raise RuntimeError('OUTER_SEED_COVERAGE_MISMATCH')
    frame=pd.DataFrame(rows)
    atomic_csv(frame.drop(columns='inner_failure_categories'),OUT/'results/paired_outer_results.csv')
    for name,col in [('original','p_original'),('m4','p_m4')]:
        atomic_csv(frame[['outer_id','status','observed_D4',col]],OUT/f'results/{name}_pvalues.csv')
    paired=frame[frame.status=='PAIRED_VALID'];N=len(paired)
    tab=dict(n00=int(((paired.reject_original==False)&(paired.reject_m4==False)).sum()),
             n01=int(((paired.reject_original==False)&(paired.reject_m4==True)).sum()),
             n10=int(((paired.reject_original==True)&(paired.reject_m4==False)).sum()),
             n11=int(((paired.reject_original==True)&(paired.reject_m4==True)).sum()))
    if sum(tab.values())!=N: raise RuntimeError('PAIRED_TABLE_SUM')
    tab['paired_n']=N
    tab['difference']=(tab['n10']-tab['n01'])/N if N else None
    if N:
        rng=np.random.default_rng(read(OUT/'protocol/frozen_hash_registry.json')['paired_ci_seed'])
        delta=paired.reject_original.astype(int).to_numpy()-paired.reject_m4.astype(int).to_numpy()
        boot=np.array([delta[rng.integers(N,size=N)].mean() for _ in range(10000)])
        tab['difference_ci95']=np.quantile(boot,[.025,.975]).tolist()
        discordant=tab['n10']+tab['n01']
        tab['mcnemar_exact_p']=binomtest(tab['n10'],discordant,.5).pvalue if discordant else 1.
    else:tab['difference_ci95']=None;tab['mcnemar_exact_p']=None
    atomic_csv(pd.DataFrame([tab]),OUT/'results/paired_rejection_table.csv')
    summary={};tables=[]
    for name,pcol,rcol in [('Original','p_original','reject_original'),('M4-HS','p_m4','reject_m4')]:
        valid=frame[pcol].notna();n=int(valid.sum());k=int(frame.loc[valid,rcol].astype(bool).sum())
        estimate=k/n if n else None
        exact=binomtest(k,n).proportion_ci(confidence_level=.95,method='exact') if n else None
        record=dict(method=name,analyzable=n,reject=k,estimate=estimate,
                    wilson95=wilson(k,n),exact95=(exact.low,exact.high) if exact else None,
                    signed_deviation=estimate-ALPHA if n else None,
                    absolute_calibration_error=abs(estimate-ALPHA) if n else None)
        summary[name]=record;tables.append(record)
    atomic_csv(pd.DataFrame(tables),OUT/'results/type1_summary.csv')
    failures=Counter()
    for row in rows:
        if row['status']=='OUTER_ANALYSIS_UNAVAILABLE':failures[row.get('reason') or 'OUTER_UNKNOWN']+=1
        if row['status']=='METHOD_UNAVAILABLE':failures[row.get('reason') or 'METHOD_UNKNOWN']+=1
        for k,v in row.get('inner_failure_categories',{}).items():failures[k]+=v
    atomic_csv(pd.DataFrame([dict(category=k,count=v) for k,v in sorted(failures.items())],columns=['category','count']),OUT/'diagnostics/failure_summary.csv')
    total_inner_attempted=sum(r['inner_attempted'] for r in rows)
    total_inner_valid=sum(r['inner_valid'] for r in rows)
    audit=dict(status='PASS',registered_outer=R,completed_outer=len(rows),paired_analyzable=N,
               outer_unavailable=int((frame.status=='OUTER_ANALYSIS_UNAVAILABLE').sum()),
               method_unavailable=int((frame.status=='METHOD_UNAVAILABLE').sum()),
               original_draws_planned=R*B,inner_draws_registered=len(inner_seeds),
               inner_draws_attempted=total_inner_attempted,inner_draws_valid=total_inner_valid,
               inner_draws_invalid=total_inner_attempted-total_inner_valid,
               missing_outer=0,duplicate_outer=0,extra_outer=0,replacement_outer=0,
               duplicate_seed=0,extra_seed=0,prior_seed_overlap=0,temp_outputs=0,
               protected=protected,no_real_heartsteps_p_used=True)
    atomic_json(OUT/'diagnostics/final_integrity_audit.json',audit)
    if all(x['estimate'] is not None for x in summary.values()):plots(frame,summary,tab)
    result=dict(design='R=200 paired stationary-null HeartSteps-like outers',primary='D4',alpha=ALPHA,
                outer_analyzable=int(frame.observed_D4.notna().sum()),paired_analyzable=N,
                Original=summary['Original'],M4_HS=summary['M4-HS'],paired=tab,
                failures=dict(failures),integrity=audit)
    atomic_json(OUT/'results/final_result.json',result)
    def pct(x):return 'unavailable' if x is None else f'{100*x:.2f}%'
    lines=['# Original versus M4-HS v0: paired Type-I results','',
           'Frozen HeartSteps-like stationary-null DGP; R=200; D4 primary; alpha=0.05.','',
           f"Completed outers: {len(rows)}/200; observed-unavailable: {audit['outer_unavailable']}; paired analyzable: {N}.",'',
           '| Method | Reject / analyzable | Type-I | Wilson 95% CI | Exact 95% CI | ACE |',
           '|---|---:|---:|---:|---:|---:|']
    for name in ('Original','M4-HS'):
        s=summary[name];wi=s['wilson95'];ex=s['exact95']
        lines.append(f"| {name} | {s['reject']}/{s['analyzable']} | {pct(s['estimate'])} | {pct(wi[0])}–{pct(wi[1])} | {pct(ex[0])}–{pct(ex[1])} | {pct(s['absolute_calibration_error'])} |")
    lines += ['',f"Paired table: n00={tab['n00']}, n01={tab['n01']}, n10={tab['n10']}, n11={tab['n11']}.",
              f"Original-minus-M4 paired difference: {pct(tab['difference'])}; 95% paired bootstrap CI: {pct(tab['difference_ci95'][0])}–{pct(tab['difference_ci95'][1])}; exact McNemar p={tab['mcnemar_exact_p']}.",'',
              f"Inner M4: {total_inner_valid}/{total_inner_attempted} valid attempts; failures and unavailable outers are never coded as non-rejections.",'',
              'This comparison tests only the frozen HeartSteps-like fitted stationary generator. The augmented scheduled generator does not prove that the four-state embedded Q is exactly stationary/Markov. No universal superiority or full misspecification robustness follows.']
    (OUT/'ORIGINAL_VS_M4_TYPE1_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    summary_txt=(f"200个outer已完成；observed不可分析{audit['outer_unavailable']}个；共同可分析{N}个。\n"
                 f"Original：{summary['Original']['reject']}/{summary['Original']['analyzable']}，Type-I={pct(summary['Original']['estimate'])}。\n"
                 f"M4-HS：{summary['M4-HS']['reject']}/{summary['M4-HS']['analyzable']}，Type-I={pct(summary['M4-HS']['estimate'])}。\n"
                 f"配对n00/n01/n10/n11={tab['n00']}/{tab['n01']}/{tab['n10']}/{tab['n11']}。结论限于冻结的HeartSteps-like DGP。\n")
    (OUT/'ORIGINAL_VS_M4_TYPE1_SUMMARY.txt').write_text(summary_txt,encoding='utf-8')
    paths=[p for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json']
    atomic_json(OUT/'manifest.json',dict(status='COMPLETE',files={str(p.relative_to(OUT)).replace('\\','/'):digest(p) for p in paths},
                                         completed_outer=R,inner_draws_attempted=total_inner_attempted,
                                         no_next_stage=True))
    print(summary_txt)

