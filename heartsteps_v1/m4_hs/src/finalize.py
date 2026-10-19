"""One final descriptive report, integrity audit, and manifest; never launches computation."""
from __future__ import annotations
import json
import platform
import sys
import numpy as np
import pandas as pd
from .common import BASE,PARENT,ROOT,sha,write_json,protected_check
from .runner import verify_freeze,verify_checkpoint

def md(df):
    def val(x):
        if pd.isna(x):return 'NA'
        if isinstance(x,(float,np.floating)):return f'{x:.6g}'
        return str(x).replace('|',' / ')
    return '| '+' | '.join(map(str,df.columns))+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(val(x) for x in row)+' |' for row in df.itertuples(index=False,name=None))+'\n'

def main():
    assert (BASE/'results/COMPUTATION_COMPLETE.json').exists()
    verify_freeze();protected=protected_check()
    seeds=json.loads((BASE/'specification/seed_registry.json').read_text());audit={};allrows={}
    for stage in ['validation','pilot']:
        root=BASE/'results'/stage;files=sorted(root.glob('draw_*/checkpoint.json')) if root.exists() else []
        planned={s['draw_id']:s for s in seeds[stage]};rows=[];seen=[]
        for path in files:
            j=int(path.parent.name.split('_')[-1]);assert j in planned
            rows.append(verify_checkpoint(path,planned[j],stage));seen.append(j)
        assert len(seen)==len(set(seen))
        audit[stage]={'registered':len(planned),'completed':len(rows),'valid':sum(r['status']=='VALID' for r in rows),'invalid':sum(r['status']!='VALID' for r in rows),'unconsumed':len(planned)-len(rows),'duplicate':0,'extra':0,'unregistered':0}
        allrows[stage]=rows
    tmp=list(BASE.rglob('*.tmp'));empty=[str(p.relative_to(BASE)) for p in BASE.rglob('*') if p.is_file() and p.stat().st_size==0]
    assert not tmp
    status=json.loads((BASE/'results/generator_validation_status.json').read_text());pilot=json.loads((BASE/'results/pilot_status.json').read_text())
    assert audit['validation']['completed']==200
    if status['pilot_allowed']:assert audit['pilot']['completed']==30
    else:assert audit['pilot']['completed']==0
    audit.update(protected=protected,temporary_files=tmp,zero_byte_files=empty,python=sys.version,platform=platform.platform(),formal_calibration_run=False,p_values_computed=False)
    write_json(BASE/'diagnostics/final_integrity_audit.json',audit)
    comparison=pd.read_csv(BASE/'diagnostics/predictive_comparison.csv')
    checks=pd.read_csv(BASE/'diagnostics/generator_gate_checks.csv');support=pd.read_csv(BASE/'diagnostics/validation_support.csv')
    survival=support.groupby('candidate').agg(survival=('admissible','mean'),n=('admissible','size'),left_n_median=('left_n','median'),right_n_median=('right_n','median'),left_A1_median=('left_A1','median'),right_A1_median=('right_A1','median')).reset_index();survival['day']=survival.candidate/1440
    survival.to_csv(BASE/'diagnostics/candidate_support_survival.csv',index=False)
    fit=json.loads((BASE/'diagnostics/fitted_models.json').read_text());family=[]
    for name in ['pre','reward']:
        m=fit['models'][name]['positive']
        for fam,q in m['comparison'].items():family.append({'outcome':name,'family':fam,'selected':fam==m['family'],'median_log_error':q['median_log_error'],'coverage90':q['coverage90'],'mean_abs_log_error':q['mean_absolute_log_error'],'converged':q['converged']})
    family=pd.DataFrame(family);family.to_csv(BASE/'diagnostics/positive_model_comparison.csv',index=False)
    coverage=pd.read_csv(BASE/'diagnostics/patient_predictive_coverage.csv').groupby('metric').covered.mean().reset_index()
    runtime=pd.DataFrame([{'draw_id':r['seed']['draw_id'],'seconds':r['runtime_seconds'],'status':r['status']} for r in allrows['validation']]);runtime.to_csv(BASE/'diagnostics/validation_runtime.csv',index=False)
    failed=checks.loc[~checks['pass']]
    report=f'''# M4-HS v0 generator validation

## Decision

{status['gate']}. Registered diagnostic panels: {audit['validation']['completed']}/200. Numerically valid/rebuilt: {audit['validation']['valid']}; invalid: {audit['validation']['invalid']}. No FQI, CUSUM statistic, p-value or critical value was calculated on these 200 panels. Pilot eligibility: {status['pilot_allowed']}.

These are predeclared predictive feasibility criteria, not validated inferential tolerances. No specification or model was adjusted after panel generation. All old HeartSteps/core hashes remain unchanged.

## What is regenerated

Each draw starts with the same 37 patient IDs and 7,741 recorded scheduled opportunities inside their original administrative/42-day horizons. Patient-specific shrunken intercepts are held fixed. A 210-opportunity burn-in is followed by regenerated binary location, hurdle pre-steps, technical eligibility, availability, randomized send (0.6 when available), hurdle post-steps and persistent observation masks. Original slot sine/cosine is deterministic. Missing published data remain NA. The unchanged builder reconstructs segments and valid available-to-next-available transitions.

Unlike Locked M4's synthetic generator and Bridge v0's original real-data family, this adapter models HeartSteps counts, five-slot periodicity, randomized treatment, availability and recurrent missingness. It neither calls their incompatible generators nor changes their estimators. There is no first-gap permanent censoring and no segment-as-subject relabeling.

## Fitting diagnostics (selected before panels)

{md(family)}
Both candidate positive-count families were compared on deterministic held-out scheduled rows. The simplest adequate primary family wins even if the other has a slightly smaller error. Full coefficients, 37 patient effects per model, fitted probability quantiles and optimizer diagnostics are in `diagnostics/fitted_models.json` and `model_coefficients.csv`. The holdout is an interleaved predictive diagnostic, not an independent external validation and not guaranteed slot-balanced.

## Observed versus generated panels

Units: counts are counts; rates are proportions; reward/pre quantiles are ln(steps+0.5); intervals are minutes; missing-episode lengths are scheduled opportunities. Reported intervals are the distribution across 200 generated panels, not confidence intervals for treatment effects.

{md(comparison)}

## Frozen feasibility checks that failed

{md(failed) if len(failed) else 'None. All prespecified mandatory checks passed.'}

## Seven frozen candidates

{md(survival)}

Every candidate was screened with the unchanged primary support rule. The detailed 1,400 candidate records retain left/right N, A0, A1, active patients, largest A1 share and rejection reasons. Passing support is not evidence that the generating family is adequate.

## Patient heterogeneity

Fraction of the 37 observed patient values falling within that same patient's simulated 5th-95th percentile interval:

{md(coverage)}

All per-patient counts, segments, missing episodes/reentries, zero fractions and medians are retained in `validation_patients.csv`; slot/action-specific distributions in `validation_strata.csv`; per-patient interval coverage in `patient_predictive_coverage.csv`. No poorly reproduced patient or candidate was removed.

## Timing, missingness and stationarity limitations

The scheduled design and administrative horizon are conditioned upon, not simulated. Recorded slot/gap irregularities remain. A fixed 90-second action/context lag simplifies real acquisition timing. Technical, travel and acquisition anomalies are merged into one persistent binary process; their distinct causes are not identified. Separate measurement-mask regressions can miss shared nonwear episodes and correlations. Biological models use complete adjacent scheduled pairs; this is not guaranteed representative under informative missingness.

Conditional coefficients are constant and contain no study-day/candidate/break predictor. This does not prove that the four-dimensional pooled available-opportunity Q is exactly stationary: omitted lags, patient heterogeneity, availability selection, changing administrative composition and imperfect burn-in remain. No type-I calibration claim follows from this diagnostic experiment. Gamma remains unchanged per available opportunity.

## Integrity and runtime

Duplicate/extra/unregistered completed seeds: 0/0/0. Checkpoint digests and component seeds verified; no temporary outputs. Total draw CPU-wall sums: {runtime.seconds.sum():.1f} seconds; median per draw {runtime.seconds.median():.2f} seconds. Parallel sum is not elapsed job runtime. Engineering differential tests matched all original builder records and seven explicit adverse cases. No old result/source was overwritten.
'''
    (BASE/'M4_HS_GENERATOR_VALIDATION.md').write_text(report,encoding='utf-8')
    if not status['pilot_allowed']:
        narrative='The 30-draw pilot was not started because the frozen generator-validation gate failed. All 30 parent pilot seeds remain unused. Attempted=0; valid=0; invalid=0. Support rejection rate and FQI failure rate for the pilot are NOT ESTIMABLE (denominator zero), not 0%. There is no pilot D0-D4 distribution or calibration p-value.'
        reason='; '.join(failed['check'].tolist())
        summary=f'M4-HS v0 已实现，重新生成37患者的完整scheduled panel、availability、随机动作、步数和缺失/重返过程。\n200个生成检查已完成：{audit["validation"]["valid"]}个数值有效。\nGenerator gate: {status["gate"]}\n未通过检查：{reason}\n7个candidate最小support survival={survival.survival.min():.1%}。\n30-draw pilot未启动，attempted=0；不是0/30失败。原30个pilot seeds均未消耗。\n不建议进入B=199。没有调整模型、阈值、candidate、seed或旧observed结果。\n'
    else:
        narrative=f"Pilot gate: {pilot['gate']}. Attempted={pilot['attempted']}; valid={pilot['valid']}; invalid={pilot['invalid']}; support failures={pilot['support_failures']}; FQI/statistic failures={pilot['fqi_statistic_failures']}."
        srows=allrows['pilot'];pd.DataFrame([{'draw_id':r['seed']['draw_id'],'status':r['status'],'failure_category':r['failure_category'],'runtime_seconds':r['runtime_seconds']} for r in srows]).to_csv(BASE/'results/pilot_draw_summary.csv',index=False)
        ss=pd.DataFrame([dict(draw_id=r['seed']['draw_id'],**s) for r in srows for s in r.get('support',[])]);ss.to_csv(BASE/'results/pilot_support.csv',index=False)
        if (BASE/'results/pilot_D0_D4.csv').exists():
            values=pd.read_csv(BASE/'results/pilot_D0_D4.csv');narrative+='\n\n'+md(values[['D0','D1','D2','D3','D4']].describe().reset_index())
        summary=f'M4-HS v0：200个generator panels已完成；{status["gate"]}。\n37患者保持原ID；step/availability/missingness/reentry重生成。\n30-draw pilot: {pilot["valid"]}/{pilot["attempted"]}有效；{pilot["gate"]}。\n不自动运行B=199；本轮不提供正式p值。\n'
    pilotreport=f'''# M4-HS v0 pilot report

{narrative}

Generator decision: {status['gate']}. See the complete predeclared check table in `diagnostics/generator_gate_checks.csv`. No failed predictive metric was relaxed. No observed statistic, candidate, support rule, evaluation state or FQI implementation changed.

The original observed D4=496.96432888717885 and descriptive argmax at 12.6 elapsed days remain untouched. A support-feasible simulated panel is not enough to establish a scientifically adequate stationary null. No Bridge/M4 p-value or significance claim is available from this stage.

## Frozen analysis code reused

`heartsteps_v1/scripts/adapter.py:build` with isolated loader binding; `src/rs_cusum/support.py:screen_candidates`; conditionally `src/rs_cusum_phase2r/core.py:fit_candidate`, `src/rs_cusum_phase2rc/c0_engine.py:observed_arrays`, `src/rs_cusum_realdata_bridge_v0/runner.py:layers`, existing patient-balanced action features and boundary scaling. The generator alone is new. No synthetic/Ohio computation ran.

## Next-stage restriction

B=199 is not authorized or started. A failed generator gate requires a separately reviewed modeling decision, not silently fitting another family or rerunning selected seeds. A passed small pilot would justify discussing a separately frozen formal plan, not claim Type-I validity. No tuning, seed replacement, threshold reduction, clipping, candidate substitution or fallback occurred.
'''
    (BASE/'M4_HS_PILOT_REPORT.md').write_text(pilotreport,encoding='utf-8');(BASE/'M4_HS_PILOT_SUMMARY.txt').write_text(summary,encoding='utf-8')
    files={str(p.relative_to(BASE)).replace('\\','/'):sha(p) for p in sorted(BASE.rglob('*')) if p.is_file() and p.name not in ['manifest.json','hash_registry_final.json']}
    write_json(BASE/'hash_registry_final.json',{'files':files,'protected':protected})
    write_json(BASE/'manifest.json',{'scope':'M4-HS-v0','generator_gate':status['gate'],'pilot_gate':pilot['gate'],'validation':audit['validation'],'pilot':audit['pilot'],'files':files,'final_registry_sha256':sha(BASE/'hash_registry_final.json'),'formal_B199_started':False,'observed_recomputed':False,'old_files_changed':False,'next_stage':'STOP'})
    print(summary,flush=True)

if __name__=='__main__':main()
