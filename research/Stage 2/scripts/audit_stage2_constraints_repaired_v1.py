"""Checkpointed observational audit of unchanged repaired Stage 2; no training."""
from pathlib import Path
from dataclasses import fields
from collections import Counter
import json, os, time, copy
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linprog
import nutrient_rule_policy as policy
import stage1_deployment as dep
import stage2_counterfactual_engine as s2
from prepare_stage1_repair_v2 import load

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'Saved Model/stage2_constraint_audit_repaired_v1'
NAMES=s2.DIET_RAW_COLS
IX={n:i for i,n in enumerate(NAMES)}

def write(path,obj):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)

def independently_check(n,low,high,raw,meta,cvd):
    n=np.asarray(n,dtype=float);v=dict(zip(NAMES,n));base=dict(zip(s2.FEATURE_COLS,raw))
    violations=[]
    for k,x,l,h in zip(NAMES,n,low,high):
        if x<l-1e-6 or x>h+1e-6:
            violations.append({'nutrient':k,'value':float(x),'lower':float(l),'upper':float(h),
                               'distance':float(max(l-x,x-h,0)),
                               'active':k not in policy.REPORT_ONLY | policy.EXPOSURE_REPORTING})
    energy=4*v['DR1TPROT']+4*v['DR1TCARB']+9*v['DR1TTFAT']+7*v['DR1TALCO']
    energy_error=abs(energy-v['DR1TKCAL'])/v['DR1TKCAL'] if v['DR1TKCAL']>0 else 1e99
    ratio_values=[v['DR1TSUGR']/(v['DR1TFIBE']+1),v['DR1TSODI']/(v['DR1TPOTA']+1),v['DR1TSFAT']/(v['DR1TMFAT']+v['DR1TPFAT']+.1)]
    ratio_limits=[max(base['SUGAR_TO_FIBER_RATIO'],1.),max(base['SODIUM_POTASSIUM_RATIO'],1.),max(base['SAT_TO_UNSAT_FAT_RATIO'],.4)]
    ratios_ok=all(a<=b+1e-6 for a,b in zip(ratio_values,ratio_limits))
    cvd_rule=cvd>=.5
    filter2=(v['DR1TKCAL']>=1200 and v['DR1TSODI']<=(1500 if cvd_rule else 2300) and v['DR1TPOTA']>=(3500 if cvd_rule else 2600) and .8<=v['DR1TPROT']/meta['weight_kg']<=2)
    return {'bound_violations':violations,'active_bounds_pass':not any(x['active'] for x in violations),
            'all_search_bounds_pass':not violations,'energy_error_fraction':float(energy_error),
            'energy_consistent':bool(energy_error<=.01+1e-9),'ratio_values':list(map(float,ratio_values)),
            'ratio_limits':list(map(float,ratio_limits)),'ratios_pass':bool(ratios_ok),'filter2_invariants_pass':bool(filter2),
            'joint_numeric_pass':bool(not violations and energy_error<=.01+1e-9 and ratios_ok and filter2)}

def feasibility(low,high,raw,meta,cvd):
    """Independent linear feasibility of envelope + energy + ratio/filter rules.

    Does not constrain model-score change, historical filter flags or meal foods.
    """
    A=[];b=[];base=dict(zip(s2.FEATURE_COLS,raw))
    def constraint(terms,rhs):
        row=np.zeros(39)
        for k,v in terms.items():row[IX[k]]=v
        A.append(row);b.append(rhs)
    constraint({'DR1TPROT':4,'DR1TCARB':4,'DR1TTFAT':9,'DR1TALCO':7,'DR1TKCAL':-1.01},0)
    constraint({'DR1TPROT':-4,'DR1TCARB':-4,'DR1TTFAT':-9,'DR1TALCO':-7,'DR1TKCAL':.99},0)
    a=max(base['SUGAR_TO_FIBER_RATIO'],1.);constraint({'DR1TSUGR':1,'DR1TFIBE':-a},a)
    a=max(base['SODIUM_POTASSIUM_RATIO'],1.);constraint({'DR1TSODI':1,'DR1TPOTA':-a},a)
    a=max(base['SAT_TO_UNSAT_FAT_RATIO'],.4);constraint({'DR1TSFAT':1,'DR1TMFAT':-a,'DR1TPFAT':-a},.1*a)
    lower=np.asarray(low,dtype=float).copy();upper=np.asarray(high,dtype=float).copy()
    lower[IX['DR1TKCAL']]=max(lower[IX['DR1TKCAL']],1200)
    lower[IX['DR1TPROT']]=max(lower[IX['DR1TPROT']],.8*meta['weight_kg'])
    upper[IX['DR1TPROT']]=min(upper[IX['DR1TPROT']],2*meta['weight_kg'])
    upper[IX['DR1TSODI']]=min(upper[IX['DR1TSODI']],1500 if cvd>=.5 else 2300)
    lower[IX['DR1TPOTA']]=max(lower[IX['DR1TPOTA']],3500 if cvd>=.5 else 2600)
    if (lower>upper).any():return {'status':'infeasible_bounds','feasible':False}
    solved=linprog(np.zeros(39),A_ub=A,b_ub=b,bounds=list(zip(lower,upper)),method='highs')
    result={'status':int(solved.status),'feasible':bool(solved.success),'message':solved.message}
    if solved.success:
        replay=independently_check(solved.x,low,high,raw,meta,cvd)
        result['witness']=solved.x.tolist();result['independent_witness_check']=replay
        # LP boundary feasibility uses standard numerical tolerance, not clinical tolerance.
        result['max_linear_residual']=float(max(0,np.max(np.asarray(A)@solved.x-np.asarray(b))))
        assert result['max_linear_residual']<=1e-6
        assert np.all(solved.x>=lower-1e-6) and np.all(solved.x<=upper+1e-6)
    return result

def select_cases():
    pp=load('stage2_audit_preprocessing','Data/preprocess_for_training.py')
    artifacts=s2._load_preprocessing_artifacts(dep.PREPROCESSOR)
    state=pp.PreprocessorState(**{f.name:artifacts[f.name] for f in fields(pp.PreprocessorState)})
    source=ROOT/'Data/stage1_repaired_v2/temporal_clean.csv'
    clean=pd.read_csv(source)
    inputs=[c for c in clean if c not in set(pp.META_COLS+pp.TARGET_COLS) and c not in pp.MODEL_EXCLUDED_RAW_FEATURES]
    flags=pp.compute_missingness_indicators(clean,inputs)
    rawsource=ROOT/'Data/NHANES_2017_2020_PrePandemic_Raw.csv'
    measured=pd.read_csv(rawsource,usecols=['SEQN','BMXWT','BMXHT','RIDAGEYR','RIAGENDR']).set_index('SEQN')
    assert measured.index.is_unique
    joined=clean[['SEQN']].join(measured,on='SEQN',validate='one_to_one')
    eligible=(flags.sum(axis=1)==0)&joined.BMXWT.between(30,180)&joined.BMXHT.between(120,210)
    eligible &= (joined.RIDAGEYR>=20)&joined.RIAGENDR.isin([1,2])
    chosen=clean[eligible].sort_values('SEQN').sample(n=32,random_state=20261122).sort_values('SEQN')
    cont=chosen[state.base_continuous_cols].astype(float)
    raw=pd.concat([cont,pp.add_clinical_features(cont),pp.encode_binary_and_nominal(chosen,state.binary_modes,state.nominal_modes),flags.loc[chosen.index]],axis=1)[list(s2.FEATURE_COLS)]
    cases=[]
    for i,row in chosen.iterrows():
        x=measured.loc[row.SEQN]
        cases.append({'case':len(cases),'SEQN':int(row.SEQN),'vector':raw.loc[i].tolist(),
          'metadata':{'weight_kg':float(x.BMXWT),'height_cm':float(x.BMXHT),'age_years':float(x.RIDAGEYR),'is_female':bool(x.RIAGENDR==2),'activity_level':'sedentary','renal_disease':None,'pregnant':None,'lactating':None}})
    return cases,int(eligible.sum()),[source,rawsource]

def run_case(case):
    snapshots={};low=None;high=None
    original={name:getattr(s2,name) for name in ['_run_counterfactual_optimization','_apply_filter_1_thermodynamic_balance','_apply_filter_2_iom_safety','_apply_filter_3_ratio_coherence']}
    def optimize(**kwargs):
        nonlocal low,high
        low=kwargs['lower'].copy();high=kwargs['upper'].copy()
        result=original['_run_counterfactual_optimization'](**kwargs)
        snapshots['optimizer_clipped']=result[0].copy();return result
    def filter_wrapper(name,label):
        def wrapped(n,*args,**kwargs):
            result=original[name](n,*args,**kwargs);snapshots[label]=n.copy();return result
        return wrapped
    s2._run_counterfactual_optimization=optimize
    for i in [1,2,3]:
        name=list(original)[i];setattr(s2,name,filter_wrapper(name,f'after_filter_{i}'))
    start=time.perf_counter()
    try:
        raw,_=s2.canonicalize_dietary_ratios(case['vector'])
        os.environ['BIONUTRIGPT_STAGE2_JSON']=str(OUT/f"dossier_{case['case']:03d}_LOCAL.json")
        dossier=s2.generate_counterfactual_prescription(raw,case['metadata'])
        p=dossier['patient_baseline']['probabilities'];cvd=p['cvd']
        checks={label:independently_check(v,low,high,raw,case['metadata'],cvd) for label,v in snapshots.items()}
        final=checks['after_filter_3']
        deltas=dossier['audit_filters']['filter_4_closed_loop_rescoring']
        closed=all(deltas['delta_'+k]<=.0015 for k in ['diabetes','obesity','cvd'])
        independent=final['joint_numeric_pass'] and closed
        return {'case':case['case'],'certified':dossier['certified_for_stage3'],'independent_joint_pass':independent,
                'false_accept':bool(dossier['certified_for_stage3'] and not independent),'snapshots':{k:v.tolist() for k,v in snapshots.items()},
                'checks':checks,'closed_loop_pass':closed,'linear_feasibility':feasibility(low,high,raw,case['metadata'],cvd),
                'seconds':time.perf_counter()-start,'error':None}
    except Exception as exc:
        return {'case':case['case'],'error':repr(exc),'seconds':time.perf_counter()-start,'certified':False}
    finally:
        for name,fn in original.items():setattr(s2,name,fn)

def main():
    assert dep.REPAIRED
    dep.verify_bundle();torch.set_num_threads(os.cpu_count() or 1)
    cases,eligible,paths=select_cases()
    paths += [Path(__file__),ROOT/'stage2_counterfactual_engine.py',ROOT/'stage1_deployment.py',ROOT/'nutrient_rule_policy.py',dep.BUNDLE/'manifest.json',dep.CHECKPOINT,dep.PREPROCESSOR,dep.BUNDLE/'calibrators.joblib',ROOT/'Data/preprocess_for_training.py']
    binding={str(p.relative_to(ROOT)):dep.sha(p) for p in paths}
    protocol={'version':'stage2_constraint_audit_repaired_v1','n':32,'eligible_complete_inputs':eligible,'seed':20261122,'selection':'Fixed random sample from complete repaired temporal inputs and plausible measured anthropometry, without selection on optimizer outcome','activity':'sedentary fixed engineering assumption, not measured activity','release':'repaired_v1','thresholds':dep.THRESHOLDS,'device':'cuda' if torch.cuda.is_available() else 'cpu','torch_threads':torch.get_num_threads(),'steps':50,'bound_tolerance':1e-6,'scope':'Diagnostic engineering pilot, unweighted; no clinical, population or Indian claims; no causal interpretation','linear_oracle':'Nutrient-space feasibility only; excludes model-score gate and food catalog','hashes':binding}
    OUT.mkdir(exist_ok=True)
    if (OUT/'protocol.json').exists():
        assert json.loads((OUT/'protocol.json').read_text())==protocol,'Changed run binding; use a new version'
        assert json.loads((OUT/'cases_LOCAL.json').read_text())==cases
    else:write(OUT/'protocol.json',protocol);write(OUT/'cases_LOCAL.json',cases)
    results=[]
    previous=os.environ.get('BIONUTRIGPT_STAGE2_JSON')
    try:
        for case in cases:
            path=OUT/f"case_{case['case']:03d}_LOCAL.json"
            result=json.loads(path.read_text()) if path.exists() else run_case(case)
            write(path,result);results.append(result)
            print(f"{len(results)}/32 certified={result['certified']} error={result['error']}",flush=True)
    finally:
        if previous is None:os.environ.pop('BIONUTRIGPT_STAGE2_JSON',None)
        else:os.environ['BIONUTRIGPT_STAGE2_JSON']=previous
    for name,h in binding.items():assert dep.sha(ROOT/name)==h,'Source changed during audit'
    valid=[r for r in results if not r['error']]
    summary={'completed_cases':len(results),'returned_dossiers':len(valid),'errors':len(results)-len(valid),'certified':sum(r['certified'] for r in results),'false_accepts':sum(r['false_accept'] for r in valid),
      'stage_bound_failures':{k:sum(not r['checks'][k]['all_search_bounds_pass'] for r in valid) for k in ['optimizer_clipped','after_filter_1','after_filter_2','after_filter_3']},
      'final_energy_failures':sum(not r['checks']['after_filter_3']['energy_consistent'] for r in valid),'final_ratio_failures':sum(not r['checks']['after_filter_3']['ratios_pass'] for r in valid),
      'closed_loop_failures':sum(not r['closed_loop_pass'] for r in valid),'linear_feasible':sum(r['linear_feasibility']['feasible'] for r in valid),
      'linear_feasible_but_rejected':sum(r['linear_feasibility']['feasible'] and not r['certified'] for r in valid),
      'final_violated_nutrients':dict(Counter(v['nutrient'] for r in valid for v in r['checks']['after_filter_3']['bound_violations'])),
      'runtime_seconds':sum(r['seconds'] for r in results),'source_hashes_unchanged':True}
    write(OUT/'summary.json',summary);print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
