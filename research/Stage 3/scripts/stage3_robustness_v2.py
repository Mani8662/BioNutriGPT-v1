"""Versioned sequential robustness runner. No production changes; defaults to dry run."""
from __future__ import annotations
import argparse, copy, hashlib, json, math, os, platform, random, time
from pathlib import Path
import numpy as np
import scipy
from scipy.stats import binomtest
import run_stage3_robustness_v1 as legacy
import verify_stage3_robustness_v1 as audit

ROOT = Path(__file__).resolve().parent
DIR = ROOT / 'Saved Model/stage3_robustness_v2'
MODES = audit.MODES
class ValidationError(ValueError): pass
def require(ok, message):
    if not ok: raise ValidationError(message)
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x): return hashlib.sha256(json.dumps(x,sort_keys=True,allow_nan=False).encode()).hexdigest()
def write(p,x):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix('.tmp'); t.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8'); t.replace(p)
def catalog(): return {f['food_id']:f for f in read(legacy.CATALOG_PATH)}

def prepare():
    require(not (DIR/'input_manifest.json').exists(),'Frozen v2 exists; do not overwrite')
    scenarios=read(ROOT/'Saved Model/stage3_robustness_v1/scenario_manifest.json')['scenarios']
    foods=catalog()
    for sid,s in scenarios.items():
        if not s['replay_evaluation_only']: continue
        delta=.05 if '5pct' in sid else .10
        rng=random.Random(20261124)  # nested severity, identical direction per food
        pert={}
        for fid,f in sorted(foods.items()):
            # One factor per food preserves ALL within-food nutrient relationships,
            # including catalog energy/macronutrient ratios; not a lab error model.
            factor=1+delta*rng.uniform(-1,1)
            pert[fid]={k:(None if v is None or k in f.get('unknown_nutrients',[]) else v*factor)
                       for k,v in f['nutrients_per_serving'].items()}
        s['perturbed_nutrients']=pert
        s['description']='One fixed nested proportional composition perturbation per severity; arithmetic replay only.'
    write(DIR/'scenario_manifest.json',{'scenarios':scenarios})
    protocol={'version':2,'status':'prepared_not_full_run','date':'2026-11-24',
      'cohort':'100 reused budget-study participants; exploratory; case 0 used for development pilot',
      'solver_calls':3000,'replays_max':1200,'workers':1,'budget_seconds':5.,
      'timing':'Sequential calls; wall time includes wrapper. 15000 seconds budget sum plus overhead is not a hard wall-clock upper bound. No short empirical runtime promised.',
      'perturbation':'One fixed catalog draw per severity, shared across participants/objectives. Common per-food scaling preserves nominal relationships and unknowns. Inference is conditional on these fixed perturbations, not over nutrient-error distributions.',
      'dairy_policy':'Explicit scenario removes dairy and permits zero dairy using existing solver policy; distinct from unchanged-category availability stress.',
      'statistics':'Equal participant means across three modes; 10000 paired bootstrap samples seed 20261124. Target error only jointly valid. Runtime all attempts with measured runtime. Exact tests within scenario/mode; Holm across 21 availability/compliance contrasts. Exploratory unadjusted intervals.',
      'independence':'Verifier recomputes arithmetic but shares category-policy helper and legacy audit constraints. Separate verification reproduces stored analyses, not independently implemented statistical formulas.',
      'hardware':platform.platform(),'cpu_count':os.cpu_count(),'numpy':np.__version__,'scipy':scipy.__version__}
    write(DIR/'protocol.json',protocol)
    paths=[ROOT/k for k in read(ROOT/'Saved Model/stage3_robustness_v1/input_manifest.json')]
    paths += [Path(__file__),DIR/'protocol.json',DIR/'scenario_manifest.json',ROOT/'test_stage3_robustness_v2.py']
    write(DIR/'input_manifest.json',{str(p.relative_to(ROOT)):sha(p) for p in paths})

def check_manifest():
    for name,h in read(DIR/'input_manifest.json').items():
        require((ROOT/name).is_file() and sha(ROOT/name)==h,'Bound input changed: '+name)

def tasks(indices=None):
    legacy.SCENARIO_MANIFEST_PATH=DIR/'scenario_manifest.json'
    legacy.CHECKPOINT_DIR=DIR/'unused_paths'
    a,b=legacy.build_workload_plan(indices)
    return a+b

def identity(t): return {'case':t[0],'SEQN':t[1],'mode':t[3],'scenario':t[4]}
def binding(t): return digest({'identity':identity(t),'scenario':t[5],'input':t[6],'manifest':sha(DIR/'input_manifest.json')})
def excluded(t): return sorted(set(t[5].get('excluded_food_ids',[]))|set(t[5].get('removed_food_ids_by_mode',{}).get(t[3],[])))
def assess(basket,t,foods):
    totals={}
    for code in legacy.s3.NUTRIENT_CODES:
        values=[]
        for item in basket:
            require(item['food_id'] in foods,'Unknown food')
            f=foods[item['food_id']]; v=f['nutrients_per_serving'].get(code)
            values.append(None if code in f.get('unknown_nutrients',[]) or v is None else float(v)*item['quantity'])
        totals[code]=None if any(v is None for v in values) else sum(values)
    try:
        valid,errors,_=audit.validate_plan_constraints(basket,t[6],foods,t[3],t[5],t[4])
    except audit.RobustnessValidationError as exc:
        valid,errors=False,[str(exc)]
    active={r['nutrient'] for r in t[6]['nutrient_constraint_envelope'] if r.get('rule_provenance',{}).get('planning_role') not in ('report_only','exposure_reporting')}
    if any(totals.get(k) is None for k in active): valid=False; errors.append('Unknown active nutrient')
    # Document first failing constraint from legacy audit, not exhaustive residuals.
    err=audit.recompute_target_error(totals,t[6]) if valid and all(v is not None for v in totals.values()) else None
    return {'valid':bool(valid),'first_failure':errors,'totals':totals,'target_error':err}

def perturbed_foods(t):
    foods=copy.deepcopy(catalog())
    for fid,f in foods.items():
        f['nutrients_per_serving']=t[5]['perturbed_nutrients'][fid]
        f['unknown_nutrients']=sorted(set(f.get('unknown_nutrients',[]))|{k for k,v in f['nutrients_per_serving'].items() if v is None})
    return foods
def path_for(base,t,v): return base/(f"{t[0]:03d}_{t[3]}_{t[4]}_{v}.json")

def validate_record(r,t,v,base):
    require(r['identity']==identity(t) and r['variant']==v,'Identity mismatch')
    require(r['binding']==binding(t),'Fingerprint mismatch')
    require(r['budget_seconds']==5.,'Budget mismatch')
    require(r['exclusions']==excluded(t),'Exclusions mismatch')
    if t[5]['solver_required']:
        measured=assess(r['basket'],t,catalog())
        require(r['audit']==measured,'Plan arithmetic/acceptance mismatch')
        require(type(r['accepted']) is bool and r['accepted']==measured['valid'],'Acceptance mismatch')
        require(math.isfinite(r['runtime']) and r['runtime']>=0,'Invalid runtime')
        if not r['accepted']: require(bool(r['failure']),'Missing failure classification')
        if r['basket']:
            saved={x['nutrient']:x['delivered'] for x in r['delivered']}
            require(set(saved)==set(measured['totals']),'Incomplete saved nutrient totals')
            for k,val in measured['totals'].items():
                if val is not None: require(math.isclose(saved[k],val,rel_tol=0,abs_tol=1e-4),'Saved nutrient mismatch')
    else:
        nominal=nominal_task(t)
        p=path_for(base,nominal,v); n=read(p); validate_record(n,nominal,v,base)
        require(r['nominal_hash']==sha(p),'Nominal source changed')
        measured=assess(n['basket'],t,perturbed_foods(t)) if n['accepted'] else None
        require(r['audit']==measured,'Perturbed arithmetic/constraint mismatch')
        require(r['accepted']==bool(measured and measured['valid']),'Replay compliance mismatch')
        require(r['nominal_accepted']==n['accepted'],'Nominal acceptance mismatch')

def nominal_task(t):
    return next(x for x in tasks([t[0]]) if x[3]==t[3] and x[4]=='nominal_baseline')

def execute(t,v,base):
    p=path_for(base,t,v)
    if p.exists():
        r=read(p); validate_record(r,t,v,base); return r
    r={'identity':identity(t),'variant':v,'binding':binding(t),'budget_seconds':5.,'exclusions':excluded(t)}
    if t[5]['solver_required']:
        started=time.perf_counter()
        try:
            out=legacy.rbs.solve_budget_meal_plan(stage2_input=t[6],food_db=legacy.s3.load_food_database(),time_limit_sec=5.,revision_spec={'dietary_mode':t[3],'excluded_food_ids':excluded(t)},objective_variant='current' if v=='a' else 'target_aware',target_weight=1.)
            r.update(runtime=time.perf_counter()-started,basket=out.solved_food_basket,delivered=out.delivered_nutrition,solver_metadata=out.solver_metadata,failure=out.failure,solver_accepted=out.accepted)
        except Exception as exc:
            r.update(runtime=time.perf_counter()-started,basket=[],delivered=[],solver_metadata={},failure={'code':'runtime_exception','type':type(exc).__name__,'message':str(exc)},solver_accepted=False)
        started=time.perf_counter(); r['audit']=assess(r['basket'],t,catalog()); r['validation_seconds']=time.perf_counter()-started
        r['accepted']=r['audit']['valid']
        if r['solver_accepted']!=r['accepted']:
            write(p.with_suffix('.quarantine'),r); raise ValidationError('Solver/audit disagreement; quarantined, no automatic retry')
    else:
        npath=path_for(base,nominal_task(t),v); n=read(npath); validate_record(n,nominal_task(t),v,base)
        r.update(nominal_hash=sha(npath),nominal_accepted=n['accepted'])
        r['audit']=assess(n['basket'],t,perturbed_foods(t)) if n['accepted'] else None
        r['accepted']=bool(r['audit'] and r['audit']['valid'])
    validate_record(r,t,v,base); write(p,r); return r

def bootstrap(values):
    if not values: return {'n':0,'mean':None,'ci95':None}
    arr=np.array(values,dtype=float); rng=np.random.default_rng(20261124)
    means=arr[rng.integers(len(arr),size=(10000,len(arr)))].mean(axis=1)
    return {'n':len(arr),'mean':float(arr.mean()),'ci95':np.percentile(means,[2.5,97.5]).tolist()}
def analyze(records):
    scenarios=sorted({r['identity']['scenario'] for r in records})
    cases=sorted({r['identity']['case'] for r in records})
    idx={(r['identity']['case'],r['identity']['mode'],r['identity']['scenario'],r['variant']):r for r in records}
    require(len(idx)==len(records),'Duplicate records')
    result={}; refs=[]
    for sid in scenarios:
        values={k:[] for k in ['accepted_pp','target_error_joint','runtime_all_attempts']}; tests={}
        for i in cases:
            local={k:[] for k in values}
            for mode in MODES:
                require((i,mode,sid,'a') in idx and (i,mode,sid,'b') in idx,'Missing matched record')
                a,b=idx[i,mode,sid,'a'],idx[i,mode,sid,'b']
                local['accepted_pp'].append(100*(int(b['accepted'])-int(a['accepted'])))
                if 'runtime' in a and 'runtime' in b: local['runtime_all_attempts'].append(b['runtime']-a['runtime'])
                if a['accepted'] and b['accepted'] and a['audit']['target_error'] is not None and b['audit']['target_error'] is not None:
                    local['target_error_joint'].append(b['audit']['target_error']-a['audit']['target_error'])
            for k in values:
                if local[k]: values[k].append(float(np.mean(local[k])))
        for mode in MODES:
            pairs=[(idx[i,mode,sid,'a']['accepted'],idx[i,mode,sid,'b']['accepted']) for i in cases]
            win=sum(b and not a for a,b in pairs); loss=sum(a and not b for a,b in pairs)
            tests[mode]={'wins':win,'losses':loss,'p':float(binomtest(win,win+loss).pvalue) if win+loss else 1.}
            refs.append(tests[mode])
        result[sid]={'contrasts':{k:bootstrap(v) for k,v in values.items()},'tests':tests,
                     'accepted':{v:sum(r['accepted'] for r in records if r['variant']==v and r['identity']['scenario']==sid) for v in ['a','b']}}
    running=0.
    for rank,r in enumerate(sorted(refs,key=lambda x:x['p'])):
        running=max(running,min(1.,r['p']*(len(refs)-rank))); r['holm_p']=running
    return result

def verify(base,ts):
    expected={path_for(base,t,v) for t in ts for v in ['a','b']}
    require(set(base.glob('*.json'))==expected,'Missing/extra checkpoint files')
    records=[]
    for t in ts:
        for v in ['a','b']:
            r=read(path_for(base,t,v)); validate_record(r,t,v,base); records.append(r)
    result=analyze(records)
    result_path=base.parent/'analysis.json'
    if result_path.exists(): require(read(result_path)==result,'Saved statistics mismatch')
    else: write(result_path,result)
    write(base.parent/'verification.json',{'status':'passed','records':len(records),'full_cohort':len(ts)==2100,'scope':'Catalog reconstruction plus deterministic statistical replay; shared audit policy and statistical implementation disclosed','checkpoint_hashes':{p.name:sha(p) for p in sorted(expected)},'analysis_sha256':sha(result_path)})

def main():
    p=argparse.ArgumentParser(); p.add_argument('--prepare',action='store_true'); p.add_argument('--pilot',action='store_true'); p.add_argument('--run-full',action='store_true'); p.add_argument('--verify',action='store_true')
    args=p.parse_args()
    if args.prepare: prepare(); return
    check_manifest(); ts=tasks([0] if args.pilot else None)
    base=DIR/('pilot' if args.pilot else 'full')/'checkpoints'
    if args.verify: verify(base,ts); print('Verification passed'); return
    if not(args.pilot or args.run_full): print(f'Dry run: {len(ts)*2} records; sequential; no solves'); return
    base.mkdir(parents=True,exist_ok=True)
    for i,t in enumerate(ts):
        for v in (['a','b'] if t[0]%2==0 else ['b','a']): execute(t,v,base)
        print(f'{i+1}/{len(ts)} paired tasks',flush=True)
    check_manifest(); verify(base,ts); print('Completed and verified',flush=True)
if __name__=='__main__': main()
