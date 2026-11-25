"""Read-only replay of the existing 900 solves. Never invokes a meal solver."""
from pathlib import Path
from collections import Counter,defaultdict
import copy,hashlib,itertools,json,math,shutil
import numpy as np
from scipy.stats import binomtest
import stage3_milp_solver as s3
import stage2_counterfactual_engine as s2
import stage1_deployment as deployment
import nutrient_rule_policy as rules
from stage2_stage3_adapter import compute_translation_fingerprint
from stage2_comparison_checks_v2 import check as check_stage2

R=Path(__file__).resolve().parent
OLD=R/'Saved Model/stage2_stage3_translation_study_v1'
OUT=R/'Saved Model/stage2_stage3_translation_audit_v2'
UP=R/'Saved Model/stage2_comparison_study_v2'
ARMS=['rules_only','minimum_change','model_guided']
MODES=['pure_vegetarian','veg_nonveg_combo','pure_non_vegetarian']
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,value): (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')

def category_limits(mode):
    # Independent transcription of the frozen study's category policy.
    variety={'cereal_grain':(1,3),'vegetable_sabzi':(2,math.inf),'dairy':(1,math.inf),'dal_pulse':(1,2)}
    servings={'cereal_grain':(3,5),'vegetable_sabzi':(2,4),'dairy':(1,2),'dal_pulse':(2,3)}
    if mode=='veg_nonveg_combo':
        variety['non_veg_protein']=(1,2); servings['non_veg_protein']=(1,3); servings['dal_pulse']=(1,3)
    elif mode=='pure_non_vegetarian':
        variety.update(non_veg_protein=(1,3),dal_pulse=(0,2),dairy=(0,math.inf))
        servings.update(non_veg_protein=(2,4),dal_pulse=(0,3),dairy=(0,2))
    elif mode!='pure_vegetarian': raise ValueError(mode)
    return variety,servings

def audit_basket(record,inp,foods):
    """Recalculate every decision; never call production validator or saved replay."""
    lookup={f.food_id:f for f in foods}; totals=dict.fromkeys(s3.NUTRIENT_CODES,0.)
    errors=[]; seen=set(); counts=Counter(); active=Counter(); unknown=set(); mass=0.
    basket=record['food_basket']; env={x['nutrient']:x for x in inp['nutrient_constraint_envelope']}
    required=[k for k,x in env.items() if x['rule_provenance']['planning_role'] not in {'report_only','exposure_reporting'}]
    if not 4<=len(basket)<=8: errors.append('dish_count')
    for x in basket:
        fid=x['food_id'];q=x['quantity']
        if fid not in lookup or fid in seen or type(q) is not int or not 1<=q<=3:
            errors.append('invalid_food_or_quantity');continue
        seen.add(fid);f=lookup[fid]; counts[f.category]+=q;active[f.category]+=1
        if record['dietary_mode']=='pure_vegetarian' and f.dietary_type=='non_vegetarian': errors.append('dietary_exclusion')
        unknown.update(f.unknown_nutrients);mass+=q*f.serving_weight_g
        for k in totals: totals[k]+=q*f.nutrients_per_serving[k]
    variety,servings=category_limits(record['dietary_mode'])
    for k,(lo,hi) in variety.items():
        if not lo<=active[k]<=hi: errors.append('variety:'+k)
    for k,(lo,hi) in servings.items():
        if not lo<=counts[k]<=hi: errors.append('servings:'+k)
    for k in required:
        if k in unknown: errors.append('unknown_active:'+k)
        if not env[k]['min_bound']-1e-6<=totals[k]<=env[k]['max_bound']+1e-6: errors.append('bound:'+k)
    E=totals['DR1TKCAL']; target=inp['energy_prescription']['target_calories']
    if E<=0 or abs(E-target)/target*100>5+1e-9: errors.append('energy')
    for k,factor,lo,hi in [('DR1TPROT',4,10,35),('DR1TCARB',4,45,65),('DR1TTFAT',9,20,35)]:
        pct=totals[k]*factor/max(E,1e-12)*100
        if not lo-1e-9<=pct<=hi+1e-9: errors.append('macro:'+k)
    if totals['DR1TSODI']/(totals['DR1TPOTA']+1)>env['DR1TSODI']['baseline']/(env['DR1TPOTA']['baseline']+1)+1e-9: errors.append('NaK')
    if totals['DR1TFIBE']/max(E/1000,1e-12)<14-1e-9: errors.append('fiber_density')
    residual=0.
    if basket:
        supplied={x['nutrient']:x for x in record['delivered_nutrition']}
        if set(supplied)!=set(totals): errors.append('missing_nutrient_rows')
        for k,x in supplied.items():
            r=abs(float(x['delivered'])-totals[k]);residual=max(residual,r)
            if not math.isfinite(r) or r>1e-4: errors.append('accounting:'+k)
    passed=not errors
    distance=None
    if passed:
        distance=math.sqrt(sum(((totals[k]-env[k]['target'])/s2.DAILY_REFERENCE_SCALES[k])**2 for k in required))
    return {'passed':passed,'errors':errors,'totals':totals,'unknown':sorted(unknown),'active_codes':required,'active_distance':distance,'max_accounting_residual':residual,'mass_g':mass}

def bootstrap(diff):
    diff=np.asarray(diff,float);rng=np.random.default_rng(20261123)
    idx=rng.integers(0,len(diff),(10000,len(diff)))
    means=np.mean(diff[idx],axis=1)
    return {'n_participants':len(diff),'mean_difference':float(diff.mean()),'ci95':[float(x) for x in np.quantile(means,[.025,.975])]}

def main():
    OUT.mkdir(exist_ok=True); original_files=list(OLD.rglob('*.json'))+list(UP.joinpath('checkpoints').glob('*.json'))
    initial={str(p.relative_to(R)):sha(p) for p in original_files}
    existing=OUT/'input_manifest.json'
    if existing.exists(): assert read(existing)==initial,'Input evidence changed since first audit'
    else: save('input_manifest.json',initial)
    proto=read(OLD/'protocol.json')
    for f,h in proto['hashes'].items(): assert sha(R/f)==h,f
    assert sha(OLD/'cases_LOCAL.json')==proto['cases_sha256']
    cases=read(OLD/'cases_LOCAL.json');assert cases==read(UP/'cases_LOCAL.json')
    foods=s3.load_food_database(); artifacts=s2._load_preprocessing_artifacts(deployment.PREPROCESSOR)
    records=[read(p) for p in sorted((OLD/'checkpoints').glob('*.json'))]
    keyed={(r['case_idx'],r['arm'],r['dietary_mode']):r for r in records}
    assert len(records)==len(keyed)==900
    assert set(keyed)==set(itertools.product(range(100),ARMS,MODES))
    out=[]; inputs={}; mismatches=[];statuses=Counter();max_residual=0.;tests=[]
    for case in cases:
        i=case['case'];seq=case['SEQN'];m=case['metadata'];w=m['weight_kg']
        cps={a:read(UP/'checkpoints'/f'case_{i:03d}_{seq}_{a}.json') for a in ARMS}
        probs=cps[ARMS[0]]['baseline_probabilities'];obs=cps[ARMS[0]]['observed_nutrients']
        bmr=10*w+6.25*m['height_cm']-5*m['age_years']+(-161 if m['is_female'] else 5)
        tdee=bmr*s2.PAL_FACTORS[m['activity_level']]; kcal=max(tdee-(500 if probs['obesity']>=.5 else 0),1200)
        lo,hi=s2._build_nutrient_bounds(artifacts=artifacts,t_kcal=kcal,weight_kg=w,is_female=m['is_female'],age_years=m['age_years'],p_diab_0=probs['diabetes'],p_obes_0=probs['obesity'],p_cvd_0=probs['cvd'])
        for arm,cp in cps.items():
            assert cp['baseline_probabilities']==probs and cp['observed_nutrients']==obs
            stage2_num=check_stage2(cp['target_nutrients'],lo,hi,s3.NUTRIENT_CODES,dict(zip(s3.NUTRIENT_CODES,obs)),m,probs['cvd'])['joint']
            stage2_pass=stage2_num and all(cp['target_probabilities'][k]-probs[k]<=.0015 for k in probs)
            assert stage2_pass==cp['overall_accepted']
            env=[{'nutrient':k,'target':cp['target_nutrients'][j],'min_bound':float(lo[j]),'max_bound':float(hi[j]),'baseline':obs[j],'rule_provenance':rules.reviewed_rule_metadata(k,float(lo[j]),float(hi[j]))} for j,k in enumerate(s3.NUTRIENT_CODES)]
            inp={'patient_metadata':dict(m),'energy_prescription':{'BMR':float(bmr),'TDEE':float(tdee),'target_calories':float(kcal)},'nutrient_constraint_envelope':env}
            inputs[i,arm]=inp
            for mode in MODES:
                r=keyed[i,arm,mode]
                assert r['SEQN']==seq and r['stage2_accepted']==stage2_pass
                fp=compute_translation_fingerprint(i,seq,arm,mode,inp,proto['catalog_hash'])
                assert fp==r['fingerprint'],('input fingerprint',i,arm,mode)
                fresh=audit_basket(r,inp,foods); max_residual=max(max_residual,fresh['max_accounting_residual'])
                e2e=bool(stage2_pass and fresh['passed'])
                if fresh['passed']!=r['stage3_accepted'] or e2e!=r['end_to_end_accepted']: mismatches.append([i,arm,mode,fresh['errors']])
                code=r['solver_metadata'].get('solver_status_code')
                status=('solver_status_0' if code==0 else 'time_limit_with_verified_plan' if code==1 and fresh['passed'] else 'time_limit_without_verified_plan' if code==1 else 'solver_reported_infeasible' if code==2 else 'other')
                statuses[status]+=1
                out.append({'case':i,'arm':arm,'mode':mode,'stage2':bool(stage2_pass),'stage3':fresh['passed'],'end_to_end':e2e,'status':status,'runtime':r['solve_time_seconds'],'distance_active_stage2_scales':fresh['active_distance'],'active_codes':fresh['active_codes'],'unknown_codes':fresh['unknown'],'errors':fresh['errors'],'mass_g':fresh['mass_g'],'recomputed_totals':fresh['totals'],'targets':{x['nutrient']:x['target'] for x in env},'absolute_target_errors':{k:abs(fresh['totals'][k]-env[j]['target']) if fresh['passed'] and k not in fresh['unknown'] else None for j,k in enumerate(s3.NUTRIENT_CODES)},'units':{x['nutrient']:x['rule_provenance']['unit'] for x in env}})
                if not tests and fresh['passed']:
                    bad=copy.deepcopy(r);bad['food_basket'][0]['quantity']=4
                    assert not audit_basket(bad,inp,foods)['passed'];tests.append('invalid_quantity')
                    bad=copy.deepcopy(r);bad['delivered_nutrition'][0]['delivered']+=10
                    assert not audit_basket(bad,inp,foods)['passed'];tests.append('corrupt_totals')
                    badinp=copy.deepcopy(inp);badinp['nutrient_constraint_envelope'][0]['max_bound']=1
                    assert not audit_basket(r,badinp,foods)['passed'];tests.append('corrupt_bound')
                    bad=copy.deepcopy(r);bad['stage3_accepted']=False
                    assert audit_basket(bad,inp,foods)['passed'];tests.append('does_not_trust_saved_acceptance')
                    bad=copy.deepcopy(r);bad['food_basket'][0]['food_id']='not_a_food'
                    assert not audit_basket(bad,inp,foods)['passed'];tests.append('unknown_food')
    save('recomputed_records_LOCAL.json',out);save('acceptance_mismatches.json',mismatches)
    assert not mismatches, f'{len(mismatches)} acceptance differences; inspect before publishing'
    index={(x['case'],x['arm'],x['mode']):x for x in out}
    summaries={}
    for arm in ARMS:
        rs=[x for x in out if x['arm']==arm]; times=[x['runtime'] for x in rs]
        summaries[arm]={'attempts':300,'stage3':sum(x['stage3'] for x in rs),'end_to_end':sum(x['end_to_end'] for x in rs),'stage2_participants':sum(index[i,arm,MODES[0]]['stage2'] for i in range(100)),'runtime_median':float(np.median(times)),'runtime_p95':float(np.quantile(times,.95)),'runtime_sum':float(sum(times)),'by_mode':{m:{'n':100,'stage3':sum(index[i,arm,m]['stage3'] for i in range(100)),'end_to_end':sum(index[i,arm,m]['end_to_end'] for i in range(100))} for m in MODES}}
    contrasts=[]
    for a,b in [('model_guided','minimum_change'),('model_guided','rules_only'),('minimum_change','rules_only')]:
        c={'arm1':a,'arm2':b,'pooled':{},'per_mode':{},'distance':{}}
        for ep in ['stage3','end_to_end']:
            diffs=[np.mean([int(index[i,a,m][ep])-int(index[i,b,m][ep]) for m in MODES])*100 for i in range(100)]
            c['pooled'][ep]=bootstrap(diffs) # Clustered participant bootstrap; no invalid pooled binomial p.
        for mode in MODES:
            c['per_mode'][mode]={}
            for ep in ['stage3','end_to_end']:
                d=np.array([int(index[i,a,mode][ep])-int(index[i,b,mode][ep]) for i in range(100)])
                wins=int(sum(d==1));losses=int(sum(d==-1));p=float(binomtest(wins,wins+losses).pvalue) if wins+losses else 1.
                c['per_mode'][mode][ep]={**bootstrap(d*100),'wins':wins,'losses':losses,'exact_p':p}
        # Only jointly successful participant-mode cells enter distance contrasts.
        pdiffs=[];ncells=0
        for i in range(100):
            ds=[index[i,a,m]['distance_active_stage2_scales']-index[i,b,m]['distance_active_stage2_scales'] for m in MODES if index[i,a,m]['stage3'] and index[i,b,m]['stage3']]
            if ds: pdiffs.append(float(np.mean(ds)));ncells+=len(ds)
        c['distance']={**bootstrap(pdiffs),'n_joint_success_cells':ncells,'scope':'Participant mean over paired jointly successful modes; active nutrients only, fixed Stage 2 scales; post-hoc exploratory replacement metric'}
        contrasts.append(c)
    # Holm adjustment over nine per-mode arm comparisons, separately by binary endpoint.
    for ep in ['stage3','end_to_end']:
        cells=[c['per_mode'][m][ep] for c in contrasts for m in MODES]; order=sorted(range(9),key=lambda i:cells[i]['exact_p']);running=0.
        for rank,i in enumerate(order):
            running=max(running,min(1.,(9-rank)*cells[i]['exact_p']));cells[i]['holm_p_9']=running
    identical=sum(len({tuple(sorted((y['food_id'],y['quantity']) for y in keyed[i,a,m]['food_basket'])) for a in ARMS})==1 for i in range(100) for m in MODES)
    analysis={'version':'translation_audit_v2','participants':100,'records':900,'solver_reexecutions':0,'statuses':dict(statuses),'arm_summaries':summaries,'contrasts':contrasts,'identical_basket_groups':identical,'matched_basket_groups':300,'catalog_foods':len(foods),'active_nutrients':out[0]['active_codes'],'max_accounting_residual':max_residual,'corruption_tests_passed':tests,'amendment':'Retrospective reanalysis of existing records. No prospectively frozen primary mode; pooled intervals cluster by participant; per-mode tests exploratory with Holm correction. No pooled exact p. Original target metric withdrawn; new active-distance metric is exploratory. Input construction and catalog parsing share production helpers; constraint and statistical decisions are recomputed. No Stage 1 neural inference rerun.'}
    save('analysis_summary.json',analysis)
    assert initial=={str(p.relative_to(R)):sha(p) for p in original_files}
    save('replay_verification.json',{'passed':True,'fresh_constraint_checks':900,'input_fingerprints_matched':900,'max_accounting_residual':max_residual,'acceptance_mismatches':0,'original_artifacts_unchanged':True,'solver_calls':0,'audit_source_sha256':sha(Path(__file__)),'limitations':['Historical protocol was rewritable; this audit cannot retroactively prove pre-run freezing.','Solver status/gap are archived solver reports, not independent optimality or infeasibility proofs.','Food loader and Stage 2 bounds builder are shared dependencies.']})
    print(json.dumps({k:analysis[k] for k in ['statuses','identical_basket_groups','catalog_foods','max_accounting_residual','corruption_tests_passed']},indent=2))

if __name__=='__main__': main()
