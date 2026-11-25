"""Existing-record audit. No solver execution; shared catalog/bounds parsing disclosed."""
import json,hashlib,math,copy,itertools
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.stats import binomtest
import repair_translation_evidence_v2 as base
from run_stage3_target_aware_experiment_v1 import compute_experiment_fingerprint
R=Path(__file__).resolve().parent;OLD=R/'Saved Model/stage3_target_aware_experiment_v1';O=R/'Saved Model/stage3_target_aware_audit_v2';UP=base.UP
read=base.read;sha=base.sha;ARMS=base.ARMS;MODES=base.MODES
def save(n,x): (O/n).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def main():
 O.mkdir(exist_ok=True);files=list(OLD.rglob('*.json'))+list((UP/'checkpoints').glob('*.json'));manifest={str(p.relative_to(R)):sha(p) for p in files}
 if (O/'input_manifest.json').exists():assert read(O/'input_manifest.json')==manifest
 else:save('input_manifest.json',manifest)
 p=read(OLD/'protocol.json')
 for f,h in p['hashes'].items():assert sha(R/f)==h,f
 assert sha(OLD/'cases_LOCAL.json')==p['cases_sha256']
 cases=read(OLD/'cases_LOCAL.json');assert cases==read(UP/'cases_LOCAL.json')
 foods=base.s3.load_food_database();art=base.s2._load_preprocessing_artifacts(base.deployment.PREPROCESSOR)
 raw=[read(f) for f in (OLD/'checkpoints').glob('case_*.json')];ix={(r['case_idx'],r['arm'],r['dietary_mode']):r for r in raw}
 assert len(raw)==len(ix)==900 and set(ix)==set(itertools.product(range(100),ARMS,MODES))
 results=[];maxres=0.;maxerr=0.;tests=[];statuses={v:Counter() for v in ['variant_a','variant_b']}
 for case in cases:
  i=case['case'];seq=case['SEQN'];m=case['metadata'];w=m['weight_kg'];cps={a:read(UP/'checkpoints'/f'case_{i:03d}_{seq}_{a}.json') for a in ARMS};pr=cps[ARMS[0]]['baseline_probabilities'];obs=cps[ARMS[0]]['observed_nutrients']
  bmr=10*w+6.25*m['height_cm']-5*m['age_years']+(-161 if m['is_female'] else 5);tdee=bmr*base.s2.PAL_FACTORS[m['activity_level']];kcal=max(tdee-(500 if pr['obesity']>=.5 else 0),1200)
  lo,hi=base.s2._build_nutrient_bounds(artifacts=art,t_kcal=kcal,weight_kg=w,is_female=m['is_female'],age_years=m['age_years'],p_diab_0=pr['diabetes'],p_obes_0=pr['obesity'],p_cvd_0=pr['cvd'])
  for arm,cp in cps.items():
   env=[{'nutrient':k,'target':cp['target_nutrients'][j],'min_bound':float(lo[j]),'max_bound':float(hi[j]),'baseline':obs[j],'rule_provenance':base.rules.reviewed_rule_metadata(k,float(lo[j]),float(hi[j]))} for j,k in enumerate(base.s3.NUTRIENT_CODES)]
   inp={'patient_metadata':dict(m),'energy_prescription':{'BMR':float(bmr),'TDEE':float(tdee),'target_calories':float(kcal)},'nutrient_constraint_envelope':env};targets={x['nutrient']:x['target'] for x in env}
   for mode in MODES:
    r=ix[i,arm,mode];assert r['SEQN']==seq and r['stage2_accepted']==cp['overall_accepted'];assert compute_experiment_fingerprint(i,seq,arm,mode,inp,p['catalog_hash'])==r['fingerprint']
    for v in ['variant_a','variant_b']:
     z=r[v];audit=base.audit_basket({**z,'dietary_mode':mode},inp,foods);assert audit['passed']==z['accepted']==z['replay_passed'],(i,arm,mode,v,audit['errors']);e2e=bool(cp['overall_accepted'] and audit['passed']);assert e2e==z['end_to_end_accepted']
     maxres=max(maxres,audit['max_accounting_residual']);codes=[k for k in audit['active_codes'] if k!='DR1TKCAL'];l1=None
     if audit['passed']:
      l1=float(np.mean([abs(audit['totals'][k]-targets[k])/p['normalization_scales'][k] for k in codes]));maxerr=max(maxerr,abs(l1-z['target_attainment']['mean_normalized_l1_target_error']));assert maxerr<1e-6
      assert len(codes)==z['target_attainment']['target_nutrient_count']
      for k in codes: assert abs(abs(audit['totals'][k]-targets[k])-z['target_attainment']['per_nutrient_absolute_error'][k])<2e-6
      if not tests:
       bad=copy.deepcopy(z);bad['food_basket'][0]['quantity']=4;assert not base.audit_basket({**bad,'dietary_mode':mode},inp,foods)['passed'];tests.append('invalid_quantity')
       bad=copy.deepcopy(z);bad['delivered_nutrition'][0]['delivered']+=10;assert not base.audit_basket({**bad,'dietary_mode':mode},inp,foods)['passed'];tests.append('corrupted_totals')
       bad=copy.deepcopy(inp);bad['nutrient_constraint_envelope'][0]['max_bound']=1;assert not base.audit_basket({**z,'dietary_mode':mode},bad,foods)['passed'];tests.append('corrupted_bound')
       bad=copy.deepcopy(z);bad['accepted']=False;assert base.audit_basket({**bad,'dietary_mode':mode},inp,foods)['passed'];tests.append('saved_flag_not_trusted')
     statuses[v][z['solver_status']]+=1
     results.append({'case':i,'arm':arm,'mode':mode,'variant':v,'accepted':audit['passed'],'e2e':e2e,'l1':l1,'runtime':z['solve_time_seconds'],'basket':sorted((x['food_id'],x['quantity']) for x in z['food_basket']),'mass_g':audit['mass_g'],'totals':audit['totals'] if audit['passed'] else None,'target_codes':codes,'targets':targets})
 save('recomputed_LOCAL.json',results);idx={(x['case'],x['arm'],x['mode'],x['variant']):x for x in results}
 summaries={};contrasts=[]
 for arm in ARMS:
  summaries[arm]={v:{'n':300,'accepted':sum(x['accepted'] for x in results if x['arm']==arm and x['variant']==v),'e2e':sum(x['e2e'] for x in results if x['arm']==arm and x['variant']==v),'median_runtime':float(np.median([x['runtime'] for x in results if x['arm']==arm and x['variant']==v]))} for v in ['variant_a','variant_b']}
 specifications=[(a,'variant_b',a,'variant_a') for a in ARMS]+[('model_guided','variant_b','minimum_change','variant_b'),('model_guided','variant_b','rules_only','variant_b'),('minimum_change','variant_b','rules_only','variant_b')]
 for a,av,b,bv in specifications:
  c={'a':a,'av':av,'b':b,'bv':bv,'binary':{},'modes':{},'distance':{}};matched=[];am=[];bm=[];n=0
  for ep in ['accepted','e2e']:
   c['binary'][ep]=base.bootstrap([100*np.mean([int(idx[i,a,m,av][ep])-int(idx[i,b,m,bv][ep]) for m in MODES]) for i in range(100)])
  for mode in MODES:
   c['modes'][mode]={}
   for ep in ['accepted','e2e']:
    d=[int(idx[i,a,mode,av][ep])-int(idx[i,b,mode,bv][ep]) for i in range(100)];w=d.count(1);l=d.count(-1);pval=float(binomtest(w,w+l).pvalue) if w+l else 1.
    c['modes'][mode][ep]={'wins':w,'losses':l,'p':pval}
  for i in range(100):
   pairs=[(idx[i,a,m,av]['l1'],idx[i,b,m,bv]['l1']) for m in MODES if idx[i,a,m,av]['accepted'] and idx[i,b,m,bv]['accepted']]
   if pairs:
    aa=float(np.mean([x for x,y in pairs]));bb=float(np.mean([y for x,y in pairs]));am.append(aa);bm.append(bb);matched.append(aa-bb);n+=len(pairs)
  c['distance']={'difference':base.bootstrap(matched),'a_mean':base.bootstrap(am),'b_mean':base.bootstrap(bm),'joint_cells':n,'relative_change_pct':100*(np.mean(am)/np.mean(bm)-1),'estimator':'Equal participant weights; average only jointly successful matched modes within each participant'}
  contrasts.append(c)
 for ep in ['accepted','e2e']:
  cells=[c['modes'][m][ep] for c in contrasts for m in MODES];ordered=sorted(cells,key=lambda x:x['p']);prev=0
  for rank,x in enumerate(ordered):prev=max(prev,min(1,(len(cells)-rank)*x['p']));x['holm_p_18']=prev
 joint=[r for r in raw if r['variant_a']['accepted'] and r['variant_b']['accepted']];same=sum(idx[r['case_idx'],r['arm'],r['dietary_mode'],'variant_a']['basket']==idx[r['case_idx'],r['arm'],r['dietary_mode'],'variant_b']['basket'] for r in joint)
 analysis={'n_participants':100,'n_outputs':1800,'solver_reruns':0,'sample_reused':True,'summary':summaries,'contrasts':contrasts,'saved_status_labels':{v:dict(s) for v,s in statuses.items()},'joint_success_pairs':len(joint),'identical_joint_baskets':same,'different_joint_baskets':len(joint)-same,'max_accounting_residual':maxres,'max_saved_l1_error':maxerr,'tests_passed':tests,'limitations':['Raw solver codes/messages/incumbent flags not saved; labels cannot independently establish optimality or absence of incumbent.','Reused sample also appeared in development tests; exploratory evidence.','Protocol rewritable; cannot retrospectively prove frozen provenance.','Fixed A-then-B order; timing claims descriptive.','Holm adjustment retrospective over 18 mode-by-contrast tests per binary endpoint; intervals pointwise.','Shared food loader, Stage 2 bounds and previous independent basket replay; no production acceptance function or saved replay flags used.']}
 save('analysis_summary.json',analysis);assert manifest=={str(p.relative_to(R)):sha(p) for p in files};save('replay_verification.json',{'passed':True,'outputs':1800,'fingerprints':900,'source_hash':sha(Path(__file__)),'original_artifacts_unchanged':True,'solver_calls':0});print(json.dumps(summaries,indent=2))
if __name__=='__main__':main()
