"""Paired engineering pilot, not clinical utility or full cohort evidence."""
import os
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import json,copy,time,hashlib
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import stage3_milp_solver as s3
R=Path(__file__).resolve().parent
O=R/'Saved Model/stage1_downstream_optimizer_pilot_v1'
def replay(out,foods,env):
 if not out.accepted:return {'has_plan':False,'passed':False}
 lookup={f.food_id:f for f in foods};tot=dict.fromkeys(s3.NUTRIENT_CODES,0.);errors=[];mass=0;counts={};active={}
 for item in out.solved_food_basket:
  q=item['quantity'];f=lookup[item['food_id']]
  if type(q)is not int or not 1<=q<=3:errors.append('portion')
  mass+=q*f.serving_weight_g;counts[f.category]=counts.get(f.category,0)+q;active[f.category]=active.get(f.category,0)+1
  for k in tot:tot[k]+=q*f.nutrients_per_serving[k]
 limits,serv=s3._category_rules_for_mode('veg_nonveg_combo')
 for k,(lo,hi) in serv.items():
  if not lo<=counts.get(k,0)<=hi:errors.append('serving:'+k)
 for k,(lo,hi) in limits.items():
  if not lo<=active.get(k,0)<=hi:errors.append('variety:'+k)
 if not 4<=len(out.solved_food_basket)<=8:errors.append('dish_count')
 for row in env['nutrient_constraint_envelope']:
  k=row['nutrient']
  if row['rule_provenance'].get('planning_role') in ['report_only','exposure_reporting']:continue
  if any(k in lookup[x['food_id']].unknown_nutrients for x in out.solved_food_basket):errors.append('unknown:'+k)
  if not row['min_bound']-1e-4<=tot[k]<=row['max_bound']+1e-4:errors.append('bound:'+k)
 kcal=tot['DR1TKCAL'];target=env['energy_prescription']['target_calories'];err=abs(kcal-target)/target
 if err>.05+1e-9:errors.append('energy')
 for k,factor,lo,hi in [('DR1TPROT',4,.1,.35),('DR1TCARB',4,.45,.65),('DR1TTFAT',9,.2,.35)]:
  if not lo-1e-9<=tot[k]*factor/kcal<=hi+1e-9:errors.append('macro:'+k)
 base={x['nutrient']:x['baseline'] for x in env['nutrient_constraint_envelope']}
 if tot['DR1TSODI']/(tot['DR1TPOTA']+1)>base['DR1TSODI']/(base['DR1TPOTA']+1)+1e-9:errors.append('sodium_potassium')
 if tot['DR1TFIBE']/(kcal/1000)<14-1e-9:errors.append('fiber_density')
 for row in out.delivered_nutrition:
  if abs(row['delivered']-tot[row['nutrient']])>1e-4:errors.append('accounting:'+row['nutrient'])
 return {'has_plan':True,'passed':not errors,'errors':errors,'totals':tot,'food_mass_g':mass,'dish_count':len(out.solved_food_basket),'calorie_error_fraction':err}
def worker(task):
 index,env,reference=task;foods=s3.load_food_database(str(R/'food_composition_cooked_sourced.json'))
 # Zero active slack is required by the production solver; targets fixed to reference.
 target={x['nutrient']:x['target'] for x in reference['nutrient_constraint_envelope']}
 for row in env['nutrient_constraint_envelope']:row['target']=target[row['nutrient']]
 start=time.perf_counter()
 try:
  out=s3.solve_meal_plan(env,foods,time_limit_sec=3.,revision_spec={'dietary_mode':'veg_nonveg_combo'})
  own=replay(out,foods,env);common=replay(out,foods,reference)
  result={'case':index,'SEQN':env['SEQN'],'arm':env['arm'],'threshold':env['threshold'],'accepted':out.accepted,'own_replay':own,'reference_replay':common,'output':out.to_dict(),'input':env,'seconds':time.perf_counter()-start}
 except Exception as exc:result={'case':index,'SEQN':env['SEQN'],'arm':env['arm'],'threshold':env['threshold'],'accepted':False,'error':repr(exc),'seconds':time.perf_counter()-start}
 (O/f'case_{index:03d}_LOCAL.json').write_text(json.dumps(result,indent=2,default=s3._json_safe));return {k:v for k,v in result.items() if k not in ['input','output']}
def main():
 O.mkdir(exist_ok=False)
 source=R/'Saved Model/stage2_research_adapter_pilot_v1/envelopes_LOCAL.json';rows=json.loads(source.read_text());refs={x['SEQN']:copy.deepcopy(x) for x in rows if x['arm']=='rules_only'}
 for ref in refs.values():
  for row in ref['nutrient_constraint_envelope']:
   row['target']=ref['energy_prescription']['target_calories'] if row['nutrient']=='DR1TKCAL' else min(max(row['baseline'],row['min_bound']),row['max_bound'])
 protocol={'n':5,'cases':80,'dietary_mode':'veg_nonveg_combo','time_limit_seconds':3,'workers':min(4,os.cpu_count() or 1),'thresholds':[.05,.1,.2,.3,.5],'targets':'Measured baseline clipped into rules-only bounds, same across paired arms; energy target identical. No counterfactual optimization.','objective':'Production calorie-deviation plus food-selection cost. Active slack fixed zero; any bound-derived slack penalty cannot contribute in accepted solves. No objective override.','scope':'Deterministic pilot only; no inferential clinical or population claims; no best-case replacement','hashes':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,R/'stage3_milp_solver.py',R/'food_composition_cooked_sourced.json',Path(__file__)]}}
 (O/'protocol.json').write_text(json.dumps(protocol,indent=2))
 tasks=[(i,copy.deepcopy(x),refs[x['SEQN']]) for i,x in enumerate(rows)];results=[]
 with ProcessPoolExecutor(max_workers=protocol['workers']) as pool:
  for f in as_completed([pool.submit(worker,t) for t in tasks]):
   results.append(f.result());print(f'Completed {len(results)}/80',flush=True)
 results.sort(key=lambda x:x['case']);(O/'summary_LOCAL.json').write_text(json.dumps(results,indent=2))
 assert len(results)==80
 for x in results:
  if x['accepted']:assert x['own_replay']['passed'],x
 (O/'verification.json').write_text(json.dumps({'complete':True,'cases':80,'accepted':sum(x['accepted'] for x in results),'accepted_own_replay_all_pass':True,'scope':'Implementation pilot only'},indent=2))
if __name__=='__main__':main()
