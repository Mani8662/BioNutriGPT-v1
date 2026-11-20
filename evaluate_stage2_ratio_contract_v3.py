"""Isolated production-artifact stability pilot. Not clinical efficacy evidence."""
import copy,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'backend'))
import pipeline_service as ps
import stage2_counterfactual_engine as s2

def main():
 dest=ROOT/'Saved Model/stage2_ratio_contract_v3';dest.mkdir(exist_ok=False)
 def write(name,obj): (dest/name).write_text(json.dumps(obj,indent=2,default=str),encoding='utf-8')
 paths=[s2.DEFAULT_CHECKPOINT,s2.DEFAULT_PREPROCESSOR,ROOT/'Saved Model/stage1_comparison_v2/bionutrigpt/trial_03/model.pt',ROOT/'Data/stage1_repaired_v2/preprocessing_artifacts.joblib',ROOT/'stage2_counterfactual_engine.py',Path(__file__)]
 hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
 write('binding.json',{'hashes':hashes,'device':'cuda' if torch.cuda.is_available() else 'cpu','cutoffs':[s2.CUTOFF_DIABETES,s2.CUTOFF_OBESITY,s2.CUTOFF_CVD],'calibration':'Production _predict_probabilities applies raw sigmoid, no repaired calibrator.','scope':'Default production artifacts, not repaired v2. Sample fixture, not a patient cohort.'})
 write('protocol.json',{'cases':['baseline','repeat'],'selection':'Existing backend sample with dietary ratios recomputed consistently in every case; no selection or retries','iterations_per_run':50,'claims':'Model-score behavior only; no health-effect inference.'})
 torch.set_num_threads(os.cpu_count() or 1)
 base=np.array(ps.SAMPLE_PATIENT_VALUES,dtype=np.float32)
 begin=len(s2.BIO_COLS);end=begin+39
 base[end:end+len(s2.DIET_RATIO_COLS)]=s2._compute_ratios_numpy(base[begin:end])
 results=[]
 prior={key:os.environ.get(key) for key in ['BIONUTRIGPT_STAGE1_CHECKPOINT','BIONUTRIGPT_PREPROCESSOR_ARTIFACT','BIONUTRIGPT_STAGE2_JSON']}
 try:
  os.environ['BIONUTRIGPT_STAGE1_CHECKPOINT']=str(s2.DEFAULT_CHECKPOINT)
  os.environ['BIONUTRIGPT_PREPROCESSOR_ARTIFACT']=str(s2.DEFAULT_PREPROCESSOR)
  for label in ['baseline','repeat']:
   vector=base.copy();meta=copy.deepcopy(ps.SAMPLE_METADATA)
   if label.startswith('weight'):meta['weight_kg']*=.99 if 'minus' in label else 1.01
   if label.startswith('diet'):
    begin=len(s2.BIO_COLS);end=begin+39
    vector[begin:end]*=.99 if 'minus' in label else 1.01
    vector[end:end+len(s2.DIET_RATIO_COLS)]=s2._compute_ratios_numpy(vector[begin:end])
   os.environ['BIONUTRIGPT_STAGE2_JSON']=str(dest/(label+'.json'))
   row={'case':label};start=time.perf_counter()
   try:
    dossier=s2.generate_counterfactual_prescription(vector,meta)
    target=np.array([x['target'] for x in dossier['nutrient_constraint_envelope']])
    candidate=s2._construct_candidate_88(torch.tensor(vector),torch.tensor(target,dtype=torch.float32)).detach().numpy()[0]
    start_diet=len(s2.BIO_COLS);end_mutable=start_diet+39+len(s2.DIET_RATIO_COLS)
    row.update(certified=dossier['certified_for_stage3'],immutable_unchanged=bool(np.array_equal(candidate[:start_diet],vector[:start_diet]) and np.array_equal(candidate[end_mutable:],vector[end_mutable:])),target=target.tolist(),bounds_violations=[x['nutrient'] for x in dossier['nutrient_constraint_envelope'] if x['target']<x['min_bound']-1e-4 or x['target']>x['max_bound']+1e-4],filters=dossier['audit_filters'])
   except Exception as exc:row['error']=repr(exc)
   row['seconds']=time.perf_counter()-start;results.append(row)
   write('results.json',results)
   print(json.dumps({k:v for k,v in row.items() if k in ['case','certified','immutable_unchanged','error','seconds','bounds_violations']}),flush=True)
 finally:
  for key,value in prior.items():
   if value is None:os.environ.pop(key,None)
   else:os.environ[key]=value
 if results and 'target' in results[0]:
  ref=np.array(results[0]['target'])
  for row in results:
   if 'target' in row:row['max_relative_target_change']=float(np.max(abs(np.array(row['target'])-ref)/np.maximum(abs(ref),1)))
 write('results.json',results)
if __name__=='__main__':main()
