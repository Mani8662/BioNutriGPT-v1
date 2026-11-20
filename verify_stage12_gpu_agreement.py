"""Read-only model agreement test; output isolated from production dossiers."""
import hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'backend'))
import pipeline_service as ps
import stage2_counterfactual_engine as s2
import infer_bionutrigpt_pipeline_Stage_1_and_2 as p12

def main():
 dest=ROOT/'Saved Model/stage12_agreement_v1';dest.mkdir(exist_ok=False)
 def write(name,obj): (dest/name).write_text(json.dumps(obj,indent=2),encoding='utf-8')
 write('protocol.json',{'scope':'One existing backend fixture; supplied ratios versus recomputed ratios; no population or clinical claim.','tolerance':1e-7,'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['stage2_counterfactual_engine.py','infer_bionutrigpt_pipeline_Stage_1_and_2.py','verify_stage12_gpu_agreement.py']}})
 raw=np.array(ps.SAMPLE_PATIENT_VALUES,dtype=np.float32);canonical,_=s2.canonicalize_dietary_ratios(raw)
 prior={key:os.environ.get(key) for key in ['BIONUTRIGPT_STAGE2_JSON','BIONUTRIGPT_STAGE1_CHECKPOINT','BIONUTRIGPT_PREPROCESSOR_ARTIFACT']}
 rows=[]
 try:
  os.environ['BIONUTRIGPT_STAGE1_CHECKPOINT']=str(s2.DEFAULT_CHECKPOINT)
  os.environ['BIONUTRIGPT_PREPROCESSOR_ARTIFACT']=str(s2.DEFAULT_PREPROCESSOR)
  for label,vector in [('supplied_ratios',raw),('canonical_ratios',canonical)]:
   start=time.perf_counter()
   screening,device=p12.run_stage1_diagnostics(vector,checkpoint_path=s2.DEFAULT_CHECKPOINT,preprocessor_path=s2.DEFAULT_PREPROCESSOR)
   os.environ['BIONUTRIGPT_STAGE2_JSON']=str(dest/(label+'_stage2.json'))
   planning=s2.generate_counterfactual_prescription(vector,dict(ps.SAMPLE_METADATA))
   write(label+'_stage1.json',screening)
   differences={k:abs(v-planning['patient_baseline']['probabilities'][k]) for k,v in screening['probabilities'].items()}
   row=dict(case=label,device=str(device),differences=differences,binding_matches=screening['model_binding']==planning['model_binding'],ratio_audit_matches=screening['derived_input_audit']==planning['derived_input_audit'],thresholds_match=all(screening['classifications'][k]['threshold']==planning['model_binding']['operating_cutoffs'][k] for k in differences),seconds=time.perf_counter()-start)
   row['passed']=max(differences.values())<=1e-7 and row['binding_matches'] and row['ratio_audit_matches'] and row['thresholds_match']
   rows.append(row);write('results.json',rows);print(json.dumps(row),flush=True)
 finally:
  for k,v in prior.items():
   if v is None:os.environ.pop(k,None)
   else:os.environ[k]=v
 assert all(r['passed'] for r in rows)
 left=json.loads((dest/'supplied_ratios_stage2.json').read_text());right=json.loads((dest/'canonical_ratios_stage2.json').read_text())
 assert left['nutrient_constraint_envelope']==right['nutrient_constraint_envelope']
 write('summary.json',{'passed':True,'cases':len(rows),'target_envelopes_identical':True,'scope':'Single fixture GPU agreement, not a model-validity/generalization test.'})
if __name__=='__main__':main()
