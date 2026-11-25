from pathlib import Path
import json,hashlib
import pandas as pd,numpy as np
from stage2_research_rule_adapter import build_research_prescription,DIET_RAW_COLS
R=Path(__file__).resolve().parent
O=R/'Saved Model/stage2_research_adapter_pilot_v1'
def main():
 O.mkdir(exist_ok=False)
 protocol={'selection':'Five lowest SEQN values in repaired temporal cohort, chosen without optimizer outcomes','thresholds':[.05,.10,.20,.30,.50],'policy':'Exploratory fixed-grid engineering sensitivity; selected after earlier model evaluation, not clinically validated or preregistered. No preferred threshold selected.','arms':['rules_only','bionutrigpt','catboost','ft_transformer'],'activity':'sedentary fixed engineering scenario in all arms, not measured activity','scope':'Adapter connection pilot, no solver/counterfactual search, no plan-quality or clinical benefit claim'}
 (O/'protocol.json').write_text(json.dumps(protocol,indent=2))
 ids=pd.read_csv(R/'Data/stage1_repaired_v2/temporal_ml_ready.csv',usecols=['SEQN']).sort_values('SEQN').head(5).SEQN
 rawpath=R/'Data/NHANES_2017_2020_PrePandemic_Raw.csv';raw=pd.read_csv(rawpath,low_memory=False).set_index('SEQN');assert raw.index.is_unique
 predictions={};paths=[rawpath,R/'stage2_research_rule_adapter.py',Path(__file__)]
 for model in protocol['arms'][1:]:
  p=R/'Saved Model/stage1_comparison_v2'/model/'temporal_predictions_LOCAL.csv';paths.append(p);predictions[model]=pd.read_csv(p).set_index('SEQN');assert predictions[model].index.is_unique
 rows=[]
 for i in ids:
  x=raw.loc[i];cv=x[['MCQ160B','MCQ160C','MCQ160D','MCQ160E','MCQ160F']]
  diabetes=True if x.DIQ010==1 else False if x.DIQ010 in (2,3) else None
  cvd=True if cv.eq(1).any() else False if cv.eq(2).all() else None
  profile=dict(weight_kg=float(x.BMXWT),height_cm=float(x.BMXHT),age_years=float(x.RIDAGEYR),is_female=bool(x.RIAGENDR==2),activity_level='sedentary',has_diabetes=diabetes,has_cvd=cvd,renal_disease=None,pregnant=None,lactating=None,baseline_nutrients={k:float(x[k]) for k in DIET_RAW_COLS})
  base=build_research_prescription(profile);base.update(SEQN=float(i),threshold=None,raw_DIQ010=float(x.DIQ010));rows.append(base)
  for model,f in predictions.items():
   probs={'diabetes':float(f.loc[i,'TARGET_DIABETES_calibrated']),'cvd':float(f.loc[i,'TARGET_CVD_calibrated'])}
   for threshold in protocol['thresholds']:
    out=build_research_prescription(profile,arm=model,probabilities=probs,thresholds={'diabetes':threshold,'cvd':threshold});out.update(SEQN=float(i),threshold=threshold,raw_DIQ010=float(x.DIQ010))
    assert out['energy_prescription']==base['energy_prescription']
    for e in out['nutrient_constraint_envelope']:assert e['min_bound']<=e['max_bound'] and np.isfinite([e['min_bound'],e['max_bound'],e['baseline']]).all()
    rows.append(out)
 (O/'envelopes_LOCAL.json').write_text(json.dumps(rows,indent=2))
 assert len(rows)==80
 report={'pilot_passed':True,'participants':5,'envelopes':len(rows),'solver_runs':0,'clinical_unknowns_preserved':True,'production_modified':False,'hashes':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
 (O/'verification.json').write_text(json.dumps(report,indent=2));print('PASS: five participants, 80 research envelopes; no solver/model training or production mutation.')
if __name__=='__main__':main()
