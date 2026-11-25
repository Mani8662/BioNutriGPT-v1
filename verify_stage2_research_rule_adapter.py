import copy,itertools,json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import stage2_counterfactual_engine as legacy
from stage2_research_rule_adapter import build_flag_bounds,build_research_prescription

def main():
 base=dict(weight_kg=75.,height_cm=170.,age_years=50.,is_female=False,activity_level='sedentary',has_diabetes=False,has_cvd=False,renal_disease=None,pregnant=None,lactating=None)
 for d,o,c in itertools.product([False,True],repeat=3):
  kw=dict(artifacts={},t_kcal=1800.,weight_kg=75.,is_female=False,age_years=50.)
  a=build_flag_bounds(**kw,diabetes_rule=d,obesity_rule=o,cvd_rule=c)
  b=legacy._build_nutrient_bounds(**kw,p_diab_0=float(d),p_obes_0=float(o),p_cvd_0=float(c))
  for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
 with patch.object(legacy,'_load_stage1_model',side_effect=AssertionError('Unexpected model loading')),patch.object(legacy,'_predict_probabilities',side_effect=AssertionError('Unexpected inference')):
  before=copy.deepcopy(base);plain=build_research_prescription(base);assert base==before
  known=dict(base,has_diabetes=True,has_cvd=True)
  k=build_research_prescription(known,arm='bionutrigpt',probabilities={'diabetes':0.,'cvd':0.},thresholds={'diabetes':.2,'cvd':.2})
  assert all(r['selected_rule'] for r in k['rule_decisions'].values())
  predicted=build_research_prescription(base,arm='catboost',probabilities={'diabetes':.2,'cvd':.19,'obesity':1.},thresholds={'diabetes':.2,'cvd':.2})
  assert predicted['rule_decisions']['diabetes']['selected_rule'] and not predicted['rule_decisions']['cvd']['selected_rule']
  assert predicted['obesity_rule']==plain['obesity_rule'] and predicted['energy_prescription']==plain['energy_prescription']
  unknown=build_research_prescription(dict(base,has_diabetes=None));assert unknown['rule_decisions']['diabetes']['observed_history'] is None
 for kw in [dict(probabilities={}),dict(arm='catboost'),dict(arm='catboost',probabilities={'diabetes':float('nan'),'cvd':0},thresholds={'diabetes':.2,'cvd':.2})]:
  try:build_research_prescription(base,**kw)
  except (ValueError,KeyError):pass
  else:raise AssertionError('Invalid input accepted')
 for key,value in [('renal_disease',True),('pregnant',True),('lactating',True),('age_years',19),('weight_kg',0),('has_cvd','no')]:
  try:build_research_prescription(dict(base,**{key:value}))
  except ValueError:pass
  else:raise AssertionError('Exclusion not enforced')
 print('PASS: eight branch-parity combinations, no model calls, diagnosis precedence, measured BMI, threshold boundary, unknown history, immutability and invalid/excluded inputs.')
if __name__=='__main__':main()
