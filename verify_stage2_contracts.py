"""Stage 2 input/consistency regression; no clinical or model efficacy claims."""
import json,unittest,sys
from pathlib import Path
import numpy as np
import stage2_counterfactual_engine as s2
from audit_stage2_current_v1 import bounds
sys.path.insert(0,str(Path(__file__).resolve().parent/'backend'))
from schemas import PatientInput,FullPipelineRequest
class Contracts(unittest.TestCase):
 def metadata(self,**kw):
  d=dict(weight_kg=70,height_cm=170,age_years=42,activity_level='sedentary',is_female=False);d.update(kw)
  return s2._validate_patient_metadata(d,np.zeros(len(s2.FEATURE_COLS)))
 def test_adult_boundary(self):
  for age in [0,12,17.999]:
   with self.assertRaises(ValueError):self.metadata(age_years=age)
  self.assertEqual(self.metadata(age_years=18)['age_years'],18)
 def test_unsupported_population(self):
  for key in ['renal_disease','pregnant','lactating']:
   with self.assertRaises(ValueError):self.metadata(**{key:True})
   with self.assertRaises(ValueError):self.metadata(**{key:'false'})
   self.assertFalse(self.metadata(**{key:False})['special_population_status'][key])
 def test_unknown_is_explicit(self):
  self.assertEqual(self.metadata()['special_population_status'],dict(renal_disease=None,pregnant=None,lactating=None))
 def test_protein_conflict(self):
  with self.assertRaisesRegex(s2.Stage2Error,'Conflicting protein'):bounds(weight_kg=25)
  bounds(weight_kg=28)
 def test_invalid_overrides(self):
  for value in [(-5,10),(20,10),(0,float('nan'))]:
   with self.assertRaises((s2.Stage2Error,ValueError)):bounds(artifacts={'nutrient_bounds':{'DR1TVB12':value}})
 def test_api_contract(self):
  for cls in [PatientInput,FullPipelineRequest]:
   with self.assertRaises(ValueError):cls(age_years=12)
   with self.assertRaises(ValueError):cls(pregnant='false')
   self.assertTrue(cls(age_years=18,pregnant=True).pregnant)
 def test_txt_population_flags(self):
  import pipeline_service as ps
  numbers=' '.join(['0']*87)
  _,meta=ps.parse_raw_text('# renal_disease=yes pregnant=no lactating=unknown\n'+numbers)
  self.assertEqual(meta,dict(renal_disease=True,pregnant=False,lactating=None))
  _,meta=ps.parse_raw_text(numbers)
  self.assertNotIn('pregnant',meta)
  for header in ['# pregnant=maybe','# pregnant=yes\n# pregnant=no','# pregnant=']:
   with self.assertRaises(ValueError):ps.parse_raw_text(header+'\n'+numbers)
  with self.assertRaises(ValueError):self.metadata(**{'pregnant':True})
 def test_ratio_canonicalization_is_idempotent_and_preserves_inputs(self):
  import pipeline_service as ps
  raw=np.array(ps.SAMPLE_PATIENT_VALUES,dtype=np.float32);original=raw.copy()
  fixed,audit=s2.canonicalize_dietary_ratios(raw)
  repeated,second=s2.canonicalize_dietary_ratios(fixed)
  np.testing.assert_array_equal(raw,original)
  np.testing.assert_array_equal(fixed,repeated)
  self.assertEqual(second['changed_fields'],[])
  begin=len(s2.BIO_COLS)+39;end=begin+len(s2.DIET_RATIO_COLS)
  np.testing.assert_array_equal(fixed[:begin],raw[:begin])
  np.testing.assert_array_equal(fixed[end:],raw[end:])
  self.assertTrue(audit['changed_fields'])
 def test_binding_changes_when_assets_change(self):
  import tempfile
  with tempfile.TemporaryDirectory() as temp:
   checkpoint=Path(temp)/'model';preprocessor=Path(temp)/'prep'
   checkpoint.write_bytes(b'one');preprocessor.write_bytes(b'prep')
   before=s2.model_binding(checkpoint,preprocessor)
   checkpoint.write_bytes(b'two');after=s2.model_binding(checkpoint,preprocessor)
   self.assertNotEqual(before['checkpoint_sha256'],after['checkpoint_sha256'])
   self.assertEqual(before['probability_transform'],'raw_sigmoid_no_calibration')
 def test_grid_preserves_nonconflicting_bounds(self):
  cases=json.loads(Path('Saved Model/stage2_current_audit_v1/boundary_grid.json').read_text())
  for row in cases:
   kw=dict(age_years=row['age'],is_female=row['female'],weight_kg=row['weight'],t_kcal=row['energy'])
   if row['protein_filter_conflict']:
    with self.assertRaisesRegex(s2.Stage2Error,'Conflicting protein'):bounds(**kw)
   else:
    low,high=bounds(**kw)
    self.assertTrue(np.isfinite(low).all() and (low<=high).all())
    self.assertAlmostEqual(float(low[s2.NUTRIENT_INDEX['DR1TPROT']]),row['protein_min'])
if __name__=='__main__':unittest.main()
