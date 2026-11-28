import ast,json,unittest
from pathlib import Path
import numpy as np
import stage2_counterfactual_engine as s
from stage2_comparison_checks_v2 import check
from analyze_stage2_comparison_study_v2 import paired_bootstrap_contrast
R=Path(__file__).resolve().parent
class CorrectionTests(unittest.TestCase):
    def setUp(self):
        p=R/'Saved Model/stage2_comparison_study_v1'
        self.case=json.loads((p/'cases_LOCAL.json').read_text())[49]
        self.rec=json.loads(next((p/'checkpoints').glob('case_049_*_minimum_change.json')).read_text())
        self.v=np.array(self.rec['target_nutrients']); self.base=dict(zip(s.DIET_RAW_COLS,self.rec['observed_nutrients']))
    def result(self):
        return check(self.v,np.zeros(39),np.ones(39)*1e9,s.DIET_RAW_COLS,self.base,self.case['metadata'],self.rec['baseline_probabilities']['cvd'])
    def test_rounding_boundary(self): self.assertTrue(self.result()['filter2_ok'])
    def test_real_shortfall_rejected(self):
        self.v[s.DIET_RAW_COLS.index('DR1TPROT')]-=.001
        self.assertFalse(self.result()['filter2_ok'])
    def test_nan_rejected(self):
        self.v[0]=np.nan
        with self.assertRaises(ValueError): self.result()
    def test_one_search_call(self):
        runner=ast.parse((R/'run_stage2_comparison_study_v2.py').read_text())
        arms=ast.parse((R/'stage2_comparison_arms_v2.py').read_text())
        count=lambda t:sum(isinstance(n,ast.Attribute) and n.attr=='_run_counterfactual_optimization' for n in ast.walk(t))
        self.assertEqual(count(runner),0); self.assertEqual(count(arms),1)
    def test_discordance_exact_p(self):
        a=np.array([1]*8+[0]+[0]*91); b=np.array([0]*8+[1]+[0]*91)
        self.assertAlmostEqual(paired_bootstrap_contrast(a,b)['two_sided_p'],.0390625)
    def test_no_discordance_p(self):
        self.assertEqual(paired_bootstrap_contrast(np.ones(10),np.ones(10))['two_sided_p'],1.)
    def test_constant_continuous_difference(self):
        res=paired_bootstrap_contrast(np.ones(100)*.25,np.zeros(100))
        self.assertEqual(res['ci_lower_pct'],25.); self.assertEqual(res['ci_upper_pct'],25.)
        self.assertIsNone(res['two_sided_p'])
if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CorrectionTests))
    (R/'Saved Model/stage2_comparison_study_v2/correction_tests.json').write_text(json.dumps({'passed':result.wasSuccessful(),'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors)}))
    raise SystemExit(not result.wasSuccessful())
