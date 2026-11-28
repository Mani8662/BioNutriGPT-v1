"""Exercise real projection failure paths without changing model or clinical rules."""
import json,os,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import stage2_counterfactual_engine as s2

R=Path(__file__).resolve().parent

class SolverStatusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        b=R/'Saved Model/stage2_constraint_audit_repaired_v1'
        cls.case=json.loads((b/'cases_LOCAL.json').read_text())[0]
        cls.d=json.loads((b/'dossier_000_LOCAL.json').read_text())
        cls.n=np.array(json.loads((b/'case_000_LOCAL.json').read_text())['snapshots']['optimizer_clipped'])
        cls.raw,_=s2.canonicalize_dietary_ratios(cls.case['vector'])
        env=cls.d['nutrient_constraint_envelope']
        cls.low=np.array([x['min_bound'] for x in env]);cls.high=np.array([x['max_bound'] for x in env])

    def project(self):
        return s2._project_to_joint_feasible_polytope(self.n,self.low,self.high,self.raw,self.case['metadata'],self.d['patient_baseline']['probabilities']['cvd'])

    def test_nonconvergence_is_explicit_feasible_fallback(self):
        def failed(fun,x0,**kwargs):
            return SimpleNamespace(x=x0,success=False,status=9,message='iteration limit',nit=500)
        with patch('scipy.optimize.minimize',side_effect=failed):
            n,a=self.project()
        self.assertTrue(a['passed']);self.assertTrue(a['fallback_used'])
        self.assertEqual(a['status'],'feasible_lp_fallback')
        self.assertEqual(a['optimization_status'],'minimum_distance_not_established')
        self.assertLessEqual(a['max_linear_residual'],1e-6)

    def test_nonfinite_slsqp_result_not_accepted_as_projection(self):
        result=SimpleNamespace(x=np.full(39,np.nan),success=True,status=0,message='success',nit=1)
        with patch('scipy.optimize.minimize',return_value=result):n,a=self.project()
        self.assertTrue(np.isfinite(n).all());self.assertTrue(a['fallback_used'])

    def test_slsqp_exception_uses_verified_fallback(self):
        with patch('scipy.optimize.minimize',side_effect=RuntimeError('injected solver failure')):n,a=self.project()
        self.assertTrue(a['fallback_used']);self.assertIn('injected',a['slsqp_message'])

    def test_invalid_lp_witness_rejected(self):
        result=SimpleNamespace(x=np.full(39,np.nan),success=True,status=0,message='success')
        with patch('scipy.optimize.linprog',return_value=result):
            with self.assertRaisesRegex(s2.Stage2Error,'invalid witness'):self.project()

    def test_lp_failure_is_not_claimed_infeasible(self):
        result=SimpleNamespace(success=False,status=1,message='time limit')
        with patch('scipy.optimize.linprog',return_value=result):
            try:self.project()
            except s2.Stage2Error as exc:
                self.assertNotIsInstance(exc,s2.Stage2InfeasibleConstraintsError)
                self.assertIn('feasibility_solver_failure',str(exc))
            else:self.fail('Unverified target accepted')

    def test_lp_infeasibility_classified(self):
        result=SimpleNamespace(success=False,status=2,message='infeasible')
        with patch('scipy.optimize.linprog',return_value=result):
            with self.assertRaises(s2.Stage2InfeasibleConstraintsError):self.project()

if __name__=='__main__':unittest.main()
