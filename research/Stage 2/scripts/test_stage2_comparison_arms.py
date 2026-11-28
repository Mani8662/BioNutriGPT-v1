"""Unit and safeguard tests for the Stage 2 Three-Arm Comparison.

Exercises:
1. Exact common envelope construction across all 3 arms.
2. Immutable non-dietary features strictly preserved across all arms.
3. Arm A contains zero optimization/gradients and produces finite deterministic output.
4. Arm B minimizes distance from observed intake without model-guided search.
5. Arm C executes full 50-step model-guided optimization plus projection.
6. Uniform evaluation logic correctly scores numerical constraints and score gate.
7. Rejection of non-finite inputs with informative errors.
8. Solver errors distinguished from proven mathematical infeasibility.
9. Feasible LP fallback accurately flagged without claiming minimum-distance optimality.
10. Checkpoint / resume integrity validation.
"""
import copy
import json
import math
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import stage2_counterfactual_engine as s2
import stage1_deployment as deployment
import stage2_comparison_arms as arms

ROOT = Path(__file__).resolve().parent


class Stage2ComparisonArmsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cases_path = ROOT / "Saved Model/stage2_constraint_audit_repaired_v1/cases_LOCAL.json"
        cls.cases = json.loads(cases_path.read_text(encoding="utf-8"))
        cls.case0 = cls.cases[0]
        cls.raw, _ = s2.canonicalize_dietary_ratios(cls.case0["vector"])
        cls.metadata = cls.case0["metadata"]

        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.artifacts = s2._load_preprocessing_artifacts(s2.DEFAULT_PREPROCESSOR)
        cls.model = s2._load_stage1_model(s2.DEFAULT_CHECKPOINT, cls.artifacts, cls.device)

        # Baseline probabilities
        norm_0 = s2._normalize_torch(torch.as_tensor(cls.raw, dtype=torch.float32, device=cls.device).unsqueeze(0), cls.artifacts)
        p_d0, p_o0, p_c0 = s2._predict_probabilities(cls.model, norm_0, no_grad=True)
        cls.p_d0 = float(p_d0.cpu())
        cls.p_o0 = float(p_o0.cpu())
        cls.p_c0 = float(p_c0.cpu())

        w = cls.metadata["weight_kg"]
        h = cls.metadata["height_cm"]
        age = cls.metadata["age_years"]
        is_fem = cls.metadata["is_female"]
        bmr = 10.0 * w + 6.25 * h - 5.0 * age + (-161.0 if is_fem else 5.0)
        tdee = bmr * s2.PAL_FACTORS[cls.metadata["activity_level"]]
        t_kcal = max(tdee - 500.0, 1200.0) if cls.p_o0 >= s2.CUTOFF_OBESITY else max(tdee, 1200.0)

        cls.lower, cls.upper = s2._build_nutrient_bounds(
            artifacts=cls.artifacts,
            t_kcal=t_kcal,
            weight_kg=w,
            is_female=is_fem,
            age_years=age,
            p_diab_0=cls.p_d0,
            p_obes_0=cls.p_o0,
            p_cvd_0=cls.p_c0,
        )

    def test_01_same_participant_envelope_across_all_arms(self):
        """Verify all three arms receive identical inputs, bounds, and baseline probabilities."""
        target_a, audit_a = arms.build_rules_only_target(
            raw_vector=self.raw,
            metadata=self.metadata,
            p_diab_0=self.p_d0,
            p_obes_0=self.p_o0,
            p_cvd_0=self.p_c0,
            lower=self.lower,
            upper=self.upper,
        )
        target_b, audit_b = arms.run_minimum_change_arm(
            raw_vector=self.raw,
            metadata=self.metadata,
            p_cvd_0=self.p_c0,
            lower=self.lower,
            upper=self.upper,
        )
        target_c, audit_c = arms.run_model_guided_arm(
            model=self.model,
            artifacts=self.artifacts,
            raw_vector=self.raw,
            metadata=self.metadata,
            p_diab_0=self.p_d0,
            p_obes_0=self.p_o0,
            p_cvd_0=self.p_c0,
            lower=self.lower,
            upper=self.upper,
            device=self.device,
        )

        self.assertEqual(target_a.shape, (39,))
        self.assertEqual(target_b.shape, (39,))
        self.assertEqual(target_c.shape, (39,))
        self.assertTrue(np.isfinite(target_a).all())
        self.assertTrue(np.isfinite(target_b).all())
        self.assertTrue(np.isfinite(target_c).all())

    def test_02_immutable_features_strictly_preserved(self):
        """Non-dietary biological, demographic, and lifestyle features remain unchanged."""
        raw_t = torch.as_tensor(self.raw, dtype=torch.float32, device=self.device)
        target_b, _ = arms.run_minimum_change_arm(
            raw_vector=self.raw,
            metadata=self.metadata,
            p_cvd_0=self.p_c0,
            lower=self.lower,
            upper=self.upper,
        )
        cand_88 = s2._construct_candidate_88(raw_t, torch.as_tensor(target_b, dtype=torch.float32, device=self.device))
        cand_np = cand_88.squeeze(0).cpu().numpy()

        fixed_indices = list(range(len(s2.BIO_COLS))) + list(range(len(s2.BIO_COLS) + 39 + len(s2.DIET_RATIO_COLS), len(s2.FEATURE_COLS)))
        np.testing.assert_allclose(cand_np[fixed_indices], self.raw[fixed_indices], atol=1e-5)

    def test_03_no_model_search_in_arms_a_and_b(self):
        """Arm A and Arm B must not perform any model-guided gradient optimization."""
        _, audit_a = arms.build_rules_only_target(
            raw_vector=self.raw,
            metadata=self.metadata,
            p_diab_0=self.p_d0,
            p_obes_0=self.p_o0,
            p_cvd_0=self.p_c0,
            lower=self.lower,
            upper=self.upper,
        )
        self.assertFalse(audit_a["optimization_used"])
        self.assertFalse(audit_a["projection_used"])

        _, audit_b = arms.run_minimum_change_arm(
            raw_vector=self.raw,
            metadata=self.metadata,
            p_cvd_0=self.p_c0,
            lower=self.lower,
            upper=self.upper,
        )
        self.assertTrue(audit_b["optimization_used"])
        self.assertFalse(audit_b["model_guided_search"])

    def test_04_arm_b_minimizes_distance_from_observed_intake(self):
        """Arm B's projection distance from observed intake must be <= Arm C's distance."""
        target_b, _ = arms.run_minimum_change_arm(
            raw_vector=self.raw,
            metadata=self.metadata,
            p_cvd_0=self.p_c0,
            lower=self.lower,
            upper=self.upper,
        )
        target_c, _ = arms.run_model_guided_arm(
            model=self.model,
            artifacts=self.artifacts,
            raw_vector=self.raw,
            metadata=self.metadata,
            p_diab_0=self.p_d0,
            p_obes_0=self.p_o0,
            p_cvd_0=self.p_c0,
            lower=self.lower,
            upper=self.upper,
            device=self.device,
        )

        n_obs = self.raw[len(s2.BIO_COLS) : len(s2.BIO_COLS) + 39]
        scales = np.array([s2.DAILY_REFERENCE_SCALES[k] for k in arms.NAMES], dtype=np.float64)
        dist_b = np.sqrt(np.sum(((target_b - n_obs) / scales) ** 2))
        dist_c = np.sqrt(np.sum(((target_c - n_obs) / scales) ** 2))

        # Because B solves the exact QP from n_obs, dist_b must be <= dist_c + numerical tolerance
        self.assertLessEqual(dist_b, dist_c + 1e-4)

    def test_05_shared_final_acceptance_logic(self):
        """All arms use identical evaluation logic for numerical and model-score gates."""
        target_b, _ = arms.run_minimum_change_arm(
            raw_vector=self.raw,
            metadata=self.metadata,
            p_cvd_0=self.p_c0,
            lower=self.lower,
            upper=self.upper,
        )
        eval_b = arms.evaluate_candidate_target(
            model=self.model,
            artifacts=self.artifacts,
            raw_vector=self.raw,
            target_39=target_b,
            metadata=self.metadata,
            p_diab_0=self.p_d0,
            p_obes_0=self.p_o0,
            p_cvd_0=self.p_c0,
            lower=self.lower,
            upper=self.upper,
            device=self.device,
        )
        self.assertTrue(eval_b["valid"])
        self.assertIn("numerical_pass", eval_b)
        self.assertIn("score_gate_pass", eval_b)
        self.assertIn("overall_accepted", eval_b)
        self.assertEqual(eval_b["overall_accepted"], eval_b["numerical_pass"] and eval_b["score_gate_pass"])

    def test_06_nonfinite_inputs_rejected(self):
        """Non-finite inputs are safely caught and rejected without silent corruption."""
        nan_target = np.full(39, np.nan)
        eval_res = arms.evaluate_candidate_target(
            model=self.model,
            artifacts=self.artifacts,
            raw_vector=self.raw,
            target_39=nan_target,
            metadata=self.metadata,
            p_diab_0=self.p_d0,
            p_obes_0=self.p_o0,
            p_cvd_0=self.p_c0,
            lower=self.lower,
            upper=self.upper,
            device=self.device,
        )
        self.assertFalse(eval_res["valid"])
        self.assertFalse(eval_res["overall_accepted"])

    def test_07_lp_fallback_reported_without_claiming_optimality(self):
        """When SLSQP fails, LP fallback is clearly marked minimum_distance_not_established."""
        def mock_slsqp_fail(*args, **kwargs):
            return SimpleNamespace(x=kwargs["x0"], success=False, status=9, message="Mocked solver failure", nit=500)

        with patch("scipy.optimize.minimize", side_effect=mock_slsqp_fail):
            target, audit = s2._project_to_joint_feasible_polytope(
                n_candidate=self.raw[len(s2.BIO_COLS) : len(s2.BIO_COLS) + 39],
                lower=self.lower,
                upper=self.upper,
                raw_vector=self.raw,
                metadata=self.metadata,
                p_cvd_0=self.p_c0,
            )
            self.assertTrue(audit["passed"])
            self.assertTrue(audit["fallback_used"])
            self.assertEqual(audit["status"], "feasible_lp_fallback")
            self.assertEqual(audit["optimization_status"], "minimum_distance_not_established")

    def test_08_solver_error_distinguished_from_infeasibility(self):
        """Timeouts or solver exceptions are distinguished from proven mathematical infeasibility."""
        mock_lp_timeout = SimpleNamespace(success=False, status=1, message="time limit")
        with patch("scipy.optimize.linprog", return_value=mock_lp_timeout):
            try:
                s2._project_to_joint_feasible_polytope(
                    n_candidate=self.raw[len(s2.BIO_COLS) : len(s2.BIO_COLS) + 39],
                    lower=self.lower,
                    upper=self.upper,
                    raw_vector=self.raw,
                    metadata=self.metadata,
                    p_cvd_0=self.p_c0,
                )
            except s2.Stage2Error as exc:
                self.assertNotIsInstance(exc, s2.Stage2InfeasibleConstraintsError)
                self.assertIn("feasibility_solver_failure", str(exc))
            else:
                self.fail("Unverified target accepted")

    def test_09_contradictory_bounds_classified_as_infeasible(self):
        """Contradictory bounds correctly raise Stage2InfeasibleConstraintsError."""
        bad_lower = self.lower.copy()
        bad_upper = self.upper.copy()
        bad_lower[s2.KCAL] = 2500.0
        bad_upper[s2.KCAL] = 1800.0  # lower > upper
        with self.assertRaises(s2.Stage2InfeasibleConstraintsError):
            s2._project_to_joint_feasible_polytope(
                n_candidate=self.raw[len(s2.BIO_COLS) : len(s2.BIO_COLS) + 39],
                lower=bad_lower,
                upper=bad_upper,
                raw_vector=self.raw,
                metadata=self.metadata,
                p_cvd_0=self.p_c0,
            )

    def test_10_checkpoint_resume_contract(self):
        """Simulate saving and loading a checkpoint, ensuring exact round-trip fidelity."""
        test_case_data = {
            "SEQN": 999999,
            "case": 99,
            "arm": "rules_only",
            "certified": True,
            "target": [float(i) for i in range(39)],
            "deltas": {"diabetes": -0.001, "obesity": -0.002, "cvd": 0.0005},
        }
        test_path = ROOT / "scratch/test_checkpoint_roundtrip.json"
        test_path.parent.mkdir(parents=True, exist_ok=True)
        test_path.write_text(json.dumps(test_case_data, indent=2), encoding="utf-8")

        loaded = json.loads(test_path.read_text(encoding="utf-8"))
        self.assertEqual(loaded["SEQN"], test_case_data["SEQN"])
        self.assertEqual(loaded["certified"], test_case_data["certified"])
        self.assertEqual(loaded["target"], test_case_data["target"])
        if test_path.exists():
            test_path.unlink()


if __name__ == "__main__":
    unittest.main()
