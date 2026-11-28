"""Unit tests verifying Stage 2 joint constraint projection and safeguards.

Validates that the repaired projection:
1. Leaves already-feasible candidates unchanged.
2. Projects energy-inconsistent candidates to feasible energy within bounds.
3. Prevents calorie inflation beyond the upper bound.
4. Prevents calorie deflation below the lower bound.
5. Explicitly rejects contradictory/infeasible constraint systems.
6. Enforces clinical ratio envelopes within component bounds.
7. Rejects non-finite inputs with informative errors.
8. Verifies rescoring through the repaired calibrated model.
9. Guarantees non-dietary features remain strictly immutable.
10. Rejects targets when the closed-loop score gate fails (delta p > 0.0015).
"""
import copy
import json
import math
import os
import unittest
from pathlib import Path

import numpy as np
import torch

import stage1_deployment as deployment
import stage2_counterfactual_engine as s2

ROOT = Path(__file__).resolve().parent


class Stage2JointRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = json.loads((ROOT / "Saved Model/stage2_constraint_audit_repaired_v1/cases_LOCAL.json").read_text(encoding="utf-8"))
        cls.case0 = cls.cases[0]
        cls.raw_vector, _ = s2.canonicalize_dietary_ratios(cls.case0["vector"])
        cls.metadata = cls.case0["metadata"]
        cls.artifacts = s2._load_preprocessing_artifacts(s2.DEFAULT_PREPROCESSOR)
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.model = s2._load_stage1_model(s2.DEFAULT_CHECKPOINT, cls.artifacts, cls.device)

        # Baseline probabilities
        norm_0 = s2._normalize_torch(torch.as_tensor(cls.raw_vector, dtype=torch.float32, device=cls.device).unsqueeze(0), cls.artifacts)
        p_d0, p_o0, p_c0 = s2._predict_probabilities(cls.model, norm_0, no_grad=True)
        cls.p_cvd_0 = float(p_c0.cpu())
        cls.p_diab_0 = float(p_d0.cpu())
        cls.p_obes_0 = float(p_o0.cpu())

        # Bounds for case 0
        weight_kg = cls.metadata["weight_kg"]
        height_cm = cls.metadata["height_cm"]
        age_years = cls.metadata["age_years"]
        is_female = cls.metadata["is_female"]
        bmr = 10.0 * weight_kg + 6.25 * height_cm - 5.0 * age_years + (-161.0 if is_female else 5.0)
        tdee = bmr * s2.PAL_FACTORS[cls.metadata["activity_level"]]
        t_kcal = max(tdee - 500.0, 1200.0) if cls.p_obes_0 >= s2.CUTOFF_OBESITY else max(tdee, 1200.0)
        cls.t_kcal = t_kcal

        cls.lower, cls.upper = s2._build_nutrient_bounds(
            artifacts=cls.artifacts,
            t_kcal=t_kcal,
            weight_kg=weight_kg,
            is_female=is_female,
            age_years=age_years,
            p_diab_0=cls.p_diab_0,
            p_obes_0=cls.p_obes_0,
            p_cvd_0=cls.p_cvd_0,
        )

    def test_01_feasible_target_already_satisfying_constraints_unchanged(self):
        """A candidate that is already strictly feasible must be returned unchanged."""
        low_tight, up_tight = s2._build_tightened_joint_bounds(self.lower, self.upper, self.metadata, self.p_cvd_0)
        A_ub, b_ub = s2._build_linear_constraint_matrices(self.raw_vector)

        # Use an LP solution as a known feasible target
        from scipy.optimize import linprog
        lp = linprog(np.zeros(39), A_ub=A_ub, b_ub=b_ub, bounds=list(zip(low_tight, up_tight)), method="highs")
        self.assertTrue(lp.success)
        feasible_target = lp.x

        projected, audit = s2._project_to_joint_feasible_polytope(
            n_candidate=feasible_target,
            lower=self.lower,
            upper=self.upper,
            raw_vector=self.raw_vector,
            metadata=self.metadata,
            p_cvd_0=self.p_cvd_0,
        )
        self.assertEqual(audit["status"], "already_feasible")
        self.assertEqual(audit["nit"], 0)
        self.assertAlmostEqual(audit["projection_distance_normalized"], 0.0, places=9)
        np.testing.assert_allclose(projected, feasible_target, atol=1e-12)

    def test_02_inconsistent_energy_projected_to_feasible_energy_within_bounds(self):
        """A candidate with inconsistent macro-energy arithmetic is projected to satisfy energy balance within bounds."""
        low_tight, up_tight = s2._build_tightened_joint_bounds(self.lower, self.upper, self.metadata, self.p_cvd_0)
        # Create an energy-inconsistent candidate by taking midpoint and distorting carbohydrates
        midpoint = (low_tight + up_tight) / 2.0
        # Compute exact energy of midpoint
        calc_e = 4.0 * midpoint[s2.PROT] + 4.0 * midpoint[s2.CARB] + 9.0 * midpoint[s2.TFAT] + 7.0 * midpoint[s2.ALCO]
        # Set KCAL to 0.70 * calc_e (30% error, clearly > 1%)
        midpoint[s2.KCAL] = np.clip(0.70 * calc_e, low_tight[s2.KCAL], up_tight[s2.KCAL])

        # Verify it violates energy consistency before projection
        rel_err_before = abs(calc_e - midpoint[s2.KCAL]) / midpoint[s2.KCAL]
        self.assertGreater(rel_err_before, 0.01)

        projected, audit = s2._project_to_joint_feasible_polytope(
            n_candidate=midpoint,
            lower=self.lower,
            upper=self.upper,
            raw_vector=self.raw_vector,
            metadata=self.metadata,
            p_cvd_0=self.p_cvd_0,
        )
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["status"], "projected_successfully")

        # Verify all bounds
        self.assertTrue(np.all(projected >= low_tight - 1e-6))
        self.assertTrue(np.all(projected <= up_tight + 1e-6))

        # Verify energy consistency after projection
        calc_e_after = 4.0 * projected[s2.PROT] + 4.0 * projected[s2.CARB] + 9.0 * projected[s2.TFAT] + 7.0 * projected[s2.ALCO]
        rel_err_after = abs(calc_e_after - projected[s2.KCAL]) / projected[s2.KCAL]
        self.assertLessEqual(rel_err_after, 0.01 + 1e-9)

    def test_03_calorie_correction_exceeding_upper_bound_constrained(self):
        """When macro energy sums to more than upper calorie bound, projected KCAL stays <= upper bound."""
        low_tight, up_tight = s2._build_tightened_joint_bounds(self.lower, self.upper, self.metadata, self.p_cvd_0)
        # Candidate at upper bounds for PROT, CARB, TFAT
        high_macro = low_tight.copy()
        high_macro[s2.PROT] = up_tight[s2.PROT]
        high_macro[s2.CARB] = up_tight[s2.CARB]
        high_macro[s2.TFAT] = up_tight[s2.TFAT]
        high_macro[s2.KCAL] = up_tight[s2.KCAL]

        macro_sum = 4.0 * high_macro[s2.PROT] + 4.0 * high_macro[s2.CARB] + 9.0 * high_macro[s2.TFAT]
        # In the unprojected sequential pipeline, Filter 1 would overwrite KCAL = macro_sum, exceeding up_tight[KCAL]
        self.assertGreater(macro_sum, up_tight[s2.KCAL])

        projected, audit = s2._project_to_joint_feasible_polytope(
            n_candidate=high_macro,
            lower=self.lower,
            upper=self.upper,
            raw_vector=self.raw_vector,
            metadata=self.metadata,
            p_cvd_0=self.p_cvd_0,
        )
        self.assertTrue(audit["passed"])
        # Projected KCAL must strictly respect upper bound
        self.assertLessEqual(projected[s2.KCAL], up_tight[s2.KCAL] + 1e-6)
        calc_e = 4.0 * projected[s2.PROT] + 4.0 * projected[s2.CARB] + 9.0 * projected[s2.TFAT] + 7.0 * projected[s2.ALCO]
        self.assertLessEqual(abs(calc_e - projected[s2.KCAL]) / projected[s2.KCAL], 0.01 + 1e-9)

    def test_04_calorie_correction_below_lower_bound_constrained(self):
        """When macro energy sums to less than lower calorie bound, projected KCAL stays >= lower bound."""
        low_tight, up_tight = s2._build_tightened_joint_bounds(self.lower, self.upper, self.metadata, self.p_cvd_0)
        low_macro = low_tight.copy()
        low_macro[s2.PROT] = low_tight[s2.PROT]
        low_macro[s2.CARB] = low_tight[s2.CARB]
        low_macro[s2.TFAT] = low_tight[s2.TFAT]
        low_macro[s2.KCAL] = low_tight[s2.KCAL]

        macro_sum = 4.0 * low_macro[s2.PROT] + 4.0 * low_macro[s2.CARB] + 9.0 * low_macro[s2.TFAT]
        self.assertLess(macro_sum, low_tight[s2.KCAL])

        projected, audit = s2._project_to_joint_feasible_polytope(
            n_candidate=low_macro,
            lower=self.lower,
            upper=self.upper,
            raw_vector=self.raw_vector,
            metadata=self.metadata,
            p_cvd_0=self.p_cvd_0,
        )
        self.assertTrue(audit["passed"])
        # Projected KCAL must strictly respect lower bound
        self.assertGreaterEqual(projected[s2.KCAL], low_tight[s2.KCAL] - 1e-6)
        calc_e = 4.0 * projected[s2.PROT] + 4.0 * projected[s2.CARB] + 9.0 * projected[s2.TFAT] + 7.0 * projected[s2.ALCO]
        self.assertLessEqual(abs(calc_e - projected[s2.KCAL]) / projected[s2.KCAL], 0.01 + 1e-9)

    def test_05_conflicting_bounds_raise_explicit_failure(self):
        """Contradictory bounds raise Stage2InfeasibleConstraintsError; no invalid target is emitted."""
        bad_lower = self.lower.copy()
        bad_upper = self.upper.copy()
        # Contradictory protein bounds: lower > upper
        bad_lower[s2.PROT] = 200.0
        bad_upper[s2.PROT] = 100.0

        with self.assertRaises(s2.Stage2InfeasibleConstraintsError):
            s2._project_to_joint_feasible_polytope(
                n_candidate=self.lower.copy(),
                lower=bad_lower,
                upper=bad_upper,
                raw_vector=self.raw_vector,
                metadata=self.metadata,
                p_cvd_0=self.p_cvd_0,
                raise_on_infeasible=True,
            )

        # With raise_on_infeasible=False, returns passed=False
        _, audit = s2._project_to_joint_feasible_polytope(
            n_candidate=self.lower.copy(),
            lower=bad_lower,
            upper=bad_upper,
            raw_vector=self.raw_vector,
            metadata=self.metadata,
            p_cvd_0=self.p_cvd_0,
            raise_on_infeasible=False,
        )
        self.assertFalse(audit["passed"])
        self.assertEqual(audit["status"], "infeasible_bounds")

    def test_06_ratio_violations_corrected_within_bounds(self):
        """A candidate that violates sugar/fiber ratio is projected to satisfy ratio within bounds."""
        low_tight, up_tight = s2._build_tightened_joint_bounds(self.lower, self.upper, self.metadata, self.p_cvd_0)
        from scipy.optimize import linprog
        A_ub, b_ub = s2._build_linear_constraint_matrices(self.raw_vector)
        lp = linprog(np.zeros(39), A_ub=A_ub, b_ub=b_ub, bounds=list(zip(low_tight, up_tight)), method="highs")
        self.assertTrue(lp.success)
        cand = lp.x.copy()

        # Artificially inflate sugar to violate sugar/fiber ratio
        cand[s2.SUGR] = up_tight[s2.SUGR]
        cand[s2.FIBE] = low_tight[s2.FIBE]
        base_sf = self.raw_vector[s2.FEATURE_COLS.index("SUGAR_TO_FIBER_RATIO")]
        a_sf = max(base_sf, 1.0)
        ratio_before = cand[s2.SUGR] / (cand[s2.FIBE] + 1.0)
        self.assertGreater(ratio_before, a_sf)

        projected, audit = s2._project_to_joint_feasible_polytope(
            n_candidate=cand,
            lower=self.lower,
            upper=self.upper,
            raw_vector=self.raw_vector,
            metadata=self.metadata,
            p_cvd_0=self.p_cvd_0,
        )
        self.assertTrue(audit["passed"])
        ratio_after = projected[s2.SUGR] / (projected[s2.FIBE] + 1.0)
        self.assertLessEqual(ratio_after, a_sf + 1e-6)
        self.assertTrue(np.all(projected >= low_tight - 1e-6))
        self.assertTrue(np.all(projected <= up_tight + 1e-6))

    def test_07_nonfinite_inputs_handled_cleanly(self):
        """Nonfinite candidate nutrients or bounds raise Stage2Error cleanly."""
        nan_candidate = self.lower.copy()
        nan_candidate[s2.KCAL] = float("nan")
        with self.assertRaises(s2.Stage2Error):
            s2._project_to_joint_feasible_polytope(
                n_candidate=nan_candidate,
                lower=self.lower,
                upper=self.upper,
                raw_vector=self.raw_vector,
                metadata=self.metadata,
                p_cvd_0=self.p_cvd_0,
            )

        inf_bounds = self.lower.copy()
        inf_bounds[s2.PROT] = float("inf")
        with self.assertRaises(s2.Stage2Error):
            s2._project_to_joint_feasible_polytope(
                n_candidate=self.lower.copy(),
                lower=inf_bounds,
                upper=self.upper,
                raw_vector=self.raw_vector,
                metadata=self.metadata,
                p_cvd_0=self.p_cvd_0,
            )

    def test_08_final_target_rescoring_uses_repaired_calibrated_model(self):
        """Verify that rescoring in repaired mode uses the calibrated Stage 1 wrapper."""
        self.assertTrue(deployment.REPAIRED)
        binding = s2.model_binding(s2.DEFAULT_CHECKPOINT, s2.DEFAULT_PREPROCESSOR)
        self.assertEqual(binding["release"], "repaired_v1")
        self.assertEqual(binding["probability_transform"], "reserved_set_logit_Platt")
        self.assertEqual(len(binding.get("calibrator_sha256", "")), 64)

    def test_09_non_dietary_features_remain_strictly_immutable(self):
        """Constructed 88-vector must preserve non-dietary features identically."""
        low_tight, up_tight = s2._build_tightened_joint_bounds(self.lower, self.upper, self.metadata, self.p_cvd_0)
        A_ub, b_ub = s2._build_linear_constraint_matrices(self.raw_vector)
        from scipy.optimize import linprog
        lp = linprog(np.zeros(39), A_ub=A_ub, b_ub=b_ub, bounds=list(zip(low_tight, up_tight)), method="highs")
        target_39 = lp.x

        raw_88_t = torch.as_tensor(self.raw_vector, dtype=torch.float32)
        target_39_t = torch.as_tensor(target_39, dtype=torch.float32)
        candidate_88 = s2._construct_candidate_88(raw_88_t, target_39_t).squeeze(0).numpy()

        bio_len = len(s2.BIO_COLS)
        diet_len = len(s2.DIET_RAW_COLS)
        ratio_len = len(s2.DIET_RATIO_COLS)
        other_start = bio_len + diet_len + ratio_len

        # Biological features (0..bio_len) immutable
        np.testing.assert_allclose(candidate_88[:bio_len], self.raw_vector[:bio_len], atol=1e-5)
        # Lifestyle and demographic features (other_start..88) immutable
        np.testing.assert_allclose(candidate_88[other_start:], self.raw_vector[other_start:], atol=1e-5)

    def test_10_rejection_when_score_gate_fails(self):
        """When delta p exceeds 0.0015 on any endpoint, case is rejected."""
        # Case 10 in the audit is known to fail the score gate (delta cvd > 0.0015)
        case10 = self.cases[10]
        raw10, _ = s2.canonicalize_dietary_ratios(case10["vector"])
        dossier10 = s2.generate_counterfactual_prescription(raw10, case10["metadata"])
        
        f4 = dossier10["audit_filters"]["filter_4_closed_loop_rescoring"]
        self.assertFalse(f4["passed"])
        self.assertFalse(dossier10["certified_for_stage3"])
        # Check that joint projection itself passed, but score gate correctly blocked certification
        self.assertTrue(dossier10["audit_projection"]["passed"])
        self.assertFalse(dossier10["certified_for_stage3"])


if __name__ == "__main__":
    unittest.main()
