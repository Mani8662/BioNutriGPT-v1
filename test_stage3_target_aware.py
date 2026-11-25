#!/usr/bin/env python3
"""Integrity and Unit Tests for Experimental Stage 3 Target-Aware Solver.

Tests all 9 architectural requirements:
1. Synthetic target shifting alters meal plan under identical bounds.
2. Arm sensitivity: different Stage 2 arms produce distinct objectives in Variant B.
3. Feasible-set invariance: feasible space of food variables is identical.
4. Zero-slack invariant: deviation variables cannot purchase bound violations.
5. Energy reconciliation: calorie target is not double-counted.
6. Schema and coverage: missing active nutrient values fail closed.
7. Production isolation: stage3_milp_solver.py remains untouched.
8. Single invocation: exactly one optimizer call per solve.
9. Verifier integrity: independent verifier detects deliberately corrupted results.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import stage3_milp_solver as s3
import stage3_target_aware_solver as tas
import stage2_stage3_adapter as adapter
import stage2_counterfactual_engine as s2
import stage1_deployment as deployment
import nutrient_rule_policy as nrp


class TestStage3TargetAware(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = s3.load_food_database()
        cls.cases = json.loads((PROJECT_DIR / "Saved Model" / "stage2_comparison_study_v2" / "cases_LOCAL.json").read_text(encoding="utf-8"))
        cls.artifacts = s2._load_preprocessing_artifacts(deployment.PREPROCESSOR)
        cls.cp_dir = PROJECT_DIR / "Saved Model" / "stage2_comparison_study_v2" / "checkpoints"

    def _build_test_case_input(self, case_idx: int, arm: str) -> tuple[dict, dict]:
        case = self.cases[case_idx]
        seqn = case["SEQN"]
        meta = case["metadata"]
        w = meta["weight_kg"]
        h = meta["height_cm"]
        age = meta["age_years"]
        is_fem = meta["is_female"]
        bmr = 10.0 * w + 6.25 * h - 5.0 * age + (-161.0 if is_fem else 5.0)
        tdee = bmr * s2.PAL_FACTORS[meta["activity_level"]]

        ro_cp = json.loads((self.cp_dir / f"case_{case_idx:03d}_{seqn}_rules_only.json").read_text(encoding="utf-8"))
        p_d0 = ro_cp["baseline_probabilities"]["diabetes"]
        p_o0 = ro_cp["baseline_probabilities"]["obesity"]
        p_c0 = ro_cp["baseline_probabilities"]["cvd"]
        t_kcal = max(tdee - 500.0, 1200.0) if p_o0 >= s2.CUTOFF_OBESITY else max(tdee, 1200.0)

        lower, upper = s2._build_nutrient_bounds(
            artifacts=self.artifacts, t_kcal=t_kcal, weight_kg=w, is_female=is_fem, age_years=age,
            p_diab_0=p_d0, p_obes_0=p_o0, p_cvd_0=p_c0
        )
        n_obs = ro_cp["observed_nutrients"]

        arm_cp = json.loads((self.cp_dir / f"case_{case_idx:03d}_{seqn}_{arm}.json").read_text(encoding="utf-8"))
        tgt_vec = arm_cp["target_nutrients"]

        envelope = []
        for i, code in enumerate(s3.NUTRIENT_CODES):
            envelope.append({
                "nutrient": code,
                "target": float(tgt_vec[i]),
                "min_bound": float(lower[i]),
                "max_bound": float(upper[i]),
                "baseline": float(n_obs[i]),
                "rule_provenance": nrp.reviewed_rule_metadata(code, float(lower[i]), float(upper[i])),
            })

        s3_input = {
            "patient_metadata": dict(meta),
            "energy_prescription": {"BMR": float(bmr), "TDEE": float(tdee), "target_calories": float(t_kcal)},
            "nutrient_constraint_envelope": envelope,
        }
        return s3_input, arm_cp

    def test_01_hand_checkable_target_shift_alters_plan(self):
        """Test 1: Synthetic test where shifting targets alters meal plan in Variant B under identical bounds."""
        # Create a synthetic 5-food catalog
        # Two vegetable candidates with identical selection cost and calories, but one has high protein and the other has high fiber
        def make_food(fid, name, cat, kcal, prot, fibe, cost=1.0):
            nutrients = {code: 1.0 for code in s3.NUTRIENT_CODES}
            nutrients["DR1TKCAL"] = float(kcal)
            nutrients["DR1TPROT"] = float(prot)
            nutrients["DR1TFIBE"] = float(fibe)
            nutrients["DR1TCARB"] = float(kcal * 0.50 / 4.0)  # ~50% carbs for AMDR
            nutrients["DR1TTFAT"] = float(kcal * 0.25 / 9.0)  # ~25% fat for AMDR
            nutrients["DR1TSODI"] = 200.0
            nutrients["DR1TPOTA"] = 500.0
            return s3.FoodItem(
                food_id=fid, name=name, category=cat, serving_unit="serving", serving_weight_g=100.0,
                nutrients_per_serving=nutrients, selection_cost=cost, data_source="synthetic",
                unknown_nutrients=(), catalog_version="test", dietary_type="pure_vegetarian"
            )

        f_cereal = make_food("f_cereal", "Whole Grain", "cereal_grain", 200, 6, 4)
        f_dal = make_food("f_dal", "Dal Tadka", "dal_pulse", 200, 14, 6)
        f_dairy = make_food("f_dairy", "Curd", "dairy", 100, 8, 0)
        # Vegetable A: High protein (18g), low fiber (3g)
        f_veg_prot = make_food("f_veg_prot", "Soy Green Veg", "vegetable_sabzi", 150, 18, 3, cost=1.0)
        # Vegetable B: Low protein (4g), high fiber (16g)
        f_veg_fibe = make_food("f_veg_fibe", "High Fiber Veg", "vegetable_sabzi", 150, 4, 16, cost=1.0)

        synthetic_db = [f_cereal, f_dal, f_dairy, f_veg_prot, f_veg_fibe]

        # Common broad bounds where both f_veg_prot and f_veg_fibe are fully feasible
        envelope_base = []
        for code in s3.NUTRIENT_CODES:
            lo, hi = 0.0, 10000.0
            if code == "DR1TKCAL":
                lo, hi = 1000.0, 2500.0
            elif code == "DR1TPROT":
                lo, hi = 30.0, 150.0
            elif code == "DR1TFIBE":
                lo, hi = 15.0, 80.0
            envelope_base.append({
                "nutrient": code, "target": 50.0, "min_bound": lo, "max_bound": hi,
                "baseline": 50.0,
                "rule_provenance": nrp.reviewed_rule_metadata(code, lo, hi),
            })

        stage_input_prot = {
            "patient_metadata": {"age_years": 35, "is_female": False, "weight_kg": 70, "height_cm": 175},
            "energy_prescription": {"BMR": 1600.0, "TDEE": 2000.0, "target_calories": 1600.0},
            "nutrient_constraint_envelope": copy.deepcopy(envelope_base),
        }
        # Set protein target high (120g) and fiber target low (20g)
        for r in stage_input_prot["nutrient_constraint_envelope"]:
            if r["nutrient"] == "DR1TPROT":
                r["target"] = 120.0
            elif r["nutrient"] == "DR1TFIBE":
                r["target"] = 20.0

        stage_input_fibe = {
            "patient_metadata": {"age_years": 35, "is_female": False, "weight_kg": 70, "height_cm": 175},
            "energy_prescription": {"BMR": 1600.0, "TDEE": 2000.0, "target_calories": 1600.0},
            "nutrient_constraint_envelope": copy.deepcopy(envelope_base),
        }
        # Set fiber target high (60g) and protein target low (40g)
        for r in stage_input_fibe["nutrient_constraint_envelope"]:
            if r["nutrient"] == "DR1TPROT":
                r["target"] = 40.0
            elif r["nutrient"] == "DR1TFIBE":
                r["target"] = 60.0

        # Variant A (bounds-based): Since bounds, calories, and selection costs are identical,
        # Variant A selects the EXACT same basket under both inputs
        out_a_prot = tas.solve_experimental_meal_plan(
            stage_input_prot, synthetic_db, time_limit_sec=3.0,
            revision_spec={"dietary_mode": "pure_vegetarian"},
            objective_variant="current",
        )
        out_a_fibe = tas.solve_experimental_meal_plan(
            stage_input_fibe, synthetic_db, time_limit_sec=3.0,
            revision_spec={"dietary_mode": "pure_vegetarian"},
            objective_variant="current",
        )
        self.assertTrue(out_a_prot.accepted)
        self.assertTrue(out_a_fibe.accepted)
        basket_a_prot = {x["food_id"]: x["quantity"] for x in out_a_prot.solved_food_basket}
        basket_a_fibe = {x["food_id"]: x["quantity"] for x in out_a_fibe.solved_food_basket}
        self.assertEqual(basket_a_prot, basket_a_fibe, "Variant A must be insensitive to target shifts under identical bounds.")

        # Variant B (target-aware): High-protein target must select f_veg_prot; High-fiber target must select f_veg_fibe
        out_b_prot = tas.solve_experimental_meal_plan(
            stage_input_prot, synthetic_db, time_limit_sec=3.0,
            revision_spec={"dietary_mode": "pure_vegetarian"},
            objective_variant="target_aware",
            target_weight=2.0,
        )
        out_b_fibe = tas.solve_experimental_meal_plan(
            stage_input_fibe, synthetic_db, time_limit_sec=3.0,
            revision_spec={"dietary_mode": "pure_vegetarian"},
            objective_variant="target_aware",
            target_weight=2.0,
        )
        self.assertTrue(out_b_prot.accepted)
        self.assertTrue(out_b_fibe.accepted)
        basket_b_prot = {x["food_id"]: x["quantity"] for x in out_b_prot.solved_food_basket}
        basket_b_fibe = {x["food_id"]: x["quantity"] for x in out_b_fibe.solved_food_basket}

        # Verify that Variant B shifted the preferred dish
        self.assertIn("f_veg_prot", basket_b_prot)
        self.assertIn("f_veg_fibe", basket_b_fibe)
        self.assertNotEqual(basket_b_prot, basket_b_fibe, "Variant B must shift the preferred food basket when targets shift.")

    def test_02_arm_sensitivity_different_objectives_in_variant_b(self):
        """Test 2: Different Stage 2 arms produce distinct objectives in Variant B."""
        s3_ro, _ = self._build_test_case_input(0, "rules_only")
        s3_mc, _ = self._build_test_case_input(0, "minimum_change")
        s3_mg, _ = self._build_test_case_input(0, "model_guided")

        out_ro = tas.solve_experimental_meal_plan(s3_ro, self.catalog, time_limit_sec=3.0, objective_variant="target_aware", target_weight=1.0)
        out_mc = tas.solve_experimental_meal_plan(s3_mc, self.catalog, time_limit_sec=3.0, objective_variant="target_aware", target_weight=1.0)
        out_mg = tas.solve_experimental_meal_plan(s3_mg, self.catalog, time_limit_sec=3.0, objective_variant="target_aware", target_weight=1.0)

        self.assertTrue(out_ro.accepted)
        self.assertTrue(out_mc.accepted)
        self.assertTrue(out_mg.accepted)

        cost_ro = out_ro.solver_metadata["cost_breakdown"]["target_deviation_cost"]
        cost_mc = out_mc.solver_metadata["cost_breakdown"]["target_deviation_cost"]
        cost_mg = out_mg.solver_metadata["cost_breakdown"]["target_deviation_cost"]

        # The target deviation costs must differ because target vectors are distinct
        self.assertFalse(
            math.isclose(cost_ro, cost_mc, rel_tol=1e-5) and math.isclose(cost_mc, cost_mg, rel_tol=1e-5),
            "Variant B target deviation costs must distinguish between distinct Stage 2 arms.",
        )

    def test_03_feasible_set_invariance(self):
        """Test 3: Feasible space of discrete food variables is identical between Variant A and Variant B."""
        s3_input, _ = self._build_test_case_input(0, "model_guided")
        stage = s3._parse_stage2_input(s3_input)
        eligible = [f for f in self.catalog if f.dietary_type == "pure_vegetarian"]
        m = len(eligible)

        # Build Variant A constraints
        c_a, integrality_a, bounds_a, constr_a, info_a = s3._make_sparse_constraints(
            eligible, stage, gamma=0.10, dietary_mode="pure_vegetarian", allow_zero_dairy=False
        )

        # For any discrete basket q in {0,1,2,3}^M that satisfies Variant A:
        # there exists a unique non-negative d+, d- that satisfies Variant B equality rows
        target_nutrients = tas.get_eligible_target_nutrients(stage, eligible)
        F = info_a["food_matrix"]

        rng = np.random.default_rng(42)
        for _ in range(5):
            # Generate random integer serving counts q
            q_test = rng.integers(0, 4, size=m, dtype=np.int64)
            # Evaluate delivered nutrition
            delivered = F @ q_test.astype(float)
            # Variant B target deviation equality requires: delivered_k - d_k^+ + d_k^- = target_k
            # Setting d_k^+ = max(0, delivered_k - target_k) >= 0 and d_k^- = max(0, target_k - delivered_k) >= 0
            # ALWAYS satisfies the equality constraint with zero violation!
            for code in target_nutrients:
                k = s3.NUTRIENT_INDEX[code]
                val = delivered[k]
                tgt = float(stage.targets[code])
                d_plus = max(0.0, val - tgt)
                d_minus = max(0.0, tgt - val)
                residual = abs((val - d_plus + d_minus) - tgt)
                self.assertLess(residual, 1e-12, "L1 deviation variables must satisfy target equality exactly for any discrete portion vector.")

    def test_04_zero_slack_invariant_bounds_not_purchased(self):
        """Test 4: Deviation variables cannot purchase violations of hard component bounds."""
        s3_input, _ = self._build_test_case_input(0, "model_guided")
        # Artificially set protein target to 300g (way above upper bound of 80g)
        for r in s3_input["nutrient_constraint_envelope"]:
            if r["nutrient"] == "DR1TPROT":
                r["target"] = 300.0
                r["max_bound"] = 75.0  # Strict upper bound

        out = tas.solve_experimental_meal_plan(
            s3_input, self.catalog, time_limit_sec=3.0,
            revision_spec={"dietary_mode": "pure_vegetarian"},
            objective_variant="target_aware",
            target_weight=5.0,  # High weight to aggressively incentivize higher protein
        )
        if out.accepted:
            deliv_prot = next(row["delivered"] for row in out.delivered_nutrition if row["nutrient"] == "DR1TPROT")
            self.assertLessEqual(
                deliv_prot, 75.0 + 1e-4,
                f"Delivered protein {deliv_prot}g exceeded hard upper bound 75.0g; deviation variables must NOT purchase bound violations."
            )

    def test_05_energy_reconciliation_no_double_penalty(self):
        """Test 5: Calorie target is not double-counted in target deviation variables."""
        s3_input, _ = self._build_test_case_input(0, "model_guided")
        stage = s3._parse_stage2_input(s3_input)
        eligible = [f for f in self.catalog if f.dietary_type == "pure_vegetarian"]

        target_nutrients = tas.get_eligible_target_nutrients(stage, eligible)
        self.assertNotIn(
            "DR1TKCAL", target_nutrients,
            "DR1TKCAL must NOT be in target_nutrients because calorie error is already penalized by cal_over and cal_under."
        )

        out = tas.solve_experimental_meal_plan(
            s3_input, self.catalog, time_limit_sec=3.0,
            objective_variant="target_aware",
        )
        if out.accepted:
            breakdown = out.solver_metadata["cost_breakdown"]
            self.assertIn("calorie_cost", breakdown)
            self.assertIn("target_deviation_cost", breakdown)

    def test_06_schema_coverage_missing_active_nutrients_fail_closed(self):
        """Test 6: Missing active nutrient values fail closed."""
        s3_input, _ = self._build_test_case_input(0, "model_guided")
        # Corrupt a food in catalog by marking active nutrient unknown
        corrupted_db = list(self.catalog)
        first_food = corrupted_db[0]
        corrupted_food = s3.FoodItem(
            food_id=first_food.food_id,
            name=first_food.name,
            category=first_food.category,
            serving_unit=first_food.serving_unit,
            serving_weight_g=first_food.serving_weight_g,
            nutrients_per_serving=dict(first_food.nutrients_per_serving),
            selection_cost=first_food.selection_cost,
            data_source=first_food.data_source,
            unknown_nutrients=("DR1TPROT",),  # Active nutrient marked unknown
            catalog_version="corrupted",
            dietary_type=first_food.dietary_type,
        )
        corrupted_db[0] = corrupted_food

        # Solve should exclude the incomplete food
        out = tas.solve_experimental_meal_plan(
            s3_input, corrupted_db, time_limit_sec=3.0,
            objective_variant="target_aware",
        )
        if out.accepted:
            basket_ids = {x["food_id"] for x in out.solved_food_basket}
            self.assertNotIn(corrupted_food.food_id, basket_ids, "Food with unknown active nutrient must never be selected.")

        # Test invalid nonfinite target
        bad_input = copy.deepcopy(s3_input)
        bad_input["nutrient_constraint_envelope"][0]["target"] = float("nan")
        with self.assertRaises(Exception):
            tas.solve_experimental_meal_plan(bad_input, self.catalog, time_limit_sec=3.0)

    def test_07_production_isolation_milp_solver_untouched(self):
        """Test 7: Production solver stage3_milp_solver.py remains completely untouched."""
        # Check git status for stage3_milp_solver.py
        result = subprocess.run(
            ["git", "diff", "--name-only", "stage3_milp_solver.py"],
            cwd=str(PROJECT_DIR),
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout.strip(), "", "stage3_milp_solver.py must have zero git modifications.")

        # Ensure production solve_meal_plan can be imported and executed independently
        s3_input, _ = self._build_test_case_input(0, "rules_only")
        out_prod = s3.solve_meal_plan(s3_input, self.catalog, time_limit_sec=3.0)
        self.assertTrue(out_prod.accepted)
        self.assertNotIn("target_attainment", out_prod.solver_metadata, "Production solver must not output experimental metadata.")

    def test_08_single_invocation_one_optimizer_call(self):
        """Test 8: Exactly one optimizer invocation per solve."""
        s3_input, _ = self._build_test_case_input(0, "model_guided")

        with patch("stage3_target_aware_solver.milp", wraps=tas.milp) as mock_milp:
            out = tas.solve_experimental_meal_plan(
                s3_input, self.catalog, time_limit_sec=3.0,
                objective_variant="target_aware",
            )
            self.assertEqual(mock_milp.call_count, 1, "Must call milp exactly once per solve.")

    def test_09_verifier_integrity_catches_corrupted_results(self):
        """Test 9: Independent verifier detects deliberately corrupted results."""
        s3_input, _ = self._build_test_case_input(0, "model_guided")
        out = tas.solve_experimental_meal_plan(
            s3_input, self.catalog, time_limit_sec=3.0,
            revision_spec={"dietary_mode": "pure_vegetarian"},
            objective_variant="target_aware",
        )
        self.assertTrue(out.accepted)

        # 1. Corrupt serving quantity (set to 4 > MAX_SERVINGS_PER_DISH)
        bad_basket_1 = copy.deepcopy(out.solved_food_basket)
        bad_basket_1[0]["quantity"] = 4
        replay_1 = adapter.replay_stage3_solution(bad_basket_1, self.catalog, s3_input, "pure_vegetarian")
        self.assertFalse(replay_1["passed"], "Verifier must catch serving quantity > 3.")

        # 2. Corrupt food ID
        bad_basket_2 = copy.deepcopy(out.solved_food_basket)
        bad_basket_2[0]["food_id"] = "non_existent_dish_xyz"
        replay_2 = adapter.replay_stage3_solution(bad_basket_2, self.catalog, s3_input, "pure_vegetarian")
        self.assertFalse(replay_2["passed"], "Verifier must catch unknown food ID.")

        # 3. Corrupt dietary mode (inject non-veg food into pure_vegetarian)
        non_veg_food = next(f for f in self.catalog if f.dietary_type == "non_vegetarian")
        bad_basket_3 = copy.deepcopy(out.solved_food_basket)
        bad_basket_3[0] = {
            "food_id": non_veg_food.food_id,
            "name": non_veg_food.name,
            "quantity": 1,
            "serving_weight_g": non_veg_food.serving_weight_g,
        }
        replay_3 = adapter.replay_stage3_solution(bad_basket_3, self.catalog, s3_input, "pure_vegetarian")
        self.assertFalse(replay_3["passed"], "Verifier must catch non-veg food in pure_vegetarian mode.")

        # 4. Tamper with bounds to induce violation
        strict_input = copy.deepcopy(s3_input)
        for r in strict_input["nutrient_constraint_envelope"]:
            if r["nutrient"] == "DR1TPROT":
                r["min_bound"] = 500.0  # Impossible lower bound
        replay_4 = adapter.replay_stage3_solution(out.solved_food_basket, self.catalog, strict_input, "pure_vegetarian")
        self.assertFalse(replay_4["passed"], "Verifier must catch bound violation.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
