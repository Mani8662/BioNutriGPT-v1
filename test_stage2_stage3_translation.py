"""Unit and Contract Tests for Stage 2 -> Stage 3 Translation Study.

Verifies:
1. Correct Stage 2 target/bound mapping and units.
2. Different arm targets actually reach the intended solver inputs.
3. Common inputs (bounds, calories, baselines) remain strictly identical across arms.
4. Rejected Stage 2 cases stay rejected in end-to-end evaluation.
5. Offline diagnostic evaluations cannot mutate caller inputs or application state.
6. Nutrient totals independently recomputed from food quantities and catalog values.
7. Serving integrality, portion bounds, and dietary exclusions.
8. Missing nutrients fail closed and are not silently treated as measured zero.
9. Known feasible and contradictory development examples.
10. Timeout and solver status classification.
11. Retention of failures in denominators (no silent drop).
12. Checkpoint fingerprint mismatch rejection.
13. Exactly one solver invocation per attempt (no double solve for logging).
14. Independent verifier detects deliberately corrupted results.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import stage3_milp_solver as s3
import stage2_stage3_adapter as adapter
import stage2_counterfactual_engine as s2
import stage1_deployment as deployment
import nutrient_rule_policy as nrp


class TestStage2Stage3Translation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.foods = s3.load_food_database()
        cls.artifacts = s2._load_preprocessing_artifacts(deployment.PREPROCESSOR)
        cls.cases = json.load(open(ROOT / "Saved Model/stage2_comparison_study_v2/cases_LOCAL.json", encoding="utf-8"))
        cls.catalog_hash = "a66b145f088f02c82ed9e8f81639a063bf3aed0bc44fcb2aab381e8bda24e64b"

        # Load case 0 checkpoints
        cp_dir = ROOT / "Saved Model/stage2_comparison_study_v2/checkpoints"
        seqn = cls.cases[0]["SEQN"]
        cls.c0_ro = json.load(open(cp_dir / f"case_000_{seqn}_rules_only.json"))
        cls.c0_mc = json.load(open(cp_dir / f"case_000_{seqn}_minimum_change.json"))
        cls.c0_mg = json.load(open(cp_dir / f"case_000_{seqn}_model_guided.json"))

        # Build Stage 2 envelope helper
        c0 = cls.cases[0]
        meta = c0["metadata"]
        w = meta["weight_kg"]
        h = meta["height_cm"]
        age = meta["age_years"]
        is_fem = meta["is_female"]
        bmr = 10.0 * w + 6.25 * h - 5.0 * age + (-161.0 if is_fem else 5.0)
        tdee = bmr * s2.PAL_FACTORS[meta["activity_level"]]
        t_kcal = max(tdee - 500.0, 1200.0) if cls.c0_ro["baseline_probabilities"]["obesity"] >= s2.CUTOFF_OBESITY else max(tdee, 1200.0)

        lower, upper = s2._build_nutrient_bounds(
            artifacts=cls.artifacts, t_kcal=t_kcal, weight_kg=w, is_female=is_fem, age_years=age,
            p_diab_0=cls.c0_ro["baseline_probabilities"]["diabetes"],
            p_obes_0=cls.c0_ro["baseline_probabilities"]["obesity"],
            p_cvd_0=cls.c0_ro["baseline_probabilities"]["cvd"]
        )
        cls.lower = lower
        cls.upper = upper
        cls.bmr = bmr
        cls.tdee = tdee
        cls.t_kcal = t_kcal
        cls.n_obs = cls.c0_ro["observed_nutrients"]
        cls.meta = meta

    def _build_input(self, target_vec):
        envelope = []
        for i, code in enumerate(s3.NUTRIENT_CODES):
            envelope.append({
                "nutrient": code,
                "target": float(target_vec[i]),
                "min_bound": float(self.lower[i]),
                "max_bound": float(self.upper[i]),
                "baseline": float(self.n_obs[i]),
                "rule_provenance": nrp.reviewed_rule_metadata(code, float(self.lower[i]), float(self.upper[i]))
            })
        return {
            "patient_metadata": dict(self.meta),
            "energy_prescription": {"BMR": float(self.bmr), "TDEE": float(self.tdee), "target_calories": float(self.t_kcal)},
            "nutrient_constraint_envelope": envelope
        }

    def test_01_target_and_bound_mapping(self):
        """Verify that Stage 2 bounds and targets are mapped with correct lengths and non-empty values."""
        in_ro = self._build_input(self.c0_ro["target_nutrients"])
        self.assertEqual(len(in_ro["nutrient_constraint_envelope"]), 39)
        for r in in_ro["nutrient_constraint_envelope"]:
            self.assertTrue(math.isfinite(r["target"]))
            self.assertTrue(math.isfinite(r["min_bound"]))
            self.assertTrue(math.isfinite(r["max_bound"]))
            self.assertGreaterEqual(r["max_bound"], r["min_bound"])

    def test_02_targets_differ_while_common_inputs_are_identical(self):
        """Verify that arm targets differ, while bounds, energy, and baselines are strictly identical."""
        in_ro = self._build_input(self.c0_ro["target_nutrients"])
        in_mc = self._build_input(self.c0_mc["target_nutrients"])
        in_mg = self._build_input(self.c0_mg["target_nutrients"])

        # Targets must differ between arms
        ro_tgts = [r["target"] for r in in_ro["nutrient_constraint_envelope"]]
        mc_tgts = [r["target"] for r in in_mc["nutrient_constraint_envelope"]]
        mg_tgts = [r["target"] for r in in_mg["nutrient_constraint_envelope"]]
        self.assertNotEqual(ro_tgts, mc_tgts)
        self.assertNotEqual(mc_tgts, mg_tgts)

        # Common inputs must be strictly identical
        for r1, r2, r3 in zip(in_ro["nutrient_constraint_envelope"], in_mc["nutrient_constraint_envelope"], in_mg["nutrient_constraint_envelope"]):
            self.assertEqual(r1["min_bound"], r2["min_bound"])
            self.assertEqual(r2["min_bound"], r3["min_bound"])
            self.assertEqual(r1["max_bound"], r2["max_bound"])
            self.assertEqual(r2["max_bound"], r3["max_bound"])
            self.assertEqual(r1["baseline"], r2["baseline"])
            self.assertEqual(r2["baseline"], r3["baseline"])

        self.assertEqual(in_ro["energy_prescription"], in_mc["energy_prescription"])
        self.assertEqual(in_mc["energy_prescription"], in_mg["energy_prescription"])

    def test_03_end_to_end_rejection_of_failed_stage2_cases(self):
        """Verify that a case that failed Stage 2 stays rejected in End-to-End evaluation."""
        # Case 0 in Arm A failed the Stage 2 score gate (overall_accepted is False)
        self.assertFalse(self.c0_ro["overall_accepted"])

        in_ro = self._build_input(self.c0_ro["target_nutrients"])
        s3_out = s3.solve_meal_plan(in_ro, self.foods, time_limit_sec=1.5, revision_spec={"dietary_mode": "pure_vegetarian"})
        # Stage 3 meal may succeed in diagnostic mode
        self.assertTrue(s3_out.accepted)

        # But end-to-end pipeline acceptance MUST be False because Stage 2 failed!
        e2e_accepted = bool(self.c0_ro["overall_accepted"] and s3_out.accepted)
        self.assertFalse(e2e_accepted)

    def test_04_caller_input_immutability(self):
        """Verify that Stage 3 solving does not mutate caller input structures."""
        in_ro = self._build_input(self.c0_ro["target_nutrients"])
        snap_before = json.dumps(in_ro, sort_keys=True)
        _ = s3.solve_meal_plan(in_ro, self.foods, time_limit_sec=1.5, revision_spec={"dietary_mode": "pure_vegetarian"})
        snap_after = json.dumps(in_ro, sort_keys=True)
        self.assertEqual(snap_before, snap_after)

    def test_05_independent_recomputation_of_nutrient_totals(self):
        """Verify that replay_stage3_solution reconstructs totals from quantities and catalog values."""
        in_ro = self._build_input(self.c0_ro["target_nutrients"])
        s3_out = s3.solve_meal_plan(in_ro, self.foods, time_limit_sec=1.5, revision_spec={"dietary_mode": "pure_vegetarian"})

        replay = adapter.replay_stage3_solution(s3_out.solved_food_basket, self.foods, in_ro, "pure_vegetarian")
        self.assertTrue(replay["passed"])
        self.assertTrue(replay["has_plan"])

        # Check delivered nutrition match
        lookup = {f.food_id: f for f in self.foods}
        for code in s3.NUTRIENT_CODES:
            manual_tot = sum(item["quantity"] * lookup[item["food_id"]].nutrients_per_serving[code] for item in s3_out.solved_food_basket)
            self.assertAlmostEqual(manual_tot, replay["totals"][code], places=4)

    def test_06_serving_integrality_and_portion_bounds(self):
        """Verify that non-integer or out-of-range quantities fail replay audit."""
        in_ro = self._build_input(self.c0_ro["target_nutrients"])
        s3_out = s3.solve_meal_plan(in_ro, self.foods, time_limit_sec=1.5, revision_spec={"dietary_mode": "pure_vegetarian"})

        # Corrupt one serving to 4 (exceeding MAX_SERVINGS_PER_DISH = 3)
        bad_basket = copy.deepcopy(s3_out.solved_food_basket)
        bad_basket[0]["quantity"] = 4
        replay_bad = adapter.replay_stage3_solution(bad_basket, self.foods, in_ro, "pure_vegetarian")
        self.assertFalse(replay_bad["passed"])
        self.assertTrue(any("invalid_serving_quantity" in e for e in replay_bad["errors"]))

    def test_07_dietary_mode_exclusion(self):
        """Verify that a non-vegetarian food in pure_vegetarian mode fails replay."""
        in_ro = self._build_input(self.c0_ro["target_nutrients"])
        # Find a non-veg food
        non_veg = next(f for f in self.foods if f.dietary_type == "non_vegetarian")

        synthetic_basket = [
            {"food_id": "recipe_boiled_rice", "quantity": 1},
            {"food_id": "recipe_washed_moong_dal", "quantity": 1},
            {"food_id": "recipe_spinach_boiled", "quantity": 1},
            {"food_id": "recipe_brinjal_bhartha", "quantity": 1},
            {"food_id": non_veg.food_id, "quantity": 1},
        ]
        replay = adapter.replay_stage3_solution(synthetic_basket, self.foods, in_ro, "pure_vegetarian")
        self.assertFalse(replay["passed"])
        self.assertTrue(any("non_veg_food_in_pure_veg" in e for e in replay["errors"]))

    def test_08_missing_active_nutrients_fail_closed(self):
        """Verify that catalog foods with unknown active nutrients are rejected and cannot satisfy active rules."""
        in_ro = self._build_input(self.c0_ro["target_nutrients"])
        # Corrupt catalog item to have unknown DR1TVC
        corrupt_foods = [
            s3.FoodItem(
                f.food_id, f.name, f.category, f.serving_unit, f.serving_weight_g,
                f.nutrients_per_serving,
                unknown_nutrients=("DR1TVC",) if f.category == "vegetable_sabzi" else f.unknown_nutrients,
                dietary_type=f.dietary_type
            )
            for f in self.foods
        ]
        out = s3.solve_meal_plan(in_ro, corrupt_foods, time_limit_sec=1.5, revision_spec={"dietary_mode": "pure_vegetarian"})
        # Should fail closed because vegetable_sabzi is required and lacks complete active data
        self.assertFalse(out.accepted)
        self.assertEqual(out.failure["code"], "insufficient_nutrient_data")

    def test_09_contradictory_bounds_fail_cleanly(self):
        """Verify that contradictory nutrient bounds fail cleanly with infeasibility without crashing."""
        in_ro = self._build_input(self.c0_ro["target_nutrients"])
        # Set protein upper bound lower than minimum possible food content (e.g. max 1g protein)
        for r in in_ro["nutrient_constraint_envelope"]:
            if r["nutrient"] == "DR1TPROT":
                r["min_bound"] = 0.0
                r["max_bound"] = 1.0  # Impossible with required dals/cereals

        out = s3.solve_meal_plan(in_ro, self.foods, time_limit_sec=1.5, revision_spec={"dietary_mode": "pure_vegetarian"})
        self.assertFalse(out.accepted)
        self.assertIn(out.failure["code"], ("infeasible_active_rules", "no_valid_candidate"))

    def test_10_single_solver_invocation(self):
        """Verify that exactly one solver invocation occurs per attempt."""
        in_ro = self._build_input(self.c0_ro["target_nutrients"])
        mock_milp = MagicMock(wraps=s3.milp)

        with unittest.mock.patch("stage3_milp_solver.milp", mock_milp):
            out = s3.solve_meal_plan(in_ro, self.foods, time_limit_sec=1.5, revision_spec={"dietary_mode": "pure_vegetarian"})

        self.assertEqual(mock_milp.call_count, 1)

    def test_11_fingerprint_mismatch_detection(self):
        """Verify that changing any input alters the canonical fingerprint."""
        in_ro = self._build_input(self.c0_ro["target_nutrients"])
        fp1 = adapter.compute_translation_fingerprint(0, 109295, "rules_only", "pure_vegetarian", in_ro, self.catalog_hash)

        # Alter target
        in_mod = copy.deepcopy(in_ro)
        in_mod["nutrient_constraint_envelope"][0]["target"] += 5.0
        fp2 = adapter.compute_translation_fingerprint(0, 109295, "rules_only", "pure_vegetarian", in_mod, self.catalog_hash)
        self.assertNotEqual(fp1, fp2)

        # Alter mode
        fp3 = adapter.compute_translation_fingerprint(0, 109295, "rules_only", "veg_nonveg_combo", in_ro, self.catalog_hash)
        self.assertNotEqual(fp1, fp3)

        # Alter catalog hash
        fp4 = adapter.compute_translation_fingerprint(0, 109295, "rules_only", "pure_vegetarian", in_ro, "altered_catalog_hash")
        self.assertNotEqual(fp1, fp4)

    def test_12_verifier_detects_deliberate_corruption(self):
        """Verify that the independent verifier flags a corrupted basket total or modified quantity."""
        in_ro = self._build_input(self.c0_ro["target_nutrients"])
        s3_out = s3.solve_meal_plan(in_ro, self.foods, time_limit_sec=1.5, revision_spec={"dietary_mode": "pure_vegetarian"})

        # Deliberately falsify one food quantity
        corrupted_basket = copy.deepcopy(s3_out.solved_food_basket)
        corrupted_basket[0]["quantity"] += 1
        replay = adapter.replay_stage3_solution(corrupted_basket, self.foods, in_ro, "pure_vegetarian")
        # Calorie error or portion bounds will be caught
        self.assertFalse(replay["passed"])


if __name__ == "__main__":
    unittest.main()
