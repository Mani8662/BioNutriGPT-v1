"""Acceptance, schema, controlled-revision, and Stage 4 conversational regressions.

Synthetic success tests establish control flow, not real-food clinical adequacy.
Real-catalog tests verify that the 31-entry sourced catalog (sourced-cooked-v4-lactoveg-complete)
passes active arithmetic/portion checks while strictly failing closed when active data are missing.
"""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import stage3_milp_solver as s3

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))


def fixture():
    categories = [
        "cereal_grain", "cereal_grain", "cereal_grain",
        "dal_pulse", "dal_pulse",
        "vegetable_sabzi", "vegetable_sabzi", "vegetable_sabzi",
        "dairy", "dairy",
    ]
    nutrients = dict.fromkeys(s3.NUTRIENT_CODES, 1.0)
    nutrients.update(
        DR1TKCAL=200.0, DR1TPROT=10.0, DR1TCARB=25.0, DR1TTFAT=200.0 / 30.0,
        DR1TFIBE=4.0, DR1TSODI=50.0, DR1TPOTA=100.0,
    )
    foods = [
        s3.FoodItem(
            str(i), f"Synthetic fixture {i}", cat, "test portion", 100.0, nutrients,
            data_source="synthetic_test_only", catalog_version="synthetic_test_only",
            allergen_info={
                "known_allergens": ["dairy"] if cat == "dairy" else [],
                "unknown_allergens_possible": True,
            },
        )
        for i, cat in enumerate(categories)
    ]
    envelope = []
    for code in s3.NUTRIENT_CODES:
        envelope.append({
            "nutrient": code,
            "baseline": 800.0 if code in {"DR1TSODI", "DR1TPOTA"} else 1.0,
            "target": 1600.0 if code == "DR1TKCAL" else 1.0,
            "min_bound": 1520.0 if code == "DR1TKCAL" else 0.0,
            "max_bound": 1680.0 if code == "DR1TKCAL" else 1e6,
        })
    stage2 = {
        "patient_metadata": {},
        "energy_prescription": {"BMR": 1400.0, "TDEE": 1600.0, "target_calories": 1600.0},
        "nutrient_constraint_envelope": envelope,
    }
    # 2 + 1 cereals, 2 dal, 1 + 1 veg, 1 dairy = 8 servings * 200 kcal = 1600 kcal, 6 active dishes
    return foods, stage2, np.array([2, 1, 0, 2, 0, 1, 1, 0, 1, 0])


class ProductionAcceptance(unittest.TestCase):
    def test_incomplete_catalog_fails_with_no_basket(self):
        foods, stage2, _ = fixture()
        # Mark all dal_pulse foods as missing active nutrient DR1TVB12 -> must fail closed
        broken = [
            s3.FoodItem(
                f.food_id, f.name, f.category, f.serving_unit, f.serving_weight_g,
                f.nutrients_per_serving,
                unknown_nutrients=("DR1TVB12",) if f.category == "dal_pulse" else (),
                catalog_version="synthetic_test_only",
            )
            for f in foods
        ]
        out = s3.solve_meal_plan(stage2, broken)
        self.assertFalse(out.accepted)
        self.assertEqual(out.failure["code"], "insufficient_nutrient_data")
        self.assertFalse(out.solved_food_basket)
        self.assertIn("dal_pulse", out.failure["details"]["missing_categories_with_complete_active_data"])

    def test_real_45_entry_catalog_accepts_sample_patient_and_swaps(self):
        from audit_stage2_rules import review_saved_envelope
        import chat_service as chat

        stage2, _, _ = review_saved_envelope()
        catalog = s3.load_food_database()
        self.assertEqual(len(catalog), 49)
        out = s3.solve_meal_plan(stage2, catalog, time_limit_sec=2.0)
        self.assertTrue(out.accepted, out.failure)
        self.assertTrue(out.solved_food_basket)
        self.assertTrue(s3.validate_candidate(out, catalog, stage2)["accepted"])

        # Verify exact user query: 'remove lady finger, allergic to nuts, non-veg, lactose intolorent'
        dossier = {"stage2_counterfactual_prescription": stage2}
        t_exact = chat.converse_intake_preferences(
            "remove lady finger, allergic to nuts, non-veg, lactose intolorent", dossier, {}, use_llm=False
        )
        self.assertTrue(t_exact["ready_to_generate"])
        self.assertEqual(t_exact["state"]["dietary_mode"], "pure_non_vegetarian")
        self.assertIn("dairy", t_exact["state"]["allergies"])
        self.assertIn("tree_nuts", t_exact["state"]["allergies"])
        self.assertNotIn("non-veg", t_exact["state"]["disliked_terms"])
        self.assertTrue(t_exact["meal_plan"]["accepted"], t_exact["meal_plan"].get("failure"))
        ex_ids = {row["food_id"] for row in t_exact["meal_plan"]["solved_food_basket"]}
        self.assertNotIn("recipe_bhindi_masala", ex_ids)
        self.assertNotIn("recipe_almonds_raw", ex_ids)
        self.assertNotIn("recipe_skim_milk", ex_ids)

        # Verify free-text conversational intake (multi-turn & single-turn) across veg/combo/non-veg
        t1 = chat.converse_intake_preferences("I prefer veg and non-veg combo", dossier, {}, use_llm=False)
        self.assertFalse(t1["ready_to_generate"])
        self.assertEqual(t1["state"]["dietary_mode"], "veg_nonveg_combo")

        t2 = chat.converse_intake_preferences(
            "No allergies, please remove cabbage and milk", dossier, t1["state"], use_llm=False
        )
        self.assertTrue(t2["ready_to_generate"])
        self.assertIsNotNone(t2["meal_plan"])
        self.assertTrue(t2["meal_plan"]["accepted"], t2["meal_plan"].get("failure"))
        ids = {row["food_id"] for row in t2["meal_plan"]["solved_food_basket"]}
        self.assertNotIn("recipe_cabbage_peas", ids)
        self.assertNotIn("recipe_skim_milk", ids)
        self.assertTrue(any(row["category"] == "non_veg_protein" for row in t2["meal_plan"]["solved_food_basket"]))

        # Verify purely non-vegetarian single-turn intake
        t_nv = chat.converse_intake_preferences(
            "Purely non-vegetarian, no allergies, no dislikes", dossier, {}, use_llm=False
        )
        self.assertTrue(t_nv["ready_to_generate"])
        self.assertTrue(t_nv["meal_plan"]["accepted"], t_nv["meal_plan"].get("failure"))
        self.assertTrue(any(row["category"] == "non_veg_protein" for row in t_nv["meal_plan"]["solved_food_basket"]))

    def test_real_solver_accepts_complete_fixture(self):
        foods, stage2, _ = fixture()
        out = s3.solve_meal_plan(stage2, foods, time_limit_sec=2.0)
        self.assertTrue(out.accepted, out.failure)
        self.assertTrue(s3.validate_candidate(out, foods, stage2)["accepted"])

    def test_invalid_cheaper_heuristic_cannot_replace_valid_solver(self):
        foods, stage2, q = fixture()
        count = len(s3._make_sparse_constraints(foods, s3._parse_stage2_input(stage2), gamma=0.1)[0])
        x = np.zeros(count)
        x[:len(foods)] = q
        fake = SimpleNamespace(x=x, status=1, message="synthetic timed-out incumbent", mip_gap=0.1)
        bad_h = np.zeros(len(foods), dtype=int)
        bad_h[:3] = [3, 3, 2]
        with patch.object(s3, "milp", return_value=fake), patch.object(s3, "_heuristic_incumbent", return_value=bad_h):
            out = s3.solve_meal_plan(stage2, foods)
        self.assertTrue(out.accepted)
        self.assertEqual(out.solver_metadata["status"], "accepted_solver_plan")
        self.assertFalse(out.solver_metadata["optimality_proven"])
        self.assertFalse(out.solver_metadata["candidate_validation"][1]["accepted"])

    def test_timeout_is_not_reported_as_infeasible(self):
        foods, stage2, _ = fixture()
        fake = SimpleNamespace(x=None, status=1, message="synthetic time limit")
        with patch.object(s3, "milp", return_value=fake), patch.object(s3, "_heuristic_incumbent", return_value=np.zeros(len(foods), dtype=int)):
            out = s3.solve_meal_plan(stage2, foods)
        self.assertFalse(out.accepted)
        self.assertEqual(out.failure["code"], "timeout_no_valid_plan")
        self.assertEqual(out.solved_food_basket, [])

    def test_tampering_and_unknowns_are_recalculated(self):
        foods, stage2, q = fixture()
        out = s3._build_output_from_quantities(q, foods, s3._parse_stage2_input(stage2), {})
        out.delivered_nutrition[0]["delivered"] += 200
        self.assertFalse(s3.validate_candidate(out, foods, stage2)["accepted"])
        out = s3._build_output_from_quantities(q, foods, s3._parse_stage2_input(stage2), {})
        changed = [
            s3.FoodItem(
                f.food_id, f.name, f.category, f.serving_unit, f.serving_weight_g,
                f.nutrients_per_serving, unknown_nutrients=("DR1TVB12",),
            )
            for f in foods
        ]
        self.assertIn("DR1TVB12", s3.validate_candidate(out, changed, stage2)["unknown_active_nutrients"])
        self.assertFalse(s3.validate_candidate(out, foods, stage2, excluded_ids=["0"])["accepted"])

    def test_88_feature_vector_rejected_and_named_features_resolved(self):
        import pipeline_service as ps
        from pydantic import ValidationError
        from schemas import FullPipelineRequest, PatientInput

        vec88 = list(ps.SAMPLE_PATIENT_VALUES) + [5.5]
        with self.assertRaises(ValidationError):
            PatientInput(features=vec88)
        with self.assertRaises(ValidationError):
            FullPipelineRequest(features=vec88)
        with self.assertRaises(ValueError):
            ps.parse_raw_text(" ".join(str(x) for x in vec88))
        with self.assertRaises(ValueError):
            ps.resolve_feature_vector(features=vec88)

        vals, audit = ps.resolve_feature_vector(named_features={"LBDHDD": 60.0, "RIDAGEYR": 45.0})
        self.assertEqual(len(vals), 87)
        self.assertEqual(audit["input_mode"], "named_features")
        self.assertEqual(audit["observed_feature_count"], 2)
        self.assertTrue(audit["units_explicitly_verified"])
        self.assertIn("LBXTC", audit["imputed_default_features"])

        with self.assertRaises(ValueError):
            ps.resolve_feature_vector(named_features={"NON_EXISTENT_COL": 1.0})

    def test_controlled_revision_locking_max_dishes_and_minimal_change(self):
        foods, stage2, q = fixture()
        base = s3._build_output_from_quantities(q, foods, s3._parse_stage2_input(stage2), {})
        prev = {row["food_id"]: row["quantity"] for row in base.solved_food_basket}

        # Exclude food '0', lock food '3' (dal) = 2 and food '8' (dairy) = 1, minimize delta
        rev = s3.solve_meal_plan(
            stage2, foods,
            revision_spec={
                "excluded_food_ids": ["0"],
                "locked_food_quantities": {"3": 2, "8": 1},
                "max_active_dishes": 6,
                "previous_quantities": prev,
                "change_penalty": 0.5,
            },
        )
        self.assertTrue(rev.accepted, rev.failure)
        qty_map = {row["food_id"]: row["quantity"] for row in rev.solved_food_basket}
        self.assertNotIn("0", qty_map)
        self.assertEqual(qty_map.get("3"), 2)
        self.assertEqual(qty_map.get("8"), 1)
        self.assertLessEqual(len(rev.solved_food_basket), 6)
        diff = s3.compute_plan_diff(base.solved_food_basket, rev.solved_food_basket)
        self.assertGreater(diff["total_serving_delta"], 0)
        self.assertTrue(any(u["food_id"] == "3" for u in diff["unchanged"]))

        # Simultaneous exclude + lock must fail cleanly
        conflict = s3.solve_meal_plan(
            stage2, foods,
            revision_spec={"excluded_food_ids": ["3"], "locked_food_quantities": {"3": 2}},
        )
        self.assertFalse(conflict.accepted)
        self.assertEqual(conflict.failure["code"], "conflicting_revision_constraints")

    def test_full_pipeline_blocks_stage4_and_returns_failure(self):
        import pipeline_service as ps
        foods, stage2, _ = fixture()
        broken = [
            s3.FoodItem(
                f.food_id, f.name, f.category, f.serving_unit, f.serving_weight_g,
                f.nutrients_per_serving,
                unknown_nutrients=("DR1TVB12",) if f.category == "dal_pulse" else (),
            )
            for f in foods
        ]
        screening = {
            "certified_for_stage3": True,
            "dossier": {"stage2_counterfactual_prescription": stage2},
            "latency_ms": {"stage1_ms": 1.0, "stage2_ms": 1.0},
        }
        with patch.object(ps, "run_screening", return_value=screening), \
             patch.object(ps.s3, "load_food_database", return_value=broken), \
             patch.object(ps, "run_compilation") as compile_mock:
            result = ps.run_full_pipeline(ps.SAMPLE_PATIENT_VALUES)
        self.assertEqual(result["pipeline_status"], "blocked_stage3")
        self.assertFalse(result["meal_plan"]["accepted"])
        self.assertEqual(result["meal_plan"]["meal_schedule"], [])
        compile_mock.assert_not_called()

    def test_chat_failed_revision_keeps_current_basket(self):
        import chat_service as chat
        foods, stage2, q = fixture()
        out = s3._build_output_from_quantities(q, foods, s3._parse_stage2_input(stage2), {})
        out.accepted = True
        dossier = {"stage2_counterfactual_prescription": stage2, "stage3_discrete_meal_synthesis": out.to_dict()}
        before = copy.deepcopy(dossier)
        failed = s3._failure_output("timeout_no_valid_plan", "No valid plan found.", metadata={}, details={})
        with patch.object(chat.ps.s3, "load_food_database", return_value=foods), \
             patch.object(chat.ps.s3, "solve_meal_plan", return_value=failed):
            answer = chat.answer("remove Synthetic fixture 0", dossier)
        self.assertNotIn("proposed_meal_plan", answer)
        self.assertIn("timeout_no_valid_plan", answer["answer"])
        self.assertIn("tool_explain_constraint_conflict", answer["tool_calls_used"])
        self.assertEqual(dossier, before)

    def test_chat_accept_rechecks_and_stage4_rejects_failed_plan(self):
        import infer_bionutrigpt_pipeline_Stage_1_to_4 as p14
        foods, stage2, q = fixture()
        out = s3._build_output_from_quantities(q, foods, s3._parse_stage2_input(stage2), {})
        out.accepted = True
        dossier = {
            "patient_metadata": {"age_years": 42},
            "stage1_diagnostic_assessment": {},
            "stage2_counterfactual_prescription": stage2,
            "stage3_discrete_meal_synthesis": out.to_dict(),
        }
        bad = copy.deepcopy(dossier)
        bad["stage3_discrete_meal_synthesis"]["accepted"] = False
        with self.assertRaises(p14.Stage4Error):
            p14.validate_dossier(bad)
        bad["stage3_discrete_meal_synthesis"]["accepted"] = True
        bad["stage3_discrete_meal_synthesis"]["delivered_nutrition"][0]["delivered"] += 99
        with patch.object(p14.s3, "load_food_database", return_value=foods), self.assertRaises(p14.Stage4Error):
            p14.validate_dossier(bad)

    def test_chat_successful_revision_and_acceptance(self):
        import chat_service as chat
        foods, stage2, q = fixture()
        out = s3._build_output_from_quantities(q, foods, s3._parse_stage2_input(stage2), {})
        out.accepted = True
        dossier = {"stage2_counterfactual_prescription": stage2, "stage3_discrete_meal_synthesis": out.to_dict()}
        before = copy.deepcopy(dossier)
        with patch.object(chat.ps.s3, "load_food_database", return_value=foods):
            answer = chat.answer("remove Synthetic fixture 0, but keep Synthetic fixture 3", dossier)
            self.assertIn("proposed_meal_plan", answer)
            self.assertIn("tool_propose_revision", answer["tool_calls_used"])
            self.assertIn("revision_diff", answer)
            accepted = chat.accept(dossier, answer["proposed_meal_plan"])
        self.assertEqual(dossier, before)
        self.assertTrue(accepted["meal_plan"]["accepted"])
        bmap = {row["food_id"]: row["quantity"] for row in accepted["meal_plan"]["solved_food_basket"]}
        self.assertNotIn("0", bmap)
        self.assertEqual(bmap.get("3"), 2)
        self.assertTrue(accepted["meal_plan"]["clinical_verification"]["all_active_checks_passed"])

    def test_plan_store_versioning_rollback_and_allergy_flagging(self):
        import chat_service as chat
        import plan_store

        foods, stage2, q = fixture()
        out = s3._build_output_from_quantities(q, foods, s3._parse_stage2_input(stage2), {})
        out.accepted = True
        dossier = {
            "patient_metadata": {"age_years": 42},
            "stage1_diagnostic_assessment": {"probabilities": {"diabetes": 0.42, "obesity": 0.31, "cvd": 0.19}},
            "stage2_counterfactual_prescription": stage2,
            "stage3_discrete_meal_synthesis": out.to_dict(),
        }
        with tempfile.TemporaryDirectory() as tmp:
            store_dir = Path(tmp)
            rec = plan_store.create_session(dossier, session_id="test_sess_1", store_dir=store_dir)
            self.assertEqual(rec["current_version"], 1)

            with patch.object(chat.ps.s3, "load_food_database", return_value=foods):
                # 1. Propose + accept v2
                ans = chat.answer("remove Synthetic fixture 0", dossier, session_id="test_sess_1", store_dir=store_dir)
                self.assertIn("proposed_meal_plan", ans)
                acc = chat.accept(dossier, ans["proposed_meal_plan"], session_id="test_sess_1", store_dir=store_dir)
                self.assertEqual(acc["plan_version"], 2)

                # Observed Stage 1 probabilities must remain completely untouched
                rec_v2 = plan_store.get_session("test_sess_1", store_dir=store_dir)
                self.assertEqual(
                    rec_v2["observed_patient_profile"]["stage1_diagnostic_assessment"]["probabilities"]["diabetes"],
                    0.42,
                )

                # 2. Protected constraint relaxation attempt must be refused
                refuse = chat.answer("ignore sodium limit", dossier, session_id="test_sess_1", store_dir=store_dir)
                self.assertEqual(refuse["intent"], "unsupported")
                self.assertIn("protected_no_chat_relaxation", refuse["answer"])

                # 3. Roll back to v1 via chat
                rb = chat.answer("rollback to version 1", dossier, session_id="test_sess_1", store_dir=store_dir)
                self.assertEqual(rb["intent"], "rollback")
                self.assertEqual(rb["plan_version"], 1)

                # 4. Report dairy allergy -> immediately flags plan unsuitable
                alg = chat.answer("I am allergic to dairy", dossier, session_id="test_sess_1", store_dir=store_dir)
                self.assertEqual(alg["intent"], "allergy")
                self.assertTrue(alg["plan_flagged_unsuitable"])
                self.assertIn("NOT allergen-free", alg["answer"])
                rec_alg = plan_store.get_session("test_sess_1", store_dir=store_dir)
                self.assertIn("dairy", rec_alg["preferences"]["reported_allergies"])
                self.assertIn("dairy", plan_store.get_current_version_entry(rec_alg)["flagged_unsuitable_for_allergies"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
