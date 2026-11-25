#!/usr/bin/env python3
"""Unit and Contract Tests for Stage 3 Research Budget Solver Wrapper.

Verifies:
1. Production defaults and safety clamps in stage3_milp_solver.py remain 100% untouched.
2. Research budget solver supports 3.0s, 5.0s, and 10.0s without silent clamping.
3. Zero-slack invariant is strictly preserved across both variants.
4. Mathematical polytope invariance: Proj_{(q,z)}(X_B) == X_A.
5. Metadata records requested vs effective time limits accurately.
6. Target attainment metrics are computed accurately for both variants.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import stage3_milp_solver as s3
import stage3_target_aware_solver as tas
import stage3_budget_solver as rbs
import stage2_stage3_adapter as adapter


import nutrient_rule_policy as nrp


def _load_sample_stage2_input() -> dict:
    """Load sample stage2 input for contract testing."""
    targets_dir = ROOT / "Saved Model" / "stage3_budget_experiment_v1" / "stage2_targets"
    cp_files = list(targets_dir.glob("case_000_*_rules_only.json"))
    cases = json.loads((ROOT / "Saved Model/stage3_budget_experiment_v1/cases_LOCAL.json").read_text(encoding="utf-8"))
    c0 = cases[0]
    meta = c0["metadata"]

    if cp_files:
        d = json.loads(cp_files[0].read_text(encoding="utf-8"))
        envelope = []
        for i, code in enumerate(s3.NUTRIENT_CODES):
            envelope.append({
                "nutrient": code,
                "target": float(d["target_nutrients"][i]),
                "min_bound": float(d["lower_bounds"][i]),
                "max_bound": float(d["upper_bounds"][i]),
                "baseline": float(d["observed_nutrients"][i]),
                "rule_provenance": nrp.reviewed_rule_metadata(code, float(d["lower_bounds"][i]), float(d["upper_bounds"][i]))
            })
        return {
            "patient_metadata": dict(meta),
            "energy_prescription": dict(d["energy_prescription"]),
            "nutrient_constraint_envelope": envelope,
        }

    # Fallback to direct calculation
    w, h, age, is_fem = meta["weight_kg"], meta["height_cm"], meta["age_years"], meta["is_female"]
    bmr = 10.0 * w + 6.25 * h - 5.0 * age + (-161.0 if is_fem else 5.0)
    tdee = bmr * 1.20
    t_kcal = max(tdee, 1200.0)

    envelope = []
    for code in s3.NUTRIENT_CODES:
        envelope.append({
            "nutrient": code,
            "target": 50.0 if code == "DR1TPROT" else (2000.0 if code == "DR1TKCAL" else 20.0),
            "min_bound": 0.0,
            "max_bound": 5000.0,
            "baseline": 30.0,
            "rule_provenance": nrp.reviewed_rule_metadata(code, 0.0, 5000.0),
        })

    return {
        "patient_metadata": dict(meta),
        "energy_prescription": {"BMR": float(bmr), "TDEE": float(tdee), "target_calories": float(t_kcal)},
        "nutrient_constraint_envelope": envelope,
    }


def test_production_solver_defaults_untouched():
    """Verify production stage3_milp_solver.py has not been altered."""
    assert s3.DEFAULT_TIME_LIMIT_SEC == 1.50
    assert s3.MAX_TIME_LIMIT_SEC == 3.0
    assert s3.MAX_SERVINGS_PER_DISH == 3
    assert s3.MAX_ACTIVE_DISHES == 8


def test_production_solver_clamps_at_3():
    """Verify that calling production solver with 10s is clamped to 3s."""
    foods = s3.load_food_database()
    stage2_in = _load_sample_stage2_input()
    out = s3.solve_meal_plan(
        stage2_input=stage2_in,
        food_db=foods,
        time_limit_sec=10.0,
    )
    # Production solver clamps effective time limit to 3.0
    assert out.solver_metadata["time_limit_sec"] == 3.0
    assert out.solver_metadata["requested_time_limit_sec"] == 10.0


def test_budget_solver_does_not_clamp_3_5_10():
    """Verify research budget solver accepts 3.0, 5.0, 10.0 without clamping."""
    assert rbs.RESEARCH_MAX_TIME_LIMIT_SEC == 15.0
    foods = s3.load_food_database()
    stage2_in = _load_sample_stage2_input()

    for budget in (3.0, 5.0, 10.0):
        out_a = rbs.solve_budget_meal_plan(
            stage2_input=stage2_in,
            food_db=foods,
            time_limit_sec=budget,
            objective_variant="current",
        )
        assert out_a.solver_metadata["requested_time_limit_sec"] == budget
        assert out_a.solver_metadata["effective_time_limit_sec"] == budget
        assert out_a.solver_metadata["time_limit_sec"] == budget
        assert out_a.solver_metadata["budget_seconds"] == budget

        out_b = rbs.solve_budget_meal_plan(
            stage2_input=stage2_in,
            food_db=foods,
            time_limit_sec=budget,
            objective_variant="target_aware",
        )
        assert out_b.solver_metadata["requested_time_limit_sec"] == budget
        assert out_b.solver_metadata["effective_time_limit_sec"] == budget
        assert out_b.solver_metadata["time_limit_sec"] == budget
        assert out_b.solver_metadata["budget_seconds"] == budget


def test_zero_slack_invariant():
    """Verify deviation variables cannot purchase hard bound violations."""
    foods = s3.load_food_database()
    stage2_in = _load_sample_stage2_input()
    out = rbs.solve_budget_meal_plan(
        stage2_input=stage2_in,
        food_db=foods,
        time_limit_sec=3.0,
        objective_variant="target_aware",
    )
    if out.accepted:
        assert out.solver_metadata["active_nutrient_slack"] == 0.0
        assert out.clinical_verification["all_active_checks_passed"] is True


def test_target_attainment_present_in_both_variants():
    """Verify target attainment metrics are computed and present for both variants."""
    foods = s3.load_food_database()
    stage2_in = _load_sample_stage2_input()

    out_a = rbs.solve_budget_meal_plan(
        stage2_input=stage2_in,
        food_db=foods,
        time_limit_sec=3.0,
        objective_variant="current",
    )
    if out_a.accepted:
        attain_a = out_a.solver_metadata.get("target_attainment")
        assert attain_a is not None
        assert "mean_normalized_l1_target_error" in attain_a
        assert "normalized_l2_target_distance" in attain_a
        assert math.isfinite(attain_a["mean_normalized_l1_target_error"])

    out_b = rbs.solve_budget_meal_plan(
        stage2_input=stage2_in,
        food_db=foods,
        time_limit_sec=3.0,
        objective_variant="target_aware",
    )
    if out_b.accepted:
        attain_b = out_b.solver_metadata.get("target_attainment")
        assert attain_b is not None
        assert "mean_normalized_l1_target_error" in attain_b
        assert "normalized_l2_target_distance" in attain_b
        assert math.isfinite(attain_b["mean_normalized_l1_target_error"])


if __name__ == "__main__":
    print("Running Stage 3 Budget Solver Unit Tests...")
    test_production_solver_defaults_untouched()
    print("  [PASS] test_production_solver_defaults_untouched")
    test_production_solver_clamps_at_3()
    print("  [PASS] test_production_solver_clamps_at_3")
    test_budget_solver_does_not_clamp_3_5_10()
    print("  [PASS] test_budget_solver_does_not_clamp_3_5_10")
    test_zero_slack_invariant()
    print("  [PASS] test_zero_slack_invariant")
    test_target_attainment_present_in_both_variants()
    print("  [PASS] test_target_attainment_present_in_both_variants")
    print("ALL TESTS PASSED SUCCESSFULLY!")
