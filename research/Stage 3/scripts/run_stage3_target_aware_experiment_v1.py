#!/usr/bin/env python3
"""Stage 3 Target-Aware Experiment Runner (v1).

Evaluates the experimental target-aware Stage 3 optimizer (Variant B) against
the current bounds-based Stage 3 optimizer (Variant A) across a matched cohort of
100 participants x 3 Stage 2 arms x 3 dietary modes (900 paired tasks = 1,800 solves).

Arms:
1. rules_only: Deterministic clinical reference target vector
2. minimum_change: Scale-normalized QP projection of observed intake
3. model_guided: 50-step Adam search + scale-normalized QP projection

Dietary Modes:
1. pure_vegetarian
2. veg_nonveg_combo
3. pure_non_vegetarian

Solver Variants:
A. Current Objective: Calorie deviation + food selection cost
B. Target-Aware Objective: Calorie deviation + food selection cost + weighted normalized L1 target deviations
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import stage3_milp_solver as s3
import stage3_target_aware_solver as tas
import stage2_stage3_adapter as adapter
import stage2_counterfactual_engine as s2
import stage1_deployment as deployment
import nutrient_rule_policy as nrp

STUDY_NAME = "stage3_target_aware_experiment_v1"
STUDY_DIR = ROOT / "Saved Model" / STUDY_NAME
CHECKPOINT_DIR = STUDY_DIR / "checkpoints"
V1_TRANSLATION_DIR = ROOT / "Saved Model" / "stage2_stage3_translation_study_v1" / "checkpoints"
TIME_LIMIT_SEC = 3.0
MAX_WORKERS = min(10, os.cpu_count() or 1)
TARGET_WEIGHT = 1.0


def sha(path: Path) -> str:
    """Compute SHA-256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute_experiment_fingerprint(
    case_idx: int,
    seqn: int,
    arm: str,
    dietary_mode: str,
    stage2_input: dict[str, Any],
    catalog_hash: str,
    time_limit_sec: float = TIME_LIMIT_SEC,
    target_weight: float = TARGET_WEIGHT,
) -> str:
    """Compute deterministic canonical SHA-256 fingerprint for a paired experiment task."""
    envelope_rows = []
    for r in stage2_input.get("nutrient_constraint_envelope", []):
        envelope_rows.append({
            "nutrient": str(r["nutrient"]),
            "target": float(r["target"]),
            "min_bound": float(r["min_bound"]),
            "max_bound": float(r["max_bound"]),
            "baseline": float(r["baseline"]),
            "rule_provenance": r.get("rule_provenance", {}),
        })
    envelope_rows.sort(key=lambda x: x["nutrient"])

    payload = {
        "protocol_version": STUDY_NAME,
        "case_idx": int(case_idx),
        "SEQN": int(seqn),
        "arm": str(arm),
        "dietary_mode": str(dietary_mode),
        "patient_metadata": stage2_input.get("patient_metadata", {}),
        "energy_prescription": stage2_input.get("energy_prescription", {}),
        "nutrient_constraint_envelope": envelope_rows,
        "bindings": {
            "catalog_hash": str(catalog_hash),
            "nutrient_ruleset_version": stage2_input.get("nutrient_ruleset_version", nrp.RULESET_VERSION),
        },
        "solver_settings": {
            "time_limit_sec": float(time_limit_sec),
            "target_weight": float(target_weight),
            "slack_policy": "zero_active_slack",
            "variant_a_objective": "calorie_deviation_plus_food_selection",
            "variant_b_objective": "calorie_deviation_plus_food_selection_plus_l1_target_deviation",
        },
    }
    adapter.validate_finite(payload)
    dumped = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=s3._json_safe)
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()


def freeze_protocol(catalog_hash: str) -> dict[str, Any]:
    """Freeze immutable study protocol and record cryptographic source digests."""
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    cases_source = ROOT / "Saved Model/stage2_comparison_study_v2/cases_LOCAL.json"
    cases_local = STUDY_DIR / "cases_LOCAL.json"
    if not cases_local.exists():
        cases_local.write_bytes(cases_source.read_bytes())

    bound_sources = [
        "food_composition_cooked_sourced.json",
        "stage3_milp_solver.py",
        "stage3_target_aware_solver.py",
        "test_stage3_target_aware.py",
        "stage2_stage3_adapter.py",
        "stage2_counterfactual_engine.py",
        "nutrient_rule_policy.py",
        "stage1_deployment.py",
        "run_stage3_target_aware_experiment_v1.py",
        "Saved Model/stage2_comparison_study_v2/cases_LOCAL.json",
        "Saved Model/stage2_comparison_study_v2/summary.json",
    ]

    source_hashes = {}
    for rel_path in bound_sources:
        p = ROOT / rel_path
        if not p.exists():
            raise FileNotFoundError(f"Missing bound source: {p}")
        source_hashes[rel_path] = sha(p)

    protocol = {
        "protocol_version": STUDY_NAME,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "cohort_size": 100,
        "cases_sha256": sha(cases_local),
        "catalog_hash": catalog_hash,
        "arms": ["rules_only", "minimum_change", "model_guided"],
        "dietary_modes": ["pure_vegetarian", "veg_nonveg_combo", "pure_non_vegetarian"],
        "total_planned_paired_tasks": 900,
        "total_solves": 1800,
        "time_limit_sec": TIME_LIMIT_SEC,
        "target_weight": TARGET_WEIGHT,
        "max_workers": MAX_WORKERS,
        "device": "cpu",
        "solver_backend": "scipy.optimize.milp (HiGHS)",
        "hashes": source_hashes,
        "normalization_scales": tas.DAILY_REFERENCE_SCALES,
        "nutrient_units": tas.NUTRIENT_UNITS,
        "primary_hypotheses": {
            "h1_feasibility": "Variant B maintains identical discrete feasibility to Variant A within statistical equivalence tolerances.",
            "h2_target_attainment": "Variant B significantly reduces normalized L1 target error compared to Variant A without violating hard bounds.",
            "h3_computational_cost": "Variant B achieves target tracking within the 3.0s limit without solver timeout degradation.",
        },
        "statistical_testing": {
            "binary_discordance_test": "Exact two-sided paired binomial test per dietary mode",
            "pooled_binary_uncertainty": "Participant-clustered bootstrap (10,000 resamples, seed 20261123)",
            "continuous_paired_contrasts": "Paired mean difference bootstrap percentile 95% CI (10,000 resamples)",
        },
    }

    protocol_path = STUDY_DIR / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    # Generate human-readable Protocol.md
    md_lines = [
        f"# Protocol: {STUDY_NAME}",
        "",
        "## Overview",
        "Controlled engineering evaluation comparing the experimental target-aware Stage 3 optimizer (Variant B)",
        "against the current bounds-based Stage 3 optimizer (Variant A) across 100 participants x 3 Stage 2 arms x 3 dietary modes.",
        "",
        "## Key Mathematical Invariants",
        "- Identical discrete feasible polytope: Proj_{(q,z)}(X_B) = X_A.",
        "- Zero-slack invariant: Deviation variables cannot purchase violations of hard component bounds.",
        "- Energy conservation: Calorie target penalized once via cal_over/cal_under; excluded from d_k^+/d_k^-.",
        "- Normalized L1 goal programming: Standard FDA Daily Reference Scales s_k with positive, finite units.",
        "",
        "## Study Dimensions",
        f"- Cohort: 100 adult temporal NHANES participants (disjoint from pilot and training sets)",
        f"- Arms: Rules-Only (RO), Minimum-Change (MC), Model-Guided (MG)",
        f"- Dietary Modes: Pure Vegetarian, Veg/Non-Veg Combo, Pure Non-Vegetarian",
        f"- Total Paired Tasks: 900 (1,800 total solves)",
        f"- Time Limit: {TIME_LIMIT_SEC} seconds per solve",
        f"- Target Weight (lambda): {TARGET_WEIGHT}",
        "",
        "## Primary Metrics",
        "1. Discrete Meal Feasibility Rate (% accepted by independent verifier)",
        "2. Mean Normalized L1 Target Error (NTE_L1)",
        "3. Normalized L2 Target Distance (NTD_L2)",
        "4. Core Nutrients MAPE (%)",
        "5. Solve Time (seconds) and MIP Gap",
        "6. Food Basket Composition (dish count, total servings, total mass)",
    ]
    (STUDY_DIR / "Protocol.md").write_text("\n".join(md_lines), encoding="utf-8")

    return protocol


def execute_paired_task(
    task_args: tuple[int, int, str, str, dict[str, Any], bool, str, str]
) -> dict[str, Any]:
    """Execute a single paired task (Variant A and Variant B) for one participant, arm, and mode."""
    case_idx, seqn, arm, mode, s3_input, s2_accepted, cat_hash, cp_str = task_args
    cp_path = Path(cp_str)

    expected_fp = compute_experiment_fingerprint(
        case_idx=case_idx,
        seqn=seqn,
        arm=arm,
        dietary_mode=mode,
        stage2_input=s3_input,
        catalog_hash=cat_hash,
        time_limit_sec=TIME_LIMIT_SEC,
        target_weight=TARGET_WEIGHT,
    )

    # Resume check
    if cp_path.exists():
        try:
            saved = json.loads(cp_path.read_text(encoding="utf-8"))
            if saved.get("fingerprint") == expected_fp:
                return {
                    "case_idx": case_idx,
                    "SEQN": seqn,
                    "arm": arm,
                    "dietary_mode": mode,
                    "stage2_accepted": s2_accepted,
                    "variant_a_accepted": saved.get("variant_a", {}).get("accepted", False),
                    "variant_b_accepted": saved.get("variant_b", {}).get("accepted", False),
                    "variant_a_replay_passed": saved.get("variant_a", {}).get("replay_passed", False),
                    "variant_b_replay_passed": saved.get("variant_b", {}).get("replay_passed", False),
                    "variant_a_time": saved.get("variant_a", {}).get("solve_time_seconds", 0.0),
                    "variant_b_time": saved.get("variant_b", {}).get("solve_time_seconds", 0.0),
                    "variant_a_l1_error": saved.get("variant_a", {}).get("target_attainment", {}).get("mean_normalized_l1_target_error"),
                    "variant_b_l1_error": saved.get("variant_b", {}).get("target_attainment", {}).get("mean_normalized_l1_target_error"),
                    "variant_a_l2_dist": saved.get("variant_a", {}).get("target_attainment", {}).get("normalized_l2_target_distance"),
                    "variant_b_l2_dist": saved.get("variant_b", {}).get("target_attainment", {}).get("normalized_l2_target_distance"),
                    "resumed": True,
                }
        except Exception:
            pass

    foods = s3.load_food_database()

    # --- Variant A: Current Bounds-Based Solver ---
    t0_a = time.perf_counter()
    out_a = tas.solve_experimental_meal_plan(
        stage2_input=s3_input,
        food_db=foods,
        time_limit_sec=TIME_LIMIT_SEC,
        revision_spec={"dietary_mode": mode},
        objective_variant="current",
    )
    elapsed_a = time.perf_counter() - t0_a
    replay_a = adapter.replay_stage3_solution(
        basket=out_a.solved_food_basket,
        foods=foods,
        stage2_input=s3_input,
        dietary_mode=mode,
    )

    # --- Variant B: Target-Aware Solver ---
    t0_b = time.perf_counter()
    out_b = tas.solve_experimental_meal_plan(
        stage2_input=s3_input,
        food_db=foods,
        time_limit_sec=TIME_LIMIT_SEC,
        revision_spec={"dietary_mode": mode},
        objective_variant="target_aware",
        target_weight=TARGET_WEIGHT,
    )
    elapsed_b = time.perf_counter() - t0_b
    replay_b = adapter.replay_stage3_solution(
        basket=out_b.solved_food_basket,
        foods=foods,
        stage2_input=s3_input,
        dietary_mode=mode,
    )

    # Paired metrics and basket diff
    basket_diff = s3.compute_plan_diff(out_a.solved_food_basket, out_b.solved_food_basket)
    basket_identical = (basket_diff["total_serving_delta"] == 0 and
                        len(basket_diff["added"]) == 0 and
                        len(basket_diff["removed"]) == 0 and
                        len(basket_diff["quantity_changed"]) == 0)

    attain_a = out_a.solver_metadata.get("target_attainment", {})
    attain_b = out_b.solver_metadata.get("target_attainment", {})

    delta_l1 = None
    delta_l2 = None
    if attain_a and attain_b:
        l1_a = attain_a.get("mean_normalized_l1_target_error")
        l1_b = attain_b.get("mean_normalized_l1_target_error")
        if l1_a is not None and l1_b is not None:
            delta_l1 = round(l1_b - l1_a, 6)

        l2_a = attain_a.get("normalized_l2_target_distance")
        l2_b = attain_b.get("normalized_l2_target_distance")
        if l2_a is not None and l2_b is not None:
            delta_l2 = round(l2_b - l2_a, 6)

    record = {
        "case_idx": case_idx,
        "SEQN": seqn,
        "arm": arm,
        "dietary_mode": mode,
        "fingerprint": expected_fp,
        "stage2_accepted": s2_accepted,
        "paired_comparison": {
            "basket_identical": basket_identical,
            "serving_delta": basket_diff["total_serving_delta"],
            "delta_mean_normalized_l1": delta_l1,
            "delta_normalized_l2": delta_l2,
            "delta_solve_time_seconds": round(elapsed_b - elapsed_a, 4),
            "basket_diff": basket_diff,
        },
        "variant_a": {
            "accepted": bool(out_a.accepted),
            "replay_passed": bool(replay_a.get("passed", False)),
            "end_to_end_accepted": bool(s2_accepted and out_a.accepted and replay_a.get("passed", False)),
            "solver_status": out_a.solver_metadata.get("status"),
            "solver_gap": out_a.solver_metadata.get("mip_gap"),
            "solve_time_seconds": round(elapsed_a, 4),
            "food_basket": out_a.solved_food_basket,
            "delivered_nutrition": out_a.delivered_nutrition,
            "replay_audit": replay_a,
            "target_attainment": attain_a,
            "cost_breakdown": out_a.solver_metadata.get("cost_breakdown"),
        },
        "variant_b": {
            "accepted": bool(out_b.accepted),
            "replay_passed": bool(replay_b.get("passed", False)),
            "end_to_end_accepted": bool(s2_accepted and out_b.accepted and replay_b.get("passed", False)),
            "solver_status": out_b.solver_metadata.get("status"),
            "solver_gap": out_b.solver_metadata.get("mip_gap"),
            "solve_time_seconds": round(elapsed_b, 4),
            "food_basket": out_b.solved_food_basket,
            "delivered_nutrition": out_b.delivered_nutrition,
            "replay_audit": replay_b,
            "target_attainment": attain_b,
            "cost_breakdown": out_b.solver_metadata.get("cost_breakdown"),
        },
    }

    # Save checkpoint atomically
    temp_cp = cp_path.with_suffix(".tmp")
    temp_cp.write_text(json.dumps(record, indent=2, default=s3._json_safe), encoding="utf-8")
    temp_cp.replace(cp_path)

    return {
        "case_idx": case_idx,
        "SEQN": seqn,
        "arm": arm,
        "dietary_mode": mode,
        "stage2_accepted": s2_accepted,
        "variant_a_accepted": bool(out_a.accepted),
        "variant_b_accepted": bool(out_b.accepted),
        "variant_a_replay_passed": bool(replay_a.get("passed", False)),
        "variant_b_replay_passed": bool(replay_b.get("passed", False)),
        "variant_a_time": round(elapsed_a, 4),
        "variant_b_time": round(elapsed_b, 4),
        "variant_a_l1_error": attain_a.get("mean_normalized_l1_target_error"),
        "variant_b_l1_error": attain_b.get("mean_normalized_l1_target_error"),
        "variant_a_l2_dist": attain_a.get("normalized_l2_target_distance"),
        "variant_b_l2_dist": attain_b.get("normalized_l2_target_distance"),
        "basket_identical": basket_identical,
        "resumed": False,
    }


def run_experiment():
    """Main execution orchestrator."""
    print("=" * 70)
    print(f"Starting {STUDY_NAME} (900 paired tasks = 1,800 solves)")
    print(f"Max workers: {MAX_WORKERS}, Time limit: {TIME_LIMIT_SEC}s, Target weight: {TARGET_WEIGHT}")
    print("=" * 70)

    # 1. Load catalog and compute hash
    foods = s3.load_food_database()
    cat_path = ROOT / "food_composition_cooked_sourced.json"
    cat_hash = sha(cat_path)
    print(f"Loaded food catalog: {len(foods)} dishes (hash: {cat_hash[:16]}...)")

    # 2. Freeze protocol
    protocol = freeze_protocol(cat_hash)
    print(f"Protocol frozen to {STUDY_DIR / 'protocol.json'}")

    # 3. Load cases and Stage 2 checkpoints
    cases = json.load(open(STUDY_DIR / "cases_LOCAL.json", encoding="utf-8"))
    cp_dir = ROOT / "Saved Model/stage2_comparison_study_v2/checkpoints"
    artifacts = s2._load_preprocessing_artifacts(deployment.PREPROCESSOR)

    arms = ["rules_only", "minimum_change", "model_guided"]
    modes = ["pure_vegetarian", "veg_nonveg_combo", "pure_non_vegetarian"]

    tasks = []
    print("Preparing 900 paired tasks from frozen cases and checkpoints...")
    for c in cases:
        case_idx = c["case"]
        seqn = c["SEQN"]
        meta = c["metadata"]

        w = meta["weight_kg"]
        h = meta["height_cm"]
        age = meta["age_years"]
        is_fem = meta["is_female"]
        bmr = 10.0 * w + 6.25 * h - 5.0 * age + (-161.0 if is_fem else 5.0)
        tdee = bmr * s2.PAL_FACTORS[meta["activity_level"]]

        ro_cp = json.load(open(cp_dir / f"case_{case_idx:03d}_{seqn}_rules_only.json", encoding="utf-8"))
        p_d0 = ro_cp["baseline_probabilities"]["diabetes"]
        p_o0 = ro_cp["baseline_probabilities"]["obesity"]
        p_c0 = ro_cp["baseline_probabilities"]["cvd"]
        t_kcal = max(tdee - 500.0, 1200.0) if p_o0 >= s2.CUTOFF_OBESITY else max(tdee, 1200.0)

        lower, upper = s2._build_nutrient_bounds(
            artifacts=artifacts, t_kcal=t_kcal, weight_kg=w, is_female=is_fem, age_years=age,
            p_diab_0=p_d0, p_obes_0=p_o0, p_cvd_0=p_c0
        )
        n_obs = ro_cp["observed_nutrients"]

        for arm in arms:
            arm_cp = json.load(open(cp_dir / f"case_{case_idx:03d}_{seqn}_{arm}.json", encoding="utf-8"))
            s2_accepted = bool(arm_cp["overall_accepted"])
            tgt_vec = arm_cp["target_nutrients"]

            envelope = []
            for i, code in enumerate(s3.NUTRIENT_CODES):
                envelope.append({
                    "nutrient": code,
                    "target": float(tgt_vec[i]),
                    "min_bound": float(lower[i]),
                    "max_bound": float(upper[i]),
                    "baseline": float(n_obs[i]),
                    "rule_provenance": nrp.reviewed_rule_metadata(code, float(lower[i]), float(upper[i]))
                })

            s3_input = {
                "patient_metadata": dict(meta),
                "energy_prescription": {"BMR": float(bmr), "TDEE": float(tdee), "target_calories": float(t_kcal)},
                "nutrient_constraint_envelope": envelope,
            }

            for mode in modes:
                cp_file = CHECKPOINT_DIR / f"case_{case_idx:03d}_{seqn}_{arm}_{mode}.json"
                task_tuple = (case_idx, seqn, arm, mode, s3_input, s2_accepted, cat_hash, str(cp_file))
                tasks.append(task_tuple)

    print(f"Total paired tasks prepared: {len(tasks)}")

    # 4. Execute tasks in parallel
    print(f"Executing tasks with ProcessPoolExecutor (max_workers={MAX_WORKERS})...")
    start_total = time.perf_counter()
    results = []
    resumed_count = 0

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {executor.submit(execute_paired_task, t): t for t in tasks}
        completed_count = 0
        for future in as_completed(future_map):
            completed_count += 1
            res = future.result()
            results.append(res)
            if res.get("resumed"):
                resumed_count += 1
            if completed_count % 100 == 0 or completed_count == len(tasks):
                elapsed = time.perf_counter() - start_total
                rate = completed_count / elapsed if elapsed > 0 else 0
                print(f"[{completed_count:03d}/{len(tasks)}] completed ({rate:.1f} tasks/sec, {resumed_count} resumed)...")

    total_time = time.perf_counter() - start_total
    print("=" * 70)
    print(f"Execution complete in {total_time:.2f} seconds ({resumed_count} resumed, {len(tasks) - resumed_count} executed fresh).")
    print("=" * 70)

    # 5. Save execution summary
    summary_path = STUDY_DIR / "execution_summary.json"
    summary_data = {
        "protocol_version": STUDY_NAME,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "total_paired_tasks": len(results),
        "total_solves": len(results) * 2,
        "resumed_count": resumed_count,
        "wall_clock_time_seconds": round(total_time, 2),
        "workers_used": MAX_WORKERS,
    }
    summary_path.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    run_experiment()
