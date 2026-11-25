#!/usr/bin/env python3
"""Stage 3 Controlled Solver-Budget Experiment Runner.

Evaluates Variant A (current bounds-based) vs Variant B (target-aware)
across 100 participants x 3 Stage 2 arms x 3 dietary modes x 3 budgets (3.0s, 5.0s, 10.0s).
Total: 2,700 paired tasks (5,400 optimization solves).

Saves atomic checkpoints in:
Saved Model/stage3_budget_experiment_v1/checkpoints/task_{case:03d}_{seqn}_{arm}_{mode}_{budget}s.json
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
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
import stage3_budget_solver as rbs
import stage2_stage3_adapter as adapter
import nutrient_rule_policy as nrp

STUDY_NAME = "stage3_budget_experiment_v1"
STUDY_DIR = ROOT / "Saved Model" / STUDY_NAME
CHECKPOINT_DIR = STUDY_DIR / "checkpoints"
TARGETS_DIR = STUDY_DIR / "stage2_targets"

BUDGETS_SEC = [3.0, 5.0, 10.0]
TARGET_WEIGHT = 1.0
MAX_WORKERS = min(10, os.cpu_count() or 1)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute_budget_experiment_fingerprint(
    case_idx: int,
    seqn: int,
    arm: str,
    dietary_mode: str,
    budget_sec: float,
    stage2_input: dict[str, Any],
    catalog_hash: str,
    target_weight: float = TARGET_WEIGHT,
    protocol_version: str = STUDY_NAME,
) -> str:
    """Compute canonical SHA-256 fingerprint for a single paired budget task."""
    envelope_rows = []
    for r in stage2_input.get("nutrient_constraint_envelope", []):
        envelope_rows.append((
            r["nutrient"],
            round(float(r["target"]), 4),
            round(float(r["min_bound"]), 4),
            round(float(r["max_bound"]), 4),
            round(float(r["baseline"]), 4),
        ))
    envelope_rows.sort(key=lambda x: x[0])

    payload = {
        "protocol_version": protocol_version,
        "case_idx": int(case_idx),
        "SEQN": int(seqn),
        "arm": arm,
        "dietary_mode": dietary_mode,
        "budget_sec": round(float(budget_sec), 2),
        "target_weight": round(float(target_weight), 4),
        "catalog_hash": catalog_hash,
        "envelope": envelope_rows,
        "energy_prescription": {
            "target_calories": round(float(stage2_input["energy_prescription"]["target_calories"]), 2),
            "BMR": round(float(stage2_input["energy_prescription"]["BMR"]), 2),
            "TDEE": round(float(stage2_input["energy_prescription"]["TDEE"]), 2),
        },
        "patient_metadata": {
            k: stage2_input["patient_metadata"][k]
            for k in sorted(stage2_input["patient_metadata"])
        },
        "invariants": {
            "slack_policy": "zero_active_slack",
            "variant_a_objective": "calorie_deviation_plus_food_selection",
            "variant_b_objective": "calorie_deviation_plus_food_selection_plus_l1_target_deviation",
        },
    }
    adapter.validate_finite(payload)
    dumped = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=s3._json_safe)
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()


def write_json_atomic(path: Path, data: Any) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=s3._json_safe), encoding="utf-8")
    tmp.replace(path)


def execute_paired_budget_task(
    task_args: tuple[int, int, str, str, float, dict[str, Any], bool, str, str]
) -> dict[str, Any]:
    """Execute a single paired budget task (Variant A and Variant B) under independent cold starts."""
    case_idx, seqn, arm, mode, budget_sec, s3_input, s2_accepted, cat_hash, cp_str = task_args
    cp_path = Path(cp_str)

    expected_fp = compute_budget_experiment_fingerprint(
        case_idx=case_idx,
        seqn=seqn,
        arm=arm,
        dietary_mode=mode,
        budget_sec=budget_sec,
        stage2_input=s3_input,
        catalog_hash=cat_hash,
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
                    "budget_sec": budget_sec,
                    "stage2_accepted": s2_accepted,
                    "variant_a_accepted": saved.get("variant_a", {}).get("accepted", False),
                    "variant_b_accepted": saved.get("variant_b", {}).get("accepted", False),
                    "variant_a_replay_passed": saved.get("variant_a", {}).get("replay_passed", False),
                    "variant_b_replay_passed": saved.get("variant_b", {}).get("replay_passed", False),
                    "variant_a_time": saved.get("variant_a", {}).get("solve_time_seconds", 0.0),
                    "variant_b_time": saved.get("variant_b", {}).get("solve_time_seconds", 0.0),
                    "variant_a_l1_error": saved.get("variant_a", {}).get("target_attainment", {}).get("mean_normalized_l1_target_error"),
                    "variant_b_l1_error": saved.get("variant_b", {}).get("target_attainment", {}).get("mean_normalized_l1_target_error"),
                    "resumed": True,
                }
        except Exception:
            pass

    foods = s3.load_food_database()

    # --- Variant A: Current Bounds-Based Solver ---
    t0_a = time.perf_counter()
    out_a = rbs.solve_budget_meal_plan(
        stage2_input=s3_input,
        food_db=foods,
        time_limit_sec=budget_sec,
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
    out_b = rbs.solve_budget_meal_plan(
        stage2_input=s3_input,
        food_db=foods,
        time_limit_sec=budget_sec,
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

    attain_a = out_a.solver_metadata.get("target_attainment", {})
    attain_b = out_b.solver_metadata.get("target_attainment", {})

    delta_l1 = None
    delta_l2 = None
    basket_identical = None

    if out_a.accepted and out_b.accepted:
        basket_identical = (
            basket_diff["total_serving_delta"] == 0
            and len(basket_diff["added"]) == 0
            and len(basket_diff["removed"]) == 0
            and len(basket_diff["quantity_changed"]) == 0
        )
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
        "budget_seconds": budget_sec,
        "fingerprint": expected_fp,
        "stage2_accepted": s2_accepted,
        "paired_comparison": {
            "jointly_accepted": bool(out_a.accepted and out_b.accepted),
            "discordant_a_only": bool(out_a.accepted and not out_b.accepted),
            "discordant_b_only": bool(not out_a.accepted and out_b.accepted),
            "jointly_rejected": bool(not out_a.accepted and not out_b.accepted),
            "basket_identical_if_joint": basket_identical,
            "delta_mean_normalized_l1_target_error": delta_l1,
            "delta_normalized_l2_target_distance": delta_l2,
            "solve_time_delta_seconds": round(elapsed_b - elapsed_a, 4),
            "basket_diff": basket_diff,
        },
        "variant_a": {
            "accepted": out_a.accepted,
            "replay_passed": replay_a["passed"],
            "end_to_end_accepted": bool(s2_accepted and out_a.accepted and replay_a["passed"]),
            "solver_status": out_a.solver_metadata.get("status"),
            "solver_status_code": out_a.solver_metadata.get("solver_status_code"),
            "status_message": out_a.solver_metadata.get("status_message"),
            "solver_gap": out_a.solver_metadata.get("mip_gap"),
            "solve_time_seconds": round(elapsed_a, 4),
            "requested_time_limit_sec": out_a.solver_metadata.get("requested_time_limit_sec"),
            "effective_time_limit_sec": out_a.solver_metadata.get("effective_time_limit_sec"),
            "target_calories": out_a.solver_metadata.get("target_calories"),
            "delivered_calories": out_a.solver_metadata.get("delivered_calories"),
            "active_dish_count": out_a.solver_metadata.get("active_dish_count"),
            "total_servings": out_a.solver_metadata.get("total_servings"),
            "food_basket": out_a.solved_food_basket,
            "delivered_nutrition": out_a.delivered_nutrition,
            "replay_audit": replay_a,
            "target_attainment": attain_a,
            "cost_breakdown": out_a.solver_metadata.get("cost_breakdown"),
        },
        "variant_b": {
            "accepted": out_b.accepted,
            "replay_passed": replay_b["passed"],
            "end_to_end_accepted": bool(s2_accepted and out_b.accepted and replay_b["passed"]),
            "solver_status": out_b.solver_metadata.get("status"),
            "solver_status_code": out_b.solver_metadata.get("solver_status_code"),
            "status_message": out_b.solver_metadata.get("status_message"),
            "solver_gap": out_b.solver_metadata.get("mip_gap"),
            "solve_time_seconds": round(elapsed_b, 4),
            "requested_time_limit_sec": out_b.solver_metadata.get("requested_time_limit_sec"),
            "effective_time_limit_sec": out_b.solver_metadata.get("effective_time_limit_sec"),
            "target_calories": out_b.solver_metadata.get("target_calories"),
            "delivered_calories": out_b.solver_metadata.get("delivered_calories"),
            "active_dish_count": out_b.solver_metadata.get("active_dish_count"),
            "total_servings": out_b.solver_metadata.get("total_servings"),
            "food_basket": out_b.solved_food_basket,
            "delivered_nutrition": out_b.delivered_nutrition,
            "replay_audit": replay_b,
            "target_attainment": attain_b,
            "cost_breakdown": out_b.solver_metadata.get("cost_breakdown"),
        },
    }

    write_json_atomic(cp_path, record)

    return {
        "case_idx": case_idx,
        "SEQN": seqn,
        "arm": arm,
        "dietary_mode": mode,
        "budget_sec": budget_sec,
        "stage2_accepted": s2_accepted,
        "variant_a_accepted": out_a.accepted,
        "variant_b_accepted": out_b.accepted,
        "variant_a_replay_passed": replay_a["passed"],
        "variant_b_replay_passed": replay_b["passed"],
        "variant_a_time": round(elapsed_a, 4),
        "variant_b_time": round(elapsed_b, 4),
        "variant_a_l1_error": attain_a.get("mean_normalized_l1_target_error"),
        "variant_b_l1_error": attain_b.get("mean_normalized_l1_target_error"),
        "resumed": False,
    }


def main():
    print("=" * 75)
    print("Stage 3 Controlled Solver-Budget Experiment Runner")
    print(f"Cohort: 100 participants | Arms: 3 | Modes: 3 | Budgets: {BUDGETS_SEC}")
    print(f"Total Paired Tasks: 2,700 (5,400 optimization solves)")
    print("=" * 75)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    protocol_path = STUDY_DIR / "protocol.json"
    assert protocol_path.exists(), f"Missing frozen protocol: {protocol_path}"
    cases_path = STUDY_DIR / "cases_LOCAL.json"
    assert cases_path.exists(), f"Missing cohort manifest: {cases_path}"
    catalog_path = ROOT / "food_composition_cooked_sourced.json"
    assert catalog_path.exists(), f"Missing food catalog: {catalog_path}"

    cat_hash = sha(catalog_path)
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    assert len(cases) == 100

    arms = ["rules_only", "minimum_change", "model_guided"]
    modes = ["pure_vegetarian", "veg_nonveg_combo", "pure_non_vegetarian"]

    tasks = []
    for case in cases:
        case_idx = case["case"]
        seqn = case["SEQN"]
        meta = case["metadata"]

        for arm in arms:
            tgt_path = TARGETS_DIR / f"case_{case_idx:03d}_{seqn}_{arm}.json"
            assert tgt_path.exists(), f"Missing Stage 2 target checkpoint: {tgt_path}"
            tgt_data = json.loads(tgt_path.read_text(encoding="utf-8"))
            s2_accepted = bool(tgt_data["overall_accepted"])

            envelope = []
            for i, code in enumerate(s3.NUTRIENT_CODES):
                envelope.append({
                    "nutrient": code,
                    "target": float(tgt_data["target_nutrients"][i]),
                    "min_bound": float(tgt_data["lower_bounds"][i]),
                    "max_bound": float(tgt_data["upper_bounds"][i]),
                    "baseline": float(tgt_data["observed_nutrients"][i]),
                    "rule_provenance": nrp.reviewed_rule_metadata(
                        code, float(tgt_data["lower_bounds"][i]), float(tgt_data["upper_bounds"][i])
                    ),
                })

            s3_input = {
                "patient_metadata": dict(meta),
                "energy_prescription": dict(tgt_data["energy_prescription"]),
                "nutrient_constraint_envelope": envelope,
            }

            for mode in modes:
                for b_sec in BUDGETS_SEC:
                    b_int = int(b_sec)
                    cp_name = f"task_{case_idx:03d}_{seqn}_{arm}_{mode}_{b_int}s.json"
                    cp_path = CHECKPOINT_DIR / cp_name
                    tasks.append((case_idx, seqn, arm, mode, b_sec, s3_input, s2_accepted, cat_hash, str(cp_path)))

    assert len(tasks) == 2700, f"Expected 2,700 paired tasks, got {len(tasks)}"
    print(f"Total paired tasks assembled: {len(tasks)}")

    # Randomized execution order with fixed seed to balance CPU/thermal load
    random.Random(20261124).shuffle(tasks)

    print(f"Executing tasks in parallel with ProcessPoolExecutor (max_workers={MAX_WORKERS})...")
    t_start = time.perf_counter()
    completed = 0
    resumed_count = 0

    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {executor.submit(execute_paired_budget_task, t): t for t in tasks}
        for future in as_completed(future_to_task):
            res = future.result()
            results.append(res)
            completed += 1
            if res.get("resumed"):
                resumed_count += 1

            if completed % 100 == 0 or completed == len(tasks):
                elapsed = time.perf_counter() - t_start
                rate = completed / elapsed if elapsed > 0 else 0.0
                print(
                    f"[{completed:04d}/{len(tasks):04d}] Completed | "
                    f"Elapsed: {elapsed:.1f}s | Rate: {rate:.1f} tasks/s | "
                    f"Resumed: {resumed_count}",
                    flush=True,
                )

    total_time = time.perf_counter() - t_start
    print("=" * 75)
    print(f"All 2,700 Paired Tasks (5,400 Solves) Completed in {total_time:.2f} seconds!")
    print(f"Freshly solved: {completed - resumed_count} | Resumed: {resumed_count}")
    print("=" * 75)


if __name__ == "__main__":
    main()
