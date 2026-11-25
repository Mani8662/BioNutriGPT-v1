"""Stage 2 -> Stage 3 Translation Study Runner.

Executes a 100-participant x 3-arm x 3-mode matched translation study (900 planned tasks)
using HiGHS MILP on discrete cooked food items.

Arms:
1. rules_only: Deterministic clinical reference target vector
2. minimum_change: Scale-normalized QP projection of observed intake
3. model_guided: 50-step Adam search + scale-normalized QP projection

Dietary Modes:
1. pure_vegetarian
2. veg_nonveg_combo
3. pure_non_vegetarian
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import stage3_milp_solver as s3
import stage2_stage3_adapter as adapter
import stage2_counterfactual_engine as s2
import stage1_deployment as deployment
import nutrient_rule_policy as nrp

STUDY_NAME = "stage2_stage3_translation_study_v1"
STUDY_DIR = ROOT / "Saved Model" / STUDY_NAME
CHECKPOINT_DIR = STUDY_DIR / "checkpoints"
TIME_LIMIT_SEC = 3.0
MAX_WORKERS = min(10, os.cpu_count() or 1)


def sha(path: Path) -> str:
    """Compute SHA-256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def solve_translation_task(
    task: tuple[int, int, str, str, dict[str, Any], bool, str, str]
) -> dict[str, Any]:
    """Top-level worker function for executing a single translation task.
    
    Parameters:
    task: (case_idx, seqn, arm, mode, s3_input, stage2_accepted, catalog_hash, checkpoint_path_str)
    """
    case_idx, seqn, arm, mode, s3_input, s2_accepted, cat_hash, cp_str = task
    cp_path = Path(cp_str)

    expected_fp = adapter.compute_translation_fingerprint(
        case_idx=case_idx,
        seqn=seqn,
        arm=arm,
        dietary_mode=mode,
        stage2_input=s3_input,
        catalog_hash=cat_hash,
        time_limit_sec=TIME_LIMIT_SEC,
    )

    # Check for valid resume
    if cp_path.exists():
        try:
            saved = json.loads(cp_path.read_text(encoding="utf-8"))
            if saved.get("fingerprint") == expected_fp:
                # Valid resume
                return {
                    "case_idx": case_idx,
                    "SEQN": seqn,
                    "arm": arm,
                    "dietary_mode": mode,
                    "stage2_accepted": s2_accepted,
                    "stage3_accepted": saved.get("stage3_accepted", False),
                    "end_to_end_accepted": saved.get("end_to_end_accepted", False),
                    "solver_status": saved.get("solver_status"),
                    "solve_time_seconds": saved.get("solve_time_seconds", 0.0),
                    "normalized_target_distance": saved.get("normalized_target_distance"),
                    "core_nutrients_mape": saved.get("core_nutrients_mape"),
                    "resumed": True,
                }
        except Exception:
            pass  # Corrupted cache, re-solve

    # Fresh solve
    foods = s3.load_food_database()
    t0 = time.perf_counter()
    error_msg = None
    s3_out_dict: dict[str, Any] = {}
    stage3_accepted = False
    solver_status = "error"
    gap = None
    replay: dict[str, Any] = {}
    soft_devs: dict[str, Any] = {}

    try:
        out = s3.solve_meal_plan(
            stage2_input=s3_input,
            food_db=foods,
            time_limit_sec=TIME_LIMIT_SEC,
            revision_spec={"dietary_mode": mode},
        )
        elapsed = time.perf_counter() - t0
        s3_out_dict = out.to_dict()
        stage3_accepted = bool(out.accepted)
        meta = out.solver_metadata or {}
        solver_status = str(meta.get("status", "unknown"))
        gap = meta.get("mip_gap")

        # Independent replay audit
        replay = adapter.replay_stage3_solution(
            basket=out.solved_food_basket,
            foods=foods,
            stage2_input=s3_input,
            dietary_mode=mode,
        )

        # Soft target attainment
        if out.accepted and replay.get("passed"):
            soft_devs = adapter.compute_soft_target_deviations(
                delivered_totals=replay["totals"],
                target_envelope=s3_input["nutrient_constraint_envelope"],
            )

    except Exception as exc:
        elapsed = time.perf_counter() - t0
        error_msg = repr(exc)
        solver_status = "exception"
        stage3_accepted = False
        replay = {"passed": False, "has_plan": False, "errors": [repr(exc)]}

    # End-to-end acceptance requires BOTH Stage 2 accepted and Stage 3 accepted & verified
    e2e_accepted = bool(s2_accepted and stage3_accepted and replay.get("passed", False))

    record = {
        "case_idx": case_idx,
        "SEQN": seqn,
        "arm": arm,
        "dietary_mode": mode,
        "fingerprint": expected_fp,
        "stage2_accepted": s2_accepted,
        "stage3_accepted": stage3_accepted,
        "stage3_replay_passed": bool(replay.get("passed", False)),
        "end_to_end_accepted": e2e_accepted,
        "solver_status": solver_status,
        "solver_gap": gap,
        "solve_time_seconds": round(elapsed, 4),
        "normalized_target_distance": soft_devs.get("normalized_target_distance"),
        "core_nutrients_mape": soft_devs.get("core_nutrients_mape"),
        "error": error_msg,
        "food_basket": s3_out_dict.get("solved_food_basket", []),
        "delivered_nutrition": s3_out_dict.get("delivered_nutrition", []),
        "replay_audit": replay,
        "soft_target_deviations": soft_devs,
        "solver_metadata": s3_out_dict.get("solver_metadata", {}),
    }

    # Atomic write
    tmp_path = cp_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(record, indent=2, default=s3._json_safe), encoding="utf-8")
    tmp_path.replace(cp_path)

    return {
        "case_idx": case_idx,
        "SEQN": seqn,
        "arm": arm,
        "dietary_mode": mode,
        "stage2_accepted": s2_accepted,
        "stage3_accepted": stage3_accepted,
        "end_to_end_accepted": e2e_accepted,
        "solver_status": solver_status,
        "solve_time_seconds": round(elapsed, 4),
        "normalized_target_distance": soft_devs.get("normalized_target_distance"),
        "core_nutrients_mape": soft_devs.get("core_nutrients_mape"),
        "resumed": False,
    }


def freeze_protocol(catalog_hash: str) -> dict[str, Any]:
    """Freeze study protocol and source hashes before execution."""
    cases_source = ROOT / "Saved Model/stage2_comparison_study_v2/cases_LOCAL.json"
    cases_local = STUDY_DIR / "cases_LOCAL.json"
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Copy cases_LOCAL if not present
    if not cases_local.exists():
        cases_local.write_bytes(cases_source.read_bytes())

    bound_sources = [
        "food_composition_cooked_sourced.json",
        "stage3_milp_solver.py",
        "stage2_stage3_adapter.py",
        "test_stage2_stage3_translation.py",
        "stage2_counterfactual_engine.py",
        "nutrient_rule_policy.py",
        "stage1_deployment.py",
        "Saved Model/stage2_comparison_study_v2/cases_LOCAL.json",
        "Saved Model/stage2_comparison_study_v2/summary.json",
        "Saved Model/stage2_comparison_study_v2/verification.json",
        "Saved Model/stage2_comparison_study_v2/audit_complete.json",
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
        "total_planned_solves": 900,
        "time_limit_sec": TIME_LIMIT_SEC,
        "max_workers": MAX_WORKERS,
        "device": "cpu",
        "solver_backend": "scipy.optimize.milp (HiGHS)",
        "hashes": source_hashes,
        "evaluations": {
            "evaluation_a": "Diagnostic Target-to-Meal Translation (all 100 profiles x 3 arms x 3 modes)",
            "evaluation_b": "End-to-End Pipeline Acceptance (Stage 2 Accepted AND Stage 3 Accepted)",
        },
        "hypothesis_testing": {
            "primary_contrast": "Model-Guided (C) vs Minimum-Change (B)",
            "secondary_contrasts": [
                "Model-Guided (C) vs Rules-Only (A)",
                "Minimum-Change (B) vs Rules-Only (A)",
            ],
            "binary_test": "Exact two-sided paired binomial discordance test (McNemar exact)",
            "bootstrap_resamples": 10000,
            "bootstrap_seed": 20261123,
            "continuous_estimator": "Paired mean difference bootstrap percentile 95% CI",
        },
    }

    protocol_path = STUDY_DIR / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    # Generate human-readable Protocol.md
    md_lines = [
        f"# Protocol: {STUDY_NAME}",
        "",
        f"**Date:** {protocol['created_at']}  ",
        f"**Cohort:** 100 reserved temporal NHANES adult participants (`cases_LOCAL.json`, SHA-256: `{protocol['cases_sha256'][:16]}...`)  ",
        f"**Catalog:** 49 cooked dishes (`food_composition_cooked_sourced.json`, SHA-256: `{catalog_hash[:16]}...`)  ",
        f"**Total Tasks:** {protocol['total_planned_solves']} (100 cases x 3 arms x 3 dietary modes)  ",
        f"**Time Limit:** {TIME_LIMIT_SEC}s per solve via HiGHS MILP on {MAX_WORKERS} CPU worker threads.  ",
        "",
        "## Evaluated Arms",
        "1. **Arm A (Rules-Only):** Deterministic clinical reference target vector.",
        "2. **Arm B (Minimum-Change):** Scale-normalized convex quadratic projection of observed intake.",
        "3. **Arm C (Model-Guided):** 50-step Adam search on risk loss + scale-normalized QP projection.",
        "",
        "## Evaluated Dietary Modes",
        "1. `pure_vegetarian`: lacto-vegetarian, zero non-veg foods.",
        "2. `veg_nonveg_combo`: combination of vegetarian staples with non-veg protein dish.",
        "3. `pure_non_vegetarian`: non-veg protein prioritized with cereal and vegetable sabzi.",
        "",
        "## Dual Evaluation Protocol",
        "1. **Evaluation A (Diagnostic Translation):** Evaluates all 900 target-to-meal solves regardless of upstream score gate. Evaluates discrete meal feasibility, dish variety, portion compliance, and soft target attainment.",
        "2. **Evaluation B (End-to-End Acceptance):** A case succeeds if and only if upstream Stage 2 accepted AND downstream Stage 3 independently passes all checks. Denominator is all 100 participants.",
    ]
    (STUDY_DIR / "Protocol.md").write_text("\n".join(md_lines), encoding="utf-8")

    return protocol


def run_translation_study():
    """Main execution orchestrator."""
    print("=" * 70)
    print(f"Starting {STUDY_NAME} (900 planned tasks)")
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
    print("Preparing 900 tasks from frozen cases and checkpoints...")
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

        # Load RO checkpoint for baseline probabilities and observed intake
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

    print(f"Total tasks prepared: {len(tasks)}")

    # 4. Execute tasks in parallel
    print(f"Executing tasks with ProcessPoolExecutor (max_workers={MAX_WORKERS})...")
    start_total = time.perf_counter()
    results = []
    resumed_count = 0
    fresh_count = 0

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {executor.submit(solve_translation_task, t): t for t in tasks}
        for future in as_completed(future_map):
            res = future.result()
            results.append(res)
            if res.get("resumed"):
                resumed_count += 1
            else:
                fresh_count += 1
            if len(results) % 100 == 0 or len(results) == len(tasks):
                elapsed_so_far = time.perf_counter() - start_total
                print(f"  Progress: {len(results)}/{len(tasks)} completed ({resumed_count} resumed, {fresh_count} fresh) [{elapsed_so_far:.1f}s]")

    total_time = time.perf_counter() - start_total
    print(f"All {len(results)} tasks completed in {total_time:.2f}s ({total_time / 60:.2f} min)")

    # 5. Build summary.json
    summary: dict[str, Any] = {
        "study": STUDY_NAME,
        "cohort_size": 100,
        "total_tasks_attempted": len(tasks),
        "total_tasks_completed": len(results),
        "total_wallclock_seconds": round(total_time, 2),
        "max_workers": MAX_WORKERS,
        "hardware": {
            "device": "cpu",
            "cpu_cores": os.cpu_count(),
            "solver": "scipy.optimize.milp / HiGHS",
        },
        "by_arm": {},
        "by_mode": {},
        "by_arm_mode": {},
    }

    # Aggregate by arm and mode
    for arm in arms:
        arm_res = [r for r in results if r["arm"] == arm]
        s2_acc = sum(1 for r in arm_res if r["stage2_accepted"]) // len(modes)  # 100 participants
        s3_acc = sum(1 for r in arm_res if r["stage3_accepted"])
        e2e_acc = sum(1 for r in arm_res if r["end_to_end_accepted"])
        dists = [r["normalized_target_distance"] for r in arm_res if r["normalized_target_distance"] is not None]

        summary["by_arm"][arm] = {
            "attempted_solves": len(arm_res),
            "stage2_accepted_participants": s2_acc,
            "stage2_accepted_pct": round(s2_acc / 100 * 100.0, 1),
            "stage3_accepted_solves": s3_acc,
            "stage3_accepted_pct": round(s3_acc / len(arm_res) * 100.0, 1),
            "end_to_end_accepted_solves": e2e_acc,
            "end_to_end_accepted_pct": round(e2e_acc / len(arm_res) * 100.0, 1),
            "mean_normalized_target_distance": round(float(np.mean(dists)), 4) if dists else None,
            "median_normalized_target_distance": round(float(np.median(dists)), 4) if dists else None,
        }

    for mode in modes:
        mode_res = [r for r in results if r["dietary_mode"] == mode]
        s3_acc = sum(1 for r in mode_res if r["stage3_accepted"])
        e2e_acc = sum(1 for r in mode_res if r["end_to_end_accepted"])
        summary["by_mode"][mode] = {
            "attempted_solves": len(mode_res),
            "stage3_accepted_solves": s3_acc,
            "stage3_accepted_pct": round(s3_acc / len(mode_res) * 100.0, 1),
            "end_to_end_accepted_solves": e2e_acc,
            "end_to_end_accepted_pct": round(e2e_acc / len(mode_res) * 100.0, 1),
        }

    for arm in arms:
        for mode in modes:
            cell_res = [r for r in results if r["arm"] == arm and r["dietary_mode"] == mode]
            s2_acc = sum(1 for r in cell_res if r["stage2_accepted"])
            s3_acc = sum(1 for r in cell_res if r["stage3_accepted"])
            e2e_acc = sum(1 for r in cell_res if r["end_to_end_accepted"])
            dists = [r["normalized_target_distance"] for r in cell_res if r["normalized_target_distance"] is not None]
            runtimes = [r["solve_time_seconds"] for r in cell_res]

            summary["by_arm_mode"][f"{arm}__{mode}"] = {
                "n_cases": len(cell_res),
                "stage2_accepted": s2_acc,
                "stage3_accepted": s3_acc,
                "end_to_end_accepted": e2e_acc,
                "end_to_end_pct": round(e2e_acc / len(cell_res) * 100.0, 1),
                "median_runtime_seconds": round(float(np.median(runtimes)), 3),
                "median_target_distance": round(float(np.median(dists)), 4) if dists else None,
            }

    summary_path = STUDY_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary written to {summary_path}")

    return summary


if __name__ == "__main__":
    run_translation_study()
