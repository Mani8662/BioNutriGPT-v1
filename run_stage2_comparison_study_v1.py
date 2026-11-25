"""BioNutriGPT Stage 2 Three-Arm Comparative Added-Value Runner.

Executes all 100 reserved temporal NHANES profiles across the three arms:
  Arm A: Deterministic Rules-Only
  Arm B: Minimum-Change Joint Projection
  Arm C: Full Model-Guided Optimization + Joint Projection

Saves atomic checkpoints per participant and per arm with rigorous resume safeguards.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import stage1_deployment as deployment
import stage2_counterfactual_engine as s2
import stage2_comparison_arms as arms

OUT = ROOT / "Saved Model/stage2_comparison_study_v1"
CHECKPOINT_DIR = OUT / "checkpoints"


def sha(p: Path | str) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temp_path.replace(path)


def run_single_case_arm(
    *,
    case: dict[str, Any],
    arm_name: str,
    model: torch.nn.Module,
    artifacts: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    raw, _ = s2.canonicalize_dietary_ratios(case["vector"])
    meta = case["metadata"]
    seqn = int(case["SEQN"])
    case_idx = int(case["case"])

    # Baseline probabilities
    norm_0 = s2._normalize_torch(
        torch.as_tensor(raw, dtype=torch.float32, device=device).unsqueeze(0),
        artifacts,
    )
    p_d0_t, p_o0_t, p_c0_t = s2._predict_probabilities(model, norm_0, no_grad=True)
    p_d0 = float(p_d0_t.cpu())
    p_o0 = float(p_o0_t.cpu())
    p_c0 = float(p_c0_t.cpu())

    # Prescribed calories and bounds (identical across all arms)
    w = meta["weight_kg"]
    h = meta["height_cm"]
    age = meta["age_years"]
    is_fem = meta["is_female"]
    bmr = 10.0 * w + 6.25 * h - 5.0 * age + (-161.0 if is_fem else 5.0)
    tdee = bmr * s2.PAL_FACTORS[meta["activity_level"]]
    t_kcal = max(tdee - 500.0, 1200.0) if p_o0 >= s2.CUTOFF_OBESITY else max(tdee, 1200.0)

    lower, upper = s2._build_nutrient_bounds(
        artifacts=artifacts,
        t_kcal=t_kcal,
        weight_kg=w,
        is_female=is_fem,
        age_years=age,
        p_diab_0=p_d0,
        p_obes_0=p_o0,
        p_cvd_0=p_c0,
    )

    n_obs = raw[len(s2.BIO_COLS) : len(s2.BIO_COLS) + 39].tolist()

    try:
        pre_projection_candidate = None
        solver_audit = {}

        if arm_name == "rules_only":
            target_39, audit = arms.build_rules_only_target(
                raw_vector=raw,
                metadata=meta,
                p_diab_0=p_d0,
                p_obes_0=p_o0,
                p_cvd_0=p_c0,
                lower=lower,
                upper=upper,
            )
            solver_audit = audit

        elif arm_name == "minimum_change":
            pre_projection_candidate = n_obs
            target_39, audit = arms.run_minimum_change_arm(
                raw_vector=raw,
                metadata=meta,
                p_cvd_0=p_c0,
                lower=lower,
                upper=upper,
            )
            solver_audit = audit

        elif arm_name == "model_guided":
            # For logging, record the 50-step Adam candidate before projection
            p_d0_ten = torch.tensor(p_d0, dtype=torch.float32, device=device)
            p_o0_ten = torch.tensor(p_o0, dtype=torch.float32, device=device)
            p_c0_ten = torch.tensor(p_c0, dtype=torch.float32, device=device)

            n_adam, _, _, _ = s2._run_counterfactual_optimization(
                model=model,
                artifacts=artifacts,
                raw_vector=raw,
                lower=lower,
                upper=upper,
                device=device,
                p_diab_0=p_d0_ten,
                p_obes_0=p_o0_ten,
                p_cvd_0=p_c0_ten,
            )
            pre_projection_candidate = n_adam.tolist()

            target_39, audit = arms.run_model_guided_arm(
                model=model,
                artifacts=artifacts,
                raw_vector=raw,
                metadata=meta,
                p_diab_0=p_d0,
                p_obes_0=p_o0,
                p_cvd_0=p_c0,
                lower=lower,
                upper=upper,
                device=device,
            )
            solver_audit = audit
        else:
            raise ValueError(f"Unknown arm: {arm_name}")

        # Shared evaluation
        evaluation = arms.evaluate_candidate_target(
            model=model,
            artifacts=artifacts,
            raw_vector=raw,
            target_39=target_39,
            metadata=meta,
            p_diab_0=p_d0,
            p_obes_0=p_o0,
            p_cvd_0=p_c0,
            lower=lower,
            upper=upper,
            device=device,
        )

        dt = time.perf_counter() - t0
        return {
            "case": case_idx,
            "SEQN": seqn,
            "arm": arm_name,
            "success": True,
            "error": None,
            "runtime_seconds": dt,
            "baseline_probabilities": {"diabetes": p_d0, "obesity": p_o0, "cvd": p_c0},
            "observed_nutrients": n_obs,
            "pre_projection_candidate": pre_projection_candidate,
            "target_nutrients": target_39.tolist(),
            "target_probabilities": evaluation["target_probabilities"],
            "score_deltas": evaluation["score_deltas"],
            "score_gate_pass": evaluation["score_gate_pass"],
            "numerical_pass": evaluation["numerical_pass"],
            "bound_violations": evaluation["bound_violations"],
            "energy_ok": evaluation["energy_ok"],
            "ratios_ok": evaluation["ratios_ok"],
            "filter2_ok": evaluation["filter2_ok"],
            "overall_accepted": evaluation["overall_accepted"],
            "distance_from_observed_normalized": evaluation["distance_from_observed_normalized"],
            "major_nutrient_changes": evaluation["major_nutrient_changes"],
            "solver_audit": solver_audit,
        }

    except Exception as exc:
        dt = time.perf_counter() - t0
        return {
            "case": case_idx,
            "SEQN": seqn,
            "arm": arm_name,
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
            "runtime_seconds": dt,
            "baseline_probabilities": {"diabetes": p_d0, "obesity": p_o0, "cvd": p_c0},
            "observed_nutrients": n_obs,
            "pre_projection_candidate": None,
            "target_nutrients": None,
            "target_probabilities": None,
            "score_deltas": None,
            "score_gate_pass": False,
            "numerical_pass": False,
            "bound_violations": [],
            "energy_ok": False,
            "ratios_ok": False,
            "filter2_ok": False,
            "overall_accepted": False,
            "distance_from_observed_normalized": None,
            "major_nutrient_changes": None,
            "solver_audit": {"error": repr(exc)},
        }


def main():
    print("=" * 70)
    print("BioNutriGPT Stage 2 Three-Arm Comparative Added-Value Runner")
    print("=" * 70)

    # 1. Verify frozen protocol and cases
    protocol_path = OUT / "protocol.json"
    cases_path = OUT / "cases_LOCAL.json"
    assert protocol_path.exists(), f"Missing protocol: {protocol_path}"
    assert cases_path.exists(), f"Missing cases: {cases_path}"

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    for rel_path, expected_hash in protocol["hashes"].items():
        actual_hash = sha(ROOT / rel_path)
        assert actual_hash == expected_hash, f"Source changed after freeze: {rel_path} ({actual_hash} != {expected_hash})"

    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    assert len(cases) == 100, f"Expected 100 cases, got {len(cases)}"
    assert sha(cases_path) == protocol["cases_sha256"], "Cases JSON hash mismatch"

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Hardware configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(os.cpu_count() or 1)
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Torch threads: {torch.get_num_threads()}")

    artifacts = s2._load_preprocessing_artifacts(deployment.PREPROCESSOR)
    model = s2._load_stage1_model(deployment.CHECKPOINT, artifacts, device)

    arms_list = ["rules_only", "minimum_change", "model_guided"]
    total_tasks = len(cases) * len(arms_list)
    completed_tasks = 0

    results_by_arm: dict[str, list[dict[str, Any]]] = {a: [] for a in arms_list}
    t_start = time.perf_counter()

    for case in cases:
        case_idx = case["case"]
        seqn = case["SEQN"]

        for arm_name in arms_list:
            ckpt_path = CHECKPOINT_DIR / f"case_{case_idx:03d}_{seqn}_{arm_name}.json"

            if ckpt_path.exists():
                res = json.loads(ckpt_path.read_text(encoding="utf-8"))
                # Validate checkpoint schema
                assert res["case"] == case_idx
                assert res["SEQN"] == seqn
                assert res["arm"] == arm_name
            else:
                res = run_single_case_arm(
                    case=case,
                    arm_name=arm_name,
                    model=model,
                    artifacts=artifacts,
                    device=device,
                )
                write_json_atomic(ckpt_path, res)

            results_by_arm[arm_name].append(res)
            completed_tasks += 1

            if completed_tasks % 15 == 0 or completed_tasks == total_tasks:
                elapsed = time.perf_counter() - t_start
                rate = completed_tasks / elapsed if elapsed > 0 else 0
                print(
                    f"[{completed_tasks:03d}/{total_tasks:03d}] "
                    f"Case {case_idx:02d} SEQN {seqn} ({arm_name:14s}): "
                    f"accepted={res['overall_accepted']} num={res['numerical_pass']} "
                    f"score={res['score_gate_pass']} | Rate: {rate:.1f} tasks/s",
                    flush=True,
                )

    total_time = time.perf_counter() - t_start
    print(f"\nExecution finished in {total_time:.2f}s ({total_time / 60:.2f} min).")

    # Verify source hashes remained completely unchanged throughout execution
    for rel_path, expected_hash in protocol["hashes"].items():
        actual_hash = sha(ROOT / rel_path)
        assert actual_hash == expected_hash, f"Source mutated during execution: {rel_path}"

    # Compile comprehensive summary
    summary_arms: dict[str, Any] = {}
    for arm_name in arms_list:
        arm_res = results_by_arm[arm_name]
        valid = [r for r in arm_res if r.get("success")]
        accepted = [r for r in valid if r.get("overall_accepted")]
        num_pass = [r for r in valid if r.get("numerical_pass")]
        score_pass = [r for r in valid if r.get("score_gate_pass")]

        dists = [r["distance_from_observed_normalized"] for r in valid if r.get("distance_from_observed_normalized") is not None]
        times = [r["runtime_seconds"] for r in arm_res]

        # Fallbacks
        lp_fallbacks = 0
        for r in valid:
            audit = r.get("solver_audit", {})
            proj_audit = audit.get("projection_audit", {})
            if proj_audit.get("fallback_used"):
                lp_fallbacks += 1

        summary_arms[arm_name] = {
            "attempted": len(arm_res),
            "completed": len(valid),
            "errors": len(arm_res) - len(valid),
            "overall_accepted": len(accepted),
            "overall_accepted_pct": (len(accepted) / len(arm_res)) * 100.0,
            "numerical_pass": len(num_pass),
            "numerical_pass_pct": (len(num_pass) / len(arm_res)) * 100.0,
            "score_gate_pass": len(score_pass),
            "score_gate_pass_pct": (len(score_pass) / len(arm_res)) * 100.0,
            "lp_fallbacks": lp_fallbacks,
            "distance_from_observed": {
                "mean": float(np.mean(dists)) if dists else None,
                "std": float(np.std(dists)) if dists else None,
                "median": float(np.median(dists)) if dists else None,
                "iqr": float(np.percentile(dists, 75) - np.percentile(dists, 25)) if dists else None,
                "min": float(np.min(dists)) if dists else None,
                "max": float(np.max(dists)) if dists else None,
            },
            "runtime_seconds": {
                "total": float(np.sum(times)),
                "mean": float(np.mean(times)),
                "median": float(np.median(times)),
                "p95": float(np.percentile(times, 95)),
            },
        }

    summary = {
        "version": "stage2_comparison_study_v1",
        "cohort_size": 100,
        "total_tasks_attempted": total_tasks,
        "total_tasks_completed": sum(len([r for r in res if r.get("success")]) for res in results_by_arm.values()),
        "total_runtime_seconds": total_time,
        "device": str(device),
        "arms": summary_arms,
        "source_hashes_verified": True,
    }

    summary_path = OUT / "summary.json"
    write_json_atomic(summary_path, summary)
    print(f"\nSaved summary to {summary_path}")
    print(json.dumps(summary["arms"], indent=2))


if __name__ == "__main__":
    main()
