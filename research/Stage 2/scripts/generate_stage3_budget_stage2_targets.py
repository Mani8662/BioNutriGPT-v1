#!/usr/bin/env python3
"""Generate and cache Stage 2 target vectors across 3 arms for the 100 budget-experiment participants.

Arms:
1. rules_only: Deterministic clinical target construction
2. minimum_change: Scale-normalized QP projection of observed intake
3. model_guided: 50-step Adam search + scale-normalized QP projection

Saves atomic checkpoints in:
Saved Model/stage3_budget_experiment_v1/stage2_targets/case_{case:03d}_{seqn}_{arm}.json
"""
from __future__ import annotations

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
import stage2_comparison_arms_v2 as arms

STUDY_DIR = ROOT / "Saved Model" / "stage3_budget_experiment_v1"
TARGETS_DIR = STUDY_DIR / "stage2_targets"


def sha(p: Path | str) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_json_atomic(path: Path, data: Any) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def run_single_case_arm(
    *,
    case: dict[str, Any],
    arm_name: str,
    model: torch.nn.Module,
    artifacts: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    if device.type == "cuda":
        torch.cuda.synchronize()
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
            pre_projection_candidate = audit.get("pre_projection_candidate")
            solver_audit = audit
        else:
            raise ValueError(f"Unknown arm_name: {arm_name}")

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

        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        return {
            "case": case_idx,
            "SEQN": seqn,
            "arm": arm_name,
            "success": True,
            "error": None,
            "runtime_seconds": dt,
            "baseline_probabilities": {"diabetes": p_d0, "obesity": p_o0, "cvd": p_c0},
            "energy_prescription": {
                "BMR": float(bmr),
                "TDEE": float(tdee),
                "target_calories": float(t_kcal),
            },
            "lower_bounds": lower.tolist(),
            "upper_bounds": upper.tolist(),
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
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        return {
            "case": case_idx,
            "SEQN": seqn,
            "arm": arm_name,
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
            "runtime_seconds": dt,
            "baseline_probabilities": {"diabetes": p_d0, "obesity": p_o0, "cvd": p_c0},
            "energy_prescription": {
                "BMR": float(bmr),
                "TDEE": float(tdee),
                "target_calories": float(t_kcal),
            },
            "lower_bounds": lower.tolist(),
            "upper_bounds": upper.tolist(),
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
    print("Stage 2 Target Caching for Budget Experiment (100 participants x 3 arms)")
    print("=" * 70)

    TARGETS_DIR.mkdir(parents=True, exist_ok=True)
    cases_path = STUDY_DIR / "cases_LOCAL.json"
    assert cases_path.exists(), f"Missing cohort manifest: {cases_path}"
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    assert len(cases) == 100, f"Expected 100 cases, got {len(cases)}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    artifacts = s2._load_preprocessing_artifacts(deployment.PREPROCESSOR)
    model = s2._load_stage1_model(deployment.CHECKPOINT, artifacts, device)

    # Warm-up
    warm = torch.as_tensor(cases[0]["vector"], dtype=torch.float32, device=device).unsqueeze(0)
    s2._predict_probabilities(model, s2._normalize_torch(warm, artifacts), no_grad=True)
    if device.type == "cuda":
        torch.cuda.synchronize()

    arms_list = ["rules_only", "minimum_change", "model_guided"]
    total = len(cases) * len(arms_list)
    done = 0
    t_start = time.perf_counter()

    summary_counts = {a: {"attempted": 0, "accepted": 0, "numerical_pass": 0, "score_pass": 0} for a in arms_list}

    for case in cases:
        case_idx = case["case"]
        seqn = case["SEQN"]

        for arm in arms_list:
            ckpt_path = TARGETS_DIR / f"case_{case_idx:03d}_{seqn}_{arm}.json"
            if ckpt_path.exists():
                res = json.loads(ckpt_path.read_text(encoding="utf-8"))
            else:
                res = run_single_case_arm(
                    case=case,
                    arm_name=arm,
                    model=model,
                    artifacts=artifacts,
                    device=device,
                )
                write_json_atomic(ckpt_path, res)

            summary_counts[arm]["attempted"] += 1
            if res["overall_accepted"]:
                summary_counts[arm]["accepted"] += 1
            if res["numerical_pass"]:
                summary_counts[arm]["numerical_pass"] += 1
            if res["score_gate_pass"]:
                summary_counts[arm]["score_pass"] += 1

            done += 1
            if done % 30 == 0 or done == total:
                elapsed = time.perf_counter() - t_start
                rate = done / elapsed if elapsed > 0 else 0
                print(
                    f"[{done:03d}/{total:03d}] Case {case_idx:02d} SEQN {seqn} ({arm:14s}): "
                    f"acc={res['overall_accepted']} | Rate: {rate:.1f} tasks/s",
                    flush=True,
                )

    print("\nStage 2 Target Caching Complete:")
    for arm in arms_list:
        c = summary_counts[arm]
        print(
            f"  {arm:16s}: Accepted={c['accepted']}/{c['attempted']} ({c['accepted']/c['attempted']*100:.1f}%), "
            f"NumericalPass={c['numerical_pass']}/{c['attempted']}, ScorePass={c['score_pass']}/{c['attempted']}"
        )

    # Save summary
    summary_path = TARGETS_DIR / "stage2_targets_summary.json"
    summary_path.write_text(json.dumps(summary_counts, indent=2), encoding="utf-8")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
