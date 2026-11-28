#!/usr/bin/env python3
"""BioNutriGPT Stage 2 Component Ablation Engine (v1).

Implements the experimental component ablations for the Stage 2 Counterfactual
Prescription Engine under the repaired_v1 release.
Evaluates all variants under identical participant inputs, baseline predictions,
nutrient envelopes, solver budgets, and independent validation standards.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn

import stage1_deployment as deployment
import stage2_counterfactual_engine as s2
import stage2_comparison_arms_v2 as arms
from stage2_comparison_checks_v2 import check

NAMES = s2.DIET_RAW_COLS
IX = {n: i for i, n in enumerate(NAMES)}


def run_ablation_variant(
    *,
    case: Mapping[str, Any],
    variant_config: Mapping[str, Any],
    model: nn.Module,
    artifacts: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Execute one ablation variant for a single participant profile.

    variant_config must provide:
      - variant_id: str
      - include_model_objective: bool
      - include_dietary_change_penalty: bool
      - include_barrier_penalty: bool
      - include_joint_projection: bool
      - include_sequential_filters: bool
    """
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    raw, ratio_audit = s2.canonicalize_dietary_ratios(case["vector"])
    meta = case["metadata"]
    seqn = int(case["SEQN"])
    case_idx = int(case["case"])
    variant_id = str(variant_config["variant_id"])

    # 1. Baseline risk predictions
    raw_t = torch.as_tensor(raw, dtype=torch.float32, device=device).unsqueeze(0)
    norm_0 = s2._normalize_torch(raw_t, artifacts)
    p_d0_t, p_o0_t, p_c0_t = s2._predict_probabilities(model, norm_0, no_grad=True)
    p_d0 = float(p_d0_t.cpu())
    p_o0 = float(p_o0_t.cpu())
    p_c0 = float(p_c0_t.cpu())

    # 2. Prescribed caloric budget and bounds
    w = float(meta["weight_kg"])
    h = float(meta["height_cm"])
    age = float(meta["age_years"])
    is_fem = bool(meta["is_female"])
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

    bio_len = len(s2.BIO_COLS)
    diet_len = len(s2.DIET_RAW_COLS)
    n_obs = raw[bio_len : bio_len + diet_len].astype(np.float64)

    # 3. Adam optimization (50 steps)
    n_current = torch.tensor(n_obs, dtype=torch.float32, device=device)
    lower_t = torch.as_tensor(lower, dtype=torch.float32, device=device)
    upper_t = torch.as_tensor(upper, dtype=torch.float32, device=device)

    delta_N = torch.zeros(39, dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([delta_N], lr=0.05)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_adam_0 = time.perf_counter()

    final_loss_breakdown: dict[str, float] = {}

    for step in range(50):
        optimizer.zero_grad(set_to_none=True)
        n_candidate = n_current + delta_N
        candidate_88 = s2._construct_candidate_88(raw_t.squeeze(0), n_candidate)
        normalized = s2._normalize_torch(candidate_88, artifacts)
        p_diab, p_obes, p_cvd = s2._predict_probabilities(model, normalized, no_grad=False)

        loss = torch.tensor(0.0, dtype=torch.float32, device=device)

        if variant_config["include_model_objective"]:
            model_risk = 2.0 * p_cvd + 2.0 * p_diab + 1.0 * p_obes
            pareto = 50.0 * (
                torch.relu(p_diab - p_d0_t)
                + torch.relu(p_obes - p_o0_t)
                + torch.relu(p_cvd - p_c0_t)
            )
            loss = loss + model_risk + pareto
        else:
            model_risk = torch.tensor(0.0, device=device)
            pareto = torch.tensor(0.0, device=device)

        if variant_config["include_dietary_change_penalty"]:
            change_penalty = 0.001 * torch.sum(delta_N ** 2)
            loss = loss + change_penalty
        else:
            change_penalty = torch.tensor(0.0, device=device)

        if variant_config["include_barrier_penalty"]:
            barrier = s2.default_barrier_penalty(n_candidate, lower_t, upper_t)
            loss = loss + barrier.reshape(())
        else:
            barrier = torch.tensor(0.0, device=device)

        if step == 49:
            final_loss_breakdown = {
                "model_risk_term": float(model_risk.detach().cpu()),
                "pareto_penalty": float(pareto.detach().cpu()),
                "change_penalty": float(change_penalty.detach().cpu()),
                "barrier_penalty": float(barrier.detach().cpu()),
                "total_loss": float(loss.detach().cpu()),
            }

        if loss.requires_grad:
            loss.backward()
            optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_adam_ms = (time.perf_counter() - t_adam_0) * 1000.0

    with torch.no_grad():
        n_adam_raw = (n_current + delta_N.detach()).cpu().numpy().astype(np.float64)
        n_adam_clipped = torch.clamp(n_current + delta_N.detach(), lower_t, upper_t).cpu().numpy().astype(np.float64)

    # 4. Joint constraint projection
    t_proj_0 = time.perf_counter()
    if variant_config["include_joint_projection"]:
        projected, proj_audit = s2._project_to_joint_feasible_polytope(
            n_candidate=n_adam_clipped,
            lower=lower,
            upper=upper,
            raw_vector=raw,
            metadata=meta,
            p_cvd_0=p_c0,
            raise_on_infeasible=False,
        )
    else:
        projected = n_adam_clipped.copy()
        proj_audit = {
            "passed": True,
            "feasible": None,
            "status": "bypassed_by_ablation",
            "fallback_used": False,
            "runtime_ms": 0.0,
        }
    t_proj_ms = (time.perf_counter() - t_proj_0) * 1000.0

    # 5. Sequential downstream filters 1–3
    cand_filtered = projected.copy()
    if variant_config["include_sequential_filters"]:
        f1 = s2._apply_filter_1_thermodynamic_balance(cand_filtered)
        f2 = s2._apply_filter_2_iom_safety(cand_filtered, weight_kg=w, p_cvd_0=p_c0)
        base_ratios = raw[bio_len + diet_len : bio_len + diet_len + len(s2.DIET_RATIO_COLS)]
        f3 = s2._apply_filter_3_ratio_coherence(
            cand_filtered,
            baseline_ratios_from_vector=base_ratios,
            lower=lower,
            upper=upper,
        )
    else:
        f1 = {"passed": True, "bypassed": True, "changes": []}
        f2 = {"passed": True, "bypassed": True, "changes": []}
        f3 = {"passed": True, "bypassed": True, "changes": []}

    n_final = cand_filtered.copy()

    # 6. Evaluation through shared independent validator and score gate
    eval_result = arms.evaluate_candidate_target(
        model=model,
        artifacts=artifacts,
        raw_vector=raw,
        target_39=n_final,
        metadata=meta,
        p_diab_0=p_d0,
        p_obes_0=p_o0,
        p_cvd_0=p_c0,
        lower=lower,
        upper=upper,
        device=device,
    )

    t_total_ms = (time.perf_counter() - t_start) * 1000.0

    # 7. Compute explicit constraint residuals for transparency
    low_tight, up_tight = s2._build_tightened_joint_bounds(lower, upper, meta, p_c0)
    A_ub, b_ub = s2._build_linear_constraint_matrices(raw)
    max_bound_residual = float(np.max(np.maximum(0.0, np.maximum(low_tight - n_final, n_final - up_tight))))
    max_linear_residual = float(np.max(np.maximum(0.0, A_ub @ n_final - b_ub)))

    # Scale-normalized distance from observed intake
    scales = np.array([s2.DAILY_REFERENCE_SCALES[k] for k in NAMES], dtype=np.float64)
    l1_norm_dist = float(np.sum(np.abs(n_final - n_obs) / scales))
    l2_norm_dist = float(np.sqrt(np.sum(((n_final - n_obs) / scales) ** 2)))

    # Differences introduced by intermediate steps
    adam_shift_l2 = float(np.sqrt(np.sum(((n_adam_clipped - n_obs) / scales) ** 2)))
    proj_shift_l2 = float(np.sqrt(np.sum(((projected - n_adam_clipped) / scales) ** 2)))
    filter_shift_l2 = float(np.sqrt(np.sum(((n_final - projected) / scales) ** 2)))

    record = {
        "case": case_idx,
        "SEQN": seqn,
        "variant_id": variant_id,
        "variant_config": dict(variant_config),
        "timing_ms": {
            "adam_ms": t_adam_ms,
            "projection_ms": t_proj_ms,
            "total_ms": t_total_ms,
        },
        "baseline_probabilities": {
            "diabetes": p_d0,
            "obesity": p_o0,
            "cvd": p_c0,
        },
        "prescribed_energy": {
            "BMR": float(bmr),
            "TDEE": float(tdee),
            "target_calories": float(t_kcal),
        },
        "adam_optimization": {
            "steps": 50,
            "final_loss": final_loss_breakdown,
            "shift_from_obs_norm_l2": adam_shift_l2,
        },
        "projection": {
            **proj_audit,
            "shift_from_adam_norm_l2": proj_shift_l2,
        },
        "filters": {
            "filter_1": f1,
            "filter_2": f2,
            "filter_3": f3,
            "shift_from_proj_norm_l2": filter_shift_l2,
        },
        "residuals": {
            "max_bound_residual": max_bound_residual,
            "max_linear_residual": max_linear_residual,
            "energy_ok": eval_result.get("energy_ok", False),
            "ratios_ok": eval_result.get("ratios_ok", False),
            "filter2_ok": eval_result.get("filter2_ok", False),
            "bound_violations": eval_result.get("bound_violations", []),
        },
        "evaluation": {
            "numerical_pass": eval_result["numerical_pass"],
            "score_gate_pass": eval_result["score_gate_pass"],
            "overall_accepted": eval_result["overall_accepted"],
            "target_probabilities": eval_result["target_probabilities"],
            "score_deltas": eval_result["score_deltas"],
            "distance_from_observed_l1_norm": l1_norm_dist,
            "distance_from_observed_l2_norm": l2_norm_dist,
            "major_nutrient_changes": eval_result.get("major_nutrient_changes", {}),
        },
        "vectors": {
            "observed_nutrients": n_obs.tolist(),
            "pre_projection_adam_clipped": n_adam_clipped.tolist(),
            "post_projection": projected.tolist(),
            "final_target": n_final.tolist(),
            "lower_bounds": lower.tolist(),
            "upper_bounds": upper.tolist(),
        },
    }
    return record
