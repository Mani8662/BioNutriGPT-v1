"""BioNutriGPT Stage 2 Three-Arm Comparison Module.

Implements the three nutrient-target construction methods:
  Arm A: Deterministic rules-only nutrient targets based strictly on clinical
         guidelines, AMDRs, and DRI reference values without optimization.
  Arm B: Minimum-change convex quadratic projection of observed intake vector
         onto the joint feasible polytope without model-guided search.
  Arm C: Full deployed Stage 2 model-guided Adam optimization followed by
         the identical joint projection.

All three arms evaluate under the exact same participant envelope,
tightened bounds, energy balance, ratio rules, and closed-loop score gate.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

import stage2_counterfactual_engine as s2
import stage1_deployment as deployment
from verify_stage2_constraint_audit_repaired_v1 import check

NAMES = s2.DIET_RAW_COLS
IX = {n: i for i, n in enumerate(NAMES)}


def build_rules_only_target(
    *,
    raw_vector: np.ndarray,
    metadata: Mapping[str, Any],
    p_diab_0: float,
    p_obes_0: float,
    p_cvd_0: float,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Arm A: Pure deterministic nutrient target construction.

    Contains ZERO gradient descent, ZERO model-score optimization, and
    ZERO quadratic/linear programming projection. Targets are derived in closed
    form from standard clinical guidelines, AMDR intervals, and DRI references.
    """
    t0 = time.perf_counter()

    # Exact prescribed calorie target
    t_kcal = float(lower[IX["DR1TKCAL"]] / 0.95)
    weight_kg = float(metadata["weight_kg"])
    age_years = float(metadata["age_years"])
    is_female = bool(metadata["is_female"])

    target = np.zeros(39, dtype=np.float64)

    # 1. KCAL: Prescribed energy requirement
    target[IX["DR1TKCAL"]] = t_kcal

    # 2. Protein: Standard clinical guideline target: 1.0 g/kg body weight, bounded by envelope
    prot_target = max(1.0 * weight_kg, lower[IX["DR1TPROT"]])
    prot_target = min(prot_target, upper[IX["DR1TPROT"]])
    target[IX["DR1TPROT"]] = prot_target

    # 3. Fat: 25% of energy, or 30% under diabetes routing to allow lower carbohydrate fraction
    fat_pct = 0.30 if p_diab_0 >= s2.CUTOFF_DIABETES else 0.25
    target[IX["DR1TTFAT"]] = (fat_pct * t_kcal) / 9.0

    # 4. Alcohol: 0.0 g (abstinence recommendation for therapeutic nutrition)
    target[IX["DR1TALCO"]] = 0.0

    # 5. Carbohydrate: Closed-form thermodynamic energy balance
    carb_kcal = t_kcal - (
        4.0 * target[IX["DR1TPROT"]]
        + 9.0 * target[IX["DR1TTFAT"]]
        + 7.0 * target[IX["DR1TALCO"]]
    )
    target[IX["DR1TCARB"]] = carb_kcal / 4.0

    # Under diabetes routing, if carb exceeds the 45% envelope ceiling, adjust fat within AMDR (up to 35%)
    if target[IX["DR1TCARB"]] > upper[IX["DR1TCARB"]]:
        target[IX["DR1TCARB"]] = upper[IX["DR1TCARB"]]
        rem_kcal = t_kcal - (4.0 * target[IX["DR1TPROT"]] + 4.0 * target[IX["DR1TCARB"]])
        target[IX["DR1TTFAT"]] = rem_kcal / 9.0

    # Baseline ratio limits for personal coherence
    base_sf = float(raw_vector[s2.FEATURE_COLS.index("SUGAR_TO_FIBER_RATIO")])
    base_nk = float(raw_vector[s2.FEATURE_COLS.index("SODIUM_POTASSIUM_RATIO")])
    base_su = float(raw_vector[s2.FEATURE_COLS.index("SAT_TO_UNSAT_FAT_RATIO")])
    a_sf = max(base_sf, 1.0)
    a_nk = max(base_nk, 1.0)
    a_su = max(base_su, 0.4)

    # 6. Dietary Fiber: Reference guideline target (lower bound floor: 14g/1000kcal or 28g/35g)
    target[IX["DR1TFIBE"]] = lower[IX["DR1TFIBE"]]

    # 7. Sugar: Constrained by WHO guideline (<10% kcal) and participant's sugar/fiber ratio envelope
    max_sugar_ratio = a_sf * (target[IX["DR1TFIBE"]] + 1.0) * 0.99
    target[IX["DR1TSUGR"]] = min((0.10 * t_kcal) / 4.0, max_sugar_ratio)

    # 8. Polyunsaturated Fat (PUFA): Standard AMDR target (7% of energy)
    target[IX["DR1TPFAT"]] = (0.07 * t_kcal) / 9.0

    # 9. Saturated Fat: Capped by CVD guideline (5% or 7% of energy) and sat/unsat fat ratio limit
    sfat_cap = ((0.05 if p_cvd_0 >= s2.CUTOFF_CVD else 0.07) * t_kcal) / 9.0
    sfat_ratio_cap = (a_su / (1.0 + a_su)) * (target[IX["DR1TTFAT"]] + 0.1) * 0.99
    target[IX["DR1TSFAT"]] = min(sfat_cap, sfat_ratio_cap)

    # 10. Monounsaturated Fat (MUFA): Remainder of total fat (TTFAT - SFAT - PFAT)
    target[IX["DR1TMFAT"]] = max(
        0.0, target[IX["DR1TTFAT"]] - target[IX["DR1TSFAT"]] - target[IX["DR1TPFAT"]]
    )

    # 11. Cholesterol: Standard guideline target 100 mg (CVD) or 150 mg (general)
    target[IX["DR1TCHOL"]] = 100.0 if p_cvd_0 >= s2.CUTOFF_CVD else 150.0

    # 12. Potassium: Reference floor (3500 mg if CVD routing, else 2600 mg)
    target[IX["DR1TPOTA"]] = 3500.0 if p_cvd_0 >= s2.CUTOFF_CVD else 2600.0

    # 13. Sodium: Dietary guideline ceiling (1500 mg if CVD, else 2000 mg), bounded by ratio
    sodi_cap = 1500.0 if p_cvd_0 >= s2.CUTOFF_CVD else 2000.0
    sodi_ratio_cap = a_nk * (target[IX["DR1TPOTA"]] + 1.0) * 0.99
    target[IX["DR1TSODI"]] = min(sodi_cap, sodi_ratio_cap)

    # 14. Folate forms: DFE RDA reference 400 ug DFE; synthetic folic acid 0.0 ug; food/total folate 400 ug
    target[IX["DR1TFDFE"]] = 400.0
    target[IX["DR1TFF"]] = 400.0
    target[IX["DR1TFOLA"]] = 400.0
    target[IX["DR1TFA"]] = 0.0

    # 15. Retinol & Exposure reporting: Preformed retinol 0.0 ug; Caffeine 0.0 mg
    target[IX["DR1TRET"]] = 0.0
    target[IX["DR1TCAFF"]] = 0.0

    # 16. Non-essential carotenoids: 0.0 ug
    for code in ("DR1TACAR", "DR1TBCAR", "DR1TLYCO", "DR1TLZ"):
        target[IX[code]] = 0.0

    # 17. Remaining 17 vitamins and minerals with positive RDAs/AIs: Set to lower bound (RDA/AI reference)
    for code in NAMES:
        idx = IX[code]
        if target[idx] > 0.0:
            continue
        low_val = lower[idx]
        if low_val > 0.0:
            target[idx] = low_val
        else:
            target[idx] = 0.0

    dt = time.perf_counter() - t0
    audit = {
        "arm": "rules_only",
        "method": "deterministic_clinical_reference_rules",
        "optimization_used": False,
        "projection_used": False,
        "runtime_ms": dt * 1000.0,
    }
    return target, audit


def run_minimum_change_arm(
    *,
    raw_vector: np.ndarray,
    metadata: Mapping[str, Any],
    p_cvd_0: float,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Arm B: Minimum-change joint constraint projection of observed intake.

    Contains ZERO model-guided gradient optimization. Projects the observed
    nutrient vector onto the joint feasible polytope using fixed engineering scales.
    """
    bio_len = len(s2.BIO_COLS)
    n_obs = raw_vector[bio_len : bio_len + 39].astype(np.float64).copy()

    projected, proj_audit = s2._project_to_joint_feasible_polytope(
        n_candidate=n_obs,
        lower=lower,
        upper=upper,
        raw_vector=raw_vector,
        metadata=metadata,
        p_cvd_0=p_cvd_0,
    )

    f1 = s2._apply_filter_1_thermodynamic_balance(projected)
    f2 = s2._apply_filter_2_iom_safety(projected, weight_kg=metadata["weight_kg"], p_cvd_0=p_cvd_0)
    base_ratios = raw_vector[bio_len + 39 : bio_len + 39 + len(s2.DIET_RATIO_COLS)]
    f3 = s2._apply_filter_3_ratio_coherence(
        projected,
        baseline_ratios_from_vector=base_ratios,
        lower=lower,
        upper=upper,
    )

    audit = {
        "arm": "minimum_change",
        "method": "scale_normalized_qp_projection_from_observed",
        "optimization_used": True,
        "model_guided_search": False,
        "projection_audit": proj_audit,
        "filter_1": f1,
        "filter_2": f2,
        "filter_3": f3,
    }
    return projected, audit


def run_model_guided_arm(
    *,
    model: torch.nn.Module,
    artifacts: Mapping[str, Any],
    raw_vector: np.ndarray,
    metadata: Mapping[str, Any],
    p_diab_0: float,
    p_obes_0: float,
    p_cvd_0: float,
    lower: np.ndarray,
    upper: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Arm C: Full deployed Stage 2 model-guided search + joint projection."""
    t0 = time.perf_counter()

    p_d0_t = torch.tensor(p_diab_0, dtype=torch.float32, device=device)
    p_o0_t = torch.tensor(p_obes_0, dtype=torch.float32, device=device)
    p_c0_t = torch.tensor(p_cvd_0, dtype=torch.float32, device=device)

    # Step 1: 50-step Adam search guided by model risk gradients
    n_adam_clipped, _, _, _ = s2._run_counterfactual_optimization(
        model=model,
        artifacts=artifacts,
        raw_vector=raw_vector,
        lower=lower,
        upper=upper,
        device=device,
        p_diab_0=p_d0_t,
        p_obes_0=p_o0_t,
        p_cvd_0=p_c0_t,
    )

    # Step 2: Identical joint constraint projection
    projected, proj_audit = s2._project_to_joint_feasible_polytope(
        n_candidate=n_adam_clipped,
        lower=lower,
        upper=upper,
        raw_vector=raw_vector,
        metadata=metadata,
        p_cvd_0=p_cvd_0,
    )

    f1 = s2._apply_filter_1_thermodynamic_balance(projected)
    f2 = s2._apply_filter_2_iom_safety(projected, weight_kg=metadata["weight_kg"], p_cvd_0=p_cvd_0)
    base_ratios = raw_vector[len(s2.BIO_COLS) + 39 : len(s2.BIO_COLS) + 39 + len(s2.DIET_RATIO_COLS)]
    f3 = s2._apply_filter_3_ratio_coherence(
        projected,
        baseline_ratios_from_vector=base_ratios,
        lower=lower,
        upper=upper,
    )

    dt = time.perf_counter() - t0
    audit = {
        "arm": "model_guided",
        "method": "50_step_adam_plus_qp_projection",
        "optimization_used": True,
        "model_guided_search": True,
        "adam_steps": 50,
        "total_runtime_ms": dt * 1000.0,
        "projection_audit": proj_audit,
        "filter_1": f1,
        "filter_2": f2,
        "filter_3": f3,
    }
    return projected, audit


def evaluate_candidate_target(
    *,
    model: torch.nn.Module,
    artifacts: Mapping[str, Any],
    raw_vector: np.ndarray,
    target_39: np.ndarray,
    metadata: Mapping[str, Any],
    p_diab_0: float,
    p_obes_0: float,
    p_cvd_0: float,
    lower: np.ndarray,
    upper: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    """Shared, uniform evaluation across all three arms.

    Replays:
      1. Independent numerical constraints (bounds, energy, ratios, Filter 2 invariants).
      2. Differentiable normalization and Stage 1 calibrated model rescoring.
      3. Closed-loop score gate check (Delta p <= 0.0015 across all 3 outcomes).
      4. Standardized scale-normalized distance from observed intake.
      5. Overall certification decision (numerical AND score gate).
    """
    target = np.asarray(target_39, dtype=np.float64)
    if target.shape != (39,) or not np.isfinite(target).all():
        return {
            "valid": False,
            "error": "Target contains non-finite values or invalid shape",
            "numerical_pass": False,
            "score_gate_pass": False,
            "overall_accepted": False,
        }

    # 1. Independent numerical check
    base_dict = {k: float(raw_vector[IX[k] + len(s2.BIO_COLS)]) for k in NAMES}
    num_check = check(target, lower, upper, NAMES, base_dict, metadata, p_cvd_0)
    numerical_pass = bool(num_check["joint"])

    # 2. Frozen-model rescoring
    raw_t = torch.as_tensor(raw_vector, dtype=torch.float32, device=device)
    target_t = torch.as_tensor(target, dtype=torch.float32, device=device)
    cand_88 = s2._construct_candidate_88(raw_t, target_t)
    norm_cand = s2._normalize_torch(cand_88, artifacts)

    p_d_cand, p_o_cand, p_c_cand = s2._predict_probabilities(model, norm_cand, no_grad=True)
    p_d = float(p_d_cand.cpu())
    p_o = float(p_o_cand.cpu())
    p_c = float(p_c_cand.cpu())

    delta_d = float(p_d - p_diab_0)
    delta_o = float(p_o - p_obes_0)
    delta_c = float(p_c - p_cvd_0)

    score_gate_pass = bool(delta_d <= 0.0015 and delta_o <= 0.0015 and delta_c <= 0.0015)
    overall_accepted = bool(numerical_pass and score_gate_pass)

    # 3. Scale-normalized distance from observed intake
    n_obs = raw_vector[len(s2.BIO_COLS) : len(s2.BIO_COLS) + 39].astype(np.float64)
    scales = np.array([s2.DAILY_REFERENCE_SCALES[k] for k in NAMES], dtype=np.float64)
    dist_norm = float(np.sqrt(np.sum(((target - n_obs) / scales) ** 2)))

    # Major nutrient changes
    major_changes = {
        "KCAL": {"observed": float(n_obs[IX["DR1TKCAL"]]), "target": float(target[IX["DR1TKCAL"]]), "delta": float(target[IX["DR1TKCAL"]] - n_obs[IX["DR1TKCAL"]]), "unit": "kcal"},
        "PROT": {"observed": float(n_obs[IX["DR1TPROT"]]), "target": float(target[IX["DR1TPROT"]]), "delta": float(target[IX["DR1TPROT"]] - n_obs[IX["DR1TPROT"]]), "unit": "g"},
        "CARB": {"observed": float(n_obs[IX["DR1TCARB"]]), "target": float(target[IX["DR1TCARB"]]), "delta": float(target[IX["DR1TCARB"]] - n_obs[IX["DR1TCARB"]]), "unit": "g"},
        "TFAT": {"observed": float(n_obs[IX["DR1TTFAT"]]), "target": float(target[IX["DR1TTFAT"]]), "delta": float(target[IX["DR1TTFAT"]] - n_obs[IX["DR1TTFAT"]]), "unit": "g"},
        "SODI": {"observed": float(n_obs[IX["DR1TSODI"]]), "target": float(target[IX["DR1TSODI"]]), "delta": float(target[IX["DR1TSODI"]] - n_obs[IX["DR1TSODI"]]), "unit": "mg"},
        "POTA": {"observed": float(n_obs[IX["DR1TPOTA"]]), "target": float(target[IX["DR1TPOTA"]]), "delta": float(target[IX["DR1TPOTA"]] - n_obs[IX["DR1TPOTA"]]), "unit": "mg"},
    }

    return {
        "valid": True,
        "numerical_pass": numerical_pass,
        "bound_violations": num_check["bound_names"],
        "energy_ok": num_check["energy_ok"],
        "ratios_ok": num_check["ratios_ok"],
        "filter2_ok": num_check["filter2_ok"],
        "target_probabilities": {"diabetes": p_d, "obesity": p_o, "cvd": p_c},
        "score_deltas": {"diabetes": delta_d, "obesity": delta_o, "cvd": delta_c},
        "score_gate_pass": score_gate_pass,
        "overall_accepted": overall_accepted,
        "distance_from_observed_normalized": dist_norm,
        "major_nutrient_changes": major_changes,
    }
